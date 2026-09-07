#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
09_executable_portfolio_backtest.py

China A-Share Cross-Sectional Alpha Research
Stage 9: Executable Walk-Forward Long-Only Portfolio Backtest

PURPOSE
-------
Convert the frozen Stage-7/8 predictive models into an executable A-share
long-only portfolio experiment under explicit next-open trading constraints
and transaction costs.

This is the first stage that should be interpreted as a self-financing trading
backtest rather than a predictive diagnostic.

CRITICAL LOOK-AHEAD SAFEGUARD
-----------------------------
Stage-7/8 stock-level prediction files were intentionally built on
`model_ready_5d`, which requires the future t+6 target to be observable.
That is appropriate for evaluating IC / RankIC, but it MUST NOT define the
tradable stock universe.

Therefore Stage 9 DOES NOT rank directly from those saved prediction files.

Instead, on every signal date it reconstructs the ex-ante prediction universe:

    liquid_universe_primary == True
    AND all 11 frozen z-scored features are finite

No future target availability, future suspension, or future exit tradability is
used for stock selection.

The frozen models are then re-applied:

    XGBOOST
        Stage-8 selected annual specification, retrained at the same annual
        expanding-window cutoffs using the Stage-8 purged training cache.

    RIDGE
        Stage-7 selected MONTHLY coefficient path.

    OLS
        Stage-7 MONTHLY coefficient path.

RESEARCH SAMPLE
---------------
OOS only:
    2019-01-01 to 2025-12-31

REBALANCE SCHEDULE
------------------
Non-overlapping 5-market-day cadence.

If the signal date has market-calendar index j:

    signal at close:        t_j
    execute at open:        t_{j+1}
    next signal:            t_{j+5}
    next execution / exit:  t_{j+6}

Thus a portfolio entered at t+1 is normally held until t+6, matching the frozen
5-day forward-return horizon without overlapping cohorts.

The schedule is anchored at the first OOS market date and then takes every
fifth market date.

PORTFOLIOS
----------
Primary:
    XGB_TOP20
    RIDGE_TOP20
    OLS_TOP20

Robustness:
    XGB_TOP10
    RIDGE_TOP10
    OLS_TOP10

Benchmark:
    UNIVERSE_EW
        Equal-weight ex-ante prediction-ready Primary Universe.

For a top-p portfolio, K = ceil(p * N_signal_universe).

At the next open, the ranking is scanned from best to worst. A new stock can
enter only if it is actually buyable at that open. If a top-ranked stock is
limit-up / suspended / otherwise not buyable, the algorithm fills from the
next-ranked executable stock. Existing holdings may remain selected without
requiring a new buy.

A-SHARE EXECUTION CONSTRAINTS
-----------------------------
At each execution open:

    Suspended / missing daily row:
        cannot buy or sell.

    Open at up-limit:
        cannot buy (conservative).

    Open at down-limit:
        cannot sell (conservative).

    Current ST:
        no NEW buy; existing holdings may be sold if otherwise sellable.

    T+1:
        satisfied mechanically by the weekly holding schedule; newly bought
        positions are never sold on the same day.

Positions that cannot be sold when they leave the target set remain frozen and
are carried forward until a later scheduled rebalance when they become sellable.
They are NOT silently dropped.

PORTFOLIO ACCOUNTING
--------------------
Self-financing portfolio with:
    cash
    fractional adjusted-price shares
    open-to-open marking at each scheduled execution date

Fractional shares are used because adjusted prices are synthetic total-return
prices; enforcing 100-share board lots on adjusted-share units would be
internally inconsistent. For institutional-size portfolios this omission is
economically negligible relative to the requested 0/5/10/20 bps cost grid.

Retained holdings are rebalanced toward equal weights when execution permits.
Locked positions remain at their current marked value, and the remaining NAV is
allocated across flexible target holdings.

TRANSACTION COSTS
-----------------
One-way proportional cost scenarios on traded notional:

    0 bps
    5 bps
    10 bps
    20 bps

The same cost rate is applied to buys and sells.

This generic grid is deliberately used instead of hard-coding broker fees /
stamp duty so it remains aligned with the precommitted research design.

OUTPUT
------
data/backtest_reports/
    backtest_period_returns.csv
    backtest_equity_curve.csv
    backtest_summary.csv
    backtest_yearly.csv
    backtest_execution_diagnostics.csv
    backtest_model_comparison.csv
    backtest_metadata.json
    09_executable_portfolio_backtest.log

RUN IN ANACONDA PROMPT
----------------------
python 09_executable_portfolio_backtest.py

Optional:
python 09_executable_portfolio_backtest.py --cost-bps 0,5,10,20
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import platform
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import xgboost as xgb
except ImportError as exc:
    raise RuntimeError(
        "Stage 9 requires XGBoost because the frozen Stage-8 model must be "
        "re-applied to the ex-ante trading universe.\n"
        "In Anaconda Prompt run:\n"
        "    pip install xgboost"
    ) from exc


# ======================================================================================
# Frozen specification
# ======================================================================================

REQUIRED_STAGE5_VERSION = "v4_final_positional"

OOS_START = "20190101"
OOS_END = "20251231"

FEATURE_NAMES = [
    "mom5",
    "mom20",
    "mom60",
    "vol5",
    "vol20",
    "vol60",
    "vol5_vol60",
    "volume_ratio20",
    "amount_ratio20",
    "range1",
    "range20",
]

FEATURE_COLUMNS = [
    f"{x}_z"
    for x in FEATURE_NAMES
]

P = len(FEATURE_COLUMNS)

TOP_FRACTIONS = (
    0.20,
    0.10,
)

DEFAULT_COST_BPS = (
    0.0,
    5.0,
    10.0,
    20.0,
)

REBALANCE_STEP_MARKET_DAYS = 5

INITIAL_NAV = 1.0

# Conservative exact-limit tolerance.
LIMIT_TOL = 1e-10

STAGE9_SPEC_VERSION = "v1_exante_weekly_self_financing"


# ======================================================================================
# Configuration
# ======================================================================================

@dataclass
class Config:
    data_root: Path
    cost_bps: Tuple[float, ...]
    nthread: int

    @property
    def signal_root(self) -> Path:
        return self.data_root / "processed" / "signals"

    @property
    def stage7_report_root(self) -> Path:
        return self.data_root / "linear_model_reports"

    @property
    def stage8_report_root(self) -> Path:
        return self.data_root / "xgb_model_reports"

    @property
    def stage8_cache_root(self) -> Path:
        return self.data_root / "cache" / "stage8_xgb"

    @property
    def report_root(self) -> Path:
        return self.data_root / "backtest_reports"


# ======================================================================================
# Logging / generic utilities
# ======================================================================================

def setup_logging(report_root: Path) -> logging.Logger:
    report_root.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("stage9_backtest")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(
        report_root / "09_executable_portfolio_backtest.log",
        mode="a",
        encoding="utf-8",
    )
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


def package_version(name: str) -> str:
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:
        return "unknown"


def normalize_date_series(s: pd.Series) -> pd.Series:
    return (
        s.astype("string")
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(8)
    )


def safe_float(x: Any) -> float:
    try:
        y = float(x)
    except Exception:
        return np.nan

    return y if np.isfinite(y) else np.nan


def atomic_write_json(
    obj: Dict[str, Any],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")

    if tmp.exists():
        tmp.unlink()

    with tmp.open("w", encoding="utf-8") as f:
        json.dump(
            obj,
            f,
            ensure_ascii=False,
            indent=2,
            default=str,
        )

    tmp.replace(path)


def parse_month_from_path(
    path: Path,
) -> Tuple[int, int, str]:
    m = re.search(
        r"signals_(\d{4})(\d{2})\.parquet$",
        path.name,
    )

    if not m:
        raise ValueError(
            f"Cannot parse month from {path}"
        )

    y = int(m.group(1))
    mo = int(m.group(2))

    return y, mo, f"{y:04d}-{mo:02d}"


def month_label_from_date(
    date_str: str,
) -> str:
    return (
        f"{date_str[:4]}-"
        f"{date_str[4:6]}"
    )


# ======================================================================================
# Stage-5 discovery and calendar
# ======================================================================================

def discover_signal_files(
    signal_root: Path,
) -> List[Path]:
    files = sorted(
        signal_root.glob(
            "*/*/signals_*.parquet"
        )
    )

    if not files:
        raise FileNotFoundError(
            f"No Stage-5 signal files found under {signal_root}"
        )

    return files


def oos_signal_files(
    files: Sequence[Path],
) -> List[Path]:
    out = []

    for p in files:
        y, m, _ = parse_month_from_path(p)
        ym = f"{y:04d}{m:02d}"

        if (
            ym >= OOS_START[:6]
            and ym <= OOS_END[:6]
        ):
            out.append(p)

    return out


def validate_stage5_version(
    path: Path,
) -> None:
    v = pd.read_parquet(
        path,
        columns=[
            "stage5_spec_version",
        ],
    )

    versions = (
        v["stage5_spec_version"]
        .dropna()
        .astype(str)
        .unique()
    )

    if (
        len(versions) != 1
        or versions[0]
        != REQUIRED_STAGE5_VERSION
    ):
        raise ValueError(
            f"{path}: expected Stage-5 version "
            f"{REQUIRED_STAGE5_VERSION!r}, got {versions.tolist()}"
        )


def build_oos_market_calendar(
    files: Sequence[Path],
    *,
    logger: logging.Logger,
) -> List[str]:
    dates: List[str] = []

    for i, path in enumerate(
        oos_signal_files(files),
        1,
    ):
        validate_stage5_version(path)

        x = pd.read_parquet(
            path,
            columns=[
                "trade_date",
            ],
        )

        d = normalize_date_series(
            x["trade_date"]
        )

        d = d.loc[
            (d >= OOS_START)
            & (d <= OOS_END)
        ]

        dates.extend(
            d.drop_duplicates().tolist()
        )

        logger.info(
            "CALENDAR %d | %s | dates=%d",
            i,
            path.name,
            d.nunique(),
        )

    dates = sorted(
        set(dates)
    )

    if not dates:
        raise RuntimeError(
            "No OOS market dates found."
        )

    return dates


@dataclass(frozen=True)
class RebalanceEvent:
    event_id: int
    signal_date: str
    execution_date: str
    scheduled_exit_date: str


def build_rebalance_schedule(
    market_dates: Sequence[str],
) -> List[RebalanceEvent]:
    """
    Anchor at first OOS market date.

    signal index j
    execution j+1
    scheduled exit j+6
    next signal j+5
    """
    dates = list(market_dates)

    events: List[RebalanceEvent] = []

    event_id = 0

    for j in range(
        0,
        len(dates),
        REBALANCE_STEP_MARKET_DAYS,
    ):
        if j + 6 >= len(dates):
            break

        events.append(
            RebalanceEvent(
                event_id=event_id,
                signal_date=dates[j],
                execution_date=dates[j + 1],
                scheduled_exit_date=dates[j + 6],
            )
        )

        event_id += 1

    if not events:
        raise RuntimeError(
            "No valid non-overlapping OOS rebalance events."
        )

    # Exact non-overlap identity:
    # next execution equals prior scheduled exit.
    for a, b in zip(
        events[:-1],
        events[1:],
    ):
        if (
            b.execution_date
            != a.scheduled_exit_date
        ):
            raise RuntimeError(
                "Rebalance schedule is not exactly non-overlapping: "
                f"{a} -> {b}"
            )

    return events


# ======================================================================================
# Ex-ante signal-universe snapshots and execution-market snapshots
# ======================================================================================

SIGNAL_READ_COLUMNS = [
    "ts_code",
    "trade_date",
    "liquid_universe_primary",
    *FEATURE_COLUMNS,
]

MARKET_READ_COLUMNS = [
    "ts_code",
    "trade_date",
    "open",
    "adj_open",
    "up_limit",
    "down_limit",
    "is_st",
]


@dataclass
class MarketRow:
    price: float
    buyable: bool
    sellable: bool
    is_st: bool


def infer_market_row(
    row: pd.Series,
) -> Optional[MarketRow]:
    raw_open = safe_float(
        row.get("open")
    )
    adj_open = safe_float(
        row.get("adj_open")
    )

    if not (
        np.isfinite(raw_open)
        and raw_open > 0
        and np.isfinite(adj_open)
        and adj_open > 0
    ):
        return None

    up = safe_float(
        row.get("up_limit")
    )
    down = safe_float(
        row.get("down_limit")
    )

    is_st_raw = row.get(
        "is_st",
        False,
    )

    try:
        is_st = bool(
            False
            if pd.isna(is_st_raw)
            else is_st_raw
        )
    except Exception:
        is_st = False

    at_up = (
        np.isfinite(up)
        and raw_open
        >= up - LIMIT_TOL
    )

    at_down = (
        np.isfinite(down)
        and raw_open
        <= down + LIMIT_TOL
    )

    buyable = (
        (not at_up)
        and (not is_st)
    )

    sellable = (
        not at_down
    )

    return MarketRow(
        price=float(adj_open),
        buyable=bool(buyable),
        sellable=bool(sellable),
        is_st=bool(is_st),
    )


def build_snapshots(
    files: Sequence[Path],
    events: Sequence[RebalanceEvent],
    *,
    logger: logging.Logger,
) -> Tuple[
    Dict[str, pd.DataFrame],
    Dict[str, Dict[str, MarketRow]],
]:
    signal_dates = {
        e.signal_date
        for e in events
    }

    execution_dates = {
        e.execution_date
        for e in events
    }

    execution_dates.add(
        events[-1].scheduled_exit_date
    )

    signal_snapshots: Dict[
        str,
        pd.DataFrame
    ] = {}

    market_snapshots: Dict[
        str,
        Dict[str, MarketRow]
    ] = {}

    for i, path in enumerate(
        oos_signal_files(files),
        1,
    ):
        # Read the union once.
        cols = sorted(
            set(
                SIGNAL_READ_COLUMNS
                + MARKET_READ_COLUMNS
            )
        )

        try:
            df = pd.read_parquet(
                path,
                columns=cols,
            )
        except Exception as exc:
            raise RuntimeError(
                f"{path} does not contain all Stage-9 required columns. "
                "Stage 9 expects current-day open/adj_open/limit/ST fields "
                "to be preserved in the frozen Stage-5 signal panels."
            ) from exc

        df["trade_date"] = (
            normalize_date_series(
                df["trade_date"]
            )
        )

        # --------------------------------------------------------------
        # Ex-ante signal universes.
        # --------------------------------------------------------------
        sub_sig = df.loc[
            df["trade_date"].isin(
                signal_dates
            )
        ].copy()

        if not sub_sig.empty:
            for d, g in sub_sig.groupby(
                "trade_date",
                sort=False,
            ):
                primary = (
                    g[
                        "liquid_universe_primary"
                    ]
                    .fillna(False)
                    .astype(bool)
                )

                x = g.loc[
                    primary,
                    [
                        "ts_code",
                        *FEATURE_COLUMNS,
                    ],
                ].copy()

                for c in FEATURE_COLUMNS:
                    x[c] = pd.to_numeric(
                        x[c],
                        errors="coerce",
                    )

                finite = np.ones(
                    len(x),
                    dtype=bool,
                )

                for c in FEATURE_COLUMNS:
                    finite &= np.isfinite(
                        x[c].to_numpy(
                            dtype=float
                        )
                    )

                x = x.loc[
                    finite
                ].copy()

                if x["ts_code"].duplicated().any():
                    raise RuntimeError(
                        f"{d}: duplicate ts_code in signal universe."
                    )

                x = x.sort_values(
                    "ts_code"
                ).reset_index(drop=True)

                signal_snapshots[
                    str(d)
                ] = x

        # --------------------------------------------------------------
        # Execution-open market snapshots.
        # Missing stock row = suspended / unavailable.
        # --------------------------------------------------------------
        sub_mkt = df.loc[
            df["trade_date"].isin(
                execution_dates
            )
        ].copy()

        if not sub_mkt.empty:
            for d, g in sub_mkt.groupby(
                "trade_date",
                sort=False,
            ):
                snap = (
                    market_snapshots
                    .setdefault(
                        str(d),
                        {},
                    )
                )

                for row in g.itertuples(
                    index=False,
                ):
                    s = pd.Series(
                        row._asdict()
                    )

                    mr = infer_market_row(
                        s
                    )

                    if mr is not None:
                        code = str(
                            s["ts_code"]
                        )

                        snap[
                            code
                        ] = mr

        logger.info(
            "SNAPSHOTS %d | %s | signal_dates=%d market_dates=%d",
            i,
            path.name,
            sub_sig[
                "trade_date"
            ].nunique()
            if not sub_sig.empty
            else 0,
            sub_mkt[
                "trade_date"
            ].nunique()
            if not sub_mkt.empty
            else 0,
        )

        del (
            df,
            sub_sig,
            sub_mkt,
        )
        gc.collect()

    missing_signal = sorted(
        signal_dates
        - set(
            signal_snapshots
        )
    )

    missing_market = sorted(
        execution_dates
        - set(
            market_snapshots
        )
    )

    if missing_signal:
        raise RuntimeError(
            f"Missing signal snapshots for dates: {missing_signal[:20]}"
        )

    # An execution date may have no rows only if the whole market is absent,
    # which should never happen.
    if missing_market:
        raise RuntimeError(
            f"Missing market snapshots for execution dates: {missing_market[:20]}"
        )

    return (
        signal_snapshots,
        market_snapshots,
    )


# ======================================================================================
# Stage-7 monthly linear models
# ======================================================================================

@dataclass
class LinearBetas:
    ridge: np.ndarray
    ols: np.ndarray


def load_linear_beta_path(
    cfg: Config,
) -> Dict[str, LinearBetas]:
    path = (
        cfg.stage7_report_root
        / "linear_model_coefficients.csv"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Missing Stage-7 coefficient path: {path}"
        )

    df = pd.read_csv(
        path
    )

    beta_cols = [
        f"beta_{x}"
        for x in FEATURE_NAMES
    ]

    required = [
        "prediction_month",
        "model",
        *beta_cols,
    ]

    missing = [
        c
        for c in required
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Stage-7 coefficient file missing columns: {missing}"
        )

    out: Dict[
        str,
        Dict[str, np.ndarray]
    ] = {}

    for row in df.itertuples(
        index=False,
    ):
        month = str(
            row.prediction_month
        )

        model = str(
            row.model
        ).upper()

        if model not in {
            "RIDGE",
            "OLS",
        }:
            continue

        beta = np.array(
            [
                float(
                    getattr(
                        row,
                        f"beta_{x}",
                    )
                )
                for x in FEATURE_NAMES
            ],
            dtype=float,
        )

        if not np.isfinite(
            beta
        ).all():
            raise RuntimeError(
                f"{month} {model}: nonfinite Stage-7 beta."
            )

        out.setdefault(
            month,
            {},
        )[model] = beta

    final: Dict[
        str,
        LinearBetas
    ] = {}

    for month, d in out.items():
        if not {
            "RIDGE",
            "OLS",
        }.issubset(
            d
        ):
            continue

        final[
            month
        ] = LinearBetas(
            ridge=d["RIDGE"],
            ols=d["OLS"],
        )

    return final


# ======================================================================================
# Stage-8 frozen XGBoost model
# ======================================================================================

@dataclass
class Stage8Cache:
    X: np.ndarray
    y: np.ndarray
    exit_date: np.ndarray


def load_stage8_metadata(
    cfg: Config,
) -> Dict[str, Any]:
    path = (
        cfg.stage8_report_root
        / "xgb_metadata.json"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Missing Stage-8 metadata: {path}"
        )

    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def open_stage8_cache(
    cfg: Config,
) -> Stage8Cache:
    X_path = (
        cfg.stage8_cache_root
        / "X.npy"
    )
    y_path = (
        cfg.stage8_cache_root
        / "y_xs.npy"
    )
    exit_path = (
        cfg.stage8_cache_root
        / "exit_date.npy"
    )

    for p in [
        X_path,
        y_path,
        exit_path,
    ]:
        if not p.exists():
            raise FileNotFoundError(
                f"Missing Stage-8 training cache file: {p}"
            )

    X = np.load(
        X_path,
        mmap_mode="r",
    )
    y = np.load(
        y_path,
        mmap_mode="r",
    )
    exit_date = np.load(
        exit_path,
        mmap_mode="r",
    )

    if (
        X.ndim != 2
        or X.shape[1] != P
        or y.shape[0] != X.shape[0]
        or exit_date.shape[0]
        != X.shape[0]
    ):
        raise RuntimeError(
            "Stage-8 training cache shape mismatch."
        )

    return Stage8Cache(
        X=X,
        y=y,
        exit_date=exit_date,
    )


def training_prefix_end(
    cache: Stage8Cache,
    cutoff_date: str,
) -> int:
    cutoff = int(
        cutoff_date
    )

    idx = int(
        np.searchsorted(
            cache.exit_date,
            cutoff,
            side="left",
        )
    )

    if idx <= P:
        raise RuntimeError(
            f"Insufficient purged training observations before {cutoff_date}."
        )

    if int(
        cache.exit_date[
            idx - 1
        ]
    ) >= cutoff:
        raise RuntimeError(
            "Stage-8 cache label purge check failed."
        )

    return idx


def build_frozen_xgb_params(
    meta: Mapping[str, Any],
    *,
    cfg: Config,
) -> Tuple[
    Dict[str, Any],
    int,
    str,
]:
    candidate = dict(
        meta[
            "selected_xgb_candidate"
        ]
    )

    fixed = dict(
        meta[
            "xgb_fixed_params"
        ]
    )

    rounds = int(
        meta[
            "num_boost_round"
        ]
    )

    params = {
        **fixed,
        "max_depth": int(
            candidate[
                "max_depth"
            ]
        ),
        "min_child_weight": float(
            candidate[
                "min_child_weight"
            ]
        ),
        "reg_lambda": float(
            candidate[
                "reg_lambda"
            ]
        ),
        "seed": int(
            meta.get(
                "random_seed",
                20260906,
            )
        ),
        "nthread": cfg.nthread,
        "verbosity": 0,
    }

    # Stage 8 was frozen on CPU by default. Reproduce metadata device if safely
    # available; otherwise CPU is deterministic and valid.
    device = str(
        meta.get(
            "device",
            "cpu",
        )
    )

    if (
        device != "cpu"
        and int(
            str(
                xgb.__version__
            ).split(".")[0]
        )
        >= 2
    ):
        params[
            "device"
        ] = device

    return (
        params,
        rounds,
        str(
            candidate[
                "candidate_id"
            ]
        ),
    )


def fit_xgb_for_year(
    *,
    cache: Stage8Cache,
    cutoff_date: str,
    params: Mapping[str, Any],
    rounds: int,
    cfg: Config,
) -> Tuple[
    xgb.Booster,
    int,
    str,
]:
    n_train = training_prefix_end(
        cache,
        cutoff_date,
    )

    last_exit = str(
        int(
            cache.exit_date[
                n_train - 1
            ]
        )
    )

    dtrain = xgb.DMatrix(
        cache.X[
            :n_train
        ],
        label=cache.y[
            :n_train
        ],
        feature_names=FEATURE_NAMES,
        nthread=cfg.nthread,
    )

    booster = xgb.train(
        params=dict(params),
        dtrain=dtrain,
        num_boost_round=rounds,
        verbose_eval=False,
    )

    return (
        booster,
        n_train,
        last_exit,
    )


# ======================================================================================
# Score all ex-ante weekly signal universes
# ======================================================================================

@dataclass
class ScoreSnapshot:
    signal_date: str
    universe_codes: List[str]
    scores: Dict[str, Dict[str, float]]
    n_universe: int
    xgb_train_n: int
    xgb_last_exit: str


def score_signal_snapshots(
    *,
    events: Sequence[RebalanceEvent],
    signal_snapshots: Mapping[str, pd.DataFrame],
    linear_betas: Mapping[str, LinearBetas],
    stage8_cache: Stage8Cache,
    stage8_meta: Mapping[str, Any],
    cfg: Config,
    logger: logging.Logger,
) -> Dict[str, ScoreSnapshot]:
    params, rounds, candidate_id = (
        build_frozen_xgb_params(
            stage8_meta,
            cfg=cfg,
        )
    )

    first_signal_by_year: Dict[
        int,
        str
    ] = {}

    for e in events:
        y = int(
            e.signal_date[:4]
        )
        first_signal_by_year.setdefault(
            y,
            e.signal_date,
        )

    boosters: Dict[
        int,
        xgb.Booster
    ] = {}
    train_info: Dict[
        int,
        Tuple[int, str]
    ] = {}

    for year in sorted(
        first_signal_by_year
    ):
        cutoff = first_signal_by_year[
            year
        ]

        booster, n_train, last_exit = (
            fit_xgb_for_year(
                cache=stage8_cache,
                cutoff_date=cutoff,
                params=params,
                rounds=rounds,
                cfg=cfg,
            )
        )

        boosters[
            year
        ] = booster

        train_info[
            year
        ] = (
            n_train,
            last_exit,
        )

        logger.info(
            "XGB REFIT | year=%d cutoff=%s train_n=%d last_exit=%s candidate=%s",
            year,
            cutoff,
            n_train,
            last_exit,
            candidate_id,
        )

    out: Dict[
        str,
        ScoreSnapshot
    ] = {}

    for i, e in enumerate(
        events,
        1,
    ):
        d = e.signal_date
        year = int(
            d[:4]
        )

        month = (
            f"{d[:4]}-"
            f"{d[4:6]}"
        )

        if month not in linear_betas:
            raise RuntimeError(
                f"No Stage-7 monthly beta for {month}"
            )

        x = signal_snapshots[
            d
        ]

        X = x[
            FEATURE_COLUMNS
        ].to_numpy(
            dtype=np.float32
        )

        codes = x[
            "ts_code"
        ].astype(
            str
        ).tolist()

        booster = boosters[
            year
        ]

        dtest = xgb.DMatrix(
            X,
            feature_names=FEATURE_NAMES,
            nthread=cfg.nthread,
        )

        score_xgb = booster.predict(
            dtest
        ).astype(
            float
        )

        betas = linear_betas[
            month
        ]

        X64 = X.astype(
            np.float64
        )

        score_ridge = (
            X64
            @ betas.ridge
        )

        score_ols = (
            X64
            @ betas.ols
        )

        if not (
            np.isfinite(
                score_xgb
            ).all()
            and np.isfinite(
                score_ridge
            ).all()
            and np.isfinite(
                score_ols
            ).all()
        ):
            raise RuntimeError(
                f"{d}: nonfinite ex-ante model score."
            )

        model_scores: Dict[
            str,
            Dict[str, float]
        ] = {
            "XGB": dict(
                zip(
                    codes,
                    score_xgb.tolist(),
                )
            ),
            "RIDGE": dict(
                zip(
                    codes,
                    score_ridge.tolist(),
                )
            ),
            "OLS": dict(
                zip(
                    codes,
                    score_ols.tolist(),
                )
            ),
        }

        n_train, last_exit = train_info[
            year
        ]

        out[
            d
        ] = ScoreSnapshot(
            signal_date=d,
            universe_codes=codes,
            scores=model_scores,
            n_universe=len(
                codes
            ),
            xgb_train_n=n_train,
            xgb_last_exit=last_exit,
        )

        logger.info(
            "SCORE %d/%d | %s | universe=%d",
            i,
            len(events),
            d,
            len(codes),
        )

    return out


# ======================================================================================
# Portfolio simulator
# ======================================================================================

@dataclass
class Position:
    shares: float
    last_price: float


@dataclass
class PortfolioState:
    cash: float = INITIAL_NAV
    positions: Dict[str, Position] = field(
        default_factory=dict
    )
    prev_posttrade_nav: Optional[float] = None


@dataclass
class RebalanceResult:
    pretrade_nav: float
    posttrade_nav: float
    trade_notional: float
    buy_notional: float
    sell_notional: float
    transaction_cost: float
    one_way_turnover: float
    holdings_count: int
    desired_count: int
    frozen_count: int
    frozen_weight: float
    cash_weight: float
    blocked_sell_count: int
    blocked_buy_count: int
    executable_fill_rate: float
    net_period_return: float
    gross_mark_to_market_return: float


def mark_portfolio(
    state: PortfolioState,
    market: Mapping[str, MarketRow],
) -> Tuple[
    Dict[str, float],
    float,
]:
    values: Dict[
        str,
        float
    ] = {}

    total = float(
        state.cash
    )

    for code, pos in list(
        state.positions.items()
    ):
        mr = market.get(
            code
        )

        if (
            mr is not None
            and np.isfinite(
                mr.price
            )
            and mr.price > 0
        ):
            pos.last_price = float(
                mr.price
            )

        value = (
            pos.shares
            * pos.last_price
        )

        if (
            not np.isfinite(
                value
            )
            or value < -1e-12
        ):
            raise RuntimeError(
                f"Invalid marked position value for {code}: {value}"
            )

        values[
            code
        ] = float(
            max(
                value,
                0.0,
            )
        )

        total += values[
            code
        ]

    return (
        values,
        float(total),
    )


def choose_desired_names(
    *,
    ranked_codes: Sequence[str],
    k: int,
    state: PortfolioState,
    market: Mapping[str, MarketRow],
) -> List[str]:
    desired: List[str] = []

    held = set(
        state.positions
    )

    for code in ranked_codes:
        if len(desired) >= k:
            break

        if code in held:
            desired.append(
                code
            )
            continue

        mr = market.get(
            code
        )

        if (
            mr is not None
            and mr.buyable
        ):
            desired.append(
                code
            )

    return desired


def rebalance_equal_weight(
    *,
    state: PortfolioState,
    desired_names: Sequence[str],
    market: Mapping[str, MarketRow],
    cost_bps: float,
) -> RebalanceResult:
    """
    Rebalance toward equal weights subject to execution constraints.

    Locked positions:
        - off-target positions that cannot be sold;
        - desired positions that cannot be adjusted in the required direction.

    The remaining NAV is equal-weighted across flexible desired holdings.
    """
    cost_rate = (
        float(cost_bps)
        / 10000.0
    )

    current_values, pre_nav = (
        mark_portfolio(
            state,
            market,
        )
    )

    if pre_nav <= 0:
        raise RuntimeError(
            f"Nonpositive pretrade NAV: {pre_nav}"
        )

    previous_post = (
        state.prev_posttrade_nav
    )

    gross_ret = (
        pre_nav
        / previous_post
        - 1.0
        if (
            previous_post is not None
            and previous_post > 0
        )
        else np.nan
    )

    desired_set = set(
        desired_names
    )

    trade_notional = 0.0
    buy_notional = 0.0
    sell_notional = 0.0
    total_cost = 0.0
    blocked_sell_count = 0
    blocked_buy_count = 0

    # ------------------------------------------------------------------
    # 1) Sell off-target positions whenever possible.
    # ------------------------------------------------------------------
    locked_values: Dict[
        str,
        float
    ] = {}

    for code in list(
        state.positions
    ):
        if code in desired_set:
            continue

        pos = state.positions[
            code
        ]

        mr = market.get(
            code
        )

        value = current_values.get(
            code,
            pos.shares
            * pos.last_price,
        )

        if (
            mr is not None
            and mr.sellable
        ):
            notional = float(
                value
            )

            cost = (
                notional
                * cost_rate
            )

            state.cash += (
                notional
                - cost
            )

            sell_notional += notional
            trade_notional += notional
            total_cost += cost

            del state.positions[
                code
            ]
            current_values.pop(
                code,
                None,
            )
        else:
            locked_values[
                code
            ] = float(
                value
            )
            blocked_sell_count += 1

    # ------------------------------------------------------------------
    # 2) Determine which desired holdings can flex to the equal-weight target.
    # ------------------------------------------------------------------
    desired = list(
        desired_names
    )

    active = set(
        desired
    )

    # Desired positions with no executable current price are locked.
    for code in list(
        active
    ):
        if code in state.positions:
            mr = market.get(
                code
            )

            if mr is None:
                value = current_values[
                    code
                ]
                locked_values[
                    code
                ] = value
                active.remove(
                    code
                )

    # Iteratively identify directionally constrained desired holdings.
    for _ in range(
        len(desired) + 2
    ):
        locked_total = float(
            sum(
                locked_values.values()
            )
        )

        residual_nav = max(
            pre_nav
            - locked_total
            - total_cost,
            0.0,
        )

        if not active:
            target_value = 0.0
            break

        target_value = (
            residual_nav
            / len(active)
        )

        newly_locked: List[
            str
        ] = []

        for code in list(
            active
        ):
            cur = current_values.get(
                code,
                0.0,
            )

            mr = market.get(
                code
            )

            # New positions were pre-screened buyable.
            if code not in state.positions:
                if (
                    mr is None
                    or not mr.buyable
                ):
                    locked_values[
                        code
                    ] = 0.0
                    newly_locked.append(
                        code
                    )
                    blocked_buy_count += 1
                continue

            if mr is None:
                locked_values[
                    code
                ] = cur
                newly_locked.append(
                    code
                )
                continue

            if (
                cur > target_value
                + 1e-15
                and not mr.sellable
            ):
                locked_values[
                    code
                ] = cur
                newly_locked.append(
                    code
                )
                blocked_sell_count += 1

            elif (
                cur < target_value
                - 1e-15
                and not mr.buyable
            ):
                locked_values[
                    code
                ] = cur
                newly_locked.append(
                    code
                )
                blocked_buy_count += 1

        if not newly_locked:
            break

        for code in newly_locked:
            active.discard(
                code
            )

    locked_total = float(
        sum(
            locked_values.values()
        )
    )

    residual_nav = max(
        pre_nav
        - locked_total
        - total_cost,
        0.0,
    )

    target_value = (
        residual_nav
        / len(active)
        if active
        else 0.0
    )

    # ------------------------------------------------------------------
    # 3) Active desired sells first.
    # ------------------------------------------------------------------
    for code in list(
        active
    ):
        if code not in state.positions:
            continue

        cur = current_values.get(
            code,
            0.0,
        )

        if cur <= target_value:
            continue

        mr = market.get(
            code
        )

        if (
            mr is None
            or not mr.sellable
        ):
            # Should have been caught by locking.
            blocked_sell_count += 1
            continue

        sell_value = float(
            cur
            - target_value
        )

        if sell_value <= 0:
            continue

        shares_to_sell = (
            sell_value
            / mr.price
        )

        pos = state.positions[
            code
        ]

        shares_to_sell = min(
            shares_to_sell,
            pos.shares,
        )

        actual_notional = (
            shares_to_sell
            * mr.price
        )

        cost = (
            actual_notional
            * cost_rate
        )

        pos.shares -= (
            shares_to_sell
        )

        state.cash += (
            actual_notional
            - cost
        )

        sell_notional += (
            actual_notional
        )
        trade_notional += (
            actual_notional
        )
        total_cost += cost

        if pos.shares <= 1e-16:
            del state.positions[
                code
            ]
            current_values[
                code
            ] = 0.0
        else:
            current_values[
                code
            ] = (
                pos.shares
                * mr.price
            )

    # ------------------------------------------------------------------
    # 4) Compute required active buys and scale to available cash.
    # ------------------------------------------------------------------
    buy_requests: Dict[
        str,
        float
    ] = {}

    for code in active:
        cur = current_values.get(
            code,
            0.0,
        )

        if cur >= target_value:
            continue

        mr = market.get(
            code
        )

        if (
            mr is None
            or not mr.buyable
        ):
            blocked_buy_count += 1
            continue

        req = float(
            target_value
            - cur
        )

        if req > 0:
            buy_requests[
                code
            ] = req

    total_requested = float(
        sum(
            buy_requests.values()
        )
    )

    if total_requested > 0:
        max_affordable_notional = (
            state.cash
            / (
                1.0
                + cost_rate
            )
        )

        scale = min(
            1.0,
            max_affordable_notional
            / total_requested
            if total_requested > 0
            else 1.0,
        )

        for code, req in buy_requests.items():
            mr = market[
                code
            ]

            notional = float(
                req
                * scale
            )

            if notional <= 0:
                continue

            cost = (
                notional
                * cost_rate
            )

            cash_needed = (
                notional
                + cost
            )

            if (
                cash_needed
                > state.cash
                + 1e-14
            ):
                cash_needed = state.cash
                notional = (
                    cash_needed
                    / (
                        1.0
                        + cost_rate
                    )
                )
                cost = (
                    notional
                    * cost_rate
                )

            shares = (
                notional
                / mr.price
            )

            if code in state.positions:
                pos = state.positions[
                    code
                ]
                pos.shares += shares
                pos.last_price = mr.price
            else:
                state.positions[
                    code
                ] = Position(
                    shares=shares,
                    last_price=mr.price,
                )

            state.cash -= (
                notional
                + cost
            )

            buy_notional += notional
            trade_notional += notional
            total_cost += cost

    # Numerical cash guard.
    if (
        state.cash < -1e-10
    ):
        raise RuntimeError(
            f"Negative cash after rebalance: {state.cash}"
        )

    if state.cash < 0:
        state.cash = 0.0

    # ------------------------------------------------------------------
    # 5) Posttrade accounting.
    # ------------------------------------------------------------------
    _, post_nav = mark_portfolio(
        state,
        market,
    )

    identity_error = (
        post_nav
        - (
            pre_nav
            - total_cost
        )
    )

    if abs(
        identity_error
    ) > max(
        1e-10,
        1e-8
        * pre_nav,
    ):
        raise RuntimeError(
            "Self-financing identity failed: "
            f"post={post_nav}, pre={pre_nav}, cost={total_cost}, "
            f"error={identity_error}"
        )

    turnover = (
        trade_notional
        / pre_nav
    )

    frozen_value = float(
        sum(
            locked_values.values()
        )
    )

    frozen_weight = (
        frozen_value
        / pre_nav
    )

    cash_weight = (
        state.cash
        / post_nav
        if post_nav > 0
        else np.nan
    )

    fill_rate = (
        len(
            desired_set
            & set(
                state.positions
            )
        )
        / len(
            desired_set
        )
        if desired_set
        else np.nan
    )

    net_ret = (
        post_nav
        / previous_post
        - 1.0
        if (
            previous_post is not None
            and previous_post > 0
        )
        else np.nan
    )

    state.prev_posttrade_nav = (
        post_nav
    )

    return RebalanceResult(
        pretrade_nav=float(
            pre_nav
        ),
        posttrade_nav=float(
            post_nav
        ),
        trade_notional=float(
            trade_notional
        ),
        buy_notional=float(
            buy_notional
        ),
        sell_notional=float(
            sell_notional
        ),
        transaction_cost=float(
            total_cost
        ),
        one_way_turnover=float(
            turnover
        ),
        holdings_count=len(
            state.positions
        ),
        desired_count=len(
            desired_set
        ),
        frozen_count=len(
            locked_values
        ),
        frozen_weight=float(
            frozen_weight
        ),
        cash_weight=float(
            cash_weight
        ),
        blocked_sell_count=int(
            blocked_sell_count
        ),
        blocked_buy_count=int(
            blocked_buy_count
        ),
        executable_fill_rate=float(
            fill_rate
        )
        if np.isfinite(
            fill_rate
        )
        else np.nan,
        net_period_return=float(
            net_ret
        )
        if np.isfinite(
            net_ret
        )
        else np.nan,
        gross_mark_to_market_return=float(
            gross_ret
        )
        if np.isfinite(
            gross_ret
        )
        else np.nan,
    )


# ======================================================================================
# Strategy definitions
# ======================================================================================

@dataclass(frozen=True)
class StrategySpec:
    strategy: str
    model_key: str
    top_fraction: Optional[float]


def strategy_specs() -> List[StrategySpec]:
    out = [
        StrategySpec(
            strategy="UNIVERSE_EW",
            model_key="BENCHMARK",
            top_fraction=None,
        )
    ]

    for model_key in [
        "XGB",
        "RIDGE",
        "OLS",
    ]:
        for frac in TOP_FRACTIONS:
            pct = int(
                round(
                    frac
                    * 100
                )
            )

            out.append(
                StrategySpec(
                    strategy=(
                        f"{model_key}_TOP{pct}"
                    ),
                    model_key=model_key,
                    top_fraction=frac,
                )
            )

    return out


def ranked_codes_for_strategy(
    *,
    spec: StrategySpec,
    snapshot: ScoreSnapshot,
) -> Tuple[
    List[str],
    int,
]:
    if spec.model_key == "BENCHMARK":
        codes = sorted(
            snapshot.universe_codes
        )

        return (
            codes,
            len(codes),
        )

    scores = snapshot.scores[
        spec.model_key
    ]

    ranked = sorted(
        snapshot.universe_codes,
        key=lambda c: (
            -scores[c],
            c,
        ),
    )

    if spec.top_fraction is None:
        k = len(
            ranked
        )
    else:
        k = max(
            1,
            int(
                math.ceil(
                    spec.top_fraction
                    * len(
                        ranked
                    )
                )
            ),
        )

    return (
        ranked,
        k,
    )


# ======================================================================================
# Backtest engine
# ======================================================================================

def run_one_strategy_cost(
    *,
    spec: StrategySpec,
    cost_bps: float,
    events: Sequence[RebalanceEvent],
    score_snapshots: Mapping[str, ScoreSnapshot],
    market_snapshots: Mapping[str, Mapping[str, MarketRow]],
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    state = PortfolioState()

    rows: List[
        Dict[str, Any]
    ] = []

    equity_rows: List[
        Dict[str, Any]
    ] = []

    for e in events:
        score = score_snapshots[
            e.signal_date
        ]

        market = market_snapshots[
            e.execution_date
        ]

        ranked, k = ranked_codes_for_strategy(
            spec=spec,
            snapshot=score,
        )

        desired = choose_desired_names(
            ranked_codes=ranked,
            k=k,
            state=state,
            market=market,
        )

        result = rebalance_equal_weight(
            state=state,
            desired_names=desired,
            market=market,
            cost_bps=cost_bps,
        )

        rows.append({
            "strategy": spec.strategy,
            "cost_bps": float(
                cost_bps
            ),
            "event_id": int(
                e.event_id
            ),
            "signal_date": e.signal_date,
            "execution_date": e.execution_date,
            "scheduled_exit_date": e.scheduled_exit_date,
            "signal_universe_n": int(
                score.n_universe
            ),
            "target_k": int(
                k
            ),
            "desired_count_after_entry_filter": int(
                len(
                    desired
                )
            ),
            "xgb_train_n": int(
                score.xgb_train_n
            ),
            "xgb_last_training_exit_date": (
                score.xgb_last_exit
            ),
            **result.__dict__,
        })

        equity_rows.append({
            "strategy": spec.strategy,
            "cost_bps": float(
                cost_bps
            ),
            "date": e.execution_date,
            "nav": float(
                result.posttrade_nav
            ),
            "pretrade_nav": float(
                result.pretrade_nav
            ),
            "period_net_return": (
                result.net_period_return
            ),
            "period_gross_mark_to_market_return": (
                result.gross_mark_to_market_return
            ),
        })

    # ------------------------------------------------------------------
    # Final scheduled exit / liquidation attempt.
    # ------------------------------------------------------------------
    final_date = (
        events[-1]
        .scheduled_exit_date
    )

    market = market_snapshots[
        final_date
    ]

    # Desired set empty: sell everything that can be sold, carry blocked names.
    final_result = rebalance_equal_weight(
        state=state,
        desired_names=[],
        market=market,
        cost_bps=cost_bps,
    )

    rows.append({
        "strategy": spec.strategy,
        "cost_bps": float(
            cost_bps
        ),
        "event_id": int(
            len(
                events
            )
        ),
        "signal_date": "",
        "execution_date": final_date,
        "scheduled_exit_date": final_date,
        "signal_universe_n": np.nan,
        "target_k": 0,
        "desired_count_after_entry_filter": 0,
        "xgb_train_n": np.nan,
        "xgb_last_training_exit_date": "",
        "is_final_liquidation": True,
        **final_result.__dict__,
    })

    equity_rows.append({
        "strategy": spec.strategy,
        "cost_bps": float(
            cost_bps
        ),
        "date": final_date,
        "nav": float(
            final_result.posttrade_nav
        ),
        "pretrade_nav": float(
            final_result.pretrade_nav
        ),
        "period_net_return": (
            final_result.net_period_return
        ),
        "period_gross_mark_to_market_return": (
            final_result.gross_mark_to_market_return
        ),
    })

    period = pd.DataFrame(
        rows
    )

    if "is_final_liquidation" not in period.columns:
        period[
            "is_final_liquidation"
        ] = False

    period[
        "is_final_liquidation"
    ] = (
        period[
            "is_final_liquidation"
        ]
        .fillna(False)
        .astype(bool)
    )

    equity = pd.DataFrame(
        equity_rows
    )

    return (
        period,
        equity,
    )


# ======================================================================================
# Performance statistics
# ======================================================================================

def max_drawdown(
    nav: np.ndarray,
) -> float:
    nav = np.asarray(
        nav,
        dtype=float,
    )

    nav = nav[
        np.isfinite(
            nav
        )
    ]

    if len(nav) == 0:
        return np.nan

    peak = np.maximum.accumulate(
        nav
    )

    dd = (
        nav
        / peak
        - 1.0
    )

    return float(
        np.min(
            dd
        )
    )


def cagr_from_nav(
    dates: Sequence[str],
    nav: Sequence[float],
) -> float:
    if len(dates) < 2:
        return np.nan

    d0 = pd.Timestamp(
        str(
            dates[0]
        )
    )
    d1 = pd.Timestamp(
        str(
            dates[-1]
        )
    )

    years = (
        (d1 - d0).days
        / 365.25
    )

    if years <= 0:
        return np.nan

    nav0 = float(
        nav[0]
    )
    nav1 = float(
        nav[-1]
    )

    if (
        nav0 <= 0
        or nav1 <= 0
    ):
        return np.nan

    return float(
        (
            nav1
            / nav0
        )
        ** (
            1.0
            / years
        )
        - 1.0
    )


def build_summary(
    period: pd.DataFrame,
    equity: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    periods_per_year = (
        252.0
        / REBALANCE_STEP_MARKET_DAYS
    )

    for (
        strategy,
        cost_bps,
    ), g in period.groupby(
        [
            "strategy",
            "cost_bps",
        ],
        sort=False,
    ):
        e = equity.loc[
            (
                equity[
                    "strategy"
                ]
                == strategy
            )
            & (
                equity[
                    "cost_bps"
                ]
                == cost_bps
            )
        ].sort_values(
            "date"
        )

        rets = (
            pd.to_numeric(
                e[
                    "period_net_return"
                ],
                errors="coerce",
            )
            .dropna()
            .to_numpy(
                dtype=float
            )
        )

        mean_r = (
            float(
                np.mean(
                    rets
                )
            )
            if len(
                rets
            )
            else np.nan
        )

        sd_r = (
            float(
                np.std(
                    rets,
                    ddof=1,
                )
            )
            if len(
                rets
            )
            > 1
            else np.nan
        )

        ann_vol = (
            sd_r
            * math.sqrt(
                periods_per_year
            )
            if np.isfinite(
                sd_r
            )
            else np.nan
        )

        sharpe = (
            mean_r
            / sd_r
            * math.sqrt(
                periods_per_year
            )
            if (
                np.isfinite(
                    mean_r
                )
                and np.isfinite(
                    sd_r
                )
                and sd_r > 0
            )
            else np.nan
        )

        nav = pd.to_numeric(
            e["nav"],
            errors="coerce",
        ).to_numpy(
            dtype=float
        )

        dates = e[
            "date"
        ].astype(
            str
        ).tolist()

        final_row = g.loc[
            g[
                "is_final_liquidation"
            ]
        ]

        final_unliq_weight = (
            safe_float(
                final_row.iloc[-1][
                    "frozen_weight"
                ]
            )
            if not final_row.empty
            else np.nan
        )

        nonfinal = g.loc[
            ~g[
                "is_final_liquidation"
            ]
        ]

        rows.append({
            "strategy": strategy,
            "cost_bps": float(
                cost_bps
            ),
            "n_rebalances": int(
                len(
                    nonfinal
                )
            ),
            "start_date": (
                dates[0]
                if dates
                else ""
            ),
            "end_date": (
                dates[-1]
                if dates
                else ""
            ),
            "final_nav": (
                float(
                    nav[-1]
                )
                if len(
                    nav
                )
                else np.nan
            ),
            "cagr": cagr_from_nav(
                dates,
                nav,
            ),
            "annualized_volatility": ann_vol,
            "sharpe_zero_rf": sharpe,
            "max_drawdown": max_drawdown(
                nav
            ),
            "mean_period_net_return": mean_r,
            "mean_period_net_return_bps": (
                mean_r
                * 10000.0
                if np.isfinite(
                    mean_r
                )
                else np.nan
            ),
            "positive_period_share": (
                float(
                    np.mean(
                        rets > 0
                    )
                )
                if len(
                    rets
                )
                else np.nan
            ),
            "mean_one_way_turnover": safe_float(
                pd.to_numeric(
                    nonfinal[
                        "one_way_turnover"
                    ],
                    errors="coerce",
                ).mean()
            ),
            "median_one_way_turnover": safe_float(
                pd.to_numeric(
                    nonfinal[
                        "one_way_turnover"
                    ],
                    errors="coerce",
                ).median()
            ),
            "mean_holdings_count": safe_float(
                pd.to_numeric(
                    nonfinal[
                        "holdings_count"
                    ],
                    errors="coerce",
                ).mean()
            ),
            "mean_cash_weight": safe_float(
                pd.to_numeric(
                    nonfinal[
                        "cash_weight"
                    ],
                    errors="coerce",
                ).mean()
            ),
            "mean_frozen_weight": safe_float(
                pd.to_numeric(
                    nonfinal[
                        "frozen_weight"
                    ],
                    errors="coerce",
                ).mean()
            ),
            "mean_executable_fill_rate": safe_float(
                pd.to_numeric(
                    nonfinal[
                        "executable_fill_rate"
                    ],
                    errors="coerce",
                ).mean()
            ),
            "total_transaction_cost_nav": safe_float(
                pd.to_numeric(
                    g[
                        "transaction_cost"
                    ],
                    errors="coerce",
                ).sum()
            ),
            "blocked_sell_events": int(
                pd.to_numeric(
                    nonfinal[
                        "blocked_sell_count"
                    ],
                    errors="coerce",
                )
                .fillna(0)
                .sum()
            ),
            "blocked_buy_events": int(
                pd.to_numeric(
                    nonfinal[
                        "blocked_buy_count"
                    ],
                    errors="coerce",
                )
                .fillna(0)
                .sum()
            ),
            "final_unliquidated_weight": (
                final_unliq_weight
            ),
        })

    return pd.DataFrame(
        rows
    )


def build_yearly(
    equity: pd.DataFrame,
) -> pd.DataFrame:
    x = equity.copy()

    x["year"] = (
        x["date"]
        .astype(
            str
        )
        .str[:4]
        .astype(
            int
        )
    )

    rows = []

    for (
        strategy,
        cost_bps,
        year,
    ), g in x.groupby(
        [
            "strategy",
            "cost_bps",
            "year",
        ],
        sort=True,
    ):
        g = g.sort_values(
            "date"
        )

        rets = (
            pd.to_numeric(
                g[
                    "period_net_return"
                ],
                errors="coerce",
            )
            .dropna()
            .to_numpy(
                dtype=float
            )
        )

        if len(
            rets
        ):
            compounded = float(
                np.prod(
                    1.0
                    + rets
                )
                - 1.0
            )

            sd = (
                float(
                    np.std(
                        rets,
                        ddof=1,
                    )
                )
                if len(
                    rets
                )
                > 1
                else np.nan
            )

            mean_r = float(
                np.mean(
                    rets
                )
            )

            sharpe = (
                mean_r
                / sd
                * math.sqrt(
                    252.0
                    / REBALANCE_STEP_MARKET_DAYS
                )
                if (
                    np.isfinite(
                        sd
                    )
                    and sd > 0
                )
                else np.nan
            )
        else:
            compounded = np.nan
            sharpe = np.nan

        rows.append({
            "strategy": strategy,
            "cost_bps": float(
                cost_bps
            ),
            "year": int(
                year
            ),
            "n_periods": int(
                len(
                    rets
                )
            ),
            "year_return": compounded,
            "year_return_bps": (
                compounded
                * 10000.0
                if np.isfinite(
                    compounded
                )
                else np.nan
            ),
            "year_sharpe_zero_rf": sharpe,
        })

    return pd.DataFrame(
        rows
    )


def build_execution_diagnostics(
    period: pd.DataFrame,
) -> pd.DataFrame:
    nonfinal = period.loc[
        ~period[
            "is_final_liquidation"
        ]
    ].copy()

    rows = []

    for (
        strategy,
        cost_bps,
    ), g in nonfinal.groupby(
        [
            "strategy",
            "cost_bps",
        ],
        sort=False,
    ):
        rows.append({
            "strategy": strategy,
            "cost_bps": float(
                cost_bps
            ),
            "n_rebalances": int(
                len(
                    g
                )
            ),
            "mean_signal_universe_n": safe_float(
                pd.to_numeric(
                    g[
                        "signal_universe_n"
                    ],
                    errors="coerce",
                ).mean()
            ),
            "mean_target_k": safe_float(
                pd.to_numeric(
                    g[
                        "target_k"
                    ],
                    errors="coerce",
                ).mean()
            ),
            "mean_desired_count_after_entry_filter": safe_float(
                pd.to_numeric(
                    g[
                        "desired_count_after_entry_filter"
                    ],
                    errors="coerce",
                ).mean()
            ),
            "mean_fill_rate": safe_float(
                pd.to_numeric(
                    g[
                        "executable_fill_rate"
                    ],
                    errors="coerce",
                ).mean()
            ),
            "min_fill_rate": safe_float(
                pd.to_numeric(
                    g[
                        "executable_fill_rate"
                    ],
                    errors="coerce",
                ).min()
            ),
            "mean_frozen_weight": safe_float(
                pd.to_numeric(
                    g[
                        "frozen_weight"
                    ],
                    errors="coerce",
                ).mean()
            ),
            "max_frozen_weight": safe_float(
                pd.to_numeric(
                    g[
                        "frozen_weight"
                    ],
                    errors="coerce",
                ).max()
            ),
            "mean_cash_weight": safe_float(
                pd.to_numeric(
                    g[
                        "cash_weight"
                    ],
                    errors="coerce",
                ).mean()
            ),
            "blocked_sell_count_total": int(
                pd.to_numeric(
                    g[
                        "blocked_sell_count"
                    ],
                    errors="coerce",
                )
                .fillna(0)
                .sum()
            ),
            "blocked_buy_count_total": int(
                pd.to_numeric(
                    g[
                        "blocked_buy_count"
                    ],
                    errors="coerce",
                )
                .fillna(0)
                .sum()
            ),
            "mean_one_way_turnover": safe_float(
                pd.to_numeric(
                    g[
                        "one_way_turnover"
                    ],
                    errors="coerce",
                ).mean()
            ),
        })

    return pd.DataFrame(
        rows
    )


def build_model_comparison(
    summary: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for frac in TOP_FRACTIONS:
        pct = int(
            round(
                100
                * frac
            )
        )

        xgb_name = (
            f"XGB_TOP{pct}"
        )

        ridge_name = (
            f"RIDGE_TOP{pct}"
        )

        ols_name = (
            f"OLS_TOP{pct}"
        )

        for cost_bps in sorted(
            summary[
                "cost_bps"
            ].unique()
        ):
            s = summary.loc[
                summary[
                    "cost_bps"
                ]
                == cost_bps
            ].set_index(
                "strategy"
            )

            if (
                xgb_name not in s.index
                or ridge_name not in s.index
            ):
                continue

            x = s.loc[
                xgb_name
            ]
            r = s.loc[
                ridge_name
            ]

            row = {
                "top_fraction": float(
                    frac
                ),
                "cost_bps": float(
                    cost_bps
                ),
                "xgb_strategy": xgb_name,
                "ridge_strategy": ridge_name,
                "xgb_cagr": x[
                    "cagr"
                ],
                "ridge_cagr": r[
                    "cagr"
                ],
                "delta_cagr_xgb_minus_ridge": (
                    x[
                        "cagr"
                    ]
                    - r[
                        "cagr"
                    ]
                ),
                "xgb_sharpe": x[
                    "sharpe_zero_rf"
                ],
                "ridge_sharpe": r[
                    "sharpe_zero_rf"
                ],
                "delta_sharpe_xgb_minus_ridge": (
                    x[
                        "sharpe_zero_rf"
                    ]
                    - r[
                        "sharpe_zero_rf"
                    ]
                ),
                "xgb_max_drawdown": x[
                    "max_drawdown"
                ],
                "ridge_max_drawdown": r[
                    "max_drawdown"
                ],
                "xgb_mean_turnover": x[
                    "mean_one_way_turnover"
                ],
                "ridge_mean_turnover": r[
                    "mean_one_way_turnover"
                ],
            }

            if ols_name in s.index:
                o = s.loc[
                    ols_name
                ]

                row.update({
                    "ols_strategy": ols_name,
                    "ols_cagr": o[
                        "cagr"
                    ],
                    "ols_sharpe": o[
                        "sharpe_zero_rf"
                    ],
                })

            if "UNIVERSE_EW" in s.index:
                b = s.loc[
                    "UNIVERSE_EW"
                ]

                row.update({
                    "benchmark_cagr": b[
                        "cagr"
                    ],
                    "xgb_cagr_minus_benchmark": (
                        x[
                            "cagr"
                        ]
                        - b[
                            "cagr"
                        ]
                    ),
                    "ridge_cagr_minus_benchmark": (
                        r[
                            "cagr"
                        ]
                        - b[
                            "cagr"
                        ]
                    ),
                })

            rows.append(
                row
            )

    return pd.DataFrame(
        rows
    )


# ======================================================================================
# QA
# ======================================================================================

def final_qa(
    *,
    events: Sequence[RebalanceEvent],
    score_snapshots: Mapping[str, ScoreSnapshot],
    period: pd.DataFrame,
    equity: pd.DataFrame,
    summary: pd.DataFrame,
) -> List[str]:
    issues: List[str] = []

    if not events:
        issues.append(
            "No rebalance events."
        )
        return issues

    # Ex-ante XGB purge.
    for d, s in score_snapshots.items():
        if int(
            s.xgb_last_exit
        ) >= int(
            d
        ):
            issues.append(
                f"{d}: XGB training label purge violation."
            )

    # Signal < execution < scheduled exit.
    for e in events:
        if not (
            int(
                e.signal_date
            )
            < int(
                e.execution_date
            )
            < int(
                e.scheduled_exit_date
            )
        ):
            issues.append(
                f"Invalid event chronology: {e}"
            )

    if period.empty:
        issues.append(
            "Backtest period output is empty."
        )
        return issues

    if equity.empty:
        issues.append(
            "Backtest equity output is empty."
        )

    # NAV must be positive / finite.
    nav = pd.to_numeric(
        equity[
            "nav"
        ],
        errors="coerce",
    ).to_numpy(
        dtype=float
    )

    if not (
        np.isfinite(
            nav
        ).all()
        and (
            nav > 0
        ).all()
    ):
        issues.append(
            "Equity curve contains nonfinite/nonpositive NAV."
        )

    # Self-financing outputs must have nonnegative costs and turnover.
    if (
        pd.to_numeric(
            period[
                "transaction_cost"
            ],
            errors="coerce",
        )
        < -1e-14
    ).any():
        issues.append(
            "Negative transaction cost found."
        )

    if (
        pd.to_numeric(
            period[
                "one_way_turnover"
            ],
            errors="coerce",
        )
        < -1e-14
    ).any():
        issues.append(
            "Negative turnover found."
        )

    # Expected strategy count.
    expected = {
        x.strategy
        for x in strategy_specs()
    }

    found = set(
        summary[
            "strategy"
        ].unique()
    )

    if found != expected:
        issues.append(
            f"Strategy set mismatch. expected={sorted(expected)} found={sorted(found)}"
        )

    # Every strategy x cost should exist exactly once.
    expected_rows = (
        len(
            expected
        )
        * summary[
            "cost_bps"
        ].nunique()
    )

    if len(
        summary
    ) != expected_rows:
        issues.append(
            f"Unexpected summary row count {len(summary)} vs {expected_rows}."
        )

    # Costs should not mechanically improve final NAV by a meaningful amount
    # for the same strategy.
    for strategy, g in summary.groupby(
        "strategy"
    ):
        g = g.sort_values(
            "cost_bps"
        )

        final_nav = pd.to_numeric(
            g[
                "final_nav"
            ],
            errors="coerce",
        ).to_numpy(
            dtype=float
        )

        if len(
            final_nav
        ) > 1:
            increases = np.diff(
                final_nav
            )

            if np.any(
                increases > 1e-8
            ):
                issues.append(
                    f"{strategy}: final NAV increases materially with higher costs."
                )

    return issues


# ======================================================================================
# CLI
# ======================================================================================

def parse_cost_bps(
    raw: str,
) -> Tuple[float, ...]:
    vals = []

    for item in raw.split(","):
        item = item.strip()

        if not item:
            continue

        x = float(
            item
        )

        if x < 0:
            raise ValueError(
                "Transaction-cost bps cannot be negative."
            )

        vals.append(
            x
        )

    if not vals:
        raise ValueError(
            "Empty transaction-cost grid."
        )

    return tuple(
        sorted(
            set(
                vals
            )
        )
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Stage 9 executable weekly long-only A-share portfolio backtest."
        )
    )

    p.add_argument(
        "--data-root",
        default="data",
    )

    p.add_argument(
        "--cost-bps",
        default=",".join(
            str(
                x
            )
            for x in DEFAULT_COST_BPS
        ),
    )

    p.add_argument(
        "--nthread",
        type=int,
        default=-1,
    )

    return p.parse_args()


# ======================================================================================
# Main
# ======================================================================================

def main() -> int:
    args = parse_args()

    cfg = Config(
        data_root=Path(
            args.data_root
        ),
        cost_bps=parse_cost_bps(
            args.cost_bps
        ),
        nthread=int(
            args.nthread
        ),
    )

    cfg.report_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    logger = setup_logging(
        cfg.report_root
    )

    logger.info("=" * 116)
    logger.info("STAGE 9 | EXECUTABLE WALK-FORWARD PORTFOLIO BACKTEST")
    logger.info(
        "OOS=%s..%s | rebalance every %d market days | costs=%s bps",
        OOS_START,
        OOS_END,
        REBALANCE_STEP_MARKET_DAYS,
        ", ".join(
            f"{x:g}"
            for x in cfg.cost_bps
        ),
    )
    logger.info(
        "Selection universe is EX-ANTE: Primary Universe + finite frozen features only. "
        "Future target/exit availability is never used for ranking."
    )
    logger.info(
        "Execution: next open; limit-up no buy; limit-down no sell; current ST no new buy; "
        "blocked sells are carried."
    )
    logger.info("=" * 116)

    try:
        files = discover_signal_files(
            cfg.signal_root
        )

        market_dates = build_oos_market_calendar(
            files,
            logger=logger,
        )

        events = build_rebalance_schedule(
            market_dates
        )

        logger.info(
            "Rebalance events=%d | first signal=%s first execution=%s | "
            "last signal=%s final scheduled exit=%s",
            len(events),
            events[0].signal_date,
            events[0].execution_date,
            events[-1].signal_date,
            events[-1].scheduled_exit_date,
        )

        # ------------------------------------------------------------------
        # Ex-ante signal and execution-open market snapshots.
        # ------------------------------------------------------------------
        (
            signal_snapshots,
            market_snapshots,
        ) = build_snapshots(
            files,
            events,
            logger=logger,
        )

        # ------------------------------------------------------------------
        # Frozen Stage-7 / Stage-8 model objects.
        # ------------------------------------------------------------------
        linear_betas = load_linear_beta_path(
            cfg
        )

        stage8_meta = load_stage8_metadata(
            cfg
        )

        stage8_cache = open_stage8_cache(
            cfg
        )

        score_snapshots = score_signal_snapshots(
            events=events,
            signal_snapshots=signal_snapshots,
            linear_betas=linear_betas,
            stage8_cache=stage8_cache,
            stage8_meta=stage8_meta,
            cfg=cfg,
            logger=logger,
        )

        # ------------------------------------------------------------------
        # Simulate all strategy / cost combinations.
        # ------------------------------------------------------------------
        all_period: List[
            pd.DataFrame
        ] = []

        all_equity: List[
            pd.DataFrame
        ] = []

        specs = strategy_specs()

        total_runs = (
            len(
                specs
            )
            * len(
                cfg.cost_bps
            )
        )

        run_id = 0

        for spec in specs:
            for cost_bps in cfg.cost_bps:
                run_id += 1

                logger.info(
                    "BACKTEST %d/%d | %s | cost=%.1f bps",
                    run_id,
                    total_runs,
                    spec.strategy,
                    cost_bps,
                )

                p, e = run_one_strategy_cost(
                    spec=spec,
                    cost_bps=cost_bps,
                    events=events,
                    score_snapshots=score_snapshots,
                    market_snapshots=market_snapshots,
                )

                all_period.append(
                    p
                )
                all_equity.append(
                    e
                )

        period = pd.concat(
            all_period,
            ignore_index=True,
            sort=False,
        )

        equity = pd.concat(
            all_equity,
            ignore_index=True,
            sort=False,
        )

        period = period.sort_values(
            [
                "strategy",
                "cost_bps",
                "execution_date",
            ]
        ).reset_index(drop=True)

        equity = equity.sort_values(
            [
                "strategy",
                "cost_bps",
                "date",
            ]
        ).reset_index(drop=True)

        summary = build_summary(
            period,
            equity,
        )

        yearly = build_yearly(
            equity
        )

        execution = build_execution_diagnostics(
            period
        )

        comparison = build_model_comparison(
            summary
        )

        issues = final_qa(
            events=events,
            score_snapshots=score_snapshots,
            period=period,
            equity=equity,
            summary=summary,
        )

        # ------------------------------------------------------------------
        # Save reports.
        # ------------------------------------------------------------------
        period_path = (
            cfg.report_root
            / "backtest_period_returns.csv"
        )
        equity_path = (
            cfg.report_root
            / "backtest_equity_curve.csv"
        )
        summary_path = (
            cfg.report_root
            / "backtest_summary.csv"
        )
        yearly_path = (
            cfg.report_root
            / "backtest_yearly.csv"
        )
        execution_path = (
            cfg.report_root
            / "backtest_execution_diagnostics.csv"
        )
        comparison_path = (
            cfg.report_root
            / "backtest_model_comparison.csv"
        )

        period.to_csv(
            period_path,
            index=False,
            encoding="utf-8-sig",
        )
        equity.to_csv(
            equity_path,
            index=False,
            encoding="utf-8-sig",
        )
        summary.to_csv(
            summary_path,
            index=False,
            encoding="utf-8-sig",
        )
        yearly.to_csv(
            yearly_path,
            index=False,
            encoding="utf-8-sig",
        )
        execution.to_csv(
            execution_path,
            index=False,
            encoding="utf-8-sig",
        )
        comparison.to_csv(
            comparison_path,
            index=False,
            encoding="utf-8-sig",
        )

        selected_xgb = (
            stage8_meta.get(
                "selected_xgb_candidate",
                {},
            )
        )

        metadata = {
            "project": (
                "China A-Share Cross-Sectional Alpha Research"
            ),
            "script": (
                "09_executable_portfolio_backtest.py"
            ),
            "stage9_spec_version": (
                STAGE9_SPEC_VERSION
            ),
            "generated_at_utc": datetime.now(
                timezone.utc
            ).isoformat(),
            "required_stage5_version": (
                REQUIRED_STAGE5_VERSION
            ),
            "oos_period": [
                OOS_START,
                OOS_END,
            ],
            "features": FEATURE_NAMES,
            "signal_universe_rule": (
                "liquid_universe_primary == True AND all 11 frozen z-score "
                "features finite; target availability and future tradability "
                "are explicitly excluded from selection"
            ),
            "models": [
                "Frozen Stage-8 XGBoost annual specification retrained with the frozen purge rule",
                "Frozen Stage-7 monthly Ridge coefficient path",
                "Frozen Stage-7 monthly OLS coefficient path",
            ],
            "selected_xgb_candidate": (
                selected_xgb
            ),
            "rebalance_step_market_days": (
                REBALANCE_STEP_MARKET_DAYS
            ),
            "n_rebalance_events": len(
                events
            ),
            "first_signal_date": (
                events[0].signal_date
            ),
            "first_execution_date": (
                events[0].execution_date
            ),
            "last_signal_date": (
                events[-1].signal_date
            ),
            "final_scheduled_exit_date": (
                events[-1]
                .scheduled_exit_date
            ),
            "top_fractions": list(
                TOP_FRACTIONS
            ),
            "transaction_cost_bps_one_way": list(
                cfg.cost_bps
            ),
            "initial_nav": INITIAL_NAV,
            "execution_rules": {
                "new_buy_at_up_limit": False,
                "sell_at_down_limit": False,
                "new_buy_current_ST": False,
                "missing_daily_row": (
                    "cannot trade; existing position carried at last marked adjusted price"
                ),
                "blocked_sell": (
                    "position remains in portfolio until a later scheduled rebalance"
                ),
                "T_plus_1": (
                    "satisfied mechanically by 5-market-day rebalance cadence"
                ),
            },
            "share_accounting": (
                "fractional shares on adjusted open total-return prices; 100-share "
                "board-lot discretization intentionally omitted"
            ),
            "cost_interpretation": (
                "symmetric one-way proportional cost on absolute traded notional"
            ),
            "qa_issue_count": len(
                issues
            ),
            "qa_issues": issues,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "pandas_version": package_version(
                "pandas"
            ),
            "numpy_version": package_version(
                "numpy"
            ),
            "pyarrow_version": package_version(
                "pyarrow"
            ),
            "xgboost_version": str(
                xgb.__version__
            ),
            "research_notes": [
                "Stage 9 is the first self-financing portfolio stage; earlier Q5/Q1 results were predictive diagnostics.",
                "No future t+6 target availability is used to define the signal-date investment universe.",
                "No future exit tradability is used to decide whether a stock is bought.",
                "Entry-time tradability is allowed because it is observed at the actual execution open.",
                "If a desired new name cannot be bought, the ranking is filled from the next executable name.",
                "If an existing name cannot be sold, it is frozen and carried rather than dropped.",
                "Top-20% is the primary long-only portfolio; Top-10% is a precommitted concentration robustness check.",
                "Costs 0/5/10/20 bps are one-way and are deducted from cash at each actual buy/sell.",
            ],
        }

        atomic_write_json(
            metadata,
            cfg.report_root
            / "backtest_metadata.json",
        )

        logger.info("=" * 116)
        logger.info(
            "STAGE 9 COMPLETE | summary rows=%d | yearly rows=%d | "
            "period rows=%d",
            len(summary),
            len(yearly),
            len(period),
        )
        logger.info(
            "Reports: %s | %s | %s | %s | %s | %s",
            period_path,
            equity_path,
            summary_path,
            yearly_path,
            execution_path,
            comparison_path,
        )

        if issues:
            logger.error(
                "STAGE 9 QA FAIL | %d issue(s)",
                len(issues),
            )

            for issue in issues:
                logger.error(
                    "QA | %s",
                    issue,
                )

            logger.info("=" * 116)
            return 1

        logger.info(
            "STAGE 9 QA: PASS"
        )
        logger.info("=" * 116)

        return 0

    except KeyboardInterrupt:
        logger.warning(
            "Interrupted by user."
        )
        return 130

    except Exception:
        logger.exception(
            "Fatal error during Stage-9 executable backtest."
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
