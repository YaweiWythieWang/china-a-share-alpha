#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
10_robustness_final_diagnostics.py

China A-Share Cross-Sectional Alpha Research
Stage 10: Robustness & Final Research Diagnostics

PURPOSE
-------
Stress-test the FROZEN Stage-7/8/9 conclusions without any further model tuning.

OOS has already been opened. Therefore Stage 10 is deliberately diagnostic:
    - no new feature engineering;
    - no XGBoost hyperparameter changes;
    - no Ridge retuning;
    - no ex-post factor deletion;
    - no sign flipping.

ROBUSTNESS BLOCKS
-----------------
A) Liquidity universe robustness
   Baseline:
       Stage-9 Primary Top-1500 ADV20 universe.
   Robustness:
       Restrict the same frozen model scores to the lagged-ADV20 Top-1000
       ex-ante universe.

   Important:
       Stage-5 Top-1500 z-scores and the frozen models are NOT re-estimated.
       This isolates investability/liquidity-universe sensitivity rather than
       creating a new post-OOS model.

B) Portfolio concentration & transaction-cost robustness
   Already precommitted in Stage 9:
       Top 20%
       Top 10%
       0 / 5 / 10 / 20 bps one-way cost.

   Stage 10 places Top-1500 and Top-1000 results in one unified table.

C) Forward-horizon robustness
   Frozen model scores on the same non-overlapping 5-market-day SIGNAL schedule
   are tested against open-to-open forward returns over:

       1, 5, 10, 20 market days.

   For horizon H:
       entry = next market open t+1
       exit  = market open t+1+H
       return = AdjOpen_exit / AdjOpen_entry - 1

   This is a PREDICTIVE robustness check. Models remain the frozen 5-day-target
   models. We do NOT retrain separate 1/10/20-day models after seeing OOS.

D) Regime robustness
   Regimes are descriptive only, not trading rules.

   Using the 0-bps Top-1500 UNIVERSE_EW non-overlapping period return:
       DOWN    = bottom tercile
       NEUTRAL = middle tercile
       UP      = top tercile

   We report each strategy's active return relative to its same-liquidity
   benchmark inside each regime.

E) Paired inference
   For the non-overlapping executable backtest periods:
       strategy - same-universe benchmark
       XGB - Ridge

   Mean differences use Newey-West/HAC lag 5 for conservative inference.

DEPENDENCY
----------
This script reuses the frozen Stage-9 execution engine. Put:

    09_executable_portfolio_backtest.py
    10_robustness_final_diagnostics.py

in the SAME project directory.

INPUTS
------
Stage-5 frozen signal panels:
    data/processed/signals/YYYY/MM/signals_YYYYMM.parquet

Stage-7 reports:
    data/linear_model_reports/linear_model_coefficients.csv

Stage-8:
    data/xgb_model_reports/xgb_metadata.json
    data/cache/stage8_xgb/

Stage-9 baseline reports:
    data/backtest_reports/

OUTPUT
------
data/robustness_reports/
    robustness_liquidity_summary.csv
    robustness_liquidity_yearly.csv
    robustness_execution_summary.csv
    robustness_horizon_daily.csv
    robustness_horizon_summary.csv
    robustness_regime_results.csv
    robustness_paired_tests.csv
    robustness_final_scorecard.csv
    robustness_metadata.json
    10_robustness_final_diagnostics.log

RUN IN ANACONDA PROMPT
----------------------
python 10_robustness_final_diagnostics.py
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import logging
import math
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Set, Tuple

import numpy as np
import pandas as pd


STAGE10_SPEC_VERSION = "v1_post_oos_frozen_robustness"

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

HORIZONS = (1, 5, 10, 20)
DEFAULT_COST_BPS = (0.0, 5.0, 10.0, 20.0)
HAC_LAG = 5


class Config:
    def __init__(self, data_root: Path, nthread: int):
        self.data_root = data_root
        self.nthread = nthread

    @property
    def signal_root(self) -> Path:
        return self.data_root / "processed" / "signals"

    @property
    def stage9_report_root(self) -> Path:
        return self.data_root / "backtest_reports"

    @property
    def report_root(self) -> Path:
        return self.data_root / "robustness_reports"


def setup_logging(report_root: Path) -> logging.Logger:
    report_root.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("stage10_robustness")
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
        report_root / "10_robustness_final_diagnostics.log",
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


def atomic_write_json(obj: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)
    tmp.replace(path)


def load_stage9_module():
    candidates = [
        Path(__file__).resolve().with_name("09_executable_portfolio_backtest.py"),
        Path.cwd() / "09_executable_portfolio_backtest.py",
    ]

    path = next((p for p in candidates if p.exists()), None)

    if path is None:
        raise FileNotFoundError(
            "Cannot find 09_executable_portfolio_backtest.py. "
            "Place Stage-9 and Stage-10 scripts in the same project directory."
        )

    spec = importlib.util.spec_from_file_location("stage9_engine", path)

    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import Stage-9 engine from {path}")

    mod = importlib.util.module_from_spec(spec)
    sys.modules["stage9_engine"] = mod
    spec.loader.exec_module(mod)

    return mod


def build_top1000_code_sets(
    files: Sequence[Path],
    signal_dates: Set[str],
    *,
    logger: logging.Logger,
) -> Dict[str, Set[str]]:
    out: Dict[str, Set[str]] = {}

    read_cols = [
        "ts_code",
        "trade_date",
        "base_eligible_signal_day",
        "adv20_lagged",
    ]

    for i, path in enumerate(files, 1):
        df = pd.read_parquet(path, columns=read_cols)
        df["trade_date"] = normalize_date_series(df["trade_date"])

        x = df.loc[df["trade_date"].isin(signal_dates)].copy()

        if x.empty:
            continue

        x["adv20_lagged"] = pd.to_numeric(
            x["adv20_lagged"],
            errors="coerce",
        )

        for d, g in x.groupby("trade_date", sort=False):
            eligible = (
                g["base_eligible_signal_day"]
                .fillna(False)
                .astype(bool)
            )

            z = g.loc[
                eligible
                & np.isfinite(
                    g["adv20_lagged"].to_numpy(dtype=float)
                ),
                ["ts_code", "adv20_lagged"],
            ].copy()

            z["ts_code"] = z["ts_code"].astype(str)

            z = z.sort_values(
                ["adv20_lagged", "ts_code"],
                ascending=[False, True],
                kind="stable",
            )

            out[str(d)] = set(
                z.head(1000)["ts_code"].tolist()
            )

        logger.info(
            "TOP1000 %d | %s | relevant dates=%d",
            i,
            path.name,
            x["trade_date"].nunique(),
        )

        del df, x
        gc.collect()

    missing = signal_dates - set(out)

    if missing:
        raise RuntimeError(
            f"Missing Top-1000 code sets for {len(missing)} dates; "
            f"examples={sorted(missing)[:10]}"
        )

    return out


def subset_score_snapshots_to_top1000(
    stage9,
    *,
    top1500_scores: Mapping[str, Any],
    top1000_codes: Mapping[str, Set[str]],
) -> Dict[str, Any]:
    out = {}

    for d, snap in top1500_scores.items():
        raw1000 = top1000_codes[d]
        available = set(snap.universe_codes)
        codes = sorted(raw1000 & available)

        model_scores = {}

        for model_key, score_map in snap.scores.items():
            model_scores[model_key] = {
                c: score_map[c]
                for c in codes
            }

        out[d] = stage9.ScoreSnapshot(
            signal_date=d,
            universe_codes=codes,
            scores=model_scores,
            n_universe=len(codes),
            xgb_train_n=snap.xgb_train_n,
            xgb_last_exit=snap.xgb_last_exit,
        )

    return out


def run_top1000_backtest(
    stage9,
    *,
    cfg9,
    events,
    top1000_scores,
    market_snapshots,
    logger: logging.Logger,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    all_period = []
    all_equity = []

    specs = stage9.strategy_specs()
    total = len(specs) * len(cfg9.cost_bps)
    run_id = 0

    for spec in specs:
        for cost in cfg9.cost_bps:
            run_id += 1

            logger.info(
                "TOP1000 BACKTEST %d/%d | %s | cost=%.1f bps",
                run_id,
                total,
                spec.strategy,
                cost,
            )

            p, e = stage9.run_one_strategy_cost(
                spec=spec,
                cost_bps=cost,
                events=events,
                score_snapshots=top1000_scores,
                market_snapshots=market_snapshots,
            )

            p["liquidity_universe"] = "TOP1000"
            e["liquidity_universe"] = "TOP1000"

            all_period.append(p)
            all_equity.append(e)

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

    summary = stage9.build_summary(period, equity)
    yearly = stage9.build_yearly(equity)
    execution = stage9.build_execution_diagnostics(period)

    summary["liquidity_universe"] = "TOP1000"
    yearly["liquidity_universe"] = "TOP1000"
    execution["liquidity_universe"] = "TOP1000"

    return period, summary, yearly, execution


def load_stage9_baseline(
    cfg: Config,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    names = {
        "period": "backtest_period_returns.csv",
        "summary": "backtest_summary.csv",
        "yearly": "backtest_yearly.csv",
        "execution": "backtest_execution_diagnostics.csv",
    }

    frames = {}

    for k, name in names.items():
        path = cfg.stage9_report_root / name

        if not path.exists():
            raise FileNotFoundError(
                f"Missing Stage-9 baseline report: {path}"
            )

        frames[k] = pd.read_csv(
            path,
            dtype={
                "execution_date": str,
                "signal_date": str,
            },
        )

        frames[k]["liquidity_universe"] = "TOP1500"

    return (
        frames["period"],
        frames["summary"],
        frames["yearly"],
        frames["execution"],
    )


def build_horizon_date_plan(
    *,
    market_dates: Sequence[str],
    events,
    score_snapshots,
) -> Tuple[Dict[str, Set[str]], List[Dict[str, Any]]]:
    idx = {d: j for j, d in enumerate(market_dates)}

    needed: Dict[str, Set[str]] = {}
    plan: List[Dict[str, Any]] = []

    for e in events:
        j = idx[e.signal_date]
        entry_idx = j + 1

        if entry_idx >= len(market_dates):
            continue

        entry_date = market_dates[entry_idx]
        codes = set(
            score_snapshots[e.signal_date].universe_codes
        )

        for h in HORIZONS:
            exit_idx = entry_idx + h

            if exit_idx >= len(market_dates):
                continue

            exit_date = market_dates[exit_idx]

            needed.setdefault(
                entry_date,
                set(),
            ).update(codes)

            needed.setdefault(
                exit_date,
                set(),
            ).update(codes)

            plan.append({
                "signal_date": e.signal_date,
                "entry_date": entry_date,
                "exit_date": exit_date,
                "horizon_days": int(h),
            })

    return needed, plan


def collect_needed_adj_open(
    files: Sequence[Path],
    needed_codes_by_date: Mapping[str, Set[str]],
    *,
    logger: logging.Logger,
) -> Dict[str, Dict[str, float]]:
    needed_dates = set(needed_codes_by_date)

    out: Dict[str, Dict[str, float]] = {
        d: {}
        for d in needed_dates
    }

    for i, path in enumerate(files, 1):
        df = pd.read_parquet(
            path,
            columns=[
                "ts_code",
                "trade_date",
                "adj_open",
            ],
        )

        df["trade_date"] = normalize_date_series(
            df["trade_date"]
        )

        x = df.loc[
            df["trade_date"].isin(needed_dates)
        ].copy()

        if x.empty:
            continue

        x["adj_open"] = pd.to_numeric(
            x["adj_open"],
            errors="coerce",
        )

        for d, g in x.groupby(
            "trade_date",
            sort=False,
        ):
            need = needed_codes_by_date[str(d)]

            z = g.loc[
                g["ts_code"].astype(str).isin(need),
                ["ts_code", "adj_open"],
            ]

            target = out[str(d)]

            for row in z.itertuples(index=False):
                code = str(row.ts_code)
                px = safe_float(row.adj_open)

                if np.isfinite(px) and px > 0:
                    target[code] = float(px)

        logger.info(
            "HORIZON PRICE %d | %s | relevant dates=%d",
            i,
            path.name,
            x["trade_date"].nunique()
            if not x.empty
            else 0,
        )

        del df, x
        gc.collect()

    return out


def pearson_corr(
    x: np.ndarray,
    y: np.ndarray,
) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    valid = (
        np.isfinite(x)
        & np.isfinite(y)
    )

    x = x[valid]
    y = y[valid]

    if len(x) < 3:
        return np.nan

    sx = float(np.std(x, ddof=0))
    sy = float(np.std(y, ddof=0))

    if sx <= 0 or sy <= 0:
        return np.nan

    return float(
        np.corrcoef(x, y)[0, 1]
    )


def spearman_corr(
    x: np.ndarray,
    y: np.ndarray,
) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    valid = (
        np.isfinite(x)
        & np.isfinite(y)
    )

    x = x[valid]
    y = y[valid]

    if len(x) < 3:
        return np.nan

    rx = (
        pd.Series(x)
        .rank(method="average")
        .to_numpy(dtype=float)
    )

    ry = (
        pd.Series(y)
        .rank(method="average")
        .to_numpy(dtype=float)
    )

    return pearson_corr(rx, ry)


def newey_west_mean_stats(
    values: Sequence[float],
    lag: int = HAC_LAG,
) -> Tuple[float, float, float, int]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]

    n = len(x)

    if n == 0:
        return np.nan, np.nan, np.nan, 0

    mean = float(np.mean(x))

    if n < 2:
        return mean, np.nan, np.nan, n

    e = x - mean
    L = min(int(lag), n - 1)

    lrv = float(
        np.dot(e, e) / n
    )

    for ell in range(1, L + 1):
        gamma = float(
            np.dot(
                e[ell:],
                e[:-ell],
            )
            / n
        )

        weight = (
            1.0
            - ell
            / (L + 1.0)
        )

        lrv += (
            2.0
            * weight
            * gamma
        )

    lrv = max(
        lrv,
        0.0,
    )

    se = math.sqrt(
        lrv / n
    )

    t = (
        mean / se
        if se > 0
        else np.nan
    )

    return mean, se, t, n


def equal_count_quantiles(
    score: np.ndarray,
    codes: np.ndarray,
    q: int = 5,
) -> np.ndarray:
    score = np.asarray(score, dtype=float)
    codes = np.asarray(codes, dtype=str)

    n = len(score)
    out = np.full(
        n,
        np.nan,
        dtype=float,
    )

    valid = np.isfinite(score)
    idx = np.flatnonzero(valid)

    if len(idx) < q:
        return out

    order_local = np.lexsort(
        (
            codes[idx],
            score[idx],
        )
    )

    ordered = idx[order_local]
    m = len(ordered)

    groups = (
        np.floor(
            np.arange(m, dtype=float)
            * q
            / m
        )
        .astype(int)
        + 1
    )

    groups = np.minimum(
        groups,
        q,
    )

    out[ordered] = groups

    return out


def build_horizon_daily(
    *,
    plan: Sequence[Mapping[str, Any]],
    prices: Mapping[str, Mapping[str, float]],
    score_snapshots: Mapping[str, Any],
) -> pd.DataFrame:
    rows = []
    models = ["XGB", "RIDGE", "OLS"]

    for item in plan:
        signal_date = str(item["signal_date"])
        entry_date = str(item["entry_date"])
        exit_date = str(item["exit_date"])
        h = int(item["horizon_days"])

        snap = score_snapshots[signal_date]
        entry = prices[entry_date]
        exit_ = prices[exit_date]

        for model in models:
            score_map = snap.scores[model]

            codes = []
            scores = []
            rets = []

            for code in snap.universe_codes:
                p0 = entry.get(code)
                p1 = exit_.get(code)

                if (
                    p0 is None
                    or p1 is None
                    or p0 <= 0
                    or p1 <= 0
                ):
                    continue

                s = score_map[code]

                if not np.isfinite(s):
                    continue

                r = p1 / p0 - 1.0

                if not np.isfinite(r):
                    continue

                codes.append(code)
                scores.append(float(s))
                rets.append(float(r))

            n = len(rets)

            if n < 100:
                continue

            codes_arr = np.asarray(codes, dtype=str)
            score_arr = np.asarray(scores, dtype=float)
            ret_arr = np.asarray(rets, dtype=float)

            q = equal_count_quantiles(
                score_arr,
                codes_arr,
                q=5,
            )

            q1 = float(
                np.mean(
                    ret_arr[q == 1]
                )
            )
            q5 = float(
                np.mean(
                    ret_arr[q == 5]
                )
            )

            universe = float(
                np.mean(ret_arr)
            )

            order = np.lexsort(
                (
                    codes_arr,
                    -score_arr,
                )
            )

            k20 = int(
                math.ceil(
                    0.20 * n
                )
            )

            k10 = int(
                math.ceil(
                    0.10 * n
                )
            )

            top20 = float(
                np.mean(
                    ret_arr[
                        order[:k20]
                    ]
                )
            )

            top10 = float(
                np.mean(
                    ret_arr[
                        order[:k10]
                    ]
                )
            )

            rows.append({
                "signal_date": signal_date,
                "entry_date": entry_date,
                "exit_date": exit_date,
                "horizon_days": h,
                "model": model,
                "n": int(n),
                "coverage_rate": (
                    n / snap.n_universe
                    if snap.n_universe > 0
                    else np.nan
                ),
                "ic": pearson_corr(
                    score_arr,
                    ret_arr,
                ),
                "rank_ic": spearman_corr(
                    score_arr,
                    ret_arr,
                ),
                "q5_q1_ret": q5 - q1,
                "top20_minus_universe": (
                    top20 - universe
                ),
                "top10_minus_universe": (
                    top10 - universe
                ),
            })

    return pd.DataFrame(rows)


def summarize_horizon(
    daily: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for (
        model,
        h,
    ), g in daily.groupby(
        [
            "model",
            "horizon_days",
        ],
        sort=True,
    ):
        mean_ic, ic_se, ic_t, _ = (
            newey_west_mean_stats(
                g["ic"],
                lag=HAC_LAG,
            )
        )

        mean_ric, ric_se, ric_t, _ = (
            newey_west_mean_stats(
                g["rank_ic"],
                lag=HAC_LAG,
            )
        )

        spread, spread_se, spread_t, _ = (
            newey_west_mean_stats(
                g["q5_q1_ret"],
                lag=HAC_LAG,
            )
        )

        top20, top20_se, top20_t, _ = (
            newey_west_mean_stats(
                g["top20_minus_universe"],
                lag=HAC_LAG,
            )
        )

        top10, top10_se, top10_t, _ = (
            newey_west_mean_stats(
                g["top10_minus_universe"],
                lag=HAC_LAG,
            )
        )

        rows.append({
            "model": model,
            "horizon_days": int(h),
            "n_signal_dates": int(
                g["signal_date"].nunique()
            ),
            "mean_coverage_rate": safe_float(
                pd.to_numeric(
                    g["coverage_rate"],
                    errors="coerce",
                ).mean()
            ),
            "mean_ic": mean_ic,
            "ic_hac_se": ic_se,
            "ic_hac_t": ic_t,
            "mean_rank_ic": mean_ric,
            "rank_ic_hac_se": ric_se,
            "rank_ic_hac_t": ric_t,
            "mean_q5_q1": spread,
            "mean_q5_q1_bps": (
                spread * 10000.0
                if np.isfinite(spread)
                else np.nan
            ),
            "q5_q1_hac_t": spread_t,
            "mean_top20_minus_universe": top20,
            "mean_top20_minus_universe_bps": (
                top20 * 10000.0
                if np.isfinite(top20)
                else np.nan
            ),
            "top20_minus_universe_hac_t": top20_t,
            "mean_top10_minus_universe": top10,
            "mean_top10_minus_universe_bps": (
                top10 * 10000.0
                if np.isfinite(top10)
                else np.nan
            ),
            "top10_minus_universe_hac_t": top10_t,
        })

    return pd.DataFrame(rows)


def prepare_period_returns(
    df: pd.DataFrame,
    liquidity: str,
) -> pd.DataFrame:
    x = df.copy()
    x["liquidity_universe"] = liquidity
    x["execution_date"] = normalize_date_series(
        x["execution_date"]
    )
    x["net_period_return"] = pd.to_numeric(
        x["net_period_return"],
        errors="coerce",
    )

    return x.loc[
        x["net_period_return"].notna()
    ].copy()


def paired_tests(
    all_period: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for liquidity in sorted(
        all_period[
            "liquidity_universe"
        ].unique()
    ):
        xu = all_period.loc[
            all_period[
                "liquidity_universe"
            ]
            == liquidity
        ]

        for cost in sorted(
            xu["cost_bps"].unique()
        ):
            xc = xu.loc[
                xu["cost_bps"] == cost
            ]

            bench = xc.loc[
                xc["strategy"] == "UNIVERSE_EW",
                [
                    "execution_date",
                    "net_period_return",
                ],
            ].rename(
                columns={
                    "net_period_return":
                    "benchmark_return"
                }
            )

            for strategy in sorted(
                set(xc["strategy"])
                - {"UNIVERSE_EW"}
            ):
                s = xc.loc[
                    xc["strategy"] == strategy,
                    [
                        "execution_date",
                        "net_period_return",
                    ],
                ].rename(
                    columns={
                        "net_period_return":
                        "strategy_return"
                    }
                )

                m = s.merge(
                    bench,
                    on="execution_date",
                    how="inner",
                )

                diff = (
                    m["strategy_return"]
                    - m["benchmark_return"]
                )

                mean, se, t, n = (
                    newey_west_mean_stats(
                        diff,
                        lag=HAC_LAG,
                    )
                )

                rows.append({
                    "test_type":
                    "strategy_minus_benchmark",
                    "liquidity_universe":
                    liquidity,
                    "cost_bps":
                    float(cost),
                    "strategy":
                    strategy,
                    "benchmark":
                    "UNIVERSE_EW",
                    "n_periods":
                    int(n),
                    "mean_difference":
                    mean,
                    "mean_difference_bps":
                    (
                        mean * 10000.0
                        if np.isfinite(mean)
                        else np.nan
                    ),
                    "hac_se":
                    se,
                    "hac_t":
                    t,
                })

            for pct in (20, 10):
                xgb_name = f"XGB_TOP{pct}"
                ridge_name = f"RIDGE_TOP{pct}"

                xgb = xc.loc[
                    xc["strategy"] == xgb_name,
                    [
                        "execution_date",
                        "net_period_return",
                    ],
                ].rename(
                    columns={
                        "net_period_return":
                        "xgb_return"
                    }
                )

                ridge = xc.loc[
                    xc["strategy"] == ridge_name,
                    [
                        "execution_date",
                        "net_period_return",
                    ],
                ].rename(
                    columns={
                        "net_period_return":
                        "ridge_return"
                    }
                )

                m = xgb.merge(
                    ridge,
                    on="execution_date",
                    how="inner",
                )

                diff = (
                    m["xgb_return"]
                    - m["ridge_return"]
                )

                mean, se, t, n = (
                    newey_west_mean_stats(
                        diff,
                        lag=HAC_LAG,
                    )
                )

                rows.append({
                    "test_type":
                    "xgb_minus_ridge",
                    "liquidity_universe":
                    liquidity,
                    "cost_bps":
                    float(cost),
                    "strategy":
                    xgb_name,
                    "benchmark":
                    ridge_name,
                    "n_periods":
                    int(n),
                    "mean_difference":
                    mean,
                    "mean_difference_bps":
                    (
                        mean * 10000.0
                        if np.isfinite(mean)
                        else np.nan
                    ),
                    "hac_se":
                    se,
                    "hac_t":
                    t,
                })

    return pd.DataFrame(rows)


def build_regime_results(
    all_period: pd.DataFrame,
) -> pd.DataFrame:
    baseline = all_period.loc[
        (
            all_period[
                "liquidity_universe"
            ]
            == "TOP1500"
        )
        & (
            all_period["strategy"]
            == "UNIVERSE_EW"
        )
        & (
            all_period["cost_bps"]
            == 0
        ),
        [
            "execution_date",
            "net_period_return",
        ],
    ].copy()

    baseline = baseline.dropna(
        subset=[
            "net_period_return"
        ]
    )

    q1 = float(
        baseline[
            "net_period_return"
        ].quantile(
            1.0 / 3.0
        )
    )

    q2 = float(
        baseline[
            "net_period_return"
        ].quantile(
            2.0 / 3.0
        )
    )

    def classify(x: float) -> str:
        if x <= q1:
            return "DOWN"
        if x >= q2:
            return "UP"
        return "NEUTRAL"

    baseline["regime"] = (
        baseline[
            "net_period_return"
        ].map(classify)
    )

    regime_map = baseline[
        [
            "execution_date",
            "regime",
        ]
    ]

    rows = []

    for liquidity in sorted(
        all_period[
            "liquidity_universe"
        ].unique()
    ):
        xu = all_period.loc[
            all_period[
                "liquidity_universe"
            ]
            == liquidity
        ]

        for cost in sorted(
            xu["cost_bps"].unique()
        ):
            xc = xu.loc[
                xu["cost_bps"] == cost
            ]

            bench = xc.loc[
                xc["strategy"] == "UNIVERSE_EW",
                [
                    "execution_date",
                    "net_period_return",
                ],
            ].rename(
                columns={
                    "net_period_return":
                    "benchmark_return"
                }
            )

            for strategy in sorted(
                set(xc["strategy"])
                - {"UNIVERSE_EW"}
            ):
                s = xc.loc[
                    xc["strategy"] == strategy,
                    [
                        "execution_date",
                        "net_period_return",
                    ],
                ].rename(
                    columns={
                        "net_period_return":
                        "strategy_return"
                    }
                )

                m = (
                    s.merge(
                        bench,
                        on="execution_date",
                        how="inner",
                    )
                    .merge(
                        regime_map,
                        on="execution_date",
                        how="inner",
                    )
                )

                m["active_return"] = (
                    m["strategy_return"]
                    - m["benchmark_return"]
                )

                for regime, g in m.groupby(
                    "regime",
                    sort=False,
                ):
                    mean, se, t, n = (
                        newey_west_mean_stats(
                            g["active_return"],
                            lag=HAC_LAG,
                        )
                    )

                    rows.append({
                        "liquidity_universe":
                        liquidity,
                        "cost_bps":
                        float(cost),
                        "strategy":
                        strategy,
                        "regime":
                        regime,
                        "regime_definition":
                        (
                            "Top1500 0-bps "
                            "UNIVERSE_EW non-overlapping "
                            "5-day return terciles"
                        ),
                        "down_cutoff":
                        q1,
                        "up_cutoff":
                        q2,
                        "n_periods":
                        int(n),
                        "mean_strategy_return":
                        safe_float(
                            pd.to_numeric(
                                g[
                                    "strategy_return"
                                ],
                                errors="coerce",
                            ).mean()
                        ),
                        "mean_benchmark_return":
                        safe_float(
                            pd.to_numeric(
                                g[
                                    "benchmark_return"
                                ],
                                errors="coerce",
                            ).mean()
                        ),
                        "mean_active_return":
                        mean,
                        "mean_active_return_bps":
                        (
                            mean * 10000.0
                            if np.isfinite(mean)
                            else np.nan
                        ),
                        "active_hac_se":
                        se,
                        "active_hac_t":
                        t,
                        "active_positive_share":
                        safe_float(
                            (
                                g[
                                    "active_return"
                                ]
                                > 0
                            ).mean()
                        ),
                    })

    return pd.DataFrame(rows)


def build_final_scorecard(
    liquidity_summary: pd.DataFrame,
    paired: pd.DataFrame,
    horizon_summary: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for liquidity in (
        "TOP1500",
        "TOP1000",
    ):
        s = liquidity_summary.loc[
            (
                liquidity_summary[
                    "liquidity_universe"
                ]
                == liquidity
            )
            & (
                liquidity_summary[
                    "strategy"
                ]
                == "XGB_TOP20"
            )
            & (
                liquidity_summary[
                    "cost_bps"
                ]
                == 5
            )
        ]

        if len(s) == 1:
            r = s.iloc[0]

            test = paired.loc[
                (
                    paired["test_type"]
                    == "strategy_minus_benchmark"
                )
                & (
                    paired[
                        "liquidity_universe"
                    ]
                    == liquidity
                )
                & (
                    paired["cost_bps"]
                    == 5
                )
                & (
                    paired["strategy"]
                    == "XGB_TOP20"
                )
            ]

            active_t = (
                safe_float(
                    test.iloc[0]["hac_t"]
                )
                if len(test) == 1
                else np.nan
            )

            rows.append({
                "check":
                f"XGB_TOP20_5bps_{liquidity}",
                "category":
                "executable_liquidity_robustness",
                "value_1_name":
                "cagr",
                "value_1":
                r["cagr"],
                "value_2_name":
                "sharpe",
                "value_2":
                r["sharpe_zero_rf"],
                "value_3_name":
                "active_return_hac_t",
                "value_3":
                active_t,
            })

    xgb_h = horizon_summary.loc[
        horizon_summary["model"]
        == "XGB"
    ]

    for row in xgb_h.itertuples(
        index=False
    ):
        rows.append({
            "check":
            f"XGB_HORIZON_{int(row.horizon_days)}D",
            "category":
            "predictive_horizon_robustness",
            "value_1_name":
            "rank_ic",
            "value_1":
            row.mean_rank_ic,
            "value_2_name":
            "top20_minus_universe_bps",
            "value_2":
            row.mean_top20_minus_universe_bps,
            "value_3_name":
            "top20_hac_t",
            "value_3":
            row.top20_minus_universe_hac_t,
        })

    return pd.DataFrame(rows)


def final_qa(
    *,
    top1500_scores,
    top1000_scores,
    liquidity_summary,
    horizon_summary,
    paired,
) -> List[str]:
    issues = []

    for d in top1500_scores:
        n15 = top1500_scores[d].n_universe
        n10 = top1000_scores[d].n_universe

        if n10 > n15:
            issues.append(
                f"{d}: Top1000 ready universe {n10} exceeds Top1500 {n15}"
            )

        if n10 <= 0:
            issues.append(
                f"{d}: empty Top1000 ready universe"
            )

    expected_liquidity = {
        "TOP1500",
        "TOP1000",
    }

    if set(
        liquidity_summary[
            "liquidity_universe"
        ]
    ) != expected_liquidity:
        issues.append(
            "Liquidity summary does not contain both TOP1500 and TOP1000."
        )

    expected_h = set(HORIZONS)

    found_h = set(
        horizon_summary[
            "horizon_days"
        ].astype(int)
    )

    if found_h != expected_h:
        issues.append(
            f"Horizon summary mismatch: "
            f"expected={sorted(expected_h)} "
            f"found={sorted(found_h)}"
        )

    for model in (
        "XGB",
        "RIDGE",
        "OLS",
    ):
        found = set(
            horizon_summary.loc[
                horizon_summary[
                    "model"
                ]
                == model,
                "horizon_days",
            ].astype(int)
        )

        if found != expected_h:
            issues.append(
                f"{model}: incomplete horizon coverage {sorted(found)}"
            )

    if paired.empty:
        issues.append(
            "Paired-test output is empty."
        )

    return issues


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Stage 10 frozen-model robustness and final diagnostics."
        )
    )

    p.add_argument(
        "--data-root",
        default="data",
    )

    p.add_argument(
        "--nthread",
        type=int,
        default=-1,
    )

    return p.parse_args()


def main() -> int:
    args = parse_args()

    cfg = Config(
        data_root=Path(args.data_root),
        nthread=int(args.nthread),
    )

    cfg.report_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    logger = setup_logging(
        cfg.report_root
    )

    logger.info("=" * 116)
    logger.info(
        "STAGE 10 | ROBUSTNESS & FINAL DIAGNOSTICS"
    )
    logger.info(
        "Post-OOS discipline: frozen features, frozen Ridge/OLS, "
        "frozen XGBoost specification; NO further model tuning."
    )
    logger.info(
        "Robustness: Top1500 vs Top1000 | Top20 vs Top10 | "
        "0/5/10/20 bps | horizons 1/5/10/20 | return regimes."
    )
    logger.info("=" * 116)

    try:
        stage9 = load_stage9_module()

        cfg9 = stage9.Config(
            data_root=cfg.data_root,
            cost_bps=DEFAULT_COST_BPS,
            nthread=cfg.nthread,
        )

        files = stage9.discover_signal_files(
            cfg.signal_root
        )

        market_dates = (
            stage9.build_oos_market_calendar(
                files,
                logger=logger,
            )
        )

        events = stage9.build_rebalance_schedule(
            market_dates
        )

        signal_dates = {
            e.signal_date
            for e in events
        }

        logger.info(
            "Baseline 5-day events=%d | signal dates=%d",
            len(events),
            len(signal_dates),
        )

        (
            signal_snapshots,
            market_snapshots,
        ) = stage9.build_snapshots(
            files,
            events,
            logger=logger,
        )

        linear_betas = (
            stage9.load_linear_beta_path(
                cfg9
            )
        )

        stage8_meta = (
            stage9.load_stage8_metadata(
                cfg9
            )
        )

        stage8_cache = (
            stage9.open_stage8_cache(
                cfg9
            )
        )

        top1500_scores = (
            stage9.score_signal_snapshots(
                events=events,
                signal_snapshots=signal_snapshots,
                linear_betas=linear_betas,
                stage8_cache=stage8_cache,
                stage8_meta=stage8_meta,
                cfg=cfg9,
                logger=logger,
            )
        )

        top1000_codes = (
            build_top1000_code_sets(
                files,
                signal_dates,
                logger=logger,
            )
        )

        top1000_scores = (
            subset_score_snapshots_to_top1000(
                stage9,
                top1500_scores=top1500_scores,
                top1000_codes=top1000_codes,
            )
        )

        (
            period1000,
            summary1000,
            yearly1000,
            execution1000,
        ) = run_top1000_backtest(
            stage9,
            cfg9=cfg9,
            events=events,
            top1000_scores=top1000_scores,
            market_snapshots=market_snapshots,
            logger=logger,
        )

        (
            period1500,
            summary1500,
            yearly1500,
            execution1500,
        ) = load_stage9_baseline(
            cfg
        )

        period1500 = prepare_period_returns(
            period1500,
            "TOP1500",
        )

        period1000 = prepare_period_returns(
            period1000,
            "TOP1000",
        )

        liquidity_summary = pd.concat(
            [
                summary1500,
                summary1000,
            ],
            ignore_index=True,
            sort=False,
        )

        liquidity_yearly = pd.concat(
            [
                yearly1500,
                yearly1000,
            ],
            ignore_index=True,
            sort=False,
        )

        execution_summary = pd.concat(
            [
                execution1500,
                execution1000,
            ],
            ignore_index=True,
            sort=False,
        )

        all_period = pd.concat(
            [
                period1500,
                period1000,
            ],
            ignore_index=True,
            sort=False,
        )

        if (
            "mean_one_way_turnover"
            in liquidity_summary.columns
        ):
            liquidity_summary[
                "mean_gross_traded_notional_ratio"
            ] = liquidity_summary[
                "mean_one_way_turnover"
            ]

            liquidity_summary[
                "approx_one_way_turnover"
            ] = (
                liquidity_summary[
                    "mean_one_way_turnover"
                ]
                / 2.0
            )

        if (
            "mean_one_way_turnover"
            in execution_summary.columns
        ):
            execution_summary[
                "mean_gross_traded_notional_ratio"
            ] = execution_summary[
                "mean_one_way_turnover"
            ]

            execution_summary[
                "approx_one_way_turnover"
            ] = (
                execution_summary[
                    "mean_one_way_turnover"
                ]
                / 2.0
            )

        paired = paired_tests(
            all_period
        )

        regime = build_regime_results(
            all_period
        )

        (
            needed_codes_by_date,
            horizon_plan,
        ) = build_horizon_date_plan(
            market_dates=market_dates,
            events=events,
            score_snapshots=top1500_scores,
        )

        horizon_prices = collect_needed_adj_open(
            files,
            needed_codes_by_date,
            logger=logger,
        )

        horizon_daily = build_horizon_daily(
            plan=horizon_plan,
            prices=horizon_prices,
            score_snapshots=top1500_scores,
        )

        horizon_summary = summarize_horizon(
            horizon_daily
        )

        scorecard = build_final_scorecard(
            liquidity_summary,
            paired,
            horizon_summary,
        )

        issues = final_qa(
            top1500_scores=top1500_scores,
            top1000_scores=top1000_scores,
            liquidity_summary=liquidity_summary,
            horizon_summary=horizon_summary,
            paired=paired,
        )

        paths = {
            "liquidity_summary":
            cfg.report_root
            / "robustness_liquidity_summary.csv",

            "liquidity_yearly":
            cfg.report_root
            / "robustness_liquidity_yearly.csv",

            "execution":
            cfg.report_root
            / "robustness_execution_summary.csv",

            "horizon_daily":
            cfg.report_root
            / "robustness_horizon_daily.csv",

            "horizon_summary":
            cfg.report_root
            / "robustness_horizon_summary.csv",

            "regime":
            cfg.report_root
            / "robustness_regime_results.csv",

            "paired":
            cfg.report_root
            / "robustness_paired_tests.csv",

            "scorecard":
            cfg.report_root
            / "robustness_final_scorecard.csv",
        }

        liquidity_summary.to_csv(
            paths["liquidity_summary"],
            index=False,
            encoding="utf-8-sig",
        )

        liquidity_yearly.to_csv(
            paths["liquidity_yearly"],
            index=False,
            encoding="utf-8-sig",
        )

        execution_summary.to_csv(
            paths["execution"],
            index=False,
            encoding="utf-8-sig",
        )

        horizon_daily.to_csv(
            paths["horizon_daily"],
            index=False,
            encoding="utf-8-sig",
        )

        horizon_summary.to_csv(
            paths["horizon_summary"],
            index=False,
            encoding="utf-8-sig",
        )

        regime.to_csv(
            paths["regime"],
            index=False,
            encoding="utf-8-sig",
        )

        paired.to_csv(
            paths["paired"],
            index=False,
            encoding="utf-8-sig",
        )

        scorecard.to_csv(
            paths["scorecard"],
            index=False,
            encoding="utf-8-sig",
        )

        top1000_sizes = np.array(
            [
                s.n_universe
                for s in top1000_scores.values()
            ],
            dtype=float,
        )

        metadata = {
            "project":
            "China A-Share Cross-Sectional Alpha Research",

            "script":
            "10_robustness_final_diagnostics.py",

            "stage10_spec_version":
            STAGE10_SPEC_VERSION,

            "generated_at_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

            "post_oos_rule":
            (
                "No model tuning, feature deletion, factor sign flipping, "
                "or hyperparameter expansion is permitted in Stage 10."
            ),

            "liquidity_robustness": {
                "baseline":
                "Frozen Stage-9 Primary Top1500 ADV20 universe",

                "robustness":
                (
                    "Lagged-ADV20 Top1000 restriction, intersected with "
                    "frozen feature-ready Top1500 scores"
                ),

                "zscore_rule":
                (
                    "Retain frozen Stage-5 Top1500-based daily z-scores; "
                    "do not re-standardize or retrain after OOS"
                ),

                "top1000_mean_ready_n":
                float(
                    np.mean(
                        top1000_sizes
                    )
                ),

                "top1000_min_ready_n":
                float(
                    np.min(
                        top1000_sizes
                    )
                ),
            },

            "portfolio_robustness": {
                "top_fractions":
                [
                    0.20,
                    0.10,
                ],

                "cost_bps_one_way":
                list(
                    DEFAULT_COST_BPS
                ),
            },

            "horizon_robustness": {
                "horizons_market_days":
                list(
                    HORIZONS
                ),

                "signal_schedule":
                "same frozen non-overlapping every-5-market-day signal dates",

                "models":
                [
                    "frozen XGB",
                    "frozen monthly Ridge",
                    "frozen monthly OLS",
                ],

                "interpretation":
                (
                    "predictive persistence of the frozen 5-day model scores; "
                    "not horizon-specific retraining"
                ),
            },

            "regime_robustness":
            (
                "Descriptive DOWN/NEUTRAL/UP terciles of Top1500 "
                "0-bps UNIVERSE_EW 5-day non-overlapping returns"
            ),

            "paired_inference":
            (
                "Newey-West/HAC lag 5 on non-overlapping executable "
                "period-return differences"
            ),

            "turnover_reporting_correction":
            (
                "Stage-9 column mean_one_way_turnover equals "
                "(buy notional + sell notional)/NAV and is therefore gross "
                "traded notional ratio. Stage-10 reports both the gross ratio "
                "and gross/2 as an approximate conventional one-way turnover."
            ),

            "qa_issue_count":
            len(issues),

            "qa_issues":
            issues,

            "python_version":
            platform.python_version(),

            "platform":
            platform.platform(),

            "pandas_version":
            package_version("pandas"),

            "numpy_version":
            package_version("numpy"),
        }

        atomic_write_json(
            metadata,
            cfg.report_root
            / "robustness_metadata.json",
        )

        logger.info("=" * 116)
        logger.info(
            "STAGE 10 COMPLETE | liquidity summary=%d rows | "
            "horizon summary=%d rows | paired tests=%d rows | regimes=%d rows",
            len(liquidity_summary),
            len(horizon_summary),
            len(paired),
            len(regime),
        )

        if issues:
            logger.error(
                "STAGE 10 QA FAIL | %d issue(s)",
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
            "STAGE 10 QA: PASS"
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
            "Fatal error during Stage-10 robustness diagnostics."
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
