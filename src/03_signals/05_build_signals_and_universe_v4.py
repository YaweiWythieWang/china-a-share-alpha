#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
05_build_signals_and_universe.py

China A-Share Cross-Sectional Alpha Research
Stage 5 v4: Dynamic liquid universe + price-volume signals + 5-day forward target.

INPUT
-----
Created by 04_build_full_panel.py:

data/processed/full_panel/YYYY/MM/full_panel_YYYYMM.parquet

Also uses:
data/raw_full/trade_cal/trade_cal_20090601_20251231.parquet

DEFAULT PERIOD
--------------
Warm-up/raw data:
    2009-06-01 to 2025-12-31
Formal research sample:
    2010-01-01 to 2025-12-31

OUTPUT
------
Monthly signal panel:
data/processed/signals/YYYY/MM/signals_YYYYMM.parquet

Diagnostics:
data/signal_reports/
    signal_monthly_quality.csv
    signal_daily_universe_summary.csv
    signal_feature_missingness.csv
    signal_feature_summary.csv
    signal_target_quality.csv
    signal_metadata.json
    05_build_signals_and_universe.log

PRIMARY RESEARCH DESIGN
-----------------------
Signal formation:
    At market close on date t.

Execution:
    Earliest entry at market open on t+1.

Target:
    5-day forward tradable-window open-to-open return

        target_ret_5d(t)
          = AdjOpen(t+6) / AdjOpen(t+1) - 1

    where t+1 and t+6 are MARKET trading dates, not per-stock next observations.

Universe:
    Start from Stage-4 `base_eligible_signal_day`:
        common Shanghai/Shenzhen A-share
        & seasoned >=120 observed trading days
        & not ST
        & not suspended on signal day
        & close >= 5 RMB
        & valid adjustment factor

    Liquidity ranking uses LAGGED information only:

        ADV20_lagged(t)
          = mean(AmountRMB over previous 20 observed stock trading rows)

    Primary liquid universe:
        top N by ADV20_lagged each date among base-eligible stocks.
        Default N = 1500.

    Robustness flags are also emitted for Top 1000 and Top 1500.

IMPORTANT LOOK-AHEAD RULE
-------------------------
`can_buy_next_open` is NOT used to select the signal-day universe or train the model,
because next-day open tradability is future information at close t.

It is retained only as an execution diagnostic for later backtesting.

FEATURES
--------
All are known by close t.

Momentum:
    mom5  = AdjClose_t / AdjClose_{t-5 observed rows} - 1
    mom20 = AdjClose_t / AdjClose_{t-20 observed rows} - 1
    mom60 = AdjClose_t / AdjClose_{t-60 observed rows} - 1

Volatility:
    ret1_obs = AdjClose_t / AdjClose_{previous observed row} - 1
    vol5, vol20, vol60 = rolling standard deviation of ret1_obs
    vol5_vol60 = vol5 / vol60

Volume / amount:
    volume_ratio20 = VolumeShares_t / mean(previous 20 observed rows)
    amount_ratio20 = AmountRMB_t / mean(previous 20 observed rows)

Range:
    range1 = (High - Low) / Close
    range20 = mean(range1 over current + prior 19 observed rows)

Liquidity:
    adv20_lagged = mean(previous 20 observed AmountRMB rows)

NOTE ON SUSPENSIONS
-------------------
The raw daily backbone has no rows on suspension dates. Therefore Stage 5 defines
rolling stock features over the stock's OBSERVED trading rows. A 252-market-day
history buffer is loaded for each target month so the 60-observation features are
not mechanically lost after ordinary suspensions. This is transparent
and conservative for the first research version. Market-calendar dates are still
used for the forward target and execution dates. The project can later add a
calendarized-suspension robustness specification if needed.

CROSS-SECTIONAL PREPROCESSING
-----------------------------
Performed DAILY and ONLY inside the PRIMARY liquid universe:
    1% / 99% winsorization
    z-score standardization (population std, ddof=0)

For each raw feature X:
    X_w   = winsorized feature
    X_z   = z-scored winsorized feature

Rows outside the primary universe keep X_w / X_z as NaN.

INSTALL
-------
pip install pandas numpy pyarrow

RUN
---
python 05_build_signals_and_universe.py

Alternative primary universe:
python 05_build_signals_and_universe.py --top-n 1000

Resume:
Re-run the same command; valid monthly output files are reused.

Force rebuild:
python 05_build_signals_and_universe.py --force
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import platform
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# ======================================================================================
# Defaults
# ======================================================================================

DEFAULT_START = "20090601"
DEFAULT_END = "20251231"
FORMAL_SAMPLE_START = "20100101"
DEFAULT_TOP_N = 1500

FEATURE_HISTORY_BUFFER_MARKET_DAYS = 252
FORWARD_MARKET_DAYS = 6

KEY = ["ts_code", "trade_date"]

RAW_FEATURES = [
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

CORE_MODEL_FEATURES = [
    "mom5",
    "mom20",
    "mom60",
    "vol5",
    "vol20",
    "vol60",
    "vol5_vol60",
    "volume_ratio20",
    "amount_ratio20",
    "range20",
]


# ======================================================================================
# Configuration
# ======================================================================================

@dataclass
class Config:
    start: str
    end: str
    formal_sample_start: str
    top_n: int
    data_root: Path
    winsor_lower: float
    winsor_upper: float
    min_cross_section: int
    force: bool

    @property
    def panel_root(self) -> Path:
        return self.data_root / "processed" / "full_panel"

    @property
    def output_root(self) -> Path:
        return self.data_root / "processed" / "signals"

    @property
    def report_root(self) -> Path:
        return self.data_root / "signal_reports"

    @property
    def trade_cal_path(self) -> Path:
        return (
            self.data_root
            / "raw_full"
            / "trade_cal"
            / f"trade_cal_{self.start}_{self.end}.parquet"
        )


# ======================================================================================
# Generic helpers
# ======================================================================================

def setup_logging(report_root: Path) -> logging.Logger:
    report_root.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("signals_universe")
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
        report_root / "05_build_signals_and_universe.log",
        mode="a",
        encoding="utf-8",
    )
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


def ensure_parquet_engine() -> None:
    try:
        import pyarrow  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "Parquet I/O requires pyarrow.\n"
            "Install with: pip install pyarrow"
        ) from exc


def validate_yyyymmdd(value: str, name: str) -> None:
    try:
        datetime.strptime(value, "%Y%m%d")
    except ValueError as exc:
        raise ValueError(f"{name} must be YYYYMMDD, got {value!r}") from exc


def normalize_date_series(s: pd.Series) -> pd.Series:
    return (
        s.astype("string")
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(8)
    )


def atomic_write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")

    if tmp.exists():
        tmp.unlink()

    df.to_parquet(
        tmp,
        index=False,
        engine="pyarrow",
        compression="snappy",
    )
    tmp.replace(path)


def atomic_write_json(obj: Dict[str, Any], path: Path) -> None:
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


def valid_signal_file(path: Path, top_n: int) -> bool:
    if not path.exists() or path.stat().st_size <= 0:
        return False

    required = [
        "ts_code",
        "trade_date",
        "adv20_lagged",
        "liquid_top1000",
        "liquid_top1500",
        "liquid_universe_primary",
        "mom60",
        "vol60",
        "target_ret_5d",
        "target_entry_trade_date",
        "target_exit_trade_date",
        "stage5_spec_version",
    ]

    try:
        df = pd.read_parquet(path)
    except Exception:
        return False

    if any(c not in df.columns for c in required):
        return False

    # Existing output is valid only if its primary-universe flag was produced
    # under the same requested top_n. We store the chosen N in a column.
    if "primary_top_n" not in df.columns:
        return False

    vals = pd.to_numeric(
        df["primary_top_n"],
        errors="coerce",
    ).dropna().unique()

    versions = df["stage5_spec_version"].dropna().astype(str).unique()
    return (
        len(vals) == 1
        and int(vals[0]) == int(top_n)
        and len(versions) == 1
        and versions[0] == STAGE5_SPEC_VERSION
    )


def package_version(name: str) -> str:
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:
        return "unknown"


def assert_unique_key(
    df: pd.DataFrame,
    keys: Sequence[str],
    *,
    name: str,
) -> None:
    missing = [c for c in keys if c not in df.columns]

    if missing:
        raise ValueError(f"{name} missing key columns: {missing}")

    dup = df.duplicated(list(keys), keep=False)
    n_dup = int(dup.sum())

    if n_dup:
        sample = df.loc[dup, list(keys)].head(20)
        raise ValueError(
            f"{name} has {n_dup} duplicate rows on {list(keys)}.\n"
            f"Sample:\n{sample.to_string(index=False)}"
        )


# ======================================================================================
# Calendar / month utilities
# ======================================================================================

def load_open_dates(
    cfg: Config,
    logger: logging.Logger,
) -> Tuple[List[str], Dict[str, int]]:
    if not cfg.trade_cal_path.exists():
        raise FileNotFoundError(cfg.trade_cal_path)

    cal = pd.read_parquet(cfg.trade_cal_path)

    if not {"cal_date", "is_open"}.issubset(cal.columns):
        raise ValueError("trade_cal missing cal_date/is_open.")

    cal["cal_date"] = normalize_date_series(cal["cal_date"])
    cal["is_open"] = pd.to_numeric(
        cal["is_open"],
        errors="coerce",
    )

    open_dates = sorted(
        cal.loc[
            (cal["is_open"] == 1)
            & (cal["cal_date"] >= cfg.start)
            & (cal["cal_date"] <= cfg.end),
            "cal_date",
        ]
        .dropna()
        .drop_duplicates()
        .tolist()
    )

    if not open_dates:
        raise RuntimeError("No market open dates found.")

    trade_index = {
        d: i
        for i, d in enumerate(open_dates)
    }

    logger.info(
        "TRADE CALENDAR | dates=%d | %s -> %s",
        len(open_dates),
        open_dates[0],
        open_dates[-1],
    )

    return open_dates, trade_index


def build_month_buckets(
    open_dates: Sequence[str],
) -> List[Tuple[int, int, List[str]]]:
    buckets: Dict[Tuple[int, int], List[str]] = {}

    for d in open_dates:
        key = (int(d[:4]), int(d[4:6]))
        buckets.setdefault(key, []).append(d)

    return [
        (year, month, sorted(dates))
        for (year, month), dates in sorted(buckets.items())
    ]


def month_output_path(
    root: Path,
    year: int,
    month: int,
) -> Path:
    return (
        root
        / f"{year:04d}"
        / f"{month:02d}"
        / f"signals_{year:04d}{month:02d}.parquet"
    )


def panel_month_path(
    root: Path,
    year: int,
    month: int,
) -> Path:
    return (
        root
        / f"{year:04d}"
        / f"{month:02d}"
        / f"full_panel_{year:04d}{month:02d}.parquet"
    )


def dates_to_month_keys(
    dates: Sequence[str],
) -> List[Tuple[int, int]]:
    return sorted(
        {
            (int(d[:4]), int(d[4:6]))
            for d in dates
        }
    )


# ======================================================================================
# Buffered panel loading
# ======================================================================================

def load_buffered_panel(
    cfg: Config,
    *,
    buffer_dates: Sequence[str],
    logger: logging.Logger,
) -> pd.DataFrame:
    month_keys = dates_to_month_keys(buffer_dates)
    frames: List[pd.DataFrame] = []

    required_cols = [
        "ts_code",
        "trade_date",
        "open",
        "high",
        "low",
        "close",
        "adj_open",
        "adj_close",
        "volume_shares",
        "amount_rmb",
        "up_limit",
        "down_limit",
        "is_st",
        "is_suspended",
        "base_eligible_signal_day",
        "is_common_a_share",
        "seasoned_120d",
        "in_formal_research_sample",
        "can_buy_next_open",
        "can_sell_next_open",
    ]

    for year, month in month_keys:
        path = panel_month_path(
            cfg.panel_root,
            year,
            month,
        )

        if not path.exists():
            raise FileNotFoundError(
                f"Missing Stage-4 panel month: {path}"
            )

        df = pd.read_parquet(path)

        missing = [
            c for c in required_cols
            if c not in df.columns
        ]
        if missing:
            raise ValueError(
                f"{path} missing required columns: {missing}"
            )

        frames.append(df)

    panel = pd.concat(
        frames,
        ignore_index=True,
        sort=False,
    )

    panel["trade_date"] = normalize_date_series(
        panel["trade_date"]
    )

    date_set = set(buffer_dates)
    panel = panel.loc[
        panel["trade_date"].isin(date_set)
    ].copy()

    panel = panel.sort_values(
        ["ts_code", "trade_date"]
    ).reset_index(drop=True)

    assert_unique_key(
        panel,
        KEY,
        name="buffered full panel",
    )

    logger.info(
        "BUFFER LOAD | months=%d dates=%d rows=%d codes=%d",
        len(month_keys),
        len(buffer_dates),
        len(panel),
        panel["ts_code"].nunique(),
    )

    return panel


# ======================================================================================
# Rolling feature construction — robust NumPy implementation
# ======================================================================================

STAGE5_SPEC_VERSION = "v4_final_positional"


def _rolling_mean_current(
    values: np.ndarray,
    window: int,
) -> np.ndarray:
    """
    Rolling mean over [i-window+1, ..., i], requiring exactly `window`
    finite observations. Returns NaN otherwise.
    """
    values = np.asarray(values, dtype=float)
    n = len(values)
    out = np.full(n, np.nan, dtype=float)

    finite = np.isfinite(values)
    filled = np.where(finite, values, 0.0)

    psum = np.concatenate(([0.0], np.cumsum(filled)))
    pcount = np.concatenate(([0], np.cumsum(finite.astype(np.int64))))

    i = np.arange(n)
    start = i - window + 1
    eligible = start >= 0

    if not eligible.any():
        return out

    ii = i[eligible]
    ss = start[eligible]

    counts = pcount[ii + 1] - pcount[ss]
    sums = psum[ii + 1] - psum[ss]

    ok = counts == window
    out[ii[ok]] = sums[ok] / float(window)

    return out


def _rolling_mean_previous(
    values: np.ndarray,
    window: int,
) -> np.ndarray:
    """
    Mean of the PREVIOUS `window` observations, excluding the current row:
        mean(values[i-window : i])
    Requires all `window` observations to be finite.
    """
    values = np.asarray(values, dtype=float)
    n = len(values)
    out = np.full(n, np.nan, dtype=float)

    finite = np.isfinite(values)
    filled = np.where(finite, values, 0.0)

    psum = np.concatenate(([0.0], np.cumsum(filled)))
    pcount = np.concatenate(([0], np.cumsum(finite.astype(np.int64))))

    i = np.arange(n)
    start = i - window
    eligible = start >= 0

    if not eligible.any():
        return out

    ii = i[eligible]
    ss = start[eligible]

    counts = pcount[ii] - pcount[ss]
    sums = psum[ii] - psum[ss]

    ok = counts == window
    out[ii[ok]] = sums[ok] / float(window)

    return out


def _rolling_std_current(
    values: np.ndarray,
    window: int,
) -> np.ndarray:
    """
    Population rolling standard deviation (ddof=0) over the current and
    previous window-1 values.

    Uses rolling first/second moments rather than pandas GroupBy.rolling.
    Values are small daily returns, so this calculation is numerically stable.
    Small negative variances from floating-point round-off are clamped to zero.
    """
    values = np.asarray(values, dtype=float)
    n = len(values)
    out = np.full(n, np.nan, dtype=float)

    finite = np.isfinite(values)
    filled = np.where(finite, values, 0.0)

    psum = np.concatenate(([0.0], np.cumsum(filled)))
    psq = np.concatenate(([0.0], np.cumsum(filled * filled)))
    pcount = np.concatenate(([0], np.cumsum(finite.astype(np.int64))))

    i = np.arange(n)
    start = i - window + 1
    eligible = start >= 0

    if not eligible.any():
        return out

    ii = i[eligible]
    ss = start[eligible]

    counts = pcount[ii + 1] - pcount[ss]
    sums = psum[ii + 1] - psum[ss]
    sums2 = psq[ii + 1] - psq[ss]

    ok = counts == window
    if not ok.any():
        return out

    means = sums[ok] / float(window)
    variances = sums2[ok] / float(window) - means * means
    variances = np.maximum(variances, 0.0)

    out[ii[ok]] = np.sqrt(variances)
    return out


def construct_raw_features(
    panel: pd.DataFrame,
) -> pd.DataFrame:
    """
    Construct all stock-time rolling features explicitly by security.

    IMPORTANT
    ---------
    This v3 implementation intentionally avoids:
        pandas GroupBy.shift
        pandas GroupBy.rolling
        pandas GroupBy.transform(lambda rolling...)

    Earlier diagnostics found rare month-specific silent degeneracy in those
    paths under the user's environment:
        2021-04: MOM20 became identically zero and AmountRatio20 all missing
        2025-03: VOL5/VOL20/VOL60 became identically zero

    The underlying Stage-4 prices were verified independently to be healthy.
    This implementation therefore uses contiguous per-stock NumPy arrays and
    explicit prefix-sum rolling calculations.
    """
    x = panel.copy()

    x = x.sort_values(
        ["ts_code", "trade_date"]
    ).reset_index(drop=True)

    for c in [
        "adj_close",
        "adj_open",
        "open",
        "high",
        "low",
        "close",
        "volume_shares",
        "amount_rmb",
    ]:
        x[c] = pd.to_numeric(
            x[c],
            errors="coerce",
        )

    n = len(x)

    # Preallocate every result by POSITION, not by pandas index alignment.
    out_arrays: Dict[str, np.ndarray] = {
        "mom5": np.full(n, np.nan, dtype=float),
        "mom20": np.full(n, np.nan, dtype=float),
        "mom60": np.full(n, np.nan, dtype=float),
        "ret1_obs": np.full(n, np.nan, dtype=float),
        "vol5": np.full(n, np.nan, dtype=float),
        "vol20": np.full(n, np.nan, dtype=float),
        "vol60": np.full(n, np.nan, dtype=float),
        "adv20_lagged": np.full(n, np.nan, dtype=float),
        "volume_ratio20": np.full(n, np.nan, dtype=float),
        "amount_ratio20": np.full(n, np.nan, dtype=float),
        "range1": np.full(n, np.nan, dtype=float),
        "range20": np.full(n, np.nan, dtype=float),
    }

    # Because x has a RangeIndex after sorting/reset_index, groupby.indices
    # gives integer row positions that can safely index NumPy arrays.
    groups = x.groupby(
        "ts_code",
        sort=False,
    ).indices

    adj_all = x["adj_close"].to_numpy(dtype=float)
    close_all = x["close"].to_numpy(dtype=float)
    high_all = x["high"].to_numpy(dtype=float)
    low_all = x["low"].to_numpy(dtype=float)
    volume_all = x["volume_shares"].to_numpy(dtype=float)
    amount_all = x["amount_rmb"].to_numpy(dtype=float)

    for _, pos in groups.items():
        pos = np.asarray(pos, dtype=np.int64)

        adj = adj_all[pos]
        close = close_all[pos]
        high = high_all[pos]
        low = low_all[pos]
        volume = volume_all[pos]
        amount = amount_all[pos]

        m = len(pos)

        # --------------------------------------------------------------
        # Momentum: current adjusted close vs h-th previous observed row.
        # --------------------------------------------------------------
        for h in (5, 20, 60):
            arr = np.full(m, np.nan, dtype=float)

            if m > h:
                cur = adj[h:]
                lag = adj[:-h]
                valid = (
                    np.isfinite(cur)
                    & np.isfinite(lag)
                    & (lag != 0)
                )

                vals = np.full(m - h, np.nan, dtype=float)
                vals[valid] = cur[valid] / lag[valid] - 1.0
                arr[h:] = vals

            out_arrays[f"mom{h}"][pos] = arr

        # --------------------------------------------------------------
        # One-observation return.
        # --------------------------------------------------------------
        ret = np.full(m, np.nan, dtype=float)

        if m > 1:
            cur = adj[1:]
            prev = adj[:-1]
            valid = (
                np.isfinite(cur)
                & np.isfinite(prev)
                & (prev != 0)
            )
            vals = np.full(m - 1, np.nan, dtype=float)
            vals[valid] = cur[valid] / prev[valid] - 1.0
            ret[1:] = vals

        out_arrays["ret1_obs"][pos] = ret

        # --------------------------------------------------------------
        # Rolling return volatility.
        # --------------------------------------------------------------
        for w in (5, 20, 60):
            out_arrays[f"vol{w}"][pos] = _rolling_std_current(
                ret,
                w,
            )

        # --------------------------------------------------------------
        # Lagged liquidity/activity means: previous 20 observed rows,
        # excluding current date.
        # --------------------------------------------------------------
        adv20 = _rolling_mean_previous(
            amount,
            20,
        )
        avg_vol20 = _rolling_mean_previous(
            volume,
            20,
        )

        out_arrays["adv20_lagged"][pos] = adv20

        vr = np.full(m, np.nan, dtype=float)
        valid_v = (
            np.isfinite(volume)
            & np.isfinite(avg_vol20)
            & (avg_vol20 > 0)
        )
        vr[valid_v] = (
            volume[valid_v]
            / avg_vol20[valid_v]
        )
        out_arrays["volume_ratio20"][pos] = vr

        ar = np.full(m, np.nan, dtype=float)
        valid_a = (
            np.isfinite(amount)
            & np.isfinite(adv20)
            & (adv20 > 0)
        )
        ar[valid_a] = (
            amount[valid_a]
            / adv20[valid_a]
        )
        out_arrays["amount_ratio20"][pos] = ar

        # --------------------------------------------------------------
        # Intraday range and 20-observation mean range.
        # --------------------------------------------------------------
        r1 = np.full(m, np.nan, dtype=float)
        valid_r = (
            np.isfinite(high)
            & np.isfinite(low)
            & np.isfinite(close)
            & (close > 0)
        )
        r1[valid_r] = (
            (high[valid_r] - low[valid_r])
            / close[valid_r]
        )

        out_arrays["range1"][pos] = r1
        out_arrays["range20"][pos] = _rolling_mean_current(
            r1,
            20,
        )

    # Assign by position once all groups are complete.
    for name, arr in out_arrays.items():
        x[name] = arr

    v5 = pd.to_numeric(x["vol5"], errors="coerce").to_numpy(dtype=float)
    v60 = pd.to_numeric(x["vol60"], errors="coerce").to_numpy(dtype=float)
    ratio = np.full(len(x), np.nan, dtype=float)
    valid_ratio = (
        np.isfinite(v5)
        & np.isfinite(v60)
        & (v60 > 0)
    )
    ratio[valid_ratio] = v5[valid_ratio] / v60[valid_ratio]
    x["vol5_vol60"] = ratio

    return x


# ======================================================================================
# Forward target construction
# ======================================================================================

def map_forward_dates(
    dates: pd.Series,
    *,
    trade_index: Dict[str, int],
    open_dates: Sequence[str],
    steps: int,
) -> pd.Series:
    def one(d: Any) -> Any:
        if pd.isna(d):
            return pd.NA

        ds = str(d)
        idx = trade_index.get(ds)

        if idx is None:
            return pd.NA

        j = idx + steps

        if j >= len(open_dates):
            return pd.NA

        return open_dates[j]

    return dates.map(one).astype("string")


def construct_forward_target(
    panel: pd.DataFrame,
    *,
    trade_index: Dict[str, int],
    open_dates: Sequence[str],
) -> pd.DataFrame:
    x = panel.copy()

    x["target_entry_trade_date"] = map_forward_dates(
        x["trade_date"],
        trade_index=trade_index,
        open_dates=open_dates,
        steps=1,
    )

    x["target_exit_trade_date"] = map_forward_dates(
        x["trade_date"],
        trade_index=trade_index,
        open_dates=open_dates,
        steps=6,
    )

    # --------------------------------------------------------------
    # Entry open lookup: market t+1
    # --------------------------------------------------------------
    entry_lookup = x[
        [
            "ts_code",
            "trade_date",
            "open",
            "adj_open",
            "up_limit",
            "down_limit",
            "is_suspended",
        ]
    ].copy()

    entry_lookup = entry_lookup.rename(
        columns={
            "trade_date": "target_entry_trade_date",
            "open": "target_entry_open",
            "adj_open": "target_entry_adj_open",
            "up_limit": "target_entry_up_limit",
            "down_limit": "target_entry_down_limit",
            "is_suspended": "target_entry_is_suspended",
        }
    )

    x = x.merge(
        entry_lookup,
        on=[
            "ts_code",
            "target_entry_trade_date",
        ],
        how="left",
        validate="many_to_one",
    )

    # --------------------------------------------------------------
    # Exit open lookup: market t+6
    # --------------------------------------------------------------
    exit_lookup = panel[
        [
            "ts_code",
            "trade_date",
            "open",
            "adj_open",
            "up_limit",
            "down_limit",
            "is_suspended",
        ]
    ].copy()

    exit_lookup = exit_lookup.rename(
        columns={
            "trade_date": "target_exit_trade_date",
            "open": "target_exit_open",
            "adj_open": "target_exit_adj_open",
            "up_limit": "target_exit_up_limit",
            "down_limit": "target_exit_down_limit",
            "is_suspended": "target_exit_is_suspended",
        }
    )

    x = x.merge(
        exit_lookup,
        on=[
            "ts_code",
            "target_exit_trade_date",
        ],
        how="left",
        validate="many_to_one",
    )

    # --------------------------------------------------------------
    # Target return
    # --------------------------------------------------------------
    entry_adj = pd.to_numeric(
        x["target_entry_adj_open"],
        errors="coerce",
    ).to_numpy(dtype=float)

    exit_adj = pd.to_numeric(
        x["target_exit_adj_open"],
        errors="coerce",
    ).to_numpy(dtype=float)

    target = np.full(len(x), np.nan, dtype=float)
    valid_target = (
        np.isfinite(entry_adj)
        & np.isfinite(exit_adj)
        & (entry_adj > 0)
    )
    target[valid_target] = (
        exit_adj[valid_target]
        / entry_adj[valid_target]
        - 1.0
    )
    x["target_ret_5d"] = target

    # --------------------------------------------------------------
    # Execution diagnostics
    # --------------------------------------------------------------
    eps = 1e-10

    x["target_entry_has_quote"] = (
        x["target_entry_open"].notna()
    )

    x["target_exit_has_quote"] = (
        x["target_exit_open"].notna()
    )

    # IMPORTANT:
    # This is future information and MUST NOT define the signal-day universe.
    x["target_entry_tradable"] = (
        x["target_entry_trade_date"].notna()
        & x["target_entry_has_quote"]
        & (~x["target_entry_is_suspended"].fillna(False))
        & (
            x["target_entry_up_limit"].isna()
            | (
                x["target_entry_open"]
                < x["target_entry_up_limit"] - eps
            )
        )
    )

    x["target_exit_tradable"] = (
        x["target_exit_trade_date"].notna()
        & x["target_exit_has_quote"]
        & (~x["target_exit_is_suspended"].fillna(False))
        & (
            x["target_exit_down_limit"].isna()
            | (
                x["target_exit_open"]
                > x["target_exit_down_limit"] + eps
            )
        )
    )

    x["target_fully_tradable_5d"] = (
        x["target_entry_tradable"]
        & x["target_exit_tradable"]
        & x["target_ret_5d"].notna()
    )

    return x



def recompute_final_algebraic_columns(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Final safety pass AFTER target-month slicing.

    Recomputes derived algebraic columns from their saved source columns by
    pure NumPy POSITION, so they cannot inherit any pandas index alignment
    behavior from earlier intermediate operations.

    Recomputed:
        vol5_vol60
        amount_ratio20
        target_ret_5d
    """
    x = df.copy().reset_index(drop=True)

    # VOL5 / VOL60
    v5 = pd.to_numeric(x["vol5"], errors="coerce").to_numpy(dtype=float)
    v60 = pd.to_numeric(x["vol60"], errors="coerce").to_numpy(dtype=float)
    ratio = np.full(len(x), np.nan, dtype=float)
    ok = np.isfinite(v5) & np.isfinite(v60) & (v60 > 0)
    ratio[ok] = v5[ok] / v60[ok]
    x["vol5_vol60"] = ratio

    # Amount / lagged ADV20
    amount = pd.to_numeric(
        x["amount_rmb"], errors="coerce"
    ).to_numpy(dtype=float)
    adv = pd.to_numeric(
        x["adv20_lagged"], errors="coerce"
    ).to_numpy(dtype=float)
    ar = np.full(len(x), np.nan, dtype=float)
    ok = np.isfinite(amount) & np.isfinite(adv) & (adv > 0)
    ar[ok] = amount[ok] / adv[ok]
    x["amount_ratio20"] = ar

    # t+1 -> t+6 adjusted-open return
    entry = pd.to_numeric(
        x["target_entry_adj_open"], errors="coerce"
    ).to_numpy(dtype=float)
    exit_ = pd.to_numeric(
        x["target_exit_adj_open"], errors="coerce"
    ).to_numpy(dtype=float)

    target = np.full(len(x), np.nan, dtype=float)
    ok = np.isfinite(entry) & np.isfinite(exit_) & (entry > 0)
    target[ok] = exit_[ok] / entry[ok] - 1.0
    x["target_ret_5d"] = target

    # Update downstream target flag because target_ret_5d was recomputed.
    if {
        "target_entry_tradable",
        "target_exit_tradable",
    }.issubset(x.columns):
        x["target_fully_tradable_5d"] = (
            x["target_entry_tradable"].fillna(False)
            & x["target_exit_tradable"].fillna(False)
            & x["target_ret_5d"].notna()
        )

    return x


# ======================================================================================
# Dynamic liquidity universe
# ======================================================================================

def build_daily_liquid_universe(
    panel: pd.DataFrame,
    *,
    top_n: int,
) -> pd.DataFrame:
    x = panel.copy()

    candidate = (
        x["base_eligible_signal_day"].fillna(False)
        & x["adv20_lagged"].notna()
        & np.isfinite(x["adv20_lagged"])
        & (x["adv20_lagged"] > 0)
    )

    x["liquidity_candidate"] = candidate

    # Deterministic tie-breaking: panel is sorted by date and code before ranking.
    x = x.sort_values(
        ["trade_date", "ts_code"]
    ).reset_index(drop=True)

    rank = pd.Series(
        np.nan,
        index=x.index,
        dtype="float64",
    )

    candidate_idx = x.index[x["liquidity_candidate"]]

    if len(candidate_idx):
        rank.loc[candidate_idx] = (
            x.loc[candidate_idx]
            .groupby(
                "trade_date",
                sort=False,
            )["adv20_lagged"]
            .rank(
                ascending=False,
                method="first",
            )
        )

    x["adv20_rank"] = rank

    # Percentile rank is computed within candidates; highest liquidity approaches 1.
    pct = pd.Series(
        np.nan,
        index=x.index,
        dtype="float64",
    )

    if len(candidate_idx):
        pct.loc[candidate_idx] = (
            x.loc[candidate_idx]
            .groupby(
                "trade_date",
                sort=False,
            )["adv20_lagged"]
            .rank(
                ascending=True,
                pct=True,
                method="average",
            )
        )

    x["adv20_pct_rank"] = pct

    x["liquid_top1000"] = (
        x["liquidity_candidate"]
        & (x["adv20_rank"] <= 1000)
    )

    x["liquid_top1500"] = (
        x["liquidity_candidate"]
        & (x["adv20_rank"] <= 1500)
    )

    x["primary_top_n"] = int(top_n)

    x["liquid_universe_primary"] = (
        x["liquidity_candidate"]
        & (x["adv20_rank"] <= int(top_n))
    )

    return x


# ======================================================================================
# Cross-sectional preprocessing
# ======================================================================================

def preprocess_one_feature(
    df: pd.DataFrame,
    *,
    feature: str,
    universe_col: str,
    lower_q: float,
    upper_q: float,
    min_cross_section: int,
) -> Tuple[pd.Series, pd.Series]:
    wins = pd.Series(
        np.nan,
        index=df.index,
        dtype="float64",
    )

    z = pd.Series(
        np.nan,
        index=df.index,
        dtype="float64",
    )

    universe = (
        df[universe_col].fillna(False)
        & df[feature].notna()
        & np.isfinite(df[feature])
    )

    work = df.loc[
        universe,
        ["trade_date", feature],
    ].copy()

    if work.empty:
        return wins, z

    grouped = work.groupby(
        "trade_date",
        sort=False,
    )

    counts = grouped[feature].transform("count")

    lo = grouped[feature].transform(
        lambda s: s.quantile(lower_q)
    )
    hi = grouped[feature].transform(
        lambda s: s.quantile(upper_q)
    )

    clipped = work[feature].clip(
        lower=lo,
        upper=hi,
    )

    # Standardize clipped value within each cross-section.
    temp = pd.DataFrame(
        {
            "trade_date": work["trade_date"],
            "clipped": clipped,
        },
        index=work.index,
    )

    g2 = temp.groupby(
        "trade_date",
        sort=False,
    )["clipped"]

    mu = g2.transform("mean")
    sd = g2.transform(
        lambda s: s.std(ddof=0)
    )

    valid = (
        (counts >= min_cross_section)
        & sd.notna()
        & (sd > 0)
    )

    wins.loc[work.index] = clipped
    z.loc[work.index[valid]] = (
        (
            clipped.loc[valid]
            - mu.loc[valid]
        )
        / sd.loc[valid]
    )

    return wins, z


def cross_sectional_preprocess(
    panel: pd.DataFrame,
    *,
    cfg: Config,
) -> pd.DataFrame:
    x = panel.copy()

    for feature in RAW_FEATURES:
        w, z = preprocess_one_feature(
            x,
            feature=feature,
            universe_col="liquid_universe_primary",
            lower_q=cfg.winsor_lower,
            upper_q=cfg.winsor_upper,
            min_cross_section=cfg.min_cross_section,
        )

        x[f"{feature}_w"] = w
        x[f"{feature}_z"] = z

    return x


# ======================================================================================
# Model-ready flags
# ======================================================================================

def add_model_ready_flags(
    panel: pd.DataFrame,
) -> pd.DataFrame:
    x = panel.copy()

    raw_complete = x[
        CORE_MODEL_FEATURES
    ].notna().all(axis=1)

    z_cols = [
        f"{f}_z"
        for f in CORE_MODEL_FEATURES
    ]

    z_complete = x[
        z_cols
    ].notna().all(axis=1)

    x["core_raw_features_complete"] = raw_complete
    x["core_z_features_complete"] = z_complete

    x["stage5_spec_version"] = STAGE5_SPEC_VERSION

    # Prediction research sample.
    # DO NOT require future tradability here: that would condition the statistical
    # training sample on future execution outcomes.
    x["model_ready_5d"] = (
        x["in_formal_research_sample"].fillna(False)
        & x["liquid_universe_primary"].fillna(False)
        & x["core_z_features_complete"]
        & x["target_ret_5d"].notna()
    )

    # Separate execution-ready flag for later portfolio backtesting.
    x["backtest_execution_ready_5d"] = (
        x["in_formal_research_sample"].fillna(False)
        & x["liquid_universe_primary"].fillna(False)
        & x["core_z_features_complete"]
        & x["target_fully_tradable_5d"].fillna(False)
    )

    return x


# ======================================================================================
# Diagnostics
# ======================================================================================

def build_daily_universe_summary(
    df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for trade_date, g in df.groupby(
        "trade_date",
        sort=True,
    ):
        cand = g["liquidity_candidate"].fillna(False)
        primary = g["liquid_universe_primary"].fillna(False)

        row = {
            "trade_date": trade_date,
            "n_daily_rows": int(len(g)),
            "n_common_a_share": int(
                g["is_common_a_share"].fillna(False).sum()
            ),
            "n_base_eligible": int(
                g["base_eligible_signal_day"].fillna(False).sum()
            ),
            "n_liquidity_candidates": int(cand.sum()),
            "n_top1000": int(
                g["liquid_top1000"].fillna(False).sum()
            ),
            "n_top1500": int(
                g["liquid_top1500"].fillna(False).sum()
            ),
            "n_primary_universe": int(primary.sum()),
            "n_model_ready_5d": int(
                g["model_ready_5d"].fillna(False).sum()
            ),
            "n_execution_ready_5d": int(
                g["backtest_execution_ready_5d"]
                .fillna(False)
                .sum()
            ),
            "median_adv20_rmb_primary": (
                float(
                    g.loc[
                        primary,
                        "adv20_lagged",
                    ].median()
                )
                if primary.any()
                else np.nan
            ),
            "min_adv20_rmb_primary": (
                float(
                    g.loc[
                        primary,
                        "adv20_lagged",
                    ].min()
                )
                if primary.any()
                else np.nan
            ),
        }

        rows.append(row)

    return pd.DataFrame(rows)


def build_monthly_quality(
    df: pd.DataFrame,
    *,
    year: int,
    month: int,
) -> Dict[str, Any]:
    formal = df["in_formal_research_sample"].fillna(False)
    primary = df["liquid_universe_primary"].fillna(False)
    formal_primary = formal & primary

    row: Dict[str, Any] = {
        "month": f"{year:04d}-{month:02d}",
        "rows": int(len(df)),
        "unique_codes": int(
            df["ts_code"].nunique()
        ),
        "duplicate_key_rows": int(
            df.duplicated(KEY, keep=False).sum()
        ),
        "formal_rows": int(formal.sum()),
        "base_eligible_rows": int(
            df["base_eligible_signal_day"]
            .fillna(False)
            .sum()
        ),
        "liquidity_candidate_rows": int(
            df["liquidity_candidate"]
            .fillna(False)
            .sum()
        ),
        "primary_universe_rows": int(primary.sum()),
        "model_ready_rows": int(
            df["model_ready_5d"]
            .fillna(False)
            .sum()
        ),
        "execution_ready_rows": int(
            df["backtest_execution_ready_5d"]
            .fillna(False)
            .sum()
        ),
    }

    if formal_primary.any():
        row["target_5d_nonmissing_rate_primary"] = float(
            df.loc[
                formal_primary,
                "target_ret_5d",
            ].notna().mean()
        )
        row["entry_tradable_rate_primary"] = float(
            df.loc[
                formal_primary,
                "target_entry_tradable",
            ].mean()
        )
        row["exit_tradable_rate_primary"] = float(
            df.loc[
                formal_primary,
                "target_exit_tradable",
            ].mean()
        )
        row["core_z_complete_rate_primary"] = float(
            df.loc[
                formal_primary,
                "core_z_features_complete",
            ].mean()
        )
    else:
        row["target_5d_nonmissing_rate_primary"] = np.nan
        row["entry_tradable_rate_primary"] = np.nan
        row["exit_tradable_rate_primary"] = np.nan
        row["core_z_complete_rate_primary"] = np.nan

    # Feature infinities must never survive.
    inf_count = 0
    for c in RAW_FEATURES:
        values = pd.to_numeric(
            df[c],
            errors="coerce",
        )
        inf_count += int(
            np.isinf(values.to_numpy()).sum()
        )

    row["raw_feature_infinite_cells"] = inf_count

    # --------------------------------------------------------------
    # Hard degeneracy QA inside the formal primary universe.
    # A real cross-sectional market factor must not be all missing or
    # identically constant for an entire formal-sample month.
    # --------------------------------------------------------------
    degenerate_features: List[str] = []

    if formal_primary.any():
        for c in CORE_MODEL_FEATURES:
            s = (
                pd.to_numeric(
                    df.loc[formal_primary, c],
                    errors="coerce",
                )
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
            )

            if len(s) == 0:
                degenerate_features.append(
                    f"{c}:all_missing"
                )
                continue

            if len(s) >= 100:
                sd = float(s.std(ddof=0))
                if (not np.isfinite(sd)) or sd <= 1e-12:
                    degenerate_features.append(
                        f"{c}:constant"
                    )

    row["degenerate_core_feature_count"] = len(
        degenerate_features
    )
    row["degenerate_core_features"] = "|".join(
        degenerate_features
    )

    # --------------------------------------------------------------
    # Algebraic QA:
    # amount_ratio20 MUST equal amount_rmb / adv20_lagged wherever
    # all values exist and ADV20 is positive.
    # --------------------------------------------------------------
    ratio_mask = (
        df["amount_ratio20"].notna()
        & df["amount_rmb"].notna()
        & df["adv20_lagged"].notna()
        & (df["adv20_lagged"] > 0)
    )

    if ratio_mask.any():
        implied = (
            df.loc[ratio_mask, "amount_rmb"]
            / df.loc[ratio_mask, "adv20_lagged"]
        )
        ratio_err = (
            df.loc[ratio_mask, "amount_ratio20"]
            - implied
        ).abs()
        row["amount_ratio20_max_identity_error"] = float(
            ratio_err.max()
        )
    else:
        row["amount_ratio20_max_identity_error"] = np.nan

    # --------------------------------------------------------------
    # VOL ratio identity QA.
    # --------------------------------------------------------------
    v5 = pd.to_numeric(df["vol5"], errors="coerce").to_numpy(dtype=float)
    v60 = pd.to_numeric(df["vol60"], errors="coerce").to_numpy(dtype=float)
    stored_vratio = pd.to_numeric(
        df["vol5_vol60"], errors="coerce"
    ).to_numpy(dtype=float)

    implied_mask = np.isfinite(v5) & np.isfinite(v60) & (v60 > 0)
    implied_vratio = np.full(len(df), np.nan, dtype=float)
    implied_vratio[implied_mask] = v5[implied_mask] / v60[implied_mask]

    missing_vratio_when_defined = int(
        (implied_mask & ~np.isfinite(stored_vratio)).sum()
    )
    row["vol5_vol60_missing_when_defined"] = missing_vratio_when_defined

    both_vratio = implied_mask & np.isfinite(stored_vratio)
    if both_vratio.any():
        row["vol5_vol60_max_identity_error"] = float(
            np.max(
                np.abs(
                    stored_vratio[both_vratio]
                    - implied_vratio[both_vratio]
                )
            )
        )
    else:
        row["vol5_vol60_max_identity_error"] = np.nan

    # --------------------------------------------------------------
    # Forward-target identity QA.
    # --------------------------------------------------------------
    entry = pd.to_numeric(
        df["target_entry_adj_open"], errors="coerce"
    ).to_numpy(dtype=float)
    exit_ = pd.to_numeric(
        df["target_exit_adj_open"], errors="coerce"
    ).to_numpy(dtype=float)
    stored_target = pd.to_numeric(
        df["target_ret_5d"], errors="coerce"
    ).to_numpy(dtype=float)

    implied_target_mask = (
        np.isfinite(entry)
        & np.isfinite(exit_)
        & (entry > 0)
    )
    implied_target = np.full(len(df), np.nan, dtype=float)
    implied_target[implied_target_mask] = (
        exit_[implied_target_mask]
        / entry[implied_target_mask]
        - 1.0
    )

    row["target_missing_when_defined"] = int(
        (
            implied_target_mask
            & ~np.isfinite(stored_target)
        ).sum()
    )

    both_target = (
        implied_target_mask
        & np.isfinite(stored_target)
    )
    if both_target.any():
        row["target_ret_5d_max_identity_error"] = float(
            np.max(
                np.abs(
                    stored_target[both_target]
                    - implied_target[both_target]
                )
            )
        )
    else:
        row["target_ret_5d_max_identity_error"] = np.nan

    # Exact -100% is impossible when both adjusted opens are finite and positive.
    row["target_ret_le_minus_one_rows"] = int(
        (
            np.isfinite(stored_target)
            & (stored_target <= -1.0)
        ).sum()
    )

    # Monthly target degeneracy in the formal primary universe.
    primary_target = pd.to_numeric(
        df.loc[formal_primary, "target_ret_5d"],
        errors="coerce",
    ).replace([np.inf, -np.inf], np.nan).dropna()

    if len(primary_target) >= 100:
        row["target_ret_5d_std_primary"] = float(
            primary_target.std(ddof=0)
        )
    else:
        row["target_ret_5d_std_primary"] = np.nan

    # Volatility cannot be negative.
    neg_vol = 0
    for c in ["vol5", "vol20", "vol60"]:
        neg_vol += int(
            (
                pd.to_numeric(
                    df[c],
                    errors="coerce",
                ) < -1e-15
            )
            .fillna(False)
            .sum()
        )

    row["negative_volatility_cells"] = neg_vol

    hard_fail = (
        row["duplicate_key_rows"] > 0
        or row["raw_feature_infinite_cells"] > 0
        or row["degenerate_core_feature_count"] > 0
        or row["negative_volatility_cells"] > 0
        or row["vol5_vol60_missing_when_defined"] > 0
        or row["target_missing_when_defined"] > 0
        or row["target_ret_le_minus_one_rows"] > 0
        or (
            pd.notna(row["target_ret_5d_std_primary"])
            and row["target_ret_5d_std_primary"] <= 1e-12
        )
        or (
            pd.notna(
                row["amount_ratio20_max_identity_error"]
            )
            and row[
                "amount_ratio20_max_identity_error"
            ] > 1e-10
        )
        or (
            pd.notna(
                row["vol5_vol60_max_identity_error"]
            )
            and row["vol5_vol60_max_identity_error"] > 1e-12
        )
        or (
            pd.notna(
                row["target_ret_5d_max_identity_error"]
            )
            and row["target_ret_5d_max_identity_error"] > 1e-12
        )
    )

    row["status"] = (
        "FAIL" if hard_fail else "PASS"
    )

    return row


def update_missingness(
    df: pd.DataFrame,
    accumulator: Dict[str, Dict[str, Any]],
) -> None:
    n = len(df)

    tracked = [
        "adv20_lagged",
        *RAW_FEATURES,
        *[f"{f}_w" for f in RAW_FEATURES],
        *[f"{f}_z" for f in RAW_FEATURES],
        "target_entry_adj_open",
        "target_exit_adj_open",
        "target_ret_5d",
    ]

    for col in tracked:
        if col not in df.columns:
            continue

        if col not in accumulator:
            accumulator[col] = {
                "column": col,
                "total_rows": 0,
                "missing_cells": 0,
            }

        accumulator[col]["total_rows"] += n
        accumulator[col]["missing_cells"] += int(
            df[col].isna().sum()
        )


def finalize_missingness(
    accumulator: Dict[str, Dict[str, Any]],
) -> pd.DataFrame:
    rows = []

    for col, x in accumulator.items():
        total = int(x["total_rows"])
        miss = int(x["missing_cells"])

        rows.append(
            {
                "column": col,
                "total_rows": total,
                "missing_cells": miss,
                "missing_rate": (
                    miss / total
                    if total
                    else np.nan
                ),
                "nonmissing_cells": total - miss,
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            ["missing_rate", "column"],
            ascending=[False, True],
        )
        .reset_index(drop=True)
    )


def update_feature_summary(
    df: pd.DataFrame,
    rows: List[Dict[str, Any]],
    *,
    month_label: str,
) -> None:
    primary = df["liquid_universe_primary"].fillna(False)

    for feature in RAW_FEATURES:
        s = pd.to_numeric(
            df.loc[primary, feature],
            errors="coerce",
        ).replace(
            [np.inf, -np.inf],
            np.nan,
        ).dropna()

        if s.empty:
            rows.append(
                {
                    "month": month_label,
                    "feature": feature,
                    "n": 0,
                    "mean": np.nan,
                    "std": np.nan,
                    "p01": np.nan,
                    "p50": np.nan,
                    "p99": np.nan,
                }
            )
            continue

        rows.append(
            {
                "month": month_label,
                "feature": feature,
                "n": int(len(s)),
                "mean": float(s.mean()),
                "std": float(s.std(ddof=0)),
                "p01": float(s.quantile(0.01)),
                "p50": float(s.quantile(0.50)),
                "p99": float(s.quantile(0.99)),
            }
        )


def build_target_quality_row(
    df: pd.DataFrame,
    *,
    month_label: str,
) -> Dict[str, Any]:
    formal_primary = (
        df["in_formal_research_sample"].fillna(False)
        & df["liquid_universe_primary"].fillna(False)
    )

    g = df.loc[formal_primary].copy()

    if g.empty:
        return {
            "month": month_label,
            "n_primary": 0,
            "n_target_nonmissing": 0,
            "target_nonmissing_rate": np.nan,
            "n_entry_tradable": 0,
            "n_exit_tradable": 0,
            "n_fully_tradable": 0,
            "target_mean": np.nan,
            "target_std": np.nan,
            "target_p01": np.nan,
            "target_p50": np.nan,
            "target_p99": np.nan,
        }

    target = pd.to_numeric(
        g["target_ret_5d"],
        errors="coerce",
    ).replace(
        [np.inf, -np.inf],
        np.nan,
    )

    nonmiss = target.dropna()

    return {
        "month": month_label,
        "n_primary": int(len(g)),
        "n_target_nonmissing": int(target.notna().sum()),
        "target_nonmissing_rate": float(target.notna().mean()),
        "n_entry_tradable": int(
            g["target_entry_tradable"]
            .fillna(False)
            .sum()
        ),
        "n_exit_tradable": int(
            g["target_exit_tradable"]
            .fillna(False)
            .sum()
        ),
        "n_fully_tradable": int(
            g["target_fully_tradable_5d"]
            .fillna(False)
            .sum()
        ),
        "target_mean": (
            float(nonmiss.mean())
            if len(nonmiss)
            else np.nan
        ),
        "target_std": (
            float(nonmiss.std(ddof=0))
            if len(nonmiss)
            else np.nan
        ),
        "target_p01": (
            float(nonmiss.quantile(0.01))
            if len(nonmiss)
            else np.nan
        ),
        "target_p50": (
            float(nonmiss.quantile(0.50))
            if len(nonmiss)
            else np.nan
        ),
        "target_p99": (
            float(nonmiss.quantile(0.99))
            if len(nonmiss)
            else np.nan
        ),
    }


# ======================================================================================
# One-month build
# ======================================================================================

def build_one_target_month(
    cfg: Config,
    *,
    year: int,
    month: int,
    target_dates: Sequence[str],
    open_dates: Sequence[str],
    trade_index: Dict[str, int],
    logger: logging.Logger,
) -> pd.DataFrame:
    first_idx = trade_index[target_dates[0]]
    last_idx = trade_index[target_dates[-1]]

    buffer_start_idx = max(
        0,
        first_idx - FEATURE_HISTORY_BUFFER_MARKET_DAYS,
    )
    buffer_end_idx = min(
        len(open_dates) - 1,
        last_idx + FORWARD_MARKET_DAYS,
    )

    buffer_dates = open_dates[
        buffer_start_idx : buffer_end_idx + 1
    ]

    x = load_buffered_panel(
        cfg,
        buffer_dates=buffer_dates,
        logger=logger,
    )

    # 1) rolling raw features
    x = construct_raw_features(x)

    # 2) forward t+1 to t+6 target using MARKET dates
    x = construct_forward_target(
        x,
        trade_index=trade_index,
        open_dates=open_dates,
    )

    # Keep target month only AFTER features and targets have been built.
    target_set = set(target_dates)

    out = x.loc[
        x["trade_date"].isin(target_set)
    ].copy()

    # HARD QA: every market-open target date must have at least one daily row.
    # This catches the Stage-3 failure mode where an empty Parquet partition
    # existed and therefore looked "complete" by filename alone.
    observed_target_dates = set(out["trade_date"].astype(str).unique())
    missing_market_dates = sorted(target_set - observed_target_dates)
    if missing_market_dates:
        raise RuntimeError(
            f"{year:04d}-{month:02d}: market-open dates with ZERO panel rows: "
            f"{missing_market_dates}. Repair raw data before proceeding."
        )

    out = out.sort_values(
        ["trade_date", "ts_code"]
    ).reset_index(drop=True)

    # Final positional recomputation of algebraically derived columns.
    out = recompute_final_algebraic_columns(out)

    # 3) dynamic liquidity universe
    out = build_daily_liquid_universe(
        out,
        top_n=cfg.top_n,
    )

    # 4) daily winsorization / z-score inside primary universe
    out = cross_sectional_preprocess(
        out,
        cfg=cfg,
    )

    # 5) model/backtest readiness
    out = add_model_ready_flags(out)

    assert_unique_key(
        out,
        KEY,
        name=f"signals {year}-{month:02d}",
    )

    return out


# ======================================================================================
# CLI
# ======================================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Build dynamic liquid universe, price-volume signals, "
            "and 5-day forward target."
        )
    )

    p.add_argument(
        "--start",
        default=DEFAULT_START,
    )
    p.add_argument(
        "--end",
        default=DEFAULT_END,
    )
    p.add_argument(
        "--formal-sample-start",
        default=FORMAL_SAMPLE_START,
    )
    p.add_argument(
        "--data-root",
        default="data",
    )
    p.add_argument(
        "--top-n",
        type=int,
        default=DEFAULT_TOP_N,
        help="Primary ADV20 liquidity universe size (default 1500)",
    )
    p.add_argument(
        "--winsor-lower",
        type=float,
        default=0.01,
    )
    p.add_argument(
        "--winsor-upper",
        type=float,
        default=0.99,
    )
    p.add_argument(
        "--min-cross-section",
        type=int,
        default=30,
        help="Minimum names needed to z-score a daily cross-section.",
    )
    p.add_argument(
        "--force",
        action="store_true",
    )

    return p.parse_args()


# ======================================================================================
# Main
# ======================================================================================

def main() -> int:
    args = parse_args()

    validate_yyyymmdd(args.start, "--start")
    validate_yyyymmdd(args.end, "--end")
    validate_yyyymmdd(
        args.formal_sample_start,
        "--formal-sample-start",
    )

    if args.start > args.end:
        raise ValueError("--start must be <= --end")

    if args.top_n <= 0:
        raise ValueError("--top-n must be positive.")

    if not (
        0 <= args.winsor_lower
        < args.winsor_upper
        <= 1
    ):
        raise ValueError(
            "Require 0 <= winsor_lower < winsor_upper <= 1."
        )

    ensure_parquet_engine()

    cfg = Config(
        start=args.start,
        end=args.end,
        formal_sample_start=args.formal_sample_start,
        top_n=int(args.top_n),
        data_root=Path(args.data_root),
        winsor_lower=float(args.winsor_lower),
        winsor_upper=float(args.winsor_upper),
        min_cross_section=max(
            2,
            int(args.min_cross_section),
        ),
        force=bool(args.force),
    )

    cfg.output_root.mkdir(
        parents=True,
        exist_ok=True,
    )
    cfg.report_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    logger = setup_logging(
        cfg.report_root
    )

    logger.info("=" * 104)
    logger.info("STAGE 5 | SIGNALS + DYNAMIC LIQUID UNIVERSE")
    logger.info(
        "Raw period: %s -> %s",
        cfg.start,
        cfg.end,
    )
    logger.info(
        "Formal sample starts: %s",
        cfg.formal_sample_start,
    )
    logger.info(
        "Primary liquidity universe: ADV20 Top %d",
        cfg.top_n,
    )
    logger.info(
        "Winsorization: %.2f%% / %.2f%%",
        100 * cfg.winsor_lower,
        100 * cfg.winsor_upper,
    )
    logger.info(
        "Panel root: %s",
        cfg.panel_root.resolve(),
    )
    logger.info(
        "Output root: %s",
        cfg.output_root.resolve(),
    )
    logger.info(
        "force=%s",
        cfg.force,
    )
    logger.info("=" * 104)

    try:
        open_dates, trade_index = load_open_dates(
            cfg,
            logger,
        )

        buckets = build_month_buckets(
            open_dates
        )

        monthly_quality_rows: List[Dict[str, Any]] = []
        daily_summary_frames: List[pd.DataFrame] = []
        feature_summary_rows: List[Dict[str, Any]] = []
        target_quality_rows: List[Dict[str, Any]] = []
        missing_acc: Dict[str, Dict[str, Any]] = {}

        total_rows = 0
        total_model_ready = 0
        total_execution_ready = 0

        for i, (year, month, target_dates) in enumerate(
            buckets,
            1,
        ):
            month_label = f"{year:04d}-{month:02d}"

            out_path = month_output_path(
                cfg.output_root,
                year,
                month,
            )

            logger.info(
                "MONTH %d/%d | %s | target_dates=%d",
                i,
                len(buckets),
                month_label,
                len(target_dates),
            )

            if (
                valid_signal_file(
                    out_path,
                    cfg.top_n,
                )
                and not cfg.force
            ):
                logger.info(
                    "REUSE %s",
                    out_path,
                )

                df = pd.read_parquet(
                    out_path
                )

            else:
                df = build_one_target_month(
                    cfg,
                    year=year,
                    month=month,
                    target_dates=target_dates,
                    open_dates=open_dates,
                    trade_index=trade_index,
                    logger=logger,
                )

                atomic_write_parquet(
                    df,
                    out_path,
                )

                logger.info(
                    "SAVE %s | rows=%d codes=%d primary=%d model_ready=%d",
                    out_path,
                    len(df),
                    df["ts_code"].nunique(),
                    int(
                        df[
                            "liquid_universe_primary"
                        ].sum()
                    ),
                    int(
                        df["model_ready_5d"].sum()
                    ),
                )

            q = build_monthly_quality(
                df,
                year=year,
                month=month,
            )
            monthly_quality_rows.append(q)

            dsum = build_daily_universe_summary(
                df
            )
            daily_summary_frames.append(
                dsum
            )

            update_missingness(
                df,
                missing_acc,
            )

            update_feature_summary(
                df,
                feature_summary_rows,
                month_label=month_label,
            )

            target_quality_rows.append(
                build_target_quality_row(
                    df,
                    month_label=month_label,
                )
            )

            total_rows += len(df)
            total_model_ready += int(
                df["model_ready_5d"].sum()
            )
            total_execution_ready += int(
                df[
                    "backtest_execution_ready_5d"
                ].sum()
            )

            if q["status"] == "FAIL":
                logger.error(
                    "MONTHLY QA FAIL | %s | %s",
                    month_label,
                    q,
                )
            else:
                logger.info(
                    "MONTHLY QA PASS | %s | "
                    "primary=%d model_ready=%d target=%.4f z_complete=%.4f",
                    month_label,
                    q["primary_universe_rows"],
                    q["model_ready_rows"],
                    q["target_5d_nonmissing_rate_primary"]
                    if pd.notna(
                        q["target_5d_nonmissing_rate_primary"]
                    )
                    else float("nan"),
                    q["core_z_complete_rate_primary"]
                    if pd.notna(
                        q["core_z_complete_rate_primary"]
                    )
                    else float("nan"),
                )

            del df, dsum
            gc.collect()

        # ------------------------------------------------------------------
        # Reports
        # ------------------------------------------------------------------
        monthly_quality_df = pd.DataFrame(
            monthly_quality_rows
        )

        daily_summary_df = pd.concat(
            daily_summary_frames,
            ignore_index=True,
            sort=False,
        )

        missingness_df = finalize_missingness(
            missing_acc
        )

        feature_summary_df = pd.DataFrame(
            feature_summary_rows
        )

        target_quality_df = pd.DataFrame(
            target_quality_rows
        )

        monthly_quality_path = (
            cfg.report_root
            / "signal_monthly_quality.csv"
        )

        daily_summary_path = (
            cfg.report_root
            / "signal_daily_universe_summary.csv"
        )

        missingness_path = (
            cfg.report_root
            / "signal_feature_missingness.csv"
        )

        feature_summary_path = (
            cfg.report_root
            / "signal_feature_summary.csv"
        )

        target_quality_path = (
            cfg.report_root
            / "signal_target_quality.csv"
        )

        monthly_quality_df.to_csv(
            monthly_quality_path,
            index=False,
            encoding="utf-8-sig",
        )

        daily_summary_df.to_csv(
            daily_summary_path,
            index=False,
            encoding="utf-8-sig",
        )

        missingness_df.to_csv(
            missingness_path,
            index=False,
            encoding="utf-8-sig",
        )

        feature_summary_df.to_csv(
            feature_summary_path,
            index=False,
            encoding="utf-8-sig",
        )

        target_quality_df.to_csv(
            target_quality_path,
            index=False,
            encoding="utf-8-sig",
        )

        failed_months = monthly_quality_df.loc[
            monthly_quality_df["status"] == "FAIL"
        ]

        metadata = {
            "project": (
                "China A-Share Cross-Sectional Alpha Research"
            ),
            "script": "05_build_signals_and_universe_v4.py",
            "generated_at_utc": datetime.now(
                timezone.utc
            ).isoformat(),
            "raw_start": cfg.start,
            "raw_end": cfg.end,
            "formal_sample_start": cfg.formal_sample_start,
            "open_market_dates": len(open_dates),
            "months_processed": len(buckets),
            "primary_top_n": cfg.top_n,
            "winsor_lower": cfg.winsor_lower,
            "winsor_upper": cfg.winsor_upper,
            "min_cross_section": cfg.min_cross_section,
            "feature_history_buffer_market_days": FEATURE_HISTORY_BUFFER_MARKET_DAYS,
            "forward_market_days_buffer": FORWARD_MARKET_DAYS,
            "total_output_rows": total_rows,
            "total_model_ready_rows": total_model_ready,
            "total_execution_ready_rows": total_execution_ready,
            "monthly_qa_fail_count": int(
                len(failed_months)
            ),
            "raw_features": RAW_FEATURES,
            "core_model_features": CORE_MODEL_FEATURES,
            "target_definition": (
                "AdjOpen(t+6) / AdjOpen(t+1) - 1, "
                "where dates are global market trading dates."
            ),
            "liquidity_definition": (
                "ADV20_lagged = mean AmountRMB over previous "
                "20 observed security trading rows."
            ),
            "primary_universe_definition": (
                "base_eligible_signal_day & valid positive ADV20_lagged "
                f"& daily ADV20 rank <= {cfg.top_n}"
            ),
            "preprocessing_definition": (
                f"daily {100*cfg.winsor_lower:.1f}%/"
                f"{100*cfg.winsor_upper:.1f}% winsorization + "
                "daily z-score within primary liquid universe only"
            ),
            "important_no_lookahead_notes": [
                "ADV20 is lagged and excludes signal-day amount.",
                "Cross-sectional preprocessing is done after universe selection.",
                "Next-day can_buy/tradability is not used to select or train the signal-day universe.",
                "Future tradability is retained only for later execution/backtest diagnostics.",
                "The forward target uses market t+1 entry and market t+6 exit dates.",
            ],
            "stage5_spec_version": STAGE5_SPEC_VERSION,
            "final_algebraic_recompute": (
                "vol5_vol60, amount_ratio20, and target_ret_5d are recomputed "
                "after target-month slicing using positional NumPy arrays, then "
                "validated by exact algebraic identity QA."
            ),
            "rolling_window_note": (
                "Stock rolling features use observed security trading rows and are computed "
                "with explicit per-security NumPy prefix-sum routines, avoiding pandas "
                "GroupBy.rolling/shift alignment paths. "
                "Each target month loads a 252-market-day historical buffer so "
                "ordinary suspensions do not create artificial month-boundary "
                "missingness for 60-observation features. Securities with fewer "
                "than 60 observed rows even within this one-year buffer remain "
                "missing by design; these are unusually stale histories and can "
                "be studied in robustness checks."
            ),
            "right_boundary_note": (
                "Because raw data end at 2025-12-31, the last six market dates "
                "cannot have a complete t+6 target and are naturally not model-ready."
            ),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "pandas_version": package_version("pandas"),
            "numpy_version": package_version("numpy"),
            "pyarrow_version": package_version("pyarrow"),
        }

        atomic_write_json(
            metadata,
            cfg.report_root
            / "signal_metadata.json",
        )

        logger.info("=" * 104)
        logger.info(
            "STAGE 5 COMPLETE | rows=%d model_ready=%d execution_ready=%d",
            total_rows,
            total_model_ready,
            total_execution_ready,
        )
        logger.info(
            "Reports: %s | %s | %s | %s | %s",
            monthly_quality_path,
            daily_summary_path,
            missingness_path,
            feature_summary_path,
            target_quality_path,
        )

        if not failed_months.empty:
            logger.error(
                "SIGNAL QA HAS %d FAILED MONTHS.",
                len(failed_months),
            )
            logger.error(
                "Do NOT proceed to single-factor analysis until reviewed."
            )
            logger.info("=" * 104)
            return 1

        logger.info(
            "SIGNAL + UNIVERSE MONTHLY QA: PASS"
        )
        logger.info("=" * 104)

        return 0

    except KeyboardInterrupt:
        logger.warning(
            "Interrupted by user. Existing valid monthly signal files are preserved. "
            "Re-run the same command to resume."
        )
        return 130

    except Exception:
        logger.exception(
            "Fatal error while building signals/universe."
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
