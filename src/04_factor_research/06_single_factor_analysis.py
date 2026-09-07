#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
06_single_factor_analysis.py

China A-Share Cross-Sectional Alpha Research
Stage 6: Single-Factor Predictive Diagnostics

PURPOSE
-------
Evaluate whether each frozen Stage-5 price-volume signal predicts future
cross-sectional A-share returns.

This is NOT yet the final tradable portfolio backtest. In particular:
- no transaction costs are applied here;
- no next-open tradability filter is used for the prediction sample;
- no factor direction is flipped based on realized results;
- daily 5-day targets overlap, so inference uses Newey-West/HAC standard errors.

INPUT
-----
Stage-5 v4 monthly signal panels:

data/processed/signals/YYYY/MM/signals_YYYYMM.parquet

Required Stage-5 specification:
    stage5_spec_version == "v4_final_positional"

FORMAL SAMPLE
-------------
2010-01-01 to 2025-12-31, subject to model_ready_5d == True.

Precommitted research periods:
    TRAIN      2010-01-01 to 2016-12-31
    VALIDATION 2017-01-01 to 2018-12-31
    OOS        2019-01-01 to 2025-12-31
    FULL       2010-01-01 to 2025-12-31

FACTORS
-------
All 11 frozen Stage-5 signals are evaluated with their daily cross-sectional
winsorized + z-scored values:

    mom5_z
    mom20_z
    mom60_z
    vol5_z
    vol20_z
    vol60_z
    vol5_vol60_z
    volume_ratio20_z
    amount_ratio20_z
    range1_z
    range20_z

Q5 ALWAYS means the highest raw factor value and Q1 the lowest.
We do NOT flip signs after looking at results.

DAILY STATISTICS
----------------
For each date and factor:
    Pearson IC   = Corr(factor_z, target_ret_5d)
    Rank IC      = Spearman Corr(factor_z, target_ret_5d)

Quintiles:
    Deterministic equal-count groups Q1...Q5 by factor rank.
    Ties are broken deterministically by ts_code.

Returns:
    Q1...Q5 = equal-weight mean 5-day forward return in each quintile
    Q5-Q1   = high-factor minus low-factor predictive spread
    Q5-EW   = Q5 minus equal-weight return of the factor-valid universe

INFERENCE
---------
Because adjacent signal dates use overlapping 5-day targets, conventional
iid t-statistics are inappropriate.

This script reports Newey-West / HAC t-statistics with default lag = 5 for:
    mean IC
    mean Rank IC
    mean Q5-Q1 spread
    mean Q5-EW spread

ICIR:
    icir_daily = mean(IC) / std(IC)
    icir_ann_sqrt252 = icir_daily * sqrt(252)

The annualized version is reported for familiarity, but it should not be
interpreted as a portfolio Sharpe ratio.

OUTPUT
------
data/single_factor_reports/
    single_factor_daily_metrics.csv
    single_factor_summary.csv
    single_factor_yearly.csv
    single_factor_period_stability.csv
    single_factor_factor_coverage.csv
    single_factor_metadata.json
    06_single_factor_analysis.log

RUN IN ANACONDA PROMPT
----------------------
python 06_single_factor_analysis.py

Optional:
python 06_single_factor_analysis.py --hac-lag 5
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import platform
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd


# ======================================================================================
# Frozen research specification
# ======================================================================================

REQUIRED_STAGE5_VERSION = "v4_final_positional"

FORMAL_START = "20100101"
FORMAL_END = "20251231"

PERIODS = {
    "TRAIN_2010_2016": ("20100101", "20161231"),
    "VALIDATION_2017_2018": ("20170101", "20181231"),
    "OOS_2019_2025": ("20190101", "20251231"),
    "FULL_2010_2025": ("20100101", "20251231"),
}

FACTORS = [
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

FACTOR_COLUMNS = {
    f: f"{f}_z"
    for f in FACTORS
}

TARGET = "target_ret_5d"
N_QUANTILES = 5


# ======================================================================================
# Configuration
# ======================================================================================

@dataclass
class Config:
    data_root: Path
    hac_lag: int
    min_daily_n: int

    @property
    def signal_root(self) -> Path:
        return self.data_root / "processed" / "signals"

    @property
    def report_root(self) -> Path:
        return self.data_root / "single_factor_reports"


# ======================================================================================
# Generic helpers
# ======================================================================================

def setup_logging(report_root: Path) -> logging.Logger:
    report_root.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("single_factor_analysis")
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
        report_root / "06_single_factor_analysis.log",
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
        json.dump(
            obj,
            f,
            ensure_ascii=False,
            indent=2,
            default=str,
        )

    tmp.replace(path)


# ======================================================================================
# File discovery / validation
# ======================================================================================

def discover_signal_files(
    signal_root: Path,
) -> List[Path]:
    files = sorted(
        signal_root.glob("*/*/signals_*.parquet")
    )

    if not files:
        raise FileNotFoundError(
            f"No Stage-5 signal files found under {signal_root}"
        )

    return files


def validate_stage5_file(
    path: Path,
) -> None:
    required = [
        "ts_code",
        "trade_date",
        "model_ready_5d",
        TARGET,
        "stage5_spec_version",
        *FACTOR_COLUMNS.values(),
    ]

    try:
        df = pd.read_parquet(
            path,
            columns=required,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Cannot read required Stage-5 columns from {path}"
        ) from exc

    missing = [
        c for c in required
        if c not in df.columns
    ]
    if missing:
        raise ValueError(
            f"{path} missing required columns: {missing}"
        )

    versions = (
        df["stage5_spec_version"]
        .dropna()
        .astype(str)
        .unique()
    )

    if (
        len(versions) != 1
        or versions[0] != REQUIRED_STAGE5_VERSION
    ):
        raise ValueError(
            f"{path}: expected stage5_spec_version="
            f"{REQUIRED_STAGE5_VERSION!r}, got {versions.tolist()}"
        )


# ======================================================================================
# Statistics
# ======================================================================================

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


def rankdata_average(
    x: np.ndarray,
) -> np.ndarray:
    """
    Average ranks for ties, equivalent in spirit to scipy.stats.rankdata(method='average'),
    implemented with pandas Series.rank to avoid an extra scipy dependency.
    """
    return (
        pd.Series(x)
        .rank(
            method="average",
            ascending=True,
        )
        .to_numpy(dtype=float)
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

    rx = rankdata_average(x)
    ry = rankdata_average(y)

    return pearson_corr(
        rx,
        ry,
    )


def deterministic_quantiles(
    factor: np.ndarray,
    ts_codes: np.ndarray,
    n_quantiles: int = N_QUANTILES,
) -> np.ndarray:
    """
    Equal-count deterministic quantile assignment.

    Sorting:
        factor ascending, then ts_code ascending.

    The lowest factor values receive Q1 and highest receive Q5.
    Ties are deterministically broken by ts_code.

    Group sizes differ by at most one observation.
    """
    factor = np.asarray(factor, dtype=float)
    ts_codes = np.asarray(ts_codes, dtype=str)

    n = len(factor)
    out = np.full(n, np.nan, dtype=float)

    valid = np.isfinite(factor)
    idx = np.flatnonzero(valid)

    if len(idx) < n_quantiles:
        return out

    # np.lexsort: last key is primary.
    order_local = np.lexsort(
        (
            ts_codes[idx],
            factor[idx],
        )
    )
    ordered_idx = idx[order_local]

    m = len(ordered_idx)

    # Position 0..m-1 -> Q1..Q5.
    q = (
        np.floor(
            np.arange(m, dtype=float)
            * n_quantiles
            / m
        )
        .astype(int)
        + 1
    )

    q = np.minimum(
        q,
        n_quantiles,
    )

    out[ordered_idx] = q.astype(float)
    return out


def newey_west_mean_stats(
    series: Sequence[float],
    lag: int,
) -> Tuple[float, float, float, int]:
    """
    Returns:
        mean, HAC standard error of mean, HAC t-stat, n

    Bartlett weights:
        w_l = 1 - l/(L+1)

    Long-run variance:
        gamma_0 + 2 sum_l w_l gamma_l

    Variance(mean) = long_run_variance / T
    """
    x = np.asarray(series, dtype=float)
    x = x[np.isfinite(x)]

    n = len(x)

    if n == 0:
        return np.nan, np.nan, np.nan, 0

    mean = float(np.mean(x))

    if n < 2:
        return mean, np.nan, np.nan, n

    e = x - mean
    L = min(
        int(lag),
        n - 1,
    )

    gamma0 = float(
        np.dot(e, e) / n
    )
    long_run_var = gamma0

    for ell in range(1, L + 1):
        gamma_l = float(
            np.dot(
                e[ell:],
                e[:-ell],
            )
            / n
        )

        weight = 1.0 - ell / (L + 1.0)

        long_run_var += (
            2.0
            * weight
            * gamma_l
        )

    # Numerical guard.
    long_run_var = max(
        long_run_var,
        0.0,
    )

    se = math.sqrt(
        long_run_var / n
    )

    t = (
        mean / se
        if se > 0
        else np.nan
    )

    return mean, se, t, n


def plain_std(
    series: Sequence[float],
) -> float:
    x = np.asarray(series, dtype=float)
    x = x[np.isfinite(x)]

    if len(x) < 2:
        return np.nan

    return float(
        np.std(
            x,
            ddof=1,
        )
    )


def positive_share(
    series: Sequence[float],
) -> float:
    x = np.asarray(series, dtype=float)
    x = x[np.isfinite(x)]

    if len(x) == 0:
        return np.nan

    return float(
        np.mean(x > 0)
    )


def quantile_monotonicity_corr(
    q_means: Sequence[float],
) -> float:
    y = np.asarray(q_means, dtype=float)

    if (
        len(y) != N_QUANTILES
        or not np.isfinite(y).all()
    ):
        return np.nan

    x = np.arange(
        1,
        N_QUANTILES + 1,
        dtype=float,
    )

    return pearson_corr(
        x,
        y,
    )


# ======================================================================================
# Daily analysis
# ======================================================================================

def analyze_one_day(
    day: pd.DataFrame,
    *,
    trade_date: str,
    factor_name: str,
    factor_col: str,
    min_daily_n: int,
) -> Dict[str, Any]:
    cols = [
        "ts_code",
        factor_col,
        TARGET,
    ]

    g = day[cols].copy()

    g[factor_col] = pd.to_numeric(
        g[factor_col],
        errors="coerce",
    )
    g[TARGET] = pd.to_numeric(
        g[TARGET],
        errors="coerce",
    )

    valid = (
        g[factor_col].notna()
        & g[TARGET].notna()
        & np.isfinite(g[factor_col])
        & np.isfinite(g[TARGET])
    )

    g = g.loc[valid].copy()

    n = len(g)

    row: Dict[str, Any] = {
        "trade_date": trade_date,
        "factor": factor_name,
        "n": int(n),
    }

    if n < min_daily_n:
        row.update({
            "ic": np.nan,
            "rank_ic": np.nan,
            "universe_ew_ret_5d": np.nan,
            "q1_ret_5d": np.nan,
            "q2_ret_5d": np.nan,
            "q3_ret_5d": np.nan,
            "q4_ret_5d": np.nan,
            "q5_ret_5d": np.nan,
            "q5_q1_ret_5d": np.nan,
            "q5_minus_universe_ret_5d": np.nan,
            "q1_n": 0,
            "q2_n": 0,
            "q3_n": 0,
            "q4_n": 0,
            "q5_n": 0,
        })
        return row

    x = g[factor_col].to_numpy(dtype=float)
    y = g[TARGET].to_numpy(dtype=float)
    codes = g["ts_code"].astype(str).to_numpy()

    row["ic"] = pearson_corr(
        x,
        y,
    )
    row["rank_ic"] = spearman_corr(
        x,
        y,
    )

    row["universe_ew_ret_5d"] = float(
        np.mean(y)
    )

    q = deterministic_quantiles(
        x,
        codes,
        n_quantiles=N_QUANTILES,
    )

    q_means = []

    for j in range(1, N_QUANTILES + 1):
        mask = q == j

        qn = int(mask.sum())

        qret = (
            float(np.mean(y[mask]))
            if qn
            else np.nan
        )

        row[f"q{j}_n"] = qn
        row[f"q{j}_ret_5d"] = qret

        q_means.append(qret)

    if (
        np.isfinite(row["q5_ret_5d"])
        and np.isfinite(row["q1_ret_5d"])
    ):
        row["q5_q1_ret_5d"] = (
            row["q5_ret_5d"]
            - row["q1_ret_5d"]
        )
    else:
        row["q5_q1_ret_5d"] = np.nan

    if np.isfinite(row["q5_ret_5d"]):
        row["q5_minus_universe_ret_5d"] = (
            row["q5_ret_5d"]
            - row["universe_ew_ret_5d"]
        )
    else:
        row["q5_minus_universe_ret_5d"] = np.nan

    return row


def analyze_month_file(
    path: Path,
    *,
    cfg: Config,
    logger: logging.Logger,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    required_cols = [
        "ts_code",
        "trade_date",
        "model_ready_5d",
        TARGET,
        "stage5_spec_version",
        *FACTOR_COLUMNS.values(),
    ]

    df = pd.read_parquet(
        path,
        columns=required_cols,
    )

    df["trade_date"] = normalize_date_series(
        df["trade_date"]
    )

    versions = (
        df["stage5_spec_version"]
        .dropna()
        .astype(str)
        .unique()
    )

    if (
        len(versions) != 1
        or versions[0] != REQUIRED_STAGE5_VERSION
    ):
        raise ValueError(
            f"{path} does not contain frozen Stage-5 v4 output."
        )

    formal = (
        (df["trade_date"] >= FORMAL_START)
        & (df["trade_date"] <= FORMAL_END)
    )

    ready = (
        df["model_ready_5d"]
        .fillna(False)
        .astype(bool)
    )

    df = df.loc[
        formal & ready
    ].copy()

    if df.empty:
        return [], []

    if df.duplicated(
        ["ts_code", "trade_date"],
        keep=False,
    ).any():
        raise RuntimeError(
            f"Duplicate security-date keys in {path}"
        )

    daily_rows: List[Dict[str, Any]] = []
    coverage_rows: List[Dict[str, Any]] = []

    for trade_date, day in df.groupby(
        "trade_date",
        sort=True,
    ):
        base_n = len(day)

        for factor_name, factor_col in FACTOR_COLUMNS.items():
            valid_factor = (
                pd.to_numeric(
                    day[factor_col],
                    errors="coerce",
                )
                .replace(
                    [np.inf, -np.inf],
                    np.nan,
                )
                .notna()
            )

            valid_target = (
                pd.to_numeric(
                    day[TARGET],
                    errors="coerce",
                )
                .replace(
                    [np.inf, -np.inf],
                    np.nan,
                )
                .notna()
            )

            n_valid = int(
                (valid_factor & valid_target).sum()
            )

            coverage_rows.append({
                "trade_date": trade_date,
                "factor": factor_name,
                "model_ready_rows": int(base_n),
                "factor_target_valid_rows": n_valid,
                "coverage_rate": (
                    n_valid / base_n
                    if base_n
                    else np.nan
                ),
            })

            daily_rows.append(
                analyze_one_day(
                    day,
                    trade_date=trade_date,
                    factor_name=factor_name,
                    factor_col=factor_col,
                    min_daily_n=cfg.min_daily_n,
                )
            )

    logger.info(
        "%s | model-ready rows=%d dates=%d daily-factor records=%d",
        path.name,
        len(df),
        df["trade_date"].nunique(),
        len(daily_rows),
    )

    return daily_rows, coverage_rows


# ======================================================================================
# Period summaries
# ======================================================================================

def summarize_factor_period(
    g: pd.DataFrame,
    *,
    factor: str,
    period: str,
    start_date: str,
    end_date: str,
    hac_lag: int,
) -> Dict[str, Any]:
    x = g.loc[
        (g["trade_date"] >= start_date)
        & (g["trade_date"] <= end_date)
    ].copy()

    row: Dict[str, Any] = {
        "factor": factor,
        "period": period,
        "start_date": start_date,
        "end_date": end_date,
        "n_dates": int(
            x["trade_date"].nunique()
        ),
    }

    # IC
    ic_mean, ic_se, ic_t, ic_n = newey_west_mean_stats(
        x["ic"],
        lag=hac_lag,
    )
    ic_sd = plain_std(
        x["ic"]
    )

    row["mean_ic"] = ic_mean
    row["sd_ic"] = ic_sd
    row["ic_hac_se"] = ic_se
    row["ic_hac_t"] = ic_t
    row["ic_n_dates"] = ic_n
    row["ic_positive_share"] = positive_share(
        x["ic"]
    )

    row["icir_daily"] = (
        ic_mean / ic_sd
        if (
            np.isfinite(ic_mean)
            and np.isfinite(ic_sd)
            and ic_sd > 0
        )
        else np.nan
    )
    row["icir_ann_sqrt252"] = (
        row["icir_daily"] * math.sqrt(252.0)
        if np.isfinite(row["icir_daily"])
        else np.nan
    )

    # Rank IC
    ric_mean, ric_se, ric_t, ric_n = newey_west_mean_stats(
        x["rank_ic"],
        lag=hac_lag,
    )
    ric_sd = plain_std(
        x["rank_ic"]
    )

    row["mean_rank_ic"] = ric_mean
    row["sd_rank_ic"] = ric_sd
    row["rank_ic_hac_se"] = ric_se
    row["rank_ic_hac_t"] = ric_t
    row["rank_ic_n_dates"] = ric_n
    row["rank_ic_positive_share"] = positive_share(
        x["rank_ic"]
    )

    row["rank_icir_daily"] = (
        ric_mean / ric_sd
        if (
            np.isfinite(ric_mean)
            and np.isfinite(ric_sd)
            and ric_sd > 0
        )
        else np.nan
    )
    row["rank_icir_ann_sqrt252"] = (
        row["rank_icir_daily"] * math.sqrt(252.0)
        if np.isfinite(row["rank_icir_daily"])
        else np.nan
    )

    # Quintiles
    q_means = []

    for j in range(1, N_QUANTILES + 1):
        col = f"q{j}_ret_5d"
        mean_q = safe_float(
            pd.to_numeric(
                x[col],
                errors="coerce",
            ).mean()
        )

        row[f"mean_q{j}_ret_5d"] = mean_q
        row[f"mean_q{j}_ret_5d_bps"] = (
            mean_q * 10000.0
            if np.isfinite(mean_q)
            else np.nan
        )

        q_means.append(mean_q)

    spread_mean, spread_se, spread_t, spread_n = newey_west_mean_stats(
        x["q5_q1_ret_5d"],
        lag=hac_lag,
    )

    row["mean_q5_q1_ret_5d"] = spread_mean
    row["mean_q5_q1_ret_5d_bps"] = (
        spread_mean * 10000.0
        if np.isfinite(spread_mean)
        else np.nan
    )
    row["q5_q1_hac_se"] = spread_se
    row["q5_q1_hac_t"] = spread_t
    row["q5_q1_n_dates"] = spread_n
    row["q5_q1_positive_share"] = positive_share(
        x["q5_q1_ret_5d"]
    )

    q5ew_mean, q5ew_se, q5ew_t, q5ew_n = newey_west_mean_stats(
        x["q5_minus_universe_ret_5d"],
        lag=hac_lag,
    )

    row["mean_q5_minus_universe_ret_5d"] = q5ew_mean
    row["mean_q5_minus_universe_ret_5d_bps"] = (
        q5ew_mean * 10000.0
        if np.isfinite(q5ew_mean)
        else np.nan
    )
    row["q5_minus_universe_hac_se"] = q5ew_se
    row["q5_minus_universe_hac_t"] = q5ew_t
    row["q5_minus_universe_n_dates"] = q5ew_n

    row["quantile_monotonicity_corr"] = (
        quantile_monotonicity_corr(
            q_means
        )
    )

    row["mean_daily_n"] = safe_float(
        pd.to_numeric(
            x["n"],
            errors="coerce",
        ).mean()
    )

    return row


def build_period_summary(
    daily: pd.DataFrame,
    *,
    hac_lag: int,
) -> pd.DataFrame:
    rows = []

    for factor in FACTORS:
        g = daily.loc[
            daily["factor"] == factor
        ].copy()

        for period, (
            start_date,
            end_date,
        ) in PERIODS.items():
            rows.append(
                summarize_factor_period(
                    g,
                    factor=factor,
                    period=period,
                    start_date=start_date,
                    end_date=end_date,
                    hac_lag=hac_lag,
                )
            )

    return pd.DataFrame(rows)


def build_yearly_summary(
    daily: pd.DataFrame,
    *,
    hac_lag: int,
) -> pd.DataFrame:
    rows = []

    temp = daily.copy()
    temp["year"] = temp[
        "trade_date"
    ].str[:4].astype(int)

    for factor in FACTORS:
        gf = temp.loc[
            temp["factor"] == factor
        ].copy()

        for year, gy in gf.groupby(
            "year",
            sort=True,
        ):
            start_date = f"{year:04d}0101"
            end_date = f"{year:04d}1231"

            row = summarize_factor_period(
                gy,
                factor=factor,
                period=f"YEAR_{year}",
                start_date=start_date,
                end_date=end_date,
                hac_lag=hac_lag,
            )
            row["year"] = int(year)
            rows.append(row)

    return pd.DataFrame(rows)


def build_period_stability(
    summary: pd.DataFrame,
) -> pd.DataFrame:
    """
    One row per factor with train/validation/OOS directional consistency.

    Important:
    This does NOT flip the factor. It only reports whether the raw sign is stable.
    """
    rows = []

    for factor in FACTORS:
        g = summary.loc[
            summary["factor"] == factor
        ].set_index("period")

        def get(period: str, col: str) -> float:
            if (
                period not in g.index
                or col not in g.columns
            ):
                return np.nan
            return safe_float(
                g.loc[period, col]
            )

        train_ric = get(
            "TRAIN_2010_2016",
            "mean_rank_ic",
        )
        val_ric = get(
            "VALIDATION_2017_2018",
            "mean_rank_ic",
        )
        oos_ric = get(
            "OOS_2019_2025",
            "mean_rank_ic",
        )

        train_spread = get(
            "TRAIN_2010_2016",
            "mean_q5_q1_ret_5d",
        )
        val_spread = get(
            "VALIDATION_2017_2018",
            "mean_q5_q1_ret_5d",
        )
        oos_spread = get(
            "OOS_2019_2025",
            "mean_q5_q1_ret_5d",
        )

        def same_nonzero_sign(a: float, b: float) -> Any:
            if not (
                np.isfinite(a)
                and np.isfinite(b)
            ):
                return pd.NA

            sa = np.sign(a)
            sb = np.sign(b)

            if sa == 0 or sb == 0:
                return False

            return bool(sa == sb)

        rows.append({
            "factor": factor,
            "train_mean_rank_ic": train_ric,
            "validation_mean_rank_ic": val_ric,
            "oos_mean_rank_ic": oos_ric,
            "rank_ic_train_validation_same_sign": (
                same_nonzero_sign(
                    train_ric,
                    val_ric,
                )
            ),
            "rank_ic_train_oos_same_sign": (
                same_nonzero_sign(
                    train_ric,
                    oos_ric,
                )
            ),
            "train_mean_q5_q1_ret_5d": train_spread,
            "validation_mean_q5_q1_ret_5d": val_spread,
            "oos_mean_q5_q1_ret_5d": oos_spread,
            "spread_train_validation_same_sign": (
                same_nonzero_sign(
                    train_spread,
                    val_spread,
                )
            ),
            "spread_train_oos_same_sign": (
                same_nonzero_sign(
                    train_spread,
                    oos_spread,
                )
            ),
        })

    return pd.DataFrame(rows)


# ======================================================================================
# Coverage summary
# ======================================================================================

def build_coverage_summary(
    coverage_daily: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for factor, g in coverage_daily.groupby(
        "factor",
        sort=False,
    ):
        rates = pd.to_numeric(
            g["coverage_rate"],
            errors="coerce",
        )

        rows.append({
            "factor": factor,
            "n_dates": int(
                g["trade_date"].nunique()
            ),
            "mean_model_ready_rows": safe_float(
                pd.to_numeric(
                    g["model_ready_rows"],
                    errors="coerce",
                ).mean()
            ),
            "mean_factor_target_valid_rows": safe_float(
                pd.to_numeric(
                    g["factor_target_valid_rows"],
                    errors="coerce",
                ).mean()
            ),
            "mean_coverage_rate": safe_float(
                rates.mean()
            ),
            "min_coverage_rate": safe_float(
                rates.min()
            ),
            "dates_below_99pct_coverage": int(
                (rates < 0.99).sum()
            ),
            "dates_below_95pct_coverage": int(
                (rates < 0.95).sum()
            ),
        })

    return pd.DataFrame(rows)


# ======================================================================================
# QA
# ======================================================================================

def final_qa(
    daily: pd.DataFrame,
    summary: pd.DataFrame,
    coverage: pd.DataFrame,
) -> List[str]:
    issues: List[str] = []

    if daily.empty:
        issues.append("daily metrics output is empty")
        return issues

    if daily.duplicated(
        ["trade_date", "factor"],
        keep=False,
    ).any():
        issues.append(
            "duplicate (trade_date, factor) rows in daily metrics"
        )

    # Every factor should appear.
    missing_factors = sorted(
        set(FACTORS)
        - set(daily["factor"].unique())
    )
    if missing_factors:
        issues.append(
            f"missing factors in daily metrics: {missing_factors}"
        )

    # Formal date coverage should be the same across factors because model_ready is common.
    date_counts = (
        daily.groupby("factor")["trade_date"]
        .nunique()
    )

    if date_counts.nunique() != 1:
        issues.append(
            f"factor date counts differ: {date_counts.to_dict()}"
        )

    # At least some finite IC and RankIC for each factor.
    for factor in FACTORS:
        g = daily.loc[
            daily["factor"] == factor
        ]

        if pd.to_numeric(
            g["ic"],
            errors="coerce",
        ).notna().sum() == 0:
            issues.append(
                f"{factor}: no finite daily IC"
            )

        if pd.to_numeric(
            g["rank_ic"],
            errors="coerce",
        ).notna().sum() == 0:
            issues.append(
                f"{factor}: no finite daily RankIC"
            )

        if pd.to_numeric(
            g["q5_q1_ret_5d"],
            errors="coerce",
        ).notna().sum() == 0:
            issues.append(
                f"{factor}: no finite daily Q5-Q1 spread"
            )

    # Coverage in the frozen model-ready sample should be almost complete.
    for row in coverage.itertuples(index=False):
        if (
            np.isfinite(row.mean_coverage_rate)
            and row.mean_coverage_rate < 0.99
        ):
            issues.append(
                f"{row.factor}: mean coverage below 99% "
                f"({row.mean_coverage_rate:.6f})"
            )

    # Full period summary should exist for every factor.
    full = summary.loc[
        summary["period"] == "FULL_2010_2025"
    ]

    if len(full) != len(FACTORS):
        issues.append(
            f"FULL summary has {len(full)} factors; expected {len(FACTORS)}"
        )

    return issues


# ======================================================================================
# CLI
# ======================================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Stage 6 single-factor IC / RankIC / quintile analysis."
        )
    )

    p.add_argument(
        "--data-root",
        default="data",
    )
    p.add_argument(
        "--hac-lag",
        type=int,
        default=5,
        help=(
            "Newey-West/HAC lag for overlapping 5-day targets "
            "(default 5)."
        ),
    )
    p.add_argument(
        "--min-daily-n",
        type=int,
        default=100,
        help=(
            "Minimum factor-target observations required for daily "
            "IC/quintile statistics (default 100)."
        ),
    )

    return p.parse_args()


# ======================================================================================
# Main
# ======================================================================================

def main() -> int:
    args = parse_args()

    if args.hac_lag < 0:
        raise ValueError("--hac-lag must be >= 0.")

    if args.min_daily_n < 10:
        raise ValueError(
            "--min-daily-n must be >= 10."
        )

    cfg = Config(
        data_root=Path(args.data_root),
        hac_lag=int(args.hac_lag),
        min_daily_n=int(args.min_daily_n),
    )

    cfg.report_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    logger = setup_logging(
        cfg.report_root
    )

    logger.info("=" * 104)
    logger.info("STAGE 6 | SINGLE-FACTOR PREDICTIVE DIAGNOSTICS")
    logger.info(
        "Frozen Stage-5 spec: %s",
        REQUIRED_STAGE5_VERSION,
    )
    logger.info(
        "Formal sample: %s -> %s",
        FORMAL_START,
        FORMAL_END,
    )
    logger.info(
        "Factors: %s",
        ", ".join(FACTORS),
    )
    logger.info(
        "HAC lag=%d | min daily n=%d",
        cfg.hac_lag,
        cfg.min_daily_n,
    )
    logger.info(
        "IMPORTANT: no factor sign flipping; Q5 always highest factor values."
    )
    logger.info("=" * 104)

    try:
        files = discover_signal_files(
            cfg.signal_root
        )

        logger.info(
            "Discovered %d monthly Stage-5 files.",
            len(files),
        )

        # Validate all monthly files before analysis.
        for path in files:
            validate_stage5_file(
                path
            )

        daily_rows: List[Dict[str, Any]] = []
        coverage_rows: List[Dict[str, Any]] = []

        for i, path in enumerate(files, 1):
            logger.info(
                "MONTH %d/%d | %s",
                i,
                len(files),
                path.name,
            )

            drows, crows = analyze_month_file(
                path,
                cfg=cfg,
                logger=logger,
            )

            daily_rows.extend(
                drows
            )
            coverage_rows.extend(
                crows
            )

            gc.collect()

        daily = pd.DataFrame(
            daily_rows
        )

        coverage_daily = pd.DataFrame(
            coverage_rows
        )

        if daily.empty:
            raise RuntimeError(
                "No formal model-ready observations were analyzed."
            )

        daily = daily.sort_values(
            ["trade_date", "factor"]
        ).reset_index(drop=True)

        coverage_daily = coverage_daily.sort_values(
            ["trade_date", "factor"]
        ).reset_index(drop=True)

        # ------------------------------------------------------------------
        # Aggregates
        # ------------------------------------------------------------------
        summary = build_period_summary(
            daily,
            hac_lag=cfg.hac_lag,
        )

        yearly = build_yearly_summary(
            daily,
            hac_lag=cfg.hac_lag,
        )

        stability = build_period_stability(
            summary
        )

        coverage_summary = build_coverage_summary(
            coverage_daily
        )

        # ------------------------------------------------------------------
        # QA
        # ------------------------------------------------------------------
        issues = final_qa(
            daily,
            summary,
            coverage_summary,
        )

        # ------------------------------------------------------------------
        # Save
        # ------------------------------------------------------------------
        daily_path = (
            cfg.report_root
            / "single_factor_daily_metrics.csv"
        )

        summary_path = (
            cfg.report_root
            / "single_factor_summary.csv"
        )

        yearly_path = (
            cfg.report_root
            / "single_factor_yearly.csv"
        )

        stability_path = (
            cfg.report_root
            / "single_factor_period_stability.csv"
        )

        coverage_path = (
            cfg.report_root
            / "single_factor_factor_coverage.csv"
        )

        daily.to_csv(
            daily_path,
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

        stability.to_csv(
            stability_path,
            index=False,
            encoding="utf-8-sig",
        )

        coverage_summary.to_csv(
            coverage_path,
            index=False,
            encoding="utf-8-sig",
        )

        metadata = {
            "project": (
                "China A-Share Cross-Sectional Alpha Research"
            ),
            "script": "06_single_factor_analysis.py",
            "generated_at_utc": datetime.now(
                timezone.utc
            ).isoformat(),
            "required_stage5_spec_version": REQUIRED_STAGE5_VERSION,
            "formal_sample_start": FORMAL_START,
            "formal_sample_end": FORMAL_END,
            "periods": PERIODS,
            "factors": FACTORS,
            "factor_columns": FACTOR_COLUMNS,
            "target": TARGET,
            "n_quantiles": N_QUANTILES,
            "hac_lag": cfg.hac_lag,
            "min_daily_n": cfg.min_daily_n,
            "daily_metric_rows": int(
                len(daily)
            ),
            "daily_unique_dates": int(
                daily["trade_date"].nunique()
            ),
            "summary_rows": int(
                len(summary)
            ),
            "yearly_rows": int(
                len(yearly)
            ),
            "qa_issue_count": int(
                len(issues)
            ),
            "qa_issues": issues,
            "research_notes": [
                "Prediction sample uses model_ready_5d and does not condition on future tradability.",
                "Q5 always means highest raw factor values; no ex-post sign flipping is performed.",
                "Daily quintile returns are predictive diagnostics based on overlapping 5-day forward returns, not a self-financing daily portfolio.",
                "Newey-West/HAC lag 5 is used for mean IC, mean RankIC, Q5-Q1, and Q5-minus-universe inference.",
                "Transaction costs and executable portfolio accounting are deferred to a later backtest stage.",
                "Train/validation/OOS splits are precommitted: 2010-2016 / 2017-2018 / 2019-2025.",
            ],
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "pandas_version": package_version("pandas"),
            "numpy_version": package_version("numpy"),
            "pyarrow_version": package_version("pyarrow"),
        }

        atomic_write_json(
            metadata,
            cfg.report_root
            / "single_factor_metadata.json",
        )

        logger.info("=" * 104)
        logger.info(
            "STAGE 6 COMPLETE | daily dates=%d daily-factor rows=%d",
            daily["trade_date"].nunique(),
            len(daily),
        )
        logger.info(
            "Outputs: %s | %s | %s | %s | %s",
            daily_path,
            summary_path,
            yearly_path,
            stability_path,
            coverage_path,
        )

        if issues:
            logger.error(
                "STAGE 6 QA FAIL | %d issue(s)",
                len(issues),
            )

            for issue in issues:
                logger.error(
                    "QA | %s",
                    issue,
                )

            logger.info("=" * 104)
            return 1

        logger.info(
            "STAGE 6 QA: PASS"
        )
        logger.info("=" * 104)

        return 0

    except KeyboardInterrupt:
        logger.warning(
            "Interrupted by user."
        )
        return 130

    except Exception:
        logger.exception(
            "Fatal error during Stage-6 single-factor analysis."
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
