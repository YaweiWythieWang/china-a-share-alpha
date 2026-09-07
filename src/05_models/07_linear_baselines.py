#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
07_linear_baselines.py

China A-Share Cross-Sectional Alpha Research
Stage 7: Multivariate Linear Baselines

PURPOSE
-------
Move from single-factor diagnostics to multivariate cross-sectional prediction
using two transparent linear baselines:

    1) Pooled OLS
    2) Ridge Regression

The design is strictly time ordered and avoids label leakage from the overlapping
5-day forward target.

FROZEN INPUT
------------
Stage-5 v4 signal panels:

    data/processed/signals/YYYY/MM/signals_YYYYMM.parquet

Required:
    stage5_spec_version == "v4_final_positional"

FEATURES
--------
All 11 frozen Stage-6 signals are retained. We do NOT delete weak factors after
seeing Stage-6 results.

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

TARGET USED FOR FITTING
-----------------------
For each signal date t:

    y_xs(i,t)
      = target_ret_5d(i,t)
        - mean_i[target_ret_5d(i,t)]

This daily cross-sectional demeaning removes the common market component from
the pooled regression and makes the fitting objective align with the project's
cross-sectional prediction goal.

Predictions are still evaluated against the original raw 5-day forward returns.
Within a date, demeaning the target does not change Pearson/Spearman cross-sectional
correlations.

MODEL FORM
----------
No intercept is fit:

    y_xs = X beta + error

because Stage-5 factor columns are daily cross-sectional z-scores and y_xs is
daily demeaned.

OLS:
    beta_OLS = argmin mean (y - X beta)^2

Ridge:
    beta_Ridge(lambda)
      = argmin mean (y - X beta)^2 + lambda ||beta||_2^2

The Ridge penalty is normalized by sample size, so lambda has a stable meaning
as the expanding sample grows.

TIME SPLIT
----------
TRAIN:
    2010-01-01 to 2016-12-31

VALIDATION:
    2017-01-01 to 2018-12-31

OOS:
    2019-01-01 to 2025-12-31

Ridge lambda is selected ONLY from VALIDATION.

WALK-FORWARD / LABEL PURGING
----------------------------
Models are retrained MONTHLY with an expanding window.

For a prediction month beginning on date T, a historical observation is eligible
for training only if:

    target_exit_trade_date < T

This is stricter than simply requiring signal_date < T. It guarantees that the
full t+1 -> t+6 label would actually have been observable when the model was fit.

No current-month labels are used because the model is fixed for the whole month.

RIDGE TUNING
------------
Default lambda grid:

    1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10

Each lambda is evaluated via monthly expanding-window predictions during
2017-2018.

Primary selection criterion:
    highest VALIDATION mean daily RankIC

Tie-breakers:
    higher mean daily IC
    then smaller lambda

Once selected, lambda is frozen for 2019-2025 OOS.

EVALUATION
----------
For OLS and selected Ridge:

    daily Pearson IC
    daily Spearman RankIC
    Q1...Q5 equal-weight 5-day forward returns
    Q5-Q1 predictive spread
    Q5 - eligible-universe EW spread

Higher model score always means higher predicted return.

Because 5-day targets overlap, summary t-statistics use Newey-West / HAC lag 5.

FACTOR REDUNDANCY DIAGNOSTICS
-----------------------------
Using TRAIN only:
    pooled factor correlation matrix
    eigenvalues / condition number
    VIF-style diagnostics
    high-correlation factor pairs

OUTPUT
------
data/linear_model_reports/
    linear_factor_correlation_train.csv
    linear_factor_redundancy_train.csv
    ridge_validation_tuning.csv
    linear_model_coefficients.csv
    linear_model_daily_metrics.csv
    linear_model_summary.csv
    linear_model_yearly.csv
    linear_model_prediction_coverage.csv
    linear_model_metadata.json
    07_linear_baselines.log

Monthly predictions:
data/processed/linear_predictions/YYYY/MM/linear_predictions_YYYYMM.parquet

RUN IN ANACONDA PROMPT
----------------------
python 07_linear_baselines.py

Optional:
python 07_linear_baselines.py --hac-lag 5
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
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# ======================================================================================
# Frozen specification
# ======================================================================================

REQUIRED_STAGE5_VERSION = "v4_final_positional"
STAGE7_SPEC_VERSION = "v1_linear_walkforward"

TRAIN_START = "20100101"
TRAIN_END = "20161231"

VALIDATION_START = "20170101"
VALIDATION_END = "20181231"

OOS_START = "20190101"
OOS_END = "20251231"

PREDICTION_START = VALIDATION_START
PREDICTION_END = OOS_END

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

TARGET = "target_ret_5d"
EXIT_DATE_COL = "target_exit_trade_date"

DEFAULT_RIDGE_GRID = (
    1e-6,
    1e-5,
    1e-4,
    1e-3,
    1e-2,
    1e-1,
    1.0,
    10.0,
)

N_QUANTILES = 5


# ======================================================================================
# Configuration
# ======================================================================================

@dataclass
class Config:
    data_root: Path
    hac_lag: int
    min_daily_n: int
    ridge_grid: Tuple[float, ...]
    high_corr_threshold: float
    force: bool

    @property
    def signal_root(self) -> Path:
        return self.data_root / "processed" / "signals"

    @property
    def prediction_root(self) -> Path:
        return self.data_root / "processed" / "linear_predictions"

    @property
    def report_root(self) -> Path:
        return self.data_root / "linear_model_reports"


# ======================================================================================
# Logging / generic utilities
# ======================================================================================

def setup_logging(report_root: Path) -> logging.Logger:
    report_root.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("linear_baselines")
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
        report_root / "07_linear_baselines.log",
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


def atomic_write_parquet(
    df: pd.DataFrame,
    path: Path,
) -> None:
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


def safe_float(x: Any) -> float:
    try:
        y = float(x)
    except Exception:
        return np.nan

    return y if np.isfinite(y) else np.nan


def parse_month_from_path(
    path: Path,
) -> Tuple[int, int, str]:
    m = re.search(
        r"signals_(\d{4})(\d{2})\.parquet$",
        path.name,
    )

    if not m:
        raise ValueError(
            f"Cannot parse year/month from {path}"
        )

    year = int(m.group(1))
    month = int(m.group(2))
    label = f"{year:04d}-{month:02d}"

    return year, month, label


def month_key(
    date_str: str,
) -> str:
    return f"{date_str[:4]}-{date_str[4:6]}"


# ======================================================================================
# File discovery / loading
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


def required_signal_columns() -> List[str]:
    cols = [
        "ts_code",
        "trade_date",
        "model_ready_5d",
        "stage5_spec_version",
        TARGET,
        EXIT_DATE_COL,
        "target_entry_trade_date",
        "target_entry_tradable",
        "target_exit_tradable",
        "target_fully_tradable_5d",
        *FEATURE_COLUMNS,
    ]

    return cols


def load_signal_month(
    path: Path,
) -> pd.DataFrame:
    cols = required_signal_columns()

    try:
        df = pd.read_parquet(
            path,
            columns=cols,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Cannot read required Stage-5 columns from {path}"
        ) from exc

    missing = [
        c for c in cols
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{path} missing columns: {missing}"
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
            f"{path}: expected Stage-5 version "
            f"{REQUIRED_STAGE5_VERSION!r}, got {versions.tolist()}"
        )

    df["trade_date"] = normalize_date_series(
        df["trade_date"]
    )
    df[EXIT_DATE_COL] = normalize_date_series(
        df[EXIT_DATE_COL]
    )
    df["target_entry_trade_date"] = normalize_date_series(
        df["target_entry_trade_date"]
    )

    return df


def model_ready_month(
    df: pd.DataFrame,
) -> pd.DataFrame:
    ready = (
        df["model_ready_5d"]
        .fillna(False)
        .astype(bool)
    )

    x = df.loc[
        ready
    ].copy()

    for c in [
        *FEATURE_COLUMNS,
        TARGET,
    ]:
        x[c] = pd.to_numeric(
            x[c],
            errors="coerce",
        )

    finite = np.ones(
        len(x),
        dtype=bool,
    )

    for c in [
        *FEATURE_COLUMNS,
        TARGET,
    ]:
        finite &= np.isfinite(
            x[c].to_numpy(dtype=float)
        )

    x = x.loc[
        finite
    ].copy()

    if x.duplicated(
        ["ts_code", "trade_date"],
        keep=False,
    ).any():
        raise RuntimeError(
            "Duplicate security-date keys in model-ready month."
        )

    return x


# ======================================================================================
# Daily sufficient statistics
# ======================================================================================

@dataclass
class DayStats:
    signal_date: str
    exit_date: str
    n: int
    xtx: np.ndarray
    xty: np.ndarray
    yty: float


def demean_target_by_date(
    df: pd.DataFrame,
) -> pd.DataFrame:
    x = df.copy()

    daily_mean = (
        x.groupby(
            "trade_date",
            sort=False,
        )[TARGET]
        .transform("mean")
    )

    x["target_xs"] = (
        x[TARGET]
        - daily_mean
    )

    return x


def day_sufficient_stats(
    day: pd.DataFrame,
) -> DayStats:
    signal_dates = day["trade_date"].unique()
    exit_dates = (
        day[EXIT_DATE_COL]
        .dropna()
        .unique()
    )

    if len(signal_dates) != 1:
        raise RuntimeError(
            "Expected one signal date per daily group."
        )

    if len(exit_dates) != 1:
        raise RuntimeError(
            f"{signal_dates[0]}: expected one target exit date, "
            f"got {exit_dates.tolist()}"
        )

    X = day[
        FEATURE_COLUMNS
    ].to_numpy(dtype=float)

    y = day[
        "target_xs"
    ].to_numpy(dtype=float)

    if not (
        np.isfinite(X).all()
        and np.isfinite(y).all()
    ):
        raise RuntimeError(
            f"{signal_dates[0]}: nonfinite training array."
        )

    return DayStats(
        signal_date=str(signal_dates[0]),
        exit_date=str(exit_dates[0]),
        n=len(day),
        xtx=X.T @ X,
        xty=X.T @ y,
        yty=float(y @ y),
    )


def build_day_stats_and_train_moments(
    files: Sequence[Path],
    *,
    logger: logging.Logger,
) -> Tuple[List[DayStats], Dict[str, Any]]:
    """
    First pass over Stage-5 data.

    Produces:
      - one sufficient-stat object per signal date;
      - pooled TRAIN feature moments for correlation / redundancy diagnostics.
    """
    stats: List[DayStats] = []

    train_n = 0
    train_sum = np.zeros(
        P,
        dtype=float,
    )
    train_cross = np.zeros(
        (P, P),
        dtype=float,
    )

    for i, path in enumerate(
        files,
        1,
    ):
        df = load_signal_month(
            path
        )

        x = model_ready_month(
            df
        )

        if x.empty:
            continue

        x = demean_target_by_date(
            x
        )

        for _, day in x.groupby(
            "trade_date",
            sort=True,
        ):
            stats.append(
                day_sufficient_stats(
                    day
                )
            )

        train_mask = (
            (x["trade_date"] >= TRAIN_START)
            & (x["trade_date"] <= TRAIN_END)
        )

        xt = x.loc[
            train_mask,
            FEATURE_COLUMNS,
        ]

        if not xt.empty:
            X = xt.to_numpy(
                dtype=float
            )

            train_n += len(X)
            train_sum += X.sum(
                axis=0
            )
            train_cross += X.T @ X

        logger.info(
            "SUFF STATS %d/%d | %s | model-ready rows=%d dates=%d",
            i,
            len(files),
            path.name,
            len(x),
            x["trade_date"].nunique(),
        )

        del df, x
        gc.collect()

    stats.sort(
        key=lambda s: (
            s.exit_date,
            s.signal_date,
        )
    )

    train_moments = {
        "n": train_n,
        "sum": train_sum,
        "cross": train_cross,
    }

    return stats, train_moments


# ======================================================================================
# Correlation / redundancy diagnostics
# ======================================================================================

def correlation_from_moments(
    n: int,
    sum_x: np.ndarray,
    cross: np.ndarray,
) -> np.ndarray:
    if n <= 1:
        raise RuntimeError(
            "Insufficient TRAIN observations for correlation."
        )

    mean = sum_x / n

    cov = (
        cross / n
        - np.outer(
            mean,
            mean,
        )
    )

    var = np.diag(
        cov
    )

    sd = np.sqrt(
        np.maximum(
            var,
            0.0,
        )
    )

    denom = np.outer(
        sd,
        sd,
    )

    corr = np.divide(
        cov,
        denom,
        out=np.full_like(
            cov,
            np.nan,
            dtype=float,
        ),
        where=denom > 0,
    )

    np.fill_diagonal(
        corr,
        1.0,
    )

    return corr


def build_redundancy_report(
    corr: np.ndarray,
    *,
    threshold: float,
) -> pd.DataFrame:
    eigvals = np.linalg.eigvalsh(
        corr
    )

    min_eig = float(
        np.min(eigvals)
    )
    max_eig = float(
        np.max(eigvals)
    )

    condition = (
        max_eig / min_eig
        if min_eig > 0
        else np.inf
    )

    inv_corr = np.linalg.pinv(
        corr,
        rcond=1e-12,
    )

    vif = np.diag(
        inv_corr
    )

    max_abs_other = []

    for j in range(P):
        vals = np.abs(
            np.delete(
                corr[j],
                j,
            )
        )
        max_abs_other.append(
            float(np.nanmax(vals))
        )

    rows = []

    for j, name in enumerate(
        FEATURE_NAMES
    ):
        rows.append({
            "record_type": "factor",
            "factor": name,
            "factor_2": "",
            "correlation": np.nan,
            "abs_correlation": np.nan,
            "vif_style": float(vif[j]),
            "max_abs_corr_with_other": max_abs_other[j],
            "train_corr_min_eigenvalue": min_eig,
            "train_corr_max_eigenvalue": max_eig,
            "train_corr_condition_number": condition,
        })

    for i in range(P):
        for j in range(
            i + 1,
            P,
        ):
            c = float(
                corr[i, j]
            )

            if abs(c) >= threshold:
                rows.append({
                    "record_type": "high_corr_pair",
                    "factor": FEATURE_NAMES[i],
                    "factor_2": FEATURE_NAMES[j],
                    "correlation": c,
                    "abs_correlation": abs(c),
                    "vif_style": np.nan,
                    "max_abs_corr_with_other": np.nan,
                    "train_corr_min_eigenvalue": min_eig,
                    "train_corr_max_eigenvalue": max_eig,
                    "train_corr_condition_number": condition,
                })

    return pd.DataFrame(
        rows
    )


# ======================================================================================
# Expanding sufficient-stat accumulator
# ======================================================================================

class ExpandingStats:
    def __init__(
        self,
        day_stats: Sequence[DayStats],
    ):
        self.day_stats = list(
            day_stats
        )
        self.pointer = 0
        self.n = 0
        self.xtx = np.zeros(
            (P, P),
            dtype=float,
        )
        self.xty = np.zeros(
            P,
            dtype=float,
        )
        self.yty = 0.0
        self.last_included_exit_date: Optional[str] = None

    def advance_to(
        self,
        cutoff_date: str,
    ) -> None:
        """
        Include all historical observations whose full label had become
        observable strictly before cutoff_date:
            exit_date < cutoff_date
        """
        while (
            self.pointer
            < len(self.day_stats)
            and self.day_stats[
                self.pointer
            ].exit_date < cutoff_date
        ):
            s = self.day_stats[
                self.pointer
            ]

            self.n += s.n
            self.xtx += s.xtx
            self.xty += s.xty
            self.yty += s.yty
            self.last_included_exit_date = (
                s.exit_date
            )

            self.pointer += 1

    def normalized_moments(
        self,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self.n <= P:
            raise RuntimeError(
                f"Insufficient expanding training sample n={self.n}"
            )

        return (
            self.xtx / self.n,
            self.xty / self.n,
        )


# ======================================================================================
# Model fitting
# ======================================================================================

def solve_ols(
    xtx_mean: np.ndarray,
    xty_mean: np.ndarray,
) -> Tuple[np.ndarray, float]:
    condition = float(
        np.linalg.cond(
            xtx_mean
        )
    )

    beta = (
        np.linalg.pinv(
            xtx_mean,
            rcond=1e-12,
        )
        @ xty_mean
    )

    return beta, condition


def solve_ridge(
    xtx_mean: np.ndarray,
    xty_mean: np.ndarray,
    lam: float,
) -> Tuple[np.ndarray, float]:
    A = (
        xtx_mean
        + float(lam)
        * np.eye(
            P,
            dtype=float,
        )
    )

    condition = float(
        np.linalg.cond(
            A
        )
    )

    try:
        beta = np.linalg.solve(
            A,
            xty_mean,
        )
    except np.linalg.LinAlgError:
        beta = (
            np.linalg.pinv(
                A,
                rcond=1e-12,
            )
            @ xty_mean
        )

    return beta, condition


# ======================================================================================
# Daily evaluation utilities
# ======================================================================================

def pearson_corr(
    x: np.ndarray,
    y: np.ndarray,
) -> float:
    x = np.asarray(
        x,
        dtype=float,
    )
    y = np.asarray(
        y,
        dtype=float,
    )

    valid = (
        np.isfinite(x)
        & np.isfinite(y)
    )

    x = x[valid]
    y = y[valid]

    if len(x) < 3:
        return np.nan

    sx = float(
        np.std(
            x,
            ddof=0,
        )
    )
    sy = float(
        np.std(
            y,
            ddof=0,
        )
    )

    if sx <= 0 or sy <= 0:
        return np.nan

    return float(
        np.corrcoef(
            x,
            y,
        )[0, 1]
    )


def spearman_corr(
    x: np.ndarray,
    y: np.ndarray,
) -> float:
    x = np.asarray(
        x,
        dtype=float,
    )
    y = np.asarray(
        y,
        dtype=float,
    )

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
        .rank(
            method="average",
        )
        .to_numpy(dtype=float)
    )
    ry = (
        pd.Series(y)
        .rank(
            method="average",
        )
        .to_numpy(dtype=float)
    )

    return pearson_corr(
        rx,
        ry,
    )


def deterministic_quantiles(
    score: np.ndarray,
    ts_codes: np.ndarray,
    n_quantiles: int = N_QUANTILES,
) -> np.ndarray:
    score = np.asarray(
        score,
        dtype=float,
    )
    ts_codes = np.asarray(
        ts_codes,
        dtype=str,
    )

    n = len(score)
    out = np.full(
        n,
        np.nan,
        dtype=float,
    )

    valid = np.isfinite(
        score
    )
    idx = np.flatnonzero(
        valid
    )

    if len(idx) < n_quantiles:
        return out

    order_local = np.lexsort(
        (
            ts_codes[idx],
            score[idx],
        )
    )
    ordered_idx = idx[
        order_local
    ]

    m = len(
        ordered_idx
    )

    q = (
        np.floor(
            np.arange(
                m,
                dtype=float,
            )
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

    out[
        ordered_idx
    ] = q

    return out


def evaluate_scores_by_day(
    df: pd.DataFrame,
    *,
    score_col: str,
    model_name: str,
    min_daily_n: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for trade_date, day in df.groupby(
        "trade_date",
        sort=True,
    ):
        score = pd.to_numeric(
            day[score_col],
            errors="coerce",
        ).to_numpy(dtype=float)

        target = pd.to_numeric(
            day[TARGET],
            errors="coerce",
        ).to_numpy(dtype=float)

        codes = (
            day["ts_code"]
            .astype(str)
            .to_numpy()
        )

        valid = (
            np.isfinite(score)
            & np.isfinite(target)
        )

        score = score[valid]
        target = target[valid]
        codes = codes[valid]

        n = len(
            score
        )

        row: Dict[str, Any] = {
            "trade_date": trade_date,
            "model": model_name,
            "n": int(n),
        }

        if n < min_daily_n:
            row.update({
                "ic": np.nan,
                "rank_ic": np.nan,
                "score_mean": np.nan,
                "score_std": np.nan,
                "target_mean": np.nan,
                "q1_ret_5d": np.nan,
                "q2_ret_5d": np.nan,
                "q3_ret_5d": np.nan,
                "q4_ret_5d": np.nan,
                "q5_ret_5d": np.nan,
                "q5_q1_ret_5d": np.nan,
                "q5_minus_universe_ret_5d": np.nan,
            })

            rows.append(
                row
            )
            continue

        row["ic"] = pearson_corr(
            score,
            target,
        )
        row["rank_ic"] = spearman_corr(
            score,
            target,
        )

        row["score_mean"] = float(
            np.mean(
                score
            )
        )
        row["score_std"] = float(
            np.std(
                score,
                ddof=0,
            )
        )
        row["target_mean"] = float(
            np.mean(
                target
            )
        )

        universe_ret = float(
            np.mean(
                target
            )
        )
        row[
            "universe_ew_ret_5d"
        ] = universe_ret

        q = deterministic_quantiles(
            score,
            codes,
            n_quantiles=N_QUANTILES,
        )

        for j in range(
            1,
            N_QUANTILES + 1,
        ):
            mask = (
                q == j
            )

            row[
                f"q{j}_ret_5d"
            ] = (
                float(
                    np.mean(
                        target[mask]
                    )
                )
                if mask.any()
                else np.nan
            )

            row[
                f"q{j}_n"
            ] = int(
                mask.sum()
            )

        row["q5_q1_ret_5d"] = (
            row["q5_ret_5d"]
            - row["q1_ret_5d"]
        )

        row[
            "q5_minus_universe_ret_5d"
        ] = (
            row["q5_ret_5d"]
            - universe_ret
        )

        rows.append(
            row
        )

    return rows


def newey_west_mean_stats(
    values: Sequence[float],
    lag: int,
) -> Tuple[float, float, float, int]:
    x = np.asarray(
        values,
        dtype=float,
    )
    x = x[
        np.isfinite(x)
    ]

    n = len(
        x
    )

    if n == 0:
        return (
            np.nan,
            np.nan,
            np.nan,
            0,
        )

    mean = float(
        np.mean(
            x
        )
    )

    if n < 2:
        return (
            mean,
            np.nan,
            np.nan,
            n,
        )

    e = x - mean

    L = min(
        int(lag),
        n - 1,
    )

    gamma0 = float(
        np.dot(
            e,
            e,
        )
        / n
    )

    lrv = gamma0

    for ell in range(
        1,
        L + 1,
    ):
        gamma_l = float(
            np.dot(
                e[ell:],
                e[:-ell],
            )
            / n
        )

        weight = (
            1.0
            - ell
            / (
                L + 1.0
            )
        )

        lrv += (
            2.0
            * weight
            * gamma_l
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

    return (
        mean,
        se,
        t,
        n,
    )


def simple_sd(
    values: Sequence[float],
) -> float:
    x = np.asarray(
        values,
        dtype=float,
    )
    x = x[
        np.isfinite(x)
    ]

    if len(x) < 2:
        return np.nan

    return float(
        np.std(
            x,
            ddof=1,
        )
    )


# ======================================================================================
# Ridge validation tuning
# ======================================================================================

def prediction_month_files(
    files: Sequence[Path],
    start_date: str,
    end_date: str,
) -> List[Path]:
    out = []

    for path in files:
        year, month, _ = parse_month_from_path(
            path
        )
        ym = f"{year:04d}{month:02d}"

        start_ym = start_date[:6]
        end_ym = end_date[:6]

        if (
            ym >= start_ym
            and ym <= end_ym
        ):
            out.append(
                path
            )

    return out


def build_validation_tuning(
    validation_files: Sequence[Path],
    *,
    day_stats: Sequence[DayStats],
    cfg: Config,
    logger: logging.Logger,
) -> pd.DataFrame:
    """
    Walk-forward validation for all Ridge lambda candidates.
    """
    accumulator = ExpandingStats(
        day_stats
    )

    daily_by_lambda: Dict[
        float,
        List[Dict[str, Any]]
    ] = {
        lam: []
        for lam in cfg.ridge_grid
    }

    mse_sse: Dict[
        float,
        float
    ] = {
        lam: 0.0
        for lam in cfg.ridge_grid
    }
    mse_n: Dict[
        float,
        int
    ] = {
        lam: 0
        for lam in cfg.ridge_grid
    }

    for i, path in enumerate(
        validation_files,
        1,
    ):
        df = load_signal_month(
            path
        )
        x = model_ready_month(
            df
        )

        x = x.loc[
            (x["trade_date"] >= VALIDATION_START)
            & (x["trade_date"] <= VALIDATION_END)
        ].copy()

        if x.empty:
            continue

        cutoff = str(
            x["trade_date"].min()
        )

        accumulator.advance_to(
            cutoff
        )

        xtx_mean, xty_mean = (
            accumulator.normalized_moments()
        )

        X = x[
            FEATURE_COLUMNS
        ].to_numpy(dtype=float)

        raw_y = x[
            TARGET
        ].to_numpy(dtype=float)

        daily_mean = (
            x.groupby(
                "trade_date",
                sort=False,
            )[TARGET]
            .transform("mean")
            .to_numpy(dtype=float)
        )

        y_xs = (
            raw_y
            - daily_mean
        )

        codes = x[
            "ts_code"
        ].astype(str).to_numpy()

        for lam in cfg.ridge_grid:
            beta, _ = solve_ridge(
                xtx_mean,
                xty_mean,
                lam,
            )

            score = (
                X @ beta
            )

            col = (
                f"_ridge_{lam:g}"
            )
            x[col] = score

            pred_err = (
                y_xs
                - score
            )

            mse_sse[lam] += float(
                pred_err @ pred_err
            )
            mse_n[lam] += len(
                pred_err
            )

            daily_by_lambda[
                lam
            ].extend(
                evaluate_scores_by_day(
                    x,
                    score_col=col,
                    model_name=f"RIDGE_{lam:g}",
                    min_daily_n=cfg.min_daily_n,
                )
            )

        logger.info(
            "RIDGE VALIDATION %d/%d | %s | cutoff=%s train_n=%d pred_rows=%d",
            i,
            len(validation_files),
            path.name,
            cutoff,
            accumulator.n,
            len(x),
        )

        del df, x
        gc.collect()

    rows = []

    for lam in cfg.ridge_grid:
        d = pd.DataFrame(
            daily_by_lambda[
                lam
            ]
        )

        mean_ic, ic_se, ic_t, ic_n = (
            newey_west_mean_stats(
                d["ic"],
                lag=cfg.hac_lag,
            )
        )

        (
            mean_rank_ic,
            ric_se,
            ric_t,
            ric_n,
        ) = newey_west_mean_stats(
            d["rank_ic"],
            lag=cfg.hac_lag,
        )

        (
            mean_spread,
            spread_se,
            spread_t,
            spread_n,
        ) = newey_west_mean_stats(
            d["q5_q1_ret_5d"],
            lag=cfg.hac_lag,
        )

        rows.append({
            "lambda": float(lam),
            "n_prediction_dates": int(
                d["trade_date"].nunique()
            ),
            "validation_mean_ic": mean_ic,
            "validation_ic_hac_se": ic_se,
            "validation_ic_hac_t": ic_t,
            "validation_mean_rank_ic": mean_rank_ic,
            "validation_rank_ic_hac_se": ric_se,
            "validation_rank_ic_hac_t": ric_t,
            "validation_mean_q5_q1_ret_5d": mean_spread,
            "validation_mean_q5_q1_ret_5d_bps": (
                mean_spread * 10000.0
                if np.isfinite(
                    mean_spread
                )
                else np.nan
            ),
            "validation_q5_q1_hac_se": spread_se,
            "validation_q5_q1_hac_t": spread_t,
            "validation_mse_target_xs": (
                mse_sse[lam]
                / mse_n[lam]
                if mse_n[lam] > 0
                else np.nan
            ),
        })

    tuning = pd.DataFrame(
        rows
    )

    # Primary: maximum validation mean RankIC.
    # Tie-break: maximum mean IC.
    # Final tie-break: smaller lambda.
    tuning = tuning.sort_values(
        [
            "validation_mean_rank_ic",
            "validation_mean_ic",
            "lambda",
        ],
        ascending=[
            False,
            False,
            True,
        ],
        kind="stable",
    ).reset_index(drop=True)

    tuning[
        "selected"
    ] = False

    if not tuning.empty:
        tuning.loc[
            0,
            "selected",
        ] = True

    return tuning


# ======================================================================================
# Final walk-forward predictions: OLS + selected Ridge
# ======================================================================================

def prediction_output_path(
    root: Path,
    year: int,
    month: int,
) -> Path:
    return (
        root
        / f"{year:04d}"
        / f"{month:02d}"
        / f"linear_predictions_{year:04d}{month:02d}.parquet"
    )


def build_final_walkforward(
    prediction_files: Sequence[Path],
    *,
    day_stats: Sequence[DayStats],
    selected_lambda: float,
    cfg: Config,
    logger: logging.Logger,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Returns:
        daily metrics,
        coefficient path,
        monthly prediction coverage
    """
    accumulator = ExpandingStats(
        day_stats
    )

    daily_rows: List[
        Dict[str, Any]
    ] = []
    coef_rows: List[
        Dict[str, Any]
    ] = []
    coverage_rows: List[
        Dict[str, Any]
    ] = []

    for i, path in enumerate(
        prediction_files,
        1,
    ):
        year, month, label = (
            parse_month_from_path(
                path
            )
        )

        df = load_signal_month(
            path
        )
        x = model_ready_month(
            df
        )

        x = x.loc[
            (x["trade_date"] >= PREDICTION_START)
            & (x["trade_date"] <= PREDICTION_END)
        ].copy()

        if x.empty:
            continue

        cutoff = str(
            x["trade_date"].min()
        )

        accumulator.advance_to(
            cutoff
        )

        xtx_mean, xty_mean = (
            accumulator.normalized_moments()
        )

        beta_ols, cond_ols = solve_ols(
            xtx_mean,
            xty_mean,
        )
        beta_ridge, cond_ridge = solve_ridge(
            xtx_mean,
            xty_mean,
            selected_lambda,
        )

        X = x[
            FEATURE_COLUMNS
        ].to_numpy(dtype=float)

        score_ols = (
            X @ beta_ols
        )
        score_ridge = (
            X @ beta_ridge
        )

        if not (
            np.isfinite(score_ols).all()
            and np.isfinite(score_ridge).all()
        ):
            raise RuntimeError(
                f"{label}: nonfinite model predictions."
            )

        x[
            "score_ols"
        ] = score_ols
        x[
            "score_ridge"
        ] = score_ridge

        daily_rows.extend(
            evaluate_scores_by_day(
                x,
                score_col="score_ols",
                model_name="OLS",
                min_daily_n=cfg.min_daily_n,
            )
        )

        daily_rows.extend(
            evaluate_scores_by_day(
                x,
                score_col="score_ridge",
                model_name="RIDGE",
                min_daily_n=cfg.min_daily_n,
            )
        )

        for model_name, beta, condition in [
            (
                "OLS",
                beta_ols,
                cond_ols,
            ),
            (
                "RIDGE",
                beta_ridge,
                cond_ridge,
            ),
        ]:
            row: Dict[str, Any] = {
                "prediction_month": label,
                "prediction_first_trade_date": cutoff,
                "model": model_name,
                "ridge_lambda": (
                    selected_lambda
                    if model_name == "RIDGE"
                    else 0.0
                ),
                "train_n": int(
                    accumulator.n
                ),
                "last_included_label_exit_date": (
                    accumulator.last_included_exit_date
                ),
                "matrix_condition_number": float(
                    condition
                ),
            }

            for feature_name, b in zip(
                FEATURE_NAMES,
                beta,
            ):
                row[
                    f"beta_{feature_name}"
                ] = float(
                    b
                )

            coef_rows.append(
                row
            )

        # Save compact stock-level predictions for later model comparison/backtest.
        save_cols = [
            "ts_code",
            "trade_date",
            TARGET,
            "target_entry_trade_date",
            EXIT_DATE_COL,
            "target_entry_tradable",
            "target_exit_tradable",
            "target_fully_tradable_5d",
        ]

        pred = x[
            save_cols
        ].copy()

        pred["score_ols"] = (
            score_ols
        )
        pred["score_ridge"] = (
            score_ridge
        )
        pred["selected_ridge_lambda"] = float(
            selected_lambda
        )
        pred["stage7_spec_version"] = (
            STAGE7_SPEC_VERSION
        )

        out_path = prediction_output_path(
            cfg.prediction_root,
            year,
            month,
        )

        atomic_write_parquet(
            pred,
            out_path,
        )

        coverage_rows.append({
            "prediction_month": label,
            "first_trade_date": cutoff,
            "last_trade_date": str(
                x["trade_date"].max()
            ),
            "prediction_rows": int(
                len(x)
            ),
            "prediction_dates": int(
                x["trade_date"].nunique()
            ),
            "train_n_at_month_start": int(
                accumulator.n
            ),
            "last_included_label_exit_date": (
                accumulator.last_included_exit_date
            ),
            "ols_score_missing_rows": int(
                np.isnan(
                    score_ols
                ).sum()
            ),
            "ridge_score_missing_rows": int(
                np.isnan(
                    score_ridge
                ).sum()
            ),
            "output_path": str(
                out_path
            ),
        })

        logger.info(
            "FINAL WF %d/%d | %s | cutoff=%s train_n=%d "
            "OLS cond=%.2e Ridge cond=%.2e pred_rows=%d",
            i,
            len(prediction_files),
            label,
            cutoff,
            accumulator.n,
            cond_ols,
            cond_ridge,
            len(x),
        )

        del df, x, pred
        gc.collect()

    daily = pd.DataFrame(
        daily_rows
    )
    coef = pd.DataFrame(
        coef_rows
    )
    coverage = pd.DataFrame(
        coverage_rows
    )

    return (
        daily,
        coef,
        coverage,
    )


# ======================================================================================
# Summary tables
# ======================================================================================

def summarize_model_period(
    g: pd.DataFrame,
    *,
    model: str,
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
        "model": model,
        "period": period,
        "start_date": start_date,
        "end_date": end_date,
        "n_dates": int(
            x["trade_date"].nunique()
        ),
        "mean_daily_n": safe_float(
            pd.to_numeric(
                x["n"],
                errors="coerce",
            ).mean()
        ),
    }

    (
        mean_ic,
        ic_se,
        ic_t,
        ic_n,
    ) = newey_west_mean_stats(
        x["ic"],
        lag=hac_lag,
    )

    sd_ic = simple_sd(
        x["ic"]
    )

    row["mean_ic"] = mean_ic
    row["sd_ic"] = sd_ic
    row["ic_hac_se"] = ic_se
    row["ic_hac_t"] = ic_t
    row["ic_n_dates"] = ic_n
    row["icir_daily"] = (
        mean_ic / sd_ic
        if (
            np.isfinite(mean_ic)
            and np.isfinite(sd_ic)
            and sd_ic > 0
        )
        else np.nan
    )
    row["icir_ann_sqrt252"] = (
        row["icir_daily"]
        * math.sqrt(
            252.0
        )
        if np.isfinite(
            row["icir_daily"]
        )
        else np.nan
    )

    (
        mean_ric,
        ric_se,
        ric_t,
        ric_n,
    ) = newey_west_mean_stats(
        x["rank_ic"],
        lag=hac_lag,
    )

    sd_ric = simple_sd(
        x["rank_ic"]
    )

    row["mean_rank_ic"] = mean_ric
    row["sd_rank_ic"] = sd_ric
    row["rank_ic_hac_se"] = ric_se
    row["rank_ic_hac_t"] = ric_t
    row["rank_ic_n_dates"] = ric_n
    row["rank_icir_daily"] = (
        mean_ric / sd_ric
        if (
            np.isfinite(mean_ric)
            and np.isfinite(sd_ric)
            and sd_ric > 0
        )
        else np.nan
    )
    row[
        "rank_icir_ann_sqrt252"
    ] = (
        row["rank_icir_daily"]
        * math.sqrt(
            252.0
        )
        if np.isfinite(
            row["rank_icir_daily"]
        )
        else np.nan
    )

    for j in range(
        1,
        N_QUANTILES + 1,
    ):
        qmean = safe_float(
            pd.to_numeric(
                x[f"q{j}_ret_5d"],
                errors="coerce",
            ).mean()
        )
        row[
            f"mean_q{j}_ret_5d"
        ] = qmean
        row[
            f"mean_q{j}_ret_5d_bps"
        ] = (
            qmean * 10000.0
            if np.isfinite(qmean)
            else np.nan
        )

    (
        spread_mean,
        spread_se,
        spread_t,
        spread_n,
    ) = newey_west_mean_stats(
        x["q5_q1_ret_5d"],
        lag=hac_lag,
    )

    row[
        "mean_q5_q1_ret_5d"
    ] = spread_mean
    row[
        "mean_q5_q1_ret_5d_bps"
    ] = (
        spread_mean * 10000.0
        if np.isfinite(
            spread_mean
        )
        else np.nan
    )
    row[
        "q5_q1_hac_se"
    ] = spread_se
    row[
        "q5_q1_hac_t"
    ] = spread_t
    row[
        "q5_q1_n_dates"
    ] = spread_n

    (
        q5ew_mean,
        q5ew_se,
        q5ew_t,
        q5ew_n,
    ) = newey_west_mean_stats(
        x[
            "q5_minus_universe_ret_5d"
        ],
        lag=hac_lag,
    )

    row[
        "mean_q5_minus_universe_ret_5d"
    ] = q5ew_mean
    row[
        "mean_q5_minus_universe_ret_5d_bps"
    ] = (
        q5ew_mean * 10000.0
        if np.isfinite(
            q5ew_mean
        )
        else np.nan
    )
    row[
        "q5_minus_universe_hac_se"
    ] = q5ew_se
    row[
        "q5_minus_universe_hac_t"
    ] = q5ew_t
    row[
        "q5_minus_universe_n_dates"
    ] = q5ew_n

    return row


def build_model_summary(
    daily: pd.DataFrame,
    *,
    hac_lag: int,
) -> pd.DataFrame:
    periods = {
        "VALIDATION_2017_2018": (
            VALIDATION_START,
            VALIDATION_END,
        ),
        "OOS_2019_2025": (
            OOS_START,
            OOS_END,
        ),
        "WF_2017_2025": (
            VALIDATION_START,
            OOS_END,
        ),
    }

    rows = []

    for model in [
        "OLS",
        "RIDGE",
    ]:
        g = daily.loc[
            daily["model"] == model
        ].copy()

        for period, (
            start_date,
            end_date,
        ) in periods.items():
            rows.append(
                summarize_model_period(
                    g,
                    model=model,
                    period=period,
                    start_date=start_date,
                    end_date=end_date,
                    hac_lag=hac_lag,
                )
            )

    return pd.DataFrame(
        rows
    )


def build_model_yearly(
    daily: pd.DataFrame,
    *,
    hac_lag: int,
) -> pd.DataFrame:
    x = daily.copy()
    x["year"] = (
        x["trade_date"]
        .str[:4]
        .astype(int)
    )

    rows = []

    for model in [
        "OLS",
        "RIDGE",
    ]:
        gm = x.loc[
            x["model"] == model
        ]

        for year, gy in gm.groupby(
            "year",
            sort=True,
        ):
            row = summarize_model_period(
                gy,
                model=model,
                period=f"YEAR_{year}",
                start_date=f"{year:04d}0101",
                end_date=f"{year:04d}1231",
                hac_lag=hac_lag,
            )
            row["year"] = int(
                year
            )
            rows.append(
                row
            )

    return pd.DataFrame(
        rows
    )


# ======================================================================================
# Final QA
# ======================================================================================

def final_qa(
    tuning: pd.DataFrame,
    daily: pd.DataFrame,
    summary: pd.DataFrame,
    coefficients: pd.DataFrame,
    coverage: pd.DataFrame,
) -> List[str]:
    issues: List[str] = []

    selected = tuning.loc[
        tuning["selected"] == True  # noqa: E712
    ]

    if len(selected) != 1:
        issues.append(
            f"Expected exactly one selected Ridge lambda, got {len(selected)}"
        )

    if daily.empty:
        issues.append(
            "daily model metrics are empty"
        )
        return issues

    if daily.duplicated(
        ["trade_date", "model"],
        keep=False,
    ).any():
        issues.append(
            "duplicate (trade_date, model) rows in daily metrics"
        )

    models = set(
        daily["model"].unique()
    )

    if models != {
        "OLS",
        "RIDGE",
    }:
        issues.append(
            f"Unexpected model set: {models}"
        )

    date_counts = (
        daily.groupby(
            "model"
        )["trade_date"]
        .nunique()
    )

    if date_counts.nunique() != 1:
        issues.append(
            f"OLS/Ridge date counts differ: {date_counts.to_dict()}"
        )

    for model in [
        "OLS",
        "RIDGE",
    ]:
        g = daily.loc[
            daily["model"] == model
        ]

        if pd.to_numeric(
            g["rank_ic"],
            errors="coerce",
        ).notna().sum() == 0:
            issues.append(
                f"{model}: no finite RankIC"
            )

        if pd.to_numeric(
            g["q5_q1_ret_5d"],
            errors="coerce",
        ).notna().sum() == 0:
            issues.append(
                f"{model}: no finite Q5-Q1"
            )

    if coefficients.empty:
        issues.append(
            "coefficient path is empty"
        )
    else:
        beta_cols = [
            f"beta_{x}"
            for x in FEATURE_NAMES
        ]

        for c in beta_cols:
            if not np.isfinite(
                pd.to_numeric(
                    coefficients[c],
                    errors="coerce",
                ).to_numpy(dtype=float)
            ).all():
                issues.append(
                    f"nonfinite coefficient column: {c}"
                )

    if coverage.empty:
        issues.append(
            "prediction coverage is empty"
        )
    else:
        if (
            coverage[
                "ols_score_missing_rows"
            ].sum()
            != 0
        ):
            issues.append(
                "OLS predictions contain missing scores"
            )

        if (
            coverage[
                "ridge_score_missing_rows"
            ].sum()
            != 0
        ):
            issues.append(
                "Ridge predictions contain missing scores"
            )

    oos = summary.loc[
        summary[
            "period"
        ] == "OOS_2019_2025"
    ]

    if len(oos) != 2:
        issues.append(
            f"Expected 2 OOS summary rows, got {len(oos)}"
        )

    return issues


# ======================================================================================
# CLI
# ======================================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Stage 7 pooled OLS + Ridge expanding-window linear baselines."
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
    )
    p.add_argument(
        "--min-daily-n",
        type=int,
        default=100,
    )
    p.add_argument(
        "--ridge-grid",
        default=",".join(
            str(x)
            for x in DEFAULT_RIDGE_GRID
        ),
        help=(
            "Comma-separated Ridge lambdas, default "
            + ",".join(
                str(x)
                for x in DEFAULT_RIDGE_GRID
            )
        ),
    )
    p.add_argument(
        "--high-corr-threshold",
        type=float,
        default=0.80,
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Recompute and overwrite Stage-7 monthly prediction files.",
    )

    return p.parse_args()


def parse_ridge_grid(
    raw: str,
) -> Tuple[float, ...]:
    vals = []

    for item in raw.split(","):
        item = item.strip()

        if not item:
            continue

        val = float(
            item
        )

        if val <= 0:
            raise ValueError(
                "Ridge lambdas must be strictly positive."
            )

        vals.append(
            val
        )

    if not vals:
        raise ValueError(
            "Ridge grid is empty."
        )

    return tuple(
        sorted(
            set(
                vals
            )
        )
    )


# ======================================================================================
# Main
# ======================================================================================

def main() -> int:
    args = parse_args()

    if args.hac_lag < 0:
        raise ValueError(
            "--hac-lag must be >= 0."
        )

    if args.min_daily_n < 10:
        raise ValueError(
            "--min-daily-n must be >= 10."
        )

    if not (
        0 < args.high_corr_threshold < 1
    ):
        raise ValueError(
            "--high-corr-threshold must lie in (0,1)."
        )

    ridge_grid = parse_ridge_grid(
        args.ridge_grid
    )

    cfg = Config(
        data_root=Path(
            args.data_root
        ),
        hac_lag=int(
            args.hac_lag
        ),
        min_daily_n=int(
            args.min_daily_n
        ),
        ridge_grid=ridge_grid,
        high_corr_threshold=float(
            args.high_corr_threshold
        ),
        force=bool(
            args.force
        ),
    )

    cfg.report_root.mkdir(
        parents=True,
        exist_ok=True,
    )
    cfg.prediction_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    logger = setup_logging(
        cfg.report_root
    )

    logger.info("=" * 104)
    logger.info("STAGE 7 | MULTIVARIATE LINEAR BASELINES")
    logger.info(
        "Frozen Stage-5 version: %s",
        REQUIRED_STAGE5_VERSION,
    )
    logger.info(
        "Features (%d): %s",
        P,
        ", ".join(
            FEATURE_NAMES
        ),
    )
    logger.info(
        "TRAIN=%s..%s | VALIDATION=%s..%s | OOS=%s..%s",
        TRAIN_START,
        TRAIN_END,
        VALIDATION_START,
        VALIDATION_END,
        OOS_START,
        OOS_END,
    )
    logger.info(
        "Ridge grid: %s",
        ", ".join(
            f"{x:g}"
            for x in cfg.ridge_grid
        ),
    )
    logger.info(
        "Monthly expanding refit with label purge: "
        "training observation allowed only when target_exit_trade_date < month cutoff."
    )
    logger.info(
        "Fit target: daily cross-sectionally demeaned 5-day return; no intercept."
    )
    logger.info("=" * 104)

    try:
        files = discover_signal_files(
            cfg.signal_root
        )

        logger.info(
            "Discovered %d Stage-5 monthly files.",
            len(files),
        )

        # ------------------------------------------------------------------
        # Pass 1: sufficient stats + TRAIN correlation moments
        # ------------------------------------------------------------------
        (
            day_stats,
            train_moments,
        ) = build_day_stats_and_train_moments(
            files,
            logger=logger,
        )

        if not day_stats:
            raise RuntimeError(
                "No model-ready day sufficient statistics found."
            )

        logger.info(
            "Built daily sufficient statistics for %d signal dates.",
            len(day_stats),
        )

        # TRAIN correlation / redundancy
        corr = correlation_from_moments(
            train_moments["n"],
            train_moments["sum"],
            train_moments["cross"],
        )

        corr_df = pd.DataFrame(
            corr,
            index=FEATURE_NAMES,
            columns=FEATURE_NAMES,
        )
        corr_df.index.name = "factor"

        redundancy_df = build_redundancy_report(
            corr,
            threshold=cfg.high_corr_threshold,
        )

        corr_path = (
            cfg.report_root
            / "linear_factor_correlation_train.csv"
        )
        redundancy_path = (
            cfg.report_root
            / "linear_factor_redundancy_train.csv"
        )

        corr_df.to_csv(
            corr_path,
            encoding="utf-8-sig",
        )
        redundancy_df.to_csv(
            redundancy_path,
            index=False,
            encoding="utf-8-sig",
        )

        # ------------------------------------------------------------------
        # Validation-only Ridge tuning
        # ------------------------------------------------------------------
        validation_files = prediction_month_files(
            files,
            VALIDATION_START,
            VALIDATION_END,
        )

        tuning = build_validation_tuning(
            validation_files,
            day_stats=day_stats,
            cfg=cfg,
            logger=logger,
        )

        selected_rows = tuning.loc[
            tuning["selected"] == True  # noqa: E712
        ]

        if len(selected_rows) != 1:
            raise RuntimeError(
                "Ridge tuning did not produce exactly one selected lambda."
            )

        selected_lambda = float(
            selected_rows.iloc[0][
                "lambda"
            ]
        )

        logger.info(
            "SELECTED RIDGE LAMBDA = %.10g | validation RankIC=%.6f | "
            "IC=%.6f | Q5-Q1=%.2f bps",
            selected_lambda,
            selected_rows.iloc[0][
                "validation_mean_rank_ic"
            ],
            selected_rows.iloc[0][
                "validation_mean_ic"
            ],
            selected_rows.iloc[0][
                "validation_mean_q5_q1_ret_5d_bps"
            ],
        )

        tuning_path = (
            cfg.report_root
            / "ridge_validation_tuning.csv"
        )

        tuning.to_csv(
            tuning_path,
            index=False,
            encoding="utf-8-sig",
        )

        # ------------------------------------------------------------------
        # Final walk-forward OLS + selected Ridge: Validation + OOS
        # ------------------------------------------------------------------
        pred_files = prediction_month_files(
            files,
            PREDICTION_START,
            PREDICTION_END,
        )

        (
            daily,
            coefficients,
            coverage,
        ) = build_final_walkforward(
            pred_files,
            day_stats=day_stats,
            selected_lambda=selected_lambda,
            cfg=cfg,
            logger=logger,
        )

        daily = daily.sort_values(
            [
                "trade_date",
                "model",
            ]
        ).reset_index(drop=True)

        coefficients = coefficients.sort_values(
            [
                "prediction_month",
                "model",
            ]
        ).reset_index(drop=True)

        coverage = coverage.sort_values(
            "prediction_month"
        ).reset_index(drop=True)

        summary = build_model_summary(
            daily,
            hac_lag=cfg.hac_lag,
        )

        yearly = build_model_yearly(
            daily,
            hac_lag=cfg.hac_lag,
        )

        # ------------------------------------------------------------------
        # QA
        # ------------------------------------------------------------------
        issues = final_qa(
            tuning,
            daily,
            summary,
            coefficients,
            coverage,
        )

        # ------------------------------------------------------------------
        # Save reports
        # ------------------------------------------------------------------
        coef_path = (
            cfg.report_root
            / "linear_model_coefficients.csv"
        )
        daily_path = (
            cfg.report_root
            / "linear_model_daily_metrics.csv"
        )
        summary_path = (
            cfg.report_root
            / "linear_model_summary.csv"
        )
        yearly_path = (
            cfg.report_root
            / "linear_model_yearly.csv"
        )
        coverage_path = (
            cfg.report_root
            / "linear_model_prediction_coverage.csv"
        )

        coefficients.to_csv(
            coef_path,
            index=False,
            encoding="utf-8-sig",
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
        coverage.to_csv(
            coverage_path,
            index=False,
            encoding="utf-8-sig",
        )

        # Metadata
        eigvals = np.linalg.eigvalsh(
            corr
        )

        metadata = {
            "project": (
                "China A-Share Cross-Sectional Alpha Research"
            ),
            "script": "07_linear_baselines.py",
            "stage7_spec_version": STAGE7_SPEC_VERSION,
            "generated_at_utc": datetime.now(
                timezone.utc
            ).isoformat(),
            "required_stage5_spec_version": REQUIRED_STAGE5_VERSION,
            "features": FEATURE_NAMES,
            "feature_columns": FEATURE_COLUMNS,
            "target_raw": TARGET,
            "fit_target": (
                "target_xs = target_ret_5d - same-day model-ready cross-sectional mean"
            ),
            "intercept": False,
            "train_period": [
                TRAIN_START,
                TRAIN_END,
            ],
            "validation_period": [
                VALIDATION_START,
                VALIDATION_END,
            ],
            "oos_period": [
                OOS_START,
                OOS_END,
            ],
            "refit_frequency": "monthly",
            "label_purge_rule": (
                "include training row only if target_exit_trade_date "
                "< first prediction trade date of current month"
            ),
            "ridge_grid": list(
                cfg.ridge_grid
            ),
            "ridge_selection_metric": (
                "maximum validation mean daily RankIC; tie-break mean IC, then smaller lambda"
            ),
            "selected_ridge_lambda": selected_lambda,
            "hac_lag": cfg.hac_lag,
            "min_daily_n": cfg.min_daily_n,
            "train_factor_correlation_min_eigenvalue": float(
                np.min(
                    eigvals
                )
            ),
            "train_factor_correlation_max_eigenvalue": float(
                np.max(
                    eigvals
                )
            ),
            "train_factor_correlation_condition_number": float(
                np.linalg.cond(
                    corr
                )
            ),
            "qa_issue_count": len(
                issues
            ),
            "qa_issues": issues,
            "research_notes": [
                "All 11 frozen Stage-6 factors are retained; no ex-post feature deletion or sign flipping.",
                "Model scores are trained to predict higher future cross-sectional returns, so higher score means higher predicted return.",
                "Ridge lambda is selected using validation 2017-2018 only; OOS 2019-2025 is not used for tuning.",
                "Monthly refits use expanding historical data but purge observations whose t+6 label was not yet observable at the monthly cutoff.",
                "Prediction evaluation uses raw forward returns; fit target is daily demeaned only to remove common market return from pooled regression.",
                "Stage 7 remains a predictive-model comparison. Transaction costs and self-financing portfolio accounting are deferred to the executable backtest stage.",
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
            / "linear_model_metadata.json",
        )

        logger.info("=" * 104)
        logger.info(
            "STAGE 7 COMPLETE | selected Ridge lambda=%.10g | "
            "daily model rows=%d | coefficient rows=%d",
            selected_lambda,
            len(daily),
            len(coefficients),
        )
        logger.info(
            "Reports: %s | %s | %s | %s | %s | %s | %s",
            corr_path,
            redundancy_path,
            tuning_path,
            coef_path,
            daily_path,
            summary_path,
            yearly_path,
        )

        if issues:
            logger.error(
                "STAGE 7 QA FAIL | %d issue(s)",
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
            "STAGE 7 QA: PASS"
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
            "Fatal error during Stage-7 linear baseline analysis."
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
