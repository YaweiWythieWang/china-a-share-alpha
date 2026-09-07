#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
08_xgboost_nonlinear.py

China A-Share Cross-Sectional Alpha Research
Stage 8: Nonlinear Model — XGBoost

PURPOSE
-------
Test whether a nonlinear tree-boosting model provides incremental out-of-sample
cross-sectional predictive value beyond the frozen linear baselines.

The experiment is intentionally conservative:

    Frozen inputs:
        - exactly the same 11 Stage-5 v4 features;
        - the same 5-day forward target;
        - the same daily cross-sectional target demeaning used in Stage 7;
        - no ex-post feature deletion or sign flipping.

    Validation:
        2017-2018 only.

    OOS:
        2019-2025 only.

    Label purge:
        A historical observation may enter training only when its complete
        t+1 -> t+6 target is observable strictly before the current refit date.

WHY ANNUAL REFITS?
------------------
The Stage-5 model-ready sample contains several million stock-date rows.
Unlike Ridge, XGBoost has no low-dimensional sufficient-statistic update that
makes a complete monthly expanding refit nearly free.

Therefore Stage 8 precommits to ANNUAL expanding refits:

    Validation refits:
        first trading day of 2017
        first trading day of 2018

    OOS refits:
        first trading day of 2019, ..., first trading day of 2025

For an apples-to-apples refit-frequency comparison, the script also computes:

    RIDGE_ANNUAL
        Stage-7 selected Ridge penalty, but refit annually using exactly the
        same training cutoffs as XGBoost.

In addition, when available, the script imports Stage-7 daily metrics for:

    RIDGE_STAGE7_MONTHLY

which is the stronger monthly-refit linear benchmark.

XGBOOST MODEL SELECTION
-----------------------
A small PRECOMMITTED candidate set is evaluated on 2017-2018 Validation only.
This is not a broad hyperparameter search.

All candidates use:
    objective          = reg:squarederror
    tree_method        = hist
    learning_rate      = 0.05
    num_boost_round    = 250
    subsample          = 0.80
    colsample_bytree   = 0.90
    max_bin            = 256
    reg_alpha          = 0
    fixed random seed

Candidate structural complexity:
    XGB_A_SHALLOW
        max_depth=2, min_child_weight=100, reg_lambda=10

    XGB_B_BALANCED
        max_depth=3, min_child_weight=100, reg_lambda=10

    XGB_C_STRONG_REG
        max_depth=3, min_child_weight=300, reg_lambda=30

    XGB_D_DEEPER
        max_depth=4, min_child_weight=100, reg_lambda=10

Primary selection criterion:
    maximum Validation mean daily RankIC

Tie-breakers:
    higher Validation mean daily IC
    shallower max_depth
    larger min_child_weight
    larger reg_lambda
    lexicographically smaller candidate id

No OOS result is used for model selection.

TARGET
------
For fitting:
    target_xs(i,t)
      = target_ret_5d(i,t)
        - same-day cross-sectional mean(target_ret_5d)

For evaluation:
    original raw target_ret_5d.

FEATURES
--------
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

TRAINING CACHE
--------------
To keep memory usage predictable, Stage 8 builds an on-disk NumPy memmap cache:

    data/cache/stage8_xgb/

The cache stores:
    X float32
    target_xs float32
    signal_date int32 YYYYMMDD
    target_exit_date int32 YYYYMMDD

Rows are sorted chronologically. Because target exit date is monotone with signal
date, each annual purged training sample is a prefix of the cache.

PREDICTIONS
-----------
Monthly stock-level XGBoost and annual-Ridge scores are saved under:

    data/processed/xgb_predictions/YYYY/MM/xgb_predictions_YYYYMM.parquet

EVALUATION
----------
For each model:
    daily Pearson IC
    daily Spearman RankIC
    Q1...Q5 equal-weight forward returns
    Q5-Q1 predictive spread
    Q5 - universe EW spread

Higher model score always means higher predicted future return.

Because adjacent 5-day targets overlap, summary inference uses Newey-West/HAC
with default lag 5.

OUTPUT
------
data/xgb_model_reports/
    xgb_validation_tuning.csv
    xgb_model_daily_metrics.csv
    xgb_model_summary.csv
    xgb_model_yearly.csv
    xgb_feature_importance.csv
    xgb_prediction_coverage.csv
    xgb_vs_linear_comparison.csv
    xgb_metadata.json
    08_xgboost_nonlinear.log

RUN IN ANACONDA PROMPT
----------------------
python 08_xgboost_nonlinear.py

If XGBoost is not installed:
    pip install xgboost

Optional GPU on modern XGBoost:
    python 08_xgboost_nonlinear.py --device cuda

To rebuild the training cache:
    python 08_xgboost_nonlinear.py --rebuild-cache
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import platform
import re
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import xgboost as xgb
except ImportError as exc:
    raise RuntimeError(
        "Stage 8 requires XGBoost. In Anaconda Prompt run:\n"
        "    pip install xgboost\n"
        "then rerun this script."
    ) from exc


# ======================================================================================
# Frozen research specification
# ======================================================================================

REQUIRED_STAGE5_VERSION = "v4_final_positional"
STAGE8_SPEC_VERSION = "v1_xgb_annual_walkforward"

TRAIN_START = "20100101"
TRAIN_END = "20161231"

VALIDATION_START = "20170101"
VALIDATION_END = "20181231"

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

TARGET = "target_ret_5d"
EXIT_DATE_COL = "target_exit_trade_date"
ENTRY_DATE_COL = "target_entry_trade_date"

N_QUANTILES = 5
RANDOM_SEED = 20260906

# This is only a fallback. The script first tries to read the selected Ridge
# lambda from Stage-7 metadata / tuning output.
STAGE7_RIDGE_LAMBDA_FALLBACK = 100000.0


XGB_CANDIDATES: Tuple[Dict[str, Any], ...] = (
    {
        "candidate_id": "XGB_A_SHALLOW",
        "max_depth": 2,
        "min_child_weight": 100.0,
        "reg_lambda": 10.0,
    },
    {
        "candidate_id": "XGB_B_BALANCED",
        "max_depth": 3,
        "min_child_weight": 100.0,
        "reg_lambda": 10.0,
    },
    {
        "candidate_id": "XGB_C_STRONG_REG",
        "max_depth": 3,
        "min_child_weight": 300.0,
        "reg_lambda": 30.0,
    },
    {
        "candidate_id": "XGB_D_DEEPER",
        "max_depth": 4,
        "min_child_weight": 100.0,
        "reg_lambda": 10.0,
    },
)

FIXED_XGB_PARAMS = {
    "objective": "reg:squarederror",
    "tree_method": "hist",
    "eta": 0.05,
    "subsample": 0.80,
    "colsample_bytree": 0.90,
    "max_bin": 256,
    "reg_alpha": 0.0,
    "base_score": 0.0,
}

NUM_BOOST_ROUND = 250


# ======================================================================================
# Configuration
# ======================================================================================

@dataclass
class Config:
    data_root: Path
    hac_lag: int
    min_daily_n: int
    nthread: int
    device: str
    rebuild_cache: bool

    @property
    def signal_root(self) -> Path:
        return self.data_root / "processed" / "signals"

    @property
    def cache_root(self) -> Path:
        return self.data_root / "cache" / "stage8_xgb"

    @property
    def prediction_root(self) -> Path:
        return self.data_root / "processed" / "xgb_predictions"

    @property
    def report_root(self) -> Path:
        return self.data_root / "xgb_model_reports"

    @property
    def stage7_report_root(self) -> Path:
        return self.data_root / "linear_model_reports"


# ======================================================================================
# Logging / utility
# ======================================================================================

def setup_logging(report_root: Path) -> logging.Logger:
    report_root.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("xgb_stage8")
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
        report_root / "08_xgboost_nonlinear.log",
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


def date_to_int(s: pd.Series) -> np.ndarray:
    return (
        normalize_date_series(s)
        .astype("Int64")
        .to_numpy(dtype=np.int64, na_value=-1)
        .astype(np.int32)
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


def parse_month_from_path(path: Path) -> Tuple[int, int, str]:
    m = re.search(
        r"signals_(\d{4})(\d{2})\.parquet$",
        path.name,
    )

    if not m:
        raise ValueError(f"Cannot parse month from {path}")

    year = int(m.group(1))
    month = int(m.group(2))
    return year, month, f"{year:04d}-{month:02d}"


def xgb_major_version() -> int:
    raw = str(xgb.__version__)
    m = re.match(r"(\d+)", raw)

    if not m:
        return 0

    return int(m.group(1))


# ======================================================================================
# Stage-5 loading
# ======================================================================================

def discover_signal_files(signal_root: Path) -> List[Path]:
    files = sorted(
        signal_root.glob("*/*/signals_*.parquet")
    )

    if not files:
        raise FileNotFoundError(
            f"No Stage-5 signal files under {signal_root}"
        )

    return files


def required_columns() -> List[str]:
    return [
        "ts_code",
        "trade_date",
        "model_ready_5d",
        "stage5_spec_version",
        TARGET,
        ENTRY_DATE_COL,
        EXIT_DATE_COL,
        "target_entry_tradable",
        "target_exit_tradable",
        "target_fully_tradable_5d",
        *FEATURE_COLUMNS,
    ]


def load_signal_month(path: Path) -> pd.DataFrame:
    cols = required_columns()

    try:
        df = pd.read_parquet(
            path,
            columns=cols,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Cannot read Stage-5 monthly file {path}"
        ) from exc

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
            f"{path}: expected {REQUIRED_STAGE5_VERSION!r}, "
            f"got {versions.tolist()}"
        )

    df["trade_date"] = normalize_date_series(df["trade_date"])
    df[ENTRY_DATE_COL] = normalize_date_series(df[ENTRY_DATE_COL])
    df[EXIT_DATE_COL] = normalize_date_series(df[EXIT_DATE_COL])

    return df


def model_ready_month(df: pd.DataFrame) -> pd.DataFrame:
    ready = (
        df["model_ready_5d"]
        .fillna(False)
        .astype(bool)
    )

    x = df.loc[ready].copy()

    for c in [*FEATURE_COLUMNS, TARGET]:
        x[c] = pd.to_numeric(
            x[c],
            errors="coerce",
        )

    finite = np.ones(len(x), dtype=bool)

    for c in [*FEATURE_COLUMNS, TARGET]:
        finite &= np.isfinite(
            x[c].to_numpy(dtype=float)
        )

    x = x.loc[finite].copy()

    if x.duplicated(
        ["ts_code", "trade_date"],
        keep=False,
    ).any():
        raise RuntimeError(
            "Duplicate security-date keys in model-ready Stage-5 month."
        )

    x = x.sort_values(
        ["trade_date", "ts_code"]
    ).reset_index(drop=True)

    return x


def add_cross_sectional_target(df: pd.DataFrame) -> pd.DataFrame:
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


# ======================================================================================
# On-disk training cache
# ======================================================================================

CACHE_META_NAME = "cache_meta.json"
CACHE_X_NAME = "X.npy"
CACHE_Y_NAME = "y_xs.npy"
CACHE_SIGNAL_DATE_NAME = "signal_date.npy"
CACHE_EXIT_DATE_NAME = "exit_date.npy"


def cache_meta_path(cfg: Config) -> Path:
    return cfg.cache_root / CACHE_META_NAME


def expected_cache_signature() -> Dict[str, Any]:
    return {
        "stage8_cache_version": "v1",
        "required_stage5_version": REQUIRED_STAGE5_VERSION,
        "feature_names": FEATURE_NAMES,
        "feature_columns": FEATURE_COLUMNS,
        "dtype_X": "float32",
        "dtype_y": "float32",
        "dtype_dates": "int32",
    }


def cache_is_valid(cfg: Config) -> bool:
    meta_path = cache_meta_path(cfg)

    required_files = [
        cfg.cache_root / CACHE_X_NAME,
        cfg.cache_root / CACHE_Y_NAME,
        cfg.cache_root / CACHE_SIGNAL_DATE_NAME,
        cfg.cache_root / CACHE_EXIT_DATE_NAME,
    ]

    if (
        not meta_path.exists()
        or not all(p.exists() for p in required_files)
    ):
        return False

    try:
        meta = json.loads(
            meta_path.read_text(encoding="utf-8")
        )
    except Exception:
        return False

    sig = expected_cache_signature()

    for k, v in sig.items():
        if meta.get(k) != v:
            return False

    try:
        total_rows = int(meta["total_rows"])
        X = np.load(
            cfg.cache_root / CACHE_X_NAME,
            mmap_mode="r",
        )
        y = np.load(
            cfg.cache_root / CACHE_Y_NAME,
            mmap_mode="r",
        )
        sd = np.load(
            cfg.cache_root / CACHE_SIGNAL_DATE_NAME,
            mmap_mode="r",
        )
        ed = np.load(
            cfg.cache_root / CACHE_EXIT_DATE_NAME,
            mmap_mode="r",
        )

        return (
            X.shape == (total_rows, P)
            and y.shape == (total_rows,)
            and sd.shape == (total_rows,)
            and ed.shape == (total_rows,)
        )
    except Exception:
        return False


def build_training_cache(
    files: Sequence[Path],
    *,
    cfg: Config,
    logger: logging.Logger,
) -> Dict[str, Any]:
    if cfg.rebuild_cache and cfg.cache_root.exists():
        logger.info(
            "Removing existing Stage-8 cache because --rebuild-cache was requested."
        )
        shutil.rmtree(cfg.cache_root)

    if cache_is_valid(cfg):
        meta = json.loads(
            cache_meta_path(cfg).read_text(encoding="utf-8")
        )
        logger.info(
            "REUSE Stage-8 training cache | rows=%d",
            int(meta["total_rows"]),
        )
        return meta

    cfg.cache_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------------------
    # Pass 1: row count.
    # ------------------------------------------------------------------
    total_rows = 0
    month_counts: List[Dict[str, Any]] = []

    for i, path in enumerate(files, 1):
        df = load_signal_month(path)
        x = model_ready_month(df)

        n = len(x)
        total_rows += n

        year, month, label = parse_month_from_path(path)
        month_counts.append({
            "month": label,
            "rows": n,
        })

        logger.info(
            "CACHE COUNT %d/%d | %s | rows=%d cumulative=%d",
            i,
            len(files),
            path.name,
            n,
            total_rows,
        )

        del df, x
        gc.collect()

    if total_rows <= 0:
        raise RuntimeError("Stage-8 cache has zero rows.")

    # ------------------------------------------------------------------
    # Allocate .npy memmaps.
    # ------------------------------------------------------------------
    X_mm = np.lib.format.open_memmap(
        cfg.cache_root / CACHE_X_NAME,
        mode="w+",
        dtype=np.float32,
        shape=(total_rows, P),
    )
    y_mm = np.lib.format.open_memmap(
        cfg.cache_root / CACHE_Y_NAME,
        mode="w+",
        dtype=np.float32,
        shape=(total_rows,),
    )
    signal_mm = np.lib.format.open_memmap(
        cfg.cache_root / CACHE_SIGNAL_DATE_NAME,
        mode="w+",
        dtype=np.int32,
        shape=(total_rows,),
    )
    exit_mm = np.lib.format.open_memmap(
        cfg.cache_root / CACHE_EXIT_DATE_NAME,
        mode="w+",
        dtype=np.int32,
        shape=(total_rows,),
    )

    # ------------------------------------------------------------------
    # Pass 2: fill in chronological order.
    # ------------------------------------------------------------------
    pos = 0
    first_signal_date: Optional[int] = None
    last_signal_date: Optional[int] = None
    first_exit_date: Optional[int] = None
    last_exit_date: Optional[int] = None

    for i, path in enumerate(files, 1):
        df = load_signal_month(path)
        x = model_ready_month(df)

        if x.empty:
            continue

        x = add_cross_sectional_target(x)

        n = len(x)
        sl = slice(pos, pos + n)

        X_arr = x[
            FEATURE_COLUMNS
        ].to_numpy(dtype=np.float32)

        y_arr = x[
            "target_xs"
        ].to_numpy(dtype=np.float32)

        signal_arr = date_to_int(
            x["trade_date"]
        )
        exit_arr = date_to_int(
            x[EXIT_DATE_COL]
        )

        if not (
            np.isfinite(X_arr).all()
            and np.isfinite(y_arr).all()
            and (signal_arr > 0).all()
            and (exit_arr > 0).all()
        ):
            raise RuntimeError(
                f"{path}: nonfinite or invalid cache input."
            )

        if len(exit_arr) > 1 and np.any(
            np.diff(exit_arr.astype(np.int64)) < 0
        ):
            raise RuntimeError(
                f"{path}: exit dates are not monotone within month."
            )

        if pos > 0:
            if int(exit_arr[0]) < int(exit_mm[pos - 1]):
                raise RuntimeError(
                    f"{path}: cache exit-date chronology is not monotone."
                )

        X_mm[sl] = X_arr
        y_mm[sl] = y_arr
        signal_mm[sl] = signal_arr
        exit_mm[sl] = exit_arr

        if first_signal_date is None:
            first_signal_date = int(signal_arr[0])
            first_exit_date = int(exit_arr[0])

        last_signal_date = int(signal_arr[-1])
        last_exit_date = int(exit_arr[-1])

        pos += n

        logger.info(
            "CACHE FILL %d/%d | %s | rows=%d written=%d/%d",
            i,
            len(files),
            path.name,
            n,
            pos,
            total_rows,
        )

        del (
            df,
            x,
            X_arr,
            y_arr,
            signal_arr,
            exit_arr,
        )
        gc.collect()

    if pos != total_rows:
        raise RuntimeError(
            f"Cache row count mismatch: expected {total_rows}, wrote {pos}."
        )

    X_mm.flush()
    y_mm.flush()
    signal_mm.flush()
    exit_mm.flush()

    meta = {
        **expected_cache_signature(),
        "total_rows": int(total_rows),
        "first_signal_date": int(first_signal_date),
        "last_signal_date": int(last_signal_date),
        "first_exit_date": int(first_exit_date),
        "last_exit_date": int(last_exit_date),
        "month_counts": month_counts,
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
    }

    atomic_write_json(
        meta,
        cache_meta_path(cfg),
    )

    logger.info(
        "CACHE BUILD COMPLETE | rows=%d | signal=%s..%s | exit=%s..%s",
        total_rows,
        first_signal_date,
        last_signal_date,
        first_exit_date,
        last_exit_date,
    )

    return meta


@dataclass
class CacheArrays:
    X: np.ndarray
    y: np.ndarray
    signal_date: np.ndarray
    exit_date: np.ndarray


def open_training_cache(cfg: Config) -> CacheArrays:
    return CacheArrays(
        X=np.load(
            cfg.cache_root / CACHE_X_NAME,
            mmap_mode="r",
        ),
        y=np.load(
            cfg.cache_root / CACHE_Y_NAME,
            mmap_mode="r",
        ),
        signal_date=np.load(
            cfg.cache_root / CACHE_SIGNAL_DATE_NAME,
            mmap_mode="r",
        ),
        exit_date=np.load(
            cfg.cache_root / CACHE_EXIT_DATE_NAME,
            mmap_mode="r",
        ),
    )


def training_prefix_end(
    cache: CacheArrays,
    cutoff_date: str,
) -> int:
    cutoff = int(cutoff_date)

    # Training rule:
    #     target_exit_trade_date < cutoff_date
    # Since exit_date is monotone, the eligible training set is a prefix.
    idx = int(
        np.searchsorted(
            cache.exit_date,
            cutoff,
            side="left",
        )
    )

    if idx <= P:
        raise RuntimeError(
            f"Insufficient purged training sample before {cutoff_date}: n={idx}"
        )

    if idx > 0:
        last_exit = int(cache.exit_date[idx - 1])

        if last_exit >= cutoff:
            raise RuntimeError(
                f"Label purge failed: last_exit={last_exit}, cutoff={cutoff}"
            )

    return idx


# ======================================================================================
# Refit schedule
# ======================================================================================

def files_for_year(
    files: Sequence[Path],
    year: int,
) -> List[Path]:
    out = []

    for p in files:
        y, _, _ = parse_month_from_path(p)

        if y == year:
            out.append(p)

    return out


def load_prediction_year(
    files: Sequence[Path],
    year: int,
) -> pd.DataFrame:
    frames = []

    for path in files_for_year(files, year):
        df = load_signal_month(path)
        x = model_ready_month(df)

        if not x.empty:
            frames.append(x)

    if not frames:
        raise RuntimeError(
            f"No model-ready prediction rows for year {year}."
        )

    out = pd.concat(
        frames,
        ignore_index=True,
        sort=False,
    )

    out = out.sort_values(
        ["trade_date", "ts_code"]
    ).reset_index(drop=True)

    return out


def first_prediction_trade_date(
    files: Sequence[Path],
    year: int,
) -> str:
    x = load_prediction_year(files, year)

    try:
        return str(x["trade_date"].min())
    finally:
        del x
        gc.collect()


# ======================================================================================
# XGBoost
# ======================================================================================

def xgb_params_for_candidate(
    candidate: Dict[str, Any],
    *,
    cfg: Config,
) -> Dict[str, Any]:
    params = {
        **FIXED_XGB_PARAMS,
        "max_depth": int(candidate["max_depth"]),
        "min_child_weight": float(candidate["min_child_weight"]),
        "reg_lambda": float(candidate["reg_lambda"]),
        "seed": RANDOM_SEED,
        "nthread": cfg.nthread,
        "verbosity": 0,
    }

    if cfg.device != "cpu":
        if xgb_major_version() < 2:
            raise RuntimeError(
                f"--device {cfg.device!r} requires a modern XGBoost version. "
                f"Detected xgboost {xgb.__version__}. "
                "Upgrade xgboost or use --device cpu."
            )

        params["device"] = cfg.device

    return params


def fit_xgb(
    cache: CacheArrays,
    *,
    n_train: int,
    candidate: Dict[str, Any],
    cfg: Config,
) -> xgb.Booster:
    X_train = cache.X[:n_train]
    y_train = cache.y[:n_train]

    dtrain = xgb.DMatrix(
        X_train,
        label=y_train,
        feature_names=FEATURE_NAMES,
        nthread=cfg.nthread,
    )

    params = xgb_params_for_candidate(
        candidate,
        cfg=cfg,
    )

    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=NUM_BOOST_ROUND,
        verbose_eval=False,
    )

    return booster


def predict_xgb(
    booster: xgb.Booster,
    X: np.ndarray,
    *,
    cfg: Config,
) -> np.ndarray:
    dtest = xgb.DMatrix(
        X,
        feature_names=FEATURE_NAMES,
        nthread=cfg.nthread,
    )

    pred = booster.predict(dtest)

    return np.asarray(
        pred,
        dtype=float,
    )


# ======================================================================================
# Same-frequency Ridge
# ======================================================================================

def read_stage7_selected_lambda(cfg: Config) -> float:
    meta_path = (
        cfg.stage7_report_root
        / "linear_model_metadata.json"
    )

    if meta_path.exists():
        try:
            meta = json.loads(
                meta_path.read_text(
                    encoding="utf-8"
                )
            )
            val = float(
                meta["selected_ridge_lambda"]
            )

            if val > 0 and np.isfinite(val):
                return val
        except Exception:
            pass

    tuning_path = (
        cfg.stage7_report_root
        / "ridge_validation_tuning.csv"
    )

    if tuning_path.exists():
        try:
            t = pd.read_csv(tuning_path)
            sel = t.loc[
                t["selected"].astype(str).str.lower().isin(
                    ["true", "1"]
                )
            ]

            if len(sel) == 1:
                val = float(sel.iloc[0]["lambda"])

                if val > 0 and np.isfinite(val):
                    return val
        except Exception:
            pass

    return STAGE7_RIDGE_LAMBDA_FALLBACK


def fit_ridge_from_cache(
    cache: CacheArrays,
    *,
    n_train: int,
    lam: float,
) -> Tuple[np.ndarray, float]:
    X = np.asarray(
        cache.X[:n_train],
        dtype=np.float64,
    )
    y = np.asarray(
        cache.y[:n_train],
        dtype=np.float64,
    )

    xtx_mean = (
        X.T @ X
        / n_train
    )
    xty_mean = (
        X.T @ y
        / n_train
    )

    A = (
        xtx_mean
        + float(lam)
        * np.eye(
            P,
            dtype=float,
        )
    )

    cond = float(
        np.linalg.cond(A)
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

    return beta, cond


# ======================================================================================
# Daily evaluation
# ======================================================================================

def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]

    if len(x) < 3:
        return np.nan

    sx = float(np.std(x, ddof=0))
    sy = float(np.std(y, ddof=0))

    if sx <= 0 or sy <= 0:
        return np.nan

    return float(np.corrcoef(x, y)[0, 1])


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    valid = np.isfinite(x) & np.isfinite(y)
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


def deterministic_quantiles(
    score: np.ndarray,
    ts_codes: np.ndarray,
    n_quantiles: int = N_QUANTILES,
) -> np.ndarray:
    score = np.asarray(score, dtype=float)
    ts_codes = np.asarray(ts_codes, dtype=str)

    n = len(score)
    out = np.full(n, np.nan, dtype=float)

    valid = np.isfinite(score)
    idx = np.flatnonzero(valid)

    if len(idx) < n_quantiles:
        return out

    order_local = np.lexsort(
        (
            ts_codes[idx],
            score[idx],
        )
    )
    ordered_idx = idx[order_local]

    m = len(ordered_idx)

    q = (
        np.floor(
            np.arange(m, dtype=float)
            * n_quantiles
            / m
        )
        .astype(int)
        + 1
    )

    q = np.minimum(q, n_quantiles)

    out[ordered_idx] = q.astype(float)

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

        n = len(score)

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
                "universe_ew_ret_5d": np.nan,
                "q1_ret_5d": np.nan,
                "q2_ret_5d": np.nan,
                "q3_ret_5d": np.nan,
                "q4_ret_5d": np.nan,
                "q5_ret_5d": np.nan,
                "q5_q1_ret_5d": np.nan,
                "q5_minus_universe_ret_5d": np.nan,
            })

            rows.append(row)
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
            np.mean(score)
        )
        row["score_std"] = float(
            np.std(score, ddof=0)
        )

        universe_ret = float(
            np.mean(target)
        )
        row["universe_ew_ret_5d"] = universe_ret

        q = deterministic_quantiles(
            score,
            codes,
            n_quantiles=N_QUANTILES,
        )

        for j in range(1, N_QUANTILES + 1):
            mask = q == j

            row[f"q{j}_n"] = int(mask.sum())
            row[f"q{j}_ret_5d"] = (
                float(np.mean(target[mask]))
                if mask.any()
                else np.nan
            )

        row["q5_q1_ret_5d"] = (
            row["q5_ret_5d"]
            - row["q1_ret_5d"]
        )

        row["q5_minus_universe_ret_5d"] = (
            row["q5_ret_5d"]
            - universe_ret
        )

        rows.append(row)

    return rows


def newey_west_mean_stats(
    values: Sequence[float],
    lag: int,
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

    gamma0 = float(
        np.dot(e, e) / n
    )
    lrv = gamma0

    for ell in range(1, L + 1):
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
            / (L + 1.0)
        )

        lrv += (
            2.0
            * weight
            * gamma_l
        )

    lrv = max(lrv, 0.0)

    se = math.sqrt(
        lrv / n
    )

    t = (
        mean / se
        if se > 0
        else np.nan
    )

    return mean, se, t, n


def simple_sd(values: Sequence[float]) -> float:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]

    if len(x) < 2:
        return np.nan

    return float(np.std(x, ddof=1))


# ======================================================================================
# Summary
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

    mean_ic, ic_se, ic_t, ic_n = (
        newey_west_mean_stats(
            x["ic"],
            lag=hac_lag,
        )
    )

    sd_ic = simple_sd(x["ic"])

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
        row["icir_daily"] * math.sqrt(252.0)
        if np.isfinite(row["icir_daily"])
        else np.nan
    )

    mean_ric, ric_se, ric_t, ric_n = (
        newey_west_mean_stats(
            x["rank_ic"],
            lag=hac_lag,
        )
    )

    sd_ric = simple_sd(x["rank_ic"])

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
    row["rank_icir_ann_sqrt252"] = (
        row["rank_icir_daily"]
        * math.sqrt(252.0)
        if np.isfinite(row["rank_icir_daily"])
        else np.nan
    )

    for j in range(1, N_QUANTILES + 1):
        qmean = safe_float(
            pd.to_numeric(
                x[f"q{j}_ret_5d"],
                errors="coerce",
            ).mean()
        )

        row[f"mean_q{j}_ret_5d"] = qmean
        row[f"mean_q{j}_ret_5d_bps"] = (
            qmean * 10000.0
            if np.isfinite(qmean)
            else np.nan
        )

    spread_mean, spread_se, spread_t, spread_n = (
        newey_west_mean_stats(
            x["q5_q1_ret_5d"],
            lag=hac_lag,
        )
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

    q5ew_mean, q5ew_se, q5ew_t, q5ew_n = (
        newey_west_mean_stats(
            x["q5_minus_universe_ret_5d"],
            lag=hac_lag,
        )
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

    return row


def build_summary(
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

    for model in sorted(daily["model"].unique()):
        g = daily.loc[
            daily["model"] == model
        ]

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

    return pd.DataFrame(rows)


def build_yearly(
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

    for model in sorted(x["model"].unique()):
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
            row["year"] = int(year)
            rows.append(row)

    return pd.DataFrame(rows)


# ======================================================================================
# Validation tuning
# ======================================================================================

def fit_predict_candidate_year(
    *,
    candidate: Dict[str, Any],
    year: int,
    files: Sequence[Path],
    cache: CacheArrays,
    cfg: Config,
    logger: logging.Logger,
) -> Tuple[
    pd.DataFrame,
    xgb.Booster,
    int,
    str,
    str,
]:
    pred = load_prediction_year(
        files,
        year,
    )

    cutoff = str(
        pred["trade_date"].min()
    )

    n_train = training_prefix_end(
        cache,
        cutoff,
    )

    last_exit = str(
        int(cache.exit_date[n_train - 1])
    )

    logger.info(
        "XGB FIT | candidate=%s | year=%d | cutoff=%s | train_n=%d | last_exit=%s",
        candidate["candidate_id"],
        year,
        cutoff,
        n_train,
        last_exit,
    )

    booster = fit_xgb(
        cache,
        n_train=n_train,
        candidate=candidate,
        cfg=cfg,
    )

    X_pred = pred[
        FEATURE_COLUMNS
    ].to_numpy(dtype=np.float32)

    score = predict_xgb(
        booster,
        X_pred,
        cfg=cfg,
    )

    if not np.isfinite(score).all():
        raise RuntimeError(
            f"{candidate['candidate_id']} {year}: nonfinite XGBoost scores."
        )

    pred["score_xgb"] = score

    return (
        pred,
        booster,
        n_train,
        cutoff,
        last_exit,
    )


def tune_xgb(
    *,
    files: Sequence[Path],
    cache: CacheArrays,
    cfg: Config,
    logger: logging.Logger,
) -> pd.DataFrame:
    rows = []

    for candidate in XGB_CANDIDATES:
        daily_rows: List[Dict[str, Any]] = []

        for year in (2017, 2018):
            pred, booster, n_train, cutoff, last_exit = (
                fit_predict_candidate_year(
                    candidate=candidate,
                    year=year,
                    files=files,
                    cache=cache,
                    cfg=cfg,
                    logger=logger,
                )
            )

            daily_rows.extend(
                evaluate_scores_by_day(
                    pred,
                    score_col="score_xgb",
                    model_name=candidate["candidate_id"],
                    min_daily_n=cfg.min_daily_n,
                )
            )

            del pred, booster
            gc.collect()

        daily = pd.DataFrame(daily_rows)

        mean_ic, ic_se, ic_t, ic_n = (
            newey_west_mean_stats(
                daily["ic"],
                lag=cfg.hac_lag,
            )
        )

        mean_ric, ric_se, ric_t, ric_n = (
            newey_west_mean_stats(
                daily["rank_ic"],
                lag=cfg.hac_lag,
            )
        )

        spread_mean, spread_se, spread_t, spread_n = (
            newey_west_mean_stats(
                daily["q5_q1_ret_5d"],
                lag=cfg.hac_lag,
            )
        )

        rows.append({
            **candidate,
            "learning_rate": FIXED_XGB_PARAMS["eta"],
            "num_boost_round": NUM_BOOST_ROUND,
            "subsample": FIXED_XGB_PARAMS["subsample"],
            "colsample_bytree": FIXED_XGB_PARAMS["colsample_bytree"],
            "max_bin": FIXED_XGB_PARAMS["max_bin"],
            "validation_n_dates": int(
                daily["trade_date"].nunique()
            ),
            "validation_mean_ic": mean_ic,
            "validation_ic_hac_se": ic_se,
            "validation_ic_hac_t": ic_t,
            "validation_mean_rank_ic": mean_ric,
            "validation_rank_ic_hac_se": ric_se,
            "validation_rank_ic_hac_t": ric_t,
            "validation_mean_q5_q1_ret_5d": spread_mean,
            "validation_mean_q5_q1_ret_5d_bps": (
                spread_mean * 10000.0
                if np.isfinite(spread_mean)
                else np.nan
            ),
            "validation_q5_q1_hac_se": spread_se,
            "validation_q5_q1_hac_t": spread_t,
        })

    tuning = pd.DataFrame(rows)

    tuning = tuning.sort_values(
        [
            "validation_mean_rank_ic",
            "validation_mean_ic",
            "max_depth",
            "min_child_weight",
            "reg_lambda",
            "candidate_id",
        ],
        ascending=[
            False,
            False,
            True,
            False,
            False,
            True,
        ],
        kind="stable",
    ).reset_index(drop=True)

    tuning["selected"] = False

    if not tuning.empty:
        tuning.loc[0, "selected"] = True

    return tuning


# ======================================================================================
# Feature importance
# ======================================================================================

def extract_importance(
    booster: xgb.Booster,
    *,
    refit_year: int,
    candidate_id: str,
    n_train: int,
) -> List[Dict[str, Any]]:
    gain_raw = booster.get_score(
        importance_type="gain"
    )
    weight_raw = booster.get_score(
        importance_type="weight"
    )
    cover_raw = booster.get_score(
        importance_type="cover"
    )

    # With explicit DMatrix feature_names, keys should be actual feature names.
    total_gain = sum(
        float(gain_raw.get(f, 0.0))
        for f in FEATURE_NAMES
    )

    rows = []

    for f in FEATURE_NAMES:
        gain = float(
            gain_raw.get(f, 0.0)
        )
        weight = float(
            weight_raw.get(f, 0.0)
        )
        cover = float(
            cover_raw.get(f, 0.0)
        )

        rows.append({
            "refit_year": int(refit_year),
            "candidate_id": candidate_id,
            "train_n": int(n_train),
            "factor": f,
            "gain": gain,
            "gain_share": (
                gain / total_gain
                if total_gain > 0
                else np.nan
            ),
            "split_count": weight,
            "mean_cover": cover,
        })

    return rows


# ======================================================================================
# Final annual walk-forward
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
        / f"xgb_predictions_{year:04d}{month:02d}.parquet"
    )


def run_final_walkforward(
    *,
    files: Sequence[Path],
    cache: CacheArrays,
    selected_candidate: Dict[str, Any],
    ridge_lambda: float,
    cfg: Config,
    logger: logging.Logger,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    daily_rows: List[Dict[str, Any]] = []
    importance_rows: List[Dict[str, Any]] = []
    coverage_rows: List[Dict[str, Any]] = []

    for year in range(2017, 2026):
        pred = load_prediction_year(
            files,
            year,
        )

        cutoff = str(
            pred["trade_date"].min()
        )

        n_train = training_prefix_end(
            cache,
            cutoff,
        )

        last_exit = str(
            int(cache.exit_date[n_train - 1])
        )

        logger.info(
            "FINAL YEAR %d | cutoff=%s | train_n=%d | last_exit=%s | "
            "XGB=%s | Ridge lambda=%.10g",
            year,
            cutoff,
            n_train,
            last_exit,
            selected_candidate["candidate_id"],
            ridge_lambda,
        )

        # XGBoost annual fit.
        booster = fit_xgb(
            cache,
            n_train=n_train,
            candidate=selected_candidate,
            cfg=cfg,
        )

        X_pred = pred[
            FEATURE_COLUMNS
        ].to_numpy(dtype=np.float32)

        xgb_score = predict_xgb(
            booster,
            X_pred,
            cfg=cfg,
        )

        # Same-frequency annual Ridge.
        beta_ridge, ridge_cond = fit_ridge_from_cache(
            cache,
            n_train=n_train,
            lam=ridge_lambda,
        )

        ridge_score = (
            X_pred.astype(np.float64)
            @ beta_ridge
        )

        if not (
            np.isfinite(xgb_score).all()
            and np.isfinite(ridge_score).all()
        ):
            raise RuntimeError(
                f"{year}: nonfinite final model scores."
            )

        pred["score_xgb_annual"] = xgb_score
        pred["score_ridge_annual"] = ridge_score

        daily_rows.extend(
            evaluate_scores_by_day(
                pred,
                score_col="score_xgb_annual",
                model_name="XGBOOST_ANNUAL",
                min_daily_n=cfg.min_daily_n,
            )
        )

        daily_rows.extend(
            evaluate_scores_by_day(
                pred,
                score_col="score_ridge_annual",
                model_name="RIDGE_ANNUAL",
                min_daily_n=cfg.min_daily_n,
            )
        )

        importance_rows.extend(
            extract_importance(
                booster,
                refit_year=year,
                candidate_id=selected_candidate[
                    "candidate_id"
                ],
                n_train=n_train,
            )
        )

        # Save one monthly prediction file at a time.
        pred["prediction_month"] = (
            pred["trade_date"].str[:6]
        )

        for ym, gm in pred.groupby(
            "prediction_month",
            sort=True,
        ):
            y = int(ym[:4])
            m = int(ym[4:6])

            save_cols = [
                "ts_code",
                "trade_date",
                TARGET,
                ENTRY_DATE_COL,
                EXIT_DATE_COL,
                "target_entry_tradable",
                "target_exit_tradable",
                "target_fully_tradable_5d",
            ]

            out = gm[
                save_cols
            ].copy()

            out["score_xgb_annual"] = gm[
                "score_xgb_annual"
            ].to_numpy(dtype=float)

            out["score_ridge_annual"] = gm[
                "score_ridge_annual"
            ].to_numpy(dtype=float)

            out["xgb_candidate_id"] = (
                selected_candidate["candidate_id"]
            )
            out["ridge_lambda"] = float(
                ridge_lambda
            )
            out["refit_year"] = int(year)
            out["refit_cutoff_date"] = cutoff
            out["last_included_label_exit_date"] = (
                last_exit
            )
            out["stage8_spec_version"] = (
                STAGE8_SPEC_VERSION
            )

            out_path = prediction_output_path(
                cfg.prediction_root,
                y,
                m,
            )

            atomic_write_parquet(
                out,
                out_path,
            )

            coverage_rows.append({
                "prediction_month": f"{y:04d}-{m:02d}",
                "refit_year": int(year),
                "refit_cutoff_date": cutoff,
                "train_n": int(n_train),
                "last_included_label_exit_date": last_exit,
                "prediction_rows": int(len(out)),
                "prediction_dates": int(
                    out["trade_date"].nunique()
                ),
                "xgb_missing_scores": int(
                    pd.to_numeric(
                        out["score_xgb_annual"],
                        errors="coerce",
                    ).isna().sum()
                ),
                "ridge_annual_missing_scores": int(
                    pd.to_numeric(
                        out["score_ridge_annual"],
                        errors="coerce",
                    ).isna().sum()
                ),
                "ridge_annual_condition_number": float(
                    ridge_cond
                ),
                "output_path": str(out_path),
            })

        del (
            pred,
            booster,
            X_pred,
            xgb_score,
            beta_ridge,
            ridge_score,
        )
        gc.collect()

    daily = pd.DataFrame(daily_rows)
    importance = pd.DataFrame(importance_rows)
    coverage = pd.DataFrame(coverage_rows)

    return daily, importance, coverage


# ======================================================================================
# Import Stage-7 monthly Ridge benchmark
# ======================================================================================

def load_stage7_monthly_ridge_daily(
    cfg: Config,
) -> Optional[pd.DataFrame]:
    path = (
        cfg.stage7_report_root
        / "linear_model_daily_metrics.csv"
    )

    if not path.exists():
        return None

    try:
        df = pd.read_csv(
            path,
            dtype={
                "trade_date": str,
                "model": str,
            },
        )
    except Exception:
        return None

    if (
        "model" not in df.columns
        or "trade_date" not in df.columns
    ):
        return None

    ridge = df.loc[
        df["model"] == "RIDGE"
    ].copy()

    if ridge.empty:
        return None

    ridge["trade_date"] = normalize_date_series(
        ridge["trade_date"]
    )
    ridge["model"] = "RIDGE_STAGE7_MONTHLY"

    required = [
        "trade_date",
        "model",
        "n",
        "ic",
        "rank_ic",
        "universe_ew_ret_5d",
        "q1_ret_5d",
        "q2_ret_5d",
        "q3_ret_5d",
        "q4_ret_5d",
        "q5_ret_5d",
        "q5_q1_ret_5d",
        "q5_minus_universe_ret_5d",
    ]

    missing = [
        c for c in required
        if c not in ridge.columns
    ]

    if missing:
        return None

    return ridge[required].copy()


# ======================================================================================
# Comparison
# ======================================================================================

def build_incremental_comparison(
    summary: pd.DataFrame,
) -> pd.DataFrame:
    oos = summary.loc[
        summary["period"] == "OOS_2019_2025"
    ].copy()

    if oos.empty:
        return pd.DataFrame()

    xgb_row = oos.loc[
        oos["model"] == "XGBOOST_ANNUAL"
    ]

    if len(xgb_row) != 1:
        return pd.DataFrame()

    x = xgb_row.iloc[0]

    rows = []

    for benchmark in [
        "RIDGE_ANNUAL",
        "RIDGE_STAGE7_MONTHLY",
    ]:
        b = oos.loc[
            oos["model"] == benchmark
        ]

        if len(b) != 1:
            continue

        b = b.iloc[0]

        rows.append({
            "model": "XGBOOST_ANNUAL",
            "benchmark": benchmark,
            "oos_xgb_mean_ic": x["mean_ic"],
            "oos_benchmark_mean_ic": b["mean_ic"],
            "delta_mean_ic": (
                x["mean_ic"]
                - b["mean_ic"]
            ),
            "oos_xgb_mean_rank_ic": x["mean_rank_ic"],
            "oos_benchmark_mean_rank_ic": b["mean_rank_ic"],
            "delta_mean_rank_ic": (
                x["mean_rank_ic"]
                - b["mean_rank_ic"]
            ),
            "relative_rank_ic_improvement": (
                (
                    x["mean_rank_ic"]
                    / b["mean_rank_ic"]
                    - 1.0
                )
                if (
                    np.isfinite(b["mean_rank_ic"])
                    and b["mean_rank_ic"] != 0
                )
                else np.nan
            ),
            "oos_xgb_q5_q1_bps": x[
                "mean_q5_q1_ret_5d_bps"
            ],
            "oos_benchmark_q5_q1_bps": b[
                "mean_q5_q1_ret_5d_bps"
            ],
            "delta_q5_q1_bps": (
                x[
                    "mean_q5_q1_ret_5d_bps"
                ]
                - b[
                    "mean_q5_q1_ret_5d_bps"
                ]
            ),
            "oos_xgb_q5_minus_universe_bps": x[
                "mean_q5_minus_universe_ret_5d_bps"
            ],
            "oos_benchmark_q5_minus_universe_bps": b[
                "mean_q5_minus_universe_ret_5d_bps"
            ],
            "delta_q5_minus_universe_bps": (
                x[
                    "mean_q5_minus_universe_ret_5d_bps"
                ]
                - b[
                    "mean_q5_minus_universe_ret_5d_bps"
                ]
            ),
        })

    return pd.DataFrame(rows)


# ======================================================================================
# Final QA
# ======================================================================================

def final_qa(
    *,
    tuning: pd.DataFrame,
    daily: pd.DataFrame,
    summary: pd.DataFrame,
    coverage: pd.DataFrame,
    importance: pd.DataFrame,
) -> List[str]:
    issues: List[str] = []

    selected = tuning.loc[
        tuning["selected"] == True  # noqa: E712
    ]

    if len(selected) != 1:
        issues.append(
            f"Expected exactly one selected XGB candidate, got {len(selected)}."
        )

    if daily.empty:
        issues.append(
            "Stage-8 daily metrics are empty."
        )
        return issues

    if daily.duplicated(
        ["trade_date", "model"],
        keep=False,
    ).any():
        issues.append(
            "Duplicate (trade_date, model) rows."
        )

    required_models = {
        "XGBOOST_ANNUAL",
        "RIDGE_ANNUAL",
    }

    missing_models = (
        required_models
        - set(daily["model"].unique())
    )

    if missing_models:
        issues.append(
            f"Missing required models: {sorted(missing_models)}"
        )

    for model in required_models:
        g = daily.loc[
            daily["model"] == model
        ]

        oos = g.loc[
            (g["trade_date"] >= OOS_START)
            & (g["trade_date"] <= OOS_END)
        ]

        years = sorted(
            oos["trade_date"]
            .str[:4]
            .unique()
            .tolist()
        )

        if years != [
            "2019",
            "2020",
            "2021",
            "2022",
            "2023",
            "2024",
            "2025",
        ]:
            issues.append(
                f"{model}: incomplete OOS year coverage {years}"
            )

        score_std = pd.to_numeric(
            g["score_std"],
            errors="coerce",
        )

        if (
            score_std.notna().sum() == 0
            or (score_std <= 0).any()
        ):
            issues.append(
                f"{model}: nonpositive or missing daily score dispersion."
            )

    if coverage.empty:
        issues.append(
            "Prediction coverage is empty."
        )
    else:
        if coverage["xgb_missing_scores"].sum() != 0:
            issues.append(
                "XGBoost predictions contain missing scores."
            )

        if (
            coverage["ridge_annual_missing_scores"].sum()
            != 0
        ):
            issues.append(
                "Annual Ridge predictions contain missing scores."
            )

        for row in coverage.itertuples(index=False):
            try:
                if int(row.last_included_label_exit_date) >= int(
                    row.refit_cutoff_date
                ):
                    issues.append(
                        f"{row.prediction_month}: label purge violation."
                    )
            except Exception:
                issues.append(
                    f"{row.prediction_month}: invalid purge dates."
                )

    if importance.empty:
        issues.append(
            "XGBoost feature importance is empty."
        )
    else:
        oos_years = set(
            importance.loc[
                importance["refit_year"].between(
                    2019,
                    2025,
                ),
                "refit_year",
            ].astype(int)
        )

        if oos_years != set(
            range(2019, 2026)
        ):
            issues.append(
                f"Incomplete OOS feature-importance years: {sorted(oos_years)}"
            )

    oos_summary = summary.loc[
        summary["period"] == "OOS_2019_2025"
    ]

    if (
        oos_summary["model"]
        .eq("XGBOOST_ANNUAL")
        .sum()
        != 1
    ):
        issues.append(
            "Expected one XGBOOST_ANNUAL OOS summary row."
        )

    return issues


# ======================================================================================
# CLI
# ======================================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Stage 8 XGBoost nonlinear expanding-window alpha model."
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
        "--nthread",
        type=int,
        default=-1,
        help=(
            "XGBoost CPU threads. -1 uses XGBoost default/all available threads."
        ),
    )
    p.add_argument(
        "--device",
        default="cpu",
        choices=[
            "cpu",
            "cuda",
        ],
        help=(
            "Default cpu. Use cuda only with a modern GPU-enabled XGBoost install."
        ),
    )
    p.add_argument(
        "--rebuild-cache",
        action="store_true",
    )

    return p.parse_args()


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

    cfg = Config(
        data_root=Path(args.data_root),
        hac_lag=int(args.hac_lag),
        min_daily_n=int(args.min_daily_n),
        nthread=int(args.nthread),
        device=str(args.device),
        rebuild_cache=bool(args.rebuild_cache),
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

    logger.info("=" * 112)
    logger.info("STAGE 8 | XGBOOST NONLINEAR MODEL")
    logger.info(
        "Stage-5 frozen version: %s",
        REQUIRED_STAGE5_VERSION,
    )
    logger.info(
        "Features (%d): %s",
        P,
        ", ".join(FEATURE_NAMES),
    )
    logger.info(
        "Validation=%s..%s | OOS=%s..%s",
        VALIDATION_START,
        VALIDATION_END,
        OOS_START,
        OOS_END,
    )
    logger.info(
        "Annual expanding refits | label purge requires exit_date < annual cutoff."
    )
    logger.info(
        "XGBoost version=%s | tree_method=hist | rounds=%d | device=%s",
        xgb.__version__,
        NUM_BOOST_ROUND,
        cfg.device,
    )
    logger.info(
        "Candidate set: %s",
        ", ".join(
            x["candidate_id"]
            for x in XGB_CANDIDATES
        ),
    )
    logger.info("=" * 112)

    try:
        files = discover_signal_files(
            cfg.signal_root
        )

        logger.info(
            "Discovered %d Stage-5 monthly files.",
            len(files),
        )

        # ------------------------------------------------------------------
        # Build/reuse compact training cache.
        # ------------------------------------------------------------------
        cache_meta = build_training_cache(
            files,
            cfg=cfg,
            logger=logger,
        )

        cache = open_training_cache(
            cfg
        )

        # ------------------------------------------------------------------
        # Validation-only candidate selection.
        # ------------------------------------------------------------------
        tuning = tune_xgb(
            files=files,
            cache=cache,
            cfg=cfg,
            logger=logger,
        )

        selected_rows = tuning.loc[
            tuning["selected"] == True  # noqa: E712
        ]

        if len(selected_rows) != 1:
            raise RuntimeError(
                "XGBoost validation tuning did not select exactly one candidate."
            )

        selected_id = str(
            selected_rows.iloc[0][
                "candidate_id"
            ]
        )

        selected_candidate = next(
            x for x in XGB_CANDIDATES
            if x["candidate_id"] == selected_id
        )

        logger.info(
            "SELECTED XGB = %s | depth=%d | min_child_weight=%.1f | "
            "reg_lambda=%.1f | validation RankIC=%.6f | IC=%.6f | "
            "Q5-Q1=%.2f bps",
            selected_id,
            selected_candidate["max_depth"],
            selected_candidate["min_child_weight"],
            selected_candidate["reg_lambda"],
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
            / "xgb_validation_tuning.csv"
        )

        tuning.to_csv(
            tuning_path,
            index=False,
            encoding="utf-8-sig",
        )

        # ------------------------------------------------------------------
        # Stage-7 Ridge penalty.
        # ------------------------------------------------------------------
        ridge_lambda = read_stage7_selected_lambda(
            cfg
        )

        logger.info(
            "Stage-7 selected Ridge lambda used for same-frequency benchmark: %.10g",
            ridge_lambda,
        )

        # ------------------------------------------------------------------
        # Final Validation + OOS annual walk-forward.
        # ------------------------------------------------------------------
        daily, importance, coverage = (
            run_final_walkforward(
                files=files,
                cache=cache,
                selected_candidate=selected_candidate,
                ridge_lambda=ridge_lambda,
                cfg=cfg,
                logger=logger,
            )
        )

        # Add stronger monthly Stage-7 Ridge benchmark if available.
        stage7_monthly = (
            load_stage7_monthly_ridge_daily(
                cfg
            )
        )

        if stage7_monthly is not None:
            logger.info(
                "Imported Stage-7 monthly Ridge daily metrics for stronger benchmark."
            )

            # Align union of columns safely.
            daily = pd.concat(
                [
                    daily,
                    stage7_monthly,
                ],
                ignore_index=True,
                sort=False,
            )
        else:
            logger.warning(
                "Stage-7 monthly Ridge daily metrics not found/compatible; "
                "comparison will use RIDGE_ANNUAL only."
            )

        daily["trade_date"] = normalize_date_series(
            daily["trade_date"]
        )

        daily = daily.sort_values(
            ["trade_date", "model"]
        ).reset_index(drop=True)

        importance = importance.sort_values(
            ["refit_year", "factor"]
        ).reset_index(drop=True)

        coverage = coverage.sort_values(
            "prediction_month"
        ).reset_index(drop=True)

        summary = build_summary(
            daily,
            hac_lag=cfg.hac_lag,
        )

        yearly = build_yearly(
            daily,
            hac_lag=cfg.hac_lag,
        )

        comparison = build_incremental_comparison(
            summary
        )

        # ------------------------------------------------------------------
        # QA
        # ------------------------------------------------------------------
        issues = final_qa(
            tuning=tuning,
            daily=daily,
            summary=summary,
            coverage=coverage,
            importance=importance,
        )

        # ------------------------------------------------------------------
        # Save reports
        # ------------------------------------------------------------------
        daily_path = (
            cfg.report_root
            / "xgb_model_daily_metrics.csv"
        )
        summary_path = (
            cfg.report_root
            / "xgb_model_summary.csv"
        )
        yearly_path = (
            cfg.report_root
            / "xgb_model_yearly.csv"
        )
        importance_path = (
            cfg.report_root
            / "xgb_feature_importance.csv"
        )
        coverage_path = (
            cfg.report_root
            / "xgb_prediction_coverage.csv"
        )
        comparison_path = (
            cfg.report_root
            / "xgb_vs_linear_comparison.csv"
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
        importance.to_csv(
            importance_path,
            index=False,
            encoding="utf-8-sig",
        )
        coverage.to_csv(
            coverage_path,
            index=False,
            encoding="utf-8-sig",
        )
        comparison.to_csv(
            comparison_path,
            index=False,
            encoding="utf-8-sig",
        )

        # Mean importance across OOS refits for convenience.
        oos_imp = importance.loc[
            importance["refit_year"].between(
                2019,
                2025,
            )
        ]

        mean_oos_importance = (
            oos_imp.groupby(
                "factor",
                as_index=False,
            )
            .agg(
                mean_gain_share=(
                    "gain_share",
                    "mean",
                ),
                mean_split_count=(
                    "split_count",
                    "mean",
                ),
            )
            .sort_values(
                "mean_gain_share",
                ascending=False,
            )
            .to_dict(
                orient="records"
            )
        )

        metadata = {
            "project": (
                "China A-Share Cross-Sectional Alpha Research"
            ),
            "script": "08_xgboost_nonlinear.py",
            "stage8_spec_version": STAGE8_SPEC_VERSION,
            "generated_at_utc": datetime.now(
                timezone.utc
            ).isoformat(),
            "required_stage5_spec_version": REQUIRED_STAGE5_VERSION,
            "features": FEATURE_NAMES,
            "feature_columns": FEATURE_COLUMNS,
            "fit_target": (
                "target_xs = target_ret_5d - same-day model-ready cross-sectional mean"
            ),
            "evaluation_target": TARGET,
            "validation_period": [
                VALIDATION_START,
                VALIDATION_END,
            ],
            "oos_period": [
                OOS_START,
                OOS_END,
            ],
            "xgb_refit_frequency": "annual",
            "xgb_refit_years": list(
                range(2017, 2026)
            ),
            "label_purge_rule": (
                "training row allowed only when target_exit_trade_date "
                "< first model-ready trade date of refit year"
            ),
            "xgb_fixed_params": FIXED_XGB_PARAMS,
            "num_boost_round": NUM_BOOST_ROUND,
            "xgb_candidates": list(
                XGB_CANDIDATES
            ),
            "xgb_selection_metric": (
                "maximum Validation 2017-2018 mean daily RankIC; "
                "tie-break mean IC, shallower depth, larger min_child_weight, "
                "larger reg_lambda, candidate id"
            ),
            "selected_xgb_candidate": selected_candidate,
            "selected_stage7_ridge_lambda": ridge_lambda,
            "benchmarks": [
                "RIDGE_ANNUAL same annual refit cutoffs",
                "RIDGE_STAGE7_MONTHLY if Stage-7 daily metrics are available",
            ],
            "training_cache_meta": cache_meta,
            "oos_mean_feature_importance": mean_oos_importance,
            "hac_lag": cfg.hac_lag,
            "min_daily_n": cfg.min_daily_n,
            "random_seed": RANDOM_SEED,
            "qa_issue_count": len(issues),
            "qa_issues": issues,
            "research_notes": [
                "No OOS observation is used for XGBoost candidate selection.",
                "All 11 frozen predictors are retained; no Stage-6/7 result is used to delete or flip a feature.",
                "Annual XGBoost refitting is precommitted for computational tractability on the multi-million-row expanding sample.",
                "RIDGE_ANNUAL is included to separate nonlinear-model value from refit-frequency effects.",
                "Stage-7 monthly Ridge is included as a stronger linear benchmark whenever its daily report is available.",
                "Stage 8 is still a prediction study. Transaction costs, self-financing accounting, and next-open execution constraints belong to the executable backtest stage.",
            ],
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "pandas_version": package_version("pandas"),
            "numpy_version": package_version("numpy"),
            "pyarrow_version": package_version("pyarrow"),
            "xgboost_version": str(
                xgb.__version__
            ),
            "device": cfg.device,
            "nthread": cfg.nthread,
        }

        atomic_write_json(
            metadata,
            cfg.report_root
            / "xgb_metadata.json",
        )

        logger.info("=" * 112)
        logger.info(
            "STAGE 8 COMPLETE | selected=%s | daily rows=%d | "
            "importance rows=%d | prediction months=%d",
            selected_id,
            len(daily),
            len(importance),
            len(coverage),
        )
        logger.info(
            "Reports: %s | %s | %s | %s | %s | %s | %s",
            tuning_path,
            daily_path,
            summary_path,
            yearly_path,
            importance_path,
            coverage_path,
            comparison_path,
        )

        if issues:
            logger.error(
                "STAGE 8 QA FAIL | %d issue(s)",
                len(issues),
            )

            for issue in issues:
                logger.error(
                    "QA | %s",
                    issue,
                )

            logger.info("=" * 112)
            return 1

        logger.info(
            "STAGE 8 QA: PASS"
        )
        logger.info("=" * 112)

        return 0

    except KeyboardInterrupt:
        logger.warning(
            "Interrupted by user."
        )
        return 130

    except Exception:
        logger.exception(
            "Fatal error during Stage-8 XGBoost analysis."
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
