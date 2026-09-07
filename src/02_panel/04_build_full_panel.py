#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
04_build_full_panel.py

China A-Share Cross-Sectional Alpha Research
Stage 4: Build the full Shanghai + Shenzhen daily master panel, month by month.

INPUT
-----
Created by 03_download_full_history.py:

data/raw_full/
    daily/YYYY/MM/daily_YYYYMMDD.parquet
    adj_factor/YYYY/MM/adj_factor_YYYYMMDD.parquet
    daily_basic/YYYY/MM/daily_basic_YYYYMMDD.parquet
    stock_st/YYYY/MM/stock_st_YYYYMMDD.parquet
    suspend_d/YYYY/MM/suspend_d_YYYYMMDD.parquet
    stk_limit/YYYY/MM/stk_limit_YYYYMMDD.parquet
    stock_basic/stock_basic_SH_SZ_all.parquet
    trade_cal/trade_cal_20090601_20251231.parquet

DEFAULT PERIOD
--------------
Raw / warm-up:
    2009-06-01 to 2025-12-31
Formal research sample:
    2010-01-01 to 2025-12-31

OUTPUT
------
Monthly full panel:
data/processed/full_panel/YYYY/MM/full_panel_YYYYMM.parquet

Derived security history:
data/interim_full/security_history_20090601_20251231.parquet

Diagnostics:
data/full_panel_reports/
    full_panel_merge_coverage.csv
    full_panel_monthly_quality.csv
    full_panel_missingness_summary.csv
    full_panel_security_type_diagnostics.csv
    full_panel_stock_basic_duplicates.csv
    full_panel_metadata.json
    04_build_full_panel.log

CORE DESIGN
-----------
1) `daily` is the row backbone.
2) The build is MONTH-BY-MONTH to avoid loading 10M+ rows at once.
3) Next trading day comes from the MARKET calendar, not per-stock shift(-1).
4) Cross-month next-open lookups use a one-market-day forward buffer.
5) Adjusted prices:
       adj_price = raw_price * adj_factor
   The absolute scale is arbitrary, while within-stock price ratios are valid.
6) Historical security existence is anchored to observed `daily` history.
   `stock_basic` is auxiliary metadata, NOT the sole historical-universe truth.
7) Trading-day seasoning:
       observed_trading_age = market_trade_index - first_observed_trade_index
       seasoned_120d = observed_trading_age >= 120
   For securities already trading on the warm-up boundary, the observed age is
   left-censored. By the 2010 formal-sample start they have had ~7 months of
   observed market history, so the 120-trading-day seasoning rule is usable.
8) Common-A-share classification is a transparent CODE-RULE FLAG, not a row
   deletion. Downstream research can inspect/modify it.
   Included prefixes:
       Shanghai: 600, 601, 603, 605, 688
       Shenzhen: 000, 001, 002, 003, 300, 301, 302
   This excludes obvious B-share prefixes such as 900.SH and 200/201.SZ.
9) Base eligibility flag at signal date:
       is_common_a_share
       & seasoned_120d
       & ~is_st
       & ~is_suspended
       & close >= 5 RMB
       & has_adj_factor
   Liquidity/ADV20 filters are deliberately deferred to Stage 5.
10) No signals, targets, IC, ML models, portfolio construction, or backtest here.

INSTALL
-------
pip install pandas numpy pyarrow

RUN
---
python 04_build_full_panel.py

Resume:
    Re-run the same command. Existing valid monthly panels are reused.

Force rebuild:
    python 04_build_full_panel.py --force

Optional custom root:
    python 04_build_full_panel.py --data-root data
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import platform
import sys
from collections import defaultdict
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

KEY = ["ts_code", "trade_date"]

COMMON_A_PREFIXES_SH = ("600", "601", "603", "605", "688")
COMMON_A_PREFIXES_SZ = ("000", "001", "002", "003", "300", "301", "302")

DAILY_NUMERIC = [
    "open", "high", "low", "close", "pre_close",
    "change", "pct_chg", "vol", "amount",
]

DAILY_BASIC_KEEP = [
    "ts_code",
    "trade_date",
    "turnover_rate",
    "turnover_rate_f",
    "volume_ratio",
    "total_share",
    "float_share",
    "free_share",
    "total_mv",
    "circ_mv",
]

DAILY_BASIC_NUMERIC = [
    "turnover_rate",
    "turnover_rate_f",
    "volume_ratio",
    "total_share",
    "float_share",
    "free_share",
    "total_mv",
    "circ_mv",
]

STOCK_BASIC_KEEP = [
    "ts_code",
    "symbol",
    "name",
    "area",
    "industry",
    "market",
    "exchange",
    "list_status",
    "list_date",
    "delist_date",
]


# ======================================================================================
# Configuration
# ======================================================================================

@dataclass
class Config:
    start: str
    end: str
    formal_sample_start: str
    data_root: Path
    force: bool

    @property
    def raw_root(self) -> Path:
        return self.data_root / "raw_full"

    @property
    def interim_root(self) -> Path:
        return self.data_root / "interim_full"

    @property
    def output_root(self) -> Path:
        return self.data_root / "processed" / "full_panel"

    @property
    def report_root(self) -> Path:
        return self.data_root / "full_panel_reports"

    @property
    def security_history_path(self) -> Path:
        return (
            self.interim_root
            / f"security_history_{self.start}_{self.end}.parquet"
        )


# ======================================================================================
# Generic helpers
# ======================================================================================

def setup_logging(report_root: Path) -> logging.Logger:
    report_root.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("build_full_panel")
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
        report_root / "04_build_full_panel.log",
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


def normalize_trade_date(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "trade_date" in out.columns:
        out["trade_date"] = normalize_date_series(out["trade_date"])
    return out


def numeric_coerce(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


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


def valid_parquet(
    path: Path,
    required_cols: Optional[Sequence[str]] = None,
) -> bool:
    if not path.exists() or path.stat().st_size <= 0:
        return False

    try:
        df = pd.read_parquet(path)
    except Exception:
        return False

    if required_cols:
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            return False

    return True


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


def package_version(name: str) -> str:
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:
        return "unknown"


# ======================================================================================
# Raw-data path / loading
# ======================================================================================

def raw_partition_path(
    raw_root: Path,
    dataset: str,
    trade_date: str,
) -> Path:
    yyyy = trade_date[:4]
    mm = trade_date[4:6]

    return (
        raw_root
        / dataset
        / yyyy
        / mm
        / f"{dataset}_{trade_date}.parquet"
    )


def load_one_date(
    raw_root: Path,
    dataset: str,
    trade_date: str,
    *,
    allow_missing: bool = False,
) -> pd.DataFrame:
    path = raw_partition_path(raw_root, dataset, trade_date)

    if not path.exists():
        if allow_missing:
            return pd.DataFrame()
        raise FileNotFoundError(path)

    try:
        df = pd.read_parquet(path)
    except Exception as exc:
        raise RuntimeError(f"Cannot read {path}") from exc

    return normalize_trade_date(df)


def load_month(
    raw_root: Path,
    dataset: str,
    year: int,
    month: int,
    *,
    expected_dates: Sequence[str],
    allow_empty: bool = True,
) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []

    for trade_date in expected_dates:
        path = raw_partition_path(raw_root, dataset, trade_date)

        if not path.exists():
            raise FileNotFoundError(
                f"Expected raw partition is missing: {path}"
            )

        try:
            frames.append(pd.read_parquet(path))
        except Exception as exc:
            raise RuntimeError(
                f"Cannot read raw partition: {path}"
            ) from exc

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True, sort=False)
    out = normalize_trade_date(out)

    if not allow_empty and out.empty:
        raise RuntimeError(
            f"{dataset} {year}-{month:02d} loaded zero rows."
        )

    return out


# ======================================================================================
# Trading calendar
# ======================================================================================

def load_trade_calendar(
    cfg: Config,
    logger: logging.Logger,
) -> Tuple[pd.DataFrame, List[str], Dict[str, int], Dict[str, Optional[str]]]:
    path = (
        cfg.raw_root
        / "trade_cal"
        / f"trade_cal_{cfg.start}_{cfg.end}.parquet"
    )

    if not path.exists():
        raise FileNotFoundError(path)

    cal = pd.read_parquet(path)

    if not {"cal_date", "is_open"}.issubset(cal.columns):
        raise ValueError("trade_cal missing cal_date/is_open.")

    cal["cal_date"] = normalize_date_series(cal["cal_date"])
    cal["is_open"] = pd.to_numeric(cal["is_open"], errors="coerce")

    # SSE and SZSE should share the ordinary A-share open calendar.
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
        raise RuntimeError("No open dates found.")

    trade_index = {
        d: i
        for i, d in enumerate(open_dates)
    }

    next_trade_date: Dict[str, Optional[str]] = {}
    for i, d in enumerate(open_dates):
        next_trade_date[d] = (
            open_dates[i + 1]
            if i + 1 < len(open_dates)
            else None
        )

    logger.info(
        "TRADE CALENDAR open_dates=%d | %s -> %s",
        len(open_dates),
        open_dates[0],
        open_dates[-1],
    )

    return cal, open_dates, trade_index, next_trade_date


def month_groups(open_dates: Sequence[str]) -> List[Tuple[int, int, List[str]]]:
    buckets: Dict[Tuple[int, int], List[str]] = defaultdict(list)

    for d in open_dates:
        buckets[(int(d[:4]), int(d[4:6]))].append(d)

    return [
        (year, month, sorted(dates))
        for (year, month), dates in sorted(buckets.items())
    ]


# ======================================================================================
# Security classification / stock_basic canonical metadata
# ======================================================================================

def classify_common_a_share(ts_code: pd.Series) -> pd.Series:
    s = ts_code.astype("string")
    code = s.str.split(".").str[0]
    suffix = s.str.split(".").str[-1]

    sh = (
        suffix.eq("SH")
        & code.str.startswith(COMMON_A_PREFIXES_SH)
    )

    sz = (
        suffix.eq("SZ")
        & code.str.startswith(COMMON_A_PREFIXES_SZ)
    )

    return (sh | sz).fillna(False)


def security_prefix(ts_code: pd.Series) -> pd.Series:
    s = ts_code.astype("string")
    return s.str.split(".").str[0].str[:3]


def prepare_stock_basic(
    cfg: Config,
    logger: logging.Logger,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    path = (
        cfg.raw_root
        / "stock_basic"
        / "stock_basic_SH_SZ_all.parquet"
    )

    if not path.exists():
        raise FileNotFoundError(path)

    sb = pd.read_parquet(path)
    keep = [c for c in STOCK_BASIC_KEEP if c in sb.columns]
    sb = sb[keep].drop_duplicates().reset_index(drop=True)

    if "ts_code" not in sb.columns:
        raise ValueError("stock_basic missing ts_code.")

    for c in ["list_date", "delist_date"]:
        if c in sb.columns:
            sb[c] = (
                sb[c]
                .astype("string")
                .str.replace(r"\.0$", "", regex=True)
                .replace({
                    "<NA>": pd.NA,
                    "nan": pd.NA,
                    "None": pd.NA,
                    "": pd.NA,
                })
            )

    dup_mask = sb.duplicated("ts_code", keep=False)
    duplicates = sb.loc[dup_mask].copy()

    # Build one deterministic auxiliary metadata row per ts_code.
    # IMPORTANT: this does NOT define the historical universe.
    status_priority = {"L": 3, "P": 2, "D": 1}

    tmp = sb.copy()
    tmp["_status_priority"] = (
        tmp.get("list_status", pd.Series(index=tmp.index, dtype="object"))
        .map(status_priority)
        .fillna(0)
    )

    metadata_cols = [
        c for c in [
            "name", "area", "industry", "market",
            "exchange", "list_date", "delist_date",
        ]
        if c in tmp.columns
    ]

    if metadata_cols:
        tmp["_metadata_nonnull"] = tmp[metadata_cols].notna().sum(axis=1)
    else:
        tmp["_metadata_nonnull"] = 0

    tmp = tmp.sort_values(
        ["ts_code", "_status_priority", "_metadata_nonnull"],
        ascending=[True, False, False],
        kind="stable",
    )

    canonical = (
        tmp.drop_duplicates("ts_code", keep="first")
        .drop(
            columns=["_status_priority", "_metadata_nonnull"],
            errors="ignore",
        )
        .reset_index(drop=True)
    )

    duplicates.to_csv(
        cfg.report_root / "full_panel_stock_basic_duplicates.csv",
        index=False,
        encoding="utf-8-sig",
    )

    logger.info(
        "STOCK BASIC rows=%d canonical_codes=%d duplicate_rows=%d",
        len(sb),
        canonical["ts_code"].nunique(),
        len(duplicates),
    )

    return canonical, duplicates


# ======================================================================================
# First-observed security history
# ======================================================================================

def build_or_load_security_history(
    cfg: Config,
    *,
    month_buckets: Sequence[Tuple[int, int, List[str]]],
    trade_index: Dict[str, int],
    logger: logging.Logger,
) -> pd.DataFrame:
    path = cfg.security_history_path

    if valid_parquet(
        path,
        [
            "ts_code",
            "first_observed_trade_date",
            "last_observed_trade_date",
            "first_observed_trade_index",
            "n_daily_rows",
            "is_common_a_share",
        ],
    ) and not cfg.force:
        logger.info("REUSE security history -> %s", path)
        return pd.read_parquet(path)

    logger.info("BUILD SECURITY HISTORY | scanning all daily months once")

    first_seen: Dict[str, str] = {}
    last_seen: Dict[str, str] = {}
    counts: Dict[str, int] = defaultdict(int)

    for i, (year, month, dates) in enumerate(month_buckets, 1):
        daily = load_month(
            cfg.raw_root,
            "daily",
            year,
            month,
            expected_dates=dates,
            allow_empty=False,
        )

        if not {"ts_code", "trade_date"}.issubset(daily.columns):
            raise ValueError(
                f"daily {year}-{month:02d} missing ts_code/trade_date"
            )

        daily["trade_date"] = normalize_date_series(daily["trade_date"])

        grouped = (
            daily.groupby("ts_code", sort=False)["trade_date"]
            .agg(["min", "max", "size"])
            .reset_index()
        )

        for row in grouped.itertuples(index=False):
            code = str(row.ts_code)
            dmin = str(row.min)
            dmax = str(row.max)
            n = int(row.size)

            if code not in first_seen or dmin < first_seen[code]:
                first_seen[code] = dmin
            if code not in last_seen or dmax > last_seen[code]:
                last_seen[code] = dmax

            counts[code] += n

        logger.info(
            "SECURITY HISTORY %d/%d | %04d-%02d rows=%d codes=%d",
            i,
            len(month_buckets),
            year,
            month,
            len(daily),
            daily["ts_code"].nunique(),
        )

        del daily, grouped
        gc.collect()

    rows = []

    for code in sorted(counts):
        first_date = first_seen[code]
        last_date = last_seen[code]

        rows.append(
            {
                "ts_code": code,
                "first_observed_trade_date": first_date,
                "last_observed_trade_date": last_date,
                "first_observed_trade_index": trade_index.get(
                    first_date,
                    np.nan,
                ),
                "last_observed_trade_index": trade_index.get(
                    last_date,
                    np.nan,
                ),
                "n_daily_rows": counts[code],
            }
        )

    hist = pd.DataFrame(rows)
    hist["is_common_a_share"] = classify_common_a_share(hist["ts_code"])
    hist["code_prefix3"] = security_prefix(hist["ts_code"])

    atomic_write_parquet(hist, path)

    logger.info(
        "SAVE SECURITY HISTORY rows=%d common_A=%d -> %s",
        len(hist),
        int(hist["is_common_a_share"].sum()),
        path,
    )

    return hist


# ======================================================================================
# Monthly preparation helpers
# ======================================================================================

def prepare_sparse_flag(
    df: pd.DataFrame,
    flag_name: str,
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=KEY + [flag_name])

    x = normalize_trade_date(df)

    if not set(KEY).issubset(x.columns):
        raise ValueError(
            f"{flag_name} source missing key columns."
        )

    assert_unique_key(x, KEY, name=flag_name)

    x = x[KEY].copy()
    x[flag_name] = True

    return x


def prepare_adj_factor(df: pd.DataFrame) -> pd.DataFrame:
    x = normalize_trade_date(df)

    if x.empty:
        return pd.DataFrame(
            columns=["ts_code", "trade_date", "adj_factor"]
        )

    assert_unique_key(x, KEY, name="adj_factor")

    cols = [c for c in ["ts_code", "trade_date", "adj_factor"] if c in x.columns]
    x = x[cols].copy()
    x = numeric_coerce(x, ["adj_factor"])

    return x


def prepare_daily_basic(df: pd.DataFrame) -> pd.DataFrame:
    x = normalize_trade_date(df)

    if x.empty:
        return pd.DataFrame(columns=DAILY_BASIC_KEEP)

    assert_unique_key(x, KEY, name="daily_basic")

    cols = [c for c in DAILY_BASIC_KEEP if c in x.columns]
    x = x[cols].copy()
    x = numeric_coerce(x, DAILY_BASIC_NUMERIC)

    return x


def prepare_limits(df: pd.DataFrame) -> pd.DataFrame:
    x = normalize_trade_date(df)

    if x.empty:
        return pd.DataFrame(
            columns=["ts_code", "trade_date", "up_limit", "down_limit"]
        )

    assert_unique_key(x, KEY, name="stk_limit")

    cols = [
        c for c in [
            "ts_code",
            "trade_date",
            "up_limit",
            "down_limit",
        ]
        if c in x.columns
    ]

    x = x[cols].copy()
    x = numeric_coerce(x, ["up_limit", "down_limit"])

    return x


def left_merge_one_to_one(
    panel: pd.DataFrame,
    right: pd.DataFrame,
    *,
    name: str,
    coverage_rows: List[Dict[str, Any]],
    match_col: Optional[str] = None,
) -> pd.DataFrame:
    before = len(panel)

    probe = right[KEY].copy()
    probe["_merge_probe"] = 1

    match = panel[KEY].merge(
        probe,
        on=KEY,
        how="left",
        validate="one_to_one",
    )["_merge_probe"].notna()

    out = panel.merge(
        right,
        on=KEY,
        how="left",
        validate="one_to_one",
    )

    if len(out) != before:
        raise RuntimeError(
            f"Row count changed after {name}: {before} -> {len(out)}"
        )

    coverage_rows.append(
        {
            "merge": name,
            "left_rows": before,
            "right_rows": len(right),
            "matched_rows": int(match.sum()),
            "unmatched_rows": int((~match).sum()),
            "match_rate": float(match.mean()) if before else np.nan,
        }
    )

    return out


# ======================================================================================
# Forward-buffer lookups
# ======================================================================================

def boundary_next_date_for_month(
    month_dates: Sequence[str],
    next_trade_date_map: Dict[str, Optional[str]],
) -> Optional[str]:
    if not month_dates:
        return None

    last_date = month_dates[-1]
    nxt = next_trade_date_map.get(last_date)

    if nxt is None:
        return None

    if nxt[:6] == last_date[:6]:
        return None

    return nxt


def build_boundary_next_lookup(
    cfg: Config,
    boundary_date: Optional[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns:
        boundary_price: ts_code, trade_date, open, adj_open
        boundary_limits: ts_code, trade_date, up_limit, down_limit
        boundary_suspend: ts_code, trade_date, is_suspended
    """
    if boundary_date is None:
        return (
            pd.DataFrame(
                columns=["ts_code", "trade_date", "open", "adj_open"]
            ),
            pd.DataFrame(
                columns=["ts_code", "trade_date", "up_limit", "down_limit"]
            ),
            pd.DataFrame(
                columns=["ts_code", "trade_date", "is_suspended"]
            ),
        )

    d = load_one_date(
        cfg.raw_root,
        "daily",
        boundary_date,
        allow_missing=False,
    )

    af = prepare_adj_factor(
        load_one_date(
            cfg.raw_root,
            "adj_factor",
            boundary_date,
            allow_missing=False,
        )
    )

    lim = prepare_limits(
        load_one_date(
            cfg.raw_root,
            "stk_limit",
            boundary_date,
            allow_missing=False,
        )
    )

    sus_raw = load_one_date(
        cfg.raw_root,
        "suspend_d",
        boundary_date,
        allow_missing=False,
    )
    sus = prepare_sparse_flag(
        sus_raw,
        "is_suspended",
    )

    d = numeric_coerce(d, ["open"])
    assert_unique_key(d, KEY, name=f"daily boundary {boundary_date}")

    price = d[["ts_code", "trade_date", "open"]].merge(
        af,
        on=KEY,
        how="left",
        validate="one_to_one",
    )
    price["adj_open"] = price["open"] * price["adj_factor"]

    return (
        price[["ts_code", "trade_date", "open", "adj_open"]],
        lim,
        sus,
    )


# ======================================================================================
# Monthly panel build
# ======================================================================================

def build_one_month(
    cfg: Config,
    *,
    year: int,
    month: int,
    month_dates: Sequence[str],
    trade_index: Dict[str, int],
    next_trade_date_map: Dict[str, Optional[str]],
    security_history: pd.DataFrame,
    stock_basic: pd.DataFrame,
    logger: logging.Logger,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    coverage_rows: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Load current month raw datasets
    # ------------------------------------------------------------------
    daily = load_month(
        cfg.raw_root,
        "daily",
        year,
        month,
        expected_dates=month_dates,
        allow_empty=False,
    )
    daily = normalize_trade_date(daily)
    daily = numeric_coerce(daily, DAILY_NUMERIC)
    assert_unique_key(daily, KEY, name=f"daily {year}-{month:02d}")

    adj = prepare_adj_factor(
        load_month(
            cfg.raw_root,
            "adj_factor",
            year,
            month,
            expected_dates=month_dates,
        )
    )

    basic = prepare_daily_basic(
        load_month(
            cfg.raw_root,
            "daily_basic",
            year,
            month,
            expected_dates=month_dates,
        )
    )

    st = prepare_sparse_flag(
        load_month(
            cfg.raw_root,
            "stock_st",
            year,
            month,
            expected_dates=month_dates,
        ),
        "is_st",
    )

    sus = prepare_sparse_flag(
        load_month(
            cfg.raw_root,
            "suspend_d",
            year,
            month,
            expected_dates=month_dates,
        ),
        "is_suspended",
    )

    limits = prepare_limits(
        load_month(
            cfg.raw_root,
            "stk_limit",
            year,
            month,
            expected_dates=month_dates,
        )
    )

    # ------------------------------------------------------------------
    # Daily backbone
    # ------------------------------------------------------------------
    panel = daily.sort_values(KEY).reset_index(drop=True)

    panel["market_trade_index"] = (
        panel["trade_date"]
        .map(trade_index)
        .astype("Int64")
    )

    if panel["market_trade_index"].isna().any():
        raise RuntimeError(
            f"{year}-{month:02d}: daily rows not found in market calendar."
        )

    panel["next_trade_date"] = (
        panel["trade_date"]
        .map(next_trade_date_map)
        .astype("string")
    )

    # ------------------------------------------------------------------
    # Adjustment factor + adjusted OHLC
    # ------------------------------------------------------------------
    panel = left_merge_one_to_one(
        panel,
        adj,
        name="adj_factor",
        coverage_rows=coverage_rows,
    )
    panel["has_adj_factor"] = panel["adj_factor"].notna()

    for c in ["open", "high", "low", "close", "pre_close"]:
        if c in panel.columns:
            panel[f"adj_{c}"] = panel[c] * panel["adj_factor"]

    # Useful normalized units for later liquidity work
    if "vol" in panel.columns:
        # Tushare daily vol is in hands; 1 hand = 100 shares.
        panel["volume_shares"] = panel["vol"] * 100.0

    if "amount" in panel.columns:
        # Tushare daily amount is in thousand RMB.
        panel["amount_rmb"] = panel["amount"] * 1000.0

    # ------------------------------------------------------------------
    # daily_basic
    # ------------------------------------------------------------------
    panel = left_merge_one_to_one(
        panel,
        basic,
        name="daily_basic",
        coverage_rows=coverage_rows,
    )

    if "turnover_rate" in panel.columns:
        panel["has_daily_basic"] = panel["turnover_rate"].notna()
    else:
        panel["has_daily_basic"] = False

    # ------------------------------------------------------------------
    # Sparse current status flags
    # ------------------------------------------------------------------
    before = len(panel)
    panel = panel.merge(
        st,
        on=KEY,
        how="left",
        validate="one_to_one",
    )
    if len(panel) != before:
        raise RuntimeError("Row count changed after stock_st merge.")
    panel["is_st"] = panel["is_st"].fillna(False).astype(bool)

    before = len(panel)
    panel = panel.merge(
        sus,
        on=KEY,
        how="left",
        validate="one_to_one",
    )
    if len(panel) != before:
        raise RuntimeError("Row count changed after suspension merge.")
    panel["is_suspended"] = (
        panel["is_suspended"]
        .fillna(False)
        .astype(bool)
    )

    # ------------------------------------------------------------------
    # Current price limits
    # ------------------------------------------------------------------
    panel = left_merge_one_to_one(
        panel,
        limits,
        name="stk_limit",
        coverage_rows=coverage_rows,
    )
    panel["has_price_limit"] = (
        panel["up_limit"].notna()
        if "up_limit" in panel.columns
        else False
    )

    # ------------------------------------------------------------------
    # Derived security history: historical existence / trading age
    # ------------------------------------------------------------------
    hist_keep = [
        "ts_code",
        "first_observed_trade_date",
        "last_observed_trade_date",
        "first_observed_trade_index",
        "last_observed_trade_index",
        "n_daily_rows",
        "is_common_a_share",
        "code_prefix3",
    ]

    panel = panel.merge(
        security_history[hist_keep],
        on="ts_code",
        how="left",
        validate="many_to_one",
    )

    if panel["first_observed_trade_index"].isna().any():
        raise RuntimeError(
            f"{year}-{month:02d}: some daily codes are absent from security_history."
        )

    panel["observed_trading_age"] = (
        panel["market_trade_index"].astype("Int64")
        - panel["first_observed_trade_index"].astype("Int64")
    )

    panel["seasoned_120d"] = (
        panel["observed_trading_age"] >= 120
    ).fillna(False)

    panel["first_observed_at_warmup_boundary"] = (
        panel["first_observed_trade_date"] == cfg.start
    )

    # ------------------------------------------------------------------
    # Auxiliary stock_basic metadata
    # ------------------------------------------------------------------
    sb = stock_basic.copy()

    rename_map = {}
    for c in sb.columns:
        if c == "ts_code":
            continue
        rename_map[c] = f"meta_{c}"

    sb = sb.rename(columns=rename_map)

    before = len(panel)
    panel = panel.merge(
        sb,
        on="ts_code",
        how="left",
        validate="many_to_one",
    )
    if len(panel) != before:
        raise RuntimeError("Row count changed after stock_basic metadata merge.")

    panel["has_stock_basic_metadata"] = (
        panel["meta_list_date"].notna()
        if "meta_list_date" in panel.columns
        else False
    )

    # Metadata conflict is a diagnostic only, NOT a delete rule.
    if "meta_list_date" in panel.columns:
        trade_dt = pd.to_datetime(
            panel["trade_date"],
            format="%Y%m%d",
            errors="coerce",
        )
        list_dt = pd.to_datetime(
            panel["meta_list_date"],
            format="%Y%m%d",
            errors="coerce",
        )

        panel["metadata_trade_before_list_date"] = (
            list_dt.notna()
            & trade_dt.notna()
            & (trade_dt < list_dt)
        )

    else:
        panel["metadata_trade_before_list_date"] = False

    # ------------------------------------------------------------------
    # Signal-date base flags
    # ------------------------------------------------------------------
    panel["price_ge_5"] = (
        panel["close"].notna()
        & (panel["close"] >= 5.0)
    )

    panel["base_eligible_signal_day"] = (
        panel["is_common_a_share"].fillna(False)
        & panel["seasoned_120d"].fillna(False)
        & (~panel["is_st"])
        & (~panel["is_suspended"])
        & panel["price_ge_5"]
        & panel["has_adj_factor"]
    )

    # ------------------------------------------------------------------
    # Cross-month next-open lookup
    # ------------------------------------------------------------------
    boundary_date = boundary_next_date_for_month(
        month_dates,
        next_trade_date_map,
    )

    (
        boundary_price,
        boundary_limits,
        boundary_suspend,
    ) = build_boundary_next_lookup(
        cfg,
        boundary_date,
    )

    current_price = panel[
        ["ts_code", "trade_date", "open", "adj_open"]
    ].copy()

    price_lookup = pd.concat(
        [current_price, boundary_price],
        ignore_index=True,
        sort=False,
    )
    price_lookup = (
        price_lookup
        .drop_duplicates(["ts_code", "trade_date"])
        .reset_index(drop=True)
    )
    assert_unique_key(
        price_lookup,
        KEY,
        name=f"next price lookup {year}-{month:02d}",
    )

    price_lookup = price_lookup.rename(
        columns={
            "trade_date": "next_trade_date",
            "open": "next_open",
            "adj_open": "next_adj_open",
        }
    )

    panel = panel.merge(
        price_lookup,
        on=["ts_code", "next_trade_date"],
        how="left",
        validate="one_to_one",
    )

    current_limits = limits.copy()
    limit_lookup = pd.concat(
        [current_limits, boundary_limits],
        ignore_index=True,
        sort=False,
    )
    limit_lookup = (
        limit_lookup
        .drop_duplicates(KEY)
        .reset_index(drop=True)
    )
    assert_unique_key(
        limit_lookup,
        KEY,
        name=f"next limit lookup {year}-{month:02d}",
    )

    limit_lookup = limit_lookup.rename(
        columns={
            "trade_date": "next_trade_date",
            "up_limit": "next_up_limit",
            "down_limit": "next_down_limit",
        }
    )

    panel = panel.merge(
        limit_lookup,
        on=["ts_code", "next_trade_date"],
        how="left",
        validate="one_to_one",
    )

    current_suspend = sus.copy()
    suspend_lookup = pd.concat(
        [current_suspend, boundary_suspend],
        ignore_index=True,
        sort=False,
    )
    suspend_lookup = (
        suspend_lookup
        .drop_duplicates(KEY)
        .reset_index(drop=True)
    )
    assert_unique_key(
        suspend_lookup,
        KEY,
        name=f"next suspension lookup {year}-{month:02d}",
    )

    suspend_lookup = suspend_lookup.rename(
        columns={
            "trade_date": "next_trade_date",
            "is_suspended": "next_is_suspended",
        }
    )

    panel = panel.merge(
        suspend_lookup,
        on=["ts_code", "next_trade_date"],
        how="left",
        validate="one_to_one",
    )

    panel["next_is_suspended"] = (
        panel["next_is_suspended"]
        .fillna(False)
        .astype(bool)
    )

    panel["has_next_open_quote"] = panel["next_open"].notna()

    eps = 1e-10

    panel["opens_at_or_above_up_limit"] = (
        panel["next_open"].notna()
        & panel["next_up_limit"].notna()
        & (
            panel["next_open"]
            >= panel["next_up_limit"] - eps
        )
    )

    panel["opens_at_or_below_down_limit"] = (
        panel["next_open"].notna()
        & panel["next_down_limit"].notna()
        & (
            panel["next_open"]
            <= panel["next_down_limit"] + eps
        )
    )

    panel["can_buy_next_open"] = (
        panel["next_trade_date"].notna()
        & panel["has_next_open_quote"]
        & (~panel["next_is_suspended"])
        & (~panel["opens_at_or_above_up_limit"])
    )

    panel["can_sell_next_open"] = (
        panel["next_trade_date"].notna()
        & panel["has_next_open_quote"]
        & (~panel["next_is_suspended"])
        & (~panel["opens_at_or_below_down_limit"])
    )

    # ------------------------------------------------------------------
    # Formal sample flag (do not drop warm-up)
    # ------------------------------------------------------------------
    panel["in_formal_research_sample"] = (
        panel["trade_date"] >= cfg.formal_sample_start
    )

    # ------------------------------------------------------------------
    # Final integrity
    # ------------------------------------------------------------------
    panel = panel.sort_values(KEY).reset_index(drop=True)
    assert_unique_key(
        panel,
        KEY,
        name=f"full panel {year}-{month:02d}",
    )

    coverage_df = pd.DataFrame(coverage_rows)
    coverage_df.insert(0, "month", f"{year:04d}-{month:02d}")

    return panel, coverage_df


# ======================================================================================
# Diagnostics aggregation
# ======================================================================================

def monthly_quality(
    panel: pd.DataFrame,
    *,
    year: int,
    month: int,
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "month": f"{year:04d}-{month:02d}",
        "rows": int(len(panel)),
        "unique_ts_codes": int(
            panel["ts_code"].nunique()
        ),
        "duplicate_key_rows": int(
            panel.duplicated(KEY, keep=False).sum()
        ),
    }

    bad_ohlc = (
        (panel["high"] < panel["low"])
        | (panel["open"] < panel["low"])
        | (panel["open"] > panel["high"])
        | (panel["close"] < panel["low"])
        | (panel["close"] > panel["high"])
    ).fillna(False)

    row["ohlc_violations"] = int(bad_ohlc.sum())
    row["nonpositive_adj_factor_rows"] = int(
        (panel["adj_factor"] <= 0)
        .fillna(False)
        .sum()
    )

    row["adj_factor_match_rate"] = float(
        panel["has_adj_factor"].mean()
    )
    row["daily_basic_match_rate"] = float(
        panel["has_daily_basic"].mean()
    )
    row["price_limit_match_rate"] = float(
        panel["has_price_limit"].mean()
    )
    row["stock_basic_metadata_match_rate"] = float(
        panel["has_stock_basic_metadata"].mean()
    )

    row["common_a_share_rate"] = float(
        panel["is_common_a_share"].mean()
    )
    row["seasoned_120d_rate"] = float(
        panel["seasoned_120d"].mean()
    )
    row["base_eligible_rate"] = float(
        panel["base_eligible_signal_day"].mean()
    )

    row["is_st_rows"] = int(panel["is_st"].sum())
    row["is_suspended_rows_on_daily_backbone"] = int(
        panel["is_suspended"].sum()
    )

    row["metadata_trade_before_list_date_rows"] = int(
        panel["metadata_trade_before_list_date"].sum()
    )

    nonterminal = panel["next_trade_date"].notna()
    row["nonterminal_rows"] = int(nonterminal.sum())

    if nonterminal.any():
        row["next_open_quote_rate_nonterminal"] = float(
            panel.loc[
                nonterminal,
                "has_next_open_quote",
            ].mean()
        )
        row["can_buy_next_open_rate_nonterminal"] = float(
            panel.loc[
                nonterminal,
                "can_buy_next_open",
            ].mean()
        )
        row["can_sell_next_open_rate_nonterminal"] = float(
            panel.loc[
                nonterminal,
                "can_sell_next_open",
            ].mean()
        )
    else:
        row["next_open_quote_rate_nonterminal"] = np.nan
        row["can_buy_next_open_rate_nonterminal"] = np.nan
        row["can_sell_next_open_rate_nonterminal"] = np.nan

    row["opens_at_or_above_up_limit_rows"] = int(
        panel["opens_at_or_above_up_limit"].sum()
    )
    row["opens_at_or_below_down_limit_rows"] = int(
        panel["opens_at_or_below_down_limit"].sum()
    )

    hard_fail = any(
        [
            row["duplicate_key_rows"] > 0,
            row["ohlc_violations"] > 0,
            row["nonpositive_adj_factor_rows"] > 0,
            row["adj_factor_match_rate"] < 0.995,
        ]
    )

    row["status"] = "FAIL" if hard_fail else "PASS"

    return row


def update_missingness_accumulator(
    panel: pd.DataFrame,
    accumulator: Dict[str, Dict[str, Any]],
) -> None:
    n = len(panel)

    for col in panel.columns:
        miss = int(panel[col].isna().sum())

        if col not in accumulator:
            accumulator[col] = {
                "column": col,
                "dtype": str(panel[col].dtype),
                "total_rows": 0,
                "missing_cells": 0,
            }

        accumulator[col]["total_rows"] += n
        accumulator[col]["missing_cells"] += miss


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
                "dtype": x["dtype"],
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


def security_type_diagnostics(
    security_history: pd.DataFrame,
) -> pd.DataFrame:
    x = security_history.copy()

    suffix = (
        x["ts_code"]
        .astype("string")
        .str.split(".")
        .str[-1]
    )

    x["suffix"] = suffix

    out = (
        x.groupby(
            ["suffix", "code_prefix3", "is_common_a_share"],
            dropna=False,
        )
        .agg(
            n_codes=("ts_code", "nunique"),
            total_daily_rows=("n_daily_rows", "sum"),
            min_first_observed=(
                "first_observed_trade_date",
                "min",
            ),
            max_last_observed=(
                "last_observed_trade_date",
                "max",
            ),
        )
        .reset_index()
        .sort_values(
            ["is_common_a_share", "total_daily_rows"],
            ascending=[True, False],
        )
    )

    return out


# ======================================================================================
# CLI
# ======================================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Build the full Shanghai/Shenzhen A-share master panel "
            "month by month."
        )
    )

    p.add_argument(
        "--start",
        default=DEFAULT_START,
        help=f"Raw start YYYYMMDD (default {DEFAULT_START})",
    )
    p.add_argument(
        "--end",
        default=DEFAULT_END,
        help=f"Raw end YYYYMMDD (default {DEFAULT_END})",
    )
    p.add_argument(
        "--formal-sample-start",
        default=FORMAL_SAMPLE_START,
        help=(
            "Formal research sample start YYYYMMDD "
            f"(default {FORMAL_SAMPLE_START})"
        ),
    )
    p.add_argument(
        "--data-root",
        default="data",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Rebuild existing monthly panel files and security history.",
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

    ensure_parquet_engine()

    cfg = Config(
        start=args.start,
        end=args.end,
        formal_sample_start=args.formal_sample_start,
        data_root=Path(args.data_root),
        force=bool(args.force),
    )

    cfg.interim_root.mkdir(
        parents=True,
        exist_ok=True,
    )
    cfg.output_root.mkdir(
        parents=True,
        exist_ok=True,
    )
    cfg.report_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    logger = setup_logging(cfg.report_root)

    logger.info("=" * 104)
    logger.info("STAGE 4 | BUILD FULL MASTER PANEL")
    logger.info("Raw period: %s -> %s", cfg.start, cfg.end)
    logger.info(
        "Formal sample starts: %s",
        cfg.formal_sample_start,
    )
    logger.info(
        "Raw root: %s",
        cfg.raw_root.resolve(),
    )
    logger.info(
        "Output root: %s",
        cfg.output_root.resolve(),
    )
    logger.info("force=%s", cfg.force)
    logger.info("=" * 104)

    try:
        (
            trade_cal,
            open_dates,
            trade_index,
            next_trade_date_map,
        ) = load_trade_calendar(
            cfg,
            logger,
        )

        buckets = month_groups(open_dates)

        stock_basic, sb_duplicates = prepare_stock_basic(
            cfg,
            logger,
        )

        security_history = build_or_load_security_history(
            cfg,
            month_buckets=buckets,
            trade_index=trade_index,
            logger=logger,
        )

        sec_diag = security_type_diagnostics(
            security_history
        )
        sec_diag.to_csv(
            cfg.report_root
            / "full_panel_security_type_diagnostics.csv",
            index=False,
            encoding="utf-8-sig",
        )

        merge_frames: List[pd.DataFrame] = []
        quality_rows: List[Dict[str, Any]] = []
        missing_accumulator: Dict[str, Dict[str, Any]] = {}

        total_panel_rows = 0
        formal_sample_rows = 0

        for i, (year, month, dates) in enumerate(buckets, 1):
            month_label = f"{year:04d}-{month:02d}"

            output_path = (
                cfg.output_root
                / f"{year:04d}"
                / f"{month:02d}"
                / f"full_panel_{year:04d}{month:02d}.parquet"
            )

            logger.info(
                "MONTH %d/%d | %s | dates=%d",
                i,
                len(buckets),
                month_label,
                len(dates),
            )

            if valid_parquet(
                output_path,
                [
                    "ts_code",
                    "trade_date",
                    "adj_factor",
                    "is_common_a_share",
                    "observed_trading_age",
                    "seasoned_120d",
                    "next_trade_date",
                    "can_buy_next_open",
                ],
            ) and not cfg.force:
                logger.info(
                    "REUSE %s",
                    output_path,
                )

                panel = pd.read_parquet(
                    output_path
                )

                # Merge coverage from reused outputs can be reconstructed
                # from match flags.
                coverage = pd.DataFrame(
                    [
                        {
                            "month": month_label,
                            "merge": "adj_factor",
                            "left_rows": len(panel),
                            "right_rows": np.nan,
                            "matched_rows": int(
                                panel["has_adj_factor"].sum()
                            ),
                            "unmatched_rows": int(
                                (~panel["has_adj_factor"]).sum()
                            ),
                            "match_rate": float(
                                panel["has_adj_factor"].mean()
                            ),
                        },
                        {
                            "month": month_label,
                            "merge": "daily_basic",
                            "left_rows": len(panel),
                            "right_rows": np.nan,
                            "matched_rows": int(
                                panel["has_daily_basic"].sum()
                            ),
                            "unmatched_rows": int(
                                (~panel["has_daily_basic"]).sum()
                            ),
                            "match_rate": float(
                                panel["has_daily_basic"].mean()
                            ),
                        },
                        {
                            "month": month_label,
                            "merge": "stk_limit",
                            "left_rows": len(panel),
                            "right_rows": np.nan,
                            "matched_rows": int(
                                panel["has_price_limit"].sum()
                            ),
                            "unmatched_rows": int(
                                (~panel["has_price_limit"]).sum()
                            ),
                            "match_rate": float(
                                panel["has_price_limit"].mean()
                            ),
                        },
                    ]
                )

            else:
                panel, coverage = build_one_month(
                    cfg,
                    year=year,
                    month=month,
                    month_dates=dates,
                    trade_index=trade_index,
                    next_trade_date_map=next_trade_date_map,
                    security_history=security_history,
                    stock_basic=stock_basic,
                    logger=logger,
                )

                atomic_write_parquet(
                    panel,
                    output_path,
                )

                logger.info(
                    "SAVE %s rows=%d codes=%d",
                    output_path,
                    len(panel),
                    panel["ts_code"].nunique(),
                )

            merge_frames.append(coverage)

            q = monthly_quality(
                panel,
                year=year,
                month=month,
            )
            quality_rows.append(q)

            update_missingness_accumulator(
                panel,
                missing_accumulator,
            )

            total_panel_rows += len(panel)
            formal_sample_rows += int(
                panel[
                    "in_formal_research_sample"
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
                    "adj=%.4f basic=%.4f limit=%.4f "
                    "commonA=%.4f seasoned=%.4f eligible=%.4f",
                    month_label,
                    q["adj_factor_match_rate"],
                    q["daily_basic_match_rate"],
                    q["price_limit_match_rate"],
                    q["common_a_share_rate"],
                    q["seasoned_120d_rate"],
                    q["base_eligible_rate"],
                )

            del panel, coverage
            gc.collect()

        merge_df = (
            pd.concat(
                merge_frames,
                ignore_index=True,
                sort=False,
            )
            if merge_frames
            else pd.DataFrame()
        )

        quality_df = pd.DataFrame(quality_rows)
        missingness_df = finalize_missingness(
            missing_accumulator
        )

        merge_path = (
            cfg.report_root
            / "full_panel_merge_coverage.csv"
        )
        quality_path = (
            cfg.report_root
            / "full_panel_monthly_quality.csv"
        )
        missingness_path = (
            cfg.report_root
            / "full_panel_missingness_summary.csv"
        )

        merge_df.to_csv(
            merge_path,
            index=False,
            encoding="utf-8-sig",
        )
        quality_df.to_csv(
            quality_path,
            index=False,
            encoding="utf-8-sig",
        )
        missingness_df.to_csv(
            missingness_path,
            index=False,
            encoding="utf-8-sig",
        )

        hard_fail_months = quality_df.loc[
            quality_df["status"] == "FAIL"
        ]

        metadata = {
            "project": (
                "China A-Share Cross-Sectional Alpha Research"
            ),
            "script": "04_build_full_panel.py",
            "generated_at_utc": datetime.now(
                timezone.utc
            ).isoformat(),
            "raw_start": cfg.start,
            "raw_end": cfg.end,
            "formal_sample_start": cfg.formal_sample_start,
            "raw_open_market_dates": len(open_dates),
            "months_built_or_reused": len(buckets),
            "total_panel_rows": total_panel_rows,
            "formal_sample_rows": formal_sample_rows,
            "security_history_codes": int(
                security_history["ts_code"].nunique()
            ),
            "common_a_share_codes_by_rule": int(
                security_history[
                    "is_common_a_share"
                ].sum()
            ),
            "stock_basic_canonical_codes": int(
                stock_basic["ts_code"].nunique()
            ),
            "stock_basic_duplicate_rows_reported": int(
                len(sb_duplicates)
            ),
            "monthly_qa_fail_count": int(
                len(hard_fail_months)
            ),
            "output_root": str(
                cfg.output_root.resolve()
            ),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "pandas_version": package_version("pandas"),
            "numpy_version": package_version("numpy"),
            "pyarrow_version": package_version("pyarrow"),
            "common_a_share_code_rule": {
                "SH": list(COMMON_A_PREFIXES_SH),
                "SZ": list(COMMON_A_PREFIXES_SZ),
            },
            "base_eligible_signal_day_definition": (
                "is_common_a_share & seasoned_120d & ~is_st "
                "& ~is_suspended & close>=5 & has_adj_factor"
            ),
            "notes": [
                "daily is the row backbone.",
                "Panel is built month by month for memory safety.",
                "Next trading date comes from the global market calendar.",
                "Cross-month next-open lookup uses one next-market-day boundary buffer.",
                "Historical daily observations define observed security existence.",
                "stock_basic is auxiliary metadata and does not define the historical universe.",
                "metadata_trade_before_list_date is diagnostic only and does not delete rows.",
                "is_common_a_share is a transparent code-rule flag; rows are not dropped.",
                "ADV20/liquidity ranking is intentionally deferred to Stage 5.",
                "No return target, signal, IC, ML model, portfolio, or backtest is built here.",
                "The right boundary 2025-12-31 has no 2026 forward buffer in the downloaded raw period; final forward-target dates will therefore be dropped later.",
            ],
        }

        atomic_write_json(
            metadata,
            cfg.report_root
            / "full_panel_metadata.json",
        )

        logger.info("=" * 104)
        logger.info(
            "FULL PANEL COMPLETE | total_rows=%d formal_rows=%d months=%d",
            total_panel_rows,
            formal_sample_rows,
            len(buckets),
        )
        logger.info(
            "Reports: %s | %s | %s",
            merge_path,
            quality_path,
            missingness_path,
        )

        if not hard_fail_months.empty:
            logger.error(
                "FULL PANEL QA HAS %d FAILED MONTHS.",
                len(hard_fail_months),
            )
            logger.error(
                "Do NOT proceed to Stage 5 until these months are reviewed."
            )
            logger.info("=" * 104)
            return 1

        logger.info("FULL PANEL MONTHLY QA: PASS")
        logger.info("=" * 104)
        return 0

    except KeyboardInterrupt:
        logger.warning(
            "Interrupted by user. Existing valid monthly panel files are preserved. "
            "Re-run the same command to resume."
        )
        return 130

    except Exception:
        logger.exception(
            "Fatal error while building full panel."
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
