#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
03_download_full_history.py

China A-Share Cross-Sectional Alpha Research
Stage 3: Full-history downloader for Shanghai + Shenzhen A-shares.

Default raw-data period:
    2009-06-01 to 2025-12-31

Formal research sample:
    2010-01-01 to 2025-12-31
Warm-up period:
    2009-06-01 to 2009-12-31

Universe:
    Shanghai + Shenzhen securities only (*.SH, *.SZ)
    Beijing Stock Exchange / NEEQ-mapped *.BJ observations are excluded.

Data downloaded:
    1) stock_basic: L / D / P
    2) trade_cal: SSE + SZSE
    3) daily
    4) adj_factor
    5) daily_basic
    6) stock_st
    7) suspend_d
    8) stk_limit

Robustness / engineering features:
    - Token comes from environment variable; never hard-coded.
    - Tushare-compatible custom HTTP endpoint.
    - Daily cross-sectional download by trade_date.
    - Save by dataset/YYYY/MM/dataset_YYYYMMDD.parquet.
    - Atomic Parquet writes.
    - Strong resume: valid existing partitions are skipped.
    - Retry with capped exponential backoff + jitter.
    - Circuit breaker after repeated terminal network/SSL failures.
    - Re-create API client after circuit break.
    - Failure queue exported to CSV.
    - --repair-only mode to retry only tasks recorded as failed previously.
    - Periodic manifest checkpoints.
    - Full completeness QA by expected market open dates.
    - Filters all date-level tables to *.SH / *.SZ before saving.
    - Does NOT construct factors, universes, adjusted prices, or backtests.

Recommended packages:
    pip install tushare pandas pyarrow requests urllib3 certifi

PowerShell:
    $env:TUSHARE_TOKEN="YOUR_TOKEN"
    python 03_download_full_history.py

Optional endpoint override:
    $env:TUSHARE_HTTP_URL="https://t.xiaodefa.top/"

Repair only:
    python 03_download_full_history.py --repair-only

Safer proxy settings:
    python 03_download_full_history.py --sleep 0.8 --max-retries 8 \
        --backoff-base 2 --circuit-breaker-threshold 3 --circuit-breaker-seconds 90
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import platform
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import pandas as pd
import requests
import tushare as ts


# ======================================================================================
# Defaults
# ======================================================================================

DEFAULT_START = "20090601"
DEFAULT_END = "20251231"
DEFAULT_HTTP_URL = "https://t.xiaodefa.top/"

DATE_TABLES = [
    "daily",
    "adj_factor",
    "daily_basic",
    "stock_st",
    "suspend_d",
    "stk_limit",
]

FIELDS: Dict[str, str] = {
    "stock_basic": (
        "ts_code,symbol,name,area,industry,market,exchange,"
        "list_status,list_date,delist_date"
    ),
    "trade_cal": "exchange,cal_date,is_open,pretrade_date",
    "daily": (
        "ts_code,trade_date,open,high,low,close,pre_close,"
        "change,pct_chg,vol,amount"
    ),
    "adj_factor": "ts_code,trade_date,adj_factor",
    "daily_basic": (
        "ts_code,trade_date,close,turnover_rate,turnover_rate_f,"
        "volume_ratio,total_share,float_share,free_share,"
        "total_mv,circ_mv"
    ),
    "stock_st": "ts_code,name,trade_date,type,type_name",
    "suspend_d": "ts_code,trade_date,suspend_timing,suspend_type",
    "stk_limit": "trade_date,ts_code,pre_close,up_limit,down_limit",
}

REQUIRED_COLUMNS: Dict[str, Sequence[str]] = {
    "stock_basic": ["ts_code", "exchange", "list_status", "list_date"],
    "trade_cal": ["exchange", "cal_date", "is_open"],
    "daily": [
        "ts_code", "trade_date", "open", "high", "low", "close",
        "pre_close", "vol", "amount",
    ],
    "adj_factor": ["ts_code", "trade_date", "adj_factor"],
    "daily_basic": [
        "ts_code", "trade_date", "turnover_rate", "volume_ratio",
        "total_mv", "circ_mv",
    ],
    "stock_st": ["ts_code", "trade_date", "type", "type_name"],
    "suspend_d": ["ts_code", "trade_date", "suspend_type"],
    "stk_limit": ["ts_code", "trade_date", "up_limit", "down_limit"],
}

NETWORK_EXCEPTIONS = (
    requests.exceptions.SSLError,
    requests.exceptions.ConnectionError,
    requests.exceptions.Timeout,
    requests.exceptions.ChunkedEncodingError,
)

TASK_COLUMNS = [
    "dataset",
    "trade_date",
    "segment",
    "status",
    "rows",
    "file_path",
    "attempted_at_utc",
    "error_type",
    "error",
]


# ======================================================================================
# Configuration / state
# ======================================================================================

@dataclass
class Config:
    start: str
    end: str
    data_root: Path
    http_url: str
    token: str
    sleep_seconds: float
    max_retries: int
    backoff_base_seconds: float
    backoff_cap_seconds: float
    circuit_breaker_threshold: int
    circuit_breaker_seconds: float
    checkpoint_every: int
    repair_only: bool
    force: bool

    @property
    def raw_root(self) -> Path:
        return self.data_root / "raw_full"

    @property
    def report_root(self) -> Path:
        return self.data_root / "full_reports"

    @property
    def manifest_path(self) -> Path:
        return self.report_root / "full_download_manifest.csv"

    @property
    def failures_path(self) -> Path:
        return self.report_root / "full_download_failures.csv"

    @property
    def summary_path(self) -> Path:
        return self.report_root / "full_download_summary.csv"

    @property
    def quality_path(self) -> Path:
        return self.report_root / "full_download_quality.csv"

    @property
    def metadata_path(self) -> Path:
        return self.report_root / "full_download_metadata.json"


@dataclass
class RuntimeState:
    consecutive_terminal_network_failures: int = 0
    tasks_seen: int = 0
    tasks_downloaded: int = 0
    tasks_skipped: int = 0
    tasks_failed: int = 0
    client_rebuilds: int = 0


# ======================================================================================
# Logging / file helpers
# ======================================================================================

def setup_logging(report_root: Path) -> logging.Logger:
    report_root.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("full_download")
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
        report_root / "03_download_full_history.log",
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
            "Parquet output requires pyarrow.\n"
            "Install with: pip install pyarrow"
        ) from exc


def atomic_write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    df.to_parquet(tmp, index=False, engine="pyarrow", compression="snappy")
    tmp.replace(path)


def atomic_write_json(obj: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)
    tmp.replace(path)


def valid_parquet(path: Path, required_cols: Optional[Sequence[str]] = None) -> bool:
    if not path.exists() or path.stat().st_size <= 0:
        return False
    try:
        df = pd.read_parquet(path)
        if required_cols:
            missing = [c for c in required_cols if c not in df.columns]
            if missing and not df.empty:
                return False
        return True
    except Exception:
        return False


def append_manifest_rows(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=TASK_COLUMNS)
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in TASK_COLUMNS})


def dataset_partition_path(raw_root: Path, dataset: str, trade_date: str) -> Path:
    yyyy = trade_date[:4]
    mm = trade_date[4:6]
    return raw_root / dataset / yyyy / mm / f"{dataset}_{trade_date}.parquet"


# ======================================================================================
# API helpers
# ======================================================================================

def make_pro_client(token: str, http_url: str):
    pro = ts.pro_api(token)
    pro._DataApi__http_url = http_url
    return pro


def is_network_exception(exc: BaseException) -> bool:
    cur: Optional[BaseException] = exc
    depth = 0
    while cur is not None and depth < 8:
        if isinstance(cur, NETWORK_EXCEPTIONS):
            return True
        text = repr(cur).lower()
        if any(
            s in text
            for s in [
                "ssl",
                "unexpected_eof",
                "connection reset",
                "connection aborted",
                "max retries exceeded",
                "remote end closed",
                "timed out",
            ]
        ):
            return True
        cur = cur.__cause__ or cur.__context__
        depth += 1
    return False


def api_call_with_retry(
    func: Callable[..., pd.DataFrame],
    *,
    api_name: str,
    logger: logging.Logger,
    cfg: Config,
    **kwargs: Any,
) -> pd.DataFrame:
    last_exc: Optional[BaseException] = None

    for attempt in range(1, cfg.max_retries + 1):
        try:
            df = func(**kwargs)
            if df is None:
                df = pd.DataFrame()
            if not isinstance(df, pd.DataFrame):
                df = pd.DataFrame(df)

            if cfg.sleep_seconds > 0:
                time.sleep(cfg.sleep_seconds)
            return df

        except KeyboardInterrupt:
            raise

        except BaseException as exc:
            last_exc = exc
            network = is_network_exception(exc)

            if attempt >= cfg.max_retries:
                break

            wait = min(
                cfg.backoff_cap_seconds,
                cfg.backoff_base_seconds * (2 ** (attempt - 1)),
            ) + random.uniform(0.0, 1.0)

            logger.warning(
                "%s failed attempt %d/%d | network=%s | %s | retry in %.2fs",
                api_name,
                attempt,
                cfg.max_retries,
                network,
                repr(exc),
                wait,
            )
            time.sleep(wait)

    assert last_exc is not None
    raise RuntimeError(
        f"{api_name} failed after {cfg.max_retries} attempts"
    ) from last_exc


def apply_circuit_breaker_if_needed(
    *,
    state: RuntimeState,
    cfg: Config,
    logger: logging.Logger,
) -> bool:
    if (
        state.consecutive_terminal_network_failures
        < cfg.circuit_breaker_threshold
    ):
        return False

    logger.error(
        "CIRCUIT BREAKER OPEN | consecutive terminal network failures=%d | sleep %.1fs",
        state.consecutive_terminal_network_failures,
        cfg.circuit_breaker_seconds,
    )
    time.sleep(cfg.circuit_breaker_seconds)
    state.consecutive_terminal_network_failures = 0
    state.client_rebuilds += 1
    logger.info("CIRCUIT BREAKER CLOSED | API client will be rebuilt.")
    return True


# ======================================================================================
# Data normalization
# ======================================================================================

def filter_sh_sz(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "ts_code" not in df.columns:
        return df.copy()
    mask = df["ts_code"].astype(str).str.endswith((".SH", ".SZ"))
    return df.loc[mask].reset_index(drop=True)


def normalize_date_string(s: pd.Series) -> pd.Series:
    return (
        s.astype("string")
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(8)
    )


def validate_yyyymmdd(value: str, name: str) -> None:
    try:
        datetime.strptime(value, "%Y%m%d")
    except ValueError as exc:
        raise ValueError(f"{name} must be YYYYMMDD, got {value!r}") from exc


# ======================================================================================
# Static/master downloads
# ======================================================================================

def download_stock_basic(
    pro,
    cfg: Config,
    logger: logging.Logger,
) -> pd.DataFrame:
    out_dir = cfg.raw_root / "stock_basic"
    out_dir.mkdir(parents=True, exist_ok=True)
    combined_path = out_dir / "stock_basic_SH_SZ_all.parquet"

    if valid_parquet(combined_path, ["ts_code", "list_status"]) and not cfg.force:
        logger.info("SKIP stock_basic combined -> %s", combined_path)
        return pd.read_parquet(combined_path)

    frames: List[pd.DataFrame] = []

    for status in ("L", "D", "P"):
        path = out_dir / f"stock_basic_{status}.parquet"

        if valid_parquet(path, ["ts_code"]) and not cfg.force:
            logger.info("SKIP stock_basic status=%s", status)
            frames.append(pd.read_parquet(path))
            continue

        logger.info("GET  stock_basic status=%s", status)
        df = api_call_with_retry(
            pro.stock_basic,
            api_name=f"stock_basic[{status}]",
            logger=logger,
            cfg=cfg,
            exchange="",
            list_status=status,
            fields=FIELDS["stock_basic"],
        )
        if "list_status" not in df.columns:
            df["list_status"] = status

        df = filter_sh_sz(df)
        atomic_write_parquet(df, path)
        frames.append(df)
        logger.info("SAVE stock_basic status=%s rows=%d", status, len(df))

    all_basic = pd.concat(frames, ignore_index=True, sort=False)
    all_basic = all_basic.drop_duplicates().reset_index(drop=True)

    if "ts_code" in all_basic.columns:
        dup = int(all_basic.duplicated("ts_code", keep=False).sum())
        if dup:
            logger.warning(
                "stock_basic has %d rows on duplicated ts_code after combining L/D/P. "
                "This is retained for diagnostics; do not silently drop by ts_code.",
                dup,
            )

    atomic_write_parquet(all_basic, combined_path)
    logger.info("SAVE stock_basic combined rows=%d", len(all_basic))
    return all_basic


def download_trade_cal(
    pro,
    cfg: Config,
    logger: logging.Logger,
) -> pd.DataFrame:
    out_dir = cfg.raw_root / "trade_cal"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"trade_cal_{cfg.start}_{cfg.end}.parquet"

    if valid_parquet(path, ["exchange", "cal_date", "is_open"]) and not cfg.force:
        logger.info("SKIP trade_cal -> %s", path)
        return pd.read_parquet(path)

    frames = []
    for exchange in ("SSE", "SZSE"):
        logger.info("GET  trade_cal exchange=%s", exchange)
        df = api_call_with_retry(
            pro.trade_cal,
            api_name=f"trade_cal[{exchange}]",
            logger=logger,
            cfg=cfg,
            exchange=exchange,
            start_date=cfg.start,
            end_date=cfg.end,
            fields=FIELDS["trade_cal"],
        )
        if "exchange" not in df.columns:
            df["exchange"] = exchange
        frames.append(df)

    cal = pd.concat(frames, ignore_index=True, sort=False)
    cal = cal.drop_duplicates().reset_index(drop=True)
    atomic_write_parquet(cal, path)
    logger.info("SAVE trade_cal rows=%d", len(cal))
    return cal


def get_open_dates(
    trade_cal: pd.DataFrame,
    start: str,
    end: str,
) -> List[str]:
    cal = trade_cal.copy()
    cal["cal_date"] = normalize_date_string(cal["cal_date"])
    cal["is_open"] = pd.to_numeric(cal["is_open"], errors="coerce")

    dates = (
        cal.loc[
            (cal["is_open"] == 1)
            & (cal["cal_date"] >= start)
            & (cal["cal_date"] <= end),
            "cal_date",
        ]
        .dropna()
        .drop_duplicates()
        .sort_values()
        .tolist()
    )
    if not dates:
        raise RuntimeError("No open market dates found.")
    return dates


# ======================================================================================
# Repair task loading
# ======================================================================================

def load_repair_tasks(cfg: Config, logger: logging.Logger) -> Optional[Set[Tuple[str, str]]]:
    if not cfg.repair_only:
        return None

    if not cfg.failures_path.exists():
        raise FileNotFoundError(
            f"--repair-only requested but failure file does not exist: {cfg.failures_path}"
        )

    fail = pd.read_csv(cfg.failures_path, dtype=str)
    if fail.empty:
        logger.info("Repair file is empty. Nothing to repair.")
        return set()

    if not {"dataset", "trade_date"}.issubset(fail.columns):
        raise ValueError(
            f"Failure file missing dataset/trade_date columns: {cfg.failures_path}"
        )

    tasks = {
        (str(r.dataset), str(r.trade_date).zfill(8))
        for r in fail.itertuples(index=False)
        if str(r.dataset) in DATE_TABLES and str(r.trade_date) not in ("", "nan")
    }
    logger.info("REPAIR MODE tasks=%d", len(tasks))
    return tasks


# ======================================================================================
# Download one task
# ======================================================================================

def download_one_date_table(
    *,
    pro,
    dataset: str,
    trade_date: str,
    cfg: Config,
    logger: logging.Logger,
    state: RuntimeState,
) -> Tuple[str, Dict[str, Any], Any]:
    """
    Returns:
        status: skipped / downloaded / failed
        manifest row
        pro: possibly rebuilt client
    """
    path = dataset_partition_path(cfg.raw_root, dataset, trade_date)
    state.tasks_seen += 1

    if valid_parquet(path, REQUIRED_COLUMNS.get(dataset)) and not cfg.force:
        state.tasks_skipped += 1
        row = {
            "dataset": dataset,
            "trade_date": trade_date,
            "segment": "",
            "status": "skipped",
            "rows": "",
            "file_path": str(path),
            "attempted_at_utc": datetime.now(timezone.utc).isoformat(),
            "error_type": "",
            "error": "",
        }
        return "skipped", row, pro

    func = getattr(pro, dataset)

    kwargs: Dict[str, Any] = {
        "trade_date": trade_date,
        "fields": FIELDS[dataset],
    }
    if dataset == "suspend_d":
        kwargs["suspend_type"] = "S"

    logger.info("GET  %-12s %s", dataset, trade_date)

    try:
        df = api_call_with_retry(
            func,
            api_name=f"{dataset}[{trade_date}]",
            logger=logger,
            cfg=cfg,
            **kwargs,
        )

        if not df.empty and "trade_date" not in df.columns:
            df["trade_date"] = trade_date

        df = filter_sh_sz(df)
        atomic_write_parquet(df, path)

        state.tasks_downloaded += 1
        state.consecutive_terminal_network_failures = 0

        row = {
            "dataset": dataset,
            "trade_date": trade_date,
            "segment": "",
            "status": "downloaded",
            "rows": len(df),
            "file_path": str(path),
            "attempted_at_utc": datetime.now(timezone.utc).isoformat(),
            "error_type": "",
            "error": "",
        }

        logger.info("SAVE %-12s %s rows=%d", dataset, trade_date, len(df))
        return "downloaded", row, pro

    except Exception as exc:
        state.tasks_failed += 1
        network = is_network_exception(exc)

        if network:
            state.consecutive_terminal_network_failures += 1
        else:
            state.consecutive_terminal_network_failures = 0

        row = {
            "dataset": dataset,
            "trade_date": trade_date,
            "segment": "",
            "status": "failed",
            "rows": "",
            "file_path": str(path),
            "attempted_at_utc": datetime.now(timezone.utc).isoformat(),
            "error_type": type(exc).__name__,
            "error": repr(exc),
        }

        logger.exception(
            "FAILED %-12s %s | terminal_network=%s | consecutive_network_failures=%d",
            dataset,
            trade_date,
            network,
            state.consecutive_terminal_network_failures,
        )

        if apply_circuit_breaker_if_needed(
            state=state,
            cfg=cfg,
            logger=logger,
        ):
            pro = make_pro_client(cfg.token, cfg.http_url)

        return "failed", row, pro


# ======================================================================================
# Main date-task loop
# ======================================================================================

def run_date_downloads(
    *,
    pro,
    open_dates: Sequence[str],
    cfg: Config,
    logger: logging.Logger,
    state: RuntimeState,
    repair_tasks: Optional[Set[Tuple[str, str]]],
) -> Tuple[Any, List[Dict[str, Any]]]:
    pending_manifest: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []

    all_tasks: List[Tuple[str, str]] = [
        (dataset, trade_date)
        for trade_date in open_dates
        for dataset in DATE_TABLES
    ]

    if repair_tasks is not None:
        all_tasks = [task for task in all_tasks if task in repair_tasks]

    total = len(all_tasks)
    logger.info("DATE TASKS total=%d", total)

    for idx, (dataset, trade_date) in enumerate(all_tasks, 1):
        logger.info(
            "Progress %d/%d | date=%s | dataset=%s",
            idx,
            total,
            trade_date,
            dataset,
        )

        status, manifest_row, pro = download_one_date_table(
            pro=pro,
            dataset=dataset,
            trade_date=trade_date,
            cfg=cfg,
            logger=logger,
            state=state,
        )
        pending_manifest.append(manifest_row)

        if status == "failed":
            failures.append(manifest_row)

        if (
            len(pending_manifest) >= cfg.checkpoint_every
            or idx == total
        ):
            append_manifest_rows(cfg.manifest_path, pending_manifest)
            pending_manifest.clear()
            logger.info(
                "CHECKPOINT | seen=%d downloaded=%d skipped=%d failed=%d",
                state.tasks_seen,
                state.tasks_downloaded,
                state.tasks_skipped,
                state.tasks_failed,
            )

    return pro, failures


# ======================================================================================
# Completeness / quality reports
# ======================================================================================

def scan_dataset_partitions(
    raw_root: Path,
    dataset: str,
    expected_dates: Sequence[str],
) -> Dict[str, Any]:
    found_dates: Set[str] = set()
    corrupt_files: List[str] = []
    total_rows = 0
    unique_ts: Set[str] = set()
    duplicate_rows = 0
    missing_required_columns_count = 0
    ohlc_violations = 0
    negative_vol = 0
    negative_amount = 0
    nonpositive_adj_factor = 0
    non_sh_sz_rows = 0

    paths = sorted((raw_root / dataset).glob("*/*/*.parquet"))

    for path in paths:
        try:
            df = pd.read_parquet(path)
        except Exception:
            corrupt_files.append(str(path))
            continue

        stem = path.stem
        date = stem.rsplit("_", 1)[-1]
        if len(date) == 8 and date.isdigit():
            found_dates.add(date)

        total_rows += len(df)

        req = REQUIRED_COLUMNS.get(dataset, [])
        missing = [c for c in req if c not in df.columns]
        if missing and not df.empty:
            missing_required_columns_count += 1

        if "ts_code" in df.columns:
            ts = df["ts_code"].astype(str)
            unique_ts.update(ts.dropna().tolist())
            non_sh_sz_rows += int(
                (~ts.str.endswith((".SH", ".SZ"))).fillna(False).sum()
            )

        if {"ts_code", "trade_date"}.issubset(df.columns):
            duplicate_rows += int(
                df.duplicated(["ts_code", "trade_date"], keep=False).sum()
            )

        if dataset == "daily":
            d = df.copy()
            for c in ["open", "high", "low", "close", "vol", "amount"]:
                if c in d.columns:
                    d[c] = pd.to_numeric(d[c], errors="coerce")
            if {"open", "high", "low", "close"}.issubset(d.columns):
                bad = (
                    (d["high"] < d["low"])
                    | (d["open"] < d["low"])
                    | (d["open"] > d["high"])
                    | (d["close"] < d["low"])
                    | (d["close"] > d["high"])
                ).fillna(False)
                ohlc_violations += int(bad.sum())
            if "vol" in d.columns:
                negative_vol += int((d["vol"] < 0).fillna(False).sum())
            if "amount" in d.columns:
                negative_amount += int((d["amount"] < 0).fillna(False).sum())

        if dataset == "adj_factor" and "adj_factor" in df.columns:
            x = pd.to_numeric(df["adj_factor"], errors="coerce")
            nonpositive_adj_factor += int((x <= 0).fillna(False).sum())

    expected_set = set(expected_dates)
    missing_dates = sorted(expected_set - found_dates)
    extra_dates = sorted(found_dates - expected_set)

    return {
        "dataset": dataset,
        "expected_partitions": len(expected_dates),
        "found_partitions": len(found_dates),
        "missing_partitions": len(missing_dates),
        "extra_partitions": len(extra_dates),
        "total_rows": total_rows,
        "unique_ts_codes": len(unique_ts),
        "corrupt_files": len(corrupt_files),
        "missing_required_column_partitions": missing_required_columns_count,
        "duplicate_rows_on_key": duplicate_rows,
        "non_sh_sz_rows": non_sh_sz_rows,
        "ohlc_violations": ohlc_violations if dataset == "daily" else None,
        "negative_vol_rows": negative_vol if dataset == "daily" else None,
        "negative_amount_rows": negative_amount if dataset == "daily" else None,
        "nonpositive_adj_factor_rows": (
            nonpositive_adj_factor if dataset == "adj_factor" else None
        ),
        "missing_dates_sample": ",".join(missing_dates[:30]),
        "extra_dates_sample": ",".join(extra_dates[:30]),
        "corrupt_files_sample": ",".join(corrupt_files[:10]),
    }


def write_full_reports(
    *,
    cfg: Config,
    open_dates: Sequence[str],
    current_run_failures: Sequence[Dict[str, Any]],
    logger: logging.Logger,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cfg.report_root.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    quality_rows = []

    for dataset in DATE_TABLES:
        x = scan_dataset_partitions(cfg.raw_root, dataset, open_dates)
        summary_rows.append(x)

        hard_fail = (
            x["missing_partitions"] > 0
            or x["corrupt_files"] > 0
            or x["duplicate_rows_on_key"] > 0
            or x["non_sh_sz_rows"] > 0
            or x["missing_required_column_partitions"] > 0
        )

        if dataset == "daily":
            hard_fail = hard_fail or any(
                [
                    (x["ohlc_violations"] or 0) > 0,
                    (x["negative_vol_rows"] or 0) > 0,
                    (x["negative_amount_rows"] or 0) > 0,
                ]
            )

        if dataset == "adj_factor":
            hard_fail = hard_fail or (
                (x["nonpositive_adj_factor_rows"] or 0) > 0
            )

        quality_rows.append(
            {
                "dataset": dataset,
                "status": "FAIL" if hard_fail else "PASS",
                "expected_partitions": x["expected_partitions"],
                "found_partitions": x["found_partitions"],
                "missing_partitions": x["missing_partitions"],
                "corrupt_files": x["corrupt_files"],
                "duplicate_rows_on_key": x["duplicate_rows_on_key"],
                "non_sh_sz_rows": x["non_sh_sz_rows"],
                "missing_required_column_partitions": x[
                    "missing_required_column_partitions"
                ],
                "ohlc_violations": x["ohlc_violations"],
                "negative_vol_rows": x["negative_vol_rows"],
                "negative_amount_rows": x["negative_amount_rows"],
                "nonpositive_adj_factor_rows": x["nonpositive_adj_factor_rows"],
                "missing_dates_sample": x["missing_dates_sample"],
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    quality_df = pd.DataFrame(quality_rows)

    summary_df.to_csv(
        cfg.summary_path,
        index=False,
        encoding="utf-8-sig",
    )
    quality_df.to_csv(
        cfg.quality_path,
        index=False,
        encoding="utf-8-sig",
    )

    failure_columns = TASK_COLUMNS
    fail_df = pd.DataFrame(
        current_run_failures,
        columns=failure_columns,
    )
    fail_df.to_csv(
        cfg.failures_path,
        index=False,
        encoding="utf-8-sig",
    )

    logger.info("WROTE %s", cfg.summary_path)
    logger.info("WROTE %s", cfg.quality_path)
    logger.info("WROTE %s", cfg.failures_path)

    return summary_df, quality_df


def write_metadata(
    *,
    cfg: Config,
    open_dates: Sequence[str],
    state: RuntimeState,
    logger: logging.Logger,
) -> None:
    def version(name: str) -> str:
        try:
            from importlib.metadata import version as _v
            return _v(name)
        except Exception:
            return "unknown"

    obj = {
        "project": "China A-Share Cross-Sectional Alpha Research",
        "script": "03_download_full_history.py",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "raw_start": cfg.start,
        "raw_end": cfg.end,
        "warmup_period": "2009-06-01 to 2009-12-31",
        "formal_research_sample": "2010-01-01 to 2025-12-31",
        "universe": "Shanghai + Shenzhen securities only (*.SH, *.SZ)",
        "bse_excluded": True,
        "open_market_dates": len(open_dates),
        "http_url": cfg.http_url,
        "token": "***REDACTED***",
        "raw_root": str(cfg.raw_root.resolve()),
        "repair_only": cfg.repair_only,
        "force": cfg.force,
        "sleep_seconds": cfg.sleep_seconds,
        "max_retries": cfg.max_retries,
        "backoff_base_seconds": cfg.backoff_base_seconds,
        "backoff_cap_seconds": cfg.backoff_cap_seconds,
        "circuit_breaker_threshold": cfg.circuit_breaker_threshold,
        "circuit_breaker_seconds": cfg.circuit_breaker_seconds,
        "runtime_state": {
            "tasks_seen": state.tasks_seen,
            "tasks_downloaded": state.tasks_downloaded,
            "tasks_skipped": state.tasks_skipped,
            "tasks_failed": state.tasks_failed,
            "client_rebuilds": state.client_rebuilds,
        },
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "pandas_version": version("pandas"),
        "tushare_version": version("tushare"),
        "pyarrow_version": version("pyarrow"),
        "requests_version": version("requests"),
        "notes": [
            "All date-level datasets are filtered to *.SH/*.SZ before storage.",
            "The historical .BJ / NEEQ-mapped records discovered in pilot are excluded.",
            "stock_basic is auxiliary metadata and is not treated as the sole historical universe truth.",
            "Raw daily partitions are retained at day granularity for robust resume/repair.",
            "Exact trading-day IPO seasoning is deferred to panel construction.",
            "No signals, portfolio construction, ML models, or backtests are created here.",
        ],
    }

    atomic_write_json(obj, cfg.metadata_path)
    logger.info("WROTE %s", cfg.metadata_path)


# ======================================================================================
# CLI
# ======================================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Download full 2009H2-2025 Shanghai/Shenzhen A-share raw history "
            "from a Tushare-compatible endpoint."
        )
    )

    p.add_argument("--start", default=DEFAULT_START)
    p.add_argument("--end", default=DEFAULT_END)
    p.add_argument("--data-root", default="data")
    p.add_argument(
        "--http-url",
        default=os.getenv("TUSHARE_HTTP_URL", DEFAULT_HTTP_URL),
    )
    p.add_argument(
        "--sleep",
        type=float,
        default=0.8,
        help="Sleep after each successful API call (default: 0.8s)",
    )
    p.add_argument(
        "--max-retries",
        type=int,
        default=8,
        help="Retry attempts per API task (default: 8)",
    )
    p.add_argument(
        "--backoff-base",
        type=float,
        default=2.0,
        help="Exponential backoff base seconds (default: 2)",
    )
    p.add_argument(
        "--backoff-cap",
        type=float,
        default=60.0,
        help="Maximum per-retry backoff seconds (default: 60)",
    )
    p.add_argument(
        "--circuit-breaker-threshold",
        type=int,
        default=3,
        help="Open breaker after N terminal network failures (default: 3)",
    )
    p.add_argument(
        "--circuit-breaker-seconds",
        type=float,
        default=90.0,
        help="Circuit-breaker cooldown seconds (default: 90)",
    )
    p.add_argument(
        "--checkpoint-every",
        type=int,
        default=50,
        help="Append manifest after N tasks (default: 50)",
    )
    p.add_argument(
        "--repair-only",
        action="store_true",
        help="Retry only dataset/date tasks listed in full_download_failures.csv",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-download and overwrite existing valid partitions",
    )

    return p.parse_args()


# ======================================================================================
# Main
# ======================================================================================

def main() -> int:
    args = parse_args()

    token = os.getenv("TUSHARE_TOKEN", "").strip()
    if not token:
        print(
            "ERROR: TUSHARE_TOKEN is not set.\n\n"
            "PowerShell:\n"
            '  $env:TUSHARE_TOKEN="YOUR_TOKEN"\n'
            "  python 03_download_full_history.py\n",
            file=sys.stderr,
        )
        return 2

    validate_yyyymmdd(args.start, "--start")
    validate_yyyymmdd(args.end, "--end")
    if args.start > args.end:
        raise ValueError("--start must be <= --end")

    ensure_parquet_engine()

    cfg = Config(
        start=args.start,
        end=args.end,
        data_root=Path(args.data_root),
        http_url=args.http_url,
        token=token,
        sleep_seconds=max(0.0, args.sleep),
        max_retries=max(1, args.max_retries),
        backoff_base_seconds=max(0.1, args.backoff_base),
        backoff_cap_seconds=max(1.0, args.backoff_cap),
        circuit_breaker_threshold=max(1, args.circuit_breaker_threshold),
        circuit_breaker_seconds=max(1.0, args.circuit_breaker_seconds),
        checkpoint_every=max(1, args.checkpoint_every),
        repair_only=bool(args.repair_only),
        force=bool(args.force),
    )

    cfg.raw_root.mkdir(parents=True, exist_ok=True)
    cfg.report_root.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(cfg.report_root)
    state = RuntimeState()

    logger.info("=" * 100)
    logger.info("FULL-HISTORY DOWNLOAD | SHANGHAI + SHENZHEN ONLY")
    logger.info("Raw period: %s -> %s", cfg.start, cfg.end)
    logger.info("Formal sample: 20100101 -> 20251231")
    logger.info("Warm-up: 20090601 -> 20091231")
    logger.info("Endpoint: %s", cfg.http_url)
    logger.info("Raw root: %s", cfg.raw_root.resolve())
    logger.info("repair_only=%s force=%s", cfg.repair_only, cfg.force)
    logger.info("Token: ***REDACTED***")
    logger.info("=" * 100)

    try:
        pro = make_pro_client(cfg.token, cfg.http_url)

        # Static/master data.
        # In repair mode we still need trade_cal, but stock_basic can simply be reused.
        download_stock_basic(pro, cfg, logger)
        trade_cal = download_trade_cal(pro, cfg, logger)
        open_dates = get_open_dates(trade_cal, cfg.start, cfg.end)

        logger.info(
            "OPEN MARKET DATES=%d | first=%s | last=%s",
            len(open_dates),
            open_dates[0],
            open_dates[-1],
        )

        repair_tasks = load_repair_tasks(cfg, logger)
        if repair_tasks == set():
            # Nothing explicitly failed previously; still run completeness reports.
            logger.info("No failed tasks to repair.")

        pro, current_run_failures = run_date_downloads(
            pro=pro,
            open_dates=open_dates,
            cfg=cfg,
            logger=logger,
            state=state,
            repair_tasks=repair_tasks,
        )

        summary_df, quality_df = write_full_reports(
            cfg=cfg,
            open_dates=open_dates,
            current_run_failures=current_run_failures,
            logger=logger,
        )
        write_metadata(
            cfg=cfg,
            open_dates=open_dates,
            state=state,
            logger=logger,
        )

        logger.info("=" * 100)
        logger.info(
            "RUN FINISHED | seen=%d downloaded=%d skipped=%d failed=%d client_rebuilds=%d",
            state.tasks_seen,
            state.tasks_downloaded,
            state.tasks_skipped,
            state.tasks_failed,
            state.client_rebuilds,
        )

        for row in summary_df.to_dict("records"):
            logger.info(
                "  %-12s partitions=%d/%d missing=%d rows=%d unique_ts=%d",
                row["dataset"],
                row["found_partitions"],
                row["expected_partitions"],
                row["missing_partitions"],
                row["total_rows"],
                row["unique_ts_codes"],
            )

        hard_fails = quality_df.loc[quality_df["status"] == "FAIL"]
        if not hard_fails.empty:
            logger.error("FULL DOWNLOAD QA HAS FAILURES.")
            logger.error(
                "Run again normally to fill missing partitions, or use --repair-only "
                "after inspecting full_download_failures.csv."
            )
            logger.info("=" * 100)
            return 1

        if current_run_failures:
            logger.error(
                "Current run had failures even though partition QA passed unexpectedly. "
                "Inspect full_download_failures.csv."
            )
            logger.info("=" * 100)
            return 1

        logger.info("FULL DOWNLOAD COMPLETENESS QA: PASS")
        logger.info("=" * 100)
        return 0

    except KeyboardInterrupt:
        logger.warning(
            "Interrupted by user. Existing Parquet partitions and manifest checkpoints "
            "are preserved. Re-run the same command to resume."
        )
        return 130

    except Exception:
        logger.exception("Fatal error.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
