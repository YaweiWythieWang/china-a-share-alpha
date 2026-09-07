#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
03b_repair_20130410.py

Targeted repair for the market-wide empty/missing daily partition discovered
during Stage 5 diagnostics.

Repairs all six date-level datasets for 2013-04-10 and overwrites the existing
raw_full partitions:
    daily
    adj_factor
    daily_basic
    stock_st
    suspend_d
    stk_limit

Token:
    environment variable TUSHARE_TOKEN

Endpoint:
    TUSHARE_HTTP_URL or default https://t.xiaodefa.top/

Run:
    python 03b_repair_20130410.py
"""

from __future__ import annotations

import os
import random
import time
from pathlib import Path

import pandas as pd
import tushare as ts

TRADE_DATE = "20130410"
DEFAULT_HTTP_URL = "https://t.xiaodefa.top/"
DATA_ROOT = Path("data")
RAW_ROOT = DATA_ROOT / "raw_full"

FIELDS = {
    "daily": (
        "ts_code,trade_date,open,high,low,close,pre_close,"
        "change,pct_chg,vol,amount"
    ),
    "adj_factor": "ts_code,trade_date,adj_factor",
    "daily_basic": (
        "ts_code,trade_date,close,turnover_rate,turnover_rate_f,"
        "volume_ratio,total_share,float_share,free_share,total_mv,circ_mv"
    ),
    "stock_st": "ts_code,name,trade_date,type,type_name",
    "suspend_d": "ts_code,trade_date,suspend_timing,suspend_type",
    "stk_limit": "trade_date,ts_code,pre_close,up_limit,down_limit",
}


def filter_sh_sz(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "ts_code" not in df.columns:
        return df.copy()
    mask = df["ts_code"].astype(str).str.endswith((".SH", ".SZ"))
    return df.loc[mask].reset_index(drop=True)


def output_path(dataset: str) -> Path:
    return (
        RAW_ROOT
        / dataset
        / TRADE_DATE[:4]
        / TRADE_DATE[4:6]
        / f"{dataset}_{TRADE_DATE}.parquet"
    )


def atomic_write(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    df.to_parquet(tmp, index=False, engine="pyarrow", compression="snappy")
    tmp.replace(path)


def call_with_retry(func, name: str, max_retries: int = 8, **kwargs):
    last = None
    for attempt in range(1, max_retries + 1):
        try:
            df = func(**kwargs)
            if df is None:
                df = pd.DataFrame()
            time.sleep(0.8)
            return df
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            last = exc
            if attempt == max_retries:
                break
            wait = min(60.0, 2.0 * (2 ** (attempt - 1))) + random.uniform(0, 1)
            print(
                f"[WARN] {name} attempt {attempt}/{max_retries} failed: "
                f"{exc!r}; retry in {wait:.1f}s"
            )
            time.sleep(wait)
    raise RuntimeError(f"{name} failed after {max_retries} attempts") from last


def main():
    token = os.getenv("TUSHARE_TOKEN", "").strip()
    if not token:
        raise RuntimeError(
            'TUSHARE_TOKEN is not set.\n'
            'PowerShell: $env:TUSHARE_TOKEN="YOUR_TOKEN"'
        )

    http_url = os.getenv("TUSHARE_HTTP_URL", DEFAULT_HTTP_URL)

    pro = ts.pro_api(token)
    pro._DataApi__http_url = http_url

    print("=" * 80)
    print(f"Repairing market date {TRADE_DATE}")
    print(f"Endpoint: {http_url}")
    print("=" * 80)

    results = {}

    for dataset in [
        "daily",
        "adj_factor",
        "daily_basic",
        "stock_st",
        "suspend_d",
        "stk_limit",
    ]:
        func = getattr(pro, dataset)

        kwargs = {
            "trade_date": TRADE_DATE,
            "fields": FIELDS[dataset],
        }
        if dataset == "suspend_d":
            kwargs["suspend_type"] = "S"

        print(f"[GET] {dataset} {TRADE_DATE}")

        df = call_with_retry(
            func,
            f"{dataset}[{TRADE_DATE}]",
            **kwargs,
        )

        if not df.empty and "trade_date" not in df.columns:
            df["trade_date"] = TRADE_DATE

        df = filter_sh_sz(df)

        # On a genuine mainland market-open date, these core tables must be nonempty.
        if dataset in {"daily", "adj_factor", "daily_basic", "stk_limit"} and df.empty:
            raise RuntimeError(
                f"{dataset} returned ZERO rows for {TRADE_DATE}. "
                "Do not overwrite the raw partition; proxy/API is still unhealthy."
            )

        path = output_path(dataset)
        atomic_write(df, path)

        # Read-back validation.
        check = pd.read_parquet(path)
        if len(check) != len(df):
            raise RuntimeError(f"Read-back row count mismatch for {path}")

        results[dataset] = len(df)
        print(f"[SAVE] {path} rows={len(df):,}")

    print("=" * 80)
    print("REPAIR COMPLETE")
    for k, v in results.items():
        print(f"{k:12s}: {v:,} rows")

    if results["daily"] <= 0:
        raise RuntimeError("daily repair is unexpectedly empty.")

    print(
        "\nNext:\n"
        "1) Delete only data/processed/full_panel/2013/04/full_panel_201304.parquet\n"
        "2) Re-run: python 04_build_full_panel.py\n"
        "3) Replace Stage-5 script with the corrected version and run it with --force."
    )


if __name__ == "__main__":
    main()
