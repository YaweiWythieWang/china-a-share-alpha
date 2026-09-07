#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
11_build_final_research_package.py

China A-Share Cross-Sectional Alpha Research
Stage 11: Final Research Package Builder

PURPOSE
-------
Create one canonical set of final tables and figures from the FROZEN Stage 6-10
research outputs. No model is retrained and no OOS result is used to modify the
research design.

The resulting files are intended to be the single source of truth for:
    - final Quant Research report
    - GitHub README
    - resume bullets
    - interview presentation / talking points

INPUTS
------
Stage 6:
    data/single_factor_reports/
        single_factor_summary.csv
        single_factor_yearly.csv

Stage 7:
    data/linear_model_reports/
        linear_model_summary.csv

Stage 8:
    data/xgb_model_reports/
        xgb_model_summary.csv
        xgb_model_yearly.csv
        xgb_feature_importance.csv

Stage 9:
    data/backtest_reports/
        backtest_summary.csv
        backtest_yearly.csv
        backtest_equity_curve.csv

Stage 10:
    data/robustness_reports/
        robustness_liquidity_summary.csv
        robustness_horizon_summary.csv
        robustness_regime_results.csv
        robustness_paired_tests.csv

OUTPUT
------
data/final_package/
    tables/
        final_key_results.csv
        final_model_comparison.csv
        final_cost_sensitivity.csv
        final_liquidity_robustness.csv
        final_horizon_robustness.csv
        final_regime_robustness.csv
        final_yearly_performance.csv
        final_feature_importance.csv

    figures/
        01_single_factor_oos_rankic.png
        02_model_oos_rankic.png
        03_backtest_equity_curve_5bps.png
        04_cost_sensitivity_cagr.png
        05_liquidity_robustness_5bps.png
        06_horizon_robustness_top20_active.png
        07_regime_active_return_5bps.png
        08_yearly_returns_xgb_vs_benchmark_5bps.png

    final_research_summary.json
    11_build_final_research_package.log

RUN
---
python 11_build_final_research_package.py
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import platform
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ======================================================================================
# Paths / config
# ======================================================================================

class Config:
    def __init__(self, data_root: Path):
        self.data_root = data_root

    @property
    def stage6_root(self) -> Path:
        return self.data_root / "single_factor_reports"

    @property
    def stage7_root(self) -> Path:
        return self.data_root / "linear_model_reports"

    @property
    def stage8_root(self) -> Path:
        return self.data_root / "xgb_model_reports"

    @property
    def stage9_root(self) -> Path:
        return self.data_root / "backtest_reports"

    @property
    def stage10_root(self) -> Path:
        return self.data_root / "robustness_reports"

    @property
    def output_root(self) -> Path:
        return self.data_root / "final_package"

    @property
    def table_root(self) -> Path:
        return self.output_root / "tables"

    @property
    def figure_root(self) -> Path:
        return self.output_root / "figures"


def setup_logging(output_root: Path) -> logging.Logger:
    output_root.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("stage11_final_package")
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
        output_root / "11_build_final_research_package.log",
        mode="w",
        encoding="utf-8",
    )
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


def read_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return pd.read_csv(path)


def safe_float(x: Any) -> float:
    try:
        y = float(x)
    except Exception:
        return np.nan
    return y if np.isfinite(y) else np.nan


def save_json(obj: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def pct(x: float) -> float:
    return 100.0 * float(x)


# ======================================================================================
# Load frozen research outputs
# ======================================================================================

def load_inputs(cfg: Config) -> Dict[str, pd.DataFrame]:
    return {
        "sf_summary": read_required(
            cfg.stage6_root / "single_factor_summary.csv"
        ),
        "sf_yearly": read_required(
            cfg.stage6_root / "single_factor_yearly.csv"
        ),
        "linear_summary": read_required(
            cfg.stage7_root / "linear_model_summary.csv"
        ),
        "xgb_summary": read_required(
            cfg.stage8_root / "xgb_model_summary.csv"
        ),
        "xgb_yearly": read_required(
            cfg.stage8_root / "xgb_model_yearly.csv"
        ),
        "xgb_importance": read_required(
            cfg.stage8_root / "xgb_feature_importance.csv"
        ),
        "bt_summary": read_required(
            cfg.stage9_root / "backtest_summary.csv"
        ),
        "bt_yearly": read_required(
            cfg.stage9_root / "backtest_yearly.csv"
        ),
        "bt_equity": read_required(
            cfg.stage9_root / "backtest_equity_curve.csv"
        ),
        "rob_liquidity": read_required(
            cfg.stage10_root / "robustness_liquidity_summary.csv"
        ),
        "rob_horizon": read_required(
            cfg.stage10_root / "robustness_horizon_summary.csv"
        ),
        "rob_regime": read_required(
            cfg.stage10_root / "robustness_regime_results.csv"
        ),
        "rob_paired": read_required(
            cfg.stage10_root / "robustness_paired_tests.csv"
        ),
    }


# ======================================================================================
# Canonical final tables
# ======================================================================================

def build_key_results(d: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    # Strongest single-factor OOS RankIC by absolute value.
    sf = d["sf_summary"].copy()
    sf_oos = sf.loc[sf["period"] == "OOS_2019_2025"].copy()
    sf_oos["abs_rank_ic"] = pd.to_numeric(
        sf_oos["mean_rank_ic"], errors="coerce"
    ).abs()
    sf_oos = sf_oos.sort_values("abs_rank_ic", ascending=False)

    if not sf_oos.empty:
        r = sf_oos.iloc[0]
        rows.append({
            "category": "Single factor",
            "metric": "Strongest OOS RankIC",
            "specification": str(r["factor"]),
            "value": safe_float(r["mean_rank_ic"]),
            "unit": "correlation",
            "secondary_metric": "HAC t",
            "secondary_value": safe_float(r["rank_ic_hac_t"]),
        })

    # Stage-7 monthly Ridge OOS.
    lin = d["linear_summary"]
    r = lin.loc[
        (lin["model"] == "RIDGE")
        & (lin["period"] == "OOS_2019_2025")
    ]
    if len(r) == 1:
        r = r.iloc[0]
        rows.append({
            "category": "Linear model",
            "metric": "Ridge OOS RankIC",
            "specification": "Monthly expanding Ridge",
            "value": safe_float(r["mean_rank_ic"]),
            "unit": "correlation",
            "secondary_metric": "HAC t",
            "secondary_value": safe_float(r["rank_ic_hac_t"]),
        })

    # Stage-8 XGB OOS.
    xs = d["xgb_summary"]
    r = xs.loc[
        (xs["model"] == "XGBOOST_ANNUAL")
        & (xs["period"] == "OOS_2019_2025")
    ]
    if len(r) == 1:
        r = r.iloc[0]
        rows.append({
            "category": "Nonlinear model",
            "metric": "XGBoost OOS RankIC",
            "specification": "Annual expanding XGBoost",
            "value": safe_float(r["mean_rank_ic"]),
            "unit": "correlation",
            "secondary_metric": "Q5-Universe bps/5d",
            "secondary_value": safe_float(
                r["mean_q5_minus_universe_ret_5d_bps"]
            ),
        })

    # Stage-9 primary executable strategy at 5 bps.
    bt = d["bt_summary"]
    for strategy in ["XGB_TOP20", "RIDGE_TOP20", "UNIVERSE_EW"]:
        r = bt.loc[
            (bt["strategy"] == strategy)
            & (pd.to_numeric(bt["cost_bps"], errors="coerce") == 5.0)
        ]
        if len(r) == 1:
            r = r.iloc[0]
            rows.append({
                "category": "Executable backtest",
                "metric": "CAGR @ 5 bps",
                "specification": strategy,
                "value": pct(safe_float(r["cagr"])),
                "unit": "%",
                "secondary_metric": "Sharpe",
                "secondary_value": safe_float(r["sharpe_zero_rf"]),
            })

    # Stage-10 Top1000 primary robustness.
    rl = d["rob_liquidity"]
    r = rl.loc[
        (rl["liquidity_universe"] == "TOP1000")
        & (rl["strategy"] == "XGB_TOP20")
        & (pd.to_numeric(rl["cost_bps"], errors="coerce") == 5.0)
    ]
    if len(r) == 1:
        r = r.iloc[0]
        rows.append({
            "category": "Liquidity robustness",
            "metric": "CAGR @ 5 bps",
            "specification": "XGB_TOP20 / TOP1000",
            "value": pct(safe_float(r["cagr"])),
            "unit": "%",
            "secondary_metric": "Sharpe",
            "secondary_value": safe_float(r["sharpe_zero_rf"]),
        })

    # Horizon persistence.
    rh = d["rob_horizon"]
    for h in [1, 5, 10, 20]:
        r = rh.loc[
            (rh["model"] == "XGB")
            & (pd.to_numeric(rh["horizon_days"], errors="coerce") == h)
        ]
        if len(r) == 1:
            r = r.iloc[0]
            rows.append({
                "category": "Horizon robustness",
                "metric": f"Top20-Universe @ {h}d",
                "specification": "Frozen XGB score",
                "value": safe_float(
                    r["mean_top20_minus_universe_bps"]
                ),
                "unit": "bps",
                "secondary_metric": "HAC t",
                "secondary_value": safe_float(
                    r["top20_minus_universe_hac_t"]
                ),
            })

    return pd.DataFrame(rows)


def build_model_comparison(d: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []

    lin = d["linear_summary"]
    xgb = d["xgb_summary"]

    for model, source in [
        ("OLS", lin),
        ("RIDGE", lin),
        ("XGBOOST_ANNUAL", xgb),
    ]:
        r = source.loc[
            (source["model"] == model)
            & (source["period"] == "OOS_2019_2025")
        ]
        if len(r) != 1:
            continue

        r = r.iloc[0]

        rows.append({
            "model": model,
            "mean_ic": safe_float(r["mean_ic"]),
            "ic_hac_t": safe_float(r["ic_hac_t"]),
            "mean_rank_ic": safe_float(r["mean_rank_ic"]),
            "rank_ic_hac_t": safe_float(r["rank_ic_hac_t"]),
            "q5_q1_bps": safe_float(r["mean_q5_q1_ret_5d_bps"]),
            "q5_q1_hac_t": safe_float(r["q5_q1_hac_t"]),
            "q5_minus_universe_bps": safe_float(
                r["mean_q5_minus_universe_ret_5d_bps"]
            ),
            "q5_minus_universe_hac_t": safe_float(
                r["q5_minus_universe_hac_t"]
            ),
        })

    return pd.DataFrame(rows)


def build_cost_sensitivity(d: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    bt = d["bt_summary"].copy()

    keep = bt.loc[
        bt["strategy"].isin(
            ["XGB_TOP20", "RIDGE_TOP20", "OLS_TOP20", "UNIVERSE_EW"]
        ),
        [
            "strategy",
            "cost_bps",
            "cagr",
            "sharpe_zero_rf",
            "max_drawdown",
            "mean_one_way_turnover",
        ],
    ].copy()

    keep["cagr_pct"] = 100.0 * pd.to_numeric(
        keep["cagr"], errors="coerce"
    )
    keep["max_drawdown_pct"] = 100.0 * pd.to_numeric(
        keep["max_drawdown"], errors="coerce"
    )
    keep["gross_traded_notional_ratio"] = pd.to_numeric(
        keep["mean_one_way_turnover"], errors="coerce"
    )
    keep["approx_one_way_turnover"] = (
        keep["gross_traded_notional_ratio"] / 2.0
    )

    return keep


def build_liquidity_table(d: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rl = d["rob_liquidity"].copy()

    out = rl.loc[
        rl["strategy"].isin(["XGB_TOP20", "RIDGE_TOP20", "UNIVERSE_EW"]),
        [
            "liquidity_universe",
            "strategy",
            "cost_bps",
            "cagr",
            "sharpe_zero_rf",
            "max_drawdown",
            "mean_one_way_turnover",
        ],
    ].copy()

    out["cagr_pct"] = 100.0 * pd.to_numeric(
        out["cagr"], errors="coerce"
    )
    out["max_drawdown_pct"] = 100.0 * pd.to_numeric(
        out["max_drawdown"], errors="coerce"
    )
    out["gross_traded_notional_ratio"] = pd.to_numeric(
        out["mean_one_way_turnover"], errors="coerce"
    )
    out["approx_one_way_turnover"] = (
        out["gross_traded_notional_ratio"] / 2.0
    )

    return out


def build_horizon_table(d: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rh = d["rob_horizon"].copy()

    cols = [
        "model",
        "horizon_days",
        "mean_ic",
        "ic_hac_t",
        "mean_rank_ic",
        "rank_ic_hac_t",
        "mean_q5_q1_bps",
        "q5_q1_hac_t",
        "mean_top20_minus_universe_bps",
        "top20_minus_universe_hac_t",
        "mean_top10_minus_universe_bps",
        "top10_minus_universe_hac_t",
    ]

    return rh[cols].copy()


def build_regime_table(d: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rr = d["rob_regime"].copy()

    out = rr.loc[
        (rr["liquidity_universe"] == "TOP1500")
        & (rr["strategy"] == "XGB_TOP20")
        & (pd.to_numeric(rr["cost_bps"], errors="coerce") == 5.0),
        [
            "regime",
            "n_periods",
            "mean_strategy_return",
            "mean_benchmark_return",
            "mean_active_return_bps",
            "active_hac_t",
            "active_positive_share",
        ],
    ].copy()

    out["mean_strategy_return_pct"] = (
        100.0 * pd.to_numeric(
            out["mean_strategy_return"], errors="coerce"
        )
    )
    out["mean_benchmark_return_pct"] = (
        100.0 * pd.to_numeric(
            out["mean_benchmark_return"], errors="coerce"
        )
    )

    return out


def build_yearly_table(d: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    by = d["bt_yearly"].copy()

    out = by.loc[
        by["strategy"].isin(["XGB_TOP20", "RIDGE_TOP20", "UNIVERSE_EW"])
        & (pd.to_numeric(by["cost_bps"], errors="coerce") == 5.0),
        [
            "strategy",
            "year",
            "year_return",
            "year_sharpe_zero_rf",
        ],
    ].copy()

    out["year_return_pct"] = (
        100.0 * pd.to_numeric(
            out["year_return"], errors="coerce"
        )
    )

    return out


def build_importance_table(d: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    imp = d["xgb_importance"].copy()

    oos = imp.loc[
        pd.to_numeric(
            imp["refit_year"], errors="coerce"
        ).between(2019, 2025)
    ]

    out = (
        oos.groupby("factor", as_index=False)
        .agg(
            mean_gain_share=("gain_share", "mean"),
            sd_gain_share=("gain_share", "std"),
            mean_split_count=("split_count", "mean"),
        )
        .sort_values("mean_gain_share", ascending=False)
        .reset_index(drop=True)
    )

    out["mean_gain_share_pct"] = (
        100.0 * out["mean_gain_share"]
    )

    return out


# ======================================================================================
# Figures
# ======================================================================================

def save_current_figure(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()


def fig_single_factor_rankic(
    d: Dict[str, pd.DataFrame],
    path: Path,
) -> None:
    x = d["sf_summary"].copy()
    x = x.loc[
        x["period"] == "OOS_2019_2025"
    ].copy()
    x["mean_rank_ic"] = pd.to_numeric(
        x["mean_rank_ic"], errors="coerce"
    )
    x = x.sort_values("mean_rank_ic")

    plt.figure(figsize=(9, 6))
    plt.barh(x["factor"], x["mean_rank_ic"])
    plt.axvline(0.0, linewidth=1)
    plt.xlabel("Mean OOS RankIC")
    plt.ylabel("Factor")
    plt.title("Single-Factor OOS RankIC (2019–2025)")
    save_current_figure(path)


def fig_model_rankic(
    model_table: pd.DataFrame,
    path: Path,
) -> None:
    x = model_table.copy()
    names = x["model"].replace(
        {
            "XGBOOST_ANNUAL": "XGBoost",
            "RIDGE": "Ridge",
            "OLS": "OLS",
        }
    )

    plt.figure(figsize=(7, 5))
    plt.bar(names, x["mean_rank_ic"])
    plt.ylabel("Mean OOS RankIC")
    plt.title("OOS Cross-Sectional Ranking: Linear vs Nonlinear Models")
    save_current_figure(path)


def fig_equity_curve(
    d: Dict[str, pd.DataFrame],
    path: Path,
) -> None:
    x = d["bt_equity"].copy()
    x = x.loc[
        x["strategy"].isin(
            ["XGB_TOP20", "RIDGE_TOP20", "OLS_TOP20", "UNIVERSE_EW"]
        )
        & (pd.to_numeric(x["cost_bps"], errors="coerce") == 5.0)
    ].copy()

    x["date"] = pd.to_datetime(
        x["date"].astype(str),
        format="%Y%m%d",
        errors="coerce",
    )

    plt.figure(figsize=(10, 6))

    for strategy, g in x.groupby("strategy", sort=False):
        g = g.sort_values("date")
        plt.plot(g["date"], g["nav"], label=strategy)

    plt.ylabel("Cumulative NAV")
    plt.xlabel("Date")
    plt.title("Executable OOS Equity Curves — 5 bps One-Way Cost")
    plt.legend()
    save_current_figure(path)


def fig_cost_sensitivity(
    cost_table: pd.DataFrame,
    path: Path,
) -> None:
    plt.figure(figsize=(8, 5))

    for strategy, g in cost_table.groupby(
        "strategy",
        sort=False,
    ):
        g = g.sort_values("cost_bps")
        plt.plot(
            g["cost_bps"],
            g["cagr_pct"],
            marker="o",
            label=strategy,
        )

    plt.axhline(0.0, linewidth=1)
    plt.xlabel("One-Way Transaction Cost (bps)")
    plt.ylabel("CAGR (%)")
    plt.title("Transaction-Cost Sensitivity")
    plt.legend()
    save_current_figure(path)


def fig_liquidity_robustness(
    liquidity_table: pd.DataFrame,
    path: Path,
) -> None:
    x = liquidity_table.loc[
        (pd.to_numeric(
            liquidity_table["cost_bps"],
            errors="coerce",
        ) == 5.0)
        & liquidity_table["strategy"].isin(
            ["XGB_TOP20", "RIDGE_TOP20", "UNIVERSE_EW"]
        )
    ].copy()

    labels = [
        f"{u}\n{s}"
        for u, s in zip(
            x["liquidity_universe"],
            x["strategy"],
        )
    ]

    plt.figure(figsize=(10, 5))
    plt.bar(labels, x["cagr_pct"])
    plt.axhline(0.0, linewidth=1)
    plt.ylabel("CAGR (%)")
    plt.title("Liquidity-Universe Robustness — 5 bps")
    plt.xticks(rotation=30, ha="right")
    save_current_figure(path)


def fig_horizon_robustness(
    horizon_table: pd.DataFrame,
    path: Path,
) -> None:
    x = horizon_table.loc[
        horizon_table["model"] == "XGB"
    ].copy()
    x = x.sort_values("horizon_days")

    plt.figure(figsize=(8, 5))
    plt.plot(
        x["horizon_days"],
        x["mean_top20_minus_universe_bps"],
        marker="o",
    )
    plt.axhline(0.0, linewidth=1)
    plt.xlabel("Forward Horizon (Market Days)")
    plt.ylabel("Top20 − Universe (bps)")
    plt.title("Frozen XGBoost Score: Horizon Persistence")
    save_current_figure(path)


def fig_regime(
    regime_table: pd.DataFrame,
    path: Path,
) -> None:
    order = ["DOWN", "NEUTRAL", "UP"]
    x = (
        regime_table.set_index("regime")
        .reindex(order)
        .reset_index()
    )

    plt.figure(figsize=(7, 5))
    plt.bar(
        x["regime"],
        x["mean_active_return_bps"],
    )
    plt.axhline(0.0, linewidth=1)
    plt.ylabel("Mean Active Return (bps / 5-day period)")
    plt.title("XGB Top20 Active Return by Market Regime — 5 bps")
    save_current_figure(path)


def fig_yearly(
    yearly_table: pd.DataFrame,
    path: Path,
) -> None:
    pivot = yearly_table.pivot(
        index="year",
        columns="strategy",
        values="year_return_pct",
    ).sort_index()

    plt.figure(figsize=(10, 6))

    for strategy in [
        "XGB_TOP20",
        "RIDGE_TOP20",
        "UNIVERSE_EW",
    ]:
        if strategy in pivot.columns:
            plt.plot(
                pivot.index,
                pivot[strategy],
                marker="o",
                label=strategy,
            )

    plt.axhline(0.0, linewidth=1)
    plt.xlabel("Year")
    plt.ylabel("Annual Return (%)")
    plt.title("OOS Year-by-Year Returns — 5 bps")
    plt.legend()
    save_current_figure(path)


# ======================================================================================
# Final frozen-research summary
# ======================================================================================

def build_research_summary(
    key_results: pd.DataFrame,
    model_table: pd.DataFrame,
    cost_table: pd.DataFrame,
    liquidity_table: pd.DataFrame,
    horizon_table: pd.DataFrame,
    regime_table: pd.DataFrame,
) -> Dict[str, Any]:
    def one(df: pd.DataFrame, mask) -> Optional[pd.Series]:
        x = df.loc[mask]
        return x.iloc[0] if len(x) == 1 else None

    result: Dict[str, Any] = {
        "project": "China A-Share Cross-Sectional Alpha Research",
        "research_status": "FROZEN_AFTER_STAGE10",
        "research_question": (
            "Can price-volume signals predict cross-sectional A-share returns "
            "out of sample, and do nonlinear ML models provide incremental "
            "economic value over linear baselines after transaction costs?"
        ),
        "final_primary_specification": {
            "model": "XGBoost",
            "portfolio": "Top 20% long-only equal weight",
            "liquidity_universe": "ADV20 Top1500",
            "rebalance": "Every 5 market days",
            "execution": "Next market open",
            "primary_cost_case_bps_one_way": 5,
        },
        "main_findings": [],
        "limitations": [
            "High turnover makes performance materially transaction-cost sensitive.",
            "The strategy does not survive the 20 bps one-way cost stress test.",
            "XGBoost active returns are strongly regime-dependent and concentrated in UP periods.",
            "XGBoost has lower broad OOS RankIC than Ridge despite stronger top-tail economics.",
            "XGBoost-minus-Ridge executable return differences are economically positive but not strongly statistically significant.",
            "The strategy remains long-only and retains substantial market risk and drawdown exposure.",
        ],
    }

    xgb = one(
        model_table,
        model_table["model"] == "XGBOOST_ANNUAL",
    )
    ridge = one(
        model_table,
        model_table["model"] == "RIDGE",
    )

    if xgb is not None and ridge is not None:
        result["main_findings"].append({
            "finding": (
                "Ridge delivers stronger broad cross-sectional ranking, "
                "while XGBoost concentrates more predictive value in the top tail."
            ),
            "xgb_oos_rank_ic": safe_float(xgb["mean_rank_ic"]),
            "ridge_oos_rank_ic": safe_float(ridge["mean_rank_ic"]),
            "xgb_q5_minus_universe_bps": safe_float(
                xgb["q5_minus_universe_bps"]
            ),
            "ridge_q5_minus_universe_bps": safe_float(
                ridge["q5_minus_universe_bps"]
            ),
        })

    primary = one(
        cost_table,
        (cost_table["strategy"] == "XGB_TOP20")
        & (pd.to_numeric(
            cost_table["cost_bps"], errors="coerce"
        ) == 5.0),
    )

    benchmark = one(
        cost_table,
        (cost_table["strategy"] == "UNIVERSE_EW")
        & (pd.to_numeric(
            cost_table["cost_bps"], errors="coerce"
        ) == 5.0),
    )

    if primary is not None and benchmark is not None:
        result["main_findings"].append({
            "finding": (
                "The executable XGBoost Top20 portfolio outperforms the "
                "same-universe equal-weight benchmark at 5 bps one-way cost."
            ),
            "xgb_cagr_pct": safe_float(primary["cagr_pct"]),
            "benchmark_cagr_pct": safe_float(benchmark["cagr_pct"]),
            "xgb_sharpe": safe_float(primary["sharpe_zero_rf"]),
            "xgb_max_drawdown_pct": safe_float(
                primary["max_drawdown_pct"]
            ),
        })

    top1000 = one(
        liquidity_table,
        (liquidity_table["liquidity_universe"] == "TOP1000")
        & (liquidity_table["strategy"] == "XGB_TOP20")
        & (pd.to_numeric(
            liquidity_table["cost_bps"], errors="coerce"
        ) == 5.0),
    )

    if top1000 is not None:
        result["main_findings"].append({
            "finding": (
                "The main result survives the stricter ADV20 Top1000 liquidity universe."
            ),
            "top1000_xgb_cagr_pct": safe_float(top1000["cagr_pct"]),
            "top1000_xgb_sharpe": safe_float(
                top1000["sharpe_zero_rf"]
            ),
        })

    xgb_h = horizon_table.loc[
        horizon_table["model"] == "XGB"
    ].sort_values("horizon_days")

    if not xgb_h.empty:
        result["main_findings"].append({
            "finding": (
                "Frozen XGBoost scores retain positive top-tail predictive value "
                "from 1 to 20 market days, with the effect strongest per day at "
                "short horizons and decaying gradually."
            ),
            "horizon_top20_minus_universe_bps": {
                str(int(r.horizon_days)):
                safe_float(r.mean_top20_minus_universe_bps)
                for r in xgb_h.itertuples(index=False)
            },
        })

    if not regime_table.empty:
        reg = {
            str(r.regime): safe_float(r.mean_active_return_bps)
            for r in regime_table.itertuples(index=False)
        }

        result["main_findings"].append({
            "finding": (
                "Active performance is strongly regime dependent: strongest in "
                "UP periods, weak in neutral periods, and negative in DOWN periods."
            ),
            "active_return_bps_by_regime": reg,
        })

    return result


# ======================================================================================
# QA
# ======================================================================================

def qa_outputs(
    model_table: pd.DataFrame,
    cost_table: pd.DataFrame,
    liquidity_table: pd.DataFrame,
    horizon_table: pd.DataFrame,
    regime_table: pd.DataFrame,
) -> List[str]:
    issues: List[str] = []

    if set(model_table["model"]) != {
        "OLS",
        "RIDGE",
        "XGBOOST_ANNUAL",
    }:
        issues.append(
            f"Unexpected model comparison set: {model_table['model'].tolist()}"
        )

    expected_costs = {0.0, 5.0, 10.0, 20.0}
    found_costs = set(
        pd.to_numeric(
            cost_table["cost_bps"], errors="coerce"
        ).dropna().astype(float)
    )
    if found_costs != expected_costs:
        issues.append(
            f"Cost grid mismatch: {sorted(found_costs)}"
        )

    if set(
        liquidity_table["liquidity_universe"].dropna().unique()
    ) != {"TOP1500", "TOP1000"}:
        issues.append(
            "Liquidity table must contain TOP1500 and TOP1000."
        )

    found_h = set(
        pd.to_numeric(
            horizon_table["horizon_days"], errors="coerce"
        ).dropna().astype(int)
    )
    if found_h != {1, 5, 10, 20}:
        issues.append(
            f"Horizon grid mismatch: {sorted(found_h)}"
        )

    found_regimes = set(
        regime_table["regime"].dropna().astype(str)
    )
    if found_regimes != {"DOWN", "NEUTRAL", "UP"}:
        issues.append(
            f"Regime set mismatch: {sorted(found_regimes)}"
        )

    return issues


# ======================================================================================
# Main
# ======================================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build canonical final tables and figures for the frozen quant research project."
    )
    p.add_argument(
        "--data-root",
        default="data",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = Config(Path(args.data_root))

    cfg.table_root.mkdir(parents=True, exist_ok=True)
    cfg.figure_root.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(cfg.output_root)

    logger.info("=" * 100)
    logger.info("STAGE 11 | FINAL RESEARCH PACKAGE BUILDER")
    logger.info("Frozen research only: no model fitting or parameter changes.")
    logger.info("=" * 100)

    try:
        d = load_inputs(cfg)

        key_results = build_key_results(d)
        model_table = build_model_comparison(d)
        cost_table = build_cost_sensitivity(d)
        liquidity_table = build_liquidity_table(d)
        horizon_table = build_horizon_table(d)
        regime_table = build_regime_table(d)
        yearly_table = build_yearly_table(d)
        importance_table = build_importance_table(d)

        issues = qa_outputs(
            model_table,
            cost_table,
            liquidity_table,
            horizon_table,
            regime_table,
        )

        tables = {
            "final_key_results.csv": key_results,
            "final_model_comparison.csv": model_table,
            "final_cost_sensitivity.csv": cost_table,
            "final_liquidity_robustness.csv": liquidity_table,
            "final_horizon_robustness.csv": horizon_table,
            "final_regime_robustness.csv": regime_table,
            "final_yearly_performance.csv": yearly_table,
            "final_feature_importance.csv": importance_table,
        }

        for name, df in tables.items():
            path = cfg.table_root / name
            df.to_csv(
                path,
                index=False,
                encoding="utf-8-sig",
            )
            logger.info(
                "TABLE | %s | rows=%d",
                path,
                len(df),
            )

        fig_single_factor_rankic(
            d,
            cfg.figure_root / "01_single_factor_oos_rankic.png",
        )
        fig_model_rankic(
            model_table,
            cfg.figure_root / "02_model_oos_rankic.png",
        )
        fig_equity_curve(
            d,
            cfg.figure_root / "03_backtest_equity_curve_5bps.png",
        )
        fig_cost_sensitivity(
            cost_table,
            cfg.figure_root / "04_cost_sensitivity_cagr.png",
        )
        fig_liquidity_robustness(
            liquidity_table,
            cfg.figure_root / "05_liquidity_robustness_5bps.png",
        )
        fig_horizon_robustness(
            horizon_table,
            cfg.figure_root / "06_horizon_robustness_top20_active.png",
        )
        fig_regime(
            regime_table,
            cfg.figure_root / "07_regime_active_return_5bps.png",
        )
        fig_yearly(
            yearly_table,
            cfg.figure_root / "08_yearly_returns_xgb_vs_benchmark_5bps.png",
        )

        summary = build_research_summary(
            key_results,
            model_table,
            cost_table,
            liquidity_table,
            horizon_table,
            regime_table,
        )

        summary["qa_issue_count"] = len(issues)
        summary["qa_issues"] = issues
        summary["python_version"] = platform.python_version()
        summary["pandas_version"] = pd.__version__
        summary["numpy_version"] = np.__version__

        save_json(
            summary,
            cfg.output_root / "final_research_summary.json",
        )

        logger.info("FIGURES | 8 files written to %s", cfg.figure_root)

        if issues:
            logger.error(
                "STAGE 11 QA FAIL | %d issue(s)",
                len(issues),
            )
            for issue in issues:
                logger.error("QA | %s", issue)
            return 1

        logger.info("STAGE 11 QA: PASS")
        logger.info("=" * 100)
        return 0

    except KeyboardInterrupt:
        logger.warning("Interrupted by user.")
        return 130

    except Exception:
        logger.exception(
            "Fatal error during Stage-11 final package generation."
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
