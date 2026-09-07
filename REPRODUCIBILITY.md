# Reproducibility Guide

## Environment

The project was developed on Windows using Anaconda Prompt.

```cmd
conda create -n a_share_alpha python=3.13
conda activate a_share_alpha
pip install -r requirements.txt
```

On the original research machine, save an exact environment snapshot:

```cmd
pip freeze > requirements-lock.txt
```

Do **not** commit API tokens.

Use environment variables:

```cmd
set TUSHARE_TOKEN=YOUR_TOKEN
set TUSHARE_HTTP_URL=https://t.xiaodefa.top/
```

## Data

Raw and intermediate market data are excluded because of file size and provider terms.

Expected local directories:

```text
data/
├─ raw/
├─ processed/
├─ cache/
├─ single_factor_reports/
├─ linear_model_reports/
├─ xgb_model_reports/
├─ backtest_reports/
├─ robustness_reports/
└─ final_package/
```

## Pipeline

Run the final scripts in numerical order:

```text
full-history download / QA
→ full panel
→ Stage-5 v4 signals and dynamic universe
→ Stage 6 single-factor research
→ Stage 7 OLS / Ridge
→ Stage 8 XGBoost
→ Stage 9 executable backtest
→ Stage 10 robustness
→ Stage 11 final tables and figures
```

## Post-OOS Freeze

After opening the 2019–2025 OOS sample:

- no factor is dropped,
- no raw signal is sign-flipped,
- Ridge is not retuned on OOS,
- XGBoost hyperparameters are not expanded using OOS,
- the Top20 primary portfolio is not changed,
- DOWN-regime weakness is not used to add a market-timing filter.

## Before Publishing

Check:

```cmd
git status
git diff --cached
git ls-files
```

Verify that:

- no token appears in any staged file,
- no parquet/raw data are staged,
- README images render,
- final numbers match Stage 11,
- limitations remain visible.
