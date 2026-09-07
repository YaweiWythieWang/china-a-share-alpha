# Recommended Repository Structure

```text
china-a-share-alpha/
│
├─ README.md
├─ README_CN.md
├─ requirements.txt
├─ requirements-lock.txt
├─ .gitignore
├─ REPRODUCIBILITY.md
├─ PROJECT_STRUCTURE.md
│
├─ src/
│  ├─ 01_data/
│  ├─ 02_panel/
│  ├─ 03_signals/
│  ├─ 04_factor_research/
│  ├─ 05_models/
│  ├─ 06_backtest/
│  └─ 07_reporting/
│
├─ docs/
│  ├─ reports/
│  │  ├─ quant_research_report_en.tex
│  │  ├─ quant_research_report_en.pdf
│  │  ├─ quant_research_report_cn.tex
│  │  └─ quant_research_report_cn.pdf
│  └─ figures/
│
├─ results/
│  └─ final_tables/
│
└─ data/
   └─ README.md
```

## Commit

- final research scripts,
- final figures,
- final compact CSV tables,
- reports,
- documentation,
- dependency files.

## Do Not Commit

- Tushare token,
- raw history,
- large parquet panels,
- caches,
- stock-level prediction files,
- temporary logs,
- zip archives.
