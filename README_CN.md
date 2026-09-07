# 中国 A 股横截面 Alpha 研究

> **基于价格–成交量信号、Ridge 正则化、XGBoost 头部选股与可执行样本外回测的完整量化研究项目。**

## 研究问题

**价格–成交量信号能否在样本外预测 A 股横截面未来收益？在考虑真实执行约束和交易成本后，非线性机器学习模型能否相对线性基准提供额外经济价值？**

项目链条：

**数据工程 → 动态股票池 → 信号构造 → 单因子分析 → OLS/Ridge → XGBoost → 可执行回测 → 稳健性检验**

2019–2025 OOS 打开后，研究设计冻结，不再根据 OOS 结果删因子、翻转方向、调 XGBoost 或增加 regime filter。

## 核心结论

- 最强 standalone OOS RankIC：**Range20 = -0.0811**，HAC t = **-8.61**。
- Ridge 的 broad OOS RankIC 最强：**0.0842**。
- XGBoost broad RankIC 为 **0.0610**，但 Q5–Universe 为 **24.94 bps/5-day**，明显高于 Ridge 的 **7.60 bps**。
- 主策略 XGB Top20 @ 5 bps：**10.43% CAGR，0.473 Sharpe**；等权股票池 CAGR 为 **3.64%**。
- Top1000 流动性约束下：XGB Top20 @ 5 bps CAGR **8.89%**。
- 冻结 XGB score 在 1–20 日 horizon 上均有正的 top-tail 预测能力。
- 20 bps 单边成本下策略失效。
- 超额收益主要集中在 UP regime，DOWN regime 平均主动收益为负。

## 最终主策略

- Model：XGBoost
- Portfolio：Top20% long-only equal-weight
- Universe：lagged ADV20 Top1500
- Rebalance：每 5 个市场交易日
- Execution：下一市场开盘
- Primary cost：单边 5 bps

## 主要限制

- 高换手；
- 对交易成本敏感；
- 最大回撤约 -48.5%；
- long-only，市场风险较高；
- XGBoost broad RankIC 低于 Ridge；
- XGB 相对 Ridge 的 executable return 增量统计证据有限；
- alpha 明显依赖上涨市场状态。

## 项目目录

```text
china-a-share-alpha/
├─ README.md
├─ README_CN.md
├─ requirements.txt
├─ .gitignore
├─ REPRODUCIBILITY.md
├─ PROJECT_STRUCTURE.md
├─ src/
├─ docs/
│  ├─ reports/
│  └─ figures/
├─ results/
│  └─ final_tables/
└─ data/
   └─ README.md
```

原始行情、大型 parquet、中间缓存和任何 API 凭证均不上传 GitHub。

## 最终结论

> **简单价格–成交量信号在 A 股中具有稳定样本外横截面预测能力。Ridge 在高共线性下提供最强的整体排序，而 XGBoost 的增量经济价值主要来自非线性头部股票筛选。该优势在真实执行和中等交易成本下仍存在，但伴随高换手、较大回撤和明显的市场状态依赖。**

本项目仅用于研究、学习与求职展示，不构成投资建议。
