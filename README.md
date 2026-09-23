# 存款特许权、货币政策与银行风险承担

**Deposit Franchise, Monetary Policy and Risk-taking Behavior of Banks**
Jingjing Yan · University of Oxford · MPhil in Economics · 2026

货币紧缩如何通过银行的负债端竞争优势，影响其对高风险企业的贷款定价？本项目结合美国 2001—2024 年银团贷款、银行监管报表、存款市场和企业财务数据，研究存款特许权与信用风险定价之间的联系。

**核心发现：**在论文首选设定中，存款市场集中度更高的银行在货币紧缩后，对更高风险借款人提高相对贷款利差的幅度更大。低未保险存款暴露银行的估计反应更强；组间差异的证据在 10% 水平显著，在 5% 水平不显著。

[研究摘要与投资分析启示](docs/research_brief.md) · [数据与变量](docs/data_and_methods.md) · [数据处理流程](docs/pipeline.md) · [实证设计](docs/empirical_design.md) · [代码导读](docs/code_guide.md) · [复现说明](docs/reproduction.md)

## 主要结果

| 实证设定 | 三重交互项系数 | 聚类标准误 | 观测数 |
|---|---:|---:|---:|
| 借款人×年份固定效应 | 0.0413 | 0.0127 | 40,702 |
| 借款人×年份 + 银行×年份固定效应（首选） | **0.0500** | **0.0155** | **40,641** |
| 借款人×年份 + 银行×企业固定效应 | 0.0628 | 0.0210 | 37,756 |
| JK 货币政策冲击，首选固定效应 | 0.2825 | 0.1398 | 40,166 |

三重交互项为 **存款 HHI × 滞后货币政策变化 × 借款人违约概率**。HHI 与违约概率标准化，因变量为贷款利差的自然对数；标准误按银行和季度双向聚类。JK 冲击与联邦基金利率变化的尺度不同，不能直接比较系数大小。


在其他条件不变、两家银行 HHI 相差一个标准差的比较中，利率提高 1 个百分点时，借款人违约风险提高一个标准差所对应的额外利差反应约为 **5.1%**；以样本平均利差 262bp 换算，约为 **13.4bp**。这是交互效应的条件化换算，不是所有贷款的平均利率变化。

## 研究工作

- **多源数据处理：**清洗 DealScan、FDIC Call Reports、Summary of Deposits、Compustat 与 CRSP 数据，衔接贷款、借款人、银行和时间标识，检查重复键、匹配覆盖及时间对齐。
- **经济指标构建：**由县域存款份额构造 HHI，再按银行存款分布加权；结合市场和财务数据构造违约风险，整理银行资本、负债结构和贷款合同变量。
- **实证识别：**估计包含完整低阶项的三重交互模型，以借款人×年份、银行×年份或银行×企业固定效应控制需求、银行状况和稳定的银企匹配差异。
- **结果检验：**比较固定效应设定，以高频货币政策冲击做稳健性检验，并检验未保险存款暴露的异质性。

## 与投资分析的联系

项目从银行负债结构出发，将竞争优势、利率敏感性和信用风险连接起来。研究提示，分析银行的存款优势时，需要同时考察其稳定性，以及这些特征如何影响客户融资成本和风险溢价。它展示了从经济机制、数据证据到研究判断的完整过程；样本结论适用于所研究的美国银团贷款市场，不能直接外推为中国银行或个股投资建议。

## 项目结构

```text
code/
  data/          数据清洗、跨库匹配、面板构建和质量检查
  risk/          市场数据衔接、Bharath–Shumway 风险计算及 PiT 导出
  analysis/      最终论文主回归、JK 检验及未保险存款异质性
  reporting/     将回归输出整理为 LaTeX 表格
  check_project.py  无需数据的代码与文件检查
docs/            研究摘要、方法、来源及复现说明
data/README.md   本地数据配置说明
```

## 运行说明

本仓库仅提供代码和研究说明，不上传原始数据、衍生面板、回归数据文件、论文 PDF 或文献附件。上方结果为最终论文报告的汇总发现，不是由无数据环境重新估计得到。

无需数据即可运行代码检查：

```bash
python3 code/check_project.py
```

运行实证估计需要自行准备有权限的数据、上游构造文件，以及 Stata 17 和 `reghdfe`。数据处理代码保留原研究逻辑，并做了路径配置和脚本化整理。完整数据输入、运行顺序和复现边界见[复现说明](docs/reproduction.md)。

## English abstract

This project studies how deposit franchise shapes banks’ pricing of borrower risk during monetary tightening. Using U.S. syndicated loans from 2001–2024, it links loan contracts to bank regulatory reports, local deposit-market concentration, borrower financials and market-based default risk. A triple-interaction specification with high-dimensional fixed effects finds a stronger risk-pricing response among banks with higher deposit HHI. The estimated response is larger for banks with lower uninsured-deposit exposure, although the between-group difference is significant only at the 10% level. The repository contains selected data-processing and analysis code, documentation of the reported findings and a table-rendering utility. No datasets or paper attachments are distributed.
