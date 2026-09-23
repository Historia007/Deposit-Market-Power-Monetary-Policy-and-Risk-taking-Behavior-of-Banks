# 复现说明

## 1. 仓库范围与代码检查

仓库只包含代码及说明。数据、已保存的回归 CSV、论文 PDF 和图表附件均不上传。README 的系数与样本量已与最终论文及本地汇总输出核对。

Python 3.10 或更高版本；从仓库根目录运行：

```bash
python3 code/check_project.py
```

该检查验证 Python 语法、源文件完整性、文档链接和上传文件范围，不运行回归。

| 最终论文表号 | 内容 | 对应程序 |
|---|---|---|
| Table 2 | 主回归三列 | `code/analysis/208_generate_thesis_main_tables.do` |
| Table 3 | FFR 与 JK 比较 | 同上 |
| Table 4 | 未保险存款异质性 | `code/analysis/206_cr_bankyear_unins_split_full_tables.do` |

原分析脚本208在 2026-05-09 的输出对应最终主表与 JK 表；脚本206在 2026-05-06 的中位数分组输出对应异质性表。历史文件名中的 table_4/5/6 对应最终论文的2/3/4。

## 2. 有授权数据的回归估计

需要 Stata 17+、`reghdfe` 及其依赖 `ftools`。在 Stata 中配置数据工作区和输出目录，再运行脚本：

```stata
ssc install ftools, replace
ssc install reghdfe, replace
cd "/path/to/this/repository"
global BASE "/path/to/empirical-data-workspace"
global RESULTS "results/reestimated"
do code/analysis/208_generate_thesis_main_tables.do
do code/analysis/206_cr_bankyear_unins_split_full_tables.do
```

基础输入：

```text
data/final/regression_panel_lender_level_ppml_input.dta
data/final/regression_panel_lender_level_lender_spread_controls.dta
data/final/regression_panel_lender_level_ppml_structure_controls.dta
data/final/regression_panel_extra_bank_controls.dta
data/final/regression_panel_lender_level_ppml_macro_controls.dta
data/intermediate/merged/pd_pit_origination_2001_2025_for_stata.dta
data/final/bank_uninsured_deposits.dta              # 异质性分析
```

文件名中的 `ppml` 是历史输入命名，最终主表实际用 `reghdfe` 估计对数利差，不是 PPML。

## 3. 上游数据处理

```bash
python3 -m pip install -r requirements-analysis.txt
export MPHIL_THESIS_ROOT="/path/to/empirical-data-workspace"
```

- `code/data/01`—`05`：分别清洗监管、存款、贷款、企业数据并连接借款人。
- `06`—`08`：衔接银行数据、检查映射、审核手工链接。
- `09`：构建用于实证分析的面板；`10`、`11`：重复键与流程检查。
- `40`、`56`：生成利差附加数据和 Stata 输入；`178`：构建未保险存款数据。
- `code/risk/09`、`10`、`11`：授权市场数据取得、CCM 衔接及 Bharath–Shumway 风险计算；`14`：导出既有发起日 PiT 数据。

这里的数字保留原研究编号，并非跨目录连续运行序号。需先准备源数据及链接表，按程序输入输出依赖运行。市场数据下载脚本需自行配置 WRDS 凭证，本仓库不附带凭证。

**复现边界：**目前归档没有完整独立重建精确发起日 PiT 风险数据的脚本；部分额外控制变量和手工链接输入也需已有研究工作区。该项目提供核心处理和最终估计代码，不能仅用 GitHub 文件从零重建所有底层数据。本次整理验证的是语法、文档链接、上传范围以及本地保存结果与最终论文的一致性，未重跑全部商业数据清洗和 Stata 回归。

## 4. 整理范围

数据清洗 notebook 导出为 `.py`，移除嵌入的数据展示、编辑指令注释和 Jupyter magic；个人绝对路径改为环境变量配置。最终估计脚本改用显式数据根目录和单独结果目录。原估计公式、样本筛选和风险定义保留。Bharath–Shumway 文件中额外依赖的历史迭代模型比较段已删除，保留季度风险计算和输出。

源码的相对来源及 SHA-256 记录在 `source_manifest.json`。历史探索性设定、导师往来、早期草稿、后续独立数据交付、第三方文献、日志和缓存未纳入本项目。公开版本仅保留本次整理的代码和说明，不继承包含文献附件的旧历史。

## 5. 回归后的表格生成

运行估计后，用实际输出路径生成 LaTeX 表格（将日期替换为运行日期）：

```bash
python3 code/reporting/render_thesis_tables.py \
  --key results/reestimated/spec208_thesis_main_tables_key_YYYY-MM-DD.csv \
  --coefs results/reestimated/spec208_thesis_main_tables_full_coefs_YYYY-MM-DD.csv \
  --unins-key results/reestimated/spec206_cr_bankyear_unins_split_key_results_YYYY-MM-DD.csv \
  --unins-coefs results/reestimated/spec206_cr_bankyear_unins_split_full_coefs_YYYY-MM-DD.csv \
  --outdir results/tables
```

此步骤只需 Python 标准库。生成的表格写入被 Git 忽略的本地 `results/`；不会将结果数据纳入代码仓库。
