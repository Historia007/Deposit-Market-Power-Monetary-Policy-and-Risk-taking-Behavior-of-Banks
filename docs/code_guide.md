# 代码导读

## 建议阅读顺序

先阅读 [研究摘要](research_brief.md)，明确问题和结论；再阅读 [实证设计](empirical_design.md)，理解三重交互和固定效应；随后用 [数据流程](pipeline.md) 对照代码。需要实际运行时，使用 [复现说明](reproduction.md)。

## 数据处理目录

| 文件 | 主要工作 | 阅读重点 |
|---|---|---|
| `01_clean_fdic_call_reports.py` | 监管报表组件清洗、字段合并、银行变量 | 报表口径、重复报送、金额单位 |
| `02_clean_fdic_sod.py` | 县域及银行加权存款 HHI | 地理标识、存款权重、垄断县 |
| `03_clean_dealscan.py` | 贷款及贷款行参与记录清洗 | 观察层级、期限、利差与金额 |
| `04_clean_compustat.py` | 企业财务变量 | 规模、杠杆、盈利与异常值 |
| `05_link_dealscan_compustat.py` | 借款人链接及财务日期匹配 | 来源优先级、置信信息、365天窗口 |
| `06_merge_four_tables.py` | 银行映射及多源合并 | RSSD、链接有效期、后继关系、日期规则 |
| `07_verify_lender_bank_matching.py` | 映射质量检查 | 标识、日期与覆盖的独立检查 |
| `08_manual_linking_audit.py` | 手工链接审计 | 名称一致性和不确定映射 |
| `09_prepare_final_regression_panel.py` | 基础研究面板 | 衍生字段、筛选和贷款层级汇总 |
| `10_investigate_analytical_key_duplicates.py` | 重复键检查 | 一对多结构与意外重复的区分 |
| `11_pipeline_integrity_checks.py` | 流程完整性检查 | 输入输出及关键字段一致性 |
| `40_build_lender_specific_spread_controls.py` | 贷款行利差附加输入 | 严格口径与回退字段；金额相关旧辅助字段不等于最终因变量 |
| `56_refresh_stata_inputs_from_final_panel.py` | 转换 Stata 数据输入 | 字段名、数值类型、宏观与结构附加表 |
| `178_build_balance_table_and_uninsured.py` | 未保险存款变量及银行汇总 | RCON5597与RCON2200的分母口径 |

这些程序位于 `code/data/`。数字保留原研究编号，便于对应来源，不代表编号之间的程序全部被省略为隐含依赖。

## 风险指标目录

`code/risk/` 包含四个文件：授权市场输入下载、Compustat/CRSP/CCM 合并、季度 Bharath–Shumway 风险计算，以及已存在发起日风险输入的导出器。

市场下载器读取自行配置的 WRDS 登录信息；只会在主动运行时联网。仓库不含用户名或密码。季度风险计算与贷款发起日风险的区别及复现缺口已在方法文档明确说明。

## 最终估计目录

`code/analysis/208_generate_thesis_main_tables.do` 是主表入口。按顺序完成输入合并、样本筛选、标识和滞后变量生成、标准化、控制变量整理、固定效应回归和结果导出。关注 `main_sample` 的形成顺序，以及实际使用的 `reghdfe` 命令。

`code/analysis/206_cr_bankyear_unins_split_full_tables.do` 构建未保险存款分组，运行分组回归及 pooled Wald 检验。文件也包含三分位分组的辅助估计；最终论文异质性表采用中位数分组。不能混用 pooled 标准误和 separate-subsample 标准误。

## 表格与项目检查

`code/reporting/render_thesis_tables.py` 读取使用者本地的汇总回归输出，生成三张 LaTeX 表格。它只负责排版，不替代回归，也不重新解释模型。

`code/check_project.py` 无需数据和第三方 Python 库，检查 Python 语法、源码校验值、Markdown 相对链接，以及是否出现不应上传的数据或附件。

## 原始研究逻辑与整理改动

本次整理将选定 notebook 导出为 Python，删除嵌入数据输出及 notebook 专属 magic，统一个人路径配置；最终 Stata 程序的数据与结果目录分离。Bharath–Shumway 文件中与论文主结果无关的历史迭代模型比较段已移除，计算公式保留。没有重新挑选回归设定、修改估计系数或改变样本以追求显著性。

来源表记录于 `source_manifest.json`，保存相对来源路径、原文件 SHA-256 和当前打包文件 SHA-256。较晚整理的数据代码与最终论文估计输出不是同一时间的全量环境快照，故本项目不声称完整、逐位一致的从零复现。
