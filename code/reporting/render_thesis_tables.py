#!/usr/bin/env python3
"""Render thesis-facing LaTeX tables from script-208 and script-207 artifacts."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


DISPLAY_TEX = {
    "HHI x Delta FFR x Borrower PD": r"$HHI_{i,t}\times \Delta FFR_{t-1}\times PD_{b,t}$",
    "HHI x Delta FFR": r"$HHI_{i,t}\times \Delta FFR_{t-1}$",
    "HHI x Borrower PD": r"$HHI_{i,t}\times PD_{b,t}$",
    "Delta FFR x Borrower PD": r"$\Delta FFR_{t-1}\times PD_{b,t}$",
    "HHI x MP x Borrower PD": r"$HHI_{i,t}\times MP_{t-1}\times PD_{b,t}$",
    "HHI x MP": r"$HHI_{i,t}\times MP_{t-1}$",
    "MP x Borrower PD": r"$MP_{t-1}\times PD_{b,t}$",
    "HHI (z)": r"$HHI_{i,t}$ (z)",
    "Delta FFR, lagged": r"$\Delta FFR_{t-1}$",
    "MP, lagged": r"$MP_{t-1}$",
    "Borrower PD (Naive BS, z)": r"$PD_{b,t}$ (Bharath--Shumway, z)",
    "Log maturity": r"Log maturity",
    "Credit line": r"Credit line",
    "Log number of lenders": r"Log number of lenders",
    "Secured": r"Secured",
    "Corporate/general purpose": r"Corporate/general purpose",
    "Debt repayment/refinance": r"Debt repayment/refinance",
    "Takeover/acquisition": r"Takeover/acquisition",
    "Log bank assets": r"Log bank assets",
    "Bank capital ratio": r"Bank capital ratio",
    "Deposits/assets": r"Deposits/assets",
    "Loans/assets": r"Loans/assets",
    "Log borrower sales/assets": r"Log borrower sales/assets",
    "Borrower tangibility": r"Borrower tangibility",
    "EBITDA margin": r"EBITDA margin",
    "Borrower leverage": r"Borrower leverage",
    "CPI, lagged": r"CPI, lagged",
    "GDP growth, lagged": r"GDP growth, lagged",
    "HHI x Delta FFR x PD": r"$HHI_{i,t}\times \Delta FFR_{t-1}\times PD_{b,t}$",
    "HHI x PD": r"$HHI_{i,t}\times PD_{b,t}$",
    "Delta FFR x PD": r"$\Delta FFR_{t-1}\times PD_{b,t}$",
    "Borrower PD (z)": r"$PD_{b,t}$ (Bharath--Shumway, z)",
}

def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def fnum(value: str | float | None, digits: int = 4) -> str:
    if value is None or value == "":
        return ""
    return f"{float(value):.{digits}f}"


def fp(value: str | float | None) -> str:
    if value is None or value == "":
        return ""
    val = float(value)
    if val < 0.001:
        return r"$<0.001$"
    return f"{val:.3f}"


def fint(value: str | float | None) -> str:
    if value is None or value == "":
        return ""
    return f"{int(round(float(value))):,}".replace(",", r"{,}")


def stars(pvalue: str | float | None) -> str:
    if pvalue is None or pvalue == "":
        return ""
    p = float(pvalue)
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return ""


def coef(row: dict[str, str] | None) -> str:
    if not row or row.get("omitted") == "1":
        return "--"
    return f"{fnum(row['coef'])}{stars(row['pval'])}"


def se(row: dict[str, str] | None) -> str:
    if not row or row.get("omitted") == "1":
        return ""
    return f"[{fnum(row['se'])}]"


def key_coef(row: dict[str, str] | None) -> str:
    if not row:
        return "--"
    return f"{fnum(row['coef'])}{stars(row['pval'])}"


def key_se(row: dict[str, str] | None) -> str:
    if not row:
        return ""
    return f"[{fnum(row['se'])}]"


def write(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def render_main_table(coefs: list[dict[str, str]], key: list[dict[str, str]], out: Path) -> None:
    rows = {
        (int(r["col_id"]), int(r["row_order"])): r
        for r in coefs
        if r["table"] == "main"
    }
    key_rows = {int(r["col_id"]): r for r in key if r["table"] == "main"}

    lines = [
        r"\begin{table}[!p]",
        r"\centering",
        r"\caption{Main Results: Deposit Market Power, Monetary Policy, and Loan Spreads}",
        r"\label{tab:table_4_main_results}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{5.0pt}",
        r"\setlength{\aboverulesep}{0.35ex}",
        r"\setlength{\belowrulesep}{0.35ex}",
        r"\renewcommand{\arraystretch}{0.70}",
        r"\begin{tabular}{>{\raggedright\arraybackslash}p{6.65cm}ccc}",
        r"\toprule",
        r"Dependent variable: $\ln$(all-in drawn spread) & (1) & (2) & (3) \\",
        r"\midrule",
    ]

    row_orders = sorted(order for col, order in rows if col == 1)
    for order in row_orders:
        label = DISPLAY_TEX.get(rows[(1, order)]["display"], rows[(1, order)]["display"])
        model_rows = [rows.get((col, order)) for col in range(1, 4)]
        lines.append(f"{label} & " + " & ".join(coef(r) for r in model_rows) + r" \\")
        lines.append(" & " + " & ".join(se(r) for r in model_rows) + r" \\")
        if order in {7, 14, 18, 22}:
            lines.append(r"\addlinespace")

    lines.extend(
        [
            r"\midrule",
            r"Borrower-year fixed effects & Yes & Yes & Yes \\",
            r"Bank-year fixed effects & No & Yes & No \\",
            r"Bank-firm fixed effects & No & No & Yes \\",
            r"\midrule",
            "Observations & " + " & ".join(fint(key_rows[col]["N"]) for col in range(1, 4)) + r" \\",
            "Adjusted $R^2$ & " + " & ".join(fnum(key_rows[col]["r2_adj"], 3) for col in range(1, 4)) + r" \\",
            "Within $R^2$ & " + " & ".join(fnum(key_rows[col]["r2_within"], 3) for col in range(1, 4)) + r" \\",
            r"\bottomrule",
            r"\end{tabular}",
            r"\vspace{0.5ex}",
            r"\begin{minipage}{\textwidth}",
            r"\footnotesize",
            r"\textit{Notes:} Standard errors in square brackets are two-way clustered by bank and quarter. "
            r"$^{***}p<0.01$, $^{**}p<0.05$, $^{*}p<0.10$. All columns are estimated with \texttt{reghdfe}. "
            r"The monetary-policy variable is the lagged "
            r"quarterly change in the federal funds rate. HHI and borrower PD are standardized.",
            r"\par\smallskip",
            r"Borrower PD is the probability of default computed following Bharath and Shumway (2008); "
            r"BS in the source variable name refers to Bharath--Shumway, not Black--Scholes.",
            r"\end{minipage}",
            r"\end{table}",
        ]
    )
    write(out, lines)


def render_jk_table(coefs: list[dict[str, str]], key: list[dict[str, str]], out: Path) -> None:
    main_row = next(
        r for r in key
        if r["table"] == "main" and r["fe_label"] == "Borrower-year + bank-year"
    )
    jk_row = next(
        r for r in key
        if r["table"] == "jk" and r["fe_label"] == "Borrower-year + bank-year"
    )
    main_rows = {
        int(r["row_order"]): r
        for r in coefs
        if r["table"] == "main" and int(r["col_id"]) == 2
    }
    jk_rows = {
        int(r["row_order"]): r
        for r in coefs
        if r["table"] == "jk" and int(r["col_id"]) == 1
    }
    lines = [
        r"\begin{table}[!p]",
        r"\centering",
        r"\caption{Robustness: Jarocinski--Karadi Monetary-Policy Shock}",
        r"\label{tab:table_5_jk_robustness}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{5.0pt}",
        r"\setlength{\aboverulesep}{0.35ex}",
        r"\setlength{\belowrulesep}{0.35ex}",
        r"\renewcommand{\arraystretch}{0.70}",
        r"\begin{tabular}{>{\raggedright\arraybackslash}p{8.6cm}cc}",
        r"\toprule",
        r"\rule{0pt}{2.2ex} & (1) & (2) \\",
        r"\rule[-0.7ex]{0pt}{2.8ex}Dependent variable: $\ln$(all-in drawn spread) & $\Delta FFR_{t-1}$ & JK shock$_{t-1}$ \\",
        r"\midrule",
    ]
    if jk_rows:
        for order in sorted(main_rows):
            if order not in jk_rows:
                continue
            label = DISPLAY_TEX.get(jk_rows[order]["display"], jk_rows[order]["display"])
            lines.append(f"{label} & " + coef(main_rows.get(order)) + " & " + coef(jk_rows.get(order)) + r" \\")
            lines.append(" & " + se(main_rows.get(order)) + " & " + se(jk_rows.get(order)) + r" \\")
            if order in {7, 14, 18, 22}:
                lines.append(r"\addlinespace")
    else:
        lines.append(
            r"$HHI_{i,t}\times MP_{t-1}\times PD_{b,t}$ & "
            + key_coef(main_row) + " & " + key_coef(jk_row) + r" \\"
        )
        lines.append(" & " + key_se(main_row) + " & " + key_se(jk_row) + r" \\")

    lines.extend(
        [
            r"\midrule",
            "Observations & " + fint(main_row["N"]) + " & " + fint(jk_row["N"]) + r" \\",
            "Adjusted $R^2$ & " + fnum(main_row["r2_adj"], 3) + " & " + fnum(jk_row["r2_adj"], 3) + r" \\",
            "Within $R^2$ & " + fnum(main_row["r2_within"], 3) + " & " + fnum(jk_row["r2_within"], 3) + r" \\",
            r"\bottomrule",
            r"\end{tabular}",
            r"\vspace{0.5ex}",
            r"\begin{minipage}{\textwidth}",
            r"\footnotesize",
            r"\textit{Notes:} Standard errors in square brackets are two-way clustered by bank and quarter. "
            r"$^{***}p<0.01$, $^{**}p<0.05$, $^{*}p<0.10$. Both columns use borrower-year and bank-year fixed effects; "
            r"neither column includes bank-firm fixed effects. "
            r"Column (1) reproduces the preferred main-results specification with $\Delta FFR_{t-1}$; Column (2) replaces it with the lagged "
            r"Jarocinski--Karadi monetary-policy shock. HHI and borrower PD are standardized.",
            r"\par\smallskip",
            r"Borrower PD is the probability of default computed following Bharath and Shumway (2008); "
            r"BS in the source variable name refers to Bharath--Shumway, not Black--Scholes.",
            r"\end{minipage}",
            r"\end{table}",
        ]
    )
    write(out, lines)


def render_uninsured_table(
    unins_coefs: list[dict[str, str]], unins_key: list[dict[str, str]], out: Path
) -> None:
    groups = [
        ("low_unins_high_insured", "Low uninsured"),
        ("high_unins_low_insured", "High uninsured"),
    ]
    rows = {
        (r["subgroup"], int(float(r["row_order"]))): r
        for r in unins_coefs
        if r["spec"] == "median"
    }
    key = {
        (r["result"], r["group"]): r
        for r in unins_key
        if r["spec"] == "median"
    }
    subkey = {
        r["group"]: r
        for r in unins_key
        if r["spec"] == "median" and r["result"] == "subsample_triple"
    }
    wald = key[("wald_diff_low_minus_high", "low_minus_high")]

    lines = [
        r"\begin{table}[!p]",
        r"\centering",
        r"\caption{Uninsured Deposit Split: Median Bank-Year Call-Report Share}",
        r"\label{tab:table_6_uninsured_deposit_split}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{5.0pt}",
        r"\setlength{\aboverulesep}{0.35ex}",
        r"\setlength{\belowrulesep}{0.35ex}",
        r"\renewcommand{\arraystretch}{0.70}",
        r"\begin{tabular}{>{\raggedright\arraybackslash}p{8.4cm}cc}",
        r"\toprule",
        r" & (1) & (2) \\",
        r"Dependent variable: $\ln$(all-in drawn spread) & Low uninsured & High uninsured \\",
        r"\midrule",
    ]

    row_orders = sorted({
        order
        for group, order in rows
        if group == groups[0][0] or group == groups[1][0]
    })
    for order in row_orders:
        base_row = rows.get((groups[0][0], order)) or rows.get((groups[1][0], order))
        if not base_row:
            continue
        label = DISPLAY_TEX.get(base_row["display"], base_row["display"])
        model_rows = [rows.get((group, order)) for group, _ in groups]
        lines.append(f"{label} & " + " & ".join(coef(r) for r in model_rows) + r" \\")
        lines.append(" & " + " & ".join(se(r) for r in model_rows) + r" \\")
        if order in {7, 14, 18, 22}:
            lines.append(r"\addlinespace")

    lines.extend(
        [
            r"\midrule",
            "Observations & " + " & ".join(fint(subkey[group]["N"]) for group, _ in groups) + r" \\",
            "Adjusted $R^2$ & " + " & ".join(fnum(subkey[group]["r2a"], 3) for group, _ in groups) + r" \\",
            rf"Wald test: equal triple coefficients & \multicolumn{{2}}{{c}}{{$F({int(float(wald['df_num']))},{int(float(wald['df_den']))})={fnum(wald['Fstat'], 3)}$, $p={fp(wald['pval'])}$}} \\",
            r"\bottomrule",
            r"\end{tabular}",
            r"\vspace{0.5ex}",
            r"\begin{minipage}{\textwidth}",
            r"\footnotesize",
            rf"\textit{{Notes:}} Standard errors in square brackets are two-way clustered by bank and quarter. "
            rf"$^{{***}}p<0.01$, $^{{**}}p<0.05$, $^{{*}}p<0.10$. The split is based on the Call-Report bank-year "
            rf"mean uninsured-deposit share, \texttt{{RCON5597}}/\texttt{{RCON2200}}, with the cutoff computed over "
            rf"unweighted bank-years in the final estimation-panel backbone. The median cutoff is {fnum(wald['cut_low'], 3)}.",
            r"\par\smallskip",
            r"The Wald statistic comes from a pooled fully interacted model with split-specific bank-year and "
            r"borrower-year fixed effects. Column coefficients are from separate median-split subsample regressions.",
            r"\par\smallskip",
            r"Borrower PD is the probability of default computed following Bharath and Shumway (2008); BS in the "
            r"source variable name refers to Bharath--Shumway, not Black--Scholes.",
            r"\end{minipage}",
            r"\end{table}",
        ]
    )
    write(out, lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--key", type=Path, required=True)
    parser.add_argument("--coefs", type=Path, required=True)
    parser.add_argument("--unins-key", type=Path, required=True)
    parser.add_argument("--unins-coefs", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()

    key = read_csv(args.key)
    coefs = read_csv(args.coefs)
    unins_key = read_csv(args.unins_key)
    unins_coefs = read_csv(args.unins_coefs)

    render_main_table(coefs, key, args.outdir / "table_4_main_results.tex")
    render_jk_table(coefs, key, args.outdir / "table_5_jk_robustness.tex")
    render_uninsured_table(unins_coefs, unins_key, args.outdir / "table_6_uninsured_deposit_split.tex")


if __name__ == "__main__":
    main()
