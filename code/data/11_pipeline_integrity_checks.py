#!/usr/bin/env python3
"""
End-to-end integrity checks for the current data pipeline.

Covers:
- clean datasets
- DealScan-Compustat linking
- four-table merge outputs
- final regression panels
"""

from __future__ import annotations
import os

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd


def _resolve_base() -> Path:
    """Resolve project base path across legacy/current folder naming."""
    here = Path(__file__).resolve()
    candidates = [
        Path.cwd(),
        here.parent.parent,
        Path(os.environ.get("MPHIL_THESIS_ROOT", Path(__file__).resolve().parents[2])),
        Path(os.environ.get("MPHIL_THESIS_ROOT", Path(__file__).resolve().parents[2])),
    ]
    for p in candidates:
        if (p / "data").exists() and (p / "scripts").exists():
            return p
    return here.parent.parent


BASE = _resolve_base()

P_CALL_CLEAN = BASE / "data/intermediate/clean/fdic_call_reports/fdic_call_reports_clean.parquet"
P_SOD_CLEAN = BASE / "data/intermediate/clean/fdic_sod/fdic_sod_cleaned.parquet"
P_DEALSCAN_CLEAN = BASE / "data/intermediate/clean/dealscan/dealscan_cleaned.parquet"
P_COMPUSTAT_CLEAN = BASE / "data/intermediate/clean/compustat/compustat_quarterly_cleaned.parquet"

P_DS_COMP_LINKED = BASE / "data/intermediate/merged/dealscan_compustat_linked.parquet"
P_DS_COMP_PANEL = BASE / "data/intermediate/merged/dealscan_compustat_loan_borrower_panel.parquet"
P_LINK_SUMMARY = BASE / "data/intermediate/merged/dealscan_compustat_linking_summary_metrics.csv"
P_LINK_LAG = BASE / "data/intermediate/merged/dealscan_compustat_panel_quarter_lag_distribution.csv"

P_MERGED_LENDER = BASE / "data/intermediate/merged/dealscan_compustat_call_sod_macro_lenderlevel.parquet"
P_MERGED_LOAN = BASE / "data/intermediate/merged/dealscan_compustat_call_sod_macro_loan_borrower_panel.parquet"
P_LENDER_COVERAGE = BASE / "data/intermediate/merged/lender_bank_hhi_coverage_summary.csv"
P_LOAN_COVERAGE = BASE / "data/intermediate/merged/loan_lender_match_coverage_summary.csv"

P_FINAL_LENDER = BASE / "data/final/regression_panel_lender_level.parquet"
P_FINAL_LOAN = BASE / "data/final/regression_panel_loan_level.parquet"

OUT_REPORT = BASE / "data/documentation/data_processing_reports/PIPELINE_INTEGRITY_CHECK_REPORT_2026-02-09.md"


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str


def _safe_ratio(a: float, b: float) -> float:
    return float(a) / float(b) if b else float("nan")


def _first_existing_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def run_checks() -> List[CheckResult]:
    res: List[CheckResult] = []

    # 1) Required files exist
    required = [
        P_CALL_CLEAN,
        P_SOD_CLEAN,
        P_DEALSCAN_CLEAN,
        P_COMPUSTAT_CLEAN,
        P_DS_COMP_LINKED,
        P_DS_COMP_PANEL,
        P_LINK_SUMMARY,
        P_LINK_LAG,
        P_MERGED_LENDER,
        P_MERGED_LOAN,
        P_LENDER_COVERAGE,
        P_LOAN_COVERAGE,
        P_FINAL_LENDER,
        P_FINAL_LOAN,
    ]
    missing = [str(p) for p in required if not p.exists()]
    res.append(
        CheckResult(
            "required_files_exist",
            len(missing) == 0,
            "all required outputs exist" if not missing else f"missing: {missing}",
        )
    )
    if missing:
        return res

    # Load core datasets
    call = pd.read_parquet(P_CALL_CLEAN, columns=["RSSDID", "REPDTE", "year_qtr"])
    sod = pd.read_parquet(P_SOD_CLEAN, columns=["YEAR", "RSSDID", "hhi_bank"])
    ds = pd.read_parquet(P_DEALSCAN_CLEAN, columns=["lpc_deal_id", "lpc_tranche_id", "deal_date", "borrower_id"])
    comp = pd.read_parquet(P_COMPUSTAT_CLEAN, columns=["gvkey", "datadate", "year_qtr"])
    ds_comp = pd.read_parquet(P_DS_COMP_LINKED, columns=["lpc_deal_id", "borrower_id", "gvkey", "datadate"])
    # Load full panel first to support schema variants across runs.
    ds_comp_panel = pd.read_parquet(P_DS_COMP_PANEL)
    merged_lender = pd.read_parquet(
        P_MERGED_LENDER,
        columns=[
            "ds_row_id",
            "deal_date",
            "mapped_rssd_id",
            "lender_rssd_match_method",
            "flag_lender_rssd_mapped",
            "flag_lender_call_matched",
            "flag_lender_hhi_matched",
            "hhi_bank_q",
            "ffr_q",
            "macro_ffr_q",
            "gvkey",
        ],
    )
    merged_loan = pd.read_parquet(P_MERGED_LOAN, columns=["lpc_deal_id", "gvkey", "deal_date"])
    final_lender = pd.read_parquet(P_FINAL_LENDER)
    final_loan = pd.read_parquet(P_FINAL_LOAN)
    if "lpc_deal_id" not in final_loan.columns and "deal_id" in final_loan.columns:
        final_loan = final_loan.rename(columns={"deal_id": "lpc_deal_id"})

    # 2) Time ranges
    ds_dates = pd.to_datetime(ds["deal_date"], errors="coerce")
    final_dates = pd.to_datetime(final_lender["deal_date"], errors="coerce")
    ok_time = (ds_dates.min() >= pd.Timestamp("2001-01-01")) and (ds_dates.max() <= pd.Timestamp("2025-12-31"))
    ok_final_time = (final_dates.min() >= pd.Timestamp("2001-01-01")) and (final_dates.max() <= pd.Timestamp("2025-12-31"))
    res.append(
        CheckResult(
            "time_window_2001_2025",
            bool(ok_time and ok_final_time),
            f"dealscan={ds_dates.min()}..{ds_dates.max()}, final={final_dates.min()}..{final_dates.max()}",
        )
    )

    # 3) Key uniqueness checks
    ds_panel_dup = ds_comp_panel.duplicated(["lpc_deal_id", "lpc_tranche_id", "borrower_id", "deal_date", "gvkey"]).sum()
    res.append(
        CheckResult(
            "dealscan_compustat_panel_key_unique",
            int(ds_panel_dup) == 0,
            f"duplicate panel keys={int(ds_panel_dup)}",
        )
    )
    final_loan_dup = final_loan.duplicated(["lpc_deal_id", "gvkey", "deal_date"]).sum()
    res.append(
        CheckResult(
            "final_loan_panel_key_unique",
            int(final_loan_dup) == 0,
            f"duplicate final loan keys={int(final_loan_dup)}",
        )
    )

    # 4) Compustat linking coverage consistency
    link_summary = pd.read_csv(P_LINK_SUMMARY)
    sm = {r["metric"]: float(r["value"]) for _, r in link_summary.iterrows()}
    gvkey_rows = float(ds_comp["gvkey"].notna().sum())
    comp_rows = float(ds_comp["datadate"].notna().sum())
    panel_comp_rows = float(ds_comp_panel["datadate"].notna().sum())
    ok_link = (
        int(sm.get("obs_total", -1)) == int(len(ds_comp))
        and int(sm.get("obs_with_gvkey", -1)) == int(gvkey_rows)
        and int(sm.get("obs_with_compustat", -1)) == int(comp_rows)
        and int(sm.get("panel_with_compustat", -1)) == int(panel_comp_rows)
    )
    res.append(
        CheckResult(
            "compustat_link_summary_consistent",
            bool(ok_link),
            f"summary vs parquet: obs={len(ds_comp)}, gvkey={int(gvkey_rows)}, comp={int(comp_rows)}, panel_comp={int(panel_comp_rows)}",
        )
    )

    # 5) Quarter-lag distribution consistency
    lag_csv = pd.read_csv(P_LINK_LAG)
    lag_total_csv = int(lag_csv["count"].sum())

    if "quarters_lag" in ds_comp_panel.columns:
        lag_total_panel = int(ds_comp_panel["quarters_lag"].notna().sum())
    elif "deal_year_qtr" in ds_comp_panel.columns and "comp_year_qtr" in ds_comp_panel.columns:
        # Fallback for panel schema where lag is represented by year-quarter strings.
        # Count only rows with aligned Compustat quarter (datadate non-missing).
        if "datadate" in ds_comp_panel.columns:
            lag_total_panel = int(ds_comp_panel["datadate"].notna().sum())
        else:
            lag_total_panel = int(ds_comp_panel["comp_year_qtr"].notna().sum())
    else:
        lag_total_panel = -1
    res.append(
        CheckResult(
            "compustat_quarter_lag_distribution_consistent",
            lag_total_csv == lag_total_panel,
            f"csv_total={lag_total_csv}, panel_nonmissing_lag={lag_total_panel}",
        )
    )

    # 6) Merge flags consistency
    mapped = merged_lender["mapped_rssd_id"].notna()
    ok_flag_map = (mapped.astype(int) == merged_lender["flag_lender_rssd_mapped"].fillna(0).astype(int)).all()
    call_flag = merged_lender["flag_lender_call_matched"].fillna(0).astype(int)
    hhi_flag = merged_lender["flag_lender_hhi_matched"].fillna(0).astype(int)
    hhi_nonmissing = merged_lender["hhi_bank_q"].notna().astype(int)
    ok_hhi_flag = (hhi_flag == hhi_nonmissing).all()
    ok_hhi_implies_call = ((hhi_flag <= call_flag).all())
    res.append(CheckResult("merge_flag_rssd_consistent", bool(ok_flag_map), "flag_lender_rssd_mapped equals mapped_rssd_id non-null"))
    res.append(CheckResult("merge_flag_hhi_consistent", bool(ok_hhi_flag), "flag_lender_hhi_matched equals hhi_bank_q non-null"))
    res.append(CheckResult("merge_hhi_implies_call", bool(ok_hhi_implies_call), "HHI matched rows are subset of call-matched rows"))

    # 7) Merge coverage summary consistency
    lender_cov = pd.read_csv(P_LENDER_COVERAGE)
    cov_rows = int(lender_cov["rows"].sum())
    cov_hhi = int(lender_cov["hhi_matched"].sum())
    ok_cov = (cov_rows == len(merged_lender)) and (cov_hhi == int(merged_lender["flag_lender_hhi_matched"].sum()))
    res.append(
        CheckResult(
            "lender_coverage_csv_consistent",
            bool(ok_cov),
            f"csv_rows={cov_rows}, parquet_rows={len(merged_lender)}, csv_hhi={cov_hhi}, parquet_hhi={int(merged_lender['flag_lender_hhi_matched'].sum())}",
        )
    )

    # 8) Macro ffr consistency where both exist
    both = merged_lender["ffr_q"].notna() & merged_lender["macro_ffr_q"].notna()
    max_diff = float((merged_lender.loc[both, "ffr_q"] - merged_lender.loc[both, "macro_ffr_q"]).abs().max()) if both.any() else np.nan
    ok_ffr = (not both.any()) or (max_diff < 1e-8)
    res.append(CheckResult("macro_ffr_alignment", bool(ok_ffr), f"checked_rows={int(both.sum())}, max_abs_diff={max_diff}"))

    # 9) Final panel rules checks (schema-flexible for trimmed variable names)
    spread_col = _first_existing_column(final_lender, ["all_in_spread_drawn_bps"])
    hhi_raw_col = _first_existing_column(final_lender, ["hhi_bank_q"])
    hhi_trim_col = _first_existing_column(final_lender, ["hhi_bank_q_trim_p1p99", "hhi_bank_q_trim1"])
    hhi_trim_flag_col = _first_existing_column(final_lender, ["flag_hhi_bank_q_trimmed_p1p99", "flag_hhi_bank_q_trim1"])
    jk_col = _first_existing_column(final_lender, ["jk_MP_pm_sum"])

    if spread_col is None:
        res.append(CheckResult("final_spread_column_exists", False, "missing all_in_spread_drawn_bps in final lender panel"))
    else:
        spread_s = pd.to_numeric(final_lender[spread_col], errors="coerce")
        spread_ok = (spread_s.dropna() > 0).all()
        res.append(CheckResult("final_spread_positive_nonmissing", bool(spread_ok), f"{spread_col} > 0 for all non-missing rows"))

    if hhi_raw_col is None:
        res.append(CheckResult("final_hhi_raw_exists", False, "missing hhi_bank_q in final lender panel"))
    else:
        hhi_raw = pd.to_numeric(final_lender[hhi_raw_col], errors="coerce")
        hhi_bounds = hhi_raw.dropna().between(0, 1, inclusive="both").all()
        res.append(CheckResult("final_hhi_bounds_raw", bool(hhi_bounds), f"{hhi_raw_col} in [0,1] for non-missing rows"))

    if hhi_trim_col is None:
        res.append(CheckResult("final_hhi_trim_exists", False, "missing trimmed hhi variable in final lender panel"))
    else:
        hhi_trim = pd.to_numeric(final_lender[hhi_trim_col], errors="coerce")
        trim_bounds = hhi_trim.dropna().between(0, 1, inclusive="both").all()
        res.append(CheckResult("final_hhi_bounds_trimmed", bool(trim_bounds), f"{hhi_trim_col} in [0,1] for non-missing rows"))

    if hhi_raw_col is not None and hhi_trim_col is not None and hhi_trim_flag_col is not None:
        raw = pd.to_numeric(final_lender[hhi_raw_col], errors="coerce")
        trm = pd.to_numeric(final_lender[hhi_trim_col], errors="coerce")
        flg = pd.to_numeric(final_lender[hhi_trim_flag_col], errors="coerce").fillna(0).astype(int)
        both_nonmissing = raw.notna() & trm.notna()
        changed = (raw != trm) & both_nonmissing
        trim_flag_consistent = (changed.astype(int) == flg).all()
        res.append(
            CheckResult(
                "final_hhi_trim_flag_consistent",
                bool(trim_flag_consistent),
                f"trim flag {hhi_trim_flag_col} matches {hhi_raw_col} vs {hhi_trim_col}",
            )
        )
    else:
        res.append(
            CheckResult(
                "final_hhi_trim_flag_consistent",
                False,
                f"cannot evaluate (raw={hhi_raw_col}, trim={hhi_trim_col}, flag={hhi_trim_flag_col})",
            )
        )

    # JK has known missingness in recent years; enforce coverage threshold instead of full non-missing.
    if jk_col is None:
        res.append(CheckResult("final_jk_column_exists", False, "missing jk_MP_pm_sum in final lender panel"))
    else:
        jk_nonmissing_share = float(final_lender[jk_col].notna().mean())
        res.append(
            CheckResult(
                "final_jk_nonmissing_share_ge_95pct",
                jk_nonmissing_share >= 0.95,
                f"{jk_col} non-missing share={jk_nonmissing_share:.4f}",
            )
        )

    # 10) Final panel key duplication (diagnostic)
    key_cols = ["lpc_deal_id", "lpc_tranche_id", "lender_id", "gvkey", "deal_date"]
    key_cols = [k for k in key_cols if k in final_lender.columns]
    dup_n = int(final_lender.duplicated(key_cols).sum())
    # This is not necessarily a hard fail in lender-level data structure.
    res.append(
        CheckResult(
            "final_lender_key_duplicates_diagnostic",
            True,
            f"duplicate rows on {key_cols}: {dup_n} (diagnostic only; can reflect syndicate/tranche structure)",
        )
    )

    return res


def write_report(results: List[CheckResult]) -> None:
    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    total = len(results)
    passed = sum(r.passed for r in results)
    failed = total - passed

    lines = [
        "# Pipeline Integrity Check Report",
        "",
        f"- Total checks: `{total}`",
        f"- Passed: `{passed}`",
        f"- Failed: `{failed}`",
        "",
        "## Check Results",
    ]
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        lines.append(f"- **{status}** `{r.name}`: {r.detail}")

    lines += [
        "",
        "## Verdict",
        "- Pipeline is considered consistent if all hard checks pass.",
        "- Any `diagnostic` check is informational and not treated as failure.",
    ]

    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report written to: {OUT_REPORT}")
    print(f"Passed: {passed}, Failed: {failed}")


def main() -> int:
    results = run_checks()
    write_report(results)
    failed = [r for r in results if not r.passed]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
