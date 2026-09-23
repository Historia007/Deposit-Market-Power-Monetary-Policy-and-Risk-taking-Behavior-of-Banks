#!/usr/bin/env python3
"""
Verification checks for lender-bank matching outputs.

This script validates:
1) lender->RSSD mapping integrity
2) call-report/SOD quarter alignment integrity
3) manual mapping window/successor logic integrity
4) coverage CSV consistency with parquet outputs
5) macro + FFR merge consistency
"""

from __future__ import annotations
import os

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


BASE = Path(os.environ.get("MPHIL_THESIS_ROOT", Path(__file__).resolve().parents[2]))

IN_CALL = BASE / "data/intermediate/clean/fdic_call_reports/fdic_call_reports_clean.parquet"

IN_LENDER_PANEL = BASE / "data/intermediate/merged/dealscan_compustat_call_sod_macro_lenderlevel.parquet"
IN_LOAN_PANEL = BASE / "data/intermediate/merged/dealscan_compustat_call_sod_macro_loan_borrower_panel.parquet"
IN_LENDER_MAP = BASE / "data/intermediate/merged/dealscan_lender_rssd_mapping.parquet"
IN_LENDER_COVERAGE = BASE / "data/intermediate/merged/lender_bank_hhi_coverage_summary.csv"
IN_LOAN_COVERAGE = BASE / "data/intermediate/merged/loan_lender_match_coverage_summary.csv"
IN_YEAR_COVERAGE = BASE / "data/intermediate/merged/lender_bank_hhi_coverage_by_year.csv"

OUT_REPORT = BASE / "data/documentation/data_processing_reports/lender_bank_matching_verification_report.md"


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str


def _load_merge_module():
    mod_path = BASE / "scripts/06_merge_four_tables.py"
    spec = importlib.util.spec_from_file_location("merge_module", mod_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _within_window(year: pd.Series, start: pd.Series, end: pd.Series) -> pd.Series:
    ok = year.notna()
    ok &= start.isna() | (year >= start)
    ok &= end.isna() | (year <= end)
    return ok


def _add_check(results: List[CheckResult], name: str, passed: bool, detail: str) -> None:
    results.append(CheckResult(name=name, passed=passed, detail=detail))


def run_verification() -> int:
    results: List[CheckResult] = []

    required_files = [
        IN_CALL,
        IN_LENDER_PANEL,
        IN_LOAN_PANEL,
        IN_LENDER_MAP,
        IN_LENDER_COVERAGE,
        IN_LOAN_COVERAGE,
        IN_YEAR_COVERAGE,
    ]
    missing = [str(p) for p in required_files if not p.exists()]
    if missing:
        _add_check(results, "required_files_exist", False, f"Missing files: {missing}")
        return _write_and_exit(results)
    _add_check(results, "required_files_exist", True, "All required input files exist.")

    lender = pd.read_parquet(
        IN_LENDER_PANEL,
        columns=[
            "ds_row_id",
            "deal_date",
            "mapped_rssd_id",
            "lender_id",
            "lender_rssd_match_method",
            "flag_lender_rssd_mapped",
            "flag_lender_call_matched",
            "flag_lender_hhi_matched",
            "flag_rssd_extrapolated_post2016",
            "REPDTE",
            "hhi_bank_q",
            "ffr_q",
            "macro_ffr_q",
            "flag_sod_sameyear",
            "flag_sod_ffill_used",
            "manual_rule_start_year",
            "manual_rule_end_year",
            "manual_rule_primary_rssd_id",
            "manual_rule_successor_rssd_id",
            "manual_rule_successor_start_year",
            "manual_rule_successor_end_year",
        ],
    )
    loan = pd.read_parquet(
        IN_LOAN_PANEL,
        columns=[
            "flag_any_lender_unmatched_rssd",
            "flag_all_lenders_unmatched_rssd",
            "flag_any_lender_unmatched_hhi",
            "flag_all_lenders_unmatched_hhi",
        ],
    )
    lender_map = pd.read_parquet(
        IN_LENDER_MAP,
        columns=[
            "ds_row_id",
            "deal_year",
            "lender_id",
            "lender_rssd_match_method",
            "mapped_rssd_id",
            "flag_lender_rssd_mapped",
            "manual_rule_start_year",
            "manual_rule_end_year",
            "manual_rule_primary_rssd_id",
            "manual_rule_successor_rssd_id",
            "manual_rule_successor_start_year",
            "manual_rule_successor_end_year",
        ],
    )
    lender_cov = pd.read_csv(IN_LENDER_COVERAGE)
    loan_cov = pd.read_csv(IN_LOAN_COVERAGE)
    year_cov = pd.read_csv(IN_YEAR_COVERAGE)

    call = pd.read_parquet(IN_CALL, columns=["RSSDID", "REPDTE"])
    call["RSSDID"] = pd.to_numeric(call["RSSDID"], errors="coerce").astype("Int64")
    call["REPDTE"] = pd.to_datetime(call["REPDTE"], errors="coerce")
    call_rssd = set(call["RSSDID"].dropna().astype("int64").unique())

    # ------------------------------------------------------------------
    # Basic row/ID integrity.
    # ------------------------------------------------------------------
    _add_check(
        results,
        "map_row_count_matches_lender_panel",
        len(lender_map) == len(lender),
        f"lender_map rows={len(lender_map):,}, lender_panel rows={len(lender):,}",
    )
    _add_check(
        results,
        "map_ds_row_id_unique",
        lender_map["ds_row_id"].nunique(dropna=False) == len(lender_map),
        f"unique ds_row_id in map={lender_map['ds_row_id'].nunique(dropna=False):,}",
    )
    _add_check(
        results,
        "lender_panel_ds_row_id_unique",
        lender["ds_row_id"].nunique(dropna=False) == len(lender),
        f"unique ds_row_id in lender panel={lender['ds_row_id'].nunique(dropna=False):,}",
    )

    # ------------------------------------------------------------------
    # Flag consistency.
    # ------------------------------------------------------------------
    lender["mapped_rssd_id"] = pd.to_numeric(lender["mapped_rssd_id"], errors="coerce").astype("Int64")
    lender["flag_lender_rssd_mapped"] = pd.to_numeric(lender["flag_lender_rssd_mapped"], errors="coerce").fillna(0).astype(int)
    lender["flag_lender_call_matched"] = pd.to_numeric(lender["flag_lender_call_matched"], errors="coerce").fillna(0).astype(int)
    lender["flag_lender_hhi_matched"] = pd.to_numeric(lender["flag_lender_hhi_matched"], errors="coerce").fillna(0).astype(int)

    rssd_flag_ok = ((lender["mapped_rssd_id"].notna().astype(int)) == lender["flag_lender_rssd_mapped"]).all()
    _add_check(results, "flag_lender_rssd_mapped_consistent", bool(rssd_flag_ok), "flag_lender_rssd_mapped equals mapped_rssd_id non-null.")

    lender["REPDTE"] = pd.to_datetime(lender["REPDTE"], errors="coerce")
    call_flag_ok = ((lender["REPDTE"].notna().astype(int)) == lender["flag_lender_call_matched"]).all()
    _add_check(results, "flag_lender_call_matched_consistent", bool(call_flag_ok), "flag_lender_call_matched equals REPDTE non-null.")

    hhi_flag_ok = ((lender["hhi_bank_q"].notna().astype(int)) == lender["flag_lender_hhi_matched"]).all()
    _add_check(results, "flag_lender_hhi_matched_consistent", bool(hhi_flag_ok), "flag_lender_hhi_matched equals hhi_bank_q non-null.")

    hhi_implies_call = (lender["flag_lender_hhi_matched"] <= lender["flag_lender_call_matched"]).all()
    _add_check(results, "hhi_implies_call_match", bool(hhi_implies_call), "HHI match implies call match for every row.")

    # ------------------------------------------------------------------
    # Time alignment checks for merge_asof.
    # ------------------------------------------------------------------
    lender["deal_date"] = pd.to_datetime(lender["deal_date"], errors="coerce")
    matched = lender[lender["flag_lender_call_matched"] == 1].copy()
    delta = matched["deal_date"] - matched["REPDTE"]
    # Must mirror merge logic in scripts/06_merge_four_tables.py:
    # backward asof within 370 days OR conservative forward fallback up to 120 days.
    time_ok = (delta >= pd.Timedelta(days=-120)) & (delta <= pd.Timedelta(days=370))
    _add_check(
        results,
        "call_asof_time_window_valid",
        bool(time_ok.all()),
        f"violations={(~time_ok).sum():,} out of {len(matched):,} call-matched rows",
    )

    mapped = lender[lender["mapped_rssd_id"].notna()].copy()
    rssd_universe_ok = mapped["mapped_rssd_id"].astype("int64").isin(call_rssd).all()
    _add_check(
        results,
        "mapped_rssd_in_call_universe",
        bool(rssd_universe_ok),
        f"violations={(~mapped['mapped_rssd_id'].astype('int64').isin(call_rssd)).sum():,} out of {len(mapped):,} mapped rows",
    )

    # ------------------------------------------------------------------
    # Manual mapping window + successor validation.
    # ------------------------------------------------------------------
    merge_mod = _load_merge_module()
    manual_rules = merge_mod.build_manual_lender_rules(call)
    manual_rules = manual_rules.rename(columns={"lender_id_num": "lender_id_num_rule"})

    lender_map["lender_id_num_rule"] = pd.to_numeric(lender_map["lender_id"], errors="coerce").astype("Int64")
    lender_map["deal_year"] = pd.to_numeric(lender_map["deal_year"], errors="coerce").astype("Int64")
    lender_map["mapped_rssd_id"] = pd.to_numeric(lender_map["mapped_rssd_id"], errors="coerce").astype("Int64")
    manual_join = lender_map.merge(manual_rules, on="lender_id_num_rule", how="left")

    m_primary = manual_join[manual_join["lender_rssd_match_method"] == "manual_top_lender"].copy()
    primary_year_ok = _within_window(m_primary["deal_year"], m_primary["start_year"], m_primary["end_year"])
    primary_rssd_ok = m_primary["mapped_rssd_id"] == m_primary["manual_rssd_id"]
    _add_check(
        results,
        "manual_primary_window_and_rssd_valid",
        bool((primary_year_ok & primary_rssd_ok).all()),
        f"violations={int((~(primary_year_ok & primary_rssd_ok)).sum()):,} out of {len(m_primary):,} manual_top_lender rows",
    )

    m_succ = manual_join[manual_join["lender_rssd_match_method"] == "manual_top_lender_successor"].copy()
    if len(m_succ) == 0:
        _add_check(results, "manual_successor_window_and_rssd_valid", True, "No successor rows in this run.")
    else:
        succ_year_ok = _within_window(m_succ["deal_year"], m_succ["successor_start_year"], m_succ["successor_end_year"])
        succ_rssd_ok = m_succ["mapped_rssd_id"] == m_succ["successor_rssd_id"]
        _add_check(
            results,
            "manual_successor_window_and_rssd_valid",
            bool((succ_year_ok & succ_rssd_ok).all()),
            f"violations={int((~(succ_year_ok & succ_rssd_ok)).sum()):,} out of {len(m_succ):,} manual successor rows",
        )

    # ------------------------------------------------------------------
    # Coverage CSV consistency checks.
    # ------------------------------------------------------------------
    recalc = (
        lender.groupby("lender_rssd_match_method", dropna=False)
        .agg(
            rows=("ds_row_id", "size"),
            rssd_mapped=("flag_lender_rssd_mapped", "sum"),
            call_matched=("flag_lender_call_matched", "sum"),
            hhi_matched=("flag_lender_hhi_matched", "sum"),
            extrapolated=("flag_rssd_extrapolated_post2016", "sum"),
        )
        .reset_index()
    )
    cov_cmp = recalc.merge(
        lender_cov[
            [
                "lender_rssd_match_method",
                "rows",
                "rssd_mapped",
                "call_matched",
                "hhi_matched",
                "extrapolated",
            ]
        ],
        on="lender_rssd_match_method",
        how="outer",
        suffixes=("_recalc", "_csv"),
    ).fillna(0)
    cov_diff = (
        (cov_cmp["rows_recalc"] != cov_cmp["rows_csv"])
        | (cov_cmp["rssd_mapped_recalc"] != cov_cmp["rssd_mapped_csv"])
        | (cov_cmp["call_matched_recalc"] != cov_cmp["call_matched_csv"])
        | (cov_cmp["hhi_matched_recalc"] != cov_cmp["hhi_matched_csv"])
        | (cov_cmp["extrapolated_recalc"] != cov_cmp["extrapolated_csv"])
    )
    _add_check(
        results,
        "lender_coverage_csv_consistent_with_panel",
        bool((~cov_diff).all()),
        f"mismatched methods={int(cov_diff.sum())}",
    )

    loan_metric = dict(zip(loan_cov["metric"], loan_cov["value"]))
    tol = 1e-12
    checks_loan = {
        "loan_rows": float(len(loan)),
        "share_any_lender_unmatched_rssd": float(loan["flag_any_lender_unmatched_rssd"].mean()),
        "share_all_lenders_unmatched_rssd": float(loan["flag_all_lenders_unmatched_rssd"].mean()),
        "share_any_lender_unmatched_hhi": float(loan["flag_any_lender_unmatched_hhi"].mean()),
        "share_all_lenders_unmatched_hhi": float(loan["flag_all_lenders_unmatched_hhi"].mean()),
    }
    loan_ok = True
    loan_diff_msg = []
    for k, v in checks_loan.items():
        got = float(loan_metric.get(k, np.nan))
        if np.isnan(got) or abs(got - v) > tol:
            loan_ok = False
            loan_diff_msg.append(f"{k}: csv={got}, recalculated={v}")
    _add_check(
        results,
        "loan_coverage_csv_consistent_with_panel",
        loan_ok,
        "; ".join(loan_diff_msg) if loan_diff_msg else "All loan metrics match.",
    )

    year_calc = (
        lender.assign(deal_year=pd.to_datetime(lender["deal_date"], errors="coerce").dt.year)
        .dropna(subset=["deal_year"])
        .groupby("deal_year")
        .agg(rows=("ds_row_id", "size"))
        .reset_index()
    )
    year_cmp = year_calc.merge(year_cov[["deal_year", "rows"]], on="deal_year", how="outer", suffixes=("_recalc", "_csv")).fillna(0)
    year_ok = (year_cmp["rows_recalc"] == year_cmp["rows_csv"]).all()
    _add_check(
        results,
        "year_coverage_rows_consistent",
        bool(year_ok),
        f"mismatched years={int((year_cmp['rows_recalc'] != year_cmp['rows_csv']).sum())}",
    )

    # ------------------------------------------------------------------
    # Macro/FFR and SOD flags.
    # ------------------------------------------------------------------
    if "macro_ffr_q" in lender.columns and "ffr_q" in lender.columns:
        both = lender["macro_ffr_q"].notna() & lender["ffr_q"].notna()
        macro_ffr_ok = (lender.loc[both, "macro_ffr_q"] == lender.loc[both, "ffr_q"]).all()
        _add_check(results, "ffr_q_equals_macro_ffr_q_when_both_present", bool(macro_ffr_ok), f"checked rows={int(both.sum()):,}")

    sod_ffill_rows = lender["flag_sod_ffill_used"] == 1
    sod_ffill_ok = (
        (lender.loc[sod_ffill_rows, "flag_sod_sameyear"] == 0)
        & lender.loc[sod_ffill_rows, "hhi_bank_q"].notna()
    ).all()
    _add_check(results, "sod_ffill_flags_consistent", bool(sod_ffill_ok), f"checked rows={int(sod_ffill_rows.sum()):,}")
    if "sod_ffill_days" in lender.columns:
        ff_days = pd.to_numeric(lender.loc[sod_ffill_rows, "sod_ffill_days"], errors="coerce")
        sod_days_ok = ((ff_days >= 0) & (ff_days <= 360)).all()
        max_days = int(ff_days.max()) if ff_days.notna().any() else -1
        _add_check(results, "sod_ffill_days_le_360", bool(sod_days_ok), f"checked rows={int(sod_ffill_rows.sum()):,}, max_days={max_days}")

    return _write_and_exit(results)


def _write_and_exit(results: List[CheckResult]) -> int:
    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed)

    lines: List[str] = []
    lines.append("# Lender-Bank Matching Verification Report")
    lines.append("")
    lines.append(f"- Total checks: `{len(results)}`")
    lines.append(f"- Passed: `{passed}`")
    lines.append(f"- Failed: `{failed}`")
    lines.append("")
    lines.append("## Check Results")
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        lines.append(f"- **{status}** `{r.name}`: {r.detail}")
    lines.append("")
    lines.append("## Verdict")
    if failed == 0:
        lines.append("- All automated integrity checks passed.")
    else:
        lines.append("- One or more checks failed. Review failing items above before analysis.")

    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUT_REPORT.write_text("\n".join(lines))
    print(f"Verification report written to: {OUT_REPORT}")
    print(f"Passed: {passed}, Failed: {failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(run_verification())
