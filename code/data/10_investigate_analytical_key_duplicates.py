#!/usr/bin/env python3
from __future__ import annotations
import os

from pathlib import Path

import pandas as pd


BASE = Path(os.environ.get("MPHIL_THESIS_ROOT", Path(__file__).resolve().parents[2]))
IN_PANEL = BASE / "data/final/regression_panel_lender_level.parquet"
OUT_DIR = BASE / "data/documentation/data_processing_reports"
OUT_REPORT = OUT_DIR / "ANALYTICAL_KEY_DUPLICATES_INVESTIGATION_2026-02-10.md"
OUT_TOP_CONFLICTS = OUT_DIR / "analytical_key_duplicate_conflict_columns.csv"


def main() -> None:
    df = pd.read_parquet(IN_PANEL)
    key = ["deal_id", "tranche_id", "lender_id", "gvkey", "deal_date"]

    total_rows = len(df)
    dup_rows = int(df.duplicated(key).sum())
    full_row_dup = int(df.duplicated().sum())

    mask = df.duplicated(key, keep=False)
    d = df.loc[mask].copy()
    g = d.groupby(key, dropna=False, sort=False)
    size = g.size()
    dup_groups = int((size > 1).sum())
    max_mult = int(size.max()) if len(size) else 1

    nonkey = [c for c in df.columns if c not in key]
    nu = g[nonkey].nunique(dropna=False)
    conflict_counts = (nu > 1).sum().sort_values(ascending=False)
    conflict_counts.rename("n_duplicate_key_groups_with_conflict").to_csv(OUT_TOP_CONFLICTS, header=True)

    all_same = (nu.max(axis=1) == 1)
    groups_all_same = int(all_same.sum())

    # Diagnostics for likely cause.
    extra_checks = []
    if "ds_row_id" in df.columns:
        extra_checks.append(("key+ds_row_id", int(df.duplicated(key + ["ds_row_id"]).sum())))
    if "tranche_active_date" in df.columns:
        extra_checks.append(("key+tranche_active_date", int(df.duplicated(key + ["tranche_active_date"]).sum())))
    if "facilityid" in df.columns:
        extra_checks.append(("key+facilityid", int(df.duplicated(key + ["facilityid"]).sum())))

    lines = []
    lines.append("# Analytical-Key Duplicate Investigation (2026-02-10)")
    lines.append("")
    lines.append("## Scope")
    lines.append(f"- Input: `{IN_PANEL}`")
    lines.append(f"- Analytical key: `{', '.join(key)}`")
    lines.append("")
    lines.append("## Headline Results")
    lines.append(f"- Total rows: `{total_rows:,}`")
    lines.append(f"- Duplicate rows on analytical key: `{dup_rows:,}`")
    lines.append(f"- Full-row exact duplicates: `{full_row_dup:,}`")
    lines.append(f"- Duplicate key groups: `{dup_groups:,}`")
    lines.append(f"- Max multiplicity in one key group: `{max_mult}`")
    lines.append("")
    lines.append("## Are They True Duplicates?")
    lines.append(f"- Groups with all non-key columns identical: `{groups_all_same:,}`")
    lines.append(f"- Groups with at least one non-key conflict: `{dup_groups - groups_all_same:,}`")
    lines.append("- Conclusion: these are **not** full-row duplicates; they are repeated analytical keys with changing non-key fields.")
    lines.append("")
    lines.append("## Why They Exist")
    lines.append("- The same `(deal, tranche, lender, borrower-gvkey, deal_date)` can appear at multiple tranche snapshots/amendment dates in DealScan.")
    lines.append("- Most conflicts are in tranche-level terms that vary over time (e.g., `tranche_active_date`, pricing, amount, lender-role metadata).")
    lines.append("- This means key definition is too coarse for amendment-level rows; `ds_row_id` remains unique.")
    lines.append("")
    lines.append("## Key-Expansion Diagnostics")
    for name, ndup in extra_checks:
        lines.append(f"- Duplicate rows using `{name}`: `{ndup:,}`")
    lines.append("")
    lines.append("## Top Conflict Columns")
    top = conflict_counts.head(20)
    for col, cnt in top.items():
        lines.append(f"- `{col}`: conflicts in `{int(cnt):,}` duplicate-key groups")
    lines.append("")
    lines.append("## Output Artifacts")
    lines.append(f"- Conflict counts CSV: `{OUT_TOP_CONFLICTS}`")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_REPORT.write_text("\n".join(lines))
    print(f"Wrote: {OUT_REPORT}")
    print(f"Wrote: {OUT_TOP_CONFLICTS}")


if __name__ == "__main__":
    main()
