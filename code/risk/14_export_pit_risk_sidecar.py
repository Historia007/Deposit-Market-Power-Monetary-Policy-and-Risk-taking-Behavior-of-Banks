from __future__ import annotations

from datetime import date
import os
from pathlib import Path

import numpy as np
import pandas as pd


BASE = Path(os.environ.get("MPHIL_THESIS_ROOT", Path(__file__).resolve().parents[2]))

# Preferred canonical PiT-at-origination panel (already computed upstream).
IN_PIT = BASE / "data/intermediate/merged/pd_pit_origination_2001_2025.parquet"

# Fallback source that contains lender-level rows with PiT columns.
IN_PANEL_WITH_PIT = BASE / "data/intermediate/merged/regression_panel_lender_level_with_pit_pd.parquet"

OUT_PIT = BASE / "data/intermediate/merged/pd_pit_origination_2001_2025.parquet"
OUT_PIT_STATA = BASE / "data/intermediate/merged/pd_pit_origination_2001_2025_for_stata.dta"
OUT_PIT_STD_STATA = BASE / "data/intermediate/merged/pd_pit_origination_2001_2025_std_for_stata.dta"
OUT_COVERAGE = BASE / f"regression/overview_diagnostics/pd_pit_origination_coverage_{date.today().isoformat()}.csv"

PIT_COLS = [
    "gvkey",
    "deal_date",
    "year",
    "quarter",
    "pd_naive_bs2008_pit",
    "pd_iterative_kmv_pit",
    "pd_chs_12m_pit",
]


def _zscore(series: pd.Series) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce")
    mu = x.mean(skipna=True)
    sd = x.std(skipna=True, ddof=0)
    if pd.isna(sd) or sd == 0:
        return pd.Series(np.nan, index=x.index, dtype=float)
    return (x - mu) / sd


def _build_from_fallback_panel() -> pd.DataFrame:
    if not IN_PANEL_WITH_PIT.exists():
        raise FileNotFoundError(
            "No PiT source found. Missing both:\n"
            f"- {IN_PIT}\n"
            f"- {IN_PANEL_WITH_PIT}"
        )

    df = pd.read_parquet(
        IN_PANEL_WITH_PIT,
        columns=[
            "gvkey",
            "deal_date",
            "year",
            "quarter",
            "pd_naive_bs2008_pit",
            "pd_iterative_kmv_pit",
            "pd_chs_12m_pit",
        ],
    ).copy()

    df["gvkey"] = df["gvkey"].astype(str).str.strip()
    df["deal_date"] = pd.to_datetime(df["deal_date"], errors="coerce")
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["quarter"] = pd.to_numeric(df["quarter"], errors="coerce")

    # Collapse lender-level duplicates to one borrower-deal origination row.
    out = (
        df.groupby(["gvkey", "deal_date", "year", "quarter"], as_index=False)[
            ["pd_naive_bs2008_pit", "pd_iterative_kmv_pit", "pd_chs_12m_pit"]
        ]
        .mean()
        .sort_values(["gvkey", "deal_date"])
        .reset_index(drop=True)
    )
    return out


def load_or_build_pit() -> pd.DataFrame:
    if IN_PIT.exists():
        df = pd.read_parquet(IN_PIT).copy()
    else:
        df = _build_from_fallback_panel()

    missing = [c for c in PIT_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in PiT panel: {missing}")

    df = df[PIT_COLS].copy()
    df["gvkey"] = df["gvkey"].astype(str).str.strip()
    df["deal_date"] = pd.to_datetime(df["deal_date"], errors="coerce")
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["quarter"] = pd.to_numeric(df["quarter"], errors="coerce")

    for c in ["pd_naive_bs2008_pit", "pd_iterative_kmv_pit", "pd_chs_12m_pit"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Enforce one row per (gvkey, deal_date) while keeping year/quarter consistency.
    df = (
        df.groupby(["gvkey", "deal_date"], as_index=False)
        .agg(
            year=("year", "first"),
            quarter=("quarter", "first"),
            pd_naive_bs2008_pit=("pd_naive_bs2008_pit", "mean"),
            pd_iterative_kmv_pit=("pd_iterative_kmv_pit", "mean"),
            pd_chs_12m_pit=("pd_chs_12m_pit", "mean"),
        )
        .sort_values(["gvkey", "deal_date"])
        .reset_index(drop=True)
    )
    return df


def write_outputs(df: pd.DataFrame) -> None:
    OUT_PIT.parent.mkdir(parents=True, exist_ok=True)
    OUT_PIT_STATA.parent.mkdir(parents=True, exist_ok=True)
    OUT_COVERAGE.parent.mkdir(parents=True, exist_ok=True)

    # Canonical parquet used by downstream scripts.
    df.to_parquet(OUT_PIT, index=False)

    # Stata-friendly (raw PiT PDs).
    df_stata = df.copy()
    df_stata.to_stata(OUT_PIT_STATA, write_index=False, version=118)

    # Standardized PiT PDs (global z-score).
    std = df.copy()
    std["pd_naive_std"] = _zscore(std["pd_naive_bs2008_pit"])
    std["pd_iter_std"] = _zscore(std["pd_iterative_kmv_pit"])
    std["pd_chs_std"] = _zscore(std["pd_chs_12m_pit"])
    std.to_stata(OUT_PIT_STD_STATA, write_index=False, version=118)

    cov = pd.DataFrame(
        [
            {
                "rows": int(len(df)),
                "unique_gvkey": int(df["gvkey"].nunique()),
                "year_min": float(df["year"].min()) if len(df) else np.nan,
                "year_max": float(df["year"].max()) if len(df) else np.nan,
                "cov_pd_naive_bs2008_pit": float(df["pd_naive_bs2008_pit"].notna().mean()) if len(df) else np.nan,
                "cov_pd_iterative_kmv_pit": float(df["pd_iterative_kmv_pit"].notna().mean()) if len(df) else np.nan,
                "cov_pd_chs_12m_pit": float(df["pd_chs_12m_pit"].notna().mean()) if len(df) else np.nan,
            }
        ]
    )
    cov.to_csv(OUT_COVERAGE, index=False)

    print(f"Wrote: {OUT_PIT}")
    print(f"Wrote: {OUT_PIT_STATA}")
    print(f"Wrote: {OUT_PIT_STD_STATA}")
    print(f"Wrote: {OUT_COVERAGE}")


def main() -> None:
    df = load_or_build_pit()
    write_outputs(df)


if __name__ == "__main__":
    main()
