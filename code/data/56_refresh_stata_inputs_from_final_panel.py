#!/usr/bin/env python3
"""
Rebuild legacy Stata regression inputs from the refreshed final lender-level panel.

This keeps downstream .do files unchanged while ensuring they run on the latest
corrected data vintage.
"""

from __future__ import annotations
import os

import hashlib
import re
from datetime import date, datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


BASE = Path(os.environ.get("MPHIL_THESIS_ROOT", Path(__file__).resolve().parents[2]))
IN_PANEL = BASE / "data/final/regression_panel_lender_level.parquet"

OUT_INPUT = BASE / "data/final/regression_panel_lender_level_ppml_input.dta"
OUT_STRUCT = BASE / "data/final/regression_panel_lender_level_ppml_structure_controls.dta"
OUT_MACRO = BASE / "data/final/regression_panel_lender_level_ppml_macro_controls.dta"

BACKUP_DIR = BASE / "data/final/archive_stata_inputs"
OUT_DIAG = (
    BASE
    / "regression/overview_diagnostics"
    / f"stata_input_refresh_summary_{date.today().isoformat()}.csv"
)
OUT_NAME_MAP = (
    BASE
    / "regression/overview_diagnostics"
    / f"stata_input_variable_name_map_{date.today().isoformat()}.csv"
)

KEYS = ["deal_id", "tranche_id", "lender_id", "gvkey", "year", "quarter"]
PLACEHOLDERS = {"", "nan", "none", "null", "na", "n/a", ".", "-", "--"}


def _to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _trim_p1p99(s: pd.Series, positive_only: bool = False) -> pd.Series:
    x = _to_num(s)
    base = x[(x > 0) & x.notna()] if positive_only else x[x.notna()]
    if base.empty:
        return x
    q01 = base.quantile(0.01)
    q99 = base.quantile(0.99)
    return x.clip(lower=q01, upper=q99)


def _parse_lender_share(s: pd.Series) -> pd.Series:
    x = s.astype("string").str.strip()
    x = x.str.replace("%", "", regex=False).str.replace(",", "", regex=False)
    x = x.where(~x.str.lower().isin(PLACEHOLDERS), np.nan)
    num = pd.to_numeric(x, errors="coerce").astype("float64")
    # DealScan can appear as 35 or 0.35. Harmonize to [0,1] when clear.
    mask_pct = num.gt(1) & num.le(100)
    num = num.where(~mask_pct, num / 100.0)
    return num.astype("float64")


def _coalesce(df: pd.DataFrame, names: Iterable[str]) -> pd.Series:
    out = pd.Series(np.nan, index=df.index, dtype="float64")
    for n in names:
        if n in df.columns:
            out = out.where(out.notna(), _to_num(df[n]))
    return out


def _coalesce_binary(df: pd.DataFrame, names: Iterable[str]) -> pd.Series:
    """
    Coalesce binary fields that may be stored as 0/1 or Yes/No text.
    """
    out = pd.Series(np.nan, index=df.index, dtype="float64")
    true_tokens = {"1", "true", "t", "yes", "y"}
    false_tokens = {"0", "false", "f", "no", "n"}

    for n in names:
        if n not in df.columns:
            continue
        raw = df[n]
        num = pd.to_numeric(raw, errors="coerce")
        txt = raw.astype("string").str.strip().str.lower()
        mapped = pd.Series(np.nan, index=df.index, dtype="float64")
        mapped = mapped.where(~txt.isin(true_tokens), 1.0)
        mapped = mapped.where(~txt.isin(false_tokens), 0.0)
        candidate = num.where(num.notna(), mapped)
        out = out.where(out.notna(), candidate)
    return out


def _backup_if_exists(path: Path, stamp: str) -> None:
    if not path.exists():
        return
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    bk = BACKUP_DIR / f"{path.stem}_{stamp}{path.suffix}"
    path.replace(bk)


def _coerce_for_stata(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        dt = str(out[c].dtype)
        if dt in {"Float64", "Int64", "boolean"}:
            out[c] = out[c].astype("float64")
        elif dt.startswith("string"):
            out[c] = out[c].astype("string").fillna("").str.slice(0, 120).astype("object")
        elif dt == "object":
            out[c] = out[c].astype("string").fillna("").str.slice(0, 120).astype("object")
    # Stata cannot store +/-inf.
    num_cols = out.select_dtypes(include=[np.number]).columns
    if len(num_cols) > 0:
        out[num_cols] = out[num_cols].replace([np.inf, -np.inf], np.nan)
    return out


def _as_string(series: pd.Series) -> pd.Series:
    s = series.astype("string").fillna("")
    # Keep Stata-safe text; trim very long labels to avoid downstream issues.
    return s.str.slice(0, 120)


def _stata_safe_names(columns: Iterable[str]) -> dict[str, str]:
    """Return deterministic Stata-safe names (<=32 chars, unique)."""
    mapping: dict[str, str] = {}
    used: set[str] = set()
    for col in columns:
        name = str(col)
        name = re.sub(r"[^A-Za-z0-9_]", "_", name)
        if not re.match(r"^[A-Za-z_]", name):
            name = f"v_{name}"
        if len(name) > 32:
            suf = hashlib.md5(str(col).encode("utf-8")).hexdigest()[:6]
            name = f"{name[:25]}_{suf}"
        base = name
        i = 1
        while name in used:
            tail = f"_{i}"
            name = f"{base[: 32 - len(tail)]}{tail}"
            i += 1
        used.add(name)
        mapping[str(col)] = name
    return mapping


def main() -> None:
    if not IN_PANEL.exists():
        raise FileNotFoundError(f"Missing input panel: {IN_PANEL}")

    df = pd.read_parquet(IN_PANEL)
    n_rows = len(df)

    # ------------------------------------------------------------------
    # Build ppml_input.dta
    # ------------------------------------------------------------------
    # Start from the full final panel so no existing variables are silently omitted.
    out = df.copy()
    for k in KEYS:
        if k not in out.columns:
            out[k] = np.nan

    out["year"] = _to_num(out["year"])
    out["quarter"] = _to_num(out["quarter"])
    y = out["year"].astype("Int64")
    q = out["quarter"].astype("Int64")
    out["year_quarter"] = np.where(y.notna() & q.notna(), y.astype(str) + "Q" + q.astype(str), "")

    out["lender_category"] = df["lender_category"] if "lender_category" in df.columns else ""
    out["mapped_rssd_id"] = _to_num(df["mapped_rssd_id"]) if "mapped_rssd_id" in df.columns else np.nan
    out["rssdid"] = _coalesce(df, ["RSSDID_int", "RSSDID", "mapped_rssd_id_int", "mapped_rssd_id"])

    out["spread_bps"] = _coalesce(df, ["all_in_spread_drawn_bps"])
    out["spread_pct"] = _coalesce(df, ["all_in_spread_drawn_pct", "all_in_spread_drawn_bps"]).where(
        pd.notna(_coalesce(df, ["all_in_spread_drawn_pct"])),
        _coalesce(df, ["all_in_spread_drawn_bps"]) / 100.0,
    )
    out["spread_bps_t"] = _coalesce(df, ["all_in_spread_drawn_bps_trim_p1p99"])
    out["spread_pct_t"] = _coalesce(df, ["all_in_spread_drawn_pct_trim_p1p99"])

    out["tenor_m"] = _coalesce(df, ["tenor_maturity"])
    out["log_tenor"] = _coalesce(df, ["log_tenor_maturity"])
    out["tenor_m_t"] = _coalesce(df, ["tenor_maturity_trim_p1p99"])
    out["log_tenor_t"] = _coalesce(df, ["log_tenor_maturity_trim_p1p99"])

    out["loan_amt"] = _coalesce(df, ["loan_amount_mil"])
    out["log_loan"] = _coalesce(df, ["log_loan_amount"])
    out["loan_amt_t"] = _coalesce(df, ["loan_amount_mil_trim_p1p99"])
    out["log_loan_t"] = _coalesce(df, ["log_loan_amount_trim_p1p99"])

    out["deal_amt"] = _coalesce(df, ["deal_amount_converted"])
    out["log_deal_amt"] = np.where(out["deal_amt"] > 0, np.log(out["deal_amt"]), np.nan)
    out["deal_amt_t"] = _coalesce(df, ["deal_amount_converted_trim_p1p99"])
    out["log_deal_amt_t"] = np.where(out["deal_amt_t"] > 0, np.log(out["deal_amt_t"]), np.nan)

    out["secured"] = _coalesce_binary(df, ["secured", "secured_flag"])
    out["covenants"] = _coalesce_binary(df, ["covenants"])
    out["tranche_type"] = _as_string(df["tranche_type"]) if "tranche_type" in df.columns else ""
    out["deal_purpose"] = _as_string(df["deal_purpose"]) if "deal_purpose" in df.columns else ""
    out["primary_purpose"] = _as_string(df["primary_purpose"]) if "primary_purpose" in df.columns else ""
    out["secondary_purpose"] = _as_string(df["secondary_purpose"]) if "secondary_purpose" in df.columns else ""
    out["base_reference_rate"] = _as_string(df["base_reference_rate"]) if "base_reference_rate" in df.columns else ""
    out["is_fixed_rate"] = (
        out["base_reference_rate"].str.contains("fixed", case=False, regex=False).astype("float64")
        if "base_reference_rate" in out.columns
        else np.nan
    )

    out["hhi"] = _coalesce(df, ["hhi_bank_q"])
    out["hhi_t"] = _coalesce(df, ["hhi_bank_q_trim_p1p99"])
    out["ffr"] = _coalesce(df, ["macro_ffr_q", "ffr_q"])
    out["jk_mp"] = _coalesce(df, ["jk_MP_pm_sum", "jk_mp"])
    out["jk_mp_mean"] = _coalesce(df, ["jk_MP_pm_mean", "jk_mp_mean"])

    out["log_ba"] = _coalesce(df, ["log_bank_assets"])
    out["log_ba_t"] = _coalesce(df, ["log_bank_assets_trim_p1p99"])
    out["cap_ratio"] = _coalesce(df, ["call_capital_ratio"])
    out["cap_ratio_t"] = _coalesce(df, ["call_capital_ratio_trim_p1p99"])
    out["lta"] = _coalesce(df, ["call_loans_to_assets"])
    out["lta_t"] = _coalesce(df, ["call_loans_to_assets_trim_p1p99"])
    out["dta"] = _coalesce(df, ["call_deposits_to_assets"])
    out["dta_t"] = _coalesce(df, ["call_deposits_to_assets_trim_p1p99"])
    out["cash_ratio"] = _coalesce(df, ["cash_ratio"])
    out["cash_ratio_t"] = _coalesce(df, ["cash_ratio_trim_p1p99"])

    out["call_assets"] = _coalesce(df, ["call_total_assets_mil"])
    out["call_assets_t"] = _coalesce(df, ["call_total_assets_mil_trim_p1p99"])
    out["call_loans"] = _coalesce(df, ["call_total_loans_mil"])
    out["call_loans_t"] = _coalesce(df, ["call_total_loans_mil_trim_p1p99"])
    out["call_deposits"] = _coalesce(df, ["call_total_deposits_mil"])
    out["call_deposits_t"] = _coalesce(df, ["call_total_deposits_mil_trim_p1p99"])
    out["call_equity"] = _coalesce(df, ["call_equity_capital_mil"])
    out["call_equity_t"] = _coalesce(df, ["call_equity_capital_mil_trim_p1p99"])

    out["log_assets"] = _coalesce(df, ["log_assets"])
    out["log_assets_t"] = _coalesce(df, ["log_assets_trim_p1p99"])
    out["lev"] = _coalesce(df, ["leverage"])
    out["lev_t"] = _coalesce(df, ["leverage_trim_p1p99"])
    out["roa"] = _coalesce(df, ["roa"])
    out["roa_t"] = _coalesce(df, ["roa_trim_p1p99"])
    out["roe"] = _coalesce(df, ["roe"])
    out["roe_t"] = _coalesce(df, ["roe_trim_p1p99"])
    out["zscore_full"] = _coalesce(df, ["zscore_full"])
    out["zscore_nosales"] = _coalesce(df, ["zscore_nosales"])
    out["zscore_full_t"] = _coalesce(df, ["zscore_full_trim_p1p99"])
    out["zscore_nosales_t"] = _coalesce(df, ["zscore_nosales_trim_p1p99"])
    out["mtb"] = _coalesce(df, ["market_to_book"])
    out["mtb_t"] = _coalesce(df, ["market_to_book_trim_p1p99"])
    out["tang"] = _coalesce(df, ["tangibility"])
    out["tang_t"] = _coalesce(df, ["tangibility_trim_p1p99"])

    # Deal structure controls at deal level.
    if "deal_id" in out.columns and "tranche_id" in out.columns:
        key_df = out[["deal_id", "tranche_id"]].copy()
        key_df = key_df.dropna().drop_duplicates()
        tranche_count = key_df.groupby("deal_id")["tranche_id"].nunique()
        out["n_tranches_deal"] = out["deal_id"].map(tranche_count).astype("float64")
    else:
        out["n_tranches_deal"] = np.nan
    out["log_n_tranches_deal"] = np.where(out["n_tranches_deal"] > 0, np.log(out["n_tranches_deal"]), np.nan)
    out["log_n_tranches_deal_t"] = np.nan

    # Fill trimmed counterparts if absent in source.
    trim_pos = [
        "spread_bps_t",
        "spread_pct_t",
        "tenor_m_t",
        "log_tenor_t",
        "loan_amt_t",
        "log_loan_t",
        "deal_amt_t",
        "log_deal_amt_t",
        "log_ba_t",
        "log_assets_t",
        "call_assets_t",
        "call_loans_t",
        "call_deposits_t",
        "call_equity_t",
    ]
    trim_any = [
        "cap_ratio_t",
        "lta_t",
        "dta_t",
        "cash_ratio_t",
        "lev_t",
        "roa_t",
        "roe_t",
        "zscore_full_t",
        "zscore_nosales_t",
        "mtb_t",
        "tang_t",
        "hhi_t",
        "log_n_tranches_deal_t",
    ]
    for v in trim_pos:
        if out[v].isna().all():
            base = v[:-2]
            out[v] = _trim_p1p99(out[base], positive_only=True)
    for v in trim_any:
        if out[v].isna().all():
            base = v[:-2]
            out[v] = _trim_p1p99(out[base], positive_only=False)

    # ------------------------------------------------------------------
    # Build ppml_structure_controls.dta
    # ------------------------------------------------------------------
    struct = out[KEYS].copy()
    struct["n_lenders"] = _coalesce(df, ["number_of_lenders"])
    struct["n_leads"] = _coalesce(df, ["number_of_lead_arrangers"])
    struct["prior_access"] = _coalesce(df, ["prior_synd_loan_access"])
    struct["retained_share"] = _parse_lender_share(df["lender_share"]) if "lender_share" in df.columns else np.nan

    struct["n_lenders_t"] = _trim_p1p99(struct["n_lenders"], positive_only=True)
    struct["n_leads_t"] = _trim_p1p99(struct["n_leads"], positive_only=True)
    struct["log_n_lenders"] = np.where(struct["n_lenders"] > 0, np.log(struct["n_lenders"]), np.nan)
    struct["log_n_lenders_t"] = _trim_p1p99(struct["log_n_lenders"], positive_only=False)
    struct["log_n_leads"] = np.where(struct["n_leads"] > 0, np.log(struct["n_leads"]), np.nan)
    struct["log_n_leads_t"] = _trim_p1p99(struct["log_n_leads"], positive_only=False)
    struct["retained_share_t"] = _trim_p1p99(struct["retained_share"], positive_only=False)
    struct["has_retained_share"] = struct["retained_share"].notna().astype("float64")

    # Enforce key uniqueness for using-side merge.
    struct = (
        struct.sort_values(KEYS)
        .drop_duplicates(KEYS, keep="first")
        .reset_index(drop=True)
    )

    # ------------------------------------------------------------------
    # Build ppml_macro_controls.dta
    # ------------------------------------------------------------------
    macro_cols = [
        "year",
        "quarter",
        "macro_unrate_q",
        "macro_gdp_growth_q",
        "macro_cpi_q",
        "macro_baa_aaa_q",
        "macro_gs10_q",
        "macro_tb3ms_q",
    ]
    macro = pd.DataFrame()
    for c in macro_cols:
        # Avoid merge collisions by only carrying vars not already present in main input.
        if c in {"year", "quarter"} or c not in out.columns:
            macro[c] = _to_num(df[c]) if c in df.columns else np.nan
    macro = macro.drop_duplicates(["year", "quarter"]).sort_values(["year", "quarter"]).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Save with backups
    # ------------------------------------------------------------------
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    _backup_if_exists(OUT_INPUT, stamp)
    _backup_if_exists(OUT_STRUCT, stamp)
    _backup_if_exists(OUT_MACRO, stamp)

    OUT_INPUT.parent.mkdir(parents=True, exist_ok=True)
    OUT_DIAG.parent.mkdir(parents=True, exist_ok=True)

    out_name_map = _stata_safe_names(out.columns)
    struct_name_map = _stata_safe_names(struct.columns)
    macro_name_map = _stata_safe_names(macro.columns)

    out_stata = _coerce_for_stata(out.rename(columns=out_name_map))
    struct_stata = _coerce_for_stata(struct.rename(columns=struct_name_map))
    macro_stata = _coerce_for_stata(macro.rename(columns=macro_name_map))

    out_stata.to_stata(OUT_INPUT, write_index=False, version=118)
    struct_stata.to_stata(OUT_STRUCT, write_index=False, version=118)
    macro_stata.to_stata(OUT_MACRO, write_index=False, version=118)

    name_map = pd.concat(
        [
            pd.DataFrame(
                {
                    "dataset": "input",
                    "source_name": list(out_name_map.keys()),
                    "stata_name": list(out_name_map.values()),
                }
            ),
            pd.DataFrame(
                {
                    "dataset": "structure",
                    "source_name": list(struct_name_map.keys()),
                    "stata_name": list(struct_name_map.values()),
                }
            ),
            pd.DataFrame(
                {
                    "dataset": "macro",
                    "source_name": list(macro_name_map.keys()),
                    "stata_name": list(macro_name_map.values()),
                }
            ),
        ],
        ignore_index=True,
    )
    name_map.to_csv(OUT_NAME_MAP, index=False)

    diag = pd.DataFrame(
        [
            {
                "dataset": "input",
                "rows": float(len(out)),
                "cols": float(len(out.columns)),
                "key_dups": float(out.duplicated(KEYS).sum()),
                "renamed_cols": float(sum(k != v for k, v in out_name_map.items())),
            },
            {
                "dataset": "structure",
                "rows": float(len(struct)),
                "cols": float(len(struct.columns)),
                "key_dups": float(struct.duplicated(KEYS).sum()),
                "renamed_cols": float(sum(k != v for k, v in struct_name_map.items())),
            },
            {
                "dataset": "macro",
                "rows": float(len(macro)),
                "cols": float(len(macro.columns)),
                "key_dups": float(macro.duplicated(["year", "quarter"]).sum()),
                "renamed_cols": float(sum(k != v for k, v in macro_name_map.items())),
            },
            {
                "dataset": "source_panel",
                "rows": float(n_rows),
                "cols": float(df.shape[1]),
                "key_dups": float(df.duplicated(KEYS).sum()),
                "renamed_cols": np.nan,
            },
        ]
    )
    diag.to_csv(OUT_DIAG, index=False)

    print(f"Saved: {OUT_INPUT}")
    print(f"Saved: {OUT_STRUCT}")
    print(f"Saved: {OUT_MACRO}")
    print(f"Backup dir: {BACKUP_DIR}")
    print(f"Diagnostics: {OUT_DIAG}")
    print(f"Name map: {OUT_NAME_MAP}")


if __name__ == "__main__":
    main()
