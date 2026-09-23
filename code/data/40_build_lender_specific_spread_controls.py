#!/usr/bin/env python3
"""
Build lender-specific spread controls from lender-level panel.

Output:
  - data/final/regression_panel_lender_level_lender_spread_controls.dta
  - regression/overview_diagnostics/lender_specific_spread_coverage_YYYY-MM-DD.csv

Design:
  - `spread_pct_lender_strict`: spread observed only when lender share is usable on
    the row (share > 0 and tranche amount > 0).
  - `spread_pct_lender_w`: lender-level commitment-weighted spread within
    (deal_id, lender_id, gvkey, year, quarter), mapped back to rows.
  - `_t` variants are p1/p99 trimmed copies for PPML use.
  - Lender commitment is built from DealScan `lender_commit` when available,
    converted to USD millions using tranche FX (tranche_amount / tranche_amount_converted).
    If raw commitment is unavailable, fallback uses lender_share * tranche amount.
  - Additional fallback rules are used for lender commitment construction:
    residual-single-missing, single-lender-full-tranche, and role-aware
    60/40 split (with equal-split fallback when no lead is identified) when
    a tranche has no lender-level commitment/share breakdown at all.
  - Additional hierarchy-based variants are exported using lead-role rankings:
    60/40, fixed-average (14.3%, 26%), and weighted sample-average lead share.
"""

from __future__ import annotations
import os

from datetime import date
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


BASE = Path(os.environ.get("MPHIL_THESIS_ROOT", Path(__file__).resolve().parents[2]))
IN_PANEL = BASE / "data/final/regression_panel_lender_level.parquet"
IN_RAW = BASE / "data/raw/dealscan/dealscan__2001_2025.parquet"
OUT_DTA = BASE / "data/final/regression_panel_lender_level_lender_spread_controls.dta"
OUT_DIAG = (
    BASE
    / "regression/overview_diagnostics"
    / f"lender_specific_spread_coverage_{date.today().isoformat()}.csv"
)

MERGE_KEYS = ["deal_id", "tranche_id", "lender_id", "gvkey", "year", "quarter"]
GROUP_KEYS = ["deal_id", "lender_id", "gvkey", "year", "quarter"]

PLACEHOLDERS = {"", "nan", "none", "null", "na", "n/a", ".", "-", "--"}

# Ten-rank hierarchy used to identify lead lenders; lower rank is higher priority.
ROLE_RANKS = {
    "administrative agent": 1,
    "admin agent": 1,
    "agent": 2,
    "lead bank": 3,
    "lead arranger": 4,
    "mandated lead arranger": 5,
    "mandated arranger": 6,
    "arranger": 7,
    "bookrunner": 8,
    "book runner": 8,
    "book-runner": 8,
    "lead manager": 9,
    "manager": 10,
}

IVASHINA_CHAIN = [
    {"administrative agent", "admin agent"},
    {"agent"},
    {"arranger"},
    {"bookrunner", "book runner", "book-runner"},
    {"lead arranger"},
    {"lead bank"},
    {"lead manager"},
]


def _require_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _parse_lender_share(series: pd.Series) -> pd.Series:
    s = series.astype("string").str.strip()
    s = s.str.replace("%", "", regex=False).str.replace(",", "", regex=False)
    lower = s.str.lower()
    s = s.where(~lower.isin(PLACEHOLDERS), np.nan)
    return pd.to_numeric(s, errors="coerce") / 100.0


def _normalize_key(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip()


def _parse_lender_commit(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    s = series.astype("string").str.strip()
    m = s.str.extract(r"^\s*([A-Za-z]{3})\s*([+-]?[0-9][0-9,]*\.?[0-9]*)\s*$")
    ccy = m[0].str.upper()
    val = pd.to_numeric(m[1].str.replace(",", "", regex=False), errors="coerce")
    return ccy, val


def _norm_role_token(token: str) -> str:
    t = str(token).strip().lower()
    if t in PLACEHOLDERS:
        return ""
    t = t.replace("/", " ").replace("-", " ")
    t = " ".join(t.split())
    return t


def _role_tokens(primary: pd.Series, additional: pd.Series) -> list[set[str]]:
    tokens: list[set[str]] = []
    p = primary.fillna("").astype("string")
    a = additional.fillna("").astype("string")
    for pv, av in zip(p, a):
        row: set[str] = set()
        parts = []
        if pv:
            parts.append(str(pv))
        if av:
            parts.append(str(av))
        joined = ",".join(parts)
        for tok in joined.replace(";", ",").split(","):
            nt = _norm_role_token(tok)
            if nt:
                row.add(nt)
        tokens.append(row)
    return tokens


def _lead_rank_features(primary: pd.Series, additional: pd.Series) -> pd.DataFrame:
    toks = _role_tokens(primary, additional)
    role_ranks_norm: dict[str, int] = {}
    for k, v in ROLE_RANKS.items():
        nk = _norm_role_token(k)
        if not nk:
            continue
        cur = role_ranks_norm.get(nk)
        role_ranks_norm[nk] = int(v) if cur is None else int(min(cur, int(v)))
    ivashina_chain_norm: list[set[str]] = []
    for bucket in IVASHINA_CHAIN:
        nb = {_norm_role_token(t) for t in bucket}
        nb.discard("")
        ivashina_chain_norm.append(nb)
    best_rank = []
    fallback_stage = []
    for row in toks:
        ranks = [role_ranks_norm[t] for t in row if t in role_ranks_norm]
        best_rank.append(float(min(ranks)) if ranks else np.nan)
        stage = np.nan
        if not ranks:
            for i, bucket in enumerate(ivashina_chain_norm, start=1):
                if any(t in bucket for t in row):
                    stage = float(i)
                    break
        fallback_stage.append(stage)
    return pd.DataFrame(
        {
            "lead_rank_best": pd.to_numeric(best_rank, errors="coerce"),
            "lead_stage_fallback": pd.to_numeric(fallback_stage, errors="coerce"),
        }
    )


def _build_raw_commit_sidecar() -> pd.DataFrame:
    if not IN_RAW.exists():
        raise FileNotFoundError(f"Raw DealScan file not found: {IN_RAW}")

    raw_cols = [
        "lpc_deal_id",
        "lpc_tranche_id",
        "lender_id",
        "lender_commit",
        "tranche_amount",
        "tranche_amount_converted",
        "primary_role",
        "additional_roles",
        "lead_arranger",
    ]
    raw = pd.read_parquet(IN_RAW, columns=raw_cols)
    raw["_k_deal"] = _normalize_key(raw["lpc_deal_id"])
    raw["_k_tranche"] = _normalize_key(raw["lpc_tranche_id"])
    raw["_k_lender"] = _normalize_key(raw["lender_id"])

    raw["lender_commit_ccy"], raw["lender_commit_value"] = _parse_lender_commit(
        raw["lender_commit"]
    )
    raw["lender_commit_value"] = pd.to_numeric(
        raw["lender_commit_value"], errors="coerce"
    ).astype("float64")
    tr_amt = pd.to_numeric(raw["tranche_amount"], errors="coerce")
    tr_amt_usd = pd.to_numeric(raw["tranche_amount_converted"], errors="coerce")
    raw["fx_lcy_per_usd"] = np.where(
        (tr_amt > 0) & (tr_amt_usd > 0), tr_amt / tr_amt_usd, np.nan
    )
    cond_commit = raw["lender_commit_value"].gt(0) & raw["fx_lcy_per_usd"].gt(0)
    raw["lender_commit_usd_mil"] = np.where(
        cond_commit,
        raw["lender_commit_value"] / raw["fx_lcy_per_usd"],
        np.nan,
    )
    # USD labels can be used without conversion when fx is unavailable.
    usd_like = raw["lender_commit_ccy"].isin(["USD", "USS"]).fillna(False)
    raw["lender_commit_usd_mil"] = raw["lender_commit_usd_mil"].where(
        raw["lender_commit_usd_mil"].notna(),
        np.where(
            usd_like & raw["lender_commit_value"].gt(0), raw["lender_commit_value"], np.nan
        ),
    )

    # Deduplicate key collisions by preferring rows with valid commitment.
    raw = raw.sort_values(
        ["_k_deal", "_k_tranche", "_k_lender", "lender_commit_usd_mil"],
        na_position="first",
    )
    raw = raw.drop_duplicates(["_k_deal", "_k_tranche", "_k_lender"], keep="last")
    out = raw[
        [
            "_k_deal",
            "_k_tranche",
            "_k_lender",
            "lender_commit_ccy",
            "lender_commit_value",
            "fx_lcy_per_usd",
            "lender_commit_usd_mil",
            "primary_role",
            "additional_roles",
            "lead_arranger",
        ]
    ].copy()
    return out.rename(
        columns={
            "primary_role": "primary_role_raw",
            "additional_roles": "additional_roles_raw",
            "lead_arranger": "lead_arranger_raw",
        }
    )


def _trim_p1p99(series: pd.Series) -> tuple[pd.Series, float, float]:
    s = pd.to_numeric(series, errors="coerce")
    base = s[s.notna() & (s > 0)]
    if base.empty:
        return s, np.nan, np.nan
    q01 = float(base.quantile(0.01))
    q99 = float(base.quantile(0.99))
    return s.clip(lower=q01, upper=q99), q01, q99


def _allocate_commitments_within_tranche(lt: pd.DataFrame) -> pd.DataFrame:
    """
    Reconcile lender commitments to tranche totals at lender-tranche granularity.

    Baseline commitment:
      - Raw commit (USD) preferred over share-implied commit.
      - Fallback rules (existing baseline): residual single-missing, single-lender,
        and role-aware 60/40 split when no lender-level breakdown exists
        (equal split fallback if no lead is identified).

    Hierarchy variants:
      - Preserve observed + residual/single-lender fallback first.
      - Allocate remaining residual by lead/participant sets identified from
        role hierarchy and Ivashina fallback.
      - Export 60/40, fixed-average (14.3%, 26%), and weighted-average variants.
    """
    lt = lt.copy()
    gcols = ["_k_deal", "_k_tranche", "_k_year", "_k_quarter"]

    lt["tranche_amt_eff_mil"] = pd.to_numeric(lt["tranche_amt_eff_mil"], errors="coerce")
    lt["lender_share_num"] = pd.to_numeric(lt["lender_share_num"], errors="coerce")
    lt["lender_commit_raw_usd_mil"] = pd.to_numeric(
        lt["lender_commit_raw_usd_mil"], errors="coerce"
    )
    lt["lender_commit_share_mil"] = pd.to_numeric(
        lt["lender_commit_share_mil"], errors="coerce"
    )
    lt["primary_role"] = lt.get("primary_role", pd.Series("", index=lt.index)).astype(
        "string"
    )
    lt["additional_roles"] = lt.get(
        "additional_roles", pd.Series("", index=lt.index)
    ).astype("string")

    rolef = _lead_rank_features(lt["primary_role"], lt["additional_roles"])
    lt["lead_rank_best"] = rolef["lead_rank_best"]
    lt["lead_stage_fallback"] = rolef["lead_stage_fallback"]

    g = lt.groupby(gcols, observed=True)
    lt["n_lenders_tranche"] = g["_k_lender"].transform("nunique").astype("float64")
    lt["_tranche_amt"] = g["tranche_amt_eff_mil"].transform("max")
    lt["_min_rank_tranche"] = g["lead_rank_best"].transform("min")
    lt["is_lead_ranked"] = (
        lt["lead_rank_best"].notna()
        & lt["_min_rank_tranche"].notna()
        & lt["lead_rank_best"].eq(lt["_min_rank_tranche"])
    ).astype("int8")
    lt["_has_ranked_lead_tranche"] = g["is_lead_ranked"].transform("max")
    lt["_min_fallback_stage_tranche"] = g["lead_stage_fallback"].transform("min")
    lt["is_lead_ivashina_fallback"] = (
        lt["_has_ranked_lead_tranche"].eq(0)
        & lt["lead_stage_fallback"].notna()
        & lt["_min_fallback_stage_tranche"].notna()
        & lt["lead_stage_fallback"].eq(lt["_min_fallback_stage_tranche"])
    ).astype("int8")
    lt["is_lead_final"] = (
        lt["is_lead_ranked"].eq(1) | lt["is_lead_ivashina_fallback"].eq(1)
    ).astype("int8")
    lt["is_participant"] = (lt["is_lead_final"].eq(0)).astype("int8")
    lt["n_leads_tranche"] = g["is_lead_final"].transform("sum").astype("float64")
    lt["n_participants_tranche"] = (
        lt["n_lenders_tranche"] - lt["n_leads_tranche"]
    ).astype("float64")

    raw = lt["lender_commit_raw_usd_mil"].where(lambda s: s > 0)
    share = lt["lender_share_num"].where(lambda s: s > 0)
    lt["_raw"] = raw.fillna(0.0)
    lt["_share"] = share.fillna(0.0)
    lt["_raw_pos"] = raw.notna().astype("int8")
    lt["_share_pos"] = share.notna().astype("int8")

    lt["lender_commit_obs_raw"] = raw.where(raw.notna(), lt["lender_commit_share_mil"])
    lt["_obs_raw"] = pd.to_numeric(lt["lender_commit_obs_raw"], errors="coerce").where(
        lambda s: s > 0
    )
    lt["_obs_raw_pos"] = lt["_obs_raw"].fillna(0.0)
    lt["commit_source_raw"] = (
        raw.notna() & lt["lender_commit_obs_raw"].notna()
    ).astype("int8")
    lt["commit_source_share"] = (
        raw.isna() & lt["lender_commit_share_mil"].notna()
    ).astype("int8")

    # Strict observed commitments: no scaling to tranche total.
    lt["lender_commit_tranche_mil"] = lt["_obs_raw"]
    lt["flag_lender_commit_obs"] = (
        pd.Series(lt["lender_commit_tranche_mil"]).gt(0).fillna(False).astype("int8")
    )

    # Imputed commitment fallback — rules applied in order:
    #
    # Rule 1 (residual): if exactly one lender in the tranche is missing a
    #   commitment AND the remaining lenders' observed sum is < tranche total,
    #   assign the positive residual (tranche_total - observed_sum) to that
    #   lender.  No minimum size threshold — any positive residual is assigned.
    #
    # Rule 2 (single-lender): if a tranche (or deal acting as a single tranche)
    #   has exactly ONE lender row and that lender has no observed commitment,
    #   assign the full tranche amount.  This covers solo-lender bilateral loans.
    #
    # Rule 3 (role-aware no-breakdown): if a tranche has no lender-level
    #   observed breakdown at all (no raw commit and no share-implied commit for
    #   every lender row), use role-aware 60/40 split across identified
    #   leads/participants; if no lead is identified, fall back to equal split.
    lt["_obs_sum_tranche"] = g["_obs_raw_pos"].transform("sum").astype("float64")
    lt["_n_lenders_tranche"] = g["_k_lender"].transform("nunique").astype("float64")
    lt["_n_missing_tranche"] = (
        g["flag_lender_commit_obs"].transform("size")
        - g["flag_lender_commit_obs"].transform("sum")
    ).astype("float64")
    lt["_n_raw_pos_tranche"] = g["_raw_pos"].transform("sum").astype("float64")
    lt["_n_share_pos_tranche"] = g["_share_pos"].transform("sum").astype("float64")
    lt["_residual_tranche"] = lt["_tranche_amt"] - lt["_obs_sum_tranche"]

    # Rule 1: exactly one missing lender, residual > 0
    can_single_resid = (
        lt["_obs_raw"].isna()
        & lt["_tranche_amt"].gt(0)
        & lt["_n_missing_tranche"].eq(1)
        & lt["_residual_tranche"].gt(0)
    )

    # Rule 2: single-lender tranche, no observed commit — assign full tranche amount
    can_single_lender = (
        lt["_obs_raw"].isna()
        & lt["_n_lenders_tranche"].eq(1)
        & lt["_tranche_amt"].gt(0)
    )

    # Rule 3 condition: no lender-level breakdown exists
    no_breakdown_tranche = lt["_n_raw_pos_tranche"].eq(0) & lt["_n_share_pos_tranche"].eq(0)
    can_equal_split = (
        lt["_obs_raw"].isna()
        & lt["_tranche_amt"].gt(0)
        & lt["_n_lenders_tranche"].gt(1)
        & no_breakdown_tranche
    )
    can_hierarchy_split = can_equal_split & lt["n_leads_tranche"].gt(0)
    can_equal_split_nolead = can_equal_split & lt["n_leads_tranche"].eq(0)

    # Rule 3 allocation:
    #  - if lead identified: 60/40 lead/participant split
    #  - if no participants among missing rows: allocate all to leads
    #  - if no lead identified: equal split across all lenders
    rule3_alloc = pd.Series(np.nan, index=lt.index, dtype="float64")
    lead_row = lt["is_lead_final"].eq(1)
    has_participant = lt["n_participants_tranche"].gt(0)
    has_no_participant = lt["n_participants_tranche"].eq(0)

    r3_both = can_hierarchy_split & has_participant
    rule3_alloc.loc[r3_both & lead_row] = (
        0.60
        * lt.loc[r3_both & lead_row, "_tranche_amt"]
        / lt.loc[r3_both & lead_row, "n_leads_tranche"]
    )
    rule3_alloc.loc[r3_both & (~lead_row)] = (
        0.40
        * lt.loc[r3_both & (~lead_row), "_tranche_amt"]
        / lt.loc[r3_both & (~lead_row), "n_participants_tranche"]
    )

    r3_all_leads = can_hierarchy_split & has_no_participant & lead_row
    rule3_alloc.loc[r3_all_leads] = (
        lt.loc[r3_all_leads, "_tranche_amt"] / lt.loc[r3_all_leads, "n_leads_tranche"]
    )

    rule3_alloc.loc[can_equal_split_nolead] = (
        lt.loc[can_equal_split_nolead, "_tranche_amt"]
        / lt.loc[can_equal_split_nolead, "_n_lenders_tranche"]
    )

    # Apply rules: Rule 1 first, then Rule 2, then Rule 3 for remaining missing
    _imp = lt["_obs_raw"].copy()
    _imp = _imp.where(_imp.notna(), np.where(can_single_resid, lt["_residual_tranche"], np.nan))
    _imp = _imp.where(_imp.notna(), np.where(can_single_lender, lt["_tranche_amt"], np.nan))
    _imp = _imp.where(
        _imp.notna(),
        np.where(can_equal_split, rule3_alloc, np.nan),
    )

    lt["lender_commit_tranche_mil_imp"] = _imp
    lt["commit_source_prorata"] = can_single_resid.astype("int8")
    lt["commit_source_singlelender"] = can_single_lender.astype("int8")
    lt["commit_source_equalsplit"] = can_equal_split.astype("int8")
    lt["commit_source_hierarchysplit"] = can_hierarchy_split.astype("int8")
    lt["commit_source_equalsplit_nolead"] = can_equal_split_nolead.astype("int8")
    lt["lender_commit_pro_rata_mil"] = np.nan
    lt["flag_lender_commit_imp_obs"] = (
        pd.Series(lt["lender_commit_tranche_mil_imp"]).gt(0).fillna(False).astype("int8")
    )

    # ------------------------------------------------------------------
    # Hierarchy variants: observed + residual/single first, then allocate
    # remaining residual by lead/participant rules.
    # ------------------------------------------------------------------
    imp_pre = lt["_obs_raw"].copy()
    imp_pre = imp_pre.where(
        imp_pre.notna(), np.where(can_single_resid, lt["_residual_tranche"], np.nan)
    )
    imp_pre = imp_pre.where(
        imp_pre.notna(), np.where(can_single_lender, lt["_tranche_amt"], np.nan)
    )
    lt["_known_pre"] = imp_pre.fillna(0.0)
    lt["_known_pre_sum_tranche"] = g["_known_pre"].transform("sum").astype("float64")
    lt["_resid_after_pre"] = (
        lt["_tranche_amt"] - lt["_known_pre_sum_tranche"]
    ).clip(lower=0.0)
    lt["_is_missing_pre"] = imp_pre.isna() & lt["_tranche_amt"].gt(0)
    lt["_is_missing_lead_pre"] = lt["_is_missing_pre"] & lt["is_lead_final"].eq(1)
    lt["_is_missing_participant_pre"] = lt["_is_missing_pre"] & lt["is_participant"].eq(1)
    lt["_n_missing_pre"] = g["_is_missing_pre"].transform("sum").astype("float64")
    lt["_n_missing_lead_pre"] = g["_is_missing_lead_pre"].transform("sum").astype("float64")
    lt["_n_missing_participant_pre"] = (
        g["_is_missing_participant_pre"].transform("sum").astype("float64")
    )

    # Weighted lead-share alpha from observed-share tranches.
    # NOTE: This is a global weighted-average lead share, not a bank-specific
    # lender-history estimate. Bank-specific history would require iterating over
    # each lender's prior tranches. Global average is used as a tractable
    # approximation.
    lt["_lead_share_pos"] = np.where(lt["is_lead_final"].eq(1), lt["_share"], 0.0)
    tr = (
        lt.groupby(gcols, observed=True, as_index=False)
        .agg(
            tranche_amt=("_tranche_amt", "max"),
            share_sum=("_share", "sum"),
            lead_share_sum=("_lead_share_pos", "sum"),
            has_lead=("is_lead_final", "max"),
        )
        .copy()
    )
    elig = (
        tr["tranche_amt"].gt(0)
        & tr["share_sum"].gt(0)
        & tr["has_lead"].gt(0)
        & tr["lead_share_sum"].ge(0)
    )
    alpha_weighted = np.nan
    if bool(elig.any()):
        ratios = (tr.loc[elig, "lead_share_sum"] / tr.loc[elig, "share_sum"]).clip(0.0, 1.0)
        weights = tr.loc[elig, "tranche_amt"]
        if float(weights.sum()) > 0:
            alpha_weighted = float(np.average(ratios, weights=weights))
    alpha_weighted_used = (
        float(np.clip(alpha_weighted, 0.0, 1.0)) if pd.notna(alpha_weighted) else 0.60
    )
    lt["lead_share_alpha_weighted"] = alpha_weighted_used
    lt["lead_share_alpha_weighted_raw"] = alpha_weighted

    def _alloc_hierarchy(alpha: float) -> tuple[pd.Series, pd.Series]:
        alpha = float(np.clip(alpha, 0.0, 1.0))
        out = imp_pre.copy()

        missing = lt["_is_missing_pre"].copy()
        resid = lt["_resid_after_pre"].copy()
        n_missing = lt["_n_missing_pre"].copy()
        n_ml = lt["_n_missing_lead_pre"].copy()
        n_mp = lt["_n_missing_participant_pre"].copy()
        n_leads = lt["n_leads_tranche"].copy()

        active = missing & resid.gt(0) & n_missing.gt(0)
        lead_row = lt["is_lead_final"].eq(1)
        no_leads = n_leads.eq(0)

        w = pd.Series(0.0, index=lt.index, dtype="float64")
        # No lead identified: equal split among missing lenders.
        m0 = active & no_leads
        w.loc[m0] = 1.0 / n_missing.loc[m0]

        # Both missing lead and participant rows present: alpha split.
        mb = active & n_ml.gt(0) & n_mp.gt(0) & (~no_leads)
        w.loc[mb & lead_row] = alpha / n_ml.loc[mb & lead_row]
        w.loc[mb & (~lead_row)] = (1.0 - alpha) / n_mp.loc[mb & (~lead_row)]

        # Only missing leads remain: allocate all residual to missing leads.
        ml = active & n_ml.gt(0) & n_mp.eq(0) & (~no_leads)
        w.loc[ml & lead_row] = 1.0 / n_ml.loc[ml & lead_row]

        # Only missing participants remain: allocate all residual to participants.
        mp = active & n_mp.gt(0) & n_ml.eq(0) & (~no_leads)
        w.loc[mp & (~lead_row)] = 1.0 / n_mp.loc[mp & (~lead_row)]

        alloc = resid * w
        source = (active & alloc.gt(0)).astype("int8")
        out = out.where(out.notna(), np.where(source.eq(1), alloc, np.nan))
        return out, source

    imp_h6040, src_h6040 = _alloc_hierarchy(0.60)
    imp_h143, src_h143 = _alloc_hierarchy(0.143)
    imp_h260, src_h260 = _alloc_hierarchy(0.260)
    imp_hw, src_hw = _alloc_hierarchy(alpha_weighted_used)

    lt["lender_commit_imp_h6040"] = imp_h6040
    lt["lender_commit_imp_h143"] = imp_h143
    lt["lender_commit_imp_h260"] = imp_h260
    lt["lender_commit_imp_hw"] = imp_hw
    lt["src_hier_6040"] = src_h6040
    lt["src_hier_143"] = src_h143
    lt["src_hier_260"] = src_h260
    lt["src_hier_hw"] = src_hw

    keep_cols = [
        "_k_deal",
        "_k_tranche",
        "_k_year",
        "_k_quarter",
        "_k_lender",
        "n_lenders_tranche",
        "lender_commit_tranche_mil",
        "lender_commit_pro_rata_mil",
        "lender_commit_tranche_mil_imp",
        "flag_lender_commit_obs",
        "flag_lender_commit_imp_obs",
        "lead_rank_best",
        "is_lead_ranked",
        "is_lead_ivashina_fallback",
        "is_participant",
        "n_leads_tranche",
        "n_participants_tranche",
        "lead_share_alpha_weighted",
        "lead_share_alpha_weighted_raw",
        "commit_source_raw",
        "commit_source_share",
        "commit_source_prorata",
        "commit_source_singlelender",
        "commit_source_equalsplit",
        "commit_source_hierarchysplit",
        "commit_source_equalsplit_nolead",
        "lender_commit_imp_h6040",
        "lender_commit_imp_h143",
        "lender_commit_imp_h260",
        "lender_commit_imp_hw",
        "src_hier_6040",
        "src_hier_143",
        "src_hier_260",
        "src_hier_hw",
    ]
    return lt[keep_cols].copy()


def main() -> None:
    if not IN_PANEL.exists():
        raise FileNotFoundError(f"Input panel not found: {IN_PANEL}")

    cols = [
        *MERGE_KEYS,
        "deal_date",
        "all_in_spread_drawn_bps",
        "all_in_spread_drawn_pct",
        "tranche_amount_converted",
        "loan_amount_mil",
        "lender_share",
        "primary_role",
        "lead_arranger",
    ]
    df = pd.read_parquet(IN_PANEL, columns=cols)
    _require_columns(df, MERGE_KEYS)
    df["_k_deal"] = _normalize_key(df["deal_id"])
    df["_k_tranche"] = _normalize_key(df["tranche_id"])
    df["_k_lender"] = _normalize_key(df["lender_id"])
    df["_k_year"] = pd.to_numeric(df["year"], errors="coerce")
    df["_k_quarter"] = pd.to_numeric(df["quarter"], errors="coerce")

    raw_commit = _build_raw_commit_sidecar()
    df = df.merge(
        raw_commit,
        on=["_k_deal", "_k_tranche", "_k_lender"],
        how="left",
        validate="many_to_one",
    )
    df["primary_role_use"] = (
        df["primary_role"]
        .astype("string")
        .where(df["primary_role"].astype("string").str.strip().ne(""), df["primary_role_raw"])
    )
    df["additional_roles_use"] = df["additional_roles_raw"].astype("string")

    # Use tranche amount as primary, fallback to loan_amount_mil.
    tranche_amt = pd.to_numeric(df["tranche_amount_converted"], errors="coerce")
    loan_amt = pd.to_numeric(df["loan_amount_mil"], errors="coerce")
    df["tranche_amt_eff_mil"] = tranche_amt.where(tranche_amt > 0)
    tranche_fill = (
        df.groupby(
            ["_k_deal", "_k_tranche", "_k_year", "_k_quarter"], observed=True
        )["tranche_amt_eff_mil"]
        .transform("max")
        .astype("float64")
    )
    df["tranche_amt_eff_mil"] = df["tranche_amt_eff_mil"].where(
        df["tranche_amt_eff_mil"].notna(), tranche_fill
    )
    # Last-resort fallback only when tranche amount is unavailable for the full tranche.
    df["tranche_amt_eff_mil"] = df["tranche_amt_eff_mil"].where(
        df["tranche_amt_eff_mil"].notna(), loan_amt
    )

    spread_bps = pd.to_numeric(df["all_in_spread_drawn_bps"], errors="coerce")
    spread_pct = pd.to_numeric(df["all_in_spread_drawn_pct"], errors="coerce")
    spread_pct = spread_pct.where(spread_pct.notna(), spread_bps / 100.0)

    df["lender_share_num"] = _parse_lender_share(df["lender_share"])
    df["lender_commit_share_mil"] = (
        df["lender_share_num"] * df["tranche_amt_eff_mil"]
    ).where((df["lender_share_num"] > 0) & (df["tranche_amt_eff_mil"] > 0))
    df["lender_commit_raw_usd_mil"] = pd.to_numeric(
        df["lender_commit_usd_mil"], errors="coerce"
    ).where(lambda s: s > 0)

    # Build commitments at lender-tranche level, then map back to row-level panel.
    lt = (
        df.groupby(
            ["_k_deal", "_k_tranche", "_k_year", "_k_quarter", "_k_lender"],
            observed=True,
            as_index=False,
        )
        .agg(
            tranche_amt_eff_mil=("tranche_amt_eff_mil", "max"),
            lender_share_num=("lender_share_num", "max"),
            lender_commit_share_mil=("lender_commit_share_mil", "max"),
            lender_commit_raw_usd_mil=("lender_commit_raw_usd_mil", "max"),
            primary_role=("primary_role_use", "first"),
            additional_roles=("additional_roles_use", "first"),
        )
        .copy()
    )
    lt_alloc = _allocate_commitments_within_tranche(lt)
    alloc_cols = [
        "_k_deal",
        "_k_tranche",
        "_k_year",
        "_k_quarter",
        "_k_lender",
        "n_lenders_tranche",
        "n_leads_tranche",
        "n_participants_tranche",
        "lead_rank_best",
        "is_lead_ranked",
        "is_lead_ivashina_fallback",
        "is_participant",
        "lead_share_alpha_weighted",
        "lead_share_alpha_weighted_raw",
        "commit_source_raw",
        "commit_source_share",
        "commit_source_prorata",
        "commit_source_singlelender",
        "commit_source_equalsplit",
        "commit_source_hierarchysplit",
        "commit_source_equalsplit_nolead",
        "src_hier_6040",
        "src_hier_143",
        "src_hier_260",
        "src_hier_hw",
        "flag_lender_commit_obs",
        "flag_lender_commit_imp_obs",
        "lender_commit_tranche_mil",
        "lender_commit_pro_rata_mil",
        "lender_commit_tranche_mil_imp",
        "lender_commit_imp_h6040",
        "lender_commit_imp_h143",
        "lender_commit_imp_h260",
        "lender_commit_imp_hw",
    ]
    df = df.merge(
        lt_alloc[alloc_cols],
        on=["_k_deal", "_k_tranche", "_k_year", "_k_quarter", "_k_lender"],
        how="left",
        validate="many_to_one",
    )

    # Strict lender-specific spread: only when lender commitment is known and positive.
    df["spread_bps_lender_strict"] = spread_bps.where(df["flag_lender_commit_obs"] == 1)
    df["spread_pct_lender_strict"] = spread_pct.where(df["flag_lender_commit_obs"] == 1)
    # Imputed lender-specific spread (now identical to strict; no pro-rata fallback).
    df["spread_bps_lender_imputed"] = spread_bps.where(df["flag_lender_commit_imp_obs"] == 1)
    df["spread_pct_lender_imputed"] = spread_pct.where(df["flag_lender_commit_imp_obs"] == 1)

    # Commitment-weighted lender spread within deal-lender-borrower-time cell.
    df["_wx"] = spread_bps * df["lender_commit_tranche_mil"]
    df["_wx_imp"] = spread_bps * df["lender_commit_tranche_mil_imp"]
    grp = df.groupby(GROUP_KEYS, observed=True, as_index=False).agg(
        lender_commit_deal_mil_sum=("lender_commit_tranche_mil", "sum"),
        lender_commit_deal_mil_sum_imp=("lender_commit_tranche_mil_imp", "sum"),
        _wx_sum=("_wx", "sum"),
        _wx_imp_sum=("_wx_imp", "sum"),
        n_tranche_obs_deal_lender=("tranche_id", "nunique"),
        n_tranche_with_commit=("flag_lender_commit_obs", "sum"),
        n_tranche_with_commit_imp=("flag_lender_commit_imp_obs", "sum"),
    )
    den_w = pd.to_numeric(grp["lender_commit_deal_mil_sum"], errors="coerce").replace(
        0.0, np.nan
    )
    grp["spread_bps_lender_w"] = pd.to_numeric(
        grp["_wx_sum"], errors="coerce"
    ) / den_w
    grp["spread_pct_lender_w"] = grp["spread_bps_lender_w"] / 100.0
    den_w_imp = pd.to_numeric(
        grp["lender_commit_deal_mil_sum_imp"], errors="coerce"
    ).replace(0.0, np.nan)
    grp["spread_bps_lender_w_imp"] = pd.to_numeric(
        grp["_wx_imp_sum"], errors="coerce"
    ) / den_w_imp
    grp["spread_pct_lender_w_imp"] = grp["spread_bps_lender_w_imp"] / 100.0
    grp["flag_lender_w_available"] = (grp["lender_commit_deal_mil_sum"] > 0).astype("int8")
    grp["flag_lender_w_imp_available"] = (
        grp["lender_commit_deal_mil_sum_imp"] > 0
    ).astype("int8")
    grp = grp.drop(columns=["_wx_sum", "_wx_imp_sum"])

    df = df.merge(grp, on=GROUP_KEYS, how="left", validate="many_to_one")

    df["spread_pct_lender_strict_t"], strict_q01, strict_q99 = _trim_p1p99(
        df["spread_pct_lender_strict"]
    )
    df["spread_pct_lender_w_t"], w_q01, w_q99 = _trim_p1p99(df["spread_pct_lender_w"])
    df["spread_pct_lender_imputed_t"], imp_q01, imp_q99 = _trim_p1p99(
        df["spread_pct_lender_imputed"]
    )
    df["spread_pct_lender_w_imp_t"], w_imp_q01, w_imp_q99 = _trim_p1p99(
        df["spread_pct_lender_w_imp"]
    )

    # Keep only merge keys + lender-spread variables as sidecar controls.
    out_cols = [
        *MERGE_KEYS,
        "deal_date",
        "lender_share_num",
        "lender_commit_ccy",
        "lender_commit_value",
        "fx_lcy_per_usd",
        "lender_commit_raw_usd_mil",
        "lender_commit_share_mil",
        "commit_source_raw",
        "commit_source_share",
        "commit_source_prorata",
        "commit_source_singlelender",
        "commit_source_equalsplit",
        "commit_source_hierarchysplit",
        "commit_source_equalsplit_nolead",
        "src_hier_6040",
        "src_hier_143",
        "src_hier_260",
        "src_hier_hw",
        "n_lenders_tranche",
        "n_leads_tranche",
        "n_participants_tranche",
        "lead_rank_best",
        "is_lead_ranked",
        "is_lead_ivashina_fallback",
        "is_participant",
        "lead_share_alpha_weighted",
        "lead_share_alpha_weighted_raw",
        "flag_lender_commit_obs",
        "flag_lender_commit_imp_obs",
        "lender_commit_tranche_mil",
        "lender_commit_pro_rata_mil",
        "lender_commit_tranche_mil_imp",
        "lender_commit_imp_h6040",
        "lender_commit_imp_h143",
        "lender_commit_imp_h260",
        "lender_commit_imp_hw",
        "lender_commit_deal_mil_sum",
        "lender_commit_deal_mil_sum_imp",
        "n_tranche_obs_deal_lender",
        "n_tranche_with_commit",
        "n_tranche_with_commit_imp",
        "flag_lender_w_available",
        "flag_lender_w_imp_available",
        "spread_bps_lender_strict",
        "spread_pct_lender_strict",
        "spread_pct_lender_strict_t",
        "spread_bps_lender_imputed",
        "spread_pct_lender_imputed",
        "spread_pct_lender_imputed_t",
        "spread_bps_lender_w",
        "spread_pct_lender_w",
        "spread_pct_lender_w_t",
        "spread_bps_lender_w_imp",
        "spread_pct_lender_w_imp",
        "spread_pct_lender_w_imp_t",
    ]
    out = df[out_cols].copy()
    if "lender_commit_ccy" in out.columns:
        out["lender_commit_ccy"] = out["lender_commit_ccy"].astype(object)

    # Resolve known duplicate key rows by keeping latest observed deal_date.
    out["deal_date"] = pd.to_datetime(out["deal_date"], errors="coerce")
    dup_pre = out.duplicated(MERGE_KEYS).sum()
    if dup_pre > 0:
        out = (
            out.sort_values(MERGE_KEYS + ["deal_date"], na_position="first")
            .drop_duplicates(MERGE_KEYS, keep="last")
            .copy()
        )
    out = out.drop(columns=["deal_date"])
    for c in out.columns:
        if c in MERGE_KEYS or c == "lender_commit_ccy":
            continue
        out[c] = pd.to_numeric(out[c], errors="coerce").astype("float64")

    # Safety: merge keys must be unique in sidecar after dedupe.
    dup = out.duplicated(MERGE_KEYS).sum()
    if dup > 0:
        raise ValueError(f"Sidecar merge keys are not unique: {dup} duplicates.")

    OUT_DTA.parent.mkdir(parents=True, exist_ok=True)
    OUT_DIAG.parent.mkdir(parents=True, exist_ok=True)

    out.to_stata(OUT_DTA, write_index=False, version=118)

    strict_avail = out["spread_pct_lender_strict"].notna()
    imputed_avail = out["spread_pct_lender_imputed"].notna()
    weighted_avail = out["spread_pct_lender_w"].notna()
    weighted_imp_avail = out["spread_pct_lender_w_imp"].notna()
    raw_commit_avail = out["lender_commit_raw_usd_mil"].notna()
    share_commit_avail = out["lender_commit_share_mil"].notna()
    any_commit_avail = out["lender_commit_tranche_mil"].notna()
    any_commit_imp_avail = out["lender_commit_tranche_mil_imp"].notna()
    any_commit_h6040_avail = out["lender_commit_imp_h6040"].notna()
    any_commit_havg143_avail = out["lender_commit_imp_h143"].notna()
    any_commit_havg260_avail = out["lender_commit_imp_h260"].notna()
    any_commit_havgw_avail = out["lender_commit_imp_hw"].notna()
    prorata_used = out["commit_source_prorata"] == 1
    equalsplit_used = out["commit_source_equalsplit"] == 1
    hier_6040_used = out["src_hier_6040"] == 1
    hier_avg143_used = out["src_hier_143"] == 1
    hier_avg260_used = out["src_hier_260"] == 1
    hier_avgw_used = out["src_hier_hw"] == 1
    lead_ranked = out["is_lead_ranked"] == 1
    lead_ivashina = out["is_lead_ivashina_fallback"] == 1

    diag = pd.DataFrame(
        [
            {"metric": "rows_total", "value": float(len(out))},
            {
                "metric": "raw_commit_available_rows",
                "value": float(raw_commit_avail.sum()),
            },
            {
                "metric": "raw_commit_available_pct",
                "value": float(raw_commit_avail.mean() * 100),
            },
            {
                "metric": "share_commit_available_rows",
                "value": float(share_commit_avail.sum()),
            },
            {
                "metric": "share_commit_available_pct",
                "value": float(share_commit_avail.mean() * 100),
            },
            {
                "metric": "any_commit_available_rows",
                "value": float(any_commit_avail.sum()),
            },
            {
                "metric": "any_commit_available_pct",
                "value": float(any_commit_avail.mean() * 100),
            },
            {
                "metric": "any_commit_imputed_available_rows",
                "value": float(any_commit_imp_avail.sum()),
            },
            {
                "metric": "any_commit_imputed_available_pct",
                "value": float(any_commit_imp_avail.mean() * 100),
            },
            {
                "metric": "any_commit_h6040_available_rows",
                "value": float(any_commit_h6040_avail.sum()),
            },
            {
                "metric": "any_commit_h6040_available_pct",
                "value": float(any_commit_h6040_avail.mean() * 100),
            },
            {
                "metric": "any_commit_havg143_available_rows",
                "value": float(any_commit_havg143_avail.sum()),
            },
            {
                "metric": "any_commit_havg143_available_pct",
                "value": float(any_commit_havg143_avail.mean() * 100),
            },
            {
                "metric": "any_commit_havg260_available_rows",
                "value": float(any_commit_havg260_avail.sum()),
            },
            {
                "metric": "any_commit_havg260_available_pct",
                "value": float(any_commit_havg260_avail.mean() * 100),
            },
            {
                "metric": "any_commit_havgw_available_rows",
                "value": float(any_commit_havgw_avail.sum()),
            },
            {
                "metric": "any_commit_havgw_available_pct",
                "value": float(any_commit_havgw_avail.mean() * 100),
            },
            {
                "metric": "prorata_used_rows",
                "value": float(prorata_used.sum()),
            },
            {
                "metric": "prorata_used_pct",
                "value": float(prorata_used.mean() * 100),
            },
            {
                "metric": "equalsplit_used_rows",
                "value": float(equalsplit_used.sum()),
            },
            {
                "metric": "equalsplit_used_pct",
                "value": float(equalsplit_used.mean() * 100),
            },
            {
                "metric": "hier_6040_used_rows",
                "value": float(hier_6040_used.sum()),
            },
            {
                "metric": "hier_6040_used_pct",
                "value": float(hier_6040_used.mean() * 100),
            },
            {
                "metric": "hier_avg143_used_rows",
                "value": float(hier_avg143_used.sum()),
            },
            {
                "metric": "hier_avg143_used_pct",
                "value": float(hier_avg143_used.mean() * 100),
            },
            {
                "metric": "hier_avg260_used_rows",
                "value": float(hier_avg260_used.sum()),
            },
            {
                "metric": "hier_avg260_used_pct",
                "value": float(hier_avg260_used.mean() * 100),
            },
            {
                "metric": "hier_avgw_used_rows",
                "value": float(hier_avgw_used.sum()),
            },
            {
                "metric": "hier_avgw_used_pct",
                "value": float(hier_avgw_used.mean() * 100),
            },
            {
                "metric": "lead_ranked_rows",
                "value": float(lead_ranked.sum()),
            },
            {
                "metric": "lead_ranked_pct",
                "value": float(lead_ranked.mean() * 100),
            },
            {
                "metric": "lead_ivashina_fallback_rows",
                "value": float(lead_ivashina.sum()),
            },
            {
                "metric": "lead_ivashina_fallback_pct",
                "value": float(lead_ivashina.mean() * 100),
            },
            {
                "metric": "alpha_weighted_used",
                "value": float(out["lead_share_alpha_weighted"].dropna().iloc[0])
                if out["lead_share_alpha_weighted"].notna().any()
                else np.nan,
            },
            {
                "metric": "alpha_weighted_raw",
                "value": float(out["lead_share_alpha_weighted_raw"].dropna().iloc[0])
                if out["lead_share_alpha_weighted_raw"].notna().any()
                else np.nan,
            },
            {
                "metric": "strict_available_rows",
                "value": float(strict_avail.sum()),
            },
            {
                "metric": "strict_available_pct",
                "value": float(strict_avail.mean() * 100),
            },
            {
                "metric": "imputed_available_rows",
                "value": float(imputed_avail.sum()),
            },
            {
                "metric": "imputed_available_pct",
                "value": float(imputed_avail.mean() * 100),
            },
            {
                "metric": "weighted_available_rows",
                "value": float(weighted_avail.sum()),
            },
            {
                "metric": "weighted_available_pct",
                "value": float(weighted_avail.mean() * 100),
            },
            {
                "metric": "weighted_imputed_available_rows",
                "value": float(weighted_imp_avail.sum()),
            },
            {
                "metric": "weighted_imputed_available_pct",
                "value": float(weighted_imp_avail.mean() * 100),
            },
            {"metric": "strict_trim_q01", "value": strict_q01},
            {"metric": "strict_trim_q99", "value": strict_q99},
            {"metric": "weighted_trim_q01", "value": w_q01},
            {"metric": "weighted_trim_q99", "value": w_q99},
            {"metric": "imputed_trim_q01", "value": imp_q01},
            {"metric": "imputed_trim_q99", "value": imp_q99},
            {"metric": "weighted_imputed_trim_q01", "value": w_imp_q01},
            {"metric": "weighted_imputed_trim_q99", "value": w_imp_q99},
        ]
    )
    diag.to_csv(OUT_DIAG, index=False)

    print(f"Saved lender spread controls: {OUT_DTA}")
    print(f"Saved diagnostics: {OUT_DIAG}")
    print(f"Merge-key duplicates resolved: {dup_pre}")
    print(
        "Coverage: raw_commit={:.2f}% share_commit={:.2f}% any_commit={:.2f}% any_commit_imputed={:.2f}% h6040={:.2f}% havg143={:.2f}% havg260={:.2f}% havgw={:.2f}% strict={:.2f}% imputed={:.2f}% weighted={:.2f}% weighted_imputed={:.2f}%".format(
            raw_commit_avail.mean() * 100,
            share_commit_avail.mean() * 100,
            any_commit_avail.mean() * 100,
            any_commit_imp_avail.mean() * 100,
            any_commit_h6040_avail.mean() * 100,
            any_commit_havg143_avail.mean() * 100,
            any_commit_havg260_avail.mean() * 100,
            any_commit_havgw_avail.mean() * 100,
            strict_avail.mean() * 100,
            imputed_avail.mean() * 100,
            weighted_avail.mean() * 100,
            weighted_imp_avail.mean() * 100,
        )
    )


if __name__ == "__main__":
    main()
