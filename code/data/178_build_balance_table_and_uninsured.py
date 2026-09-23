#!/usr/bin/env python3
"""
178_build_balance_table_and_uninsured.py

Purpose:
  1. Construct bank-quarter-level balance sheet variables for a sample
     balance table (summary statistics of lender characteristics).
  2. Construct uninsured deposit share from Call Reports and create a
     bank-quarter panel for Stata regression merge.

Uninsured deposit definitions:
  Primary DSSW-style measure:
    unins_share_dss = RCON5597 / RCON2200

  Numerator:
    Estimated uninsured deposits from Schedule RC-O, Memorandum item 2,
    i.e. MDRM code RCON5597 ("Estimated amount of uninsured assessable
    deposits in domestic offices of the bank and in insured branches in
    Puerto Rico and U.S. territories and possessions, including related
    interest accrued and unpaid"). This is the direct Call Report field
    for deposits not covered by federal deposit insurance, reported by
    banks subject to the RC-O uninsured-deposit reporting requirement.

    RCONF049 (amount of non-retirement accounts of $250,000 or less,
    i.e. Schedule RC-O Memorandum 1.a.(1)) is NOT the uninsured-deposit
    estimate. F049 is a small-account measure and is kept only as a
    diagnostic field.

  Denominator:
    (A) DSSW: RCON2200 (total domestic deposits) — primary
    (B) Emin et al. (2025): RCON2948 (total liabilities) — robustness

  DSSW define UninsuredDepositShare as the bank's ratio of uninsured
  deposits to domestic deposits, averaged over the beta-estimation period.
  Here the same Call Report ratio is constructed at bank-quarter frequency
  and can then be averaged to the bank-year or bank level before merging
  to loan-level regressions. The complementary "insured share" used in
  heterogeneity tests is 1 - unins_share_dss, an inferred complement to
  the DSSW-style uninsured share rather than a separately observed insured
  dollar amount.

  Both shares are winsorized at 1st and 99th percentiles. The thesis
  regressions keep direct RCON5597 observations from 2006 onward for loan
  panel coverage; an exact DSSW replication should restrict to their
  June 2015-March 2024 Call Report sample and additional bank-type screens.
  References: Drechsler, Savov, Schnabl, and Wang (2023, 2026);
              Emin, Harris, Kaplan, Klingler (2025);
              Jiang, Matvos, Piskorski, Seru (2024).
  FFIEC 031 Schedule RC-O Memorandum item 2 maps to MDRM code RCON5597
  (see data/documentation/fdic_call_reports/FFIEC031_202512_f.pdf line 3067).

Balance table variables:
  Total assets ($M), Deposits/Assets, Loans/Assets, Net Income/Assets,
  Capital ratio, Equity/Assets, ROA (annualized), ROE (annualized),
  NPL/Assets, LLP/Assets (annualized), Liquidity ratio,
  Uninsured deposit share.

Outputs:
  - data/final/bank_uninsured_deposits.dta  (bank x quarter, for Stata)
  - regression/overview_diagnostics/balance_table.csv
  - regression/overview_diagnostics/balance_table_by_hhi.csv
  - regression/overview_diagnostics/balance_table_by_uninsured.csv
"""

import pandas as pd
import numpy as np
import os

from pathlib import Path
BASE = str(Path(os.environ.get("MPHIL_THESIS_ROOT", Path(__file__).resolve().parents[2])))

# ═══════════════════════════════════════════════════════════════
# Section 1: Load data
# ═══════════════════════════════════════════════════════════════

print("── Section 1: Loading data ──")

cr_cols = ['RSSDID', 'year', 'quarter',
           'RCON2170', 'RCFD2170',   # domestic / consolidated assets ($000s)
           'RCON2200',                # total domestic deposits ($000s)
           'RCON2122',                # total loans and leases ($000s)
           'RCON3210',                # total equity capital ($000s)
           'RIAD4340',                # net income (YTD, $000s)
           'RIAD4230',                # provision for loan and lease losses (YTD, $000s)
           'RCON0010',                # cash and due from depository institutions ($000s)
           'RCON1403', 'RCFD1403',    # loans past due 90+ days, still accruing
           'RCON1407', 'RCFD1407',    # nonaccrual loans
           'total_assets_mil', 'total_deposits_mil', 'total_loans_mil',
           'equity_capital_mil', 'capital_ratio', 'deposits_to_assets',
           'loans_to_assets']

cr = pd.read_parquet(
    f'{BASE}/data/intermediate/clean/fdic_call_reports/fdic_call_reports_clean.parquet',
    columns=cr_cols)
print(f"  Clean Call Reports: {len(cr):,} rows, {cr['RSSDID'].nunique():,} banks")

# Load uninsured deposit components from raw files.
# Primary numerator:
#   RCON5597 (Schedule RC-O Memorandum item 2):
#     direct Call Report estimate of deposits not covered by federal deposit
#     insurance. This script uses reported RCON5597 directly and does not
#     construct uninsured deposits as a residual from small-account fields.
# Diagnostic fields:
#   RCONF049 = amount of non-retirement accounts of $250,000 or less
#              (Memorandum 1.a.(1)); kept for diagnostics only. This is
#              NOT the uninsured estimate.
#   RCONF051 = amount of non-retirement accounts of more than $250,000
#              (Memorandum 1.b.(1)); proxy cap-exceeding balance share.
#   RCONF047 = amount of retirement accounts > $250,000 (Memorandum 1.d.(1)).
print("  Loading RCON5597 + RCONF049/F051/F047 from raw RCON file 1...")
raw1 = pd.read_parquet(
    f'{BASE}/data/raw/fdic_call_reports/wrds_call_rcon_1__2001_2025.parquet',
    columns=['rssd9001', 'wrdsreportdate',
             'rcon5597', 'rconf049', 'rconf051', 'rconf047'])
print("  Loading RCON2200 + RCON2948 from raw RCON file 2...")
raw2_dep = pd.read_parquet(
    f'{BASE}/data/raw/fdic_call_reports/wrds_call_rcon_2__2001_2025.parquet',
    columns=['rssd9001', 'wrdsreportdate', 'rcon2200', 'rcon2948'])
raw1 = raw1.merge(raw2_dep, on=['rssd9001', 'wrdsreportdate'], how='left')
raw1['wrdsreportdate'] = pd.to_datetime(raw1['wrdsreportdate'])
raw1['year'] = raw1['wrdsreportdate'].dt.year
raw1['quarter'] = raw1['wrdsreportdate'].dt.quarter
raw1.rename(columns={
    'rssd9001': 'RSSDID',
    'rcon5597': 'RCON5597_raw',
    'rconf049': 'RCONF049_diag',
    'rconf051': 'RCONF051_diag',
    'rconf047': 'RCONF047_diag',
    'rcon2200': 'RCON2200_raw',
    'rcon2948': 'RCON2948',
}, inplace=True)
# Deduplicate (some bank-quarters have multiple filings)
raw1 = raw1.sort_values('wrdsreportdate').drop_duplicates(
    subset=['RSSDID', 'year', 'quarter'], keep='last')
raw1 = raw1[['RSSDID', 'year', 'quarter',
             'RCON5597_raw', 'RCONF049_diag', 'RCONF051_diag', 'RCONF047_diag',
             'RCON2200_raw', 'RCON2948']]
print(f"  RCON5597  (RC-O M.2, estimated uninsured deposits): "
      f"{raw1['RCON5597_raw'].notna().sum():,} non-null out of {len(raw1):,}")
print(f"  RCONF049  (diag, small accounts): "
      f"{raw1['RCONF049_diag'].notna().sum():,} non-null")
print(f"  RCONF051  (diag, large accounts): "
      f"{raw1['RCONF051_diag'].notna().sum():,} non-null")
print(f"  RCON2948  (total liabilities): "
      f"{raw1['RCON2948'].notna().sum():,} non-null")

# ═══════════════════════════════════════════════════════════════
# Section 2: Merge RCONB993 into clean Call Reports
# ═══════════════════════════════════════════════════════════════

print("\n── Section 2: Merging raw uninsured deposit fields ──")
cr = cr.merge(raw1, on=['RSSDID', 'year', 'quarter'], how='left')
print(f"  After merge: {cr['RCON5597_raw'].notna().sum():,} rows have RCON5597")

# ═══════════════════════════════════════════════════════════════
# Section 3: Construct uninsured deposit share
# ═══════════════════════════════════════════════════════════════

print("\n── Section 3: Constructing uninsured deposit share ──")

# Numerator: direct RCON5597 (Schedule RC-O Memorandum item 2).
# The thesis-facing loan regressions use 2006+ direct observations for
# coverage, while DSSW's own estimation sample starts in 2015Q2 and applies
# additional bank-type screens. Do not residual-backfill uninsured deposits
# from RCONF049/F051/F047; those fields are diagnostics, not the numerator.
#
# Denominators: DSSW (RCON2200, primary) and Emin (RCON2948, robustness).
# Both shares are winsorized at the 1st and 99th percentiles.

liabilities = cr['RCON2948'].copy()
deposits = cr['RCON2200'].copy()
deposits_raw = cr['RCON2200_raw'].copy()
deposits = deposits.fillna(deposits_raw)

cr['uninsured_deposits'] = np.nan

mask_direct = (cr['year'] >= 2006) & cr['RCON5597_raw'].notna()
cr.loc[mask_direct, 'uninsured_deposits'] = cr.loc[mask_direct, 'RCON5597_raw']
print(f"  Direct RCON5597 (2006+): {mask_direct.sum():,} bank-quarters")
print(f"  Pre-2006 bank-quarters left missing: "
      f"{((cr['year'] < 2006)).sum():,}")
print(f"  Total bank-quarters with uninsured_deposits: "
      f"{cr['uninsured_deposits'].notna().sum():,} / {len(cr):,}")

# --- Uninsured share (A): divide by total domestic deposits (DSSW primary) ---
cr['unins_share_dss'] = cr['uninsured_deposits'] / deposits
cr.loc[cr['unins_share_dss'] < 0, 'unins_share_dss'] = np.nan
cr.loc[cr['unins_share_dss'] > 1, 'unins_share_dss'] = np.nan

# --- Uninsured share (B): divide by total liabilities (Emin et al. 2025 robustness) ---
cr['unins_share'] = cr['uninsured_deposits'] / liabilities
cr.loc[cr['unins_share'] < 0, 'unins_share'] = np.nan
cr.loc[cr['unins_share'] > 1, 'unins_share'] = np.nan

# --- Winsorize DSS definition at 1st and 99th percentiles ---
p01_dss = cr['unins_share_dss'].quantile(0.01)
p99_dss = cr['unins_share_dss'].quantile(0.99)
cr['unins_share_dss'] = cr['unins_share_dss'].clip(lower=p01_dss, upper=p99_dss)
print(f"  DSSW-style: Winsorized at [{p01_dss:.4f}, {p99_dss:.4f}]")

# --- Winsorize Emin definition at 1st and 99th percentiles ---
p01 = cr['unins_share'].quantile(0.01)
p99 = cr['unins_share'].quantile(0.99)
cr['unins_share'] = cr['unins_share'].clip(lower=p01, upper=p99)
print(f"  Emin: Winsorized at [{p01:.4f}, {p99:.4f}]")

n_valid_dss = cr['unins_share_dss'].notna().sum()
n_valid = cr['unins_share'].notna().sum()
print(f"  Valid uninsured share (DSSW-style, primary): {n_valid_dss:,} / {len(cr):,}")
print(f"  Valid uninsured share (Emin, robust): {n_valid:,} / {len(cr):,}")
print(f"  DSSW — Mean: {cr['unins_share_dss'].mean():.4f}, "
      f"Median: {cr['unins_share_dss'].median():.4f}, "
      f"SD: {cr['unins_share_dss'].std():.4f}")
print(f"  Emin — Mean: {cr['unins_share'].mean():.4f}, "
      f"Median: {cr['unins_share'].median():.4f}, "
      f"SD: {cr['unins_share'].std():.4f}")

# ═══════════════════════════════════════════════════════════════
# Section 4: De-cumulate YTD variables (RIAD4340, RIAD4230)
# ═══════════════════════════════════════════════════════════════

print("\n── Section 4: De-cumulating YTD variables ──")
cr = cr.sort_values(['RSSDID', 'year', 'quarter']).reset_index(drop=True)

for col in ['RIAD4340', 'RIAD4230']:
    cr[f'{col}_q'] = cr[col].copy()
    mask_q234 = cr['quarter'].isin([2, 3, 4])
    prev_ytd = cr.groupby(['RSSDID', 'year'])[col].shift(1)
    prev_qtr = cr.groupby(['RSSDID', 'year'])['quarter'].shift(1)
    # Only de-cumulate when previous quarter is exactly quarter - 1
    valid_decum = mask_q234 & (prev_qtr == cr['quarter'] - 1)
    cr.loc[valid_decum, f'{col}_q'] = (
        cr.loc[valid_decum, col] - prev_ytd[valid_decum])
    # Gap rows (missing quarter within year): set to NaN
    gap_rows = mask_q234 & (prev_qtr != cr['quarter'] - 1) & prev_qtr.notna()
    cr.loc[gap_rows, f'{col}_q'] = np.nan
    print(f"  {col}: {gap_rows.sum():,} gap rows → NaN, "
          f"{valid_decum.sum():,} de-cumulated")

# ═══════════════════════════════════════════════════════════════
# Section 5: Construct balance sheet ratios
# ═══════════════════════════════════════════════════════════════

print("\n── Section 5: Constructing balance sheet variables ──")

# Assets ($000s) with RCFD fallback
cr['assets_000'] = cr['RCON2170'].fillna(cr['RCFD2170'])

# ROA (annualized, %) = (quarterly net income / assets) x 4 x 100
cr['roa_ann'] = (cr['RIAD4340_q'] / cr['assets_000']) * 4 * 100

# Equity / Assets
cr['equity_to_assets'] = cr['RCON3210'] / cr['assets_000']

# ROE (annualized, %) = (quarterly net income / equity) x 4 x 100
cr['roe_ann'] = (cr['RIAD4340_q'] / cr['RCON3210']) * 4 * 100

# Net Income / Assets (quarterly, not annualized — for the balance table)
cr['net_income_to_assets'] = cr['RIAD4340_q'] / cr['assets_000']

# NPL / Assets
# RCON1403 = past due 90+ days still accruing; RCON1407 = nonaccrual
# Use RCFD fallback for consolidated filers
cr['npl_90plus'] = cr['RCON1403'].fillna(cr['RCFD1403'])
cr['npl_nonaccrual'] = cr['RCON1407'].fillna(cr['RCFD1407'])
cr['npl_to_assets'] = (cr['npl_90plus'] + cr['npl_nonaccrual']) / cr['assets_000']
# Mark as NaN where both RCON and RCFD are missing
both_missing = cr['RCON1403'].isna() & cr['RCFD1403'].isna()
cr.loc[both_missing, 'npl_to_assets'] = np.nan

# LLP / Assets (annualized, %) = (quarterly provision / assets) x 4 x 100
cr['llp_to_assets_ann'] = (cr['RIAD4230_q'] / cr['assets_000']) * 4 * 100

# Liquidity ratio = cash and due from depositories / assets
cr['liquidity_ratio'] = cr['RCON0010'] / cr['assets_000']

# ═══════════════════════════════════════════════════════════════
# Section 6: Identify regression panel banks and filter
# ═══════════════════════════════════════════════════════════════

print("\n── Section 6: Filtering to regression panel banks ──")
panel = pd.read_parquet(
    f'{BASE}/data/final/regression_panel_lender_level.parquet',
    columns=['mapped_rssd_id'])
panel_banks = panel['mapped_rssd_id'].dropna().unique()
print(f"  Unique banks in regression panel: {len(panel_banks)}")

cr_sample = cr[cr['RSSDID'].isin(panel_banks)].copy()
print(f"  Bank-quarters matched: {len(cr_sample):,}")
print(f"  Banks matched: {cr_sample['RSSDID'].nunique()}")
print(f"  Uninsured share coverage: "
      f"{cr_sample['unins_share'].notna().sum():,} / {len(cr_sample):,}")

# ═══════════════════════════════════════════════════════════════
# Section 6B: Merge HHI data for balance table split
# ═══════════════════════════════════════════════════════════════

print("\n── Section 6B: Merging HHI for balance table split ──")
hhi_data = pd.read_parquet(
    f'{BASE}/data/final/regression_panel_lender_level.parquet',
    columns=['mapped_rssd_id', 'year', 'quarter', 'hhi_bank_q'])
hhi_bq = hhi_data.dropna(subset=['mapped_rssd_id', 'year', 'quarter']).copy()
hhi_bq['mapped_rssd_id'] = hhi_bq['mapped_rssd_id'].astype('int64')
hhi_bq['year'] = hhi_bq['year'].astype('int64')
hhi_bq['quarter'] = hhi_bq['quarter'].astype('int64')
hhi_bq = hhi_bq.drop_duplicates(subset=['mapped_rssd_id', 'year', 'quarter'])
hhi_bq = hhi_bq.rename(columns={'mapped_rssd_id': 'RSSDID'})
cr_sample = cr_sample.merge(hhi_bq[['RSSDID', 'year', 'quarter', 'hhi_bank_q']],
                            on=['RSSDID', 'year', 'quarter'], how='left')
print(f"  HHI coverage: {cr_sample['hhi_bank_q'].notna().sum():,} / {len(cr_sample):,}")

# ═══════════════════════════════════════════════════════════════
# Section 7: Balance table — overall summary statistics
# ═══════════════════════════════════════════════════════════════

print("\n── Section 7: Balance table summary statistics ──")

balance_vars = {
    'total_assets_mil':    'Total Assets ($M)',
    'deposits_to_assets':  'Deposits / Assets',
    'loans_to_assets':     'Loans / Assets',
    'net_income_to_assets':'Net Income / Assets',
    'capital_ratio':       'Capital Ratio',
    'equity_to_assets':    'Equity / Assets',
    'roa_ann':             'ROA (ann., %)',
    'roe_ann':             'ROE (ann., %)',
    'npl_to_assets':       'NPL / Assets',
    'llp_to_assets_ann':   'LLP / Assets (ann., %)',
    'liquidity_ratio':     'Liquidity Ratio',
    'unins_share':         'Uninsured Deposit Share',
}

stats_rows = []
for var, label in balance_vars.items():
    s = cr_sample[var].dropna()
    stats_rows.append({
        'Variable': label,
        'N': len(s),
        'Mean': s.mean(),
        'SD': s.std(),
        'P25': s.quantile(0.25),
        'Median': s.median(),
        'P75': s.quantile(0.75),
    })

balance_df = pd.DataFrame(stats_rows)
out_path = f'{BASE}/regression/overview_diagnostics/balance_table.csv'
balance_df.to_csv(out_path, index=False, float_format='%.4f')
print(f"  Saved: {out_path}")
print(balance_df.to_string(index=False, float_format=lambda x: f'{x:.4f}'))

# ═══════════════════════════════════════════════════════════════
# Section 8: Balance table split by HIGH/LOW HHI
# ═══════════════════════════════════════════════════════════════

print("\n── Section 8: Balance table by HHI group ──")

median_hhi = cr_sample['hhi_bank_q'].median()
print(f"  Sample median HHI: {median_hhi:.4f}")

cr_sample['high_hhi'] = np.where(
    cr_sample['hhi_bank_q'].notna(),
    (cr_sample['hhi_bank_q'] > median_hhi).astype(int),
    np.nan)

for g in [0, 1]:
    n = (cr_sample['high_hhi'] == g).sum()
    print(f"  high_hhi={g}: {n:,} bank-quarters")

from scipy import stats as sp_stats

hhi_split_rows = []
for var, label in balance_vars.items():
    for g, gname in [(0, 'Low HHI'), (1, 'High HHI')]:
        sub = cr_sample.loc[cr_sample['high_hhi'] == g, var].dropna()
        hhi_split_rows.append({
            'Variable': label,
            'Group': gname,
            'N': len(sub),
            'Mean': sub.mean(),
            'SD': sub.std(),
        })
    low_vals = cr_sample.loc[cr_sample['high_hhi'] == 0, var].dropna()
    high_vals = cr_sample.loc[cr_sample['high_hhi'] == 1, var].dropna()
    if len(low_vals) > 1 and len(high_vals) > 1:
        tstat, pval = sp_stats.ttest_ind(low_vals, high_vals, equal_var=False)
        hhi_split_rows.append({
            'Variable': label,
            'Group': 'Diff p-value',
            'N': 0,
            'Mean': high_vals.mean() - low_vals.mean(),
            'SD': pval,
        })

hhi_split_df = pd.DataFrame(hhi_split_rows)
out_path_hhi = f'{BASE}/regression/overview_diagnostics/balance_table_by_hhi.csv'
hhi_split_df.to_csv(out_path_hhi, index=False, float_format='%.4f')
print(f"  Saved: {out_path_hhi}")

# ═══════════════════════════════════════════════════════════════
# Section 8B: Balance table split by high/low uninsured dummy
# ═══════════════════════════════════════════════════════════════

print("\n── Section 8B: Balance table by uninsured deposit group ──")

median_unins = cr_sample['unins_share'].median()
print(f"  Sample median uninsured share: {median_unins:.4f}")

cr_sample['high_unins'] = np.where(
    cr_sample['unins_share'].notna(),
    (cr_sample['unins_share'] > median_unins).astype(int),
    np.nan)

for g in [0, 1]:
    n = (cr_sample['high_unins'] == g).sum()
    print(f"  high_unins={g}: {n:,} bank-quarters")

split_rows = []
for var, label in balance_vars.items():
    for g, gname in [(0, 'Low Unins'), (1, 'High Unins')]:
        sub = cr_sample.loc[cr_sample['high_unins'] == g, var].dropna()
        split_rows.append({
            'Variable': label,
            'Group': gname,
            'N': len(sub),
            'Mean': sub.mean(),
            'SD': sub.std(),
        })
    low_vals = cr_sample.loc[cr_sample['high_unins'] == 0, var].dropna()
    high_vals = cr_sample.loc[cr_sample['high_unins'] == 1, var].dropna()
    if len(low_vals) > 1 and len(high_vals) > 1:
        tstat, pval = sp_stats.ttest_ind(low_vals, high_vals, equal_var=False)
        split_rows.append({
            'Variable': label,
            'Group': 'Diff p-value',
            'N': 0,
            'Mean': high_vals.mean() - low_vals.mean(),
            'SD': pval,
        })

split_df = pd.DataFrame(split_rows)
out_path2 = f'{BASE}/regression/overview_diagnostics/balance_table_by_uninsured.csv'
split_df.to_csv(out_path2, index=False, float_format='%.4f')
print(f"  Saved: {out_path2}")

# ═══════════════════════════════════════════════════════════════
# Section 9: Save Stata file for regression merge
# ═══════════════════════════════════════════════════════════════

print("\n── Section 9: Saving Stata file ──")

# Save ALL banks (not just panel) so the merge covers everything
out_cols = ['RSSDID', 'year', 'quarter', 'unins_share', 'unins_share_dss',
            'uninsured_deposits',
            'roa_ann', 'roe_ann', 'equity_to_assets', 'npl_to_assets',
            'llp_to_assets_ann', 'liquidity_ratio', 'net_income_to_assets']
out = cr[out_cols].copy()
out.rename(columns={'RSSDID': 'mapped_rssd_id'}, inplace=True)

# Convert types for Stata compatibility
out['mapped_rssd_id'] = out['mapped_rssd_id'].astype('int64')
out['year'] = out['year'].astype('int32')
out['quarter'] = out['quarter'].astype('int32')

out_dta = f'{BASE}/data/final/bank_uninsured_deposits.dta'
out.to_stata(out_dta, write_index=False, version=118)
print(f"  Saved: {out_dta} ({len(out):,} rows, "
      f"{out['mapped_rssd_id'].nunique():,} banks)")

# ═══════════════════════════════════════════════════════════════
# Section 10: Diagnostics for panel banks
# ═══════════════════════════════════════════════════════════════

print("\n── Section 10: Panel bank diagnostics ──")

diag = cr_sample.groupby('RSSDID').agg(
    n_qtrs=('unins_share', 'count'),
    n_valid_unins=('unins_share', lambda x: x.notna().sum()),
    mean_unins=('unins_share', 'mean'),
    mean_assets=('total_assets_mil', 'mean'),
).reset_index()
diag = diag.sort_values('mean_assets', ascending=False)

out_diag = f'{BASE}/regression/overview_diagnostics/uninsured_deposit_panel_banks.csv'
diag.to_csv(out_diag, index=False, float_format='%.4f')
print(f"  Saved: {out_diag}")
print(f"  Panel banks with valid uninsured share: "
      f"{(diag['n_valid_unins'] > 0).sum()} / {len(diag)}")
print(f"  Mean uninsured share (panel banks): "
      f"{cr_sample['unins_share'].mean():.4f}")

print("\n── Done ──")
