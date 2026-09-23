import os
# # Final Regression Panel Preparation
#
# **Purpose**: Prepare clean regression-ready dataset with trimming, sample selection, and derived variables.
#
# **Input**:
# - `data/intermediate/merged/dealscan_compustat_call_sod_macro_lenderlevel.parquet` (598,217 HHI-matched rows)
#
# **Output**:
# - `data/final/regression_panel_lender_level.parquet` (regression-ready lender-level panel)
# - `data/final/regression_panel_loan_level.parquet` (regression-ready loan-borrower panel with aggregated bank vars)
# - `data/documentation/sample_attrition_table.csv` (sample flow table)
# - `data/documentation/final_panel_summary_statistics.csv` (Table 1 ready)
#
# **Key decisions** (based on Degerli 2019, Drechsler et al. 2017, and empirical finance best practices):
# 1. **Original + trimmed outputs** - keep originals and create P1-P99 trimmed copies for core continuous variables
# 2. **HHI and monetary policy shocks** - Keep full distribution (core identification variables)
# 3. **Bank characteristics** - Log transform size, trim ratios at P1-P99
# 4. **Borrower characteristics** - Trim extreme leverage/ROA/M-B, handle missing carefully
# 5. **Loan characteristics update (2026-02-10)** - add spread in percentage points and ln(maturity) while keeping originals
#
# **References**:
# - Degerli (2019): Monetary Policy Exposure of Banks and Loan Contracting
# - Drechsler et al. (2017): The Deposits Channel of Monetary Policy
# - [FDIC Working Paper 2022-06](https://www.fdic.gov/analysis/cfr/working-papers/2022/cfr-wp2022-06.pdf): Bank Concentration and Monetary Policy Pass-Through
# - [BIS Working Paper 1169](https://www.bis.org/publ/work1169.pdf): Risk-based Pricing in Competitive Lending Markets

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
try:
    from IPython.display import display
except Exception:
    def display(x):
        print(x)

pd.set_option('display.max_columns', 120)
pd.set_option('display.width', 1200)
pd.set_option('display.max_rows', 100)

BASE = Path(os.environ.get("MPHIL_THESIS_ROOT", Path(__file__).resolve().parents[2]))
IN_LENDER_PANEL = BASE / 'data/intermediate/merged/dealscan_compustat_call_sod_macro_lenderlevel.parquet'
OUT_DIR = BASE / 'data/final'
OUT_DOC_DIR = BASE / 'data/documentation'

OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_DOC_DIR.mkdir(parents=True, exist_ok=True)


def parse_sic4(series: pd.Series) -> pd.Series:
    """Parse 4-digit SIC code from numeric or text SIC columns."""
    if series is None:
        return pd.Series(dtype='float64')
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors='coerce')
    extracted = series.astype(str).str.extract(r'(\d{4})', expand=False)
    return pd.to_numeric(extracted, errors='coerce')


# Baseline sample policy (supervision update, 2026-02-10)
APPLY_STRICT_KEY_DEDUPE = True
DROP_FOREIGN_BANKS = True
TRIM_LOWER_Q = 0.01
TRIM_UPPER_Q = 0.99
DEDUPE_REFERENCE_DATE_COLS = ['tranche_active_date', 'deal_active_date', 'mandated_date', 'deal_date']


# ## 1. Load Data and Initial Sample Selection

# Load merged lender-level panel
raw_df = pd.read_parquet(IN_LENDER_PANEL)
df = raw_df.copy()

# Canonical keys (make notebook robust to upstream naming)
loan_id_col = 'deal_id' if 'deal_id' in df.columns else 'lpc_deal_id'
tranche_id_col = 'tranche_id' if 'tranche_id' in df.columns else ('lpc_tranche_id' if 'lpc_tranche_id' in df.columns else None)

df['deal_id'] = df[loan_id_col]
if tranche_id_col is not None:
    df['tranche_id'] = df[tranche_id_col]

if not np.issubdtype(df['deal_date'].dtype, np.datetime64):
    df['deal_date'] = pd.to_datetime(df['deal_date'], errors='coerce')

# Borrower SIC parsing for financial-firm exclusion
sic_source = None
for c in ['sic_code_num', 'sic', 'sic_code']:
    if c in df.columns:
        sic_source = c
        break

if sic_source is not None:
    df['borrower_sic4'] = parse_sic4(df[sic_source])
    df['is_financial_borrower'] = df['borrower_sic4'].between(6000, 6999, inclusive='both')
    df['is_utility_borrower'] = df['borrower_sic4'].between(4900, 4999, inclusive='both')
else:
    df['borrower_sic4'] = np.nan
    df['is_financial_borrower'] = False
    df['is_utility_borrower'] = False

print(f"Raw merged panel: {len(df):,} rows x {df.shape[1]} columns")
print(f"Time range: {df['deal_date'].min()} to {df['deal_date'].max()}")
print(f"Loan key source: {loan_id_col}")
print(f"Tranche key source: {tranche_id_col}")
print(f"Unique deals: {df['deal_id'].nunique():,}")
print(f"Unique lenders: {df['lender_id'].nunique():,}")
print(f"Unique borrowers (gvkey): {df['gvkey'].nunique():,}")
if sic_source is not None:
    fin_share = df['is_financial_borrower'].mean() * 100
    util_share = df['is_utility_borrower'].mean() * 100
    print(f"SIC source: {sic_source}; financial borrowers (SIC 6000-6999): {fin_share:.2f}%")
    print(f"SIC source: {sic_source}; utility borrowers (SIC 4900-4999): {util_share:.2f}%")
    pre_filter_rows = len(df)
    df = df[~(df['is_financial_borrower'] | df['is_utility_borrower'])].copy()
    print(f"Applied SIC exclusions (financial + utilities): dropped {pre_filter_rows - len(df):,} rows; remaining {len(df):,}")



# Initialize attrition tracking
attrition = []


def _nunique_safe(data: pd.DataFrame, col: str) -> int:
    return data[col].nunique(dropna=True) if col in data.columns else np.nan


def track_sample(label, condition=None):
    """Track sample size at each filtering step."""
    global df, attrition
    if condition is not None:
        df = df.loc[condition].copy()
    attrition.append({
        'step': len(attrition) + 1,
        'filter': label,
        'n_obs': len(df),
        'n_loans': _nunique_safe(df, 'deal_id'),
        'n_tranches': _nunique_safe(df, 'tranche_id'),
        'n_lenders': _nunique_safe(df, 'lender_id'),
        'n_borrowers': _nunique_safe(df, 'gvkey')
    })
    print(f"{label}: {len(df):,} obs ({_nunique_safe(df, 'deal_id'):,} loans)")


track_sample('Raw merged panel')



# ## 2. Core Sample Selection (Matching Flags)

# Time window consistency only (keep all rows; no NA-based filtering)
track_sample(
    'Deal date in [2001-01-01, 2025-12-31]',
    (df['deal_date'] >= pd.Timestamp('2001-01-01')) &
    (df['deal_date'] <= pd.Timestamp('2025-12-31'))
)

# NOTE:
# Do NOT filter on missingness for HHI, gvkey, monetary policy, or any controls.
# Keep all observations for flexible downstream regression experiments.

# Do not force borrower industry exclusions at this stage.

# Drop duplicate identifiers only if they are true duplicate IDs
if 'ds_row_id' in df.columns:
    dup_rowid = df.duplicated(['ds_row_id']).sum()
    print(f"Duplicate ds_row_id rows before filtering: {dup_rowid:,}")
    if dup_rowid > 0:
        track_sample('Drop duplicate ds_row_id rows', ~df.duplicated(['ds_row_id']))

if DROP_FOREIGN_BANKS and 'lender_category' in df.columns:
    before = len(df)
    track_sample(
        'Drop foreign-bank lenders (lender_category != foreign_bank)',
        df['lender_category'].fillna('') != 'foreign_bank'
    )
    print(f"Foreign-bank rows dropped: {before - len(df):,}")
else:
    print('Foreign-bank filter skipped (lender_category missing or disabled).')

# Keep exact duplicates as-is unless explicitly requested later.
exact_dups = df.duplicated().sum()
print(f"Exact duplicate rows retained: {exact_dups:,}")


# ## 3. Variable Trimming
#
# ### 3.1 Dependent Variable: Loan Spread

# Diagnose spread distribution before basic cleaning
spread_stats = df['all_in_spread_drawn_bps'].describe(percentiles=[0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
print()
print('Loan spread distribution (raw before basic cleaning):')
print(spread_stats)

print()
print(f"Negative spread: {(df['all_in_spread_drawn_bps'] < 0).sum():,} obs")
print(f"Spread == 0: {(df['all_in_spread_drawn_bps'] == 0).sum():,} obs")
print(f"Spread > 1500 bps (flag only): {(df['all_in_spread_drawn_bps'] > 1500).sum():,} obs")
print(f"Missing spread: {df['all_in_spread_drawn_bps'].isna().sum():,} obs")



# Keep all rows; do not filter by spread missingness/value at this stage.
track_sample('No spread filter applied (keep NA/non-positive for later experiments)')


# ### 3.2 Bank Competition: HHI (NO TRIMMING - Core ID Variable)

# HHI is bounded [0, 1] by construction - no trimming needed
# Just verify distribution

hhi_stats = df['hhi_bank_q'].describe(percentiles=[0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
print("\nBank HHI distribution (no trimming):")
print(hhi_stats)

print(f"\nHHI = 1 (monopoly): {(df['hhi_bank_q'] == 1.0).sum():,} obs ({(df['hhi_bank_q'] == 1.0).mean()*100:.2f}%)")
print(f"HHI > 0.25 (highly concentrated): {(df['hhi_bank_q'] > 0.25).sum():,} obs ({(df['hhi_bank_q'] > 0.25).mean()*100:.2f}%)")

# Optional: Exclude perfect monopoly markets (HHI=1) if they represent data errors
# Uncomment if needed:
# track_sample('HHI < 1 (exclude monopoly)', df['hhi_bank_q'] < 1.0)

# ### 3.3 Monetary Policy: JK Shocks (NO TRIMMING - Core ID Variable)

# JK shocks are optional controls; keep NA for flexible specs.
mp_vars = ['jk_MP_pm_sum', 'jk_MP_pm_mean', 'jk_CBI_pm_sum', 'jk_CBI_pm_mean', 'ffr_q']

for var in mp_vars:
    if var in df.columns:
        stats = df[var].describe(percentiles=[0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
        print()
        print(f'{var} distribution (no filtering):')
        print(stats[['mean', 'std', 'min', '1%', '50%', '99%', 'max']])

track_sample('No monetary policy NA filter applied')


# ### 3.4 Bank Characteristics

# Bank controls: sanitize out-of-range values to NaN (do not drop rows)
assets_pos = pd.to_numeric(df['call_total_assets_mil'], errors='coerce')
cap = pd.to_numeric(df['call_capital_ratio'], errors='coerce')
lta = pd.to_numeric(df['call_loans_to_assets'], errors='coerce')

# Key transformed variable
df['log_bank_assets'] = np.where(assets_pos > 0, np.log(assets_pos), np.nan)

# Keep controls but null-out implausible values instead of trimming sample
df['call_capital_ratio'] = cap.where(cap.between(0, 0.5))
df['call_loans_to_assets'] = lta.where(lta.between(0, 1))

print()
print('Bank-control sanitization applied (no row drop):')
print(f"  call_total_assets_mil <= 0 (log set NaN): {(assets_pos <= 0).sum():,}")
print(f"  call_capital_ratio outside [0, 0.5] -> NaN: {(~cap.between(0,0.5) & cap.notna()).sum():,}")
print(f"  call_loans_to_assets outside [0, 1] -> NaN: {(~lta.between(0,1) & lta.notna()).sum():,}")



# ### 3.5 Borrower Characteristics

# Borrower controls: sanitize out-of-range values to NaN (do not drop rows)
lev = pd.to_numeric(df['leverage'], errors='coerce')
roa = pd.to_numeric(df['roa'], errors='coerce')
mtb = pd.to_numeric(df['market_to_book'], errors='coerce')

# Plausibility ranges (broad)
df['leverage'] = lev.where(lev.between(-1, 10))
df['roa'] = roa.where(roa.between(-1, 1))
df['market_to_book'] = mtb.where(mtb.between(0, 20))

print()
print('Borrower-control sanitization applied (no row drop):')
print(f"  leverage outside [-1, 10] -> NaN: {(~lev.between(-1,10) & lev.notna()).sum():,}")
print(f"  roa outside [-1, 1] -> NaN: {(~roa.between(-1,1) & roa.notna()).sum():,}")
print(f"  market_to_book outside [0, 20] -> NaN: {(~mtb.between(0,20) & mtb.notna()).sum():,}")



# ### 3.6 Loan Characteristics

# No NA/value filtering on loan characteristics at this stage.
track_sample('No loan-characteristic filter applied (keep NA/non-finite for experiments)')

# Analytical-key duplication diagnostics and original-only dedupe
analysis_key = ['deal_id', 'tranche_id', 'lender_id', 'gvkey', 'deal_date']
analysis_key = [k for k in analysis_key if k in df.columns]

dup_before = df.duplicated(analysis_key).sum()
print(f'Duplicate analytical key rows before original-only dedupe: {dup_before:,}')

strict_keep_idx = None
if dup_before > 0:
    if 'ds_row_id' not in df.columns:
        df['ds_row_id'] = np.arange(len(df))

    version_col = '_version_date_for_dedupe'
    df[version_col] = pd.NaT

    for c in DEDUPE_REFERENCE_DATE_COLS:
        if c in df.columns:
            dt = pd.to_datetime(df[c], errors='coerce')
            df[version_col] = df[version_col].where(df[version_col].notna(), dt)

    # Diagnostic: how far apart are original and amended snapshots?
    dup_mask = df.duplicated(analysis_key, keep=False)
    dup_df = df.loc[dup_mask, analysis_key + [version_col]].copy()
    if not dup_df.empty:
        gap_by_key = (
            dup_df.groupby(analysis_key, dropna=False)[version_col]
            .agg(version_min='min', version_max='max', n_versions='size')
            .reset_index()
        )
        gap_by_key['amendment_span_days'] = (
            gap_by_key['version_max'] - gap_by_key['version_min']
        ).dt.days

        gap_summary = gap_by_key['amendment_span_days'].describe(
            percentiles=[0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
        )
        gap_summary_df = gap_summary.to_frame('value').reset_index().rename(columns={'index': 'stat'})

        bins = [-1, 0, 30, 90, 180, 365, 730, 1825, np.inf]
        labels = ['0', '1-30', '31-90', '91-180', '181-365', '366-730', '731-1825', '>1825']
        gap_by_key['span_bucket_days'] = pd.cut(gap_by_key['amendment_span_days'], bins=bins, labels=labels)
        gap_bucket = (
            gap_by_key['span_bucket_days']
            .value_counts(dropna=False, sort=False)
            .rename_axis('span_bucket_days')
            .reset_index(name='n_key_groups')
        )
        gap_bucket['share_pct'] = gap_bucket['n_key_groups'] / gap_bucket['n_key_groups'].sum() * 100

        gap_summary_path = OUT_DOC_DIR / 'amendment_original_vs_amended_gap_summary.csv'
        gap_bucket_path = OUT_DOC_DIR / 'amendment_original_vs_amended_gap_bucket.csv'
        gap_summary_df.to_csv(gap_summary_path, index=False)
        gap_bucket.to_csv(gap_bucket_path, index=False)

        print('Amendment span diagnostics (days between earliest and latest snapshot within key):')
        print(gap_summary_df.to_string(index=False))
        print(f'Gap summary saved: {gap_summary_path}')
        print(f'Gap bucket saved: {gap_bucket_path}')

    strict_keep_idx = (
        df.sort_values(analysis_key + [version_col, 'ds_row_id'], na_position='last')
        .groupby(analysis_key, dropna=False)
        .head(1)
        .index
    )

    strict_n = len(strict_keep_idx)
    print(f'Original-only dedupe candidate size: {strict_n:,} (would drop {len(df)-strict_n:,})')

    if APPLY_STRICT_KEY_DEDUPE:
        track_sample(
            'Deduplicate analytical key (keep earliest/original snapshot)',
            df.index.isin(strict_keep_idx)
        )
        strict_keep_idx = df.index
        print(f'Duplicate analytical key rows after applied dedupe: {df.duplicated(analysis_key).sum():,}')
    else:
        print('Original-only dedupe NOT applied (diagnostic only).')

    df = df.drop(columns=[version_col], errors='ignore')


# ## 4. Create Derived Variables

# Feb-10 variable updates: spread in percentage points + ln(maturity), keep originals
spread_bps = pd.to_numeric(df['all_in_spread_drawn_bps'], errors='coerce') if 'all_in_spread_drawn_bps' in df.columns else pd.Series(np.nan, index=df.index)
df['all_in_spread_drawn_pct'] = spread_bps / 100.0

maturity = pd.to_numeric(df['tenor_maturity'], errors='coerce') if 'tenor_maturity' in df.columns else pd.Series(np.nan, index=df.index)
df['log_tenor_maturity'] = np.where(maturity > 0, np.log(maturity), np.nan)

# 1%/99% trimming variables (winsor-style capped copies, originals preserved)
trim_candidates = [
    'all_in_spread_drawn_bps',
    'all_in_spread_drawn_pct',
    'tenor_maturity',
    'log_tenor_maturity',
    'log_loan_amount',
    'loan_amount_mil',
    'hhi_bank_q',
    'jk_MP_pm_sum',
    'jk_MP_pm_mean',
    'ffr_q',
    'call_total_assets_mil',
    'log_bank_assets',
    'call_capital_ratio',
    'call_loans_to_assets',
    'call_deposits_to_assets',
    'log_assets',
    'leverage',
    'roa',
    'market_to_book',
    'tangibility'
]

trim_records = []
for var in trim_candidates:
    if var not in df.columns:
        continue
    s = pd.to_numeric(df[var], errors='coerce')
    nonmiss = s.dropna()
    if len(nonmiss) == 0:
        continue

    q1 = float(nonmiss.quantile(TRIM_LOWER_Q))
    q99 = float(nonmiss.quantile(TRIM_UPPER_Q))
    out_col = f'{var}_trim_p1p99'
    flag_col = f'flag_{var}_trimmed_p1p99'

    df[out_col] = s.clip(lower=q1, upper=q99)
    df[flag_col] = ((s < q1) | (s > q99)).fillna(False).astype(int)

    trim_records.append({
        'variable': var,
        'q01': q1,
        'q99': q99,
        'n_nonmissing': int(nonmiss.notna().sum()),
        'n_trimmed': int(df[flag_col].sum()),
        'share_trimmed_pct': float(df[flag_col].mean() * 100)
    })

trim_bounds_df = pd.DataFrame(trim_records).sort_values('variable')
trim_bounds_path = OUT_DOC_DIR / 'trim_bounds_p1p99.csv'
trim_bounds_df.to_csv(trim_bounds_path, index=False)

print('Added variables: all_in_spread_drawn_pct, log_tenor_maturity')
print(f'Trim bounds saved to: {trim_bounds_path}')
if not trim_bounds_df.empty:
    display(trim_bounds_df)

# Time variables for fixed effects
df['year'] = df['deal_date'].dt.year
df['quarter'] = df['deal_date'].dt.quarter
df['year_quarter'] = df['year'].astype(str) + 'Q' + df['quarter'].astype(str)

# Time trend (years since 2001)
df['time_trend'] = (df['deal_date'] - pd.Timestamp('2001-01-01')).dt.days / 365.25

# Crisis period indicators (for robustness checks)
df['crisis_2007_2009'] = ((df['year'] >= 2007) & (df['year'] <= 2009)).astype(int)
df['post_crisis'] = (df['year'] >= 2010).astype(int)

# Bank size terciles (for heterogeneity analysis)
df['bank_size_tercile'] = pd.qcut(df['call_total_assets_mil'], q=3, labels=['Small', 'Medium', 'Large'])

# HHI concentration categories
df['hhi_category'] = pd.cut(
    df['hhi_bank_q'],
    bins=[0, 0.15, 0.25, 1.0],
    labels=['Low Concentration', 'Moderate Concentration', 'High Concentration']
)

analysis_key = ['deal_id', 'tranche_id', 'lender_id', 'gvkey', 'deal_date']
analysis_key = [k for k in analysis_key if k in df.columns]
dup_key = df.duplicated(analysis_key).sum()

print()
print(f'Final regression sample: {len(df):,} observations')
print(f"Unique deals: {df['deal_id'].nunique():,}")
if 'tranche_id' in df.columns:
    print(f"Unique tranches: {df['tranche_id'].nunique():,}")
print(f"Unique lenders: {df['lender_id'].nunique():,}")
print(f"Unique borrowers: {df['gvkey'].nunique():,}")
print(f"Time range: {df['year'].min()} - {df['year'].max()}")
print(f'Duplicate analytical key rows ({analysis_key}): {dup_key:,}')
if 'lender_category' in df.columns:
    print(f"Foreign-bank rows remaining: {(df['lender_category'] == 'foreign_bank').sum():,}")


# ## 5. Summary Statistics

# Variable groups for summary statistics table
var_groups = {
    'Dependent Variable': [
        'all_in_spread_drawn_bps',
        'all_in_spread_drawn_pct',
        'all_in_spread_drawn_bps_trim_p1p99',
        'all_in_spread_drawn_pct_trim_p1p99'
    ],
    'Bank Competition': [
        'hhi_bank_q',
        'hhi_bank_q_trim_p1p99'
    ],
    'Monetary Policy': [
        'jk_MP_pm_sum',
        'jk_MP_pm_sum_trim_p1p99',
        'ffr_q',
        'ffr_q_trim_p1p99'
    ],
    'Bank Characteristics': [
        'log_bank_assets',
        'log_bank_assets_trim_p1p99',
        'call_capital_ratio',
        'call_capital_ratio_trim_p1p99',
        'call_loans_to_assets',
        'call_loans_to_assets_trim_p1p99',
        'call_deposits_to_assets',
        'call_deposits_to_assets_trim_p1p99'
    ],
    'Borrower Characteristics': [
        'log_assets',
        'log_assets_trim_p1p99',
        'leverage',
        'leverage_trim_p1p99',
        'roa',
        'roa_trim_p1p99',
        'market_to_book',
        'market_to_book_trim_p1p99',
        'tangibility',
        'tangibility_trim_p1p99'
    ],
    'Loan Characteristics': [
        'log_loan_amount',
        'log_loan_amount_trim_p1p99',
        'tenor_maturity',
        'tenor_maturity_trim_p1p99',
        'log_tenor_maturity',
        'log_tenor_maturity_trim_p1p99',
        'secured_flag'
    ]
}

# Generate detailed summary statistics
summary_stats = []

for group, vars_list in var_groups.items():
    for var in vars_list:
        if var not in df.columns:
            continue

        s_all = pd.to_numeric(df[var], errors='coerce')
        s = s_all.dropna()
        if len(s) == 0:
            continue

        summary_stats.append({
            'Variable Group': group,
            'Variable': var,
            'N_total': len(s_all),
            'N_nonmissing': len(s),
            'Missing_pct': (s_all.isna().mean() * 100),
            'Mean': s.mean(),
            'Std': s.std(),
            'P1': s.quantile(0.01),
            'P5': s.quantile(0.05),
            'P25': s.quantile(0.25),
            'Median': s.quantile(0.50),
            'P75': s.quantile(0.75),
            'P95': s.quantile(0.95),
            'P99': s.quantile(0.99),
            'Min': s.min(),
            'Max': s.max()
        })

summary_df = pd.DataFrame(summary_stats)
display(summary_df)


# Save summary statistics
summary_path = OUT_DOC_DIR / 'key_variable_summary_statistics_basic_cleaning.csv'
summary_df.to_csv(summary_path, index=False)

# keep legacy filename for compatibility
summary_df.to_csv(OUT_DOC_DIR / 'final_panel_summary_statistics.csv', index=False)

print()
print(f'Summary statistics saved to: {summary_path}')



# ## 6. Sample Attrition Table

# Create attrition table
attrition_df = pd.DataFrame(attrition)

# Calculate percentage drops
attrition_df['pct_drop_obs'] = (
    (attrition_df['n_obs'].shift(1) - attrition_df['n_obs']) /
    attrition_df['n_obs'].shift(1) * 100
).fillna(0)

attrition_df['pct_of_raw'] = (
    attrition_df['n_obs'] / attrition_df.iloc[0]['n_obs'] * 100
)

display(attrition_df)

# Save attrition table
attrition_df.to_csv(OUT_DOC_DIR / 'sample_attrition_table.csv', index=False)
print(f"\nAttrition table saved to: {OUT_DOC_DIR / 'sample_attrition_table.csv'}")

# ## 7. Distribution Plots (Before vs After Trimming)
#
# Visual verification that trimming removed only extreme outliers.

# Key variable histograms (post basic-cleaning sample)
hist_vars = [
    ('all_in_spread_drawn_bps', 'Loan Spread (bps)'),
    ('all_in_spread_drawn_pct', 'Loan Spread (percentage points)'),
    ('tenor_maturity', 'Loan Maturity (months)'),
    ('log_tenor_maturity', 'ln(Loan Maturity)'),
    ('log_loan_amount', 'Log Loan Amount'),
    ('hhi_bank_q', 'Bank HHI'),
    ('jk_MP_pm_sum', 'JK Monetary Shock'),
    ('ffr_q', 'Federal Funds Rate (Quarterly)'),
    ('log_bank_assets', 'Log Bank Assets'),
    ('call_capital_ratio', 'Bank Capital Ratio'),
    ('log_assets', 'Borrower Log Assets'),
    ('leverage', 'Borrower Leverage')
]

fig, axes = plt.subplots(4, 3, figsize=(18, 18))
fig.suptitle('Key Variable Histograms (Basic Cleaning, Original + Transformed)', fontsize=16, y=0.995)

for idx, (var, label) in enumerate(hist_vars):
    ax = axes[idx // 3, idx % 3]

    if var not in df.columns:
        ax.text(0.5, 0.5, f'{var}\nNot Available', ha='center', va='center')
        ax.axis('off')
        continue

    s = pd.to_numeric(df[var], errors='coerce').dropna()
    if len(s) == 0:
        ax.text(0.5, 0.5, f'{var}\nAll Missing', ha='center', va='center')
        ax.axis('off')
        continue

    ax.hist(s, bins=60, alpha=0.75, density=True)
    p1, p99 = s.quantile(0.01), s.quantile(0.99)
    ax.axvline(p1, color='red', linestyle='--', linewidth=1, label='P1/P99')
    ax.axvline(p99, color='red', linestyle='--', linewidth=1)
    ax.set_title(label)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

for j in range(len(hist_vars), 12):
    axes[j // 3, j % 3].axis('off')

plt.tight_layout()
hist_path = OUT_DOC_DIR / 'key_variable_histograms_basic_cleaning.png'
plt.savefig(hist_path, dpi=300, bbox_inches='tight')
plt.show()

print()
print(f'Key variable histogram saved to: {hist_path}')


# ## 8. Save Final Regression Panel

# Save lender-level panel (baseline)
out_file_lender = OUT_DIR / 'regression_panel_lender_level.parquet'
df.to_parquet(out_file_lender, index=False)

print()
print('=' * 80)
print('FINAL REGRESSION PANEL SAVED')
print('=' * 80)
print()
print(f'Baseline lender-level panel: {out_file_lender}')
print(f'  - Observations: {len(df):,}')
print(f'  - Variables: {df.shape[1]}')
print(f"  - Unique loans: {df['deal_id'].nunique():,}")
if 'tranche_id' in df.columns:
    print(f"  - Unique tranches: {df['tranche_id'].nunique():,}")
print(f"  - Unique lenders: {df['lender_id'].nunique():,}")
print(f"  - Unique borrowers: {df['gvkey'].nunique():,}")
print(f"  - Time span: {df['year'].min()}-{df['year'].max()}")
if 'lender_category' in df.columns:
    print(f"  - Foreign-bank rows in baseline: {(df['lender_category'] == 'foreign_bank').sum():,}")
print()
print(f"  - Duplicate analytical key rows: {df.duplicated([k for k in ['deal_id','tranche_id','lender_id','gvkey','deal_date'] if k in df.columns]).sum():,}")
print()
print('Documentation:')
print(f"  - Summary stats: {OUT_DOC_DIR / 'key_variable_summary_statistics_basic_cleaning.csv'}")
print(f"  - Attrition table: {OUT_DOC_DIR / 'sample_attrition_table.csv'}")
print(f"  - Histograms: {OUT_DOC_DIR / 'key_variable_histograms_basic_cleaning.png'}")
print(f"  - Trim bounds: {OUT_DOC_DIR / 'trim_bounds_p1p99.csv'}")
print(f"  - Amendment-gap summary: {OUT_DOC_DIR / 'amendment_original_vs_amended_gap_summary.csv'}")


# ## 9. Create Loan-Borrower Panel (Aggregated Bank Variables)
#
# For loan-level regressions, aggregate bank characteristics across syndicate members.

# Identify bank variables to aggregate
bank_agg_vars = [
    'hhi_bank_q',
    'hhi_bank_q_trim_p1p99',
    'call_total_assets_mil',
    'log_bank_assets',
    'call_capital_ratio',
    'call_loans_to_assets',
    'call_deposits_to_assets'
]

# Aggregate at loan-borrower-date level
agg_dict = {}
for var in bank_agg_vars:
    if var in df.columns:
        agg_dict[f'{var}_lender_mean'] = (var, 'mean')
        agg_dict[f'{var}_lender_min'] = (var, 'min')
        agg_dict[f'{var}_lender_max'] = (var, 'max')

loan_keys = ['deal_id', 'gvkey', 'deal_date']
loan_level = (
    df.groupby(loan_keys, as_index=False)
    .agg(**agg_dict)
)

# Keep one deterministic row per loan-borrower-date for loan/borrower controls
loan_borrower_vars = [
    'deal_id', 'gvkey', 'deal_date',
    'all_in_spread_drawn_bps',
    'all_in_spread_drawn_pct',
    'all_in_spread_drawn_bps_trim_p1p99',
    'all_in_spread_drawn_pct_trim_p1p99',
    'log_loan_amount',
    'log_loan_amount_trim_p1p99',
    'tenor_maturity',
    'tenor_maturity_trim_p1p99',
    'log_tenor_maturity',
    'log_tenor_maturity_trim_p1p99',
    'secured_flag',
    'log_assets',
    'leverage',
    'roa',
    'market_to_book',
    'tangibility',
    'jk_MP_pm_sum',
    'jk_MP_pm_mean',
    'ffr_q',
    'year', 'quarter', 'year_quarter', 'time_trend',
    'crisis_2007_2009', 'post_crisis'
]
loan_borrower_vars = [c for c in loan_borrower_vars if c in df.columns]

loan_static = (
    df[loan_borrower_vars]
    .sort_values(loan_keys)
    .drop_duplicates(loan_keys, keep='first')
)

loan_level = loan_level.merge(
    loan_static,
    on=loan_keys,
    how='left',
    validate='one_to_one'
)

dup_loan_keys = loan_level.duplicated(loan_keys).sum()
print()
print(f'Loan-borrower panel: {len(loan_level):,} rows')
print(f'Variables: {loan_level.shape[1]}')
print(f'Duplicate loan keys after merge: {dup_loan_keys:,}')


# Save loan-borrower panel
out_file_loan = OUT_DIR / 'regression_panel_loan_level.parquet'
loan_level.to_parquet(out_file_loan, index=False)

print(f"\nLoan-borrower panel saved to: {out_file_loan}")
print(f"  - Observations: {len(loan_level):,}")
print(f"  - Variables: {loan_level.shape[1]}")

# ## 10. Validation Checks

print()
print('=' * 80)
print('FINAL PANEL VALIDATION CHECKS')
print('=' * 80)

# Check 1: Missingness profile (diagnostic only)
key_vars_required = [
    'all_in_spread_drawn_bps',
    'all_in_spread_drawn_pct',
    'hhi_bank_q',
    'jk_MP_pm_sum',
    'log_bank_assets',
    'log_tenor_maturity',
    'log_assets'
]

print()
print('CHECK 1: Key variables missingness profile (no dropping)')
for var in key_vars_required:
    if var in df.columns:
        missing = df[var].isna().sum()
        pct = missing / len(df) * 100
        print(f'  [INFO] {var}: {missing:,} missing ({pct:.2f}%)')

# Check 2: Impossible-range checks only (diagnostic)
print()
print('CHECK 2: Impossible-range checks (no dropping)')
impossible_checks = [
    ('hhi_bank_q', 0, 1),
    ('call_capital_ratio', 0, 1),
    ('call_loans_to_assets', 0, 1),
    ('tenor_maturity', 0, None)
]

for var, lower, upper in impossible_checks:
    if var in df.columns:
        s = pd.to_numeric(df[var], errors='coerce')
        below = (s < lower).sum() if lower is not None else 0
        above = (s > upper).sum() if upper is not None else 0
        print(f'  [INFO] {var}: {below} below {lower}, {above} above {upper}')

# Check 3: Duplicates
print()
print('CHECK 3: Duplicates')
print(f'  Exact duplicate rows: {df.duplicated().sum():,}')
if 'ds_row_id' in df.columns:
    print(f"  Duplicate ds_row_id rows: {df.duplicated(['ds_row_id']).sum():,}")

analysis_key = ['deal_id', 'tranche_id', 'lender_id', 'gvkey', 'deal_date']
analysis_key = [k for k in analysis_key if k in df.columns]
print(f'  Duplicate analytical key rows ({analysis_key}): {df.duplicated(analysis_key).sum():,}')

if 'df_strict' in globals():
    strict_key = [k for k in ['deal_id', 'tranche_id', 'lender_id', 'gvkey', 'deal_date'] if k in df_strict.columns]
    print(f'  Strict alternative duplicate analytical key rows ({strict_key}): {df_strict.duplicated(strict_key).sum():,}')

if 'lender_category' in df.columns:
    print(f"  Foreign-bank rows remaining: {(df['lender_category'] == 'foreign_bank').sum():,}")

# Check 4: Time coverage
print()
print('CHECK 4: Time coverage')
print(f"  Year range: {df['year'].min()} - {df['year'].max()}")
print('  Observations per year:')
year_counts = df.groupby('year').size().sort_index()
for year, count in year_counts.items():
    print(f'    {year}: {count:,}')
