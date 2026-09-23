import os
# # FDIC Call Reports Data Cleaning
#
# **Purpose:** Clean and prepare FDIC Call Reports data for analysis of bank risk-taking and deposit market power
#
# **Input:**
# - `data/raw/fdic_call_reports/wrds_call_rcon_1__2001_2025.parquet`
# - `data/raw/fdic_call_reports/wrds_call_rcon_2__2001_2025.parquet`
# - `data/raw/fdic_call_reports/wrds_call_rcfd_2__2001_2025.parquet`
# - `data/raw/fdic_call_reports/wrds_call_riad_1__2001_2025.parquet`
#
# **Output:**
# - `data/intermediate/clean/call_reports_cleaned.parquet`
#
# **References:**
# - Drechsler, Savov & Schnabl (2017, QJE) - Winsorization and sample selection
# - Dell'Ariccia, Laeven & Suarez (2017, JF) - Merger/acquisition filtering
# - Boyd & De Nicoló (2005, JF) - Z-score construction
#
# **Last Updated:** 2026-01-18

# ## 1. Load and prepare each dataset separately (no early merge)
# We only load the required columns and filter each dataset before any merge.

# ### 1.1 Load Raw Data
#
# Load the specific Call Reports files needed for the selected variables.
#
# **Note:** We load RCON, RCFD, RIAD, and RCOW files separately and only merge after per-file selection.


import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# Define paths
base_path = Path(os.environ.get("MPHIL_THESIS_ROOT", Path(__file__).resolve().parents[2]))
raw_path = base_path / 'data/raw/fdic_call_reports'
clean_path = base_path / 'data/intermediate/clean'

# Select only needed columns to reduce memory use
id_cols = [
    'RSSD9001', 'WRDSREPORTDATE', 'RSSD9017', 'RSSD9050', 'RSSD9999',
    'RSSDFININSTFILINGTYPE'
]
merge_id_cols = ['RSSD9001', 'WRDSREPORTDATE']
dedupe_cols = ['RSSDSUBMISSIONDATE']

rcon1_vars = [
    'RCON0010', 'RCON1350', 'RCON1754', 'RCON1766', 'RCON1773', 'RCON5367',
    'RCON5368', 'RCON5597', 'RCON6810', 'RCONA555', 'RCONA557', 'RCONF055',
    'RCONF056', 'RCONF057', 'RCONF058'
]

rcon2_vars = [
    'RCON1403', 'RCON1407', 'RCON1797', 'RCON2122', 'RCON2170', 'RCON2200',
    'RCON2215', 'RCON2604', 'RCON2948', 'RCON3190', 'RCON3200', 'RCON3210',
    'RCON3545', 'RCON6648', 'RCON7204', 'RCON8274', 'RCONA223', 'RCONA241',
    'RCONA242', 'RCONA549', 'RCONA550', 'RCONA551', 'RCONA552', 'RCONA553',
    'RCONA554', 'RCONA556', 'RCONA558', 'RCONA559', 'RCONA560', 'RCONA570',
    'RCONA571', 'RCONA572', 'RCONA573', 'RCONA574', 'RCONA575', 'RCONA579',
    'RCONA580', 'RCONA581', 'RCONA582', 'RCONA584', 'RCONA585', 'RCONA586',
    'RCONA587', 'RCONF060', 'RCONF061', 'RCONF062', 'RCONF063', 'RCONHK11',
    'RCONK221', 'RCONK222'
]

rcfd2_vars = [
    'RCFD1403', 'RCFD1407', 'RCFD2122', 'RCFD2170', 'RCFD3210', 'RCFD7204',
    'RCFD8274', 'RCFDA223', 'RCFDA549', 'RCFDA550', 'RCFDA551', 'RCFDA552',
    'RCFDA553', 'RCFDA554'
]

rcfd1_vars = [
    'RCFDA555', 'RCFDA556', 'RCFDA557', 'RCFDA558', 'RCFDA559', 'RCFDA560',
    'RCFDA570', 'RCFDA571', 'RCFDA572', 'RCFDA573', 'RCFDA574', 'RCFDA575',
    'RCFDF055', 'RCFDF056', 'RCFDF057', 'RCFDF058'
]

rcow1_vars = ['RCOW3792', 'RCOW7205', 'RCOW7206', 'RCOWA223']

riad1_vars = [
    'RIAD4073', 'RIAD4074', 'RIAD4079', 'RIAD4107', 'RIAD4230', 'RIAD4301',
    'RIAD4340', 'RIAD4605', 'RIAD4635'
]

# Lowercase versions for parquet read
id_cols_read = [c.lower() for c in id_cols + dedupe_cols]
merge_id_cols_read = [c.lower() for c in merge_id_cols + dedupe_cols]

rcon1_cols = id_cols_read + [c.lower() for c in rcon1_vars]
rcon2_cols = merge_id_cols_read + [c.lower() for c in rcon2_vars]
rcfd1_cols = merge_id_cols_read + [c.lower() for c in rcfd1_vars]
rcfd2_cols = merge_id_cols_read + [c.lower() for c in rcfd2_vars]
rcow1_cols = merge_id_cols_read + [c.lower() for c in rcow1_vars]
riad1_cols = merge_id_cols_read + [c.lower() for c in riad1_vars]

# Load files (selected columns only)
print("Loading RCON_1 (domestic balance sheet items)...")
rcon1 = pd.read_parquet(raw_path / 'wrds_call_rcon_1__2001_2025.parquet', columns=rcon1_cols)
print(f"  Shape: {rcon1.shape}")

print("\nLoading RCON_2 (domestic balance sheet items)...")
rcon2 = pd.read_parquet(raw_path / 'wrds_call_rcon_2__2001_2025.parquet', columns=rcon2_cols)
print(f"  Shape: {rcon2.shape}")

print("\nLoading RCFD_1 (consolidated balance sheet items)...")
rcfd1 = pd.read_parquet(raw_path / 'wrds_call_rcfd_1__2001_2025.parquet', columns=rcfd1_cols)
print(f"  Shape: {rcfd1.shape}")

print("\nLoading RCFD_2 (consolidated balance sheet items)...")
rcfd2 = pd.read_parquet(raw_path / 'wrds_call_rcfd_2__2001_2025.parquet', columns=rcfd2_cols)
print(f"  Shape: {rcfd2.shape}")

print("\nLoading RCOW_1 (risk-based capital items)...")
rcow1 = pd.read_parquet(raw_path / 'wrds_call_rcow_1__2001_2025.parquet', columns=rcow1_cols)
print(f"  Shape: {rcow1.shape}")

print("\nLoading RIAD_1 (income statement items)...")
riad1 = pd.read_parquet(raw_path / 'wrds_call_riad_1__2001_2025.parquet', columns=riad1_cols)
print(f"  Shape: {riad1.shape}")

print("\n" + "="*80)
print("Raw data loaded successfully (selected columns)")

# Standardize column names for downstream steps
for _df in [rcon1, rcon2, rcfd1, rcfd2, rcow1, riad1]:
    _df.columns = _df.columns.str.upper()
    _df.rename(columns={'RSSD9001': 'RSSDID', 'WRDSREPORTDATE': 'REPDTE'}, inplace=True)

merge_keys = ['RSSDID', 'REPDTE']
extra_ids = ['RSSD9017', 'RSSD9050', 'RSSD9999', 'RSSDFININSTFILINGTYPE']
id_cols = merge_keys + extra_ids

def _dedupe_latest(df, keys=('RSSDID', 'REPDTE'), ts='RSSDSUBMISSIONDATE'):
    if ts not in df.columns:
        return df
    df = df.sort_values(list(keys) + [ts])
    return df.drop_duplicates(subset=list(keys), keep='last')

print("\nDeduplicating filings using latest RSSDSUBMISSIONDATE...")
rcon1 = _dedupe_latest(rcon1)
rcon2 = _dedupe_latest(rcon2)
rcfd1 = _dedupe_latest(rcfd1)
rcfd2 = _dedupe_latest(rcfd2)
rcow1 = _dedupe_latest(rcow1)
riad1 = _dedupe_latest(riad1)

# Map each variable to its source dataset (for reference)
dataset_map = {**{v: 'rcon1' for v in rcon1_vars},
               **{v: 'rcon2' for v in rcon2_vars},
               **{v: 'rcfd1' for v in rcfd1_vars},
               **{v: 'rcfd2' for v in rcfd2_vars},
               **{v: 'rcow1' for v in rcow1_vars},
               **{v: 'riad1' for v in riad1_vars}}
dataset_map = (pd.DataFrame({'column': list(dataset_map.keys()), 'source': list(dataset_map.values())})
               .sort_values(['source', 'column']))
dataset_map.head()


def _plot_hist(series, title, bins=50, q=(0.01, 0.99), log=False):
    series = series.dropna()
    if series.empty:
        return
    if log:
        series = series[series >= 0]
        if series.empty:
            return
        series = np.log10(series + 1)
        title = f"{title} (log10(1+x))"
    if q:
        lo, hi = series.quantile([q[0], q[1]])
        series = series[(series >= lo) & (series <= hi)]
        if series.empty:
            return
    plt.figure(figsize=(6, 4))
    plt.hist(series, bins=bins, color='#4C78A8', alpha=0.8)
    plt.title(title)
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.show()


# Subset after standardizing columns
rcon1_raw = rcon1.copy()
rcon2_raw = rcon2.copy()
rcfd1_raw = rcfd1.copy()
rcfd2_raw = rcfd2.copy()
rcow1_raw = rcow1.copy()
riad1_raw = riad1.copy()

print("Raw datasets (deduped, before subsetting):")
print("rcon1_raw:", rcon1_raw.shape)
print("rcon2_raw:", rcon2_raw.shape)
print("rcfd1_raw:", rcfd1_raw.shape)
print("rcfd2_raw:", rcfd2_raw.shape)
print("rcow1_raw:", rcow1_raw.shape)
print("riad1_raw:", riad1_raw.shape)

rcon1 = rcon1[id_cols + rcon1_vars].copy()
rcon2 = rcon2[merge_keys + rcon2_vars].copy()
rcfd1 = rcfd1[merge_keys + rcfd1_vars].copy()
rcfd2 = rcfd2[merge_keys + rcfd2_vars].copy()
rcow1 = rcow1[merge_keys + rcow1_vars].copy()
riad1 = riad1[merge_keys + riad1_vars].copy()

print("Prepared per-dataset subsets:")
print("rcon1:", rcon1.shape)
print("rcon2:", rcon2.shape)
print("rcfd1:", rcfd1.shape)
print("rcfd2:", rcfd2.shape)
print("rcow1:", rcow1.shape)
print("riad1:", riad1.shape)

rcon1.head()


rcon2.head()
rcfd1.head()
rcfd2.head()
rcow1.head()
riad1.head()


# ## 2. Merge Files (after per-file selection)
#
# Merge all files on RSSDID (bank identifier) and REPDTE (report date).
#
# **Cleaning rule:** Use full outer join to preserve all observations, then filter based on data availability.

# Merge reduced datasets only after per-dataset filtering
print(f"Merging on: {merge_keys}")

df = rcon1.merge(rcon2, on=merge_keys, how='outer')
df = df.merge(rcfd1, on=merge_keys, how='outer')
df = df.merge(rcfd2, on=merge_keys, how='outer')
df = df.merge(riad1, on=merge_keys, how='outer')
df = df.merge(rcow1, on=merge_keys, how='outer')

print(f"Final merged data: {df.shape}")
print(f"Date range: {df['REPDTE'].min()} to {df['REPDTE'].max()}")


df.head()

print(f"Date range: {df['REPDTE'].min()} to {df['REPDTE'].max()}")
print(f"Number of unique dates: {df['REPDTE'].nunique()}")
print(f"\nDate breakdown:")
print(df['REPDTE'].value_counts().sort_index())

# ## 3. Missingness after merge
# Check missing values for all merged variables before cleaning.

missing = df.isna().mean().sort_values(ascending=False)
missing.to_frame('missing_share')

# Build full column labels for reference (df columns remain MDRM codes)
mdrm_path = base_path / 'data/documentation/fdic_call_reports/MDRM_CSV.xlsx'
mdrm = pd.read_excel(mdrm_path, header=1)
mdrm = mdrm.rename(columns=lambda c: str(c).strip())
mdrm['Mnemonic'] = mdrm['Mnemonic'].astype(str).str.strip().str.upper()
mdrm['Item Code'] = mdrm['Item Code'].astype(str).str.strip().str.upper()
mdrm['Item Name'] = mdrm['Item Name'].astype(str).str.strip()

def _normalize_code(mnemonic, item_code):
    if item_code.isdigit():
        item_code = item_code.zfill(4)
    return f"{mnemonic}{item_code}"

mdrm['code'] = [
    _normalize_code(mn, code)
    for mn, code in zip(mdrm['Mnemonic'], mdrm['Item Code'])
]
label_map = {
    code: f"{name} ({code})"
    for code, name in zip(mdrm['code'], mdrm['Item Name'])
}
label_map = {c: label_map[c] for c in df.columns if c in label_map}
column_labels = (
    pd.DataFrame({'code': list(label_map.keys()), 'label': list(label_map.values())})
    .sort_values('code')
)
column_labels.head()


# df = df.rename(columns={'REPDTE': 'Date'})
label_map['RCON0010']


df.head()

# ## 3. Initial Sample Selection
#
# **Reference:** Drechsler et al. (2017) - Exclude non-commercial banks and territories
#
# **Rules:**
# 1. Require non-missing RSSDID (bank identifier)
# 2. Require valid report date (REPDTE)
# 3. Sample period: 2001-Q1 to 2025-Q4
# 4. Exclude U.S. territories

initial_count = len(df)
print(f"Initial observations: {initial_count:,}\n")

# 1. Require non-missing RSSDID
df = df[df['RSSDID'].notna()]
print(f"After requiring RSSDID: {len(df):,} ({(1-len(df)/initial_count)*100:.2f}% dropped)")

# 2. Convert REPDTE to datetime
df['REPDTE'] = pd.to_datetime(df['REPDTE'])
df = df[df['REPDTE'].notna()]
print(f"After requiring valid date: {len(df):,}")

# 3. Filter time period: 2001-2025
start_date = pd.to_datetime('2001-01-01')
end_date = pd.to_datetime('2025-12-31')
before = len(df)
df = df[(df['REPDTE'] >= start_date) & (df['REPDTE'] <= end_date)]
print(f"After time filter (2001-2025): {len(df):,} (dropped {before - len(df):,})")

# 4. Create quarter variable
df['year'] = df['REPDTE'].dt.year
df['quarter'] = df['REPDTE'].dt.quarter
df['year_qtr'] = df['year'].astype(str) + 'Q' + df['quarter'].astype(str)

print(f"\nDate range: {df['REPDTE'].min()} to {df['REPDTE'].max()}")
print(f"Unique banks (RSSDID): {df['RSSDID'].nunique():,}")
print(f"Unique quarters: {df['year_qtr'].nunique()}")

# ## 4. Balance Sheet Variable Cleaning
#
# ### 4.1 Total Assets (RCFD2170 or RCON2170)
#
# **Reference:** Drechsler et al. (2017) - Exclude banks with assets < $10M
#
# **Cleaning rules:**
# - Drop if assets ≤ 0
# - Drop if assets < $10 million (not for now, but consider later)
# - Use consolidated (RCFD2170) if available, otherwise domestic (RCON2170)

# Use consolidated assets if available, otherwise domestic
df['total_assets'] = df['RCFD2170'].fillna(df['RCON2170'])

print('Total Assets Cleaning:')
print(f"  Missing assets: {df['total_assets'].isna().sum():,}")
print(f"  Non-positive assets: {(df['total_assets'] <= 0).sum():,}")
print(f"  Assets < $10M (thousands): {(df['total_assets'] < 10000).sum():,}")

df = df[df['total_assets'] > 0]
# df = df[df['total_assets'] >= 10000]
df['total_assets_mil'] = df['total_assets'] / 1000
print(f"  Remaining after filters: {len(df):,}")

if 'total_assets_mil' in df.columns:
    _plot_hist(df['total_assets_mil'], 'Total Assets (USD mil)', log = True)


# ### 4.2 Filter Extreme Asset Growth (Mergers & Acquisitions)
#
#
# > "We exclude bank-quarters with asset growth exceeding 100% to avoid contamination from mergers and acquisitions."
#
# **Rule:** Flag (not drop) banks with quarterly asset growth > 100% or < -50%

# Calculate quarter-over-quarter asset growth
df = df.sort_values(['RSSDID', 'REPDTE'])
df['assets_lag1'] = df.groupby('RSSDID')['total_assets'].shift(1)
df['asset_growth_qoq'] = (df['total_assets'] - df['assets_lag1']) / df['assets_lag1']

# Flag extreme growth (likely M&A)
df['flag_extreme_growth'] = ((df['asset_growth_qoq'] > 1.0) |
                               (df['asset_growth_qoq'] < -0.5)).astype(int)

print("Asset Growth Filtering:")
print(f"  Obs with > 100% growth: {(df['asset_growth_qoq'] > 1.0).sum():,}")
print(f"  Obs with < -50% shrinkage: {(df['asset_growth_qoq'] < -0.5).sum():,}")
print(f"  Total flagged for extreme growth: {df['flag_extreme_growth'].sum():,}")
print(f"  % of sample: {df['flag_extreme_growth'].mean()*100:.2f}%")

print("\nExamples of extreme growth:")
extreme = df[df['flag_extreme_growth'] == 1][['RSSDID', 'year_qtr', 'total_assets_mil',
                                                'assets_lag1', 'asset_growth_qoq']].head(10)
print(extreme)

if 'asset_growth_qoq' in df.columns:
    _plot_hist(df['asset_growth_qoq'], 'Asset Growth QoQ')


# ### 4.3 Total Deposits (RCON2200)
#
# **Cleaning rules:**
# - Drop if deposits ≤ 0
# - Drop if deposits/assets > 1.1 (data error)
# - Calculate deposits-to-assets ratio for analysis

df['total_deposits'] = df['RCON2200']

print('Total Deposits Cleaning:')
print(f"  Missing deposits: {df['total_deposits'].isna().sum():,}")
print(f"  Non-positive deposits: {(df['total_deposits'] <= 0).sum():,}")

df = df[df['total_deposits'] > 0]
df['total_deposits_mil'] = df['total_deposits'] / 1000
df['deposits_to_assets'] = np.where(
    df['total_assets'] > 0,
    df['total_deposits'] / df['total_assets'],
    np.nan
)
df = df[(df['deposits_to_assets'].isna()) | (df['deposits_to_assets'] <= 1.1)]
print(f"  Remaining after filters: {len(df):,}")


if 'total_deposits_mil' in df.columns:
    _plot_hist(df['total_deposits_mil'], 'Total Deposits (USD mil)', log=True)


# ### 4.4 Equity Capital (RCON3210 / RCFD3210)
#
# **Reference:** Dell'Ariccia et al. (2017)
# > "We retain negative equity observations as they indicate financial distress, a key part of our analysis."
#
# **Cleaning rules:**
# - Keep negative equity (indicates distress)
# - Drop if equity/assets > 0.5 (unusual for commercial banks)
# - Flag negative equity observations

df['equity_capital'] = df['RCFD3210'].fillna(df['RCON3210'])
df['equity_capital_mil'] = df['equity_capital'] / 1000
df['capital_ratio'] = np.where(
    df['total_assets'] > 0,
    df['equity_capital'] / df['total_assets'],
    np.nan
)
df['flag_negative_equity'] = (df['equity_capital'] < 0).astype(int)

print('Equity Capital Cleaning:')
print(f"  Missing equity: {df['equity_capital'].isna().sum():,}")
print(f"  Negative equity: {df['flag_negative_equity'].sum():,}")

df = df[(df['capital_ratio'].isna()) | (df['capital_ratio'] <= 0.5)]
print(f"  Remaining after filters: {len(df):,}")


if 'equity_capital_mil' in df.columns:
    _plot_hist(df['equity_capital_mil'], 'Equity Capital (USD mil)', log=True)


# ### 4.5 Total Loans (RCON2122 / RCFD2122)
#
# **Cleaning rules:**
# - Required for NPL ratio and loan-to-deposit ratio
# - Drop if missing or ≤ 0

# Use RCFD2122 (consolidated) if available, otherwise RCON2122 (domestic)
df['total_loans'] = df['RCFD2122'].fillna(df['RCON2122'])

print("Total Loans Cleaning:")
print(f"  Missing loans: {df['total_loans'].isna().sum():,}")
print(f"  Zero loans: {(df['total_loans'] == 0).sum():,}")
print(f"  Negative loans: {(df['total_loans'] < 0).sum():,}")

# Keep observations with missing loans (banks may not have loans)
# But calculate ratios only when non-missing
df['total_loans_mil'] = df['total_loans'] / 1000
df['loans_to_assets'] = np.where(df['total_loans'].notna() & (df['total_assets'] > 0),
                                  df['total_loans'] / df['total_assets'],
                                  np.nan)

print("\nLoans-to-Assets Ratio (for banks with loans):")
print(df['loans_to_assets'].describe())

print(f"\nBanks with no lending activity (loans = 0 or missing): {df['total_loans'].isna().sum() + (df['total_loans'] == 0).sum():,}")


if 'total_loans_mil' in df.columns:
    _plot_hist(df['total_loans_mil'], 'Total Loans (USD mil)', log=True)


# ### 4.3 C&I Loans (RCON1766)
#
# **Reference:** Call Reports Schedule RC-C
#
# **Cleaning rules:**
# - Use domestic C&I loans (RCON1766)
# - Compute C&I loans in millions and ratios to assets/total loans

# print("C&I Loans Cleaning:")
# print(f"  Missing RCON1766: {df['RCON1766'].isna().sum():,}")
# print(f"  Zero C&I loans: {(df['RCON1766'] == 0).sum():,}")

# C&I loans (domestic)
# df['ci_loans'] = df['RCON1766']
# df['ci_loans_mil'] = df['ci_loans'] / 1000

# df['ci_loans_to_assets'] = np.where(
    # df['ci_loans'].notna() & (df['total_assets'] > 0),
    # df['ci_loans'] / df['total_assets'],
    # np.nan
# )

# df['ci_loans_share'] = np.where(
    # df['ci_loans'].notna() & (df['total_loans'] > 0),
    # df['ci_loans'] / df['total_loans'],
    # np.nan
# )

# print("C&I loans ratios (before winsorization):")
# print(df[['ci_loans_to_assets', 'ci_loans_share']].describe())

if 'ci_loans_mil' in df.columns:
    _plot_hist(df['ci_loans_mil'], 'C&I Loans (USD mil)', log=True)


# ## 5. Income Statement Variables
#
# ### 5.1 Net Income (RIAD4340)
#
# **Reference:** Drechsler et al. (2017)
# > "We keep negative values (losses) as they are informative."
#
# **Cleaning rules:**
# - Keep negative values (losses)
# - Calculate ROA (annualized quarterly)

# print("Net Income Cleaning:")
# print(f"  Missing net income: {df['RIAD4340'].isna().sum():,}")
# print(f"  Negative net income (losses): {(df['RIAD4340'] < 0).sum():,}")

# Convert to millions
# df['net_income_mil'] = df['RIAD4340'] / 1000

# Calculate ROA (annualized)
# ROA = (Net Income / Average Assets) * 400 (for quarterly data)
# df['assets_avg'] = (df['total_assets'] + df['assets_lag1']) / 2
# df['ROA'] = np.where((df['RIAD4340'].notna()) & (df['assets_avg'] > 0),
                     # (df['RIAD4340'] / df['assets_avg']) * 400,  # Annualize quarterly
                     # np.nan)

# print("\nROA (annualized %, before winsorization):")
# print(df['ROA'].describe())

# Identify extreme ROA values
# extreme_roa = df[(df['ROA'] < -10) | (df['ROA'] > 10)]
# print(f"\nObservations with |ROA| > 10%: {len(extreme_roa):,}")
# if len(extreme_roa) > 0:
    # print("Examples:")
    # print(extreme_roa[['RSSDID', 'year_qtr', 'net_income_mil', 'total_assets_mil', 'ROA']].head())

if 'ROA' in df.columns:
    _plot_hist(df['ROA'], 'ROA (annualized %)')


# ### 5.2 Interest Income and Expense
#
# **Cleaning rules:**
# - Must be ≥ 0
# - Calculate Net Interest Margin (NIM)

# print("Interest Income/Expense Cleaning:")
# print(f"  Missing interest income (RIAD4074): {df['RIAD4074'].isna().sum():,}")
# print(f"  Missing interest expense (RIAD4079): {df['RIAD4079'].isna().sum():,}")
# print(f"  Negative interest income: {(df['RIAD4074'] < 0).sum():,}")
# print(f"  Negative interest expense: {(df['RIAD4079'] < 0).sum():,}")

# Flag negative values (should investigate)
# df['flag_negative_int_income'] = (df['RIAD4074'] < 0).astype(int)
# df['flag_negative_int_expense'] = (df['RIAD4079'] < 0).astype(int)

# Set negative to missing (data error)
# df.loc[df['RIAD4074'] < 0, 'RIAD4074'] = np.nan
# df.loc[df['RIAD4079'] < 0, 'RIAD4079'] = np.nan

# Calculate Net Interest Income
# df['net_interest_income'] = df['RIAD4074'] - df['RIAD4079']

# Calculate Net Interest Margin (NIM) - annualized
# df['NIM'] = np.where((df['net_interest_income'].notna()) & (df['assets_avg'] > 0),
                     # (df['net_interest_income'] / df['assets_avg']) * 400,
                     # np.nan)

# print("\nNet Interest Margin (%, before winsorization):")
# print(df['NIM'].describe())

if 'NIM' in df.columns:
    _plot_hist(df['NIM'], 'Net Interest Margin (%)')


# ## 6. Credit Quality Variables
#
# ### 6.1 NPL Ratio
#
# **Reference:** Paligorova & Santos (2017)
# > "We focus on the upper tail of the risk distribution, so we winsorize NPL ratios more aggressively at the 99th percentile."
#
# **Construction:** NPL Ratio = (Past Due 90+ + Nonaccrual Loans) / Total Loans

# print("NPL Ratio Construction:")

# Check components
# print(f"  Missing RCFD1403 (Past Due 90+): {df['RCFD1403'].isna().sum():,}")
# print(f"  Missing RCON1403 (Past Due 90+): {df['RCON1403'].isna().sum():,}")
# print(f"  Missing RCFD1407 (Nonaccrual): {df['RCFD1407'].isna().sum():,}")
# print(f"  Missing RCON1407 (Nonaccrual): {df['RCON1407'].isna().sum():,}")
# print(f"  Missing total loans: {df['total_loans'].isna().sum():,}")

# Use consolidated values if available, otherwise domestic
# Do not coerce missing components to zero
# df['past_due_90'] = df['RCFD1403'].combine_first(df['RCON1403'])
# df['nonaccrual'] = df['RCFD1407'].combine_first(df['RCON1407'])

# Calculate NPL ratio only when both components exist and total loans > 0
# npl_ok = (
    # df['past_due_90'].notna()
    # & df['nonaccrual'].notna()
    # & df['total_loans'].notna()
    # & (df['total_loans'] > 0)
# )
# df['npl_ratio'] = np.where(npl_ok,
                           # (df['past_due_90'] + df['nonaccrual']) / df['total_loans'],
                           # np.nan)

# print("\nNPL Ratio (before winsorization):")
# print(df['npl_ratio'].describe())

# Identify extreme NPL ratios
# extreme_npl = df[df['npl_ratio'] > 0.5]
# print(f"\nObservations with NPL ratio > 50%: {len(extreme_npl):,}")
# if len(extreme_npl) > 0:
    # print("Examples (likely distressed banks):")
    # print(extreme_npl[['RSSDID', 'year_qtr', 'total_loans_mil', 'npl_ratio']].head())


if 'npl_ratio' in df.columns:
    _plot_hist(df['npl_ratio'], 'NPL Ratio')


# ## 7. Derived Variables
#
# ### 7.1 Liquidity Ratio
#
# **Reference:** Drechsler et al. (2017)
# > "We define liquid assets as cash, Treasuries, and agency securities."
#
# **Proxy (if Treasuries not available):** Cash / Total Assets

# print("Liquidity Ratio Construction:")
# print(f"  Missing RCON0010 (Cash): {df['RCON0010'].isna().sum():,}")
# print(f"  Missing RCON1754 (Treasuries): {df['RCON1754'].isna().sum():,}")
# print(f"  Missing RCON1773 (Agency securities): {df['RCON1773'].isna().sum():,}")

# Cash + Treasuries + Agency securities / Total Assets
# if 'total_assets' not in df.columns:
    # df['total_assets'] = df['RCFD2170'].fillna(df['RCON2170'])

# liq_numerator = df['RCON0010'].fillna(0) + df['RCON1754'].fillna(0) + df['RCON1773'].fillna(0)
# has_liq_component = df[['RCON0010', 'RCON1754', 'RCON1773']].notna().any(axis=1)

# df['liquidity_ratio'] = np.where(has_liq_component & (df['total_assets'] > 0),
                                 # liq_numerator / df['total_assets'],
                                 # np.nan)

# Validate range [0, 1]
# errors = df[(df['liquidity_ratio'] < 0) | (df['liquidity_ratio'] > 1)]
# print(f"  Observations with liquidity ratio outside [0,1]: {len(errors):,}")

# print("\nLiquidity Ratio (before winsorization):")
# print(df['liquidity_ratio'].describe())

if 'liquidity_ratio' in df.columns:
    _plot_hist(df['liquidity_ratio'], 'Liquidity Ratio')

# ### 7.2 Z-Score
#
# **Reference:** Boyd & De Nicoló (2005, JF)
# > "Z-score requires at least 8 quarters to obtain stable volatility estimates."
#
# **Construction:** Z-score = (ROA + Capital Ratio) / σ(ROA)
#
# where σ(ROA) is rolling 12-quarter standard deviation

# print("Z-Score Construction:")

# Sort by bank and date
# df = df.sort_values(['RSSDID', 'REPDTE'])

# Calculate rolling 12-quarter SD of ROA (require min 8 observations)
# df['roa_sd_12q'] = df.groupby('RSSDID')['ROA'].transform(
    # lambda x: x.rolling(window=12, min_periods=8).std()
# )

# print(f"  Observations with sufficient data for SD calculation: {df['roa_sd_12q'].notna().sum():,}")
# print(f"  % of sample: {df['roa_sd_12q'].notna().mean()*100:.2f}%")

# Calculate Z-score
# df['z_score'] = np.where((df['ROA'].notna()) &
                         # (df['capital_ratio'].notna()) &
                         # (df['roa_sd_12q'].notna()) &
                         # (df['roa_sd_12q'] > 0),
                         # (df['ROA'] + df['capital_ratio'] * 100) / df['roa_sd_12q'],
                         # np.nan)

# print("\nZ-Score (before winsorization):")
# print(df['z_score'].describe())

# print(f"\nObservations with Z-score: {df['z_score'].notna().sum():,}")
# print(f"% of sample: {df['z_score'].notna().mean()*100:.2f}%")

if 'z_score' in df.columns:
    _plot_hist(df['z_score'], 'Z-Score')


# ## 8. Winsorization
#
# **Reference:** Drechsler et al. (2017, QJE)
# > "We winsorize all bank-level variables at the 1st and 99th percentiles to mitigate the influence of outliers."
#
# **Rule:** Winsorize continuous variables at 1% and 99% within each quarter

# def winsorize_by_group(data, var, group, lower=0.01, upper=0.99):
    # """
    # Winsorize variable within groups at specified percentiles.

    # Parameters:
    # - data: DataFrame
    # - var: variable name to winsorize
    # - group: grouping variable (e.g., 'year_qtr')
    # - lower, upper: percentiles (default: 1% and 99%)

    # Returns:
    # - Series with winsorized values
    # """
    # return data.groupby(group)[var].transform(
        # lambda x: x.clip(lower=x.quantile(lower), upper=x.quantile(upper))
    # )

# print("Winsorizing variables at 1% and 99% within each quarter...\n")

# List of variables to winsorize
# vars_to_winsorize = [
    # ('total_assets_mil', 'total_assets_mil_w'),
    # ('total_deposits_mil', 'total_deposits_mil_w'),
    # ('equity_capital_mil', 'equity_capital_mil_w'),
    # ('total_loans_mil', 'total_loans_mil_w'),
    # ('ci_loans_mil', 'ci_loans_mil_w'),
    # ('deposits_to_assets', 'deposits_to_assets_w'),
    # ('capital_ratio', 'capital_ratio_w'),
    # ('loans_to_assets', 'loans_to_assets_w'),
    # ('ci_loans_to_assets', 'ci_loans_to_assets_w'),
    # ('ci_loans_share', 'ci_loans_share_w'),
    # ('ROA', 'ROA_w'),
    # ('NIM', 'NIM_w'),
    # ('liquidity_ratio', 'liquidity_ratio_w'),
# ]

# Winsorize each variable
# for orig_var, wins_var in vars_to_winsorize:
    # if orig_var in df.columns:
        # df[wins_var] = winsorize_by_group(df, orig_var, 'year_qtr')

        # Show effect of winsorization
        # before = df[orig_var].describe()
        # after = df[wins_var].describe()

        # print(f"{orig_var}:")
        # print(f"  Mean: {before['mean']:.4f} → {after['mean']:.4f}")
        # print(f"  Std:  {before['std']:.4f} → {after['std']:.4f}")
        # print(f"  Min:  {before['min']:.4f} → {after['min']:.4f}")
        # print(f"  Max:  {before['max']:.4f} → {after['max']:.4f}")
        # print()

# Winsorize NPL ratio at 95% and 99% (asymmetric - focus on upper tail)
# print("NPL Ratio (asymmetric winsorization at 95% and 99%):")
# df['npl_ratio_w'] = winsorize_by_group(df, 'npl_ratio', 'year_qtr', lower=0.0, upper=0.99)
# print(df[['npl_ratio', 'npl_ratio_w']].describe())

# Winsorize Z-score at 1% and 99%
# print("\nZ-Score winsorization:")
# df['z_score_w'] = winsorize_by_group(df, 'z_score', 'year_qtr')
# print(df[['z_score', 'z_score_w']].describe())

# ## 5. Distribution plots (key cleaned variables)
# Visual check of winsorized variables for outliers and skewness.


import matplotlib.pyplot as plt

plots_dir = clean_path / 'fdic_call_reports' / 'plots'
plots_dir.mkdir(parents=True, exist_ok=True)

log_vars = {
    'total_assets_mil',
    'total_deposits_mil',
    'equity_capital_mil',
    'total_loans_mil'
}

plot_vars = [
    'total_assets_mil',
    'total_deposits_mil',
    'equity_capital_mil',
    'total_loans_mil',
    'deposits_to_assets',
    'capital_ratio',
    'loans_to_assets',
    'ROA',
    'NIM',
    'npl_ratio',
    'liquidity_ratio',
    'z_score'
]

plot_vars = [v for v in plot_vars if v in df.columns]

n = len(plot_vars)
cols = 3
rows = (n + cols - 1) // cols

plt.figure(figsize=(cols * 5, rows * 3.5))
for i, var in enumerate(plot_vars, 1):
    ax = plt.subplot(rows, cols, i)
    series = df[var].dropna()
    if var in log_vars:
        series = series[series >= 0]
        series = np.log10(series + 1)
        title = f"log10(1+{var})"
    else:
        title = var
    ax.hist(series, bins=50, color='#4C78A8', alpha=0.8)
    ax.set_title(title)
    ax.grid(alpha=0.2)

plt.tight_layout()
plt.savefig(plots_dir / 'dist_key_vars.png', dpi=150, bbox_inches='tight')
plt.show()

# ## 5.1 Distribution plots by section
# Each section plots a small set of related variables to make skewness easier to spot.


import matplotlib.pyplot as plt

plots_dir = clean_path / 'fdic_call_reports' / 'plots'
plots_dir.mkdir(parents=True, exist_ok=True)

log_vars = {
    'total_assets_mil',
    'total_deposits_mil',
    'equity_capital_mil',
    'total_loans_mil'
}

sections = {
    'Balance sheet levels': [
        'total_assets_mil',
        'total_deposits_mil',
        'equity_capital_mil',
        'total_loans_mil'
    ],
    'Ratios': [
        'deposits_to_assets',
        'capital_ratio',
        'loans_to_assets',
        'liquidity_ratio'
    ],
    'Profitability / interest': [
        'ROA',
        'NIM'
    ],
    'Risk': [
        'npl_ratio',
        'z_score'
    ]
}

for section_title, vars_list in sections.items():
    vars_list = [v for v in vars_list if v in df.columns]
    if not vars_list:
        continue

    plot_vars = []
    for v in vars_list:
        series = df[v].dropna()
        if not series.empty:
            plot_vars.append(v)

    if not plot_vars:
        continue

    cols = 2
    rows = (len(plot_vars) + cols - 1) // cols
    plt.figure(figsize=(cols * 5, rows * 3.5))
    for i, var in enumerate(plot_vars, 1):
        ax = plt.subplot(rows, cols, i)
        series = df[var].dropna()
        if var in log_vars:
            series = series[series >= 0]
            series = np.log10(series + 1)
            plot_title = f"log10(1+{var})"
        else:
            plot_title = var
        ax.hist(series, bins=50, color='#4C78A8', alpha=0.8)
        ax.set_title(plot_title)
        ax.grid(alpha=0.2)

    plt.suptitle(section_title, y=1.02, fontsize=12)
    plt.tight_layout()
    section_slug = section_title.lower().replace(' ', '_').replace('/', '_')
    plt.savefig(plots_dir / f"dist_section_{section_slug}.png", dpi=150, bbox_inches='tight')
    plt.show()


# ## 9. Final Sample Selection
#
# ### 9.1 Minimum Continuity Requirement
#
# **Rule:** Banks must have at least 4 consecutive quarters of non-missing data

print("Minimum Continuity Filter:")
print(f"  Before filter: {len(df):,} observations")
print(f"  Unique banks: {df['RSSDID'].nunique():,}")

# Count observations per bank
bank_counts = df.groupby('RSSDID').size()
print(f"\n  Banks with < 4 quarters: {(bank_counts < 4).sum():,}")
print(f"  Banks with ≥ 4 quarters: {(bank_counts >= 4).sum():,}")
print(f"  Banks with ≥ 8 quarters: {(bank_counts >= 8).sum():,}")

# Keep banks with at least 4 quarters
banks_to_keep = bank_counts[bank_counts >= 4].index
df = df[df['RSSDID'].isin(banks_to_keep)]

print(f"\n  After requiring ≥ 4 quarters: {len(df):,} observations")
print(f"  Unique banks: {df['RSSDID'].nunique():,}")

# Report distribution of observations per bank
bank_counts_final = df.groupby('RSSDID').size()
print("\n  Distribution of observations per bank:")
print(bank_counts_final.describe())

# ## 10. Data Quality Summary
#
# Generate summary statistics and data quality report before saving.

print("="*80)
print("FINAL CLEANED DATASET SUMMARY")
print("="*80)

print(f"\nTotal observations: {len(df):,}")
print(f"Unique banks (RSSDID): {df['RSSDID'].nunique():,}")
print(f"Date range: {df['REPDTE'].min()} to {df['REPDTE'].max()}")
print(f"Unique quarters: {df['year_qtr'].nunique()}")

print("\nSample composition by year:")
year_summary = df.groupby('year').agg({
    'RSSDID': 'nunique',
    'total_assets_mil': 'sum'
}).rename(columns={'RSSDID': 'num_banks', 'total_assets_mil': 'total_assets_bil'})
year_summary['total_assets_bil'] = year_summary['total_assets_bil'] / 1000
print(year_summary)

print("\n" + "="*80)
print("KEY VARIABLES - MISSINGNESS")
print("="*80)

key_vars = [
    'total_assets_mil',
    'total_deposits_mil',
    'equity_capital_mil',
    'total_loans_mil',
    'deposits_to_assets',
    'capital_ratio',
    'ROA',
    'NIM',
    'npl_ratio',
    'liquidity_ratio',
    'z_score'
]

for var in key_vars:
    if var in df.columns:
        missing = df[var].isna().sum()
        pct_missing = missing / len(df) * 100
        print(f"{var:30s}: {missing:8,} missing ({pct_missing:5.2f}%)")

print("\n" + "="*80)
print("FLAGGED OBSERVATIONS")
print("="*80)

flag_vars = [
    'flag_extreme_growth',
    'flag_negative_equity',
    'flag_negative_int_income',
    'flag_negative_int_expense'
]

for var in flag_vars:
    if var in df.columns:
        count = df[var].sum()
        pct = count / len(df) * 100
        print(f"{var:30s}: {count:8,} ({pct:5.2f}%)")

# ## 11. Select Final Variables and Save
#
# Select key variables for analysis and save cleaned dataset.

# Define final variable list
raw_selected_cols = [
    # RCON 1
    'RCON0010', 'RCON1350', 'RCON1766', 'RCON5367', 'RCON5368', 'RCON5597',
    'RCON6810', 'RCONA555', 'RCONA557', 'RCONF055', 'RCONF056', 'RCONF057',
    'RCONF058',
    # RCON 2
    'RCON1403', 'RCON1407', 'RCON1797', 'RCON2122', 'RCON2170', 'RCON2200',
    'RCON2215', 'RCON2604', 'RCON2948', 'RCON3190', 'RCON3200', 'RCON3210',
    'RCON3545', 'RCON6648', 'RCON7204', 'RCON8274', 'RCONA223', 'RCONA241',
    'RCONA242', 'RCONA549', 'RCONA550', 'RCONA551', 'RCONA552', 'RCONA553',
    'RCONA554', 'RCONA556', 'RCONA558', 'RCONA559', 'RCONA560', 'RCONA570',
    'RCONA571', 'RCONA572', 'RCONA573', 'RCONA574', 'RCONA575', 'RCONA579',
    'RCONA580', 'RCONA581', 'RCONA582', 'RCONA584', 'RCONA585', 'RCONA586',
    'RCONA587', 'RCONF060', 'RCONF061', 'RCONF062', 'RCONF063', 'RCONHK11',
    'RCONK221', 'RCONK222',
    # RCFD 1/2
    'RCFD1403', 'RCFD1407', 'RCFD2122', 'RCFD2170', 'RCFD3210', 'RCFD7204',
    'RCFD8274', 'RCFDA223', 'RCFDA549', 'RCFDA550', 'RCFDA551', 'RCFDA552',
    'RCFDA553', 'RCFDA554', 'RCFDA555', 'RCFDA556', 'RCFDA557', 'RCFDA558',
    'RCFDA559', 'RCFDA560', 'RCFDA570', 'RCFDA571', 'RCFDA572', 'RCFDA573',
    'RCFDA574', 'RCFDA575', 'RCFDF055', 'RCFDF056', 'RCFDF057', 'RCFDF058',
    # RCOW
    'RCOW3792', 'RCOW7205', 'RCOW7206', 'RCOWA223',
    # RIAD
    'RIAD4073', 'RIAD4074', 'RIAD4079', 'RIAD4107', 'RIAD4230', 'RIAD4301',
    'RIAD4340', 'RIAD4605', 'RIAD4635'
]
raw_selected_cols = [c for c in raw_selected_cols if c in df.columns]

id_vars = [
    'RSSDID', 'REPDTE', 'RSSD9017', 'RSSD9050', 'RSSD9999',
    'RSSDFININSTFILINGTYPE', 'year', 'quarter', 'year_qtr'
]

# Derived and cleaned variables (non-winsorized)
derived_vars = [
    # Balance sheet
    'total_assets_mil', 'total_deposits_mil', 'equity_capital_mil',
    'total_loans_mil', 'ci_loans_mil',
    # Ratios (non-winsorized)
    'deposits_to_assets', 'capital_ratio', 'loans_to_assets',
    'ci_loans_to_assets', 'ci_loans_share', 'ROA', 'NIM',
    'npl_ratio', 'liquidity_ratio', 'z_score',
    # Growth and flags
    'asset_growth_qoq', 'flag_extreme_growth', 'flag_negative_equity',
    'flag_negative_int_income', 'flag_negative_int_expense',
    # Additional useful variables
    'net_income_mil', 'net_interest_income', 'roa_sd_12q'
]

final_vars = id_vars + raw_selected_cols + derived_vars
final_vars = [c for c in final_vars if c in df.columns]

# De-duplicate while preserving order
seen = set()
final_vars = [c for c in final_vars if not (c in seen or seen.add(c))]

# Select final dataset
df_clean = df[final_vars].copy()

print(f"Final cleaned dataset shape: {df_clean.shape}")
print(f"\nVariables included: {len(final_vars)}")


# ## 12. Preview Cleaned Data
#
# Show sample of cleaned data for verification.

print("Sample of cleaned data (first 20 rows):")
print(df_clean.head(20))

print("\nSummary statistics (key variables):")
summary_cols = [
    'total_assets_mil', 'total_deposits_mil', 'equity_capital_mil',
    'total_loans_mil', 'ci_loans_mil', 'deposits_to_assets',
    'capital_ratio', 'loans_to_assets', 'ci_loans_to_assets',
    'ci_loans_share', 'ROA', 'NIM', 'npl_ratio', 'liquidity_ratio',
    'z_score'
]
summary_cols = [c for c in summary_cols if c in df_clean.columns]
print(df_clean[summary_cols].describe())

print("\nData types (key variables):")
print(df_clean[summary_cols].dtypes)


# Save cleaned dataset (non-winsorized)
output_dir = clean_path / "fdic_call_reports"
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / "fdic_call_reports_clean.parquet"
df_clean.to_parquet(output_path, index=False)
print(f"Saved cleaned dataset to: {output_path}")
print(f"Rows: {len(df_clean):,} | Columns: {df_clean.shape[1]:,}")
