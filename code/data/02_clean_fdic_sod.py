import os
# # FDIC Summary of Deposits (SOD) Cleaning
#
# Purpose: Basic cleaning and HHI construction following Drechsler, Savov, and Schnabl (2017).
# Notes:
# - Save steps are commented out until you review the outputs.
# - This notebook keeps monopoly counties (HHI can be 1.0).

# ## 1. Setup and load data
# Load the raw SOD parquet file and confirm basic structure.

# DEPSUMBR: Branch deposits (in thousands of dollars). Branch office deposits as of June 30.
# STNUMBR: State number (branch). State number that corresponds to the state in which the branch is located.
# CNTYNUMB: County number (branch). County number that corresponds to the county in which the branch is located.
# STCNTYBR: State & County number (branch). The state and county FIPS code associated with the specific branch location.
# STALPBR	: State (branch). The two‑letter U.S. Postal Service abbreviation for the state in which the branch is located.
# RSSDID (或 CERT): Bank identifier to aggregate branch deposits to bank‑county.
# UNINUMBR: Urban influence code (branch). A classification scheme that distinguishes metropolitan counties by the population size of their metro area, and nonmetropolitan counties by their degree of urbanization and adjacency to metro areas.
# YEAR: Report year (annual, June 30).

import pandas as pd
import numpy as np
from pathlib import Path

pd.set_option('display.max_columns', 120)
pd.set_option('display.max_rows', 100)

base_path = Path(os.environ.get("MPHIL_THESIS_ROOT", Path(__file__).resolve().parents[2]))
raw_file = base_path / 'data/raw/fdic_sod/fdic_sod__2001_2025.parquet'
output_file = base_path / 'data/intermediate/clean/fdic_sod/fdic_sod_cleaned.parquet'

selected_cols = [
    'CERT', 'CNTYNUMB', 'STNUMBR', 'DEPSUMBR', 'RSSDID', 'STCNTYBR', 'UNINUMBR',
    'YEAR', 'BKCLASS'
]

df = pd.read_parquet(raw_file, columns=selected_cols)
print(f"Initial shape: {df.shape}")
df.head()


# ## 2. Standardize columns and check required fields
# We standardize column names and verify required fields for HHI construction.

df.columns = df.columns.str.upper()

required_cols = ['CERT', 'CNTYNUMB', 'STNUMBR', 'DEPSUMBR', 'RSSDID', 'STCNTYBR', 'UNINUMBR','YEAR']
missing_required = [c for c in required_cols if c not in df.columns]
print("Missing required columns:", missing_required if missing_required else "None")

missing_counts = df[required_cols].isna().sum().sort_values(ascending=False)
missing_counts

# ## 3. Sample selection and deposit cleaning
# Apply the 2001-2025 window and drop branches with missing or non-positive deposits.

df['YEAR'] = pd.to_numeric(df['YEAR'], errors='coerce')
df['DEPSUMBR'] = pd.to_numeric(df['DEPSUMBR'], errors='coerce')

before = len(df)
df = df[(df['YEAR'] >= 2001) & (df['YEAR'] <= 2025)]
print(f"After time filter: {len(df):,} (dropped {before - len(df):,})")

before = len(df)
df = df[(df['DEPSUMBR'] > 0) & (df['DEPSUMBR'].notna())]
print(f"After deposit filter: {len(df):,} (dropped {before - len(df):,})")

df['RSSDID'] = pd.to_numeric(df['RSSDID'], errors='coerce')
before = len(df)
df = df[df['RSSDID'].notna() & (df['RSSDID'] > 0)]
print(f"After RSSDID filter: {len(df):,} (dropped {before - len(df):,})")


df['DEPSUMBR'].describe()

# ## 4. Geography filter and county identifier
# Exclude territories and build a county identifier (STCNTYBR if available).

# STNUMBR: State number (branch). State number that corresponds to the state in which the branch is located.
# CNTYNUMB: County number (branch). County number that corresponds to the county in which the branch is located.
# STCNTYBR: State & County number (branch). The state and county FIPS code associated with the specific branch location.
# STALPBR	: State (branch). The two‑letter U.S. Postal Service abbreviation for the state in which the branch is located.

# STCNTYBR is preferred as it directly maps to county FIPS codes. If missing, we construct it using STNUMBR and CNTYNUMB.

before = len(df)
print("Before generating county_id:", len(df))
if 'STCNTYBR' in df.columns:
    stcnty = pd.to_numeric(df['STCNTYBR'], errors='coerce')
    df['county_id'] = stcnty.astype('Int64').astype(str).str.zfill(5)
    print("Using STCNTYBR as county_id")

else:
    stnum = pd.to_numeric(df['STNUMBR'], errors='coerce').astype('Int64')
    cnty = pd.to_numeric(df['CNTYNUMB'], errors='coerce').astype('Int64')
    df['county_id'] = stnum.astype(str).str.zfill(2) + cnty.astype(str).str.zfill(3)
    df.loc[stnum.isna() | cnty.isna(), 'county_id'] = np.nan
    print("Using STNUMBR + CNTYNUMB as county_id")


print("Missing county_id:", df['county_id'].isna().sum())
before = len(df)
df = df[df['county_id'].notna()]
print(f"After requiring county_id: {len(df):,} (dropped {before - len(df):,})")


# Exclude territories using FIPS state codes
territory_fips = {'60','66','69','72','78'}  # AS, GU, MP, PR, VI
before = len(df)
stnum = pd.to_numeric(df['STNUMBR'], errors='coerce').astype('Int64')
df = df[stnum.notna()]
df = df[~stnum.astype(str).str.zfill(2).isin(territory_fips)]
print(f"After excluding territories: {len(df):,} (dropped {before - len(df):,})")

df[['STNUMBR', 'CNTYNUMB', 'county_id']].head()

# ## 5. County HHI calculation
# Compute county-level HHI from bank market shares within each county.

# Aggregate to bank-county level (use RSSDID)
bank_county = df.groupby(['YEAR', 'county_id', 'RSSDID'], as_index=False)['DEPSUMBR'].sum()

# County totals and market shares
county_totals = bank_county.groupby(['YEAR', 'county_id'], as_index=False)['DEPSUMBR'].sum()
county_totals = county_totals.rename(columns={'DEPSUMBR': 'county_total_deposits'})

bank_county = bank_county.merge(county_totals, on=['YEAR', 'county_id'], how='left')
bank_county['market_share'] = bank_county['DEPSUMBR'] / bank_county['county_total_deposits']

# County HHI
county_hhi = (
    bank_county.groupby(['YEAR', 'county_id'])['market_share']
    .apply(lambda x: (x ** 2).sum())
    .reset_index(name='hhi_county')
)

county_hhi['hhi_county'].describe()

county_hhi.head()

import matplotlib.pyplot as plt

plt.figure(figsize=(6,4))
plt.hist(county_hhi['hhi_county'].dropna(), bins=50, color='#4C78A8', alpha=0.8)
plt.title('County HHI Distribution')
plt.xlabel('HHI')
plt.ylabel('Count')
plt.grid(alpha=0.2)
plt.tight_layout()
plt.show()

hhi_mean = county_hhi['hhi_county'].mean()
hhi_std = county_hhi['hhi_county'].std()

print(f"HHI mean: {hhi_mean:.6f}")
print(f"HHI std:  {hhi_std:.6f}")

# 1) HHI = 1 的县
hhi_one = county_hhi[county_hhi['hhi_county'] == 1].copy()
print("HHI=1 counties:", len(hhi_one))

# 2) 查看年份分布
print(hhi_one['YEAR'].value_counts().sort_index())

# 3) 在这些县里，银行数是否为 1
bc_one = bank_county.merge(hhi_one[['YEAR', 'county_id']], on=['YEAR', 'county_id'], how='inner')
bc_one['bank_count'] = bc_one.groupby(['YEAR', 'county_id'])['RSSDID'].transform('nunique')
print(bc_one['bank_count'].value_counts().sort_index())



# ## 6. Bank-level deposit-weighted HHI
# Aggregate county HHIs to the bank level using deposit weights.

# Bank totals
bank_totals = bank_county.groupby(['YEAR', 'RSSDID'], as_index=False)['DEPSUMBR'].sum()
bank_totals = bank_totals.rename(columns={'DEPSUMBR': 'bank_total_deposits'})

bank_county = bank_county.merge(bank_totals, on=['YEAR', 'RSSDID'], how='left')
bank_county = bank_county.merge(county_hhi, on=['YEAR', 'county_id'], how='left')

bank_county['deposit_weight'] = np.where(
    bank_county['bank_total_deposits'] > 0,
    bank_county['DEPSUMBR'] / bank_county['bank_total_deposits'],
    np.nan
)
bank_county['weighted_hhi'] = bank_county['deposit_weight'] * bank_county['hhi_county']

bank_hhi = (
    bank_county.groupby(['YEAR', 'RSSDID'], as_index=False)['weighted_hhi']
    .sum()
    .rename(columns={'weighted_hhi': 'hhi_bank'})
)

# Attach CERT if available (match on YEAR + RSSDID)
if 'CERT' in df.columns:
    cert_rssd = df[['YEAR', 'CERT', 'RSSDID']].drop_duplicates()
    bank_hhi = bank_hhi.merge(cert_rssd, on=['YEAR', 'RSSDID'], how='left')

# Attach bank type if available
if 'BKCLASS' in df.columns:
    bank_type = df[['YEAR', 'RSSDID', 'BKCLASS']].drop_duplicates()
    bank_hhi = bank_hhi.merge(bank_type, on=['YEAR', 'RSSDID'], how='left')

# Descriptive stats
print(bank_hhi['hhi_bank'].describe())

bank_totals.head(10)


# Histogram
import matplotlib.pyplot as plt
output_dir = base_path / 'data/intermediate/clean/fdic_sod'
output_dir.mkdir(parents=True, exist_ok=True)

plt.figure(figsize=(6,4))
plt.hist(bank_hhi['hhi_bank'].dropna(), bins=50, color='#4C78A8', alpha=0.8)
plt.title('Bank HHI Distribution')
plt.xlabel('HHI (bank)')
plt.ylabel('Count')
plt.grid(alpha=0.2)
plt.tight_layout()
plt.show()


# ## 7. Winsorize and finalize
# Winsorize HHI at 1% and 99% by year and prepare final fields.

bank_hhi = bank_hhi.merge(bank_totals, on=['YEAR', 'RSSDID'], how='left')
bank_hhi['bank_total_deposits_mil'] = bank_hhi['bank_total_deposits'] / 1000

# Winsorization disabled - using raw HHI
# bank_hhi['hhi_bank_w'] = bank_hhi.groupby('YEAR')['hhi_bank'].transform(
#     lambda x: x.clip(lower=x.quantile(0.01), upper=x.quantile(0.99))
# )

bank_hhi[['hhi_bank']].describe()

# Summary statistics and distribution plots for key variables (non-winsorized)
key_vars = ['hhi_bank', 'bank_total_deposits_mil']
key_vars = [c for c in key_vars if c in bank_hhi.columns]

print("\nSummary statistics (key variables):")
print(bank_hhi[key_vars].describe())

import matplotlib.pyplot as plt

n = len(key_vars)
cols = 2
rows = (n + cols - 1) // cols

plt.figure(figsize=(cols * 5, rows * 3.5))
for i, var in enumerate(key_vars, 1):
    ax = plt.subplot(rows, cols, i)
    series = bank_hhi[var].dropna()
    if var == 'bank_total_deposits_mil':
        series = series[series >= 0]
        series = np.log10(series + 1)
        title = 'log10(1+bank_total_deposits_mil)'
    else:
        title = var
    ax.hist(series, bins=50, color='#4C78A8', alpha=0.8)
    ax.set_title(title)
    ax.grid(alpha=0.2)
plt.savefig(output_dir / 'bank_hhi_hist.png', dpi=150, bbox_inches='tight')
plt.tight_layout()
plt.show()
print(f"Saved histogram to: {output_dir / 'bank_hhi_hist.png'}")


# Save summary stats for key variables
summary_path = output_dir / 'bank_hhi_keyvars_summary_stats.csv'
bank_hhi[key_vars].describe().to_csv(summary_path)
print(f"Saved summary stats to: {summary_path}")
bank_hhi.head()


# Save summary stats for key variables
summary_path = output_dir / 'bank_hhi_keyvars_summary_stats.csv'
bank_hhi[key_vars].describe().to_csv(summary_path)
print(f"Saved summary stats to: {summary_path}")

bank_hhi.head()

# ## 8. Save output (commented)
# Review the outputs first. Uncomment the save line when you are ready.

# Ensure unique bank-year key before save
before = len(bank_hhi)
sort_cols = ['YEAR', 'RSSDID'] + (['CERT'] if 'CERT' in bank_hhi.columns else [])
bank_hhi = bank_hhi.sort_values(sort_cols, na_position='last').drop_duplicates(['YEAR', 'RSSDID'], keep='first')
print(f"After enforcing unique YEAR-RSSDID: {len(bank_hhi):,} (dropped {before - len(bank_hhi):,})")

output_file
bank_hhi.to_parquet(output_file, index=False)
