import os
# # DealScan Cleaning
#
# Purpose: Basic loan-level cleaning for DealScan following Paligorova and Santos (2017) and Sufi (2007).
# Notes:
# - Save steps are commented out until you review the outputs.
# - The dataset is loan-lender level; filtering can reduce the sample substantially.

# ## 1. Setup and load data
# Load the raw DealScan parquet file and check basic structure.

import pandas as pd
import numpy as np
from pathlib import Path

pd.set_option('display.max_columns', 120)
pd.set_option('display.max_rows', 100)

base_path = Path(os.environ.get("MPHIL_THESIS_ROOT", Path(__file__).resolve().parents[2]))
raw_file = base_path / 'data/raw/dealscan/dealscan__2001_2025.parquet'
output_file = base_path / 'data/intermediate/clean/dealscan/dealscan_cleaned.parquet'

selected_cols = [
     'borrower_id', 'borrower_name', 'all_in_spread_drawn_bps',
    'base_reference_rate', 'collateral_security_type', 'country', 'covenants',
    'deal_active_date', 'deal_amount_converted', 'deal_purpose',
    'lead_arranger', 'lender_id', 'lender_name', 'lender_parent_id',
    'lender_share', 'lender_commit', 'lpc_deal_id', 'lpc_tranche_id',
    'mandated_date', 'margin_bps', 'number_of_lead_arrangers',
    'number_of_lenders', 'perm_id', 'primary_purpose', 'primary_role',
    'secondary_purpose', 'secured', 'sic_code', 'tenor_maturity', 'ticker',
    'tranche_active_date', 'tranche_amount_converted', 'tranche_type'
]

df = pd.read_parquet(raw_file, columns=selected_cols)
print(f"Initial shape: {df.shape}")
df.head()


# ## 2. Parse dates and build deal_date
# We use deal_active_date when available; otherwise we fall back to tranche_active_date.

df['deal_active_date'] = pd.to_datetime(df['deal_active_date'], errors='coerce')
df['tranche_active_date'] = pd.to_datetime(df['tranche_active_date'], errors='coerce')

df['deal_date'] = df['deal_active_date'].fillna(df['tranche_active_date'])
print("Missing deal_date:", df['deal_date'].isna().sum())
print("Date range:", df['deal_date'].min(), "to", df['deal_date'].max())

df[['deal_active_date', 'tranche_active_date', 'deal_date']].head()

# ## 2.1 Prior Syndicated Loan Access
# Indicator = 1 if the borrower has any earlier DealScan loan before the current deal_date.

# Prior syndicated loan access (binary)
# Compute at borrower-deal level to avoid double-counting across tranches/lenders
mask = df['borrower_id'].notna() & df['deal_date'].notna()
deal_hist = df.loc[mask, ['borrower_id', 'deal_date']].drop_duplicates()
deal_hist = deal_hist.sort_values(['borrower_id', 'deal_date'])
first_date = deal_hist.groupby('borrower_id')['deal_date'].transform('min')
deal_hist['prior_synd_loan_access'] = (deal_hist['deal_date'] > first_date).astype(int)

df = df.drop(columns=['prior_synd_loan_access'], errors='ignore')
df = df.merge(deal_hist, on=['borrower_id', 'deal_date'], how='left')

print("Prior syndicated loan access (share):", df['prior_synd_loan_access'].mean())


# ## 3. Sample selection
# Keep US borrowers and restrict to 2001-2025.

# US borrowers only
before = len(df)
df['country'] = df['country'].astype(str).str.strip()
df = df[df['country'] == 'United States']
print(f"US borrowers only: {len(df):,} (dropped {before - len(df):,})")

# Time period 2001-2025 (use unified deal_date)
before = len(df)
df = df[df['deal_date'].between('2001-01-01', '2025-12-31')]
print(f"Time period 2001-2025: {len(df):,} (dropped {before - len(df):,})")

df['year'] = df['deal_date'].dt.year


# ## 4. Industry filtering
# _Skipped for now; apply a unified industry filter after merging._

df['sic_code'].isna().sum()

sic_raw = df['sic_code'].astype('string').str.strip()
parts = sic_raw.str.split(':', n=1, expand=True)

df['sic_code_num'] = pd.to_numeric(parts[0].str.extract(r'(\d{2,4})')[0], errors='coerce')
df['sic_desc'] = parts[1].str.strip()
df['sic_desc'] = df['sic_desc'].fillna(sic_raw)

print("Missing sic_code:", df['sic_code_num'].isna().sum())

# Industry filtering is deferred to the final estimation sample.


df.head()

# ## 5. Spread cleaning
# Filter all-in-spread-drawn values and winsorize by year (not for now).

import matplotlib.pyplot as plt

series = df['all_in_spread_drawn_bps'].dropna()

plt.figure(figsize=(6, 4))
plt.hist(series, bins=50, color='#4C78A8', alpha=0.8)
plt.title('All-in Spread Drawn (bps)')
plt.grid(alpha=0.2)
plt.tight_layout()
plt.show()

# Spread filter
# before = len(df)
# df = df[(df['all_in_spread_drawn_bps'] >= 0) & (df['all_in_spread_drawn_bps'] <= 1500)]
# print(f"After spread filter: {len(df):,} (dropped {before - len(df):,})")

# Winsorize by year
# df['year'] = df['deal_date'].dt.year
# df['spread_w'] = df.groupby('year')['all_in_spread_drawn_bps'].transform(
     # lambda x: x.clip(lower=x.quantile(0.01), upper=x.quantile(0.99))
# )

# df[['all_in_spread_drawn_bps', 'spread_w']].describe()

# import matplotlib.pyplot as plt

# fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# axes[0].hist(df['all_in_spread_drawn_bps'].dropna(), bins=50, color='#4C78A8', alpha=0.8)
# axes[0].set_title('All-in Spread (raw)')
# axes[0].grid(alpha=0.2)

# axes[1].hist(df['spread_w'].dropna(), bins=30, color='#4C78A8', alpha=0.8)
# axes[1].set_title('All-in Spread (winsorized)')
# axes[1].grid(alpha=0.2)

# plt.tight_layout()
# plt.show()


# ## 6. Maturity cleaning
# Restrict maturity to (0, 180] months and compute maturity in years.

df['tenor_maturity'].describe()

series = df['tenor_maturity'].dropna()

plt.figure(figsize=(6, 4))
plt.hist(series, bins=50, color='#4C78A8', alpha=0.8)
plt.title('Tenor Maturity (months)')
plt.grid(alpha=0.2)
plt.tight_layout()
plt.show()

before = len(df)
df = df[(df['tenor_maturity'] > 0) & (df['tenor_maturity'] <= 180)]
print(f"After maturity filter: {len(df):,} (dropped {before - len(df):,})")

df['maturity_years'] = df['tenor_maturity'] / 12

df[['tenor_maturity', 'maturity_years']].describe()

series = df['maturity_years'].dropna()

plt.figure(figsize=(6, 4))
plt.hist(series, bins=50, color='#4C78A8', alpha=0.8)
plt.title('Tenor Maturity (years)')
plt.grid(alpha=0.2)
plt.tight_layout()
plt.show()

# ## 7. Amount cleaning
# Keep positive tranche amounts and log-transform loan size.

import matplotlib.pyplot as plt
import numpy as np

var = 'loan_amount_mil' if 'loan_amount_mil' in df.columns else 'tranche_amount_converted'
series = df[var].dropna()

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].hist(series, bins=50, color='#4C78A8', alpha=0.8)
axes[0].set_title(f'{var} (raw)')
axes[0].grid(alpha=0.2)

axes[1].hist(np.log10(series + 1), bins=50, color='#4C78A8', alpha=0.8)
axes[1].set_title(f'log10(1+{var})')
axes[1].grid(alpha=0.2)

plt.tight_layout()
plt.show()

df['tranche_amount_converted'].describe()

df.iloc[10]



valid_commit = df["lender_commit"].notna().sum()
total = len(df)
valid_commit, total, valid_commit / total

(df["lender_share"] != "N/A").sum()/total

# Parse lender_share to numeric (percent -> fraction)
ls = df['lender_share']
if ls.dtype == object:
    ls_clean = ls.astype(str).str.replace('%', '', regex=False).str.strip()
    ls_num = pd.to_numeric(ls_clean, errors='coerce')
else:
    ls_num = pd.to_numeric(ls, errors='coerce')

df['lender_share_num'] = ls_num / 100.0

# Parse lender_commit: format is "CUR amount" (e.g. "USD 27") — extract USD rows only
lc_raw = df['lender_commit'].astype(str).str.strip()
lc_split = lc_raw.str.split(' ', n=1, expand=True)
lc_currency = lc_split[0]
lc_amount   = pd.to_numeric(lc_split[1], errors='coerce')

df['lender_commit_usd_mil'] = np.where(lc_currency == 'USD', lc_amount, np.nan)

# Derived lender amount: lender_commit_usd_mil (primary) → lender_share × tranche_amount (fallback)
df['lender_share_mil']  = df['lender_share_num'] * df['tranche_amount_converted']
df['lender_commit_mil'] = df['lender_commit_usd_mil'].where(df['lender_commit_usd_mil'].notna())
df['lender_commit_mil'] = df['lender_commit_mil'].fillna(df['lender_share_mil'])

# Log with guard against log(0)
df['log_lender_commit'] = np.log(df['lender_commit_mil'].where(df['lender_commit_mil'] > 0))
df['log_lender_commit'] = df['log_lender_commit'].replace([np.inf, -np.inf], np.nan)

# Coverage report
n = len(df)
n_usd   = df['lender_commit_usd_mil'].notna().sum()
n_share = df['lender_share_mil'].notna().sum()
n_any   = df['lender_commit_mil'].notna().sum()
print(f"lender_commit_usd_mil (primary):  {n_usd:>9,} / {n:,} = {100*n_usd/n:.1f}%")
print(f"lender_share_mil      (fallback): {n_share:>9,} / {n:,} = {100*n_share/n:.1f}%")
print(f"lender_commit_mil     (combined): {n_any:>9,} / {n:,} = {100*n_any/n:.1f}%")
print(f"log_lender_commit     (positive): {df['log_lender_commit'].notna().sum():>9,} / {n:,}")
df[['lender_commit_usd_mil', 'lender_share_mil', 'lender_commit_mil', 'log_lender_commit']].describe()

import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(df['lender_commit_mil'].dropna(), bins=50, color='seagreen', edgecolor='white')
axes[0].set_title('Lender Commitment (mil USD)')
axes[0].set_xlabel('Commitment (USD, millions)')
axes[0].set_ylabel('Count')

axes[1].hist(df['log_lender_commit'].dropna(), bins=50, color='slateblue', edgecolor='white')
axes[1].set_title('Log Lender Commitment')
axes[1].set_xlabel('log(Commitment)')
axes[1].set_ylabel('Count')

plt.tight_layout()
plt.show()

# import matplotlib.pyplot as plt

# fig, axes = plt.subplots(1, 2, figsize=(12, 4))
# axes[0].hist(df['loan_amount_mil'].dropna(), bins=50, color='steelblue', edgecolor='white')
# axes[0].set_title('Loan Amount (mil)')
# axes[0].set_xlabel('Loan Amount (USD, millions)')
# axes[0].set_ylabel('Count')

# axes[1].hist(df['log_loan_amount_w'].dropna(), bins=50, color='darkorange', edgecolor='white')
# axes[1].set_title('Log Loan Amount (winsorized)')
# axes[1].set_xlabel('log(Loan Amount)')
# axes[1].set_ylabel('Count')

# plt.tight_layout()
# plt.show()


# ## 8. Derived variables
# Create secured indicator (whether collateralized) and lead retention proxy.

# Secured indicator
secured_raw = df['secured'].astype(str).str.upper().str.strip()
df['secured_flag'] = secured_raw.isin(['YES', 'Y', '1', 'TRUE', 'SECURED']).astype(int)

# Derived amount variables (no winsorization, no extra filtering)
df['loan_amount_mil'] = df['tranche_amount_converted']
df['log_loan_amount'] = np.log(df['loan_amount_mil'].where(df['loan_amount_mil'] > 0))

print("Secured share:", df['secured_flag'].mean())


# ## 9. Final dataset preview
# Select key variables for downstream analysis.

raw_selected_cols = [
    'all_in_spread_drawn_bps', 'base_reference_rate', 'borrower_id',
    'borrower_name', 'collateral_security_type', 'country', 'covenants',
    'deal_active_date', 'deal_amount_converted', 'deal_purpose',
    'lead_arranger', 'lender_id', 'lender_name', 'lender_parent_id',
    'lender_share', 'lpc_deal_id', 'lpc_tranche_id', 'mandated_date',
    'lender_commit',
    'margin_bps', 'number_of_lead_arrangers', 'number_of_lenders',
    'perm_id', 'primary_purpose', 'primary_role', 'secondary_purpose',
    'secured', 'sic_code', 'tenor_maturity', 'ticker',
    'tranche_active_date', 'tranche_amount_converted', 'tranche_type'
]
raw_selected_cols = [c for c in raw_selected_cols if c in df.columns]

# Use non-winsorized variables
derived_cols = [
    'deal_date', 'year', 'maturity_years', 'loan_amount_mil',
    'log_loan_amount', 'secured_flag', 'prior_synd_loan_access',
    'sic_code_num',
    # Lender-specific amount (primary: lender_commit USD; fallback: lender_share × tranche_amount)
    'lender_share_num', 'lender_commit_usd_mil', 'lender_share_mil',
    'lender_commit_mil', 'log_lender_commit',
]

final_cols = raw_selected_cols + derived_cols
final_cols = [c for c in final_cols if c in df.columns]

# De-duplicate while preserving order
seen = set()
final_cols = [c for c in final_cols if not (c in seen or seen.add(c))]

df_clean = df[final_cols].copy()
print("Final shape:", df_clean.shape)

df_clean.head()

# Summary statistics and distribution plots for key variables (non-winsorized)
key_vars = [
    'all_in_spread_drawn_bps', 'tenor_maturity', 'maturity_years',
    'loan_amount_mil', 'log_loan_amount', 'number_of_lenders',
    'secured_flag', 'lender_commit_mil', 'log_lender_commit',
]
key_vars = [c for c in key_vars if c in df_clean.columns]

print("\nSummary statistics (key variables):")
print(df_clean[key_vars].describe())

import matplotlib.pyplot as plt

n = len(key_vars)
cols = 3
rows = (n + cols - 1) // cols

plt.figure(figsize=(cols * 5, rows * 3.5))
for i, var in enumerate(key_vars, 1):
    ax = plt.subplot(rows, cols, i)
    series = df_clean[var].dropna()
    ax.hist(series, bins=50, color='#4C78A8', alpha=0.8)
    ax.set_title(var)
    ax.grid(alpha=0.2)

plt.tight_layout()
plt.show()

# ## 10. Save output (commented)
# Review the outputs first. Uncomment the save line when you are ready.

output_file
df_clean.to_parquet(output_file, index=False)
