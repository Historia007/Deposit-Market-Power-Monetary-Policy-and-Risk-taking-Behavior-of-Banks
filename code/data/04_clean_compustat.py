import os
# # Compustat Cleaning
#
# **Purpose**: Clean and prepare Compustat data for bank lending research.
#
# **Data Sources**:
# - `funda`: Annual fundamentals (balance sheet, income statement)
# - `fundq`: Quarterly fundamentals (better temporal alignment with loan origination)
# - `company`: Company descriptive info (GICS codes, industry classifications)
# - `names`: Historical identifier tracking
# - `security`: Security identifiers for linking
#
# **Key Variables Constructed**:
# - Standard firm characteristics (size, leverage, profitability, tangibility)
# - **Distance-to-Default (Merton model)**: Key ex-ante risk measure for loan pricing research (currently disabled)
# - Market-to-book, interest coverage, R&D intensity
#
# **Notes**:
# - Quarterly data (`fundq`) is preferred for loan-level analysis
# - Annual data (`funda`) available for robustness checks
# - Save steps are commented out until you review the outputs

# ## 1. Setup and Load Data

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import norm
from scipy.optimize import brentq
import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', 120)
pd.set_option('display.max_rows', 100)

base_path = Path(os.environ.get("MPHIL_THESIS_ROOT", Path(__file__).resolve().parents[2]))

# Raw data paths
raw_path = base_path / 'data/raw/compustat'
funda_file = raw_path / 'funda__2001_2025.parquet'
fundq_file = raw_path / 'fundq__2001_2025.parquet'
company_file = raw_path / 'company__2001_2025.parquet'
names_file = raw_path / 'names__2001_2025.parquet'
security_file = raw_path / 'security__2001_2025.parquet'

# Output paths
output_path = base_path / 'data/intermediate/clean/compustat'
output_path.mkdir(parents=True, exist_ok=True)

output_annual = output_path / 'compustat_annual_cleaned.parquet'
output_quarterly = output_path / 'compustat_quarterly_cleaned.parquet'
output_company = output_path / 'compustat_company.parquet'

print("Paths configured")

# ## 2. Load Company Data (Industry Classifications)
#
# The `company` file contains GICS codes and better industry classifications than what's in fundamentals.

# Load company descriptive data
company_cols = [
    'gvkey', 'conm', 'cik', 'sic', 'naics',
    'gsector', 'ggroup', 'gind', 'gsubind',  # GICS codes
    'fic', 'state', 'incorp', 'loc',
    'ipodate', 'dldte', 'costat'
]

df_company = pd.read_parquet(company_file)
df_company = df_company[[c for c in company_cols if c in df_company.columns]]

print(f"Company data shape: {df_company.shape}")
print(f"\nGICS sector distribution:")
if 'gsector' in df_company.columns:
    print(df_company['gsector'].value_counts())

df_company.head()

# ## 3. Load Names Data (Historical Identifiers)
#
# The `names` file tracks historical company identifiers - important for linking to DealScan borrowers.

# Load names data for historical identifier tracking
df_names = pd.read_parquet(names_file)

print(f"Names data shape: {df_names.shape}")
print(f"\nColumns: {df_names.columns.tolist()}")
print(f"\nYear range coverage: {df_names['year1'].min()} to {df_names['year2'].max()}")

df_names.head()

# ## 4. Load and Clean Quarterly Fundamentals (Primary Dataset)
#
# Quarterly data is preferred for syndicated loan research because:
# 1. Better temporal alignment with loan origination dates
# 2. More timely market values for distance-to-default calculation
# 3. Can capture within-year variation in borrower characteristics

# Selected quarterly columns
# See WRDS Compustat documentation for variable definitions
fundq_cols = [
    # Identifiers
    'gvkey', 'datadate', 'fyearq', 'fqtr', 'fyr', 'tic', 'cusip', 'conm', 'cik',
    'indfmt', 'consol', 'popsrc', 'datafmt', 'fic', 'costat',

    # Balance sheet items
    'atq',      # Total assets
    'actq',     # Current assets
    'ltq',      # Total liabilities
    'lctq',     # Current liabilities
    'dlcq',     # Debt in current liabilities (short-term debt)
    'dlttq',    # Long-term debt
    'ceqq',     # Common equity
    'seqq',     # Stockholders' equity
    'ppentq',   # Property, plant & equipment (net)
    'cheq',     # Cash and short-term investments
    'rectq',    # Receivables
    'invtq',    # Inventories
    'req',      # Retained earnings
    'wcapq',    # Working capital
    'intanq',   # Intangible assets
    'gdwlq',    # Goodwill

    # Income statement items (quarterly)
    'saleq',    # Sales/revenue
    'cogsq',    # Cost of goods sold
    'xsgaq',    # SG&A expense
    'xintq',    # Interest expense
    'xrdq',     # R&D expense
    'dpq',      # Depreciation
    'niq',      # Net income
    'ibq',      # Income before extraordinary items
    'oibdpq',   # Operating income before depreciation (EBITDA)
    'oiadpq',   # Operating income after depreciation
    'piq',      # Pretax income

    # Market data
    'prccq',    # Price close - quarter
    'prchq',    # Price high - quarter
    'prclq',    # Price low - quarter
    'cshoq',    # Common shares outstanding
    'mkvaltq',  # Market value of equity

    # Cash flow items (year-to-date)
    'oancfy',   # Operating cash flow (YTD)
    'capxy',    # Capital expenditures (YTD)
    'dvy',      # Dividends (YTD)
]

# Load quarterly data
df_q = pd.read_parquet(fundq_file)
available_cols = [c for c in fundq_cols if c in df_q.columns]
df_q = df_q[available_cols]

# Ensure required columns exist for calculations
calc_cols = [
    'atq', 'dlcq', 'dlttq', 'ceqq', 'seqq', 'req', 'ibq', 'oiadpq', 'oibdpq', 'dpq', 'saleq',
    'xintq', 'xrdq', 'cheq', 'lctq', 'actq', 'ppentq', 'prccq', 'cshoq',
    'mkvaltq'
]
for col in calc_cols:
    if col not in df_q.columns:
        df_q[col] = np.nan


print(f"Quarterly data initial shape: {df_q.shape}")
print(f"Columns loaded: {len(available_cols)}")
print(f"Missing columns: {set(fundq_cols) - set(available_cols)}")

# Parse dates and apply standard Compustat filters
df_q['datadate'] = pd.to_datetime(df_q['datadate'], errors='coerce')
df_q = df_q[df_q['datadate'].notna()]
df_q['year'] = df_q['datadate'].dt.year
df_q['quarter'] = df_q['fqtr'].astype('Int64')
df_q['year_qtr'] = df_q['year'].astype(str) + 'Q' + df_q['quarter'].astype(str)

print(f"Date range: {df_q['datadate'].min()} to {df_q['datadate'].max()}")

# Standard Compustat filters
filters = [
    ('indfmt', ['INDL']),      # Industrial format (no SIC-based industry exclusion applied in cleaning)
    ('consol', ['C']),         # Consolidated statements
    ('popsrc', ['D']),         # Domestic population (North American Compustat)
    ('datafmt', ['STD']),      # Standard format (excludes restated)
]

for col, allowed in filters:
    if col in df_q.columns:
        before = len(df_q)
        df_q = df_q[df_q[col].isin(allowed)]
        print(f"{col} in {allowed}: {len(df_q):,} (dropped {before - len(df_q):,})")

# FIX (2026-02-24): Removed fic=='USA' filter.
# popsrc=='D' already restricts to the North American Compustat universe.
# The fic=='USA' filter was dropping 1,201 gvkeys (141k panel rows) for
# non-US-incorporated firms that are DealScan borrowers (e.g. Canadian
# firms, Bermuda/Cayman-incorporated companies). These firms have valid
# quarterly data in Compustat and are linked via the Roberts/Chava crosswalk.
if 'fic' in df_q.columns:
    non_usa = (df_q['fic'] != 'USA').sum()
    print(f"Non-USA fic firms retained: {non_usa:,} (previously dropped)")
    print(f"fic distribution:")
    print(df_q['fic'].value_counts().head(10))

# ## 5. Merge with Company Data for Industry Classifications

df_company.head()

df_q.head()

# Merge company info
company_merge_cols = ['gvkey', 'sic', 'naics', 'gsector', 'ggroup', 'gind', 'gsubind', 'state']
company_merge_cols = [c for c in company_merge_cols if c in df_company.columns]

df_q = df_q.merge(
    df_company[company_merge_cols],
    on='gvkey',
    how='left',
    suffixes=('', '_company')
)

print(f"After merge with company data: {df_q.shape}")
print(f"\nSIC coverage:")
if 'sic' in df_q.columns:
    print(df_q['sic'].value_counts(dropna=False))

missing = df_q['sic'].isna().sum()
total = len(df_q)
missing, total, missing/total

df_q['gsector'].isna().sum(), df_q['gsector'].notna().mean()

missing = df_q['gsector'].isna().sum()
total = len(df_q)
missing, total, missing/total

import pandas as pd
g = pd.to_numeric(df_q['gsector'], errors='coerce')
(g.eq(40).sum(), g.eq(40).mean())

sic = pd.to_numeric(df_q['sic'], errors='coerce')
(sic.between(6000, 6999).sum(), sic.between(6000, 6999).mean())

# ## 6. Industry Filtering
#
# _Skipped for now; apply a unified industry filter after merging._

# Industry filtering is deferred to the final estimation sample.
if 'sic' in df_q.columns:
    df_q['sic_num'] = pd.to_numeric(df_q['sic'], errors='coerce')

print(f"After industry filter: {len(df_q):,}")

pd.to_numeric(df_q['sic'], errors='coerce').between(6000, 6999).sum(), pd.to_numeric(df_q['sic'], errors='coerce').between(4900, 4999).sum()

# ## 7. Asset and Time Filters

df_q['atq'].replace([np.inf, -np.inf], np.nan).dropna().describe()
(df_q['atq'] <= 1).mean(), df_q['atq'].isna().mean()

# NOTE: Do not drop small firms by asset size
# (No asset-size filter applied here)

# Time filter: 2001-2025
before = len(df_q)
df_q = df_q[(df_q['year'] >= 2001) & (df_q['year'] <= 2025)]
print(f"After time filter (2001-2025): {len(df_q):,} (dropped {before - len(df_q):,})")

# Ensure unique firm-quarter key
dedup_before = len(df_q)
sort_cols = ['gvkey', 'datadate'] + ([c for c in ['fyearq', 'fqtr'] if c in df_q.columns])
df_q = df_q.sort_values(sort_cols).drop_duplicates(['gvkey', 'datadate'], keep='last')
print(f"After deduplicating gvkey-datadate: {len(df_q):,} (dropped {dedup_before - len(df_q):,})")

print(f"\nFinal quarterly sample:")
print(f"  Unique firms (gvkey): {df_q['gvkey'].nunique():,}")
print(f"  Total firm-quarters: {len(df_q):,}")
print(f"  Years covered: {df_q['year'].min()} - {df_q['year'].max()}")

# ## 8. Construct Core Financial Variables

# === Size ===
df_q['log_assets'] = np.log(df_q['atq'])

# === Leverage ===
# FIX (2026-02-13): Changed min_count from 2 to 1.
# Previously required BOTH dlcq and dlttq to be non-missing, which reduced
# leverage coverage unnecessarily. Standard practice treats missing component
# as zero (Dell'Ariccia et al. 2017, Paligorova & Santos 2017).
df_q['total_debt'] = df_q[['dlcq', 'dlttq']].sum(axis=1, min_count=1)
df_q['leverage'] = df_q['total_debt'] / df_q['atq']

# === Profitability ===
df_q['roa'] = df_q['ibq'] / df_q['atq']  # Income before extraordinary / Assets
df_q['roe'] = np.where(df_q['ceqq'] > 0, df_q['ibq'] / df_q['ceqq'], np.nan)

# EBITDA margin
df_q['ebitda_margin'] = np.where(
    df_q['saleq'] > 0,
    df_q['oibdpq'] / df_q['saleq'],
    np.nan
)

# === Market-to-Book ===
# FIX (2026-02-13): Added fallback to prccq * cshoq when mkvaltq is missing.
# Previously only used mkvaltq which has high missingness in Compustat.
# This substantially improves market_to_book coverage.
if 'mkvaltq' in df_q.columns:
    df_q['market_cap'] = df_q['mkvaltq'].fillna(df_q['prccq'] * df_q['cshoq'])
else:
    df_q['market_cap'] = df_q['prccq'] * df_q['cshoq']

df_q['market_to_book'] = np.where(
    (df_q['ceqq'] > 0) & (df_q['market_cap'] > 0),
    df_q['market_cap'] / df_q['ceqq'],
    np.nan
)

# === Tangibility ===
df_q['tangibility'] = np.where(
    df_q['atq'] > 0,
    df_q['ppentq'] / df_q['atq'],
    np.nan
)

# === Interest Coverage ===
df_q['interest_coverage'] = np.where(
    (df_q['xintq'] > 0) & (df_q['oibdpq'].notna()),
    np.log(1 + df_q['oibdpq'] / df_q['xintq']),
    np.nan
)

# === R&D Intensity ===
df_q['rd_to_assets'] = np.where(
    (df_q['atq'] > 0) & (df_q['xrdq'].notna()),
    df_q['xrdq'] / df_q['atq'],
    np.nan
)

df_q['rd_to_sales'] = np.where(
    (df_q['saleq'] > 0) & (df_q['xrdq'].notna()),
    df_q['xrdq'] / df_q['saleq'],
    np.nan
)

# === Cash Holdings ===
df_q['cash_ratio'] = np.where(
    (df_q['atq'] > 0) & (df_q['cheq'].notna()),
    df_q['cheq'] / df_q['atq'],
    np.nan
)

# === Current Ratio ===
df_q['current_ratio'] = np.where(
    df_q['lctq'] > 0,
    df_q['actq'] / df_q['lctq'],
    np.nan
)

# Preview
constructed_vars = [
    'log_assets', 'leverage', 'roa', 'roe', 'ebitda_margin',
    'market_to_book', 'tangibility', 'interest_coverage',
    'rd_to_assets', 'cash_ratio', 'current_ratio'
]
df_q[constructed_vars].apply(lambda s: s.replace([np.inf, -np.inf], np.nan).dropna().describe()).T


# ## 9. Altman Z-score (Accounting-Based Risk Measure)
#
# Z-score is an ex-ante default risk proxy based on balance-sheet and income statement items (Altman, 1968). It can be computed at the quarterly frequency because all inputs are available in fundq.
#
# Original Altman (1968) formula:
#
# $$Z = \frac{1.2\,WC + 1.4\,RE + 3.3\,EBIT + 0.999\,Sales}{TA}$$
#
# Where:
# - $WC$ = Working capital = current assets - current liabilities
# - $RE$ = Retained earnings
# - $EBIT$ = Earnings before interest and taxes (computed as OIBDP - DP in Compustat)
# - $TA$ = Total assets
#
# Following Paligorova & Santos (2017), we also compute a no-sales version because Sales is used separately as a control variable in the loan spread regressions.

# === Altman Z-score (quarterly) ===
df_q['working_capital'] = df_q['actq'] - df_q['lctq']
df_q['retained_earnings_q'] = df_q['req']

# EBIT proxy: use operating income before depreciation minus depreciation
df_q['ebit_q'] = df_q['oibdpq'] - df_q['dpq']

denom = df_q['atq']

# Full Altman Z-score (includes Sales)
df_q['zscore_full'] = np.where(
    denom > 0,
    (1.2 * df_q['working_capital'] + 1.4 * df_q['retained_earnings_q'] + 3.3 * df_q['ebit_q'] + 0.999 * df_q['saleq']) / denom,
    np.nan
)

# No-sales version (Paligorova & Santos, 2017)
df_q['zscore_nosales'] = np.where(
    denom > 0,
    (1.2 * df_q['working_capital'] + 1.4 * df_q['retained_earnings_q'] + 3.3 * df_q['ebit_q']) / denom,
    np.nan
)

df_q[['zscore_full', 'zscore_nosales']].apply(lambda s: s.replace([np.inf, -np.inf], np.nan).dropna().describe()).T

# Histogram of Z-score distributions (exclude NaN/inf)
import matplotlib.pyplot as plt

z_cols = ['zscore_full', 'zscore_nosales']

# Adjust these quantile ranges to control x-axis limits
trim_q = (0.05, 0.95)

fig, axes = plt.subplots(2, 1, figsize=(8, 8))
if not hasattr(axes, '__len__'):
    axes = [axes]

for i, col in enumerate(z_cols):
    s = df_q[col].replace([np.inf, -np.inf], np.nan).dropna()

    # Trimmed view for readability (trim_q)
    lo, hi = s.quantile(list(trim_q))
    s_trim = s[(s >= lo) & (s <= hi)]
    ax = axes[i]
    ax.hist(s_trim, bins=100, color='#F58518', alpha=0.8, edgecolor='white')
    ax.set_xlim(lo, hi)
    ax.set_title(f"{col} ({int(trim_q[0]*100)}-{int(trim_q[1]*100)}% view)")
    ax.grid(alpha=0.2)

plt.tight_layout()
plt.show()


# ## 10. Distance-to-Default (Merton Model)
#
# _Temporarily disabled in this clean script._
#
# **Critical for your research**: Distance-to-default is a key ex-ante risk measure mentioned in your proposal.
#
# The Merton (1974) model treats equity as a call option on the firm's assets:
#
# $$DD = \frac{\ln(V_A/D) + (\mu - 0.5\sigma_A^2)T}{\sigma_A \sqrt{T}}$$
#
# Where:
# - $V_A$ = Market value of assets
# - $D$ = Face value of debt (book value of total debt)
# - $\mu$ = Expected return on assets (often set to risk-free rate)
# - $\sigma_A$ = Asset volatility (derived from equity volatility)
# - $T$ = Time horizon (typically 1 year)
#
# **Implementation**: We use the iterative approach from Bharath & Shumway (2008) which is common in the literature.

def compute_equity_volatility(df, gvkey_col='gvkey', price_col='prccq', date_col='datadate', window=12):
    """
    Compute annualized equity volatility from quarterly stock returns.
    Uses trailing 12 quarters (3 years) of data.
    """
    df = df.sort_values([gvkey_col, date_col])

    # Compute quarterly returns
    df['price_lag'] = df.groupby(gvkey_col)[price_col].shift(1)
    df['ret_q'] = np.where(
        (df['price_lag'] > 0) & (df[price_col] > 0),
        np.log(df[price_col] / df['price_lag']),
        np.nan
    )

    # Rolling standard deviation of quarterly returns
    df['sigma_e_q'] = df.groupby(gvkey_col)['ret_q'].transform(
        lambda x: x.rolling(window=window, min_periods=4).std()
    )

    # Annualize: multiply by sqrt(4) for quarterly to annual
    df['sigma_e'] = df['sigma_e_q'] * np.sqrt(4)

    return df


def merton_dd_naive(E, D, sigma_e, r=0.02, T=1):
    """
    Naive (Bharath-Shumway style) distance-to-default calculation.

    Parameters:
    - E: Market value of equity
    - D: Face value of debt
    - sigma_e: Equity volatility (annualized)
    - r: Risk-free rate (default 2%)
    - T: Time horizon in years (default 1)

    Returns:
    - DD: Distance-to-default
    - PD_naive: Naive probability of default = N(-DD)
    """
    # Naive approximation: V_A ≈ E + D, sigma_A ≈ sigma_e * E / (E + D)
    V_A = E + D
    sigma_A = sigma_e * (E / V_A)

    # Distance-to-default
    if V_A <= 0 or D <= 0 or sigma_A <= 0:
        return np.nan, np.nan

    DD = (np.log(V_A / D) + (r - 0.5 * sigma_A**2) * T) / (sigma_A * np.sqrt(T))
    PD_naive = norm.cdf(-DD)

    return DD, PD_naive


# Apply equity volatility calculation (disabled for now)
# print("Computing equity volatility...")
# df_q = compute_equity_volatility(df_q)
# print(f"Equity volatility computed for {df_q['sigma_e'].notna().sum():,} observations")

# Merton distance-to-default / PD disabled for now
# (Previously computed distance_to_default and prob_default_naive here.)

# ## 11. Data Quality Filters
#
# Remove definitional impossibilities while preserving maximum observations for merging with DealScan.
#
# **Minimal filters applied:**
# - Assets ≥ $10M (removes micro-cap data errors)
# - Leverage between -1 and 10 (removes impossible ratios)
# - ROA between -200% and +200% (removes division-by-zero errors)
# - Market-to-book ≥ 0 (definitional requirement)
#
# These filters remove ~5-10% of clearly erroneous observations while preserving 90-95% of data for merging.

# ============================================================================
# DATA QUALITY FILTERS: Remove Definitional Impossibilities
# ============================================================================

print("\n" + "="*70)
print("APPLYING MINIMAL DATA QUALITY FILTERS")
print("="*70)

initial_obs = len(df_q)
print(f"Starting observations: {initial_obs:,}\n")

# Filter 1: Leverage between -1 and 10
before = len(df_q)
df_q = df_q[
    df_q['leverage'].isna() |
    ((df_q['leverage'] >= -1) & (df_q['leverage'] <= 10))
].copy()
dropped = before - len(df_q)
print(f"Filter 1 - Leverage ∈ [-1, 10]:")
print(f"  Dropped: {dropped:,} obs ({dropped/before*100:.2f}%)")
print(f"  Remaining: {len(df_q):,}\n")

# Filter 2: ROA between -200% and +200%
before = len(df_q)
df_q = df_q[
    df_q['roa'].isna() |
    ((df_q['roa'] >= -2) & (df_q['roa'] <= 2))
].copy()
dropped = before - len(df_q)
print(f"Filter 2 - ROA ∈ [-200%, +200%]:")
print(f"  Dropped: {dropped:,} obs ({dropped/before*100:.2f}%)")
print(f"  Remaining: {len(df_q):,}\n")

# Filter 3: Market-to-book ≥ 0 (if not missing)
before = len(df_q)
df_q = df_q[
    df_q['market_to_book'].isna() |
    (df_q['market_to_book'] >= 0)
].copy()
dropped = before - len(df_q)
print(f"Filter 3 - Market-to-book ≥ 0:")
print(f"  Dropped: {dropped:,} obs ({dropped/before*100:.2f}%)")
print(f"  Remaining: {len(df_q):,}\n")

# Summary
total_dropped = initial_obs - len(df_q)
print("="*70)
print(f"TOTAL DROPPED: {total_dropped:,} obs ({total_dropped/initial_obs*100:.2f}%)")
print(f"FINAL SAMPLE: {len(df_q):,} obs ({len(df_q)/initial_obs*100:.2f}% retained)")
print(f"Unique firms: {df_q['gvkey'].nunique():,}")
print("="*70)

# Show improved summary statistics
print("\nSummary statistics after quality filters:")
quality_vars = ['log_assets', 'leverage', 'roa', 'market_to_book']
print(df_q[quality_vars].apply(lambda s: s.replace([np.inf, -np.inf], np.nan).dropna().describe()).T)


# ## 11. Winsorize Variables
#
# Winsorize at 1% and 99% within each year-quarter to reduce the impact of outliers.

# vars_to_winsorize = [
    # 'log_assets', 'leverage', 'roa', 'roe', 'ebitda_margin',
    # 'market_to_book', 'tangibility', 'interest_coverage',
    # 'rd_to_assets', 'rd_to_sales', 'cash_ratio', 'current_ratio',
    # 'zscore_full', 'zscore_nosales',
    # 'distance_to_default', 'sigma_e'
# ]

# for var in vars_to_winsorize:
    # if var in df_q.columns:
        # df_q[f'{var}_w'] = df_q.groupby('year_qtr')[var].transform(
            # lambda x: x.clip(lower=x.quantile(0.01), upper=x.quantile(0.99))
        # )

# winsor_cols = [f'{v}_w' for v in vars_to_winsorize if f'{v}_w' in df_q.columns]
# print(f"Created {len(winsor_cols)} winsorized variables")
# df_q[winsor_cols].describe()

# ## 12. Final Dataset Preview

# Define final columns to keep
id_cols = ['gvkey', 'tic', 'cusip', 'cik', 'conm', 'datadate', 'year', 'quarter', 'year_qtr', 'fyearq', 'fqtr']

industry_cols = ['sic', 'naics', 'gsector', 'ggroup', 'gind', 'gsubind', 'state']

raw_financial_cols = [
    'atq', 'ltq', 'dlcq', 'dlttq', 'total_debt', 'ceqq', 'seqq',
    'ppentq', 'cheq', 'actq', 'lctq', 'wcapq',
    'saleq', 'ibq', 'niq', 'oibdpq', 'xintq', 'xrdq',
    'prccq', 'cshoq', 'market_cap'
]

derived_cols = [
    'log_assets', 'leverage', 'roa', 'roe', 'ebitda_margin',
    'market_to_book', 'tangibility', 'interest_coverage',
    'rd_to_assets', 'rd_to_sales', 'cash_ratio', 'current_ratio',
    'zscore_full', 'zscore_nosales',
    'sigma_e', 'distance_to_default', 'prob_default_naive'
]

# Note: winsorized columns not created since winsorization is disabled
winsorized_cols = []

# Combine all columns
all_cols = id_cols + industry_cols + raw_financial_cols + derived_cols + winsorized_cols
final_cols = [c for c in all_cols if c in df_q.columns]

# Remove duplicates while preserving order
seen = set()
final_cols = [c for c in final_cols if not (c in seen or seen.add(c))]

df_q_final = df_q[final_cols].copy()

print(f"Final quarterly dataset shape: {df_q_final.shape}")
print(f"\nColumns ({len(final_cols)}):")
for i, col in enumerate(final_cols):
    print(f"  {i+1}. {col}")

# Summary statistics for key variables
key_vars = ['log_assets', 'leverage', 'roa', 'market_to_book']
key_vars = [c for c in key_vars if c in df_q_final.columns]

print("Summary statistics (key variables):")
print(df_q_final[key_vars].apply(lambda s: s.replace([np.inf, -np.inf], np.nan).dropna().describe()).T)

# Visualize distributions with better outlier handling
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

plot_vars = key_vars + ['tangibility', 'cash_ratio'] if 'tangibility' in df_q_final.columns else key_vars

# Tailored quantile ranges for each variable type
var_settings = {
    'log_assets': {'q': (0.01, 0.99), 'bins': 60, 'color': '#4C78A8'},
    'leverage': {'q': (0.00, 0.95), 'bins': 50, 'color': '#F28E2B'},  # Tighter for leverage
    'roa': {'q': (0.05, 0.95), 'bins': 50, 'color': '#E15759'},  # Much tighter for ROA
    'market_to_book': {'q': (0.05, 0.95), 'bins': 50, 'color': '#76B7B2'},  # Tighter for M/B
    'tangibility': {'q': (0.01, 0.99), 'bins': 50, 'color': '#59A14F'},
    'cash_ratio': {'q': (0.01, 0.99), 'bins': 50, 'color': '#EDC948'}
}

for i, var in enumerate(plot_vars[:6]):
    ax = axes[i]
    series = df_q_final[var].replace([np.inf, -np.inf], np.nan).dropna()

    if not series.empty:
        # Get variable-specific settings or use defaults
        settings = var_settings.get(var, {'q': (0.05, 0.95), 'bins': 50, 'color': '#4C78A8'})
        q_low, q_high = settings['q']

        # Calculate quantiles and filter data for plotting
        lo, hi = series.quantile([q_low, q_high])
        plot_data = series[(series >= lo) & (series <= hi)]

        # Create histogram
        n, bins, patches = ax.hist(plot_data, bins=settings['bins'],
                                    color=settings['color'], alpha=0.7, edgecolor='white', linewidth=0.5)

        # Add median line
        median_val = series.median()
        if lo <= median_val <= hi:
            ax.axvline(median_val, color='darkred', linestyle='--', linewidth=2,
                      label=f'Median: {median_val:.2f}', alpha=0.8)

        # Set title with sample info
        n_obs = len(series)
        n_outliers = len(series) - len(plot_data)
        pct_outliers = (n_outliers / n_obs * 100) if n_obs > 0 else 0

        ax.set_title(f'{var}\n(showing {int(q_low*100)}-{int(q_high*100)}% range, '
                    f'{pct_outliers:.1f}% outliers hidden)',
                    fontsize=10, fontweight='bold')

        # Add stats box
        stats_text = f'N={n_obs:,}\nMean={series.mean():.2f}\nSD={series.std():.2f}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               fontsize=8, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))

        if lo <= median_val <= hi:
            ax.legend(fontsize=8, loc='upper right')

        ax.set_xlim(lo, hi)
    else:
        ax.set_title(var, fontsize=10)

    ax.set_xlabel('')
    ax.set_ylabel('Frequency', fontsize=9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# Hide unused subplots
for j in range(len(plot_vars), 6):
    axes[j].axis('off')

plt.suptitle('Distribution of Key Firm Characteristics (Quarterly) - Compustat',
            fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
plt.show()

# Additional: Show extreme outlier counts
print("\n" + "="*60)
print("OUTLIER ANALYSIS (observations outside reasonable ranges)")
print("="*60)

outlier_checks = {
    'log_assets': (-10, 20),
    'leverage': (-5, 10),
    'roa': (-1, 1),
    'market_to_book': (-10, 100)
}

for var in key_vars:
    if var in df_q_final.columns and var in outlier_checks:
        series = df_q_final[var].replace([np.inf, -np.inf], np.nan).dropna()
        lo, hi = outlier_checks[var]
        n_low = (series < lo).sum()
        n_high = (series > hi).sum()
        n_total = len(series)
        pct_outliers = ((n_low + n_high) / n_total * 100) if n_total > 0 else 0
        print(f"{var:20s}: {n_low:,} too low (< {lo}), {n_high:,} too high (> {hi}) "
              f"= {pct_outliers:.2f}% outliers")

# ## 12. Load and Clean Annual Data (Alternative)
#
# Annual data (`funda`) is useful for:
# - Robustness checks
# - Variables only available annually
# - Matching with annual bank-level data

# NOTE: Annual Compustat pipeline disabled for now.
# # Load annual fundamentals
# funda_cols = [
#     'gvkey', 'datadate', 'fyear', 'fyr', 'tic', 'cusip', 'conm', 'cik',
#     'indfmt', 'consol', 'popsrc', 'datafmt', 'fic', 'sich', 'costat',
#     'at', 'lt', 'dlc', 'dltt', 'ceq', 'seq', 'ppent', 'che', 'act', 'lct',
#     'sale', 'ni', 'ib', 'oibdp', 'xint', 'xrd', 'xad', 'xsga', 'dp',
#     'prcc_f', 'csho', 'mkvalt',
#     'oancf', 'capx', 'dv', 'wcap', 're',
#     'emp'
# ]

# df_a = pd.read_parquet(funda_file)
# available_cols_a = [c for c in funda_cols if c in df_a.columns]
# df_a = df_a[available_cols_a]

# # Ensure required columns exist for calculations
# calc_cols_a = [
#     'at', 'dlc', 'dltt', 'seq', 'ni', 'oibdp', 'xint', 'ppent', 'mkvalt',
#     'prcc_f', 'csho'
# ]
# for col in calc_cols_a:
#     if col not in df_a.columns:
#         df_a[col] = np.nan


# print(f"Annual data initial shape: {df_a.shape}")


# NOTE: Annual Compustat pipeline disabled for now.
# # Apply same filters as quarterly
# df_a['datadate'] = pd.to_datetime(df_a['datadate'], errors='coerce')
# df_a = df_a[df_a['datadate'].notna()]
# df_a['year'] = df_a['datadate'].dt.year

# # Standard filters
# for col, allowed in [('indfmt', ['INDL']), ('consol', ['C']), ('popsrc', ['D']), ('datafmt', ['STD'])]:
#     if col in df_a.columns:
#         df_a = df_a[df_a[col].isin(allowed)]

# # US firms
# if 'fic' in df_a.columns:
#     df_a = df_a[df_a['fic'] == 'USA']

# # Industry exclusion intentionally disabled for annual pipeline as well
# if 'sich' in df_a.columns:
#     df_a['sic_num'] = pd.to_numeric(df_a['sich'], errors='coerce')
#     sic_exclude = df_a['sic_num'].between(6000, 6999) | df_a['sic_num'].between(4900, 4999)
#     df_a = df_a[~sic_exclude]

# # Asset and time filters
# df_a = df_a[(df_a['at'] > 1) & (df_a['at'].notna())]
# df_a = df_a[(df_a['year'] >= 2001) & (df_a['year'] <= 2025)]

# print(f"Annual data after filters: {len(df_a):,}")
# print(f"Unique firms: {df_a['gvkey'].nunique():,}")


# NOTE: Annual Compustat pipeline disabled for now.
# # Construct annual variables (same as quarterly)
# df_a['log_assets'] = np.log(df_a['at'])
# df_a['total_debt'] = df_a[['dlc', 'dltt']].sum(axis=1, min_count=2)
# df_a['leverage'] = df_a['total_debt'] / df_a['at']
# df_a['roa'] = df_a['ni'] / df_a['at']

# # Market-to-book
# if 'mkvalt' in df_a.columns:
#     df_a['market_cap'] = df_a['mkvalt']
# else:
#     df_a['market_cap'] = df_a['prcc_f'] * df_a['csho']

# df_a['market_to_book'] = np.where(
#     (df_a['seq'] > 0) & (df_a['market_cap'] > 0),
#     df_a['market_cap'] / df_a['seq'],
#     np.nan
# )

# df_a['tangibility'] = np.where(
#     df_a['at'] > 0,
#     df_a['ppent'] / df_a['at'],
#     np.nan
# )
# df_a['interest_coverage'] = np.where(
#     (df_a['xint'] > 0) & (df_a['oibdp'].notna()),
#     np.log(1 + df_a['oibdp'] / df_a['xint']),
#     np.nan
# )

# # Note: Winsorization disabled - using raw variables
# print(f"Annual data shape: {df_a.shape}")
# df_a[['log_assets', 'leverage', 'roa', 'market_to_book', 'tangibility']].apply(lambda s: s.replace([np.inf, -np.inf], np.nan).dropna().describe()).T


# ## 13. Column Definitions
#
# ### Identifiers
# | Column | Description |
# |--------|-------------|
# | gvkey | Compustat permanent firm identifier (primary key for linking) |
# | tic | Ticker symbol |
# | cusip | CUSIP identifier (may change over time) |
# | cik | SEC CIK identifier |
# | conm | Company name |
#
# ### Industry Classifications
# | Column | Description |
# |--------|-------------|
# | sic | 4-digit SIC code |
# | naics | 6-digit NAICS code |
# | gsector | GICS sector (2-digit): 10=Energy, 15=Materials, 20=Industrials, 25=Consumer Discretionary, 30=Consumer Staples, 35=Health Care, 40=Financials, 45=IT, 50=Communication, 55=Utilities, 60=Real Estate |
# | gind | GICS industry (6-digit) |
# | gsubind | GICS sub-industry (8-digit) |
#
# ### Balance Sheet (Quarterly: suffix q)
# | Column | Description |
# |--------|-------------|
# | atq | Total assets |
# | ltq | Total liabilities |
# | dlcq | Debt in current liabilities (short-term debt) |
# | dlttq | Long-term debt |
# | ceqq | Common equity |
# | seqq | Stockholders' equity |
# | ppentq | Property, plant & equipment (net) |
# | cheq | Cash and short-term investments |
#
# ### Income Statement (Quarterly)
# | Column | Description |
# |--------|-------------|
# | saleq | Sales/revenue |
# | ibq | Income before extraordinary items |
# | niq | Net income |
# | oibdpq | Operating income before depreciation (EBITDA proxy) |
# | xintq | Interest expense |
# | xrdq | R&D expense |
#
# ### Market Data
# | Column | Description |
# |--------|-------------|
# | prccq | Stock price (close, end of quarter) |
# | cshoq | Common shares outstanding (millions) |
# | market_cap | Market value of equity = prccq × cshoq |
#
# ### Derived Variables
# | Column | Description |
# |--------|-------------|
# | log_assets | log(total assets) - firm size proxy |
# | leverage | (short-term debt + long-term debt) / total assets |
# | roa | Net income / total assets - profitability |
# | market_to_book | Market cap / book equity - growth opportunities |
# | tangibility | PP&E / total assets - asset tangibility |
# | interest_coverage | log(1 + EBITDA/interest expense) - debt service capacity |
# | sigma_e | Annualized equity volatility (from quarterly returns) - currently disabled |
# | **distance_to_default** | Merton (1974) DD - number of std devs from default (disabled) |
# | prob_default_naive | N(-DD) - naive default probability (disabled) |
#
# ### Suffix Conventions
# | Suffix | Meaning |
# |--------|-------------|
# | _w | Winsorized at 1%/99% by year-quarter |

# ## 14. Save Outputs (Commented)
#
# Review the outputs first. Uncomment the save lines when ready.

print("Output paths:")
print(f"  Quarterly: {output_quarterly}")
print(f"  Annual: {output_annual}")
print(f"  Company: {output_company}")

# Save outputs
# NOTE: Annual Compustat save disabled for now.
df_q_final.to_parquet(output_quarterly, index=False)
print(f"Saved quarterly data: {len(df_q_final):,} rows")

# df_a.to_parquet(output_annual, index=False)
# print(f"Saved annual data: {len(df_a):,} rows")

df_company.to_parquet(output_company, index=False)
print(f"Saved company data: {len(df_company):,} rows")


# ## 15. Linking to DealScan
#
# For your research, you'll need to link Compustat borrowers to DealScan loans. The standard approach uses the **Roberts Dealscan-Compustat Linking Database** (Chava & Roberts, 2008) available on WRDS.
#
# Key linking variables:
# - `gvkey` (Compustat) → `gvkey` in linking table
# - `facilityid` (DealScan) → `facilityid` in linking table
#
# The linking table coverage is 1987-2017. For loans after 2017, you may need to use fuzzy matching on company names or CIK codes.

# Check identifier availability for linking
print("Identifier availability for linking:")
print(f"  gvkey: {df_q_final['gvkey'].notna().sum():,} / {len(df_q_final):,}")
print(f"  cusip: {df_q_final['cusip'].notna().sum():,} / {len(df_q_final):,}")
print(f"  cik: {df_q_final['cik'].notna().sum():,} / {len(df_q_final):,}")
print(f"  tic: {df_q_final['tic'].notna().sum():,} / {len(df_q_final):,}")
