import os
# # DealScan–Compustat Linking Notebook
#
# This notebook links cleaned DealScan loans to cleaned Compustat firm financials.

# Linkage retains firms regardless of asset size; no additional size cutoff is applied.

# ## Literature-Backed Linking Approach (Summary)
#
# **Standard practice:** Use the WRDS **Roberts DealScan–Compustat Linking Database** (Chava & Roberts, 2008) as the primary mapping between DealScan loan IDs and Compustat GVKEY. This is the canonical link used in the literature and distributed via WRDS.
#
# **Coverage/extensions:** The WRDS link has been extended beyond the original sample period and includes match-quality fields (e.g., confidence scores). For years beyond the link’s coverage, researchers typically supplement with **ticker/CUSIP** matches and **name-based matching** (token/fuzzy matching with manual verification).
#
# We implement this hierarchy below and keep *all* firms regardless of asset size.

# ## 1. Setup & Paths
#
# Imports packages, sets display options, and defines input/output paths including the WRDS link table. Update `base_path` if you move the project.

import pandas as pd
import numpy as np
from pathlib import Path
import re
from difflib import SequenceMatcher
import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', 50)

base_path = Path(os.environ.get("MPHIL_THESIS_ROOT", Path(__file__).resolve().parents[2]))

# Cleaned inputs
dealscan_file = base_path / 'data/intermediate/clean/dealscan/dealscan_cleaned.parquet'
compustat_q_file = base_path / 'data/intermediate/clean/compustat/compustat_quarterly_cleaned.parquet'

# Link table (Roberts/Chava WRDS link)
link_table_path = base_path / 'data/intermediate/crosswalks/Dealscan-Compustat_Linking_Database012024.xlsx'

# Output paths
output_path = base_path / 'data/intermediate/merged'
output_path.mkdir(parents=True, exist_ok=True)
output_file = output_path / 'dealscan_compustat_linked.parquet'
output_panel_file = output_path / 'dealscan_compustat_loan_borrower_panel.parquet'
output_borrower_link_file = output_path / 'dealscan_borrower_gvkey_link.parquet'


def normalize_gvkey(x):
    """Normalize GVKEY to 6-digit string (Compustat convention)."""
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    if s.endswith('.0'):
        s = s[:-2]
    s = re.sub(r'[^0-9]', '', s)
    return s.zfill(6) if s else np.nan


print('Input files:')
print(f'  Dealscan: {dealscan_file.exists()}')
print(f'  Compustat: {compustat_q_file.exists()}')
print(f'Link table: {link_table_path.exists()} -> {link_table_path}')
print('\nOutput files:')
print(f'  Lender-level merged: {output_file}')
print(f'  Loan-borrower panel: {output_panel_file}')
print(f'  Borrower-gvkey map: {output_borrower_link_file}')



# ## 2. Load Cleaned Data
#
# Loads the cleaned DealScan and Compustat quarterly files, converts dates, and prints basic coverage plus identifier availability.

# Load cleaned datasets
df_ds = pd.read_parquet(dealscan_file)
df_q = pd.read_parquet(compustat_q_file)

print('=' * 70)
print('DATA LOADED')
print('=' * 70)
print(f'\nDealScan shape: {df_ds.shape}')
print(f'  Unique deals: {df_ds["lpc_deal_id"].nunique():,}')
print(f'  Unique borrowers: {df_ds["borrower_id"].nunique():,}')
print(f'  Date range: {df_ds["deal_date"].min()} to {df_ds["deal_date"].max()}')

print(f'\nCompustat quarterly shape: {df_q.shape}')
print(f'  Unique firms (gvkey): {df_q["gvkey"].nunique():,}')
print(f'  Date range: {df_q["datadate"].min()} to {df_q["datadate"].max()}')

# Convert dates
df_ds['deal_date'] = pd.to_datetime(df_ds['deal_date'])
df_q['datadate'] = pd.to_datetime(df_q['datadate'])

# Check identifier availability
print('\n' + '=' * 70)
print('IDENTIFIER AVAILABILITY FOR LINKING')
print('=' * 70)
print('\nDealScan identifiers:')
print(f'  ticker: {df_ds["ticker"].notna().sum():,} / {len(df_ds):,} ({df_ds["ticker"].notna().mean()*100:.1f}%)')
print(f'  borrower_name: {df_ds["borrower_name"].notna().sum():,} / {len(df_ds):,} ({df_ds["borrower_name"].notna().mean()*100:.1f}%)')

print('\nCompustat identifiers:')
print(f'  tic: {df_q["tic"].notna().sum():,} / {len(df_q):,} ({df_q["tic"].notna().mean()*100:.1f}%)')
print(f'  conm: {df_q["conm"].notna().sum():,} / {len(df_q):,} ({df_q["conm"].notna().mean()*100:.1f}%)')
print(f'  cusip: {df_q["cusip"].notna().sum():,} / {len(df_q):,} ({df_q["cusip"].notna().mean()*100:.1f}%)')

# ## 3. Load WRDS Link Table
#
# Loads the WRDS Roberts/Chava link table (`links` sheet) used as the primary mapping from DealScan IDs to Compustat GVKEY.

# Load WRDS DealScan–Compustat link table (Roberts/Chava)
link_df = None
if link_table_path.exists():
    link_df = pd.read_excel(link_table_path, sheet_name='links')
    print('Link table loaded:', link_df.shape)
    print('Link table columns:', link_df.columns.tolist())
else:
    print('Link table not found. Please confirm the path.')


# ## 4. Borrower and Firm Lookups
#
# Builds borrower-level identifiers from DealScan and firm-level identifiers from Compustat for matching.

# ============================================================================
# STEP 1: Create Borrower-Level Lookup from DealScan
# ============================================================================
# Aggregate to borrower level to get unique borrowers with their identifiers

borrower_df = df_ds.groupby('borrower_id').agg({
    'borrower_name': 'first',
    'ticker': lambda x: x.dropna().iloc[0] if x.notna().any() else None,
    'lpc_deal_id': 'nunique',
    'deal_date': ['min', 'max']
}).reset_index()

borrower_df.columns = ['borrower_id', 'borrower_name', 'ticker', 'num_deals', 'first_deal', 'last_deal']
borrower_df['borrower_id_num'] = pd.to_numeric(borrower_df['borrower_id'], errors='coerce')

print('=' * 70)
print('STEP 1: DEALSCAN BORROWER SUMMARY')
print('=' * 70)
print(f'\nUnique borrowers: {len(borrower_df):,}')
print(f'Borrowers with ticker: {borrower_df["ticker"].notna().sum():,} ({borrower_df["ticker"].notna().mean()*100:.1f}%)')
print(f'Borrowers without ticker: {borrower_df["ticker"].isna().sum():,} ({borrower_df["ticker"].isna().mean()*100:.1f}%)')

# ============================================================================
# STEP 2: Create Firm-Level Lookup from Compustat
# ============================================================================
# Get unique firms with their most recent identifiers

firm_df = df_q.sort_values('datadate').groupby('gvkey').agg({
    'tic': 'last',
    'conm': 'last',
    'cusip': 'last',
    'cik': 'last',
    'sic': 'first',
    'datadate': ['min', 'max']
}).reset_index()

firm_df.columns = ['gvkey', 'tic', 'conm', 'cusip', 'cik', 'sic', 'first_quarter', 'last_quarter']
firm_df['gvkey'] = firm_df['gvkey'].apply(normalize_gvkey)
firm_df = firm_df[firm_df['gvkey'].notna()].copy()

print('\n' + '=' * 70)
print('STEP 2: COMPUSTAT FIRM SUMMARY')
print('=' * 70)
print(f'\nUnique firms (gvkey): {len(firm_df):,}')
print(f'Firms with ticker: {firm_df["tic"].notna().sum():,} ({firm_df["tic"].notna().mean()*100:.1f}%)')

# Preview
print('\nSample Compustat firms:')
print(firm_df[['gvkey', 'tic', 'conm']].head(10))



# ## 5. Link Table Matching (Primary)
#
# Matches using WRDS `facilityid` and `borrowercompanyid`, selecting the best GVKEY per borrower by frequency and confidence score.

# ============================================================================
# STEP 3: LINK TABLE MATCHING (WRDS Roberts/Chava)
# ============================================================================
# Use the WRDS DealScan–Compustat linking table to map DealScan IDs -> GVKEY.
# The link file includes facility-level IDs and borrowercompanyid mappings.

print('=' * 70)
print('STEP 3: LINK TABLE MATCHING (WRDS)')
print('=' * 70)

link_facility_borrower = pd.DataFrame(columns=['borrower_id', 'borrower_name', 'gvkey', 'match_method', 'confidence_score'])
link_borrowerid = pd.DataFrame(columns=['borrower_id', 'borrower_name', 'gvkey', 'match_method', 'confidence_score'])

if link_df is not None:
    # Standardize IDs for matching
    df_ds['facilityid'] = pd.to_numeric(df_ds['lpc_tranche_id'], errors='coerce')

    link_df['facilityid'] = pd.to_numeric(link_df['facilityid'], errors='coerce')
    link_df['borrowercompanyid'] = pd.to_numeric(link_df['borrowercompanyid'], errors='coerce')
    link_df['gvkey'] = link_df['gvkey'].apply(normalize_gvkey)

    # Facility-level link: DealScan tranche/facility ID -> GVKEY
    facility_link = df_ds[['borrower_id', 'borrower_name', 'facilityid']].merge(
        link_df[['facilityid', 'gvkey', 'borrowercompanyid', 'confidence_score', 'score_company_match', 'score_ticker_match']],
        on='facilityid',
        how='left'
    )

    # Choose best gvkey per borrower based on frequency, then confidence
    facility_link = facility_link[facility_link['gvkey'].notna()].copy()
    if len(facility_link) > 0:
        agg = facility_link.groupby(['borrower_id', 'gvkey']).agg(
            n_facilities=('facilityid', 'nunique'),
            max_conf=('confidence_score', 'max'),
            borrower_name=('borrower_name', 'first')
        ).reset_index()
        agg = agg.sort_values(['borrower_id', 'n_facilities', 'max_conf'], ascending=[True, False, False])
        best = agg.drop_duplicates('borrower_id', keep='first')
        link_facility_borrower = best[['borrower_id', 'borrower_name', 'gvkey', 'max_conf']].copy()
        link_facility_borrower.rename(columns={'max_conf': 'confidence_score'}, inplace=True)
        link_facility_borrower['match_method'] = 'linktable_facilityid'

    # Borrowercompanyid link: DealScan borrower_id_num -> GVKEY
    borrower_link_df = borrower_df.merge(
        link_df[['borrowercompanyid', 'gvkey', 'confidence_score']].dropna(subset=['borrowercompanyid', 'gvkey']),
        left_on='borrower_id_num',
        right_on='borrowercompanyid',
        how='left'
    )

    borrower_link_df = borrower_link_df[borrower_link_df['gvkey'].notna()].copy()
    if len(borrower_link_df) > 0:
        agg_b = borrower_link_df.groupby(['borrower_id', 'gvkey']).agg(
            max_conf=('confidence_score', 'max'),
            borrower_name=('borrower_name', 'first')
        ).reset_index()
        agg_b = agg_b.sort_values(['borrower_id', 'max_conf'], ascending=[True, False])
        best_b = agg_b.drop_duplicates('borrower_id', keep='first')
        link_borrowerid = best_b[['borrower_id', 'borrower_name', 'gvkey', 'max_conf']].copy()
        link_borrowerid.rename(columns={'max_conf': 'confidence_score'}, inplace=True)
        link_borrowerid['match_method'] = 'linktable_borrowerid'

    print(f'Facility-level link matches: {len(link_facility_borrower):,} borrowers')
    print(f'Borrowercompanyid link matches: {len(link_borrowerid):,} borrowers')
else:
    print('Link table not available; skipping link-table matching.')



# ## 6. Ticker Matching (Fallback)
#
# Matches remaining borrowers by cleaned ticker symbols as a secondary fallback.

# ============================================================================
# STEP 4: TICKER-BASED MATCHING (Primary Method)
# ============================================================================
# Match DealScan borrowers to Compustat firms using ticker symbols

print('=' * 70)
print('STEP 4: TICKER-BASED MATCHING')
print('=' * 70)

# Standardize tickers for matching
def clean_ticker(ticker):
    """Clean ticker symbol for matching."""
    if pd.isna(ticker):
        return None
    ticker = str(ticker).upper().strip()
    # Remove common suffixes
    ticker = re.sub(r'\.\d+$', '', ticker)  # Remove .1, .2, etc.
    ticker = re.sub(r'[^A-Z0-9]', '', ticker)  # Keep only alphanumeric
    return ticker if ticker else None

borrower_df['ticker_clean'] = borrower_df['ticker'].apply(clean_ticker)
firm_df['tic_clean'] = firm_df['tic'].apply(clean_ticker)

# Match on cleaned ticker
ticker_match = borrower_df[borrower_df['ticker_clean'].notna()].merge(
    firm_df[['gvkey', 'tic_clean', 'conm']].rename(columns={'conm': 'compustat_name'}),
    left_on='ticker_clean',
    right_on='tic_clean',
    how='left'
)

# Count matches
n_with_ticker = borrower_df['ticker_clean'].notna().sum()
n_matched = ticker_match['gvkey'].notna().sum()

print(f'\nBorrowers with ticker: {n_with_ticker:,}')
print(f'Matched to Compustat via ticker: {n_matched:,} ({n_matched/n_with_ticker*100:.1f}%)')
print(f'Unmatched (ticker not in Compustat): {n_with_ticker - n_matched:,}')

# Create the borrower-to-gvkey mapping from ticker matches
ticker_link = ticker_match[ticker_match['gvkey'].notna()][
    ['borrower_id', 'borrower_name', 'gvkey', 'compustat_name']
].copy()
ticker_link['match_method'] = 'ticker'
# no confidence score from ticker match
ticker_link['confidence_score'] = np.nan

print(f'\n--- Sample Ticker Matches ---')
print(ticker_link.head(10))

# ## 7. Name Matching (Exact + Fuzzy Backup)
#
# Performs exact name matches first, then fuzzy matching for the remaining unmatched borrowers. Stores similarity as a confidence proxy.

# ============================================================================
# STEP 5: NAME-BASED MATCHING (Supplementary Method)
# ============================================================================
# For borrowers without link-table or ticker matches, try name matching

print('=' * 70)
print('STEP 5: NAME-BASED MATCHING (for unmatched borrowers)')
print('=' * 70)

def clean_company_name(name):
    """Standardize company name for matching."""
    if pd.isna(name):
        return ''
    name = str(name).upper().strip()
    # Remove common suffixes
    suffixes = [' INC', ' CORP', ' CO', ' LLC', ' LP', ' LTD', ' PLC', ' HOLDINGS',
                ' GROUP', ' INTL', ' INTERNATIONAL', ' COMPANIES', ' COMPANY',
                ' THE', ',', '.', '-', "'"]
    for suffix in suffixes:
        name = name.replace(suffix, ' ')
    # Clean up whitespace
    name = ' '.join(name.split())
    return name

# Get unmatched borrowers (those not matched via link table or ticker)
matched_borrower_ids = set(ticker_link['borrower_id']).union(
    set(link_facility_borrower['borrower_id'])
).union(
    set(link_borrowerid['borrower_id'])
)
unmatched_borrowers = borrower_df[~borrower_df['borrower_id'].isin(matched_borrower_ids)].copy()

print(f'Borrowers not matched via link table or ticker: {len(unmatched_borrowers):,}')

# Clean names
unmatched_borrowers['name_clean'] = unmatched_borrowers['borrower_name'].apply(clean_company_name)
firm_df['conm_clean'] = firm_df['conm'].apply(clean_company_name)

# Exact name match first
exact_name_match = unmatched_borrowers.merge(
    firm_df[['gvkey', 'conm_clean', 'conm']].rename(columns={'conm': 'compustat_name'}),
    left_on='name_clean',
    right_on='conm_clean',
    how='inner'
)

print(f'Exact name matches: {len(exact_name_match):,}')

# Create name-based link
name_link = exact_name_match[['borrower_id', 'borrower_name', 'gvkey', 'compustat_name']].copy()
name_link['match_method'] = 'name_exact'
name_link['confidence_score'] = 1.0

# ---------------------------------------------------------------------------
# Fuzzy match as back-up for remaining unmatched
# ---------------------------------------------------------------------------
# Use a simple blocking key to keep runtime reasonable
fuzzy_threshold = 0.92

# Remove already name-matched borrowers
already_matched = set(name_link['borrower_id'])
remaining = unmatched_borrowers[~unmatched_borrowers['borrower_id'].isin(already_matched)].copy()

# Create blocking key from first token
firm_df['name_key'] = firm_df['conm_clean'].str.split().str[0].str[:4]
remaining['name_key'] = remaining['name_clean'].str.split().str[0].str[:4]

# Build lookup dict
firm_groups = {k: v for k, v in firm_df.groupby('name_key')}

fuzzy_rows = []
for _, row in remaining.iterrows():
    key = row['name_key']
    if key not in firm_groups:
        continue
    candidates = firm_groups[key]
    best_gvkey = None
    best_name = None
    best_score = 0.0
    for _, crow in candidates.iterrows():
        score = SequenceMatcher(None, row['name_clean'], crow['conm_clean']).ratio()
        if score > best_score:
            best_score = score
            best_gvkey = crow['gvkey']
            best_name = crow['conm']
    if best_score >= fuzzy_threshold:
        fuzzy_rows.append({
            'borrower_id': row['borrower_id'],
            'borrower_name': row['borrower_name'],
            'gvkey': best_gvkey,
            'compustat_name': best_name,
            'match_method': 'name_fuzzy',
            'confidence_score': best_score
        })

fuzzy_link = pd.DataFrame(fuzzy_rows)
print(f'Fuzzy name matches (threshold {fuzzy_threshold}): {len(fuzzy_link):,}')


# ## 8. Combine Matches with Priority
#
# Combines all match sources with a clear priority order and keeps one GVKEY per borrower. Computes overall match rates.

# ============================================================================
# STEP 6: COMBINE ALL MATCHES
# ============================================================================

print('\n' + '=' * 70)
print('STEP 6: COMBINE ALL MATCHES')
print('=' * 70)

# Combine link-table, ticker, exact-name, and fuzzy-name matches
borrower_link = pd.concat([
    link_facility_borrower,
    link_borrowerid,
    ticker_link,
    name_link,
    fuzzy_link
], ignore_index=True)

borrower_link['gvkey'] = borrower_link['gvkey'].apply(normalize_gvkey)
borrower_link = borrower_link[borrower_link['gvkey'].notna()].copy()

# Priority order (lower = higher priority)
priority = {
    'linktable_facilityid': 0,
    'linktable_borrowerid': 1,
    'ticker': 2,
    'name_exact': 3,
    'name_fuzzy': 4
}
borrower_link['match_priority'] = borrower_link['match_method'].map(priority).fillna(9)
borrower_link['confidence_rank'] = pd.to_numeric(borrower_link['confidence_score'], errors='coerce').fillna(-1)

# Remove duplicates (keep highest-priority match, then highest confidence)
borrower_link = borrower_link.sort_values(['match_priority', 'confidence_rank'], ascending=[True, False]).drop_duplicates('borrower_id', keep='first')

print(f'\nTotal unique borrowers matched: {len(borrower_link):,}')
print(f'  - Via linktable (facilityid): {(borrower_link["match_method"] == "linktable_facilityid").sum():,}')
print(f'  - Via linktable (borrowercompanyid): {(borrower_link["match_method"] == "linktable_borrowerid").sum():,}')
print(f'  - Via ticker: {(borrower_link["match_method"] == "ticker").sum():,}')
print(f'  - Via name (exact): {(borrower_link["match_method"] == "name_exact").sum():,}')
print(f'  - Via name (fuzzy): {(borrower_link["match_method"] == "name_fuzzy").sum():,}')

# Calculate overall match rate
total_borrowers = len(borrower_df)
matched_borrowers = len(borrower_link)
print(f'\nOverall match rate: {matched_borrowers:,} / {total_borrowers:,} = {matched_borrowers/total_borrowers*100:.1f}%')

# Save borrower -> gvkey mapping
borrower_link.to_parquet(output_borrower_link_file, index=False)
print(f'Saved borrower-gvkey link map to: {output_borrower_link_file}')



# ## 9. Merge and Time Alignment
#
# Merges the borrower-GVKEY link back to DealScan and aligns each loan to the most recent Compustat quarter (merge_asof). No asset-size filtering is applied.

# ============================================================================
# STEP 7: MERGE DEALSCAN WITH BORROWER-GVKEY LINK
# ============================================================================

print('=' * 70)
print('STEP 7: MERGE DEALSCAN WITH GVKEY')
print('=' * 70)

# Keep source row id so we can merge aligned Compustat back to full sample
df_ds_work = df_ds.reset_index().rename(columns={'index': 'ds_row_id'})

# Add gvkey to DealScan
df_ds_linked = df_ds_work.merge(
    borrower_link[['borrower_id', 'gvkey', 'match_method', 'confidence_score']],
    on='borrower_id',
    how='left'
)

df_ds_linked['gvkey'] = df_ds_linked['gvkey'].apply(normalize_gvkey)

# Calculate deal-level match rate
n_deals_total = df_ds_linked['lpc_deal_id'].nunique()
n_deals_matched = df_ds_linked[df_ds_linked['gvkey'].notna()]['lpc_deal_id'].nunique()

print(f'\nDeal-level match rate:')
print(f'  Total unique deals: {n_deals_total:,}')
print(f'  Deals with gvkey: {n_deals_matched:,} ({n_deals_matched/n_deals_total*100:.1f}%)')
print(f'  Deals without gvkey: {n_deals_total - n_deals_matched:,}')

# Observation-level match rate
n_obs_total = len(df_ds_linked)
n_obs_matched = df_ds_linked['gvkey'].notna().sum()
print(f'\nObservation-level match rate:')
print(f'  Total observations: {n_obs_total:,}')
print(f'  Matched observations: {n_obs_matched:,} ({n_obs_matched/n_obs_total*100:.1f}%)')

# ============================================================================
# STEP 8: TEMPORAL ALIGNMENT WITH COMPUSTAT
# ============================================================================
# Match each DealScan row to the most recent Compustat quarter <= deal_date

print('\n' + '=' * 70)
print('STEP 8: TEMPORAL ALIGNMENT (merge_asof)')
print('=' * 70)

# Keep only gvkey-matched records for merge_asof
df_ds_matched = df_ds_linked[df_ds_linked['gvkey'].notna()].copy()
df_ds_matched = df_ds_matched.sort_values(['deal_date', 'gvkey']).reset_index(drop=True)

# Prepare Compustat for merge
q_cols = df_q.columns.tolist()
df_q_sorted = df_q.copy()
df_q_sorted['gvkey'] = df_q_sorted['gvkey'].apply(normalize_gvkey)
df_q_sorted = df_q_sorted[df_q_sorted['gvkey'].notna()].sort_values(['datadate', 'gvkey']).reset_index(drop=True)

# Merge using merge_asof: find most recent Compustat quarter <= deal_date
# Keep all Compustat vars (overlap cols get _comp suffix)
df_merged_matched = pd.merge_asof(
    df_ds_matched,
    df_q_sorted,
    left_on='deal_date',
    right_on='datadate',
    by='gvkey',
    direction='backward',
    tolerance=pd.Timedelta('365 days'),
    suffixes=('', '_comp')
)

# Check merge success among gvkey-matched rows
n_with_compustat = df_merged_matched['datadate'].notna().sum()
print(f'\nGVKEY-matched rows with aligned Compustat quarter: {n_with_compustat:,} / {len(df_merged_matched):,} ({n_with_compustat/len(df_merged_matched)*100:.1f}%)')

# Calculate time gap between deal and Compustat quarter
df_merged_matched['days_to_compustat'] = (df_merged_matched['deal_date'] - df_merged_matched['datadate']).dt.days
print(f'\nTime gap statistics (deal_date - datadate):')
print(df_merged_matched['days_to_compustat'].describe())

# Merge aligned Compustat back to full DealScan sample
align_cols = ['ds_row_id', 'datadate', 'days_to_compustat']
align_cols += [c for c in df_merged_matched.columns if c in q_cols and c != 'gvkey']
align_cols = list(dict.fromkeys(align_cols))

aligned_subset = df_merged_matched[align_cols].copy()

# Full lender-level linked panel (includes unmatched rows with NaN Compustat vars)
df_merged = df_ds_linked.merge(aligned_subset, on='ds_row_id', how='left')
print(f'\nFinal lender-level dataset shape (all DealScan rows): {df_merged.shape}')
print(f'Rows with aligned Compustat quarter in full sample: {df_merged["datadate"].notna().sum():,} / {len(df_merged):,} ({df_merged["datadate"].notna().mean()*100:.1f}%)')

# Save lender-level output
output_file.parent.mkdir(parents=True, exist_ok=True)
df_merged.to_parquet(output_file, index=False)
print(f'Saved lender-level merged dataset to: {output_file}')

# ============================================================================
# STEP 9: BUILD LOAN-BORROWER PANEL (Quarter-aligned, pre-filter)
# ============================================================================
print('\n' + '=' * 70)
print('STEP 9: BUILD LOAN-BORROWER PANEL')
print('=' * 70)

panel_keys = ['lpc_deal_id', 'lpc_tranche_id', 'borrower_id', 'deal_date', 'gvkey']

# One row per loan-tranche-borrower-date-gvkey
panel_base = df_merged.sort_values(panel_keys + ['datadate']).drop_duplicates(panel_keys, keep='first').copy()

# Syndicate diagnostics from lender-level rows
df_merged['lender_share_num'] = pd.to_numeric(df_merged['lender_share'], errors='coerce')
if 'lead_arranger' in df_merged.columns:
    lead_arranger_flag = df_merged['lead_arranger'].astype(str).str.upper().isin(['YES', 'Y', '1', 'TRUE'])
else:
    lead_arranger_flag = pd.Series(False, index=df_merged.index)
df_merged['lead_arranger_flag'] = lead_arranger_flag.astype(int)

synd_stats = df_merged.groupby(panel_keys, dropna=False).agg(
    n_lender_rows=('borrower_id', 'size'),
    n_lenders=('lender_id', lambda x: x.nunique(dropna=True)),
    n_lender_parents=('lender_parent_id', lambda x: x.nunique(dropna=True)),
    lender_share_nonmissing=('lender_share_num', 'count'),
    lender_share_sum=('lender_share_num', 'sum'),
    lender_share_mean=('lender_share_num', 'mean'),
    lead_arranger_rows=('lead_arranger_flag', 'sum')
).reset_index()

panel = panel_base.merge(synd_stats, on=panel_keys, how='left')
panel['deal_year_qtr'] = panel['deal_date'].dt.to_period('Q').astype(str)
panel['comp_year_qtr'] = panel['datadate'].dt.to_period('Q').astype(str)

print(f'Loan-borrower panel shape: {panel.shape}')
print(f'Unique loan-borrower units: {len(panel):,}')
print(f'Panel rows with gvkey: {panel["gvkey"].notna().sum():,} ({panel["gvkey"].notna().mean()*100:.1f}%)')
print(f'Panel rows with aligned Compustat quarter: {panel["datadate"].notna().sum():,} ({panel["datadate"].notna().mean()*100:.1f}%)')

panel.to_parquet(output_panel_file, index=False)
print(f'Saved loan-borrower panel to: {output_panel_file}')




# ## Linkage Explanation (Detailed)
#
# ### 1. Use the WRDS/Roberts–Chava Link Table First (Best Practice)
# The standard approach in the literature is to use the **Roberts DealScan–Compustat Linking Database** (Chava & Roberts, 2008), distributed through WRDS. This file maps DealScan identifiers to Compustat **GVKEY**, and includes both **facility-level IDs** (`facilityid`) and **borrowercompanyid** mappings plus match-quality fields (e.g., confidence scores). We load the WRDS link table (`Dealscan-Compustat_Linking_Database012024.xlsx`, sheet `links`) and prioritize its matches.
#
# ### 2. Facility-Level Match (Most Precise)
# We first match **DealScan `lpc_tranche_id` → link table `facilityid` → Compustat `gvkey`**. This provides the most precise facility-level borrower mapping. We keep all matches regardless of asset size.
#
# ### 3. Borrowercompanyid Match (Secondary)
# If a facility-level match is missing, we fall back to **DealScan `borrower_id` → link table `borrowercompanyid` → `gvkey`**. This is a borrower-level link and typically broader but still based on the WRDS link table.
#
# ### 4. Ticker Match (Fallback)
# For borrowers not linked by the WRDS table, we match **DealScan `ticker`** to **Compustat `tic`** after cleaning (uppercasing, stripping punctuation, removing suffixes). This is a common fallback in the literature when link tables are incomplete.
#
# ### 5. Name Match (Last Resort)
# Finally, we use standardized company names for an **exact name match** (`borrower_name` ↔ `conm`). This captures additional matches but should be treated as lower confidence. If needed, a token/fuzzy match can be added with manual verification for ambiguous cases.
#
# ### 6. Priority Rules
# We keep **one GVKEY per borrower** and apply a priority order: facility-level link table > borrowercompanyid link table > ticker > name. This prevents duplicate matches and keeps the most reliable mapping.
#
# ### 7. Time Alignment to Compustat
# After assigning `gvkey`, we align each DealScan observation to the **most recent Compustat quarter ≤ deal date** using `merge_asof` with a 1‑year tolerance. This ensures financials are lagged and available at origination.
#
# ### 8. Diagnostics
# We report match rates at the borrower, deal, and observation levels, and compute date-gap statistics between loan origination and Compustat quarter ends. This helps validate match quality.
#
# **Important constraint:** we do **not** drop firms based on asset size (e.g., `atq < 1`) at any step of the linking pipeline.
#
# ### References
# - Chava, S., & Roberts, M. (2008). *How Does Financing Impact Investment? The Role of Debt Covenants.*
# - Roberts DealScan–Compustat Linking Database (WRDS).
# - Ferracuti & Morris (2022) extension notes (DealScan–Compustat link update).
# - Cohen et al. (2015) NBER working paper on matching DealScan to Compustat using tickers and names.
#
#
# ### 9. Confidence Scores and Fuzzy Backup
# We carry over **link-table confidence scores** (when provided by WRDS). For fuzzy-name matches, we store the **string similarity score** as a proxy confidence. These fields can be used to filter to higher-confidence links in robustness checks.
