from IPython.display import display
import os
# # Manual Linking Audit Notebook
#
# This notebook is a clean manual-audit workspace for your final merged datasets.
#

# ## Table Of Contents
#
# - 0) Final Dataset Structure (Quick Access)
# - 1) Load Data
# - 2) Coverage Snapshots
# - 3) Final Analysis Dataset Structure (Detailed)
# - 4) Mapping Quality Checks
# - 5) Name Comparison: DealScan Lender vs Call/SOD Bank
# - 6) Unique Name-Mapping Table
# - 7) Loan-Level Coverage Checks
# - 8) Manual Audit Helpers
# - 9) Summary Stats and Missingness
#
# If markdown links do not jump in your IDE, use the Outline panel (left side).
#

# ## 0) Final Dataset Structure (Quick Access)

from pathlib import Path
import re
import numpy as np
import pandas as pd

pd.set_option('display.max_columns', 200)
pd.set_option('display.max_rows', 200)
pd.set_option('display.width', 240)

BASE = Path(os.environ.get("MPHIL_THESIS_ROOT", Path(__file__).resolve().parents[2]))

P_LENDER_PANEL = BASE / 'data/intermediate/merged/dealscan_compustat_call_sod_macro_lenderlevel.parquet'
P_LOAN_PANEL = BASE / 'data/intermediate/merged/dealscan_compustat_call_sod_macro_loan_borrower_panel.parquet'
P_LENDER_MAP = BASE / 'data/intermediate/merged/dealscan_lender_rssd_mapping.parquet'
P_LENDER_COV = BASE / 'data/intermediate/merged/lender_bank_hhi_coverage_summary.csv'
P_LOAN_COV = BASE / 'data/intermediate/merged/loan_lender_match_coverage_summary.csv'
P_YEAR_COV = BASE / 'data/intermediate/merged/lender_bank_hhi_coverage_by_year.csv'
P_CALL = BASE / 'data/intermediate/clean/fdic_call_reports/fdic_call_reports_clean.parquet'
P_SOD = BASE / 'data/intermediate/clean/fdic_sod/fdic_sod_cleaned.parquet'

print('Main files:')
for p in [P_LENDER_PANEL, P_LOAN_PANEL, P_LENDER_MAP, P_LENDER_COV, P_LOAN_COV, P_YEAR_COV, P_CALL, P_SOD]:
    print(' -', p.name, '| exists =', p.exists())


# ## 1) Load Data

lender_panel = pd.read_parquet(P_LENDER_PANEL)
loan_panel = pd.read_parquet(P_LOAN_PANEL)
lender_map = pd.read_parquet(P_LENDER_MAP)
lender_cov = pd.read_csv(P_LENDER_COV)
loan_cov = pd.read_csv(P_LOAN_COV)
year_cov = pd.read_csv(P_YEAR_COV)
call_df = pd.read_parquet(P_CALL)
sod_df = pd.read_parquet(P_SOD)

print('Loaded shapes:')
print('  lender_panel:', lender_panel.shape)
print('  loan_panel  :', loan_panel.shape)
print('  lender_map  :', lender_map.shape)
print('  call_df     :', call_df.shape)
print('  sod_df      :', sod_df.shape)


# ## 2) Coverage Snapshots

display(lender_cov.sort_values('rows', ascending=False))
display(loan_cov)
display(year_cov.sort_values('deal_year'))


# ## 3) Final Analysis Dataset Structure (Detailed)

lender_key = [c for c in ['lpc_deal_id','lpc_tranche_id','borrower_id','lender_id','deal_date'] if c in lender_panel.columns]
loan_key = [c for c in ['lpc_deal_id','lpc_tranche_id','borrower_id','deal_date','gvkey'] if c in loan_panel.columns]

print('Lender-level panel (main regression panel)')
print('  Unit of observation: deal-tranche-borrower-lender')
print('  Rows/Cols:', lender_panel.shape)
if lender_key:
    print('  Key:', lender_key)
    print('  Unique key combos:', f"{lender_panel[lender_key].drop_duplicates().shape[0]:,}")

print('\nLoan-borrower panel (aggregated panel)')
print('  Unit of observation: deal-tranche-borrower')
print('  Rows/Cols:', loan_panel.shape)
if loan_key:
    print('  Key:', loan_key)
    print('  Unique key combos:', f"{loan_panel[loan_key].drop_duplicates().shape[0]:,}")

if 'deal_date' in lender_panel.columns and 'deal_date' in loan_panel.columns:
    d1 = pd.to_datetime(lender_panel['deal_date'], errors='coerce')
    d2 = pd.to_datetime(loan_panel['deal_date'], errors='coerce')
    overview = pd.DataFrame({
        'dataset': ['lender_panel', 'loan_panel'],
        'date_min': [d1.min(), d2.min()],
        'date_max': [d1.max(), d2.max()],
        'n_rows': [len(lender_panel), len(loan_panel)],
        'n_cols': [lender_panel.shape[1], loan_panel.shape[1]],
    })
    display(overview)

id_cols = [c for c in ['ds_row_id','lpc_deal_id','lpc_tranche_id','borrower_id','lender_id','gvkey','mapped_rssd_id','deal_date','deal_quarter_end'] if c in lender_panel.columns]
print('Sample IDs from lender_panel:')
display(lender_panel[id_cols].head(20))


# ## 4) Mapping Quality Checks

summary = {
    'rows_lender_panel': len(lender_panel),
    'rows_lender_map': len(lender_map),
    'unique_ds_row_id_lender_panel': lender_panel['ds_row_id'].nunique() if 'ds_row_id' in lender_panel.columns else np.nan,
    'unique_ds_row_id_lender_map': lender_map['ds_row_id'].nunique() if 'ds_row_id' in lender_map.columns else np.nan,
    'rssd_mapped_rows': int(lender_panel['flag_lender_rssd_mapped'].sum()) if 'flag_lender_rssd_mapped' in lender_panel.columns else np.nan,
    'call_matched_rows': int(lender_panel['flag_lender_call_matched'].sum()) if 'flag_lender_call_matched' in lender_panel.columns else np.nan,
    'hhi_matched_rows': int(lender_panel['flag_lender_hhi_matched'].sum()) if 'flag_lender_hhi_matched' in lender_panel.columns else np.nan,
}
display(pd.Series(summary).to_frame('value'))

if 'lender_rssd_match_method' in lender_panel.columns:
    display(lender_panel['lender_rssd_match_method'].value_counts(dropna=False).to_frame('rows'))


# ## 5) Name Comparison: DealScan Lender vs Call/SOD Bank

def clean_name_for_compare(x):
    if pd.isna(x):
        return ''
    s = str(x).upper().strip()
    s = re.sub(r'[^A-Z0-9]+', ' ', s)
    tokens = [t for t in s.split() if t not in {
        'NATIONAL','ASSOCIATION','NA','N','A','BANK','BANCORP','CORP','CORPORATION',
        'INC','LIMITED','LTD','LLC','CO','COMPANY','THE','TRUST','HOLDINGS','HOLDING','SA','PLC'
    }]
    return ' '.join(tokens)

# Call latest bank name per RSSD
call_name_col = None
for c in ['RSSD9017', 'bank_name', 'NAME', 'name']:
    if c in call_df.columns:
        call_name_col = c
        break

call_names = pd.DataFrame(columns=['mapped_rssd_id','call_bank_name_latest'])
if call_name_col is not None and {'RSSDID','REPDTE'}.issubset(call_df.columns):
    tmp = call_df[['RSSDID','REPDTE',call_name_col]].copy()
    tmp['RSSDID'] = pd.to_numeric(tmp['RSSDID'], errors='coerce')
    tmp['REPDTE'] = pd.to_datetime(tmp['REPDTE'], errors='coerce')
    tmp = tmp.dropna(subset=['RSSDID', call_name_col]).sort_values(['RSSDID','REPDTE'])
    tmp = tmp.groupby('RSSDID').tail(1)[['RSSDID', call_name_col]]
    call_names = tmp.rename(columns={'RSSDID':'mapped_rssd_id', call_name_col:'call_bank_name_latest'})

# SOD latest bank name per RSSD (best effort)
sod_name_col = None
for c in ['BANKNAME','bank_name','NAMEFULL','NAME','name']:
    if c in sod_df.columns:
        sod_name_col = c
        break

sod_names = pd.DataFrame(columns=['mapped_rssd_id','sod_bank_name_latest'])
if sod_name_col is not None and 'RSSDID' in sod_df.columns:
    cols = ['RSSDID', sod_name_col] + ([c for c in ['YEAR'] if c in sod_df.columns])
    tmp = sod_df[cols].copy()
    tmp['RSSDID'] = pd.to_numeric(tmp['RSSDID'], errors='coerce')
    if 'YEAR' in tmp.columns:
        tmp['YEAR'] = pd.to_numeric(tmp['YEAR'], errors='coerce')
        tmp = tmp.dropna(subset=['RSSDID', sod_name_col]).sort_values(['RSSDID','YEAR'])
    else:
        tmp = tmp.dropna(subset=['RSSDID', sod_name_col]).sort_values(['RSSDID'])
    tmp = tmp.groupby('RSSDID').tail(1)[['RSSDID', sod_name_col]]
    sod_names = tmp.rename(columns={'RSSDID':'mapped_rssd_id', sod_name_col:'sod_bank_name_latest'})

name_compare = lender_panel[['ds_row_id','deal_date','lender_id','lender_name','mapped_rssd_id','lender_rssd_match_method','lender_category']].copy()
name_compare['mapped_rssd_id'] = pd.to_numeric(name_compare['mapped_rssd_id'], errors='coerce')
name_compare = name_compare.merge(call_names, on='mapped_rssd_id', how='left')
name_compare = name_compare.merge(sod_names, on='mapped_rssd_id', how='left')

name_compare['lender_name_clean'] = name_compare['lender_name'].map(clean_name_for_compare)
name_compare['call_name_clean'] = name_compare['call_bank_name_latest'].map(clean_name_for_compare)
name_compare['sod_name_clean'] = name_compare['sod_bank_name_latest'].map(clean_name_for_compare)
name_compare['clean_name_equal_call'] = (name_compare['lender_name_clean'] == name_compare['call_name_clean']) & (name_compare['call_name_clean'] != '')

print('call name column used:', call_name_col)
print('sod name column used :', sod_name_col)

display(name_compare[name_compare['mapped_rssd_id'].notna()][[
    'deal_date','lender_id','lender_name','mapped_rssd_id','lender_rssd_match_method',
    'call_bank_name_latest','sod_bank_name_latest','lender_name_clean','call_name_clean','sod_name_clean','clean_name_equal_call'
]].head(100))


# ## 6) Unique Name-Mapping Table

unique_name_compare = name_compare[name_compare['mapped_rssd_id'].notna()].copy()
unique_name_compare = unique_name_compare.sort_values(['lender_id','mapped_rssd_id','deal_date'])
unique_name_compare = unique_name_compare.drop_duplicates(
    ['lender_id','lender_name','mapped_rssd_id','call_bank_name_latest','sod_bank_name_latest'], keep='first'
)

display(unique_name_compare[[
    'lender_id','lender_name','mapped_rssd_id','lender_rssd_match_method','lender_category',
    'call_bank_name_latest','sod_bank_name_latest','lender_name_clean','call_name_clean','sod_name_clean','clean_name_equal_call'
]].head(500))

print('Unique lender-name mapping rows:', len(unique_name_compare))

if 'lender_rssd_match_method' in unique_name_compare.columns:
    summary = unique_name_compare.groupby('lender_rssd_match_method').agg(
        unique_rows=('lender_id','size'),
        exact_clean_name_match=('clean_name_equal_call','sum')
    ).reset_index()
    summary['exact_clean_name_match_rate'] = summary['exact_clean_name_match'] / summary['unique_rows']
    display(summary.sort_values('unique_rows', ascending=False))


# ## 7) Loan-Level Coverage Checks

loan_summary = {
    'loan_rows': len(loan_panel),
    'share_any_lender_unmatched_rssd': float(loan_panel['flag_any_lender_unmatched_rssd'].mean()) if 'flag_any_lender_unmatched_rssd' in loan_panel.columns else np.nan,
    'share_all_lender_unmatched_rssd': float(loan_panel['flag_all_lenders_unmatched_rssd'].mean()) if 'flag_all_lenders_unmatched_rssd' in loan_panel.columns else np.nan,
    'share_any_lender_unmatched_hhi': float(loan_panel['flag_any_lender_unmatched_hhi'].mean()) if 'flag_any_lender_unmatched_hhi' in loan_panel.columns else np.nan,
    'share_all_lender_unmatched_hhi': float(loan_panel['flag_all_lenders_unmatched_hhi'].mean()) if 'flag_all_lenders_unmatched_hhi' in loan_panel.columns else np.nan,
}
display(pd.Series(loan_summary).to_frame('value'))

cols = [
    'lpc_deal_id','lpc_tranche_id','borrower_id','deal_date','gvkey',
    'n_lender_rows','n_lenders','n_lenders_rssd_mapped','n_lenders_hhi_matched',
    'share_lenders_rssd_mapped','share_lenders_hhi_matched',
    'flag_any_lender_unmatched_rssd','flag_all_lenders_unmatched_rssd',
    'flag_any_lender_unmatched_hhi','flag_all_lenders_unmatched_hhi'
]
cols = [c for c in cols if c in loan_panel.columns]

display(loan_panel[cols].sort_values(cols[-1] if cols else 'deal_date', ascending=False).head(100))


# ## 8) Manual Audit Helpers

def audit_lender(lender_id=None, lender_name_contains=None, method=None, unique_only=True, n=200):
    df = name_compare.copy()
    if lender_id is not None:
        df = df[pd.to_numeric(df['lender_id'], errors='coerce') == lender_id]
    if lender_name_contains is not None:
        df = df[df['lender_name'].astype(str).str.contains(lender_name_contains, case=False, na=False)]
    if method is not None:
        df = df[df['lender_rssd_match_method'] == method]
    if unique_only:
        df = df.sort_values(['lender_id','mapped_rssd_id','deal_date'])
        df = df.drop_duplicates(['lender_id','lender_name','mapped_rssd_id','call_bank_name_latest','sod_bank_name_latest'], keep='first')

    cols = [
        'deal_date','lender_id','lender_name','mapped_rssd_id','lender_rssd_match_method','lender_category',
        'call_bank_name_latest','sod_bank_name_latest','lender_name_clean','call_name_clean','sod_name_clean','clean_name_equal_call'
    ]
    return df[cols].head(n)


def audit_rssd(rssd_id, n=200):
    df = lender_panel[pd.to_numeric(lender_panel['mapped_rssd_id'], errors='coerce') == rssd_id].copy()
    cols = [
        'deal_date','lender_id','lender_name','mapped_rssd_id','lender_rssd_match_method',
        'flag_lender_rssd_mapped','flag_lender_call_matched','flag_lender_hhi_matched','REPDTE','hhi_bank_q'
    ]
    cols = [c for c in cols if c in df.columns]
    return df[cols].sort_values('deal_date').head(n)


def audit_deal(lpc_deal_id=None, lpc_tranche_id=None, borrower_id=None, n=200):
    df = lender_panel.copy()
    if lpc_deal_id is not None:
        df = df[df['lpc_deal_id'] == lpc_deal_id]
    if lpc_tranche_id is not None:
        df = df[df['lpc_tranche_id'] == lpc_tranche_id]
    if borrower_id is not None:
        df = df[df['borrower_id'] == borrower_id]
    cols = [
        'lpc_deal_id','lpc_tranche_id','borrower_id','gvkey','deal_date',
        'lender_id','lender_name','mapped_rssd_id','lender_rssd_match_method',
        'lender_share','flag_lender_call_matched','flag_lender_hhi_matched','hhi_bank_q'
    ]
    cols = [c for c in cols if c in df.columns]
    return df[cols].sort_values(['lpc_deal_id','lpc_tranche_id']).head(n)

print('Helpers loaded:')
print(' - audit_lender(lender_id=..., lender_name_contains=..., method=..., unique_only=True, n=...)')
print(' - audit_rssd(rssd_id=..., n=...)')
print(' - audit_deal(lpc_deal_id=..., lpc_tranche_id=..., borrower_id=..., n=...)')


# Example usage (uncomment to run)
# display(audit_lender(lender_name_contains='Wachovia', unique_only=True, n=100))
# display(audit_lender(method='manual_top_lender', unique_only=True, n=200))
# display(audit_rssd(451965, n=100))
# display(audit_deal(lpc_deal_id=12345, n=100))


# ## 9) Summary Stats and Missingness

print('Summary stats (selected variables)')
loan_vars = [v for v in ['all_in_spread_drawn_bps','tenor_maturity','tranche_amount_converted','secured_flag','covenants'] if v in lender_panel.columns]
bank_vars = [v for v in ['call_total_assets_mil','call_capital_ratio','call_loans_to_assets','hhi_bank_q'] if v in lender_panel.columns]
borrower_vars = [v for v in ['log_assets','leverage','roa','market_to_book','tangibility'] if v in lender_panel.columns]
macro_vars = [v for v in ['macro_ffr_q','jk_MP_median_sum','jk_CBI_median_sum','macro_gdp_growth_q'] if v in lender_panel.columns]
for label, vv in [('Loan', loan_vars), ('Bank', bank_vars), ('Borrower', borrower_vars), ('Macro', macro_vars)]:
    if vv:
        print('\n', label, 'variables')
        display(lender_panel[vv].describe().T)
print('\nMissingness (selected analysis vars)')
check_vars = list(dict.fromkeys(loan_vars + bank_vars + borrower_vars + macro_vars + ['mapped_rssd_id', 'gvkey']))
miss = []
for v in check_vars:
    if v in lender_panel.columns:
        n = len(lender_panel)
        nn = lender_panel[v].notna().sum()
        miss.append({'variable': v, 'non_missing': int(nn), 'coverage_pct': float(nn / n * 100)})
display(pd.DataFrame(miss).sort_values('coverage_pct'))
