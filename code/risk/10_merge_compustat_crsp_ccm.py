from __future__ import annotations

from datetime import date
import os
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(os.environ.get('MPHIL_THESIS_ROOT', Path(__file__).resolve().parents[2]))
P_FUNDQ = BASE / 'data/raw/compustat/fundq__2001_2025.parquet'
P_CCM = BASE / 'data/raw/ccm/ccmxpf_lnkhist__2001_2025.parquet'
P_MSF_DIR = BASE / 'data/raw/crsp/msf'
P_DSF_DIR = BASE / 'data/raw/crsp/dsf'

OUT_MERGED = BASE / 'data/intermediate/merged/compustat_crsp_ccm_quarterly_2001_2025.parquet'
OUT_QUALITY = BASE / 'data/documentation/compustat' / f'compustat_crsp_merge_quality_{date.today().isoformat()}.csv'

OUT_MERGED.parent.mkdir(parents=True, exist_ok=True)
OUT_QUALITY.parent.mkdir(parents=True, exist_ok=True)

print('Paths ready')
print('OUT_MERGED =', OUT_MERGED)
print('OUT_QUALITY =', OUT_QUALITY)


# 1) Load Compustat quarter-level base
fund_cols = [
    'gvkey', 'datadate', 'fyearq', 'fqtr', 'tic', 'conm',
    'dlcq', 'dlttq', 'atq', 'ceqq', 'ltq', 'niq', 'oibdpq', 'xintq'
]
fundq = pd.read_parquet(P_FUNDQ, columns=fund_cols)
fundq['gvkey'] = fundq['gvkey'].astype(str).str.strip()
fundq['datadate'] = pd.to_datetime(fundq['datadate'], errors='coerce')
fundq = fundq.dropna(subset=['gvkey', 'datadate']).copy()
fundq = fundq[(fundq['datadate'] >= '2001-01-01') & (fundq['datadate'] <= '2025-12-31')].copy()
fundq['qtr'] = fundq['datadate'].dt.to_period('Q')

# one row per gvkey-quarter for merge quality accounting
fundq_q = (
    fundq.sort_values(['gvkey', 'datadate'])
    .drop_duplicates(['gvkey', 'qtr'], keep='last')
    .reset_index(drop=True)
)

print('Compustat gvkey-quarter rows:', f"{len(fundq_q):,}")
print('Compustat unique gvkeys:', f"{fundq_q['gvkey'].nunique():,}")
print('Quarter range:', fundq_q['qtr'].min(), 'to', fundq_q['qtr'].max())


# 2) Load CCM and apply standard WRDS filters
ccm = pd.read_parquet(P_CCM)
ccm['gvkey'] = ccm['gvkey'].astype(str).str.strip()
ccm['lpermno'] = pd.to_numeric(ccm['lpermno'], errors='coerce').astype('Int64')
ccm['linkdt'] = pd.to_datetime(ccm['linkdt'], errors='coerce')
ccm['linkenddt'] = pd.to_datetime(ccm['linkenddt'], errors='coerce')
ccm['linkenddt'] = ccm['linkenddt'].fillna(pd.Timestamp('2099-12-31'))

ccm_std = ccm[
    ccm['lpermno'].notna()
    & ccm['linktype'].isin(['LC', 'LU', 'LS', 'LD', 'LN'])
    & ccm['linkprim'].isin(['P', 'C'])
].copy()
ccm_std['permno'] = ccm_std['lpermno'].astype(int)
ccm_std['prim_rank'] = np.where(ccm_std['linkprim'].eq('P'), 0, 1)

print('CCM rows (raw):', f"{len(ccm):,}")
print('CCM rows (standard filter):', f"{len(ccm_std):,}")
print('CCM unique gvkeys (std):', f"{ccm_std['gvkey'].nunique():,}")
print('CCM unique permnos (std):', f"{ccm_std['permno'].nunique():,}")


# 3) Merge Compustat quarter rows to CCM by valid link date window
tmp = fundq_q.merge(
    ccm_std[['gvkey', 'permno', 'linkdt', 'linkenddt', 'prim_rank']],
    on='gvkey',
    how='left'
)
tmp = tmp[(tmp['datadate'] >= tmp['linkdt']) & (tmp['datadate'] <= tmp['linkenddt'])].copy()
tmp = tmp.sort_values(['gvkey', 'qtr', 'prim_rank', 'linkdt'], ascending=[True, True, True, False])
comp_ccm = tmp.drop_duplicates(['gvkey', 'qtr'], keep='first').reset_index(drop=True)

base_rows = len(fundq_q)
matched_ccm_rows = len(comp_ccm)
ccm_match_rate = matched_ccm_rows / base_rows if base_rows else np.nan

print('Rows after CCM date-valid mapping:', f"{matched_ccm_rows:,}")
print('CCM match rate:', f"{ccm_match_rate:.2%}")


# 4) Build CRSP monthly quarter-end features
msf_files = sorted(P_MSF_DIR.glob('msf_*__borrower_permnos.parquet'))
msf_parts = []
for fp in msf_files:
    d = pd.read_parquet(fp, columns=['permno', 'date', 'ret', 'prc', 'shrout', 'vol'])
    msf_parts.append(d)
msf = pd.concat(msf_parts, ignore_index=True)
msf['permno'] = pd.to_numeric(msf['permno'], errors='coerce').astype('Int64')
msf['date'] = pd.to_datetime(msf['date'], errors='coerce')
msf = msf.dropna(subset=['permno', 'date']).copy()
msf['permno'] = msf['permno'].astype(int)
msf['qtr'] = msf['date'].dt.to_period('Q')
msf = msf.sort_values(['permno', 'date'])
msf_qe = msf.groupby(['permno', 'qtr'], as_index=False).tail(1).copy()
msf_qe['mcap_qe_m'] = msf_qe['prc'].abs() * msf_qe['shrout'] * 1000.0
msf_qe = msf_qe.rename(columns={
    'date': 'msf_qe_date',
    'ret': 'msf_qe_ret',
    'prc': 'msf_qe_prc',
    'shrout': 'msf_qe_shrout',
    'vol': 'msf_qe_vol'
})[['permno', 'qtr', 'msf_qe_date', 'msf_qe_ret', 'msf_qe_prc', 'msf_qe_shrout', 'msf_qe_vol', 'mcap_qe_m']]

print('MSF quarter-end rows:', f"{len(msf_qe):,}")


# 5) Build CRSP daily quarter features (per year to control memory)
def agg_dsf_year(fp: Path) -> pd.DataFrame:
    d = pd.read_parquet(fp, columns=['permno', 'date', 'ret', 'prc', 'shrout', 'vol'])
    if d.empty:
        return pd.DataFrame(columns=['permno', 'qtr', 'dsf_days', 'dsf_ret_std_q', 'dsf_qe_date', 'dsf_qe_prc', 'dsf_qe_shrout', 'dsf_qe_vol', 'mcap_qe_d'])
    d['permno'] = pd.to_numeric(d['permno'], errors='coerce').astype('Int64')
    d['date'] = pd.to_datetime(d['date'], errors='coerce')
    d = d.dropna(subset=['permno', 'date']).copy()
    d['permno'] = d['permno'].astype(int)
    d['qtr'] = d['date'].dt.to_period('Q')
    d = d.sort_values(['permno', 'date'])
    g = d.groupby(['permno', 'qtr'], as_index=False)
    stat = g.agg(dsf_days=('date', 'count'), dsf_ret_std_q=('ret', 'std'))
    qe = g.tail(1).rename(columns={
        'date': 'dsf_qe_date',
        'prc': 'dsf_qe_prc',
        'shrout': 'dsf_qe_shrout',
        'vol': 'dsf_qe_vol'
    })[['permno', 'qtr', 'dsf_qe_date', 'dsf_qe_prc', 'dsf_qe_shrout', 'dsf_qe_vol']]
    out = stat.merge(qe, on=['permno', 'qtr'], how='left')
    out['mcap_qe_d'] = out['dsf_qe_prc'].abs() * out['dsf_qe_shrout'] * 1000.0
    return out

dsf_files = sorted(P_DSF_DIR.glob('dsf_*__borrower_permnos.parquet'))
dsf_q_parts = []
for fp in dsf_files:
    dsf_q_parts.append(agg_dsf_year(fp))
dsf_q = pd.concat(dsf_q_parts, ignore_index=True) if dsf_q_parts else pd.DataFrame()

print('DSF quarter features rows:', f"{len(dsf_q):,}")


# 6) Merge everything and save intermediate dataset
merged = comp_ccm.merge(msf_qe, on=['permno', 'qtr'], how='left')
merged = merged.merge(dsf_q, on=['permno', 'qtr'], how='left')
merged['year'] = merged['qtr'].dt.year
merged['quarter'] = merged['qtr'].dt.quarter

merged = merged.sort_values(['gvkey', 'year', 'quarter']).reset_index(drop=True)
merged.to_parquet(OUT_MERGED, index=False)

print('Saved merged file:', OUT_MERGED)
print('Merged rows:', f"{len(merged):,}")
print('Merged cols:', len(merged.columns))


# 7) Merge quality report and matching rates
msf_match = merged['msf_qe_date'].notna().mean() if len(merged) else np.nan
dsf_match = merged['dsf_qe_date'].notna().mean() if len(merged) else np.nan
daily_days_nonzero = (merged['dsf_days'].fillna(0) > 0).mean() if len(merged) else np.nan

quality = pd.DataFrame([
    {'metric': 'compustat_base_rows_gvkey_qtr', 'value': base_rows},
    {'metric': 'compustat_unique_gvkey', 'value': fundq_q['gvkey'].nunique()},
    {'metric': 'ccm_mapped_rows', 'value': matched_ccm_rows},
    {'metric': 'ccm_match_rate', 'value': ccm_match_rate},
    {'metric': 'merged_rows', 'value': len(merged)},
    {'metric': 'msf_quarter_end_match_rate', 'value': msf_match},
    {'metric': 'dsf_quarter_end_match_rate', 'value': dsf_match},
    {'metric': 'dsf_days_gt_zero_rate', 'value': daily_days_nonzero},
    {'metric': 'merged_unique_gvkey', 'value': merged['gvkey'].nunique()},
    {'metric': 'merged_unique_permno', 'value': merged['permno'].nunique()},
    {'metric': 'quarter_min', 'value': str(merged['qtr'].min()) if len(merged) else ''},
    {'metric': 'quarter_max', 'value': str(merged['qtr'].max()) if len(merged) else ''},
])
quality.to_csv(OUT_QUALITY, index=False)
quality
