from __future__ import annotations

from datetime import date
import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm

BASE = Path(os.environ.get('MPHIL_THESIS_ROOT', Path(__file__).resolve().parents[2]))
IN_MERGED = BASE / 'data/intermediate/merged/compustat_crsp_ccm_quarterly_2001_2025.parquet'

OUT_DTD_PQ = BASE / 'data/intermediate/merged/bs2008_naive_dtd_quarterly_2001_2025.parquet'
OUT_DTD_CSV = BASE / 'data/intermediate/merged/bs2008_naive_dtd_quarterly_2001_2025.csv'
OUT_SUMMARY = BASE / 'data/intermediate/crsp' / f'bs2008_naive_dtd_quality_{date.today().isoformat()}.csv'

OUT_DTD_PQ.parent.mkdir(parents=True, exist_ok=True)
OUT_SUMMARY.parent.mkdir(parents=True, exist_ok=True)

IN_MERGED, OUT_DTD_PQ, OUT_SUMMARY

df = pd.read_parquet(IN_MERGED)

required = [
    'gvkey', 'permno', 'qtr', 'year', 'quarter', 'datadate',
    'dlcq', 'dlttq', 'mcap_qe_d', 'mcap_qe_m', 'dsf_ret_std_q', 'dsf_days'
]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f'Missing required columns: {missing}')

for c in ['dlcq', 'dlttq', 'mcap_qe_d', 'mcap_qe_m', 'dsf_ret_std_q', 'dsf_days']:
    df[c] = pd.to_numeric(df[c], errors='coerce')

df['gvkey'] = df['gvkey'].astype(str).str.strip()
df['qtr'] = pd.PeriodIndex(df['qtr'], freq='Q')
df = df.sort_values(['gvkey', 'qtr']).reset_index(drop=True)

print('Rows:', f"{len(df):,}")
print('Unique gvkey:', f"{df['gvkey'].nunique():,}")
print('Quarter range:', df['qtr'].min(), 'to', df['qtr'].max())

print('\nNull share by key DtD inputs:')
for c in ['mcap_qe_d', 'mcap_qe_m', 'dlcq', 'dlttq', 'dsf_ret_std_q', 'dsf_days']:
    print(f"  {c}: {df[c].isna().mean():.2%}")

# Equity in USD: daily quarter-end preferred; monthly fallback
E_usd = df['mcap_qe_d'].copy()
E_usd = E_usd.where(E_usd.notna(), df['mcap_qe_m'])

# Harmonized units: thousand USD
df['E_kusd'] = E_usd / 1000.0

# Debt proxy (thousand USD)
df['F_kusd'] = 1000.0 * (df['dlcq'].fillna(0.0) + 0.5 * df['dlttq'].fillna(0.0))

# One-year lagged log return using quarter-end equity
df['E_kusd_lag4'] = df.groupby('gvkey')['E_kusd'].shift(4)
df['ret_1y_log'] = np.where(
    (df['E_kusd'] > 0) & (df['E_kusd_lag4'] > 0),
    np.log(df['E_kusd'] / df['E_kusd_lag4']),
    np.nan,
)

print('E_kusd non-missing share:', f"{df['E_kusd'].notna().mean():.2%}")
print('F_kusd > 0 share:', f"{(df['F_kusd'] > 0).mean():.2%}")
print('ret_1y_log non-missing share:', f"{df['ret_1y_log'].notna().mean():.2%}")

def compute_sigma_e_1y(group: pd.DataFrame) -> pd.DataFrame:
    g = group.sort_values('qtr').copy()

    n = g['dsf_days'].where(g['dsf_days'] > 1)
    s = g['dsf_ret_std_q'].where(g['dsf_ret_std_q'] > 0)

    a = (n - 1.0) * (s ** 2)
    b = (n - 1.0)

    valid = (a.notna() & b.notna()).astype(float)
    g['sigma_e_valid_quarters'] = valid.rolling(window=4, min_periods=1).sum()

    a_roll = a.rolling(window=4, min_periods=2).sum()
    b_roll = b.rolling(window=4, min_periods=2).sum()
    var_daily_1y = a_roll / b_roll

    g['sigma_e_1y_ann'] = np.sqrt(252.0 * var_daily_1y)
    return g


df = (
    df.groupby('gvkey', group_keys=False)
      .apply(compute_sigma_e_1y)
      .reset_index(drop=True)
)

print('sigma_e_1y_ann non-missing share:', f"{df['sigma_e_1y_ann'].notna().mean():.2%}")
print('sigma_e_1y_ann median:', round(float(df['sigma_e_1y_ann'].median(skipna=True)), 4))

df['sigma_d_naive'] = 0.05 + 0.25 * df['sigma_e_1y_ann']
df['V_naive_kusd'] = df['E_kusd'] + df['F_kusd']

wE = df['E_kusd'] / df['V_naive_kusd']
wF = df['F_kusd'] / df['V_naive_kusd']
df['sigma_v_naive'] = wE * df['sigma_e_1y_ann'] + wF * df['sigma_d_naive']

valid = (
    (df['E_kusd'] > 0)
    & (df['F_kusd'] > 0)
    & (df['V_naive_kusd'] > 0)
    & (df['sigma_e_1y_ann'] > 0)
    & (df['sigma_v_naive'] > 0)
    & df['ret_1y_log'].notna()
)

num = np.log(df['V_naive_kusd'] / df['F_kusd']) + (df['ret_1y_log'] - 0.5 * (df['sigma_v_naive'] ** 2))
den = df['sigma_v_naive']

df['dd_naive_bs2008'] = np.where(valid, num / den, np.nan)
df['pi_naive_bs2008'] = np.where(df['dd_naive_bs2008'].notna(), norm.cdf(-df['dd_naive_bs2008']), np.nan)

print('dd_naive non-missing share:', f"{df['dd_naive_bs2008'].notna().mean():.2%}")
print('dd_naive median:', round(float(df['dd_naive_bs2008'].median(skipna=True)), 4))
print('pi_naive median:', round(float(df['pi_naive_bs2008'].median(skipna=True)), 4))

# Export the thesis-relevant naive measure; historical iterative-model comparison omitted.

out_cols = [
    'gvkey', 'permno', 'year', 'quarter', 'qtr', 'datadate',
    'dlcq', 'dlttq',
    'mcap_qe_d', 'mcap_qe_m',
    'E_kusd', 'F_kusd', 'E_kusd_lag4', 'ret_1y_log',
    'dsf_days', 'dsf_ret_std_q', 'sigma_e_valid_quarters', 'sigma_e_1y_ann',
    'sigma_d_naive', 'V_naive_kusd', 'sigma_v_naive',
    'dd_naive_bs2008', 'pi_naive_bs2008'
]

out = df[out_cols].copy().sort_values(['gvkey', 'year', 'quarter'])
out.to_parquet(OUT_DTD_PQ, index=False)
out.to_csv(OUT_DTD_CSV, index=False)

summary = pd.DataFrame([
    {'metric': 'rows_output', 'value': len(out)},
    {'metric': 'unique_gvkey', 'value': out['gvkey'].nunique()},
    {'metric': 'unique_permno', 'value': out['permno'].nunique()},
    {'metric': 'quarter_min', 'value': str(out['qtr'].min()) if len(out) else ''},
    {'metric': 'quarter_max', 'value': str(out['qtr'].max()) if len(out) else ''},
    {'metric': 'share_E_positive', 'value': float((out['E_kusd'] > 0).mean()) if len(out) else np.nan},
    {'metric': 'share_F_positive', 'value': float((out['F_kusd'] > 0).mean()) if len(out) else np.nan},
    {'metric': 'share_ret_1y_available', 'value': float(out['ret_1y_log'].notna().mean()) if len(out) else np.nan},
    {'metric': 'share_sigma_e_available', 'value': float(out['sigma_e_1y_ann'].notna().mean()) if len(out) else np.nan},
    {'metric': 'share_dd_naive_available', 'value': float(out['dd_naive_bs2008'].notna().mean()) if len(out) else np.nan},
    {'metric': 'median_dd_naive', 'value': float(out['dd_naive_bs2008'].median(skipna=True)) if out['dd_naive_bs2008'].notna().any() else np.nan},
    {'metric': 'median_pi_naive', 'value': float(out['pi_naive_bs2008'].median(skipna=True)) if out['pi_naive_bs2008'].notna().any() else np.nan},
])
summary.to_csv(OUT_SUMMARY, index=False)

print('Saved parquet:', OUT_DTD_PQ)
print('Saved csv:', OUT_DTD_CSV)
print('Saved summary:', OUT_SUMMARY)
summary
