#!/usr/bin/env python3
"""
Merge pipeline for:
1) DealScan-Compustat linked lender-level data
2) FDIC Call Reports (quarterly)
3) FDIC SOD HHI (annual -> forward-fill to quarterly)
4) Macro + Federal Funds Rate (quarterly) and JK monetary shocks (quarterly aggregate)

Key update (2026-02-09):
- External Keil/LCOID crosswalk methods can be toggled via `USE_EXTERNAL_CROSSWALKS`
- Preserve direct-ID/manual/name fallback matching methods
- Keep explicit flags for post-2016 extrapolated mappings (when crosswalks are enabled)

Outputs:
- Lender-level merged panel with lender matching flags
- Loan-borrower merged panel with lender coverage diagnostics
- Coverage summary CSVs + markdown report
"""

from __future__ import annotations
import os

import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


BASE = Path(os.environ.get("MPHIL_THESIS_ROOT", Path(__file__).resolve().parents[2]))


IN_LENDER_LEVEL = BASE / "data/intermediate/merged/dealscan_compustat_linked.parquet"
IN_CALL = BASE / "data/intermediate/clean/fdic_call_reports/fdic_call_reports_clean.parquet"
IN_SOD = BASE / "data/intermediate/clean/fdic_sod/fdic_sod_cleaned.parquet"
IN_MACRO = BASE / "data/raw/monetary_policy/fred_macro_quarterly_2001_2025.parquet"
IN_JK_MONTHLY = BASE / "data/raw/monetary_policy/jkshocks_update_fed_202401/shocks_fed_jk_m.csv"

# External crosswalks
IN_RSSD_LENDERID_CW = BASE / "data/intermediate/crosswalks/rssd_lenderid.csv"
IN_RSSD_ULTIMATEPARENT_CW = BASE / "data/intermediate/crosswalks/rssd_ultimateparentid.csv"
IN_LCOID_RSSDHCR_CW = BASE / "data/intermediate/crosswalks/lcoid_rssdhcr_match_external_oct2019.csv"
USE_EXTERNAL_CROSSWALKS = False
USE_DIRECT_ID_MATCH = False
# When True, restrict RSSD mapping to manual lender_id rules only
# (manual_top_lender + manual_top_lender_successor). This disables
# crosswalk, direct-ID, and name-based fallback matching.
USE_MANUAL_MATCH_ONLY = True

OUT_DIR = BASE / "data/intermediate/merged"
OUT_DOC_DIR = BASE / "data/documentation"

OUT_LENDER_PANEL = OUT_DIR / "dealscan_compustat_call_sod_macro_lenderlevel.parquet"
OUT_LOAN_PANEL = OUT_DIR / "dealscan_compustat_call_sod_macro_loan_borrower_panel.parquet"
OUT_LENDER_COVERAGE = OUT_DIR / "lender_bank_hhi_coverage_summary.csv"
OUT_LOAN_COVERAGE = OUT_DIR / "loan_lender_match_coverage_summary.csv"
OUT_YEAR_COVERAGE = OUT_DIR / "lender_bank_hhi_coverage_by_year.csv"
OUT_LENDER_MAP = OUT_DIR / "dealscan_lender_rssd_mapping.parquet"
OUT_REPORT = OUT_DOC_DIR / "FOUR_TABLE_MERGE_REPORT_2026-02-09.md"
OUT_DOC_REPORTS_DIR = OUT_DOC_DIR / "data_processing_reports"
OUT_LENDER_MATCH_REPORT = OUT_DOC_REPORTS_DIR / "lender_bank_matching_coverage_report.md"
OUT_DIAG_DIR = BASE / "regression" / "overview_diagnostics"
OUT_MANUAL_AUDIT_SUMMARY = OUT_DIAG_DIR / "manual_match_name_date_summary_2026-02-15.csv"
OUT_MANUAL_AUDIT_BY_LENDER = OUT_DIAG_DIR / "manual_match_name_date_by_lender_2026-02-15.csv"
OUT_MANUAL_AUDIT_SUSPECT = OUT_DIAG_DIR / "manual_match_name_date_suspect_pairs_2026-02-15.csv"


_SINGLE_TOKEN_STOP_WORDS = {
    "NA",
    "BANK",
    "BANCORP",
    "CORP",
    "CORPORATION",
    "INC",
    "LIMITED",
    "LTD",
    "LLC",
    "CO",
    "COMPANY",
    "THE",
    "TRUST",
    "HOLDINGS",
    "HOLDING",
    "SA",
    "PLC",
}
_MULTI_TOKEN_STOP_WORDS = {
    ("NATIONAL", "ASSOCIATION"),
    ("N", "A"),
    ("S", "A"),
}


def strict_normalize_name(name: object) -> str:
    """Strict name normalization: keep all tokens, remove punctuation only."""
    if pd.isna(name):
        return ""
    s = str(name).upper().strip()
    s = re.sub(r"[^A-Z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def clean_name(name: object) -> str:
    """Conservative token-based name cleaning for exact matching."""
    if pd.isna(name):
        return ""

    s = str(name).upper().strip()
    s = re.sub(r"[^A-Z0-9]+", " ", s)
    if not s:
        return ""

    tokens = s.split()
    kept: List[str] = []
    i = 0
    while i < len(tokens):
        if i + 1 < len(tokens) and (tokens[i], tokens[i + 1]) in _MULTI_TOKEN_STOP_WORDS:
            i += 2
            continue
        tok = tokens[i]
        if tok in _SINGLE_TOKEN_STOP_WORDS:
            i += 1
            continue
        kept.append(tok)
        i += 1

    return " ".join(kept)


# ---------------------------------------------------------------------------
# Manual lender_id -> RSSD mapping for top unmatched US banks
# Each entry verified against FDIC SOD / Call Reports bank names.
# ---------------------------------------------------------------------------
MANUAL_LENDER_RSSD: Dict[int, int] = {
    # --- JPMorgan Chase (main bank charter: RSSD 852218) ---
    14261: 852218,   # JP Morgan
    17665: 852218,   # JP Morgan Chase Bank NA
    46786: 852218,   # JP Morgan & Co
    14262: 852218,   # JP Morgan Chase & Co
    14263: 852218,   # JP Morgan Chase
    49436: 852218,   # JP Morgan Securities Inc
    24680: 852218,   # JP Morgan Chase Bank NA Toronto

    # --- Bank of America (main bank charter: RSSD 480228) ---
    15447: 480228,   # Bank of America
    46975: 480228,   # Bank of America NA
    15450: 480228,   # Bank of America Merrill Lynch
    62312: 480228,   # Bank of America Securities
    151226: 480228,  # BofA Securities
    51306: 480228,   # Banc of America Securities LLC
    82570: 480228,   # Banc of America Bridge LLC

    # --- Bank One legacy (pre-JPM merger) ---
    48180: 173333,   # BANK ONE Corp
    19300: 173333,   # Bank One NA
    46946: 173333,   # Bank One Corp (legacy id)

    # --- Wells Fargo (main bank charter: RSSD 451965) ---
    19499: 451965,   # Wells Fargo & Co
    27004: 451965,   # Wells Fargo Bank NA
    27003: 451965,   # Wells Fargo Bank
    46784: 451965,   # Wells Fargo Foothill Inc
    118702: 451965,  # Wells Fargo Securities LLC
    92489: 451965,   # Wells Fargo Retail Finance
    57425: 451965,   # Wells Fargo Capital Finance LLC

    # --- US Bank (main bank charter: RSSD 504713) ---
    32997: 504713,   # US Bank NA
    32998: 504713,   # US Bank National Association
    87883: 504713,   # US Bank Corporate Trust Services

    # --- Citibank / Citigroup (main bank charter: RSSD 476810) ---
    13703: 476810,   # Citigroup
    13704: 476810,   # Citi
    29606: 476810,   # Citibank
    29607: 476810,   # Citibank NA
    51189: 476810,   # Citicorp North America Inc
    60604: 476810,   # Citicorp USA Inc
    48798: 476810,   # Citigroup Global Markets Inc
    52699: 476810,   # Citibank/Salomon Smith Barney

    # --- Goldman Sachs (bank charter: RSSD 2182786) ---
    43219: 2182786,  # Goldman Sachs & Co
    49902: 2182786,  # Goldman Sachs Bank USA
    64313: 2182786,  # Goldman Sachs Credit Partners LP
    57167: 2182786,  # Goldman Sachs Lending Partners LLC
    26805: 2182786,  # Goldman Sachs Group Inc

    # --- Fifth Third Bank (main: RSSD 723112) ---
    29604: 723112,   # Fifth Third Bank
    85702: 723112,   # Fifth Third Bank of Columbus

    # --- SunTrust Bank (main: RSSD 675332) ---
    49246: 675332,   # SunTrust Bank
    52004: 675332,   # SunTrust Robinson Humphrey
    17673: 675332,   # SunTrust Banks Inc

    # --- Truist Bank (main: RSSD 852320) ---
    150321: 852320,  # Truist Financial
    33768: 852320,   # Branch Banking & Trust Co
    22047: 852320,   # BB&T Corp

    # --- PNC Bank (main: RSSD 817824) ---
    31840: 817824,   # PNC Bank
    31841: 817824,   # PNC Bank NA
    15279: 259518,   # National City Bank (legacy PNC acquisition target)

    # --- TD Bank (main: RSSD 497404) ---
    14846: 497404,   # Toronto Dominion Bank
    16412: 497404,   # TD Securities
    138579: 497404,  # Toronto-Dominion Bank
    93054: 497404,   # Toronto Dominion Bank [Texas]
    29598: 497404,   # Toronto Dominion New York
    50527: 497404,   # TD Banknorth NA
    50531: 497404,   # TD Bank NA

    # --- BMO (main US charter: RSSD 75633 = BMO Bank National Association) ---
    12677: 75633,    # Bank of Montreal
    18837: 75633,    # BMO Capital Markets Financing Inc
    101691: 75633,   # BMO Harris Bank NA
    104020: 75633,   # BMO Bank of Montreal
    54483: 75633,    # Bank of Montreal Chicago
    26798: 93039,    # Harris Trust & Savings Bank (legacy BMO charter)

    # --- Regions Bank (main: RSSD 233031) ---
    49928: 233031,   # Regions Bank
    101682: 606046,  # Cadence Bank NA
    52356: 917742,   # Associated Bank NA

    # --- Huntington (main: RSSD 12311) ---
    17131: 12311,    # Huntington Bank
    17130: 12311,    # Huntington National Bank
    17317: 12311,    # Huntington Bancshares

    # --- Citizens Bank (main: RSSD 617051 = Citizens Bank, NA) ---
    52640: 617051,   # Citizens Bank NA
    116754: 617051,  # Citizens Financial Group

    # --- KeyBank (main: RSSD 280110) ---
    26205: 280110,   # Key Bank NA
    26206: 280110,   # Keybank NA
    51838: 280110,   # KeyBank
    51639: 280110,   # Keybanc Capital Markets

    # --- Comerica Bank (main: RSSD 60143) ---
    31836: 60143,    # Comerica Bank
    31838: 60143,    # Comerica Bank NA

    # --- Wachovia / First Union legacy (main pre-merger operating charter: RSSD 484422) ---
    # Using 484422 materially improves 2001-2008 call/SOD continuity relative to 392620.
    45499: 484422,   # Wachovia Bank
    45500: 484422,   # Wachovia Bank NA
    92610: 484422,   # Wachovia Securities
    50062: 484422,   # First Union National Bank (renamed Wachovia)

    # --- Northern Trust (main: RSSD 210434) ---
    12940: 210434,   # Northern Trust
    12941: 210434,   # Northern Trust Corp
    118729: 210434,  # The Northern Trust Company
    34537: 210434,   # Northern Trust Co

    # --- Bank of New York Mellon (main: RSSD 541101) ---
    50202: 541101,   # Bank of New York
    50203: 541101,   # Bank of New York Mellon
    18074: 541101,   # Bank of New York Co Inc [BNY]
    26419: 934329,   # Mellon Bank

    # --- Morgan Stanley (bank: RSSD 1456501) ---
    17669: 1456501,  # Morgan Stanley
    92232: 1456501,  # Morgan Stanley Bank
    103799: 1456501, # Morgan Stanley Bank NA
    69677: 1456501,  # Morgan Stanley Senior Funding Inc
    21942: 1456501,  # Morgan Stanley Bank AG

    # --- Deutsche Bank Trust Company Americas (RSSD 214807) ---
    10669: 214807,   # Deutsche Bank AG
    11283: 214807,   # Deutsche Bank AG New York Branch
    54986: 214807,   # Deutsche Bank Trust Co Americas
    55155: 214807,   # Deutsche Bank Alex Brown
    11282: 214807,   # Deutsche Bank New York

    # --- HSBC Bank USA (RSSD 413208) ---
    23803: 413208,   # HSBC
    50505: 413208,   # HSBC Bank USA NA
    10903: 413208,   # HSBC Bank Plc
    39777: 413208,   # HSBC Banking Group
    61661: 413208,   # HSBC Business Credit (USA) Inc

    # --- Barclays (US charter: RSSD 2980209 = Barclays Bank Delaware) ---
    11130: 2980209,  # Barclays Bank Plc
    103475: 2980209, # Barclays
    17594: 2980209,  # Barclays Capital

    # --- Mizuho Bank USA (RSSD 229913) ---
    23894: 229913,   # Mizuho Bank Ltd
    36755: 229913,   # Mizuho Corporate Bank Ltd
    48055: 229913,   # Mizuho Financial Group Inc
    49338: 229913,   # Mizuho Corporate Bank USA

    # --- UBS Bank USA (RSSD 3212149) ---
    11126: 3212149,  # UBS AG
    53833: 3212149,  # UBS Loan Finance LLC
    22172: 3212149,  # UBS Securities LLC

    # --- MUFG Union Bank (RSSD 212465) ---
    116485: 212465,  # MUFG Union Bank NA
    138429: 212465,  # MUFG Bank Ltd
    46801: 212465,   # Bank of Tokyo-Mitsubishi UFJ Ltd [BTMU]
    46803: 212465,   # MUFG Bank Ltd [ex-Bank of Tokyo-Mitsubishi]
    49683: 212465,   # Bank of Tokyo-Mitsubishi UFJ Trust Co
    49513: 212465,   # Bank of Tokyo-Mitsubishi Group
    55889: 212465,   # Union Bank NA
    12181: 212465,   # Union Bank of California
    12182: 212465,   # Union Bank of California NA

    # --- CIBC (US charter: RSSD 1842065 = CIBC Bank USA) ---
    12303: 1842065,  # Canadian Imperial Bank of Commerce
    19017: 1842065,  # CIBC World Markets
    52408: 1842065,  # CIBC Inc
    12302: 1842065,  # CIBC [Canadian Imperial Bank of Commerce]
    105527: 1842065, # CIBC
    27230: 1842065,  # Canadian Imperial Bank of Commerce New York

    # --- Sumitomo Mitsui Trust Bank USA (RSSD 925411) ---
    31824: 925411,   # Sumitomo Mitsui Banking Corp
    103322: 925411,  # Sumitomo Mitsui Trust Bank Ltd

    # --- Zions Bank (RSSD 276579) ---
    72806: 276579,   # Zions Bank
    54162: 276579,   # Zions First National Bank

    # --- Santander Bank NA (RSSD 722777) ---
    111453: 3269590, # Santander Bank NA (Sovereign legacy charter)
    11671: 722777,   # Banco Santander SA
    62906: 722777,   # Banco Santander SA New York
    11670: 722777,   # Banco Santander SA [Ex-Santander Central Hispano]
    51547: 3269590,  # Sovereign Bank (legacy name / pre-Santander transition)

    # --- RBS Citizens NA (RSSD 617051 = Citizens Bank NA) ---
    51237: 617051,   # RBS Citizens NA

    # --- First Republic Bank (RSSD 131360) ---
    # Already matched if in call reports

    # --- Goldman Sachs additional ---
    52941: 2182786,  # Goldman Sachs Capital Partners

    # --- Compass / BBVA / PNC chain ---
    38705: 697633,   # Compass Bank (legacy BBVA USA charter)
    57073: 697633,   # BBVA Compass
    50489: 804963,   # Bank of the West (legacy; sold to BMO in 2023)

    # --- M&T Bank ---
    48436: 501105,   # M&T Bank
    24589: 501105,   # Manufacturers & Traders Trust Co

    # --- First Horizon legacy names ---
    29835: 485559,   # First Tennessee Bank
    47897: 485559,   # First Tennessee Bank NA
    73447: 485559,   # First Horizon Bank

    # --- Commerce Bank ---
    52309: 601050,   # Commerce Bank NA

    # --- Raymond James Bank ---
    73235: 2193616,  # Raymond James Bank FSB
    111253: 2193616, # Raymond James Bank NA

    # --- City National / BOKF / State Street / Synovus ---
    23674: 63069,    # City National Bank
    49942: 676656,   # Amegy Bank NA
    48050: 765505,   # Webster Bank NA (legacy Webster Five Cents charter)
    20992: 339858,   # Bank of Oklahoma (BOKF, NA)
    88345: 35301,    # State Street Bank
    103840: 395238,  # Synovus Bank

    # --- CIT Bank legacy chain ---
    15808: 2950677,  # CIT Bank Ltd
    # --- Capital One ---
    46962: 112837,   # Capital One Bank
    118531: 112837,  # Capital One NA
    # --- Silicon Valley Bank ---
    53329: 802866,   # Silicon Valley Bank
    # --- Legacy BofA chain ---
    45419: 455534,   # LaSalle Bank NA
    45540: 76201,    # Fleet National Bank
    50950: 1225800,  # Merrill Lynch Bank USA
}

# Year-window and successor overrides applied on top of call-data active windows.
# The base start/end years are inferred from Call Reports coverage of each RSSD.
MANUAL_LENDER_RULE_OVERRIDES: Dict[int, Dict[str, int]] = {
    # Legacy Bank One IDs should roll into JPM after 2004 merger.
    48180: {"end_year": 2004, "successor_rssd_id": 852218, "successor_start_year": 2005},
    19300: {"end_year": 2004, "successor_rssd_id": 852218, "successor_start_year": 2005},
    46946: {"end_year": 2004, "successor_rssd_id": 852218, "successor_start_year": 2005},
    # Wachovia entities should roll into Wells Fargo after the 2008 merger.
    45499: {"end_year": 2008, "successor_rssd_id": 451965, "successor_start_year": 2009},
    45500: {"end_year": 2008, "successor_rssd_id": 451965, "successor_start_year": 2009},
    92610: {"end_year": 2008, "successor_rssd_id": 451965, "successor_start_year": 2009},
    # SunTrust entities should roll into Truist after the BB&T/SunTrust merger.
    49246: {"end_year": 2019, "successor_rssd_id": 852320, "successor_start_year": 2020},
    52004: {"end_year": 2019, "successor_rssd_id": 852320, "successor_start_year": 2020},
    17673: {"end_year": 2019, "successor_rssd_id": 852320, "successor_start_year": 2020},
    # Compass/BBVA legacy should roll into PNC after BBVA USA acquisition.
    38705: {"end_year": 2021, "successor_rssd_id": 817824, "successor_start_year": 2022},
    57073: {"end_year": 2021, "successor_rssd_id": 817824, "successor_start_year": 2022},
    # Bank of the West sold to BMO in 2023.
    50489: {"end_year": 2022, "successor_rssd_id": 75633, "successor_start_year": 2023},
    # National City merged into PNC.
    15279: {"end_year": 2009, "successor_rssd_id": 817824, "successor_start_year": 2010},
    # LaSalle and Fleet legacy entities into BofA.
    45419: {"end_year": 2008, "successor_rssd_id": 480228, "successor_start_year": 2009},
    45540: {"end_year": 2005, "successor_rssd_id": 480228, "successor_start_year": 2006},
    # Merrill Lynch Bank USA rolled into BofA structure.
    50950: {"end_year": 2009, "successor_rssd_id": 480228, "successor_start_year": 2010},
    # Santander (Sovereign) charter transition.
    111453: {"end_year": 2011, "successor_rssd_id": 722777, "successor_start_year": 2012},
    # Webster charter transition to national association.
    48050: {"end_year": 2003, "successor_rssd_id": 761806, "successor_start_year": 2004},
    # Union Bank legacy should roll into U.S. Bank after MUFG sale completion.
    138429: {"end_year": 2023, "successor_rssd_id": 504713, "successor_start_year": 2024},
    55889: {"end_year": 2023, "successor_rssd_id": 504713, "successor_start_year": 2024},
    # Harris Trust folded into BMO operating charter.
    26798: {"end_year": 2002, "successor_rssd_id": 75633, "successor_start_year": 2003},
    # First Union renamed/merged through Wachovia to Wells Fargo.
    50062: {"end_year": 2008, "successor_rssd_id": 451965, "successor_start_year": 2009},
    # CIT Bank charter conversion.
    15808: {"end_year": 2015, "successor_rssd_id": 3918898, "successor_start_year": 2016},
    # Capital One charter migration.
    46962: {"end_year": 2022, "successor_rssd_id": 112837, "successor_start_year": 2023},
    118531: {"end_year": 2022, "successor_rssd_id": 112837, "successor_start_year": 2023},
}


# ---------------------------------------------------------------------------
# Lender classification: foreign_bank, non_bank_lender, or us_bank_unmatched
# Patterns applied to lender_name after all matching tiers fail.
# ---------------------------------------------------------------------------
_FOREIGN_BANK_PATTERNS: List[str] = [
    # Canadian banks (no US charter for parent)
    r"(?i)\broyal bank of canada\b",
    r"(?i)\brbc capital markets\b",
    r"(?i)\bbank of nova scotia\b",
    r"(?i)\bscotiabank\b",
    # French banks
    r"(?i)\bbnp paribas\b",
    r"(?i)\bsociete generale\b",
    r"(?i)\bcredit agricole\b",
    r"(?i)\bcalyon\b",
    r"(?i)\bnatixis\b",
    # Swiss banks
    r"(?i)\bcredit suisse\b",
    # UK banks
    r"(?i)\broyal bank of scotland\b",
    r"(?i)\bbank of scotland\b",
    r"(?i)\bbank of ireland\b",
    r"(?i)\ballied irish\b",
    r"(?i)\brbs (?!citizens)\b",
    r"(?i)\blloyds\b",
    r"(?i)\bstandard chartered\b",
    # German banks
    r"(?i)\bcommerzbank\b",
    # Dutch banks
    r"(?i)\babn amro\b",
    r"(?i)\brabobank\b",
    r"(?i)\bfortis\b",
    # Italian banks
    r"(?i)\bintesa sanpaolo\b",
    r"(?i)\bunicredit\b",
    # Spanish banks (for parent entities, not US Santander Bank)
    r"(?i)\bbanco santander.*(?:suisse|brasil|brazil|central hispano)\b",
    # Japanese banks (parent only; US subs already manually mapped)
    r"(?i)\bnomura\b",
    r"(?i)\bdaiwa\b",
    # Australian banks
    r"(?i)\bcommonwealth bank of australia\b",
    r"(?i)\bwestpac\b",
    r"(?i)\banz\b",
    r"(?i)\bnational australia bank\b",
    r"(?i)\bmacquarie\b",
    # Chinese banks (parent)
    r"(?i)\bbank of china\b(?!.*usa)",
    r"(?i)\bindustrial.*commercial.*bank.*china(?!.*usa)\b",
    r"(?i)\bchina construction bank\b",
    r"(?i)\bagricultural bank of china\b",
    r"(?i)\bchina merchants\b",
    # Korean banks
    r"(?i)\bkorea development bank\b",
    r"(?i)\bkeb hana\b",
    r"(?i)\bshinhan bank\b",
    r"(?i)\bwoori bank\b",
    # Other
    r"(?i)\bkbc bank\b",
    r"(?i)\bportigon\b",
    r"(?i)\bcredit lyonnais\b",
    r"(?i)\ble credit lyonnais\b",
    r"(?i)\bscotia capital\b",
    r"(?i)\bsumitomo bank\b",
    r"(?i)\bing bank\b",
    r"(?i)\bing capital\b",
    r"(?i)^ing$",
    r"(?i)\bdbs bank\b",
    r"(?i)\bocbc bank\b",
    r"(?i)\bnordea\b",
    r"(?i)\bseb\b",
    r"(?i)\bswedbank\b",
    r"(?i)\bdanske bank\b",
    r"(?i)\bfirst abu dhabi\b",
    r"(?i)\bqatar national bank\b",
    r"(?i)\bemirates nbd\b",
    r"(?i)\bbank leumi\b",
    r"(?i)\bhapoalim\b",
    r"(?i)\bcaixa\b",
    r"(?i)\bbilbao vizcaya\b",
    r"(?i)\bbbva\b(?!.*usa)",
]

_NON_BANK_PATTERNS: List[str] = [
    r"(?i)\bblackrock\b",
    r"(?i)\bapollo\b",
    r"(?i)\bkkr\b",
    r"(?i)\bcarlyle\b",
    r"(?i)\bares management\b",
    r"(?i)\bgoldman sachs asset management\b",
    r"(?i)\bneuberger berman\b",
    r"(?i)\blazard\b",
    r"(?i)\bprudential\b(?!.*bank)",
    r"(?i)\bmetlife\b(?!.*bank)",
    r"(?i)\baig\b",
    r"(?i)\ballstate\b(?!.*bank)",
    r"(?i)\bgeneral electric capital\b",
    r"(?i)\bge capital\b",
    r"(?i)\bcit group\b",
    r"(?i)\bheller financial\b",
    r"(?i)\bamerican express\b(?!.*bank)",
    r"(?i)\bfidelity\b(?!.*bank)",
    r"(?i)\bvanguard\b",
    r"(?i)\boaktree\b",
    r"(?i)\bgolub capital\b",
    r"(?i)\bblue mountain\b",
    r"(?i)\bantares capital\b",
    r"(?i)\bantares holdings\b",
    r"(?i)\bchurchill asset\b",
    r"(?i)\bbridgepoint\b",
    r"(?i)\bingalls\b",
    r"(?i)\bnxt capital\b",
    r"(?i)\bares capital\b",
    r"(?i)\bsiemens financial services\b",
    r"(?i)\blehman brothers\b",
    r"(?i)\bmerrill lynch capital\b",
    r"(?i)\bjefferies finance\b",
    r"(?i)\bjefferies\b",
    r"(?i)\bcobank\b",
    r"(?i)\bfarm credit\b",
    r"(?i)\bmadison capital\b",
    r"(?i)\bnewstar financial\b",
    r"(?i)\bmonroe capital\b",
    r"(?i)\btcw\b",
    r"(?i)\bwhite oak\b",
    r"(?i)\bbain capital\b",
    r"(?i)\bvaragon\b",
    r"(?i)\bowl rock\b",
]


def classify_unmatched_lender(name: str) -> str:
    """Return 'foreign_bank', 'non_bank_lender', or 'us_bank_unmatched'."""
    if pd.isna(name) or name == "":
        return "us_bank_unmatched"
    for pat in _FOREIGN_BANK_PATTERNS:
        if re.search(pat, name):
            return "foreign_bank"
    for pat in _NON_BANK_PATTERNS:
        if re.search(pat, name):
            return "non_bank_lender"
    return "us_bank_unmatched"


def to_int_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype("Int64")


def safe_ratio(num: float, den: float) -> float:
    if den == 0:
        return np.nan
    return float(num) / float(den)


def build_call_rssd_year_window(call_df: pd.DataFrame) -> pd.DataFrame:
    """Infer active year/date windows for each RSSD in Call Reports."""
    c = call_df[["RSSDID", "REPDTE"]].copy()
    c["RSSDID"] = to_int_series(c["RSSDID"])
    c["REPDTE"] = pd.to_datetime(c["REPDTE"], errors="coerce")
    c["year"] = c["REPDTE"].dt.year.astype("Int64")
    c = c.dropna(subset=["RSSDID", "year"])

    # Use DataFrameGroupBy aggregation for compatibility with older pandas.
    win = (
        c.groupby("RSSDID", as_index=False)
        .agg(
            call_start_year=("year", "min"),
            call_end_year=("year", "max"),
            call_start_repdte=("REPDTE", "min"),
            call_end_repdte=("REPDTE", "max"),
        )
        .rename(columns={"RSSDID": "rssd_id"})
    )
    win["rssd_id"] = to_int_series(win["rssd_id"])
    win["call_start_year"] = to_int_series(win["call_start_year"])
    win["call_end_year"] = to_int_series(win["call_end_year"])
    win["call_start_repdte"] = pd.to_datetime(win["call_start_repdte"], errors="coerce")
    win["call_end_repdte"] = pd.to_datetime(win["call_end_repdte"], errors="coerce")
    return win


def _within_year_window(year: pd.Series, start: pd.Series, end: pd.Series) -> pd.Series:
    """Return True where year lies in [start, end] with open bounds allowed."""
    ok = year.notna()
    ok &= start.isna() | (year >= start)
    ok &= end.isna() | (year <= end)
    return ok


def _within_repdte_window(
    deal_date: pd.Series,
    start_repdte: pd.Series,
    end_repdte: pd.Series,
    lead_days: int = 120,
    lag_days: int = 370,
) -> pd.Series:
    """
    Return True where deal_date falls in [start_repdte-lead_days, end_repdte+lag_days].
    Open bounds are allowed when start/end dates are missing.
    """
    ok = deal_date.notna()
    lower = start_repdte - pd.to_timedelta(lead_days, unit="D")
    upper = end_repdte + pd.to_timedelta(lag_days, unit="D")
    ok &= start_repdte.isna() | (deal_date >= lower)
    ok &= end_repdte.isna() | (deal_date <= upper)
    return ok


def build_manual_lender_rules(call_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build manual lender rules with explicit year windows and optional successor RSSD.

    Base windows come from Call Reports active years for each RSSD; explicit overrides
    can tighten windows and/or define successor mappings post-merger.
    """
    rules = pd.DataFrame(
        {
            "lender_id_num": list(MANUAL_LENDER_RSSD.keys()),
            "manual_rssd_id": list(MANUAL_LENDER_RSSD.values()),
        }
    )
    rules["lender_id_num"] = to_int_series(rules["lender_id_num"])
    rules["manual_rssd_id"] = to_int_series(rules["manual_rssd_id"])

    call_win = build_call_rssd_year_window(call_df)
    base_win = call_win.rename(
        columns={
            "rssd_id": "manual_rssd_id",
            "call_start_year": "start_year",
            "call_end_year": "end_year",
            "call_start_repdte": "start_repdte",
            "call_end_repdte": "end_repdte",
        }
    )
    rules = rules.merge(base_win, on="manual_rssd_id", how="left")

    rules["successor_rssd_id"] = pd.Series([pd.NA] * len(rules), dtype="Int64")
    rules["successor_start_year"] = pd.Series([pd.NA] * len(rules), dtype="Int64")
    rules["successor_end_year"] = pd.Series([pd.NA] * len(rules), dtype="Int64")
    rules["successor_start_repdte"] = pd.NaT
    rules["successor_end_repdte"] = pd.NaT

    for lender_id, params in MANUAL_LENDER_RULE_OVERRIDES.items():
        hit = rules["lender_id_num"] == lender_id
        if not hit.any():
            continue
        if "start_year" in params:
            rules.loc[hit, "start_year"] = params["start_year"]
        if "end_year" in params:
            rules.loc[hit, "end_year"] = params["end_year"]
        if "successor_rssd_id" in params:
            rules.loc[hit, "successor_rssd_id"] = params["successor_rssd_id"]
        if "successor_start_year" in params:
            rules.loc[hit, "successor_start_year"] = params["successor_start_year"]
        if "successor_end_year" in params:
            rules.loc[hit, "successor_end_year"] = params["successor_end_year"]

    succ_win = call_win.rename(
        columns={
            "rssd_id": "successor_rssd_id",
            "call_start_year": "succ_start_auto",
            "call_end_year": "succ_end_auto",
            "call_start_repdte": "succ_start_repdte_auto",
            "call_end_repdte": "succ_end_repdte_auto",
        }
    )
    rules = rules.merge(succ_win, on="successor_rssd_id", how="left")
    rules["successor_start_year"] = rules["successor_start_year"].fillna(rules["succ_start_auto"])
    rules["successor_end_year"] = rules["successor_end_year"].fillna(rules["succ_end_auto"])
    rules["successor_start_repdte"] = rules["successor_start_repdte"].fillna(rules["succ_start_repdte_auto"])
    rules["successor_end_repdte"] = rules["successor_end_repdte"].fillna(rules["succ_end_repdte_auto"])
    rules = rules.drop(columns=["succ_start_auto", "succ_end_auto", "succ_start_repdte_auto", "succ_end_repdte_auto"])

    for c in [
        "lender_id_num",
        "manual_rssd_id",
        "start_year",
        "end_year",
        "successor_rssd_id",
        "successor_start_year",
        "successor_end_year",
    ]:
        rules[c] = to_int_series(rules[c])
    rules["start_repdte"] = pd.to_datetime(rules["start_repdte"], errors="coerce")
    rules["end_repdte"] = pd.to_datetime(rules["end_repdte"], errors="coerce")
    rules["successor_start_repdte"] = pd.to_datetime(rules["successor_start_repdte"], errors="coerce")
    rules["successor_end_repdte"] = pd.to_datetime(rules["successor_end_repdte"], errors="coerce")
    return rules


def detect_col(cols: Iterable[str], patterns: Iterable[str]) -> Optional[str]:
    lower = {c.lower(): c for c in cols}
    for p in patterns:
        for lc, orig in lower.items():
            if p in lc:
                return orig
    return None


def load_keil_crosswalk(path: Path, id_patterns: Iterable[str], id_out_col: str) -> pd.DataFrame:
    """
    Load and standardize Keil crosswalk files:
    - rssd_lenderid.csv (lenderid + year -> rssd)
    - rssd_ultimateparentid.csv (ultimateparentid + year -> rssd)

    Keep one best row per (id, year), prioritizing:
    best_match, callReports_sample, summaryOfDeposits_sample, federalReserve_sample.
    """
    if not path.exists():
        return pd.DataFrame(columns=[id_out_col, "year", "rssd_id"])

    cw = pd.read_csv(path)

    id_col = detect_col(cw.columns, id_patterns)
    year_col = detect_col(cw.columns, ["year"])
    rssd_col = detect_col(cw.columns, ["rssd", "rssd9001"])

    if id_col is None or year_col is None or rssd_col is None:
        return pd.DataFrame(columns=[id_out_col, "year", "rssd_id"])

    keep_cols = [id_col, year_col, rssd_col]
    for c in ["best_match", "callReports_sample", "summaryOfDeposits_sample", "federalReserve_sample"]:
        if c in cw.columns:
            keep_cols.append(c)

    out = cw[keep_cols].copy()
    out = out.rename(columns={id_col: id_out_col, year_col: "year", rssd_col: "rssd_id"})

    out[id_out_col] = to_int_series(out[id_out_col])
    out["year"] = to_int_series(out["year"])
    out["rssd_id"] = to_int_series(out["rssd_id"])

    for c in ["best_match", "callReports_sample", "summaryOfDeposits_sample", "federalReserve_sample"]:
        if c not in out.columns:
            out[c] = 0
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0).astype(int)

    out = out.dropna(subset=[id_out_col, "year", "rssd_id"])
    out = out.sort_values(
        [id_out_col, "year", "best_match", "callReports_sample", "summaryOfDeposits_sample", "federalReserve_sample", "rssd_id"],
        ascending=[True, True, False, False, False, False, True],
    )
    out = out.drop_duplicates([id_out_col, "year"], keep="first")
    return out[[id_out_col, "year", "rssd_id"]].copy()


def load_mackinlay_lcoid_crosswalk(path: Path) -> pd.DataFrame:
    """Load lcoid -> rssd9348 crosswalk with start/end year ranges."""
    if not path.exists():
        return pd.DataFrame(columns=["lcoid", "rssd_id", "start_year", "end_year"])

    cw = pd.read_csv(path)
    required = ["lcoid", "rssd9348", "rssd9348_startyear", "rssd9348_endyear"]
    if not set(required).issubset(set(cw.columns)):
        return pd.DataFrame(columns=["lcoid", "rssd_id", "start_year", "end_year"])

    out = cw[required].copy()
    out.columns = ["lcoid", "rssd_id", "start_year", "end_year"]
    out["lcoid"] = to_int_series(out["lcoid"])
    out["rssd_id"] = to_int_series(out["rssd_id"])
    out["start_year"] = to_int_series(out["start_year"])
    out["end_year"] = to_int_series(out["end_year"])
    out = out.dropna(subset=["lcoid", "rssd_id", "start_year", "end_year"])
    out = out.sort_values(["lcoid", "start_year", "end_year", "rssd_id"], ascending=[True, True, True, True])
    return out


def _merge_candidate_exact(
    rows: pd.DataFrame,
    left_id_col: str,
    left_year_col: str,
    cw: pd.DataFrame,
    cw_id_col: str,
) -> pd.DataFrame:
    """Build candidate mapping by exact (id, year)."""
    if cw.empty:
        return pd.DataFrame(columns=["ds_row_id", "rssd_id"])

    m = rows[["ds_row_id", left_id_col, left_year_col]].merge(
        cw[[cw_id_col, "year", "rssd_id"]],
        left_on=[left_id_col, left_year_col],
        right_on=[cw_id_col, "year"],
        how="left",
    )
    m = m[["ds_row_id", "rssd_id"]].dropna(subset=["rssd_id"]).drop_duplicates("ds_row_id")
    return m


def _merge_candidate_postmax(
    rows: pd.DataFrame,
    left_id_col: str,
    left_year_col: str,
    cw: pd.DataFrame,
    cw_id_col: str,
) -> pd.DataFrame:
    """For years above crosswalk max year, carry forward the latest known mapping per id."""
    if cw.empty:
        return pd.DataFrame(columns=["ds_row_id", "rssd_id"])

    max_year = int(cw["year"].max())
    last = (
        cw.sort_values([cw_id_col, "year", "rssd_id"])
        .groupby(cw_id_col, as_index=False)
        .tail(1)[[cw_id_col, "rssd_id"]]
        .drop_duplicates(cw_id_col)
    )

    m = rows[["ds_row_id", left_id_col, left_year_col]].merge(
        last,
        left_on=left_id_col,
        right_on=cw_id_col,
        how="left",
    )
    m = m[m[left_year_col] > max_year]
    m = m[["ds_row_id", "rssd_id"]].dropna(subset=["rssd_id"]).drop_duplicates("ds_row_id")
    return m


def _merge_candidate_range(
    rows: pd.DataFrame,
    left_id_col: str,
    left_year_col: str,
    cw_range: pd.DataFrame,
) -> pd.DataFrame:
    """Build candidate mapping by (id, start_year<=year<=end_year)."""
    if cw_range.empty:
        return pd.DataFrame(columns=["ds_row_id", "rssd_id"])

    m = rows[["ds_row_id", left_id_col, left_year_col]].merge(
        cw_range[["lcoid", "rssd_id", "start_year", "end_year"]],
        left_on=left_id_col,
        right_on="lcoid",
        how="left",
    )
    m = m[(m[left_year_col] >= m["start_year"]) & (m[left_year_col] <= m["end_year"])]
    m = m.sort_values(["ds_row_id", "end_year", "start_year", "rssd_id"], ascending=[True, False, False, True])
    m = m[["ds_row_id", "rssd_id"]].dropna(subset=["rssd_id"]).drop_duplicates("ds_row_id")
    return m


def _merge_candidate_range_postmax(
    rows: pd.DataFrame,
    left_id_col: str,
    left_year_col: str,
    cw_range: pd.DataFrame,
) -> pd.DataFrame:
    """For years above range max, carry forward the latest range mapping per id."""
    if cw_range.empty:
        return pd.DataFrame(columns=["ds_row_id", "rssd_id"])

    max_year = int(cw_range["end_year"].max())
    last = (
        cw_range.sort_values(["lcoid", "end_year", "start_year", "rssd_id"])
        .groupby("lcoid", as_index=False)
        .tail(1)[["lcoid", "rssd_id"]]
        .drop_duplicates("lcoid")
    )

    m = rows[["ds_row_id", left_id_col, left_year_col]].merge(
        last,
        left_on=left_id_col,
        right_on="lcoid",
        how="left",
    )
    m = m[m[left_year_col] > max_year]
    m = m[["ds_row_id", "rssd_id"]].dropna(subset=["rssd_id"]).drop_duplicates("ds_row_id")
    return m


def build_lender_rssd_map(df_lender: pd.DataFrame, call_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build row-level lender->RSSD mapping following crosswalk documentation:
    1) parentid + year
    2) lenderid + year
    3) post-2016 carry-forward for parentid/lenderid
    plus conservative fallback methods.
    """
    rows = df_lender[["ds_row_id", "lender_id", "lender_parent_id", "lender_name", "deal_date"]].copy()
    rows["deal_date"] = pd.to_datetime(rows["deal_date"], errors="coerce")
    rows["deal_year"] = rows["deal_date"].dt.year.astype("Int64")
    rows["lender_id_num"] = to_int_series(rows["lender_id"])
    rows["lender_parent_id_num"] = to_int_series(rows["lender_parent_id"])

    map_df = rows[["ds_row_id", "lender_id", "lender_parent_id", "lender_name", "deal_year", "lender_id_num"]].copy()
    map_df["mapped_rssd_id"] = pd.Series([pd.NA] * len(map_df), dtype="Int64")
    map_df["lender_rssd_match_method"] = "unmatched"
    map_df["flag_rssd_extrapolated_post2016"] = 0

    # Build Call Report bank universe first.
    call_name = call_df[["RSSDID", "RSSD9017", "REPDTE"]].dropna(subset=["RSSDID", "RSSD9017"]).copy()
    call_name["RSSDID"] = to_int_series(call_name["RSSDID"])
    call_name = call_name.dropna(subset=["RSSDID"]).sort_values(["RSSDID", "REPDTE"])
    call_latest = call_name.groupby("RSSDID").tail(1)[["RSSDID", "RSSD9017"]].copy()
    call_latest["name_clean"] = call_latest["RSSD9017"].map(clean_name)
    call_latest["name_strict"] = call_latest["RSSD9017"].map(strict_normalize_name)
    rssd_set = set(call_latest["RSSDID"].dropna().astype("int64").unique())

    # Resolve effective matching regime.
    use_external = bool(USE_EXTERNAL_CROSSWALKS and (not USE_MANUAL_MATCH_ONLY))
    use_direct = bool(USE_DIRECT_ID_MATCH and (not USE_MANUAL_MATCH_ONLY))

    # Load crosswalks only when enabled in the effective regime.
    if use_external:
        cw_parent = load_keil_crosswalk(IN_RSSD_ULTIMATEPARENT_CW, ["ultimateparentid"], "id_num")
        cw_lender = load_keil_crosswalk(IN_RSSD_LENDERID_CW, ["lenderid", "lender_id"], "id_num")
        cw_lcoid = load_mackinlay_lcoid_crosswalk(IN_LCOID_RSSDHCR_CW)
    else:
        cw_parent = pd.DataFrame(columns=["id_num", "year", "rssd_id"])
        cw_lender = pd.DataFrame(columns=["id_num", "year", "rssd_id"])
        cw_lcoid = pd.DataFrame(columns=["lcoid", "rssd_id", "start_year", "end_year"])
    manual_rules = build_manual_lender_rules(call_df)

    if not manual_rules.empty:
        manual_meta = manual_rules.rename(
            columns={
                "start_year": "manual_rule_start_year",
                "end_year": "manual_rule_end_year",
                "manual_rssd_id": "manual_rule_primary_rssd_id",
                "successor_rssd_id": "manual_rule_successor_rssd_id",
                "successor_start_year": "manual_rule_successor_start_year",
                "successor_end_year": "manual_rule_successor_end_year",
            }
        )
        map_df = map_df.merge(
            manual_meta[
                [
                    "lender_id_num",
                    "manual_rule_start_year",
                    "manual_rule_end_year",
                    "manual_rule_primary_rssd_id",
                    "manual_rule_successor_rssd_id",
                    "manual_rule_successor_start_year",
                    "manual_rule_successor_end_year",
                    "start_repdte",
                    "end_repdte",
                    "successor_start_repdte",
                    "successor_end_repdte",
                ]
            ],
            on="lender_id_num",
            how="left",
        )
        map_df = map_df.rename(
            columns={
                "start_repdte": "manual_rule_start_repdte",
                "end_repdte": "manual_rule_end_repdte",
                "successor_start_repdte": "manual_rule_successor_start_repdte",
                "successor_end_repdte": "manual_rule_successor_end_repdte",
            }
        )

    def apply_candidate(cand: pd.DataFrame, method: str, extrapolated: bool = False) -> None:
        nonlocal map_df
        if cand.empty:
            return
        # Keep only mappings that can be linked to current Call Reports bank universe.
        cand = cand[cand["rssd_id"].isin(rssd_set)].copy()
        if cand.empty:
            return
        cname = f"cand_{method}"
        m = cand.rename(columns={"rssd_id": cname})
        map_df = map_df.merge(m[["ds_row_id", cname]], on="ds_row_id", how="left")
        hit = map_df["mapped_rssd_id"].isna() & map_df[cname].notna()
        map_df.loc[hit, "mapped_rssd_id"] = map_df.loc[hit, cname].astype("Int64")
        map_df.loc[hit, "lender_rssd_match_method"] = method
        if extrapolated:
            map_df.loc[hit, "flag_rssd_extrapolated_post2016"] = 1
        map_df = map_df.drop(columns=[cname])

    if use_external:
        # Crosswalk methods in documented order.
        apply_candidate(
            _merge_candidate_exact(rows, "lender_parent_id_num", "deal_year", cw_parent, "id_num"),
            "keil_parentid_year",
        )
        apply_candidate(
            _merge_candidate_exact(rows, "lender_id_num", "deal_year", cw_lender, "id_num"),
            "keil_lenderid_year",
        )
        apply_candidate(
            _merge_candidate_postmax(rows, "lender_parent_id_num", "deal_year", cw_parent, "id_num"),
            "keil_parentid_post2016_carry",
            extrapolated=True,
        )
        apply_candidate(
            _merge_candidate_postmax(rows, "lender_id_num", "deal_year", cw_lender, "id_num"),
            "keil_lenderid_post2016_carry",
            extrapolated=True,
        )

        # Additional external range crosswalk (optional supplement).
        apply_candidate(
            _merge_candidate_range(rows, "lender_parent_id_num", "deal_year", cw_lcoid),
            "mackinlay_lcoid_range_parent",
        )
        apply_candidate(
            _merge_candidate_range(rows, "lender_id_num", "deal_year", cw_lcoid),
            "mackinlay_lcoid_range_lender",
        )
        apply_candidate(
            _merge_candidate_range_postmax(rows, "lender_parent_id_num", "deal_year", cw_lcoid),
            "mackinlay_lcoid_post2014_carry_parent",
            extrapolated=True,
        )
        apply_candidate(
            _merge_candidate_range_postmax(rows, "lender_id_num", "deal_year", cw_lcoid),
            "mackinlay_lcoid_post2014_carry_lender",
            extrapolated=True,
        )

    # Optional direct numeric ID matching.
    if use_direct:
        parent_direct = rows[rows["lender_parent_id_num"].isin(rssd_set)][["ds_row_id", "lender_parent_id_num"]].copy()
        parent_direct = parent_direct.rename(columns={"lender_parent_id_num": "rssd_id"})
        apply_candidate(parent_direct, "parent_id_direct_rssd")

        lender_direct = rows[rows["lender_id_num"].isin(rssd_set)][["ds_row_id", "lender_id_num"]].copy()
        lender_direct = lender_direct.rename(columns={"lender_id_num": "rssd_id"})
        apply_candidate(lender_direct, "lender_id_direct_rssd")

    # Manual mapping with year windows and optional successor charter after mergers.
    if not manual_rules.empty:
        mj = rows[["ds_row_id", "lender_id_num", "deal_year", "deal_date"]].merge(
            manual_rules, on="lender_id_num", how="inner"
        )
        in_primary = _within_year_window(mj["deal_year"], mj["start_year"], mj["end_year"]) & _within_repdte_window(
            mj["deal_date"], mj["start_repdte"], mj["end_repdte"]
        )
        primary = mj[in_primary][["ds_row_id", "manual_rssd_id"]].rename(columns={"manual_rssd_id": "rssd_id"})
        primary = primary.dropna(subset=["rssd_id"]).drop_duplicates("ds_row_id")
        apply_candidate(primary, "manual_top_lender")

        in_successor = (
            (~in_primary)
            & mj["successor_rssd_id"].notna()
            & _within_year_window(mj["deal_year"], mj["successor_start_year"], mj["successor_end_year"])
            & _within_repdte_window(mj["deal_date"], mj["successor_start_repdte"], mj["successor_end_repdte"])
        )
        succ = mj[in_successor][["ds_row_id", "successor_rssd_id"]].rename(columns={"successor_rssd_id": "rssd_id"})
        succ = succ.dropna(subset=["rssd_id"]).drop_duplicates("ds_row_id")
        apply_candidate(succ, "manual_top_lender_successor")

    # -----------------------------------------------------------------------
    # FIX (2026-02-24): lender_parent_id fallback for manual rules.
    # For still-unmatched rows where lender_parent_id != lender_id, try
    # looking up lender_parent_id in the manual rules. This recovers ~21k
    # rows from major banks whose subsidiaries (lender_id) were not in
    # MANUAL_LENDER_RSSD but whose parent (lender_parent_id) is.
    # -----------------------------------------------------------------------
    if not manual_rules.empty:
        parent_unmatched = map_df["mapped_rssd_id"].isna()
        parent_differs = rows["lender_parent_id_num"] != rows["lender_id_num"]
        parent_rows = rows[parent_unmatched.values & parent_differs.values].copy()
        if len(parent_rows) > 0:
            mj_p = parent_rows[["ds_row_id", "lender_parent_id_num", "deal_year", "deal_date"]].merge(
                manual_rules.rename(columns={"lender_id_num": "lender_parent_id_num"}),
                on="lender_parent_id_num",
                how="inner",
            )
            if len(mj_p) > 0:
                in_primary_p = _within_year_window(
                    mj_p["deal_year"], mj_p["start_year"], mj_p["end_year"]
                ) & _within_repdte_window(
                    mj_p["deal_date"], mj_p["start_repdte"], mj_p["end_repdte"]
                )
                primary_p = mj_p[in_primary_p][["ds_row_id", "manual_rssd_id"]].rename(
                    columns={"manual_rssd_id": "rssd_id"}
                )
                primary_p = primary_p.dropna(subset=["rssd_id"]).drop_duplicates("ds_row_id")
                apply_candidate(primary_p, "manual_top_lender_parent")

                in_succ_p = (
                    (~in_primary_p)
                    & mj_p["successor_rssd_id"].notna()
                    & _within_year_window(
                        mj_p["deal_year"], mj_p["successor_start_year"], mj_p["successor_end_year"]
                    )
                    & _within_repdte_window(
                        mj_p["deal_date"], mj_p["successor_start_repdte"], mj_p["successor_end_repdte"]
                    )
                )
                succ_p = mj_p[in_succ_p][["ds_row_id", "successor_rssd_id"]].rename(
                    columns={"successor_rssd_id": "rssd_id"}
                )
                succ_p = succ_p.dropna(subset=["rssd_id"]).drop_duplicates("ds_row_id")
                apply_candidate(succ_p, "manual_top_lender_parent_successor")

    if not USE_MANUAL_MATCH_ONLY:
        # Conservative strict-name exact match (unique only).
        n_by_strict = call_latest.groupby("name_strict")["RSSDID"].nunique().reset_index(name="n")
        uniq_strict = set(n_by_strict[n_by_strict["n"] == 1]["name_strict"])
        strict_name_map = (
            call_latest[call_latest["name_strict"].isin(uniq_strict)][["name_strict", "RSSDID"]]
            .drop_duplicates("name_strict")
            .rename(columns={"RSSDID": "rssd_id"})
        )

        nm_strict = rows[["ds_row_id", "lender_name"]].copy()
        nm_strict["name_strict"] = nm_strict["lender_name"].map(strict_normalize_name)
        nm_strict = nm_strict.merge(strict_name_map, on="name_strict", how="left")
        nm_strict = nm_strict[["ds_row_id", "rssd_id"]].dropna(subset=["rssd_id"]).drop_duplicates("ds_row_id")
        apply_candidate(nm_strict, "lender_name_exact_strict_unique")

        # Conservative cleaned-name exact match (unique only).
        n_by_clean = call_latest.groupby("name_clean")["RSSDID"].nunique().reset_index(name="n")
        uniq_clean = set(n_by_clean[n_by_clean["n"] == 1]["name_clean"])
        name_map = (
            call_latest[call_latest["name_clean"].isin(uniq_clean)][["name_clean", "RSSDID"]]
            .drop_duplicates("name_clean")
            .rename(columns={"RSSDID": "rssd_id"})
        )

        nm = rows[["ds_row_id", "lender_name"]].copy()
        nm["name_clean"] = nm["lender_name"].map(clean_name)
        nm = nm.merge(name_map, on="name_clean", how="left")
        nm = nm[["ds_row_id", "rssd_id"]].dropna(subset=["rssd_id"]).drop_duplicates("ds_row_id")
        apply_candidate(nm, "lender_name_exact_clean_unique")

    # --- Classify remaining unmatched lenders ---
    still_unmatched = map_df["lender_rssd_match_method"] == "unmatched"
    map_df.loc[still_unmatched, "lender_category"] = map_df.loc[still_unmatched, "lender_name"].apply(
        classify_unmatched_lender
    )
    map_df.loc[~still_unmatched, "lender_category"] = "matched"

    map_df["flag_lender_rssd_mapped"] = map_df["mapped_rssd_id"].notna().astype(int)
    return map_df


def build_bank_quarter_table(call_df: pd.DataFrame, sod_df: pd.DataFrame) -> pd.DataFrame:
    call_vars = [
        "total_assets_mil",
        "total_deposits_mil",
        "equity_capital_mil",
        "total_loans_mil",
        "deposits_to_assets",
        "capital_ratio",
        "loans_to_assets",
        "flag_extreme_growth",
        "flag_negative_equity",
    ]
    call_keep = ["RSSDID", "REPDTE", "year", "quarter", "year_qtr"] + [c for c in call_vars if c in call_df.columns]
    call_q = call_df[call_keep].copy()
    call_q["RSSDID"] = to_int_series(call_q["RSSDID"])
    call_q["REPDTE"] = pd.to_datetime(call_q["REPDTE"], errors="coerce")
    call_q = call_q.dropna(subset=["RSSDID", "REPDTE"]).drop_duplicates(["RSSDID", "REPDTE"])

    rename_call = {c: f"call_{c}" for c in call_q.columns if c not in ["RSSDID", "REPDTE", "year", "quarter", "year_qtr"]}
    call_q = call_q.rename(columns=rename_call)

    sod_keep = ["RSSDID", "YEAR", "hhi_bank", "bank_total_deposits_mil", "BKCLASS", "CERT"]
    sod = sod_df[[c for c in sod_keep if c in sod_df.columns]].copy()
    sod["RSSDID"] = to_int_series(sod["RSSDID"])
    sod["YEAR"] = pd.to_numeric(sod["YEAR"], errors="coerce").astype("Int64")
    if "CERT" in sod.columns:
        sod["CERT"] = to_int_series(sod["CERT"])
    sod = sod.dropna(subset=["RSSDID", "YEAR"]).drop_duplicates(["RSSDID", "YEAR"])

    bank_q = call_q.merge(sod, left_on=["RSSDID", "year"], right_on=["RSSDID", "YEAR"], how="left")

    bank_q["hhi_bank_sameyear"] = bank_q["hhi_bank"]

    ff_cols = [c for c in ["hhi_bank", "bank_total_deposits_mil", "BKCLASS", "CERT"] if c in bank_q.columns]

    # Fallback only when same-year SOD is missing: use nearest prior annual SOD
    # observation within 360 days to avoid long-range carries.
    sod_asof = sod.copy()
    sod_asof["sod_asof_date"] = pd.to_datetime(sod_asof["YEAR"].astype(str) + "-06-30", errors="coerce")
    sod_asof = sod_asof.dropna(subset=["RSSDID", "sod_asof_date"]).sort_values(["sod_asof_date", "RSSDID"])
    sod_asof = sod_asof.rename(columns={c: f"{c}_asof" for c in ff_cols})

    left = bank_q[["RSSDID", "REPDTE"]].dropna(subset=["RSSDID", "REPDTE"]).copy()
    left = left.sort_values(["REPDTE", "RSSDID"]).reset_index(drop=True)
    right_cols = ["RSSDID", "sod_asof_date"] + [f"{c}_asof" for c in ff_cols]
    right = sod_asof[right_cols].sort_values(["sod_asof_date", "RSSDID"]).reset_index(drop=True)

    asof = pd.merge_asof(
        left,
        right,
        left_on="REPDTE",
        right_on="sod_asof_date",
        left_by="RSSDID",
        right_by="RSSDID",
        direction="backward",
        tolerance=pd.Timedelta("360 days"),
    )
    bank_q = bank_q.merge(asof, on=["RSSDID", "REPDTE"], how="left")

    for c in ff_cols:
        bank_q[f"{c}_q"] = bank_q[c]
        fallback_col = f"{c}_asof"
        if fallback_col in bank_q.columns:
            bank_q[f"{c}_q"] = bank_q[f"{c}_q"].fillna(bank_q[fallback_col])

    bank_q["flag_sod_sameyear"] = bank_q["hhi_bank_sameyear"].notna().astype(int)
    bank_q["flag_sod_ffill_used"] = ((bank_q["hhi_bank_sameyear"].isna()) & (bank_q["hhi_bank_q"].notna())).astype(int)
    bank_q["sod_ffill_days"] = pd.NA
    used_ff = bank_q["flag_sod_ffill_used"] == 1
    if "sod_asof_date" in bank_q.columns:
        bank_q.loc[used_ff, "sod_ffill_days"] = (
            bank_q.loc[used_ff, "REPDTE"] - bank_q.loc[used_ff, "sod_asof_date"]
        ).dt.days

    keep_cols = ["RSSDID", "REPDTE", "year", "quarter", "year_qtr"] + list(rename_call.values())
    keep_cols += [
        c
        for c in [
            "hhi_bank_q",
            "bank_total_deposits_mil_q",
            "BKCLASS_q",
            "CERT_q",
            "flag_sod_sameyear",
            "flag_sod_ffill_used",
            "sod_ffill_days",
        ]
        if c in bank_q.columns
    ]
    bank_q = bank_q[keep_cols].copy()

    return bank_q


def align_lender_to_bank_quarter(
    df_lender: pd.DataFrame,
    lender_map: pd.DataFrame,
    bank_q: pd.DataFrame,
) -> pd.DataFrame:
    map_cols = [
        "ds_row_id",
        "mapped_rssd_id",
        "lender_rssd_match_method",
        "flag_lender_rssd_mapped",
        "flag_rssd_extrapolated_post2016",
    ]
    if "lender_category" in lender_map.columns:
        map_cols.append("lender_category")
    for c in [
        "manual_rule_start_year",
        "manual_rule_end_year",
        "manual_rule_start_repdte",
        "manual_rule_end_repdte",
        "manual_rule_primary_rssd_id",
        "manual_rule_successor_rssd_id",
        "manual_rule_successor_start_year",
        "manual_rule_successor_end_year",
        "manual_rule_successor_start_repdte",
        "manual_rule_successor_end_repdte",
    ]:
        if c in lender_map.columns:
            map_cols.append(c)
    out = df_lender.merge(
        lender_map[map_cols],
        on="ds_row_id",
        how="left",
    )

    out["deal_date"] = pd.to_datetime(out["deal_date"], errors="coerce")
    out["deal_quarter_end"] = out["deal_date"].dt.to_period("Q").dt.to_timestamp("Q")

    left = out[out["mapped_rssd_id"].notna() & out["deal_date"].notna()].copy()
    left["mapped_rssd_id_int"] = left["mapped_rssd_id"].astype("int64")

    right = bank_q.copy()
    right["RSSDID_int"] = right["RSSDID"].astype("int64")

    left = left.sort_values(["deal_date", "mapped_rssd_id_int"]).reset_index(drop=True)
    right = right.sort_values(["REPDTE", "RSSDID_int"]).reset_index(drop=True)

    # Primary alignment: most recent prior Call Report within ~1 year.
    aligned = pd.merge_asof(
        left,
        right,
        left_on="deal_date",
        right_on="REPDTE",
        left_by="mapped_rssd_id_int",
        right_by="RSSDID_int",
        direction="backward",
        tolerance=pd.Timedelta("370 days"),
    )

    # Conservative fallback: if no backward match, allow next filing within ~120 days.
    # This recovers early-quarter deals (e.g., Jan/Feb) that precede the quarter-end call date.
    miss_back = aligned["REPDTE"].isna()
    if miss_back.any():
        left_fwd = left.loc[miss_back].copy()
        left_fwd = left_fwd.sort_values(["deal_date", "mapped_rssd_id_int"]).reset_index(drop=True)

        aligned_fwd = pd.merge_asof(
            left_fwd,
            right,
            left_on="deal_date",
            right_on="REPDTE",
            left_by="mapped_rssd_id_int",
            right_by="RSSDID_int",
            direction="forward",
            tolerance=pd.Timedelta("120 days"),
        )

        fwd_cols = [c for c in aligned_fwd.columns if c in right.columns or c == "REPDTE"]
        aligned_fwd = aligned_fwd[["ds_row_id"] + fwd_cols].copy()
        aligned = aligned.merge(
            aligned_fwd.rename(columns={c: f"{c}__fwd" for c in fwd_cols}),
            on="ds_row_id",
            how="left",
        )
        for c in fwd_cols:
            cf = f"{c}__fwd"
            if cf in aligned.columns:
                aligned[c] = aligned[c].where(aligned[c].notna(), aligned[cf])
        drop_fwd = [c for c in aligned.columns if c.endswith("__fwd")]
        if drop_fwd:
            aligned = aligned.drop(columns=drop_fwd)

    bank_cols = [c for c in aligned.columns if c not in out.columns or c in ["REPDTE"]]
    aligned_subset = aligned[["ds_row_id"] + bank_cols].copy()
    out = out.merge(aligned_subset, on="ds_row_id", how="left")

    out["flag_lender_call_matched"] = out["REPDTE"].notna().astype(int)
    out["flag_lender_hhi_matched"] = out["hhi_bank_q"].notna().astype(int)
    out["flag_lender_hhi_unmatched"] = (1 - out["flag_lender_hhi_matched"]).astype(int)
    return out


def add_macro_and_shocks(df: pd.DataFrame) -> pd.DataFrame:
    macro = pd.read_parquet(IN_MACRO).copy()
    macro = macro.reset_index().rename(columns={"index": "deal_quarter_end"})
    macro["deal_quarter_end"] = pd.to_datetime(macro["deal_quarter_end"], errors="coerce")
    macro_cols = [c for c in macro.columns if c != "deal_quarter_end"]
    macro = macro.rename(columns={c: f"macro_{c}" for c in macro_cols})

    out = df.merge(macro, on="deal_quarter_end", how="left")
    if "macro_ffr_q" in out.columns:
        out["ffr_q"] = out["macro_ffr_q"]

    if IN_JK_MONTHLY.exists():
        jk = pd.read_csv(IN_JK_MONTHLY)
        jk["month"] = pd.to_numeric(jk["month"], errors="coerce")
        jk["year"] = pd.to_numeric(jk["year"], errors="coerce")
        jk = jk.dropna(subset=["year", "month"]).copy()
        jk["date"] = pd.to_datetime(
            jk["year"].astype(int).astype(str) + "-" + jk["month"].astype(int).astype(str).str.zfill(2) + "-01",
            errors="coerce",
        )
        jk["deal_quarter_end"] = jk["date"].dt.to_period("Q").dt.to_timestamp("Q")

        shock_cols = [c for c in ["MP_pm", "CBI_pm", "MP_median", "CBI_median"] if c in jk.columns]
        if shock_cols:
            agg = jk.groupby("deal_quarter_end")[shock_cols].agg(["sum", "mean"])
            agg.columns = [f"jk_{c}_{stat}" for c, stat in agg.columns]
            agg = agg.reset_index()
            out = out.merge(agg, on="deal_quarter_end", how="left")

    return out


def build_loan_borrower_panel(df_lender_merged: pd.DataFrame) -> pd.DataFrame:
    keys = ["lpc_deal_id", "lpc_tranche_id", "borrower_id", "deal_date", "gvkey"]

    base = (
        df_lender_merged.sort_values(keys + ["ds_row_id"])
        .drop_duplicates(keys, keep="first")
        .copy()
    )

    g = df_lender_merged.groupby(keys, dropna=False)

    cover = g.agg(
        n_lender_rows=("lender_id", "size"),
        n_lenders=("lender_id", lambda x: x.nunique(dropna=True)),
        n_lender_parents=("lender_parent_id", lambda x: x.nunique(dropna=True)),
        n_lenders_rssd_mapped=("flag_lender_rssd_mapped", "sum"),
        n_lenders_call_matched=("flag_lender_call_matched", "sum"),
        n_lenders_hhi_matched=("flag_lender_hhi_matched", "sum"),
        n_lenders_post2016_extrapolated=("flag_rssd_extrapolated_post2016", "sum"),
        lender_hhi_mean=("hhi_bank_q", "mean"),
        lender_hhi_median=("hhi_bank_q", "median"),
        call_capital_ratio_lender_mean=("call_capital_ratio", "mean"),
        call_loans_to_assets_lender_mean=("call_loans_to_assets", "mean"),
        call_total_assets_mil_lender_mean=("call_total_assets_mil", "mean"),
    ).reset_index()

    cover["share_lenders_rssd_mapped"] = cover["n_lenders_rssd_mapped"] / cover["n_lender_rows"].replace(0, np.nan)
    cover["share_lenders_call_matched"] = cover["n_lenders_call_matched"] / cover["n_lender_rows"].replace(0, np.nan)
    cover["share_lenders_hhi_matched"] = cover["n_lenders_hhi_matched"] / cover["n_lender_rows"].replace(0, np.nan)

    cover["flag_any_lender_unmatched_rssd"] = (cover["n_lenders_rssd_mapped"] < cover["n_lender_rows"]).astype(int)
    cover["flag_any_lender_unmatched_hhi"] = (cover["n_lenders_hhi_matched"] < cover["n_lender_rows"]).astype(int)
    cover["flag_all_lenders_unmatched_rssd"] = (cover["n_lenders_rssd_mapped"] == 0).astype(int)
    cover["flag_all_lenders_unmatched_hhi"] = (cover["n_lenders_hhi_matched"] == 0).astype(int)

    tmp = df_lender_merged[keys + ["hhi_bank_q", "lender_share"]].copy()
    tmp["lender_share_num"] = pd.to_numeric(tmp["lender_share"], errors="coerce")
    tmp["w_valid"] = tmp["lender_share_num"].where(tmp["hhi_bank_q"].notna())
    tmp["hhi_w"] = tmp["hhi_bank_q"] * tmp["w_valid"]
    gw = tmp.groupby(keys, dropna=False).agg(
        hhi_w_sum=("hhi_w", "sum"),
        w_sum=("w_valid", "sum"),
    ).reset_index()
    gw["lender_hhi_weighted_lender_share"] = gw["hhi_w_sum"] / gw["w_sum"].replace(0, np.nan)
    gw = gw[keys + ["lender_hhi_weighted_lender_share"]]

    panel = base.merge(cover, on=keys, how="left").merge(gw, on=keys, how="left")
    return panel


def write_report(df_lender: pd.DataFrame, df_loan: pd.DataFrame, lender_map: pd.DataFrame) -> None:
    total_rows = len(df_lender)
    rssd_mapped_rows = int(df_lender["flag_lender_rssd_mapped"].sum())
    hhi_matched_rows = int(df_lender["flag_lender_hhi_matched"].sum())
    call_matched_rows = int(df_lender["flag_lender_call_matched"].sum())
    extrap_rows = int(df_lender.get("flag_rssd_extrapolated_post2016", pd.Series(dtype=float)).fillna(0).sum())

    method_counts_rows = lender_map["lender_rssd_match_method"].value_counts(dropna=False)
    method_counts_lender = (
        lender_map[["lender_id", "lender_rssd_match_method"]]
        .drop_duplicates()
        .groupby("lender_rssd_match_method", dropna=False)["lender_id"]
        .nunique()
        .sort_values(ascending=False)
    )

    loan_total = len(df_loan)
    loan_any_unmatched_hhi = int(df_loan["flag_any_lender_unmatched_hhi"].sum())
    loan_all_unmatched_hhi = int(df_loan["flag_all_lenders_unmatched_hhi"].sum())

    lines: List[str] = []
    lines.append("# Four-Table Merge Report (2026-02-09)")
    lines.append("")
    lines.append("## 1) Merge scope")
    lines.append("- Base lender-level data: `dealscan_compustat_linked.parquet`")
    lines.append("- Added lender-side bank data: `Call Reports + SOD(HHI)`")
    lines.append("- Added macro + FFR: quarterly FRED macro parquet")
    lines.append("- Added monetary shocks: JK monthly shocks aggregated to quarter (sum + mean)")
    lines.append("")
    lines.append("## 2) Lender->Bank mapping strategy used")
    lines.append("Priority order:")
    if USE_MANUAL_MATCH_ONLY:
        lines.append("1. `manual_top_lender` (hand-verified lender_id -> RSSD within year/date window)")
        lines.append("2. `manual_top_lender_successor` (post-merger successor RSSD within year/date window)")
        lines.append("3. `unmatched` -> classified as `foreign_bank` / `non_bank_lender` / `us_bank_unmatched`")
    elif USE_EXTERNAL_CROSSWALKS:
        lines.append("1. `keil_parentid_year` (ultimateparentid + year)")
        lines.append("2. `keil_lenderid_year` (lenderid + year)")
        lines.append("3. `keil_parentid_post2016_carry`")
        lines.append("4. `keil_lenderid_post2016_carry`")
        lines.append("5. `mackinlay_lcoid_*` (range-based optional supplement)")
        if USE_DIRECT_ID_MATCH:
            lines.append("6. `parent_id_direct_rssd`")
            lines.append("7. `lender_id_direct_rssd`")
            lines.append("8. `manual_top_lender` (hand-verified lender_id -> RSSD within year window)")
            lines.append("9. `manual_top_lender_successor` (post-merger successor RSSD)")
            lines.append("10. `lender_name_exact_strict_unique`")
            lines.append("11. `lender_name_exact_clean_unique`")
            lines.append("12. `unmatched` -> classified as `foreign_bank` / `non_bank_lender` / `us_bank_unmatched`")
        else:
            lines.append("6. `manual_top_lender` (hand-verified lender_id -> RSSD within year window)")
            lines.append("7. `manual_top_lender_successor` (post-merger successor RSSD)")
            lines.append("8. `lender_name_exact_strict_unique`")
            lines.append("9. `lender_name_exact_clean_unique`")
            lines.append("10. `unmatched` -> classified as `foreign_bank` / `non_bank_lender` / `us_bank_unmatched`")
    else:
        if USE_DIRECT_ID_MATCH:
            lines.append("1. `parent_id_direct_rssd`")
            lines.append("2. `lender_id_direct_rssd`")
            lines.append("3. `manual_top_lender` (hand-verified lender_id -> RSSD within year window)")
            lines.append("4. `manual_top_lender_successor` (post-merger successor RSSD)")
            lines.append("5. `lender_name_exact_strict_unique`")
            lines.append("6. `lender_name_exact_clean_unique`")
            lines.append("7. `unmatched` -> classified as `foreign_bank` / `non_bank_lender` / `us_bank_unmatched`")
        else:
            lines.append("1. `manual_top_lender` (hand-verified lender_id -> RSSD within year window)")
            lines.append("2. `manual_top_lender_successor` (post-merger successor RSSD)")
            lines.append("3. `lender_name_exact_strict_unique`")
            lines.append("4. `lender_name_exact_clean_unique`")
            lines.append("5. `unmatched` -> classified as `foreign_bank` / `non_bank_lender` / `us_bank_unmatched`")
    lines.append("")
    lines.append("### Why this approach")
    if USE_MANUAL_MATCH_ONLY:
        lines.append("- Manual-only regime is enabled (`USE_MANUAL_MATCH_ONLY = True`).")
        lines.append("- Crosswalk, direct-ID, and name-based fallbacks are intentionally disabled.")
        lines.append("- Mapping relies exclusively on manual lender-ID rules with explicit primary/successor windows.")
    elif USE_EXTERNAL_CROSSWALKS:
        lines.append("- Follows crosswalk documentation: first identify parent-level RSSD by year, then lender-level RSSD by year.")
        lines.append("- Years after crosswalk end are explicitly carried forward and flagged (`flag_rssd_extrapolated_post2016`).")
    else:
        lines.append("- External Keil/LCOID crosswalks are explicitly disabled in this run.")
        if USE_DIRECT_ID_MATCH:
            lines.append("- Mapping relies on direct RSSD IDs, manual lender-ID rules with year windows/successor mapping, and conservative exact name matching.")
        else:
            lines.append("- Mapping relies on manual lender-ID rules with year windows/successor mapping and conservative exact name matching.")
    lines.append("- Manual mapping added for major US banks whose DealScan names differ from FDIC legal names.")
    lines.append("- Unmatched lenders classified by type (foreign / non-bank / US unmatched) for transparency.")
    lines.append("")
    lines.append("## 3) Coverage diagnostics")
    lines.append(f"- Lender-level rows: `{total_rows:,}`")
    lines.append(f"- Rows with mapped RSSDID: `{rssd_mapped_rows:,}` ({safe_ratio(rssd_mapped_rows, total_rows)*100:.2f}%)")
    lines.append(f"- Rows with matched Call Report quarter: `{call_matched_rows:,}` ({safe_ratio(call_matched_rows, total_rows)*100:.2f}%)")
    lines.append(f"- Rows with matched HHI: `{hhi_matched_rows:,}` ({safe_ratio(hhi_matched_rows, total_rows)*100:.2f}%)")
    lines.append(f"- Rows using post-cutoff extrapolated mapping: `{extrap_rows:,}` ({safe_ratio(extrap_rows, total_rows)*100:.2f}%)")
    lines.append("")
    lines.append("### Mapping method counts (row-level)")
    for k, v in method_counts_rows.items():
        lines.append(f"- `{k}`: `{int(v):,}`")
    lines.append("")
    lines.append("### Mapping method counts (unique lender_id)")
    for k, v in method_counts_lender.items():
        lines.append(f"- `{k}`: `{int(v):,}`")
    lines.append("")
    lines.append("### Loan-level unmatched flags")
    lines.append(f"- Loan-borrower rows: `{loan_total:,}`")
    lines.append(
        f"- Rows with at least one unmatched lender HHI: `{loan_any_unmatched_hhi:,}` "
        f"({safe_ratio(loan_any_unmatched_hhi, loan_total)*100:.2f}%)"
    )
    lines.append(
        f"- Rows with all lenders unmatched HHI: `{loan_all_unmatched_hhi:,}` "
        f"({safe_ratio(loan_all_unmatched_hhi, loan_total)*100:.2f}%)"
    )

    if "lender_category" in lender_map.columns:
        lines.append("")
        lines.append("### Unmatched lender classification")
        cat_counts_rows = (
            lender_map.groupby("lender_category", dropna=False)["ds_row_id"]
            .count()
            .sort_values(ascending=False)
        )
        for cat, cnt in cat_counts_rows.items():
            lines.append(f"- `{cat}`: `{int(cnt):,}` rows ({safe_ratio(cnt, total_rows)*100:.2f}%)")
        cat_counts_lender = (
            lender_map[["lender_id", "lender_category"]]
            .drop_duplicates()
            .groupby("lender_category", dropna=False)["lender_id"]
            .nunique()
            .sort_values(ascending=False)
        )
        lines.append("")
        lines.append("### Unmatched lender classification (unique lender_id)")
        for cat, cnt in cat_counts_lender.items():
            lines.append(f"- `{cat}`: `{int(cnt):,}`")
    lines.append("")
    lines.append("## 4) Crosswalk documentation used")
    if USE_EXTERNAL_CROSSWALKS:
        lines.append("- `data/intermediate/crosswalks/Documentation RSSD Link.pdf`")
        lines.append("- `data/intermediate/crosswalks/rssd_lenderid.csv`")
        lines.append("- `data/intermediate/crosswalks/rssd_ultimateparentid.csv`")
        lines.append("- `data/intermediate/crosswalks/lcoid_rssdhcr_linkfile_README.txt`")
        lines.append("- `data/intermediate/crosswalks/lcoid_rssdhcr_match_external_oct2019.csv`")
    else:
        lines.append("- External crosswalk methods disabled (`USE_EXTERNAL_CROSSWALKS = False`).")
    lines.append("")
    lines.append("## 5) Output files")
    lines.append("- `data/intermediate/merged/dealscan_compustat_call_sod_macro_lenderlevel.parquet`")
    lines.append("- `data/intermediate/merged/dealscan_compustat_call_sod_macro_loan_borrower_panel.parquet`")
    lines.append("- `data/intermediate/merged/dealscan_lender_rssd_mapping.parquet`")
    lines.append("- `data/intermediate/merged/lender_bank_hhi_coverage_summary.csv`")
    lines.append("- `data/intermediate/merged/loan_lender_match_coverage_summary.csv`")
    lines.append("- `data/intermediate/merged/lender_bank_hhi_coverage_by_year.csv`")

    OUT_REPORT.write_text("\n".join(lines))


def regenerate_lender_matching_coverage_report() -> None:
    """Regenerate lender bank matching report directly from output CSV/parquet artifacts."""
    if not (OUT_LENDER_COVERAGE.exists() and OUT_LOAN_COVERAGE.exists() and OUT_YEAR_COVERAGE.exists() and OUT_LENDER_MAP.exists()):
        return

    lender_cov = pd.read_csv(OUT_LENDER_COVERAGE)
    loan_cov = pd.read_csv(OUT_LOAN_COVERAGE)
    year_cov = pd.read_csv(OUT_YEAR_COVERAGE)

    map_cols = ["lender_id", "flag_lender_rssd_mapped"]
    cat_available = False
    try:
        lender_map = pd.read_parquet(OUT_LENDER_MAP)
        cat_available = "lender_category" in lender_map.columns
    except Exception:
        lender_map = pd.DataFrame(columns=map_cols)
    if not lender_map.empty:
        map_cols = [c for c in map_cols if c in lender_map.columns]
        if map_cols:
            lender_map = lender_map[map_cols + (["lender_category"] if cat_available else [])].copy()

    total_rows = int(lender_cov["rows"].sum())
    rssd_mapped_rows = int(lender_cov["rssd_mapped"].sum())
    call_matched_rows = int(lender_cov["call_matched"].sum())
    hhi_matched_rows = int(lender_cov["hhi_matched"].sum())
    extrap_rows = int(lender_cov["extrapolated"].sum())

    matched_lenders = np.nan
    if not lender_map.empty and "flag_lender_rssd_mapped" in lender_map.columns:
        matched_lenders = int(
            lender_map[lender_map["flag_lender_rssd_mapped"] == 1]["lender_id"].nunique(dropna=True)
        )

    loan_metric = dict(zip(loan_cov["metric"], loan_cov["value"]))
    loan_rows = int(float(loan_metric.get("loan_rows", np.nan)))
    share_any_unmatched_hhi = float(loan_metric.get("share_any_lender_unmatched_hhi", np.nan))
    share_all_unmatched_hhi = float(loan_metric.get("share_all_lenders_unmatched_hhi", np.nan))

    lines: List[str] = []
    lines.append("# Lender-Bank Matching Coverage Report")
    lines.append("")
    lines.append(f"**Generated from outputs on**: {pd.Timestamp.today().date().isoformat()}")
    lines.append("**Source files**:")
    lines.append(f"- `{OUT_LENDER_COVERAGE}`")
    lines.append(f"- `{OUT_LOAN_COVERAGE}`")
    lines.append(f"- `{OUT_YEAR_COVERAGE}`")
    lines.append(f"- `{OUT_LENDER_MAP}`")
    lines.append("")
    lines.append("## 1) Headline coverage")
    lines.append(f"- Lender-level rows: `{total_rows:,}`")
    lines.append(f"- RSSD mapped rows: `{rssd_mapped_rows:,}` ({safe_ratio(rssd_mapped_rows, total_rows)*100:.2f}%)")
    lines.append(f"- Call Report matched rows: `{call_matched_rows:,}` ({safe_ratio(call_matched_rows, total_rows)*100:.2f}%)")
    lines.append(f"- HHI matched rows: `{hhi_matched_rows:,}` ({safe_ratio(hhi_matched_rows, total_rows)*100:.2f}%)")
    lines.append(f"- Post-cutoff extrapolated rows: `{extrap_rows:,}` ({safe_ratio(extrap_rows, total_rows)*100:.2f}%)")
    if not np.isnan(matched_lenders):
        lines.append(f"- Unique matched lenders: `{int(matched_lenders):,}`")
    lines.append("")
    lines.append("## 2) Match method breakdown (row-level)")
    for _, r in lender_cov.sort_values("rows", ascending=False).iterrows():
        method = str(r["lender_rssd_match_method"])
        rows = int(r["rows"])
        call_rate = float(r["call_matched_rate"]) * 100 if "call_matched_rate" in lender_cov.columns else np.nan
        hhi_rate = float(r["hhi_matched_rate"]) * 100 if "hhi_matched_rate" in lender_cov.columns else np.nan
        lines.append(
            f"- `{method}`: `{rows:,}` rows ({safe_ratio(rows, total_rows)*100:.2f}%), "
            f"Call `{call_rate:.2f}%`, HHI `{hhi_rate:.2f}%`"
        )

    lines.append("")
    lines.append("## 3) Loan-level coverage")
    lines.append(f"- Loan-borrower rows: `{loan_rows:,}`")
    lines.append(
        f"- Loans with at least one unmatched lender HHI: "
        f"`{share_any_unmatched_hhi*100:.2f}%`"
    )
    lines.append(
        f"- Loans with all lenders unmatched HHI: "
        f"`{share_all_unmatched_hhi*100:.2f}%`"
    )
    lines.append(
        f"- Loans with at least one lender HHI matched: "
        f"`{(1.0 - share_all_unmatched_hhi)*100:.2f}%`"
    )
    lines.append(
        f"- Loans with all lenders HHI matched: "
        f"`{(1.0 - share_any_unmatched_hhi)*100:.2f}%`"
    )

    if cat_available:
        lines.append("")
        lines.append("## 4) Unmatched lender classification")
        cat_rows = (
            lender_map.groupby("lender_category", dropna=False)
            .size()
            .sort_values(ascending=False)
        )
        cat_lenders = (
            lender_map[["lender_id", "lender_category"]]
            .drop_duplicates()
            .groupby("lender_category", dropna=False)["lender_id"]
            .nunique()
            .sort_values(ascending=False)
        )
        for cat, cnt in cat_rows.items():
            unique_l = int(cat_lenders.get(cat, 0))
            lines.append(
                f"- `{cat}`: `{int(cnt):,}` rows ({safe_ratio(int(cnt), total_rows)*100:.2f}%), "
                f"`{unique_l:,}` unique lenders"
            )

    lines.append("")
    lines.append("## 5) Year coverage snapshot")
    for _, r in year_cov.sort_values("deal_year").iterrows():
        y = int(r["deal_year"])
        rr = int(r["rows"])
        rssd_rate = float(r["rssd_mapped_rate"]) * 100
        hhi_rate = float(r["hhi_matched_rate"]) * 100
        lines.append(f"- `{y}`: rows `{rr:,}`, RSSD `{rssd_rate:.2f}%`, HHI `{hhi_rate:.2f}%`")

    OUT_DOC_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_LENDER_MATCH_REPORT.write_text("\n".join(lines))


def _audit_name_tokens(name: object) -> set:
    """Tokenize names for manual mapping audit (conservative stopword removal)."""
    s = strict_normalize_name(name)
    if not s:
        return set()
    toks = s.split()
    stop = {
        "BANK",
        "NATIONAL",
        "ASSOCIATION",
        "NA",
        "N",
        "A",
        "CORP",
        "CORPORATION",
        "CO",
        "COMPANY",
        "INC",
        "LLC",
        "LTD",
        "THE",
        "GROUP",
        "HOLDINGS",
        "HOLDING",
        "TRUST",
        "USA",
        "US",
        "AND",
    }
    return {t for t in toks if t not in stop}


def _safe_name_jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    u = a | b
    if not u:
        return 0.0
    return float(len(a & b)) / float(len(u))


def _safe_name_containment(a: set, b: set) -> float:
    """Share of lender-name tokens found in mapped-bank tokens."""
    if not a:
        return 0.0
    return float(len(a & b)) / float(len(a))


def write_manual_match_name_date_audit(lender_merged: pd.DataFrame, call_df: pd.DataFrame) -> None:
    """
    Audit manual lender->bank matches using:
    1) date alignment to Call Report REPDTE
    2) lender name vs mapped bank legal-name consistency.
    """
    manual_methods = {"manual_top_lender", "manual_top_lender_successor", "manual_top_lender_parent", "manual_top_lender_parent_successor"}
    m = lender_merged[lender_merged["lender_rssd_match_method"].isin(manual_methods)].copy()
    if m.empty:
        return

    call_name = call_df[["RSSDID", "REPDTE", "RSSD9017"]].copy()
    call_name["RSSDID"] = to_int_series(call_name["RSSDID"])
    call_name["REPDTE"] = pd.to_datetime(call_name["REPDTE"], errors="coerce")
    call_name = call_name.dropna(subset=["RSSDID", "REPDTE"]).drop_duplicates(["RSSDID", "REPDTE"])

    m["mapped_rssd_id"] = to_int_series(m["mapped_rssd_id"])
    m["REPDTE"] = pd.to_datetime(m["REPDTE"], errors="coerce")
    m["deal_date"] = pd.to_datetime(m["deal_date"], errors="coerce")

    m = m.merge(
        call_name.rename(columns={"RSSDID": "mapped_rssd_id"}),
        on=["mapped_rssd_id", "REPDTE"],
        how="left",
    )

    m["deal_to_repdte_days"] = (m["deal_date"] - m["REPDTE"]).dt.days
    m["abs_deal_to_repdte_days"] = m["deal_to_repdte_days"].abs()
    m["flag_call_repdte_missing"] = m["REPDTE"].isna().astype(int)
    m["flag_bank_name_missing"] = m["RSSD9017"].isna().astype(int)
    m["flag_date_out_of_alignment_window"] = (
        m["deal_to_repdte_days"].notna()
        & ((m["deal_to_repdte_days"] < -120) | (m["deal_to_repdte_days"] > 370))
    ).astype(int)

    m["lender_name_clean"] = m["lender_name"].map(clean_name)
    m["bank_name_clean"] = m["RSSD9017"].map(clean_name)
    m["flag_name_exact_clean"] = (
        m["lender_name_clean"].notna()
        & m["bank_name_clean"].notna()
        & (m["lender_name_clean"] != "")
        & (m["lender_name_clean"] == m["bank_name_clean"])
    ).astype(int)

    m["lender_tokens"] = m["lender_name"].map(_audit_name_tokens)
    m["bank_tokens"] = m["RSSD9017"].map(_audit_name_tokens)
    m["name_jaccard"] = m.apply(lambda r: _safe_name_jaccard(r["lender_tokens"], r["bank_tokens"]), axis=1)
    m["name_containment"] = m.apply(
        lambda r: _safe_name_containment(r["lender_tokens"], r["bank_tokens"]), axis=1
    )
    m["flag_low_name_overlap"] = (
        m["RSSD9017"].notna()
        & m["lender_tokens"].map(len).gt(0)
        & m["name_containment"].lt(0.34)
    ).astype(int)

    summary = pd.DataFrame(
        {
            "metric": [
                "manual_rows",
                "manual_call_repdte_missing_rows",
                "manual_call_repdte_missing_pct",
                "manual_bank_name_missing_rows",
                "manual_bank_name_missing_pct",
                "manual_date_out_of_alignment_window_rows",
                "manual_date_out_of_alignment_window_pct",
                "manual_name_exact_clean_pct",
                "manual_low_name_overlap_pct",
                "manual_abs_deal_to_repdte_days_p50",
                "manual_abs_deal_to_repdte_days_p95",
            ],
            "value": [
                len(m),
                int(m["flag_call_repdte_missing"].sum()),
                float(m["flag_call_repdte_missing"].mean() * 100.0),
                int(m["flag_bank_name_missing"].sum()),
                float(m["flag_bank_name_missing"].mean() * 100.0),
                int(m["flag_date_out_of_alignment_window"].sum()),
                float(m["flag_date_out_of_alignment_window"].mean() * 100.0),
                float(m["flag_name_exact_clean"].mean() * 100.0),
                float(m["flag_low_name_overlap"].mean() * 100.0),
                float(m["abs_deal_to_repdte_days"].median(skipna=True)),
                float(m["abs_deal_to_repdte_days"].quantile(0.95)),
            ],
        }
    )

    by_lender = (
        m.groupby(
            ["lender_id", "lender_name", "mapped_rssd_id", "RSSD9017", "lender_rssd_match_method"],
            dropna=False,
        )
        .agg(
            rows=("ds_row_id", "size"),
            call_repdte_missing_rate=("flag_call_repdte_missing", "mean"),
            bank_name_missing_rate=("flag_bank_name_missing", "mean"),
            date_out_of_window_rate=("flag_date_out_of_alignment_window", "mean"),
            name_exact_clean_rate=("flag_name_exact_clean", "mean"),
            low_name_overlap_rate=("flag_low_name_overlap", "mean"),
            name_containment_median=("name_containment", "median"),
            abs_deal_to_repdte_days_p50=("abs_deal_to_repdte_days", "median"),
            abs_deal_to_repdte_days_p95=("abs_deal_to_repdte_days", lambda x: x.quantile(0.95)),
            deal_min=("deal_date", "min"),
            deal_max=("deal_date", "max"),
            repdte_min=("REPDTE", "min"),
            repdte_max=("REPDTE", "max"),
        )
        .reset_index()
    )
    by_lender = by_lender.sort_values(["rows", "call_repdte_missing_rate"], ascending=[False, True])

    suspect = by_lender[
        (by_lender["rows"] >= 25)
        & (
            (by_lender["call_repdte_missing_rate"] > 0.00)
            | (by_lender["date_out_of_window_rate"] > 0.00)
            | (
                (by_lender["rows"] >= 500)
                & (by_lender["low_name_overlap_rate"] > 0.95)
                & (by_lender["name_exact_clean_rate"] < 0.05)
            )
        )
    ].copy()

    OUT_DIAG_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT_MANUAL_AUDIT_SUMMARY, index=False)
    by_lender.to_csv(OUT_MANUAL_AUDIT_BY_LENDER, index=False)
    suspect.to_csv(OUT_MANUAL_AUDIT_SUSPECT, index=False)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DOC_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DOC_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIAG_DIR.mkdir(parents=True, exist_ok=True)

    df_lender = pd.read_parquet(IN_LENDER_LEVEL)
    call_df = pd.read_parquet(IN_CALL)
    sod_df = pd.read_parquet(IN_SOD)

    lender_map = build_lender_rssd_map(df_lender, call_df)
    lender_map.to_parquet(OUT_LENDER_MAP, index=False)

    bank_q = build_bank_quarter_table(call_df, sod_df)

    lender_merged = align_lender_to_bank_quarter(df_lender, lender_map, bank_q)

    lender_merged = add_macro_and_shocks(lender_merged)
    lender_merged.to_parquet(OUT_LENDER_PANEL, index=False)

    loan_panel = build_loan_borrower_panel(lender_merged)
    loan_panel.to_parquet(OUT_LOAN_PANEL, index=False)

    lender_cov = (
        lender_merged.groupby("lender_rssd_match_method", dropna=False)
        .agg(
            rows=("ds_row_id", "size"),
            rssd_mapped=("flag_lender_rssd_mapped", "sum"),
            call_matched=("flag_lender_call_matched", "sum"),
            hhi_matched=("flag_lender_hhi_matched", "sum"),
            extrapolated=("flag_rssd_extrapolated_post2016", "sum"),
        )
        .reset_index()
    )
    lender_cov["rssd_mapped_rate"] = lender_cov["rssd_mapped"] / lender_cov["rows"].replace(0, np.nan)
    lender_cov["call_matched_rate"] = lender_cov["call_matched"] / lender_cov["rows"].replace(0, np.nan)
    lender_cov["hhi_matched_rate"] = lender_cov["hhi_matched"] / lender_cov["rows"].replace(0, np.nan)
    lender_cov["extrapolated_rate"] = lender_cov["extrapolated"] / lender_cov["rows"].replace(0, np.nan)
    lender_cov.to_csv(OUT_LENDER_COVERAGE, index=False)

    loan_cov = pd.DataFrame(
        {
            "metric": [
                "loan_rows",
                "share_any_lender_unmatched_rssd",
                "share_all_lenders_unmatched_rssd",
                "share_any_lender_unmatched_hhi",
                "share_all_lenders_unmatched_hhi",
            ],
            "value": [
                len(loan_panel),
                float(loan_panel["flag_any_lender_unmatched_rssd"].mean()),
                float(loan_panel["flag_all_lenders_unmatched_rssd"].mean()),
                float(loan_panel["flag_any_lender_unmatched_hhi"].mean()),
                float(loan_panel["flag_all_lenders_unmatched_hhi"].mean()),
            ],
        }
    )
    loan_cov.to_csv(OUT_LOAN_COVERAGE, index=False)

    lender_merged["deal_year"] = pd.to_datetime(lender_merged["deal_date"], errors="coerce").dt.year
    year_cov = (
        lender_merged.dropna(subset=["deal_year"])
        .groupby("deal_year")
        .agg(
            rows=("ds_row_id", "size"),
            rssd_mapped=("flag_lender_rssd_mapped", "sum"),
            call_matched=("flag_lender_call_matched", "sum"),
            hhi_matched=("flag_lender_hhi_matched", "sum"),
            extrapolated=("flag_rssd_extrapolated_post2016", "sum"),
        )
        .reset_index()
        .sort_values("deal_year")
    )
    year_cov["rssd_mapped_rate"] = year_cov["rssd_mapped"] / year_cov["rows"].replace(0, np.nan)
    year_cov["call_matched_rate"] = year_cov["call_matched"] / year_cov["rows"].replace(0, np.nan)
    year_cov["hhi_matched_rate"] = year_cov["hhi_matched"] / year_cov["rows"].replace(0, np.nan)
    year_cov["extrapolated_rate"] = year_cov["extrapolated"] / year_cov["rows"].replace(0, np.nan)
    year_cov.to_csv(OUT_YEAR_COVERAGE, index=False)

    write_report(lender_merged, loan_panel, lender_map)
    regenerate_lender_matching_coverage_report()
    write_manual_match_name_date_audit(lender_merged, call_df)

    print("Done.")
    print("Lender-level output:", OUT_LENDER_PANEL)
    print("Loan-borrower output:", OUT_LOAN_PANEL)
    print("Lender coverage:", OUT_LENDER_COVERAGE)
    print("Loan coverage:", OUT_LOAN_COVERAGE)
    print("Year coverage:", OUT_YEAR_COVERAGE)
    print("Report:", OUT_REPORT)
    print("Manual match audit summary:", OUT_MANUAL_AUDIT_SUMMARY)
    print("Manual match audit by lender:", OUT_MANUAL_AUDIT_BY_LENDER)
    print("Manual match audit suspect pairs:", OUT_MANUAL_AUDIT_SUSPECT)


if __name__ == "__main__":
    main()
