/*==============================================================================
  Script 208: Thesis table artifacts for main results, JK robustness, and
              uninsured-deposit median split

  Purpose:
    Create dated regression artifacts used by the thesis-facing table fragments
    requested on 2026-05-07. This script does not modify the historical proposal
    or supervision-report table artifacts.

    Main table:
      (1) Borrower-year FE
      (2) Borrower-year + bank-year FE
      (3) Borrower-year + bank-firm FE
      MP = lagged change in FFR.

    JK robustness table:
      (1) Borrower-year + bank-year FE
      MP = lagged Jarocinski-Karadi shock.

    All specifications use the same dependent variable, control set including
    purpose dummies, risk measure, and clustering as scripts 189/191:
    broad lead-bank, US non-financial borrower, 2001-2024;
    DV = ln(all_in_spread_bps); risk = risk_p_naive_z; HHI = hhi_std_y;
    vce(cluster bank_id quarter_id).

  Output:
    regression/overview_diagnostics/spec208_thesis_main_tables_full_coefs_YYYY-MM-DD.csv
    regression/overview_diagnostics/spec208_thesis_main_tables_key_YYYY-MM-DD.csv
==============================================================================*/
set more off
clear all

if "$BASE" == "" {
    display as error "Set global BASE to your local empirical data workspace before running."
    exit 198
}
local __d = date("`c(current_date)'", "DMY")
local run_date : display %tdCCYY-NN-DD `__d'
if "$RESULTS" == "" global RESULTS "results/reestimated"
capture mkdir "$RESULTS"
local OUT "$RESULTS"

timer clear 1
timer on 1

/*==============================================================================
  SECTION 1: LOAD AND MERGE  (identical to scripts 188/189)
==============================================================================*/

local in_dta        "$BASE/data/final/regression_panel_lender_level_ppml_input.dta"
local in_lspread    "$BASE/data/final/regression_panel_lender_level_lender_spread_controls.dta"
local in_pit        "$BASE/data/intermediate/merged/pd_pit_origination_2001_2025_for_stata.dta"
local in_bank_extra "$BASE/data/final/regression_panel_extra_bank_controls.dta"
local in_macro      "$BASE/data/final/regression_panel_lender_level_ppml_macro_controls.dta"
local in_struct     "$BASE/data/final/regression_panel_lender_level_ppml_structure_controls.dta"

use "`in_dta'", clear
di as result "Loaded main panel: N = " _N

merge m:1 deal_id tranche_id lender_id gvkey year quarter using "`in_lspread'", nogen keep(master match)

capture confirm file "`in_struct'"
if !_rc {
    merge m:1 deal_id tranche_id lender_id gvkey year quarter using "`in_struct'", nogen keep(master match)
}

capture confirm file "`in_bank_extra'"
if !_rc {
    merge m:1 mapped_rssd_id year quarter using "`in_bank_extra'", nogen keep(master match)
}

capture confirm file "`in_macro'"
if !_rc {
    merge m:1 year quarter using "`in_macro'", nogen keep(master match)
}

preserve
use "`in_pit'", clear
capture confirm numeric variable gvkey
if !_rc tostring gvkey, replace format(%12.0g)
keep gvkey deal_date pd_naive_bs2008_pit pd_iterative_kmv_pit pd_chs_12m_pit
collapse (mean) pd_naive_bs2008_pit pd_iterative_kmv_pit pd_chs_12m_pit, by(gvkey deal_date)
tempfile pitu
save `pitu'
restore

capture confirm numeric variable gvkey
if !_rc tostring gvkey, replace format(%12.0g)
merge m:1 gvkey deal_date using `pitu', nogen keep(master match)

di as result "After all merges: N = " _N

/*==============================================================================
  SECTION 2: SAMPLE FILTERS  (identical to scripts 188/189)
==============================================================================*/

keep if !missing(gvkey)

capture confirm variable country
if !_rc {
    gen str30 __country = strlower(strtrim(country))
    keep if inlist(__country, "united states", "united states of america", "us", "usa", "u.s.", "u.s.a.")
    drop __country
}

capture confirm variable is_financial_borrower
if !_rc {
    keep if is_financial_borrower != 1 & !missing(is_financial_borrower)
}

gen double bank_rssd = mapped_rssd_id
replace bank_rssd = rssdid if missing(bank_rssd) & !missing(rssdid)
keep if !missing(bank_rssd) & !missing(gvkey)

gen byte is_lead_broad = 0
capture confirm variable primary_role
if !_rc {
    gen str80 __role = strlower(strtrim(primary_role))
    replace is_lead_broad = 1 if inlist(__role, "admin agent", "arranger", "lead arranger", "mandated lead arranger")
    replace is_lead_broad = 1 if inlist(__role, "bookrunner", "lead manager", "syndication agent", "syndications agent")
    replace is_lead_broad = 1 if inlist(__role, "managing agent", "sole lender", "lead left", "mandated arranger")
    replace is_lead_broad = 1 if inlist(__role, "co-arranger", "co-agent", "co-lead manager", "co-lead arranger")
    replace is_lead_broad = 1 if inlist(__role, "lead bank", "senior lead manager", "coordinating arranger", "documentation agent")
    replace is_lead_broad = 1 if inlist(__role, "agent")
    drop __role
}

capture confirm variable lead_arranger
if !_rc {
    gen strL __lead = strlower(strtrim(lead_arranger))
    replace __lead = subinstr(__lead, ";", ",", .)
    replace __lead = subinstr(__lead, "/", ",", .)
    gen strL __lead_compact = subinstr(__lead, " ", "", .)
    gen strL __lend_compact = subinstr(strlower(strtrim(lender_name)), " ", "", .)
    gen byte lead_arranger_match = ///
        (strpos("," + __lead_compact + ",", "," + __lend_compact + ",") > 0) ///
        if !missing(__lead_compact, __lend_compact)
    replace lead_arranger_match = 0 if missing(lead_arranger_match)
    replace is_lead_broad = 1 if lead_arranger_match == 1
    drop __lead __lead_compact __lend_compact lead_arranger_match
}

keep if is_lead_broad == 1
drop is_lead_broad
keep if inrange(year, 2001, 2024)

di as result "After broad lead-bank sample filters: N = " _N

/*==============================================================================
  SECTION 3: IDS, LAGS, VARIABLES  (identical to scripts 188/189)
==============================================================================*/

capture drop bank_id
egen long bank_id = group(bank_rssd)
capture drop borrower_id
egen long borrower_id = group(gvkey)
keep if !missing(borrower_id)

gen int quarter_id = yq(year, quarter)
format quarter_id %tq
egen long borrower_year = group(borrower_id year)
egen long bank_year     = group(bank_id year)
egen long bank_firm     = group(bank_id borrower_id)

preserve
keep quarter_id ffr jk_mp
duplicates drop
sort quarter_id
tsset quarter_id
gen double ffr_l1   = L.ffr
gen double dffr     = D.ffr
gen double dffr_l1  = L.dffr
gen double jk_mp_l1 = L.jk_mp
tempfile mpl
save `mpl'
restore
merge m:1 quarter_id using `mpl', nogen keep(master match)

capture drop macro_cpi_l1 macro_gdp_growth_l1
preserve
keep quarter_id macro_cpi_q macro_gdp_growth_q
duplicates drop
sort quarter_id
tsset quarter_id
gen double macro_cpi_l1        = L.macro_cpi_q
gen double macro_gdp_growth_l1 = L.macro_gdp_growth_q
tempfile macrolag
save `macrolag'
restore
merge m:1 quarter_id using `macrolag', nogen keep(master match)

gen double risk_p_naive = pd_naive_bs2008_pit
capture drop risk_p_naive_z
quietly summarize risk_p_naive if risk_p_naive < .
if r(sd) > 0 {
    gen double risk_p_naive_z = (risk_p_naive - r(mean)) / r(sd)
}
else {
    gen double risk_p_naive_z = .
}

capture drop hhi_std_y
quietly summarize hhi_t if hhi_t < .
gen double hhi_std_y = (hhi_t - r(mean)) / r(sd) if r(sd) > 0

capture drop ln_spread_bps_dep all_in_spread_bps summary_spread_bps
capture confirm variable all_in_spread_bps
if _rc {
    gen double all_in_spread_bps = .
}
capture confirm variable spread_bps_lender_strict
if !_rc {
    replace all_in_spread_bps = spread_bps_lender_strict if missing(all_in_spread_bps) & spread_bps_lender_strict < .
}
replace all_in_spread_bps = spread_pct_t * 10000 if missing(all_in_spread_bps) & spread_pct_t < .
gen double ln_spread_bps_dep = ln(all_in_spread_bps) if all_in_spread_bps > 0 & all_in_spread_bps < .

capture confirm variable is_credit_line
if _rc {
    gen byte is_credit_line = 0
    capture confirm variable tranche_type
    if !_rc {
        replace is_credit_line = 1 if strpos(strlower(tranche_type), "revolver") > 0
        replace is_credit_line = 1 if strpos(strlower(tranche_type), "line") > 0
    }
}

capture confirm variable purp_corpgen
if _rc {
    gen byte purp_corpgen = 0
    gen byte purp_debtrepay = 0
    gen byte purp_takeover = 0
    capture confirm variable primary_purpose
    if !_rc {
        gen str80 __purpose = strlower(strtrim(primary_purpose))
        replace purp_corpgen   = 1 if ///
            inlist(__purpose, "corp. purposes", "general purpose", "working cap.") | ///
            strpos(__purpose, "corporate") > 0 | ///
            strpos(__purpose, "working cap") > 0
        replace purp_debtrepay = 1 if ///
            inlist(__purpose, "debt repay.", "debt repay") | ///
            strpos(__purpose, "debt repay") > 0 | strpos(__purpose, "refin") > 0
        replace purp_takeover  = 1 if ///
            inlist(__purpose, "takeover", "lbo", "mbo", "acquis. line") | ///
            strpos(__purpose, "takeover") > 0 | strpos(__purpose, "acquisition") > 0 | ///
            strpos(__purpose, "acquis") > 0 | strpos(__purpose, "lbo") > 0 | ///
            strpos(__purpose, "mbo") > 0
        drop __purpose
    }
}

capture confirm variable log_n_lenders
if _rc {
    capture confirm variable number_of_lenders
    if !_rc {
        gen double log_n_lenders = ln(number_of_lenders) if number_of_lenders > 0
    }
    else {
        gen double log_n_lenders = .
    }
}

capture confirm variable tangibility
if _rc {
    capture confirm variable tang_t
    if !_rc rename tang_t tangibility
}

capture confirm variable log_sales_t
if _rc {
    capture confirm variable log_assets_t
    if !_rc {
        gen double log_sales_t = log_assets_t
    }
    else {
        capture confirm variable log_assets
        if !_rc gen double log_sales_t = log_assets
    }
}

/*==============================================================================
  SECTION 4: CONTROL SETS -- loan-purpose dummies always included
==============================================================================*/

local LOAN_CTRL    "c.log_tenor_maturity i.is_credit_line c.log_n_lenders i.secured_flag"
local BANK_CTRL    "c.log_bank_assets c.call_capital_ratio c.call_deposits_to_assets c.call_loans_to_assets"
local FIRM_CTRL    "c.log_sales_t c.tangibility c.ebitda_margin c.leverage"
local MACRO_CTRL   "c.macro_cpi_l1 c.macro_gdp_growth_l1"
local PURPOSE_CTRL "i.purp_corpgen i.purp_debtrepay i.purp_takeover"
local FULL_CTRL_P  "`LOAN_CTRL' `BANK_CTRL' `FIRM_CTRL' `MACRO_CTRL' `PURPOSE_CTRL'"

gen byte main_sample = 1
foreach v in ln_spread_bps_dep hhi_t hhi_std_y dffr_l1 risk_p_naive risk_p_naive_z ///
             log_tenor_maturity is_credit_line log_n_lenders secured_flag ///
             log_bank_assets call_capital_ratio call_deposits_to_assets ///
             call_loans_to_assets log_sales_t tangibility ebitda_margin leverage ///
             macro_cpi_l1 macro_gdp_growth_l1 {
    replace main_sample = 0 if missing(`v')
}

quietly count if main_sample == 1
di as result "Pre-FE main_sample N: " r(N)

/*==============================================================================
  SECTION 5: RESULTS STORE
==============================================================================*/

local key_results  "`OUT'/spec208_thesis_main_tables_key_`run_date'.csv"
local coef_results "`OUT'/spec208_thesis_main_tables_full_coefs_`run_date'.csv"

tempname pk pc
postfile `pk' str12 table int col_id str48 col_label str16 mp_var str48 fe_label ///
    double(coef se pval N r2_adj r2_within df_r) ///
    using "`key_results'", replace

postfile `pc' str12 table int col_id str48 col_label str16 mp_var ///
    int row_order str80 display str80 raw_term ///
    double(coef se pval N r2_adj r2_within df_r) byte omitted ///
    using "`coef_results'", replace

local dv "ln_spread_bps_dep"

/* Full coefficient rows for the main FFR table. */
local nrows 24
local row1_display  "HHI x Delta FFR x Borrower PD"
local row1_term     "c.hhi_std_y#c.dffr_l1#c.risk_p_naive_z"
local row2_display  "HHI x Delta FFR"
local row2_term     "c.hhi_std_y#c.dffr_l1"
local row3_display  "HHI x Borrower PD"
local row3_term     "c.hhi_std_y#c.risk_p_naive_z"
local row4_display  "Delta FFR x Borrower PD"
local row4_term     "c.dffr_l1#c.risk_p_naive_z"
local row5_display  "HHI (z)"
local row5_term     "hhi_std_y"
local row6_display  "Delta FFR, lagged"
local row6_term     "dffr_l1"
local row7_display  "Borrower PD (Naive BS, z)"
local row7_term     "risk_p_naive_z"
local row8_display  "Log maturity"
local row8_term     "log_tenor_maturity"
local row9_display  "Credit line"
local row9_term     "1.is_credit_line"
local row10_display "Log number of lenders"
local row10_term    "log_n_lenders"
local row11_display "Secured"
local row11_term    "1.secured_flag"
local row12_display "Corporate/general purpose"
local row12_term    "1.purp_corpgen"
local row13_display "Debt repayment/refinance"
local row13_term    "1.purp_debtrepay"
local row14_display "Takeover/acquisition"
local row14_term    "1.purp_takeover"
local row15_display "Log bank assets"
local row15_term    "log_bank_assets"
local row16_display "Bank capital ratio"
local row16_term    "call_capital_ratio"
local row17_display "Deposits/assets"
local row17_term    "call_deposits_to_assets"
local row18_display "Loans/assets"
local row18_term    "call_loans_to_assets"
local row19_display "Log borrower sales/assets"
local row19_term    "log_sales_t"
local row20_display "Borrower tangibility"
local row20_term    "tangibility"
local row21_display "EBITDA margin"
local row21_term    "ebitda_margin"
local row22_display "Borrower leverage"
local row22_term    "leverage"
local row23_display "CPI, lagged"
local row23_term    "macro_cpi_l1"
local row24_display "GDP growth, lagged"
local row24_term    "macro_gdp_growth_l1"

/*==============================================================================
  SECTION 6: MAIN FFR TABLE -- THREE COLUMNS
==============================================================================*/

forvalues col = 1/3 {
    if `col' == 1 {
        local col_label "Borrower-year FE"
        local fe_label "Borrower-year"
        local absorb_opt "absorb(borrower_year)"
    }
    else if `col' == 2 {
        local col_label "Borrower-year + bank-year FE"
        local fe_label "Borrower-year + bank-year"
        local absorb_opt "absorb(borrower_year bank_year)"
    }
    else {
        local col_label "Borrower-year + bank-firm FE"
        local fe_label "Borrower-year + bank-firm"
        local absorb_opt "absorb(borrower_year bank_firm)"
    }

    di as result _n "=== MAIN TABLE COL (`col'): `col_label' ==="
    reghdfe `dv' c.hhi_std_y##c.dffr_l1##c.risk_p_naive_z `FULL_CTRL_P' ///
        if main_sample == 1, `absorb_opt' vce(cluster bank_id quarter_id)

    scalar __triple = _b[c.hhi_std_y#c.dffr_l1#c.risk_p_naive_z]
    scalar __trise = _se[c.hhi_std_y#c.dffr_l1#c.risk_p_naive_z]
    scalar __trip = 2 * ttail(e(df_r), abs(__triple / __trise))
    post `pk' ("main") (`col') ("`col_label'") ("dffr_l1") ("`fe_label'") ///
        (__triple) (__trise) (__trip) (e(N)) (e(r2_a)) (e(r2_within)) (e(df_r))

    forvalues r = 1/`nrows' {
        local disp "`row`r'_display'"
        local term "`row`r'_term'"
        scalar __b = .
        scalar __se = .
        scalar __p = .
        local __omit = 1
        capture scalar __b = _b[`term']
        if !_rc {
            capture scalar __se = _se[`term']
            if !_rc {
                if __se < . & __se > 0 {
                    scalar __p = 2 * ttail(e(df_r), abs(__b / __se))
                    local __omit = 0
                }
            }
        }
        post `pc' ("main") (`col') ("`col_label'") ("dffr_l1") ///
            (`r') ("`disp'") ("`term'") ///
            (__b) (__se) (__p) (e(N)) (e(r2_a)) (e(r2_within)) (e(df_r)) (`__omit')
    }
}

/*==============================================================================
  SECTION 7: JK ROBUSTNESS TABLE -- PREFERRED FE COLUMN ONLY
==============================================================================*/

local col = 1
local col_label "Borrower-year + bank-year FE"
local fe_label "Borrower-year + bank-year"
local absorb_opt "absorb(borrower_year bank_year)"

di as result _n "=== JK ROBUSTNESS COL (`col'): `col_label' ==="
reghdfe `dv' c.hhi_std_y##c.jk_mp_l1##c.risk_p_naive_z `FULL_CTRL_P' ///
    if main_sample == 1, `absorb_opt' vce(cluster bank_id quarter_id)

scalar __triple = _b[c.hhi_std_y#c.jk_mp_l1#c.risk_p_naive_z]
scalar __trise = _se[c.hhi_std_y#c.jk_mp_l1#c.risk_p_naive_z]
scalar __trip = 2 * ttail(e(df_r), abs(__triple / __trise))
post `pk' ("jk") (`col') ("`col_label'") ("jk_mp_l1") ("`fe_label'") ///
    (__triple) (__trise) (__trip) (e(N)) (e(r2_a)) (e(r2_within)) (e(df_r))

local jkrow1_display  "HHI x MP x Borrower PD"
local jkrow1_term     "c.hhi_std_y#c.jk_mp_l1#c.risk_p_naive_z"
local jkrow2_display  "HHI x MP"
local jkrow2_term     "c.hhi_std_y#c.jk_mp_l1"
local jkrow3_display  "HHI x Borrower PD"
local jkrow3_term     "c.hhi_std_y#c.risk_p_naive_z"
local jkrow4_display  "MP x Borrower PD"
local jkrow4_term     "c.jk_mp_l1#c.risk_p_naive_z"
local jkrow5_display  "HHI (z)"
local jkrow5_term     "hhi_std_y"
local jkrow6_display  "MP, lagged"
local jkrow6_term     "jk_mp_l1"
local jkrow7_display  "Borrower PD (Naive BS, z)"
local jkrow7_term     "risk_p_naive_z"
forvalues r = 8/`nrows' {
    local jkrow`r'_display "`row`r'_display'"
    local jkrow`r'_term "`row`r'_term'"
}

forvalues r = 1/`nrows' {
    local disp "`jkrow`r'_display'"
    local term "`jkrow`r'_term'"
    scalar __b = .
    scalar __se = .
    scalar __p = .
    local __omit = 1
    capture scalar __b = _b[`term']
    if !_rc {
        capture scalar __se = _se[`term']
        if !_rc {
            if __se < . & __se > 0 {
                scalar __p = 2 * ttail(e(df_r), abs(__b / __se))
                local __omit = 0
            }
        }
    }
    post `pc' ("jk") (`col') ("`col_label'") ("jk_mp_l1") ///
        (`r') ("`disp'") ("`term'") ///
        (__b) (__se) (__p) (e(N)) (e(r2_a)) (e(r2_within)) (e(df_r)) (`__omit')
}

/*==============================================================================
  SECTION 8: EXPORT AND SANITY ANCHORS
==============================================================================*/

postclose `pk'
postclose `pc'

use "`key_results'", clear
list, clean noobs
export delimited using "`key_results'", replace
di as result "Saved key results to: `key_results'"

use "`coef_results'", clear
export delimited using "`coef_results'", replace
di as result "Saved full coefficients to: `coef_results'"

di as result _n "=== ANCHOR CHECKS ==="
di as result "Expected main col (1), borrower-year: theta about 0.04134, N 40702."
di as result "Expected main col (2), borrower-year + bank-year: theta about 0.05003, N 40641."
di as result "Expected main col (3), borrower-year + bank-firm: theta about 0.06277, N 37756."
di as result "Expected JK col (1), borrower-year + bank-year: theta about 0.28250, N 40166."

timer off 1
timer list 1

di as result _n "=== Script 208 complete ==="
