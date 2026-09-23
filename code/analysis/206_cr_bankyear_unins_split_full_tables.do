/*==============================================================================
  Script 206: Call-Report Bank-Year Uninsured Split -- Full Coefficient Tables

  Purpose:
    Produce full coefficient tables for the two thesis-facing split definitions:
      1. cr_bankyear_unweighted_median
      2. cr_bankyear_unweighted_tercile_tails

  Guardrails:
    - New dated outputs only; no historical table is overwritten.
    - Same dependent variable, controls, FE, clustering, and interaction
      structure as script 204.
    - Split variable is the Call Report bank-year mean DSSW-style uninsured
      deposit share, with cutoffs computed over unique bank-years in the final
      estimation-panel backbone.
    - Target group is low uninsured / high inferred insured deposit share.

  Outputs:
    regression/overview_diagnostics/spec206_cr_bankyear_unins_split_full_coefs_YYYY-MM-DD.csv
    regression/overview_diagnostics/spec206_cr_bankyear_unins_split_key_results_YYYY-MM-DD.csv
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
  Helper: post all displayed non-absorbed coefficients from the current model.
==============================================================================*/

capture program drop post_all_coefs
program define post_all_coefs
    args posthandle spec subgroup

    matrix __b = e(b)
    matrix __V = e(V)
    local cn : colnames __b

    local df = e(df_r)
    if missing(`df') local df = e(N) - 1

    scalar __N = e(N)
    scalar __r2a = e(r2_a)
    capture scalar __r2w = e(r2_within)
    if _rc scalar __r2w = .

    local j = 0
    foreach term of local cn {
        local ++j
        local skip = 0
        if "`term'" == "_cons" local skip = 1
        if strpos("`term'", "o.") > 0 local skip = 1
        if strpos("`term'", "b.") > 0 local skip = 1

        if `skip' == 0 {
            scalar __coef = __b[1, `j']
            scalar __se = sqrt(__V[`j', `j'])

            if !missing(__coef) & !missing(__se) & __se > 0 {
                scalar __t = __coef / __se
                scalar __p = 2 * ttail(`df', abs(__t))

                local stars ""
                if __p < 0.01 local stars "***"
                else if __p < 0.05 local stars "**"
                else if __p < 0.10 local stars "*"

                local display "`term'"
                local ord 999
                if "`term'" == "c.hhi_std_y#c.dffr_l1#c.risk_p_naive_z" {
                    local display "HHI x Delta FFR x PD"
                    local ord 1
                }
                if "`term'" == "c.hhi_std_y#c.dffr_l1" {
                    local display "HHI x Delta FFR"
                    local ord 2
                }
                if "`term'" == "c.hhi_std_y#c.risk_p_naive_z" {
                    local display "HHI x PD"
                    local ord 3
                }
                if "`term'" == "c.dffr_l1#c.risk_p_naive_z" {
                    local display "Delta FFR x PD"
                    local ord 4
                }
                if "`term'" == "hhi_std_y" {
                    local display "HHI (z)"
                    local ord 5
                }
                if "`term'" == "dffr_l1" {
                    local display "Delta FFR, lagged"
                    local ord 6
                }
                if "`term'" == "risk_p_naive_z" {
                    local display "Borrower PD (z)"
                    local ord 7
                }
                if "`term'" == "log_tenor_maturity" {
                    local display "Log maturity"
                    local ord 8
                }
                if "`term'" == "1.is_credit_line" {
                    local display "Credit line"
                    local ord 9
                }
                if "`term'" == "log_n_lenders" {
                    local display "Log number of lenders"
                    local ord 10
                }
                if "`term'" == "1.secured_flag" {
                    local display "Secured"
                    local ord 11
                }
                if "`term'" == "1.purp_corpgen" {
                    local display "Corporate/general purpose"
                    local ord 12
                }
                if "`term'" == "1.purp_debtrepay" {
                    local display "Debt repayment/refinance"
                    local ord 13
                }
                if "`term'" == "1.purp_takeover" {
                    local display "Takeover/acquisition"
                    local ord 14
                }
                if "`term'" == "log_bank_assets" {
                    local display "Log bank assets"
                    local ord 15
                }
                if "`term'" == "call_capital_ratio" {
                    local display "Bank capital ratio"
                    local ord 16
                }
                if "`term'" == "call_deposits_to_assets" {
                    local display "Deposits/assets"
                    local ord 17
                }
                if "`term'" == "call_loans_to_assets" {
                    local display "Loans/assets"
                    local ord 18
                }
                if "`term'" == "log_sales_t" {
                    local display "Log borrower sales/assets"
                    local ord 19
                }
                if "`term'" == "tangibility" {
                    local display "Borrower tangibility"
                    local ord 20
                }
                if "`term'" == "ebitda_margin" {
                    local display "EBITDA margin"
                    local ord 21
                }
                if "`term'" == "leverage" {
                    local display "Borrower leverage"
                    local ord 22
                }
                if "`term'" == "macro_cpi_l1" {
                    local display "Lagged CPI inflation"
                    local ord 23
                }
                if "`term'" == "macro_gdp_growth_l1" {
                    local display "Lagged GDP growth"
                    local ord 24
                }

                capture post `posthandle' ("`spec'") ("`subgroup'") (`ord') ("`display'") ///
                    ("`term'") (__coef) (__se) (__t) (__p) ("`stars'") ///
                    (__N) (__r2a) (__r2w) (`df')
                if _rc {
                    di as error "post_all_coefs failed: spec=`spec' subgroup=`subgroup' term=`term' display=`display' order=`ord' rc=" _rc
                    exit _rc
                }
            }
        }
    }
end

/*==============================================================================
  Data preparation: aligned with script 204.
==============================================================================*/

local in_dta        "$BASE/data/final/regression_panel_lender_level_ppml_input.dta"
local in_lspread    "$BASE/data/final/regression_panel_lender_level_lender_spread_controls.dta"
local in_pit        "$BASE/data/intermediate/merged/pd_pit_origination_2001_2025_for_stata.dta"
local in_bank_extra "$BASE/data/final/regression_panel_extra_bank_controls.dta"
local in_macro      "$BASE/data/final/regression_panel_lender_level_ppml_macro_controls.dta"
local in_struct     "$BASE/data/final/regression_panel_lender_level_ppml_structure_controls.dta"
local in_unins      "$BASE/data/final/bank_uninsured_deposits.dta"

use "`in_dta'", clear
di as result "Loaded main panel: N = " _N

merge m:1 deal_id tranche_id lender_id gvkey year quarter using "`in_lspread'", nogen keep(master match)
capture confirm file "`in_struct'"
if !_rc merge m:1 deal_id tranche_id lender_id gvkey year quarter using "`in_struct'", nogen keep(master match)
capture confirm file "`in_bank_extra'"
if !_rc merge m:1 mapped_rssd_id year quarter using "`in_bank_extra'", nogen keep(master match)
capture confirm file "`in_macro'"
if !_rc merge m:1 year quarter using "`in_macro'", nogen keep(master match)

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

keep if !missing(gvkey)
capture confirm variable country
if !_rc {
    gen str30 __country = strlower(strtrim(country))
    keep if inlist(__country, "united states", "united states of america", "us", "usa", "u.s.", "u.s.a.")
    drop __country
}
capture confirm variable is_financial_borrower
if !_rc keep if is_financial_borrower != 1 & !missing(is_financial_borrower)

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

capture drop bank_id
egen long bank_id = group(bank_rssd)
capture drop borrower_id
egen long borrower_id = group(gvkey)
keep if !missing(borrower_id)

gen int quarter_id = yq(year, quarter)
format quarter_id %tq
egen long borrower_year = group(borrower_id year)
egen long bank_year     = group(bank_id year)

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

capture drop macro_slope_q macro_cpi_l1 macro_gdp_growth_l1 macro_slope_l1
gen double macro_slope_q = macro_gs10_q - macro_tb3ms_q if !missing(macro_gs10_q, macro_tb3ms_q)
preserve
keep quarter_id macro_cpi_q macro_gdp_growth_q macro_slope_q
duplicates drop
sort quarter_id
tsset quarter_id
gen double macro_cpi_l1        = L.macro_cpi_q
gen double macro_gdp_growth_l1 = L.macro_gdp_growth_q
gen double macro_slope_l1      = L.macro_slope_q
tempfile macrolag
save `macrolag'
restore
merge m:1 quarter_id using `macrolag', nogen keep(master match)

gen double risk_p_naive = pd_naive_bs2008_pit
capture drop risk_p_naive_z
quietly summarize risk_p_naive if risk_p_naive < .
gen double risk_p_naive_z = (risk_p_naive - r(mean)) / r(sd) if r(sd) > 0

capture drop hhi_std_y
quietly summarize hhi_t if hhi_t < .
gen double hhi_std_y = (hhi_t - r(mean)) / r(sd) if r(sd) > 0

capture drop ln_spread_bps_dep all_in_spread_bps
capture confirm variable all_in_spread_bps
if _rc gen double all_in_spread_bps = .
capture confirm variable spread_bps_lender_strict
if !_rc replace all_in_spread_bps = spread_bps_lender_strict if missing(all_in_spread_bps) & spread_bps_lender_strict < .
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

capture confirm variable secured_flag
if _rc {
    gen byte secured_flag = 0
    capture confirm variable secured
    if !_rc replace secured_flag = 1 if inlist(strlower(strtrim(secured)), "yes", "y", "secured")
}

gen byte purp_corpgen = 0
gen byte purp_debtrepay = 0
gen byte purp_takeover = 0
capture confirm variable primary_purpose
if !_rc {
    gen str80 __purpose = strlower(strtrim(primary_purpose))
    replace purp_corpgen = 1 if inlist(__purpose, "corp. purposes", "general purpose", "working cap.") | strpos(__purpose, "corporate") > 0 | strpos(__purpose, "working cap") > 0
    replace purp_debtrepay = 1 if inlist(__purpose, "debt repay.", "debt repay") | strpos(__purpose, "debt repay") > 0 | strpos(__purpose, "refin") > 0
    replace purp_takeover = 1 if inlist(__purpose, "takeover", "lbo", "mbo", "acquis. line") | strpos(__purpose, "takeover") > 0 | strpos(__purpose, "acquisition") > 0 | strpos(__purpose, "acquis") > 0 | strpos(__purpose, "lbo") > 0 | strpos(__purpose, "mbo") > 0
    drop __purpose
}

capture confirm variable log_n_lenders
if _rc {
    capture confirm variable number_of_lenders
    if !_rc gen double log_n_lenders = ln(number_of_lenders) if number_of_lenders > 0
    else gen double log_n_lenders = .
}

capture confirm variable tangibility
if _rc {
    capture confirm variable tang_t
    if !_rc rename tang_t tangibility
}

capture confirm variable log_sales_t
if _rc {
    capture confirm variable log_assets_t
    if !_rc gen double log_sales_t = log_assets_t
    else {
        capture confirm variable log_assets
        if !_rc gen double log_sales_t = log_assets
    }
}

/*==============================================================================
  Call-Report bank-year uninsured share split variable.
==============================================================================*/

preserve
use "`in_unins'", clear
rename mapped_rssd_id bank_rssd
keep if inrange(year, 2006, 2024)
keep if !missing(bank_rssd, year, quarter, unins_share_dss)
duplicates drop bank_rssd year quarter, force
collapse (mean) unins_cr_bankyear=unins_share_dss ///
         (count) unins_cr_bankyear_nq=unins_share_dss, by(bank_rssd year)
tempfile cr_by
save `cr_by'
restore
merge m:1 bank_rssd year using `cr_by', nogen keep(master match)
label var unins_cr_bankyear "Call Report bank-year mean DSSW-style uninsured share"

local LOAN_CTRL  "c.log_tenor_maturity i.is_credit_line c.log_n_lenders i.secured_flag i.purp_corpgen i.purp_debtrepay i.purp_takeover"
local BANK_CTRL  "c.log_bank_assets c.call_capital_ratio c.call_deposits_to_assets c.call_loans_to_assets"
local FIRM_CTRL  "c.log_sales_t c.tangibility c.ebitda_margin c.leverage"
local MACRO_CTRL "c.macro_cpi_l1 c.macro_gdp_growth_l1"
local FULL_CTRL  "`LOAN_CTRL' `BANK_CTRL' `FIRM_CTRL' `MACRO_CTRL'"

local dv "ln_spread_bps_dep"
local mp "dffr_l1"
local risk "risk_p_naive_z"
local INTERACT_BASE "c.hhi_std_y##c.`mp'##c.`risk'"
local triple "c.hhi_std_y#c.`mp'#c.`risk'"

mark __base_esample
markout __base_esample `dv' hhi_std_y `mp' `risk' ///
    log_tenor_maturity is_credit_line log_n_lenders secured_flag ///
    purp_corpgen purp_debtrepay purp_takeover ///
    log_bank_assets call_capital_ratio call_deposits_to_assets call_loans_to_assets ///
    log_sales_t tangibility ebitda_margin leverage ///
    macro_cpi_l1 macro_gdp_growth_l1

quietly count if __base_esample
local N_main = r(N)
di as result "Final estimation-panel backbone before split availability: " `N_main'

gen byte __cr_by_esample = __base_esample & !missing(unins_cr_bankyear)
quietly count if __cr_by_esample
local N_unins = r(N)
di as result "Backbone with Call-Report bank-year uninsured share: " `N_unins'

preserve
keep if __cr_by_esample
keep bank_id year unins_cr_bankyear
duplicates drop bank_id year, force
quietly count
local N_unique_bankyears = r(N)
quietly _pctile unins_cr_bankyear, p(33.333333 50 66.666667)
local terc_low = r(r1)
local med = r(r2)
local terc_high = r(r3)
quietly summarize unins_cr_bankyear
local mean_by = r(mean)
local sd_by = r(sd)
restore

di as result "Unique bank-years used for cutoffs: " `N_unique_bankyears'
di as result "Median cutoff = " %9.6f `med'
di as result "Tercile cutoffs = " %9.6f `terc_low' " / " %9.6f `terc_high'

gen byte split_median = .
replace split_median = 1 if __cr_by_esample & unins_cr_bankyear <= `med'
replace split_median = 0 if __cr_by_esample & unins_cr_bankyear >  `med'

gen byte split_tercile_tails = .
replace split_tercile_tails = 1 if __cr_by_esample & unins_cr_bankyear <= `terc_low'
replace split_tercile_tails = 0 if __cr_by_esample & unins_cr_bankyear >= `terc_high'

/*==============================================================================
  Output handles.
==============================================================================*/

local pf_coefs "`OUT'/spec206_cr_bankyear_unins_split_full_coefs_`run_date'.csv"
local pf_key   "`OUT'/spec206_cr_bankyear_unins_split_key_results_`run_date'.csv"

tempfile coefdta keydta
tempname ph_coefs ph_key

postfile `ph_coefs' str24 spec str32 subgroup int row_order str80 display ///
    str120 raw_term double coef double se double tstat double pval str5 stars ///
    long N double r2a double r2_within double df_r using `coefdta', replace

postfile `ph_key' str24 spec str32 result str32 group double coef double se ///
    double tstat double pval str5 stars double Fstat double df_num double df_den ///
    long N double r2a double r2_within double cut_low double cut_high ///
    double mean_unins double sd_unins long N_total long N_target ///
    long N_comparator long N_unique_bankyears str180 note using `keydta', replace

/*==============================================================================
  Run requested split specifications.
==============================================================================*/

foreach spec in median tercile_tails {
    local splitvar "split_`spec'"
    local cut_low = `med'
    local cut_high = `med'
    if "`spec'" == "tercile_tails" {
        local cut_low = `terc_low'
        local cut_high = `terc_high'
    }

    quietly count if !missing(`splitvar')
    local N_total = r(N)
    quietly count if `splitvar' == 1
    local N_target = r(N)
    quietly count if `splitvar' == 0
    local N_comp = r(N)

    di as result _n "Running `spec' low-uninsured/high-insured subsample..."
    reghdfe `dv' `INTERACT_BASE' `FULL_CTRL' if `splitvar' == 1, ///
        absorb(bank_year borrower_year) vce(cluster bank_id quarter_id)
    post_all_coefs `ph_coefs' "`spec'" "low_unins_high_insured"
    local b_t = _b[`triple']
    local se_t = _se[`triple']
    local df_t = e(df_r)
    local t_t = `b_t' / `se_t'
    local p_t = 2 * ttail(`df_t', abs(`t_t'))
    local stars_t ""
    if `p_t' < 0.01 local stars_t "***"
    else if `p_t' < 0.05 local stars_t "**"
    else if `p_t' < 0.10 local stars_t "*"
    post `ph_key' ("`spec'") ("subsample_triple") ("low_unins_high_insured") ///
        (`b_t') (`se_t') (`t_t') (`p_t') ("`stars_t'") ///
        (.) (.) (`df_t') (e(N)) (e(r2_a)) (e(r2_within)) ///
        (`cut_low') (`cut_high') (`mean_by') (`sd_by') ///
        (`N_total') (`N_target') (`N_comp') (`N_unique_bankyears') ///
        ("Separate low-uninsured/high-insured subsample regression")

    di as result _n "Running `spec' high-uninsured/low-insured subsample..."
    reghdfe `dv' `INTERACT_BASE' `FULL_CTRL' if `splitvar' == 0, ///
        absorb(bank_year borrower_year) vce(cluster bank_id quarter_id)
    post_all_coefs `ph_coefs' "`spec'" "high_unins_low_insured"
    local b_c = _b[`triple']
    local se_c = _se[`triple']
    local df_c = e(df_r)
    local t_c = `b_c' / `se_c'
    local p_c = 2 * ttail(`df_c', abs(`t_c'))
    local stars_c ""
    if `p_c' < 0.01 local stars_c "***"
    else if `p_c' < 0.05 local stars_c "**"
    else if `p_c' < 0.10 local stars_c "*"
    post `ph_key' ("`spec'") ("subsample_triple") ("high_unins_low_insured") ///
        (`b_c') (`se_c') (`t_c') (`p_c') ("`stars_c'") ///
        (.) (.) (`df_c') (e(N)) (e(r2_a)) (e(r2_within)) ///
        (`cut_low') (`cut_high') (`mean_by') (`sd_by') ///
        (`N_total') (`N_target') (`N_comp') (`N_unique_bankyears') ///
        ("Separate high-uninsured/low-insured subsample regression")

    tempvar g_bankyear g_borroweryear
    egen long `g_bankyear' = group(`splitvar' bank_year) if !missing(`splitvar')
    egen long `g_borroweryear' = group(`splitvar' borrower_year) if !missing(`splitvar')

    local INTERACT_POOL "i.`splitvar'##(c.hhi_std_y##c.`mp'##c.`risk')"
    local CTRL_POOL "i.`splitvar'##(c.log_tenor_maturity c.log_n_lenders c.log_bank_assets c.call_capital_ratio c.call_deposits_to_assets c.call_loans_to_assets c.log_sales_t c.tangibility c.ebitda_margin c.leverage c.macro_cpi_l1 c.macro_gdp_growth_l1)"
    local CTRL_POOL_IND "i.`splitvar'##i.is_credit_line i.`splitvar'##i.secured_flag i.`splitvar'##i.purp_corpgen i.`splitvar'##i.purp_debtrepay i.`splitvar'##i.purp_takeover"
    local diffterm "1.`splitvar'#c.hhi_std_y#c.`mp'#c.`risk'"

    di as result _n "Running `spec' pooled fully interacted Wald model..."
    reghdfe `dv' `INTERACT_POOL' `CTRL_POOL' `CTRL_POOL_IND' if !missing(`splitvar'), ///
        absorb(`g_bankyear' `g_borroweryear') vce(cluster bank_id quarter_id)

    local b_base = _b[`triple']
    local se_base = _se[`triple']
    local df_pool = e(df_r)
    local t_base = `b_base' / `se_base'
    local p_base = 2 * ttail(`df_pool', abs(`t_base'))

    local b_diff = _b[`diffterm']
    local se_diff = _se[`diffterm']
    local t_diff = `b_diff' / `se_diff'
    local p_diff_t = 2 * ttail(`df_pool', abs(`t_diff'))
    local stars_diff ""
    if `p_diff_t' < 0.01 local stars_diff "***"
    else if `p_diff_t' < 0.05 local stars_diff "**"
    else if `p_diff_t' < 0.10 local stars_diff "*"

    test `diffterm' = 0
    local F_diff = r(F)
    local p_wald = r(p)
    local df_num = r(df)
    local df_den = r(df_r)

    lincom `triple' + `diffterm'
    local b_pool_target = r(estimate)
    local se_pool_target = r(se)
    local t_pool_target = r(t)
    local p_pool_target = r(p)

    post `ph_key' ("`spec'") ("wald_diff_low_minus_high") ("low_minus_high") ///
        (`b_diff') (`se_diff') (`t_diff') (`p_wald') ("`stars_diff'") ///
        (`F_diff') (`df_num') (`df_den') (e(N)) (e(r2_a)) (e(r2_within)) ///
        (`cut_low') (`cut_high') (`mean_by') (`sd_by') ///
        (`N_total') (`N_target') (`N_comp') (`N_unique_bankyears') ///
        ("Pooled fully interacted model; H0: low-uninsured triple equals high-uninsured triple")

    post `ph_key' ("`spec'") ("pooled_total_triple") ("low_unins_high_insured") ///
        (`b_pool_target') (`se_pool_target') (`t_pool_target') (`p_pool_target') ("") ///
        (.) (.) (`df_pool') (e(N)) (e(r2_a)) (e(r2_within)) ///
        (`cut_low') (`cut_high') (`mean_by') (`sd_by') ///
        (`N_total') (`N_target') (`N_comp') (`N_unique_bankyears') ///
        ("Low-uninsured total triple from pooled model")

    post `ph_key' ("`spec'") ("pooled_base_triple") ("high_unins_low_insured") ///
        (`b_base') (`se_base') (`t_base') (`p_base') ("") ///
        (.) (.) (`df_pool') (e(N)) (e(r2_a)) (e(r2_within)) ///
        (`cut_low') (`cut_high') (`mean_by') (`sd_by') ///
        (`N_total') (`N_target') (`N_comp') (`N_unique_bankyears') ///
        ("High-uninsured base triple from pooled model")

    drop `g_bankyear' `g_borroweryear'
}

postclose `ph_coefs'
postclose `ph_key'

preserve
use `coefdta', clear
sort spec row_order subgroup raw_term
export delimited using "`pf_coefs'", replace
restore

preserve
use `keydta', clear
sort spec result group
export delimited using "`pf_key'", replace
restore

di as result _n "Full coefficient table saved to: `pf_coefs'"
di as result "Key results and Wald test saved to: `pf_key'"

timer off 1
timer list 1
di as result _n "Script 206 completed."
