#!/usr/bin/env python3
"""
Download WRDS inputs required for Distance-to-Default construction.

Outputs:
  - data/raw/ccm/ccmxpf_lnkhist__2001_2025.parquet
  - data/raw/crsp/dsf/dsf_YYYY__borrower_permnos.parquet
  - data/raw/crsp/msf/msf_YYYY__borrower_permnos.parquet
  - data/raw/monetary_policy/ff_factors_daily__2001_2025.parquet (if accessible)
  - data/raw/monetary_policy/ff_factors_monthly__2001_2025.parquet (if accessible)
  - data/documentation/wrds_dd_download_summary_YYYY-MM-DD.csv
"""

from __future__ import annotations

import argparse
import os
from datetime import date
from pathlib import Path
from typing import Iterable

import pandas as pd
import psycopg2


BASE = Path(os.environ.get("MPHIL_THESIS_ROOT", Path(__file__).resolve().parents[2]))
COMP_FUNDQ = BASE / "data/raw/compustat/fundq__2001_2025.parquet"
OUT_CCM = BASE / "data/raw/ccm/ccmxpf_lnkhist__2001_2025.parquet"
OUT_DSF_DIR = BASE / "data/raw/crsp/dsf"
OUT_MSF_DIR = BASE / "data/raw/crsp/msf"
OUT_FF_D = BASE / "data/raw/monetary_policy/ff_factors_daily__2001_2025.parquet"
OUT_FF_M = BASE / "data/raw/monetary_policy/ff_factors_monthly__2001_2025.parquet"
OUT_SUMMARY = (
    BASE
    / "data/documentation"
    / f"wrds_dd_download_summary_{date.today().isoformat()}.csv"
)


def load_pgpass() -> tuple[str, str, str, str, str]:
    pgpass = Path.home() / ".pgpass"
    if not pgpass.exists():
        raise FileNotFoundError("~/.pgpass not found. Configure WRDS credentials first.")
    line = pgpass.read_text().strip().splitlines()[0]
    parts = line.split(":")
    if len(parts) < 5:
        raise ValueError("~/.pgpass first line has invalid format.")
    host, port, dbname, user, pwd = parts[:5]
    return host, port, dbname, user, pwd


def connect_wrds() -> psycopg2.extensions.connection:
    host, port, dbname, user, pwd = load_pgpass()
    return psycopg2.connect(
        host=host,
        port=port,
        dbname=dbname,
        user=user,
        password=pwd,
        sslmode="require",
    )


def query_df(conn: psycopg2.extensions.connection, sql: str, params: tuple | None = None) -> pd.DataFrame:
    return pd.read_sql_query(sql, conn, params=params)


def chunked(seq: list[int], n: int) -> Iterable[list[int]]:
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def load_gvkeys() -> list[str]:
    if not COMP_FUNDQ.exists():
        raise FileNotFoundError(f"Missing Compustat fundamentals file: {COMP_FUNDQ}")
    df = pd.read_parquet(COMP_FUNDQ, columns=["gvkey"])
    vals = (
        df["gvkey"]
        .dropna()
        .astype(str)
        .str.strip()
    )
    vals = vals[vals.ne("")]
    return sorted(vals.unique().tolist())


def fetch_ccm(conn: psycopg2.extensions.connection, gvkeys: list[str]) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for gchunk in chunked(gvkeys, 5000):
        sql = """
            select
                gvkey,
                lpermno,
                lpermco,
                linktype,
                linkprim,
                linkdt,
                linkenddt
            from crsp.ccmxpf_lnkhist
            where gvkey = any(%s)
        """
        part = query_df(conn, sql, (gchunk,))
        parts.append(part)
    ccm = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if ccm.empty:
        return ccm
    ccm["gvkey"] = ccm["gvkey"].astype(str).str.strip()
    ccm["lpermno"] = pd.to_numeric(ccm["lpermno"], errors="coerce").astype("Int64")
    ccm["linkdt"] = pd.to_datetime(ccm["linkdt"], errors="coerce")
    ccm["linkenddt"] = pd.to_datetime(ccm["linkenddt"], errors="coerce")
    return ccm


def fetch_crsp_year(
    conn: psycopg2.extensions.connection,
    table: str,
    permnos: list[int],
    year: int,
) -> pd.DataFrame:
    # Minimal fields for DD construction (price/return/shares and adjustments).
    cols = "permno, date, ret, prc, shrout, cfacpr, cfacshr, vol"
    sql = f"""
        select {cols}
        from crsp.{table}
        where permno = any(%s)
          and date >= %s::date
          and date < %s::date
    """
    start = f"{year}-01-01"
    end = f"{year + 1}-01-01"
    out = query_df(conn, sql, (permnos, start, end))
    if out.empty:
        return out
    out["permno"] = pd.to_numeric(out["permno"], errors="coerce").astype("Int64")
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    for c in ["ret", "prc", "shrout", "cfacpr", "cfacshr", "vol"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def fetch_ff(conn: psycopg2.extensions.connection, table: str, start_year: int, end_year: int) -> pd.DataFrame:
    # WRDS FF library is commonly "ff".
    sql = f"""
        select *
        from ff.{table}
        where date >= %s::date
          and date < %s::date
    """
    return query_df(conn, sql, (f"{start_year}-01-01", f"{end_year + 1}-01-01"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-year", type=int, default=2001)
    ap.add_argument("--end-year", type=int, default=2025)
    args = ap.parse_args()

    OUT_CCM.parent.mkdir(parents=True, exist_ok=True)
    OUT_DSF_DIR.mkdir(parents=True, exist_ok=True)
    OUT_MSF_DIR.mkdir(parents=True, exist_ok=True)
    OUT_SUMMARY.parent.mkdir(parents=True, exist_ok=True)

    gvkeys = load_gvkeys()
    print(f"Compustat gvkeys loaded: {len(gvkeys):,}", flush=True)

    conn = connect_wrds()
    summary_rows: list[dict[str, object]] = []
    try:
        ccm = fetch_ccm(conn, gvkeys)
        ccm.to_parquet(OUT_CCM, index=False)
        print(f"Saved CCM: {OUT_CCM} ({len(ccm):,} rows)", flush=True)
        summary_rows.append({"dataset": "ccm_link", "rows": len(ccm), "path": str(OUT_CCM)})

        if ccm.empty:
            raise RuntimeError("CCM link download returned zero rows; cannot continue with CRSP pulls.")

        ccm_keep = ccm[
            ccm["linktype"].isin(["LC", "LU", "LS", "LD", "LN"]) &
            ccm["linkprim"].isin(["P", "C"])
        ].copy()
        permnos = (
            ccm_keep["lpermno"]
            .dropna()
            .astype(int)
            .drop_duplicates()
            .sort_values()
            .tolist()
        )
        print(f"Borrower-linked CRSP permnos: {len(permnos):,}", flush=True)
        summary_rows.append({"dataset": "borrower_permnos", "rows": len(permnos), "path": "derived from ccm"})

        for yr in range(args.start_year, args.end_year + 1):
            p_dsf = OUT_DSF_DIR / f"dsf_{yr}__borrower_permnos.parquet"
            if p_dsf.exists():
                dsf_rows = len(pd.read_parquet(p_dsf, columns=["permno"]))
                print(f"Skip DSF {yr} (exists): {dsf_rows:,} rows", flush=True)
                summary_rows.append({"dataset": f"crsp_dsf_{yr}", "rows": dsf_rows, "path": str(p_dsf)})
            else:
                dsf = fetch_crsp_year(conn, "dsf", permnos, yr)
                dsf.to_parquet(p_dsf, index=False)
                print(f"Saved DSF {yr}: {len(dsf):,} rows", flush=True)
                summary_rows.append({"dataset": f"crsp_dsf_{yr}", "rows": len(dsf), "path": str(p_dsf)})

            p_msf = OUT_MSF_DIR / f"msf_{yr}__borrower_permnos.parquet"
            if p_msf.exists():
                msf_rows = len(pd.read_parquet(p_msf, columns=["permno"]))
                print(f"Skip MSF {yr} (exists): {msf_rows:,} rows", flush=True)
                summary_rows.append({"dataset": f"crsp_msf_{yr}", "rows": msf_rows, "path": str(p_msf)})
            else:
                msf = fetch_crsp_year(conn, "msf", permnos, yr)
                msf.to_parquet(p_msf, index=False)
                print(f"Saved MSF {yr}: {len(msf):,} rows", flush=True)
                summary_rows.append({"dataset": f"crsp_msf_{yr}", "rows": len(msf), "path": str(p_msf)})

        # Fama-French factors (RF included). Non-fatal if access is unavailable.
        try:
            ff_d = fetch_ff(conn, "factors_daily", args.start_year, args.end_year)
            ff_d.to_parquet(OUT_FF_D, index=False)
            print(f"Saved FF daily factors: {len(ff_d):,} rows", flush=True)
            summary_rows.append({"dataset": "ff_factors_daily", "rows": len(ff_d), "path": str(OUT_FF_D)})
        except Exception as exc:
            print(f"Skipped ff.factors_daily ({type(exc).__name__}): {exc}", flush=True)
            summary_rows.append({"dataset": "ff_factors_daily", "rows": -1, "path": "SKIPPED"})

        try:
            ff_m = fetch_ff(conn, "factors_monthly", args.start_year, args.end_year)
            ff_m.to_parquet(OUT_FF_M, index=False)
            print(f"Saved FF monthly factors: {len(ff_m):,} rows", flush=True)
            summary_rows.append({"dataset": "ff_factors_monthly", "rows": len(ff_m), "path": str(OUT_FF_M)})
        except Exception as exc:
            print(f"Skipped ff.factors_monthly ({type(exc).__name__}): {exc}", flush=True)
            summary_rows.append({"dataset": "ff_factors_monthly", "rows": -1, "path": "SKIPPED"})

    finally:
        conn.close()

    pd.DataFrame(summary_rows).to_csv(OUT_SUMMARY, index=False)
    print(f"Saved summary: {OUT_SUMMARY}", flush=True)


if __name__ == "__main__":
    main()
