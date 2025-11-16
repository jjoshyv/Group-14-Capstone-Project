#!/usr/bin/env python3
"""Clean EPA O3 CSV files and produce a monthly cleaned CSV.

Usage:
    python 01_clean_epa_o3.py --input-glob "EPAair_O3_*_raw.csv" --out "data/cleaned/Cleaned_EPA_O3_Monthly.csv"
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import glob
import os
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PPM_TO_UGM3_O3 = 2140.0  # approximate conversion factor for O3

def convert_to_ug_m3(value, unit):
    try:
        if isinstance(unit, str) and 'ppm' in unit.lower():
            return float(value) * PPM_TO_UGM3_O3
        return float(value)
    except Exception:
        return None

def load_and_concat(glob_pattern: str) -> pd.DataFrame:
    files = sorted(glob.glob(glob_pattern))
    if not files:
        raise FileNotFoundError(f"No files match pattern: {glob_pattern}")
    frames = []
    for f in files:
        df = pd.read_csv(f)
        df['source_file'] = os.path.basename(f)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)

def clean_epa(df: pd.DataFrame) -> pd.DataFrame:
    cols = {c.lower(): c for c in df.columns}
    date_col = next((cols[k] for k in cols if 'date' in k), None)
    value_col = next((cols[k] for k in cols if any(x in k for x in ['arithmetic mean','mean','o3','value'])), None)
    unit_col = next((cols[k] for k in cols if 'unit' in k or 'measure' in k), None)

    if date_col is None or value_col is None:
        raise ValueError("Could not detect date or value columns automatically. Inspect the dataframe columns.")

    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=[date_col, value_col])
    if unit_col and unit_col in df.columns:
        df['O3_ug_m3'] = df.apply(lambda r: convert_to_ug_m3(r[value_col], r[unit_col]), axis=1)
    else:
        df['O3_ug_m3'] = pd.to_numeric(df[value_col], errors='coerce')

    df = df.dropna(subset=['O3_ug_m3'])
    df = df.rename(columns={date_col: 'Date'})
    df = df[['Date', 'O3_ug_m3']]
    return df

def aggregate_monthly(df: pd.DataFrame) -> pd.DataFrame:
    df = df.set_index('Date').sort_index()
    monthly = df['O3_ug_m3'].resample('M').mean().reset_index()
    monthly['Date'] = monthly['Date'].dt.to_period('M').dt.to_timestamp()
    return monthly

def parse_args():
    p = argparse.ArgumentParser(description="Clean EPA O3 CSVs and produce monthly CSV.")
    p.add_argument("--input-glob", default="EPAair_O3_*_raw.csv", help="Glob pattern for raw EPA CSVs")
    p.add_argument("--out", default="data/cleaned/Cleaned_EPA_O3_Monthly.csv", help="Output CSV path")
    return p.parse_args()

def main():
    args = parse_args()
    outpath = Path(args.out)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Loading files with pattern: %s", args.input_glob)
    df = load_and_concat(args.input_glob)
    logger.info("Raw rows loaded: %d", len(df))
    cleaned = clean_epa(df)
    monthly = aggregate_monthly(cleaned)
    monthly.to_csv(outpath, index=False)
    logger.info("Saved cleaned monthly data to: %s (rows=%d)", outpath, len(monthly))

if __name__ == "__main__":
    main()
