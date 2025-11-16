"""
etl/py_etl_parquet.py

Chunked pandas -> pyarrow Parquet ingestion with partitioning.
Auto-discovers cleaned CSVs under project root unless CLEANED_CSV is set.
Partitions by year/month and a location column (configurable).
Writes parquet dataset with pyarrow.parquet.write_to_dataset so partitions
are created as folder structure like:
  data_lake/epa_o3_parquet/year=2020/month=01/region=Kerala/part-*.parquet
"""

import os
from pathlib import Path
import sys
import logging
import argparse
import math
from typing import Optional, List

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# ---------- USER CONFIG ----------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
# If you want to force a specific cleaned CSV, set CLEANED_CSV to its absolute path:
CLEANED_CSV: Optional[Path] = None
OUT_DIR = PROJECT_ROOT / "data_lake" / "epa_o3_parquet"
# Column names (after normalization). If your cleaned CSV uses different names,
# set these accordingly.
DATE_COL_CANDIDATES = ["date", "measurement_date", "dt"]
LOCATION_COL_CANDIDATES = ["site_id", "region", "country", "site", "location"]
MEASURE_COL_KEYWORDS = ["o3", "ozone"]
# Chunking for pandas.read_csv
CHUNK_SIZE = 200_000   # tune downward if you get OOM
# -------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger("py_etl_parquet")


def find_cleaned_csv(root: Path) -> List[Path]:
    patterns = ["Cleaned_*.csv", "*Cleaned*.csv", "*epa*.csv", "*o3*.csv"]
    found = []
    for pat in patterns:
        found.extend(root.glob(pat))
    # return unique sorted list
    seen = set()
    out = []
    for p in sorted(found, key=lambda p: (len(p.parts), str(p))):
        if p.resolve() not in seen:
            out.append(p)
            seen.add(p.resolve())
    return out


def choose_column(cols: List[str], candidates: List[str]) -> Optional[str]:
    lcols = [c.lower() for c in cols]
    for cand in candidates:
        if cand in lcols:
            # return actual column name with original case
            return cols[lcols.index(cand)]
    return None


def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    # strip whitespace and lowercase columns
    mapping = {c: c.strip() for c in df.columns}
    df = df.rename(columns=mapping)
    return df


def to_pyarrow_table(df: pd.DataFrame) -> pa.Table:
    # Convert pandas df to pyarrow Table with safe conversion
    try:
        return pa.Table.from_pandas(df, preserve_index=False)
    except Exception as e:
        # fallback: coerce object columns to string
        for c in df.select_dtypes(include="object").columns:
            df[c] = df[c].astype("string")
        return pa.Table.from_pandas(df, preserve_index=False)


def write_chunk_to_dataset(table: pa.Table, root_path: str, partition_cols: List[str]):
    """
    Use pyarrow parquet.write_to_dataset to append a chunk into a partitioned dataset.
    If dataset doesn't exist, it will be created.
    """
    pq.write_to_dataset(
        table,
        root_path=root_path,
        partition_cols=partition_cols,
        filesystem=None,  # local filesystem
        use_threads=True
    )


def process_csv(path: Path, out_dir: Path, chunk_size=CHUNK_SIZE):
    log.info(f"Processing CSV: {path}")
    first_chunk = True
    total_rows = 0

    # pandas read in chunks
    for chunk_no, chunk in enumerate(pd.read_csv(path, chunksize=chunk_size, low_memory=False)):
        log.info(f"Read chunk #{chunk_no} rows={len(chunk)}")
        chunk = normalize_cols(chunk)

        # normalize column names to lower-case references for searching
        cols = list(chunk.columns)
        lcols = [c.lower() for c in cols]

        # find date col
        date_col = choose_column(cols, DATE_COL_CANDIDATES)
        if not date_col:
            raise ValueError("No date column found among candidates: " + ", ".join(DATE_COL_CANDIDATES))
        # ensure date format
        chunk[date_col] = pd.to_datetime(chunk[date_col], errors="coerce")
        # drop rows without date
        chunk = chunk.dropna(subset=[date_col])
        # create year/month columns
        chunk["year"] = chunk[date_col].dt.year.astype("Int64")
        chunk["month"] = chunk[date_col].dt.month.astype("Int64")

        # choose location column
        loc_col = choose_column(cols, LOCATION_COL_CANDIDATES)
        if loc_col:
            # convert to string category for partitions
            chunk[loc_col] = chunk[loc_col].astype("string")
        else:
            # create synthetic location value so partitioning by location still works
            loc_col = None

        # measurement columns detection (not necessary but informative)
        measure_cols = [c for c in cols if any(k in c.lower() for k in MEASURE_COL_KEYWORDS)]
        if not measure_cols:
            log.warning("No obvious measurement columns found (o3/ozone). Proceeding with available columns.")

        # decide partition columns (year, month, location if exists)
        partition_cols = ["year", "month"]
        if loc_col:
            partition_cols.append(loc_col)

        # prepare arrow table
        # remove pandas extension dtypes for pyarrow if required: convert Int64 to int or nullable strings
        # convert pandas Int64 (nullable) to int64 with nulls handled by pyarrow
        for ic in chunk.select_dtypes(include=["Int64"]).columns:
            chunk[ic] = chunk[ic].astype("Int64")

        table = to_pyarrow_table(chunk)

        # write chunk into partitioned dataset
        write_chunk_to_dataset(table, str(out_dir), partition_cols)
        rows = len(chunk)
        total_rows += rows
        log.info(f"Wrote chunk #{chunk_no} -> rows={rows} total_rows={total_rows}")

    log.info(f"Finished processing {path}. Total rows written: {total_rows}")


def main(explicit_csv: Optional[str] = None, out_dir: Optional[str] = None):
    if explicit_csv:
        csv_path = Path(explicit_csv)
        if not csv_path.exists():
            raise FileNotFoundError(f"Explicit CSV not found: {csv_path}")
        csv_list = [csv_path]
    elif CLEANED_CSV:
        csv_list = [Path(CLEANED_CSV)]
    else:
        found = find_cleaned_csv(PROJECT_ROOT)
        if not found:
            raise FileNotFoundError(f"No cleaned CSV found under {PROJECT_ROOT}. Put your Cleaned_*.csv in project root or pass --csv.")
        csv_list = [found[0]]
        log.info(f"Auto-discovered cleaned CSV: {csv_list[0]}")

    out_dir = Path(out_dir) if out_dir else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"Output dataset root: {out_dir}")

    for csv in csv_list:
        process_csv(csv, out_dir)

    log.info("All done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chunked CSV -> partitioned Parquet dataset (pyarrow)")
    parser.add_argument("--csv", help="Explicit CSV path (optional)")
    parser.add_argument("--out", help="Output dataset root (optional)")
    args = parser.parse_args()
    main(args.csv, args.out)
