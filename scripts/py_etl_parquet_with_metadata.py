#!/usr/bin/env python3
"""
py_etl_parquet_with_metadata.py

Simple chunked CSV -> partitioned Parquet ETL with:
 - case-insensitive header normalization (maps "Date" -> "date" usage)
 - required-column validation (strict mode)
 - optional auto-creation of location column with default
 - partitioning by year/month and optional location column
 - writes metadata JSON and logs to a small metadata sqlite DB (or duckdb if available)

Usage examples:
    python etl/py_etl_parquet_with_metadata.py
    python etl/py_etl_parquet_with_metadata.py --csv "Cleaned_EPA_O3_Monthly.csv" --date-col "Date" --location-col region --auto-create-location
    python etl/py_etl_parquet_with_metadata.py --csv "Cleaned_EPA_O3_Monthly.csv" --date-col "Date" --strict
"""

from __future__ import annotations
import argparse
import json
import logging
import os
import sqlite3
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# try duckdb if present (optional)
try:
    import duckdb  # type: ignore
    HAVE_DUCKDB = True
except Exception:
    HAVE_DUCKDB = False

# ------------------------------
# Logging
# ------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ------------------------------
# Helpers: column normalization
# ------------------------------
def normalize_df_columns(df: pd.DataFrame) -> None:
    """
    Strip whitespace from column names and normalize them (but keep original case mapping).
    This modifies df in place.
    """
    new_cols = []
    for c in df.columns:
        if isinstance(c, str):
            new_cols.append(c.strip())
        else:
            new_cols.append(c)
    df.columns = new_cols


def find_actual_columns(df: pd.DataFrame, requested: List[Optional[str]]) -> Dict[Optional[str], Optional[str]]:
    """
    For each requested column name (may be None), find the actual column name in df case-insensitively.
    Returns mapping requested -> actual_name (or None if not found).
    """
    normalize_df_columns(df)
    lower_map = {str(c).lower(): c for c in df.columns if isinstance(c, str)}
    resolved: Dict[Optional[str], Optional[str]] = {}
    for req in requested:
        if req is None:
            resolved[req] = None
            continue
        # exact match first
        if req in df.columns:
            resolved[req] = req
            continue
        # case-insensitive
        lower = req.lower()
        resolved[req] = lower_map.get(lower)
    return resolved

# ------------------------------
# Schema validation
# ------------------------------
def validate_schema(df: pd.DataFrame, required_cols: List[str], case_sensitive: bool = False) -> Tuple[bool, List[str]]:
    """
    Check df for required columns. Return (is_valid, missing_list).
    If case_sensitive=False (default) then use case-insensitive matching.
    """
    normalize_df_columns(df)
    if case_sensitive:
        existing = set(df.columns)
        missing = [c for c in required_cols if c not in existing]
        return (len(missing) == 0, missing)
    # case-insensitive
    lower_map = {str(c).lower() for c in df.columns if isinstance(c, str)}
    missing = [c for c in required_cols if str(c).lower() not in lower_map]
    return (len(missing) == 0, missing)

# ------------------------------
# Partition helpers
# ------------------------------
def ensure_date_columns(df: pd.DataFrame, real_date_col: str) -> pd.DataFrame:
    """
    Ensure date parsing; create year and month columns (strings, zero padded month).
    Modifies and returns df.
    """
    # parse date column robustly - allow pandas to infer format
    df = df.copy()
    df[real_date_col] = pd.to_datetime(df[real_date_col], errors="coerce")
    if df[real_date_col].isnull().any():
        log.debug("Some date values failed to parse (they will be NaT).")
    df["year"] = df[real_date_col].dt.year.astype("Int64")
    df["month"] = df[real_date_col].dt.month.astype("Int64")
    # convert to simple strings for partitioning (avoid pandas nullable ints in pyarrow)
    df["year"] = df["year"].astype("Int64").astype("str").replace("<NA>", "unknown")
    df["month"] = df["month"].astype("Int64").astype("str").replace("<NA>", "unknown")
    # zero-pad months to 2 digits when numeric
    def pad_month(m):
        try:
            mi = int(m)
            return f"{mi:02d}"
        except Exception:
            return m
    df["month"] = df["month"].apply(pad_month)
    return df

# ------------------------------
# Parquet writing (pyarrow)
# ------------------------------
def write_parquet_partitioned(df: pd.DataFrame, out_root: Path, dataset_name: str, partition_cols: List[str]) -> None:
    """
    Write the pandas DataFrame to parquet using pyarrow, partitioned by partition_cols.
    Will append files (pyarrow.write_to_dataset will create files under out_root/dataset_name).
    """
    if df.empty:
        log.info("Chunk is empty; nothing to write.")
        return

    dest_dir = Path(out_root) / dataset_name
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Convert pandas -> pyarrow table (let pyarrow infer schema)
    table = pa.Table.from_pandas(df, preserve_index=False)

    # Write to dataset; when partition_cols contains columns not present pyarrow will fail,
    # so ensure partition_cols exist in df
    missing_part = [c for c in partition_cols if c not in df.columns]
    if missing_part:
        raise ValueError(f"Missing partition columns in DataFrame: {missing_part}")

    pq.write_to_dataset(table, root_path=str(dest_dir), partition_cols=partition_cols)
    log.info(f"Wrote chunk -> {dest_dir} (partitions: {partition_cols})")


# ------------------------------
# Metadata recording
# ------------------------------
def write_metadata_json(meta: dict, out_root: Path, dataset_name: str) -> Path:
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    meta_dir = out_root / dataset_name / "_metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    meta_path = meta_dir / f"meta_{meta['run_id']}.json"
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, default=str)
    log.info(f"Metadata recorded: {meta_path}")
    return meta_path


def record_metadata_sqlite(meta: dict, db_path: Path):
    """
    Record a compact metadata summary into a small sqlite DB for quick lookups later.
    If duckdb is present we'll use duckdb (works similarly), otherwise sqlite3.
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if HAVE_DUCKDB:
        try:
            conn = duckdb.connect(database=str(db_path), read_only=False)
            # create table if not exists
            conn.execute("""
                CREATE TABLE IF NOT EXISTS etl_metadata (
                    run_id VARCHAR,
                    dataset VARCHAR,
                    source_file VARCHAR,
                    rows_written INTEGER,
                    partition_cols VARCHAR,
                    location_col VARCHAR,
                    out_root VARCHAR,
                    timestamp VARCHAR
                )
            """)
            conn.execute("INSERT INTO etl_metadata VALUES (?, ?, ?, ?, ?, ?, ?, ?)", [
                meta.get("run_id"), meta.get("dataset"), meta.get("source_file"),
                meta.get("rows_written"), json.dumps(meta.get("partition_cols")),
                meta.get("location_col"), meta.get("out_root"), meta.get("timestamp")
            ])
            conn.close()
            log.info(f"Metadata recorded into DuckDB: {db_path}")
            return
        except Exception as e:
            log.warning("DuckDB write failed, falling back to sqlite3: %s", e)

    # fallback sqlite3
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS etl_metadata (
            run_id TEXT,
            dataset TEXT,
            source_file TEXT,
            rows_written INTEGER,
            partition_cols TEXT,
            location_col TEXT,
            out_root TEXT,
            timestamp TEXT
        )
    """)
    cur.execute("""
        INSERT INTO etl_metadata (run_id, dataset, source_file, rows_written, partition_cols, location_col, out_root, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        meta.get("run_id"), meta.get("dataset"), meta.get("source_file"),
        meta.get("rows_written"), json.dumps(meta.get("partition_cols")),
        meta.get("location_col"), meta.get("out_root"), meta.get("timestamp")
    ))
    conn.commit()
    conn.close()
    log.info(f"Metadata recorded into sqlite DB: {db_path}")


# ------------------------------
# Main ETL orchestration
# ------------------------------
def etl_csv_to_parquet(
    csv_path: Path,
    out_root: Path,
    dataset_name: Optional[str] = None,
    date_col: Optional[str] = "date",
    location_col: Optional[str] = None,
    auto_create_location: bool = False,
    location_default: str = "Unknown",
    partition_cols: Optional[List[str]] = None,
    chunksize: int = 200000,
    strict: bool = False,
) -> dict:
    """
    Main ETL loop:
     - read CSV in chunks
     - normalize headers and validate required columns
     - create year/month columns
     - optionally create location column with default
     - write each chunk to parquet dataset partitioned by partition_cols
    Returns metadata dict.
    """
    csv_path = Path(csv_path)
    out_root = Path(out_root)
    if dataset_name is None:
        dataset_name = csv_path.stem

    if partition_cols is None:
        # default partitioning
        partition_cols = ["year", "month"]
        if location_col is not None:
            partition_cols.append(location_col)

    run_id = str(uuid.uuid4())
    rows_written_total = 0
    meta = {
        "run_id": run_id,
        "dataset": dataset_name,
        "source_file": str(csv_path.name),
        "rows_written": 0,
        "partition_cols": partition_cols,
        "location_col": location_col,
        "out_root": str(out_root),
        "timestamp": datetime.utcnow().isoformat(),
    }

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    # read in chunks
    reader = pd.read_csv(csv_path, chunksize=chunksize)
    chunk_idx = 0
    for chunk in reader:
        chunk_idx += 1
        log.info(f"Processing chunk #{chunk_idx} -> rows={len(chunk)}")

        # normalize header and resolve requested column names case-insensitively
        resolved = find_actual_columns(chunk, [date_col, location_col])
        real_date_col = resolved.get(date_col) or date_col
        real_location_col = resolved.get(location_col)  # may be None

        # If location requested but missing:
        if location_col and real_location_col is None:
            if auto_create_location:
                log.warning(f"Location column '{location_col}' missing; creating with default '{location_default}'.")
                chunk[location_col] = location_default
                real_location_col = location_col
            elif strict:
                raise ValueError(f"Missing required location column: {location_col}")
            else:
                # we will not include location in partitions if missing and not strict
                log.warning(f"Location column '{location_col}' missing; proceeding without it (not strict).")
                if location_col in partition_cols:
                    partition_cols = [c for c in partition_cols if c != location_col]

        # Validate required columns: date is always required
        required = [real_date_col]
        if strict and location_col:
            # if strict and the user expects the location column, require it
            required.append(location_col)

        valid, missing = validate_schema(chunk, required, case_sensitive=False)
        if not valid:
            raise ValueError(f"Missing required columns: {missing}")

        # If the real_date_col is different from canonical expected name, rename to canonical 'date' for internal processing
        # (but we do not overwrite original header; we'll just work with the real name)
        if real_date_col not in chunk.columns:
            raise ValueError(f"Resolved date column '{real_date_col}' not found in chunk")

        # create year/month columns from date
        chunk = ensure_date_columns(chunk, real_date_col)

        # If location exists and its name differs from partition column entry, ensure partition list uses actual name
        # (partition_cols may have been built earlier using the requested location_col)
        if location_col and real_location_col and location_col != real_location_col:
            # replace requested location_col with real_location_col in partition_cols
            partition_cols = [real_location_col if c == location_col else c for c in partition_cols]

        # Write chunk to parquet dataset. If location column was dropped because missing, partition_cols adjusted above.
        write_parquet_partitioned(chunk, out_root, dataset_name, partition_cols)

        rows_written_total += len(chunk)

    meta["rows_written"] = rows_written_total
    # finalize metadata to include final timestamp
    meta["timestamp_end"] = datetime.utcnow().isoformat()
    return meta

# ------------------------------
# CLI
# ------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="ETL CSV -> partitioned Parquet with metadata")
    p.add_argument("--csv", "-c", default=None, help="CSV filename (auto-discover if omitted)")
    p.add_argument("--date-col", default="date", help="Name of the date column in the CSV (case-insensitive). Default 'date'")
    p.add_argument("--location-col", default=None, help="Name of location column (region/country/site_id) to partition by")
    p.add_argument("--auto-create-location", action="store_true", help="If location-col missing, create it filled with default 'Unknown'")
    p.add_argument("--location-default", default="Unknown", help="Default value when auto-creating location column")
    p.add_argument("--out-root", default="data_lake", help="Root output directory for parquet datasets")
    p.add_argument("--dataset-name", default=None, help="Optional dataset name (defaults to CSV stem)")
    p.add_argument("--chunksize", type=int, default=200000, help="CSV chunksize")
    p.add_argument("--strict", action="store_true", help="Require required columns (date and location if provided), fail if missing")
    p.add_argument("--meta-db", default="etl_metadata.db", help="Path to sqlite/duckdb metadata DB to record a summary row")
    p.add_argument("--meta-json", default=True, type=bool, help="Write per-run metadata JSON (True/False)")
    return p.parse_args()


def auto_discover_csv(root: Path) -> Optional[Path]:
    candidates = list(root.glob("Cleaned_*.csv"))
    if not candidates:
        return None
    # prefer exact match if only one or choose the newest
    candidates = sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def main():
    args = parse_args()
    project_root = Path.cwd()

    # find CSV
    csv_path = None
    if args.csv:
        csv_path = Path(args.csv)
        if not csv_path.is_absolute():
            csv_path = project_root / csv_path
    else:
        discovered = auto_discover_csv(project_root)
        if discovered:
            csv_path = discovered
            log.info(f"Auto-discovered cleaned CSV: {csv_path.name}")
        else:
            raise FileNotFoundError("No CSV provided and none discovered with pattern Cleaned_*.csv in project root.")

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    log.info(f"Processing CSV: {csv_path.name}")
    try:
        metadata = etl_csv_to_parquet(
            csv_path=csv_path,
            out_root=Path(args.out_root),
            dataset_name=args.dataset_name,
            date_col=args.date_col,
            location_col=args.location_col,
            auto_create_location=args.auto_create_location,
            location_default=args.location_default,
            chunksize=args.chunksize,
            strict=args.strict,
        )
        # write metadata JSON
        if args.meta_json:
            write_metadata_json(metadata, Path(args.out_root), metadata["dataset"])
        # record to sqlite/duckdb
        record_metadata_sqlite(metadata, Path(args.meta_db))
        log.info("ETL completed successfully.")
        log.info(json.dumps(metadata, indent=2))
    except Exception as e:
        log.error("ETL process failed: %s", e)
        # re-raise so caller sees failure code
        raise


if __name__ == "__main__":
    main()
