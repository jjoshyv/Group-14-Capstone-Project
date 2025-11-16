#!/usr/bin/env python3
"""
etl/feature_engineering.py

Robust feature engineering entrypoint for time series sensor data.

Usage:
    python etl/feature_engineering.py \
        --input "<path-or-filename-or-glob>" \
        --value-col "03_ug_m3" \
        --date-col "date" \
        --location-col "location" \
        --out-root "data_lake/feature_sets" \
        --rolling-window-months 60
"""

from __future__ import annotations
import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional, Sequence
import tempfile
import shutil
import itertools

import pandas as pd
import numpy as np

# Optional import for parquet writing
try:
    import pyarrow  # noqa: F401
except Exception:
    # pyarrow optional; pandas fallback will try fastparquet / pyarrow
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s - %(message)s"
)
logger = logging.getLogger("feature_engineering")


# -------------------------
# Helpers: input resolution
# -------------------------
def resolve_input_path(input_str: str, cwd: Optional[Path] = None) -> Optional[Path]:
    """
    Try to resolve the provided input string to a real file path.

    Strategies (in order):
      - If input_str exists as absolute/relative path -> return
      - Try interpreting input_str as a glob pattern in cwd
      - If just a filename, look for matching files in cwd (non-recursive)
      - If not found return None
    """
    cwd = Path.cwd() if cwd is None else cwd
    p = Path(input_str)
    # direct existence (absolute or relative)
    if p.exists():
        logger.info("Input file found as given: %s", str(p))
        return p.resolve()

    # If input string contains glob chars, try glob in cwd
    if any(ch in input_str for ch in ["*", "?", "["]):
        matches = list(cwd.glob(input_str))
        if matches:
            logger.info("Found %d matches for glob '%s' under %s. Using first: %s",
                        len(matches), input_str, cwd, matches[0])
            return matches[0].resolve()

    # try as a filename inside cwd (non-recursive)
    candidate = cwd / input_str
    if candidate.exists():
        logger.info("Found file in current directory: %s", candidate)
        return candidate.resolve()

    # try to find cleaned_* style files if user passed a short name
    base = Path(input_str).name
    guess_glob = f"*{base}*"
    matches = list(cwd.glob(guess_glob))
    if matches:
        logger.warning("Input not found exactly; using match %s for pattern %s", matches[0], guess_glob)
        return matches[0].resolve()

    # last resort: search top-level for files starting with "Cleaned" and containing base
    for f in cwd.iterdir():
        if f.is_file() and "cleaned" in f.name.lower() and base.lower() in f.name.lower():
            logger.warning("Found probable input file: %s", f)
            return f.resolve()

    logger.warning("Input file not found under cwd=%s for input_str=%s", cwd, input_str)
    return None


# -------------------------
# Data reading helpers
# -------------------------
def try_read_csv(path: Path | str, parse_dates: Optional[Sequence[str]] = None, nrows: Optional[int] = None) -> Optional[pd.DataFrame]:
    """
    Read CSV robustly with fallback encodings/engines.
    Returns DataFrame or None if not found/readable.
    """
    p = Path(path)
    if not p.exists():
        logger.debug("try_read_csv: path does not exist: %s", p)
        return None

    parse_dates = list(parse_dates) if parse_dates else None

    # Try a list of encodings and parsers
    encodings = ["utf-8", "utf-8-sig", "latin1", "cp1252"]
    engines = ["python", "c"]
    for enc, eng in itertools.product(encodings, engines):
        try:
            logger.debug("Trying read_csv path=%s encoding=%s engine=%s nrows=%s", p, enc, eng, nrows)
            df = pd.read_csv(p, encoding=enc, engine=eng, parse_dates=parse_dates, nrows=nrows)
            logger.info("CSV loaded: %s (encoding=%s engine=%s rows=%d cols=%d)", p, enc, eng, len(df), len(df.columns))
            return df
        except Exception as e:
            logger.debug("read_csv failed for encoding=%s engine=%s: %s", enc, eng, e)
            continue

    logger.error("Unable to read CSV: %s (tried multiple encodings/engines)", p)
    return None


def try_read_parquet(path: Path | str) -> Optional[pd.DataFrame]:
    p = Path(path)
    if not p.exists():
        return None
    try:
        df = pd.read_parquet(p)
        logger.info("Parquet loaded: %s rows=%d cols=%d", p, len(df), len(df.columns))
        return df
    except Exception as e:
        logger.warning("Failed to read parquet %s: %s", p, e)
        return None


# -------------------------
# Column matching helpers
# -------------------------
def _normalize_colname(col: str) -> str:
    return "".join(ch for ch in col.lower() if ch.isalnum())


def find_best_column(target: str, columns: Sequence[str]) -> Optional[str]:
    """
    Find best candidate column in columns that matches target.
    Matching is case-insensitive and strips punctuation/whitespace/underscores.
    Returns matched column name or None.
    """
    if target in columns:
        return target
    target_n = _normalize_colname(target)
    # Build dict normalized -> original
    normalized = { _normalize_colname(c): c for c in columns }
    if target_n in normalized:
        return normalized[target_n]
    # If exact substring (case-insensitive) match
    target_l = target.lower()
    for c in columns:
        if target_l in c.lower() or c.lower() in target_l:
            return c
    # No match
    return None


# -------------------------
# Feature engineering logic
# -------------------------
def compute_features(df: pd.DataFrame,
                     value_col: str,
                     date_col: str = "date",
                     location_col: Optional[str] = None,
                     rolling_window_months: int = 60) -> pd.DataFrame:
    """
    Compute a small set of features:
      - rolling mean over rolling_window_months (per location if location_col provided)
      - monthly mean across locations and yearly mean across locations
    Returns a feature dataframe with partition columns year, month and location (if available).
    """
    if date_col not in df.columns:
        raise ValueError(f"Date column '{date_col}' not found in dataframe columns: {list(df.columns)}")
    if value_col not in df.columns:
        raise ValueError(f"Value column '{value_col}' not found in dataframe columns: {list(df.columns)}")

    # Ensure date dtype
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    # drop rows without date or value
    df = df.dropna(subset=[date_col, value_col]).reset_index(drop=True)

    # fill location if missing
    if location_col is None or location_col not in df.columns:
        # create fallback single-location column "location_inferred"
        df["location_inferred"] = "all"
        location_col = "location_inferred"

    # add time dimensions
    df["year"] = df[date_col].dt.year
    df["month"] = df[date_col].dt.month
    df["year_month"] = df[date_col].dt.to_period("M").astype(str)

    # compute rolling average per location
    # we will compute rolling mean using a groupby + resample-like approach
    try:
        # ensure each group sorted by date
        rolled_frames = []
        for loc, g in df.groupby(location_col):
            g = g.sort_values(date_col).reset_index(drop=True)
            # set index to date for rolling by time
            g_indexed = g.set_index(date_col).asfreq("D")  # daily frequency fill gaps
            # forward/backfill value_col missing days to avoid empty rolling windows => use NaN (we avoid fill)
            # compute monthly-window approximate in days
            window_days = max(1, rolling_window_months * 30)
            # compute rolling mean
            g_indexed[f"{value_col}_rolling_{rolling_window_months}m"] = g_indexed[value_col].rolling(window=f"{window_days}d", min_periods=1).mean()
            # reset index and recover location
            g_out = g_indexed.reset_index()
            g_out[location_col] = loc
            rolled_frames.append(g_out)
        rolled = pd.concat(rolled_frames, ignore_index=True)
        # join rolling column back to original rows based on date and location (merge)
        merged = pd.merge(df, rolled[[date_col, location_col, f"{value_col}_rolling_{rolling_window_months}m"]],
                          how="left", on=[date_col, location_col])
    except Exception as e:
        # fallback: rolling per group by integer-based window (simpler, not time aware)
        logger.warning("Time-based rolling failed; falling back to group rolling by rows: %s", e)
        def _rolling_group(g):
            return g.sort_values(date_col)[value_col].rolling(window=rolling_window_months, min_periods=1).mean()
        rolled_series = df.groupby(location_col, group_keys=False).apply(lambda g: _rolling_group(g))
        merged = df.copy()
        merged[f"{value_col}_rolling_{rolling_window_months}m"] = rolled_series.values

    # spatial aggregates
    # monthly mean across locations
    monthly = merged.groupby(pd.Grouper(key=date_col, freq="M"))[value_col].mean().reset_index().rename(columns={value_col: value_col + "_monthly_mean"})
    monthly["year"] = monthly[date_col].dt.year
    monthly["month"] = monthly[date_col].dt.month

    # yearly mean across locations
    yearly = merged.groupby(pd.Grouper(key=date_col, freq="Y"))[value_col].mean().reset_index().rename(columns={value_col: value_col + "_yearly_mean"})
    yearly["year"] = yearly[date_col].dt.year

    # return merged features (original rows + rolling column)
    return merged, monthly, yearly


# -------------------------
# File writing helpers
# -------------------------
def ensure_dir(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)


def write_parquet(df: pd.DataFrame, path: Path, partition_cols: Optional[Sequence[str]] = None):
    ensure_dir(path)
    try:
        if partition_cols:
            # write partitioned: use a temporary folder then move
            tmp = Path(tempfile.mkdtemp(prefix="fe_")) / "data"
            tmp.mkdir(parents=True, exist_ok=True)
            df.to_parquet(tmp, index=False, partition_cols=list(partition_cols))
            # move/create target dir
            if path.exists():
                shutil.rmtree(path)
            shutil.move(str(tmp), str(path))
        else:
            df.to_parquet(path, index=False)
        logger.info("Wrote Parquet: %s", path)
    except Exception as e:
        logger.error("Failed to write parquet %s: %s", path, e)
        # attempt single-file fallback
        try:
            df.to_parquet(path, index=False)
            logger.info("Wrote Parquet (fallback single-file): %s", path)
        except Exception as e2:
            logger.exception("Final parquet write failed: %s", e2)
            raise


# -------------------------
# Main CLI + pipeline
# -------------------------
def main(argv=None):
    parser = argparse.ArgumentParser(description="Feature engineering pipeline (robust file resolution & CSV parsing)")
    parser.add_argument("--input", required=True, help="Input cleaned CSV path, filename, or glob")
    parser.add_argument("--value-col", required=True, help="Name of the column containing target numeric values (e.g. '03_ug_m3')")
    parser.add_argument("--date-col", default="date", help="Name of the date column (default: date)")
    parser.add_argument("--location-col", default=None, help="Name of the location column (if not provided will infer or create 'location_inferred')")
    parser.add_argument("--out-root", default="data_lake/feature_sets", help="Output root folder")
    parser.add_argument("--rolling-window-months", type=int, default=60, help="Rolling window in months for rolling features")
    parser.add_argument("--preview", action="store_true", help="If set, read only first N rows (for debugging)")
    args = parser.parse_args(argv)

    cwd = Path.cwd()
    logger.info("Detected project root: %s", cwd)
    in_path = resolve_input_path(args.input, cwd=cwd)
    if in_path is None:
        logger.error("Input path could not be resolved. Please check the file name or run from the correct working directory.")
        sys.exit(2)

    # attempt reading CSV
    df = try_read_csv(in_path, parse_dates=[args.date_col] if args.date_col else None, nrows=(5 if args.preview else None))
    if df is None:
        logger.error("Failed to read input CSV: %s", in_path)
        sys.exit(3)

    # If preview mode, re-read full if needed
    if args.preview:
        df_full = try_read_csv(in_path, parse_dates=[args.date_col] if args.date_col else None)
        if df_full is not None:
            df = df_full

    # Normalize and find columns
    # fix common problems: spaces, uppercase etc.
    col_value = find_best_column(args.value_col, df.columns)
    col_date = find_best_column(args.date_col, df.columns) if args.date_col else None
    col_location = find_best_column(args.location_col, df.columns) if args.location_col else None

    if col_value is None:
        # attempt to find a numeric column automatically if only one numeric column present
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if len(numeric_cols) == 1:
            col_value = numeric_cols[0]
            logger.warning("Value column not found; using only numeric column: %s", col_value)
        else:
            logger.error("Value column '%s' not found and cannot auto-select. Columns: %s", args.value_col, list(df.columns))
            sys.exit(4)

    if col_date is None:
        # attempt to auto-detect date-like columns
        date_candidates = [c for c in df.columns if "date" in c.lower() or "time" in c.lower() or c.lower() in ("ds", "timestamp")]
        col_date = date_candidates[0] if date_candidates else None
        if col_date:
            logger.warning("Date column not provided or not found; using inferred: %s", col_date)
        else:
            logger.error("Date column not found. Columns: %s", list(df.columns))
            sys.exit(5)

    if col_location is None:
        # fallback: infer 'location' or 'station' or create location_inferred later
        loc_candidates = [c for c in df.columns if any(k in c.lower() for k in ("loc", "station", "site", "region", "city"))]
        col_location = loc_candidates[0] if loc_candidates else None
        if col_location:
            logger.warning("Location column inferred as: %s", col_location)
        else:
            logger.info("No location column found; pipeline will create a single 'location_inferred' for aggregation.")

    # Clean up column names mapping (preserve original columns in df)
    logger.info("Using columns -> date: %s, value: %s, location: %s", col_date, col_value, col_location)

    # Ensure date parsing
    try:
        df[col_date] = pd.to_datetime(df[col_date], errors="coerce")
    except Exception:
        # leave as-is; compute_features will coerce
        pass

    # run feature computation
    try:
        features_df, monthly_df, yearly_df = compute_features(
            df,
            value_col=col_value,
            date_col=col_date,
            location_col=col_location,
            rolling_window_months=args.rolling_window_months
        )
    except Exception as e:
        logger.exception("Feature computation failed: %s", e)
        sys.exit(6)

    # Prepare outputs
    out_root = Path(args.out_root)
    features_path = out_root / "features.parquet"
    monthly_path = out_root / "monthly_region.parquet"
    yearly_path = out_root / "yearly_region.parquet"
    manifest_path = out_root / "feature_ingest_metadata.json"
    try:
        # Write features (single-file parquet)
        ensure_dir(features_path)
        features_df.to_parquet(features_path, index=False)
        logger.info("Wrote features.parquet: %s", features_path)

        # monthly/yearly parquet (single-file)
        ensure_dir(monthly_path)
        monthly_df.to_parquet(monthly_path, index=False)
        logger.info("Wrote monthly parquet: %s", monthly_path)

        ensure_dir(yearly_path)
        yearly_df.to_parquet(yearly_path, index=False)
        logger.info("Wrote yearly parquet: %s", yearly_path)
    except Exception as e:
        logger.exception("Parquet write failed: %s", e)
        sys.exit(7)

    # write metadata manifest
    manifest = {
        "input": str(in_path),
        "rows_input": int(len(df)),
        "rows_features": int(len(features_df)),
        "value_col": col_value,
        "date_col": col_date,
        "location_col": col_location if col_location else "location_inferred",
        "partition_cols": ["year", "month", col_location if col_location else "location_inferred"],
        "rolling_window_months": args.rolling_window_months,
    }
    ensure_dir(manifest_path)
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)
    logger.info("Wrote metadata manifest: %s", manifest_path)
    logger.info("Feature engineering completed. Outputs in: %s", out_root)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
