#!/usr/bin/env python3
"""

Compute simple linear trends by location and save:
 - per-location PNG plots (scatter + fitted line)
 - a CSV summary of slopes and statistics
 - a small JSON metadata file describing the run

Design goals:
 - Robust to column case (auto-detects common names)
 - No SciPy dependency; uses NumPy for slope + R
 - Safe path handling (relative metadata paths with fallback)

Usage examples
--------------
# Auto-discover columns (case-insensitive), default input and output:
python analysis/7_trend_analysis.py

# Explicit columns & custom out root:
python analysis/7_trend_analysis.py \
  --value-col o3_ug_m3 --date-col date --location-col location \
  --out-root analysis_outputs/trends
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# --------------------------- column helpers ---------------------------

LOCATION_CANDIDATES = [
    "location", "region", "site", "station", "state", "country", "city", "area"
]
DATE_CANDIDATES = ["date", "dt", "day", "timestamp"]
VALUE_HINTS = ["value", "target", "measurement", "concentration", "o3", "pm25", "pm2_5"]


def _ci_map(columns: List[str]) -> Dict[str, str]:
    """Build case-insensitive map of df columns -> canonical name."""
    return {c.lower(): c for c in columns}


def find_col_ci(df: pd.DataFrame, desired: Optional[str], candidates: List[str]) -> Optional[str]:
    """
    Find a column in df case-insensitively.

    Priority:
      1) explicit 'desired' (case-insensitive exact match)
      2) first candidate that matches (case-insensitive)
    Returns canonical DataFrame column name or None.
    """
    cmap = _ci_map(list(df.columns))
    if desired:
        key = desired.lower()
        if key in cmap:
            return cmap[key]
    for name in candidates:
        key = name.lower()
        if key in cmap:
            return cmap[key]
    return None


# --------------------------- trend math ---------------------------

def to_year_float(dts: pd.Series) -> np.ndarray:
    """Convert datetime series to fractional years (e.g., 2015.5)."""
    y = dts.dt.year.values.astype(float)
    doy = dts.dt.dayofyear.values.astype(float)
    return y + doy / 365.25


def fit_trend(x_years: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """
    Fit y = a*x + b using NumPy polyfit, report:
     - slope_per_year (a)
     - intercept (b)
     - r_value (Pearson)
     - stderr (SE of slope; quick OLS formula)
     - n (observations)
    """
    n = len(y)
    if n < 2:
        return {"slope": np.nan, "intercept": np.nan, "r_value": np.nan, "stderr": np.nan, "n": n}

    # Fit line
    a, b = np.polyfit(x_years, y, 1)

    # R (Pearson)
    r = np.corrcoef(x_years, y)[0, 1]

    # Std err of slope (classic OLS formula)
    y_hat = a * x_years + b
    resid = y - y_hat
    sse = np.sum(resid ** 2)
    s_xx = np.sum((x_years - x_years.mean()) ** 2)
    if n > 2 and s_xx > 0:
        stderr = np.sqrt(sse / (n - 2)) / np.sqrt(s_xx)
    else:
        stderr = np.nan

    return {"slope": float(a), "intercept": float(b), "r_value": float(r), "stderr": float(stderr), "n": int(n)}


# --------------------------- plotting ---------------------------

def plot_series_with_trend(
    dates: pd.Series,
    values: pd.Series,
    slope: float,
    intercept: float,
    out_path: Path,
    title: str
) -> None:
    """Save a simple scatter + fitted line plot."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    x = dates
    y = values

    fig = plt.figure(figsize=(9, 4.8))
    ax = plt.gca()
    ax.scatter(x, y, s=10, alpha=0.7)
    try:
        # trend line
        x_years = to_year_float(dates)
        x_line = np.linspace(x_years.min(), x_years.max(), 100)
        y_line = slope * x_line + intercept
        # convert fractional years back to datetimes for x-axis
        years_int = np.floor(x_line).astype(int)
        frac = x_line - years_int
        # approximate date: Jan 1 + frac*365 days
        line_dates = pd.to_datetime(years_int, format="%Y") + pd.to_timedelta((frac * 365.25), unit="D")
        ax.plot(line_dates, y_line, linewidth=2)
    except Exception:
        # If anything goes wrong, skip the line
        pass

    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# --------------------------- core analysis ---------------------------

def run_trend_analysis(
    input_path: Path,
    out_root: Path,
    value_col: Optional[str] = None,
    date_col: Optional[str] = None,
    location_col: Optional[str] = None,
    min_obs: int = 12,
) -> Dict:
    """
    Execute trend analysis and return a metadata dict summarizing the run.
    """
    out_root = out_root.resolve()
    plots_dir = out_root / "plots"
    out_root.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Load input
    df = pd.read_parquet(input_path)

    # Detect columns case-insensitively
    date_c = find_col_ci(df, date_col, DATE_CANDIDATES)
    if not date_c:
        raise ValueError(f"Could not find a date column (tried: {DATE_CANDIDATES} and --date-col).")

    # Heuristic: if not provided, try common value names first, else any numeric column
    value_c = find_col_ci(df, value_col, VALUE_HINTS)
    if not value_c:
        # pick the first numeric column that isn't date or location-like
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        blacklist = set([date_c] + [x for x in df.columns if x.lower() in [n.lower() for n in LOCATION_CANDIDATES]])
        num_cols = [c for c in num_cols if c not in blacklist]
        if not num_cols:
            raise ValueError("Could not find a numeric value column; specify with --value-col.")
        value_c = num_cols[0]

    loc_c = find_col_ci(df, location_col, LOCATION_CANDIDATES)
    if not loc_c:
        # create a default location so grouping still works
        loc_c = "location"
        df[loc_c] = "Unknown"

    # Normalize and prepare
    df = df[[date_c, value_c, loc_c]].copy()
    df.rename(columns={date_c: "date", value_c: "value", loc_c: "location"}, inplace=True)

    # Coerce datetime
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values(["location", "date"])

    # Guard: min obs per location
    groups = []
    for loc, sub in df.groupby("location", dropna=False):
        if len(sub) >= min_obs:
            groups.append((loc, sub.copy()))

    if not groups:
        raise ValueError(f"No locations with at least {min_obs} observations.")

    # Compute trends + plots
    summary_rows: List[Dict] = []
    for loc, sub in groups:
        x_years = to_year_float(sub["date"])
        y = sub["value"].astype(float).values
        stats = fit_trend(x_years, y)

        # Save plot
        safe_name = str(loc).replace(" ", "_")
        plot_path = plots_dir / f"trend_{safe_name}.png"
        title = f"Trend for {loc} ({value_c})"
        plot_series_with_trend(sub["date"], sub["value"], stats["slope"], stats["intercept"], plot_path, title)

        # robust relative path for metadata row
        try:
            plot_rel = os.path.relpath(plot_path, start=Path.cwd())
        except Exception:
            plot_rel = str(plot_path)

        summary_rows.append({
            "location": loc,
            "n_obs": stats["n"],
            "slope_per_year": stats["slope"],
            "intercept": stats["intercept"],
            "r_value": stats["r_value"],
            "stderr": stats["stderr"],
            "plot": plot_rel,
        })

    # Write CSV summary
    summary_df = pd.DataFrame(summary_rows).sort_values(["location"]).reset_index(drop=True)
    summary_csv = out_root / "trend_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    # Metadata JSON
    meta = {
        "input": str(input_path),
        "rows_used": int(sum(row["n_obs"] for row in summary_rows)),
        "locations": int(len(summary_rows)),
        "value_col": value_c,
        "date_col": date_c,
        "location_col": loc_c,
        "min_obs": int(min_obs),
        "outputs": {
            "summary_csv": _safe_rel(summary_csv),
            "plots_dir": _safe_rel(plots_dir),
        }
    }

    meta_path = out_root / "trend_run_metadata.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return meta


def _safe_rel(path: Path) -> str:
    """Return a nice relative path if possible; otherwise absolute string."""
    try:
        return os.path.relpath(path, start=Path.cwd())
    except Exception:
        return str(path)


# --------------------------- CLI ---------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute per-location linear trends and save plots + summary.")
    p.add_argument(
        "-i", "--input",
        default="data_lake/feature_sets/features.parquet",
        type=str,
        help="Input parquet with columns including date, value, location."
    )
    p.add_argument("--out-root", default="analysis_outputs/trends", type=str, help="Output directory root.")
    p.add_argument("--value-col", default=None, type=str, help="Value column (case-insensitive).")
    p.add_argument("--date-col", default=None, type=str, help="Date column (case-insensitive).")
    p.add_argument("--location-col", default=None, type=str, help="Location column (case-insensitive).")
    p.add_argument("--min-obs", default=12, type=int, help="Minimum observations per location.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    out_root = Path(args.out_root)

    print(f"INFO - Loading input: {input_path}")
    meta = run_trend_analysis(
        input_path=input_path,
        out_root=out_root,
        value_col=args.value_col,
        date_col=args.date_col,
        location_col=args.location_col,
        min_obs=args.min_obs,
    )

    print("INFO - Trend analysis completed. Outputs:")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
