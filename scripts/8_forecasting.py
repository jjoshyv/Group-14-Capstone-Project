#!/usr/bin/env python3
"""
analysis/8_forecasting.py

Forecasting module for Capstone Project.

Features:
 - Reads features parquet (default: data_lake/feature_sets/features.parquet)
 - Auto-detects date/target/location columns case-insensitively
 - Supports Prophet (if installed), SARIMAX, or Holt-Winters
 - Forecast horizon in years (monthly frequency by default)
 - Saves forecast CSVs, plots (with uncertainty when available), and a run metadata JSON
 - Evaluates forecasts on a holdout period if requested (MAE, RMSE)

Usage examples:
  # Prophet forecast next 10 years for all locations (monthly)
  python analysis/8_forecasting.py --target-col o3_ug_m3 --date-col date --location-col location --engine prophet --forecast-years 10

  # SARIMAX forecast, single location "Kerala", evaluate on last 24 months
  python analysis/8_forecasting.py --engine sarimax --location Kerala --evaluate-months 24 --forecast-years 10
"""

from __future__ import annotations
import argparse
import json
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# try prophet (new package 'prophet' or older 'fbprophet')
try:
    from prophet import Prophet  # type: ignore
    PROPHET_AVAILABLE = True
except Exception:
    try:
        from fbprophet import Prophet  # type: ignore
        PROPHET_AVAILABLE = True
    except Exception:
        PROPHET_AVAILABLE = False

# statsmodels
try:
    import statsmodels.api as sm  # SARIMAX
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    STATSMODELS_AVAILABLE = True
except Exception:
    STATSMODELS_AVAILABLE = False

from sklearn.metrics import mean_absolute_error, mean_squared_error
import math
import warnings
warnings.filterwarnings("ignore")

# logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")
log = logging.getLogger("forecasting")

# -------------------------
# Helpers
# -------------------------
def safe_read_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    df = pd.read_parquet(path)
    # normalize columns to lowercase underscored form for robust matching
    df.columns = (
        pd.Series(df.columns)
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"[^a-z0-9_]+", "_", regex=True)
        .tolist()
    )
    log.info("Loaded dataframe with columns: %s", list(df.columns))
    return df

def find_col_case_insensitive(df: pd.DataFrame, desired: str) -> Optional[str]:
    if desired in df.columns:
        return desired
    lowmap = {c.lower(): c for c in df.columns}
    return lowmap.get(desired.lower())

def ensure_datetime(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    if date_col not in df.columns:
        raise ValueError(f"Date column '{date_col}' not found. Available: {list(df.columns)}")
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    if df[date_col].isna().all():
        raise ValueError(f"All values in date column '{date_col}' parsed as NaT.")
    return df

def resample_monthly(df: pd.DataFrame, date_col: str, target_col: str) -> pd.DataFrame:
    """Aggregate to monthly mean and return DataFrame indexed by month start."""
    tmp = df[[date_col, target_col]].dropna().copy()
    tmp[date_col] = pd.to_datetime(tmp[date_col])
    tmp = tmp.set_index(date_col).resample("MS").mean().reset_index()
    return tmp

def train_prophet(series_df: pd.DataFrame, target_col: str, periods: int, freq: str = "M", yearly_seasonality: Optional[bool] = True) -> Tuple[pd.DataFrame, dict]:
    """
    Train Prophet model on series_df which must have columns [date,target_col] (date named 'ds' for Prophet).
    Returns forecast DataFrame (with yhat, yhat_lower, yhat_upper where available) and metadata.
    """
    if not PROPHET_AVAILABLE:
        raise RuntimeError("Prophet is not installed on this environment. Install 'prophet' or 'fbprophet' to use this engine.")
    df = series_df.rename(columns={series_df.columns[0]: "ds", series_df.columns[1]: "y"})[["ds", "y"]].dropna()
    m = Prophet(yearly_seasonality=yearly_seasonality, daily_seasonality=False, weekly_seasonality=False)
    m.fit(df)
    future = m.make_future_dataframe(periods=periods, freq='M')
    forecast = m.predict(future)
    meta = {"engine": "prophet", "model": "Prophet()", "n_obs": len(df)}
    return forecast, meta

def train_sarimax(series_df: pd.DataFrame, target_col: str, periods: int, seasonal_periods: int = 12) -> Tuple[pd.DataFrame, dict]:
    """
    Train a simple SARIMAX model on monthly series, with order default (1,1,1) and seasonal (0,1,1,seasonal_periods).
    Returns forecast dataframe with columns ['ds','yhat','yhat_lower','yhat_upper'] where conf_int available.
    """
    if not STATSMODELS_AVAILABLE:
        raise RuntimeError("statsmodels not available; cannot run SARIMAX.")
    s = series_df.copy()
    s = s.set_index(s.columns[0])[s.columns[1]].asfreq('MS')  # expecting monthly start index
    # fallback fill for missing values
    s = s.fillna(method='ffill').fillna(method='bfill')
    # simple params: (p,d,q) = (1,1,1), seasonal (P,D,Q,s) = (0,1,1,seasonal_periods)
    order = (1, 1, 1)
    seasonal_order = (0, 1, 1, seasonal_periods)
    model = sm.tsa.statespace.SARIMAX(s, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
    res = model.fit(disp=False)
    # get prediction
    start = len(s)
    end = start + periods - 1
    pred = res.get_prediction(start=start, end=end, dynamic=False)
    pred_mean = pred.predicted_mean
    conf = pred.conf_int(alpha=0.05)
    # construct forecast index
    last_date = s.index[-1]
    future_idx = pd.date_range(start=last_date + pd.offsets.MonthBegin(1), periods=periods, freq='MS')
    df_fore = pd.DataFrame({"ds": future_idx, "yhat": pred_mean.values, "yhat_lower": conf.iloc[:, 0].values, "yhat_upper": conf.iloc[:, 1].values})
    meta = {"engine": "sarimax", "order": order, "seasonal_order": seasonal_order, "aic": float(res.aic)}
    return df_fore, meta

def train_holtwinters(series_df: pd.DataFrame, target_col: str, periods: int, seasonal_periods: int = 12) -> Tuple[pd.DataFrame, dict]:
    """
    Fit Holt-Winters Exponential Smoothing (additive seasonality) and forecast periods ahead.
    Returns forecast df with ds,yhat (no uncertainty interval).
    """
    if not STATSMODELS_AVAILABLE:
        raise RuntimeError("statsmodels not available; cannot run Holt-Winters.")
    s = series_df.copy()
    s = s.set_index(s.columns[0])[s.columns[1]].asfreq('MS')
    s = s.fillna(method='ffill').fillna(method='bfill')
    # try additive seasonal; if fails, try multiplicative
    try:
        model = ExponentialSmoothing(s, trend='add', seasonal='add', seasonal_periods=seasonal_periods)
        fit = model.fit()
    except Exception:
        model = ExponentialSmoothing(s, trend='add', seasonal='mul', seasonal_periods=seasonal_periods)
        fit = model.fit()
    future_idx = pd.date_range(start=s.index[-1] + pd.offsets.MonthBegin(1), periods=periods, freq='MS')
    preds = fit.forecast(periods)
    df_fore = pd.DataFrame({"ds": future_idx, "yhat": preds.values})
    meta = {"engine": "holtwinters", "params": str(fit.params)}
    return df_fore, meta

def evaluate_forecast(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    # remove nan pairs
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if mask.sum() == 0:
        return {"mae": None, "rmse": None}
    mae = float(mean_absolute_error(y_true[mask], y_pred[mask]))
    rmse = float(math.sqrt(mean_squared_error(y_true[mask], y_pred[mask])))
    return {"mae": mae, "rmse": rmse}

def plot_forecast(history_df: pd.DataFrame, forecast_df: pd.DataFrame, target_col: str, out_path: Path, title: str = None, show_interval: bool = True):
    """
    history_df: df with [ds, y]
    forecast_df: df with ds, yhat, optionally yhat_lower/yhat_upper
    """
    plt.figure(figsize=(10, 5))
    plt.plot(history_df.iloc[:, 0], history_df.iloc[:, 1], label="history", marker='o', linewidth=1)
    plt.plot(forecast_df["ds"], forecast_df["yhat"], label="forecast", linestyle='--')
    if show_interval and ("yhat_lower" in forecast_df.columns and "yhat_upper" in forecast_df.columns):
        plt.fill_between(forecast_df["ds"], forecast_df["yhat_lower"], forecast_df["yhat_upper"], color="gray", alpha=0.3, label="uncertainty")
    plt.xlabel("Date")
    plt.ylabel(target_col)
    if title:
        plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

# -------------------------
# Main orchestration
# -------------------------
def main():
    p = argparse.ArgumentParser(description="Forecast environmental metrics (Prophet / SARIMAX / Holt-Winters)")
    p.add_argument("--input", "-i", type=str, default="data_lake/feature_sets/features.parquet", help="Input features parquet")
    p.add_argument("--target-col", "-t", type=str, required=True, help="Target numeric column to forecast (case-insensitive)")
    p.add_argument("--date-col", "-d", type=str, default="date", help="Date column name (case-insensitive)")
    p.add_argument("--location-col", "-l", type=str, default="location", help="Location column (use 'all' to aggregate or omit to run per-location)")
    p.add_argument("--engine", type=str, default="prophet", choices=["prophet", "sarimax", "holtwinters"], help="Forecast engine")
    p.add_argument("--forecast-years", type=int, default=10, help="Forecast horizon in years (monthly frequency assumed)")
    p.add_argument("--evaluate-months", type=int, default=0, help="If >0, hold out last N months for evaluation")
    p.add_argument("--out-root", type=str, default="analysis_outputs/forecasts", help="Output folder for forecasts and plots")
    p.add_argument("--seasonal-period", type=int, default=12, help="Seasonal period in months (default 12)")
    args = p.parse_args()

    input_path = Path(args.input)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    log.info("Loading input: %s", input_path)
    df = safe_read_parquet(input_path)

    # column detection
    date_col_found = find_col_case_insensitive(df, args.date_col)
    target_col_found = find_col_case_insensitive(df, args.target_col)
    loc_col_found = find_col_case_insensitive(df, args.location_col) if args.location_col and args.location_col.lower() != "all" else None

    if date_col_found is None:
        log.error("Date column '%s' not found. Available: %s", args.date_col, list(df.columns))
        raise SystemExit(1)
    if target_col_found is None:
        log.error("Target column '%s' not found. Available: %s", args.target_col, list(df.columns))
        raise SystemExit(1)

    # ensure datetime
    df = ensure_datetime(df, date_col_found)

    # if location specified as 'all' or not found, we aggregate across all locations
    run_locations = []
    if loc_col_found is None:
        if args.location_col and args.location_col.lower() == "all":
            # aggregate on entire dataset
            run_locations = ["__ALL__"]
        else:
            # if location column not found, create 'location' with Unknown or run by 'location' if present
            if "location" in df.columns:
                loc_col_found = "location"
            else:
                df["location"] = "Unknown"
                loc_col_found = "location"
            run_locations = sorted(df[loc_col_found].dropna().unique().tolist())
    else:
        run_locations = sorted(df[loc_col_found].dropna().unique().tolist())

    log.info("Forecast will run for %d locations (engine=%s)", len(run_locations), args.engine)

    # forecast horizon in months
    periods = int(args.forecast_years * 12)

    summary_rows = []
    for loc in run_locations:
        log.info("Processing location: %s", loc)
        if loc == "__ALL__":
            series_df = df[[date_col_found, target_col_found]].copy()
        else:
            series_df = df[df[loc_col_found] == loc][[date_col_found, target_col_found]].copy()

        if series_df.empty or len(series_df) < 12:
            log.warning("Skipping location %s due to insufficient data (rows=%d)", loc, len(series_df))
            continue

        # resample monthly (mean)
        s_monthly = resample_monthly(series_df, date_col_found, target_col_found)

        # split evaluation holdout if requested
        eval_results = None
        holdout = None
        train_df = s_monthly.copy()
        if args.evaluate_months and args.evaluate_months > 0:
            holdout = train_df.tail(args.evaluate_months).copy()
            train_df = train_df.iloc[:-args.evaluate_months].copy()
            if len(train_df) < 12:
                log.warning("After holdout, insufficient training data for %s; skipping evaluation for this location.", loc)
                holdout = None

        # prepare history df for plotting/training
        history_df = train_df.copy().reset_index(drop=True)

        # Train/forecast depending on engine
        try:
            if args.engine == "prophet":
                if not PROPHET_AVAILABLE:
                    log.warning("Prophet not available; falling back to SARIMAX if available.")
                    if STATSMODELS_AVAILABLE:
                        forecast_df, meta = train_sarimax(train_df, target_col_found, periods, seasonal_periods=args.seasonal_period)
                    else:
                        forecast_df, meta = train_holtwinters(train_df, target_col_found, periods, seasonal_periods=args.seasonal_period)
                else:
                    # rename columns for prophet
                    prophet_input = train_df.rename(columns={train_df.columns[0]: "ds", train_df.columns[1]: "y"})[["ds", "y"]]
                    forecast_raw, meta = train_prophet(prophet_input, target_col_found, periods)
                    # Prophet returns a long forecast including history; we will extract future rows
                    # keep columns ds,yhat,yhat_lower,yhat_upper (Prophet: yhat, yhat_lower, yhat_upper)
                    # For consistency, extract rows where ds > last history date
                    last_train = train_df[train_df.columns[0]].max()
                    future_mask = forecast_raw["ds"] > last_train
                    forecast_df = forecast_raw.loc[future_mask, ["ds", "yhat", "yhat_lower", "yhat_upper"]].rename(columns={"yhat": "yhat"})
            elif args.engine == "sarimax":
                forecast_df, meta = train_sarimax(train_df, target_col_found, periods, seasonal_periods=args.seasonal_period)
            elif args.engine == "holtwinters":
                forecast_df, meta = train_holtwinters(train_df, target_col_found, periods, seasonal_periods=args.seasonal_period)
            else:
                raise ValueError("Invalid engine")
        except Exception as e:
            log.error("Model training failed for %s: %s", loc, e)
            continue

        # Evaluate if holdout present: we compare holdout y vs forecast first n points
        if holdout is not None and not holdout.empty:
            # ensure forecast_df sorted and aligned
            # combine forecast start dates = immediately after train last date
            # get first len(holdout) rows from forecast_df for evaluation
            eval_len = len(holdout)
            eval_fore = forecast_df.head(eval_len)
            y_true = holdout[target_col_found].to_numpy()
            y_pred = eval_fore["yhat"].to_numpy()
            eval_results = evaluate_forecast(y_true, y_pred)
            log.info("Evaluation for %s: MAE=%.4f RMSE=%.4f", loc, eval_results.get("mae"), eval_results.get("rmse"))

        # Save forecast CSV (append history with forecast for convenience)
        # Build combined df: history + forecast
        hist_for_join = history_df.rename(columns={history_df.columns[0]: "ds", history_df.columns[1]: "y"}).reset_index(drop=True)
        # forecast_df may have various column names; ensure consistent columns
        fdf = forecast_df.copy().reset_index(drop=True)
        # normalize forecast df
        if "yhat" not in fdf.columns and "y" in fdf.columns:
            fdf = fdf.rename(columns={"y": "yhat"})
        # ensure ds column present
        if "ds" not in fdf.columns:
            # try if first column is ds
            fdf = fdf.rename(columns={fdf.columns[0]: "ds"})
        out_combined = pd.concat([hist_for_join, fdf[["ds", "yhat"]]], ignore_index=True, sort=False)
        # save per-location csv
        loc_safe = str(loc).replace(" ", "_").replace("/", "_")
        csv_out = out_root / f"forecast_{loc_safe}.csv"
        out_combined.to_csv(csv_out, index=False)
        log.info("Wrote forecast CSV: %s", csv_out)

        # save plot
        plot_path = out_root / f"forecast_{loc_safe}.png"
        plot_forecast(hist_for_join, fdf, args.target_col, plot_path, title=f"{loc} forecast ({args.engine})", show_interval=("yhat_lower" in fdf.columns))
        log.info("Wrote forecast plot: %s", plot_path)

        # save metadata row
        row = {
            "location": loc,
            "n_train": int(len(history_df)),
            "n_forecast": int(len(fdf)),
            "engine": meta.get("engine", args.engine),
            "meta": meta,
        }
        if eval_results:
            row.update(eval_results)
        summary_rows.append(row)

    # Save summary CSV and JSON
    summary_df = pd.DataFrame(summary_rows)
    summary_csv = out_root / "forecast_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    with open(out_root / "forecast_run_metadata.json", "w", encoding="utf-8") as fh:
        json.dump({"input": str(input_path), "engine": args.engine, "target_col": args.target_col, "n_locations": len(summary_rows)}, fh, indent=2)
    log.info("Saved forecast summary: %s", summary_csv)
    log.info("Forecasting completed. Outputs in: %s", str(out_root))


if __name__ == "__main__":
    main()
