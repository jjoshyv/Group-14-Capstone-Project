# 5_exploratory_analysis.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress

# Load dataset
df = pd.read_csv("Merged_Dataset.csv", parse_dates=["Date"])
df = df.sort_values("Date").reset_index(drop=True)

print("\n✅ Loaded dataset with", len(df), "records from", df['Date'].min().date(), "to", df['Date'].max().date())
print("\nColumns:", df.columns.tolist())
print("\nSummary Statistics:\n", df.describe().round(2))

# --- 1️⃣ Time Series Trends ---
plt.figure(figsize=(10, 5))
plt.plot(df["Date"], df["O3_ug_m3"], color="purple", linewidth=2, label="O₃ (µg/m³)")
plt.title("Monthly O₃ Concentration (2010–2019)")
plt.xlabel("Year")
plt.ylabel("O₃ (µg/m³)")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()

# --- 2️⃣ Seasonal Patterns ---
df["Month"] = df["Date"].dt.month
season_avg = df.groupby("Month")[["O3_ug_m3", "T2M", "PRECTOTCORR"]].mean().reset_index()

plt.figure(figsize=(10, 6))
plt.plot(season_avg["Month"], season_avg["O3_ug_m3"], "-o", color="purple", label="O₃ (µg/m³)")
plt.plot(season_avg["Month"], season_avg["T2M"], "-o", color="orange", label="Temperature (°C)")
plt.title("Seasonal Variation (Monthly Average across 2010–2019)")
plt.xlabel("Month")
plt.ylabel("Value")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(df["Date"], df["O3_ug_m3"], label="O₃ (µg/m³)", color="purple")
plt.plot(df["Date"], intercept + slope*x, color="black", linestyle="--", label=f"Trend line (R²={r_value**2:.2f})")
plt.title("Trend in O₃ Concentration (2010–2019)")
plt.xlabel("Year")
plt.ylabel("O₃ (µg/m³)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()

print(f"\n📈 Trend Analysis: O₃ increases {slope:.3f} µg/m³ per month (R² = {r_value**2:.3f}, p = {p_value:.4f})")
