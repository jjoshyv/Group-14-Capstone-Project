# Capstone Project — Environmental Data Analysis & Forecasting

**Project:** Environmental Data Analysis & Predictive Modeling for Ecosphere Institute  
**Repository:** Group-14-Capstone-Project

---

## Overview

This repository contains the full pipeline for analyzing and forecasting ozone (O₃) concentration (2010–2020) and converting forecasts into Air Quality Index (AQI) interpretations. The analysis includes data cleaning, exploratory analysis, feature engineering, trend analysis, and forecasting (Prophet). Visualizations and interactive dashboards are included for reporting and interpretation.

---

## Repository Structure 

```
Capstone-Project-Ecosphere-Institute/
├── notebooks/
│   ├── visualization.ipynb
├── scripts/
│   ├── 01_clean_epa_o3.py
│   ├── exploratory_analysis.py
│   ├── feature_engineering.py
│   ├── py_etl_parquet.py
│   ├── 7_trend_analysis.py
│   └── 8_forecasting.py
├── data/
│   ├── raw/          # optional - small datasets only (do NOT commit large files >100MB)
│   └── cleaned/      # cleaned parquet files (can be committed if small)
├── visualizations/   # exported PNGs for report
├── dashboards/       # exported HTML dashboards
├── analysis/           # saved model outputs / forecasts
├── datalake/
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Quickstart (setup)

1. **Clone the repository**

```bash
git clone https://github.com/jjoshyv/Group-14-Capstone-Project.git
cd Group-14-Capstone-Project
```

2. **Create and activate a virtual environment** (recommended)

```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate         # Windows PowerShell
```


## Data & Storage

- **Do not commit very large raw datasets** (> 100 MB) to GitHub. Use external storage (Google Drive, AWS S3, or institutional storage) and link them in this README.
- Dashboards and visuals for reports are stored under `dashboards/` and `visualizations/` respectively.

---

## Contribution & Collaboration

- Use branches for major changes: `git checkout -b feature/your-feature`
- Commit changes with clear messages and open pull requests for review.
- Add unit tests for core ETL functions or CI checks if integrating into a production workflow.

---

## Licensing & Citation

If you want to publish this repository, consider adding an appropriate license (MIT, Apache 2.0, etc.). Cite the source data repositories used:

- Kateri Salk. Environmental Data Analytics 2020 — https://github.com/KateriSalk/Environmental_Data_Analytics_2020
- acgeospatial. Awesome Earth Observation Code — https://github.com/acgeospatial/awesome-earthobservation-code

---


