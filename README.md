# ⌚ Fitbit Fitness Tracker Analysis

EDA + interactive dashboard on the Fitabase export of 33 Fitbit users (12 Apr – 12 May 2016):
daily/hourly activity, sleep, weight and heart rate.

## Files
```
fitbit_project/
├── app.py               # EDA/ETL pipeline: raw CSVs -> cleaned tables -> SQLite
├── Fitbit_EDA.ipynb     # cleaning + exploratory analysis (executed, with findings)
├── dashboard.py         # Streamlit dashboard (8 tabs)
├── requirements.txt
├── data/raw/            # the 6 small source tables + hourly heart rate (~2.8 MB)
├── data/processed/      # cleaned tables as CSV (generated)
└── db/fitbit.db         # SQLite database (generated)
```

## Run
```bash
pip install -r requirements.txt
python app.py                 # rebuild db/ from data/raw/
streamlit run dashboard.py
```
To rebuild from the ORIGINAL large export (≈500 MB; only needed if you want to regenerate `data/raw`):
```bash
python app.py --source "path/to/Fitabase Data 4.12.16-5.12.16"
```
The minute/second-level files are not used: they duplicate the daily/hourly tables, and the
89 MB heart-rate file is aggregated to hourly (`data/raw/heartrate_hourly.csv`).

## Deploying on Streamlit Cloud
Main file: `dashboard.py`. **Keep the folder structure** — upload the *folders* (drag the whole
`fitbit_project` contents into GitHub), not loose files, so `data/raw/` and `db/` keep their paths.
The dashboard self-heals: if `db/fitbit.db` is missing it rebuilds it from `data/raw/`.

## Cleaning rules (see `app.py`)
- **Non-wear day:** 0 steps or < 12 h tracked minutes -> flagged, excluded by default.
- **Partial day:** 12 May (export stops ~3 pm) -> flagged, excluded by default.
- Duplicate sleep rows dropped; near-empty weight `Fat` column dropped.
- Segments: steps bands (Sedentary <5k … Highly Active 12.5k+); usage tier by valid days; WHO proxy = (fairly + 2×very active min) × 7 ≥ 150.
- Users are anonymised as `User 01–33` in all outputs.

## Dashboard tabs
Overview · Activity Patterns · Time of Day · Sleep · Users & Segments · Heart Rate & Weight · Insights · Data Quality & Explore.
Sidebar filters: date range, activity level, usage tier, weekday/weekend, specific users, exclude non-wear/partial days.
