"""
app.py — Fitbit Fitness Tracker Analysis: EDA / ETL Pipeline
================================================================
Source: Fitabase export (Fitbit Fitness Tracker Data, 12 Apr - 12 May 2016,
33 users). Builds an analysis-ready SQLite database from the SMALL tables
(daily activity, sleep, weight, hourly steps/calories/intensity, hourly heart
rate). The huge minute-level / second-level files (~500 MB) are NOT needed
and are never loaded.

Usage
-----
    python app.py                       # process files already in data/raw/
    python app.py --source "<path to 'Fitabase Data 4.12.16-5.12.16'>"
                                        # copy the needed files from the original
                                        # export into data/raw/ (and aggregate the
                                        # 89 MB heart-rate file to hourly), then process

Outputs
-------
    db/fitbit.db  (tables: daily, hourly, users, weight, data_quality)
    data/processed/*.csv  (same tables as flat files)
"""
from __future__ import annotations

import argparse
import shutil
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROC_DIR = BASE_DIR / "data" / "processed"
DB_PATH = BASE_DIR / "db" / "fitbit.db"

NEEDED_FILES = [
    "dailyActivity_merged.csv",
    "sleepDay_merged.csv",
    "weightLogInfo_merged.csv",
    "hourlySteps_merged.csv",
    "hourlyCalories_merged.csv",
    "hourlyIntensities_merged.csv",
]
HR_HOURLY_FILE = "heartrate_hourly.csv"  # pre-aggregated (small) version of heartrate_seconds

PARTIAL_DAY = pd.Timestamp("2016-05-12")  # export ends 3pm that day
MIN_WEAR_MINUTES = 720                    # < 12h of tracked minutes => treated as non-wear
DT_FMT = "%m/%d/%Y %I:%M:%S %p"


# ----------------------------------------------------------------------------
# 0. Optional: build data/raw from the original (large) export
# ----------------------------------------------------------------------------
def prepare_raw_from_source(source: Path) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for f in NEEDED_FILES:
        src = source / f
        if not src.exists():
            raise FileNotFoundError(f"Expected {src}")
        shutil.copy(src, RAW_DIR / f)
    hr_src = source / "heartrate_seconds_merged.csv"
    if hr_src.exists():
        print("Aggregating heart-rate seconds -> hourly (chunked)...")
        parts = []
        for ch in pd.read_csv(hr_src, chunksize=500_000):
            t = pd.to_datetime(ch["Time"], format=DT_FMT)
            ch["Hour"] = t.dt.floor("h")
            parts.append(ch.groupby(["Id", "Hour"])["Value"].agg(["sum", "count", "min", "max"]))
        hr = pd.concat(parts).groupby(level=[0, 1]).agg(
            {"sum": "sum", "count": "sum", "min": "min", "max": "max"}
        )
        hr["HR_Mean"] = (hr["sum"] / hr["count"]).round(1)
        out = hr.reset_index().rename(
            columns={"Hour": "ActivityHour", "min": "HR_Min", "max": "HR_Max", "count": "HR_Readings"}
        )[["Id", "ActivityHour", "HR_Mean", "HR_Min", "HR_Max", "HR_Readings"]]
        out.to_csv(RAW_DIR / HR_HOURLY_FILE, index=False)
        print(f"  wrote {HR_HOURLY_FILE}: {len(out):,} user-hours")
    print("Raw files ready in", RAW_DIR)


# ----------------------------------------------------------------------------
# 1. Load
# ----------------------------------------------------------------------------
def _find(fname: str):
    """Look in data/raw/ first, then next to app.py (so a flat GitHub upload also works)."""
    for folder in (RAW_DIR, BASE_DIR):
        if (folder / fname).exists():
            return folder / fname
    return None


def load_raw() -> dict[str, pd.DataFrame]:
    missing = [f for f in NEEDED_FILES if _find(f) is None]
    if missing:
        raise FileNotFoundError(
            f"Missing data files: {missing}. Put them in data/raw/ (or in the same folder as app.py), "
            f"or run `python app.py --source <Fitabase folder>`."
        )
    raw = {f.replace("_merged.csv", ""): pd.read_csv(_find(f)) for f in NEEDED_FILES}
    hr_path = _find(HR_HOURLY_FILE)
    raw["heartrate_hourly"] = pd.read_csv(hr_path) if hr_path else pd.DataFrame()
    return raw


# ----------------------------------------------------------------------------
# 2. Clean / transform
# ----------------------------------------------------------------------------
def activity_level(steps: float) -> str:
    """Steps/day categories (Tudor-Locke & Bassett)."""
    if pd.isna(steps):
        return np.nan
    if steps < 5000:
        return "Sedentary (<5k)"
    if steps < 7500:
        return "Low Active (5-7.5k)"
    if steps < 10000:
        return "Somewhat Active (7.5-10k)"
    if steps < 12500:
        return "Active (10-12.5k)"
    return "Highly Active (12.5k+)"


LEVEL_ORDER = ["Sedentary (<5k)", "Low Active (5-7.5k)", "Somewhat Active (7.5-10k)",
               "Active (10-12.5k)", "Highly Active (12.5k+)"]


def build_daily(raw: dict, dq: list) -> pd.DataFrame:
    da = raw["dailyActivity"].copy()
    da["Date"] = pd.to_datetime(da["ActivityDate"], format="%m/%d/%Y")
    da = da.drop(columns=["ActivityDate"])
    n0 = len(da)
    da = da.drop_duplicates(["Id", "Date"])
    dq.append(("dailyActivity: duplicate user-days", n0 - len(da), "Dropped"))

    da["Wear_Minutes"] = (da[["VeryActiveMinutes", "FairlyActiveMinutes",
                              "LightlyActiveMinutes", "SedentaryMinutes"]].sum(axis=1))
    da["Non_Wear"] = (da["TotalSteps"] == 0) | (da["Wear_Minutes"] < MIN_WEAR_MINUTES)
    da["Partial_Day"] = da["Date"] == PARTIAL_DAY
    da["Valid_Day"] = ~da["Non_Wear"] & ~da["Partial_Day"]
    dq.append(("dailyActivity: zero-step days", int((da["TotalSteps"] == 0).sum()), "Flagged Non_Wear"))
    dq.append((f"dailyActivity: < {MIN_WEAR_MINUTES} tracked minutes", int((da["Wear_Minutes"] < MIN_WEAR_MINUTES).sum()), "Flagged Non_Wear"))
    dq.append(("dailyActivity: partial last day (12 May)", int(da["Partial_Day"].sum()), "Flagged Partial_Day"))
    dq.append(("dailyActivity: valid user-days kept for analysis", int(da["Valid_Day"].sum()), "Valid_Day = True"))
    dq.append(("dailyActivity: TotalDistance != TrackerDistance", int((abs(da["TotalDistance"] - da["TrackerDistance"]) > 0.01).sum()), "Kept (logged activities)"))

    # engineered features
    da["DayOfWeek"] = da["Date"].dt.day_name()
    da["Is_Weekend"] = da["Date"].dt.dayofweek >= 5
    da["Day_Type"] = np.where(da["Is_Weekend"], "Weekend", "Weekday")
    da["MVPA_Minutes"] = da["FairlyActiveMinutes"] + da["VeryActiveMinutes"]
    da["MVPA_WHO_Equiv"] = da["FairlyActiveMinutes"] + 2 * da["VeryActiveMinutes"]
    da["Active_Minutes"] = da["MVPA_Minutes"] + da["LightlyActiveMinutes"]
    da["Sedentary_Hours"] = da["SedentaryMinutes"] / 60
    da["Calories_per_1k_Steps"] = np.where(da["TotalSteps"] > 0, da["Calories"] / (da["TotalSteps"] / 1000), np.nan)
    da["Activity_Level"] = da["TotalSteps"].apply(activity_level)

    # ---- sleep (night ending on Date) ----
    sl = raw["sleepDay"].copy()
    sl["Date"] = pd.to_datetime(sl["SleepDay"], format=DT_FMT).dt.normalize()
    sl = sl.drop(columns=["SleepDay"])
    n0 = len(sl)
    sl = sl.drop_duplicates()
    dq.append(("sleepDay: exact duplicate rows", n0 - len(sl), "Dropped"))
    dq.append(("sleepDay: users with sleep data", int(sl["Id"].nunique()), "Info (of 33)"))
    sl["Sleep_Hours"] = sl["TotalMinutesAsleep"] / 60
    sl["Time_In_Bed_Hours"] = sl["TotalTimeInBed"] / 60
    sl["Sleep_Efficiency"] = (sl["TotalMinutesAsleep"] / sl["TotalTimeInBed"] * 100).round(1)
    sl["Minutes_Awake_In_Bed"] = sl["TotalTimeInBed"] - sl["TotalMinutesAsleep"]
    sl["Is_Nap_Day"] = sl["TotalSleepRecords"] > 1
    da = da.merge(sl, on=["Id", "Date"], how="left")
    da["Has_Sleep"] = da["TotalMinutesAsleep"].notna()

    # previous-day activity (does yesterday's activity relate to last night's sleep?)
    da = da.sort_values(["Id", "Date"])
    da["Prev_Day_Steps"] = da.groupby("Id")["TotalSteps"].shift(1)
    da["Prev_Day_Steps"] = da["Prev_Day_Steps"].where(da.groupby("Id")["Date"].diff().dt.days == 1)

    # ---- weight (last log of the day) ----
    w = raw["weightLogInfo"].copy()
    w["Date"] = pd.to_datetime(w["Date"], format=DT_FMT).dt.normalize()
    dq.append(("weightLog: Fat column missing", int(w["Fat"].isna().sum()), f"Column dropped (of {len(w)} rows)"))
    dq.append(("weightLog: users with weight data", int(w["Id"].nunique()), "Info (of 33)"))
    wd = w.sort_values("LogId").groupby(["Id", "Date"], as_index=False).last()[["Id", "Date", "WeightKg", "BMI", "IsManualReport"]]
    da = da.merge(wd, on=["Id", "Date"], how="left")
    return da.reset_index(drop=True), w.drop(columns=["Fat"])


def build_hourly(raw: dict, daily: pd.DataFrame, dq: list) -> pd.DataFrame:
    def prep(df, col):
        d = df.copy()
        d["Hour_TS"] = pd.to_datetime(d["ActivityHour"], format=DT_FMT)
        return d.drop(columns=["ActivityHour"])
    hs = prep(raw["hourlySteps"], "StepTotal")
    hc = prep(raw["hourlyCalories"], "Calories")
    hi = prep(raw["hourlyIntensities"], "TotalIntensity")
    h = hs.merge(hc, on=["Id", "Hour_TS"], how="outer").merge(hi, on=["Id", "Hour_TS"], how="outer")
    h = h.drop_duplicates(["Id", "Hour_TS"])
    hr = raw["heartrate_hourly"]
    if len(hr):
        hr = hr.copy()
        hr["Hour_TS"] = pd.to_datetime(hr["ActivityHour"])
        h = h.merge(hr.drop(columns=["ActivityHour"]), on=["Id", "Hour_TS"], how="left")
        dq.append(("heart rate: users with HR data", int(hr["Id"].nunique()), "Info (of 33)"))
    else:
        for c in ["HR_Mean", "HR_Min", "HR_Max", "HR_Readings"]:
            h[c] = np.nan
    h["Date"] = h["Hour_TS"].dt.normalize()
    h["Hour"] = h["Hour_TS"].dt.hour
    h["DayOfWeek"] = h["Hour_TS"].dt.day_name()
    h["Day_Type"] = np.where(h["Hour_TS"].dt.dayofweek >= 5, "Weekend", "Weekday")
    h["Time_Of_Day"] = pd.cut(h["Hour"], bins=[-1, 5, 11, 16, 20, 23],
                              labels=["Night (0-5)", "Morning (6-11)", "Afternoon (12-16)", "Evening (17-20)", "Late (21-23)"])
    h["Time_Of_Day"] = h["Time_Of_Day"].astype(str)
    h = h.merge(daily[["Id", "Date", "Valid_Day"]], on=["Id", "Date"], how="left")
    h["Valid_Day"] = h["Valid_Day"].fillna(False)
    return h


def build_users(daily: pd.DataFrame, hourly: pd.DataFrame) -> pd.DataFrame:
    v = daily[daily["Valid_Day"]]
    u = v.groupby("Id").agg(
        Valid_Days=("Date", "count"),
        Avg_Steps=("TotalSteps", "mean"),
        Avg_Calories=("Calories", "mean"),
        Avg_Distance_Km=("TotalDistance", "mean"),
        Avg_Sedentary_Hours=("Sedentary_Hours", "mean"),
        Avg_Very_Active_Min=("VeryActiveMinutes", "mean"),
        Avg_MVPA_WHO_Equiv=("MVPA_WHO_Equiv", "mean"),
        Avg_Sleep_Hours=("Sleep_Hours", "mean"),
        Avg_Sleep_Efficiency=("Sleep_Efficiency", "mean"),
        Avg_BMI=("BMI", "mean"),
    ).reset_index()
    total_days = daily.groupby("Id").size().rename("Days_Logged")
    u = u.merge(total_days, on="Id", how="right")
    u["Valid_Days"] = u["Valid_Days"].fillna(0).astype(int)
    u["Activity_Level"] = u["Avg_Steps"].apply(activity_level)
    u["Usage_Tier"] = pd.cut(u["Valid_Days"], bins=[-1, 14, 24, 100],
                             labels=["Low (<15 days)", "Moderate (15-24)", "High (25+ days)"]).astype(str)
    u["Meets_WHO_Guideline"] = (u["Avg_MVPA_WHO_Equiv"] * 7) >= 150
    u["Has_Sleep"] = daily.groupby("Id")["Has_Sleep"].max().reindex(u["Id"]).values
    u["Has_Weight"] = daily.groupby("Id")["BMI"].apply(lambda s: s.notna().any()).reindex(u["Id"]).values
    u["Has_HR"] = hourly.groupby("Id")["HR_Mean"].apply(lambda s: s.notna().any()).reindex(u["Id"]).values
    u["User"] = "User " + (u.index + 1).astype(str).str.zfill(2)
    u["Id"] = u["Id"].astype(str)
    return u


def clean_all(raw: dict):
    dq: list = []
    daily, weight = build_daily(raw, dq)
    hourly = build_hourly(raw, daily, dq)
    users = build_users(daily, hourly)
    # friendly anonymous label on daily/hourly too
    label = users.set_index("Id")["User"]
    for df in (daily, hourly):
        df["Id"] = df["Id"].astype(str)
        df["User"] = df["Id"].map(label)
    weight["Id"] = weight["Id"].astype(str)
    weight["User"] = weight["Id"].map(label)
    dq_df = pd.DataFrame(dq, columns=["Check", "Rows/Users Affected", "Action"])
    print(f"daily={len(daily):,} (valid {int(daily['Valid_Day'].sum()):,}) | hourly={len(hourly):,} | users={len(users)} | weight={len(weight)}")
    return daily, hourly, users, weight, dq_df


# ----------------------------------------------------------------------------
# 3. Persist
# ----------------------------------------------------------------------------
def save_outputs(daily, hourly, users, weight, dq_df) -> None:
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    DB_PATH.parent.mkdir(exist_ok=True)
    tables = {"daily": daily, "hourly": hourly, "users": users, "weight": weight, "data_quality": dq_df}
    conn = sqlite3.connect(DB_PATH)
    for name, df in tables.items():
        out = df.copy()
        for c in out.columns:
            if pd.api.types.is_datetime64_any_dtype(out[c]):
                out[c] = out[c].dt.strftime("%Y-%m-%d %H:%M:%S")
        out.to_csv(PROC_DIR / f"{name}.csv", index=False)
        out.to_sql(name, conn, if_exists="replace", index=False)
    conn.commit()
    conn.close()
    print(f"Saved DB -> {DB_PATH} and CSVs -> {PROC_DIR}")


def run_pipeline():
    raw = load_raw()
    daily, hourly, users, weight, dq_df = clean_all(raw)
    save_outputs(daily, hourly, users, weight, dq_df)
    return daily, hourly, users, weight, dq_df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=str, default=None, help="Path to the original 'Fitabase Data ...' folder")
    args = ap.parse_args()
    if args.source:
        prepare_raw_from_source(Path(args.source))
    run_pipeline()
    print("Pipeline complete.")
