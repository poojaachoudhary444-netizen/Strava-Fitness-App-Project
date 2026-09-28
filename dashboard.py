"""
dashboard.py — Fitbit Fitness Tracker Analysis (Streamlit)
=============================================================
Run:  streamlit run dashboard.py
Reads db/fitbit.db (built by app.py). Self-heals: if the DB is missing and the
small raw files are in data/raw/, it builds the DB on first load.
"""
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Fitbit Activity Analysis", page_icon="⌚", layout="wide",
                   initial_sidebar_state="expanded")

BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / "db" / "fitbit.db"

DOW = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
LEVELS = ["Sedentary (<5k)", "Low Active (5-7.5k)", "Somewhat Active (7.5-10k)",
          "Active (10-12.5k)", "Highly Active (12.5k+)"]
TOD = ["Night (0-5)", "Morning (6-11)", "Afternoon (12-16)", "Evening (17-20)", "Late (21-23)"]
TIERS = ["Low (<15 days)", "Moderate (15-24)", "High (25+ days)"]
LEVEL_COLORS = dict(zip(LEVELS, ["#D64550", "#F0A04B", "#F2D16B", "#7BC47F", "#2CA58D"]))
TEAL, PURPLE = "#2CA58D", "#5B4B8A"


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def _ensure_db():
    if DB_PATH.exists():
        return
    import app as etl
    with st.spinner("First run: building database from data/raw ..."):
        etl.run_pipeline()


@st.cache_data(show_spinner=True)
def load():
    _ensure_db()
    conn = sqlite3.connect(DB_PATH)
    daily = pd.read_sql("SELECT * FROM daily", conn, parse_dates=["Date"])
    hourly = pd.read_sql("SELECT * FROM hourly", conn, parse_dates=["Date", "Hour_TS"])
    users = pd.read_sql("SELECT * FROM users", conn)
    weight = pd.read_sql("SELECT * FROM weight", conn, parse_dates=["Date"])
    dq = pd.read_sql("SELECT * FROM data_quality", conn)
    conn.close()
    for df, cols in [(daily, ["Non_Wear", "Partial_Day", "Valid_Day", "Is_Weekend", "Has_Sleep", "Is_Nap_Day"]),
                     (hourly, ["Valid_Day"]),
                     (users, ["Meets_WHO_Guideline", "Has_Sleep", "Has_Weight", "Has_HR"])]:
        for c in cols:
            if c in df.columns:
                df[c] = df[c].fillna(0).astype(bool)
    return daily, hourly, users, weight, dq


daily_all, hourly_all, users_all, weight_all, dq = load()


def ols_line(fig, x, y, name="Trend"):
    m = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(m) > 5 and m["x"].nunique() > 1:
        b, a = np.polyfit(m["x"], m["y"], 1)
        xs = np.array([m["x"].min(), m["x"].max()])
        fig.add_trace(go.Scatter(x=xs, y=a + b * xs, mode="lines", name=f"{name} (r={m['x'].corr(m['y']):.2f})",
                                 line=dict(color="black", dash="dash")))
    return fig


# ----------------------------------------------------------------------------
# Sidebar filters
# ----------------------------------------------------------------------------
st.sidebar.title("⌚ Filters")
dmin, dmax = daily_all["Date"].min().date(), daily_all["Date"].max().date()
date_rng = st.sidebar.date_input("Date range", (dmin, dmax), min_value=dmin, max_value=dmax)
if isinstance(date_rng, tuple) and len(date_rng) == 2:
    d0, d1 = pd.Timestamp(date_rng[0]), pd.Timestamp(date_rng[1])
else:
    d0, d1 = pd.Timestamp(dmin), pd.Timestamp(dmax)

sel_levels = st.sidebar.multiselect("Activity level (user avg steps)", LEVELS, default=LEVELS)
sel_tiers = st.sidebar.multiselect("Device usage tier", TIERS, default=TIERS)
day_type = st.sidebar.radio("Day type", ["All", "Weekday", "Weekend"], horizontal=True)
valid_only = st.sidebar.checkbox("Exclude non-wear & partial days (recommended)", value=True)
user_opts = users_all["User"].tolist()
sel_users = st.sidebar.multiselect("Specific users (optional)", user_opts, default=[])
st.sidebar.markdown("---")
st.sidebar.caption("Fitbit Fitness Tracker Data · 33 users · 12 Apr – 12 May 2016. "
                   "Users are anonymised as User 01–33.")

u = users_all[users_all["Activity_Level"].isin(sel_levels) & users_all["Usage_Tier"].isin(sel_tiers)]
if sel_users:
    u = u[u["User"].isin(sel_users)]
ids = set(u["Id"])

daily = daily_all[daily_all["Id"].isin(ids) & daily_all["Date"].between(d0, d1)]
hourly = hourly_all[hourly_all["Id"].isin(ids) & hourly_all["Date"].between(d0, d1)]
if valid_only:
    daily = daily[daily["Valid_Day"]]
    hourly = hourly[hourly["Valid_Day"]]
if day_type != "All":
    daily = daily[daily["Day_Type"] == day_type]
    hourly = hourly[hourly["Day_Type"] == day_type]

if daily.empty:
    st.title("⌚ Fitbit Activity Analysis")
    st.warning("No data matches the current filters. Try widening your selection.")
    st.stop()

# ----------------------------------------------------------------------------
# Header + KPIs
# ----------------------------------------------------------------------------
st.title("⌚ Fitbit Fitness Tracker Analysis")
st.caption("How do people move, sleep and use their tracker? Activity, time-of-day, sleep, segments & usage.")

sleep_df = daily[daily["Has_Sleep"]]
k = st.columns(6)
k[0].metric("Users", f"{daily['Id'].nunique()}")
k[1].metric("User-days", f"{len(daily):,}")
k[2].metric("Avg steps / day", f"{daily['TotalSteps'].mean():,.0f}")
k[3].metric("Avg calories / day", f"{daily['Calories'].mean():,.0f}")
k[4].metric("Avg sedentary hrs", f"{daily['Sedentary_Hours'].mean():.1f}")
k[5].metric("Avg sleep (h)", f"{sleep_df['Sleep_Hours'].mean():.1f}" if len(sleep_df) else "n/a")
st.markdown("---")

tabs = st.tabs(["📊 Overview", "🏃 Activity Patterns", "🕒 Time of Day", "😴 Sleep",
                "👥 Users & Segments", "❤️ Heart Rate & Weight", "💡 Insights", "🧾 Data Quality & Explore"])

# ============================ OVERVIEW ============================
with tabs[0]:
    c1, c2 = st.columns([1.4, 1])
    with c1:
        trend = daily.groupby("Date").agg(Avg_Steps=("TotalSteps", "mean"), Users=("Id", "nunique")).reset_index()
        fig = px.line(trend, x="Date", y="Avg_Steps", markers=True, hover_data=["Users"],
                      title="Average Steps per Day (across users)")
        fig.update_traces(line_color=TEAL)
        fig.add_hline(y=10000, line_dash="dot", annotation_text="10k steps", line_color="grey")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        lv = u["Activity_Level"].value_counts().reindex(LEVELS).fillna(0).reset_index()
        lv.columns = ["Level", "Users"]
        fig = px.bar(lv, x="Users", y="Level", orientation="h", color="Level",
                     color_discrete_map=LEVEL_COLORS, title="Users by Activity Level")
        fig.update_layout(showlegend=False, yaxis=dict(categoryorder="array", categoryarray=LEVELS[::-1]))
        st.plotly_chart(fig, use_container_width=True)
    c3, c4 = st.columns(2)
    with c3:
        mins = pd.DataFrame({
            "Intensity": ["Sedentary", "Lightly Active", "Fairly Active", "Very Active"],
            "Avg minutes/day": [daily["SedentaryMinutes"].mean(), daily["LightlyActiveMinutes"].mean(),
                                daily["FairlyActiveMinutes"].mean(), daily["VeryActiveMinutes"].mean()]})
        fig = px.bar(mins, x="Intensity", y="Avg minutes/day", color="Intensity", text_auto=".0f",
                     title="Where the Day Goes: Average Minutes by Intensity",
                     color_discrete_sequence=["#B0B0B0", "#7BC47F", "#F0A04B", "#D64550"])
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with c4:
        who = u["Meets_WHO_Guideline"].mean() * 100 if len(u) else 0
        fig = go.Figure(go.Indicator(mode="gauge+number", value=who, number={"suffix": "%"},
                                     title={"text": "Users meeting WHO 150 min/wk activity guideline"},
                                     gauge={"axis": {"range": [0, 100]}, "bar": {"color": TEAL}}))
        fig.update_layout(height=330)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Guideline proxy: (Fairly Active min + 2 × Very Active min) × 7 ≥ 150 per week, based on each user's daily average.")

# ============================ ACTIVITY ============================
with tabs[1]:
    c1, c2 = st.columns(2)
    with c1:
        dow = daily.groupby("DayOfWeek")["TotalSteps"].mean().reindex(DOW).reset_index()
        fig = px.bar(dow, x="DayOfWeek", y="TotalSteps", color="TotalSteps", color_continuous_scale="Teal",
                     title="Average Steps by Day of Week", text_auto=".0f")
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.histogram(daily, x="TotalSteps", nbins=40, color_discrete_sequence=[PURPLE],
                           title="Distribution of Daily Steps")
        fig.add_vline(x=10000, line_dash="dot", line_color="grey")
        st.plotly_chart(fig, use_container_width=True)
    c3, c4 = st.columns(2)
    with c3:
        fig = px.scatter(daily, x="TotalSteps", y="Calories", color="Activity_Level",
                         color_discrete_map=LEVEL_COLORS, opacity=0.6, hover_data=["User", "Date"],
                         category_orders={"Activity_Level": LEVELS}, title="Steps vs. Calories Burned")
        ols_line(fig, daily["TotalSteps"], daily["Calories"])
        st.plotly_chart(fig, use_container_width=True)
    with c4:
        fig = px.scatter(daily, x="VeryActiveMinutes", y="Calories", opacity=0.55,
                         color_discrete_sequence=[TEAL], title="Very Active Minutes vs. Calories")
        ols_line(fig, daily["VeryActiveMinutes"], daily["Calories"])
        st.plotly_chart(fig, use_container_width=True)
    c5, c6 = st.columns(2)
    with c5:
        fig = px.box(daily, x="Day_Type", y="TotalSteps", color="Day_Type", points="outliers",
                     color_discrete_sequence=[PURPLE, TEAL], title="Weekday vs. Weekend Steps")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with c6:
        fig = px.histogram(daily, x="Sedentary_Hours", nbins=30, color_discrete_sequence=["#B0B0B0"],
                           title="Daily Sedentary Hours (tracked time)")
        st.plotly_chart(fig, use_container_width=True)
    st.info(f"Correlation of steps with calories: **{daily['TotalSteps'].corr(daily['Calories']):.2f}** · "
            f"with sedentary hours: **{daily['TotalSteps'].corr(daily['Sedentary_Hours']):.2f}**. "
            "Calories are driven by body size as well as activity, so steps explain only part of the variation.")

# ============================ TIME OF DAY ============================
with tabs[2]:
    if hourly.empty:
        st.info("No hourly data for the current filters.")
    else:
        hm = hourly.groupby(["DayOfWeek", "Hour"])["StepTotal"].mean().reset_index()
        piv = hm.pivot(index="DayOfWeek", columns="Hour", values="StepTotal").reindex(DOW)
        fig = px.imshow(piv, color_continuous_scale="Teal", aspect="auto",
                        labels=dict(x="Hour of day", y="", color="Avg steps"),
                        title="When Are People Moving? Average Steps by Day × Hour")
        st.plotly_chart(fig, use_container_width=True)
        c1, c2 = st.columns(2)
        with c1:
            hh = hourly.groupby(["Hour", "Day_Type"])["StepTotal"].mean().reset_index()
            fig = px.line(hh, x="Hour", y="StepTotal", color="Day_Type", markers=True,
                          color_discrete_sequence=[PURPLE, TEAL], title="Steps by Hour: Weekday vs. Weekend")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            hc = hourly.groupby("Hour")[["Calories", "TotalIntensity"]].mean().reset_index()
            fig = go.Figure()
            fig.add_bar(x=hc["Hour"], y=hc["Calories"], name="Avg calories", marker_color="#F0A04B")
            fig.add_scatter(x=hc["Hour"], y=hc["TotalIntensity"], name="Avg intensity", yaxis="y2",
                            mode="lines+markers", line=dict(color=PURPLE))
            fig.update_layout(title="Calories & Intensity by Hour", yaxis=dict(title="Calories"),
                              yaxis2=dict(title="Intensity", overlaying="y", side="right"),
                              legend=dict(orientation="h"))
            st.plotly_chart(fig, use_container_width=True)
        share = hourly.groupby("Time_Of_Day")["StepTotal"].sum().reindex(TOD).reset_index()
        share["Share %"] = share["StepTotal"] / share["StepTotal"].sum() * 100
        fig = px.bar(share, x="Time_Of_Day", y="Share %", text_auto=".1f", color="Share %",
                     color_continuous_scale="Purples", title="Share of All Steps by Time of Day")
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
        peak = hourly.groupby("Hour")["StepTotal"].mean().idxmax()
        st.info(f"Peak step hour: **{peak}:00**. Activity is concentrated in daytime/early evening and drops sharply overnight.")

# ============================ SLEEP ============================
with tabs[3]:
    if sleep_df.empty:
        st.info("No sleep records for the current filters (only 24 of 33 users tracked sleep).")
    else:
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Nights recorded", f"{len(sleep_df):,}")
        s2.metric("Users with sleep data", sleep_df["Id"].nunique())
        s3.metric("Nights < 7 h", f"{(sleep_df['Sleep_Hours'] < 7).mean() * 100:.0f}%")
        s4.metric("Avg sleep efficiency", f"{sleep_df['Sleep_Efficiency'].mean():.1f}%")
        c1, c2 = st.columns(2)
        with c1:
            fig = px.histogram(sleep_df, x="Sleep_Hours", nbins=30, color_discrete_sequence=[PURPLE],
                               title="Hours Asleep per Night")
            fig.add_vrect(x0=7, x1=9, fillcolor="green", opacity=0.12, annotation_text="7-9h recommended")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.histogram(sleep_df, x="Sleep_Efficiency", nbins=30, color_discrete_sequence=[TEAL],
                               title="Sleep Efficiency (asleep ÷ time in bed, %)")
            st.plotly_chart(fig, use_container_width=True)
        c3, c4 = st.columns(2)
        with c3:
            fig = px.scatter(sleep_df, x="TotalSteps", y="Sleep_Hours", opacity=0.6,
                             color_discrete_sequence=[PURPLE], title="Same-day Steps vs. Sleep Hours")
            ols_line(fig, sleep_df["TotalSteps"], sleep_df["Sleep_Hours"])
            st.plotly_chart(fig, use_container_width=True)
        with c4:
            fig = px.scatter(sleep_df, x="Prev_Day_Steps", y="Sleep_Hours", opacity=0.6,
                             color_discrete_sequence=[TEAL], title="Previous-day Steps vs. Last Night's Sleep")
            ols_line(fig, sleep_df["Prev_Day_Steps"], sleep_df["Sleep_Hours"])
            st.plotly_chart(fig, use_container_width=True)
        c5, c6 = st.columns(2)
        with c5:
            sd = sleep_df.groupby("DayOfWeek")["Sleep_Hours"].mean().reindex(DOW).reset_index()
            fig = px.bar(sd, x="DayOfWeek", y="Sleep_Hours", text_auto=".2f", color="Sleep_Hours",
                         color_continuous_scale="Purples", title="Average Sleep by Night (day the night ends)")
            fig.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
        with c6:
            fig = px.box(sleep_df, x="Activity_Level", y="Sleep_Hours", color="Activity_Level",
                         color_discrete_map=LEVEL_COLORS, category_orders={"Activity_Level": LEVELS},
                         title="Sleep Hours by Activity Level")
            fig.update_layout(showlegend=False, xaxis_tickangle=-20)
            st.plotly_chart(fig, use_container_width=True)
        st.info(f"Steps ↔ sleep correlation is weak (same-day r = {sleep_df['TotalSteps'].corr(sleep_df['Sleep_Hours']):.2f}, "
                f"previous-day r = {sleep_df['Prev_Day_Steps'].corr(sleep_df['Sleep_Hours']):.2f}). "
                "Days with more sedentary time tend to have *less* recorded sleep — likely because time in bed is counted inside the tracked sedentary minutes, so this is partly a measurement effect.")

# ============================ USERS & SEGMENTS ============================
with tabs[4]:
    if u.empty:
        st.info("No users selected.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            fig = px.scatter(u, x="Avg_Steps", y="Avg_Calories", color="Activity_Level", size="Valid_Days",
                             hover_name="User", color_discrete_map=LEVEL_COLORS,
                             category_orders={"Activity_Level": LEVELS},
                             title="Each User: Avg Steps vs. Avg Calories (size = valid days)")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            ct = pd.crosstab(u["Activity_Level"], u["Usage_Tier"]).reindex(index=LEVELS, columns=TIERS).fillna(0)
            fig = px.imshow(ct, text_auto=True, color_continuous_scale="Purples", aspect="auto",
                            title="Users: Activity Level × Device Usage Tier")
            st.plotly_chart(fig, use_container_width=True)
        c3, c4 = st.columns(2)
        with c3:
            top = u.sort_values("Avg_Steps")
            fig = px.bar(top, x="Avg_Steps", y="User", orientation="h", color="Activity_Level",
                         color_discrete_map=LEVEL_COLORS, title="Average Daily Steps per User", height=700)
            st.plotly_chart(fig, use_container_width=True)
        with c4:
            fig = px.scatter(u, x="Avg_Sedentary_Hours", y="Avg_Steps", hover_name="User",
                             color="Meets_WHO_Guideline", title="Sedentary Hours vs. Steps (per user)",
                             color_discrete_map={True: TEAL, False: "#D64550"})
            st.plotly_chart(fig, use_container_width=True)
        st.subheader("Single-user drill-down")
        pick = st.selectbox("Choose a user", u["User"].tolist())
        ud = daily_all[(daily_all["User"] == pick)].sort_values("Date")
        fig = go.Figure()
        fig.add_bar(x=ud["Date"], y=ud["TotalSteps"], name="Steps",
                    marker_color=np.where(ud["Valid_Day"], TEAL, "#CCCCCC"))
        if ud["Sleep_Hours"].notna().any():
            fig.add_scatter(x=ud["Date"], y=ud["Sleep_Hours"], name="Sleep (h)", yaxis="y2",
                            mode="lines+markers", line=dict(color=PURPLE))
        fig.update_layout(title=f"{pick}: daily steps (grey = non-wear/partial) and sleep",
                          yaxis=dict(title="Steps"), yaxis2=dict(title="Sleep hours", overlaying="y", side="right"),
                          legend=dict(orientation="h"))
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(u.drop(columns=["Id"]).round(1), use_container_width=True, hide_index=True)

# ============================ HR & WEIGHT ============================
with tabs[5]:
    st.caption("⚠️ Small samples: heart-rate data exists for 14 of 33 users and weight for only 8 — treat as indicative.")
    c1, c2 = st.columns(2)
    hr = hourly[hourly["HR_Mean"].notna()]
    with c1:
        if hr.empty:
            st.info("No heart-rate data for the current filters.")
        else:
            hh = hr.groupby("Hour")["HR_Mean"].mean().reset_index()
            fig = px.line(hh, x="Hour", y="HR_Mean", markers=True, color_discrete_sequence=["#D64550"],
                          title=f"Average Heart Rate by Hour ({hr['Id'].nunique()} users)")
            st.plotly_chart(fig, use_container_width=True)
    with c2:
        if not hr.empty:
            fig = px.scatter(hr.sample(min(3000, len(hr)), random_state=1), x="TotalIntensity", y="HR_Mean",
                             opacity=0.4, color_discrete_sequence=["#D64550"],
                             title="Hourly Activity Intensity vs. Heart Rate")
            ols_line(fig, hr["TotalIntensity"], hr["HR_Mean"])
            st.plotly_chart(fig, use_container_width=True)
    wt = weight_all[weight_all["Id"].isin(ids)]
    if wt.empty:
        st.info("No weight logs for the current user selection.")
    else:
        c3, c4 = st.columns(2)
        with c3:
            bmi = wt.groupby("User")["BMI"].mean().reset_index().sort_values("BMI")
            fig = px.bar(bmi, x="User", y="BMI", color_discrete_sequence=[PURPLE], text_auto=".1f",
                         title="Average BMI per User (with weight logs)")
            fig.add_hrect(y0=18.5, y1=24.9, fillcolor="green", opacity=0.1, annotation_text="Healthy range")
            st.plotly_chart(fig, use_container_width=True)
        with c4:
            fig = px.line(wt.sort_values("Date"), x="Date", y="WeightKg", color="User", markers=True,
                          title="Weight Logs Over Time")
            st.plotly_chart(fig, use_container_width=True)
        man = wt["IsManualReport"].astype(str).str.lower().isin(["true", "1"]).mean() * 100
        st.info(f"**{man:.0f}%** of weight entries were logged manually — manual entry is a friction point for adoption.")

# ============================ INSIGHTS ============================
with tabs[6]:
    st.subheader("Key insights (computed live from the current filters)")
    dow_m = daily.groupby("DayOfWeek")["TotalSteps"].mean()
    best_d, worst_d = dow_m.idxmax(), dow_m.idxmin()
    ins = []
    ins.append(f"**Activity is modest.** Average {daily['TotalSteps'].mean():,.0f} steps/day; only "
               f"{(daily['TotalSteps'] >= 10000).mean() * 100:.0f}% of valid days reach 10,000 steps.")
    ins.append(f"**Sedentary time dominates**: {daily['Sedentary_Hours'].mean():.1f} h/day sedentary vs. "
               f"{daily['VeryActiveMinutes'].mean():.0f} very-active minutes.")
    ins.append(f"**Best day: {best_d}** ({dow_m.max():,.0f} steps); **weakest: {worst_d}** ({dow_m.min():,.0f}).")
    if not hourly.empty:
        ph = hourly.groupby("Hour")["StepTotal"].mean().idxmax()
        ins.append(f"**Peak movement hour is {ph}:00**; activity clusters around midday and early evening.")
    if len(sleep_df):
        ins.append(f"**Sleep is short for many:** {(sleep_df['Sleep_Hours'] < 7).mean() * 100:.0f}% of tracked nights are under 7 h "
                   f"(avg {sleep_df['Sleep_Hours'].mean():.1f} h), yet efficiency is high ({sleep_df['Sleep_Efficiency'].mean():.0f}%).")
    if len(u):
        ins.append(f"**{u['Meets_WHO_Guideline'].mean() * 100:.0f}%** of users meet the WHO-equivalent activity guideline, "
                   f"but {(u['Activity_Level'].isin(LEVELS[:2])).mean() * 100:.0f}% average under 7,500 steps/day.")
        ins.append(f"**Engagement gap:** only {u['Has_Sleep'].sum()} of {len(u)} users track sleep, "
                   f"{u['Has_Weight'].sum()} log weight and {u['Has_HR'].sum()} have heart-rate data.")
    for t in ins:
        st.markdown(f"- {t}")
    st.subheader("Recommendations")
    rec = pd.DataFrame([
        ("Sedentary majority of day", "Smart 'move reminders' during long sedentary blocks, especially mid-afternoon."),
        (f"{best_d} strongest / {worst_d} weakest", f"Time challenges to lift {worst_d}; use {best_d} for social/community events."),
        ("Activity peaks midday & early evening", "Send motivation nudges shortly before peak windows; evening wind-down cues later."),
        ("Short sleep despite high efficiency", "Promote bedtime routines / sleep-goal features rather than only sleep tracking."),
        ("Low sleep/weight/HR adoption", "Onboard users to sleep & auto-sync weight; reduce manual logging friction."),
        ("Activity-level segments", "Tailor goals: gentle step ramps for Sedentary/Low Active, intensity goals for Active users."),
    ], columns=["Insight", "Recommendation"])
    st.dataframe(rec, use_container_width=True, hide_index=True)
    st.caption("Sample: 33 users over 30 days (2016), self-selected — findings are indicative, not population-representative.")

# ============================ DATA QUALITY ============================
with tabs[7]:
    st.subheader("Cleaning log")
    st.dataframe(dq, use_container_width=True, hide_index=True)
    cov = pd.DataFrame({"Data type": ["Daily activity", "Sleep", "Weight", "Heart rate"],
                        "Users": [len(users_all), int(users_all["Has_Sleep"].sum()),
                                  int(users_all["Has_Weight"].sum()), int(users_all["Has_HR"].sum())]})
    fig = px.bar(cov, x="Data type", y="Users", text_auto=True, color_discrete_sequence=[PURPLE],
                 title="Data Coverage: Number of Users (of 33) per Data Type")
    st.plotly_chart(fig, use_container_width=True)
    st.subheader("Filtered daily data")
    st.dataframe(daily.drop(columns=["Id"]), use_container_width=True, height=380)
    st.download_button("Download filtered daily data (CSV)", daily.drop(columns=["Id"]).to_csv(index=False).encode(),
                       "fitbit_daily_filtered.csv", "text/csv")

st.markdown("---")
st.caption("Pipeline: app.py → SQLite (db/fitbit.db) → this dashboard.")
