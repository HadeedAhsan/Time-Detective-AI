import streamlit as st
import pandas as pd
from groq import Groq
import os
import json
import plotly.express as px
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", None)
client = Groq(api_key=api_key)

MODEL = "openai/gpt-oss-120b"
GAP_THRESHOLD_MINUTES = 60  # gaps longer than this are treated as a break, not a switch

VALID_CATEGORIES = ["Productive", "Learning", "Necessary", "Leisure", "Unproductive", "Unknown"]
REQUIRED_COLUMNS = ["row_id", "date", "activity", "duration_minutes", "category", "interpretation"]

st.set_page_config(page_title="Time Detective AI", layout="wide")

st.markdown("""
<div style="
    background: linear-gradient(135deg, #0369A1 0%, #0EA5E9 60%, #38BDF8 100%);
    padding: 2.5rem 2rem;
    border-radius: 16px;
    margin-bottom: 1.5rem;
">
    <h1 style="color: #FFFFFF; margin: 0; font-size: 2.2rem;">🕵️ Time Detective AI</h1>
    <p style="color: #E0F2FE; margin-top: 0.6rem; font-size: 1.05rem; max-width: 640px;">
        Most trackers tell you <strong>what</strong> you did. This tells you <strong>why</strong> you're actually losing time, and what to try instead.
    </p>
</div>
""", unsafe_allow_html=True)


def format_duration(minutes):
    minutes = int(minutes)
    h = minutes // 60
    m = minutes % 60
    if h > 0 and m > 0:
        return f"{h}h {m}m"
    elif h > 0:
        return f"{h}h"
    else:
        return f"{m}m"


def data_fingerprint(dataframe):
    if dataframe.empty:
        return "empty"
    cols = [c for c in ["date", "activity", "duration_minutes", "start_time"] if c in dataframe.columns]
    return str(dataframe[cols].fillna("").sort_values(by=["date", "activity"]).to_dict(orient="records"))


def compute_switching_stats(cat_df, gap_threshold=GAP_THRESHOLD_MINUTES):
    timed = cat_df.dropna(subset=["start_time"]).copy()
    timed = timed[timed["start_time"].astype(str).str.strip() != ""]
    if len(timed) < 2:
        return None

    timed["start_dt"] = pd.to_datetime(timed["date"].astype(str) + " " + timed["start_time"].astype(str), errors="coerce")
    timed = timed.dropna(subset=["start_dt"]).sort_values("start_dt").reset_index(drop=True)
    if len(timed) < 2:
        return None

    timed["end_dt"] = timed["start_dt"] + pd.to_timedelta(timed["duration_minutes"], unit="m")

    switches = []
    sessions = []
    session_start = timed.loc[0, "start_dt"]
    session_activity = timed.loc[0, "activity"]

    for i in range(1, len(timed)):
        prev_end = timed.loc[i - 1, "end_dt"]
        cur_start = timed.loc[i, "start_dt"]
        gap_minutes = (cur_start - prev_end).total_seconds() / 60
        same_activity = timed.loc[i, "activity"] == timed.loc[i - 1, "activity"]

        if gap_minutes > gap_threshold:
            sessions.append({"activity": session_activity, "minutes": (prev_end - session_start).total_seconds() / 60})
            session_start = cur_start
            session_activity = timed.loc[i, "activity"]
            continue

        if not same_activity:
            switches.append({
                "from": timed.loc[i - 1, "activity"],
                "from_category": timed.loc[i - 1, "category"],
                "to": timed.loc[i, "activity"],
            })
            sessions.append({"activity": session_activity, "minutes": (prev_end - session_start).total_seconds() / 60})
            session_start = cur_start
            session_activity = timed.loc[i, "activity"]

    sessions.append({"activity": session_activity, "minutes": (timed.loc[len(timed) - 1, "end_dt"] - session_start).total_seconds() / 60})

    if not switches:
        return {"num_switches": 0}

    avg_session = sum(s["minutes"] for s in sessions) / len(sessions)
    longest = max(sessions, key=lambda s: s["minutes"])

    post_productive = [s["to"] for s in switches if s["from_category"] == "Productive"]
    top_post_productive = pd.Series(post_productive).value_counts().idxmax() if post_productive else None

    return {
        "num_switches": len(switches),
        "avg_session_minutes": avg_session,
        "longest_session": longest,
        "top_post_productive": top_post_productive,
        "switch_log": switches,
    }


CATEGORY_COLORS = {
    "Productive": "#0072B2",
    "Learning": "#009E73",
    "Necessary": "#F0E442",
    "Leisure": "#56B4E9",
    "Unproductive": "#D55E00",
    "Unknown": "#999999"
}

base_df = pd.read_csv("data/sample_activity.csv")

if "manual_entries" not in st.session_state:
    st.session_state.manual_entries = []
if "categorized_df" not in st.session_state:
    st.session_state.categorized_df = None
if "analyzed_fingerprint" not in st.session_state:
    st.session_state.analyzed_fingerprint = None
if "show_demo_data" not in st.session_state:
    st.session_state.show_demo_data = True

st.divider()
st.subheader("📋 Your Activity")
st.caption("Showing a demo dataset below. Add your own activities anytime, on top of it, or remove the demo entirely.")

with st.expander("➕ Add your own activity", expanded=bool(st.session_state.manual_entries)):
    with st.form("manual_entry_form", clear_on_submit=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            entry_date = st.date_input("Date")
        with col2:
            entry_activity = st.text_input("Activity name (e.g. YouTube)")
        with col3:
            entry_duration = st.number_input("Duration (minutes)", min_value=0, step=5)

        time_col, skip_col = st.columns([2, 1])
        with time_col:
            entry_time = st.time_input("Start time (unlocks switching analysis below)")
        with skip_col:
            st.write("")
            st.write("")
            dont_know_time = st.checkbox("I don't know")

        submitted = st.form_submit_button("Add activity")
        if submitted and entry_activity:
            st.session_state.manual_entries.append({
                "date": str(entry_date),
                "activity": entry_activity.strip(),
                "duration_minutes": int(entry_duration),
                "start_time": None if dont_know_time else entry_time.strftime("%H:%M")
            })
            st.success(f"Added {entry_activity}")

    if st.session_state.manual_entries:
        st.markdown("**Your added activities:**")
        for i, entry in enumerate(st.session_state.manual_entries):
            row_col1, row_col2, row_col3 = st.columns([3, 2, 1])
            with row_col1:
                st.write(f"{entry['activity']}")
            with row_col2:
                time_part = f" at {entry['start_time']}" if entry.get("start_time") else ""
                st.write(f"{entry['date']}{time_part} · {entry['duration_minutes']} min")
            with row_col3:
                if st.button("🗑️", key=f"delete_manual_{i}", help="Remove this activity"):
                    st.session_state.manual_entries.pop(i)
                    st.rerun()

uploaded_file = st.file_uploader("📤 Or upload your own activity CSV", type="csv", help="Required columns: date, activity, duration_minutes. Optional: start_time (HH:MM)")
uploaded_df = pd.DataFrame(columns=["date", "activity", "duration_minutes", "start_time"])
if uploaded_file is not None:
    try:
        uploaded_df = pd.read_csv(uploaded_file)
        uploaded_df.columns = [str(c).strip().lower().replace("\ufeff", "") for c in uploaded_df.columns]

        missing_cols = [c for c in ["date", "activity", "duration_minutes"] if c not in uploaded_df.columns]
        if missing_cols:
            found = ", ".join(uploaded_df.columns) if len(uploaded_df.columns) else "(no columns detected)"
            st.error(f"Your CSV is missing required column(s): {', '.join(missing_cols)}. Columns found instead: {found}")
            uploaded_df = pd.DataFrame(columns=["date", "activity", "duration_minutes", "start_time"])
        else:
            if "start_time" not in uploaded_df.columns:
                uploaded_df["start_time"] = None
            st.success(f"Loaded {len(uploaded_df)} activities from your file.")
    except Exception as e:
        st.error(f"Couldn't read that file: {e}")

active_base_df = base_df.copy() if st.session_state.show_demo_data else pd.DataFrame(columns=["date", "activity", "duration_minutes", "start_time"])
if "start_time" not in active_base_df.columns:
    active_base_df["start_time"] = None

col_a, col_b = st.columns([5, 1])
with col_b:
    if st.session_state.show_demo_data:
        if st.button("🗑️ Remove demo data", help="Clear the sample rows and see only what you add"):
            st.session_state.show_demo_data = False
            st.rerun()
    else:
        if st.button("↩️ Restore demo data", help="Bring back the sample rows"):
            st.session_state.show_demo_data = True
            st.rerun()

frames = [active_base_df, uploaded_df]
if st.session_state.manual_entries:
    frames.append(pd.DataFrame(st.session_state.manual_entries))
df = pd.concat(frames, ignore_index=True)

if not df.empty:
    df["duration_minutes"] = pd.to_numeric(df["duration_minutes"], errors="coerce").fillna(0).astype(int)

if df.empty:
    st.info("No activity data yet. Add your own above, or restore the demo data.")
else:
    display_df = df.copy()
    display_df["duration_minutes"] = display_df["duration_minutes"].apply(format_duration)
    display_df["start_time"] = display_df["start_time"].fillna("—")
    display_df = display_df.rename(columns={"date": "Date", "activity": "Activity", "duration_minutes": "Duration", "start_time": "Start Time"})
    st.dataframe(display_df, use_container_width=True, hide_index=True)


def categorize_activities(dataframe):
    activity_list = dataframe[["row_id", "date", "activity", "duration_minutes"]].to_dict(orient="records")

    prompt = f"""You are analyzing a user's digital activity log. For each entry, classify it into exactly one of these categories: Productive, Learning, Necessary, Leisure, Unproductive, Unknown.

Use these reasoning patterns to make your best judgment call, don't default to Unknown just because the app name alone is ambiguous:
- "Learning" is for deliberate skill-building: tutorials, courses, Duolingo, educational content, even if the app is normally entertainment (e.g. a YouTube tutorial is Learning, not Productive or Leisure).
- "Productive" is hands-on work output: coding, writing, actual task execution, not passive learning.
- Duration matters: a 5-15 minute WhatsApp check is likely Necessary, but 45+ minutes leans Leisure or Unproductive.
- Assume plain YouTube (not clearly a tutorial) over 45 minutes leans Leisure.
- Instagram, TikTok, Netflix, and similar entertainment apps default to Leisure, trending Unproductive if duration exceeds an hour.
- Zoom/meetings, email, and calendar tools are Necessary.
- Only use "Unknown" for genuinely unrecognizable or nonsense activity names, not common apps you can reason about.
- Give a short one-sentence interpretation for each entry explaining your reasoning.

Each input object has a "row_id", copy it back unchanged in your output so entries can be matched up.

Return exactly one object per input entry, {len(activity_list)} in total.

Activity data:
{json.dumps(activity_list)}

Respond ONLY with valid JSON, a list of objects, each with keys: row_id, date, activity, duration_minutes, category, interpretation. No other text, no markdown formatting."""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )

    raw = response.choices[0].message.content.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    result = pd.DataFrame(json.loads(raw))

    missing = [c for c in REQUIRED_COLUMNS if c not in result.columns]
    if missing:
        raise ValueError(f"The AI's response was missing these fields: {', '.join(missing)}")

    if result.empty:
        raise ValueError("The AI returned no results.")

    result["duration_minutes"] = pd.to_numeric(result["duration_minutes"], errors="coerce").fillna(0).astype(int)
    result.loc[~result["category"].isin(VALID_CATEGORIES), "category"] = "Unknown"

    result = result.merge(dataframe[["row_id", "start_time"]], on="row_id", how="left")

    return result.drop(columns=["row_id"])


st.divider()
if st.button("🔍 Analyze My Time", type="primary", help="Let the AI figure out what your time was actually spent on"):
    if df.empty:
        st.warning("There's no activity data to analyze yet. Add an activity or restore the demo data first.")
    elif st.session_state.categorized_df is not None and data_fingerprint(df) == st.session_state.analyzed_fingerprint:
        st.info("This data was already analyzed, results below are still current. Change your activity data to run a new analysis.")
    else:
        with st.spinner("Understanding where your time actually went..."):
            try:
                df_with_id = df.reset_index(drop=True).copy()
                df_with_id["row_id"] = range(len(df_with_id))
                st.session_state.categorized_df = categorize_activities(df_with_id)
                st.session_state.analyzed_fingerprint = data_fingerprint(df)
                st.session_state.pop("why_analysis", None)
                st.session_state.pop("what_if_answer", None)
                st.session_state.pop("switch_analysis", None)
            except json.JSONDecodeError:
                st.error("The AI's response wasn't readable. Try clicking Analyze again.")
            except ValueError as e:
                st.error(f"{e} Try clicking Analyze again.")
            except Exception as e:
                st.error(f"Analysis failed. Try again in a moment. ({e})")

if st.session_state.categorized_df is not None:
    if st.session_state.analyzed_fingerprint != data_fingerprint(df):
        st.warning("Your activity data has changed since this analysis was run. Click **Analyze My Time** again to update the results below.")

    st.subheader("🗂️ Categorized Activity")
    display_cat = st.session_state.categorized_df.copy()
    display_cat["duration_minutes"] = display_cat["duration_minutes"].apply(format_duration)
    display_cat = display_cat.rename(columns={"date": "Date", "activity": "Activity", "duration_minutes": "Duration", "category": "Category", "interpretation": "AI's Reasoning"})
    st.dataframe(display_cat.drop(columns=["start_time"], errors="ignore"), use_container_width=True, hide_index=True)

    st.download_button(
        "⬇️ Download results as CSV",
        data=st.session_state.categorized_df.to_csv(index=False),
        file_name="time_detective_results.csv",
        mime="text/csv"
    )

    unknown_df = st.session_state.categorized_df[st.session_state.categorized_df["category"] == "Unknown"]

    if not unknown_df.empty:
        st.divider()
        st.subheader("🤷 A Few Activities Need Your Input")
        st.caption(f"The AI wasn't confident about {len(unknown_df)} activities. Pick the closest fit:")
        with st.form("clarify_form"):
            clarifications = {}
            for idx, row in unknown_df.iterrows():
                clarifications[idx] = st.selectbox(
                    f"{row['activity']} ({row['duration_minutes']} min on {row['date']})",
                    ["Productive", "Learning", "Necessary", "Leisure", "Unproductive"],
                    key=f"clarify_{idx}"
                )
            clarify_submitted = st.form_submit_button("Apply")
            if clarify_submitted:
                for idx, category in clarifications.items():
                    st.session_state.categorized_df.at[idx, "category"] = category
                    st.session_state.categorized_df.at[idx, "interpretation"] = "Manually clarified by user"
                st.success("Updated, your dashboard below now reflects this.")

    cat_df = st.session_state.categorized_df

    st.divider()
    st.subheader("📊 Insights Dashboard")

    total_minutes = int(cat_df["duration_minutes"].sum())
    col1, col2 = st.columns(2)

    with col1:
        summary = cat_df.groupby("category")["duration_minutes"].sum().reset_index()
        fig = px.pie(summary, values="duration_minutes", names="category",
                     title=f"Total tracked: {total_minutes // 60}h {total_minutes % 60}m",
                     color="category",
                     color_discrete_map=CATEGORY_COLORS,
                     labels={"duration_minutes": "Duration (minutes)", "category": "Category"})
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("📄 View as text (screen reader friendly)"):
            summary_sorted = summary.sort_values("duration_minutes", ascending=False)
            for _, row in summary_sorted.iterrows():
                pct = (row["duration_minutes"] / total_minutes * 100) if total_minutes else 0
                st.write(f"**{row['category']}**: {format_duration(row['duration_minutes'])} ({pct:.0f}% of total)")

    with col2:
        leak = cat_df[cat_df["category"].isin(["Unproductive", "Leisure"])] \
                .groupby("activity")["duration_minutes"].sum() \
                .sort_values(ascending=False)
        if not leak.empty:
            biggest_leak = leak.index[0]
            biggest_leak_time = int(leak.iloc[0])
            st.metric("Biggest time leak", biggest_leak, f"{biggest_leak_time} min")
        else:
            st.write("No clear time leak detected yet.")

        by_activity = cat_df.groupby("activity")["duration_minutes"].sum().sort_values(ascending=False).reset_index()
        fig2 = px.bar(by_activity, x="activity", y="duration_minutes", title="Time by activity",
                      labels={"duration_minutes": "Duration (minutes)", "activity": "Activity"})
        st.plotly_chart(fig2, use_container_width=True)

        with st.expander("📄 View as text (screen reader friendly)"):
            for _, row in by_activity.iterrows():
                st.write(f"**{row['activity']}**: {format_duration(row['duration_minutes'])}")

    st.divider()
    st.subheader("🔀 When Do I Switch?")
    st.caption(f"Computed directly from your timestamps, no AI guessing. Gaps over {GAP_THRESHOLD_MINUTES} minutes count as a break, not a switch.")

    switch_stats = compute_switching_stats(cat_df)

    if switch_stats is None:
        st.info("Add start times to at least two activities (via the form above, or a CSV with a start_time column) to unlock this section.")
    elif switch_stats["num_switches"] == 0:
        st.info("Not enough switching detected yet to find a pattern, try adding more timestamped activities.")
    else:
        sc1, sc2, sc3 = st.columns(3)
        sc1.metric("Times you switched tasks", switch_stats["num_switches"])
        sc2.metric("Avg. time before switching", format_duration(int(switch_stats["avg_session_minutes"])))
        sc3.metric("Longest uninterrupted stretch", format_duration(int(switch_stats["longest_session"]["minutes"])), switch_stats["longest_session"]["activity"])

        if switch_stats["top_post_productive"]:
            st.write(f"After a **Productive** block, you most often move straight to **{switch_stats['top_post_productive']}**.")

        if st.button("🧠 Explain my switching pattern"):
            with st.spinner("Looking at your timing patterns..."):
                try:
                    switch_prompt = f"""You are a personal productivity analyst. Based on this user's task-switching data (computed directly from their timestamps), explain what's actually happening.

Number of switches: {switch_stats['num_switches']}
Average time spent before switching to something else: {switch_stats['avg_session_minutes']:.0f} minutes
Longest uninterrupted stretch: {switch_stats['longest_session']['minutes']:.0f} minutes on {switch_stats['longest_session']['activity']}
Most common activity right after a Productive block: {switch_stats['top_post_productive'] or 'not enough data'}
Full switch log: {json.dumps(switch_stats['switch_log'])}

Write a short analysis (3-5 sentences) explaining what this switching pattern reveals, particularly whether there's a sign of avoidance (bailing to something easy right after productive work) versus just normal task variety. End with one concrete, specific suggestion. Write directly to the user as "you". Only claim what the numbers actually support."""

                    switch_response = client.chat.completions.create(
                        model=MODEL,
                        messages=[{"role": "user", "content": switch_prompt}],
                        temperature=0.4
                    )
                    st.session_state.switch_analysis = switch_response.choices[0].message.content
                except Exception as e:
                    st.error(f"Analysis failed, try again. ({e})")

        if "switch_analysis" in st.session_state:
            st.info(st.session_state.switch_analysis)

    st.divider()
    st.subheader("🤔 Why Am I Losing Time?")
    st.caption("Get a specific, numbers-based explanation, not generic advice.")

    if st.button("Explain my patterns", help="Ask the AI to find the real pattern behind your numbers"):
        with st.spinner("Analyzing your patterns..."):
            try:
                summary_stats = cat_df.groupby("category")["duration_minutes"].sum().to_dict()
                activity_stats = cat_df.groupby("activity")["duration_minutes"].sum().sort_values(ascending=False).to_dict()

                why_prompt = f"""You are a personal productivity analyst. Based on this user's categorized activity data, explain WHY they're losing time, not just what they did.

Category totals (minutes): {json.dumps(summary_stats)}
Activity totals (minutes): {json.dumps(activity_stats)}
Full log with categories: {cat_df.drop(columns=["start_time"], errors="ignore").to_json(orient="records")}

Write a short analysis (4-6 sentences) that:
1. Identifies their single biggest time-loss pattern, not just the biggest number, but a *pattern* (e.g. repeated short sessions, one dominant distraction, task-switching if the data suggests it).
2. Gives a specific estimate in hours/minutes where possible.
3. Ends with one concrete, specific recommendation for tomorrow, not generic advice like "use your phone less."

Write directly to the user as "you." Be specific and grounded in the actual numbers given, don't make up patterns the data doesn't support."""

                why_response = client.chat.completions.create(
                    model=MODEL,
                    messages=[{"role": "user", "content": why_prompt}],
                    temperature=0.4
                )
                st.session_state.why_analysis = why_response.choices[0].message.content
            except Exception as e:
                st.error(f"Analysis failed, try clicking the button again. ({e})")

    if "why_analysis" in st.session_state:
        st.info(st.session_state.why_analysis)

    st.divider()
    st.subheader("💭 What If I Changed Something?")
    st.caption("Ask a hypothetical, get a real estimate based on your tracked data.")

    what_if_question = st.text_input("Your hypothetical question", placeholder="e.g. What if I stop using Instagram during study hours?")

    if st.button("Estimate impact", help="See a real projection based on your actual averages") and what_if_question:
        with st.spinner("Running the numbers..."):
            try:
                activity_stats = cat_df.groupby("activity")["duration_minutes"].sum().to_dict()
                days_tracked = cat_df["date"].nunique()

                what_if_prompt = f"""You are a personal productivity analyst. The user has tracked {days_tracked} days of activity.

Activity totals across those {days_tracked} days (minutes): {json.dumps(activity_stats)}

The user's hypothetical question: "{what_if_question}"

Using the real numbers above, estimate the time impact of this change. Show your math briefly (e.g. "X minutes/day average, over {days_tracked} days tracked, projected to Y hours/week or Z hours/month"). Be honest that this is an estimate based on limited tracked data, not a guarantee. Keep it to 3-4 sentences, direct and specific, no generic advice."""

                what_if_response = client.chat.completions.create(
                    model=MODEL,
                    messages=[{"role": "user", "content": what_if_prompt}],
                    temperature=0.4
                )
                st.session_state.what_if_answer = what_if_response.choices[0].message.content
            except Exception as e:
                st.error(f"Estimate failed, try again. ({e})")

    if "what_if_answer" in st.session_state:
        st.success(st.session_state.what_if_answer)