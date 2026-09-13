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


CATEGORY_COLORS = {
    "Productive": "#2ecc71",
    "Learning": "#16a085",
    "Necessary": "#f1c40f",
    "Leisure": "#3498db",
    "Unproductive": "#e74c3c",
    "Unknown": "#95a5a6"
}

base_df = pd.read_csv("data/sample_activity.csv")

if "manual_entries" not in st.session_state:
    st.session_state.manual_entries = []
if "categorized_df" not in st.session_state:
    st.session_state.categorized_df = None
if "show_demo_data" not in st.session_state:
    st.session_state.show_demo_data = True

st.divider()
st.subheader("📋 Your Activity")
st.caption("Showing a demo dataset below. Add your own activities anytime, on top of it, or remove the demo entirely.")

with st.expander("➕ Add your own activity"):
    with st.form("manual_entry_form", clear_on_submit=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            entry_date = st.date_input("Date")
        with col2:
            entry_activity = st.text_input("Activity name (e.g. YouTube)")
        with col3:
            entry_duration = st.number_input("Duration (minutes)", min_value=1, step=5)
        submitted = st.form_submit_button("Add activity")
        if submitted and entry_activity:
            st.session_state.manual_entries.append({
                "date": str(entry_date),
                "activity": entry_activity,
                "duration_minutes": entry_duration
            })
            st.success(f"Added {entry_activity}")

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

active_base_df = base_df if st.session_state.show_demo_data else pd.DataFrame(columns=["date", "activity", "duration_minutes"])

if st.session_state.manual_entries:
    manual_df = pd.DataFrame(st.session_state.manual_entries)
    df = pd.concat([active_base_df, manual_df], ignore_index=True)
else:
    df = active_base_df

if df.empty:
    st.info("No activity data yet. Add your own above, or restore the demo data.")
else:
    display_df = df.copy()
    display_df["duration_minutes"] = display_df["duration_minutes"].apply(format_duration)
    display_df = display_df.rename(columns={"date": "Date", "activity": "Activity", "duration_minutes": "Duration"})
    st.dataframe(display_df, use_container_width=True, hide_index=True)


def categorize_activities(dataframe):
    activity_list = dataframe.to_dict(orient="records")

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

Activity data:
{json.dumps(activity_list)}

Respond ONLY with valid JSON, a list of objects, each with keys: date, activity, duration_minutes, category, interpretation. No other text, no markdown formatting."""

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )

    raw = response.choices[0].message.content.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    return pd.DataFrame(json.loads(raw))


st.divider()
if st.button("🔍 Analyze My Time", type="primary", help="Let the AI figure out what your time was actually spent on"):
    with st.spinner("Understanding where your time actually went..."):
        try:
            st.session_state.categorized_df = categorize_activities(df)
            st.session_state.pop("why_analysis", None)
            st.session_state.pop("what_if_answer", None)
        except json.JSONDecodeError:
            st.error("Something went wrong reading the AI's response. Try clicking Analyze again.")

if st.session_state.categorized_df is not None:
    st.subheader("🗂️ Categorized Activity")
    display_cat = st.session_state.categorized_df.copy()
    display_cat["duration_minutes"] = display_cat["duration_minutes"].apply(format_duration)
    display_cat = display_cat.rename(columns={"date": "Date", "activity": "Activity", "duration_minutes": "Duration", "category": "Category", "interpretation": "AI's Reasoning"})
    st.dataframe(display_cat, use_container_width=True, hide_index=True)

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

    total_minutes = cat_df["duration_minutes"].sum()
    col1, col2 = st.columns(2)

    with col1:
        summary = cat_df.groupby("category")["duration_minutes"].sum().reset_index()
        fig = px.pie(summary, values="duration_minutes", names="category",
                     title=f"Total tracked: {total_minutes // 60}h {total_minutes % 60}m",
                     color="category",
                     color_discrete_map=CATEGORY_COLORS)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        leak = cat_df[cat_df["category"].isin(["Unproductive", "Leisure"])] \
                .groupby("activity")["duration_minutes"].sum() \
                .sort_values(ascending=False)
        if not leak.empty:
            biggest_leak = leak.index[0]
            biggest_leak_time = leak.iloc[0]
            st.metric("Biggest time leak", biggest_leak, f"{biggest_leak_time} min")
        else:
            st.write("No clear time leak detected yet.")

        by_activity = cat_df.groupby("activity")["duration_minutes"].sum().sort_values(ascending=False).reset_index()
        fig2 = px.bar(by_activity, x="activity", y="duration_minutes", title="Time by activity")
        st.plotly_chart(fig2, use_container_width=True)

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
Full log with categories: {cat_df.to_json(orient="records")}

Write a short analysis (4-6 sentences) that:
1. Identifies their single biggest time-loss pattern, not just the biggest number, but a *pattern* (e.g. repeated short sessions, one dominant distraction, task-switching if the data suggests it).
2. Gives a specific estimate in hours/minutes where possible.
3. Ends with one concrete, specific recommendation for tomorrow, not generic advice like "use your phone less."

Write directly to the user as "you." Be specific and grounded in the actual numbers given, don't make up patterns the data doesn't support."""

                why_response = client.chat.completions.create(
                    model="openai/gpt-oss-120b",
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

    what_if_question = st.text_input("e.g. 'What if I stop using Instagram during study hours?'")

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
                    model="openai/gpt-oss-120b",
                    messages=[{"role": "user", "content": what_if_prompt}],
                    temperature=0.4
                )
                st.session_state.what_if_answer = what_if_response.choices[0].message.content
            except Exception as e:
                st.error(f"Estimate failed, try again. ({e})")

    if "what_if_answer" in st.session_state:
        st.success(st.session_state.what_if_answer)
