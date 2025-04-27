import streamlit as st
import pandas as pd
from datetime import datetime

# Set Streamlit page config
st.set_page_config(page_title="MLB NRFI Dashboard", layout="wide")

# Load Data
@st.cache_data
def load_predictions():
    return pd.read_csv("data/mlb_nrfi_predictions.csv")

@st.cache_data
def load_results():
    return pd.read_csv("data/mlb_nrfi_results_full.csv")

# Load everything
pred_df = load_predictions()
results_df = load_results()

# 🔥 Helper: Fireball Rating
def assign_fireball(prob):
    if prob >= 70:
        return "🔥🔥🔥🔥🔥"
    elif prob >= 65:
        return "🔥🔥🔥🔥"
    elif prob >= 55:
        return "🔥🔥🔥"
    elif prob >= 50:
        return "🔥🔥"
    elif prob >= 45:
        return "🔥"
    else:
        return "no value"

# Assign Fireball Rating
if 'Predicted_NRFI_Probability' in pred_df.columns:
    pred_df['Fireball_Rating'] = pred_df['Predicted_NRFI_Probability'].apply(assign_fireball)

# Merge predictions with actual results
df = pred_df.merge(
    results_df[['Game Date', 'Away Team', 'Home Team', 'Actual_1st_Inning_Runs', 'Prediction_Result']],
    on=['Game Date', 'Away Team', 'Home Team'],
    how='left'
)

# Only fill missing prediction results with "Pending"
df['Prediction_Result'] = df['Prediction_Result'].fillna("Pending")

# 🔥 IMPORTANT: Only use completed games for evaluation
evaluation_df = df[df['Actual_1st_Inning_Runs'] != "Pending"]

# Title
st.title("⚾ MLB NRFI Predictions Dashboard (Fireball Model Only)")

# View selector
view_option = st.radio("Select View:", ["Predictions Only", "Predictions vs Actual Results"])

# Date Picker
available_dates = sorted(pd.to_datetime(pred_df['Game Date']).unique())
selected_date = st.date_input(
    "Select Game Date:",
    value=datetime.today(),
    min_value=available_dates[0],
    max_value=available_dates[-1]
)

selected_date_str = selected_date.strftime('%Y-%m-%d')

# Filter by selected date
filtered_df = df[df['Game Date'] == selected_date_str]
filtered_eval_df = evaluation_df[evaluation_df['Game Date'] == selected_date_str]

# ------------------------------
# 📋 Display Predictions
# ------------------------------

if view_option == "Predictions Only":
    st.subheader(f"NRFI Predictions for {selected_date_str}")
    st.dataframe(
        filtered_df[['Away Team', 'Home Team', 'Fireball_Rating']]
        .sort_values(by="Fireball_Rating", ascending=False),
        use_container_width=True
    )

    st.subheader("🔥 Best NRFI Bets (Fireballs 4 or 5)")
    best_bets = filtered_df[filtered_df['Predicted_NRFI_Probability'] >= 70]
    if not best_bets.empty:
        st.dataframe(
            best_bets[['Away Team', 'Home Team', 'Fireball_Rating']]
            .sort_values(by="Fireball_Rating", ascending=False),
            use_container_width=True
        )
    else:
        st.write("No high-confidence NRFI games today. ❌")

elif view_option == "Predictions vs Actual Results":
    st.subheader(f"NRFI Predictions vs Actual for {selected_date_str}")
    st.dataframe(
        filtered_eval_df[['Away Team', 'Home Team', 'Fireball_Rating', 'Actual_1st_Inning_Runs', 'Prediction_Result']]
        .sort_values(by="Fireball_Rating", ascending=False),
        use_container_width=True
    )

# ------------------------------
# 📊 DAILY SUMMARY
# ------------------------------

st.markdown("---")
st.subheader("📈 Daily Summary for Selected Date")

if view_option == "Predictions vs Actual Results":

    daily_total = len(filtered_eval_df)
    daily_hits = (filtered_eval_df['Prediction_Result'] == '✅ HIT').sum()
    daily_win_rate = (daily_hits / daily_total) * 100 if daily_total > 0 else 0

    st.write(f"**Daily Model Record:** {daily_hits} Wins / {daily_total - daily_hits} Losses ({daily_win_rate:.2f}% Win Rate)")

    fireball_daily = filtered_eval_df.groupby('Fireball_Rating').agg(
        Total=('Prediction_Result', 'count'),
        Wins=('Prediction_Result', lambda x: (x == '✅ HIT').sum())
    )
    fireball_daily['Win %'] = (fireball_daily['Wins'] / fireball_daily['Total'] * 100).round(2)

    fireball_order = ["🔥🔥🔥🔥🔥", "🔥🔥🔥🔥", "🔥🔥🔥", "🔥🔥", "🔥", "no value"]
    fireball_daily = fireball_daily.reset_index()
    fireball_daily['Fireball_Rating'] = fireball_daily['Fireball_Rating'].astype(str)
    fireball_daily['SortOrder'] = fireball_daily['Fireball_Rating'].apply(lambda x: fireball_order.index(x) if x in fireball_order else len(fireball_order))
    fireball_daily = fireball_daily.sort_values(by='SortOrder').drop(columns=['SortOrder']).set_index('Fireball_Rating')

    st.dataframe(fireball_daily, use_container_width=True)

# ------------------------------
# 📊 CUMULATIVE SUMMARY
# ------------------------------

st.markdown("---")
st.subheader("📈 Cumulative Summary (All Days)")

if view_option == "Predictions vs Actual Results":

    cumulative_total = len(evaluation_df)
    cumulative_hits = (evaluation_df['Prediction_Result'] == '✅ HIT').sum()
    cumulative_win_rate = (cumulative_hits / cumulative_total) * 100 if cumulative_total > 0 else 0

    st.write(f"**Cumulative Model Record:** {cumulative_hits} Wins / {cumulative_total - cumulative_hits} Losses ({cumulative_win_rate:.2f}% Win Rate)")

    fireball_cumulative = evaluation_df.groupby('Fireball_Rating').agg(
        Total=('Prediction_Result', 'count'),
        Wins=('Prediction_Result', lambda x: (x == '✅ HIT').sum())
    )
    fireball_cumulative['Win %'] = (fireball_cumulative['Wins'] / fireball_cumulative['Total'] * 100).round(2)

    fireball_cumulative = fireball_cumulative.reset_index()
    fireball_cumulative['Fireball_Rating'] = fireball_cumulative['Fireball_Rating'].astype(str)
    fireball_cumulative['SortOrder'] = fireball_cumulative['Fireball_Rating'].apply(lambda x: fireball_order.index(x) if x in fireball_order else len(fireball_order))
    fireball_cumulative = fireball_cumulative.sort_values(by='SortOrder').drop(columns=['SortOrder']).set_index('Fireball_Rating')

    st.dataframe(fireball_cumulative, use_container_width=True)

# Footer
st.markdown("---")
st.caption("Created by ⚾ NRFI Analyzer 2025 (Fireball Model Only)")
