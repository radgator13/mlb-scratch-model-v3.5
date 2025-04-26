import streamlit as st
import pandas as pd
import joblib
from datetime import datetime
import os

# Set Streamlit page config to wide
st.set_page_config(page_title="MLB NRFI Machine Learning Predictions", layout="wide")

# Load data safely
@st.cache_data
def load_predictions():
    return pd.read_csv("data/mlb_nrfi_predictions.csv")

@st.cache_data
def load_actuals():
    return pd.read_csv("data/mlb_nrfi_results_full.csv")

@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

# Load all
pred_df = load_predictions()
results_df = load_actuals()
model = load_model()

# Title
st.title("⚾ MLB NRFI Machine Learning Predictions")

# View Toggle
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

# Merge predictions + results
df = pred_df.merge(
    results_df[['Game Date', 'Away Team', 'Home Team', 'Actual_1st_Inning_Runs', 'Prediction_Result']],
    on=['Game Date', 'Away Team', 'Home Team'],
    how='inner'
)

# Features used for model
features = ['Predicted_NRFI_Probability', 'Away_NRFI_Batting_Rate', 'Home_NRFI_Pitching_Rate']

# Predict using ML model
df['Model_Prediction'] = model.predict(df[features])

# Human readable prediction
df['Model_Prediction_Result'] = df['Model_Prediction'].apply(lambda x: "✅ HIT" if x == 1 else "❌ MISS")

# Filter by selected date
filtered_df = df[df['Game Date'] == selected_date_str]

# ------------------------------
# 📋 Display Table
# ------------------------------

if view_option == "Predictions Only":
    st.subheader(f"ML Model NRFI Predictions for {selected_date_str}")
    st.dataframe(
        filtered_df[['Away Team', 'Home Team', 'Predicted_NRFI_Probability', 'Model_Prediction_Result']],
        use_container_width=True
    )

elif view_option == "Predictions vs Actual Results":
    st.subheader(f"ML Model NRFI Predictions vs Actual Results for {selected_date_str}")
    st.dataframe(
        filtered_df[['Away Team', 'Home Team', 'Predicted_NRFI_Probability', 'Model_Prediction_Result', 'Actual_1st_Inning_Runs', 'Prediction_Result']],
        use_container_width=True
    )

# ------------------------------
# 📊 DAILY SUMMARY
# ------------------------------

st.markdown("---")
st.subheader("📈 Daily Summary for Selected Date")

if not filtered_df.empty:

    # Daily data
    daily_total = len(filtered_df)
    daily_hits = (filtered_df['Model_Prediction_Result'] == '✅ HIT').sum()
    daily_win_rate = (daily_hits / daily_total) * 100 if daily_total > 0 else 0

    st.write(f"**Daily Model Record:** {daily_hits} Wins / {daily_total - daily_hits} Losses ({daily_win_rate:.2f}% Win Rate)")

# ------------------------------
# 📊 CUMULATIVE SUMMARY
# ------------------------------

st.markdown("---")
st.subheader("📈 Cumulative Summary (All Days)")

# Cumulative up to and including selected date
cumulative_df = df[df['Game Date'] <= selected_date_str]

if not cumulative_df.empty:
    cumulative_total = len(cumulative_df)
    cumulative_hits = (cumulative_df['Model_Prediction_Result'] == '✅ HIT').sum()
    cumulative_win_rate = (cumulative_hits / cumulative_total) * 100 if cumulative_total > 0 else 0

    st.write(f"**Cumulative Model Record:** {cumulative_hits} Wins / {cumulative_total - cumulative_hits} Losses ({cumulative_win_rate:.2f}% Win Rate)")

# Footer
st.markdown("---")
st.caption("Created by ⚾ NRFI ML Model Tester 2025")
