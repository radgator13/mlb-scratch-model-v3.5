import streamlit as st
import pandas as pd
from datetime import datetime
import joblib
import matplotlib.pyplot as plt

# Set Streamlit page config
st.set_page_config(page_title="MLB NRFI Dashboard", layout="wide")

# ------------------------------
# 🎨 Global Page Center Styling (FINAL FIX)
# ------------------------------

st.markdown(
    """
    <style>
    /* Center everything inside the page */
    [data-testid="stAppViewContainer"] {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
    }

    /* Center all titles, headers, markdowns */
    h1, h2, h3, h4, h5, h6, p {
        text-align: center;
    }

    /* Center all elements inside form blocks (inputs, pickers, etc.) */
    [data-testid="stForm"] {
        align-items: center;
    }

    /* Center table text */
    .dataframe th, .dataframe td {
        text-align: center !important;
        vertical-align: middle !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)








# Load Data
@st.cache_data
def load_predictions():
    return pd.read_csv("data/mlb_nrfi_predictions.csv")

@st.cache_data
def load_results():
    return pd.read_csv("data/mlb_nrfi_results_full.csv")

# ✅ Load Random Forest model
model = joblib.load('model_rf_real.pkl')

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

# ✅ Predict using Random Forest model
features = ['Predicted_NRFI_Probability', 'Away_NRFI_Batting_Rate', 'Home_NRFI_Pitching_Rate']
df['Model_Prediction'] = model.predict(df[features])
df['Model_Prediction_Result'] = df['Model_Prediction'].apply(lambda x: "✅ HIT" if x == 1 else "❌ MISS")

# Fill missing prediction results
df['Prediction_Result'] = df['Prediction_Result'].fillna("Pending")

# ✅ Assign Confidence Label
def assign_confidence(row):
    fireball = row['Fireball_Rating']
    model_pick = row['Model_Prediction_Result']
    
    if fireball in ["🔥🔥🔥", "🔥🔥🔥🔥", "🔥🔥🔥🔥🔥"]:
        if model_pick == "✅ HIT":
            return "Strong"
        else:
            return "Caution"
    elif fireball in ["🔥🔥"]:
        if model_pick == "✅ HIT":
            return "Medium"
        else:
            return "Caution"
    else:
        return "Weak"

df['Confidence_Label'] = df.apply(assign_confidence, axis=1)

# ✅ Determine if model prediction actually matched reality
df['Model_Correct'] = (
    (df['Model_Prediction_Result'] == "✅ HIT") & 
    (df['Prediction_Result'] == "✅ HIT")
).astype(int)

# ✅ Color function
def highlight_confidence(val):
    if val == 'Strong':
        return 'color: green; font-weight: bold;'
    elif val == 'Medium':
        return 'color: orange; font-weight: bold;'
    elif val == 'Caution':
        return 'color: red; font-weight: bold;'
    elif val == 'Weak':
        return 'color: gray;'
    else:
        return ''

# Use completed games only for evaluation
evaluation_df = df[df['Actual_1st_Inning_Runs'] != "Pending"]

# Title
st.title("⚾ MLB NRFI Predictions Dashboard (Fireball + Random Forest + Confidence + Cumulative Chart)")

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
selected_date_pd = pd.Timestamp(selected_date)  # ✅ Fix type mismatch

# ✅ Fireball Confidence Toggle
fireball_filter = st.checkbox("🔥 Show Only Fireballs 3, 4, and 5 (Higher-Confidence Picks)", value=False)

# Filter by selected date
filtered_df = df[df['Game Date'] == selected_date_str]
filtered_eval_df = evaluation_df[evaluation_df['Game Date'] == selected_date_str]

# ✅ Apply Fireball filter if checked
if fireball_filter:
    filtered_df = filtered_df[filtered_df['Fireball_Rating'].isin(["🔥🔥🔥", "🔥🔥🔥🔥", "🔥🔥🔥🔥🔥"])]
    filtered_eval_df = filtered_eval_df[filtered_eval_df['Fireball_Rating'].isin(["🔥🔥🔥", "🔥🔥🔥🔥", "🔥🔥🔥🔥🔥"])]
    evaluation_df = evaluation_df[evaluation_df['Fireball_Rating'].isin(["🔥🔥🔥", "🔥🔥🔥🔥", "🔥🔥🔥🔥🔥"])]

# ------------------------------
# 📋 Display Predictions
# ------------------------------

if view_option == "Predictions Only":
    st.subheader(f"NRFI Predictions for {selected_date_str}")

    # ✅ Fix Confidence_Label sorting here
    confidence_order = ["Strong", "Medium", "Caution", "Weak"]
    filtered_df['Confidence_Label'] = pd.Categorical(
        filtered_df['Confidence_Label'],
        categories=confidence_order,
        ordered=True
    )

    styled_df = filtered_df[['Away Team', 'Home Team', 'Fireball_Rating', 'Model_Prediction_Result', 'Confidence_Label']].sort_values(by="Confidence_Label", ascending=True)
    st.dataframe(
        styled_df.style.applymap(highlight_confidence, subset=["Confidence_Label"]),
        use_container_width=True
    )

    st.subheader("🔥 Best NRFI Bets (Fireballs 4 or 5)")
    best_bets = filtered_df[filtered_df['Predicted_NRFI_Probability'] >= 70]
    if not best_bets.empty:
        styled_best = best_bets[['Away Team', 'Home Team', 'Fireball_Rating', 'Model_Prediction_Result', 'Confidence_Label']].sort_values(by="Confidence_Label", ascending=True)
        st.dataframe(
            styled_best.style.applymap(highlight_confidence, subset=["Confidence_Label"]),
            use_container_width=True
        )
    else:
        st.write("No high-confidence NRFI games today. ❌")

elif view_option == "Predictions vs Actual Results":
    st.subheader(f"NRFI Predictions vs Actual for {selected_date_str}")

    # ✅ Fix Confidence_Label sorting first
    confidence_order = ["Strong", "Medium", "Caution", "Weak"]
    filtered_eval_df['Confidence_Label'] = pd.Categorical(
        filtered_eval_df['Confidence_Label'],
        categories=confidence_order,
        ordered=True
    )

    styled_eval = filtered_eval_df[['Away Team', 'Home Team', 'Fireball_Rating', 'Model_Prediction_Result', 'Confidence_Label', 'Actual_1st_Inning_Runs', 'Prediction_Result']].sort_values(by="Confidence_Label", ascending=True)

    st.dataframe(
        styled_eval.style.applymap(highlight_confidence, subset=["Confidence_Label"]),
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

    st.write(f"**Daily Model Record (Real Outcomes):** {daily_hits} Wins / {daily_total - daily_hits} Losses ({daily_win_rate:.2f}% Win Rate)")

    fireball_daily = filtered_eval_df.groupby('Confidence_Label').agg(
        Total=('Prediction_Result', 'count'),
        Wins=('Prediction_Result', lambda x: (x == '✅ HIT').sum()),
        Model_Hits=('Model_Correct', 'sum')
    )
    fireball_daily['Real Win %'] = (fireball_daily['Wins'] / fireball_daily['Total'] * 100).round(2)
    fireball_daily['Model Hit %'] = (fireball_daily['Model_Hits'] / fireball_daily['Total'] * 100).round(2)

    confidence_order = ["Strong", "Medium", "Caution", "Weak"]
    fireball_daily = fireball_daily.reset_index()
    fireball_daily['Confidence_Label'] = pd.Categorical(fireball_daily['Confidence_Label'], categories=confidence_order, ordered=True)
    fireball_daily = fireball_daily.sort_values(by='Confidence_Label')

    st.dataframe(fireball_daily.style.applymap(highlight_confidence, subset=["Confidence_Label"]), use_container_width=True)

# ------------------------------
# 📊 CUMULATIVE SUMMARY (Corrected + Smoothed Chart)
# ------------------------------

st.markdown("---")
st.subheader("📈 Cumulative Summary (All Days Up to Selected Date)")

if view_option == "Predictions vs Actual Results":

    selected_date_pd = pd.Timestamp(selected_date)

    # ✅ Only include games up to selected date
    cumulative_df = evaluation_df[pd.to_datetime(evaluation_df['Game Date']) <= selected_date_pd]

    if not cumulative_df.empty:

        # Cumulative totals
        cumulative_total = len(cumulative_df)
        cumulative_hits = (cumulative_df['Prediction_Result'] == '✅ HIT').sum()
        cumulative_win_rate = (cumulative_hits / cumulative_total) * 100 if cumulative_total > 0 else 0

        st.write(f"**Cumulative Model Record (Real Outcomes through {selected_date_str}):** {cumulative_hits} Wins / {cumulative_total - cumulative_hits} Losses ({cumulative_win_rate:.2f}% Win Rate)")

        # Group by Confidence Label
        fireball_cumulative = cumulative_df.groupby('Confidence_Label').agg(
            Total=('Prediction_Result', 'count'),
            Wins=('Prediction_Result', lambda x: (x == '✅ HIT').sum()),
            Model_Hits=('Model_Correct', 'sum')
        )
        fireball_cumulative['Real Win %'] = (fireball_cumulative['Wins'] / fireball_cumulative['Total'] * 100).round(2)
        fireball_cumulative['Model Hit %'] = (fireball_cumulative['Model_Hits'] / fireball_cumulative['Total'] * 100).round(2)

        fireball_cumulative = fireball_cumulative.reset_index()
        fireball_cumulative['Confidence_Label'] = pd.Categorical(fireball_cumulative['Confidence_Label'], categories=["Strong", "Medium", "Caution", "Weak"], ordered=True)
        fireball_cumulative = fireball_cumulative.sort_values(by='Confidence_Label')

        st.dataframe(fireball_cumulative.style.applymap(highlight_confidence, subset=["Confidence_Label"]), use_container_width=True)

        # ------------------------------
        # 📈 Cumulative Win % Chart
        # ------------------------------

        st.markdown("---")
        st.subheader("📈 Cumulative Win % Over Time (Target: 55%)")

        cumulative_df_sorted = cumulative_df.sort_values('Game Date')
        cumulative_df_sorted['Game Date'] = pd.to_datetime(cumulative_df_sorted['Game Date'])

        cumulative_df_sorted['Rolling Wins'] = (cumulative_df_sorted['Prediction_Result'] == '✅ HIT').cumsum()
        cumulative_df_sorted['Rolling Games'] = range(1, len(cumulative_df_sorted) + 1)
        cumulative_df_sorted['Rolling Win %'] = (cumulative_df_sorted['Rolling Wins'] / cumulative_df_sorted['Rolling Games']) * 100

        # ✅ 7-game moving average smoothing
        cumulative_df_sorted['Smoothed Win %'] = cumulative_df_sorted['Rolling Win %'].rolling(window=7, min_periods=1).mean()

        fig, ax = plt.subplots(figsize=(10, 5))

        final_cum_win = cumulative_df_sorted['Rolling Win %'].iloc[-1]
        line_color = 'green' if final_cum_win >= 60 else 'red'

        # Plot raw Win % (lighter)
        ax.plot(cumulative_df_sorted['Game Date'], cumulative_df_sorted['Rolling Win %'], marker='o', color=line_color, alpha=0.4, label='Raw Cumulative Win %')

        # Plot smoothed Win % (bold)
        ax.plot(cumulative_df_sorted['Game Date'], cumulative_df_sorted['Smoothed Win %'], color=line_color, linewidth=3, label='Smoothed Win % (7 Games)')

        # Add 60% Target Line
        ax.axhline(y=55, color='blue', linestyle='--', label='Target 55% Win Rate')

        ax.set_xlabel('Game Date')
        ax.set_ylabel('Cumulative Win %')
        ax.set_title('Cumulative NRFI Model Win Rate Over Time')
        ax.grid(True)
        ax.legend()

        st.pyplot(fig)






# Footer
st.markdown("---")
st.caption("Created by ⚾ NRFI Analyzer 2025 (Fireball + Random Forest + Real Model Hit % + Chart)")
