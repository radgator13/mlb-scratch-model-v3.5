import pandas as pd

# Load data
pred_df = pd.read_csv("data/mlb_nrfi_predictions.csv")
actual_df = pd.read_csv("data/mlb_boxscores_full.csv")

# Merge based on full matchup keys
merged = pred_df.merge(
    actual_df[['Game Date', 'Away Team', 'Home Team', 'Away 1th', 'Home 1th']],
    on=['Game Date', 'Away Team', 'Home Team'],
    how='inner'
)

print(f"🔵 Total Games Merged: {len(merged)}")

# ✅ Filter out games with pending innings
merged = merged[
    (merged['Away 1th'] != "Pending") &
    (merged['Home 1th'] != "Pending")
]

# ✅ Correct: Calculate actual 1st inning runs (force numeric)
merged['Actual_1st_Inning_Runs'] = (
    merged['Away 1th'].fillna(0).apply(lambda x: int(float(x)))
    +
    merged['Home 1th'].fillna(0).apply(lambda x: int(float(x)))
)

# Evaluate prediction result
merged['Prediction_Result'] = merged['Actual_1st_Inning_Runs'].apply(lambda x: '✅ HIT' if x == 0 else '❌ MISS')

# Save FULL file (no filter)
merged.to_csv("data/mlb_nrfi_results_full.csv", index=False)
print(f"✅ Saved full NRFI results to: data/mlb_nrfi_results_full.csv")

# Preview
print("\n📊 Preview of full results:")
print(merged[['Game Date', 'Away Team', 'Home Team', 'Predicted_NRFI_Probability', 'Actual_1st_Inning_Runs', 'Prediction_Result']].head(10))
