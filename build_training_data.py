import pandas as pd

# Load prediction and results data
pred_df = pd.read_csv('data/mlb_nrfi_predictions.csv')
results_df = pd.read_csv('data/mlb_nrfi_results_full.csv')

# Merge on full game key
merged = pred_df.merge(
    results_df[['Game Date', 'Away Team', 'Home Team', 'Actual_1st_Inning_Runs']],
    on=['Game Date', 'Away Team', 'Home Team'],
    how='inner'
)

print(f"🔵 Merged {len(merged)} games with real outcomes.")

# Remove pending games
merged = merged[merged['Actual_1st_Inning_Runs'] != "Pending"]

# Convert Actual Runs to int
merged['Actual_1st_Inning_Runs'] = merged['Actual_1st_Inning_Runs'].astype(int)

# ✅ Create real NRFI_Hit
merged['NRFI_Hit'] = (merged['Actual_1st_Inning_Runs'] == 0).astype(int)

# Select final columns
final_cols = [
    'Game Date', 'Away Team', 'Home Team',
    'Predicted_NRFI_Probability', 'Away_NRFI_Batting_Rate', 'Home_NRFI_Pitching_Rate', 'NRFI_Hit'
]
training_df = merged[final_cols]

# Save real training dataset
training_df.to_csv('data/mlb_training_dataset.csv', index=False)

print(f"✅ Saved new file: data/mlb_training_dataset.csv")
print("\n📊 Preview:")
print(training_df.head(10))
