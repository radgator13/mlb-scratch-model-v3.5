import pandas as pd
from datetime import datetime

# Load needed data
games_df = pd.read_csv("data/mlb_boxscores_full.csv")
nrfi_stats = pd.read_csv("data/team_nrfi_stats_pretty.csv")

# Filter games today and forward
today = datetime.today().strftime('%Y-%m-%d')
games_df = games_df[games_df['Game Date'] >= today]

# Get matchups
games = games_df[['Game Date', 'Away Team', 'Home Team']].drop_duplicates()

# Merge Away Team NRFI Batting
games = games.merge(
    nrfi_stats[['Team', 'NRFI_Batting_Rate']],
    left_on='Away Team',
    right_on='Team',
    how='left'
).rename(columns={"NRFI_Batting_Rate": "Away_NRFI_Batting_Rate"}).drop(columns=["Team"])

# Merge Home Team NRFI Pitching
games = games.merge(
    nrfi_stats[['Team', 'NRFI_Pitching_Rate']],
    left_on='Home Team',
    right_on='Team',
    how='left'
).rename(columns={"NRFI_Pitching_Rate": "Home_NRFI_Pitching_Rate"}).drop(columns=["Team"])

# Calculate Predicted NRFI %
games['Predicted_NRFI_Probability'] = (games['Away_NRFI_Batting_Rate'] * games['Home_NRFI_Pitching_Rate']) / 100
games['Predicted_NRFI_Probability'] = games['Predicted_NRFI_Probability'].round(2)

# Save to CSV
games.to_csv("data/mlb_nrfi_predictions.csv", index=False)
print(f"✅ Saved NRFI-only predictions to: data/mlb_nrfi_predictions.csv")

# Preview
print("\n📊 NRFI Prediction Example:")
print(games[['Game Date', 'Away Team', 'Home Team', 'Predicted_NRFI_Probability']].head(10))
