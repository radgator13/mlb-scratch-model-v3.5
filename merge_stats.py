import pandas as pd

# Load needed data
games_df = pd.read_csv("data/mlb_boxscores_full.csv")
nrfi_stats = pd.read_csv("data/team_nrfi_stats_pretty.csv")

# 🔥 Keep ALL games (today, tomorrow, day after — NO filter)

# Get matchups
games = games_df[['Game Date', 'Away Team', 'Home Team']].drop_duplicates()

# Merge Away Team NRFI Batting
games = games.merge(
    nrfi_stats[['Team', 'NRFI_Batting_Rate']],
    left_on='Away Team',
    right_on='Team',
    how='left'   # Force left join to keep ALL games
).rename(columns={"NRFI_Batting_Rate": "Away_NRFI_Batting_Rate"}).drop(columns=["Team"])

# Merge Home Team NRFI Pitching
games = games.merge(
    nrfi_stats[['Team', 'NRFI_Pitching_Rate']],
    left_on='Home Team',
    right_on='Team',
    how='left'   # Force left join to keep ALL games
).rename(columns={"NRFI_Pitching_Rate": "Home_NRFI_Pitching_Rate"}).drop(columns=["Team"])

# 🔥 Fill missing stats
games['Away_NRFI_Batting_Rate'] = games['Away_NRFI_Batting_Rate'].fillna(50.0)
games['Home_NRFI_Pitching_Rate'] = games['Home_NRFI_Pitching_Rate'].fillna(50.0)

# 🔥 Predict NRFI %
games['Predicted_NRFI_Probability'] = (games['Away_NRFI_Batting_Rate'] * games['Home_NRFI_Pitching_Rate']) / 100
games['Predicted_NRFI_Probability'] = games['Predicted_NRFI_Probability'].round(2)

# 🔥 Save ALL games
games.to_csv("data/mlb_nrfi_predictions.csv", index=False)
print(f"✅ Saved NRFI predictions for ALL games — today, tomorrow, and beyond.")

# Preview
print("\n📊 NRFI Prediction Example:")
print(games[['Game Date', 'Away Team', 'Home Team', 'Predicted_NRFI_Probability']].tail(425))
