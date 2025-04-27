import pandas as pd

# Load full boxscores
df = pd.read_csv("data/mlb_boxscores_full.csv")

# -------------------------------
# 🧠 1. Create Basic NRFI Flag
# -------------------------------

if 'NRFI' not in df.columns:
    df['NRFI'] = ((df['Away 1th'] + df['Home 1th']) == 0).astype(int)

# -------------------------------
# 🧠 2. Team NRFI Batting Stats
# -------------------------------

away_batting = df[['Away Team', 'Away 1th']].copy()
away_batting['Team'] = away_batting['Away Team']
away_batting['Scored'] = (
    away_batting['Away 1th']
    .apply(lambda x: int(float(x)) if pd.notna(x) and x != "Pending" else 0)
    > 0
).astype(int)

home_batting = df[['Home Team', 'Home 1th']].copy()
home_batting['Team'] = home_batting['Home Team']
home_batting['Scored'] = (
    home_batting['Home 1th']
    .apply(lambda x: int(float(x)) if pd.notna(x) and x != "Pending" else 0)
    > 0
).astype(int)

batting = pd.concat([away_batting[['Team', 'Scored']], home_batting[['Team', 'Scored']]])

batting_summary = batting.groupby('Team').agg(
    Games_Batted=('Scored', 'count'),
    NRFI_Batting_Hits=('Scored', lambda x: (x == 0).sum())
).reset_index()

batting_summary['Away_NRFI_Batting_Rate'] = (batting_summary['NRFI_Batting_Hits'] / batting_summary['Games_Batted']) * 100
batting_summary['Away_NRFI_Batting_Rate'] = batting_summary['Away_NRFI_Batting_Rate'].round(2)

# -------------------------------
# 🧠 3. Team NRFI Pitching Stats
# -------------------------------

away_pitching = df[['Away Team', 'Home 1th']].copy()
away_pitching['Team'] = away_pitching['Away Team']
away_pitching['Allowed'] = (
    away_pitching['Home 1th']
    .apply(lambda x: int(float(x)) if pd.notna(x) and x != "Pending" else 0)
    > 0
).astype(int)

home_pitching = df[['Home Team', 'Away 1th']].copy()
home_pitching['Team'] = home_pitching['Home Team']
home_pitching['Allowed'] = (
    home_pitching['Away 1th']
    .apply(lambda x: int(float(x)) if pd.notna(x) and x != "Pending" else 0)
    > 0
).astype(int)

pitching = pd.concat([away_pitching[['Team', 'Allowed']], home_pitching[['Team', 'Allowed']]])

pitching_summary = pitching.groupby('Team').agg(
    Games_Pitched=('Allowed', 'count'),
    NRFI_Pitching_Hits=('Allowed', lambda x: (x == 0).sum())
).reset_index()

pitching_summary['Home_NRFI_Pitching_Rate'] = (pitching_summary['NRFI_Pitching_Hits'] / pitching_summary['Games_Pitched']) * 100
pitching_summary['Home_NRFI_Pitching_Rate'] = pitching_summary['Home_NRFI_Pitching_Rate'].round(2)

# -------------------------------
# 🧠 4. Merge Batting + Pitching into Game Level
# -------------------------------

# Create away team batting stats
away_stats = batting_summary[['Team', 'Away_NRFI_Batting_Rate']].rename(columns={"Team": "Away Team"})

# Create home team pitching stats
home_stats = pitching_summary[['Team', 'Home_NRFI_Pitching_Rate']].rename(columns={"Team": "Home Team"})

# Merge away batting rates
df = df.merge(away_stats, on='Away Team', how='left')

# Merge home pitching rates
df = df.merge(home_stats, on='Home Team', how='left')

# -------------------------------
# 🧠 5. Calculate Predicted NRFI Probability per game
# -------------------------------

df['Predicted_NRFI_Probability'] = (
    df['Away_NRFI_Batting_Rate'] * df['Home_NRFI_Pitching_Rate'] / 100
).round(2)

# -------------------------------
# ✅ 6. Save final output
# -------------------------------

df.to_csv("data/mlb_nrfi_predictions.csv", index=False)
print("\n✅ Saved mlb_nrfi_predictions.csv with all features!")

# Preview
print("\n📊 Sample rows with predictions:")
print(df[['Game Date', 'Away Team', 'Home Team', 'Predicted_NRFI_Probability']].head(10))
