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
away_batting['Scored'] = (away_batting['Away 1th'] > 0).astype(int)

home_batting = df[['Home Team', 'Home 1th']].copy()
home_batting['Team'] = home_batting['Home Team']
home_batting['Scored'] = (home_batting['Home 1th'] > 0).astype(int)

batting = pd.concat([away_batting[['Team', 'Scored']], home_batting[['Team', 'Scored']]])

batting_summary = batting.groupby('Team').agg(
    Games_Batted=('Scored', 'count'),
    NRFI_Batting_Hits=('Scored', lambda x: (x == 0).sum())
).reset_index()

batting_summary['NRFI_Batting_Rate'] = (batting_summary['NRFI_Batting_Hits'] / batting_summary['Games_Batted']) * 100
batting_summary['NRFI_Batting_Rate'] = batting_summary['NRFI_Batting_Rate'].round(2)

# -------------------------------
# 🧠 3. Team NRFI Pitching Stats
# -------------------------------

away_pitching = df[['Away Team', 'Home 1th']].copy()
away_pitching['Team'] = away_pitching['Away Team']
away_pitching['Allowed'] = (away_pitching['Home 1th'] > 0).astype(int)

home_pitching = df[['Home Team', 'Away 1th']].copy()
home_pitching['Team'] = home_pitching['Home Team']
home_pitching['Allowed'] = (home_pitching['Away 1th'] > 0).astype(int)

pitching = pd.concat([away_pitching[['Team', 'Allowed']], home_pitching[['Team', 'Allowed']]])

pitching_summary = pitching.groupby('Team').agg(
    Games_Pitched=('Allowed', 'count'),
    NRFI_Pitching_Hits=('Allowed', lambda x: (x == 0).sum())
).reset_index()

pitching_summary['NRFI_Pitching_Rate'] = (pitching_summary['NRFI_Pitching_Hits'] / pitching_summary['Games_Pitched']) * 100
pitching_summary['NRFI_Pitching_Rate'] = pitching_summary['NRFI_Pitching_Rate'].round(2)

# Merge batting + pitching summaries
team_nrfi = pd.merge(batting_summary, pitching_summary, on='Team')

# Keep only useful columns
team_nrfi = team_nrfi[['Team', 'NRFI_Batting_Rate', 'NRFI_Pitching_Rate']]

# Calculate Overall NRFI Probability
team_nrfi['NRFI_Overall_Probability'] = (team_nrfi['NRFI_Batting_Rate'] * team_nrfi['NRFI_Pitching_Rate']) / 100
team_nrfi['NRFI_Overall_Probability'] = team_nrfi['NRFI_Overall_Probability'].round(2)

# Sort by best NRFI teams
team_nrfi = team_nrfi.sort_values(by='NRFI_Overall_Probability', ascending=False)

# Save to CSV
team_nrfi.to_csv("data/team_nrfi_stats_pretty.csv", index=False)
print(f"✅ Saved Sparkling NRFI Stats to: data/team_nrfi_stats_pretty.csv")

# Preview
print("\n📊 Top Teams by NRFI Overall Probability:")
print(team_nrfi.head(10))

