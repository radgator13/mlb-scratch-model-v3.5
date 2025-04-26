import pandas as pd
from datetime import datetime

# Read original CSV
input_file = "data/rotowire_projstarters.csv"
df = pd.read_csv(input_file)

# Prepare output
output_rows = []

# Parse dates from the header
date_columns = df.columns[1:]  # Skip the first column ("Team")
parsed_dates = []
today_year = datetime.today().year

for col in date_columns:
    try:
        parsed_date = datetime.strptime(col.strip(), "%a %m/%d").replace(year=today_year)
        parsed_dates.append(parsed_date.date())
    except Exception as e:
        parsed_dates.append(None)

# Walk through each team block (5 rows per team now!)
for idx in range(0, len(df), 5):
    try:
        team = df.iloc[idx, 0]

        for col_idx, date in enumerate(parsed_dates):
            if date is None:
                continue

            # Extract raw text entries
            pitcher_info = str(df.iloc[idx, col_idx + 1])  # pitcher (name and throws)
            game_info = str(df.iloc[idx + 1, col_idx + 1])  # game time
            stats_info = str(df.iloc[idx + 2, col_idx + 1])  # record and ERA
            oppbat_info = str(df.iloc[idx + 3, col_idx + 1])  # opponent batting
            opppitch_info = str(df.iloc[idx + 4, col_idx + 1])  # opponent pitcher

            # Parse pitcher name and throws
            name = ""
            throws = ""
            if "(" in pitcher_info:
                try:
                    name_part, throws_part = pitcher_info.split("(")
                    name = name_part.strip()
                    throws = throws_part.replace(")", "").strip()
                except:
                    name = pitcher_info.strip()
                    throws = ""

            # Parse record and ERA
            record = ""
            era = ""
            if stats_info and "ERA" in stats_info:
                try:
                    parts = stats_info.replace("ERA", "").strip().split()
                    if len(parts) >= 2:
                        record = parts[0]
                        era = parts[1]
                except:
                    pass

            # Opponent batting and pitcher
            opp_batt = oppbat_info.strip()
            opp_pitcher = opppitch_info.strip()

            if name:
                output_rows.append({
                    "Game Date": date,
                    "Team": team,
                    "Pitcher Name": name,
                    "Throws": throws,
                    "Record": record,
                    "ERA": era,
                    "Opp Batt Avg": opp_batt,
                    "Opp Pitcher": opp_pitcher
                })

    except Exception as e:
        print(f"❌ Error processing team block starting at row {idx}: {e}")
        continue

# Save the clean normalized output
output_df = pd.DataFrame(output_rows)
output_df.to_csv("data/rotowire_projstarters_cleaned.csv", index=False)
print(f"✅ Saved cleaned file: data/rotowire_projstarters_cleaned.csv")
