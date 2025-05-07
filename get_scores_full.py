﻿import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import pandas as pd
import re
import time
import os

def get_game_ids(date_obj):
    date_str = date_obj.strftime("%Y%m%d")
    url = f"https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard?dates={date_str}"
    r = requests.get(url)
    events = r.json().get("events", [])
    return [{"gameId": e["id"], "date": date_obj.strftime("%Y-%m-%d")} for e in events]

def extract_boxscore(game_id, game_date):
    url = f"https://www.espn.com/mlb/boxscore/_/gameId/{game_id}"
    print(f"🌐 Scraping HTML: {url}")
    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    soup = BeautifulSoup(r.content, "html.parser")

    team_names = soup.select("h2.ScoreCell__TeamName")
    if len(team_names) < 2:
        print("⚠️ Team names not found.")
        return None

    away_team = team_names[0].text.strip()
    home_team = team_names[1].text.strip()

    records = soup.select("div.Gamestrip__Record")
    away_record = records[0].text.strip().split(',')[0] if len(records) > 0 else ""
    home_record = records[1].text.strip().split(',')[0] if len(records) > 1 else ""

    scores = soup.select("div.Gamestrip__Score")
    away_runs = scores[0].get_text(strip=True) if len(scores) > 0 else ""
    home_runs = scores[1].get_text(strip=True) if len(scores) > 0 else ""

    inning_data = {}

    try:
        linescore_table = soup.find("table", class_="Table Table--align-center")
        if not linescore_table:
            print("⚠️ Linescore table not found. Proceeding with 'Pending' innings.")
            inning_data = {f"Away {i}th": "Pending" for i in range(1, 10)}
            inning_data.update({f"Home {i}th": "Pending" for i in range(1, 10)})
        else:
            header_row = linescore_table.find("thead").find("tr")
            header_cells = header_row.find_all("th")
            headers = [cell.text.strip() for cell in header_cells]

            rows = linescore_table.find("tbody").find_all("tr")
            if len(rows) < 2:
                print("⚠️ Not enough team rows in linescore. Marking innings 'Pending'.")
                inning_data = {f"Away {i}th": "Pending" for i in range(1, 10)}
                inning_data.update({f"Home {i}th": "Pending" for i in range(1, 10)})
            else:
                away_cells = rows[0].find_all("td")
                home_cells = rows[1].find_all("td")
                for inning in range(1, 10):
                    try:
                        inning_index = headers.index(str(inning))
                        away_inning_score = away_cells[inning_index].text.strip()
                        home_inning_score = home_cells[inning_index].text.strip()

                        inning_data[f"Away {inning}th"] = int(away_inning_score) if away_inning_score.isdigit() else 0
                        inning_data[f"Home {inning}th"] = int(home_inning_score) if home_inning_score.isdigit() else 0
                    except ValueError:
                        inning_data[f"Away {inning}th"] = "Pending"
                        inning_data[f"Home {inning}th"] = "Pending"
                    except Exception as e:
                        print(f"⚠️ Error parsing inning {inning}: {e}")
                        inning_data[f"Away {inning}th"] = "Pending"
                        inning_data[f"Home {inning}th"] = "Pending"
    except Exception as e:
        print(f"⚠️ Error parsing inning data: {e}")

    print(f"✅ Parsed: {away_team} vs {home_team}")

    game_row = {
        "Game Date": game_date,
        "Away Team": away_team,
        "Away Record": away_record,
        "Away Score": re.sub(r"\D", "", away_runs),
        "Home Team": home_team,
        "Home Record": home_record,
        "Home Score": re.sub(r"\D", "", home_runs),
    }
    game_row.update(inning_data)

    return game_row

def scrape_range(start_date, end_date, output_file="data/mlb_boxscores_full.csv"):
    # Force innings to be read as strings so "Pending" and 0.0 behave
    dtype_spec = {f"{side} {i}th": str for i in range(1, 10) for side in ["Away", "Home"]}

    if os.path.exists(output_file):
        existing_df = pd.read_csv(output_file, dtype=dtype_spec)
        print(f"📄 Found existing file with {len(existing_df)} rows.")
    else:
        existing_df = pd.DataFrame()
        print("🆕 No previous file found. Starting fresh.")

    current = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    new_rows = []

    while current <= end:
        games = get_game_ids(current)
        for game in games:
            try:
                row = extract_boxscore(game["gameId"], game["date"])
                if row:
                    new_rows.append(row)
            except Exception as e:
                print(f"❌ Error parsing {game['gameId']}: {e}")
            time.sleep(0.75)
        current += timedelta(days=1)

    if new_rows:
        new_df = pd.DataFrame(new_rows)

        if not existing_df.empty:
            merge_keys = ["Game Date", "Away Team", "Home Team"]
            existing_df.set_index(merge_keys, inplace=True)
            new_df.set_index(merge_keys, inplace=True)

            combined = pd.concat([existing_df[~existing_df.index.isin(new_df.index)], new_df]).reset_index()
        else:
            combined = new_df.reset_index()

        combined.sort_values(by=["Game Date", "Home Team"], inplace=True)

        # Safely calculate YRFI
        if 'Away 1th' in combined.columns and 'Home 1th' in combined.columns:
            mask = (
                combined['Away 1th'].notna() & combined['Home 1th'].notna() &
                (combined['Away 1th'] != "Pending") & (combined['Home 1th'] != "Pending")
            )

            combined.loc[mask, 'Away 1th'] = combined.loc[mask, 'Away 1th'].apply(lambda x: int(float(x)))
            combined.loc[mask, 'Home 1th'] = combined.loc[mask, 'Home 1th'].apply(lambda x: int(float(x)))

            combined.loc[mask, 'YRFI'] = (
                (combined.loc[mask, 'Away 1th'] + combined.loc[mask, 'Home 1th']) > 0
            ).astype(int)

            print("✅ YRFI column created based on 1st inning runs (after cleaning types).")
        else:
            print("⚠️ Could not create YRFI column — missing 1st inning data.")

        combined.to_csv(output_file, index=False)
        print(f"\n✅ Updated and saved to {output_file} ({len(combined)} total rows)")
    else:
        print("ℹ️ No new games found to append.")

if __name__ == "__main__":
    today = datetime.today()
    #start_date = (today - timedelta(days=1)).strftime("%Y-%m-%d")
    start_date = "2025-04-30"
    end_date = (today + timedelta(days=1)).strftime("%Y-%m-%d")

    print(f"🚀 Scraping boxscores for: {start_date} to {end_date}")
    scrape_range(start_date, end_date)
