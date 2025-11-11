import requests
import pandas as pd
from datetime import datetime
import json
import time

# ==========================
# CONFIG
# ==========================
API_KEY = "d1dbe0ad91msh9508fe05e250a69p140593jsn26c5239accb7"
HOST = "live-golf-data.p.rapidapi.com"
ORG_ID = "1"     # PGA Tour
YEAR = "2024"

headers = {
    "x-rapidapi-key": API_KEY,
    "x-rapidapi-host": HOST
}

# ==========================
# HELPER FUNCTION
# ==========================
def get_json(endpoint, params=None):
    """Fetch data and handle errors gracefully"""
    url = f"https://{HOST}/{endpoint}"
    res = requests.get(url, headers=headers, params=params)
    if res.status_code != 200:
        print(f"❌ Error {res.status_code}: {endpoint} → {res.text}")
        return None
    try:
        return res.json()
    except Exception:
        return None


# ==========================
# STEP 1: WORLD RANKINGS
# ==========================
print("🌍 Fetching World Rankings (statId=186)...")
world_data = get_json("stats", {"year": YEAR, "statId": "186"})
world_df = pd.DataFrame(world_data.get("rows", [])) if world_data else pd.DataFrame()
world_df.rename(columns={"playerName": "name", "rank": "world_rank"}, inplace=True)
print(f"✅ World rankings loaded: {len(world_df)} players")


# ==========================
# STEP 2: TOURNAMENT DETAILS
# ==========================
print("📅 Fetching tournament info (Valspar Championship example)...")
tournament_data = get_json("tournament", {"orgId": ORG_ID, "tournId": "475", "year": YEAR})
tourn_df = pd.DataFrame([tournament_data]) if tournament_data else pd.DataFrame()
tourn_df["tournId"] = tournament_data.get("tournId") if tournament_data else None
print("✅ Tournament info fetched.")


# ==========================
# STEP 3: LEADERBOARD
# ==========================
print("🏆 Fetching leaderboard...")
leaderboard_data = get_json("leaderboard", {"orgId": ORG_ID, "tournId": "475", "year": YEAR})
leaderboard_df = pd.DataFrame(leaderboard_data.get("leaderboard", [])) if leaderboard_data else pd.DataFrame()
print(f"✅ Leaderboard fetched: {len(leaderboard_df)} rows")


# ==========================
# STEP 4: POINTS (FedEx)
# ==========================
print("💯 Fetching points (FedEx)...")
points_data = get_json("points", {"tournId": "475", "year": YEAR})
points_df = pd.DataFrame(points_data.get("points", [])) if points_data else pd.DataFrame()
print(f"✅ Points fetched: {len(points_df)} rows")


# ==========================
# STEP 5: PLAYER DETAILS (optional for each player)
# ==========================
def get_player_info(first_name, last_name, player_id):
    data = get_json("players", {"firstName": first_name, "lastName": last_name, "playerId": player_id})
    if not data:
        return None
    return pd.DataFrame([data])

player_profiles = []
if not leaderboard_df.empty:
    print("🧍 Fetching player profiles...")
    for _, row in leaderboard_df.head(10).iterrows():  # limit to top 10 to avoid API rate limits
        pid = row.get("playerId")
        first = row.get("firstName", "")
        last = row.get("lastName", "")
        p = get_player_info(first, last, pid)
        if p is not None:
            player_profiles.append(p)
        time.sleep(1)
    player_df = pd.concat(player_profiles, ignore_index=True) if player_profiles else pd.DataFrame()
else:
    player_df = pd.DataFrame()

print(f"✅ Player profiles fetched: {len(player_df)}")


# ==========================
# STEP 6: MERGE ALL
# ==========================
print("🔗 Merging all data sources...")

# Merge leaderboard + world ranking
merged = leaderboard_df.merge(world_df, how="left", left_on="playerName", right_on="name")

# Merge points
if not points_df.empty:
    merged = merged.merge(points_df, how="left", on="playerId")

# Merge tournament info
if not tourn_df.empty:
    for col in tourn_df.columns:
        merged[col] = tourn_df[col].iloc[0] if len(tourn_df) > 0 else None

# Merge player profiles (if available)
if not player_df.empty and "playerId" in player_df.columns:
    merged = merged.merge(player_df, on="playerId", how="left")

# ==========================
# STEP 7: SAVE
# ==========================
merged.to_csv("golf_training_dataset.csv", index=False)
print(f"\n✅ Combined dataset saved → golf_training_dataset.csv ({len(merged)} rows)")
print("Columns:", merged.columns.tolist())
