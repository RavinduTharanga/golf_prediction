# import requests
# import pandas as pd
# import time
# import logging
# from datetime import datetime
# from xgboost import XGBClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import brier_score_loss, roc_auc_score

# # ========== CONFIG ==========
# API_KEY = "e3cf5dd42289495280baabbdecf46355"
# BASE = "https://api.sportsdata.io/golf/v2/json"
# HEAD = {"Ocp-Apim-Subscription-Key": API_KEY}

# # Logging setup
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s [%(levelname)s] %(message)s",
#     datefmt="%H:%M:%S"
# )

# # ============================

# def get_json(url):
#     """Generic GET request with error handling"""
#     try:
#         r = requests.get(url, headers=HEAD)
#         r.raise_for_status()
#         return r.json()
#     except Exception as e:
#         logging.error(f"❌ API request failed: {url}\n{e}")
#         return {}

# # 1️⃣  Get tournaments in a season
# def get_tournaments(season=2025):
#     url = f"{BASE}/Tournaments/{season}?key={API_KEY}"
#     return pd.DataFrame(get_json(url))

# # 2️⃣  Get OWGR rankings
# def get_rankings(season=2025):
#     url = f"{BASE}/Rankings/{season}?key={API_KEY}"
#     return pd.DataFrame(get_json(url))

# # 3️⃣  Get leaderboard for a tournament
# def get_leaderboard(tournament_id):
#     url = f"{BASE}/Leaderboard/{tournament_id}?key={API_KEY}"
#     data = get_json(url)
#     return pd.json_normalize(data["Players"]) if "Players" in data else pd.DataFrame()

# # ================== BUILD TRAINING DATA ==================
# def build_training(season=2025):
#     logging.info(f"🔄 Starting data collection for season {season}...")
#     tournaments = get_tournaments(season)
#     rankings = get_rankings(season)

#     if tournaments.empty:
#         logging.error("❌ No tournaments found — check your API key or season value.")
#         return pd.DataFrame()

#     logging.info(f"📅 Found {len(tournaments)} tournaments for {season}.")

#     records = []
#     for i, t in tournaments.iterrows():
#         tid = t["TournamentID"]
#         name = t["Name"]
#         logging.info(f"⛳ Processing {i+1}/{len(tournaments)}: {name} (ID: {tid})")

#         try:
#             lb = get_leaderboard(tid)
#             if lb.empty:
#                 logging.warning(f"⚠️ No leaderboard data for {name}")
#                 continue

#             df = lb.merge(rankings, how="left", on="PlayerID")
#             df["TournamentID"] = tid
#             df["TournamentName"] = name
#             df["StartDate"] = t["StartDate"]
#             df["result_win"] = (df["Rank"] == 1).astype(int)
#             records.append(df)
#         except Exception as e:
#             logging.error(f"❌ Failed to process {name}: {e}")

#         time.sleep(0.3)  # Respect API limits

#     if not records:
#         logging.error("❌ No data collected for any tournament.")
#         return pd.DataFrame()

#     all_data = pd.concat(records, ignore_index=True)
#     all_data.to_csv("golf_training_2025_raw.csv", index=False)
#     logging.info(f"✅ Data collection complete. Saved {len(all_data)} rows.")
#     return all_data

# # ================== FEATURE ENGINEERING ==================
# def create_features(df):
#     logging.info("🧩 Creating training features...")

#     df["WorldGolfRank"] = pd.to_numeric(df["WorldGolfRank"], errors="coerce")
#     df["AveragePoints"] = pd.to_numeric(df["AveragePoints"], errors="coerce")

#     # Previous event’s finishing position
#     df["RecentFinish"] = df.groupby("PlayerID")["Rank"].shift(1)

#     # Count of Top-10 finishes in the past 5 events
#     df["RecentTop10Count"] = (
#         df.groupby("PlayerID")["Rank"]
#           .transform(lambda s: s.shift(1).rolling(5).apply(lambda y: (y <= 10).sum(), raw=False))
#     )

#     # Number of events played so far this season
#     df["EventsPlayed"] = df.groupby("PlayerID").cumcount()

#     df = df.fillna(0)
#     logging.info("✅ Feature engineering complete.")

#     features = [
#         "WorldGolfRank",
#         "AveragePoints",
#         "RecentFinish",
#         "RecentTop10Count",
#         "EventsPlayed",
#     ]
#     target = "result_win"
#     return df[features + [target]]

# # ================== TRAIN MODEL ==================
# def train_model(df):
#     logging.info("🏋️ Training model...")

#     X = df.drop(columns="result_win")
#     y = df["result_win"]

#     if y.sum() == 0:
#         logging.error("❌ No winners (result_win=1) found in dataset — cannot train.")
#         return None

#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, stratify=y
#     )

#     model = XGBClassifier(
#         n_estimators=500,
#         learning_rate=0.05,
#         max_depth=4,
#         subsample=0.9,
#         colsample_bytree=0.9,
#         eval_metric="logloss",
#     )
#     model.fit(X_train, y_train)
#     preds = model.predict_proba(X_test)[:, 1]

#     auc = roc_auc_score(y_test, preds)
#     brier = brier_score_loss(y_test, preds)
#     logging.info(f"✅ Model trained | AUC: {auc:.4f} | Brier: {brier:.4f}")

#     return model
import os
import requests, pandas as pd, time, logging
from datetime import datetime
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, brier_score_loss
from tqdm import tqdm


# ================== CONFIG ==================
API_KEY = "e3cf5dd42289495280baabbdecf46355"
BASE = "https://api.sportsdata.io/golf/v2/json"
HEAD = {"Ocp-Apim-Subscription-Key": API_KEY}
YEARS = [2024, 2025]       # train across 5 seasons

# Optional free weather API
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S")

# ================== HELPERS ==================
def get_json(url):
    try:
        r = requests.get(url, headers=HEAD)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logging.error(f"❌ API request failed: {url}\n{e}")
        return {}

def get_tournaments(season):
    return pd.DataFrame(get_json(f"{BASE}/Tournaments/{season}?key={API_KEY}"))

def get_rankings(season):
    return pd.DataFrame(get_json(f"{BASE}/Rankings/{season}?key={API_KEY}"))



#     return players_df
def get_leaderboard(tournament_id):
    """Fetch leaderboard including player + tournament info."""
    data = get_json(f"{BASE}/Leaderboard/{tournament_id}?key={API_KEY}")
    if not data or "Players" not in data:
        logging.warning(f"⚠️ No player data found for tournament {tournament_id}")
        return pd.DataFrame()

    # Flatten nested structure safely
    players_df = pd.json_normalize(
        data["Players"],
        sep="_"
    )

    # ✅ Handle possible nested name fields
    name_cols = [c for c in players_df.columns if "Name" in c]
    if "Name" in players_df.columns:
        players_df["Name"] = players_df["Name"].astype(str)
    elif "Player_Name" in players_df.columns:
        players_df["Name"] = players_df["Player_Name"].astype(str)
    elif "Player_Name_x" in players_df.columns:
        players_df["Name"] = players_df["Player_Name_x"].astype(str)
    elif "Player_Name_y" in players_df.columns:
        players_df["Name"] = players_df["Player_Name_y"].astype(str)
    elif "Player_PlayerName" in players_df.columns:
        players_df["Name"] = players_df["Player_PlayerName"].astype(str)
    elif len(name_cols) > 0:
        players_df["Name"] = players_df[name_cols[0]].astype(str)
    else:
        players_df["Name"] = "Unknown Player"

    # ✅ Tournament info
    if "Tournament" in data and isinstance(data["Tournament"], dict):
        tinfo = data["Tournament"]
        for key in ["TournamentID", "Name", "StartDate", "City", "State", "Par", "Yards"]:
            if key in tinfo:
                players_df[f"Tournament_{key}"] = tinfo[key]
    else:
        players_df["Tournament_TournamentID"] = tournament_id

    # 🔍 Debug: print a sample of name columns
    sample_names = players_df["Name"].head(5).tolist()
    logging.info(f"🎯 Sample player names from leaderboard {tournament_id}: {sample_names}")

    return players_df




def get_weather(city, date):
    """Returns average temp and wind for the tournament start day."""
    try:
        params = {
            "latitude": 0, "longitude": 0, "hourly": "temperature_2m,windspeed_10m"
        }
        # Quick lookup for city → coordinates (Open-Meteo geocoding)
        geo = requests.get(
            f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1"
        ).json()
        if "results" not in geo: return (None, None)
        lat, lon = geo["results"][0]["latitude"], geo["results"][0]["longitude"]
        params.update({"latitude": lat, "longitude": lon, "start_date": date, "end_date": date})
        weather = requests.get(WEATHER_URL, params=params).json()
        temps = weather.get("hourly", {}).get("temperature_2m", [])
        winds = weather.get("hourly", {}).get("windspeed_10m", [])
        if not temps: return (None, None)
        return (sum(temps)/len(temps), sum(winds)/len(winds))
    except Exception:
        return (None, None)

# ================== BUILD TRAINING ==================
def build_training_multi(years=YEARS):
    records = []
    for season in years:
        tournaments = get_tournaments(season)
        rankings = get_rankings(season)
        if tournaments.empty:
            continue

        logging.info(f"📅 {season}: {len(tournaments)} tournaments")

        for _, t in tqdm(tournaments.iterrows(), total=len(tournaments), desc=f"[{season}] Processing tournaments"):
            tid, name = t["TournamentID"], t["Name"]

            # 🧩 Get leaderboard safely
            lb = get_leaderboard(tid)
            if lb.empty or "PlayerID" not in lb.columns:
                continue  # skip if no player data

            # 🧩 Merge leaderboard + rankings
            df = lb.merge(rankings, how="left", on="PlayerID")

            # Tournament info
            df["Season"] = season
            df["TournamentID"] = df.get("Tournament_TournamentID", tid)
            df["TournamentName"] = df.get("Tournament_Name", name)
            df["StartDate"] = df.get("Tournament_StartDate", t.get("StartDate"))
            df["Par"] = df.get("Tournament_Par", t.get("Par"))
            df["Yards"] = df.get("Tournament_Yards", t.get("Yards"))
            df["City"] = df.get("Tournament_City", t.get("City"))
            df["State"] = df.get("Tournament_State", t.get("State"))

            # Player name (ensure correct field)
            if "PlayerNameFull" in df.columns:
                df["Name"] = df["PlayerNameFull"]
            elif {"FirstName", "LastName"}.issubset(df.columns):
                df["Name"] = df["FirstName"] + " " + df["LastName"]
            elif "PlayerName" in df.columns:
                df["Name"] = df["PlayerName"]
            else:
                df["Name"] = "Unknown Player"

            # Label (winner flag)
            df["result_win"] = (df["Rank"] == 1).astype(int)

            # Weather enrichment
            if pd.notna(t.get("City")) and pd.notna(t.get("StartDate")):
                date = str(t["StartDate"]).split("T")[0]
                temp, wind = get_weather(t["City"], date)
                df["AvgTemp"] = temp
                df["AvgWind"] = wind
            else:
                df["AvgTemp"] = None
                df["AvgWind"] = None

            records.append(df)
            time.sleep(0.2)

    # ✅ After all tournaments processed
    if not records:
        logging.error("No tournament data collected.")
        return pd.DataFrame()

    all_data = pd.concat(records, ignore_index=True)
    all_data.to_csv("golf_training_multi_year.csv", index=False)
    logging.info(f"✅ Saved combined dataset: {len(all_data)} rows.")
    return all_data


# # ================== FEATURES ==================

# def create_features(df):
#     logging.info("🧩 Creating enriched features...")

#     df["WorldGolfRank"] = pd.to_numeric(df["WorldGolfRank"], errors="coerce")
#     df["AveragePoints"] = pd.to_numeric(df["AveragePoints"], errors="coerce")
#     df["Par"] = pd.to_numeric(df["Par"], errors="coerce")
#     df["Yards"] = pd.to_numeric(df["Yards"], errors="coerce")
#     df["AvgTemp"] = pd.to_numeric(df["AvgTemp"], errors="coerce")
#     df["AvgWind"] = pd.to_numeric(df["AvgWind"], errors="coerce")

#     # Player form features
#     df = df.sort_values(["PlayerID", "StartDate"])
#     df["RecentFinish"] = df.groupby("PlayerID")["Rank"].shift(1)
#     df["AvgFinish_Last3"] = (
#         df.groupby("PlayerID")["Rank"].transform(lambda x: x.shift(1).rolling(3).mean())
#     )
#     df["Top10Rate_Last5"] = (
#         df.groupby("PlayerID")["Rank"].transform(lambda x: x.shift(1).rolling(5).apply(lambda y: (y<=10).sum()/5))
#     )
#     df["EventsPlayed"] = df.groupby("PlayerID").cumcount()

#     df = df.fillna(0)
#     features = [
#         "WorldGolfRank","AveragePoints","Par","Yards","AvgTemp","AvgWind",
#         "RecentFinish","AvgFinish_Last3","Top10Rate_Last5","EventsPlayed"
#     ]

#     # ✅ Keep tournament + player metadata for evaluation
#     meta_cols = ["TournamentID", "TournamentName", "StartDate", "PlayerID", "Name", "Rank"]
#     meta_cols = [c for c in meta_cols if c in df.columns]

#     return df[features + ["result_win"] + meta_cols]
def create_features(df):
    logging.info("🧩 Creating enriched features...")

    # === Convert numeric columns safely ===
    df["WorldGolfRank"] = pd.to_numeric(df.get("WorldGolfRank"), errors="coerce")
    df["AveragePoints"] = pd.to_numeric(df.get("AveragePoints"), errors="coerce")
    df["Par"] = pd.to_numeric(df.get("Par"), errors="coerce")
    df["Yards"] = pd.to_numeric(df.get("Yards"), errors="coerce")
    df["AvgTemp"] = pd.to_numeric(df.get("AvgTemp"), errors="coerce")
    df["AvgWind"] = pd.to_numeric(df.get("AvgWind"), errors="coerce")

    # === Player form features ===
    df = df.sort_values(["PlayerID", "StartDate"])
    df["RecentFinish"] = df.groupby("PlayerID")["Rank"].shift(1)
    df["AvgFinish_Last3"] = (
        df.groupby("PlayerID")["Rank"].transform(lambda x: x.shift(1).rolling(3).mean())
    )
    df["Top10Rate_Last5"] = (
        df.groupby("PlayerID")["Rank"].transform(lambda x: x.shift(1).rolling(5).apply(lambda y: (y <= 10).sum() / 5))
    )
    df["EventsPlayed"] = df.groupby("PlayerID").cumcount()

    df = df.fillna(0)

    features = [
        "WorldGolfRank", "AveragePoints", "Par", "Yards", "AvgTemp", "AvgWind",
        "RecentFinish", "AvgFinish_Last3", "Top10Rate_Last5", "EventsPlayed"
    ]

    # === ✅ Normalize player name column for evaluation ===
    if "Name_x" in df.columns:
        df["Name"] = df["Name_x"].astype(str)
    elif "Name_y" in df.columns:
        df["Name"] = df["Name_y"].astype(str)
    elif "PlayerName" in df.columns:
        df["Name"] = df["PlayerName"].astype(str)
    elif {"FirstName", "LastName"}.issubset(df.columns):
        df["Name"] = df["FirstName"].astype(str) + " " + df["LastName"].astype(str)
    else:
        df["Name"] = "Unknown Player"

    # === Keep tournament + player metadata ===
    meta_cols = ["TournamentID", "TournamentName", "StartDate", "PlayerID", "Name", "Rank"]
    meta_cols = [c for c in meta_cols if c in df.columns]

    # === Return merged feature + meta data ===
    return df[features + ["result_win"] + meta_cols]

# ================== TRAIN ==================


def train_model(df):
    logging.info("🏋️ Training multi-year model...")

    feature_cols = [
        "WorldGolfRank","AveragePoints","Par","Yards","AvgTemp","AvgWind",
        "RecentFinish","AvgFinish_Last3","Top10Rate_Last5","EventsPlayed"
    ]

    X = df[feature_cols].copy()
    y = df["result_win"]

    if y.sum() == 0:
        logging.error("No winners found.")
        return None

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

    model = XGBClassifier(
        n_estimators=700,
        learning_rate=0.03,
        max_depth=5,
        subsample=0.9,
        colsample_bytree=0.9,
        scale_pos_weight=(len(y)-y.sum())/y.sum(),
        eval_metric="logloss"
    )

    model.fit(X_train, y_train)
    preds = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, preds)
    brier = brier_score_loss(y_test, preds)
    logging.info(f"✅ Trained | AUC={auc:.4f}, Brier={brier:.4f}")

    # 💾 Save once
    model.save_model("golf_model.json")
    logging.info("💾 Saved trained model to golf_model.json")

    return model

def predict_next_week(model):
    if model is None:
        logging.error("❌ Model not available, cannot predict.")
        return pd.DataFrame()

    logging.info("🔮 Fetching next tournament for prediction...")

    tournaments = get_tournaments(2025)
    if tournaments.empty:
        logging.error("❌ No tournaments found for 2025.")
        return pd.DataFrame()

    tournaments["StartDate"] = pd.to_datetime(tournaments["StartDate"], errors="coerce")
    today = pd.Timestamp(datetime.now().date())
    upcoming = tournaments[tournaments["StartDate"] >= today].sort_values("StartDate")

    if upcoming.empty:
        logging.error("❌ No upcoming tournaments found.")
        return pd.DataFrame()

    next_event = upcoming.iloc[0]
    name = next_event["Name"]
    tid = next_event["TournamentID"]
    start = next_event["StartDate"].date()
    logging.info(f"📅 Next tournament: {name} (ID: {tid}) starting {start}")

    # --- Fetch leaderboard and rankings ---
    lb = get_leaderboard(tid)
    rankings = get_rankings(2024)

    if lb.empty:
        logging.error(f"❌ No player data found for {name}.")
        return pd.DataFrame()

    # --- Merge and build enriched features ---
    df = lb.merge(rankings, on="PlayerID", how="left")

    # Add course information if available
    df["Par"] = next_event.get("Par", 72)
    df["Yards"] = next_event.get("Yards", 7000)
    df["City"] = next_event.get("City", None)
    df["State"] = next_event.get("State", None)

    # Weather features
    if next_event.get("City"):
        date_str = str(next_event["StartDate"]).split("T")[0]
        temp, wind = get_weather(next_event["City"], date_str)
        df["AvgTemp"] = temp
        df["AvgWind"] = wind
    else:
        df["AvgTemp"] = None
        df["AvgWind"] = None

    # Fill numeric columns
    df["WorldGolfRank"] = pd.to_numeric(df["WorldGolfRank"], errors="coerce")
    df["AveragePoints"] = pd.to_numeric(df["AveragePoints"], errors="coerce")
    df["Par"] = pd.to_numeric(df["Par"], errors="coerce")
    df["Yards"] = pd.to_numeric(df["Yards"], errors="coerce")
    df["AvgTemp"] = pd.to_numeric(df["AvgTemp"], errors="coerce")
    df["AvgWind"] = pd.to_numeric(df["AvgWind"], errors="coerce")

    # Synthetic recent performance placeholders (0s for future)
    df["RecentFinish"] = 0
    df["AvgFinish_Last3"] = 0
    df["Top10Rate_Last5"] = 0
    df["EventsPlayed"] = 0

    # Detect player name
    if "Name_x" in df.columns:
        df["Name"] = df["Name_x"]
    elif "PlayerName" in df.columns:
        df["Name"] = df["PlayerName"]
    elif {"FirstName", "LastName"}.issubset(df.columns):
        df["Name"] = df["FirstName"] + " " + df["LastName"]
    else:
        df["Name"] = "Unknown Player"

    df = df.fillna(0)

    # ✅ Use same enriched feature set as training
    features = [
        "WorldGolfRank","AveragePoints","Par","Yards","AvgTemp","AvgWind",
        "RecentFinish","AvgFinish_Last3","Top10Rate_Last5","EventsPlayed"
    ]

    df["WinProbability"] = model.predict_proba(df[features])[:, 1]
    df = df[["Name","WinProbability"]].sort_values("WinProbability", ascending=False)

    logging.info(f"✅ Predictions generated for {len(df)} players.")
    df.to_csv(f"predicted_winners_{name.replace(' ', '_')}_2025.csv", index=False)
    logging.info(f"💾 Saved predictions to predicted_winners_{name.replace(' ', '_')}_2025.csv")

    return df



def evaluate_top5_accuracy(model, df):
    """
    Compare model's top-5 predicted players vs actual top-5 per tournament.
    Saves readable summary CSV with tournament name + start date + actual winner.
    """
    summary = []
    tournaments = df["TournamentID"].unique()
    total = 0
    hit_tournaments = 0
    total_overlap = 0

    # ✅ Match same feature columns used during training
    feature_cols = [
        "WorldGolfRank", "AveragePoints", "Par", "Yards", "AvgTemp", "AvgWind",
        "RecentFinish", "AvgFinish_Last3", "Top10Rate_Last5", "EventsPlayed"
    ]

    for tid in tournaments:
        subset = df[df["TournamentID"] == tid].copy()
        if subset.empty or "Rank" not in subset.columns:
            continue

        # ✅ Robust name detection (handles Name_x, Name_y, PlayerName, etc.)
        name_cols = [c for c in subset.columns if "Name" in c and "Tournament" not in c]
        if len(name_cols) > 0:
            # Prefer Name_x (leaderboard), fallback to next
            preferred = "Name_x" if "Name_x" in name_cols else name_cols[0]
            subset["Name"] = subset[preferred].astype(str)
        else:
            subset["Name"] = "Unknown Player"

        # ✅ Only numeric features for prediction
        X = subset[feature_cols].fillna(0)
        preds = model.predict_proba(X)[:, 1]
        subset["pred_score"] = preds

        tname = subset["TournamentName"].iloc[0] if "TournamentName" in subset.columns else f"Tournament {tid}"
        tdate = subset["StartDate"].iloc[0] if "StartDate" in subset.columns else None

        # ✅ Identify top predicted and actual players
        top_pred = subset.sort_values("pred_score", ascending=False).head(5)["Name"].tolist()
        actual_top5 = subset.sort_values("Rank", ascending=True).head(5)["Name"].tolist()
        actual_winner = subset.loc[subset["Rank"].idxmin(), "Name"]

        # ✅ Compare overlap
        overlap = len(set(top_pred).intersection(set(actual_top5)))
        total_overlap += overlap
        total += 1
        if overlap > 0:
            hit_tournaments += 1

        summary.append({
            "TournamentName": tname,
            "StartDate": tdate,
            "PredictedTop5": ", ".join(top_pred),
            "ActualTop5": ", ".join(actual_top5),
            "ActualWinner": actual_winner,
            "OverlapCount": overlap
        })

    # ✅ Compute global metrics
    top5_hit_rate = hit_tournaments / total if total else 0
    avg_overlap = total_overlap / total if total else 0

    # ✅ Save summary
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv("top5_tournament_summary.csv", index=False)
    logging.info("💾 Saved Top-5 summary to top5_tournament_summary.csv")
    logging.info(f"🏆 Top-5 inclusion accuracy: {top5_hit_rate*100:.2f}% ({hit_tournaments}/{total}) tournaments")
    logging.info(f"📊 Avg overlap per tournament: {avg_overlap:.2f}")

    return summary_df



# ================== MAIN RUN ==================
if __name__ == "__main__":
    import os

    logging.info("🚀 Starting Golf Prediction Pipeline (Evaluation + Next Week Prediction)...")

    # === Step 1: Load saved model or train if missing ===
    if os.path.exists("golf_model.json"):
        logging.info("📂 Loading saved model from golf_model.json...")
        model = XGBClassifier()
        model.load_model("golf_model.json")
    else:
        logging.info("🧩 No saved model found — building dataset and training model...")
        df = build_training_multi(YEARS)
        if not df.empty:
            feats = create_features(df)
            model = train_model(feats)
            model.save_model("golf_model.json")
            logging.info("💾 Saved trained model to golf_model.json")
        else:
            logging.error("❌ No data available. Exiting.")
            exit()

    # === Step 2: Evaluate top-5 accuracy using past tournaments ===
    if os.path.exists("golf_training_multi_year.csv"):
        logging.info("📂 Loading existing dataset for evaluation...")
        df = pd.read_csv("golf_training_multi_year.csv")
    else:
        logging.info("⚙️ Building dataset for evaluation...")
        df = build_training_multi(YEARS)
        df.to_csv("golf_training_multi_year.csv", index=False)

    feats = create_features(df)
    evaluate_top5_accuracy(model, feats)

    # === Step 3: Predict next week's tournament ===
    preds = predict_next_week(model)

    if not preds.empty:
        top5 = preds.head(5)
        logging.info("🏆 Next-week Top-5 Predicted Winners:")
        print(top5.to_string(index=False))
        top5.to_csv("top5_next_week.csv", index=False)
        logging.info("💾 Saved Top-5 next-week predictions to top5_next_week.csv")
    else:
        logging.warning("⚠️ No predictions available.")
