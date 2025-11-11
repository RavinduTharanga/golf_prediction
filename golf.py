import requests
import pandas as pd
import time
import logging
from datetime import datetime
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import brier_score_loss, roc_auc_score

# ========== CONFIG ==========
API_KEY = "e3cf5dd42289495280baabbdecf46355"
BASE = "https://api.sportsdata.io/golf/v2/json"
HEAD = {"Ocp-Apim-Subscription-Key": API_KEY}

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)

# ============================

def get_json(url):
    """Generic GET request with error handling"""
    try:
        r = requests.get(url, headers=HEAD)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logging.error(f"❌ API request failed: {url}\n{e}")
        return {}

# 1️⃣  Get tournaments in a season
def get_tournaments(season=2025):
    url = f"{BASE}/Tournaments/{season}?key={API_KEY}"
    return pd.DataFrame(get_json(url))

# 2️⃣  Get OWGR rankings
def get_rankings(season=2025):
    url = f"{BASE}/Rankings/{season}?key={API_KEY}"
    return pd.DataFrame(get_json(url))

# 3️⃣  Get leaderboard for a tournament
def get_leaderboard(tournament_id):
    url = f"{BASE}/Leaderboard/{tournament_id}?key={API_KEY}"
    data = get_json(url)
    return pd.json_normalize(data["Players"]) if "Players" in data else pd.DataFrame()

# ================== BUILD TRAINING DATA ==================
def build_training(season=2025):
    logging.info(f"🔄 Starting data collection for season {season}...")
    tournaments = get_tournaments(season)
    rankings = get_rankings(season)

    if tournaments.empty:
        logging.error("❌ No tournaments found — check your API key or season value.")
        return pd.DataFrame()

    logging.info(f"📅 Found {len(tournaments)} tournaments for {season}.")

    records = []
    for i, t in tournaments.iterrows():
        tid = t["TournamentID"]
        name = t["Name"]
        logging.info(f"⛳ Processing {i+1}/{len(tournaments)}: {name} (ID: {tid})")

        try:
            lb = get_leaderboard(tid)
            if lb.empty:
                logging.warning(f"⚠️ No leaderboard data for {name}")
                continue

            df = lb.merge(rankings, how="left", on="PlayerID")
            df["TournamentID"] = tid
            df["TournamentName"] = name
            df["StartDate"] = t["StartDate"]
            df["result_win"] = (df["Rank"] == 1).astype(int)
            records.append(df)
        except Exception as e:
            logging.error(f"❌ Failed to process {name}: {e}")

        time.sleep(0.3)  # Respect API limits

    if not records:
        logging.error("❌ No data collected for any tournament.")
        return pd.DataFrame()

    all_data = pd.concat(records, ignore_index=True)
    all_data.to_csv("golf_training_2025_raw.csv", index=False)
    logging.info(f"✅ Data collection complete. Saved {len(all_data)} rows.")
    return all_data

# ================== FEATURE ENGINEERING ==================
def create_features(df):
    logging.info("🧩 Creating training features...")

    df["WorldGolfRank"] = pd.to_numeric(df["WorldGolfRank"], errors="coerce")
    df["AveragePoints"] = pd.to_numeric(df["AveragePoints"], errors="coerce")

    # Previous event’s finishing position
    df["RecentFinish"] = df.groupby("PlayerID")["Rank"].shift(1)

    # Count of Top-10 finishes in the past 5 events
    df["RecentTop10Count"] = (
        df.groupby("PlayerID")["Rank"]
          .transform(lambda s: s.shift(1).rolling(5).apply(lambda y: (y <= 10).sum(), raw=False))
    )

    # Number of events played so far this season
    df["EventsPlayed"] = df.groupby("PlayerID").cumcount()

    df = df.fillna(0)
    logging.info("✅ Feature engineering complete.")

    features = [
        "WorldGolfRank",
        "AveragePoints",
        "RecentFinish",
        "RecentTop10Count",
        "EventsPlayed",
    ]
    target = "result_win"
    return df[features + [target]]

# ================== TRAIN MODEL ==================
def train_model(df):
    logging.info("🏋️ Training model...")

    X = df.drop(columns="result_win")
    y = df["result_win"]

    if y.sum() == 0:
        logging.error("❌ No winners (result_win=1) found in dataset — cannot train.")
        return None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y
    )

    model = XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
    )
    model.fit(X_train, y_train)
    preds = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, preds)
    brier = brier_score_loss(y_test, preds)
    logging.info(f"✅ Model trained | AUC: {auc:.4f} | Brier: {brier:.4f}")

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

    # --- 🧠 Pick the *next* tournament after today ---
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
    rankings = get_rankings(2024)   # latest completed OWGR data

    if lb.empty:
        logging.error(f"❌ No player data found for {name}.")
        return pd.DataFrame()

    # --- Merge and build features ---
    df = lb.merge(rankings, on="PlayerID", how="left")
    df["WorldGolfRank"] = pd.to_numeric(df["WorldGolfRank"], errors="coerce")
    df["AveragePoints"] = pd.to_numeric(df["AveragePoints"], errors="coerce")
    df["RecentFinish"] = 0
    df["RecentTop10Count"] = 0
    df["EventsPlayed"] = 0

    # --- Detect player name column ---
    if "Name_x" in df.columns:
        df["Name"] = df["Name_x"]
    elif "PlayerName" in df.columns:
        df["Name"] = df["PlayerName"]
    elif "Name_y" in df.columns:
        df["Name"] = df["Name_y"]
    elif {"FirstName", "LastName"}.issubset(df.columns):
        df["Name"] = df["FirstName"] + " " + df["LastName"]
    else:
        df["Name"] = "Unknown Player"

    df = df.fillna(0)
    features = ["WorldGolfRank", "AveragePoints",
                "RecentFinish", "RecentTop10Count", "EventsPlayed"]

    df["WinProbability"] = model.predict_proba(df[features])[:, 1]
    df = df[["Name", "WinProbability"]].sort_values("WinProbability", ascending=False)

    logging.info(f"✅ Predictions generated for {len(df)} players.")
    df.to_csv(f"predicted_winners_{name.replace(' ', '_')}_2025.csv", index=False)
    logging.info(f"💾 Saved predictions to predicted_winners_{name.replace(' ', '_')}_2025.csv")

    return df



# ================== MAIN RUN ==================
if __name__ == "__main__":
    logging.info("🚀 Starting Golf Prediction Pipeline (2025 Season)...")
    df = build_training(2025)

    if not df.empty:
        feats = create_features(df)
        model = train_model(feats)
        preds = predict_next_week(model)

        if not preds.empty:
            logging.info("🏆 Top 10 Predicted Winners:")
            print(preds.head(10))
        else:
            logging.warning("⚠️ No predictions available.")
    else:
        logging.error("❌ Pipeline stopped: No data collected.")
