import os
import requests
import pandas as pd
import time
import logging
from datetime import datetime
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, brier_score_loss
from tqdm import tqdm

# ================== CONFIG ==================
API_KEY = "e3cf5dd42289495280baabbdecf46355"
BASE = "https://api.sportsdata.io/golf/v2/json"
HEAD = {"Ocp-Apim-Subscription-Key": API_KEY}
YEARS = [2024, 2025]
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"

# --- Clean Logging Setup ---
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(level=logging.INFO, format="%(message)s")
logging.getLogger("xgboost").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

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

def get_leaderboard(tournament_id):
    """Fetch leaderboard including player + tournament info."""
    data = get_json(f"{BASE}/Leaderboard/{tournament_id}?key={API_KEY}")
    if not data or "Players" not in data:
        return pd.DataFrame()

    players_df = pd.json_normalize(data["Players"])
    players_df["Name"] = players_df.get("Name", "Unknown Player")

    if "Tournament" in data and isinstance(data["Tournament"], dict):
        tinfo = data["Tournament"]
        for key in ["TournamentID", "Name", "StartDate", "City", "State", "Par", "Yards"]:
            if key in tinfo:
                players_df[f"Tournament_{key}"] = tinfo[key]
    else:
        players_df["Tournament_TournamentID"] = tournament_id

    return players_df

def get_weather(city, date):
    try:
        params = {"latitude": 0, "longitude": 0, "hourly": "temperature_2m,windspeed_10m"}
        geo = requests.get(f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1").json()
        if "results" not in geo:
            return (None, None)
        lat, lon = geo["results"][0]["latitude"], geo["results"][0]["longitude"]
        params.update({"latitude": lat, "longitude": lon, "start_date": date, "end_date": date})
        weather = requests.get(WEATHER_URL, params=params).json()
        temps = weather.get("hourly", {}).get("temperature_2m", [])
        winds = weather.get("hourly", {}).get("windspeed_10m", [])
        if not temps:
            return (None, None)
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

        for _, t in tqdm(tournaments.iterrows(), total=len(tournaments),
                         desc=f"[{season}] Processing tournaments", leave=False):
            tid, name = t["TournamentID"], t["Name"]
            lb = get_leaderboard(tid)
            if lb.empty or "PlayerID" not in lb.columns:
                continue

            df = lb.merge(rankings, how="left", on="PlayerID")
            df["Season"] = season
            df["TournamentID"] = df.get("Tournament_TournamentID", tid)
            df["TournamentName"] = df.get("Tournament_Name", name)
            df["StartDate"] = df.get("Tournament_StartDate", t.get("StartDate"))
            df["Par"] = df.get("Tournament_Par", t.get("Par"))
            df["Yards"] = df.get("Tournament_Yards", t.get("Yards"))
            df["City"] = df.get("Tournament_City", t.get("City"))
            df["State"] = df.get("Tournament_State", t.get("State"))
            df["Name"] = df.get("Name", "Unknown Player")
            df["result_win"] = (df["Rank"] == 1).astype(int)

            if pd.notna(t.get("City")) and pd.notna(t.get("StartDate")):
                date = str(t["StartDate"]).split("T")[0]
                temp, wind = get_weather(t["City"], date)
                df["AvgTemp"] = temp
                df["AvgWind"] = wind
            else:
                df["AvgTemp"] = None
                df["AvgWind"] = None

            records.append(df)
            time.sleep(0.15)

    if not records:
        logging.error("No tournament data collected.")
        return pd.DataFrame()

    all_data = pd.concat(records, ignore_index=True)
    all_data.to_csv("golf_training_multi_year.csv", index=False)
    logging.info(f"✅ Saved combined dataset: {len(all_data)} rows.")
    return all_data

# ================== FEATURES ==================
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
    feature_cols = ["WorldGolfRank", "AveragePoints", "Par", "Yards", "AvgTemp", "AvgWind",
                    "RecentFinish", "AvgFinish_Last3", "Top10Rate_Last5", "EventsPlayed"]
    X = df[feature_cols].copy()
    y = df["result_win"]
    if y.sum() == 0:
        logging.error("No winners found.")
        return None

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
    model = XGBClassifier(
        n_estimators=700, learning_rate=0.03, max_depth=5,
        subsample=0.9, colsample_bytree=0.9,
        scale_pos_weight=(len(y)-y.sum())/y.sum(),
        eval_metric="logloss"
    )

    model.fit(X_train, y_train)
    preds = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, preds)
    brier = brier_score_loss(y_test, preds)
    logging.info(f"✅ Trained | AUC={auc:.4f}, Brier={brier:.4f}")
    model.save_model("golf_model.json")
    logging.info("💾 Saved trained model to golf_model.json")
    return model

# ================== EVALUATE ==================
def evaluate_top5_accuracy(model, df):
    summary = []
    tournaments = df["TournamentID"].unique()
    total, hit_tournaments, total_overlap = 0, 0, 0
    feature_cols = ["WorldGolfRank","AveragePoints","Par","Yards","AvgTemp","AvgWind",
                    "RecentFinish","AvgFinish_Last3","Top10Rate_Last5","EventsPlayed"]

    for tid in tournaments:
        subset = df[df["TournamentID"] == tid].copy()
        if subset.empty or "Rank" not in subset.columns:
            continue

        # Name handling
        name_cols = [c for c in subset.columns if "Name" in c and "Tournament" not in c]
        subset["Name"] = subset[name_cols[0]].astype(str) if name_cols else "Unknown Player"

        X = subset[feature_cols].fillna(0)
        preds = model.predict_proba(X)[:, 1]
        subset["pred_score"] = preds

        tname = subset["TournamentName"].iloc[0] if "TournamentName" in subset.columns else f"Tournament {tid}"
        tdate = subset["StartDate"].iloc[0] if "StartDate" in subset.columns else None

        # top_pred = subset.sort_values("pred_score", ascending=False).head(5)["Name"].tolist()
        # actual_top5 = subset.sort_values("Rank", ascending=True).head(5)["Name"].tolist()
        top_pred = subset.sort_values("pred_score", ascending=False).head(3)["Name"].tolist()
        actual_top3 = subset.sort_values("Rank", ascending=True).head(3)["Name"].tolist()

        actual_winner = subset.loc[subset["Rank"].idxmin(), "Name"]
        overlap = len(set(top_pred).intersection(set(actual_top3)))

        total += 1
        total_overlap += overlap
        if overlap > 0:
            hit_tournaments += 1

        # summary.append({
        #     "TournamentName": tname, "StartDate": tdate,
        #     "PredictedTop5": ", ".join(top_pred),
        #     "ActualTop5": ", ".join(actual_top5),
        #     "ActualWinner": actual_winner, "OverlapCount": overlap
        # })
        overlap = len(set(top_pred).intersection(set(actual_top3)))

        summary.append({
            "TournamentName": tname, "StartDate": tdate,
            "PredictedTop3": ", ".join(top_pred),
            "ActualTop3": ", ".join(actual_top3),
            "ActualWinner": actual_winner, "OverlapCount": overlap
        })


    summary_df = pd.DataFrame(summary)
    summary_df.to_csv("top3_tournament_summary.csv", index=False)
    top5_hit_rate = hit_tournaments / total if total else 0
    avg_overlap = total_overlap / total if total else 0
    logging.info("💾 Saved Top-5 summary to top5_tournament_summary.csv")
    logging.info(f"🏆 Top-5 inclusion accuracy: {top5_hit_rate*100:.2f}% ({hit_tournaments}/{total}) tournaments")
    logging.info(f"📊 Avg overlap per tournament: {avg_overlap:.2f}")
    return summary_df

# ================== PREDICT NEXT WEEK ==================
def predict_next_week(model):
    logging.info("🔮 Fetching next tournament for prediction...")
    tournaments = get_tournaments(2025)
    tournaments["StartDate"] = pd.to_datetime(tournaments["StartDate"], errors="coerce")
    today = pd.Timestamp(datetime.now().date())
    upcoming = tournaments[tournaments["StartDate"] >= today].sort_values("StartDate")

    if upcoming.empty:
        logging.error("❌ No upcoming tournaments found.")
        return pd.DataFrame()

    next_event = upcoming.iloc[0]
    name, tid = next_event["Name"], next_event["TournamentID"]
    start = next_event["StartDate"].date()
    logging.info(f"📅 Next tournament: {name} (ID: {tid}) starting {start}")

    lb = get_leaderboard(tid)
    rankings = get_rankings(2024)
    if lb.empty:
        logging.error(f"❌ No player data found for {name}.")
        return pd.DataFrame()

    df = lb.merge(rankings, on="PlayerID", how="left")
    df["Par"] = next_event.get("Par", 72)
    df["Yards"] = next_event.get("Yards", 7000)
    df["City"] = next_event.get("City", None)
    df["State"] = next_event.get("State", None)

    if next_event.get("City"):
        date_str = str(next_event["StartDate"]).split("T")[0]
        temp, wind = get_weather(next_event["City"], date_str)
        df["AvgTemp"] = temp
        df["AvgWind"] = wind
    else:
        df["AvgTemp"] = None
        df["AvgWind"] = None

    for col in ["WorldGolfRank", "AveragePoints", "Par", "Yards", "AvgTemp", "AvgWind"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["RecentFinish"] = df["AvgFinish_Last3"] = df["Top10Rate_Last5"] = df["EventsPlayed"] = 0

    if "Name_x" in df.columns:
        df["Name"] = df["Name_x"]
    elif "PlayerName" in df.columns:
        df["Name"] = df["PlayerName"]
    elif {"FirstName", "LastName"}.issubset(df.columns):
        df["Name"] = df["FirstName"] + " " + df["LastName"]
    else:
        df["Name"] = "Unknown Player"

    df = df.fillna(0)
    features = ["WorldGolfRank","AveragePoints","Par","Yards","AvgTemp","AvgWind",
                "RecentFinish","AvgFinish_Last3","Top10Rate_Last5","EventsPlayed"]

    df["WinProbability"] = model.predict_proba(df[features])[:, 1]
    df = df[["Name","WinProbability"]].sort_values("WinProbability", ascending=False)
    df.to_csv(f"predicted_winners_{name.replace(' ', '_')}_2025.csv", index=False)

    logging.info(f"✅ Predictions generated for {len(df)} players.")
    logging.info(f"💾 Saved predictions to predicted_winners_{name.replace(' ', '_')}_2025.csv")
    logging.info("🏆 Next-week Top-5 Predicted Winners:")
    print(df.head(5).to_string(index=False))
    df.head(5).to_csv("top5_next_week.csv", index=False)
    return df

# ================== MAIN ==================
if __name__ == "__main__":
    logging.info("🚀 Starting Golf Prediction Pipeline...")

    if os.path.exists("golf_model.json"):
        model = XGBClassifier()
        model.load_model("golf_model.json")
        logging.info("📂 Loaded existing model.")
    else:
        df = build_training_multi(YEARS)
        feats = create_features(df)
        model = train_model(feats)

    if os.path.exists("golf_training_multi_year.csv"):
        df = pd.read_csv("golf_training_multi_year.csv")
    else:
        df = build_training_multi(YEARS)

    feats = create_features(df)
    evaluate_top5_accuracy(model, feats)
    predict_next_week(model)
