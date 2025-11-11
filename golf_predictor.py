import requests
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import datetime

# ========== CONFIG ==========
API_KEY = "YOUR_DATAGOLF_API_KEY"   # Replace with your key
TOUR = "pga"
N_WEEKS_HISTORY = 52  # 1 year of data

# ========== STEP 1: FETCH HISTORICAL RESULTS ==========
print("Fetching historical tournament results...")
url_results = f"https://feeds.datagolf.com/preds/historical?tour={TOUR}&key={API_KEY}"
results = requests.get(url_results).json()
results_df = pd.DataFrame(results["predictions"])

# ========== STEP 2: FETCH PLAYER STATS ==========
print("Fetching player stats...")
url_stats = f"https://feeds.datagolf.com/stats/players?tour={TOUR}&key={API_KEY}"
stats = requests.get(url_stats).json()
stats_df = pd.DataFrame(stats["stats"])

# ========== STEP 3: FETCH WORLD RANKINGS ==========
print("Fetching world rankings...")
url_rank = f"https://feeds.datagolf.com/owgr?key={API_KEY}"
rank = requests.get(url_rank).json()
rank_df = pd.DataFrame(rank["rankings"])

# ========== STEP 4: MERGE & CLEAN ==========
print("Merging datasets...")
df = (
    results_df.merge(stats_df, on="player_name", how="left")
              .merge(rank_df, on="player_name", how="left")
)

# Basic cleaning
df = df.fillna(0)
df["winner"] = (df["finish_position"] == 1).astype(int)

# Create "recent form" feature (average finish over last 5 events)
df = df.sort_values(["player_name", "date"])
df["recent_form"] = df.groupby("player_name")["finish_position"].rolling(5, min_periods=1).mean().reset_index(level=0, drop=True)
df["recent_form"] = 1 / df["recent_form"]  # lower finish = higher form

# Features
features = [
    "owgr_rank",
    "strokes_gained_total",
    "strokes_gained_putting",
    "strokes_gained_off_tee",
    "strokes_gained_approach",
    "recent_form"
]
df = df.dropna(subset=["winner"])

# ========== STEP 5: TRAIN MODEL ==========
print("Training model...")
X = df[features].fillna(0)
y = df["winner"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = XGBClassifier(
    n_estimators=400,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss"
)
model.fit(X_train, y_train)
preds = model.predict_proba(X_test)[:,1]
print("Validation AUC:", roc_auc_score(y_test, preds))

# ========== STEP 6: PREDICT NEXT WEEK ==========
print("Fetching next tournament field...")
url_next = f"https://feeds.datagolf.com/preds/current?tour={TOUR}&key={API_KEY}"
next_data = requests.get(url_next).json()
next_week_df = pd.DataFrame(next_data["predictions"])

# Predict win probability
next_week_df["win_prob"] = model.predict_proba(next_week_df[features].fillna(0))[:,1]
next_week_df = next_week_df.sort_values("win_prob", ascending=False)

# ========== STEP 7: SAVE OUTPUT ==========
today = datetime.date.today().isoformat()
next_week_df.to_csv(f"predictions_{today}.csv", index=False)
print("\nTop 10 Predicted Winners:")
print(next_week_df[["player_name","win_prob"]].head(10))
print("\nFull prediction saved as:", f"predictions_{today}.csv")
