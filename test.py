from pprint import pprint

data = get_json("https://api.sportsdata.io/golf/v2/json/Leaderboard/643?key=e3cf5dd42289495280baabbdecf46355")
print("Top-level keys:", list(data.keys()))

# Inspect the first player record
if "Players" in data:
    first_player = data["Players"][0]
    print("\nKeys in 'Players' section:")
    pprint(first_player)
