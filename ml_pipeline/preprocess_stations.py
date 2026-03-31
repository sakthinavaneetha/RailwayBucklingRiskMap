import os
import random
import pandas as pd

# ----------------------------
# PATHS
# ----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
INPUT_PATH = os.path.join(BASE_DIR, "data", "Train_stations.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "stations_data.csv")

MIN_TRACK_AGE = 8
MAX_TRACK_AGE = 35

# ----------------------------
# LOAD RAW STATION DATA
# ----------------------------
df = pd.read_csv(INPUT_PATH)

# ----------------------------
# KEEP REQUIRED COLUMNS
# ----------------------------
df = df[
    [
        "station_name",
        "station_code",
        "state_name",
        "lat",
        "lng"
    ]
]

# ----------------------------
# CLEAN
# ----------------------------
df["state_name"] = df["state_name"].fillna("Unknown")

def generate_track_age():
    return random.randint(MIN_TRACK_AGE, MAX_TRACK_AGE)

df["track_age_years"] = df.apply(lambda _: generate_track_age(), axis=1)

# ----------------------------
# SAVE
# ----------------------------
df.to_csv(OUTPUT_PATH, index=False)

print("✅ Station preprocessing completed")
print(f"📁 Saved clean data to: {OUTPUT_PATH}")
print(df.head())