import os
import pandas as pd
import numpy as np

# ----------------------------
# PATHS
# ----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

FILE_1 = os.path.join(DATA_DIR, "india_2023_2026.csv")
FILE_2 = os.path.join(DATA_DIR, "feb26_feb27.csv")
OUTPUT_FILE = os.path.join(DATA_DIR, "processed_data.csv")

# ----------------------------
# DATA PREP FUNCTION
# ----------------------------
def prepare_rail_data(file_path):
    df = pd.read_csv(file_path)

    # Engineering constants
    E = 210000       # Young's Modulus (N/mm2)
    alpha = 11.5e-6  # Thermal expansion coefficient
    neutral_temp = 35

    # Convert Fahrenheit to Celsius
    df["temp_c"] = (df["temp"] - 32) * 5 / 9

    # Rail temperature approximation
    df["rail_temp"] = df["temp_c"] + 15

    # Stress calculation
    df["stress_mpa"] = E * alpha * (df["rail_temp"] - neutral_temp)
    df["stress_mpa"] = df["stress_mpa"].clip(lower=0)

    # Synthetic track age
    np.random.seed(42)
    df["track_age"] = np.random.randint(5, 40, size=len(df))

    # TMSI target
    df["tmsi"] = (df["stress_mpa"] / 150 * 0.7) + (df["track_age"] / 40 * 0.3)
    df["tmsi"] = df["tmsi"].clip(0, 1)

    features = ["temp_c", "humidity", "solarradiation", "track_age", "tmsi"]
    return df[features]

# ----------------------------
# MAIN
# ----------------------------
print("📥 Processing datasets...")

data1 = prepare_rail_data(FILE_1)
data2 = prepare_rail_data(FILE_2)

final_df = pd.concat([data1, data2], ignore_index=True)
final_df.to_csv(OUTPUT_FILE, index=False)

print(f"✅ Success! Processed {len(final_df)} rows.")
print(f"📁 Saved to: {OUTPUT_FILE}")