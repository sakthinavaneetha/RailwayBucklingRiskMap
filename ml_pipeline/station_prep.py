import os
import json
import pandas as pd

def process_station_geojson():
    BASE_DIR = os.path.dirname(os.path.dirname(__file__))
    input_path = os.path.join(BASE_DIR, "data", "stations.json")
    output_path = os.path.join(BASE_DIR, "data", "tamilnadu_stations.csv")

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []

    for feature in data["features"]:
        geometry = feature.get("geometry")
        props = feature.get("properties", {})

        if geometry and geometry.get("coordinates"):
            lon, lat = geometry["coordinates"]
            rows.append({
                "station": props.get("name"),
                "code": props.get("code"),
                "state": props.get("state"),
                "latitude": lat,
                "longitude": lon
            })

    df = pd.DataFrame(rows)

    df_tn = df[df["state"] == "Tamil Nadu"]
    df_tn.to_csv(output_path, index=False)

    print("✅ Tamil Nadu stations file created successfully")
    print(f"📁 Saved to: {output_path}")

if __name__ == "__main__":
    process_station_geojson()