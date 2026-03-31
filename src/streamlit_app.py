import os
import json
import math
import tempfile
import joblib
import folium
import numpy as np
import pandas as pd
import streamlit as st

from weather_service import get_weather_features_for_horizon

# ---------------------------------
# PATHS
# ---------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "output", "rail_stress_model.pkl")
STATIONS_JSON_PATH = os.path.join(BASE_DIR, "data", "stations.json")
TRAINS_JSON_PATH = os.path.join(BASE_DIR, "data", "trains.json")
SCHEDULES_JSON_PATH = os.path.join(BASE_DIR, "data", "schedules.json")

# ---------------------------------
# PERFORMANCE CONFIG
# ---------------------------------
MAX_ALL_STATIONS = 80
MAX_SEARCH_STATIONS = 40
MAX_ROUTE_STATIONS = 35
WEATHER_API_LIMIT = 12
ROUTE_DISTANCE_THRESHOLD = 0.18
ROUNDED_COORD_DECIMALS = 2

# ---------------------------------
# PAGE CONFIG
# ---------------------------------
st.set_page_config(
    page_title="Rail Thermal Buckling Risk Map",
    layout="wide"
)

st.title("🚆 Rail Thermal Buckling Risk Map")
st.caption("Optimized route/station search with ML-based thermal stress prediction")

# ---------------------------------
# HELPERS
# ---------------------------------
def safe_str(value):
    if value is None:
        return ""
    return str(value).strip()

def synthetic_track_age(code: str) -> int:
    if not code:
        return 20
    return 8 + (sum(ord(ch) for ch in code) % 28)

def normalize_text(value: str) -> str:
    return safe_str(value).lower()

def point_distance_deg(lat1, lon1, lat2, lon2):
    return math.sqrt((lat1 - lat2) ** 2 + (lon1 - lon2) ** 2)

def build_station_display_name(row):
    code = safe_str(row.get("code"))
    name = safe_str(row.get("name"))
    return f"{name} ({code})"

def route_bbox(route_coords):
    lons = [p[0] for p in route_coords]
    lats = [p[1] for p in route_coords]
    return min(lats), max(lats), min(lons), max(lons)

def min_distance_to_route(lat, lon, route_coords):
    best = float("inf")
    for lon2, lat2 in route_coords:
        d = point_distance_deg(lat, lon, lat2, lon2)
        if d < best:
            best = d
    return best

# ---------------------------------
# LOAD MODEL
# ---------------------------------
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found at: {MODEL_PATH}. Run train_model.py first."
        )
    return joblib.load(MODEL_PATH)

# ---------------------------------
# LOAD STATIONS
# ---------------------------------
@st.cache_data
def load_stations():
    if not os.path.exists(STATIONS_JSON_PATH):
        raise FileNotFoundError(f"stations.json not found at: {STATIONS_JSON_PATH}")

    with open(STATIONS_JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []

    for feature in data.get("features", []):
        geometry = feature.get("geometry")
        props = feature.get("properties", {})

        if not geometry or geometry.get("type") != "Point":
            continue

        coords = geometry.get("coordinates")
        if not coords or len(coords) < 2:
            continue

        lon, lat = coords[0], coords[1]
        code = safe_str(props.get("code"))

        rows.append({
            "name": safe_str(props.get("name")),
            "code": code,
            "lat": float(lat),
            "lng": float(lon),
            "track_age_years": synthetic_track_age(code),
        })

    df = pd.DataFrame(rows).drop_duplicates(subset=["code", "name"]).reset_index(drop=True)

    if df.empty:
        raise ValueError("No valid station points found in stations.json")

    df["display_name"] = df.apply(build_station_display_name, axis=1)
    return df

# ---------------------------------
# LOAD TRAINS
# ---------------------------------
@st.cache_data
def load_trains():
    if not os.path.exists(TRAINS_JSON_PATH):
        raise FileNotFoundError(f"trains.json not found at: {TRAINS_JSON_PATH}")

    with open(TRAINS_JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []

    for feature in data.get("features", []):
        geometry = feature.get("geometry")
        props = feature.get("properties", {})

        if not geometry or geometry.get("type") != "LineString":
            continue

        coords = geometry.get("coordinates", [])
        if not coords:
            continue

        rows.append({
            "train_number": safe_str(props.get("number")),
            "train_name": safe_str(props.get("name")),
            "train_type": safe_str(props.get("type")),
            "from_station_code": safe_str(props.get("from_station_code")),
            "from_station_name": safe_str(props.get("from_station_name")),
            "to_station_code": safe_str(props.get("to_station_code")),
            "to_station_name": safe_str(props.get("to_station_name")),
            "distance": props.get("distance"),
            "departure": safe_str(props.get("departure")),
            "arrival": safe_str(props.get("arrival")),
            "coordinates": coords,
        })

    df = pd.DataFrame(rows)

    if df.empty:
        raise ValueError("No valid train LineString routes found in trains.json")

    df["train_display"] = df.apply(
        lambda r: f"{r['train_number']} - {r['train_name']} ({r['from_station_code']} → {r['to_station_code']})",
        axis=1
    )

    return df.drop_duplicates(subset=["train_number", "from_station_code", "to_station_code"]).reset_index(drop=True)

# ---------------------------------
# LOAD SCHEDULES
# ---------------------------------
@st.cache_data
def load_schedules():
    if not os.path.exists(SCHEDULES_JSON_PATH):
        raise FileNotFoundError(f"schedules.json not found at: {SCHEDULES_JSON_PATH}")

    with open(SCHEDULES_JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    for item in data:
        rows.append({
            "train_number": safe_str(item.get("train_number")),
            "train_name": safe_str(item.get("train_name")),
            "station_name": safe_str(item.get("station_name")),
            "station_code": safe_str(item.get("station_code")),
        })

    df = pd.DataFrame(rows)

    if df.empty:
        raise ValueError("No valid rows found in schedules.json")

    return df

# ---------------------------------
# WEATHER CACHE
# ---------------------------------
@st.cache_data(show_spinner=False, ttl=1800)
def get_cached_weather_features(lat_bucket, lon_bucket, weather_horizon):
    return get_weather_features_for_horizon(lat_bucket, lon_bucket, weather_horizon)

def get_weather_for_station(lat, lon, weather_horizon):
    lat_bucket = round(float(lat), ROUNDED_COORD_DECIMALS)
    lon_bucket = round(float(lon), ROUNDED_COORD_DECIMALS)
    return get_cached_weather_features(lat_bucket, lon_bucket, weather_horizon)

# ---------------------------------
# FIND MATCHING TRAINS FOR SOURCE-DEST
# ---------------------------------
def find_trains_between_stations(schedules_df, source_code, dest_code):
    if not source_code or not dest_code or source_code == dest_code:
        return pd.DataFrame()

    source_trains = set(
        schedules_df.loc[schedules_df["station_code"] == source_code, "train_number"].dropna().unique()
    )
    dest_trains = set(
        schedules_df.loc[schedules_df["station_code"] == dest_code, "train_number"].dropna().unique()
    )

    common_trains = source_trains.intersection(dest_trains)
    if not common_trains:
        return pd.DataFrame()

    result = (
        schedules_df[schedules_df["train_number"].isin(common_trains)]
        [["train_number", "train_name"]]
        .drop_duplicates()
        .sort_values(["train_number", "train_name"])
        .reset_index(drop=True)
    )

    return result

# ---------------------------------
# FAST ROUTE FILTER
# ---------------------------------
def filter_stations_near_route(stations_df, route_coords, limit_count=MAX_ROUTE_STATIONS):
    if stations_df.empty or not route_coords:
        return stations_df.copy()

    min_lat, max_lat, min_lon, max_lon = route_bbox(route_coords)

    pad = 0.35
    candidates = stations_df[
        (stations_df["lat"] >= min_lat - pad) &
        (stations_df["lat"] <= max_lat + pad) &
        (stations_df["lng"] >= min_lon - pad) &
        (stations_df["lng"] <= max_lon + pad)
    ].copy()

    if candidates.empty:
        candidates = stations_df.copy()

    candidates["route_distance"] = candidates.apply(
        lambda r: min_distance_to_route(r["lat"], r["lng"], route_coords),
        axis=1
    )
    candidates = candidates.sort_values("route_distance")

    close = candidates[candidates["route_distance"] <= ROUTE_DISTANCE_THRESHOLD].copy()
    if close.empty:
        close = candidates.head(limit_count).copy()
    else:
        close = close.head(limit_count).copy()

    return close

# ---------------------------------
# SCORE STATIONS (BATCH PREDICTION)
# ---------------------------------
def score_stations(df, model, weather_horizon):
    if df.empty:
        return df.copy()

    scored = df.copy()

    temps = []
    humidities = []
    solars = []
    sources = []

    count = 0
    for _, row in scored.iterrows():
        if count < WEATHER_API_LIMIT:
            weather = get_weather_for_station(row["lat"], row["lng"], weather_horizon)
            count += 1
        else:
            weather = {
                "temp_c": float(np.random.normal(38, 3)),
                "humidity": float(np.random.normal(60, 8)),
                "solarradiation": float(np.random.normal(850, 120)),
                "weather_source": f"{weather_horizon} (Fast Fallback)"
            }

        temps.append(weather["temp_c"])
        humidities.append(weather["humidity"])
        solars.append(weather["solarradiation"])
        sources.append(weather["weather_source"])

    scored["temp_c"] = temps
    scored["humidity"] = humidities
    scored["solarradiation"] = solars
    scored["weather_source"] = sources

    feature_df = scored[["temp_c", "humidity", "solarradiation", "track_age_years"]].copy()
    feature_df.columns = ["temp_c", "humidity", "solarradiation", "track_age"]

    preds = model.predict(feature_df)
    preds = np.clip(preds, 0.0, 1.0)

    scored["predicted_tmsi"] = preds
    scored["marker_color"] = ["green" if x < 0.4 else "orange" if x < 0.7 else "red" for x in preds]
    scored["risk_level"] = ["LOW" if x < 0.4 else "MEDIUM" if x < 0.7 else "HIGH" for x in preds]

    return scored

# ---------------------------------
# MAP
# ---------------------------------
def build_map(
    stations_to_plot,
    selected_station_row=None,
    selected_train_row=None
):
    if stations_to_plot.empty:
        return folium.Map(location=[22.0, 79.0], zoom_start=5)

    center_lat = float(stations_to_plot["lat"].mean())
    center_lng = float(stations_to_plot["lng"].mean())

    if selected_station_row is not None:
        center_lat = float(selected_station_row["lat"])
        center_lng = float(selected_station_row["lng"])

    m = folium.Map(location=[center_lat, center_lng], zoom_start=6)

    if selected_train_row is not None and isinstance(selected_train_row.get("coordinates"), list):
        route_coords = selected_train_row["coordinates"]
        route_latlng = [[lat, lon] for lon, lat in route_coords]

        folium.PolyLine(
            locations=route_latlng,
            color="blue",
            weight=4,
            opacity=0.8,
            tooltip=f"{selected_train_row['train_number']} - {selected_train_row['train_name']}"
        ).add_to(m)

        if route_coords:
            start_lon, start_lat = route_coords[0]
            end_lon, end_lat = route_coords[-1]

            folium.Marker(
                [start_lat, start_lon],
                tooltip=f"Source: {selected_train_row['from_station_name']} ({selected_train_row['from_station_code']})",
                icon=folium.Icon(color="green", icon="play")
            ).add_to(m)

            folium.Marker(
                [end_lat, end_lon],
                tooltip=f"Destination: {selected_train_row['to_station_name']} ({selected_train_row['to_station_code']})",
                icon=folium.Icon(color="red", icon="stop")
            ).add_to(m)

    if selected_station_row is not None:
        folium.Marker(
            [selected_station_row["lat"], selected_station_row["lng"]],
            tooltip=f"Selected Station: {selected_station_row['name']} ({selected_station_row['code']})",
            icon=folium.Icon(color="purple", icon="info-sign")
        ).add_to(m)

    for _, row in stations_to_plot.iterrows():
        popup = f"""
        <b>{row['name']}</b><br>
        Risk: {row['risk_level']}<br>
        Humidity: {row['humidity']:.0f}%<br>
        Solar: {row['solarradiation']:.0f} W/m²<br>
        Age: {row['track_age_years']} years
        """

        folium.CircleMarker(
            location=[row["lat"], row["lng"]],
            radius=6,
            color=row["marker_color"],
            fill=True,
            fill_color=row["marker_color"],
            fill_opacity=0.85,
            popup=popup
        ).add_to(m)

    return m

# ---------------------------------
# LOAD EVERYTHING
# ---------------------------------
try:
    model = load_model()
    stations_df = load_stations()
    trains_df = load_trains()
    schedules_df = load_schedules()
except Exception as e:
    st.error(f"Startup error: {e}")
    st.stop()

# ---------------------------------
# SIDEBAR
# ---------------------------------
st.sidebar.header("⚙️ Controls")

weather_horizon = st.sidebar.selectbox(
    "Weather Window",
    ["Today", "Next 7 Days", "1 Month", "3 Months", "6 Months"],
    index=1
)

mode = st.sidebar.radio(
    "View Mode",
    [
        "All Stations",
        "Search Station",
        "Search Train",
        "Source to Destination"
    ]
)

selected_station_row = None
selected_train_row = None
stations_to_show = stations_df.copy()

# ---------------------------------
# ALL STATIONS
# ---------------------------------
if mode == "All Stations":
    stations_to_show = stations_df.head(MAX_ALL_STATIONS).copy()

# ---------------------------------
# SEARCH STATION
# ---------------------------------
elif mode == "Search Station":
    station_query = st.sidebar.text_input("Search Station Name or Code")

    if station_query:
        q = normalize_text(station_query)
        stations_to_show = stations_df[
            stations_df["name"].str.lower().str.contains(q, na=False) |
            stations_df["code"].str.lower().str.contains(q, na=False)
        ].copy()
    else:
        stations_to_show = stations_df.head(MAX_SEARCH_STATIONS).copy()

    if not stations_to_show.empty:
        station_choice = st.sidebar.selectbox(
            "Choose Station",
            stations_to_show["display_name"].tolist()
        )
        selected_station_row = stations_to_show.loc[
            stations_to_show["display_name"] == station_choice
        ].iloc[0]

        stations_df["distance_to_selected"] = (
            (stations_df["lat"] - selected_station_row["lat"]) ** 2 +
            (stations_df["lng"] - selected_station_row["lng"]) ** 2
        ) ** 0.5

        stations_to_show = stations_df.sort_values("distance_to_selected").head(MAX_SEARCH_STATIONS).copy()

# ---------------------------------
# SEARCH TRAIN
# ---------------------------------
elif mode == "Search Train":
    train_query = st.sidebar.text_input("Search Train Number or Name")

    train_candidates = trains_df.copy()
    if train_query:
        q = normalize_text(train_query)
        train_candidates = train_candidates[
            train_candidates["train_number"].str.lower().str.contains(q, na=False) |
            train_candidates["train_name"].str.lower().str.contains(q, na=False)
        ].copy()

    train_candidates = train_candidates.head(100)

    if not train_candidates.empty:
        train_display_choice = st.sidebar.selectbox(
            "Choose Train",
            train_candidates["train_display"].tolist()
        )
        selected_train_row = train_candidates.loc[
            train_candidates["train_display"] == train_display_choice
        ].iloc[0]

        stations_to_show = filter_stations_near_route(
            stations_df,
            selected_train_row["coordinates"],
            limit_count=MAX_ROUTE_STATIONS
        )

# ---------------------------------
# SOURCE TO DESTINATION
# ---------------------------------
elif mode == "Source to Destination":
    station_lookup_df = stations_df[["display_name", "name", "code"]].drop_duplicates().sort_values("display_name")

    source_display = st.sidebar.selectbox(
        "From Station",
        station_lookup_df["display_name"].tolist(),
        key="source_station"
    )
    dest_display = st.sidebar.selectbox(
        "To Station",
        station_lookup_df["display_name"].tolist(),
        key="dest_station"
    )

    source_row = station_lookup_df.loc[station_lookup_df["display_name"] == source_display].iloc[0]
    dest_row = station_lookup_df.loc[station_lookup_df["display_name"] == dest_display].iloc[0]

    matching_schedule_trains = find_trains_between_stations(
        schedules_df,
        source_row["code"],
        dest_row["code"]
    )

    if matching_schedule_trains.empty:
        st.sidebar.info("No common train found for this pair.")
        stations_to_show = stations_df[stations_df["code"].isin([source_row["code"], dest_row["code"]])].copy()
    else:
        matching_schedule_trains["display"] = matching_schedule_trains.apply(
            lambda r: f"{r['train_number']} - {r['train_name']}",
            axis=1
        )

        selected_schedule_train = st.sidebar.selectbox(
            "Choose Matching Train",
            matching_schedule_trains["display"].tolist()
        )

        selected_train_number = matching_schedule_trains.loc[
            matching_schedule_trains["display"] == selected_schedule_train, "train_number"
        ].iloc[0]

        route_matches = trains_df[trains_df["train_number"] == selected_train_number].copy()

        if not route_matches.empty:
            selected_train_row = route_matches.iloc[0]
            stations_to_show = filter_stations_near_route(
                stations_df,
                selected_train_row["coordinates"],
                limit_count=MAX_ROUTE_STATIONS
            )
        else:
            stations_to_show = stations_df[stations_df["code"].isin([source_row["code"], dest_row["code"]])].copy()

        source_station_full = stations_df[stations_df["code"] == source_row["code"]]
        if not source_station_full.empty:
            selected_station_row = source_station_full.iloc[0]

# ---------------------------------
# SCORE
# ---------------------------------
if stations_to_show.empty:
    st.warning("No stations matched your search.")
    st.stop()

with st.spinner("Scoring selected stations..."):
    scored_stations = score_stations(stations_to_show, model, weather_horizon)

# ---------------------------------
# METRICS
# ---------------------------------
col1, col2, col3, col4 = st.columns(4)
col1.metric("Stations Displayed", len(scored_stations))
col2.metric("High Risk", int((scored_stations["risk_level"] == "HIGH").sum()))
col3.metric("Medium Risk", int((scored_stations["risk_level"] == "MEDIUM").sum()))
col4.metric("Low Risk", int((scored_stations["risk_level"] == "LOW").sum()))

# ---------------------------------
# TRAIN DETAILS
# ---------------------------------
if selected_train_row is not None:
    st.subheader("🚄 Selected Train")
    train_info = pd.DataFrame([{
        "Train Number": selected_train_row["train_number"],
        "Train Name": selected_train_row["train_name"],
        "Type": selected_train_row["train_type"],
        "From": f"{selected_train_row['from_station_name']} ({selected_train_row['from_station_code']})",
        "To": f"{selected_train_row['to_station_name']} ({selected_train_row['to_station_code']})",
        "Distance": selected_train_row["distance"],
        "Departure": selected_train_row["departure"],
        "Arrival": selected_train_row["arrival"],
    }])
    st.dataframe(train_info, use_container_width=True)

# ---------------------------------
# STATION DETAILS
# ---------------------------------
if selected_station_row is not None:
    st.subheader("📍 Selected Station")
    station_info = pd.DataFrame([{
        "Station Name": selected_station_row["name"],
        "Code": selected_station_row["code"],
        "Track Age (Synthetic)": selected_station_row["track_age_years"],
    }])
    st.dataframe(station_info, use_container_width=True)

# ---------------------------------
# TABLE
# ---------------------------------
st.subheader("📊 Predicted Risk Table")
risk_table = scored_stations[[
    "name", "code",
    "temp_c", "humidity", "solarradiation",
    "track_age_years", "predicted_tmsi", "risk_level", "weather_source"
]].copy()

risk_table.columns = [
    "Station Name", "Code",
    "Temp (°C)", "Humidity", "Solar Radiation",
    "Track Age", "Predicted TMSI", "Risk Level", "Weather Source"
]

st.dataframe(
    risk_table.sort_values("Predicted TMSI", ascending=False),
    use_container_width=True,
    height=300
)

# ---------------------------------
# MAP
# ---------------------------------
st.subheader("🗺️ Route / Station Risk Map")
map_obj = build_map(
    scored_stations,
    selected_station_row=selected_station_row,
    selected_train_row=selected_train_row
)

with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as f:
    map_obj.save(f.name)
    with open(f.name, "r", encoding="utf-8") as html_file:
        st.components.v1.html(
            html_file.read(),
            height=700,
            scrolling=False
        )

# ---------------------------------
# FOOTER
# ---------------------------------
st.markdown("---")
st.caption(
    "Optimized mode: limited stations, cached weather buckets, fast route filtering, and batch XGBoost inference."
)