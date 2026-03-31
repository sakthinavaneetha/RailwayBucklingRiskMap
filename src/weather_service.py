import os
import requests
import random
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
ENV_PATH = os.path.join(BASE_DIR, ".env")
load_dotenv(dotenv_path=ENV_PATH, override=True)

API_KEY = os.getenv("OPENWEATHER_API_KEY")

def _solar_proxy_from_current(data: dict) -> float:
    clouds = data.get("clouds", {}).get("all", 40)
    solar = 1000 - (clouds * 6)
    return float(max(250, min(1050, solar)))

def _solar_proxy_from_daily(day: dict) -> float:
    clouds = day.get("clouds", 40)
    uvi = day.get("uvi", 6)
    solar = 500 + (uvi * 55) - (clouds * 2.5)
    return float(max(250, min(1100, solar)))

def get_live_weather(lat, lon):
    if not API_KEY:
        return None

    url = (
        "https://api.openweathermap.org/data/2.5/weather"
        f"?lat={lat}&lon={lon}&units=metric&appid={API_KEY}"
    )

    try:
        response = requests.get(url, timeout=8)
        data = response.json()

        if response.status_code != 200 or "main" not in data:
            return None

        return {
            "temp": float(data["main"]["temp"]),
            "humidity": float(data["main"]["humidity"]),
            "solar": _solar_proxy_from_current(data),
            "source": "Live Current"
        }
    except Exception:
        return None

def get_daily_forecast(lat, lon):
    if not API_KEY:
        return None

    url = (
        "https://api.openweathermap.org/data/3.0/onecall"
        f"?lat={lat}&lon={lon}&exclude=minutely,hourly,alerts"
        f"&units=metric&appid={API_KEY}"
    )

    try:
        response = requests.get(url, timeout=10)
        data = response.json()

        if response.status_code != 200 or "daily" not in data:
            return None

        forecast = []
        for day in data["daily"]:
            temp_block = day.get("temp", {})
            forecast.append({
                "temp_day": float(temp_block.get("day", 0.0)),
                "temp_min": float(temp_block.get("min", 0.0)),
                "temp_max": float(temp_block.get("max", 0.0)),
                "humidity": float(day.get("humidity", 0.0)),
                "clouds": float(day.get("clouds", 0.0)),
                "uvi": float(day.get("uvi", 0.0)),
                "solar": _solar_proxy_from_daily(day),
            })
        return forecast
    except Exception:
        return None

def get_weather_features_for_horizon(lat, lon, horizon="Today"):
    if horizon == "Today":
        current = get_live_weather(lat, lon)
        if current:
            return {
                "temp_c": current["temp"],
                "humidity": current["humidity"],
                "solarradiation": current["solar"],
                "weather_source": "Today (Live Current)"
            }

        return {
            "temp_c": float(random.uniform(34, 40)),
            "humidity": float(random.uniform(50, 75)),
            "solarradiation": float(random.uniform(650, 950)),
            "weather_source": "Today (Fallback)"
        }

    daily = get_daily_forecast(lat, lon)

    if not daily:
        base_temp = float(random.uniform(34, 40))
        base_humidity = float(random.uniform(50, 75))
        base_solar = float(random.uniform(650, 950))

        multipliers = {
            "Next 7 Days": 1.03,
            "1 Month": 1.08,
            "3 Months": 1.14,
            "6 Months": 1.20,
        }

        m = multipliers.get(horizon, 1.0)

        return {
            "temp_c": base_temp * m,
            "humidity": base_humidity,
            "solarradiation": min(1100.0, base_solar * m),
            "weather_source": f"{horizon} (Fallback Estimate)"
        }

    forecast_days = daily[:7]

    avg_day_temp = sum(x["temp_day"] for x in forecast_days) / len(forecast_days)
    avg_max_temp = sum(x["temp_max"] for x in forecast_days) / len(forecast_days)
    avg_humidity = sum(x["humidity"] for x in forecast_days) / len(forecast_days)
    avg_solar = sum(x["solar"] for x in forecast_days) / len(forecast_days)

    hot_days = sum(1 for x in forecast_days if x["temp_max"] >= 38.0)
    heat_ratio = hot_days / max(1, len(forecast_days))

    if horizon == "Next 7 Days":
        effective_temp = (0.6 * avg_day_temp) + (0.4 * avg_max_temp)
        return {
            "temp_c": float(effective_temp),
            "humidity": float(avg_humidity),
            "solarradiation": float(avg_solar),
            "weather_source": "Next 7 Days (Live Forecast Aggregate)"
        }

    persistence_weights = {
        "1 Month": 0.10,
        "3 Months": 0.18,
        "6 Months": 0.26,
    }

    p = persistence_weights.get(horizon, 0.0)
    persistence_multiplier = 1.0 + (p * heat_ratio)

    effective_temp = ((0.55 * avg_day_temp) + (0.45 * avg_max_temp)) * persistence_multiplier
    effective_solar = avg_solar * (1.0 + (0.5 * p * heat_ratio))

    return {
        "temp_c": float(effective_temp),
        "humidity": float(avg_humidity),
        "solarradiation": float(min(1100.0, effective_solar)),
        "weather_source": f"{horizon} (Persistence Estimate from 7-Day Forecast)"
    }