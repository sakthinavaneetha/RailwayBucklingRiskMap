# 🚆 Rail Thermal Stress Detection & Risk Visualization

A machine learning-powered system to **predict and visualize railway track buckling risk** using real-time weather data and geospatial analysis.

---

## 🌟 Overview

Railway tracks are vulnerable to **thermal expansion and buckling**, especially under extreme environmental conditions.
This project builds an intelligent system that:

* Predicts **thermal stress risk** using ML
* Integrates **weather + environmental factors**
* Visualizes risks on an **interactive map**
* Supports **route-based analysis for trains**

---

## ⚙️ Key Features

### 🧠 ML-Based Risk Prediction

* Uses **XGBoost model** to predict Thermal Mechanical Stress Index (TMSI)
* Outputs categorized risk:

  * 🟢 LOW
  * 🟠 MEDIUM
  * 🔴 HIGH

### 🌦️ Weather Integration

* Real-time / forecast-based inputs:

  * Temperature
  * Humidity
  * Solar Radiation
* Smart caching for performance optimization

### 🗺️ Interactive Visualization

* Built with **Streamlit + Folium**
* Dynamic map with:

  * Route plotting
  * Station markers
  * Risk color coding
* Clean popup showing:

  * Risk level
  * Humidity
  * Solar radiation
  * Track age

### 🚄 Smart Route Analysis

* Search trains by:

  * Train name / number
  * Source → Destination
* Displays **only stations near the route** (optimized filtering)

### ⚡ Performance Optimized

* Limited API calls
* Weather caching (coordinate bucketing)
* Batch ML predictions
* Fast rendering (~seconds)

---

## 🧪 Tech Stack

* **Python**
* **Streamlit** (UI Dashboard)
* **Folium** (Map Visualization)
* **XGBoost** (ML Model)
* **Pandas / NumPy** (Data Processing)

---

## 📂 Project Structure

```
railway-stress-dashboard/
│
├── src/
│   ├── streamlit_app.py      # Main dashboard
│   ├── weather_service.py    # Weather data integration
│
├── data/
│   ├── stations.json
│   ├── trains.json
│   ├── schedules.json
│
├── output/
│   ├── rail_stress_model.pkl
│
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run Locally

```bash
# Clone repo
git clone https://github.com/sakthinavanaeetha/railway-stress-dashboard.git
cd railway-stress-dashboard

# Create virtual environment
python -m venv venv
venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Run app
streamlit run src/streamlit_app.py
```

---

## 📊 Model Details

* Model: **XGBoost Regressor**
* Inputs:

  * Temperature (°C)
  * Humidity (%)
  * Solar Radiation (W/m²)
  * Track Age (synthetic)
* Output:

  * **TMSI score (0–1)**
  * Converted to risk categories

---

## 💡 Key Highlights

✔ Physics-informed feature engineering
✔ Synthetic data augmentation (track age)
✔ Real-world inspired problem (rail safety)
✔ Scalable + optimized architecture
✔ Clean UI + interactive analytics

---

## ⭐ Outcome

### 🗺️ Dashboard Overview
![Dashboard](./assets/dashboard.png)

### 🚄 Train Route Visualization
![Route](./assets/route.png)

### ⚠️ Risk Popup
![Popup](./assets/popup.png)

### 📊 Risk Analytics Table
![Table](./assets/table.png)
---
## 📌 Future Improvements

* Live railway API integration
* Real-time alerts system 🚨
* Deployment (Streamlit Cloud / AWS)

---

