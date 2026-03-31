import os
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

# ----------------------------
# PATHS
# ----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed_data.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
PLOT_PATH = os.path.join(OUTPUT_DIR, "model_comparison_mae.png")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------
# LOAD DATA
# ----------------------------
print("📥 Loading processed dataset...")
df = pd.read_csv(DATA_PATH)

X = df.drop("tmsi", axis=1)
y = df["tmsi"]

print(f"✅ Dataset loaded with {df.shape[0]} samples and {df.shape[1]} columns")

# ----------------------------
# SPLIT
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("🔀 Train-test split completed")

# ----------------------------
# SCALING FOR MLP
# ----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ----------------------------
# MODELS
# ----------------------------
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost": xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=5,
        random_state=42,
        objective="reg:squarederror"
    ),
    "Deep Learning (MLP)": MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        max_iter=500,
        random_state=42
    )
}

# ----------------------------
# TRAIN & EVALUATE
# ----------------------------
results = []

print("\n🔍 Comparing Models...\n")

for name, model in models.items():
    print(f"➡️ Training {name}...")

    if "Deep Learning" in name:
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)

    results.append({
        "Model": name,
        "MAE": round(mae, 4)
    })

    print(f"   ✅ MAE: {mae:.4f}")

results_df = pd.DataFrame(results).sort_values(by="MAE")

print("\n📊 Model Comparison Results:\n")
print(results_df)

# ----------------------------
# PLOT
# ----------------------------
plt.figure(figsize=(10, 5))
bars = plt.bar(results_df["Model"], results_df["MAE"], edgecolor="black")

plt.xlabel("Prediction Models", fontsize=11)
plt.ylabel("Mean Absolute Error (MAE)", fontsize=11)
plt.title(
    "Comparison of Machine Learning Models for Rail Thermal Buckling Prediction",
    fontsize=13,
    fontweight="bold"
)
plt.xticks(rotation=25, ha="right")
plt.grid(axis="y", linestyle="--", alpha=0.6)

for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height + 0.0002,
        f"{height:.4f}",
        ha="center",
        va="bottom",
        fontsize=9
    )

plt.tight_layout()
plt.savefig(PLOT_PATH, dpi=300, bbox_inches="tight")
plt.show()

print(f"📁 Plot saved to: {PLOT_PATH}")