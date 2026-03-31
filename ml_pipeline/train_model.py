import os
import joblib
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# ----------------------------
# PATHS
# ----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed_data.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
MODEL_PATH = os.path.join(OUTPUT_DIR, "rail_stress_model.pkl")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------
# LOAD DATA
# ----------------------------
print("📥 Loading processed dataset...")
df = pd.read_csv(DATA_PATH)

# ----------------------------
# FEATURES & TARGET
# ----------------------------
X = df.drop("tmsi", axis=1)
y = df["tmsi"]

# ----------------------------
# TRAIN-TEST SPLIT
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------
# MODEL
# ----------------------------
model = xgb.XGBRegressor(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=5,
    random_state=42,
    objective="reg:squarederror"
)

print("🚀 Training XGBoost model...")
model.fit(X_train, y_train)

# ----------------------------
# EVALUATE
# ----------------------------
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)

print(f"✅ Model Training Complete")
print(f"📊 Mean Absolute Error: {mae:.4f}")

# ----------------------------
# SAVE MODEL
# ----------------------------
joblib.dump(model, MODEL_PATH)
print(f"💾 Model saved to: {MODEL_PATH}")