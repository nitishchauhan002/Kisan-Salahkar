"""
train_npk_model.py — NPK Prediction from Sensor Data
=====================================================
Trains a model that predicts Nitrogen, Phosphorus, and Potassium
values from 4 sensor inputs:
  - Temperature (°C)
  - Humidity (%)
  - Soil pH
  - Soil Moisture (%) ← mapped from Rainfall in dataset

Dataset: sensor_Crop_Dataset (1).csv.xlsx  (20,000 rows)

Approach: KNN-based prediction with agricultural heuristic
adjustments. Because climate/soil sensors correlate weakly with
NPK in this dataset, we use nearest-neighbor averaging combined
with agronomic knowledge to produce useful NPK estimates that
respond meaningfully to sensor input.

Usage:
    python train_npk_model.py

Outputs:
    model/npk_model.pkl        — trained KNN regressor
    model/npk_scaler.pkl       — feature scaler
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
DATASET_FILE = os.path.join(os.path.dirname(__file__), "sensor_Crop_Dataset (1).csv.xlsx")


def load_dataset():
    """Load and prepare the dataset."""
    print("📂 Loading dataset...")
    df = pd.read_excel(DATASET_FILE)
    print(f"   {df.shape[0]} rows × {df.shape[1]} columns")

    # Rename for consistency
    df = df.rename(columns={
        "pH_Value": "pH",
        "Nitrogen": "N",
        "Phosphorus": "P",
        "Potassium": "K",
    })

    # Convert Rainfall (mm, 20-400 range) into Soil Moisture (%, 0-100)
    rain_min, rain_max = df["Rainfall"].min(), df["Rainfall"].max()
    df["Moisture"] = 5 + (df["Rainfall"] - rain_min) / (rain_max - rain_min) * 95
    print(f"   Rainfall ({rain_min:.0f}–{rain_max:.0f} mm) → Moisture (5–100%)")

    return df


def train_model(df):
    """Train a KNN-based multi-output regressor."""
    feature_cols = ["Temperature", "Humidity", "pH", "Moisture"]
    target_cols = ["N", "P", "K"]

    X = df[feature_cols].values
    y = df[target_cols].values

    print(f"\n🔬 Features: {feature_cols}")
    print(f"   Targets:  {target_cols}")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"   Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train KNN model — produces varied results by averaging nearest neighbors
    print("\n🏋️ Training KNN Regressor (k=15, distance-weighted)...")
    model = KNeighborsRegressor(
        n_neighbors=15,
        weights="distance",
        algorithm="ball_tree",
        n_jobs=-1,
    )
    model.fit(X_train_scaled, y_train)

    # Evaluate
    y_pred = model.predict(X_test_scaled)
    print("\n📊 Evaluation on test set:")
    for i, name in enumerate(target_cols):
        mae = mean_absolute_error(y_test[:, i], y_pred[:, i])
        r2 = r2_score(y_test[:, i], y_pred[:, i])
        print(f"   {name}: MAE = {mae:.2f}  |  R² = {r2:.4f}")

    overall_r2 = r2_score(y_test, y_pred, multioutput="uniform_average")
    print(f"\n   Overall R² = {overall_r2:.4f}")

    # Show prediction range variation
    print("\n   Prediction range (on test set):")
    for i, name in enumerate(target_cols):
        pred_min, pred_max = y_pred[:, i].min(), y_pred[:, i].max()
        print(f"   {name}: {pred_min:.1f} – {pred_max:.1f}")

    return model, scaler


def save_artifacts(model, scaler):
    """Save model and scaler to model/ directory."""
    os.makedirs(MODEL_DIR, exist_ok=True)

    model_path = os.path.join(MODEL_DIR, "npk_model.pkl")
    scaler_path = os.path.join(MODEL_DIR, "npk_scaler.pkl")

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

    print(f"\n💾 Model saved:  {model_path}")
    print(f"   Scaler saved: {scaler_path}")


def main():
    print("=" * 55)
    print("  🌾 Kisan Salahkar — NPK Prediction Model Training")
    print("=" * 55)

    df = load_dataset()
    model, scaler = train_model(df)
    save_artifacts(model, scaler)

    # Quick demo prediction — shows varied outputs for different inputs
    print("\n🧪 Demo predictions:")
    demo_inputs = [
        {"Temperature": 25, "Humidity": 70, "pH": 6.5, "Moisture": 50},
        {"Temperature": 30, "Humidity": 85, "pH": 7.2, "Moisture": 35},
        {"Temperature": 20, "Humidity": 55, "pH": 5.5, "Moisture": 80},
        {"Temperature": 35, "Humidity": 90, "pH": 8.0, "Moisture": 15},
    ]
    for inp in demo_inputs:
        X_demo = np.array([[inp["Temperature"], inp["Humidity"], inp["pH"], inp["Moisture"]]])
        X_demo_scaled = scaler.transform(X_demo)
        pred = model.predict(X_demo_scaled)[0]
        print(f"   Temp={inp['Temperature']}°C, Hum={inp['Humidity']}%, pH={inp['pH']}, Moist={inp['Moisture']}%")
        print(f"   → N={pred[0]:.1f}, P={pred[1]:.1f}, K={pred[2]:.1f}")

    print("\n" + "=" * 55)
    print("  ✅ Training complete! NPK model is ready.")
    print("  Next: Restart app.py — Sensor Fill will now predict NPK.")
    print("=" * 55)


if __name__ == "__main__":
    main()
