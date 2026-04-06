"""
train_model.py – Enhanced crop-recommendation model with state-specific data.
Uses realistic agricultural parameters based on ICAR & Indian gov data.
Trains RandomForest + saves model, scaler, label-encoder AND disease model.
"""

import os, json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import joblib

# ── State-specific crop suitability ──────────────────────────────────────────
STATE_CROPS = {
    "Punjab":           ["Wheat", "Rice", "Cotton", "Sugarcane", "Maize", "Potato", "Mustard"],
    "Haryana":          ["Wheat", "Rice", "Cotton", "Mustard", "Sugarcane", "Bajra"],
    "Uttar Pradesh":    ["Wheat", "Rice", "Sugarcane", "Potato", "Mustard", "Lentil", "Chickpea", "Onion"],
    "Madhya Pradesh":   ["Soybean", "Wheat", "Chickpea", "Cotton", "Lentil", "Maize"],
    "Maharashtra":      ["Cotton", "Sugarcane", "Soybean", "Onion", "Groundnut", "Rice", "Banana", "Mango"],
    "Rajasthan":        ["Mustard", "Wheat", "Bajra", "Groundnut", "Chickpea", "Cumin"],
    "Gujarat":          ["Cotton", "Groundnut", "Wheat", "Banana", "Onion", "Sugarcane", "Cumin"],
    "Karnataka":        ["Rice", "Coffee", "Coconut", "Sugarcane", "Maize", "Chili", "Mango", "Tomato"],
    "Tamil Nadu":       ["Rice", "Coconut", "Banana", "Sugarcane", "Tea", "Groundnut", "Mango"],
    "Kerala":           ["Coconut", "Rice", "Coffee", "Tea", "Banana", "Rubber"],
    "West Bengal":      ["Rice", "Jute", "Potato", "Tea", "Mustard", "Banana"],
    "Andhra Pradesh":   ["Rice", "Chili", "Cotton", "Groundnut", "Sugarcane", "Onion", "Mango", "Tomato"],
    "Telangana":        ["Rice", "Cotton", "Maize", "Chili", "Soybean", "Groundnut"],
    "Bihar":            ["Rice", "Wheat", "Maize", "Lentil", "Potato", "Sugarcane", "Banana"],
    "Odisha":           ["Rice", "Groundnut", "Sugarcane", "Jute", "Coconut"],
    "Assam":            ["Tea", "Rice", "Jute", "Potato", "Banana"],
    "Jharkhand":        ["Rice", "Wheat", "Maize", "Lentil", "Potato"],
    "Chhattisgarh":     ["Rice", "Maize", "Soybean", "Lentil", "Chickpea"],
    "Uttarakhand":      ["Rice", "Wheat", "Soybean", "Lentil", "Potato"],
    "Himachal Pradesh": ["Wheat", "Maize", "Rice", "Potato", "Tea"],
}

# ── Realistic parameter ranges per crop (based on ICAR data) ─────────────────
CROP_PARAMS = {
    "Rice":       {"N": (60, 120), "P": (35, 60),  "K": (35, 50),  "temp": (20, 27), "humidity": (80, 95), "pH": (5.0, 7.0), "rainfall": (150, 300)},
    "Wheat":      {"N": (80, 120), "P": (40, 60),  "K": (45, 65),  "temp": (15, 25), "humidity": (50, 70), "pH": (5.5, 7.5), "rainfall": (50, 100)},
    "Maize":      {"N": (60, 120), "P": (35, 65),  "K": (30, 55),  "temp": (18, 27), "humidity": (55, 75), "pH": (5.5, 7.0), "rainfall": (60, 110)},
    "Cotton":     {"N": (100, 140),"P": (45, 65),  "K": (20, 50),  "temp": (22, 30), "humidity": (60, 80), "pH": (6.0, 8.0), "rainfall": (60, 110)},
    "Sugarcane":  {"N": (80, 130), "P": (25, 55),  "K": (30, 60),  "temp": (22, 32), "humidity": (75, 90), "pH": (5.5, 7.5), "rainfall": (150, 250)},
    "Jute":       {"N": (60, 100), "P": (35, 55),  "K": (35, 55),  "temp": (24, 37), "humidity": (70, 90), "pH": (5.5, 7.5), "rainfall": (150, 250)},
    "Coffee":     {"N": (80, 120), "P": (15, 35),  "K": (25, 45),  "temp": (15, 28), "humidity": (50, 70), "pH": (5.0, 6.5), "rainfall": (100, 200)},
    "Coconut":    {"N": (10, 40),  "P": (5, 25),   "K": (25, 55),  "temp": (25, 35), "humidity": (80, 95), "pH": (5.0, 8.0), "rainfall": (130, 250)},
    "Banana":     {"N": (80, 120), "P": (70, 100), "K": (45, 65),  "temp": (25, 35), "humidity": (75, 90), "pH": (5.5, 7.0), "rainfall": (100, 175)},
    "Mango":      {"N": (10, 40),  "P": (15, 40),  "K": (25, 50),  "temp": (27, 35), "humidity": (45, 65), "pH": (5.5, 7.5), "rainfall": (50, 100)},
    "Groundnut":  {"N": (10, 40),  "P": (35, 65),  "K": (15, 35),  "temp": (25, 35), "humidity": (40, 60), "pH": (5.5, 7.0), "rainfall": (40, 80)},
    "Soybean":    {"N": (10, 40),  "P": (55, 85),  "K": (15, 35),  "temp": (20, 30), "humidity": (60, 80), "pH": (5.5, 7.0), "rainfall": (50, 100)},
    "Mustard":    {"N": (50, 90),  "P": (35, 55),  "K": (35, 60),  "temp": (10, 25), "humidity": (40, 65), "pH": (5.5, 7.5), "rainfall": (30, 60)},
    "Lentil":     {"N": (10, 30),  "P": (55, 80),  "K": (15, 35),  "temp": (15, 25), "humidity": (30, 60), "pH": (5.5, 7.5), "rainfall": (30, 60)},
    "Chickpea":   {"N": (20, 50),  "P": (55, 80),  "K": (70, 90),  "temp": (15, 25), "humidity": (15, 40), "pH": (6.0, 8.0), "rainfall": (50, 100)},
    "Potato":     {"N": (50, 100), "P": (55, 80),  "K": (45, 75),  "temp": (15, 25), "humidity": (70, 85), "pH": (4.5, 6.5), "rainfall": (40, 80)},
    "Tomato":     {"N": (70, 130), "P": (55, 85),  "K": (45, 75),  "temp": (18, 30), "humidity": (70, 85), "pH": (5.5, 7.0), "rainfall": (50, 100)},
    "Onion":      {"N": (50, 90),  "P": (55, 80),  "K": (50, 75),  "temp": (15, 30), "humidity": (60, 80), "pH": (5.5, 7.0), "rainfall": (40, 80)},
    "Chili":      {"N": (90, 140), "P": (55, 80),  "K": (45, 70),  "temp": (20, 30), "humidity": (60, 80), "pH": (5.5, 7.5), "rainfall": (50, 100)},
    "Tea":        {"N": (80, 130), "P": (15, 35),  "K": (15, 35),  "temp": (15, 25), "humidity": (75, 95), "pH": (4.5, 6.0), "rainfall": (150, 300)},
    "Bajra":      {"N": (40, 80),  "P": (20, 40),  "K": (15, 35),  "temp": (25, 35), "humidity": (30, 55), "pH": (6.5, 8.0), "rainfall": (25, 60)},
    "Cumin":      {"N": (20, 50),  "P": (20, 45),  "K": (10, 30),  "temp": (20, 30), "humidity": (30, 55), "pH": (6.5, 8.5), "rainfall": (20, 50)},
    "Rubber":     {"N": (10, 30),  "P": (10, 30),  "K": (10, 30),  "temp": (25, 34), "humidity": (75, 95), "pH": (4.5, 6.0), "rainfall": (200, 350)},
}

# ── Common crop diseases (for disease info) ──────────────────────────────────
CROP_DISEASES = {
    "Rice": ["Blast", "Brown Spot", "Bacterial Leaf Blight", "Sheath Rot"],
    "Wheat": ["Rust (Yellow/Brown/Black)", "Loose Smut", "Karnal Bunt", "Powdery Mildew"],
    "Maize": ["Northern Leaf Blight", "Stalk Rot", "Downy Mildew"],
    "Cotton": ["Bollworm", "Whitefly", "Bacterial Blight", "Fusarium Wilt"],
    "Sugarcane": ["Red Rot", "Smut", "Grassy Shoot", "Top Borer"],
    "Potato": ["Late Blight", "Early Blight", "Black Scurf", "Common Scab"],
    "Tomato": ["Early Blight", "Late Blight", "Leaf Curl Virus", "Bacterial Wilt"],
    "Onion": ["Purple Blotch", "Stemphylium Blight", "Thrips", "Basal Rot"],
    "Chili": ["Anthracnose", "Leaf Curl", "Root Rot", "Thrips"],
    "Banana": ["Panama Wilt", "Sigatoka Leaf Spot", "Bunchy Top Virus"],
    "Mango": ["Anthracnose", "Powdery Mildew", "Mango Malformation"],
    "Groundnut": ["Tikka Disease", "Collar Rot", "Stem Rot"],
    "Soybean": ["Rust", "Yellow Mosaic Virus", "Charcoal Rot"],
    "Coffee": ["Coffee Rust", "Berry Disease", "Black Rot"],
    "Coconut": ["Bud Rot", "Root Wilt", "Leaf Rot"],
    "Tea": ["Blister Blight", "Red Rust", "Grey Blight"],
    "Mustard": ["Alternaria Blight", "White Rust", "Downy Mildew"],
    "Lentil": ["Rust", "Wilt", "Ascochyta Blight"],
    "Chickpea": ["Fusarium Wilt", "Ascochyta Blight", "Pod Borer"],
    "Jute": ["Stem Rot", "Root Rot", "Anthracnose"],
    "Bajra": ["Downy Mildew", "Ergot", "Smut"],
    "Cumin": ["Wilt", "Blight", "Powdery Mildew"],
    "Rubber": ["Abnormal Leaf Fall", "Powdery Mildew", "Pink Disease"],
}


def generate_data(n_samples_per_crop: int = 300, seed: int = 42) -> pd.DataFrame:
    """Return a DataFrame with realistic crop samples including noise."""
    rng = np.random.default_rng(seed)
    rows = []
    for crop, params in CROP_PARAMS.items():
        for _ in range(n_samples_per_crop):
            row = {
                "N":           rng.uniform(*params["N"]) + rng.normal(0, 3),
                "P":           rng.uniform(*params["P"]) + rng.normal(0, 2),
                "K":           rng.uniform(*params["K"]) + rng.normal(0, 2),
                "temperature": rng.uniform(*params["temp"]) + rng.normal(0, 1),
                "humidity":    np.clip(rng.uniform(*params["humidity"]) + rng.normal(0, 2), 5, 100),
                "pH":          round(np.clip(rng.uniform(*params["pH"]) + rng.normal(0, 0.2), 3.0, 10.0), 2),
                "rainfall":    max(0, rng.uniform(*params["rainfall"]) + rng.normal(0, 10)),
                "crop":        crop,
            }
            rows.append(row)
    return pd.DataFrame(rows)


def train() -> None:
    print("🌾 Generating enhanced crop data with 23 crops …")
    df = generate_data()
    print(f"   Samples: {len(df)}  |  Crops: {df['crop'].nunique()}")

    X = df[["N", "P", "K", "temperature", "humidity", "pH", "rainfall"]]
    y = df["crop"]

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    print("🤖 Training Random Forest (300 estimators) …")
    model = RandomForestClassifier(
        n_estimators=300, max_depth=25, min_samples_split=3,
        random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n✅ Accuracy: {acc:.4f}\n")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # Cross-validation
    cv_scores = cross_val_score(model, X_scaled, y_enc, cv=5, n_jobs=-1)
    print(f"📊 Cross-validation: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    os.makedirs("model", exist_ok=True)
    joblib.dump(model, "model/crop_model.pkl")
    joblib.dump(scaler, "model/scaler.pkl")
    joblib.dump(le, "model/label_encoder.pkl")

    # Save state-crop mapping and disease info
    with open("model/state_crops.json", "w", encoding="utf-8") as f:
        json.dump(STATE_CROPS, f)
    with open("model/crop_diseases.json", "w", encoding="utf-8") as f:
        json.dump(CROP_DISEASES, f)

    print("💾 Model artefacts + state data + disease data saved to model/")


if __name__ == "__main__":
    train()
