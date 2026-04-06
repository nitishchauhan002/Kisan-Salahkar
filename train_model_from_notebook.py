"""
Train crop recommendation model using notebook-style dataset flow and
export artefacts in the format expected by app.py.

Inputs:
- Crop_recommendation.csv (required)
- apy.csv (optional, used to derive state crop mapping)

Outputs (in model/):
- crop_model.pkl
- scaler.pkl
- label_encoder.pkl
- state_crops.json
- state_district_crops.json
- crop_diseases.json
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import LabelEncoder, StandardScaler


BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "model")


# Keep names aligned with app display and tips dictionaries.
CROP_NAME_MAP = {
    "rice": "Rice",
    "wheat": "Wheat",
    "maize": "Maize",
    "cotton": "Cotton",
    "jute": "Jute",
    "coffee": "Coffee",
    "pigeonpeas": "Chickpea",
    "mothbeans": "Bajra",
    "mungbean": "Lentil",
    "blackgram": "Lentil",
    "lentil": "Lentil",
}


DEFAULT_CROP_DISEASES = {
    "Rice": ["Blast", "Brown Spot", "Bacterial Leaf Blight"],
    "Wheat": ["Rust", "Loose Smut", "Karnal Bunt"],
    "Maize": ["Leaf Blight", "Stalk Rot", "Downy Mildew"],
    "Cotton": ["Bollworm", "Whitefly", "Bacterial Blight"],
    "Jute": ["Stem Rot", "Root Rot", "Anthracnose"],
    "Coffee": ["Coffee Rust", "Berry Disease", "Black Rot"],
    "Chickpea": ["Fusarium Wilt", "Ascochyta Blight", "Pod Borer"],
    "Bajra": ["Downy Mildew", "Ergot", "Smut"],
    "Lentil": ["Rust", "Wilt", "Ascochyta Blight"],
}


def load_crop_dataset(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    needed = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall", "label"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {csv_path}: {missing}")

    # Keep notebook crop scope, then map labels to app-friendly names.
    indian_crops = set(CROP_NAME_MAP.keys())
    df = df[df["label"].isin(indian_crops)].copy()
    df["crop_name"] = df["label"].map(CROP_NAME_MAP)

    # Ensure app uses 'pH' key casing.
    df = df.rename(columns={"ph": "pH"})
    return df


def build_state_crop_map(apy_csv_path: str, allowed_crops: set[str]) -> dict:
    if not os.path.exists(apy_csv_path):
        print("[INFO] apy.csv not found. Using fallback state crop mapping.")
        return {
            "Punjab": ["Wheat", "Rice", "Cotton", "Maize"],
            "Haryana": ["Wheat", "Rice", "Cotton", "Maize"],
            "Uttar Pradesh": ["Wheat", "Rice", "Lentil", "Chickpea"],
            "Madhya Pradesh": ["Wheat", "Chickpea", "Lentil", "Maize"],
            "Rajasthan": ["Wheat", "Bajra", "Chickpea", "Cotton"],
            "Maharashtra": ["Cotton", "Maize", "Lentil"],
            "Gujarat": ["Cotton", "Wheat", "Bajra"],
            "Karnataka": ["Maize", "Coffee", "Rice"],
            "Tamil Nadu": ["Rice", "Cotton", "Maize"],
            "West Bengal": ["Rice", "Jute", "Wheat"],
            "Bihar": ["Rice", "Wheat", "Maize", "Lentil"],
            "Assam": ["Rice", "Jute"],
        }

    df = pd.read_csv(apy_csv_path)
    df.columns = df.columns.str.strip().str.lower()
    rename_map = {
        "state_name": "state",
        "district_name": "district",
        "crop": "crop",
        "production_": "production",
    }
    df = df.rename(columns=rename_map)
    for col in ["state", "crop", "production"]:
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}' in {apy_csv_path}")

    df["production"] = pd.to_numeric(df["production"], errors="coerce")
    df = df.dropna(subset=["production"]).copy()
    df["state"] = df["state"].astype(str).str.strip()
    df["crop"] = df["crop"].astype(str).str.lower().str.strip()
    df["crop_name"] = df["crop"].map(CROP_NAME_MAP)
    df = df.dropna(subset=["crop_name"])

    grouped = (
        df.groupby(["state", "crop_name"], as_index=False)["production"]
        .sum()
        .sort_values(["state", "production"], ascending=[True, False])
    )

    state_map = {}
    for state, grp in grouped.groupby("state"):
        top = [c for c in grp["crop_name"].tolist() if c in allowed_crops]
        if top:
            state_map[state] = list(dict.fromkeys(top[:7]))

    return state_map


def build_state_district_crop_map(apy_csv_path: str, allowed_crops: set[str]) -> dict:
    if not os.path.exists(apy_csv_path):
        return {
            "Madhya Pradesh": {
                "Bhopal": ["Wheat", "Chickpea", "Lentil", "Maize"],
                "Indore": ["Wheat", "Maize", "Lentil"],
            },
            "Uttar Pradesh": {
                "Kannauj": ["Wheat", "Rice", "Lentil"],
                "Lucknow": ["Wheat", "Rice", "Chickpea"],
            },
            "Punjab": {
                "Ludhiana": ["Wheat", "Rice", "Cotton"],
                "Amritsar": ["Wheat", "Rice", "Maize"],
            },
            "Rajasthan": {
                "Jaipur": ["Wheat", "Bajra", "Chickpea"],
                "Kota": ["Wheat", "Chickpea", "Cotton"],
            },
        }

    df = pd.read_csv(apy_csv_path)
    df.columns = df.columns.str.strip().str.lower()
    rename_map = {
        "state_name": "state",
        "district_name": "district",
        "crop": "crop",
        "production_": "production",
    }
    df = df.rename(columns=rename_map)
    for col in ["state", "district", "crop", "production"]:
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}' in {apy_csv_path}")

    df["production"] = pd.to_numeric(df["production"], errors="coerce")
    df = df.dropna(subset=["production"]).copy()
    df["state"] = df["state"].astype(str).str.strip()
    df["district"] = df["district"].astype(str).str.strip()
    df["crop"] = df["crop"].astype(str).str.lower().str.strip()
    df["crop_name"] = df["crop"].map(CROP_NAME_MAP)
    df = df.dropna(subset=["crop_name"])

    grouped = (
        df.groupby(["state", "district", "crop_name"], as_index=False)["production"]
        .sum()
        .sort_values(["state", "district", "production"], ascending=[True, True, False])
    )

    out = {}
    for (state, district), grp in grouped.groupby(["state", "district"]):
        top = [c for c in grp["crop_name"].tolist() if c in allowed_crops]
        if not top:
            continue
        out.setdefault(state, {})[district] = list(dict.fromkeys(top[:7]))
    return out


def train_and_export():
    crop_csv = os.path.join(BASE_DIR, "Crop_recommendation.csv")
    apy_csv = os.path.join(BASE_DIR, "apy.csv")

    if not os.path.exists(crop_csv):
        candidates = [
            os.path.join(BASE_DIR, "crop_model.pkl"),
            os.path.join(MODEL_DIR, "crop_model.pkl"),
        ]
        fallback_model_path = None
        model = None
        for cand in candidates:
            if not os.path.exists(cand):
                continue
            try:
                model = joblib.load(cand)
                fallback_model_path = cand
                break
            except Exception as exc:
                print(f"[WARN] Could not load {cand}: {exc}")

        if model is None:
            raise FileNotFoundError(
                "Crop_recommendation.csv not found and no readable fallback model found. "
                "Provide dataset or valid crop_model.pkl."
            )

        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(model, os.path.join(MODEL_DIR, "crop_model.pkl"))

        # Notebook model was trained on raw features, so keep transform as identity.
        scaler = FunctionTransformer(np.asarray, validate=False)
        joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))

        label_encoder = LabelEncoder()
        mapped_classes = [
            CROP_NAME_MAP.get(str(c).lower().strip(), str(c).title())
            for c in list(getattr(model, "classes_", []))
        ]
        label_encoder.fit(mapped_classes)
        joblib.dump(label_encoder, os.path.join(MODEL_DIR, "label_encoder.pkl"))

        allowed_crops = set(label_encoder.classes_.tolist())
        state_crops = build_state_crop_map(apy_csv, allowed_crops)
        state_district_crops = build_state_district_crop_map(apy_csv, allowed_crops)
        with open(os.path.join(MODEL_DIR, "state_crops.json"), "w", encoding="utf-8") as f:
            json.dump(state_crops, f, ensure_ascii=False, indent=2)
        with open(os.path.join(MODEL_DIR, "state_district_crops.json"), "w", encoding="utf-8") as f:
            json.dump(state_district_crops, f, ensure_ascii=False, indent=2)

        crop_diseases = {c: DEFAULT_CROP_DISEASES.get(c, []) for c in sorted(allowed_crops)}
        with open(os.path.join(MODEL_DIR, "crop_diseases.json"), "w", encoding="utf-8") as f:
            json.dump(crop_diseases, f, ensure_ascii=False, indent=2)

        print(f"[OK] Fallback mode used (from {fallback_model_path})")
        print("[OK] Saved: model/crop_model.pkl")
        print("[OK] Saved: model/scaler.pkl (identity transform)")
        print("[OK] Saved: model/label_encoder.pkl")
        print("[OK] Saved: model/state_crops.json")
        print("[OK] Saved: model/state_district_crops.json")
        print("[OK] Saved: model/crop_diseases.json")
        print(f"[OK] Classes: {list(label_encoder.classes_)}")
        return

    df = load_crop_dataset(crop_csv)
    x_cols = ["N", "P", "K", "temperature", "humidity", "pH", "rainfall"]
    X = df[x_cols]
    y = df["crop_name"]

    label_encoder = LabelEncoder()
    y_enc = label_encoder.fit_transform(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(MODEL_DIR, "crop_model.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
    joblib.dump(label_encoder, os.path.join(MODEL_DIR, "label_encoder.pkl"))

    allowed_crops = set(label_encoder.classes_.tolist())
    state_crops = build_state_crop_map(apy_csv, allowed_crops)
    state_district_crops = build_state_district_crop_map(apy_csv, allowed_crops)
    with open(os.path.join(MODEL_DIR, "state_crops.json"), "w", encoding="utf-8") as f:
        json.dump(state_crops, f, ensure_ascii=False, indent=2)
    with open(os.path.join(MODEL_DIR, "state_district_crops.json"), "w", encoding="utf-8") as f:
        json.dump(state_district_crops, f, ensure_ascii=False, indent=2)

    crop_diseases = {c: DEFAULT_CROP_DISEASES.get(c, []) for c in sorted(allowed_crops)}
    with open(os.path.join(MODEL_DIR, "crop_diseases.json"), "w", encoding="utf-8") as f:
        json.dump(crop_diseases, f, ensure_ascii=False, indent=2)

    print(f"[OK] Accuracy: {acc:.4f}")
    print("[OK] Saved: model/crop_model.pkl")
    print("[OK] Saved: model/scaler.pkl")
    print("[OK] Saved: model/label_encoder.pkl")
    print("[OK] Saved: model/state_crops.json")
    print("[OK] Saved: model/state_district_crops.json")
    print("[OK] Saved: model/crop_diseases.json")
    print(f"[OK] Classes: {list(label_encoder.classes_)}")


if __name__ == "__main__":
    train_and_export()
