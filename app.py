"""
app.py – Kisan Salahkar: Full-featured Agriculture Advisory System
Features: Auth, Crop Prediction, Weather, Soil Testing, Govt Schemes,
          Mandi Prices, Disease Detection, Community Forum, SMS Alerts,
          Multi-state support, Voice-ready APIs.
"""

import os, hashlib, sqlite3, json, time, base64, io, uuid
from datetime import datetime
import numpy as np
import joblib
import requests as http_requests
import bcrypt
import bleach
from PIL import Image
from flask import (
    Flask, render_template, request, jsonify,
    session, redirect, url_for, send_from_directory,
)

# ── Flask setup ──────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "kisan-salahkar-secret-2026")

# ── Load ML artefacts ────────────────────────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")

ACTIVE_CROP_MODEL_FILES = {
    "model": "crop_model.pkl",
    "scaler": "scaler.pkl",
    "label_encoder": "label_encoder.pkl",
    "features": "features.pkl",
}


def _resolve_artifact_path(file_name: str) -> str:
    if os.path.isabs(file_name):
        return file_name
    return os.path.join(MODEL_DIR, file_name)


def load_crop_artifacts(model_file=None, scaler_file=None, label_encoder_file=None):
    """Load crop recommendation artefacts and make them active globally."""
    global model, scaler, label_encoder, feature_columns, ACTIVE_CROP_MODEL_FILES

    selected = {
        "model": model_file or os.environ.get("CROP_MODEL_FILE") or "crop_model.pkl",
        "scaler": scaler_file or os.environ.get("CROP_SCALER_FILE") or "scaler.pkl",
        "label_encoder": label_encoder_file or os.environ.get("CROP_LABEL_ENCODER_FILE") or "label_encoder.pkl",
        "features": os.environ.get("CROP_FEATURES_FILE") or "features.pkl",
    }

    paths = {k: _resolve_artifact_path(v) for k, v in selected.items()}
    missing = [f"model={paths['model']}" for _ in [0] if not os.path.exists(paths["model"])]
    if missing:
        raise FileNotFoundError("Missing crop artefacts: " + ", ".join(missing))

    model = joblib.load(paths["model"])
    scaler = joblib.load(paths["scaler"]) if os.path.exists(paths["scaler"]) else None
    label_encoder = joblib.load(paths["label_encoder"]) if os.path.exists(paths["label_encoder"]) else None

    default_features = ["N", "P", "K", "temperature", "humidity", "pH", "rainfall"]
    feature_columns = default_features
    if os.path.exists(paths["features"]):
        try:
            loaded_features = joblib.load(paths["features"])
            loaded_features = [str(x) for x in list(loaded_features)]
            if loaded_features:
                feature_columns = loaded_features
        except Exception as exc:
            print(f"⚠️  features.pkl could not be loaded: {exc}")

    ACTIVE_CROP_MODEL_FILES = selected
    return {
        "model": paths["model"],
        "scaler": paths["scaler"] if os.path.exists(paths["scaler"]) else "",
        "label_encoder": paths["label_encoder"] if os.path.exists(paths["label_encoder"]) else "",
        "features": paths["features"] if os.path.exists(paths["features"]) else "",
    }


def load_crop_metadata():
    global STATE_CROPS, CROP_DISEASES, STATE_DISTRICT_CROPS

    with open(os.path.join(MODEL_DIR, "state_crops.json"), encoding="utf-8") as f:
        STATE_CROPS = json.load(f)
    with open(os.path.join(MODEL_DIR, "crop_diseases.json"), encoding="utf-8") as f:
        CROP_DISEASES = json.load(f)

    STATE_DISTRICT_CROPS = {}
    _state_district_path = os.path.join(MODEL_DIR, "state_district_crops.json")
    if os.path.exists(_state_district_path):
        try:
            with open(_state_district_path, encoding="utf-8") as f:
                STATE_DISTRICT_CROPS = json.load(f)
        except Exception as e:
            print(f"⚠️  state_district_crops.json not loaded: {e}")


load_crop_artifacts()
load_crop_metadata()

# ── Load Disease CNN model (if trained) ──────────────────────────────────────
DISEASE_MODEL = None
DISEASE_LABELS = None
DISEASE_IMG_SIZE = 224
DISEASE_USE_TFLITE = False

_disease_model_path = os.path.join(MODEL_DIR, "disease_model.keras")
_disease_tflite_path = os.path.join(MODEL_DIR, "disease_model.tflite")
_disease_labels_path = os.path.join(MODEL_DIR, "disease_labels.json")

if os.path.exists(_disease_labels_path):
    with open(_disease_labels_path, encoding="utf-8") as f:
        DISEASE_LABELS = json.load(f)

    # Try full Keras model first
    if os.path.exists(_disease_model_path):
        try:
            import tensorflow as tf
            DISEASE_MODEL = tf.keras.models.load_model(_disease_model_path)
            print(f"✅ Disease CNN loaded — {len(DISEASE_LABELS)} classes")
        except Exception as e:
            print(f"⚠️  Disease CNN not loaded: {e}")
            DISEASE_MODEL = None

    # Fallback to TFLite model
    if DISEASE_MODEL is None and os.path.exists(_disease_tflite_path):
        try:
            import tensorflow as tf
            DISEASE_MODEL = tf.lite.Interpreter(model_path=_disease_tflite_path)
            DISEASE_MODEL.allocate_tensors()
            DISEASE_USE_TFLITE = True
            print(f"✅ Disease TFLite model loaded — {len(DISEASE_LABELS)} classes")
        except Exception as e:
            print(f"⚠️  Disease TFLite not loaded: {e}")
            DISEASE_MODEL = None

    if DISEASE_MODEL is None:
        print("ℹ️  Disease CNN not found. Run: python train_disease_model.py")
        print("   Text-based disease detection will be used as fallback.")
else:
    print("ℹ️  Disease CNN not found. Run: python train_disease_model.py")
    print("   Text-based disease detection will be used as fallback.")

# ── Load NPK Prediction model (if trained) ──────────────────────────────────
NPK_MODEL = None
NPK_SCALER = None
_npk_model_path = os.path.join(MODEL_DIR, "npk_model.pkl")
_npk_scaler_path = os.path.join(MODEL_DIR, "npk_scaler.pkl")

if os.path.exists(_npk_model_path) and os.path.exists(_npk_scaler_path):
    try:
        NPK_MODEL = joblib.load(_npk_model_path)
        NPK_SCALER = joblib.load(_npk_scaler_path)
        print("✅ NPK prediction model loaded (sensor → N, P, K)")
    except Exception as e:
        print(f"⚠️  NPK model not loaded: {e}")
        NPK_MODEL = None
else:
    print("ℹ️  NPK model not found. Run: python train_npk_model.py")

# ── Load State Rainfall Data (seasonal averages) ────────────────────────────
STATE_RAINFALL = {}
_rainfall_path = os.path.join(MODEL_DIR, "state_rainfall.json")
if os.path.exists(_rainfall_path):
    with open(_rainfall_path, encoding="utf-8") as f:
        STATE_RAINFALL = json.load(f)
    STATE_RAINFALL.pop("_comment", None)
    print(f"✅ State rainfall data loaded — {len(STATE_RAINFALL)} states")
else:
    print("ℹ️  state_rainfall.json not found — rainfall auto-fill disabled")

# ── Crop growing tips ────────────────────────────────────────────────────────

# ── Season-to-crop mapping (Kharif=monsoon, Rabi=winter, Zaid=summer) ────────
CROP_SEASON = {
    "kharif": ["Rice", "Maize", "Cotton", "Jute", "Groundnut", "Soybean",
               "Bajra", "Sugarcane", "Chili", "Banana"],
    "rabi":   ["Wheat", "Mustard", "Lentil", "Chickpea", "Potato", "Onion",
               "Cumin", "Maize"],
    "zaid":   ["Cucumber", "Watermelon", "Muskmelon", "Banana", "Sugarcane",
               "Maize", "Groundnut", "Tomato", "Onion", "Chili"],
}
# Reverse lookup: crop → set of seasons
CROP_TO_SEASONS = {}
for _season, _crops in CROP_SEASON.items():
    for _c in _crops:
        CROP_TO_SEASONS.setdefault(_c, set()).add(_season)
# Perennial crops (grow year-round) – always allowed
PERENNIAL_CROPS = {"Coffee", "Tea", "Coconut", "Mango", "Rubber", "Banana", "Sugarcane"}

CROP_TIPS = {
    "Rice":      "Requires standing water. Transplant seedlings 20–30 days old. Apply nitrogen in 3 splits.",
    "Wheat":     "Best sown in November. Irrigate at crown-root, tillering, flowering & grain-filling stages.",
    "Maize":     "Needs well-drained soil. Apply full P & K at sowing; split N in 2–3 doses.",
    "Cotton":    "Sow after monsoon onset. Use Bt-cotton hybrids for bollworm resistance. Pick in 3–4 rounds.",
    "Sugarcane": "Plant setts in spring. Earthing-up twice. Harvest at 10–12 months for peak sugar recovery.",
    "Jute":      "Broadcast seeds in March–April. Harvest at flower-bud stage for best fibre quality.",
    "Coffee":    "Shade-grown is preferred. Prune annually. Pick only ripe red cherries.",
    "Coconut":   "Maintain basin around the palm. Apply 50 kg organic manure/tree/year. Irrigate in summer.",
    "Banana":    "Use tissue-culture plants. Support with props after bunch emergence. Harvest at 75% maturity.",
    "Mango":     "Prune after harvest. Spray calcium-based nutrients at flowering. Bag fruits for quality.",
    "Groundnut": "Inoculate seeds with Rhizobium. Earth-up at 30 & 60 days. Harvest when leaves yellow.",
    "Soybean":   "Treat seeds with thiram + Rhizobium. Sow in rows 40 cm apart. 2 irrigations critical.",
    "Mustard":   "Thin seedlings to 15 cm spacing. Apply sulphur fertilizer. Harvest when pods turn brown.",
    "Lentil":    "Zero-till after rice is ideal. Seed-treat with Rhizobium. Harvest at physiological maturity.",
    "Chickpea":  "Sow on residual moisture. One irrigation at flowering boosts yield 30%. Watch for pod-borer.",
    "Potato":    "Use certified seed tubers. Ridge planting. Store in cold-store at 2–4 °C.",
    "Tomato":    "Stake or cage plants. Mulch to conserve moisture. Harvest at breaker stage for transport.",
    "Onion":     "Transplant 6-week-old seedlings. Stop irrigation 10 days before harvest. Cure bulbs in shade.",
    "Chili":     "Seedlings transplanted at 5–6 leaf stage. Pick ripe fruits every 8–10 days.",
    "Tea":       "Prune to table height every 3–5 years. Pluck '2 leaves + a bud'. Shade trees recommended.",
    "Bajra":     "Drought tolerant, ideal for arid zones. Thin plants to 15 cm. Apply zinc sulphate if deficient.",
    "Cumin":     "Sow in winter on well-drained sandy loam. Light irrigation at flowering. Harvest when seeds brown.",
    "Rubber":    "Tapping begins at 6–7 years. Tap every alternate day. Apply fertiliser in two splits.",
}

CROP_TIPS_HI = {
    "Rice":      "खड़े पानी की ज़रूरत। 20–30 दिन की पौध रोपें। नाइट्रोजन 3 बार में दें।",
    "Wheat":     "नवंबर में बुवाई सर्वोत्तम। ताजमूल, कल्ले, फूल और दाना भरते समय सिंचाई करें।",
    "Maize":     "अच्छी जल निकासी वाली मिट्टी चाहिए। पूरा P-K बुवाई पर, N 2–3 बार में दें।",
    "Cotton":    "मानसून शुरू होने पर बोएँ। बॉलवर्म के लिए Bt-कपास उपयोग करें। 3–4 बार चुनाई।",
    "Sugarcane": "बसंत में सेट लगाएँ। दो बार मिट्टी चढ़ाएँ। 10–12 महीने पर कटाई करें।",
    "Jute":      "मार्च–अप्रैल में बीज छिटकें। फूल कली अवस्था पर कटाई करें।",
    "Coffee":    "छायादार खेती उत्तम। हर साल छँटाई करें। सिर्फ पकी लाल चेरी तोड़ें।",
    "Coconut":   "पेड़ के चारों ओर थाला बनाएँ। 50 किग्रा जैविक खाद/पेड़/वर्ष। गर्मी में सिंचाई।",
    "Banana":    "टिशू-कल्चर पौधे उपयोग करें। गुच्छा आने पर सहारा दें। 75% परिपक्वता पर तोड़ें।",
    "Mango":     "तुड़ाई के बाद छँटाई। फूल पर कैल्शियम स्प्रे। गुणवत्ता हेतु फल बैगिंग करें।",
    "Groundnut": "बीज को राइज़ोबियम से उपचारित करें। 30 और 60 दिन पर मिट्टी चढ़ाएँ।",
    "Soybean":   "थिरम + राइज़ोबियम से बीजोपचार। 40 सेमी कतार में बोएँ। 2 सिंचाई ज़रूरी।",
    "Mustard":   "15 सेमी दूरी पर पौधे रखें। सल्फर उर्वरक दें। फलियाँ भूरी होने पर काटें।",
    "Lentil":    "धान के बाद ज़ीरो-टिल आदर्श। राइज़ोबियम से बीजोपचार। परिपक्वता पर कटाई।",
    "Chickpea":  "शेष नमी पर बोएँ। फूल पर एक सिंचाई से 30% उपज बढ़ती है। फली छेदक से बचाएँ।",
    "Potato":    "प्रमाणित बीज कंद उपयोग करें। मेड़ पर बुवाई। 2–4°C पर भंडारण।",
    "Tomato":    "पौधों को सहारा दें। नमी बचाने हेतु मल्चिंग। ब्रेकर अवस्था पर तोड़ें।",
    "Onion":     "6 सप्ताह की पौध रोपें। कटाई से 10 दिन पहले सिंचाई बंद। छाया में सुखाएँ।",
    "Chili":     "5–6 पत्ती अवस्था पर रोपाई। हर 8–10 दिन पके फल तोड़ें।",
    "Tea":       "हर 3–5 साल में मेज ऊँचाई पर छँटाई। '2 पत्ती + कली' तोड़ें। छायादार पेड़ लगाएँ।",
    "Bajra":     "सूखा सहनशील, शुष्क क्षेत्रों के लिए आदर्श। पौधे 15 सेमी पर रखें। ज़िंक सल्फेट दें।",
    "Cumin":     "सर्दी में अच्छी जल निकासी वाली दोमट मिट्टी पर बोएँ। फूल पर हल्की सिंचाई।",
    "Rubber":    "6–7 साल बाद टैपिंग शुरू। एक दिन छोड़कर टैप करें। खाद दो बार में दें।",
}

# ══════════════════════════════════════════════════════════════════════════════
#  LIVE MANDI PRICES — data.gov.in Agmarknet API
# ══════════════════════════════════════════════════════════════════════════════
# Register (free) at  https://data.gov.in/user/register  to get your own key.
# Set env var  DATA_GOV_API_KEY  or paste it below.
# The default key below is the public sample key from data.gov.in docs.
DATA_GOV_API_KEY = os.environ.get(
    "DATA_GOV_API_KEY",
    "579b464db66ec23bdd000001ccb584f6da7e411a4d7919984694c9e0",
)
DATA_GOV_MANDI_URL = (
    "https://api.data.gov.in/resource/"
    "9ef84268-d588-465a-a308-a864a43d0070"
)

# Our crop names ➜ data.gov.in commodity names
COMMODITY_MAP = {
    "Rice": "Rice", "Wheat": "Wheat", "Maize": "Maize",
    "Cotton": "Cotton", "Sugarcane": "Sugarcane",
    "Potato": "Potato", "Tomato": "Tomato", "Onion": "Onion",
    "Chili": "Green Chilli", "Soybean": "Soyabean",
    "Mustard": "Mustard", "Chickpea": "Bengal Gram(Gram)(Whole)",
    "Groundnut": "Groundnut", "Lentil": "Lentil (Masur)(Whole)",
    "Banana": "Banana", "Bajra": "Bajra(Pearl Millet/Cumbu)",
}
REVERSE_COMMODITY = {}
for _ck, _cv in COMMODITY_MAP.items():
    REVERSE_COMMODITY[_cv.strip().lower()] = _ck

# MSP ₹/quintal — Kharif & Rabi 2024-25 (update yearly)
MSP_DATA = {
    "Rice": 2300, "Wheat": 2275, "Maize": 2090, "Cotton": 7121,
    "Sugarcane": 315, "Potato": 0, "Tomato": 0, "Onion": 0,
    "Chili": 0, "Soybean": 4892, "Mustard": 5650, "Chickpea": 5440,
    "Groundnut": 6377, "Lentil": 6425, "Banana": 0, "Bajra": 2500,
}

# Indian states for dropdown filter
INDIAN_STATES = [
    "Andhra Pradesh","Assam","Bihar","Chhattisgarh","Goa","Gujarat",
    "Haryana","Himachal Pradesh","Jharkhand","Karnataka","Kerala",
    "Madhya Pradesh","Maharashtra","Manipur","Meghalaya","Mizoram",
    "Nagaland","Odisha","Punjab","Rajasthan","Sikkim","Tamil Nadu",
    "Telangana","Tripura","Uttar Pradesh","Uttrakhand","West Bengal",
    "Chandigarh","NCT of Delhi","Jammu and Kashmir","Puducherry",
]

# ── In-memory cache (clears on server restart) ──
_mandi_cache: dict = {}
MANDI_CACHE_TTL = 3600          # 1 hour


def _safe_int(val):
    """Convert data.gov.in string values to int."""
    try:
        return int(float(val))
    except (TypeError, ValueError):
        return 0


def _match_crop_name(commodity_raw: str):
    """Try to map an API commodity string back to our crop name."""
    low = commodity_raw.strip().lower()
    # exact match first
    hit = REVERSE_COMMODITY.get(low)
    if hit:
        return hit
    # partial / substring match
    for api_low, our_name in REVERSE_COMMODITY.items():
        if api_low in low or low in api_low:
            return our_name
    return None


def fetch_mandi_from_gov(commodity_key="", state="", district="", market="", limit=500):
    """Fetch commodity prices from data.gov.in Agmarknet API.

    Returns ``{"records": [...], "total": int, "fetched_at": str}`` or
    ``None`` when the API is unreachable / key missing.
    """
    if not DATA_GOV_API_KEY:
        return None

    cache_key = f"{commodity_key}|{state}|{district}|{market}|{limit}"
    now = time.time()
    cached = _mandi_cache.get(cache_key)
    if cached and (now - cached["ts"]) < MANDI_CACHE_TTL:
        return cached["data"]

    params = {
        "api-key": DATA_GOV_API_KEY,
        "format": "json",
        "limit": limit,
    }
    if commodity_key and commodity_key in COMMODITY_MAP:
        params["filters[commodity]"] = COMMODITY_MAP[commodity_key]
    if state:
        params["filters[state]"] = state
    if district:
        params["filters[district]"] = district
    if market:
        params["filters[market]"] = market

    try:
        resp = http_requests.get(DATA_GOV_MANDI_URL, params=params, timeout=12)
        resp.raise_for_status()
        body = resp.json()
        raw = body.get("records", [])
        records = []
        for r in raw:
            records.append({
                "market":      r.get("market", "").strip(),
                "state":       r.get("state", "").strip(),
                "district":    r.get("district", "").strip(),
                "commodity":   r.get("commodity", "").strip(),
                "variety":     r.get("variety", "").strip(),
                "date":        r.get("arrival_date", ""),
                "min_price":   _safe_int(r.get("min_price")),
                "max_price":   _safe_int(r.get("max_price")),
                "modal_price": _safe_int(r.get("modal_price")),
            })
        result = {
            "records": records,
            "total":   body.get("total", len(records)),
            "fetched_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        }
        _mandi_cache[cache_key] = {"data": result, "ts": now}
        return result
    except Exception as exc:
        print(f"[Mandi API] {exc}")
        return None


# ── Hardcoded fallback (used when API is down / no key) ──────────────────────
MANDI_FALLBACK = {
    "Rice":      [{"name":"Karnal","state":"Haryana","price":2250,"date":"—"},{"name":"Amritsar","state":"Punjab","price":2280,"date":"—"}],
    "Wheat":     [{"name":"Hapur","state":"UP","price":2310,"date":"—"},{"name":"Indore","state":"MP","price":2290,"date":"—"}],
    "Maize":     [{"name":"Gulbarga","state":"Karnataka","price":2100,"date":"—"},{"name":"Davangere","state":"Karnataka","price":2050,"date":"—"}],
    "Cotton":    [{"name":"Rajkot","state":"Gujarat","price":7100,"date":"—"},{"name":"Nagpur","state":"Maharashtra","price":6980,"date":"—"}],
    "Sugarcane": [{"name":"Muzaffarnagar","state":"UP","price":350,"date":"—"},{"name":"Kolhapur","state":"Maharashtra","price":320,"date":"—"}],
    "Potato":    [{"name":"Agra","state":"UP","price":800,"date":"—"},{"name":"Hooghly","state":"WB","price":750,"date":"—"}],
    "Tomato":    [{"name":"Kolar","state":"Karnataka","price":1200,"date":"—"},{"name":"Nashik","state":"Maharashtra","price":1100,"date":"—"}],
    "Onion":     [{"name":"Lasalgaon","state":"Maharashtra","price":1800,"date":"—"},{"name":"Nashik","state":"Maharashtra","price":1750,"date":"—"}],
    "Chili":     [{"name":"Guntur","state":"AP","price":14000,"date":"—"},{"name":"Khammam","state":"Telangana","price":13500,"date":"—"}],
    "Soybean":   [{"name":"Indore","state":"MP","price":4700,"date":"—"},{"name":"Latur","state":"Maharashtra","price":4650,"date":"—"}],
    "Mustard":   [{"name":"Alwar","state":"Rajasthan","price":5800,"date":"—"},{"name":"Bharatpur","state":"Rajasthan","price":5750,"date":"—"}],
    "Chickpea":  [{"name":"Indore","state":"MP","price":5500,"date":"—"},{"name":"Jaipur","state":"Rajasthan","price":5480,"date":"—"}],
    "Groundnut": [{"name":"Rajkot","state":"Gujarat","price":6500,"date":"—"},{"name":"Junagadh","state":"Gujarat","price":6450,"date":"—"}],
    "Lentil":    [{"name":"Indore","state":"MP","price":6500,"date":"—"},{"name":"Rewa","state":"MP","price":6450,"date":"—"}],
    "Banana":    [{"name":"Jalgaon","state":"Maharashtra","price":900,"date":"—"},{"name":"Tiruchirappalli","state":"Tamil Nadu","price":1000,"date":"—"}],
    "Bajra":     [{"name":"Jodhpur","state":"Rajasthan","price":2550,"date":"—"},{"name":"Hisar","state":"Haryana","price":2520,"date":"—"}],
}

# ── Government Scheme Data ───────────────────────────────────────────────────
GOVT_SCHEMES = [
    {
        "id": "pmkisan",
        "name": "PM-KISAN Samman Nidhi",
        "name_hi": "पीएम-किसान सम्मान निधि",
        "desc": "₹6,000/year direct income support in 3 instalments to farmer families.",
        "desc_hi": "किसान परिवारों को ₹6,000/वर्ष सीधे 3 किस्तों में।",
        "eligibility": {"land_max_ha": 100, "income_max": 0, "categories": ["all"]},
        "url": "https://pmkisan.gov.in",
        "benefits": "₹6,000 per year",
    },
    {
        "id": "pmfby",
        "name": "PM Fasal Bima Yojana",
        "name_hi": "पीएम फसल बीमा योजना",
        "desc": "Crop insurance at 2% premium for Kharif, 1.5% for Rabi, 5% for horticulture.",
        "desc_hi": "खरीफ में 2%, रबी में 1.5%, बागवानी में 5% प्रीमियम पर फसल बीमा।",
        "eligibility": {"land_max_ha": 100, "income_max": 0, "categories": ["all"]},
        "url": "https://pmfby.gov.in",
        "benefits": "Full crop insurance coverage",
    },
    {
        "id": "kcc",
        "name": "Kisan Credit Card",
        "name_hi": "किसान क्रेडिट कार्ड",
        "desc": "Short-term loans at 4% interest rate with prompt repayment discount.",
        "desc_hi": "समय पर भुगतान पर 4% ब्याज दर पर अल्पकालिक ऋण।",
        "eligibility": {"land_max_ha": 100, "income_max": 0, "categories": ["all"]},
        "url": "https://www.nabard.org",
        "benefits": "Loan up to ₹3 lakh at 4%",
    },
    {
        "id": "soil_health",
        "name": "Soil Health Card Scheme",
        "name_hi": "मृदा स्वास्थ्य कार्ड योजना",
        "desc": "Free soil testing every 2 years with nutrient-specific fertiliser recommendations.",
        "desc_hi": "हर 2 साल मुफ्त मिट्टी परीक्षण और उर्वरक सिफारिश।",
        "eligibility": {"land_max_ha": 100, "income_max": 0, "categories": ["all"]},
        "url": "https://soilhealth.dac.gov.in",
        "benefits": "Free soil testing",
    },
    {
        "id": "pmkmy",
        "name": "PM Kisan Maandhan Yojana",
        "name_hi": "पीएम किसान मानधन योजना",
        "desc": "Pension scheme: ₹3,000/month after age 60 for small & marginal farmers.",
        "desc_hi": "60 वर्ष के बाद ₹3,000/माह पेंशन — छोटे व सीमांत किसानों के लिए।",
        "eligibility": {"land_max_ha": 2, "income_max": 0, "categories": ["small", "marginal"]},
        "url": "https://maandhan.in",
        "benefits": "₹3,000/month pension after 60",
    },
    {
        "id": "enam",
        "name": "e-NAM (National Agriculture Market)",
        "name_hi": "ई-नाम (राष्ट्रीय कृषि बाज़ार)",
        "desc": "Online trading platform for transparent, competitive crop selling across mandis.",
        "desc_hi": "ऑनलाइन पारदर्शी मंडी व्यापार मंच।",
        "eligibility": {"land_max_ha": 100, "income_max": 0, "categories": ["all"]},
        "url": "https://enam.gov.in",
        "benefits": "Better prices through competition",
    },
    {
        "id": "pkvy",
        "name": "Paramparagat Krishi Vikas Yojana",
        "name_hi": "परम्परागत कृषि विकास योजना",
        "desc": "₹50,000/ha over 3 years for organic farming clusters of 50+ farmers.",
        "desc_hi": "50+ किसानों के जैविक खेती समूह को 3 वर्ष में ₹50,000/हेक्टेयर।",
        "eligibility": {"land_max_ha": 100, "income_max": 0, "categories": ["all"]},
        "url": "https://pgsindia-ncof.gov.in",
        "benefits": "₹50,000/ha for organic farming",
    },
    {
        "id": "pmksy",
        "name": "PM Krishi Sinchai Yojana",
        "name_hi": "पीएम कृषि सिंचाई योजना",
        "desc": "55% subsidy on micro-irrigation (drip/sprinkler) for small farmers.",
        "desc_hi": "छोटे किसानों को सूक्ष्म सिंचाई पर 55% सब्सिडी।",
        "eligibility": {"land_max_ha": 5, "income_max": 0, "categories": ["small", "marginal"]},
        "url": "https://pmksy.gov.in",
        "benefits": "55% subsidy on irrigation",
    },
]

# ── Simulated soil test reports ──────────────────────────────────────────────
SOIL_REPORTS = {}  # Will be stored in SQLite

# ── Common plant diseases with treatments ────────────────────────────────────
DISEASE_DB = {
    "Bacterial Leaf Blight": {
        "crop": "Rice", "type": "Bacterial",
        "symptoms": "Water-soaked lesions on leaves that turn yellow-white, wilting leaf tips.",
        "symptoms_hi": "पत्तियों पर पानी जैसे धब्बे जो पीले-सफेद हो जाते हैं, पत्ती की नोक मुरझाना।",
        "treatment": "Use resistant varieties (Samba Mahsuri). Avoid excess N. Spray Streptocycline 0.01%.",
        "treatment_hi": "प्रतिरोधी किस्में (सांबा मसूरी) उपयोग करें। अधिक N से बचें। स्ट्रेप्टोसाइक्लिन 0.01% छिड़कें।",
        "severity": "high",
    },
    "Late Blight": {
        "crop": "Potato/Tomato", "type": "Fungal",
        "symptoms": "Dark brown spots on leaves with white mold underneath. Rapid plant death.",
        "symptoms_hi": "पत्तियों पर गहरे भूरे धब्बे, नीचे सफेद फफूंद। तेजी से पौधा सूखना।",
        "treatment": "Spray Mancozeb 0.25% at 10-day intervals. Destroy infected plants. Use resistant varieties.",
        "treatment_hi": "मैन्कोज़ेब 0.25% हर 10 दिन छिड़कें। संक्रमित पौधे नष्ट करें। प्रतिरोधी किस्में लगाएँ।",
        "severity": "high",
    },
    "Powdery Mildew": {
        "crop": "Multiple", "type": "Fungal",
        "symptoms": "White powdery coating on leaves and stems. Leaves curl and dry.",
        "symptoms_hi": "पत्तियों और तने पर सफेद पाउडर जैसी परत। पत्ते मुड़ कर सूखते हैं।",
        "treatment": "Spray Sulphur WP 0.3% or Karathane 0.05%. Ensure good air circulation.",
        "treatment_hi": "सल्फर WP 0.3% या कैराथेन 0.05% छिड़कें। हवा का अच्छा संचार सुनिश्चित करें।",
        "severity": "medium",
    },
    "Fusarium Wilt": {
        "crop": "Multiple", "type": "Fungal",
        "symptoms": "Yellowing of lower leaves, wilting on one side, brown vascular tissue.",
        "symptoms_hi": "निचली पत्तियों का पीलापन, एक तरफ मुरझाना, भूरे संवहनी ऊतक।",
        "treatment": "Use resistant varieties. Treat seed with Trichoderma. Crop rotation with cereals.",
        "treatment_hi": "प्रतिरोधी किस्में उपयोग करें। ट्राइकोडर्मा से बीजोपचार। अनाज के साथ फसल चक्र।",
        "severity": "high",
    },
    "Leaf Curl Virus": {
        "crop": "Chili/Tomato", "type": "Viral",
        "symptoms": "Upward curling of leaves, stunted growth, reduced fruit set.",
        "symptoms_hi": "पत्तियों का ऊपर की ओर मुड़ना, बौना विकास, फल कम लगना।",
        "treatment": "Control whitefly vectors with Imidacloprid. Use virus-free seedlings. Yellow sticky traps.",
        "treatment_hi": "इमिडाक्लोप्रिड से सफेद मक्खी नियंत्रण। वायरस-मुक्त पौध उपयोग करें। पीले चिपचिपे ट्रैप।",
        "severity": "medium",
    },
    "Rust": {
        "crop": "Wheat/Soybean", "type": "Fungal",
        "symptoms": "Orange-brown pustules on leaves and stems. Premature drying.",
        "symptoms_hi": "पत्तियों और तने पर नारंगी-भूरे दाने। समय से पहले सूखना।",
        "treatment": "Grow resistant varieties. Spray Propiconazole 25EC @ 0.1%. Early sowing helps.",
        "treatment_hi": "प्रतिरोधी किस्में उगाएँ। प्रोपिकोनाज़ोल 25EC @0.1% छिड़कें। जल्दी बुवाई सहायक।",
        "severity": "high",
    },
    "Anthracnose": {
        "crop": "Mango/Chili", "type": "Fungal",
        "symptoms": "Dark sunken spots on fruits and leaves. Fruit rot during storage.",
        "symptoms_hi": "फलों और पत्तियों पर गहरे धँसे धब्बे। भंडारण में फल सड़ना।",
        "treatment": "Spray Carbendazim 0.1% before and after flowering. Proper fruit handling.",
        "treatment_hi": "फूल से पहले और बाद कार्बेन्डाज़िम 0.1% छिड़कें। फलों को सही तरीके से रखें।",
        "severity": "medium",
    },
    "Bollworm": {
        "crop": "Cotton", "type": "Pest",
        "symptoms": "Holes in bolls, frass visible. Damaged bolls with larvae inside.",
        "symptoms_hi": "बॉल्स में छेद, कीट मल दिखाई देना। लार्वा वाले क्षतिग्रस्त बॉल्स।",
        "treatment": "Use Bt cotton varieties. Install pheromone traps. Spray Spinosad if severe.",
        "treatment_hi": "Bt कपास किस्में उपयोग करें। फेरोमोन ट्रैप लगाएँ। गंभीर होने पर स्पिनोसैड छिड़कें।",
        "severity": "high",
    },
}


# ── SQLite helper ─────────────────────────────────────────────────────────────
DB_PATH = os.path.join(os.path.dirname(__file__), "users.db")

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            name     TEXT NOT NULL,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            phone    TEXT DEFAULT '',
            state    TEXT DEFAULT '',
            location TEXT DEFAULT '',
            land_ha  REAL DEFAULT 0,
            category TEXT DEFAULT 'all',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    # Forum posts
    conn.execute("""
        CREATE TABLE IF NOT EXISTS forum_posts (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id    INTEGER,
            username   TEXT NOT NULL,
            title      TEXT NOT NULL,
            content    TEXT NOT NULL,
            category   TEXT DEFAULT 'general',
            likes      INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)
    # Forum replies
    conn.execute("""
        CREATE TABLE IF NOT EXISTS forum_replies (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            post_id    INTEGER,
            user_id    INTEGER,
            username   TEXT NOT NULL,
            content    TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (post_id) REFERENCES forum_posts(id),
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)
    # Soil test reports
    conn.execute("""
        CREATE TABLE IF NOT EXISTS soil_reports (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            report_id  TEXT UNIQUE NOT NULL,
            user_id    INTEGER,
            N          REAL, P REAL, K REAL,
            pH         REAL, EC REAL,
            OC         REAL, S REAL, Zn REAL, Fe REAL, Mn REAL, Cu REAL, B REAL,
            soil_type  TEXT,
            location   TEXT,
            state      TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)
    # SMS alert subscriptions
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sms_subscriptions (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id   INTEGER,
            phone     TEXT NOT NULL,
            alert_type TEXT DEFAULT 'all',
            city      TEXT DEFAULT '',
            active    INTEGER DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)
    # ── IoT Sensor Readings ──
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sensor_readings (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            device_id     TEXT NOT NULL,
            temperature   REAL,
            humidity      REAL,
            moisture      REAL,
            ph            REAL,
            moisture_raw  INTEGER DEFAULT 0,
            ph_raw        INTEGER DEFAULT 0,
            ph_voltage    REAL DEFAULT 0,
            battery_v     REAL DEFAULT 0,
            wifi_rssi     INTEGER DEFAULT 0,
            created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    # Migrate existing DB — add raw columns if missing
    try:
        conn.execute("ALTER TABLE sensor_readings ADD COLUMN moisture_raw INTEGER DEFAULT 0")
    except Exception:
        pass
    try:
        conn.execute("ALTER TABLE sensor_readings ADD COLUMN ph_raw INTEGER DEFAULT 0")
    except Exception:
        pass
    try:
        conn.execute("ALTER TABLE sensor_readings ADD COLUMN ph_voltage REAL DEFAULT 0")
    except Exception:
        pass
    # ── IoT Camera Captures ──
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sensor_images (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            device_id   TEXT NOT NULL,
            image_data  TEXT,
            diagnosis   TEXT DEFAULT '',
            confidence  REAL DEFAULT 0,
            created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    # ── Favorite Mandis ──
    conn.execute("""
        CREATE TABLE IF NOT EXISTS favorite_mandis (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id    INTEGER NOT NULL,
            state      TEXT NOT NULL DEFAULT '',
            district   TEXT NOT NULL DEFAULT '',
            market     TEXT NOT NULL DEFAULT '',
            label      TEXT DEFAULT '',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id),
            UNIQUE(user_id, state, district, market)
        )
    """)
    # ── Prediction History ──
    conn.execute("""
        CREATE TABLE IF NOT EXISTS prediction_history (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id    INTEGER NOT NULL,
            state      TEXT DEFAULT '',
            district   TEXT DEFAULT '',
            season     TEXT DEFAULT '',
            input_data TEXT DEFAULT '{}',
            top_crop   TEXT DEFAULT '',
            confidence REAL DEFAULT 0,
            all_predictions TEXT DEFAULT '[]',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)
    conn.commit()
    conn.close()

def hash_pw(pw: str) -> str:
    return bcrypt.hashpw(pw.encode(), bcrypt.gensalt()).decode()

def check_pw(pw: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(pw.encode(), hashed.encode())
    except Exception:
        # Fallback: check legacy SHA256 hash for old accounts
        return hashlib.sha256(pw.encode()).hexdigest() == hashed

def sanitize(text: str) -> str:
    """Strip all HTML tags from user input to prevent XSS."""
    return bleach.clean(text, tags=[], strip=True)


# ── Page routes ───────────────────────────────────────────────────────────────
@app.route("/")
def index():
    if "user" in session:
        return redirect(url_for("dashboard"))
    return render_template("index.html")

@app.route("/dashboard")
def dashboard():
    if "user" not in session:
        return redirect(url_for("index"))
    return render_template("dashboard.html", user=session["user"])

@app.route("/crop")
def crop():
    if "user" not in session:
        return redirect(url_for("index"))
    return render_template("crop.html", states=list(STATE_CROPS.keys()))

@app.route("/weather")
def weather():
    if "user" not in session:
        return redirect(url_for("index"))
    return render_template("weather.html")

@app.route("/info")
def info():
    if "user" not in session:
        return redirect(url_for("index"))
    return render_template("info.html")

@app.route("/soil")
def soil():
    if "user" not in session:
        return redirect(url_for("index"))
    return render_template("soil.html")

@app.route("/schemes")
def schemes():
    if "user" not in session:
        return redirect(url_for("index"))
    return render_template("schemes.html")

@app.route("/mandi")
def mandi():
    if "user" not in session:
        return redirect(url_for("index"))
    return render_template("mandi.html")

@app.route("/disease")
def disease():
    if "user" not in session:
        return redirect(url_for("index"))
    return render_template("disease.html")

@app.route("/iot")
def iot_dashboard():
    if "user" not in session:
        return redirect(url_for("index"))
    return render_template("iot.html", user=session["user"])

@app.route("/forum")
def forum():
    if "user" not in session:
        return redirect(url_for("index"))
    return render_template("forum.html", user=session["user"])

@app.route("/alerts")
def alerts():
    if "user" not in session:
        return redirect(url_for("index"))
    return render_template("alerts.html", user=session["user"])

@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("index"))


# ── PWA ───────────────────────────────────────────────────────────────────────
@app.route("/manifest.json")
def manifest():
    return send_from_directory("static", "manifest.json")

@app.route("/sw.js")
def service_worker():
    return send_from_directory("static", "sw.js")


# ── Auth API ──────────────────────────────────────────────────────────────────
@app.route("/api/signup", methods=["POST"])
def api_signup():
    data = request.get_json(force=True)
    name     = sanitize(data.get("name", "").strip())
    username = sanitize(data.get("username", "").strip())
    password = data.get("password", "").strip()
    phone    = sanitize(data.get("phone", "").strip())
    state    = sanitize(data.get("state", "").strip())
    location = sanitize(data.get("location", "").strip())

    if not username or not password:
        return jsonify(success=False, error="Username and password required"), 400
    if len(password) < 6:
        return jsonify(success=False, error="Password must be at least 6 characters"), 400

    conn = get_db()
    try:
        conn.execute(
            "INSERT INTO users (name, username, password, phone, state, location) VALUES (?, ?, ?, ?, ?, ?)",
            (name or username, username, hash_pw(password), phone, state, location),
        )
        conn.commit()
    except sqlite3.IntegrityError:
        return jsonify(success=False, error="Username already exists"), 409
    finally:
        conn.close()

    session["user"] = {"name": name or username, "username": username, "state": state, "phone": phone, "location": location}
    return jsonify(success=True)


@app.route("/api/login", methods=["POST"])
def api_login():
    data = request.get_json(force=True)
    username = data.get("username", "").strip()
    password = data.get("password", "").strip()

    conn = get_db()
    row = conn.execute(
        "SELECT * FROM users WHERE username = ?",
        (username,),
    ).fetchone()
    conn.close()

    if row is None or not check_pw(password, row["password"]):
        return jsonify(success=False, error="Invalid credentials"), 401

    session["user"] = {
        "name": row["name"], "username": row["username"],
        "state": row["state"] or "", "phone": row["phone"] or "",
        "location": row["location"] if "location" in row.keys() else ""
    }
    return jsonify(success=True)


# ── Crop prediction API ──────────────────────────────────────────────────────
@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(force=True)

    def _value_for_feature(col: str):
        if col in data:
            return data[col]
        low = col.lower()
        if low in data:
            return data[low]
        if col == "ph" and "pH" in data:
            return data["pH"]
        if col == "pH" and "ph" in data:
            return data["ph"]
        raise KeyError(col)

    try:
        features = np.array([[float(_value_for_feature(col)) for col in feature_columns]])
    except (KeyError, ValueError) as exc:
        return jsonify(success=False, error=f"Invalid input: {exc}"), 400

    state = data.get("state", "").strip()
    district = data.get("district", "").strip()
    season = data.get("season", "").strip().lower()

    # Soil moisture (not a model feature, but used for advisory)
    soil_moisture_raw = data.get("soil_moisture", "")
    soil_moisture = None
    if soil_moisture_raw not in (None, ""):
        try:
            soil_moisture = float(soil_moisture_raw)
        except (ValueError, TypeError):
            pass

    # Auto-detect season from current month if not provided
    if not season:
        month = datetime.now().month
        if month in (6, 7, 8, 9, 10):
            season = "kharif"
        elif month in (11, 12, 1, 2, 3):
            season = "rabi"
        else:
            season = "zaid"

    crop_alias = {
        "pigeonpeas": "Chickpea",
        "mothbeans": "Bajra",
        "mungbean": "Lentil",
        "blackgram": "Lentil",
    }

    def _decode_crop_label(class_id):
        # Handles both integer-encoded classes and raw string classes.
        raw = None
        if label_encoder is not None:
            try:
                raw = label_encoder.inverse_transform([int(class_id)])[0]
            except Exception:
                raw = None
        if raw is None:
            raw = str(class_id)
        raw = str(raw).strip()
        mapped = crop_alias.get(raw.lower())
        if mapped:
            return mapped
        return raw.title() if raw and raw == raw.lower() else raw

    model_input = features
    if scaler is not None and hasattr(scaler, "transform"):
        try:
            model_input = scaler.transform(features)
        except Exception as exc:
            print(f"⚠️  scaler transform failed; using raw features: {exc}")

    proba = model.predict_proba(model_input)[0]
    class_ids = np.array(getattr(model, "classes_", np.arange(len(proba))))

    district_crops = []
    if state and district and state in STATE_DISTRICT_CROPS:
        district_map = STATE_DISTRICT_CROPS[state]
        district_lookup = {k.lower().strip(): v for k, v in district_map.items()}
        district_crops = district_lookup.get(district.lower()) or []

    # Boost score for district-suitable crops (stronger than state boost)
    if district_crops:
        for i, class_id in enumerate(class_ids):
            crop_label = _decode_crop_label(class_id)
            if crop_label in district_crops:
                proba[i] *= 1.25

    # Boost score for state-suitable crops
    if state and state in STATE_CROPS:
        state_crops = STATE_CROPS[state]
        for i, class_id in enumerate(class_ids):
            crop_label = _decode_crop_label(class_id)
            if crop_label in state_crops:
                proba[i] *= 1.15  # 15% boost for state-suitable crops

    # Location-priority ranking: when location is selected and we have known
    # allowed crops for that location, only keep those classes in ranking.
    original_proba = proba.copy()
    allowed_location_crops = district_crops or (STATE_CROPS.get(state, []) if state else [])
    if allowed_location_crops:
        allowed_norm = {str(c).strip().lower() for c in allowed_location_crops}
        allowed_indices = []
        for i, class_id in enumerate(class_ids):
            crop_label = _decode_crop_label(class_id)
            if crop_label.strip().lower() in allowed_norm:
                allowed_indices.append(i)

        if allowed_indices:
            mask = np.zeros_like(proba)
            mask[allowed_indices] = 1.0
            proba = proba * mask

            # If all allowed classes ended up zero, keep only allowed classes
            # and distribute fallback score to avoid meaningless all-zero output.
            if proba.sum() == 0:
                allowed_vals = original_proba[allowed_indices]
                if allowed_vals.sum() > 0:
                    proba[allowed_indices] = allowed_vals
                else:
                    proba[allowed_indices] = 1.0

    # ── Season filtering: penalize crops not suitable for the selected season ──
    if season and season in CROP_SEASON:
        season_crops = set(CROP_SEASON[season])
        for i, class_id in enumerate(class_ids):
            crop_label = _decode_crop_label(class_id)
            if crop_label not in season_crops and crop_label not in PERENNIAL_CROPS:
                proba[i] *= 0.05  # 95% penalty for out-of-season crops

    # Renormalize
    total = proba.sum()
    if total > 0:
        proba = proba / total
    top_idx = np.argsort(proba)[::-1][:5]

    predictions = []
    for idx in top_idx:
        class_id = class_ids[idx] if idx < len(class_ids) else idx
        crop_name = _decode_crop_label(class_id)
        diseases = CROP_DISEASES.get(crop_name, [])
        crop_seasons = sorted(CROP_TO_SEASONS.get(crop_name, []))
        predictions.append({
            "crop":       crop_name,
            "confidence": round(float(proba[idx]) * 100, 2),
            "tip":        CROP_TIPS.get(crop_name, ""),
            "tip_hi":     CROP_TIPS_HI.get(crop_name, ""),
            "diseases":   diseases[:3],
            "msp":        MSP_DATA.get(crop_name, 0),
            "seasons":    crop_seasons,
        })

    debug_input = {col: float(features[0][i]) for i, col in enumerate(feature_columns)}

    # ── Save to prediction_history if user is logged in ──
    try:
        user = session.get("user")
        if user:
            conn_h = get_db()
            u_row = conn_h.execute("SELECT id FROM users WHERE username=?", (user["username"],)).fetchone()
            if u_row:
                conn_h.execute(
                    """INSERT INTO prediction_history
                       (user_id, state, district, season, input_data, top_crop, confidence, all_predictions)
                       VALUES (?,?,?,?,?,?,?,?)""",
                    (
                        u_row["id"],
                        state or "",
                        district or "",
                        season or "",
                        json.dumps(debug_input),
                        predictions[0]["crop"] if predictions else "",
                        predictions[0]["confidence"] if predictions else 0,
                        json.dumps(predictions),
                    ),
                )
                conn_h.commit()
            conn_h.close()
    except Exception:
        pass  # Don't break prediction if history save fails

    # ── Soil moisture advisory (not a model feature, but useful for farmer) ──
    moisture_advisory = None
    if soil_moisture is not None:
        if soil_moisture < 20:
            moisture_advisory = {
                "level": "danger",
                "msg": f"Soil moisture is very low ({soil_moisture:.0f}%). Irrigate before sowing!",
                "msg_hi": f"मिट्टी की नमी बहुत कम है ({soil_moisture:.0f}%)। बुवाई से पहले सिंचाई करें!",
            }
        elif soil_moisture < 35:
            moisture_advisory = {
                "level": "warning",
                "msg": f"Soil moisture is low ({soil_moisture:.0f}%). Consider irrigating soon.",
                "msg_hi": f"मिट्टी की नमी कम है ({soil_moisture:.0f}%)। जल्द सिंचाई करें।",
            }
        elif soil_moisture > 85:
            moisture_advisory = {
                "level": "warning",
                "msg": f"Soil is waterlogged ({soil_moisture:.0f}%). Ensure drainage before planting.",
                "msg_hi": f"मिट्टी जलभराव ({soil_moisture:.0f}%) है। बुवाई से पहले जल निकासी सुनिश्चित करें।",
            }
        else:
            moisture_advisory = {
                "level": "success",
                "msg": f"Soil moisture is good ({soil_moisture:.0f}%). Ready for sowing.",
                "msg_hi": f"मिट्टी की नमी अच्छी है ({soil_moisture:.0f}%)। बुवाई के लिए तैयार।",
            }

    return jsonify(
        success=True,
        predictions=predictions,
        state=state,
        district=district,
        season=season,
        soil_moisture=soil_moisture,
        moisture_advisory=moisture_advisory,
        feature_columns=feature_columns,
        input_used=debug_input,
    )


# ── Prediction History API ───────────────────────────────────────────────────
@app.route("/api/predictions/history")
def api_prediction_history():
    user = session.get("user")
    if not user:
        return jsonify(success=False, error="Not logged in"), 401
    conn = get_db()
    u_row = conn.execute("SELECT id FROM users WHERE username=?", (user["username"],)).fetchone()
    if not u_row:
        conn.close()
        return jsonify(success=False, error="User not found"), 404
    rows = conn.execute(
        """SELECT id, state, district, season, input_data, top_crop, confidence,
                  all_predictions, created_at
           FROM prediction_history WHERE user_id=? ORDER BY created_at DESC LIMIT 10""",
        (u_row["id"],),
    ).fetchall()
    conn.close()
    history = []
    for r in rows:
        history.append({
            "id": r["id"],
            "state": r["state"],
            "district": r["district"],
            "season": r["season"],
            "input_data": json.loads(r["input_data"]) if r["input_data"] else {},
            "top_crop": r["top_crop"],
            "confidence": r["confidence"],
            "all_predictions": json.loads(r["all_predictions"]) if r["all_predictions"] else [],
            "created_at": r["created_at"],
        })
    return jsonify(success=True, history=history)


# ── State crops API ──────────────────────────────────────────────────────────
@app.route("/api/state-crops")
def api_state_crops():
    state = request.args.get("state", "")
    if state in STATE_CROPS:
        return jsonify(success=True, state=state, crops=STATE_CROPS[state])
    return jsonify(success=True, states=list(STATE_CROPS.keys()))


@app.route("/api/state-districts")
def api_state_districts():
    state = request.args.get("state", "").strip()
    if not state:
        return jsonify(success=True, states=list(STATE_DISTRICT_CROPS.keys()))

    district_map = STATE_DISTRICT_CROPS.get(state, {})
    return jsonify(success=True, state=state, districts=sorted(district_map.keys()))


@app.route("/api/model/info")
def api_model_info():
    return jsonify(success=True, active=ACTIVE_CROP_MODEL_FILES, feature_columns=feature_columns)


@app.route("/api/model/reload", methods=["POST"])
def api_model_reload():
    data = request.get_json(silent=True) or {}
    try:
        loaded_paths = load_crop_artifacts(
            model_file=data.get("model"),
            scaler_file=data.get("scaler"),
            label_encoder_file=data.get("label_encoder"),
        )
        load_crop_metadata()
        return jsonify(success=True, message="Crop model reloaded", files=loaded_paths)
    except Exception as exc:
        return jsonify(success=False, error=f"Failed to reload crop model: {exc}"), 400


# ── Weather API (wttr.in – no key required) ──────────────────────────────────
@app.route("/api/weather")
def api_weather():
    city = request.args.get("city", "Delhi")
    try:
        resp = http_requests.get(
            f"https://wttr.in/{city}?format=j1",
            headers={"User-Agent": "KisanSalahkar/2.0"},
            timeout=10,
        )
        resp.raise_for_status()
        w = resp.json()

        current = w["current_condition"][0]
        forecast = []
        for day in w.get("weather", [])[:3]:
            forecast.append({
                "date":     day["date"],
                "maxTemp":  day["maxtempC"],
                "minTemp":  day["mintempC"],
                "desc":     day["hourly"][4]["weatherDesc"][0]["value"],
                "humidity": day["hourly"][4]["humidity"],
            })

        # Farming advisory based on weather
        temp = int(current["temp_C"])
        humidity = int(current["humidity"])
        advisory = []
        if temp > 40:
            advisory.append("Extreme heat! Irrigate crops in evening. Use mulch to protect roots.")
        elif temp > 35:
            advisory.append("High temperature. Increase irrigation frequency. Avoid mid-day spraying.")
        if humidity > 85:
            advisory.append("High humidity — watch for fungal diseases. Ensure proper drainage.")
        if humidity < 30:
            advisory.append("Low humidity — increase irrigation. Consider sprinkler for humidity-loving crops.")

        return jsonify(success=True, current={
            "temp":      current["temp_C"],
            "feelsLike": current["FeelsLikeC"],
            "humidity":  current["humidity"],
            "desc":      current["weatherDesc"][0]["value"],
            "windSpeed": current["windspeedKmph"],
            "windDir":   current["winddir16Point"],
            "pressure":  current["pressure"],
            "visibility":current["visibility"],
            "uv":        current.get("uvIndex", "N/A"),
        }, forecast=forecast, city=city, advisory=advisory)
    except Exception as exc:
        return jsonify(success=False, error=str(exc)), 502


# ── Rainfall Estimation API ──────────────────────────────────────────────────
@app.route("/api/rainfall/estimate")
def api_rainfall_estimate():
    """Estimate seasonal rainfall for a given state + season.
    Optionally cross-checks with live wttr.in precipitation forecast."""
    state = request.args.get("state", "").strip()
    season = request.args.get("season", "").strip().lower()
    district = request.args.get("district", "").strip()

    if not state:
        return jsonify(success=False, error="State is required"), 400

    # Auto-detect season from current month if not provided
    if not season:
        month = datetime.now().month
        if month in (6, 7, 8, 9, 10):
            season = "kharif"
        elif month in (11, 12, 1, 2, 3):
            season = "rabi"
        else:
            season = "zaid"

    # Lookup base rainfall from state data
    state_data = STATE_RAINFALL.get(state)
    if not state_data:
        return jsonify(success=False, error=f"No rainfall data for state: {state}"), 404

    base_rainfall = state_data.get(season)
    if base_rainfall is None:
        return jsonify(success=False, error=f"No rainfall data for season: {season}"), 404

    result = {
        "rainfall": base_rainfall,
        "state": state,
        "season": season,
        "source": "historical_avg",
        "unit": "mm/season",
    }
    if district:
        result["district"] = district

    # Try to refine with live weather data from wttr.in
    try:
        location = f"{district}, {state}" if district else state
        resp = http_requests.get(
            f"https://wttr.in/{location}?format=j1",
            headers={"User-Agent": "KisanSalahkar/2.0"},
            timeout=5,
        )
        if resp.status_code == 200:
            w = resp.json()
            # Sum precipitation from 3-day forecast as a daily rate
            total_precip = 0
            days = 0
            for day in w.get("weather", [])[:3]:
                for hour in day.get("hourly", []):
                    total_precip += float(hour.get("precipMM", 0))
                days += 1
            if days > 0:
                daily_avg = total_precip / days
                # Estimate seasonal rainfall: daily_avg × days_in_season
                season_days = {"kharif": 150, "rabi": 150, "zaid": 90}
                live_estimate = int(daily_avg * season_days.get(season, 120))
                # Blend: 60% historical + 40% live estimate for better accuracy
                blended = int(0.6 * base_rainfall + 0.4 * live_estimate)
                result["rainfall"] = blended
                result["source"] = "historical+live"
                result["live_daily_avg_mm"] = round(daily_avg, 1)
    except Exception:
        pass  # Fallback to historical data silently

    return jsonify(success=True, **result)


# ── Soil Testing APIs ────────────────────────────────────────────────────────
@app.route("/api/soil/submit", methods=["POST"])
def api_soil_submit():
    """Submit soil test data manually or simulate a report lookup."""
    data = request.get_json(force=True)
    report_id = data.get("report_id", "").strip()

    if report_id:
        # Lookup existing report
        conn = get_db()
        row = conn.execute("SELECT * FROM soil_reports WHERE report_id = ?", (report_id,)).fetchone()
        conn.close()
        if row:
            return jsonify(success=True, report=dict(row))
        return jsonify(success=False, error="Report not found. Please enter data manually."), 404

    # Manual entry — save and analyze
    report_id = f"SHC-{uuid.uuid4().hex[:8].upper()}"
    user_id = 0
    if "user" in session:
        conn2 = get_db()
        u = conn2.execute("SELECT id FROM users WHERE username=?", (session["user"]["username"],)).fetchone()
        if u: user_id = u["id"]
        conn2.close()

    conn = get_db()
    conn.execute("""
        INSERT INTO soil_reports (report_id, user_id, N, P, K, pH, EC, OC, S, Zn, Fe, Mn, Cu, B, soil_type, location, state)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        report_id, user_id,
        data.get("N", 0), data.get("P", 0), data.get("K", 0),
        data.get("pH", 0), data.get("EC", 0), data.get("OC", 0),
        data.get("S", 0), data.get("Zn", 0), data.get("Fe", 0),
        data.get("Mn", 0), data.get("Cu", 0), data.get("B", 0),
        data.get("soil_type", ""), data.get("location", ""), data.get("state", ""),
    ))
    conn.commit()
    conn.close()

    # Generate recommendations
    n, p, k, ph = float(data.get("N",0)), float(data.get("P",0)), float(data.get("K",0)), float(data.get("pH",6.5))
    recs = []
    if n < 250: recs.append({"nutrient": "Nitrogen", "status": "Low", "action": f"Apply {int(280-n)} kg Urea/ha"})
    elif n > 500: recs.append({"nutrient": "Nitrogen", "status": "High", "action": "Reduce nitrogen fertiliser"})
    else: recs.append({"nutrient": "Nitrogen", "status": "Medium", "action": "Maintain current application"})

    if p < 10: recs.append({"nutrient": "Phosphorus", "status": "Low", "action": "Apply 60 kg DAP/ha"})
    elif p > 25: recs.append({"nutrient": "Phosphorus", "status": "High", "action": "Skip P this season"})
    else: recs.append({"nutrient": "Phosphorus", "status": "Medium", "action": "Apply 30 kg DAP/ha"})

    if k < 120: recs.append({"nutrient": "Potassium", "status": "Low", "action": "Apply 50 kg MOP/ha"})
    elif k > 280: recs.append({"nutrient": "Potassium", "status": "High", "action": "Skip K this season"})
    else: recs.append({"nutrient": "Potassium", "status": "Medium", "action": "Apply 25 kg MOP/ha"})

    if ph < 5.5: recs.append({"nutrient": "pH", "status": "Acidic", "action": "Apply 2-4 quintal lime/ha"})
    elif ph > 8.0: recs.append({"nutrient": "pH", "status": "Alkaline", "action": "Apply 5 quintal gypsum/ha"})
    else: recs.append({"nutrient": "pH", "status": "Normal", "action": "pH is in ideal range"})

    return jsonify(success=True, report_id=report_id, recommendations=recs)


# ── Government Scheme Eligibility API ────────────────────────────────────────
@app.route("/api/schemes/check", methods=["POST"])
def api_schemes_check():
    data = request.get_json(force=True)
    land_ha = float(data.get("land_ha", 0))
    category = data.get("category", "all").lower()  # small, marginal, all
    state = data.get("state", "")

    eligible = []
    for scheme in GOVT_SCHEMES:
        elig = scheme["eligibility"]
        if land_ha <= elig["land_max_ha"]:
            if "all" in elig["categories"] or category in elig["categories"]:
                eligible.append({
                    "id": scheme["id"],
                    "name": scheme["name"],
                    "name_hi": scheme["name_hi"],
                    "desc": scheme["desc"],
                    "desc_hi": scheme["desc_hi"],
                    "benefits": scheme["benefits"],
                    "url": scheme["url"],
                })

    return jsonify(success=True, eligible=eligible, total=len(eligible))


@app.route("/api/schemes")
def api_schemes():
    return jsonify(success=True, schemes=GOVT_SCHEMES)


# ── Mandi Prices API (live from data.gov.in + fallback) ──────────────────────
@app.route("/api/mandi")
def api_mandi():
    crop     = request.args.get("crop", "").strip()
    state    = request.args.get("state", "").strip()
    district = request.args.get("district", "").strip()
    market   = request.args.get("market", "").strip()

    # ── Detail view: specific crop ───────────────────────────────────────
    if crop:
        msp = MSP_DATA.get(crop, 0)
        live = fetch_mandi_from_gov(crop, state, district, market, limit=500)

        if live and live["records"]:
            mandis = []
            seen_states, seen_districts, seen_markets = set(), set(), set()
            for r in live["records"]:
                mandis.append({
                    "name":      r["market"],
                    "state":     r["state"],
                    "district":  r["district"],
                    "variety":   r["variety"],
                    "commodity": r["commodity"],
                    "price":     r["modal_price"],
                    "min_price": r["min_price"],
                    "max_price": r["max_price"],
                    "date":      r["date"],
                })
                seen_states.add(r["state"])
                seen_districts.add(r["district"])
                seen_markets.add(r["market"])

            prices = [m["price"] for m in mandis if m["price"]]
            avg_price = round(sum(prices) / len(prices)) if prices else 0
            min_price = min(prices) if prices else 0
            max_price = max(prices) if prices else 0

            return jsonify(
                success=True, live=True, crop=crop, msp=msp,
                unit="quintal", mandis=mandis,
                fetched_at=live["fetched_at"],
                total_records=live["total"],
                avg_price=avg_price, min_price=min_price, max_price=max_price,
                available_states=sorted(seen_states),
                available_districts=sorted(seen_districts),
                available_markets=sorted(seen_markets),
            )

        # Fallback
        fb = MANDI_FALLBACK.get(crop, [])
        return jsonify(
            success=True, live=False, crop=crop, msp=msp,
            unit="quintal", mandis=fb,
            fetched_at="offline", total_records=len(fb),
            avg_price=0, min_price=0, max_price=0,
            available_states=[], available_districts=[], available_markets=[],
        )

    # ── Overview: all supported crops ────────────────────────────────────
    all_crops = []
    live_data = fetch_mandi_from_gov("", state, district, market, limit=2000)
    is_live = False

    if live_data and live_data["records"]:
        is_live = True
        grouped: dict = {}
        for r in live_data["records"]:
            crop_name = _match_crop_name(r["commodity"])
            if crop_name:
                grouped.setdefault(crop_name, []).append(r)

        for cname in COMMODITY_MAP:
            entries = grouped.get(cname, [])
            if entries:
                prices = [e["modal_price"] for e in entries if e["modal_price"]]
                avg_p = round(sum(prices) / len(prices)) if prices else 0
                num   = len(entries)
            else:
                fb = MANDI_FALLBACK.get(cname, [])
                prices = [m["price"] for m in fb]
                avg_p = round(sum(prices) / len(prices)) if prices else 0
                num   = len(fb)
            all_crops.append({
                "crop": cname, "msp": MSP_DATA.get(cname, 0),
                "unit": "quintal", "avg_price": avg_p, "num_mandis": num,
            })
        fetched_at = live_data["fetched_at"]
    else:
        for cname, fb in MANDI_FALLBACK.items():
            prices = [m["price"] for m in fb]
            avg_p = round(sum(prices) / len(prices)) if prices else 0
            all_crops.append({
                "crop": cname, "msp": MSP_DATA.get(cname, 0),
                "unit": "quintal", "avg_price": avg_p, "num_mandis": len(fb),
            })
        fetched_at = "offline"

    return jsonify(
        success=True, live=is_live, crops=all_crops,
        fetched_at=fetched_at, states=INDIAN_STATES,
    )


@app.route("/api/mandi/districts")
def api_mandi_districts():
    """Get districts for a state from live API data."""
    state = request.args.get("state", "").strip()
    if not state:
        return jsonify(success=False, message="State required")
    data = fetch_mandi_from_gov("", state, "", "", limit=2000)
    districts = set()
    if data and data["records"]:
        for r in data["records"]:
            if r["district"]:
                districts.add(r["district"])
    return jsonify(success=True, state=state, districts=sorted(districts))


@app.route("/api/mandi/markets")
def api_mandi_markets():
    """Get markets for a state+district from live API data."""
    state    = request.args.get("state", "").strip()
    district = request.args.get("district", "").strip()
    if not state:
        return jsonify(success=False, message="State required")
    data = fetch_mandi_from_gov("", state, district, "", limit=2000)
    markets = set()
    if data and data["records"]:
        for r in data["records"]:
            if r["market"]:
                markets.add(r["market"])
    return jsonify(success=True, state=state, district=district, markets=sorted(markets))


@app.route("/api/mandi/nearby")
def api_mandi_nearby():
    """Auto-detect location via IP and return state/district guess + prices."""
    try:
        ip_data = http_requests.get("https://ipapi.co/json/", timeout=5).json()
        region = ip_data.get("region", "")
        city   = ip_data.get("city", "")
        lat    = ip_data.get("latitude", 0)
        lon    = ip_data.get("longitude", 0)

        # Try to match region to our states list
        matched_state = ""
        for s in INDIAN_STATES:
            if s.lower() == region.lower() or region.lower() in s.lower() or s.lower() in region.lower():
                matched_state = s
                break

        # Fetch prices for detected state
        prices_data = None
        if matched_state:
            prices_data = fetch_mandi_from_gov("", matched_state, "", "", limit=500)

        records = []
        if prices_data and prices_data["records"]:
            records = prices_data["records"][:50]   # Limit for nearby view

        return jsonify(
            success=True,
            detected_state=matched_state,
            detected_city=city,
            lat=lat, lon=lon,
            records=records,
            fetched_at=prices_data["fetched_at"] if prices_data else "offline",
        )
    except Exception as exc:
        print(f"[Nearby] {exc}")
        return jsonify(success=False, detected_state="", detected_city="", records=[])


# ── Favorite Mandis API ─────────────────────────────────────────────────────
@app.route("/api/mandi/favorites")
def api_mandi_favorites():
    uid = session.get("user_id")
    if not uid:
        return jsonify(success=False, message="Login required", favorites=[])
    conn = get_db()
    rows = conn.execute(
        "SELECT id, state, district, market, label FROM favorite_mandis WHERE user_id=? ORDER BY created_at DESC", (uid,)
    ).fetchall()
    conn.close()
    favs = [{"id": r[0], "state": r[1], "district": r[2], "market": r[3], "label": r[4]} for r in rows]
    return jsonify(success=True, favorites=favs)


@app.route("/api/mandi/favorites", methods=["POST"])
def api_mandi_favorites_add():
    uid = session.get("user_id")
    if not uid:
        return jsonify(success=False, message="Login required")
    data = request.get_json(force=True)
    state    = data.get("state", "").strip()
    district = data.get("district", "").strip()
    market   = data.get("market", "").strip()
    label    = data.get("label", "").strip() or f"{market}, {district}"
    if not market:
        return jsonify(success=False, message="Market name required")
    try:
        conn = get_db()
        conn.execute(
            "INSERT OR IGNORE INTO favorite_mandis (user_id, state, district, market, label) VALUES (?,?,?,?,?)",
            (uid, state, district, market, label),
        )
        conn.commit()
        conn.close()
        return jsonify(success=True, message="Mandi saved to favorites")
    except Exception as exc:
        return jsonify(success=False, message=str(exc))


@app.route("/api/mandi/favorites/<int:fav_id>", methods=["DELETE"])
def api_mandi_favorites_del(fav_id):
    uid = session.get("user_id")
    if not uid:
        return jsonify(success=False, message="Login required")
    conn = get_db()
    conn.execute("DELETE FROM favorite_mandis WHERE id=? AND user_id=?", (fav_id, uid))
    conn.commit()
    conn.close()
    return jsonify(success=True, message="Removed")


@app.route("/api/mandi/refresh")
def api_mandi_refresh():
    """Clear mandi cache so next request pulls fresh data."""
    _mandi_cache.clear()
    return jsonify(success=True, message="Cache cleared")


# ── Disease Detection API ────────────────────────────────────────────────────
@app.route("/api/disease/detect", methods=["POST"])
def api_disease_detect():
    """Accept image or description and return disease diagnosis.
    If CNN model is loaded AND image is provided → CNN prediction.
    Otherwise → text-based keyword matching (fallback).
    """
    data = request.get_json(force=True)
    image_data = data.get("image", "")
    symptoms_text = data.get("symptoms", "").lower()
    crop_name = data.get("crop", "").strip()

    results = []
    used_cnn = False

    # ── CNN Image-based detection (primary) ──
    if image_data and DISEASE_MODEL and DISEASE_LABELS:
        try:
            import tensorflow as tf

            # Decode base64 image
            if "," in image_data:
                image_data = image_data.split(",")[1]  # Remove data:image/...;base64, prefix

            img_bytes = base64.b64decode(image_data)
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            img = img.resize((DISEASE_IMG_SIZE, DISEASE_IMG_SIZE))

            # Preprocess for MobileNetV2
            img_array = np.array(img, dtype=np.float32)
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

            # Predict (Keras or TFLite)
            if DISEASE_USE_TFLITE:
                input_details = DISEASE_MODEL.get_input_details()
                output_details = DISEASE_MODEL.get_output_details()
                # TFLite expects [0,1] float or uint8 — apply simple scaling
                img_array = img_array / 127.5 - 1.0
                DISEASE_MODEL.set_tensor(input_details[0]['index'], img_array.astype(np.float32))
                DISEASE_MODEL.invoke()
                predictions = DISEASE_MODEL.get_tensor(output_details[0]['index'])[0]
            else:
                import tensorflow as tf
                img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
                predictions = DISEASE_MODEL.predict(img_array, verbose=0)[0]

            # Get top 5 predictions
            top_indices = np.argsort(predictions)[::-1][:5]

            for idx in top_indices:
                idx_str = str(idx)
                if idx_str not in DISEASE_LABELS:
                    continue
                label = DISEASE_LABELS[idx_str]
                confidence = float(predictions[idx]) * 100

                if confidence < 5:  # Skip very low confidence
                    continue

                # Filter by crop if specified
                if crop_name and crop_name.lower() not in label.get("crop", "").lower():
                    # Still include but lower confidence
                    confidence *= 0.5

                severity = "low"
                if not label.get("is_healthy", False):
                    if confidence > 70: severity = "high"
                    elif confidence > 40: severity = "medium"

                results.append({
                    "disease": label["en"],
                    "disease_hi": label["hi"],
                    "crop": label.get("crop", "Unknown"),
                    "type": "CNN Detection",
                    "symptoms": "",
                    "symptoms_hi": "",
                    "treatment": label.get("treatment_en", "Consult your local agricultural extension officer."),
                    "treatment_hi": label.get("treatment_hi", "अपने स्थानीय कृषि विस्तार अधिकारी से संपर्क करें।"),
                    "severity": severity if not label.get("is_healthy") else "none",
                    "confidence": round(confidence, 1),
                    "is_healthy": label.get("is_healthy", False),
                    "source": "cnn",
                })

            used_cnn = True
            print(f"🔬 CNN prediction: {results[0]['disease']} ({results[0]['confidence']:.1f}%)" if results else "🔬 CNN: no confident match")

        except Exception as e:
            print(f"⚠️ CNN prediction error: {e}")
            used_cnn = False

    # ── Text-based detection (fallback or supplement) ──
    if not used_cnn or (symptoms_text and not results):
        text_results = []
        if symptoms_text:
            for disease_name, info in DISEASE_DB.items():
                score = 0
                keywords = info["symptoms"].lower().split()
                for word in symptoms_text.split():
                    if word in " ".join(keywords):
                        score += 1
                if crop_name and crop_name.lower() in info["crop"].lower():
                    score += 3
                if score > 0:
                    text_results.append({
                        "disease": disease_name,
                        "disease_hi": "",
                        "crop": info["crop"],
                        "type": info["type"],
                        "symptoms": info["symptoms"],
                        "symptoms_hi": info["symptoms_hi"],
                        "treatment": info["treatment"],
                        "treatment_hi": info["treatment_hi"],
                        "severity": info["severity"],
                        "confidence": min(95, score * 15 + 40),
                        "is_healthy": False,
                        "source": "text",
                    })

        if not text_results:
            for disease_name, info in DISEASE_DB.items():
                if not crop_name or crop_name.lower() in info["crop"].lower():
                    text_results.append({
                        "disease": disease_name,
                        "disease_hi": "",
                        "crop": info["crop"],
                        "type": info["type"],
                        "symptoms": info["symptoms"],
                        "symptoms_hi": info["symptoms_hi"],
                        "treatment": info["treatment"],
                        "treatment_hi": info["treatment_hi"],
                        "severity": info["severity"],
                        "confidence": 60,
                        "is_healthy": False,
                        "source": "text",
                    })

        # If CNN gave results, add text results only if they're different
        if results:
            existing_names = {r["disease"].lower() for r in results}
            for tr in text_results:
                if tr["disease"].lower() not in existing_names:
                    results.append(tr)
        else:
            results = text_results

    results.sort(key=lambda x: x["confidence"], reverse=True)
    return jsonify(
        success=True,
        diseases=results[:5],
        model_used="cnn" if used_cnn else "text",
        cnn_available=DISEASE_MODEL is not None,
    )


@app.route("/api/disease/list")
def api_disease_list():
    crop = request.args.get("crop", "")
    if crop and crop in CROP_DISEASES:
        return jsonify(success=True, crop=crop, diseases=CROP_DISEASES[crop])
    return jsonify(success=True, all_diseases=CROP_DISEASES)


# ── Community Forum APIs ─────────────────────────────────────────────────────
@app.route("/api/forum/posts")
def api_forum_posts():
    category = request.args.get("category", "")
    conn = get_db()
    if category:
        posts = conn.execute(
            "SELECT * FROM forum_posts WHERE category = ? ORDER BY created_at DESC LIMIT 50",
            (category,)
        ).fetchall()
    else:
        posts = conn.execute(
            "SELECT * FROM forum_posts ORDER BY created_at DESC LIMIT 50"
        ).fetchall()

    result = []
    for p in posts:
        replies = conn.execute(
            "SELECT COUNT(*) as cnt FROM forum_replies WHERE post_id = ?", (p["id"],)
        ).fetchone()
        result.append({
            "id": p["id"], "username": p["username"], "title": p["title"],
            "content": p["content"], "category": p["category"],
            "likes": p["likes"], "replies": replies["cnt"],
            "created_at": p["created_at"],
        })
    conn.close()
    return jsonify(success=True, posts=result)

@app.route("/api/forum/post", methods=["POST"])
def api_forum_create_post():
    if "user" not in session:
        return jsonify(success=False, error="Login required"), 401
    data = request.get_json(force=True)
    title = sanitize(data.get("title", "").strip())
    content = sanitize(data.get("content", "").strip())
    category = sanitize(data.get("category", "general"))

    if not title or not content:
        return jsonify(success=False, error="Title and content required"), 400

    conn = get_db()
    u = conn.execute("SELECT id FROM users WHERE username=?", (session["user"]["username"],)).fetchone()
    user_id = u["id"] if u else 0
    conn.execute(
        "INSERT INTO forum_posts (user_id, username, title, content, category) VALUES (?, ?, ?, ?, ?)",
        (user_id, session["user"]["name"], title, content, category)
    )
    conn.commit()
    conn.close()
    return jsonify(success=True)

@app.route("/api/forum/reply", methods=["POST"])
def api_forum_reply():
    if "user" not in session:
        return jsonify(success=False, error="Login required"), 401
    data = request.get_json(force=True)
    post_id = data.get("post_id")
    content = sanitize(data.get("content", "").strip())

    if not content:
        return jsonify(success=False, error="Reply content required"), 400

    conn = get_db()
    u = conn.execute("SELECT id FROM users WHERE username=?", (session["user"]["username"],)).fetchone()
    user_id = u["id"] if u else 0
    conn.execute(
        "INSERT INTO forum_replies (post_id, user_id, username, content) VALUES (?, ?, ?, ?)",
        (post_id, user_id, session["user"]["name"], content)
    )
    conn.commit()

    replies = conn.execute(
        "SELECT * FROM forum_replies WHERE post_id = ? ORDER BY created_at ASC", (post_id,)
    ).fetchall()
    conn.close()

    return jsonify(success=True, replies=[dict(r) for r in replies])

@app.route("/api/forum/like", methods=["POST"])
def api_forum_like():
    data = request.get_json(force=True)
    post_id = data.get("post_id")
    conn = get_db()
    conn.execute("UPDATE forum_posts SET likes = likes + 1 WHERE id = ?", (post_id,))
    conn.commit()
    conn.close()
    return jsonify(success=True)


# ── SMS Alert APIs ───────────────────────────────────────────────────────────
@app.route("/api/alerts/subscribe", methods=["POST"])
def api_alerts_subscribe():
    if "user" not in session:
        return jsonify(success=False, error="Login required"), 401
    data = request.get_json(force=True)
    phone = data.get("phone", "").strip()
    alert_type = data.get("alert_type", "all")
    city = data.get("city", "").strip()

    if not phone or len(phone) < 10:
        return jsonify(success=False, error="Valid phone number required"), 400

    conn = get_db()
    u = conn.execute("SELECT id FROM users WHERE username=?", (session["user"]["username"],)).fetchone()
    user_id = u["id"] if u else 0

    # Update user phone
    conn.execute("UPDATE users SET phone=? WHERE id=?", (phone, user_id))

    conn.execute(
        "INSERT INTO sms_subscriptions (user_id, phone, alert_type, city) VALUES (?, ?, ?, ?)",
        (user_id, phone, alert_type, city)
    )
    conn.commit()
    conn.close()
    return jsonify(success=True, message="Alert subscription active! You will receive weather & crop alerts.")

@app.route("/api/alerts/status")
def api_alerts_status():
    if "user" not in session:
        return jsonify(success=False), 401
    conn = get_db()
    u = conn.execute("SELECT id FROM users WHERE username=?", (session["user"]["username"],)).fetchone()
    if not u:
        conn.close()
        return jsonify(success=True, subscriptions=[])
    subs = conn.execute("SELECT * FROM sms_subscriptions WHERE user_id=? AND active=1", (u["id"],)).fetchall()
    conn.close()
    return jsonify(success=True, subscriptions=[dict(s) for s in subs])

@app.route("/api/alerts/unsubscribe", methods=["POST"])
def api_alerts_unsubscribe():
    data = request.get_json(force=True)
    sub_id = data.get("id")
    conn = get_db()
    conn.execute("UPDATE sms_subscriptions SET active=0 WHERE id=?", (sub_id,))
    conn.commit()
    conn.close()
    return jsonify(success=True)


# ── IoT Sensor APIs ──────────────────────────────────────────────────────────
IOT_API_KEY = os.environ.get("IOT_API_KEY", "kisan-iot-2026")

# Sensor thresholds for alerts
SENSOR_THRESHOLDS = {
    "moisture_low": 25,      # Below 25% → irrigate
    "moisture_high": 85,     # Above 85% → waterlogged
    "temp_high": 42,         # Heat stress
    "temp_low": 4,           # Frost risk
    "ph_low": 4.5,           # Too acidic
    "ph_high": 8.5,          # Too alkaline
    "humidity_high": 90,     # Fungal disease risk
    "battery_low": 6.0,      # Battery needs charging
}


@app.route("/api/sensor/push", methods=["POST"])
def api_sensor_push():
    """Receive sensor data from ESP32 via WiFi."""
    data = request.get_json(force=True)

    # Validate API key
    if data.get("api_key") != IOT_API_KEY:
        return jsonify(success=False, error="Invalid API key"), 403

    device_id   = data.get("device_id", "unknown")
    temperature = float(data.get("temperature", 0))
    humidity    = float(data.get("humidity", 0))
    moisture    = float(data.get("moisture", 0))
    ph          = float(data.get("ph", 0))
    battery_v   = float(data.get("battery_v", 0))
    wifi_rssi   = int(data.get("wifi_rssi", 0))
    moisture_raw = int(data.get("moisture_raw", 0))
    ph_raw       = int(data.get("ph_raw", 0))
    ph_voltage   = float(data.get("ph_voltage", 0))

    # Reject invalid readings (sensor error values)
    if temperature == -999 or humidity == -999:
        return jsonify(success=False, error="Invalid sensor reading"), 400

    # Handle disconnected sensors: ESP32 sends -1 (unplugged) or -2 (probe in air)
    ph_in_air = (ph == -2)
    if moisture < 0:
        moisture = None
    if ph < 0:
        ph = None

    # Store in database
    conn = get_db()
    conn.execute(
        """INSERT INTO sensor_readings
           (device_id, temperature, humidity, moisture, ph,
            moisture_raw, ph_raw, ph_voltage, battery_v, wifi_rssi)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (device_id, temperature, humidity, moisture, ph,
         moisture_raw, ph_raw, ph_voltage, battery_v, wifi_rssi)
    )
    conn.commit()
    conn.close()

    # Check thresholds and generate alerts
    alerts = []
    if moisture is not None:
        if moisture < SENSOR_THRESHOLDS["moisture_low"]:
            alerts.append(f"⚠️ Soil moisture LOW ({moisture:.0f}%) — irrigate now!")
        if moisture > SENSOR_THRESHOLDS["moisture_high"]:
            alerts.append(f"⚠️ Soil moisture HIGH ({moisture:.0f}%) — possible waterlogging!")
    else:
        alerts.append("🔌 Soil moisture sensor not connected (check GPIO 34 wiring)")
    if temperature > SENSOR_THRESHOLDS["temp_high"]:
        alerts.append(f"🔥 Extreme heat ({temperature:.1f}°C) — protect crops!")
    if temperature < SENSOR_THRESHOLDS["temp_low"]:
        alerts.append(f"❄️ Frost risk ({temperature:.1f}°C) — cover saplings!")
    if ph is not None:
        if ph < SENSOR_THRESHOLDS["ph_low"]:
            alerts.append(f"⚗️ Soil too acidic (pH {ph:.1f}) — apply lime")
        if ph > SENSOR_THRESHOLDS["ph_high"]:
            alerts.append(f"⚗️ Soil too alkaline (pH {ph:.1f}) — apply gypsum")
    elif ph_in_air:
        alerts.append("🌬️ pH probe is in air — dip it in soil or water for a reading")
    else:
        alerts.append("🔌 pH sensor not connected (check GPIO 35 wiring)")
    if humidity > SENSOR_THRESHOLDS["humidity_high"]:
        alerts.append(f"🍄 High humidity ({humidity:.0f}%) — fungal disease risk!")
    if battery_v > 0 and battery_v < SENSOR_THRESHOLDS["battery_low"]:
        alerts.append(f"🔋 Battery low ({battery_v:.1f}V) — recharge soon")

    return jsonify(
        success=True,
        message="Data received",
        device_id=device_id,
        alerts=alerts
    )


@app.route("/api/sensor/latest")
def api_sensor_latest():
    """Get the most recent sensor reading (for dashboard auto-refresh)."""
    device_id = request.args.get("device_id", "ESP32-FARM-001")
    conn = get_db()
    row = conn.execute(
        "SELECT * FROM sensor_readings WHERE device_id=? ORDER BY created_at DESC LIMIT 1",
        (device_id,)
    ).fetchone()
    conn.close()

    if not row:
        return jsonify(success=False, error="No sensor data yet. Connect your ESP32."), 404

    reading = dict(row)

    # ── Staleness check: is the ESP32 still actively sending data? ──
    # ESP32 sends every 10s, so >30s of silence = device offline
    # NOTE: SQLite CURRENT_TIMESTAMP stores UTC, so compare with UTC
    STALE_SECONDS = 30
    created_str = reading.get("created_at", "")
    try:
        last_ts = datetime.strptime(created_str, "%Y-%m-%d %H:%M:%S")
        age_seconds = (datetime.utcnow() - last_ts).total_seconds()
    except (ValueError, TypeError):
        age_seconds = 99999  # treat parse errors as stale
    reading["device_online"] = age_seconds <= STALE_SECONDS
    reading["data_age_seconds"] = int(age_seconds)

    # Generate real-time advisories based on sensor data
    advisories = []
    m = reading["moisture"]
    t = reading["temperature"]
    h = reading["humidity"]
    p = reading["ph"]

    # Sensor disconnected detection (NULL in DB = sensor sent -1 or -2)
    moisture_connected = m is not None
    ph_connected = p is not None
    # Distinguish "module unplugged" vs "probe in air":
    # If pH is NULL but ph_raw > 50, the module is connected but probe is in air
    ph_in_air = (not ph_connected) and (reading.get("ph_raw") or 0) > 50
    if m is None:
        m = 0  # default for display
        reading["moisture"] = 0
    if p is None:
        p = 7.0  # neutral default for display
        reading["ph"] = 7.0

    # Add raw debug info to reading
    reading["moisture_connected"] = moisture_connected
    reading["ph_connected"] = ph_connected
    reading["ph_in_air"] = ph_in_air

    if not moisture_connected:
        advisories.append({"type": "warning", "icon": "plug-circle-exclamation", "msg": f"Soil moisture sensor not connected (raw ADC: {reading.get('moisture_raw', 0)}). Check GPIO 34 wiring.", "msg_hi": f"मिट्टी नमी सेंसर कनेक्ट नहीं है (raw ADC: {reading.get('moisture_raw', 0)})। GPIO 34 वायरिंग जांचें।"})
    elif m < 25:
        advisories.append({"type": "danger", "icon": "droplet-slash", "msg": f"Soil moisture critically low ({m:.0f}%). Irrigate immediately!", "msg_hi": f"मिट्टी की नमी बहुत कम ({m:.0f}%)। तुरंत सिंचाई करें!"})
    elif m < 40:
        advisories.append({"type": "warning", "icon": "droplet", "msg": f"Soil moisture low ({m:.0f}%). Consider irrigating today.", "msg_hi": f"मिट्टी की नमी कम ({m:.0f}%)। आज सिंचाई करें।"})
    elif m > 85:
        advisories.append({"type": "warning", "icon": "water", "msg": f"Soil too wet ({m:.0f}%). Ensure proper drainage.", "msg_hi": f"मिट्टी बहुत गीली ({m:.0f}%)। जल निकासी सुनिश्चित करें।"})
    else:
        advisories.append({"type": "success", "icon": "check-circle", "msg": f"Soil moisture optimal ({m:.0f}%). No irrigation needed.", "msg_hi": f"मिट्टी की नमी सही ({m:.0f}%)। सिंचाई की ज़रूरत नहीं।"})

    if t > 40:
        advisories.append({"type": "danger", "icon": "temperature-high", "msg": f"Extreme heat ({t:.1f}°C). Use mulch, irrigate in evening.", "msg_hi": f"भीषण गर्मी ({t:.1f}°C)। मल्चिंग करें, शाम को सिंचाई।"})
    elif t < 5:
        advisories.append({"type": "danger", "icon": "snowflake", "msg": f"Frost risk ({t:.1f}°C). Cover sensitive crops.", "msg_hi": f"पाला पड़ने का खतरा ({t:.1f}°C)। संवेदनशील फसलें ढकें।"})

    if h > 85:
        advisories.append({"type": "warning", "icon": "virus", "msg": f"High humidity ({h:.0f}%) — fungal disease risk. Avoid spraying.", "msg_hi": f"उच्च आर्द्रता ({h:.0f}%) — फफूंद रोग का खतरा। छिड़काव न करें।"})

    if not ph_connected:
        ph_v_display = reading.get('ph_voltage') or 0
        if ph_in_air:
            advisories.append({"type": "warning", "icon": "wind", "msg": "pH probe is in air — dip it in soil or water for a reading.", "msg_hi": "pH प्रोब हवा में है — सही रीडिंग के लिए मिट्टी या पानी में डुबोएं।"})
        else:
            advisories.append({"type": "warning", "icon": "plug-circle-exclamation", "msg": f"pH sensor not connected (raw ADC: {reading.get('ph_raw', 0)}, {ph_v_display:.3f}V). Check GPIO 35 wiring.", "msg_hi": f"pH सेंसर कनेक्ट नहीं है। GPIO 35 वायरिंग जांचें।"})
    elif p < 5.5:
        advisories.append({"type": "warning", "icon": "flask-vial", "msg": f"Soil acidic (pH {p:.1f}). Apply 2-4 q lime/ha.", "msg_hi": f"मिट्टी अम्लीय (pH {p:.1f})। 2-4 क्विंटल चूना/हे. डालें।"})
    elif p > 8.0:
        advisories.append({"type": "warning", "icon": "flask-vial", "msg": f"Soil alkaline (pH {p:.1f}). Apply gypsum.", "msg_hi": f"मिट्टी क्षारीय (pH {p:.1f})। जिप्सम डालें।"})

    reading["advisories"] = advisories
    return jsonify(success=True, reading=reading)


@app.route("/api/sensor/history")
def api_sensor_history():
    """Get sensor reading history for charts (last 24h by default)."""
    device_id = request.args.get("device_id", "ESP32-FARM-001")
    hours = int(request.args.get("hours", 24))

    conn = get_db()
    rows = conn.execute(
        """SELECT * FROM sensor_readings
           WHERE device_id=? AND created_at >= datetime('now', ?)
           ORDER BY created_at ASC""",
        (device_id, f"-{hours} hours")
    ).fetchall()
    conn.close()

    history = [dict(r) for r in rows]
    return jsonify(success=True, history=history, count=len(history))


@app.route("/api/sensor/camera", methods=["POST"])
def api_sensor_camera():
    """Receive image from ESP32-CAM for disease detection.
    If CNN model is loaded, runs prediction automatically.
    """
    data = request.get_json(force=True)

    if data.get("api_key") != IOT_API_KEY:
        return jsonify(success=False, error="Invalid API key"), 403

    device_id  = data.get("device_id", "unknown")
    image_data = data.get("image", "")

    if not image_data:
        return jsonify(success=False, error="No image data"), 400

    diagnosis = "Image received — manual review needed."
    confidence = 0
    treatment = ""

    # Run CNN prediction if model is loaded
    if DISEASE_MODEL and DISEASE_LABELS:
        try:
            import tensorflow as tf

            if "," in image_data:
                image_data_clean = image_data.split(",")[1]
            else:
                image_data_clean = image_data

            img_bytes = base64.b64decode(image_data_clean)
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            img = img.resize((DISEASE_IMG_SIZE, DISEASE_IMG_SIZE))

            img_array = np.array(img, dtype=np.float32)
            img_array = np.expand_dims(img_array, axis=0)

            if DISEASE_USE_TFLITE:
                input_details = DISEASE_MODEL.get_input_details()
                output_details = DISEASE_MODEL.get_output_details()
                img_array = img_array / 127.5 - 1.0
                DISEASE_MODEL.set_tensor(input_details[0]['index'], img_array.astype(np.float32))
                DISEASE_MODEL.invoke()
                predictions = DISEASE_MODEL.get_tensor(output_details[0]['index'])[0]
            else:
                img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
                predictions = DISEASE_MODEL.predict(img_array, verbose=0)[0]
            top_idx = int(np.argmax(predictions))
            confidence = float(predictions[top_idx]) * 100

            label = DISEASE_LABELS.get(str(top_idx), {})
            diagnosis = label.get("en", "Unknown")
            treatment = label.get("treatment_en", "")
            print(f"📸 ESP32-CAM → {diagnosis} ({confidence:.1f}%)")

        except Exception as e:
            print(f"⚠️ Camera CNN error: {e}")
            diagnosis = f"Image received — CNN error: {str(e)[:50]}"

    # Store image capture record
    conn = get_db()
    conn.execute(
        "INSERT INTO sensor_images (device_id, image_data, diagnosis, confidence) VALUES (?, ?, ?, ?)",
        (device_id, image_data[:200] + "...", diagnosis, round(confidence, 1))
    )
    conn.commit()
    conn.close()

    return jsonify(
        success=True,
        message="Image analyzed" if confidence > 0 else "Image received",
        diagnosis=diagnosis,
        confidence=round(confidence, 1),
        treatment=treatment,
    )


@app.route("/api/sensor/autofill")
def api_sensor_autofill():
    """Get latest sensor data formatted for crop prediction form auto-fill."""
    device_id = request.args.get("device_id", "ESP32-FARM-001")
    conn = get_db()
    row = conn.execute(
        "SELECT * FROM sensor_readings WHERE device_id=? ORDER BY created_at DESC LIMIT 1",
        (device_id,)
    ).fetchone()
    conn.close()

    if not row:
        return jsonify(success=False, error="No sensor data available"), 404

    temperature = round(row["temperature"], 1)
    humidity = round(row["humidity"], 1)
    ph = round(row["ph"], 1) if row["ph"] is not None else 7.0
    moisture = round(row["moisture"], 1) if row["moisture"] is not None else 50.0

    # Map sensor readings to crop prediction form fields
    autofill = {
        "ph":          ph,
        "temperature": temperature,
        "humidity":    humidity,
        "moisture":    moisture,
        "timestamp":   row["created_at"],
        "device_id":   row["device_id"],
    }

    # Predict NPK from sensor values using AI model
    if NPK_MODEL and NPK_SCALER:
        try:
            X = np.array([[temperature, humidity, ph, moisture]])
            X_scaled = NPK_SCALER.transform(X)
            pred = NPK_MODEL.predict(X_scaled)[0]
            autofill["N"] = int(round(float(pred[0])))
            autofill["P"] = int(round(float(pred[1])))
            autofill["K"] = int(round(float(pred[2])))
            autofill["npk_source"] = "ai"
        except Exception as e:
            print(f"⚠️  NPK prediction failed: {e}")
            autofill["npk_source"] = "unavailable"
    else:
        autofill["npk_source"] = "unavailable"

    return jsonify(success=True, autofill=autofill)


# ── Boot ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    init_db()
    app.run(debug=True, host="0.0.0.0", port=5000)
