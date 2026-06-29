<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,14,20,25,30&height=280&section=header&text=🌾%20Kisan%20Salahkar&fontSize=62&animation=twinkling&fontAlignY=40&desc=AI-Powered%20Agriculture%20Advisory%20System%20for%20Indian%20Farmers&descAlignY=62&descSize=18&fontColor=fff" width="100%"/>

<br/>

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.0+-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge&logo=opensourceinitiative&logoColor=white)](LICENSE)

<br/>

[![GitHub repo](https://img.shields.io/badge/GitHub-Kisan--Salahkar-181717?style=flat-square&logo=github)](https://github.com/nitishchauhan002/Kisan-Salahkar)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Nitish%20Kumar%20Singh-0A66C2?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/nitish-kumar-singh-4802792bb/)
[![Made in India](https://img.shields.io/badge/Made%20in-India%20🇮🇳-FF9933?style=flat-square)](https://github.com/nitishchauhan002)

</div>

---

## 🌟 What is Kisan Salahkar?

**Kisan Salahkar** (किसान सलाहकार) is a full-stack, AI-powered agriculture advisory web application built for Indian farmers. It integrates machine learning, real-time IoT sensor data, live government APIs, and bilingual (Hindi + English) support into a single platform — helping farmers make smarter decisions from soil to market.

> *"Salahkar" means "Advisor" in Hindi — because every farmer deserves a smart advisor in their pocket.*

---

## ✨ Feature Overview

<div align="center">

| Module | Description |
|:---|:---|
| 🌱 **Crop Recommendation** | ML model predicts best crops based on NPK, pH, rainfall, temperature, humidity & season |
| 🔬 **Disease Detection** | CNN (MobileNetV2) + text-based fallback to identify plant diseases from images |
| 🌦️ **Live Weather** | Real-time weather + 3-day forecast via wttr.in with farming advisories |
| 🧪 **Soil Testing** | Submit/lookup soil reports; auto-generates NPK + pH fertiliser recommendations |
| 📈 **Mandi Prices** | Live crop prices from data.gov.in Agmarknet API with fallback data |
| 🏛️ **Govt. Schemes** | 8 major schemes (PM-KISAN, PMFBY, KCC, e-NAM, etc.) with eligibility checker |
| 🤖 **IoT Dashboard** | ESP32 sensor integration — real-time soil moisture, pH, temperature, humidity |
| 📸 **ESP32-CAM** | Auto-diagnosis of plant diseases from camera captures |
| 💬 **Community Forum** | Farmers can post questions, reply, and like posts by category |
| 📱 **SMS Alerts** | Subscribe to weather & crop alerts via phone (Twilio-ready) |
| 🔐 **Auth System** | Secure signup/login with bcrypt password hashing + SQLite storage |
| 🌐 **Bilingual UI** | Full Hindi (हिंदी) + English support throughout |

</div>

---

## 🏗️ Architecture

```
Kisan-Salahkar/
│
├── app.py                     # 🚀 Main Flask application (all routes + APIs)
├── requirements.txt           # 📦 Python dependencies
├── users.db                   # 🗄️  SQLite database (auto-created)
│
├── model/                     # 🤖 ML Model artifacts
│   ├── crop_model.pkl         # Crop recommendation model (Scikit-learn)
│   ├── scaler.pkl             # Feature scaler
│   ├── label_encoder.pkl      # Crop label encoder
│   ├── features.pkl           # Feature column names
│   ├── disease_model.keras    # CNN disease detection (TensorFlow)
│   ├── disease_model.tflite   # Lightweight TFLite version
│   ├── disease_labels.json    # Disease class labels (EN + HI + treatments)
│   ├── npk_model.pkl          # NPK prediction from IoT sensors
│   ├── state_crops.json       # State-wise crop suitability map
│   ├── state_district_crops.json  # District-level crop data
│   ├── crop_diseases.json     # Crop → disease mapping
│   └── state_rainfall.json    # Seasonal rainfall averages by state
│
├── templates/                 # 🌐 Jinja2 HTML templates
│   ├── index.html             # Landing / Login page
│   ├── dashboard.html         # Main dashboard
│   ├── crop.html              # Crop recommendation
│   ├── disease.html           # Disease detection
│   ├── weather.html           # Weather forecast
│   ├── soil.html              # Soil test
│   ├── mandi.html             # Market prices
│   ├── schemes.html           # Govt. schemes
│   ├── iot.html               # IoT sensor dashboard
│   ├── forum.html             # Community forum
│   ├── alerts.html            # SMS alerts
│   └── info.html              # Crop info & tips
│
├── static/                    # 🎨 Static assets
│   ├── manifest.json          # PWA manifest
│   └── sw.js                  # Service Worker (offline support)
│
├── train_model.py             # Training script: crop recommendation
├── train_disease_model.py     # Training script: disease CNN
├── train_npk_model.py         # Training script: NPK predictor
├── live_receiver.py           # IoT live data receiver
└── monitor_esp32.py           # ESP32 connection monitor
```

---

## 🤖 ML Models Explained

### 🌱 Crop Recommendation
- **Input features:** N, P, K (soil nutrients), Temperature, Humidity, pH, Rainfall
- **Smart boosting:** State-level (+15%) and district-level (+25%) crop suitability scoring
- **Season filtering:** Auto-detects Kharif / Rabi / Zaid season and filters unsuitable crops
- **Output:** Top 5 crops with confidence %, growing tips, MSP price, and known diseases

### 🔬 Plant Disease Detection
- **Primary:** MobileNetV2 CNN (Keras / TFLite) — image classification from camera upload
- **Fallback:** Keyword-based symptom matching from curated disease database
- **Covers:** 8+ common diseases — Bacterial Leaf Blight, Late Blight, Fusarium Wilt, Rust, Bollworm, and more
- **Bilingual:** Symptoms and treatments in both Hindi and English

### 💧 NPK Predictor (IoT)
- **Input:** Sensor readings — Temperature, Humidity, pH, Soil Moisture
- **Output:** Predicted N, P, K values for crop prediction form auto-fill

---

## 🛰️ IoT Integration (ESP32)

```
ESP32 Device  ──WiFi──▶  /api/sensor/push   (JSON payload every 10s)
ESP32-CAM     ──WiFi──▶  /api/sensor/camera (base64 image → CNN diagnosis)

Sensors connected to ESP32:
  GPIO 34 → Soil Moisture Sensor (ADC)
  GPIO 35 → pH Sensor (analog voltage)
  DHT22   → Temperature & Humidity
  ADS1115 → ADC amplifier (optional)
```

**Real-time alerts generated for:**
- Soil moisture < 25% → Irrigate immediately
- Temperature > 42°C → Heat stress warning
- pH < 4.5 or > 8.5 → Soil amendment needed
- Humidity > 90% → Fungal disease risk
- Battery voltage < 6V → Recharge warning

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/nitishchauhan002/Kisan-Salahkar.git
cd Kisan-Salahkar

# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
```

The app will be live at `http://localhost:5000`

### Train the ML Models

```bash
# Train crop recommendation model
python train_model.py

# Train disease detection CNN
python train_disease_model.py

# Train NPK predictor (requires IoT sensor dataset)
python train_npk_model.py
```

### Environment Variables (Optional)

| Variable | Description | Default |
|:---|:---|:---|
| `SECRET_KEY` | Flask session secret | `kisan-salahkar-secret-2026` |
| `DATA_GOV_API_KEY` | data.gov.in API key for live mandi prices | Public sample key |
| `IOT_API_KEY` | Auth key for ESP32 sensor data push | `kisan-iot-2026` |
| `CROP_MODEL_FILE` | Custom crop model filename | `crop_model.pkl` |

---

## 📡 API Reference

<details>
<summary><b>🌱 Crop & Prediction APIs</b></summary>

| Method | Endpoint | Description |
|:---|:---|:---|
| `POST` | `/api/predict` | Crop recommendation with ML |
| `GET` | `/api/predictions/history` | Last 10 predictions for logged-in user |
| `GET` | `/api/state-crops?state=` | State-wise suitable crops |
| `GET` | `/api/state-districts?state=` | Districts for a state |
| `GET` | `/api/model/info` | Active model file info |
| `POST` | `/api/model/reload` | Hot-reload ML model |

</details>

<details>
<summary><b>🌦️ Weather & Rainfall APIs</b></summary>

| Method | Endpoint | Description |
|:---|:---|:---|
| `GET` | `/api/weather?city=` | Live weather + 3-day forecast |
| `GET` | `/api/rainfall/estimate?state=&season=` | Seasonal rainfall estimate (historical + live blend) |

</details>

<details>
<summary><b>📈 Mandi Price APIs</b></summary>

| Method | Endpoint | Description |
|:---|:---|:---|
| `GET` | `/api/mandi` | All crops overview |
| `GET` | `/api/mandi?crop=Rice&state=Punjab` | Filtered live prices |
| `GET` | `/api/mandi/districts?state=` | Districts for a state |
| `GET` | `/api/mandi/markets?state=&district=` | Markets in a district |
| `GET` | `/api/mandi/nearby` | Auto-detect location + nearby prices |
| `GET` | `/api/mandi/refresh` | Clear price cache |

</details>

<details>
<summary><b>🔬 Disease & Soil APIs</b></summary>

| Method | Endpoint | Description |
|:---|:---|:---|
| `POST` | `/api/disease/detect` | Image/text-based disease detection |
| `GET` | `/api/disease/list?crop=` | Disease list by crop |
| `POST` | `/api/soil/submit` | Submit / lookup soil test report |

</details>

<details>
<summary><b>🛰️ IoT Sensor APIs</b></summary>

| Method | Endpoint | Description |
|:---|:---|:---|
| `POST` | `/api/sensor/push` | Receive ESP32 sensor data |
| `GET` | `/api/sensor/latest` | Latest reading + advisories |
| `GET` | `/api/sensor/history?hours=24` | Historical readings for charts |
| `POST` | `/api/sensor/camera` | Receive ESP32-CAM image + auto-diagnose |
| `GET` | `/api/sensor/autofill` | Sensor data for crop form auto-fill |

</details>

<details>
<summary><b>👥 Forum, Schemes & Alerts</b></summary>

| Method | Endpoint | Description |
|:---|:---|:---|
| `GET` | `/api/forum/posts` | Get all / category posts |
| `POST` | `/api/forum/post` | Create new post |
| `POST` | `/api/forum/reply` | Reply to a post |
| `POST` | `/api/schemes/check` | Check govt scheme eligibility |
| `GET` | `/api/schemes` | All 8 schemes |
| `POST` | `/api/alerts/subscribe` | Subscribe to SMS alerts |

</details>

---

## 🏛️ Government Schemes Covered

| Scheme | Benefit |
|:---|:---|
| 🌾 PM-KISAN Samman Nidhi | ₹6,000/year direct income support |
| 🛡️ PM Fasal Bima Yojana | Crop insurance at 1.5–5% premium |
| 💳 Kisan Credit Card | Loans up to ₹3 lakh at 4% interest |
| 🧪 Soil Health Card | Free soil testing every 2 years |
| 👴 PM Kisan Maandhan Yojana | ₹3,000/month pension after age 60 |
| 🏪 e-NAM | Online mandi for transparent crop selling |
| 🌿 Paramparagat Krishi Vikas | ₹50,000/ha for organic farming |
| 💧 PM Krishi Sinchai Yojana | 55% subsidy on drip/sprinkler irrigation |

---

## 🛠️ Tech Stack

<div align="center">

| Layer | Technologies |
|:---|:---|
| **Backend** | Python, Flask 3.0, SQLite, SQLAlchemy |
| **ML / AI** | Scikit-learn, TensorFlow/Keras, MobileNetV2, NumPy, Pandas |
| **IoT** | ESP32, ESP32-CAM, DHT22, Analog pH & Moisture sensors |
| **APIs** | wttr.in (weather), data.gov.in (mandi), ipapi.co (location) |
| **Security** | bcrypt password hashing, bleach XSS sanitization, session auth |
| **Frontend** | HTML5, CSS3, JavaScript, Jinja2 templates |
| **Infra** | Linux, Git, Vercel / CloudPanel (deployment-ready) |

</div>

---

## 📊 Database Schema

```sql
users               -- Auth + farmer profile (state, land_ha, category)
forum_posts         -- Community posts with likes
forum_replies       -- Threaded replies
soil_reports        -- N, P, K, pH, EC, OC, Zn, Fe, Mn, Cu, B + location
sms_subscriptions   -- Alert subscriptions by phone
sensor_readings     -- ESP32 data: temp, humidity, moisture, pH, battery
sensor_images       -- ESP32-CAM captures with CNN diagnosis
favorite_mandis     -- Per-user saved market watchlist
prediction_history  -- Top-10 crop prediction log per user
```

---

## 🌐 Bilingual Support

Every major advisory, recommendation, treatment, and alert is available in both:

- 🇬🇧 **English** — for educated farmers, agri-officers, and researchers
- 🇮🇳 **Hindi (हिंदी)** — for grassroots accessibility

---

## 🔮 Roadmap

- [ ] Mobile app (Flutter / React Native)
- [ ] Voice assistant (Hindi TTS/STT) for low-literacy users
- [ ] WhatsApp chatbot integration via Twilio
- [ ] Drone imagery support for field-level disease detection
- [ ] Multi-language support (Marathi, Punjabi, Tamil, Telugu)
- [ ] FPO (Farmer Producer Organisation) dashboard

---

## 👨‍💻 Author

<div align="center">

**Nitish Kumar Singh**
B.Tech CSE (AI/ML) — IILM University, Greater Noida

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/nitish-kumar-singh-4802792bb/)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/nitishchauhan002)

</div>

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<div align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,14,20,25,30&height=120&section=footer&animation=twinkling" width="100%"/>
  <sub>🌾 Built with 💚 for Indian farmers — <a href="https://github.com/nitishchauhan002/Kisan-Salahkar">Kisan Salahkar</a></sub>
</div>
