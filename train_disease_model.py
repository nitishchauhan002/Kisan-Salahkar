"""
train_disease_model.py — Plant Disease Detection CNN
=====================================================
Trains a MobileNetV2-based image classifier on the PlantVillage dataset.
38 classes of crop leaf diseases  •  ~54,000 images  •  Target accuracy: 95%+

Usage:
    python train_disease_model.py

This will:
  1. Download the PlantVillage dataset (if not already present)
  2. Train a CNN using transfer learning (MobileNetV2)
  3. Save the model to model/disease_model.keras
  4. Save class labels to model/disease_labels.json
  5. Print evaluation metrics

Requirements:
  pip install tensorflow pillow matplotlib
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")   # Non-interactive backend (no GUI needed)
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from tensorflow.keras.applications import MobileNetV2

# ── Configuration ─────────────────────────────────────────────────────────────
IMG_SIZE       = 224          # MobileNetV2 input size
BATCH_SIZE     = 32
EPOCHS_FREEZE  = 5            # Epochs with base frozen  (transfer learning phase)
EPOCHS_FINE    = 15           # Epochs with top layers unfrozen (fine-tuning phase)
LEARNING_RATE  = 1e-3
FINE_TUNE_LR   = 1e-5
VALIDATION_SPLIT = 0.2

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR   = os.path.join(BASE_DIR, "model")
DATASET_DIR = os.path.join(BASE_DIR, "dataset", "PlantVillage")

# ── Disease class → readable name + Hindi translation ─────────────────────────
# Maps PlantVillage folder names to clean labels
CLASS_INFO = {
    "Apple___Apple_scab":                   {"en": "Apple Scab",                 "hi": "सेब का पपड़ी रोग",         "crop": "Apple"},
    "Apple___Black_rot":                    {"en": "Apple Black Rot",            "hi": "सेब का काला सड़न",          "crop": "Apple"},
    "Apple___Cedar_apple_rust":             {"en": "Cedar Apple Rust",           "hi": "सेब का जंग रोग",           "crop": "Apple"},
    "Apple___healthy":                      {"en": "Apple - Healthy",            "hi": "सेब - स्वस्थ",              "crop": "Apple"},
    "Blueberry___healthy":                  {"en": "Blueberry - Healthy",        "hi": "ब्लूबेरी - स्वस्थ",         "crop": "Blueberry"},
    "Cherry_(including_sour)___Powdery_mildew": {"en": "Cherry Powdery Mildew", "hi": "चेरी का चूर्णिल आसिता",    "crop": "Cherry"},
    "Cherry_(including_sour)___healthy":    {"en": "Cherry - Healthy",           "hi": "चेरी - स्वस्थ",             "crop": "Cherry"},
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {"en": "Corn Gray Leaf Spot", "hi": "मक्का का भूरा पत्ती धब्बा", "crop": "Maize"},
    "Corn_(maize)___Common_rust_":          {"en": "Corn Common Rust",           "hi": "मक्का का सामान्य जंग",      "crop": "Maize"},
    "Corn_(maize)___Northern_Leaf_Blight":  {"en": "Corn Northern Leaf Blight",  "hi": "मक्का का उत्तरी पत्ती झुलसा", "crop": "Maize"},
    "Corn_(maize)___healthy":               {"en": "Corn - Healthy",             "hi": "मक्का - स्वस्थ",             "crop": "Maize"},
    "Grape___Black_rot":                    {"en": "Grape Black Rot",            "hi": "अंगूर का काला सड़न",         "crop": "Grape"},
    "Grape___Esca_(Black_Measles)":         {"en": "Grape Black Measles",        "hi": "अंगूर का काला खसरा",        "crop": "Grape"},
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {"en": "Grape Leaf Blight",   "hi": "अंगूर पत्ती झुलसा",          "crop": "Grape"},
    "Grape___healthy":                      {"en": "Grape - Healthy",            "hi": "अंगूर - स्वस्थ",             "crop": "Grape"},
    "Orange___Haunglongbing_(Citrus_greening)": {"en": "Citrus Greening",       "hi": "संतरा हरित रोग",             "crop": "Orange"},
    "Peach___Bacterial_spot":               {"en": "Peach Bacterial Spot",       "hi": "आड़ू का जीवाणु धब्बा",      "crop": "Peach"},
    "Peach___healthy":                      {"en": "Peach - Healthy",            "hi": "आड़ू - स्वस्थ",              "crop": "Peach"},
    "Pepper,_bell___Bacterial_spot":        {"en": "Pepper Bacterial Spot",      "hi": "शिमला मिर्च जीवाणु धब्बा",  "crop": "Pepper"},
    "Pepper,_bell___healthy":               {"en": "Pepper - Healthy",           "hi": "शिमला मिर्च - स्वस्थ",       "crop": "Pepper"},
    "Potato___Early_blight":                {"en": "Potato Early Blight",        "hi": "आलू का अगेता झुलसा",        "crop": "Potato"},
    "Potato___Late_blight":                 {"en": "Potato Late Blight",         "hi": "आलू का पछेता झुलसा",        "crop": "Potato"},
    "Potato___healthy":                     {"en": "Potato - Healthy",           "hi": "आलू - स्वस्थ",               "crop": "Potato"},
    "Raspberry___healthy":                  {"en": "Raspberry - Healthy",        "hi": "रास्पबेरी - स्वस्थ",         "crop": "Raspberry"},
    "Soybean___healthy":                    {"en": "Soybean - Healthy",          "hi": "सोयाबीन - स्वस्थ",           "crop": "Soybean"},
    "Squash___Powdery_mildew":              {"en": "Squash Powdery Mildew",      "hi": "कद्दू का चूर्णिल आसिता",    "crop": "Squash"},
    "Strawberry___Leaf_scorch":             {"en": "Strawberry Leaf Scorch",     "hi": "स्ट्रॉबेरी पत्ती झुलसा",     "crop": "Strawberry"},
    "Strawberry___healthy":                 {"en": "Strawberry - Healthy",       "hi": "स्ट्रॉबेरी - स्वस्थ",        "crop": "Strawberry"},
    "Tomato___Bacterial_spot":              {"en": "Tomato Bacterial Spot",      "hi": "टमाटर जीवाणु धब्बा",        "crop": "Tomato"},
    "Tomato___Early_blight":                {"en": "Tomato Early Blight",        "hi": "टमाटर अगेता झुलसा",         "crop": "Tomato"},
    "Tomato___Late_blight":                 {"en": "Tomato Late Blight",         "hi": "टमाटर पछेता झुलसा",         "crop": "Tomato"},
    "Tomato___Leaf_Mold":                   {"en": "Tomato Leaf Mold",           "hi": "टमाटर पत्ती फफूंद",          "crop": "Tomato"},
    "Tomato___Septoria_leaf_spot":          {"en": "Tomato Septoria Leaf Spot",  "hi": "टमाटर सेप्टोरिया धब्बा",    "crop": "Tomato"},
    "Tomato___Spider_mites Two-spotted_spider_mite": {"en": "Tomato Spider Mites", "hi": "टमाटर मकड़ी कीट",        "crop": "Tomato"},
    "Tomato___Target_Spot":                 {"en": "Tomato Target Spot",         "hi": "टमाटर लक्ष्य धब्बा",        "crop": "Tomato"},
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {"en": "Tomato Yellow Leaf Curl Virus", "hi": "टमाटर पीला पत्ती मोड़ विषाणु", "crop": "Tomato"},
    "Tomato___Tomato_mosaic_virus":         {"en": "Tomato Mosaic Virus",        "hi": "टमाटर मोज़ेक विषाणु",       "crop": "Tomato"},
    "Tomato___healthy":                     {"en": "Tomato - Healthy",           "hi": "टमाटर - स्वस्थ",             "crop": "Tomato"},
}

# Treatment database for each disease
TREATMENT_DB = {
    "Apple Scab":                  {"en": "Apply Captan or Myclobutanil fungicide. Remove fallen leaves. Prune for airflow.", "hi": "कैप्टान या माइक्लोब्यूटानिल फफूंदनाशी छिड़कें। गिरी पत्तियाँ हटाएँ।"},
    "Apple Black Rot":             {"en": "Remove mummified fruits. Apply Captan during bloom. Prune dead wood.", "hi": "सड़े फल हटाएँ। फूल आने पर कैप्टान छिड़कें। सूखी लकड़ी काटें।"},
    "Cedar Apple Rust":            {"en": "Apply Myclobutanil before infection period. Remove nearby cedar trees.", "hi": "संक्रमण से पहले माइक्लोब्यूटानिल छिड़कें। पास के देवदार हटाएँ।"},
    "Cherry Powdery Mildew":       {"en": "Spray Sulphur WP 0.3% or potassium bicarbonate. Ensure air circulation.", "hi": "सल्फर WP 0.3% छिड़कें। हवा का संचार सुनिश्चित करें।"},
    "Corn Gray Leaf Spot":         {"en": "Use resistant hybrids. Rotate crops. Apply Azoxystrobin fungicide if severe.", "hi": "प्रतिरोधी संकर उपयोग करें। फसल चक्र अपनाएँ।"},
    "Corn Common Rust":            {"en": "Plant resistant varieties. Apply Mancozeb 0.25% if severe.", "hi": "प्रतिरोधी किस्में लगाएँ। गंभीर होने पर मैन्कोज़ेब 0.25% छिड़कें।"},
    "Corn Northern Leaf Blight":   {"en": "Use resistant hybrids. Apply Propiconazole. Crop rotation helps.", "hi": "प्रतिरोधी किस्में उपयोग करें। प्रोपिकोनाज़ोल छिड़कें।"},
    "Grape Black Rot":             {"en": "Remove mummies. Apply Captan/Myclobutanil from bloom to veraison.", "hi": "सड़े फल हटाएँ। फूल से लेकर रंग बदलने तक कैप्टान छिड़कें।"},
    "Grape Black Measles":         {"en": "No cure. Remove infected vines. Avoid wound infections during pruning.", "hi": "कोई इलाज नहीं। संक्रमित बेलें हटाएँ। कटाई में घाव से बचें।"},
    "Grape Leaf Blight":           {"en": "Apply Mancozeb or Copper Oxychloride. Remove infected leaves.", "hi": "मैन्कोज़ेब या कॉपर ऑक्सीक्लोराइड छिड़कें।"},
    "Citrus Greening":             {"en": "No cure. Control psyllid vectors. Remove infected trees. Plant disease-free nursery stock.", "hi": "कोई इलाज नहीं। सिल्लिड कीट नियंत्रण करें। संक्रमित पेड़ हटाएँ।"},
    "Peach Bacterial Spot":        {"en": "Apply Copper-based bactericides. Use resistant varieties. Windbreaks help.", "hi": "तांबा-आधारित जीवाणुनाशक छिड़कें। प्रतिरोधी किस्में उपयोग करें।"},
    "Pepper Bacterial Spot":       {"en": "Use certified disease-free seeds. Apply Copper hydroxide. Crop rotation.", "hi": "प्रमाणित बीज उपयोग करें। कॉपर हाइड्रॉक्साइड छिड़कें।"},
    "Potato Early Blight":         {"en": "Spray Mancozeb 0.25% at 10-day intervals. Use resistant varieties. Remove debris.", "hi": "मैन्कोज़ेब 0.25% हर 10 दिन छिड़कें। प्रतिरोधी किस्में लगाएँ।"},
    "Potato Late Blight":          {"en": "Spray Metalaxyl + Mancozeb. Destroy infected plants. Hill up tubers.", "hi": "मेटालैक्सिल + मैन्कोज़ेब छिड़कें। संक्रमित पौधे नष्ट करें।"},
    "Squash Powdery Mildew":       {"en": "Spray Sulphur WP or Karathane. Plant resistant varieties. Ensure spacing.", "hi": "सल्फर WP या कैराथेन छिड़कें। उचित दूरी रखें।"},
    "Strawberry Leaf Scorch":      {"en": "Remove infected leaves. Apply Captan fungicide. Ensure good drainage.", "hi": "संक्रमित पत्ते हटाएँ। कैप्टान छिड़कें। अच्छी जल निकासी रखें।"},
    "Tomato Bacterial Spot":       {"en": "Use disease-free seeds. Apply Copper hydroxide + Mancozeb. Avoid overhead irrigation.", "hi": "रोग-मुक्त बीज उपयोग करें। कॉपर + मैन्कोज़ेब छिड़कें।"},
    "Tomato Early Blight":         {"en": "Spray Mancozeb/Chlorothalonil at 7-10 day intervals. Mulch to prevent splash.", "hi": "मैन्कोज़ेब हर 7-10 दिन छिड़कें। मल्चिंग करें।"},
    "Tomato Late Blight":          {"en": "Spray Metalaxyl + Mancozeb immediately. Destroy infected plants. Avoid wet foliage.", "hi": "तुरंत मेटालैक्सिल + मैन्कोज़ेब छिड़कें। संक्रमित पौधे नष्ट करें।"},
    "Tomato Leaf Mold":            {"en": "Improve ventilation. Reduce humidity. Apply Chlorothalonil.", "hi": "हवा का संचार बढ़ाएँ। नमी कम करें। क्लोरोथालोनिल छिड़कें।"},
    "Tomato Septoria Leaf Spot":   {"en": "Remove lower infected leaves. Spray Mancozeb or Copper fungicide.", "hi": "निचली संक्रमित पत्तियाँ हटाएँ। मैन्कोज़ेब छिड़कें।"},
    "Tomato Spider Mites":         {"en": "Spray Dicofol 0.05% or neem oil. Wash plants with water jet. Release predatory mites.", "hi": "डाइकोफॉल 0.05% या नीम तेल छिड़कें। पानी से धोएँ।"},
    "Tomato Target Spot":          {"en": "Apply Chlorothalonil or Mancozeb. Remove crop debris. Ensure airflow.", "hi": "क्लोरोथालोनिल या मैन्कोज़ेब छिड़कें। फसल अवशेष हटाएँ।"},
    "Tomato Yellow Leaf Curl Virus": {"en": "Control whitefly with Imidacloprid. Use reflective mulch. Plant resistant varieties.", "hi": "इमिडाक्लोप्रिड से सफेद मक्खी नियंत्रण। प्रतिबिम्बी मल्च उपयोग करें।"},
    "Tomato Mosaic Virus":         {"en": "Use virus-free seeds. Disinfect tools. Remove infected plants immediately.", "hi": "वायरस-मुक्त बीज उपयोग करें। उपकरण निर्जीवाणु करें।"},
}


def download_dataset():
    """
    Download PlantVillage dataset from TensorFlow Datasets.
    The dataset is cached after the first download.
    """
    print("\n📥 Loading PlantVillage dataset...")
    print("   (First run will download ~240 MB, then cached)\n")

    dataset_url = "https://storage.googleapis.com/plantvillage_dataset/color.zip"

    # Use tf.keras utility to download and extract
    data_dir = tf.keras.utils.get_file(
        fname="PlantVillage",
        origin=dataset_url,
        untar=False,
        extract=True,
        cache_dir=os.path.join(BASE_DIR, "dataset"),
        cache_subdir="PlantVillage"
    )

    # The extracted folder should contain class folders
    # Find the actual directory with class subfolders
    possible_dirs = [
        os.path.join(BASE_DIR, "dataset", "PlantVillage"),
        os.path.join(BASE_DIR, "dataset", "PlantVillage", "color"),
        os.path.join(BASE_DIR, "dataset", "PlantVillage", "Plant_leave_diseases_dataset_with_augmentation"),
        data_dir,
    ]

    for d in possible_dirs:
        if os.path.isdir(d):
            subdirs = [x for x in os.listdir(d) if os.path.isdir(os.path.join(d, x))]
            if len(subdirs) >= 10:  # Found class folders
                print(f"   ✅ Dataset found at: {d}")
                print(f"   📁 {len(subdirs)} classes detected\n")
                return d

    # Fallback: use TensorFlow Datasets if direct download doesn't work
    print("   ⚠ Direct download failed, trying alternative method...")
    return setup_alternative_dataset()


def setup_alternative_dataset():
    """
    Alternative: Download PlantVillage via tensorflow_datasets
    or create a minimal dataset for testing.
    """
    alt_dir = os.path.join(BASE_DIR, "dataset", "PlantVillage")
    os.makedirs(alt_dir, exist_ok=True)

    print("\n╔════════════════════════════════════════════════════════════╗")
    print("║  MANUAL DATASET SETUP REQUIRED                           ║")
    print("╠════════════════════════════════════════════════════════════╣")
    print("║                                                          ║")
    print("║  Download PlantVillage dataset from ONE of these:        ║")
    print("║                                                          ║")
    print("║  1. Kaggle (RECOMMENDED — 1.5 GB):                      ║")
    print("║     https://www.kaggle.com/datasets/emmarex/             ║")
    print("║     plantdisease                                         ║")
    print("║                                                          ║")
    print("║  2. GitHub mirror:                                       ║")
    print("║     https://github.com/spMohanty/                       ║")
    print("║     PlantVillage-Dataset                                 ║")
    print("║                                                          ║")
    print("║  After downloading, extract so the structure is:         ║")
    print("║                                                          ║")
    print(f"║  {alt_dir}")
    print("║    ├── Apple___Apple_scab/                               ║")
    print("║    │   ├── image001.jpg                                  ║")
    print("║    │   └── ...                                           ║")
    print("║    ├── Apple___Black_rot/                                ║")
    print("║    ├── Tomato___Late_blight/                             ║")
    print("║    └── ... (38 folders total)                            ║")
    print("║                                                          ║")
    print("║  Then re-run: python train_disease_model.py              ║")
    print("╚════════════════════════════════════════════════════════════╝\n")

    raise FileNotFoundError(
        f"PlantVillage dataset not found at {alt_dir}.\n"
        "Download from Kaggle: https://www.kaggle.com/datasets/emmarex/plantdisease\n"
        f"Extract to: {alt_dir}"
    )


def create_datasets(data_dir):
    """Create training and validation datasets with augmentation."""
    print("🔄 Preparing datasets with augmentation...\n")

    # Training dataset
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=VALIDATION_SPLIT,
        subset="training",
        seed=42,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        label_mode="categorical",
    )

    # Validation dataset
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=VALIDATION_SPLIT,
        subset="validation",
        seed=42,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        label_mode="categorical",
    )

    class_names = train_ds.class_names
    num_classes = len(class_names)
    print(f"   📊 {num_classes} classes found")
    print(f"   📷 Training batches: {len(train_ds)}")
    print(f"   📷 Validation batches: {len(val_ds)}\n")

    # Performance optimization
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, class_names, num_classes


def build_model(num_classes):
    """
    Build MobileNetV2-based model with transfer learning.
    MobileNetV2 is chosen because:
      - Lightweight (3.4M params vs 25M for ResNet50)
      - Fast inference (~22ms per image)
      - Good accuracy on plant disease datasets (95%+)
      - Can be converted to TFLite for mobile/edge deployment
    """
    print("🏗️  Building MobileNetV2 model...\n")

    # Data augmentation layer
    data_augmentation = keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.15),
        layers.RandomBrightness(0.1),
        layers.RandomContrast(0.1),
    ], name="data_augmentation")

    # MobileNetV2 base (pre-trained on ImageNet)
    base_model = MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False  # Freeze during initial training

    # Build full model
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = data_augmentation(inputs)                           # Augment
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)  # Normalize for MobileNetV2
    x = base_model(x, training=False)                       # Feature extraction
    x = layers.GlobalAveragePooling2D()(x)                  # Pool spatial dimensions
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)                              # Regularization
    x = layers.Dense(256, activation="relu")(x)             # Hidden layer
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)  # Classification

    model = keras.Model(inputs, outputs, name="PlantDisease_MobileNetV2")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    total_params = model.count_params()
    trainable_params = sum([tf.reduce_prod(v.shape).numpy() for v in model.trainable_variables])
    print(f"   📐 Total params:     {total_params:,}")
    print(f"   🏋️  Trainable params: {trainable_params:,}")
    print(f"   ❄️  Frozen params:    {total_params - trainable_params:,}\n")

    return model, base_model


def train_phase1(model, train_ds, val_ds):
    """Phase 1: Train with frozen base (transfer learning)."""
    print("=" * 60)
    print("📚 PHASE 1: Transfer Learning (base frozen)")
    print(f"   Epochs: {EPOCHS_FREEZE} | LR: {LEARNING_RATE}")
    print("=" * 60 + "\n")

    early_stop = callbacks.EarlyStopping(
        monitor="val_accuracy", patience=3,
        restore_best_weights=True, verbose=1
    )

    history1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_FREEZE,
        callbacks=[early_stop],
        verbose=1,
    )
    return history1


def train_phase2(model, base_model, train_ds, val_ds):
    """Phase 2: Fine-tune top layers of MobileNetV2."""
    print("\n" + "=" * 60)
    print("🎯 PHASE 2: Fine-Tuning (top 50 layers unfrozen)")
    print(f"   Epochs: {EPOCHS_FINE} | LR: {FINE_TUNE_LR}")
    print("=" * 60 + "\n")

    # Unfreeze top 50 layers of MobileNetV2
    base_model.trainable = True
    fine_tune_at = len(base_model.layers) - 50

    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    # Recompile with lower learning rate
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=FINE_TUNE_LR),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    early_stop = callbacks.EarlyStopping(
        monitor="val_accuracy", patience=5,
        restore_best_weights=True, verbose=1
    )
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5,
        patience=2, min_lr=1e-7, verbose=1
    )

    history2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_FINE,
        callbacks=[early_stop, reduce_lr],
        verbose=1,
    )
    return history2


def save_model(model, class_names):
    """Save model and class metadata."""
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Save Keras model
    model_path = os.path.join(MODEL_DIR, "disease_model.keras")
    model.save(model_path)
    print(f"\n💾 Model saved: {model_path}")

    # Save class labels with metadata
    labels = {}
    for i, class_name in enumerate(class_names):
        info = CLASS_INFO.get(class_name, {
            "en": class_name.replace("_", " "),
            "hi": class_name.replace("_", " "),
            "crop": class_name.split("___")[0] if "___" in class_name else "Unknown",
        })
        treatment = TREATMENT_DB.get(info["en"], {
            "en": "Consult your local agricultural extension officer.",
            "hi": "अपने स्थानीय कृषि विस्तार अधिकारी से संपर्क करें।",
        })
        labels[str(i)] = {
            "class_name": class_name,
            "en": info["en"],
            "hi": info["hi"],
            "crop": info["crop"],
            "is_healthy": "healthy" in class_name.lower(),
            "treatment_en": treatment.get("en", ""),
            "treatment_hi": treatment.get("hi", ""),
        }

    labels_path = os.path.join(MODEL_DIR, "disease_labels.json")
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)
    print(f"📋 Labels saved: {labels_path}  ({len(labels)} classes)")

    # Save TFLite model for mobile/edge deployment
    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        tflite_path = os.path.join(MODEL_DIR, "disease_model.tflite")
        with open(tflite_path, "wb") as f:
            f.write(tflite_model)
        print(f"📱 TFLite model saved: {tflite_path}  ({len(tflite_model) / 1024 / 1024:.1f} MB)")
    except Exception as e:
        print(f"⚠️ TFLite conversion skipped: {e}")


def save_training_plot(history1, history2):
    """Save accuracy/loss curves."""
    # Combine histories
    acc = history1.history["accuracy"] + history2.history["accuracy"]
    val_acc = history1.history["val_accuracy"] + history2.history["val_accuracy"]
    loss = history1.history["loss"] + history2.history["loss"]
    val_loss = history1.history["val_loss"] + history2.history["val_loss"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy plot
    ax1.plot(acc, label="Train Accuracy", linewidth=2)
    ax1.plot(val_acc, label="Val Accuracy", linewidth=2)
    ax1.axvline(x=len(history1.history["accuracy"]) - 0.5, color="gray", linestyle="--", label="Fine-tuning start")
    ax1.set_title("Model Accuracy", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Loss plot
    ax2.plot(loss, label="Train Loss", linewidth=2)
    ax2.plot(val_loss, label="Val Loss", linewidth=2)
    ax2.axvline(x=len(history1.history["loss"]) - 0.5, color="gray", linestyle="--", label="Fine-tuning start")
    ax2.set_title("Model Loss", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(MODEL_DIR, "disease_training_plot.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"📈 Training plot saved: {plot_path}")


def evaluate_model(model, val_ds, class_names):
    """Print evaluation metrics."""
    print("\n" + "=" * 60)
    print("📊 EVALUATION")
    print("=" * 60)

    loss, accuracy = model.evaluate(val_ds, verbose=0)
    print(f"\n   🎯 Validation Accuracy: {accuracy * 100:.2f}%")
    print(f"   📉 Validation Loss:     {loss:.4f}")

    if accuracy > 0.95:
        print("   ✅ EXCELLENT — Model is production-ready!")
    elif accuracy > 0.90:
        print("   ✅ GOOD — Model performs well. More data could help.")
    elif accuracy > 0.80:
        print("   ⚠️  FAIR — Consider more epochs or data augmentation.")
    else:
        print("   ❌ NEEDS IMPROVEMENT — Try more training data.")

    # Per-class accuracy on a sample
    print("\n   Per-class sample (first 10 classes):")
    print("   " + "-" * 45)

    y_true = []
    y_pred = []
    for images, labels in val_ds.take(10):
        predictions = model.predict(images, verbose=0)
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred.extend(np.argmax(predictions, axis=1))

    from collections import Counter
    correct = Counter()
    total = Counter()
    for t, p in zip(y_true, y_pred):
        total[t] += 1
        if t == p:
            correct[t] += 1

    for cls_idx in sorted(total.keys())[:10]:
        name = class_names[cls_idx] if cls_idx < len(class_names) else f"Class {cls_idx}"
        info = CLASS_INFO.get(name, {"en": name})
        acc = correct[cls_idx] / total[cls_idx] * 100 if total[cls_idx] else 0
        print(f"   {info['en'][:30]:30s} {acc:5.1f}% ({correct[cls_idx]}/{total[cls_idx]})")

    return accuracy


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "═" * 60)
    print("  🌿 KISAN SALAHKAR — Plant Disease Detection Training")
    print("  📐 Architecture: MobileNetV2 + Transfer Learning")
    print("  📊 Dataset: PlantVillage (38 classes, ~54K images)")
    print("═" * 60)

    # Check GPU
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print(f"\n  🚀 GPU detected: {gpus[0].name}")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        print("\n  💻 No GPU — training on CPU (will be slower)")
        print("     Tip: Install CUDA + cuDNN for GPU acceleration\n")

    # Step 1: Get dataset
    data_dir = download_dataset()

    # Step 2: Create datasets
    train_ds, val_ds, class_names, num_classes = create_datasets(data_dir)

    # Step 3: Build model
    model, base_model = build_model(num_classes)

    # Step 4: Phase 1 — Transfer learning
    history1 = train_phase1(model, train_ds, val_ds)

    # Step 5: Phase 2 — Fine-tuning
    history2 = train_phase2(model, base_model, train_ds, val_ds)

    # Step 6: Save
    save_model(model, class_names)
    save_training_plot(history1, history2)

    # Step 7: Evaluate
    accuracy = evaluate_model(model, val_ds, class_names)

    print("\n" + "═" * 60)
    print(f"  ✅ TRAINING COMPLETE — Accuracy: {accuracy * 100:.1f}%")
    print(f"  📁 Model: model/disease_model.keras")
    print(f"  📋 Labels: model/disease_labels.json")
    print(f"  📈 Plot: model/disease_training_plot.png")
    print("═" * 60)
    print("\n  Next: Run `python app.py` — the disease detection")
    print("  page will now use CNN image classification! 🚀\n")


if __name__ == "__main__":
    main()
