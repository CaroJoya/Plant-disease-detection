import sys
import os
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow.keras import backend, layers


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="PlantGuard AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# PATHS & MODEL SETTINGS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR, 'src'))

_CANDIDATE_PATHS = [
    os.path.join(BASE_DIR, 'saved_model', 'plant_disease_model.h5'),
    os.path.join(BASE_DIR, 'plant_disease_model.h5'),
    os.path.join(os.path.dirname(__file__), 'plant_disease_model.h5'),
]

MODEL_PATH = os.environ.get('MODEL_PATH', None)
if MODEL_PATH is None:
    for p in _CANDIDATE_PATHS:
        if os.path.exists(p):
            MODEL_PATH = p
            break
    if MODEL_PATH is None:
        MODEL_PATH = _CANDIDATE_PATHS[0]

IMAGE_SIZE = (224, 224)


# ============================================================
# CLASS NAMES
# ============================================================

CLASS_NAMES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
    'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy',
]


# ============================================================
# CUSTOM LAYER
# ============================================================

class FixedDropout(layers.Dropout):
    def _get_noise_shape(self, inputs):
        if self.noise_shape is None:
            return self.noise_shape
        symbolic_shape = backend.shape(inputs)
        noise_shape = [
            symbolic_shape[axis] if shape is None else shape
            for axis, shape in enumerate(self.noise_shape)
        ]
        return tuple(noise_shape)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Model file not found at: {MODEL_PATH}")
        st.stop()

    try:
        model = tf.keras.models.load_model(
            MODEL_PATH,
            custom_objects={"FixedDropout": FixedDropout},
            compile=False,
        )
    except Exception:
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)

    return model


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB").resize(IMAGE_SIZE, Image.BILINEAR)
    img_array = np.array(image, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


# ============================================================
# LABEL FORMATTING
# ============================================================

def format_label(label):
    if "___" in label:
        crop, disease = label.split("___", 1)
    else:
        crop = "Plant"
        disease = label

    crop = crop.replace("_", " ").replace(",", ", ")
    disease = disease.replace("_", " ").replace("  ", " ")

    return crop.title(), disease.title()


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #f5fbf7 0%, #eef8f1 50%, #f8fcf9 100%);
    }
    .main .block-container {
        max-width: 1150px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
    .hero { text-align: center; padding: 2rem 1rem 1.5rem 1rem; }
    .hero-icon { font-size: 3.5rem; margin-bottom: 0.2rem; }
    .hero-title {
        font-size: 3rem; font-weight: 800; letter-spacing: -1.5px;
        color: #173b2a; margin: 0;
    }
    .hero-title span { color: #2e8b57; }
    .hero-subtitle { font-size: 1.1rem; color: #62756a; margin-top: 0.5rem; }
    .pill-container {
        display: flex; justify-content: center; gap: 0.6rem;
        flex-wrap: wrap; margin: 1rem 0 2rem 0;
    }
    .pill {
        background: white; border: 1px solid #dcebe1; border-radius: 999px;
        padding: 0.45rem 0.9rem; color: #456152; font-size: 0.85rem;
        font-weight: 600; box-shadow: 0 2px 8px rgba(34, 80, 50, 0.05);
    }
    .card {
        background: rgba(255, 255, 255, 0.92); border: 1px solid #deebe2;
        border-radius: 22px; padding: 1.5rem;
        box-shadow: 0 10px 30px rgba(35, 76, 48, 0.07); margin-bottom: 1rem;
    }
    .card-title {
        font-size: 1.15rem; font-weight: 750; color: #234735; margin-bottom: 0.25rem;
    }
    .card-description { color: #718078; font-size: 0.9rem; margin-bottom: 1rem; }
    [data-testid="stFileUploader"] {
        background: #f7fbf8; border: 2px dashed #b9d8c2;
        border-radius: 18px; padding: 0.8rem;
    }
    [data-testid="stFileUploader"] section { border: none; }
    .stButton > button {
        width: 100%; border-radius: 14px; border: none;
        padding: 0.75rem 1rem; font-size: 1rem; font-weight: 700;
        color: white;
        background: linear-gradient(135deg, #2e8b57, #247447);
        box-shadow: 0 8px 18px rgba(46, 139, 87, 0.22);
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 24px rgba(46, 139, 87, 0.28);
    }
    .result-card {
        background: linear-gradient(145deg, #f5fbf7, #ffffff);
        border: 1px solid #d8e9dc; border-radius: 20px;
        padding: 1.5rem; margin-top: 1rem;
    }
    .result-label {
        color: #6a7c70; font-size: 0.8rem; text-transform: uppercase;
        letter-spacing: 1px; font-weight: 700;
    }
    .result-crop { color: #28563b; font-size: 1rem; font-weight: 650; margin-top: 0.7rem; }
    .result-disease {
        color: #173b2a; font-size: 1.7rem; font-weight: 800; margin-top: 0.15rem;
    }
    .model-info {
        background: #edf7f0; border-radius: 16px; padding: 1rem;
        margin-top: 1rem; border: 1px solid #d8eadc;
    }
    .model-info-title { font-weight: 750; color: #28563b; margin-bottom: 0.3rem; }
    .model-info-text { color: #66776c; font-size: 0.85rem; line-height: 1.5; }
    .footer {
        text-align: center; color: #849188; font-size: 0.8rem;
        padding-top: 2rem; padding-bottom: 1rem;
    }
    .footer strong { color: #4e6958; }
    div[data-testid="stImage"] { border-radius: 18px; overflow: hidden; }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HERO SECTION
# ============================================================

st.markdown(
    """
    <div class="hero">
        <div class="hero-icon">🌿</div>
        <div class="hero-title">Plant<span>Guard</span> AI</div>
        <div class="hero-subtitle">Intelligent Plant Disease Detection</div>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# FEATURE PILLS
# ============================================================

st.markdown(
    """
    <div class="pill-container">
        <div class="pill">🧠 CNN Deep Learning</div>
        <div class="pill">🌱 38 Plant Classes</div>
        <div class="pill">⚡ Fast Prediction</div>
        <div class="pill">🔬 Image Analysis</div>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# MAIN CONTENT
# ============================================================

left_column, right_column = st.columns([1, 1], gap="large")


# ============================================================
# LEFT COLUMN — UPLOAD
# ============================================================

with left_column:
    st.markdown(
        """
        <div class="card">
            <div class="card-title">📤 Upload Plant Leaf</div>
            <div class="card-description">
                Upload a clear image of a plant leaf
                to analyze it for possible disease.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")

        st.markdown("<div style='height: 0.5rem'></div>", unsafe_allow_html=True)
        st.image(image, caption="Uploaded Leaf", width=500)
        st.markdown("<div style='height: 0.8rem'></div>", unsafe_allow_html=True)

        predict_button = st.button("🔍  Analyze Leaf")
    else:
        image = None
        predict_button = False

        st.markdown(
            """
            <div class="model-info">
                <div class="model-info-title">📌 For better results</div>
                <div class="model-info-text">
                    Use a clear, well-lit image where
                    the plant leaf is visible and
                    occupies most of the frame.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# RIGHT COLUMN — RESULTS
# ============================================================

with right_column:
    st.markdown(
        """
        <div class="card">
            <div class="card-title">🔬 Analysis Result</div>
            <div class="card-description">
                Your prediction will appear here
                after image analysis.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    if uploaded_file is None:
        st.markdown(
            """
            <div class="result-card" style="text-align:center; padding:3rem 1.5rem;">
                <div style="font-size:3rem;">🌱</div>
                <div style="color:#567061; font-weight:650; margin-top:0.8rem;">
                    Waiting for an image
                </div>
                <div style="color:#87958c; font-size:0.85rem; margin-top:0.4rem;">
                    Upload a plant leaf to begin analysis.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    elif predict_button:
        with st.spinner("Analyzing leaf..."):
            try:
                model = load_model()
                img_array = preprocess_image(image)
                predictions = model.predict(img_array, verbose=0)

                result_index = int(np.argmax(predictions[0]))
                predicted_class = CLASS_NAMES[result_index]
                confidence = float(predictions[0][result_index]) * 100
                error_message = None
            except Exception as e:
                error_message = str(e)
                predicted_class = None
                confidence = None

        if error_message:
            st.error(f"❌ Prediction failed: {error_message}")
        else:
            crop_name, disease_name = format_label(predicted_class)
            is_healthy = "healthy" in predicted_class.lower()

            result_icon = "🌱" if is_healthy else "🔎"
            status_text = "Healthy Leaf" if is_healthy else "Disease Detected"

            st.markdown(
                f"""
                <div class="result-card">
                    <div style="font-size:2.4rem; margin-bottom:0.5rem;">{result_icon}</div>
                    <div class="result-label">{status_text}</div>
                    <div class="result-crop">{crop_name}</div>
                    <div class="result-disease">{disease_name}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown("<div style='height: 0.7rem'></div>", unsafe_allow_html=True)

            # ---------- Confidence gauge ----------
            if confidence >= 75:
                bar_color = "linear-gradient(90deg, #22c55e, #16a34a)"
                badge_bg = "#dcfce7"
                badge_fg = "#15803d"
                badge_txt = "HIGH CONFIDENCE"
            elif confidence >= 50:
                bar_color = "linear-gradient(90deg, #facc15, #f59e0b)"
                badge_bg = "#fef3c7"
                badge_fg = "#b45309"
                badge_txt = "MEDIUM CONFIDENCE"
            else:
                bar_color = "linear-gradient(90deg, #f87171, #dc2626)"
                badge_bg = "#fee2e2"
                badge_fg = "#b91c1c"
                badge_txt = "LOW CONFIDENCE"

            bar_width = min(max(confidence, 0), 100)

            confidence_html = (
                '<div style="background:#ffffff; border:1px solid #d8e9dc;'
                ' border-radius:16px; padding:1rem 1.1rem; margin-top:0.6rem;'
                ' box-shadow:0 4px 14px rgba(35,76,48,0.06);">'
                '<div style="display:flex; justify-content:space-between;'
                ' align-items:center; margin-bottom:0.6rem;">'
                '<span style="color:#4b5f52; font-size:0.8rem; font-weight:700;'
                ' letter-spacing:0.5px; text-transform:uppercase;">Model Confidence</span>'
                f'<span style="background:{badge_bg}; color:{badge_fg};'
                ' font-size:0.7rem; font-weight:700; padding:0.2rem 0.55rem;'
                f' border-radius:999px; letter-spacing:0.3px;">{badge_txt}</span>'
                '</div>'
                '<div style="display:flex; align-items:baseline; gap:0.5rem;'
                ' margin-bottom:0.55rem;">'
                f'<span style="color:#173b2a; font-size:2rem; font-weight:800;'
                f' line-height:1;">{confidence:.1f}%</span>'
                '</div>'
                '<div style="width:100%; height:12px; background:#eef7f0;'
                ' border-radius:999px; overflow:hidden;">'
                f'<div style="width:{bar_width}%; height:100%;'
                f' background:{bar_color}; border-radius:999px;"></div>'
                '</div>'
                '</div>'
            )

            st.markdown(confidence_html, unsafe_allow_html=True)

            st.markdown(
                f"""
                <div class="model-info">
                    <div class="model-info-title">🧠 Prediction Details</div>
                    <div class="model-info-text">
                        The model classified the uploaded
                        image as <strong>{disease_name}</strong>
                        for <strong>{crop_name}</strong>.
                        The displayed confidence represents
                        the model's probability for its
                        selected class.
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

    else:
        st.markdown(
            """
            <div class="result-card" style="text-align:center; padding:3rem 1.5rem;">
                <div style="font-size:3rem;">🔍</div>
                <div style="color:#567061; font-weight:650; margin-top:0.8rem;">
                    Ready to analyze
                </div>
                <div style="color:#87958c; font-size:0.85rem; margin-top:0.4rem;">
                    Click "Analyze Leaf" to run the trained CNN model.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# MODEL INFORMATION
# ============================================================

st.markdown("<div style='height: 1rem'></div>", unsafe_allow_html=True)

st.markdown(
    """
    <div class="card">
        <div class="card-title">⚙️ About PlantGuard AI</div>
        <div class="card-description">
            A deep-learning based image classification
            system designed to identify plant diseases
            from leaf images.
        </div>
        <div style="display:grid;
                    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
                    gap:0.8rem;">
            <div class="model-info">
                <div class="model-info-title">🧠 Architecture</div>
                <div class="model-info-text">MobileNetV2 (Transfer Learning)</div>
            </div>
            <div class="model-info">
                <div class="model-info-title">🌱 Classes</div>
                <div class="model-info-text">38 plant and disease categories</div>
            </div>
            <div class="model-info">
                <div class="model-info-title">🖼️ Input</div>
                <div class="model-info-text">224 × 224 RGB image</div>
            </div>
            <div class="model-info">
                <div class="model-info-title">⚡ Framework</div>
                <div class="model-info-text">TensorFlow / Keras</div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        🌿 <strong>PlantGuard AI</strong>
        &nbsp;•&nbsp;
        Powered by Deep Learning
    </div>
    """,
    unsafe_allow_html=True
)