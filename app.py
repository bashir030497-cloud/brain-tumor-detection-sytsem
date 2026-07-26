import streamlit as st
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from PIL import Image
import os
import base64
import gdown

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(
    page_title="NeuroScan AI | Brain Tumor Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

def get_base64_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

brain_icon_b64 = get_base64_image(os.path.join(BASE_DIR, "Brain-Memory.png"))

# ---------------- CUSTOM CSS ----------------
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
    }

    .stApp {
        background: linear-gradient(180deg, #0f1117 0%, #14161f 100%);
        color: #e5e7eb;
    }

   #MainMenu, footer, header {visibility: hidden;}
    .block-container {
        padding-top: 2rem;
        padding-bottom: 1rem;
    }

    /* Hero banner - teal/cyan theme */
    .hero-banner {
        background: linear-gradient(135deg, #0f766e 0%, #0891b2 50%, #06b6d4 100%);
        padding: 16px 30px;
        border-radius: 14px;
        color: white;
        margin-bottom: 14px;
        box-shadow: 0 6px 18px rgba(6, 182, 212, 0.2);
    }
    .hero-title {
        font-size: 22px;
        font-weight: 800;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .hero-subtitle {
        font-size: 13px;
        font-weight: 400;
        opacity: 0.9;
        margin-top: 4px;
    }
    .hero-badge {
        display: inline-block;
        background: rgba(255,255,255,0.18);
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: 600;
        margin-top: 8px;
        letter-spacing: 0.5px;
    }

    /* Section card - dark */
    .section-card {
        background: #1a1d29;
        border-radius: 14px;
        padding: 16px 18px;
        box-shadow: 0 4px 16px rgba(0,0,0,0.3);
        border: 1px solid #262a3b;
        margin-bottom: 10px;
    }
    .section-title {
        font-size: 15px;
        font-weight: 700;
        color: #f1f2f6;
        margin-bottom: 2px;
    }
    .section-caption {
        font-size: 12px;
        color: #7d8092;
        margin-bottom: 10px;
    }

    /* Text elements */
    p, span, label, .stMarkdown {
        color: #d1d3de;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: #1a1d29;
        padding: 5px;
        border-radius: 12px;
        border: 1px solid #262a3b;
    }
    .stTabs [data-baseweb="tab"] {
        height: 38px;
        border-radius: 9px;
        font-weight: 600;
        font-size: 13px;
        color: #9294a8;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #0891b2, #06b6d4) !important;
        color: white !important;
    }

    /* Metrics */
    div[data-testid="stMetric"] {
        background: #1a1d29;
        border: 1px solid #262a3b;
        border-radius: 12px;
        padding: 10px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    }
    div[data-testid="stMetricLabel"] {
        color: #7d8092;
        font-weight: 600;
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    div[data-testid="stMetricValue"] {
        color: #f1f2f6;
        font-weight: 700;
        font-size: 18px;
    }

    /* Result cards */
    .result-card {
        padding: 16px;
        border-radius: 14px;
        text-align: center;
        margin-top: 4px;
    }
    .result-icon { font-size: 28px; margin-bottom: 2px; }
    .result-title { font-size: 18px; font-weight: 800; margin-bottom: 2px; }
    .result-sub { font-size: 13px; font-weight: 500; opacity: 0.85; }

    .tumor-detected {
        background: linear-gradient(135deg, #2d1416, #3a1a1d);
        color: #fca5a5;
        border: 1px solid #5c2529;
    }
    .no-tumor {
        background: linear-gradient(135deg, #0f2d1e, #123a25);
        color: #86efac;
        border: 1px solid #1f5c38;
    }

    /* Empty state */
    .empty-state {
        padding: 30px 20px;
        text-align: center;
        color: #6b6e82;
        border: 2px dashed #2a2e42;
        border-radius: 14px;
        background: #16181f;
        font-size: 13px;
    }
    .empty-icon { font-size: 26px; margin-bottom: 6px; }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #0b0d13;
    }
    section[data-testid="stSidebar"] * {
        color: #d1d3de !important;
    }
    .sidebar-card {
        background: rgba(6, 182, 212, 0.06);
        border-radius: 12px;
        padding: 12px 14px;
        margin-bottom: 8px;
        border: 1px solid rgba(6, 182, 212, 0.12);
    }
    .sidebar-label {
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: #6b7280 !important;
        margin-bottom: 3px;
    }
    .sidebar-value {
        font-size: 14px;
        font-weight: 600;
        color: #67e8f9 !important;
    }

    /* File uploader dark styling */
    [data-testid="stFileUploader"] {
        background: #16181f;
        border-radius: 10px;
        padding: 6px;
    }

    /* Shrink uploaded/result images */
    [data-testid="stImage"] img {
        max-height: 220px;
        object-fit: contain;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------- MODEL LOADING (Google Drive auto-download) ----------------
WEIGHTS_PATH = os.path.join(BASE_DIR, "brain_tumor_weights.weights.h5")
WEIGHTS_FILE_ID = "1cikmuIAFfSAYJvGb97gYdzGc85K95HK6"

@st.cache_resource
def load_trained_model():
    if not os.path.exists(WEIGHTS_PATH):
        url = f"https://drive.google.com/uc?id={WEIGHTS_FILE_ID}"
        gdown.download(url, WEIGHTS_PATH, quiet=False)

    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
        MaxPooling2D(2,2),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Conv2D(128, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.load_weights(WEIGHTS_PATH)
    return model

model = load_trained_model()

# ---------------- SIDEBAR ----------------




# ---------------- HERO ----------------
st.markdown(f"""
    <div class="hero-banner">
        <div class="hero-title"><img src="data:image/png;base64,{brain_icon_b64}" style="height:28px; vertical-align:middle; border-radius:6px;"> NeuroScan AI</div>

    </div>
""", unsafe_allow_html=True)

# ---------------- TABS ----------------
tab1, tab2 = st.tabs(["🔍  Prediction", "📊  Model Performance"])

with tab1:
    col_upload, col_result = st.columns([1, 1])

    with col_upload:
        st.markdown("""
            <div class="section-card">
                <div class="section-title">📤 Upload MRI Scan</div>
                <div class="section-caption">Supported formats: JPG, PNG</div>
        """, unsafe_allow_html=True)

        uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

        if uploaded_file is not None:
            img = Image.open(uploaded_file).convert("RGB")
            st.image(img, caption="Uploaded MRI Image", use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

    with col_result:
        st.markdown("""
            <div class="section-card">
                <div class="section-title">🧬 Prediction Result</div>
                <div class="section-caption">AI diagnosis based on trained CNN model</div>
        """, unsafe_allow_html=True)

        if uploaded_file is not None:
            img_resized = img.resize((128, 128))
            img_array = np.array(img_resized) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            pred = model.predict(img_array)
            confidence = pred[0][0]

            if confidence > 0.5:
                st.markdown(f"""
                    <div class="result-card tumor-detected">
                        <div class="result-icon">⚠️</div>
                        <div class="result-title">Tumor Detected</div>
                        <div class="result-sub">Confidence: {confidence*100:.2f}%</div>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class="result-card no-tumor">
                        <div class="result-icon">✅</div>
                        <div class="result-title">No Tumor Detected</div>
                        <div class="result-sub">Confidence: {(1-confidence)*100:.2f}%</div>
                    </div>
                """, unsafe_allow_html=True)

            st.progress(float(confidence))

            m1, m2, m3 = st.columns(3)
            m1.metric("Model", "CNN")
            m2.metric("Input Size", "128×128")
            m3.metric("Confidence", f"{max(confidence, 1-confidence)*100:.1f}%")
        else:
            st.markdown("""
                <div class="empty-state">
                    <div class="empty-icon">🩻</div>
                    Upload an MRI image to see the prediction result here
                </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

with tab2:
    st.markdown("""
        <div class="section-card">
            <div class="section-title">📈 Training Accuracy & Loss</div>
            <div class="section-caption">Model performance across training epochs</div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.image(os.path.join(BASE_DIR, "accuracy_plot.png"), caption="Model Accuracy", use_container_width=True)
    with col2:
        st.image(os.path.join(BASE_DIR, "loss_plot.png"), caption="Model Loss", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("""
        <div class="section-card">
            <div class="section-title">🎯 Confusion Matrix</div>
            <div class="section-caption">Prediction accuracy breakdown on test data</div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.image(os.path.join(BASE_DIR, "confusion_matrix.png"), caption="Confusion Matrix", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
