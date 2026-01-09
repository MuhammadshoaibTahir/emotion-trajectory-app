# test.py
"""
Restaurant Feedback Improvement System
Developed by:
- Muhammad Shoaib Tahir (COMSATS University Islamabad)
- Prof. Haroon Nasser Abdullah Alsager (Prince Sattam Bin Abdulaziz University)

Purpose:
Turn customer feedback into clear, actionable service improvements
"""

# ===============================
# Imports
# ===============================
import json
import streamlit as st
import pandas as pd
import numpy as np
from collections import Counter

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ===============================
# NLTK FIX (STREAMLIT CLOUD SAFE)
# ===============================
import os
import nltk

NLTK_DATA_DIR = "/home/appuser/nltk_data"
os.makedirs(NLTK_DATA_DIR, exist_ok=True)
nltk.data.path.append(NLTK_DATA_DIR)

def ensure_nltk():
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", download_dir=NLTK_DATA_DIR)

    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", download_dir=NLTK_DATA_DIR)

ensure_nltk()


# ===============================
# Streamlit Config
# ===============================
st.set_page_config(
    page_title="Restaurant Feedback Improvement System",
    layout="wide"
)

st.title("🍽️ Restaurant Feedback Improvement System")
st.caption(
    "Transform customer feedback into actionable service improvements"
)

# ===============================
# Model Loading
# ===============================
MODEL_NAME = "joeddav/distilbert-base-uncased-go-emotions-student"

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()
    labels = list(model.config.id2label.values())
    return tokenizer, model, labels

tokenizer, model, EMOTION_LABELS = load_model()

# ===============================
# Restaurant Locations (MAP FIX)
# ===============================
RESTAURANTS = pd.DataFrame({
    "restaurant": ["Marsool", "Talabat", "HungerStation"],
    "lat": [24.7136, 21.4858, 26.4207],   # Riyadh, Jeddah, Dammam
    "lon": [46.6753, 39.1925, 50.0888]
})

# ===============================
# Service Issue Rules
# ===============================
SERVICE_RULES = {
    "Food Quality": {
        "keywords": ["cold", "taste", "quality", "fresh", "stale", "burnt", "raw"],
        "action": "Improve food quality, freshness, taste consistency, and packaging."
    },
    "Delivery": {
        "keywords": ["late", "delay", "slow", "delivery", "time"],
        "action": "Improve delivery speed, rider coordination, and order tracking."
    },
    "Staff Behavior": {
        "keywords": ["rude", "staff", "behavior", "attitude"],
        "action": "Train staff on customer handling and professional behavior."
    },
    "Hygiene": {
        "keywords": ["dirty", "hygiene", "clean", "smell"],
        "action": "Improve cleanliness, hygiene standards, and kitchen sanitation."
    },
    "Seating": {
        "keywords": ["seat", "sitting", "crowded", "space"],
        "action": "Improve seating comfort, spacing, and dining environment."
    },
    "Pricing": {
        "keywords": ["price", "expensive", "cheap", "cost"],
        "action": "Review pricing strategy and value for money."
    },
    "App & Support": {
        "keywords": ["app", "refund", "cancel", "support"],
        "action": "Improve app reliability, refunds, and customer support response."
    }
}

# ===============================
# Helper Functions
# ===============================
def read_uploaded_file(file):
    ext = file.name.split(".")[-1].lower()

    if ext == "txt":
        return file.read().decode("utf-8")

    if ext == "csv":
        df = pd.read_csv(file)
        return " ".join(df.astype(str).values.flatten())

    if ext in ["xls", "xlsx"]:
        df = pd.read_excel(file)
        return " ".join(df.astype(str).values.flatten())

    if ext == "json":
        return json.dumps(json.load(file))

    return ""

def detect_emotions(text):
    sentences = sent_tokenize(text)
    inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.sigmoid(logits).numpy()
    dominant = [EMOTION_LABELS[int(np.argmax(p))] for p in probs]
    return sentences, dominant

def generate_recommendations(text, emotions):
    text_l = text.lower()
    actions = set()

    for rule in SERVICE_RULES.values():
        if any(k in text_l for k in rule["keywords"]):
            actions.add(rule["action"])

    if not actions:
        actions.add("Maintain current service quality. No major issues detected.")

    return actions, Counter(emotions)

# ===============================
# UI — Input
# ===============================
st.header("📥 Upload or Paste Customer Feedback")

uploaded_file = st.file_uploader(
    "Upload feedback file (TXT, CSV, XLSX, JSON)",
    type=["txt", "csv", "xls", "xlsx", "json"]
)

manual_text = st.text_area(
    "Or paste feedback text here",
    height=200
)

# ===============================
# Run Analysis
# ===============================
if st.button("Analyze Feedback"):
    if uploaded_file:
        feedback_text = read_uploaded_file(uploaded_file)
    else:
        feedback_text = manual_text

    if not feedback_text.strip():
        st.error("Please provide customer feedback.")
        st.stop()

    with st.spinner("Analyzing feedback..."):
        sentences, emotions = detect_emotions(feedback_text)
        actions, emotion_summary = generate_recommendations(feedback_text, emotions)

    # ===============================
    # Results
    # ===============================
    st.success("Analysis Completed")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Emotion Summary")
        for emo, count in emotion_summary.items():
            st.write(f"**{emo.capitalize()}**: {count}")

    with col2:
        st.subheader("🛠️ What Restaurant Owners Should Improve")
        for action in actions:
            st.success(action)

    st.subheader("📝 Example Feedback Insights")
    for s, e in list(zip(sentences, emotions))[:10]:
        st.write(f"• *{s}* → **{e}**")

    # ===============================
    # MAP SECTION (FIXED & VISIBLE)
    # ===============================
    st.subheader("📍 Major Food Delivery Services in Saudi Arabia")
    st.map(RESTAURANTS)

# ===============================
# Footer
# ===============================
st.info(
    "This system converts raw customer feedback into actionable service improvement "
    "recommendations for restaurant owners and food delivery platforms."
)