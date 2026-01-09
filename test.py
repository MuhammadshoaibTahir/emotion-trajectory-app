# test.py
"""
Emotion Trajectory Studio — Restaurant Owner Decision Support System
Author:
- Muhammad Shoaib Tahir (COMSATS University Islamabad)
- Prof. Haroon Nasser Abdullah Alsager (Prince Sattam Bin Abdulaziz University)

Purpose:
- Analyze customer feedback (text or files)
- Detect emotions
- Identify service problems
- Provide actionable improvement recommendations
"""

# ---------------------------
# Imports
# ---------------------------
import os
import json
import streamlit as st
import numpy as np
import pandas as pd
from collections import Counter
import nltk
from nltk.tokenize import sent_tokenize

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ---------------------------
# Streamlit config
# ---------------------------
st.set_page_config(
    page_title="Restaurant Feedback Improvement System",
    layout="wide"
)

st.title("🍽️ Restaurant Feedback Improvement System")
st.caption(
    "Turn customer feedback into clear service improvement actions"
)

# ---------------------------
# Model config
# ---------------------------
MODEL_NAME = "joeddav/distilbert-base-uncased-go-emotions-student"

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()
    return tokenizer, model, model.config.id2label

tokenizer, model, ID2LABEL = load_model()
EMOTIONS = list(ID2LABEL.values())

# ---------------------------
# SERVICE ISSUE RULE ENGINE
# ---------------------------
SERVICE_RULES = {
    "food_quality": {
        "keywords": ["cold", "taste", "quality", "fresh", "stale", "burnt", "raw"],
        "action": "Improve food quality, taste consistency, and temperature control."
    },
    "delivery": {
        "keywords": ["late", "delay", "slow", "time", "delivery"],
        "action": "Improve delivery speed, rider coordination, and order tracking."
    },
    "staff_behavior": {
        "keywords": ["rude", "staff", "behavior", "attitude"],
        "action": "Train staff on customer handling and professional behavior."
    },
    "hygiene": {
        "keywords": ["dirty", "hygiene", "clean", "smell"],
        "action": "Improve cleanliness, hygiene standards, and kitchen sanitation."
    },
    "seating": {
        "keywords": ["seat", "sitting", "crowded", "space"],
        "action": "Improve seating comfort, space management, and dining environment."
    },
    "pricing": {
        "keywords": ["price", "expensive", "cheap", "cost"],
        "action": "Review pricing strategy and value for money."
    },
    "app_service": {
        "keywords": ["app", "refund", "cancel", "support"],
        "action": "Improve app reliability, refund handling, and customer support."
    }
}

# ---------------------------
# Helper functions
# ---------------------------
def read_uploaded_file(uploaded_file):
    ext = uploaded_file.name.split(".")[-1].lower()

    if ext == "txt":
        return uploaded_file.read().decode("utf-8")

    if ext == "csv":
        df = pd.read_csv(uploaded_file)
        return " ".join(df.astype(str).values.flatten())

    if ext in ["xlsx", "xls"]:
        df = pd.read_excel(uploaded_file)
        return " ".join(df.astype(str).values.flatten())

    if ext == "json":
        data = json.load(uploaded_file)
        return json.dumps(data)

    return ""

def detect_emotions(text):
    sentences = sent_tokenize(text)
    inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.sigmoid(logits).numpy()

    dominant = [ID2LABEL[int(np.argmax(p))] for p in probs]
    return sentences, dominant

def generate_recommendations(text, dominant_emotions):
    text_l = text.lower()
    actions = set()

    for rule in SERVICE_RULES.values():
        if any(k in text_l for k in rule["keywords"]):
            actions.add(rule["action"])

    if not actions:
        actions.add("Maintain current service quality. No major issues detected.")

    emotion_summary = Counter(dominant_emotions)

    return actions, emotion_summary

# ---------------------------
# INPUT SECTION
# ---------------------------
st.header("📥 Upload Customer Feedback")

uploaded_file = st.file_uploader(
    "Upload feedback file (TXT, CSV, XLSX, JSON)",
    type=["txt", "csv", "xlsx", "xls", "json"]
)

manual_text = st.text_area(
    "Or paste feedback manually",
    height=200
)

if st.button("Analyze Feedback"):
    if uploaded_file:
        feedback_text = read_uploaded_file(uploaded_file)
    else:
        feedback_text = manual_text

    if not feedback_text.strip():
        st.error("Please provide customer feedback.")
        st.stop()

    with st.spinner("Analyzing feedback..."):
        sentences, dominant_emotions = detect_emotions(feedback_text)
        actions, emotion_summary = generate_recommendations(
            feedback_text, dominant_emotions
        )

    # ---------------------------
    # RESULTS
    # ---------------------------
    st.success("Analysis Completed")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Emotion Overview")
        for emo, count in emotion_summary.items():
            st.write(f"**{emo.capitalize()}**: {count}")

    with col2:
        st.subheader("🛠️ What You Should Improve")
        for action in actions:
            st.success(action)

    st.subheader("📝 Sample Sentence Insights")
    for s, e in zip(sentences[:10], dominant_emotions[:10]):
        st.write(f"• *{s}* → **{e}**")

    st.subheader("📍 Service Platforms (Saudi Arabia)")
    map_df = pd.DataFrame({
        "lat": [24.7136, 21.4858, 26.4207],
        "lon": [46.6753, 39.1925, 50.0888],
        "service": ["Marsool", "Talabat", "HungerStation"]
    })
    st.map(map_df)

st.info(
    "This system converts raw customer feedback into clear service improvement actions for restaurant owners."
)
