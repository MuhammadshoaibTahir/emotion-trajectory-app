# test.py
"""
Emotion Trajectory Studio — Upgraded (GoEmotions 28-class, no zero-shot)
- Single-file Streamlit app
- Sentence-level 28-emotion (GoEmotions) multi-label detection
- 2D animated emotion curves + controls
- 3D Valence × Arousal × Sentence trajectory
- Word-level emotion heatmap
- Transition matrix + Sankey
- Keyword extraction (TF×IDF-like)
- Cohesion & stability metrics
- Extra visualizations: radar profile, emotion rhythm, polarity waterfall
- Optional: Whisper transcription (opt-in)
- Robust handling for Streamlit Cloud (NLTK local data fallback)
- Exports: JSON, CSV, HTML
Notes:
- This uses a strong GoEmotions model. No zero-shot used anywhere.
- Multi-label inference uses sigmoid; dominant per-sentence is the highest probability emotion.
- Heuristics included to reduce obvious mislabels (e.g., override fear when clear positive words present).
"""
# ---------------------------
# Imports & configuration
# ---------------------------
import os
import io
import json
import math
import datetime
from tempfile import NamedTemporaryFile
from typing import List, Dict, Tuple, Optional
from collections import Counter, defaultdict

import streamlit as st
st.set_page_config(page_title="Emotion Trajectory Studio", layout="wide", initial_sidebar_state="expanded")
st.markdown(
    "<h4 style='text-align:center; color:gray;'>Created by Muhammad Shoaib Tahir — Upgraded</h4>",
    unsafe_allow_html=True
)

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.io as pio
import nltk

# NLTK local data setup (important for Cloud)
try:
    base_dir = os.path.dirname(__file__) if "__file__" in globals() else os.getcwd()
except Exception:
    base_dir = os.getcwd()
nltk_local = os.path.join(base_dir, "nltk_data")
if not os.path.exists(nltk_local):
    try:
        os.makedirs(nltk_local, exist_ok=True)
    except Exception:
        pass
if nltk_local not in nltk.data.path:
    nltk.data.path.append(nltk_local)

for _res in ("tokenizers/punkt", "tokenizers/punkt_tab"):
    try:
        nltk.data.find(_res)
    except LookupError:
        try:
            nltk.download(_res.split("/")[-1], download_dir=nltk_local)
        except Exception:
            pass

from nltk.tokenize import sent_tokenize, word_tokenize

# Transformers & torch (guarded)
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

# Whisper optional
try:
    import whisper
    WHISPER_AVAILABLE = True
except Exception:
    WHISPER_AVAILABLE = False

# SciPy optional
try:
    from scipy.signal import savgol_filter
    import scipy.stats as stats
    SG_AVAILABLE = True
    STAT_AVAILABLE = True
except Exception:
    SG_AVAILABLE = False
    try:
        import scipy.stats as stats
        STAT_AVAILABLE = True
    except Exception:
        STAT_AVAILABLE = False

# ---------------------------
# Model & emotion config
# ---------------------------
# Recommended GoEmotions specialist model (student distilled version — fast & accurate for production)
MODEL_NAME = "joeddav/distilbert-base-uncased-go-emotions-student"

# We'll load labels dynamically from model config after loading;
# but provide canonical GoEmotions label order here as fallback and for VAD mapping:
GOEMOTIONS_LABELS = [
    "admiration","amusement","anger","annoyance","approval","caring","confusion","curiosity","desire","disappointment",
    "disapproval","disgust","embarrassment","excitement","fear","gratitude","grief","joy","love","nervousness",
    "optimism","pride","realization","relief","remorse","sadness","surprise","neutral"
]

# A basic Valence-Arousal-Dominance (VAD) mapping for GoEmotions labels (heuristic values 0..1)
# These are approximate and used for visualization; adjust if you have better values.
VAD_MAP = {
    "admiration": (0.7, 0.3, 0.7),
    "amusement": (0.8, 0.6, 0.7),
    "anger": (0.15, 0.8, 0.6),
    "annoyance": (0.25, 0.6, 0.5),
    "approval": (0.7, 0.3, 0.6),
    "caring": (0.6, 0.3, 0.6),
    "confusion": (0.4, 0.5, 0.4),
    "curiosity": (0.5, 0.5, 0.5),
    "desire": (0.6, 0.6, 0.5),
    "disappointment": (0.2, 0.35, 0.3),
    "disapproval": (0.25, 0.5, 0.4),
    "disgust": (0.1, 0.6, 0.4),
    "embarrassment": (0.3, 0.5, 0.4),
    "excitement": (0.85, 0.85, 0.8),
    "fear": (0.1, 0.85, 0.25),
    "gratitude": (0.8, 0.3, 0.6),
    "grief": (0.05, 0.25, 0.2),
    "joy": (0.9, 0.6, 0.8),
    "love": (0.9, 0.4, 0.7),
    "nervousness": (0.2, 0.8, 0.3),
    "optimism": (0.8, 0.4, 0.6),
    "pride": (0.8, 0.4, 0.7),
    "realization": (0.6, 0.4, 0.5),
    "relief": (0.75, 0.3, 0.6),
    "remorse": (0.2, 0.35, 0.3),
    "sadness": (0.1, 0.25, 0.2),
    "surprise": (0.6, 0.9, 0.5),
    "neutral": (0.5, 0.3, 0.5)
}

# Map GoEmotions labels to a smaller 'basic' set if needed
BASIC_MAPPING = {
    "joy":"happy","amusement":"happy","admiration":"happy","excitement":"happy","gratitude":"happy","optimism":"happy","relief":"happy","pride":"happy",
    "love":"happy","approval":"happy",
    "sadness":"sad","grief":"sad","disappointment":"sad","remorse":"sad",
    "anger":"anger","annoyance":"anger","disgust":"anger","disapproval":"anger",
    "fear":"fear","nervousness":"fear",
    "surprise":"surprise",
    "confusion":"neutral","curiosity":"neutral","realization":"neutral","neutral":"neutral","caring":"neutral","desire":"neutral","embarrassment":"neutral"
}

# UI colors (expand dynamically if more labels)
BASE_COLOR = "#f6f6f6"
EMOTION_COLORS = {lab: "#f6f6f6" for lab in GOEMOTIONS_LABELS}  # simple default; UI uses borders to indicate emotion
EMOTION_BORDER = {lab: "#cccccc" for lab in GOEMOTIONS_LABELS}
EMOTION_TEXT_COLOR = "#222222"

# small manually chosen palette for some common labels (improves display)
PALETTE = {
    "joy":"#fff8d6","love":"#ffe6f2","anger":"#ffd6d6","fear":"#e6e6ff","sadness":"#e8f0ff","surprise":"#eaffeb",
    "neutral":"#f6f6f6","excitement":"#fff0cc","disgust":"#f0e6ff"
}
for k,v in PALETTE.items():
    if k in EMOTION_COLORS:
        EMOTION_COLORS[k] = v
        EMOTION_BORDER[k] = v

# Emoji mapping for basic emotions
EMOJI = {
    "happy":"😊","sad":"😢","anger":"😡","fear":"😨","surprise":"😲","neutral":"😐"
}

DEVICE = torch.device("cuda" if (TORCH_AVAILABLE and torch.cuda.is_available()) else "cpu") if TORCH_AVAILABLE else None

# ---------------------------
# Small helpers
# ---------------------------
def _escape_html(text: str) -> str:
    if text is None:
        return ""
    return (str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br/>"))

def clean_text_basic(text: str) -> str:
    if text is None:
        return ""
    t = str(text)
    # remove excessive whitespace and control chars
    t = t.replace("\r", " ").replace("\n", " ").strip()
    # remove URLs
    t = __import__("re").sub(r"http\S+|www\.\S+", "", t)
    # normalize spaces
    t = __import__("re").sub(r"\s+", " ", t)
    return t.strip()

def safe_insert_sentence_column(df: pd.DataFrame, sentences: List[str], dominants: List[Optional[str]]) -> pd.DataFrame:
    if df is None or df.shape[0] == 0:
        df = pd.DataFrame({"sentence": sentences, "dominant": dominants})
        for lab in EMOTION_LABELS:
            if lab not in df.columns:
                df[lab] = np.nan
        cols = ["sentence", "dominant"] + EMOTION_LABELS
        return df[cols]
    n_df = df.shape[0]
    n_sent = len(sentences)
    if n_df != n_sent:
        m = min(n_df, n_sent)
        df = df.head(m).reset_index(drop=True)
        sentences = sentences[:m]
        dominants = dominants[:m]
    if "sentence" not in df.columns:
        df.insert(0, "sentence", sentences)
    else:
        df["sentence"] = sentences
    if "dominant" not in df.columns:
        df.insert(1, "dominant", dominants)
    else:
        df["dominant"] = dominants
    return df

def ensure_length_alignment(list_a: List, list_b: List) -> Tuple[List, List]:
    m = min(len(list_a), len(list_b))
    return list_a[:m], list_b[:m]

# ---------------------------
# Smoothing & numeric tools
# ---------------------------
def smooth_series(arr: np.ndarray, window: int = 3, method: str = "mean") -> np.ndarray:
    if arr is None or arr.size == 0:
        return arr
    if window <= 1:
        return arr
    if method == "mean":
        kernel = np.ones(window) / window
        return np.convolve(arr, kernel, mode='same')
    if method == "savgol" and SG_AVAILABLE:
        win = int(window) if int(window) % 2 == 1 else int(window) + 1
        win = min(win, max(3, len(arr) if len(arr) % 2 == 1 else len(arr) - 1))
        try:
            return savgol_filter(arr, window_length=max(3, win), polyorder=2, mode='nearest')
        except Exception:
            kernel = np.ones(window) / window
            return np.convolve(arr, kernel, mode='same')
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode='same')

# ---------------------------
# Keywords (TF×IDF-like)
# ---------------------------
def extract_keywords_tfidf_like(text: str, top_k: int = 15, ngram_range: Tuple[int,int] = (1,2), stopwords: Optional[set] = None) -> List[Tuple[str,float]]:
    if not text or not text.strip():
        return []
    if stopwords is None:
        stopwords = set()
    sentences = sent_tokenize(text)
    docs = []
    for s in sentences:
        toks = [w.lower() for w in word_tokenize(s) if any(c.isalpha() for c in w)]
        toks = [t for t in toks if len(t) > 1 and t not in stopwords]
        ngrams = []
        for i in range(len(toks)):
            if 1 <= ngram_range[0] <= 1:
                ngrams.append(toks[i])
            if ngram_range[1] >= 2 and i + 1 < len(toks):
                ngrams.append(toks[i] + " " + toks[i+1])
        docs.append(ngrams)
    tf = Counter()
    for d in docs:
        tf.update(d)
    df_counts = Counter()
    unique_terms = set([t for d in docs for t in set(d)])
    for term in unique_terms:
        for d in docs:
            if term in d:
                df_counts[term] += 1
    N = max(1, len(docs))
    scores = {}
    for term, count in tf.items():
        idf = math.log((N + 1)/(1 + df_counts.get(term, 0))) + 1.0
        scores[term] = count * idf
    ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return ordered

# ---------------------------
# Cohesion & stability
# ---------------------------
def cohesion_and_stability(valence_array: np.ndarray, dominants: List[Optional[str]]) -> Dict[str, float]:
    if valence_array is None or valence_array.size == 0:
        val_var = 0.0
    else:
        val_var = float(np.nanvar(valence_array))
    if dominants:
        counts = Counter([d if d is not None else "none" for d in dominants])
        probs = np.array(list(counts.values()), dtype=float) / sum(counts.values())
        if STAT_AVAILABLE:
            ent = float(stats.entropy(probs))
        else:
            ent = float(-sum(p * math.log(p + 1e-12) for p in probs))
    else:
        ent = 0.0
    stability = (1.0 / (1.0 + val_var)) * (1.0 / (1.0 + ent))
    return {"valence_variance": val_var, "dominant_entropy": ent, "stability_score": stability}

# ---------------------------
# HTML report builder
# ---------------------------
def create_html_report(title: str, text: str, figs: Dict[str, go.Figure], tables: Dict[str, pd.DataFrame], metadata: Dict) -> str:
    parts = []
    parts.append(f"<h1>{_escape_html(title)}</h1>")
    parts.append(f"<p>Generated: {_escape_html(datetime.datetime.utcnow().isoformat())} UTC</p>")
    parts.append("<h2>Input Text</h2>")
    parts.append(f"<pre style='white-space:pre-wrap'>{_escape_html(text)}</pre>")
    for name, fig in (figs or {}).items():
        try:
            frag = pio.to_html(fig, full_html=False, include_plotlyjs='cdn')
            parts.append(f"<h3>{_escape_html(name)}</h3>")
            parts.append(frag)
        except Exception as e:
            parts.append(f"<p>Failed to render figure {name}: {_escape_html(str(e))}</p>")
    for name, df in (tables or {}).items():
        parts.append(f"<h3>{_escape_html(name)}</h3>")
        try:
            parts.append(df.to_html(index=False, escape=False))
        except Exception:
            parts.append(f"<pre>{_escape_html(str(df))}</pre>")
    parts.append("<h3>Metadata</h3>")
    parts.append(f"<pre>{_escape_html(json.dumps(metadata, indent=2))}</pre>")
    html = "<html><head><meta charset='utf-8'></head><body>{}</body></html>".format("\n".join(parts))
    return html

# ---------------------------
# Sentence highlight helpers
# ---------------------------
def sentence_detail_html(index: int, sentence: str, probs: Optional[List[float]] = None, labels: Optional[List[str]] = None) -> str:
    s = _escape_html(sentence)
    prob_html = ""
    if probs is not None and labels is not None:
        rows = []
        for lab, p in zip(labels, probs):
            pct = int(max(0, min(100, round(p * 100))))
            color = EMOTION_BORDER.get(lab, "#aaaaaa")
            rows.append(
                f"<div style='display:flex;align-items:center;margin-bottom:6px;'>"
                f"<div style='width:120px;font-size:13px;color:{EMOTION_TEXT_COLOR};'>{lab}</div>"
                f"<div style='flex:1;background:#f1f1f1;border-radius:6px;height:10px;margin-left:8px;'>"
                f"<div style='width:{pct}%;height:100%;background:{color};border-radius:6px;'></div></div>"
                f"<div style='width:50px;text-align:right;font-size:12px;color:#444;margin-left:8px;'>{p:.2f}</div>"
                f"</div>"
            )
        prob_html = "<div style='margin-top:8px;'>" + "\n".join(rows) + "</div>"
    html = f"""
    <div style='padding:10px;border-radius:8px;border:1px solid #eee;background:#fff;max-width:820px;'>
      <div style='font-weight:600;margin-bottom:8px;'>Sentence {index+1} details</div>
      <div style='white-space:pre-wrap;font-size:14px;color:{EMOTION_TEXT_COLOR};'>{s}</div>
      {prob_html}
    </div>
    """
    return html

def render_sentence_highlights(sentences: List[str], dominants: List[Optional[str]], probs: Optional[np.ndarray] = None,
                               show_confidence: bool = True, include_mini_bar: bool = True):
    if not sentences:
        st.info("No sentences to display.")
        return
    confidences = []
    if probs is not None and getattr(probs, "size", 0):
        for i in range(min(len(sentences), probs.shape[0])):
            confidences.append(float(np.max(probs[i])))
    else:
        confidences = [0.0] * len(sentences)
    for i, sent in enumerate(sentences):
        emo = dominants[i] if i < len(dominants) else None
        conf = confidences[i] if i < len(confidences) else 0.0
        header_html = f"<div style='display:flex;align-items:center;gap:8px;'><div style='font-weight:700;color:{EMOTION_TEXT_COLOR};'>#{i+1}</div>"
        if emo:
            header_html += f"<div style='font-weight:600;color:{EMOTION_TEXT_COLOR};padding:2px 8px;border-radius:6px;border:1px solid {EMOTION_BORDER.get(emo,'#ddd')};background:{EMOTION_COLORS.get(emo,'#fff')};'>{emo.capitalize()}</div>"
        header_html += "</div>"
        box_html = f"""
        <div style="background:{EMOTION_COLORS.get(emo,'#f6f6f6')};border:1px solid {EMOTION_BORDER.get(emo,'#ddd')};padding:10px;border-radius:10px;margin-bottom:8px;">
            {header_html}
            <div style="margin-top:8px;color:{EMOTION_TEXT_COLOR};font-size:14px;line-height:1.45;white-space:pre-wrap;">{_escape_html(sent)}</div>
        """
        if include_mini_bar and show_confidence:
            pct = max(3, min(100, int(conf * 100)))
            box_html += f"<div style='height:8px;background:#f2f2f2;border-radius:4px;margin-top:8px;'><div style='width:{pct}%;height:100%;background:{EMOTION_BORDER.get(emo,'#cccccc')};border-radius:4px;'></div></div>"
            box_html += f"<div style='font-size:12px;color:#444;margin-top:6px;'>Confidence: {conf:.2f}</div>"
        box_html += "</div>"
        st.markdown(box_html, unsafe_allow_html=True)
        with st.expander(f"Details: Sentence {i+1}"):
            if probs is not None and getattr(probs, "size", 0) and i < probs.shape[0]:
                detail_html = sentence_detail_html(i, sent, probs[i].tolist(), EMOTION_LABELS)
                st.markdown(detail_html, unsafe_allow_html=True)
            else:
                st.write("No per-emotion probabilities available for this sentence.")

# ---------------------------
# Model loader (cached)
# ---------------------------
if not TORCH_AVAILABLE:
    st.error("Torch/Transformers not installed in this environment. Please install torch and transformers.")
    # Early exit UI (no need to crash)
else:
    @st.cache_resource(show_spinner=True)
    def load_emotion_model(model_name: str = MODEL_NAME):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        model.to(DEVICE)
        model.eval()
        # attempt to read label mapping; fall back to canonical order
        cfg = getattr(model, "config", None)
        id2label = None
        if cfg is not None and hasattr(cfg, "id2label"):
            id2label = cfg.id2label
        if id2label:
            # Map to list ordered by index keys
            try:
                labels = [id2label[i] for i in range(len(id2label))]
            except Exception:
                labels = GOEMOTIONS_LABELS
        else:
            labels = GOEMOTIONS_LABELS
        return tokenizer, model, labels

    # load once
    try:
        tokenizer, model, EMOTION_LABELS = load_emotion_model()
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        EMOTION_LABELS = GOEMOTIONS_LABELS
        tokenizer = None
        model = None

NUM_EMOTIONS = len(EMOTION_LABELS)
if not VAD_MAP:
    VAD_MAP = {lab: (0.5,0.3,0.5) for lab in EMOTION_LABELS}

# ---------------------------
# Whisper helper
# ---------------------------
def whisper_transcribe_file(uploaded_file, model_size="small"):
    if not WHISPER_AVAILABLE:
        raise RuntimeError("Whisper not installed.")
    tmp = NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1])
    tmp.write(uploaded_file.getbuffer()); tmp.flush(); tmp.close()
    w_model = whisper.load_model(model_size)
    res = w_model.transcribe(tmp.name)
    try:
        os.unlink(tmp.name)
    except Exception:
        pass
    return res.get("text", "")

# ---------------------------
# Batched inference & trajectory builder
# ---------------------------
def batched_infer(sentences: List[str], tokenizer, model, batch_size: int = 32, threshold: float = 0.25) -> np.ndarray:
    """
    Multi-label inference using sigmoid. Returns probabilities array shape (n_sentences, n_emotions).
    threshold is not applied here — returned raw probabilities; caller may threshold for binary labels.
    """
    if sentences is None or len(sentences) == 0:
        return np.zeros((0, NUM_EMOTIONS), dtype=float)
    all_probs = []
    with torch.no_grad():
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i+batch_size]
            enc = tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=256)
            enc = {k: v.to(DEVICE) for k, v in enc.items()}
            out = model(**enc)
            # For multi-label GoEmotions student models logits -> use sigmoid
            logits = out.logits
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
    return np.vstack(all_probs)

def batched_infer_words(words: List[str], tokenizer, model, batch_size: int = 64, max_words: int = 1000) -> np.ndarray:
    words = words[:max_words]
    if not words:
        return np.zeros((0, NUM_EMOTIONS), dtype=float)
    return batched_infer(words, tokenizer, model, batch_size=batch_size)

def compute_valence_arousal_from_probs(probs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Given probs (n_sentences, n_emotions), compute valence, arousal, dominance per sentence as weighted sums.
    """
    if probs is None or probs.size == 0:
        return np.array([]), np.array([]), np.array([])
    vs = []; as_ = []; ds = []
    for p in probs:
        if p.sum() == 0:
            vs.append(0.0); as_.append(0.0); ds.append(0.0)
        else:
            v = float(sum(p[i] * VAD_MAP.get(EMOTION_LABELS[i], (0.5,0.3,0.5))[0] for i in range(len(EMOTION_LABELS))))
            a = float(sum(p[i] * VAD_MAP.get(EMOTION_LABELS[i], (0.5,0.3,0.5))[1] for i in range(len(EMOTION_LABELS))))
            d = float(sum(p[i] * VAD_MAP.get(EMOTION_LABELS[i], (0.5,0.3,0.5))[2] for i in range(len(EMOTION_LABELS))))
            vs.append(v); as_.append(a); ds.append(d)
    return np.array(vs), np.array(as_), np.array(ds)

def build_trajectory(text: str, tokenizer, model,
                     word_heatmap: bool = True,
                     word_limit: int = 300,
                     smooth_window: int = 1,
                     smooth_method: str = "mean",
                     batch_size: int = 32,
                     progress_callback=None,
                     topk:int = 1,
                     override_positive_fear: bool = True) -> Dict:
    # Clean and split sentences
    text = clean_text_basic(text)
    sentences = sent_tokenize(text)
    n = len(sentences)
    if n == 0:
        sentence_scores = np.zeros((0, NUM_EMOTIONS))
    else:
        if progress_callback is None:
            sentence_scores = batched_infer(sentences, tokenizer, model, batch_size=batch_size)
        else:
            parts = []
            for i in range(0, n, batch_size):
                b = sentences[i:i+batch_size]
                p = batched_infer(b, tokenizer, model, batch_size=batch_size)
                parts.append(p)
                try:
                    progress_callback(min(1.0, (i + batch_size) / max(1, n)))
                except Exception:
                    pass
            sentence_scores = np.vstack(parts) if parts else np.zeros((0, NUM_EMOTIONS))

    # Dominant: highest probability emotion per sentence (even though multi-label)
    dominants = []
    for i, probs in enumerate(sentence_scores):
        if probs.sum() == 0:
            dominants.append(None)
        else:
            top_idx = int(np.argmax(probs))
            top_label = EMOTION_LABELS[top_idx]
            # simple heuristic: if top is 'fear' but sentence contains positive tokens, override
            if override_positive_fear and top_label == "fear":
                s = sentences[i].lower()
                positive_tokens = ["enjoy", "enjoying", "happy", "joy", "love", "loving", "delighted", "pleased", "fun"]
                if any(tok in s for tok in positive_tokens):
                    # pick next best non-fear label
                    sorted_idx = np.argsort(probs)[::-1]
                    chosen = None
                    for idx in sorted_idx:
                        lab = EMOTION_LABELS[int(idx)]
                        if lab != "fear":
                            chosen = lab; break
                    if chosen:
                        top_label = chosen
            dominants.append(top_label)

    # Word-level heatmap
    word_tokens = []
    word_scores = np.zeros((0, NUM_EMOTIONS))
    if word_heatmap and n > 0:
        words = []
        for s in sentences:
            toks = [w for w in word_tokenize(s) if any(c.isalpha() for c in w)]
            words.extend(toks)
        words = words[:word_limit]
        word_tokens = words
        if words:
            word_scores = batched_infer_words(words, tokenizer, model, batch_size=64, max_words=word_limit)
        else:
            word_scores = np.zeros((0, NUM_EMOTIONS))

    # smoothing
    if sentence_scores.size == 0:
        smoothed = sentence_scores.copy()
    else:
        if smooth_window and smooth_window > 1:
            cols = [smooth_series(sentence_scores[:, i], window=smooth_window, method=smooth_method) for i in range(sentence_scores.shape[1])]
            smoothed = np.vstack(cols).T
        else:
            smoothed = sentence_scores.copy()

    valence, arousal, dominance = compute_valence_arousal_from_probs(smoothed)
    return {
        "sentences": sentences,
        "sentence_scores": smoothed,
        "dominant": dominants,
        "word_tokens": word_tokens,
        "word_scores": word_scores,
        "valence": valence,
        "arousal": arousal,
        "dominance": dominance
    }

# ---------------------------
# Visualization builders (re-using earlier functions)
# ---------------------------
def build_2d_animated(scores: np.ndarray, labels: List[str], frame_duration:int=250) -> go.Figure:
    n_sent = scores.shape[0] if getattr(scores, "size", 0) else 0
    n_em = scores.shape[1] if getattr(scores, "size", 0) else len(labels)
    x = list(range(n_sent))
    frames = []
    for t in range(n_sent):
        data = [go.Scatter(x=x[:t+1], y=scores[:t+1,e], mode='lines+markers', name=labels[e], showlegend=(t==0)) for e in range(n_em)]
        frames.append(go.Frame(data=data, name=str(t)))
    traces = [go.Scatter(x=[], y=[], mode='lines+markers', name=labels[e]) for e in range(n_em)]
    layout = go.Layout(title="Animated Emotion Curves", xaxis=dict(title="Sentence Index"), yaxis=dict(title="Probability", range=[0,1]),
                       updatemenus=[{"type":"buttons","buttons":[{"label":"Play","method":"animate","args":[None, {"frame":{"duration":frame_duration,"redraw":True},"fromcurrent":True}]},{"label":"Pause","method":"animate","args":[[None], {"frame":{"duration":0,"redraw":False},"mode":"immediate"}]}]}])
    fig = go.Figure(data=traces, frames=frames, layout=layout)
    for e in range(n_em):
        y = scores[:,e] if getattr(scores, "size", 0) else []
        fig.add_trace(go.Scatter(x=list(range(n_sent)), y=y, mode='lines', name=labels[e], visible=True))
    return fig

def build_3d_animated(valence: np.ndarray, arousal: np.ndarray, dominants: List[Optional[str]], frame_duration:int=250) -> go.Figure:
    n = len(valence)
    if n == 0:
        fig = go.Figure(); fig.update_layout(title="3D Trajectory (no data)"); return fig
    x = list(valence); y = list(arousal); z = list(range(n))
    # color map for labels — fallback to simple gray where missing
    cmap = {}
    for lab in EMOTION_LABELS:
        cmap[lab] = EMOTION_BORDER.get(lab, "#bbbbbb")
    colors = [cmap.get(d, "#bbbbbb") for d in dominants]
    frames = []
    for t in range(n):
        frame_data = [
            go.Scatter3d(x=x[:t+1], y=y[:t+1], z=z[:t+1], mode='lines', line=dict(width=4, color='rgba(40,80,200,0.9)')),
            go.Scatter3d(x=x[:t+1], y=y[:t+1], z=z[:t+1], mode='markers', marker=dict(size=6, color=colors[:t+1]), text=[f"{i}: {dominants[i]}" for i in range(t+1)], hovertemplate="%{text}<br>x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z}<extra></extra>")
        ]
        frames.append(go.Frame(data=frame_data, name=str(t)))
    fig = go.Figure(data=[go.Scatter3d(x=[], y=[], z=[], mode='lines'), go.Scatter3d(x=[], y=[], z=[], mode='markers')], frames=frames)
    fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='lines', line=dict(width=2, color='rgba(120,120,120,0.4)'), showlegend=False))
    fig.update_layout(scene=dict(xaxis_title='Valence', yaxis_title='Arousal', zaxis_title='Sentence Index'), title="3D Animated Trajectory", updatemenus=[dict(type='buttons', showactive=False, y=1.05, x=0.1, xanchor='left', yanchor='top', pad=dict(t=0,r=10), buttons=[dict(label='Play', method='animate', args=[None, dict(frame=dict(duration=frame_duration, redraw=True), fromcurrent=True)]), dict(label='Pause', method='animate', args=[[None], dict(frame=dict(duration=0, redraw=False), mode='immediate')])])])
    return fig

def build_heatmap(words: List[str], word_scores: np.ndarray, labels: List[str], max_words_display:int=60) -> go.Figure:
    n = min(len(words), max_words_display)
    x = [f"{i}:{w}" for i, w in enumerate(words[:n])]
    z = word_scores[:n,:].T if getattr(word_scores, "size", 0) else np.zeros((len(labels), n))
    fig = go.Figure(data=go.Heatmap(z=z, x=x, y=labels, hovertemplate="Word: %{x}<br>Emotion: %{y}<br>Score: %{z:.3f}<extra></extra>"))
    fig.update_layout(title="Word-level Emotion Heatmap", xaxis_tickangle=45, height=360)
    return fig

def build_sankey_and_matrix(dominants: List[Optional[str]]) -> Tuple[go.Figure, np.ndarray, go.Figure]:
    labels = EMOTION_LABELS
    idx = {l:i for i,l in enumerate(labels)}
    n = len(labels)
    matrix = np.zeros((n,n), dtype=int)
    for a,b in zip(dominants[:-1], dominants[1:]):
        if a is None or b is None: continue
        if a not in idx or b not in idx: continue
        matrix[idx[a], idx[b]] += 1
    sources, targets, values = [], [], []
    for i in range(n):
        for j in range(n):
            if matrix[i,j] > 0:
                sources.append(i); targets.append(j); values.append(int(matrix[i,j]))
    if not values:
        sank = go.Figure(); sank.update_layout(title="Not enough transitions for Sankey")
    else:
        sank = go.Figure(data=[go.Sankey(node=dict(label=labels), link=dict(source=sources, target=targets, value=values))])
        sank.update_layout(title="Sentence-to-sentence Emotion Transitions (Sankey)", height=420)
    mat_fig = go.Figure(data=go.Heatmap(z=matrix, x=labels, y=labels, hovertemplate="From %{y} -> %{x}: %{z}<extra></extra>"))
    mat_fig.update_layout(title="Transition Matrix (counts)", height=420)
    return sank, matrix, mat_fig

def build_gauge(valence: np.ndarray) -> Tuple[go.Figure, float]:
    if getattr(valence, "size", 0):
        mean_v = float(np.nanmean(valence))
    else:
        mean_v = 0.0
    mapped = (mean_v) * 100  # valence in 0..1 -> 0..100
    fig = go.Figure(go.Indicator(mode="gauge+number+delta", value=mapped, title={"text":"Average Valence (0..100)"}, gauge={"axis":{"range":[0,100]}}))
    fig.update_layout(height=260)
    return fig, mean_v

def build_radar_profile(sent_scores: np.ndarray, labels: List[str]) -> go.Figure:
    if not getattr(sent_scores, "size", 0):
        fig = go.Figure(); fig.update_layout(title="Emotion Radar (no data)"); return fig
    avg = np.nanmean(sent_scores, axis=0).tolist()
    r = avg + [avg[0]]
    theta = labels + [labels[0]]
    fig = go.Figure(go.Scatterpolar(r=r, theta=theta, fill='toself', name='Average emotion profile'))
    fig.update_layout(title="Average Emotion Radar", polar=dict(radialaxis=dict(visible=True, range=[0,1])))
    return fig

def build_emotion_rhythm(sent_scores: np.ndarray) -> go.Figure:
    if not getattr(sent_scores, "size", 0):
        fig = go.Figure(); fig.update_layout(title="Emotion Rhythm (no data)"); return fig
    x = list(range(sent_scores.shape[0]))
    fig = go.Figure()
    for i, lab in enumerate(EMOTION_LABELS):
        fig.add_trace(go.Scatter(x=x, y=sent_scores[:,i], mode='lines', stackgroup='one', name=lab))
    fig.update_layout(title="Emotion Rhythm (stacked)", xaxis_title="Sentence", yaxis_title="Probability", height=360)
    return fig

def build_polarity_waterfall(sent_scores: np.ndarray) -> go.Figure:
    if not getattr(sent_scores, "size", 0):
        fig = go.Figure(); fig.update_layout(title="Polarity Waterfall (no data)"); return fig
    v, _, _ = compute_valence_arousal_from_probs(sent_scores)
    deltas = v - 0.5  # center around 0
    x = [f"Sent {i+1}" for i in range(len(deltas))]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=x, y=deltas, name='Polarity (valence-0.5)'))
    fig.update_layout(title="Polarity Waterfall (Valence per sentence)", xaxis_tickangle=45, height=360)
    return fig

# ---------------------------
# UI: Sidebar & Inputs
# ---------------------------
st.title("Emotion Trajectory Studio - LingXsenti")

with st.sidebar:
    st.header("Settings & Extras")
    st.write(f"Model: `{MODEL_NAME}`")
    smoothing_method = st.selectbox("Smoothing method", options=(["mean","savgol"] if SG_AVAILABLE else ["mean"]), index=0)
    smoothing_window = st.slider("Smoothing window (sentences)", min_value=1, max_value=11, value=3)
    compute_word_heatmap = st.checkbox("Compute word-level heatmap (slower)", value=True)
    max_word_count = st.number_input("Max words to analyze for heatmap", min_value=20, max_value=3000, value=300, step=20)
    batch_size = st.number_input("Inference batch size", min_value=8, max_value=256, value=32, step=8)
    animate_2d = st.checkbox("Enable animated 2D curves", value=True)
    animate_3d = st.checkbox("Enable animated 3D trajectory", value=True)
    include_word_in_json = st.checkbox("Include word-level scores in JSON export", value=False)
    top_k = st.slider("Top-K emotions per sentence (table)", min_value=1, max_value=len(EMOTION_LABELS), value=2)
    st.markdown("---")
    st.subheader("Optional extras")
    use_whisper = st.checkbox("Enable Whisper transcription (if installed)", value=False)
    st.markdown("---")
    st.subheader("Samples")
    if st.button("Load short sample"):
        st.session_state["sample_text"] = "I was nervous before the talk. Then I felt joy when people applauded. A surprise phone call made me feel loved."
        st.rerun()
    if st.button("Load story sample"):
        st.session_state["sample_text"] = ("It was a gray morning. I felt sadness and fear, but a funny moment brought a smile. "
                                           "Later, an unexpected call filled me with love and joy.")
        st.rerun()

# Input area
col1, col2 = st.columns([1, 2])
with col1:
    st.header("Input")
    input_mode = st.radio("Input mode:", ["Text", "Upload Audio"])
    text_input = ""
    uploaded_audio = None
    if input_mode == "Text":
        text_input = st.text_area("Paste your text here (or write):", value=st.session_state.get("sample_text", ""), height=220)
    else:
        uploaded_audio = st.file_uploader("Upload audio (wav/mp3/m4a/ogg)", type=["wav", "mp3", "m4a", "ogg"])
        whisper_model_choice = st.selectbox("Whisper model (if installed):", ["tiny","base","small","medium","large"], index=2)
        st.checkbox("Auto-transcribe audio (requires 'whisper' installed)", value=WHISPER_AVAILABLE, key="auto_transcribe")

with col2:
    st.header("Quick Notes & Tips")
    st.info("This app predicts 28 emotions per sentence and visualizes trajectories in valence/arousal space.")
    st.write("For long documents, increase batch size and smoothing window for smoother curves.")
    if use_whisper and not WHISPER_AVAILABLE:
        st.warning("Whisper not found. Install it if you want automatic transcription.")

# Run analysis
if st.button("Run Analysis"):
    # handle audio transcription if needed
    if input_mode == "Text":
        text = text_input
    else:
        if uploaded_audio is None:
            st.error("Please upload an audio file or switch to Text mode.")
            st.stop()
        if use_whisper:
            if not WHISPER_AVAILABLE:
                st.error("Whisper not installed.")
                st.stop()
            with st.spinner("Transcribing audio with Whisper..."):
                try:
                    text = whisper_transcribe_file(uploaded_audio, model_size=whisper_model_choice)
                    st.success("Transcription complete.")
                    st.text_area("Transcribed text (editable):", value=text, height=180)
                except Exception as e:
                    st.error(f"Whisper transcription failed: {e}. Please paste text manually.")
                    st.stop()
        else:
            st.error("Automatic transcription not available. Please paste text manually in Text mode.")
            st.stop()

    if not text or len(text.strip()) < 3:
        st.error("Please provide some text to analyze.")
        st.stop()

    if not TORCH_AVAILABLE or model is None or tokenizer is None:
        st.error("Transformers/Torch not available or model failed to load in this environment.")
        st.stop()

    with st.spinner("Analyzing text..."):
        # load model already done; call build_trajectory
        p = st.progress(0.0)
        def progress_cb(pct):
            try:
                p.progress(min(1.0, float(pct)))
            except Exception:
                pass

        traj = build_trajectory(text, tokenizer, model,
                                word_heatmap=compute_word_heatmap,
                                word_limit=int(max_word_count),
                                smooth_window=int(smoothing_window),
                                smooth_method=smoothing_method,
                                batch_size=int(batch_size),
                                progress_callback=progress_cb,
                                topk=top_k)
        p.empty()

    # unpack and safety align
    sentences = traj["sentences"]
    sent_scores = traj["sentence_scores"]
    dominants = traj["dominant"]
    word_tokens = traj["word_tokens"]
    word_scores = traj["word_scores"]
    valence = traj["valence"]
    arousal = traj["arousal"]
    dominance = traj.get("dominance", np.array([]))

    if getattr(sent_scores, "size", 0) and sent_scores.shape[0] != len(sentences):
        st.warning("Aligning sentences and score lengths.")
        m = min(sent_scores.shape[0], len(sentences))
        sent_scores = sent_scores[:m, :]
        sentences = sentences[:m]
        dominants = dominants[:m]
        valence = valence[:m]
        arousal = arousal[:m]

    # Build df
    df_sent = safe_insert_sentence_column(pd.DataFrame(sent_scores, columns=EMOTION_LABELS) if getattr(sent_scores, "size", 0) else pd.DataFrame(), sentences, dominants)

    # export JSON
    export_json = {
        "text": text,
        "sentences": [
            {"id": i, "text": sentences[i], "dominant": dominants[i] if i < len(dominants) else None,
             "scores": {EMOTION_LABELS[j]: float(sent_scores[i,j]) if getattr(sent_scores, "size", 0) else 0.0 for j in range(len(EMOTION_LABELS))}}
            for i in range(len(sentences))
        ]
    }
    if include_word_in_json:
        export_json["word_tokens"] = word_tokens
        export_json["word_scores"] = (word_scores.tolist() if getattr(word_scores, "size", 0) else [])

    keywords = extract_keywords_tfidf_like(text, top_k=25)
    cohesion = cohesion_and_stability(valence, dominants)

    # visuals
    try:
        sankey_fig, transition_matrix, matrix_fig = build_sankey_and_matrix(dominants)
    except Exception:
        sankey_fig = go.Figure(); sankey_fig.update_layout(title="Sankey failed")
        matrix_fig = go.Figure(); matrix_fig.update_layout(title="Matrix failed")
    gauge_fig, mean_val = build_gauge(valence)
    fig2d = build_2d_animated(sent_scores, EMOTION_LABELS) if (getattr(sent_scores, "size", 0) and animate_2d) else None
    fig3d = build_3d_animated(valence, arousal, dominants) if (animate_3d and getattr(valence,'size',0)) else None
    heatmap_fig = build_heatmap(word_tokens, word_scores, EMOTION_LABELS, max_words_display=200) if (compute_word_heatmap and len(word_tokens)) else None
    radar_fig = build_radar_profile(sent_scores, EMOTION_LABELS)
    rhythm_fig = build_emotion_rhythm(sent_scores)
    waterfall_fig = build_polarity_waterfall(sent_scores)

    # session store
    st.session_state["_last_analysis"] = {
        "text": text,
        "sentences": sentences,
        "sent_scores": sent_scores.tolist() if getattr(sent_scores, "size", 0) else [],
        "dominants": dominants,
        "word_tokens": word_tokens,
        "word_scores": word_scores.tolist() if getattr(word_scores, "size", 0) else [],
        "valence": valence.tolist() if getattr(valence, "size", 0) else [],
        "arousal": arousal.tolist() if getattr(arousal, "size", 0) else [],
        "dominance": dominance.tolist() if getattr(dominance, "size", 0) else [],
        "df": df_sent,
        "export_json": export_json,
        "keywords": keywords,
        "cohesion": cohesion,
        "sankey_fig": sankey_fig,
        "matrix_fig": matrix_fig,
        "gauge_fig": gauge_fig,
        "fig2d": fig2d,
        "fig3d": fig3d,
        "heatmap_fig": heatmap_fig,
        "radar_fig": radar_fig,
        "rhythm_fig": rhythm_fig,
        "waterfall_fig": waterfall_fig
    }
    st.rerun()

# ---------------------------
# Results display + exports
# ---------------------------
if "_last_analysis" in st.session_state:
    res = st.session_state["_last_analysis"]
    text = res["text"]
    sentences = res["sentences"]
    sent_scores = np.array(res["sent_scores"]) if isinstance(res["sent_scores"], list) else (np.array(res["sent_scores"]) if isinstance(res["sent_scores"], np.ndarray) else np.array([]))
    dominants = res["dominants"]
    word_tokens = res["word_tokens"]
    word_scores = np.array(res["word_scores"]) if isinstance(res["word_scores"], list) else np.array([])
    valence = np.array(res["valence"]) if isinstance(res["valence"], list) else np.array([])
    arousal = np.array(res["arousal"]) if isinstance(res["arousal"], list) else np.array([])
    dominance = np.array(res.get("dominance", []))
    df_sent = res["df"]
    export_json = res["export_json"]
    keywords = res["keywords"]
    cohesion = res["cohesion"]
    sankey_fig = res["sankey_fig"]
    matrix_fig = res["matrix_fig"]
    gauge_fig = res["gauge_fig"]
    fig2d = res["fig2d"]
    fig3d = res["fig3d"]
    heatmap_fig = res["heatmap_fig"]
    radar_fig = res["radar_fig"]
    rhythm_fig = res["rhythm_fig"]
    waterfall_fig = res["waterfall_fig"]

    st.header("Analysis Results — Visual Dashboard")

    tabs = st.tabs(["Overview", "Visuals", "Table & Highlights", "Exports"])
    with tabs[0]:
        st.subheader("Summary")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Sentences", len(sentences))
        c2.metric("Avg Valence", f"{float(np.nanmean(valence)) if getattr(valence, 'size', 0) else 0.0:.3f}")
        c3.metric("Stability", f"{cohesion.get('stability_score',0.0):.3f}")
        c4.metric("Top emotion", (Counter([d for d in dominants if d is not None]).most_common(1)[0][0] if any(d is not None for d in dominants) else "N/A"))
        st.markdown("**Explanation:** The dashboard shows sentence-by-sentence emotion probabilities (28 classes). Valence/arousal are heuristic projections used to visualize emotional flow. Stability measures how consistent the emotional arc is (lower variance + lower entropy -> higher stability).")

        st.subheader("Radar: average profile")
        st.plotly_chart(radar_fig, use_container_width=True)

        st.subheader("Emotion rhythm (stacked)")
        st.plotly_chart(rhythm_fig, use_container_width=True)

    with tabs[1]:
        leftc, midc, rightc = st.columns([1.6, 1.2, 1])
        with leftc:
            st.subheader("3D Trajectory")
            if fig3d:
                st.plotly_chart(fig3d, use_container_width=True)
            else:
                st.info("3D trajectory not available")
            st.subheader("Valence × Arousal scatter")
            va_fig = go.Figure()
            va_fig.add_trace(go.Scatter(x=valence if getattr(valence,'size',0) else [], y=arousal if getattr(arousal,'size',0) else [], mode='markers+lines', name='sentences', marker=dict(size=6)))
            for lab in EMOTION_LABELS:
                v,a = VAD_MAP.get(lab, (0.5,0.3,0.5))[0:2]
                va_fig.add_trace(go.Scatter(x=[v], y=[a], mode='markers+text', text=[lab], textposition='top center', marker=dict(size=10)))
            va_fig.update_layout(title="Valence × Arousal — sentences & centroids", height=420)
            st.plotly_chart(va_fig, use_container_width=True)

        with midc:
            st.subheader("2D Emotion curves")
            if fig2d:
                st.plotly_chart(fig2d, use_container_width=True)
            else:
                st.info("2D curves not available")
            st.subheader("Polarity Waterfall")
            st.plotly_chart(waterfall_fig, use_container_width=True)

        with rightc:
            st.subheader("Transitions & Metrics")
            st.plotly_chart(matrix_fig, use_container_width=True)
            st.plotly_chart(sankey_fig, use_container_width=True)
            st.metric("Average Valence", f"{float(np.nanmean(valence)) if getattr(valence,'size',0) else 0.0:.3f}")
            st.metric("Stability Score", f"{cohesion.get('stability_score', 0.0):.3f}")
            st.plotly_chart(gauge_fig, use_container_width=True)

    with tabs[2]:
        st.subheader("Sentence Highlights")
        if sentences:
            render_sentence_highlights(sentences, dominants, probs=sent_scores if getattr(sent_scores, "size", 0) else None, show_confidence=True, include_mini_bar=True)
        else:
            st.info("No sentences to display.")
        st.markdown("---")
        st.subheader("Sentence Table")
        if not df_sent.empty:
            def topk_str(row):
                arr = np.array([row[lab] for lab in EMOTION_LABELS], dtype=float)
                idx = np.argsort(arr)[::-1][:top_k]
                return ", ".join([f"{EMOTION_LABELS[i]}({arr[i]:.2f})" for i in idx])
            try:
                df_sent["top_k"] = df_sent.apply(topk_str, axis=1)
            except Exception:
                pass
            st.dataframe(df_sent, use_container_width=True)
        else:
            st.info("Sentence-level table is empty (text too short).")

    with tabs[3]:
        st.subheader("Exports & Sharing")
        st.download_button("Download JSON export", data=json.dumps(export_json, indent=2).encode("utf-8"), file_name="emotion_export.json", mime="application/json")
        try:
            csv_bytes = df_sent.to_csv(index=False).encode("utf-8")
        except Exception:
            csv_bytes = ("sentence,dominant\n" + "\n".join([f"\"{s}\",{d}" for s,d in zip(sentences, dominants)])).encode("utf-8")
        st.download_button("Download CSV (sentence-level)", data=csv_bytes, file_name="sentence_level_emotions.csv", mime="text/csv")
        html_report = create_html_report("Emotion Trajectory Report", text, {"3D": fig3d if fig3d else go.Figure(), "VA": va_fig}, {"Sentence table": df_sent}, {"cohesion": cohesion, "model": MODEL_NAME})
        st.download_button("Download HTML report", data=html_report.encode("utf-8"), file_name="emotion_report.html", mime="text/html")
        st.markdown("---")
        st.subheader("Top Keywords")
        if keywords:
            for term, score in keywords:
                st.write(f"- **{term}** — {score:.3f}")
        else:
            st.info("No keywords extracted.")

    st.markdown("---")
    st.info("Finished analysis. Tips: increase batch size for long documents, and disable word-level heatmap for speed.")

# End of file
