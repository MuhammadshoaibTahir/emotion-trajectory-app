# app.py
"""
Emotion Trajectory Studio — Single-file complete app
- 3D animated trajectory (Plotly)
- Readable sentence highlighting with mini-bars & expanders
- Word-level heatmap (optional)
- Transition Sankey + matrix
- Keyword extraction (no sklearn)
- Cohesion / stability metrics
- Export: JSON, CSV, HTML
- Optional Whisper transcription and zero-shot topic classification
- Batched inference for speed
- Robust safety for indexing, DataFrame insertion, and names
- Updated for modern Streamlit API (st.rerun(), width='stretch')
"""

# ---------------------------
# Imports & Configuration
# ---------------------------
import streamlit as st
st.set_page_config(layout="wide", page_title="Emotion Trajectory Studio", initial_sidebar_state="expanded")

import os
import io
import json
import math
import time
import datetime
from tempfile import NamedTemporaryFile
from typing import List, Dict, Tuple, Optional
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.io as pio

# NLTK tokenizers
import nltk
nltk.download("punkt", quiet=True)
from nltk.tokenize import sent_tokenize, word_tokenize

# Transformers / torch (lazy checked below)
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

# Whisper optional
try:
    import whisper
    WHISPER_AVAILABLE = True
except Exception:
    WHISPER_AVAILABLE = False

# SciPy optional (savgol and entropy)
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
MODEL_NAME = "bhadresh-savani/distilbert-base-uncased-emotion"
EMOTION_LABELS = ["anger", "fear", "joy", "love", "sadness", "surprise"]

# Heuristic mapping valence [-1..1], arousal [0..1]
VALENCE_AROUSAL = {
    "anger":    (-0.6, 0.9),
    "fear":     (-0.8, 0.85),
    "joy":      (0.8, 0.6),
    "love":     (0.9, 0.5),
    "sadness":  (-0.9, 0.3),
    "surprise": (0.0, 0.95)
}

# UI colors (soft pastel)
EMOTION_COLORS = {
    "anger":    "#ffd6d6",
    "fear":     "#e6e6ff",
    "joy":      "#fff8d6",
    "love":     "#ffe6f2",
    "sadness":  "#e8f0ff",
    "surprise": "#eaffeb",
    None:       "#f6f6f6"
}
EMOTION_BORDER = {
    "anger":    "#ff9b9b",
    "fear":     "#bdbdff",
    "joy":      "#ffeaa3",
    "love":     "#ffb3db",
    "sadness":  "#c7dbff",
    "surprise": "#bff0c2",
    None:       "#dddddd"
}
EMOTION_TEXT_COLOR = "#222222"

DEVICE = torch.device("cuda" if (TORCH_AVAILABLE and torch.cuda.is_available()) else "cpu") if TORCH_AVAILABLE else None

# ---------------------------
# Small helpers
# ---------------------------
def _escape_html(text: str) -> str:
    if text is None:
        return ""
    return (str(text)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace("\n", "<br/>"))

def ensure_length_alignment(list_a: List, list_b: List) -> Tuple[List, List]:
    m = min(len(list_a), len(list_b))
    return list_a[:m], list_b[:m]

def safe_insert_sentence_column(df: pd.DataFrame, sentences: List[str], dominants: List[Optional[str]]) -> pd.DataFrame:
    """Insert 'sentence' and 'dominant' safely into df, aligning lengths."""
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
# Keyword extraction (no sklearn)
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
# Sentence highlight helpers (readable)
# ---------------------------
def sentence_detail_html(index: int, sentence: str, probs: Optional[List[float]] = None, labels: Optional[List[str]] = None) -> str:
    s = _escape_html(sentence)
    prob_html = ""
    if probs is not None and labels is not None:
        rows = []
        for lab, p in zip(labels, probs):
            pct = int(max(0, min(100, round(p * 100))))
            color = EMOTION_BORDER.get(lab, "#aaaaaa")
            # row with label, bar, value
            rows.append(
                f"<div style='display:flex;align-items:center;margin-bottom:6px;'>"
                f"<div style='width:80px;font-size:13px;color:{EMOTION_TEXT_COLOR};'>{lab}</div>"
                f"<div style='flex:1;background:#f1f1f1;border-radius:6px;height:10px;margin-left:8px;'>"
                f"<div style='width:{pct}%;height:100%;background:{color};border-radius:6px;'></div></div>"
                f"<div style='width:40px;text-align:right;font-size:12px;color:#444;margin-left:8px;'>{p:.2f}</div>"
                f"</div>"
            )
        prob_html = "<div style='margin-top:8px;'>" + "\n".join(rows) + "</div>"
    html = f"""
    <div style='padding:10px;border-radius:8px;border:1px solid #eee;background:#fff;max-width:680px;'>
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
# Model loaders
# ---------------------------
if TORCH_AVAILABLE:
    @st.cache_resource(show_spinner=False)
    def load_emotion_model(model_name: str = MODEL_NAME):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        model.to(DEVICE)
        model.eval()
        return tokenizer, model

    @st.cache_resource(show_spinner=False)
    def load_zero_shot_model(model_name: str = "facebook/bart-large-mnli"):
        try:
            z = pipeline("zero-shot-classification", model=model_name, device=0 if torch.cuda.is_available() else -1)
            return z
        except Exception:
            return None
else:
    def load_emotion_model(model_name: str = MODEL_NAME):
        raise RuntimeError("Torch/Transformers not available in environment.")
    def load_zero_shot_model(model_name: str = "facebook/bart-large-mnli"):
        return None

# Whisper helper
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
def batched_infer(sentences: List[str], tokenizer, model, batch_size: int = 32) -> np.ndarray:
    if sentences is None or len(sentences) == 0:
        return np.zeros((0, len(EMOTION_LABELS)), dtype=float)
    all_probs = []
    with torch.no_grad():
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i+batch_size]
            enc = tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=256)
            enc = {k: v.to(DEVICE) for k, v in enc.items()}
            out = model(**enc)
            probs = torch.softmax(out.logits, dim=1).cpu().numpy()
            all_probs.append(probs)
    return np.vstack(all_probs)

def batched_infer_words(words: List[str], tokenizer, model, batch_size: int = 64, max_words: int = 1000) -> np.ndarray:
    words = words[:max_words]
    if not words:
        return np.zeros((0, len(EMOTION_LABELS)), dtype=float)
    return batched_infer(words, tokenizer, model, batch_size=batch_size)

def compute_valence_arousal_from_probs(probs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if probs is None or probs.size == 0:
        return np.array([]), np.array([])
    vs = []; as_ = []
    for p in probs:
        if p.sum() == 0:
            vs.append(0.0); as_.append(0.0)
        else:
            v = float(sum(p[i] * VALENCE_AROUSAL[EMOTION_LABELS[i]][0] for i in range(len(EMOTION_LABELS))))
            a = float(sum(p[i] * VALENCE_AROUSAL[EMOTION_LABELS[i]][1] for i in range(len(EMOTION_LABELS))))
            vs.append(v); as_.append(a)
    return np.array(vs), np.array(as_)

def build_trajectory(text: str, tokenizer, model,
                     word_heatmap: bool = True,
                     word_limit: int = 300,
                     smooth_window: int = 1,
                     smooth_method: str = "mean",
                     batch_size: int = 32,
                     progress_callback=None) -> Dict:
    sentences = sent_tokenize(text)
    n = len(sentences)
    if n == 0:
        sentence_scores = np.zeros((0, len(EMOTION_LABELS)))
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
            sentence_scores = np.vstack(parts) if parts else np.zeros((0, len(EMOTION_LABELS)))
    dominants = [EMOTION_LABELS[int(np.argmax(v))] if v.sum() > 0 else None for v in sentence_scores]

    word_tokens = []
    word_scores = np.zeros((0, len(EMOTION_LABELS)))
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
            word_scores = np.zeros((0, len(EMOTION_LABELS)))

    if sentence_scores.size == 0:
        smoothed = sentence_scores.copy()
    else:
        if smooth_window and smooth_window > 1:
            cols = [smooth_series(sentence_scores[:, i], window=smooth_window, method=smooth_method) for i in range(sentence_scores.shape[1])]
            smoothed = np.vstack(cols).T
        else:
            smoothed = sentence_scores.copy()

    valence, arousal = compute_valence_arousal_from_probs(smoothed)
    return {
        "sentences": sentences,
        "sentence_scores": smoothed,
        "dominant": dominants,
        "word_tokens": word_tokens,
        "word_scores": word_scores,
        "valence": valence,
        "arousal": arousal
    }

# ---------------------------
# Visualization builders
# ---------------------------
def build_2d_animated(scores: np.ndarray, labels: List[str], frame_duration:int=250) -> go.Figure:
    n_sent = scores.shape[0] if scores.size else 0
    n_em = scores.shape[1] if scores.size else len(labels)
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
        y = scores[:,e] if scores.size else []
        fig.add_trace(go.Scatter(x=list(range(n_sent)), y=y, mode='lines', name=labels[e], visible=True))
    return fig

def build_3d_animated(valence: np.ndarray, arousal: np.ndarray, dominants: List[Optional[str]], frame_duration:int=250) -> go.Figure:
    n = len(valence)
    if n == 0:
        fig = go.Figure(); fig.update_layout(title="3D Trajectory (no data)"); return fig
    x = list(valence); y = list(arousal); z = list(range(n))
    cmap = {"anger":"#ff6b6b","fear":"#ffd166","joy":"#90ee90","love":"#ff9bd6","sadness":"#9fb3ff","surprise":"#fff49c"}
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
    mapped = (mean_v + 1) * 50
    fig = go.Figure(go.Indicator(mode="gauge+number+delta", value=mapped, title={"text":"Average Valence (0..100)"}, gauge={"axis":{"range":[0,100]}}))
    fig.update_layout(height=260)
    return fig, mean_v

# ---------------------------
# UI: Sidebar & Inputs
# ---------------------------
st.title("🎛️ Emotion Trajectory Studio — Complete Edition")

with st.sidebar:
    st.header("Settings & Extras")
    st.write(f"Model: `{MODEL_NAME}`")
    smoothing_method = st.selectbox("Smoothing method", options=(["mean","savgol"] if SG_AVAILABLE else ["mean"]), index=0)
    smoothing_window = st.slider("Smoothing window (sentences)", min_value=1, max_value=11, value=3)
    compute_word_heatmap = st.checkbox("Compute word-level heatmap (slower)", value=True)
    max_word_count = st.number_input("Max words for heatmap", min_value=20, max_value=3000, value=300, step=20)
    batch_size = st.number_input("Inference batch size", min_value=8, max_value=256, value=32, step=8)
    animate_2d = st.checkbox("Animated 2D curves", value=True)
    animate_3d = st.checkbox("Animated 3D trajectory", value=True)
    include_word_in_json = st.checkbox("Include word-level scores in JSON export", value=False)
    top_k = st.slider("Top-K emotions per sentence (table)", min_value=1, max_value=len(EMOTION_LABELS), value=2)
    st.markdown("---")
    st.subheader("Optional extras")
    use_zero_shot = st.checkbox("Enable zero-shot topic classification (heavy)", value=False)
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
col_left, col_right = st.columns([1, 2])
with col_left:
    st.header("Input")
    input_mode = st.radio("Input type:", ["Text", "Upload Audio"])
    text_input = ""
    uploaded_audio = None
    if input_mode == "Text":
        default_text = st.session_state.get("sample_text", "")
        text_input = st.text_area("Paste or write your text here:", value=default_text, height=260)
    else:
        uploaded_audio = st.file_uploader("Upload audio file (wav/mp3/m4a/ogg)", type=["wav","mp3","m4a","ogg"])
        whisper_model_choice = st.selectbox("Whisper model size", options=["tiny", "base", "small", "medium", "large"], index=2)

with col_right:
    st.header("Quick Notes")
    st.info("This app predicts 6 emotions per sentence and builds trajectories in valence/arousal space.")
    st.write("For long documents, increase batch size and smoothing window for nicer curves.")
    if use_whisper and not WHISPER_AVAILABLE:
        st.warning("Whisper not installed. Disable Whisper or install it to enable transcription.")

# ---------------------------
# Action: Run Analysis
# ---------------------------
if st.button("Run Analysis"):
    # Prepare text (handle audio)
    if input_mode == "Text":
        text = text_input
    else:
        if uploaded_audio is None:
            st.error("Please upload audio or switch to Text mode.")
            st.stop()
        if use_whisper:
            if not WHISPER_AVAILABLE:
                st.error("Whisper not installed in environment.")
                st.stop()
            with st.spinner("Transcribing audio with Whisper..."):
                try:
                    text = whisper_transcribe_file(uploaded_audio, model_size=whisper_model_choice)
                    st.success("Transcription complete.")
                    st.text_area("Transcribed text (editable):", value=text, height=200)
                except Exception as e:
                    st.error(f"Whisper transcription failed: {e}")
                    st.stop()
        else:
            st.error("Whisper disabled. Please transcribe audio externally or switch to Text input.")
            st.stop()

    if not text or len(text.strip()) < 3:
        st.error("Please provide at least a few words of text.")
        st.stop()

    # Ensure torch available
    if not TORCH_AVAILABLE:
        st.error("Torch/Transformers are not installed. Install them to run analysis.")
        st.stop()
    # Load model
    with st.spinner("Loading tokenizer and model..."):
        tokenizer, model = load_emotion_model()

    # Optional zero-shot pipeline
    zsp = None
    if use_zero_shot:
        with st.spinner("Loading zero-shot pipeline..."):
            zsp = load_zero_shot_model()
            if zsp is None:
                st.warning("Zero-shot pipeline failed to load. Continuing without it.")

    # Progress bar
    p = st.progress(0.0)
    def progress_cb(pct):
        try:
            p.progress(min(1.0, float(pct)))
        except Exception:
            pass

    # Build trajectory
    with st.spinner("Analyzing text..."):
        traj = build_trajectory(text, tokenizer, model,
                                word_heatmap=compute_word_heatmap,
                                word_limit=int(max_word_count),
                                smooth_window=int(smoothing_window),
                                smooth_method=smoothing_method,
                                batch_size=int(batch_size),
                                progress_callback=progress_cb)
    p.empty()

    # Unpack
    sentences = traj["sentences"]
    sent_scores = traj["sentence_scores"]
    dominants = traj["dominant"]
    word_tokens = traj["word_tokens"]
    word_scores = traj["word_scores"]
    valence = traj["valence"]
    arousal = traj["arousal"]

    # Align lengths (safety)
    if getattr(sent_scores, "size", 0) and sent_scores.shape[0] != len(sentences):
        st.warning("Aligning sentences and score lengths.")
        m = min(sent_scores.shape[0], len(sentences))
        sent_scores = sent_scores[:m, :]
        sentences = sentences[:m]
        dominants = dominants[:m]
        valence = valence[:m]
        arousal = arousal[:m]

    # Build DataFrame safely
    df_sent = safe_insert_sentence_column(pd.DataFrame(sent_scores, columns=EMOTION_LABELS) if getattr(sent_scores, "size", 0) else pd.DataFrame(), sentences, dominants)

    # Exports
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

    # Keywords & cohesion
    keywords = extract_keywords_tfidf_like(text, top_k=25)
    cohesion = cohesion_and_stability(valence, dominants)

    # Visuals
    try:
        sankey_fig, transition_matrix, matrix_fig = build_sankey_and_matrix(dominants)
    except Exception:
        sankey_fig = go.Figure(); sankey_fig.update_layout(title="Sankey failed")
        matrix_fig = go.Figure(); matrix_fig.update_layout(title="Matrix failed")
    gauge_fig, mean_val = build_gauge(valence)
    fig2d = build_2d_animated(sent_scores, EMOTION_LABELS) if (getattr(sent_scores, "size", 0) and animate_2d) else None
    fig3d = build_3d_animated(valence, arousal, dominants) if animate_3d else None
    heatmap_fig = build_heatmap(word_tokens, word_scores, EMOTION_LABELS, max_words_display=200) if (compute_word_heatmap and len(word_tokens)) else None

    # Zero-shot classification
    z_result = None
    if zsp is not None:
        try:
            candidate_labels = ["personal", "story", "news", "politics", "technology", "health", "science", "business", "sports", "education"]
            z_result = zsp(text, candidate_labels)
        except Exception as e:
            st.warning(f"Zero-shot failed: {e}")

    # Save to session and rerun to show results area
    st.session_state["_last_analysis"] = {
        "text": text,
        "sentences": sentences,
        "sent_scores": sent_scores.tolist() if getattr(sent_scores, "size", 0) else [],
        "dominants": dominants,
        "word_tokens": word_tokens,
        "word_scores": word_scores.tolist() if getattr(word_scores, "size", 0) else [],
        "valence": valence.tolist() if getattr(valence, "size", 0) else [],
        "arousal": arousal.tolist() if getattr(arousal, "size", 0) else [],
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
        "zero_shot": z_result
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
    z_result = res.get("zero_shot", None)

    st.header("Analysis Results")

    colA, colB, colC = st.columns([2,2,1])
    with colA:
        st.subheader("3D Trajectory — Valence × Arousal × Sentence")
        if fig3d:
            st.plotly_chart(fig3d, width="stretch")
        else:
            st.info("3D trajectory not generated.")

        st.subheader("Valence × Arousal Scatter")
        va_fig = go.Figure()
        va_fig.add_trace(go.Scatter(x=valence if getattr(valence, "size", 0) else [], y=arousal if getattr(arousal, "size", 0) else [], mode='markers+lines', name='sentences', marker=dict(size=6)))
        for lab in EMOTION_LABELS:
            v,a = VALENCE_AROUSAL[lab]
            va_fig.add_trace(go.Scatter(x=[v], y=[a], mode='markers+text', text=[lab], textposition='top center', marker=dict(size=12)))
        va_fig.update_layout(title="Valence × Arousal — sentences & centroids", height=480)
        st.plotly_chart(va_fig, width="stretch")

        st.subheader("Animated Emotion Curves (2D)")
        if fig2d:
            st.plotly_chart(fig2d, width="stretch")
        else:
            st.info("2D curves not generated or not enough data.")

    with colB:
        st.subheader("Word-level Emotion Heatmap")
        if heatmap_fig:
            st.plotly_chart(heatmap_fig, width="stretch")
        else:
            st.info("Word-level heatmap not available.")

        st.subheader("Transition Visualizations")
        st.plotly_chart(matrix_fig, width="stretch")
        st.plotly_chart(sankey_fig, width="stretch")

    with colC:
        st.subheader("Summary & Metrics")
        st.metric("Average Valence", f"{float(np.nanmean(valence)) if getattr(valence, 'size', 0) else 0.0:.3f}")
        st.metric("Stability Score", f"{cohesion.get('stability_score', 0.0):.3f}")
        st.write("Valence variance:", f"{cohesion.get('valence_variance', 0.0):.4f}")
        st.write("Dominant entropy:", f"{cohesion.get('dominant_entropy', 0.0):.4f}")
        st.plotly_chart(gauge_fig, width="stretch")

    st.markdown("---")
    st.subheader("Sentence Highlights (Readable Colors & Details)")
    if sentences:
        render_sentence_highlights(sentences, dominants, probs=sent_scores if getattr(sent_scores, "size", 0) else None, show_confidence=True, include_mini_bar=True)
    else:
        st.info("No sentences to display.")

    st.markdown("---")
    st.subheader("Top Keywords (TF×IDF-like)")
    if keywords:
        for term, score in keywords:
            st.write(f"- **{term}** — {score:.3f}")
    else:
        st.info("No keywords extracted.")

    st.markdown("---")
    st.subheader("Sentence-level table")
    if not df_sent.empty:
        def topk_str(row):
            arr = np.array([row[lab] for lab in EMOTION_LABELS], dtype=float)
            idx = np.argsort(arr)[::-1][:top_k]
            return ", ".join([f"{EMOTION_LABELS[i]}({arr[i]:.2f})" for i in idx])
        try:
            df_sent["top_k"] = df_sent.apply(topk_str, axis=1)
        except Exception:
            pass
        st.dataframe(df_sent, width="stretch")
    else:
        st.info("Sentence table is empty (text too short).")

    if z_result:
        st.markdown("---")
        st.subheader("Zero-shot Topic Classification")
        st.write(z_result)

    st.markdown("---")
    st.header("Export / Download")
    st.download_button("Download JSON export", data=json.dumps(export_json, indent=2).encode("utf-8"), file_name="emotion_export.json", mime="application/json")
    try:
        csv_bytes = df_sent.to_csv(index=False).encode("utf-8")
    except Exception:
        csv_bytes = ("sentence,dominant\n" + "\n".join([f"\"{s}\",{d}" for s,d in zip(sentences, dominants)])).encode("utf-8")
    st.download_button("Download CSV (sentence-level)", data=csv_bytes, file_name="sentence_level_emotions.csv", mime="text/csv")

    figs = {}
    if fig3d: figs["3D Trajectory"] = fig3d
    if fig2d: figs["Emotion curves"] = fig2d
    figs["Valence-Arousal"] = va_fig
    if heatmap_fig: figs["Word heatmap"] = heatmap_fig
    figs["Transition Sankey"] = sankey_fig
    figs["Transition Matrix"] = matrix_fig
    tables = {"Sentence table": df_sent}
    metadata = {"cohesion": cohesion, "model": MODEL_NAME, "generated_at": datetime.datetime.utcnow().isoformat()}
    html_report = create_html_report("Emotion Trajectory Report", text, figs, tables, metadata)
    st.download_button("Download full HTML report", data=html_report.encode("utf-8"), file_name="emotion_report.html", mime="text/html")

    st.markdown("---")
    st.info("Finished analysis. Tips: increase batch size for long documents and disable word-level heatmap for speed.")

# End of app.py
