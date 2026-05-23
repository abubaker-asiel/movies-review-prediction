import streamlit as st
import pickle
import string
import nltk
import numpy as np

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Movie Review Sentiment",
    page_icon="🎬",
    layout="centered",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=DM+Sans:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: #0d0d0d;
    color: #f0ece2;
}

h1 {
    font-family: 'Playfair Display', serif !important;
    font-size: 2.8rem !important;
    letter-spacing: -1px;
    color: #f5c842 !important;
    margin-bottom: 0 !important;
}

.subtitle {
    color: #888;
    font-size: 0.95rem;
    margin-top: 0;
    margin-bottom: 2rem;
}

.stTextArea textarea {
    background: #1a1a1a !important;
    color: #f0ece2 !important;
    border: 1px solid #333 !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.95rem !important;
}

.stButton > button {
    background: #f5c842 !important;
    color: #0d0d0d !important;
    font-weight: 600 !important;
    font-family: 'DM Sans', sans-serif !important;
    border: none !important;
    border-radius: 6px !important;
    padding: 0.55rem 2rem !important;
    font-size: 0.95rem !important;
    transition: opacity 0.2s;
}

.stButton > button:hover {
    opacity: 0.85 !important;
}

.result-positive {
    background: #0f2b1a;
    border-left: 4px solid #3ecf6e;
    border-radius: 8px;
    padding: 1.2rem 1.4rem;
    margin-top: 1.5rem;
    color: #3ecf6e;
    font-size: 1.1rem;
    font-weight: 500;
}

.result-negative {
    background: #2b0f0f;
    border-left: 4px solid #e05252;
    border-radius: 8px;
    padding: 1.2rem 1.4rem;
    margin-top: 1.5rem;
    color: #e05252;
    font-size: 1.1rem;
    font-weight: 500;
}

.confidence-bar-wrap {
    background: #1a1a1a;
    border-radius: 4px;
    height: 8px;
    margin-top: 0.6rem;
    overflow: hidden;
}

.confidence-bar-fill {
    height: 100%;
    border-radius: 4px;
    transition: width 0.6s ease;
}

.conf-label {
    font-size: 0.8rem;
    color: #888;
    margin-top: 0.3rem;
}

.divider { border-top: 1px solid #222; margin: 1.5rem 0; }
</style>
""", unsafe_allow_html=True)


# ── NLTK data ──────────────────────────────────────────────────────────────────
@st.cache_resource
def load_nltk():
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    nltk.download("wordnet", quiet=True)

load_nltk()


# ── Load model & vectorizer ────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model = pickle.load(open("Review.pkl", "rb"))
    vectorizer = pickle.load(open("vectorizier.pkl", "rb"))
    return model, vectorizer

try:
    model, vectorizer = load_artifacts()
    model_loaded = True
except FileNotFoundError:
    model_loaded = False


# ── Preprocessing (mirrors notebook) ──────────────────────────────────────────
def preprocess(text: str) -> str:
    # lowercase + remove punctuation
    no_punct = "".join(c for c in text.lower() if c not in string.punctuation)
    # tokenize
    from nltk.tokenize import word_tokenize
    tokens = word_tokenize(no_punct)
    # lemmatize
    wn = nltk.WordNetLemmatizer()
    lemmas = [wn.lemmatize(w) for w in tokens]
    return " ".join(lemmas)


# ── UI ─────────────────────────────────────────────────────────────────────────
st.markdown("<h1>🎬 Review Sentiment</h1>", unsafe_allow_html=True)
st.markdown('<p class="subtitle">Paste any movie review — we\'ll tell you if it\'s positive or negative.</p>', unsafe_allow_html=True)

if not model_loaded:
    st.error("⚠️ Model files not found. Make sure `Review.pkl` and `vectorizier.pkl` are in the same directory as this app.")
    st.stop()

default_review = "This show was an amazing, fresh & innovative idea in the 70's when it first aired."
review = st.text_area("Your review", value=default_review, height=160, label_visibility="collapsed", placeholder="Write or paste a movie review here…")

col1, col2 = st.columns([1, 4])
with col1:
    predict_btn = st.button("Predict", use_container_width=True)

if predict_btn and review.strip():
    processed = preprocess(review)
    X = vectorizer.transform([processed])

    prediction = model.predict(X)[0]           # 0 = negative, 1 = positive
    proba = model.predict_proba(X)[0]           # [neg_prob, pos_prob]
    label = "positive" if prediction == 1 else "negative"
    confidence = proba[prediction] * 100

    bar_color = "#3ecf6e" if label == "positive" else "#e05252"
    icon = "✅" if label == "positive" else "❌"

    result_class = "result-positive" if label == "positive" else "result-negative"
    st.markdown(f"""
    <div class="{result_class}">
        {icon} &nbsp; This review is <strong>{label}</strong>
        <div class="confidence-bar-wrap">
            <div class="confidence-bar-fill" style="width:{confidence:.1f}%; background:{bar_color};"></div>
        </div>
        <div class="conf-label">Confidence: {confidence:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)

elif predict_btn:
    st.warning("Please enter a review first.")

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown('<p style="color:#444; font-size:0.8rem;">Built with MultinomialNB · TF-IDF · IMDB 50K dataset</p>', unsafe_allow_html=True)
