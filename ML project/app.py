"""
Real-Time Sentiment Analysis System with Web Scraping
Main Streamlit Application — PRD v1.0
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
import os
import sys

# ── Page config (must be first Streamlit call) ─────────────────────────────
st.set_page_config(
    page_title="SentimentAI — Real-Time Analysis",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* Base */
html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }
.stApp { background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%); }

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #12122b 0%, #1a1a3e 100%);
    border-right: 1px solid rgba(108,92,231,0.2);
}
[data-testid="stSidebar"] * { color: #c8c8e8 !important; }

/* Metric cards */
div[data-testid="metric-container"] {
    background: rgba(108,92,231,0.08);
    border: 1px solid rgba(108,92,231,0.25);
    border-radius: 14px;
    padding: 16px !important;
    backdrop-filter: blur(8px);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
div[data-testid="metric-container"]:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 32px rgba(108,92,231,0.25);
}
div[data-testid="metric-container"] label { color: #a8a8c8 !important; font-size: 0.78rem !important; }
div[data-testid="metric-container"] [data-testid="stMetricValue"] { font-size: 1.9rem !important; font-weight: 700 !important; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.04);
    border-radius: 12px;
    padding: 4px;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 10px !important;
    color: #a0a0c0 !important;
    font-weight: 500;
    transition: all 0.2s;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #6c5ce7, #a855f7) !important;
    color: white !important;
    font-weight: 600 !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #6c5ce7 0%, #a855f7 100%);
    color: white !important;
    border: none;
    border-radius: 10px;
    font-weight: 600;
    font-size: 0.95rem;
    padding: 0.55rem 1.5rem;
    transition: all 0.25s ease;
    box-shadow: 0 4px 15px rgba(108,92,231,0.3);
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(108,92,231,0.5);
    filter: brightness(1.1);
}

/* Text area & inputs */
.stTextArea textarea, .stTextInput input, .stSelectbox select {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(108,92,231,0.3) !important;
    border-radius: 10px !important;
    color: #e0e0f0 !important;
}
.stTextArea textarea:focus, .stTextInput input:focus {
    border-color: #6c5ce7 !important;
    box-shadow: 0 0 0 2px rgba(108,92,231,0.2) !important;
}

/* DataFrames */
.stDataFrame { border-radius: 12px; overflow: hidden; }

/* Progress */
.stProgress > div > div { background: linear-gradient(90deg, #6c5ce7, #a855f7) !important; border-radius: 4px; }

/* Alerts */
.stSuccess { background: rgba(46,204,113,0.12) !important; border: 1px solid rgba(46,204,113,0.3) !important; border-radius: 10px !important; }
.stError   { background: rgba(231,76,60,0.12) !important;  border: 1px solid rgba(231,76,60,0.3) !important;  border-radius: 10px !important; }
.stInfo    { background: rgba(108,92,231,0.12) !important; border: 1px solid rgba(108,92,231,0.3) !important; border-radius: 10px !important; }
.stWarning { background: rgba(243,156,18,0.12) !important; border: 1px solid rgba(243,156,18,0.3) !important; border-radius: 10px !important; }

/* Section headers */
.section-header {
    font-size: 1.5rem;
    font-weight: 700;
    color: #e0e0f8;
    margin-bottom: 0.5rem;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid rgba(108,92,231,0.4);
}
.sub-header {
    font-size: 1rem;
    color: #9090b8;
    margin-bottom: 1.5rem;
}

/* Sentiment badge */
.badge-positive { background:#2ecc7122; border:1px solid #2ecc71; color:#2ecc71; border-radius:20px; padding:4px 14px; font-weight:700; display:inline-block; }
.badge-negative { background:#e74c3c22; border:1px solid #e74c3c; color:#e74c3c; border-radius:20px; padding:4px 14px; font-weight:700; display:inline-block; }
.badge-neutral  { background:#f39c1222; border:1px solid #f39c12; color:#f39c12; border-radius:20px; padding:4px 14px; font-weight:700; display:inline-block; }

/* Pipeline flow */
.pipeline-step {
    background: rgba(108,92,231,0.1);
    border: 1px solid rgba(108,92,231,0.3);
    border-radius: 10px;
    padding: 12px 18px;
    margin: 6px 0;
    color: #c8c8e8;
    font-size: 0.9rem;
    display: flex;
    align-items: center;
    gap: 10px;
}

/* Hero */
.hero-title {
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(135deg, #a78bfa, #818cf8, #38bdf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.2;
    margin-bottom: 0.5rem;
}
.hero-sub {
    font-size: 1.1rem;
    color: #8888aa;
    margin-bottom: 2rem;
}

/* Card */
.glass-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 16px;
    padding: 24px;
    backdrop-filter: blur(12px);
    margin-bottom: 1rem;
}

/* Spinner override */
.stSpinner > div { border-top-color: #6c5ce7 !important; }
</style>
""", unsafe_allow_html=True)

# ── Imports (after page config) ─────────────────────────────────────────────
from preprocessor import preprocess_text, preprocess_dataframe, get_preprocessing_steps
from model import train_model, predict_single, compare_models, get_top_features, MODELS
from visualizer import (
    plot_sentiment_donut, plot_sentiment_bar, plot_confusion_matrix,
    plot_metrics_radar, plot_model_comparison, plot_top_features,
    plot_probability_gauge, plot_sentiment_timeline, SENTIMENT_COLORS,
)
from data_generator import generate_sample_dataset

# ═══════════════════════════════════════════════════════════════════════════
# SESSION STATE DEFAULTS
# ═══════════════════════════════════════════════════════════════════════════
def _init_state():
    defaults = {
        "raw_df":        None,
        "processed_df":  None,
        "train_result":  None,
        "pipeline":      None,
        "comparison_df": None,
        "data_source":   None,
        "scrape_log":    [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:12px 0 8px 0;'>
        <div style='font-size:2.5rem;'>🧠</div>
        <div style='color:#a78bfa; font-size:1.1rem; font-weight:700;'>SentimentAI</div>
        <div style='color:#666688; font-size:0.75rem; margin-top:2px;'>Real-Time Analysis System</div>
    </div>
    <hr style='border-color:rgba(108,92,231,0.2); margin:12px 0;'>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        ["🏠 Overview", "📥 Data Collection", "⚙️ Preprocessing",
         "🤖 Model Training", "🔍 Live Prediction", "📊 Visualizations"],
        label_visibility="collapsed",
    )

    st.markdown("<hr style='border-color:rgba(108,92,231,0.15);'>", unsafe_allow_html=True)

    # Status indicators
    st.markdown("**Pipeline Status**")
    def _status(label, cond):
        icon = "✅" if cond else "⭕"
        st.markdown(f"<div style='font-size:0.82rem; color:#9090b8; margin:2px 0;'>{icon} {label}</div>", unsafe_allow_html=True)

    _status("Data loaded",      st.session_state.raw_df is not None)
    _status("Preprocessed",     st.session_state.processed_df is not None)
    _status("Model trained",    st.session_state.pipeline is not None)

    if st.session_state.train_result:
        m = st.session_state.train_result["metrics"]
        st.markdown(f"""
        <div style='margin-top:12px; background:rgba(108,92,231,0.1); border-radius:10px; padding:10px;'>
        <div style='color:#a78bfa; font-size:0.78rem; font-weight:600;'>TRAINED MODEL</div>
        <div style='color:#e0e0f0; font-size:0.85rem;'>{m["model_name"].split("(")[0].strip()}</div>
        <div style='color:#2ecc71; font-size:1.3rem; font-weight:700;'>{m["accuracy"]*100:.1f}%</div>
        <div style='color:#9090b8; font-size:0.72rem;'>accuracy on test set</div>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.markdown("""
    <div class='hero-title'>Real-Time Sentiment Analysis</div>
    <div class='hero-sub'>
        Web Scraping · NLP · TF-IDF · Logistic Regression · Interactive Visualizations
    </div>
    """, unsafe_allow_html=True)

    # Quick stats
    col1, col2, col3, col4 = st.columns(4)
    n_records = len(st.session_state.raw_df) if st.session_state.raw_df is not None else 0
    n_clean   = len(st.session_state.processed_df) if st.session_state.processed_df is not None else 0
    accuracy  = (st.session_state.train_result["metrics"]["accuracy"] * 100) if st.session_state.train_result else 0
    n_models  = len(MODELS)

    col1.metric("📄 Records Loaded",    f"{n_records:,}")
    col2.metric("✨ After Preprocessing", f"{n_clean:,}")
    col3.metric("🎯 Model Accuracy",    f"{accuracy:.1f}%" if accuracy else "—")
    col4.metric("🤖 ML Models",        n_models)

    st.markdown("---")

    # Pipeline diagram
    col_a, col_b = st.columns([1.2, 1])

    with col_a:
        st.markdown("<div class='section-header'>🔄 System Pipeline</div>", unsafe_allow_html=True)
        steps = [
            ("🌐", "Web Scraping",        "BeautifulSoup · Requests"),
            ("🧹", "Data Cleaning",       "HTML · Duplicates · Noise removal"),
            ("💬", "NLP Preprocessing",   "Tokenize · Stop-words · Lemmatize"),
            ("🔢", "TF-IDF Extraction",   "Text → Numerical feature vectors"),
            ("🤖", "Model Training",      "Logistic Regression (primary)"),
            ("📊", "Evaluation",          "Accuracy · Precision · Recall · F1"),
            ("📈", "Visualization",       "Charts · Confusion Matrix · Trends"),
        ]
        for icon, title, desc in steps:
            st.markdown(f"""
            <div class='pipeline-step'>
                <span style='font-size:1.3rem;'>{icon}</span>
                <div>
                    <div style='font-weight:600; color:#c8c8e8;'>{title}</div>
                    <div style='font-size:0.77rem; color:#7070a0;'>{desc}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with col_b:
        st.markdown("<div class='section-header'>🎯 Sentiment Classes</div>", unsafe_allow_html=True)
        examples = [
            ("Positive", "#2ecc71", "😊", '"The product is absolutely amazing!"'),
            ("Negative", "#e74c3c", "😞", '"Broke after two days, terrible quality."'),
            ("Neutral",  "#f39c12", "😐", '"The product was released yesterday."'),
        ]
        for senti, color, emoji, example in examples:
            st.markdown(f"""
            <div style='background:rgba(255,255,255,0.04); border:1px solid {color}44;
                        border-left: 3px solid {color}; border-radius:10px;
                        padding:14px; margin-bottom:10px;'>
                <div style='color:{color}; font-weight:700; font-size:1rem;'>{emoji} {senti}</div>
                <div style='color:#9090b8; font-size:0.82rem; margin-top:4px; font-style:italic;'>{example}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div class='section-header' style='margin-top:1rem;'>📐 Algorithm</div>", unsafe_allow_html=True)
        st.markdown("""
        <div style='background:rgba(108,92,231,0.08); border-radius:10px; padding:14px;
                    border:1px solid rgba(108,92,231,0.2);'>
            <div style='color:#a78bfa; font-weight:700;'>Primary: Logistic Regression</div>
            <div style='color:#8888aa; font-size:0.8rem; margin-top:4px;'>
                ✅ Works well with TF-IDF sparse features<br>
                ✅ Fast training, interpretable<br>
                ✅ Supports multiclass classification<br>
                ✅ Strong baseline for text tasks
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.info("👈 **Use the sidebar** to navigate through each pipeline stage. Start with **📥 Data Collection**.")


# ═══════════════════════════════════════════════════════════════════════════
# PAGE: DATA COLLECTION
# ═══════════════════════════════════════════════════════════════════════════
elif page == "📥 Data Collection":
    st.markdown("<div class='section-header'>📥 Data Collection</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Scrape reviews from public websites or load a built-in sample dataset.</div>", unsafe_allow_html=True)

    source_tab, upload_tab, manual_tab = st.tabs(["🌐 Web Scraper", "📂 Upload CSV", "✍️ Sample Dataset"])

    # ── Web Scraper tab ──────────────────────────────────────────────────
    with source_tab:
        col1, col2 = st.columns([1.5, 1])
        with col1:
            source_choice = st.selectbox(
                "Select Data Source",
                ["📚 Books to Scrape (Recommended)", "💬 Quotes to Scrape", "🌐 Custom URL"],
            )
            is_custom = "Custom URL" in source_choice
            if is_custom:
                custom_url = st.text_input(
                    "Enter URL to scrape",
                    placeholder="https://www.imdb.com/reviews  or  https://example.com/blog",
                )
                max_items = st.slider("Max text segments to extract", 50, 500, 200, step=50)
            else:
                custom_url = None
                max_items  = 200
                max_pages  = st.slider("Number of pages to scrape", 1, 10, 3)

        with col2:
            if is_custom:
                st.markdown("""
                <div class='glass-card'>
                <div style='color:#a78bfa; font-weight:600; margin-bottom:8px;'>🤖 Auto-Labelling (TextBlob)</div>
                <div style='color:#8888aa; font-size:0.82rem;'>
                Custom URL text has <b>no star ratings</b>, so each extracted
                paragraph is automatically labelled using <b>TextBlob NLP</b>
                polarity analysis:<br><br>
                <span style='color:#2ecc71;'>● Polarity &gt; 0.08 → <b>Positive</b></span><br>
                <span style='color:#e74c3c;'>● Polarity &lt; -0.08 → <b>Negative</b></span><br>
                <span style='color:#f39c12;'>● Otherwise → <b>Neutral</b></span><br><br>
                Labels are ready for model training immediately after scraping.
                You can review and adjust them in the preview table below.
                </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class='glass-card'>
                <div style='color:#a78bfa; font-weight:600; margin-bottom:8px;'>ℹ️ Scraping Info</div>
                <div style='color:#8888aa; font-size:0.82rem;'>
                • <b>Books to Scrape</b>: book titles + star ratings<br>
                • <b>Quotes to Scrape</b>: famous quotes + authors<br><br>
                Star rating → Sentiment label:<br>
                <span style='color:#2ecc71;'>★★★★☆/★★★★★ → Positive</span><br>
                <span style='color:#f39c12;'>★★★☆☆ → Neutral</span><br>
                <span style='color:#e74c3c;'>★☆☆☆☆/★★☆☆☆ → Negative</span>
                </div>
                </div>
                """, unsafe_allow_html=True)

        if st.button("🚀 Start Scraping", use_container_width=True):
            from scraper import (
                scrape_books_to_scrape, scrape_quotes_to_scrape,
                scrape_custom_url, assign_labels_from_rating, records_to_dataframe,
            )
            progress = st.progress(0)
            status   = st.empty()

            try:
                status.info("🕷️ Initializing web scraper…")
                progress.progress(10)
                time.sleep(0.3)

                if "Books" in source_choice:
                    status.info(f"📚 Scraping Books to Scrape (pages 1–{max_pages})…")
                    records = scrape_books_to_scrape(max_pages=max_pages)
                elif "Quotes" in source_choice:
                    status.info(f"💬 Scraping Quotes to Scrape (up to {max_pages} pages)…")
                    records = scrape_quotes_to_scrape(max_pages=max_pages)
                elif is_custom:
                    if not custom_url or not custom_url.strip():
                        st.error("⚠️ Please enter a URL before scraping.")
                        st.stop()
                    status.info(f"🌐 Scraping: {custom_url}")
                    records = scrape_custom_url(custom_url, max_items=max_items)
                    progress.progress(55)
                    status.info(f"🤖 Auto-labelling {len(records)} text segments with TextBlob…")
                else:
                    records = []

                progress.progress(65)
                status.info("🔖 Assigning sentiment labels…")
                records = assign_labels_from_rating(records)

                progress.progress(85)
                raw_df = records_to_dataframe(records)
                progress.progress(100)

                if raw_df.empty:
                    status.error("❌ No data scraped. The website may be unreachable or returned no usable text.")
                else:
                    st.session_state.raw_df       = raw_df
                    st.session_state.processed_df = None
                    st.session_state.pipeline     = None
                    st.session_state.train_result = None
                    st.session_state.data_source  = source_choice

                    labeled_count = raw_df["sentiment"].notna().sum()
                    if is_custom:
                        status.success(
                            f"✅ Scraped **{len(raw_df)}** segments — all auto-labelled "
                            f"with TextBlob (Pos/Neg/Neu). Ready for training!"
                        )
                    else:
                        status.success(f"✅ Scraped **{len(raw_df)}** records — **{labeled_count}** labelled.")

            except Exception as e:
                status.error(f"❌ Scraping failed: {e}")

    # ── Upload CSV tab ───────────────────────────────────────────────────
    with upload_tab:
        st.markdown("Upload a CSV with at least a **`text`** column. An optional **`sentiment`** column (Positive/Negative/Neutral) enables model training.")
        uploaded = st.file_uploader("Choose CSV file", type=["csv"])
        if uploaded:
            try:
                df = pd.read_csv(uploaded)
                if "text" not in df.columns:
                    st.error("CSV must have a `text` column.")
                else:
                    if "sentiment" not in df.columns:
                        df["sentiment"] = None
                    if "source" not in df.columns:
                        df["source"] = "uploaded"
                    st.session_state.raw_df      = df[["text","sentiment","source"]]
                    st.session_state.processed_df = None
                    st.session_state.pipeline     = None
                    st.session_state.train_result = None
                    st.session_state.data_source  = "CSV Upload"
                    st.success(f"✅ Loaded **{len(df)}** records from CSV.")
            except Exception as e:
                st.error(f"Error reading CSV: {e}")

    # ── Sample Dataset tab ───────────────────────────────────────────────
    with manual_tab:
        st.markdown("Use a **built-in balanced dataset** (product + movie reviews) to demo the full pipeline instantly.")
        n_per_class = st.slider("Records per sentiment class", 50, 500, 200, step=50)
        if st.button("⚡ Load Sample Dataset", use_container_width=True):
            with st.spinner("Generating balanced dataset…"):
                raw_df = generate_sample_dataset(n_per_class=n_per_class)
                raw_df["source"] = "built-in sample"
                st.session_state.raw_df      = raw_df
                st.session_state.processed_df = None
                st.session_state.pipeline     = None
                st.session_state.train_result = None
                st.session_state.data_source  = "Sample Dataset"
            st.success(f"✅ Loaded **{len(raw_df)}** sample records.")

    # ── Preview ─────────────────────────────────────────────────────────
    if st.session_state.raw_df is not None:
        df = st.session_state.raw_df
        st.markdown("---")

        unlabeled = df["sentiment"].isna().sum()
        label_method = "TextBlob auto-label" if "Custom" in (st.session_state.data_source or "") else "Star rating"

        st.markdown(
            f"**Data Preview** — {len(df):,} records from: *{st.session_state.data_source}* "
            f"| Labelling: *{label_method}*"
        )

        if unlabeled > 0:
            st.warning(f"⚠️ {unlabeled} records have no sentiment label and will be skipped during training.")

        vc = df["sentiment"].value_counts()
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("✅ Positive",  int(vc.get("Positive", 0)))
        col2.metric("❌ Negative",  int(vc.get("Negative", 0)))
        col3.metric("⚪ Neutral",   int(vc.get("Neutral",  0)))
        col4.metric("❓ Unlabelled", int(unlabeled))

        # Show label distribution bar if all labelled
        if unlabeled == 0 and len(vc) > 0:
            counts = vc.to_dict()
            st.plotly_chart(plot_sentiment_bar(counts), use_container_width=True)

        st.dataframe(
            df[["sentiment", "text", "source"]].head(30),
            use_container_width=True,
            height=380,
        )


# ═══════════════════════════════════════════════════════════════════════════
# PAGE: PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════════
elif page == "⚙️ Preprocessing":
    st.markdown("<div class='section-header'>⚙️ NLP Preprocessing</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Clean and normalize raw text through the full NLP pipeline.</div>", unsafe_allow_html=True)

    # Live demo
    st.markdown("### 🔬 Live Preprocessing Demo")
    demo_text = st.text_area(
        "Enter any text to see step-by-step preprocessing:",
        value='<p>This product is absolutely <b>AMAZING</b>!! 😍 I bought it for $29.99 and it\'s the BEST thing ever!! Visit https://example.com for more info.</p>',
        height=100,
    )

    if st.button("🔄 Run Preprocessing", use_container_width=True):
        from preprocessor import (
            remove_html_tags, remove_urls, remove_special_characters,
            normalize_whitespace, tokenize, remove_stopwords, lemmatize,
        )
        steps_data = [
            ("📝 Original Text",         demo_text),
            ("🧹 After HTML Removal",     remove_html_tags(demo_text)),
            ("🔗 After URL Removal",       remove_urls(remove_html_tags(demo_text)).lower()),
            ("⬇️ After Lowercase",         remove_urls(remove_html_tags(demo_text)).lower()),
            ("✂️ After Special Char Rm",  normalize_whitespace(remove_special_characters(remove_urls(remove_html_tags(demo_text)).lower()))),
            ("🔤 Tokenized",              str(tokenize(normalize_whitespace(remove_special_characters(remove_urls(remove_html_tags(demo_text)).lower()))))),
            ("🚫 Stop-words Removed",     str(remove_stopwords(tokenize(normalize_whitespace(remove_special_characters(remove_urls(remove_html_tags(demo_text)).lower())))))),
            ("🌱 After Lemmatization",    preprocess_text(demo_text)),
        ]
        for title, content in steps_data:
            with st.expander(title, expanded=True):
                st.code(content if content else "(empty)", language=None)

    st.markdown("---")

    # Pipeline steps overview
    st.markdown("### 📋 Preprocessing Pipeline Steps")
    steps = get_preprocessing_steps()
    cols = st.columns(2)
    for i, step in enumerate(steps):
        with cols[i % 2]:
            st.markdown(f"""
            <div class='pipeline-step'>
                <div style='background:rgba(108,92,231,0.3); border-radius:6px; padding:4px 10px;
                            font-weight:700; color:#a78bfa; font-size:0.8rem;'>{step["step"].split(".")[0]}</div>
                <div>
                    <div style='font-weight:600; color:#c8c8e8;'>{" ".join(step["step"].split(" ")[1:])}</div>
                    <div style='font-size:0.78rem; color:#7070a0;'>{step["description"]}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # Process the loaded dataset
    if st.session_state.raw_df is None:
        st.warning("⚠️ No data loaded. Go to **📥 Data Collection** first.")
    else:
        st.markdown("### 🚀 Process Loaded Dataset")
        raw_df = st.session_state.raw_df
        st.info(f"Ready to preprocess **{len(raw_df):,}** records.")

        if st.button("⚙️ Run Full Preprocessing Pipeline", use_container_width=True):
            progress = st.progress(0)
            status   = st.empty()

            with st.spinner("Preprocessing text…"):
                status.info("Step 1/4: Removing HTML, URLs, and noise…")
                progress.progress(25)
                time.sleep(0.3)

                status.info("Step 2/4: Tokenizing and removing stop words…")
                progress.progress(50)
                processed_df = preprocess_dataframe(raw_df.copy(), text_col="text")
                time.sleep(0.3)

                status.info("Step 3/4: Lemmatizing tokens…")
                progress.progress(75)
                time.sleep(0.3)

                status.info("Step 4/4: Removing duplicates and empty records…")
                progress.progress(100)
                time.sleep(0.2)

            st.session_state.processed_df = processed_df
            st.session_state.pipeline     = None
            st.session_state.train_result = None
            status.success(f"✅ Preprocessing complete! **{len(processed_df):,}** clean records ready.")

        if st.session_state.processed_df is not None:
            proc_df = st.session_state.processed_df
            col1, col2, col3 = st.columns(3)
            removed = len(raw_df) - len(proc_df)
            col1.metric("Before",  f"{len(raw_df):,}")
            col2.metric("After",   f"{len(proc_df):,}")
            col3.metric("Removed (dupes/empty)", f"{removed:,}")
            st.dataframe(
                proc_df[["text","cleaned_text","sentiment"]].head(20),
                use_container_width=True, height=350,
            )


# ═══════════════════════════════════════════════════════════════════════════
# PAGE: MODEL TRAINING
# ═══════════════════════════════════════════════════════════════════════════
elif page == "🤖 Model Training":
    st.markdown("<div class='section-header'>🤖 Model Training & Evaluation</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Train Logistic Regression (primary) and compare against Naive Bayes & SVM.</div>", unsafe_allow_html=True)

    if st.session_state.processed_df is None:
        st.warning("⚠️ No preprocessed data. Complete **⚙️ Preprocessing** first.")
        st.stop()

    proc_df = st.session_state.processed_df
    labeled = proc_df.dropna(subset=["sentiment"])

    if labeled.empty:
        st.error("❌ No labeled data found. The dataset needs a `sentiment` column.")
        st.stop()

    st.info(f"**{len(labeled):,}** labeled records available for training.")

    col1, col2, col3 = st.columns(3)
    with col1:
        model_choice = st.selectbox("Select Model", list(MODELS.keys()), index=0)
    with col2:
        test_size = st.slider("Test Set Size", 10, 40, 20, step=5, format="%d%%") / 100
    with col3:
        run_compare = st.checkbox("Compare all models", value=False)

    col_a, col_b = st.columns(2)
    with col_a:
        train_btn = st.button("🚀 Train Model", use_container_width=True)
    with col_b:
        if run_compare:
            compare_btn = st.button("⚖️ Compare All Models", use_container_width=True)
        else:
            compare_btn = False

    if train_btn:
        progress = st.progress(0)
        status   = st.empty()
        try:
            status.info("🔢 Building TF-IDF feature matrix…")
            progress.progress(30)
            time.sleep(0.4)

            status.info(f"🤖 Training {model_choice}…")
            progress.progress(60)

            result = train_model(
                labeled, text_col="cleaned_text", label_col="sentiment",
                model_name=model_choice, test_size=test_size,
            )
            progress.progress(100)

            st.session_state.train_result = result
            st.session_state.pipeline     = result["pipeline"]

            m = result["metrics"]
            status.success(f"✅ Training complete! Accuracy: **{m['accuracy']*100:.2f}%**")

        except Exception as e:
            status.error(f"Training failed: {e}")

    if compare_btn:
        with st.spinner("Training all models for comparison…"):
            comp_df = compare_models(
                labeled, text_col="cleaned_text", label_col="sentiment", test_size=test_size
            )
            st.session_state.comparison_df = comp_df
        st.success("✅ Comparison complete!")
        st.dataframe(comp_df, use_container_width=True)
        st.plotly_chart(plot_model_comparison(comp_df), use_container_width=True)

    # ── Results ─────────────────────────────────────────────────────────
    if st.session_state.train_result:
        result = st.session_state.train_result
        m      = result["metrics"]

        st.markdown("---")
        st.markdown("### 📊 Evaluation Results")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("🎯 Accuracy",   f"{m['accuracy']*100:.2f}%")
        col2.metric("📐 Precision",  f"{m['precision']*100:.2f}%")
        col3.metric("🔁 Recall",     f"{m['recall']*100:.2f}%")
        col4.metric("⚖️ F1-Score",   f"{m['f1_score']*100:.2f}%")

        st.markdown(f"""
        <div style='color:#7070a0; font-size:0.82rem; margin:.5rem 0;'>
        Model: <b style='color:#a78bfa;'>{m["model_name"]}</b> &nbsp;|&nbsp;
        Train size: <b style='color:#c8c8e8;'>{m["train_size"]:,}</b> &nbsp;|&nbsp;
        Test size: <b style='color:#c8c8e8;'>{m["test_size"]:,}</b>
        </div>
        """, unsafe_allow_html=True)

        tab1, tab2, tab3, tab4 = st.tabs(["🕸️ Radar", "🗺️ Confusion Matrix", "📋 Full Report", "🏆 Top Features"])

        with tab1:
            st.plotly_chart(plot_metrics_radar(m), use_container_width=True)

        with tab2:
            st.plotly_chart(
                plot_confusion_matrix(m["confusion_matrix"], m["labels"]),
                use_container_width=True,
            )
            st.markdown("""
            <div style='background:rgba(108,92,231,0.08); border-radius:10px; padding:12px 16px; border:1px solid rgba(108,92,231,0.2); font-size:0.82rem; color:#b0b0d0;'>
                <b>💡 How to read this Confusion Matrix:</b><br>
                • <b>Rows (Actual):</b> The true ground-truth sentiment labels in the test set.<br>
                • <b>Columns (Predicted):</b> What the model predicted for those reviews.<br>
                • <b>Diagonal cells (Green):</b> Correctly classified test reviews. Percentage in parentheses is <b>Class Recall</b> (e.g. what % of all actual reviews of this class were detected).<br>
                • <b>Off-diagonal cells:</b> Misclassifications (errors).<br>
                • <b>Overall Accuracy vs. Class Recall:</b> Overall accuracy is the total correct divided by all test samples. Individual class recall can be higher or lower depending on which category had misclassifications.
            </div>
            """, unsafe_allow_html=True)

        with tab3:
            report_df = pd.DataFrame(m["classification_report"]).T.round(3)
            st.dataframe(report_df, use_container_width=True)
            st.markdown("""
            <div style='background:rgba(108,92,231,0.08); border-radius:10px; padding:12px 16px; border:1px solid rgba(108,92,231,0.2); font-size:0.82rem; color:#b0b0d0;'>
                <b>📖 Metrics Definitions:</b><br>
                • <b>Precision:</b> When the model predicts this class, how often is it right? (True Positives / All Predicted as this class).<br>
                • <b>Recall:</b> Out of all actual reviews of this class, how many did the model find? (True Positives / All Actual in this class).<br>
                • <b>F1-Score:</b> Harmonic mean of Precision and Recall.<br>
                • <b>Support:</b> The number of true test samples evaluated for that class.
            </div>
            """, unsafe_allow_html=True)

        with tab4:
            try:
                feats = get_top_features(st.session_state.pipeline, n=20)
                if feats:
                    st.plotly_chart(plot_top_features(feats), use_container_width=True)
                else:
                    st.info("Top features only available for Logistic Regression.")
            except Exception as e:
                st.info(f"Feature visualization not available: {e}")


# ═══════════════════════════════════════════════════════════════════════════
# PAGE: LIVE PREDICTION
# ═══════════════════════════════════════════════════════════════════════════
elif page == "🔍 Live Prediction":
    st.markdown("<div class='section-header'>🔍 Live Sentiment Prediction</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Type or paste any text for real-time sentiment prediction from the trained model.</div>", unsafe_allow_html=True)

    if st.session_state.pipeline is None:
        st.warning("⚠️ No model trained yet. Complete **🤖 Model Training** first.")
        st.stop()

    if "live_input_text" not in st.session_state:
        st.session_state["live_input_text"] = ""

    # Quick examples
    st.markdown("**Quick Examples:**")
    ex_col1, ex_col2, ex_col3, ex_col4 = st.columns([1, 1, 1, 1])
    examples = {
        "😊 Positive": "This product is absolutely amazing! Best purchase I've made all year.",
        "😞 Negative": "Terrible quality. It broke after just two days. Total waste of money.",
        "😐 Neutral":  "The product was released in stores last Friday across all locations.",
    }
    
    if ex_col1.button("😊 Positive Example"):
        st.session_state["live_input_text"] = examples["😊 Positive"]
        st.rerun()
    if ex_col2.button("😞 Negative Example"):
        st.session_state["live_input_text"] = examples["😞 Negative"]
        st.rerun()
    if ex_col3.button("😐 Neutral Example"):
        st.session_state["live_input_text"] = examples["😐 Neutral"]
        st.rerun()
    if ex_col4.button("🧹 Clear Input"):
        st.session_state["live_input_text"] = ""
        st.rerun()

    user_text = st.text_area(
        "Input Text",
        value=st.session_state["live_input_text"],
        placeholder="Type your review or text here...",
        height=130,
        label_visibility="collapsed",
        key="live_text_area_input",
    )
    if user_text != st.session_state["live_input_text"]:
        st.session_state["live_input_text"] = user_text

    # Real-time / Live prediction while typing
    if user_text.strip():
        result = predict_single(
            st.session_state.pipeline,
            user_text,
            preprocessor_fn=preprocess_text,
        )

        pred  = result["prediction"]
        color = SENTIMENT_COLORS.get(pred, "#aaa")
        emoji_map = {"Positive": "😊", "Negative": "😞", "Neutral": "😐"}
        emoji = emoji_map.get(pred, "🤔")

        st.markdown(f"""
        <div style='background: rgba(255,255,255,0.04); border:1px solid {color}55;
                    border-left: 4px solid {color}; border-radius:14px;
                    padding:24px; margin:16px 0; text-align:center;'>
            <div style='font-size:3rem; margin-bottom:8px;'>{emoji}</div>
            <div style='color:#9090b8; font-size:0.85rem; margin-bottom:4px;'>Predicted Sentiment (Live)</div>
            <div style='color:{color}; font-size:2.2rem; font-weight:800;'>{pred}</div>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Preprocessed Text:**")
            st.code(result["processed_text"] or "(empty after preprocessing)", language=None)
        with col2:
            if result["probabilities"]:
                st.plotly_chart(
                    plot_probability_gauge(result["probabilities"], pred),
                    use_container_width=True,
                )
            else:
                st.info("Probability estimates not available for this model.")

    # ── Batch prediction ─────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📋 Batch Prediction")
    st.markdown("Paste multiple texts (one per line) for bulk analysis.")
    batch_input = st.text_area("Batch Input (one text per line)", height=120,
                               placeholder="The laptop is fantastic!\nTerrible product, waste of money.\nDelivery arrived on Tuesday.")

    if st.button("📊 Predict All", use_container_width=True):
        lines = [l.strip() for l in batch_input.strip().split("\n") if l.strip()]
        if not lines:
            st.warning("Enter at least one line.")
        else:
            with st.spinner(f"Predicting {len(lines)} texts…"):
                rows = []
                for line in lines:
                    r = predict_single(st.session_state.pipeline, line, preprocessor_fn=preprocess_text)
                    rows.append({
                        "text": line[:80] + "…" if len(line) > 80 else line,
                        "sentiment": r["prediction"],
                        "confidence": f"{max(r['probabilities'].values())*100:.1f}%" if r["probabilities"] else "N/A",
                    })
                batch_df = pd.DataFrame(rows)

            # Color the sentiment column
            def _color(v):
                c = SENTIMENT_COLORS.get(v, "#aaa")
                return f"color: {c}; font-weight: 600;"

            st.dataframe(
                batch_df.style.applymap(_color, subset=["sentiment"]),
                use_container_width=True,
            )

            counts = batch_df["sentiment"].value_counts().to_dict()
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(plot_sentiment_donut(counts), use_container_width=True)
            with col2:
                st.plotly_chart(plot_sentiment_bar(counts), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# PAGE: VISUALIZATIONS
# ═══════════════════════════════════════════════════════════════════════════
elif page == "📊 Visualizations":
    st.markdown("<div class='section-header'>📊 Sentiment Visualizations</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Explore the full visual analysis of sentiment results.</div>", unsafe_allow_html=True)

    if st.session_state.processed_df is None:
        st.warning("⚠️ No data available. Complete preprocessing first.")
        st.stop()

    df = st.session_state.processed_df
    labeled = df.dropna(subset=["sentiment"])

    if labeled.empty:
        st.error("No labeled data for visualization.")
        st.stop()

    # Distribution charts
    st.markdown("### 🥧 Sentiment Distribution")
    counts = labeled["sentiment"].value_counts().to_dict()
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(plot_sentiment_donut(counts), use_container_width=True)
    with col2:
        st.plotly_chart(plot_sentiment_bar(counts), use_container_width=True)

    st.markdown("---")

    # Trend
    st.markdown("### 📈 Sentiment Trend")
    st.plotly_chart(plot_sentiment_timeline(labeled), use_container_width=True)

    st.markdown("---")

    # Model evaluation (only if trained)
    if st.session_state.train_result:
        m = st.session_state.train_result["metrics"]
        st.markdown("### 🎯 Model Evaluation Visuals")

        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(plot_metrics_radar(m), use_container_width=True)
        with col2:
            st.plotly_chart(
                plot_confusion_matrix(m["confusion_matrix"], m["labels"]),
                use_container_width=True,
            )

        # Top features
        st.markdown("### 🏆 Top Predictive Words")
        try:
            feats = get_top_features(st.session_state.pipeline, n=15)
            if feats:
                st.plotly_chart(plot_top_features(feats), use_container_width=True)
            else:
                st.info("Feature importances only available for Logistic Regression.")
        except Exception:
            pass

        # Summary table
        st.markdown("### 📋 Per-Class Metrics Summary")
        report = m["classification_report"]
        rows = []
        for cls in ["Positive", "Negative", "Neutral"]:
            if cls in report:
                r = report[cls]
                rows.append({
                    "Sentiment": cls,
                    "Precision": f"{r['precision']*100:.1f}%",
                    "Recall":    f"{r['recall']*100:.1f}%",
                    "F1-Score":  f"{r['f1-score']*100:.1f}%",
                    "Support":   int(r["support"]),
                })
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            st.caption("ℹ️ Precision measures accuracy of predictions for that class; Recall measures what share of true class reviews was captured. Support is the test sample count.")

    else:
        st.info("💡 Train a model on the **🤖 Model Training** page to see evaluation charts.")

