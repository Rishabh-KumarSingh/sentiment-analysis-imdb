"""
app.py
------
Streamlit app for IMDB Sentiment Analysis
Three tabs: Single Review · Batch CSV · Model Dashboard
"""

import os
import sys
import json
import io
import warnings
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

warnings.filterwarnings("ignore")

# ── Paths ───────────────────────────────────────────────────────────────────
ROOT       = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(ROOT, "models", "best_model.pkl")
TFIDF_PATH = os.path.join(ROOT, "models", "tfidf_vectorizer.pkl")
METRICS_PATH = os.path.join(ROOT, "models", "metrics.json")

sys.path.insert(0, os.path.join(ROOT, "src"))
from preprocess import clean_text

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "Sentiment Analysis",
    page_icon  = "🎬",
    layout     = "wide",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .main-title {
    font-size: 2.4rem; font-weight: 700;
    background: linear-gradient(135deg, #6C63FF, #FF6584);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 0;
  }
  .subtitle {
    color: #888; font-size: 1rem; margin-top: 0.2rem;
  }
  .pos-badge {
    background: #d4edda; color: #155724;
    padding: 0.4rem 1.2rem; border-radius: 20px;
    font-weight: 700; font-size: 1.4rem; display: inline-block;
  }
  .neg-badge {
    background: #f8d7da; color: #721c24;
    padding: 0.4rem 1.2rem; border-radius: 20px;
    font-weight: 700; font-size: 1.4rem; display: inline-block;
  }
  .metric-card {
    background: #f8f9fa; border-radius: 12px;
    padding: 1rem 1.4rem; margin-bottom: 1rem;
    border-left: 4px solid #6C63FF;
  }
  .stTabs [data-baseweb="tab-list"] { gap: 8px; }
  .stTabs [data-baseweb="tab"] {
    height: 44px; padding: 0 24px;
    border-radius: 8px 8px 0 0; font-weight: 600;
  }
</style>
""", unsafe_allow_html=True)


# ── Load model (cached) ─────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model...")
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None, None
    model      = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(TFIDF_PATH)
    return model, vectorizer


@st.cache_data
def load_metrics():
    if not os.path.exists(METRICS_PATH):
        return None
    with open(METRICS_PATH) as f:
        return json.load(f)


# ── Prediction helper ───────────────────────────────────────────────────────
def predict(text: str, model, vectorizer):
    cleaned = clean_text(text)
    X       = vectorizer.transform([cleaned])
    pred    = model.predict(X)[0]
    proba   = model.predict_proba(X)[0]
    label   = "POSITIVE" if pred == 1 else "NEGATIVE"
    conf    = float(proba[1]) if pred == 1 else float(proba[0])
    return label, conf, cleaned


def get_top_words(text_cleaned: str, vectorizer, model, n=15):
    """
    Get top contributing words from the review using TF-IDF + model coefficients.
    Works for models with coef_ (LR, LinearSVC).
    """
    try:
        X        = vectorizer.transform([text_cleaned])
        feature_names = np.array(vectorizer.get_feature_names_out())

        # Get underlying estimator if calibrated
        estimator = getattr(model, "estimator", model)
        coefs     = getattr(estimator, "coef_", None)

        if coefs is None:
            return None

        coefs = coefs[0] if coefs.ndim > 1 else coefs
        X_arr  = X.toarray()[0]
        scores = X_arr * coefs

        # Non-zero scores only
        nonzero = np.where(X_arr > 0)[0]
        if len(nonzero) == 0:
            return None

        nonzero_scores = scores[nonzero]
        nonzero_words  = feature_names[nonzero]

        # Sort by abs importance
        idx = np.argsort(np.abs(nonzero_scores))[::-1][:n]
        return pd.DataFrame({
            "word"       : nonzero_words[idx],
            "score"      : nonzero_scores[idx],
            "sentiment"  : ["Positive" if s > 0 else "Negative" for s in nonzero_scores[idx]],
        })
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<p class="main-title">🎬 Movie Sentiment Analyzer</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">IMDB Dataset · TF-IDF + LinearSVC · Pinnacle Labs — Data Science Internship</p>', unsafe_allow_html=True)
st.divider()

model, vectorizer = load_model()
metrics           = load_metrics()

if model is None:
    st.error(
        "**Model not found.** Please run training first:\n\n"
        "```bash\npython src/train.py\n```"
    )
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "🔍  Single Review",
    "📂  Batch CSV",
    "📊  Model Dashboard",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — SINGLE REVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Analyze a single movie review")

    # Sample reviews
    examples = {
        "Select an example...": "",
        "🟢 Positive example": (
            "This film was an absolute masterpiece. The direction was brilliant, "
            "the acting was phenomenal, and the storyline kept me on the edge of "
            "my seat throughout. Easily one of the best movies I have ever watched."
        ),
        "🔴 Negative example": (
            "Terrible movie. The plot made absolutely no sense and the characters "
            "were completely unlikeable. The special effects were laughably bad and "
            "the pacing was unbearably slow. Definitely not worth your time."
        ),
        "🟡 Mixed example": (
            "The movie had some interesting ideas and the visuals were decent, "
            "but the execution was poor. The first half was promising but it "
            "completely fell apart in the second half. Mediocre at best."
        ),
    }

    col_left, col_right = st.columns([2, 1])

    with col_left:
        selected = st.selectbox("Quick examples", options=list(examples.keys()))
        review_text = st.text_area(
            "Paste your review here",
            value     = examples[selected],
            height    = 180,
            placeholder="Type or paste a movie review to analyze its sentiment...",
        )
        analyze_btn = st.button("Analyze Sentiment", type="primary", use_container_width=True)

    with col_right:
        st.markdown("**How it works**")
        st.markdown("""
        1. Your text is cleaned (HTML, punctuation removed)
        2. Stopwords filtered, negations preserved
        3. WordNet lemmatization applied
        4. TF-IDF vectorized (100K features, bigrams)
        5. LinearSVC predicts sentiment
        """)

    if analyze_btn and review_text.strip():
        with st.spinner("Analyzing..."):
            label, conf, cleaned = predict(review_text, model, vectorizer)

        st.divider()
        r1, r2, r3 = st.columns(3)

        badge_class = "pos-badge" if label == "POSITIVE" else "neg-badge"
        r1.markdown(f'<div class="{badge_class}">{label}</div>', unsafe_allow_html=True)
        r2.metric("Confidence", f"{conf*100:.1f}%")
        r3.metric("Word count (cleaned)", len(cleaned.split()))

        # Confidence gauge
        fig_gauge = go.Figure(go.Indicator(
            mode  = "gauge+number",
            value = conf * 100,
            title = {"text": "Confidence %"},
            gauge = {
                "axis"  : {"range": [0, 100]},
                "bar"   : {"color": "#6C63FF"},
                "steps" : [
                    {"range": [0,  40], "color": "#fee"},
                    {"range": [40, 65], "color": "#fff3cd"},
                    {"range": [65, 100], "color": "#d4edda"},
                ],
                "threshold": {
                    "line" : {"color": "red", "width": 2},
                    "thickness": 0.75, "value": 50,
                },
            },
            number = {"suffix": "%"},
        ))
        fig_gauge.update_layout(height=280, margin=dict(t=30, b=10, l=20, r=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

        # Top contributing words
        word_df = get_top_words(cleaned, vectorizer, model)
        if word_df is not None and not word_df.empty:
            st.subheader("Top contributing words")
            color_map = {"Positive": "#28a745", "Negative": "#dc3545"}
            fig_words = px.bar(
                word_df.sort_values("score"),
                x          = "score",
                y          = "word",
                color      = "sentiment",
                color_discrete_map = color_map,
                orientation = "h",
                labels     = {"score": "TF-IDF × Model Weight", "word": ""},
                height     = 420,
            )
            fig_words.update_layout(margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig_words, use_container_width=True)

        with st.expander("See cleaned text"):
            st.write(cleaned)

    elif analyze_btn:
        st.warning("Please enter a review to analyze.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — BATCH CSV
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Batch prediction on CSV file")
    st.markdown(
        "Upload a CSV with a **`review`** column. "
        "An optional **`sentiment`** column enables accuracy reporting."
    )

    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded:
        try:
            df_batch = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
            st.stop()

        if "review" not in df_batch.columns:
            st.error("CSV must have a `review` column.")
            st.stop()

        st.write(f"**{len(df_batch):,} reviews loaded.** Preview:")
        st.dataframe(df_batch.head(5), use_container_width=True)

        if st.button("Run Batch Predictions", type="primary"):
            progress = st.progress(0, text="Starting...")
            n = len(df_batch)

            labels, confs, cleaned_texts = [], [], []
            for i, row in df_batch.iterrows():
                lbl, conf, cln = predict(str(row["review"]), model, vectorizer)
                labels.append(lbl)
                confs.append(round(conf * 100, 2))
                cleaned_texts.append(cln)
                if i % max(1, n // 50) == 0:
                    progress.progress(min(i / n, 1.0), text=f"Processing {i+1}/{n}...")

            progress.progress(1.0, text="Done!")

            df_batch["predicted_sentiment"] = labels
            df_batch["confidence_%"]        = confs

            st.divider()
            # Summary stats
            b1, b2, b3, b4 = st.columns(4)
            pos_count = (df_batch["predicted_sentiment"] == "POSITIVE").sum()
            neg_count = (df_batch["predicted_sentiment"] == "NEGATIVE").sum()
            b1.metric("Total Reviews",   f"{n:,}")
            b2.metric("Positive",        f"{pos_count:,}")
            b3.metric("Negative",        f"{neg_count:,}")
            b4.metric("Avg Confidence",  f"{df_batch['confidence_%'].mean():.1f}%")

            # Pie chart
            fig_pie = px.pie(
                names  = ["Positive", "Negative"],
                values = [pos_count, neg_count],
                color  = ["Positive", "Negative"],
                color_discrete_map = {"Positive": "#28a745", "Negative": "#dc3545"},
                title  = "Sentiment Distribution",
            )
            fig_pie.update_layout(height=300, margin=dict(t=40,b=10,l=0,r=0))
            st.plotly_chart(fig_pie, use_container_width=True)

            # Accuracy if ground truth present
            if "sentiment" in df_batch.columns:
                df_batch["_true"] = df_batch["sentiment"].str.upper()
                acc = (df_batch["predicted_sentiment"] == df_batch["_true"]).mean() * 100
                st.success(f"Accuracy vs ground truth: **{acc:.2f}%**")
                df_batch.drop(columns=["_true"], inplace=True)

            # Confidence distribution
            fig_conf = px.histogram(
                df_batch, x="confidence_%", nbins=20,
                title="Confidence Distribution",
                color_discrete_sequence=["#6C63FF"],
            )
            fig_conf.update_layout(height=260, margin=dict(t=40,b=10))
            st.plotly_chart(fig_conf, use_container_width=True)

            # Preview results
            st.subheader("Results preview")
            st.dataframe(df_batch[["review", "predicted_sentiment", "confidence_%"]].head(20), use_container_width=True)

            # Download
            csv_out = df_batch.to_csv(index=False).encode("utf-8")
            st.download_button(
                label     = "Download full predictions CSV",
                data      = csv_out,
                file_name = "sentiment_predictions.csv",
                mime      = "text/csv",
            )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — MODEL DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Model performance dashboard")

    if metrics is None:
        st.warning("No metrics found. Run training first.")
        st.stop()

    # ── Model comparison bar chart ─────────────────────────────────────────
    model_names = list(metrics.keys())
    accuracies  = [m["accuracy"] for m in metrics.values()]
    aucs        = [m["roc_auc"] for m in metrics.values()]
    times       = [m["train_time"] for m in metrics.values()]
    is_best     = [m.get("is_best", False) for m in metrics.values()]
    colors      = ["#6C63FF" if b else "#ccc" for b in is_best]

    fig_cmp = go.Figure()
    fig_cmp.add_trace(go.Bar(
        name = "Accuracy (%)",
        x    = model_names,
        y    = accuracies,
        marker_color = colors,
        text = [f"{a:.2f}%" for a in accuracies],
        textposition = "outside",
    ))
    fig_cmp.update_layout(
        title  = "Model Accuracy Comparison",
        yaxis  = dict(range=[80, 100], title="Accuracy (%)"),
        height = 380,
        margin = dict(t=50, b=80),
        xaxis_tickangle = -15,
    )
    st.plotly_chart(fig_cmp, use_container_width=True)

    # ── AUC comparison ────────────────────────────────────────────────────
    fig_auc = go.Figure()
    fig_auc.add_trace(go.Bar(
        name = "ROC-AUC",
        x    = model_names,
        y    = aucs,
        marker_color = ["#FF6584" if b else "#f0a0b0" for b in is_best],
        text = [str(a) for a in aucs],
        textposition = "outside",
    ))
    fig_auc.update_layout(
        title  = "ROC-AUC Comparison",
        yaxis  = dict(range=[0.85, 1.0], title="ROC-AUC"),
        height = 320,
        margin = dict(t=50, b=80),
        xaxis_tickangle = -15,
    )
    st.plotly_chart(fig_auc, use_container_width=True)

    st.divider()

    # ── Per-model details ─────────────────────────────────────────────────
    best_model_name = next((n for n, m in metrics.items() if m.get("is_best")), model_names[0])
    selected_model  = st.selectbox("Inspect model details", options=model_names, index=model_names.index(best_model_name))

    info = metrics[selected_model]

    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Accuracy",   f"{info['accuracy']}%")
    d2.metric("ROC-AUC",    f"{info['roc_auc']}")
    d3.metric("Train time", f"{info['train_time']}s")
    d4.metric("Best model", "✅ Yes" if info.get("is_best") else "No")

    # Confusion matrix
    st.subheader("Confusion matrix")
    cm     = info["cm"]
    cm_arr = np.array(cm)
    fig_cm = px.imshow(
        cm_arr,
        x      = ["Predicted Negative", "Predicted Positive"],
        y      = ["Actual Negative",    "Actual Positive"],
        text_auto = True,
        color_continuous_scale = "Blues",
        title  = f"Confusion Matrix — {selected_model}",
    )
    fig_cm.update_layout(height=360, margin=dict(t=50,b=30))
    st.plotly_chart(fig_cm, use_container_width=True)

    # Classification report table
    st.subheader("Classification report")
    report = info["report"]
    report_df = pd.DataFrame({
        "Class"    : ["Negative", "Positive", "Macro avg", "Weighted avg"],
        "Precision": [
            round(report["0"]["precision"], 4),
            round(report["1"]["precision"], 4),
            round(report["macro avg"]["precision"], 4),
            round(report["weighted avg"]["precision"], 4),
        ],
        "Recall"   : [
            round(report["0"]["recall"], 4),
            round(report["1"]["recall"], 4),
            round(report["macro avg"]["recall"], 4),
            round(report["weighted avg"]["recall"], 4),
        ],
        "F1-score" : [
            round(report["0"]["f1-score"], 4),
            round(report["1"]["f1-score"], 4),
            round(report["macro avg"]["f1-score"], 4),
            round(report["weighted avg"]["f1-score"], 4),
        ],
        "Support"  : [
            int(report["0"]["support"]),
            int(report["1"]["support"]),
            int(report["macro avg"]["support"]),
            int(report["weighted avg"]["support"]),
        ],
    })
    st.dataframe(report_df, use_container_width=True, hide_index=True)

    # Training time comparison
    st.subheader("Training time (seconds)")
    fig_time = px.bar(
        x = model_names, y = times,
        labels = {"x": "", "y": "Seconds"},
        color_discrete_sequence = ["#17a2b8"],
        height = 280,
        text   = [f"{t}s" for t in times],
    )
    fig_time.update_traces(textposition="outside")
    fig_time.update_layout(margin=dict(t=20,b=60), xaxis_tickangle=-15)
    st.plotly_chart(fig_time, use_container_width=True)

    # Footer
    st.divider()
    st.caption(
        "Project 2 — Sentiment Analysis · Pinnacle Labs Data Science Internship · "
        "IMDB 50K Dataset · TF-IDF + Scikit-learn"
    )
