"""
train.py
--------
Full training pipeline for IMDB Sentiment Analysis.

Pipeline:
  1. Load IMDB Dataset.csv
  2. Clean & preprocess text
  3. TF-IDF vectorization (100K features, bigrams, sublinear_tf)
  4. Train & compare 4 models
  5. Save best model + vectorizer + metrics

Usage:
  python src/train.py
"""

import os
import sys
import json
import time
import warnings
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score,
)

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT, "data", "IMDB Dataset.csv")
MODEL_DIR = os.path.join(ROOT, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

sys.path.insert(0, os.path.join(ROOT, "src"))
from preprocess import preprocess_series


# ── Helper ─────────────────────────────────────────────────────────────────
def log(msg: str):
    print(f"[train] {msg}")


# ── 1. Load data ───────────────────────────────────────────────────────────
def load_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"\n  Dataset not found at: {DATA_PATH}\n"
            "  Please download IMDB Dataset.csv from Kaggle and place it in data/"
        )
    log(f"Loading dataset from {DATA_PATH} ...")
    df = pd.read_csv(DATA_PATH)

    # Validate columns
    assert "review"    in df.columns, "CSV must have a 'review' column"
    assert "sentiment" in df.columns, "CSV must have a 'sentiment' column"

    log(f"Loaded {len(df):,} rows  |  Columns: {list(df.columns)}")
    log(f"Label distribution:\n{df['sentiment'].value_counts()}")
    return df


# ── 2. Preprocess ──────────────────────────────────────────────────────────
def preprocess(df: pd.DataFrame):
    log("Preprocessing text ...")
    df = df.copy().dropna(subset=["review", "sentiment"])
    df["clean_review"] = preprocess_series(df["review"])
    df["label"] = (df["sentiment"].str.lower() == "positive").astype(int)
    return df


# ── 3. Vectorize ───────────────────────────────────────────────────────────
def vectorize(X_train, X_test):
    log("Fitting TF-IDF vectorizer ...")
    vectorizer = TfidfVectorizer(
        max_features   = 100_000,
        ngram_range    = (1, 2),   # unigrams + bigrams
        sublinear_tf   = True,     # replace tf with 1 + log(tf)
        min_df         = 2,        # ignore very rare terms
        max_df         = 0.95,     # ignore very common terms
        strip_accents  = "unicode",
        analyzer       = "word",
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf  = vectorizer.transform(X_test)
    log(f"Vocabulary size: {len(vectorizer.vocabulary_):,} features")
    return vectorizer, X_train_tfidf, X_test_tfidf


# ── 4. Models ──────────────────────────────────────────────────────────────
def get_models():
    return {
        "Logistic Regression": LogisticRegression(
            C=1.0, max_iter=1000, solver="lbfgs",
            multi_class="auto", n_jobs=-1, random_state=42
        ),
        "LinearSVC (Calibrated)": CalibratedClassifierCV(
            LinearSVC(C=1.0, max_iter=2000, random_state=42), cv=3
        ),
        "SGD Classifier": SGDClassifier(
            loss="modified_huber", alpha=1e-4, max_iter=100,
            random_state=42, n_jobs=-1
        ),
        "Multinomial Naive Bayes": MultinomialNB(alpha=0.1),
    }


def train_and_evaluate(models, X_train, X_test, y_train, y_test):
    results = {}

    for name, model in models.items():
        log(f"Training {name} ...")
        t0 = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - t0

        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        acc     = accuracy_score(y_test, y_pred)
        auc     = roc_auc_score(y_test, y_proba)
        report  = classification_report(y_test, y_pred, output_dict=True)
        cm      = confusion_matrix(y_test, y_pred).tolist()

        results[name] = {
            "model"      : model,
            "accuracy"   : round(acc * 100, 4),
            "roc_auc"    : round(auc, 4),
            "report"     : report,
            "cm"         : cm,
            "train_time" : round(train_time, 2),
        }

        log(
            f"  {name:35s} | Acc: {acc*100:.2f}% | "
            f"AUC: {auc:.4f} | Time: {train_time:.1f}s"
        )

    return results


# ── 5. Save ────────────────────────────────────────────────────────────────
def save_artifacts(best_name, best_model, vectorizer, results):
    # Save model + vectorizer
    joblib.dump(best_model,  os.path.join(MODEL_DIR, "best_model.pkl"))
    joblib.dump(vectorizer,  os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))

    # Save metrics summary (strip model objects for JSON serialization)
    summary = {}
    for name, r in results.items():
        summary[name] = {
            "accuracy"   : r["accuracy"],
            "roc_auc"    : r["roc_auc"],
            "report"     : r["report"],
            "cm"         : r["cm"],
            "train_time" : r["train_time"],
            "is_best"    : (name == best_name),
        }
    with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)

    log(f"Saved best_model.pkl, tfidf_vectorizer.pkl, metrics.json → models/")


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    print("\n" + "="*60)
    print("   IMDB Sentiment Analysis — Training Pipeline")
    print("="*60 + "\n")

    # Load & preprocess
    df = load_data()
    df = preprocess(df)

    # Train/test split (80/20, stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_review"], df["label"],
        test_size=0.2, random_state=42, stratify=df["label"]
    )
    log(f"Split — Train: {len(X_train):,} | Test: {len(X_test):,}")

    # Vectorize
    vectorizer, X_train_v, X_test_v = vectorize(X_train, X_test)

    # Train all models
    models  = get_models()
    results = train_and_evaluate(models, X_train_v, X_test_v, y_train, y_test)

    # Pick best by accuracy
    best_name = max(results, key=lambda k: results[k]["accuracy"])
    best_info = results[best_name]

    print("\n" + "─"*60)
    log(f"BEST MODEL : {best_name}")
    log(f"Accuracy   : {best_info['accuracy']}%")
    log(f"ROC-AUC    : {best_info['roc_auc']}")
    print(
        classification_report(
            y_test,
            best_info["model"].predict(X_test_v),
            target_names=["Negative", "Positive"]
        )
    )
    print("─"*60 + "\n")

    # Save
    save_artifacts(best_name, best_info["model"], vectorizer, results)

    print("\nTraining complete! Run the app with:\n")
    print("  streamlit run app.py\n")


if __name__ == "__main__":
    main()
