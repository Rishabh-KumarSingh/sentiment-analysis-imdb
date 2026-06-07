"""
evaluate.py
-----------
Reload the saved best model and run a full evaluation report.
Use this after training to regenerate metrics or test on new data.

Usage:
  python src/evaluate.py
  python src/evaluate.py --csv path/to/custom_test.csv
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score,
    ConfusionMatrixDisplay,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from preprocess import clean_text

MODEL_PATH = os.path.join(ROOT, "models", "best_model.pkl")
TFIDF_PATH = os.path.join(ROOT, "models", "tfidf_vectorizer.pkl")


def load_artifacts():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "Model not found. Run `python src/train.py` first."
        )
    model      = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(TFIDF_PATH)
    return model, vectorizer


def evaluate_from_csv(csv_path: str):
    """Evaluate on a CSV with 'review' and 'sentiment' columns."""
    print(f"\nLoading test data from {csv_path} ...")
    df = pd.read_csv(csv_path).dropna(subset=["review", "sentiment"])
    df["clean"] = df["review"].apply(clean_text)
    df["label"] = (df["sentiment"].str.lower() == "positive").astype(int)

    model, vectorizer = load_artifacts()
    X = vectorizer.transform(df["clean"])

    y_pred  = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    acc = accuracy_score(df["label"], y_pred)
    auc = roc_auc_score(df["label"], y_proba)

    print(f"\nAccuracy : {acc*100:.4f}%")
    print(f"ROC-AUC  : {auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(
        df["label"], y_pred, target_names=["Negative", "Positive"]
    ))
    print("Confusion Matrix:")
    print(confusion_matrix(df["label"], y_pred))


def predict_single(text: str):
    """Predict sentiment for a single review text."""
    model, vectorizer = load_artifacts()
    cleaned = clean_text(text)
    X       = vectorizer.transform([cleaned])
    pred    = model.predict(X)[0]
    proba   = model.predict_proba(X)[0]

    label   = "POSITIVE" if pred == 1 else "NEGATIVE"
    conf    = proba[pred] * 100

    print(f"\nText    : {text[:100]}...")
    print(f"Cleaned : {cleaned[:100]}...")
    print(f"Label   : {label}")
    print(f"Confidence: {conf:.1f}%")
    return label, conf


def show_stored_metrics():
    """Print metrics from the last training run."""
    metrics_path = os.path.join(ROOT, "models", "metrics.json")
    if not os.path.exists(metrics_path):
        print("No metrics.json found. Run train.py first.")
        return

    with open(metrics_path) as f:
        metrics = json.load(f)

    print("\n" + "="*60)
    print("  Stored Training Metrics")
    print("="*60)
    for name, info in metrics.items():
        tag = " ← BEST" if info.get("is_best") else ""
        print(f"\n  {name}{tag}")
        print(f"  Accuracy : {info['accuracy']}%")
        print(f"  ROC-AUC  : {info['roc_auc']}")
        print(f"  Train time: {info['train_time']}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Sentiment Model")
    parser.add_argument("--csv",  type=str, default=None, help="Path to test CSV")
    parser.add_argument("--text", type=str, default=None, help="Single review to predict")
    args = parser.parse_args()

    if args.csv:
        evaluate_from_csv(args.csv)
    elif args.text:
        predict_single(args.text)
    else:
        show_stored_metrics()
