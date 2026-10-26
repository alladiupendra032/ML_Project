"""
Machine Learning Model Module
Handles TF-IDF feature extraction, model training (Logistic Regression,
Naive Bayes, SVM), model evaluation, and prediction.
"""

import numpy as np
import pandas as pd
from collections import Counter
from typing import Dict, Tuple, Any, List, Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix,
)
from sklearn.pipeline import Pipeline

LABEL_COLORS = {
    "Positive": "#2ecc71",
    "Negative": "#e74c3c",
    "Neutral":  "#f39c12",
}

# ── Factories: a fresh instance every call — avoids state contamination ───────
MODEL_FACTORIES = {
    "Logistic Regression (Primary)": lambda: LogisticRegression(max_iter=1000, C=1.0, random_state=42),
    "Naive Bayes":                   lambda: MultinomialNB(alpha=0.5),
    "SVM (Linear)":                  lambda: LinearSVC(max_iter=1000, random_state=42),
}

# Kept for UI iteration (just the names)
MODELS = {k: None for k in MODEL_FACTORIES}


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _to_str_list(arr) -> List[str]:
    """Convert any array-like (numpy, list, etc.) to a plain Python list of str."""
    return [str(x) for x in arr]


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / e.sum()


# ─── Pipeline builder ─────────────────────────────────────────────────────────

def build_pipeline(model_name: str = "Logistic Regression (Primary)") -> Pipeline:
    """Return a fresh TF-IDF → Classifier sklearn Pipeline."""
    vectorizer = TfidfVectorizer(
        max_features=15000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
        strip_accents="unicode",
    )
    factory = MODEL_FACTORIES.get(model_name, MODEL_FACTORIES["Logistic Regression (Primary)"])
    clf = factory()
    return Pipeline([("tfidf", vectorizer), ("clf", clf)])


# ─── Training ─────────────────────────────────────────────────────────────────

def train_model(
    df: pd.DataFrame,
    text_col: str = "cleaned_text",
    label_col: str = "sentiment",
    model_name: str = "Logistic Regression (Primary)",
    test_size: float = 0.20,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Train a sentiment classifier and return metrics + fitted pipeline.

    Key fixes vs the broken version:
    - X and y are plain Python lists[str] (not numpy arrays of numpy strings).
      This prevents confusion_matrix from receiving numpy string objects,
      which caused "only integer scalar arrays can be converted to a scalar index".
    - A fresh classifier instance is created each call via MODEL_FACTORIES.
    - Labels are sorted plain-str lists, safe for confusion_matrix.
    """
    df = df.dropna(subset=[text_col, label_col]).copy()
    df = df[df[text_col].astype(str).str.strip() != ""]

    # ── Plain Python list[str] — critical to avoid numpy-string confusion ──
    X: List[str] = df[text_col].astype(str).tolist()
    y: List[str] = df[label_col].astype(str).tolist()

    if len(X) < 10:
        raise ValueError(f"Not enough data to train: only {len(X)} rows after cleaning.")

    # Stratify only when every class has >= 2 samples
    label_counts = Counter(y)
    can_stratify = len(label_counts) > 1 and all(c >= 2 for c in label_counts.values())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if can_stratify else None,
    )

    pipeline = build_pipeline(model_name)
    pipeline.fit(X_train, y_train)

    # ── Convert predictions + ground truth to plain str lists ────────────
    y_pred = _to_str_list(pipeline.predict(X_test))
    y_test = _to_str_list(y_test)

    # Labels: standardized natural order for confusion matrix
    all_classes = sorted(set(y_test + y_pred))
    preferred_order = ["Positive", "Neutral", "Negative"]
    labels: List[str] = [c for c in preferred_order if c in all_classes] + [c for c in all_classes if c not in preferred_order]

    cm     = confusion_matrix(y_test, y_pred, labels=labels)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    metrics = {
        "accuracy":              accuracy_score(y_test, y_pred),
        "precision":             precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall":                recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1_score":              f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "confusion_matrix":      cm,
        "labels":                labels,
        "classification_report": report,
        "train_size":            len(X_train),
        "test_size":             len(X_test),
        "model_name":            model_name,
    }

    return {
        "pipeline": pipeline,
        "metrics":  metrics,
        "X_train":  X_train,
        "X_test":   X_test,
        "y_train":  y_train,
        "y_test":   y_test,
        "y_pred":   y_pred,
    }


# ─── Prediction ───────────────────────────────────────────────────────────────

def predict_single(pipeline: Pipeline, text: str, preprocessor_fn=None) -> Dict[str, Any]:
    """Predict sentiment for a single text string."""
    processed = preprocessor_fn(text) if preprocessor_fn else text
    # Fallback: if preprocessing emptied the text, use the raw input
    if not processed or not processed.strip():
        processed = text.strip()

    clf = pipeline.named_steps["clf"]
    prediction = str(pipeline.predict([processed])[0])

    proba   = None
    classes = None
    try:
        # Logistic Regression / Naive Bayes
        proba   = pipeline.predict_proba([processed])[0].tolist()
        classes = _to_str_list(clf.classes_)
    except AttributeError:
        try:
            # LinearSVC: decision_function → softmax approximation
            decision = np.array(pipeline.decision_function([processed])[0])
            proba    = _softmax(decision).tolist()
            classes  = _to_str_list(clf.classes_)
        except Exception:
            pass

    return {
        "text":           text,
        "processed_text": processed,
        "prediction":     prediction,
        "probabilities":  dict(zip(classes, proba)) if (proba is not None and classes) else None,
    }


def predict_batch(pipeline: Pipeline, texts: List[str], preprocessor_fn=None) -> pd.DataFrame:
    """Predict sentiments for a list of texts."""
    return pd.DataFrame([predict_single(pipeline, t, preprocessor_fn) for t in texts])


# ─── Model Comparison ─────────────────────────────────────────────────────────

def compare_models(
    df: pd.DataFrame,
    text_col: str = "cleaned_text",
    label_col: str = "sentiment",
    test_size: float = 0.20,
) -> pd.DataFrame:
    """Train every model and return a side-by-side comparison DataFrame."""
    rows = []
    for name in MODEL_FACTORIES:
        try:
            result = train_model(df, text_col, label_col, model_name=name, test_size=test_size)
            m = result["metrics"]
            rows.append({
                "Model":     name,
                "Accuracy":  round(m["accuracy"]  * 100, 2),
                "Precision": round(m["precision"] * 100, 2),
                "Recall":    round(m["recall"]    * 100, 2),
                "F1-Score":  round(m["f1_score"]  * 100, 2),
            })
        except Exception as e:
            rows.append({
                "Model": name,
                "Accuracy": 0, "Precision": 0, "Recall": 0, "F1-Score": 0,
                "Error": str(e),
            })
    return pd.DataFrame(rows)


# ─── Feature Importance ───────────────────────────────────────────────────────

def get_top_features(pipeline: Pipeline, n: int = 20) -> Dict[str, List[Tuple[str, float]]]:
    """Return top n TF-IDF token weights per class (Logistic Regression only)."""
    clf           = pipeline.named_steps["clf"]
    vec           = pipeline.named_steps["tfidf"]
    feature_names = vec.get_feature_names_out()

    if not hasattr(clf, "coef_"):
        return {}

    classes = _to_str_list(getattr(clf, "classes_", []))
    coef    = clf.coef_
    result  = {}

    for i, cls in enumerate(classes):
        top_idx    = np.argsort(coef[i])[-n:][::-1]
        result[cls] = [(feature_names[j], float(coef[i][j])) for j in top_idx]

    return result
