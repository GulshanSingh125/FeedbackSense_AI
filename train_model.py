"""Dataset download, preparation, model training, and evaluation for FeedbackSense AI."""
from pathlib import Path
import requests
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
CSV_PATH = DATA_DIR / "customer_feedback.csv"
MODEL_PATH = MODEL_DIR / "feedback_classifier.joblib"
REPORT_PATH = REPORT_DIR / "evaluation_results.txt"

DATASET_REPO = "https://huggingface.co/datasets/AnkitAI/product-reviews-sentiment"
CSV_URL = (
    "https://huggingface.co/datasets/"
    "AnkitAI/product-reviews-sentiment/"
    "resolve/main/categorized_text_reviews.csv"
)

def download_dataset(force=False):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if CSV_PATH.exists() and not force and CSV_PATH.stat().st_size > 0:
        return CSV_PATH

    response = requests.get(CSV_URL, timeout=90, allow_redirects=True)
    response.raise_for_status()
    if not response.content or b"<html" in response.content[:500].lower():
        raise RuntimeError(
            "Hugging Face did not return a CSV. Check the current file name under "
            f"{DATASET_REPO}/tree/main and update CSV_URL in train_model.py."
        )
    CSV_PATH.write_bytes(response.content)
    return CSV_PATH

def load_dataset():
    download_dataset()
    raw = pd.read_csv(CSV_PATH)
    column_lookup = {str(c).strip().lower(): c for c in raw.columns}
    text_candidates = ["review", "text", "feedback", "customer_feedback", "content"]
    label_candidates = ["category", "label", "class", "target"]
    text_col = next((column_lookup[c] for c in text_candidates if c in column_lookup), None)
    label_col = next((column_lookup[c] for c in label_candidates if c in column_lookup), None)

    if text_col is None or label_col is None:
        raise ValueError(
            f"Could not identify text/category columns. Found columns: {raw.columns.tolist()}. "
            "Update text_candidates and label_candidates in train_model.py."
        )

    df = raw[[text_col, label_col]].copy()
    df.columns = ["feedback", "category"]
    df["feedback"] = df["feedback"].astype("string").str.strip()
    df["category"] = df["category"].astype("string").str.strip()
    df = df.dropna(subset=["feedback", "category"])
    df = df[(df["feedback"] != "") & (df["category"] != "")]
    # Exact duplicate text-label pairs can inflate evaluation. We remove them before splitting.
    df = df.drop_duplicates(subset=["feedback", "category"]).reset_index(drop=True)
    return raw, df

def build_model():
    return Pipeline([
        ("features", FeatureUnion([
            ("word_tfidf", TfidfVectorizer(
                lowercase=True, strip_accents="unicode",
                ngram_range=(1, 2), min_df=1, sublinear_tf=True
            )),
            ("char_tfidf", TfidfVectorizer(
                analyzer="char", ngram_range=(3, 5),
                min_df=1, sublinear_tf=True
            ))
        ])),
        ("classifier", LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=42
        ))
    ])

def train_and_evaluate():
    _, df = load_dataset()
    if df["category"].nunique() < 2:
        raise ValueError("At least two category labels are required.")
    if df["category"].value_counts().min() < 2:
        raise ValueError("At least two unique records per category are required for a stratified split.")

    X = df["feedback"].astype(str)
    y = df["category"].astype(str)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    model = build_model()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    labels = sorted(y.unique())
    report = classification_report(y_test, predictions, labels=labels, zero_division=0)
    cm = confusion_matrix(y_test, predictions, labels=labels)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    lines = [
        "FeedbackSense AI — Model Evaluation",
        "====================================",
        f"Dataset repository: {DATASET_REPO}",
        f"Cleaned unique text-label pairs: {len(df)}",
        f"Training rows: {len(X_train)}",
        f"Testing rows: {len(X_test)}",
        f"Categories: {', '.join(labels)}",
        f"Accuracy: {accuracy:.4f} ({accuracy:.1%})",
        "",
        "Classification report:",
        report,
        "",
        "Confusion matrix (labels in this order):",
        repr(labels),
        str(cm),
        "",
        "Caution: dataset card describes data as synthetic. Template/near-duplicate examples may inflate scores."
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    return model, df, accuracy, report, cm

def ensure_model():
    if MODEL_PATH.exists():
        try:
            return joblib.load(MODEL_PATH)
        except Exception:
            MODEL_PATH.unlink(missing_ok=True)
    model, *_ = train_and_evaluate()
    return model

if __name__ == "__main__":
    model, df, accuracy, report, cm = train_and_evaluate()
    print(f"Dataset: {CSV_PATH}")
    print(f"Cleaned unique records: {len(df)}")
    print(f"Accuracy: {accuracy:.4f}")
    print(report)
    print(f"Saved model to: {MODEL_PATH}")
    print(f"Saved report to: {REPORT_PATH}")
