import json
import re
from pathlib import Path

import joblib
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "training_data.csv"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)


def preprocess_text(text: str) -> str:
    if pd.isna(text):
        return ""

    text = str(text).lower()

    # Keep URL tokens so the model learns phishing URL patterns
    # e.g. paypal-secure-login.tk, verify, login, .ru, .tk
    # Replace only characters that break tokenization but keep slashes, dots, dashes
    text = re.sub(r"[^a-z0-9\s./:@_-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_urls(text: str) -> list[str]:
    if pd.isna(text):
        return []

    text = str(text)
    return re.findall(r"https?://[^\s]+|www\.[^\s]+", text)


def extract_url_features(text: str) -> dict:
    urls = extract_urls(text)

    if not urls:
        return {
            "url_count": 0,
            "max_url_length": 0,
            "avg_url_length": 0.0,
            "has_ip_url": 0,
            "has_at_symbol": 0,
            "has_dash": 0,
            "has_many_dots": 0,
            "has_suspicious_pattern": 0,
            "uses_https": 0,
        }

    lengths = [len(url) for url in urls]

    has_ip_url = any(
        re.search(r"(?:https?://)?(?:\d{1,3}\.){3}\d{1,3}", url) for url in urls
    )
    has_at_symbol = any("@" in url for url in urls)
    has_dash = any("-" in url for url in urls)
    has_many_dots = any(url.count(".") >= 3 for url in urls)
    has_suspicious_pattern = any(
        any(token in url.lower() for token in ["verify", "login", "secure", "update", "%", "@"])
        for url in urls
    )
    uses_https = any(url.lower().startswith("https://") for url in urls)

    return {
        "url_count": len(urls),
        "max_url_length": max(lengths),
        "avg_url_length": sum(lengths) / len(lengths),
        "has_ip_url": int(has_ip_url),
        "has_at_symbol": int(has_at_symbol),
        "has_dash": int(has_dash),
        "has_many_dots": int(has_many_dots),
        "has_suspicious_pattern": int(has_suspicious_pattern),
        "uses_https": int(uses_https),
    }


def normalize_label(value: str) -> int:
    value = str(value).strip().lower()

    safe_values = {"safe", "ham", "legitimate", "normal", "0"}
    malicious_values = {"spam", "phishing", "spam/phishing", "malicious", "1"}

    if value in safe_values:
        return 0
    if value in malicious_values:
        return 1

    raise ValueError(f"Unsupported label found in dataset: {value}")


def load_and_clean_dataset() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    required_columns = {"subject", "body", "label"}
    if not required_columns.issubset(df.columns):
        raise ValueError("CSV must contain these columns: subject, body, label")

    df["subject"] = df["subject"].fillna("").astype(str)
    df["body"] = df["body"].fillna("").astype(str)
    df["label"] = df["label"].fillna("").astype(str)

    df["combined_text"] = (df["subject"] + " " + df["body"]).str.strip()

    df = df[df["combined_text"] != ""].copy()
    df = df.drop_duplicates(subset=["subject", "body", "label"]).reset_index(drop=True)

    df["label_binary"] = df["label"].apply(normalize_label)

    return df


def main() -> None:
    df = load_and_clean_dataset()

    print("\nDataset loaded successfully.")
    print(f"Total cleaned rows: {len(df)}")
    print("Label distribution:")
    print(df["label"].value_counts())

    df["clean_text"] = df["combined_text"].apply(preprocess_text)

    url_feature_df = df["body"].apply(extract_url_features).apply(pd.Series)
    df = pd.concat([df, url_feature_df], axis=1)

    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        stop_words="english",
        min_df=2,
    )

    X_text = vectorizer.fit_transform(df["clean_text"])

    url_feature_columns = [
        "url_count",
        "max_url_length",
        "avg_url_length",
        "has_ip_url",
        "has_at_symbol",
        "has_dash",
        "has_many_dots",
        "has_suspicious_pattern",
        "uses_https",
    ]

    X_url = csr_matrix(df[url_feature_columns].astype(float).values)
    X = hstack([X_text, X_url], format="csr")
    y = df["label_binary"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }

    report_text = classification_report(
        y_test,
        y_pred,
        target_names=["Safe", "Spam/Phishing"],
        zero_division=0,
    )

    joblib.dump(model, MODELS_DIR / "xgb_email_model.pkl")
    joblib.dump(vectorizer, MODELS_DIR / "tfidf_vectorizer.pkl")
    joblib.dump(url_feature_columns, MODELS_DIR / "url_feature_columns.pkl")

    with open(MODELS_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    with open(MODELS_DIR / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report_text)

    print("\nModel training completed successfully.")
    print("Saved files:")
    print("- models/xgb_email_model.pkl")
    print("- models/tfidf_vectorizer.pkl")
    print("- models/url_feature_columns.pkl")
    print("- models/metrics.json")
    print("- models/classification_report.txt")

    print("\nEvaluation Metrics:")
    print(f"Accuracy  : {metrics['accuracy']:.4f}")
    print(f"Precision : {metrics['precision']:.4f}")
    print(f"Recall    : {metrics['recall']:.4f}")
    print(f"F1-score  : {metrics['f1_score']:.4f}")
    print(f"Confusion Matrix: {metrics['confusion_matrix']}")

    print("\nSample confidence values (Spam/Phishing probability):")
    print([round(value, 4) for value in y_prob[:5]])


if __name__ == "__main__":
    main()