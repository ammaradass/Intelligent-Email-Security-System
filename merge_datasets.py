"""
Dataset Merger — All 5 Sources
Combines SpamAssassin + Kaggle Phishing + phishing_email + Nazario + Nigerian_Fraud
"""

import pandas as pd
from pathlib import Path

BASE_DIR   = Path(__file__).resolve().parent
DATA_DIR   = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

SPAM_PATH    = DATA_DIR / "training_data.csv"
PHISH_PATH   = DATA_DIR / "Phishing_Email.csv"
PHISH2_PATH  = DATA_DIR / "phishing_email_1.csv"
NAZARIO_PATH = DATA_DIR / "Nazario.csv"
NIGERIAN_PATH= DATA_DIR / "Nigerian_Fraud.csv"
OUTPUT_PATH  = DATA_DIR / "training_data.csv"


def load_spamassassin(path):
    print(f"Loading SpamAssassin: {path.name}")
    df = pd.read_csv(path, on_bad_lines="skip")
    df = df[["subject", "body", "label"]].copy()
    df["subject"] = df["subject"].fillna("").astype(str).str.strip()
    df["body"]    = df["body"].fillna("").astype(str).str.strip()
    df["label"]   = df["label"].fillna("").astype(str).str.strip().str.lower()
    df["label"]   = df["label"].replace({"safe": "ham", "legitimate": "ham", "normal": "ham"})
    df = df[df["label"].isin(["ham", "spam"])]
    print(f"  → {len(df)} rows | {df['label'].value_counts().to_dict()}")
    return df


def load_kaggle_phishing(path):
    print(f"Loading Kaggle Phishing: {path.name}")
    df = pd.read_csv(path, on_bad_lines="skip")
    df = df.rename(columns={"Email Text": "body", "Email Type": "label"})
    df["subject"] = ""
    df["body"]    = df["body"].fillna("").astype(str).str.strip()
    df["label"]   = df["label"].replace({"Safe Email": "ham", "Phishing Email": "spam"})
    df = df[["subject", "body", "label"]]
    df = df[df["label"].isin(["ham", "spam"])]
    print(f"  → {len(df)} rows | {df['label'].value_counts().to_dict()}")
    return df


def load_phishing_email(path):
    print(f"Loading phishing_email: {path.name}")
    df = pd.read_csv(path, on_bad_lines="skip")
    # columns: text_combined, label (1=phishing, 0=safe)
    df["subject"] = ""
    df["body"]    = df["text_combined"].fillna("").astype(str).str.strip()
    df["label"]   = df["label"].apply(lambda x: "spam" if int(x) == 1 else "ham")
    df = df[["subject", "body", "label"]]
    print(f"  → {len(df)} rows | {df['label'].value_counts().to_dict()}")
    return df


def load_nazario(path):
    print(f"Loading Nazario: {path.name}")
    df = pd.read_csv(path, on_bad_lines="skip")
    # columns: sender, receiver, date, subject, body, urls, label (all 1)
    df["subject"] = df["subject"].fillna("").astype(str).str.strip()
    df["body"]    = df["body"].fillna("").astype(str).str.strip()
    df["label"]   = "spam"
    df = df[["subject", "body", "label"]]
    print(f"  → {len(df)} rows | {df['label'].value_counts().to_dict()}")
    return df


def load_nigerian_fraud(path):
    print(f"Loading Nigerian_Fraud: {path.name}")
    df = pd.read_csv(path, on_bad_lines="skip")
    # columns: sender, receiver, date, subject, body, urls, label (all 1)
    df["subject"] = df["subject"].fillna("").astype(str).str.strip()
    df["body"]    = df["body"].fillna("").astype(str).str.strip()
    df["label"]   = "spam"
    df = df[["subject", "body", "label"]]
    print(f"  → {len(df)} rows | {df['label'].value_counts().to_dict()}")
    return df


def merge_and_save(frames):
    print("\nMerging all datasets...")
    combined = pd.concat(frames, ignore_index=True)

    before = len(combined)
    combined = combined.drop_duplicates(subset=["subject", "body"]).reset_index(drop=True)
    combined = combined[combined["body"].str.len() > 10].reset_index(drop=True)
    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)

    dupes = before - len(combined)
    counts = combined["label"].value_counts()

    # Save — overwrite training_data.csv
    combined.to_csv(OUTPUT_PATH, index=False)

    print("\n" + "=" * 60)
    print("  MERGE COMPLETE")
    print("=" * 60)
    print(f"  Duplicates removed : {dupes}")
    print(f"  Total rows         : {len(combined)}")
    print(f"  Ham  (safe)        : {counts.get('ham', 0)}")
    print(f"  Spam (phishing)    : {counts.get('spam', 0)}")
    print(f"  Saved to           : {OUTPUT_PATH}")
    print("=" * 60)
    print("\nNext step → run:  python train_model.py")


def main():
    print("=" * 60)
    print("  DATASET MERGER — 5 Sources")
    print("=" * 60)

    frames = []

    if SPAM_PATH.exists():
        frames.append(load_spamassassin(SPAM_PATH))
    else:
        print(f"SKIP: {SPAM_PATH.name} not found")

    if PHISH_PATH.exists():
        frames.append(load_kaggle_phishing(PHISH_PATH))
    else:
        print(f"SKIP: {PHISH_PATH.name} not found")

    if PHISH2_PATH.exists():
        frames.append(load_phishing_email(PHISH2_PATH))
    else:
        print(f"SKIP: {PHISH2_PATH.name} not found")

    if NAZARIO_PATH.exists():
        frames.append(load_nazario(NAZARIO_PATH))
    else:
        print(f"SKIP: {NAZARIO_PATH.name} not found")

    if NIGERIAN_PATH.exists():
        frames.append(load_nigerian_fraud(NIGERIAN_PATH))
    else:
        print(f"SKIP: {NIGERIAN_PATH.name} not found")

    if not frames:
        print("No datasets found!")
        return

    merge_and_save(frames)


if __name__ == "__main__":
    main()