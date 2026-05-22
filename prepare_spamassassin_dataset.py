import csv
from email import policy
from email.parser import BytesParser
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw"
OUTPUT_CSV = BASE_DIR / "data" / "training_data.csv"

DATASET_FOLDERS = [
    ("easy_ham", "safe"),
    ("hard_ham", "safe"),
    ("spam", "spam"),
]


def extract_email_content(file_path: Path) -> tuple[str, str]:
    subject = ""
    body = ""

    try:
        with open(file_path, "rb") as f:
            msg = BytesParser(policy=policy.default).parse(f)

        subject = msg.get("subject", "") or ""

        if msg.is_multipart():
            body_parts = []

            for part in msg.walk():
                content_type = part.get_content_type()
                content_disposition = str(part.get("Content-Disposition", ""))

                if content_type == "text/plain" and "attachment" not in content_disposition:
                    try:
                        content = part.get_content()
                        if content:
                            body_parts.append(str(content))
                    except Exception:
                        continue

            body = "\n".join(body_parts).strip()
        else:
            try:
                content = msg.get_content()
                body = str(content).strip() if content else ""
            except Exception:
                body = ""

    except Exception as e:
        print(f"Failed to parse {file_path}: {e}")

    return subject.strip(), body.strip()


def main():
    rows = []

    for folder_name, label in DATASET_FOLDERS:
        folder_path = RAW_DIR / folder_name

        if not folder_path.exists():
            print(f"Folder not found: {folder_path}")
            continue

        print(f"Reading: {folder_path}")

        for file_path in folder_path.iterdir():
            if not file_path.is_file():
                continue

            subject, body = extract_email_content(file_path)

            if not subject and not body:
                continue

            rows.append({
                "subject": subject,
                "body": body,
                "label": label
            })

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["subject", "body", "label"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDone. CSV saved to: {OUTPUT_CSV}")
    print(f"Total emails: {len(rows)}")


if __name__ == "__main__":
    main()