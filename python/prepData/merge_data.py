"""Merge spam.csv, Dataset_10191.csv and phishing_legit_dataset_KD_10000.csv into Data/data.csv with columns v1 (label), v2 (text)."""
import csv
from pathlib import Path

DATA_DIR = Path(__file__).parent / "Data"
OUTPUT = DATA_DIR / "data.csv"

# (path, encoding, label column, text column)
SOURCES = [
    (DATA_DIR / "Text" / "spam.csv", "latin-1", 0, 1),
    (DATA_DIR / "A Balanced Dataset for Spam and Smishing Detection" / "Dataset_10191.csv", "utf-8", 0, 1),
    (DATA_DIR / "Mail" / "phishing_legit_dataset_KD_10000.csv", "utf-8", 1, 0),
]

# smishing is treated as spam; the phishing dataset uses 1 = phishing, 0 = legitimate
LABEL_MAP = {"ham": "ham", "spam": "spam", "smishing": "spam", "1": "spam", "0": "ham"}


def clean_text(text):
    """Drop "Keywords: ..." lines and join the rest into a single line."""
    lines = [line.strip() for line in text.splitlines()]
    return " ".join(line for line in lines if line and not line.lower().startswith("keywords:"))


def read_rows(path, encoding, label_col, text_col):
    with open(path, encoding=encoding, newline="") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            label = LABEL_MAP.get(row[label_col].strip().lower())
            text = clean_text(row[text_col])
            if label and text:
                yield label, text


def main():
    total = 0
    with open(OUTPUT, "w", encoding="utf-8", newline="") as out:
        writer = csv.writer(out)
        writer.writerow(["v1", "v2"])
        for path, encoding, label_col, text_col in SOURCES:
            count = 0
            for label, text in read_rows(path, encoding, label_col, text_col):
                writer.writerow([label, text])
                count += 1
            print(f"{path.name}: {count} rows")
            total += count
    print(f"Wrote {total} rows to {OUTPUT}")


if __name__ == "__main__":
    main()
