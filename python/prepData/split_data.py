"""Split data.csv into data/train.csv (80%) and data/test.csv (20%), each with 50% ham and 50% spam."""
import csv
import random
from pathlib import Path

ROOT = Path(__file__).parent.parent
SOURCE = ROOT / "data.csv"
OUTPUT_DIR = ROOT / "data"

TRAIN_SHARE = 0.8
SEED = None  # set to a number (e.g. 42) to get the same split every run


def main():
    rng = random.Random(SEED)

    with open(SOURCE, encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    by_label = {"ham": [], "spam": []}
    for row in rows:
        by_label[row[0]].append(row)

    # Balance the classes by cutting the larger one down to the size of the smaller one
    per_label = min(len(group) for group in by_label.values())
    train_per_label = round(per_label * TRAIN_SHARE)

    train, test = [], []
    for group in by_label.values():
        rng.shuffle(group)
        train += group[:train_per_label]
        test += group[train_per_label:per_label]

    rng.shuffle(train)
    rng.shuffle(test)

    OUTPUT_DIR.mkdir(exist_ok=True)
    for name, data in (("train.csv", train), ("test.csv", test)):
        with open(OUTPUT_DIR / name, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(data)
        print(f"{name}: {len(data)} rows ({len(data) // 2} ham, {len(data) // 2} spam)")

    for label, group in by_label.items():
        if len(group) > per_label:
            print(f"Left out {len(group) - per_label} {label} rows to keep the classes balanced")


if __name__ == "__main__":
    main()
