"""Clean data.csv: remove every copy of the texts in OriginData/humanBaseline.txt
(matched by text, not line number) and keep only one copy of any other repeated text."""
import csv
import html
import re
from pathlib import Path

ROOT = Path(__file__).parent
BASELINE = ROOT / "OriginData" / "humanBaseline.txt"
DATA = ROOT / "data.csv"


def normalize(text):
    """Make texts comparable despite encoding damage, HTML escapes and whitespace."""
    text = html.unescape(text).replace("\\", "")
    text = re.sub(r"[^\x00-\x7f]", "", text)  # drops £ and the broken � characters
    return " ".join(text.split()).lower()


def load_baseline():
    texts = []
    with open(BASELINE, encoding="utf-8") as f:
        for line in f:
            match = re.match(r"Line \d+: (.*)", line.strip())
            if match:
                # Emails were written with " / " for line breaks and still contain the Keywords line
                text = re.sub(r" / Keywords:.*?(?= / |$)", "", match.group(1))
                texts.append(normalize(text.replace(" / ", " ")))
    return texts


def main():
    baseline = load_baseline()
    removed = {text: 0 for text in baseline}  # how many copies of each baseline text were removed

    with open(DATA, encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    kept = []
    seen = set()
    duplicates = 0
    for row in rows:
        text = normalize(row[1])
        if text in removed:
            removed[text] += 1
        elif text in seen:
            duplicates += 1
        else:
            seen.add(text)
            kept.append(row)

    with open(DATA, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(kept)

    absent = sum(1 for count in removed.values() if count == 0)
    print(f"Removed {sum(removed.values())} copies of {len(baseline)} baseline texts "
          f"({absent} were already absent from {DATA.name})")
    print(f"Removed {duplicates} duplicate rows")
    print(f"{len(kept)} rows left in {DATA.name}")


if __name__ == "__main__":
    main()
