import csv
from pathlib import Path
import re
import pandas as pd
from sklearn.model_selection import train_test_split

ORIGIN_DATA_PATH = Path(__file__).parent.parent / "other" / "OriginData"
MAIL_CSV_PATH = ORIGIN_DATA_PATH / "Mail" / "phishing_legit_dataset_KD_10000.csv"
TEXT_CSV_PATH = ORIGIN_DATA_PATH / "Text" / "spam.csv"
SMISHING_CSV_PATH = ORIGIN_DATA_PATH / "A Balanced Dataset for Spam and Smishing Detection" / "Dataset_10191.csv"

def merge():
    mergeText()
    mergeSmishing()
    mergeMail()
    removeDuplicates()
    splitData()
    


#utf-8
def mergeText():
    with open(TEXT_CSV_PATH, 'r',newline = "",encoding="latin-1") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            with open("data.csv", 'a',newline = "",encoding="latin-1") as d:
                writer = csv.writer(d)
                writer.writerow((row[0],row[1]))


def mergeSmishing():
    with open(SMISHING_CSV_PATH, 'r',newline = "",encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                with open("data.csv", 'a',newline = "",encoding="utf-8") as d:
                    writer = csv.writer(d)
                    label, text = row[0],row[1]
                    if label == "smishing":
                        label = "spam"
                    writer.writerow((label,text))

def mergeMail():
    with open(MAIL_CSV_PATH, 'r',newline = "",encoding="utf-8") as f:
            reader = csv.DictReader(f)
            header = next(reader)
            for row in reader:
                with open("data.csv", 'a',newline = "",encoding="utf-8") as d:
                    writer = csv.writer(d)
                    label = "spam" if row["label"] == "1" else "ham"
                    text = row["text"]
                    text = re.sub(r"^Subject:.*\n", "", text)
                    text = re.sub(r"Keywords:.*\n?", "", text)
                    text = " ".join(text.split())    
                    writer.writerow((label,text.replace("\n", "")))


def removeDuplicates():
    df = pd.read_csv("data.csv", header=None, names=["label", "text"], encoding="latin-1")
    df = df.drop_duplicates(subset=["text"])
    df["_key"] = df["text"].str.lower().str.split().str.join(" ")
    df = df.drop_duplicates(subset=["_key"]).drop(columns="_key")
    df.to_csv("data.csv", index=False, encoding="utf-8")


def splitData():
    df = pd.read_csv("data.csv")
    train_df, temp_df = train_test_split(df, test_size=0.4, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=43)
    train_df.to_csv("train.csv", index=False)
    val_df.to_csv("val.csv", index=False)
    test_df.to_csv("test.csv", index=False)
    getHumanData()


def getHumanData():
    df = pd.read_csv("train.csv")
    sample = df.sample(n=20)
    sample.to_csv("human.csv", index=False)
    df = df.drop(sample.index)
    df.to_csv("train.csv", index=False)
    sample.to_csv("test.csv", mode="a", header=False, index=False)


if __name__ == "__main__":
    #merge()
    #getHumanData()
    splitData()
    #answerdotai/ModernBERT-base
    