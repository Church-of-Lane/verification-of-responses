import argparse
import json
import os 
import numpy as np
import keras
from keras import layers
import csv
from pathlib import Path
import numpy as np
import keras
from keras import layers
import matplotlib.pyplot as plt

#matplotlib

TRAIN_DATA = Path(__file__).parent.parent.parent / "data" / "train.csv"
VAL_DATA = Path(__file__).parent.parent.parent / "data" / "val.csv"
TEST_DATA = Path(__file__).parent.parent.parent / "data" / "ground_truth.csv"
MODEL_DIR = Path(__file__).parent / "model_v1"
VOCAB_SIZE = 20000
SEQ_LEN = 128
EMBED_DIM  = 128
NUM_CLASSES = 1

# logging.set_verbosity_error()
# tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
# def train():
#     keras_hub.models.TextClassifier.from_preset("distil_bert_base_en_uncased", num_classes=2)
def load_data(pathi):
    texts, labels = [], []
    with open(pathi, 'r',newline = "",encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            label, text = row
            labels.append(1 if label == "spam" else 0)
            texts.append(text)
    vectorizer = layers.TextVectorization(
        max_tokens=VOCAB_SIZE,
        output_sequence_length=SEQ_LEN,
        standardize = None,

    )
    vectorizer.adapt(texts)
    return vectorizer, labels, texts

def save_history(history, out_dir):
    hist = {k: [float(x) for x in v] for k, v in history.history.items()}
    with open(out_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(hist, f, indent=2)
    with open(out_dir / "history.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch"] + list(hist))
        for i in range(len(hist["loss"])):
            writer.writerow([i + 1] + [hist[k][i] for k in hist])

def plot_history(history, out_dir):
    epochs = range(1, len(history.history["loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    ax1.plot(epochs, history.history["loss"], label="train")
    ax1.plot(epochs, history.history["val_loss"], label="val")
    ax1.set_title("Loss")
    ax1.set_xlabel("Epoch")
    ax1.legend()

    ax2.plot(epochs, history.history["accuracy"], label="train")
    ax2.plot(epochs, history.history["val_accuracy"], label="val")
    ax2.set_title("Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.legend()

    fig.tight_layout()
    fig.savefig(out_dir / "training_curves.png", dpi=150)
    plt.show()


def set_model():
    inputs = keras.Input(shape=(SEQ_LEN,), dtype="int64")
    h = layers.Embedding(input_dim = VOCAB_SIZE, output_dim=EMBED_DIM, mask_zero=True)(inputs)
    h = layers.Bidirectional(layers.LSTM(64))(h)
    h = layers.Dropout(0.3)(h)
    outputs = layers.Dense(NUM_CLASSES, activation="sigmoid")(h)
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model
if __name__ == "__main__":
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    vectorizer, labels, texts = load_data(TRAIN_DATA)
    train_text = vectorizer(texts)
    v, val_labels, val_text = load_data(VAL_DATA)
    val_text = vectorizer(val_text)
    model = set_model()
    history = model.fit(
        train_text, np.array(labels),
        validation_data=(val_text, np.array(val_labels)),
        batch_size=64,
        epochs=16,
        callbacks=[keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True)],
    )

    # everything from this run goes into model_v1/
    model.save(MODEL_DIR / "spam_lstm.keras")
    with open(MODEL_DIR / "vocab.json", "w", encoding="utf-8") as f:
        json.dump(vectorizer.get_vocabulary(), f)  # needed to vectorize new text the same way
    with open(MODEL_DIR / "config.json", "w", encoding="utf-8") as f:
        json.dump({"VOCAB_SIZE": VOCAB_SIZE, "SEQ_LEN": SEQ_LEN, "EMBED_DIM": EMBED_DIM,
                   "batch_size": 64, "max_epochs": 16, "epochs_run": len(history.history["loss"])}, f, indent=2)
    save_history(history, MODEL_DIR)

    _, test_labels, test_text = load_data(TEST_DATA)
    preds = (model.predict(vectorizer(test_text)) > 0.5).astype(int).ravel()
    test_acc = float((preds == np.array(test_labels)).mean())
    print("Test accuracy:", test_acc)
    with open(MODEL_DIR / "test_results.json", "w", encoding="utf-8") as f:
        json.dump({"test_accuracy": test_acc, "n_test": len(test_labels)}, f, indent=2)

    plot_history(history, MODEL_DIR)

    