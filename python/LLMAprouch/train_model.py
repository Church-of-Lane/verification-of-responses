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
MODEL_DIR = Path(__file__).parent / "model_v2"
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
        metrics=["accuracy", "AUC", "Precision", "Recall", 
                 keras.metrics.F1Score(average="micro", threshold=0.5, name="f1")],
    )
    return model

def save_history(history, out_dir):
    hist = {k: [float(x) for x in v] for k, v in history.history.items()}
    with open(out_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(hist, f, indent=2)
    with open(out_dir / "history.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch"] + list(hist))
        for i in range(len(hist["loss"])):
            writer.writerow([i + 1] + [hist[k][i] for k in hist])

class TenthEpochLogger(keras.callbacks.Callback):
    # records train and val metrics 10 times per epoch
    def __init__(self, val_x, val_y, steps_per_epoch, batch_size):
        super().__init__()
        self.val_x = val_x
        self.val_y = np.array(val_y)
        self.steps = steps_per_epoch
        self.batch_size = batch_size
        self.points = {"epoch": [], "loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch
        self.tenths_done = 0
        self.prev_batches = 0
        self.prev_loss = 0.0
        self.prev_acc = 0.0

    def on_train_batch_end(self, batch, logs=None):
        batches = batch + 1
        tenths = batches * 10 // self.steps
        if tenths == self.tenths_done:
            return
        self.tenths_done = tenths

        # keras logs a running mean for the epoch, so undo it to get this interval only
        count = batches - self.prev_batches
        loss = (float(logs["loss"]) * batches - self.prev_loss * self.prev_batches) / count
        acc = (float(logs["accuracy"]) * batches - self.prev_acc * self.prev_batches) / count
        self.prev_batches = batches
        self.prev_loss = float(logs["loss"])
        self.prev_acc = float(logs["accuracy"])

        # predict, not evaluate: evaluate would reset the running train metrics
        probs = self.model.predict(self.val_x, batch_size=self.batch_size, verbose=0).ravel()
        probs = np.clip(probs, 1e-7, 1 - 1e-7)
        val_loss = float(-np.mean(self.val_y * np.log(probs) + (1 - self.val_y) * np.log(1 - probs)))
        val_acc = float(((probs > 0.5).astype(int) == self.val_y).mean())

        self.points["epoch"].append(round(self.epoch + tenths / 10, 1))
        self.points["loss"].append(loss)
        self.points["accuracy"].append(acc)
        self.points["val_loss"].append(val_loss)
        self.points["val_accuracy"].append(val_acc)

def save_tenth_history(points, out_dir):
    with open(out_dir / "history_tenth_epoch.json", "w", encoding="utf-8") as f:
        json.dump(points, f, indent=2)
    with open(out_dir / "history_tenth_epoch.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(list(points))
        writer.writerows(zip(*points.values()))

def plot_history(points, out_dir):
    epochs = points["epoch"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    ax1.plot(epochs, points["loss"], label="train")
    ax1.plot(epochs, points["val_loss"], label="val")
    ax1.set_title("Loss")
    ax1.set_xlabel("Epoch")
    ax1.legend()

    ax2.plot(epochs, points["accuracy"], label="train")
    ax2.plot(epochs, points["val_accuracy"], label="val")
    ax2.set_title("Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.legend()

    fig.tight_layout()
    fig.savefig(out_dir / "training_curves.png", dpi=150)
    plt.show()



if __name__ == "__main__":
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    vectorizer, labels, texts = load_data(TRAIN_DATA)
    train_text = vectorizer(texts)
    v, val_labels, val_text = load_data(VAL_DATA)
    val_text = vectorizer(val_text)
    model = set_model()
    tenth_logger = TenthEpochLogger(val_text, val_labels, int(np.ceil(len(labels) / 64)), 64)
    history = model.fit(
        train_text,  np.array(labels, dtype="float32").reshape(-1, 1),
        validation_data=(val_text, np.array(val_labels, dtype="float32").reshape(-1, 1)),
        batch_size=64,
        epochs=3,
        callbacks=[tenth_logger, keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True)],
    )

    # everything from this run goes into model_v2/
    model.save(MODEL_DIR / "spam_lstm.keras")
    with open(MODEL_DIR / "vocab.json", "w", encoding="utf-8") as f:
        json.dump(vectorizer.get_vocabulary(), f)  # needed to vectorize new text the same way
    with open(MODEL_DIR / "config.json", "w", encoding="utf-8") as f:
        json.dump({"VOCAB_SIZE": VOCAB_SIZE, "SEQ_LEN": SEQ_LEN, "EMBED_DIM": EMBED_DIM,
                   "batch_size": 64, "max_epochs": 3, "epochs_run": len(history.history["loss"])}, f, indent=2)
    save_history(history, MODEL_DIR)
    save_tenth_history(tenth_logger.points, MODEL_DIR)

    _, test_labels, test_text = load_data(TEST_DATA)
    preds = (model.predict(vectorizer(test_text)) > 0.5).astype(int).ravel()
    test_acc = float((preds == np.array(test_labels)).mean())
    print("Test accuracy:", test_acc)
    with open(MODEL_DIR / "test_results.json", "w", encoding="utf-8") as f:
        json.dump({"test_accuracy": test_acc, "n_test": len(test_labels)}, f, indent=2)

    plot_history(tenth_logger.points, MODEL_DIR)

    