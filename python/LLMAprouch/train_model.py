import keras_hub


def train():
    keras_hub.models.TextClassifier.from_preset("distil_bert_base_en_uncased", num_classes=2)





if __name__ == "__main__":
    train()
    #answerdotai/ModernBERT-base
    