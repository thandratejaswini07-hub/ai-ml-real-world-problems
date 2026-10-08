"""Crop disease detection: MobileNetV2 transfer-learning CNN.

Dataset layout (e.g. PlantVillage from Kaggle):
    data/plantvillage/<class_name>/*.jpg

Run:  python train.py --data ../data/plantvillage --epochs 5
"""
import argparse
import json
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers, models

IMG = 224


def build_model(num_classes: int) -> tf.keras.Model:
    base = tf.keras.applications.MobileNetV2(
        input_shape=(IMG, IMG, 3), include_top=False, weights="imagenet"
    )
    base.trainable = False  # fine-tune later if needed
    inputs = layers.Input((IMG, IMG, 3))
    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomRotation(0.1)(x)
    x = layers.Rescaling(1 / 127.5, offset=-1)(x)  # MobileNetV2 expects [-1, 1]
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return models.Model(inputs, outputs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--out", default="../outputs/crop")
    a = ap.parse_args()

    common = dict(image_size=(IMG, IMG), batch_size=a.batch, seed=42,
                  validation_split=0.2)
    train = tf.keras.utils.image_dataset_from_directory(a.data, subset="training", **common)
    val = tf.keras.utils.image_dataset_from_directory(a.data, subset="validation", **common)
    class_names = train.class_names
    print("Classes:", len(class_names))

    train = train.prefetch(tf.data.AUTOTUNE)
    val = val.prefetch(tf.data.AUTOTUNE)

    model = build_model(len(class_names))
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    model.fit(train, validation_data=val, epochs=a.epochs)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    model.save(out / "crop_model.keras")
    (out / "class_names.json").write_text(json.dumps(class_names))
    print("Saved model to", out)


if __name__ == "__main__":
    main()
