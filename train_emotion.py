"""Train and validate the emotion model from an image-folder dataset.

Expected folders: emotion_data/train/<emotion>/*.jpg and
emotion_data/test/<emotion>/*.jpg. Download FER-2013 from Kaggle first and
arrange/export its images into this structure.
"""
import argparse
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers, models

IMAGE_SIZE = (48, 48)
BATCH_SIZE = 64
SEED = 42


def load_data(root: Path):
    train = tf.keras.utils.image_dataset_from_directory(
        root / "train", color_mode="grayscale", image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE, seed=SEED, validation_split=0.15,
        subset="training", label_mode="categorical"
    )
    validation = tf.keras.utils.image_dataset_from_directory(
        root / "train", color_mode="grayscale", image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE, seed=SEED, validation_split=0.15,
        subset="validation", label_mode="categorical"
    )
    test = tf.keras.utils.image_dataset_from_directory(
        root / "test", color_mode="grayscale", image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE, shuffle=False, label_mode="categorical"
    )
    return train.prefetch(tf.data.AUTOTUNE), validation.prefetch(
        tf.data.AUTOTUNE
    ), test.prefetch(tf.data.AUTOTUNE), train.class_names


def build_model(class_count):
    return models.Sequential([
        layers.Input(shape=(*IMAGE_SIZE, 1)),
        layers.Rescaling(1.0 / 255),
        layers.RandomFlip("horizontal"),
        layers.Conv2D(32, 3, activation="relu"), layers.BatchNormalization(),
        layers.MaxPooling2D(), layers.Dropout(0.25),
        layers.Conv2D(64, 3, activation="relu"), layers.BatchNormalization(),
        layers.MaxPooling2D(), layers.Dropout(0.25),
        layers.Conv2D(128, 3, activation="relu"), layers.BatchNormalization(),
        layers.MaxPooling2D(), layers.Dropout(0.25),
        layers.Flatten(), layers.Dense(128, activation="relu"),
        layers.Dropout(0.4), layers.Dense(class_count, activation="softmax")
    ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("emotion_data"))
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()
    train, validation, test, names = load_data(args.data)
    model = build_model(len(names))
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    callbacks = [tf.keras.callbacks.EarlyStopping(
        patience=6, restore_best_weights=True, monitor="val_accuracy"
    ), tf.keras.callbacks.ModelCheckpoint(
        "face_model.h5", save_best_only=True, monitor="val_accuracy"
    )]
    model.fit(train, validation_data=validation, epochs=args.epochs,
              callbacks=callbacks)
    loss, accuracy = model.evaluate(test, verbose=1)
    print({"classes": names, "test_loss": float(loss),
           "test_accuracy": float(accuracy)})


if __name__ == "__main__":
    main()
