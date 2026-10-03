"""Train and compare MobileNetV2 transfer learning with the original IDRiD CNN."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image, UnidentifiedImageError
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_CSV = PROJECT_DIR / "dataset" / "idrid_labels.csv"
DEFAULT_IMAGES = PROJECT_DIR / "dataset" / "images"
MODEL_PATH = PROJECT_DIR / "models" / "dr_model.keras"
OUTPUT_DIR = PROJECT_DIR / "outputs"
BASELINE_MODEL_PATH = PROJECT_DIR / "models" / "dr_model_cnn_baseline.keras"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
CLASS_NAMES = [
    "No Diabetic Retinopathy",
    "Mild",
    "Moderate",
    "Severe",
    "Proliferative Diabetic Retinopathy",
]


class DatasetError(Exception):
    """An expected, actionable problem with the supplied dataset."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate a CNN using the IDRiD diagnosis labels."
    )
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="Labels CSV path.")
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=DEFAULT_IMAGES,
        help="Root folder recursively searched for JPG, JPEG, and PNG images.",
    )
    parser.add_argument("--image-size", type=int, default=224, help="Square image size.")
    parser.add_argument("--batch-size", type=int, default=16, help="Training batch size.")
    parser.add_argument(
        "--epochs",
        type=int,
        default=12,
        help="Maximum total epochs across frozen-backbone and fine-tuning stages (1-20).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def load_dataset(
    csv_path: Path, images_dir: Path
) -> tuple[list[str], np.ndarray]:
    if not csv_path.is_file():
        raise DatasetError(
            f"Labels CSV was not found: {csv_path}. Create it with id_code and "
            "diagnosis columns."
        )
    if not images_dir.is_dir():
        raise DatasetError(
            f"Image folder was not found: {images_dir}. Put the retinal images "
            "in this folder or a nested folder beneath it."
        )

    try:
        labels_df = pd.read_csv(csv_path, dtype={"id_code": "string"})
    except (
        OSError,
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
        UnicodeDecodeError,
    ) as error:
        raise DatasetError(f"Could not read labels CSV '{csv_path}': {error}") from error

    required_columns = {"id_code", "diagnosis"}
    missing_columns = required_columns.difference(labels_df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise DatasetError(
            f"CSV is missing required column(s): {missing}. Expected id_code and "
            "diagnosis; the optional macular-edema column is not used for labels."
        )
    if labels_df.empty:
        raise DatasetError("The labels CSV contains no data rows.")

    ids = labels_df["id_code"].fillna("").astype(str).str.strip()
    numeric_labels = pd.to_numeric(labels_df["diagnosis"], errors="coerce")
    valid_labels = numeric_labels.notna() & (numeric_labels % 1 == 0)
    valid_labels &= numeric_labels.between(0, len(CLASS_NAMES) - 1)
    valid_ids = ids.ne("")
    invalid_rows = ~(valid_labels & valid_ids)
    if invalid_rows.any():
        row_numbers = (np.flatnonzero(invalid_rows.to_numpy()) + 2).tolist()
        raise DatasetError(
            "Invalid or blank id_code/diagnosis value(s) in CSV row(s) "
            f"{row_numbers[:20]}. diagnosis must be an integer from 0 to 4; "
            "correct these rows rather than assigning a guessed grade."
        )

    normalized_ids = ids.map(lambda value: Path(value).stem)
    duplicates = normalized_ids[normalized_ids.duplicated(keep=False)].unique().tolist()
    if duplicates:
        raise DatasetError(
            "The CSV contains duplicate image IDs (after removing extensions): "
            f"{duplicates[:20]}. Keep one diagnosis row per image."
        )

    image_paths = [
        path
        for path in images_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    image_index: dict[str, Path] = {}
    duplicate_image_ids: set[str] = set()
    for path in image_paths:
        image_id = path.stem
        if image_id in image_index:
            duplicate_image_ids.add(image_id)
        else:
            image_index[image_id] = path
    if duplicate_image_ids:
        raise DatasetError(
            "Multiple image files have the same filename stem, so their labels "
            f"are ambiguous: {sorted(duplicate_image_ids)[:20]}."
        )

    usable_paths: list[str] = []
    usable_labels: list[int] = []
    unmatched_ids: list[str] = []
    corrupted_ids: list[str] = []
    for image_id, label in zip(normalized_ids, numeric_labels.astype(int)):
        image_path = image_index.get(image_id)
        if image_path is None:
            unmatched_ids.append(image_id)
            continue
        try:
            with Image.open(image_path) as image:
                image.verify()
        except (
            UnidentifiedImageError,
            OSError,
            ValueError,
            SyntaxError,
            EOFError,
            Image.DecompressionBombError,
        ):
            corrupted_ids.append(image_id)
            continue
        usable_paths.append(str(image_path))
        usable_labels.append(int(label))

    label_ids = set(normalized_ids)
    unlabelled_images = sorted(set(image_index).difference(label_ids))
    print(f"Supported image files found: {len(image_paths)}")
    print(f"Matched, readable images: {len(usable_paths)}")
    print(
        "Unmatched CSV image IDs (including unreadable image files): "
        f"{len(unmatched_ids) + len(corrupted_ids)}"
    )
    print(f"Images without a CSV diagnosis: {len(unlabelled_images)}")
    if unmatched_ids:
        print(f"  Missing image files for IDs: {unmatched_ids[:20]}")
    if corrupted_ids:
        print(f"  Skipped unreadable/corrupted images: {corrupted_ids[:20]}")
    if unlabelled_images:
        print(f"  Ignored images without labels: {unlabelled_images[:20]}")
    if not usable_paths:
        raise DatasetError(
            "No readable image files matched the CSV IDs. Check filenames, "
            "extensions, and the images folder."
        )

    class_counts = np.bincount(usable_labels, minlength=len(CLASS_NAMES))
    print("Readable matched images by diagnosis:")
    for label, name in enumerate(CLASS_NAMES):
        print(f"  {label} - {name}: {class_counts[label]}")
    return usable_paths, np.asarray(usable_labels, dtype=np.int32)


def make_splits(
    labels: np.ndarray, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create non-overlapping 60/20/20 train/validation/test index sets."""
    if len(labels) < 5:
        raise DatasetError(
            f"Only {len(labels)} readable images matched. At least 5 are needed "
            "to create separate training, validation, and test sets."
        )

    indices = np.arange(len(labels))
    present_classes = np.unique(labels)
    test_count = max(1, math.ceil(len(labels) * 0.2))
    test_stratify = labels if (
        np.bincount(labels, minlength=len(CLASS_NAMES))[present_classes].min() >= 2
        and test_count >= len(present_classes)
        and len(labels) - test_count >= len(present_classes)
    ) else None
    train_validation, test = train_test_split(
        indices, test_size=test_count, random_state=seed, stratify=test_stratify
    )

    validation_count = max(1, math.ceil(len(train_validation) * 0.25))
    train_labels = labels[train_validation]
    train_classes = np.unique(train_labels)
    validation_stratify = train_labels if (
        np.bincount(train_labels, minlength=len(CLASS_NAMES))[train_classes].min() >= 2
        and validation_count >= len(train_classes)
        and len(train_validation) - validation_count >= len(train_classes)
    ) else None
    train, validation = train_test_split(
        train_validation,
        test_size=validation_count,
        random_state=seed,
        stratify=validation_stratify,
    )
    if test_stratify is None or validation_stratify is None:
        print(
            "Note: stratification was not possible for every split because one or "
            "more grades have too few examples. The data is still kept separate."
        )
    train_set, validation_set, test_set = map(
        set, (train.tolist(), validation.tolist(), test.tolist())
    )
    if (
        train_set & validation_set
        or train_set & test_set
        or validation_set & test_set
    ):
        raise DatasetError("A data-leakage check found overlapping split indices.")
    print("Split class counts:")
    for name, split_indices in (
        ("  training", train),
        ("  validation", validation),
        ("  test", test),
    ):
        counts = np.bincount(labels[split_indices], minlength=len(CLASS_NAMES))
        print(f"{name}: {counts.tolist()}")
    print("Split overlap check: passed (all image indices are disjoint).")
    return train, validation, test


def make_dataset(
    paths: list[str],
    labels: np.ndarray,
    indices: np.ndarray,
    image_size: int,
    batch_size: int,
    training: bool,
    seed: int,
) -> tf.data.Dataset:
    selected_paths = [paths[index] for index in indices]
    selected_labels = labels[indices]
    dataset = tf.data.Dataset.from_tensor_slices((selected_paths, selected_labels))

    def load_and_preprocess(path: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        image_bytes = tf.io.read_file(path)
        image = tf.io.decode_image(
            image_bytes, channels=3, expand_animations=False
        )
        image.set_shape([None, None, 3])
        image = tf.image.resize(image, [image_size, image_size])
        image = tf.cast(image, tf.float32) / 255.0
        return image, tf.cast(label, tf.int32)

    if training:
        dataset = dataset.shuffle(
            buffer_size=len(selected_paths), seed=seed, reshuffle_each_iteration=True
        )
    dataset = dataset.map(
        load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE
    )
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def build_model(image_size: int) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(image_size, image_size, 3), name="fundus_image")
    augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.04),
            tf.keras.layers.RandomZoom(0.08),
        ],
        name="training_augmentation",
    )
    x = augmentation(inputs)
    x = tf.keras.layers.Rescaling(
        scale=2.0, offset=-1.0, name="mobilenetv2_rescaling"
    )(x)
    backbone = tf.keras.applications.MobileNetV2(
        input_shape=(image_size, image_size, 3),
        include_top=False,
        weights="imagenet",
        name="mobilenetv2_backbone",
    )
    backbone.trainable = False
    x = backbone(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.35)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    outputs = tf.keras.layers.Dense(len(CLASS_NAMES), activation="softmax", name="grade")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="idrid_mobilenetv2")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.get_layer(backbone.name).trainable = False
    return model


def save_training_plots(history: dict[str, list[float]]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history["accuracy"]) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["accuracy"], label="Training accuracy")
    plt.plot(epochs, history["val_accuracy"], label="Validation accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and validation accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "accuracy.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["loss"], label="Training loss")
    plt.plot(epochs, history["val_loss"], label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and validation loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "loss.png", dpi=150)
    plt.close()


def save_confusion_matrix(
    actual: np.ndarray,
    predicted: np.ndarray,
    output_path: Path,
    title: str,
) -> None:
    matrix = confusion_matrix(actual, predicted, labels=np.arange(len(CLASS_NAMES)))
    figure, axis = plt.subplots(figsize=(9, 7))
    display = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
    figure.colorbar(display, ax=axis)
    axis.set(
        xticks=np.arange(len(CLASS_NAMES)),
        yticks=np.arange(len(CLASS_NAMES)),
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        xlabel="Predicted grade",
        ylabel="True grade",
        title=title,
    )
    plt.setp(axis.get_xticklabels(), rotation=35, ha="right", rotation_mode="anchor")
    threshold = matrix.max() / 2 if matrix.size else 0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
                color="white" if matrix[row, column] > threshold else "black",
            )
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def calculate_metrics(
    actual: np.ndarray, probabilities: np.ndarray
) -> tuple[dict[str, float], np.ndarray, str]:
    predicted = np.argmax(probabilities, axis=1)
    report = classification_report(
        actual,
        predicted,
        labels=np.arange(len(CLASS_NAMES)),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    text_report = classification_report(
        actual,
        predicted,
        labels=np.arange(len(CLASS_NAMES)),
        target_names=CLASS_NAMES,
        zero_division=0,
    )
    metrics = {
        "accuracy": float(accuracy_score(actual, predicted)),
        "macro_precision": float(report["macro avg"]["precision"]),
        "macro_recall": float(report["macro avg"]["recall"]),
        "macro_f1": float(report["macro avg"]["f1-score"]),
    }
    return metrics, predicted, text_report


def training_callbacks(checkpoint_path: Path) -> list[tf.keras.callbacks.Callback]:
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.3, patience=2, min_lr=1e-7, verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
    ]


def combine_histories(
    histories: list[tf.keras.callbacks.History],
) -> dict[str, list[float]]:
    combined = {"accuracy": [], "val_accuracy": [], "loss": [], "val_loss": []}
    for history in histories:
        for key in combined:
            combined[key].extend(history.history.get(key, []))
    return combined


def train(args: argparse.Namespace) -> None:
    if args.image_size < 32:
        raise DatasetError("--image-size must be at least 32.")
    if args.batch_size < 1:
        raise DatasetError("--batch-size must be at least 1.")
    if args.epochs < 1 or args.epochs > 20:
        raise DatasetError("--epochs must be between 1 and 20.")

    tf.keras.utils.set_random_seed(args.seed)
    paths, labels = load_dataset(args.csv, args.images_dir)
    train_indices, validation_indices, test_indices = make_splits(labels, args.seed)
    print(
        f"Split sizes - training: {len(train_indices)}, validation: "
        f"{len(validation_indices)}, test: {len(test_indices)}"
    )
    train_data = make_dataset(
        paths, labels, train_indices, args.image_size, args.batch_size, True, args.seed
    )
    validation_data = make_dataset(
        paths, labels, validation_indices, args.image_size, args.batch_size, False, args.seed
    )
    test_data = make_dataset(
        paths, labels, test_indices, args.image_size, args.batch_size, False, args.seed
    )

    training_classes = np.unique(labels[train_indices])
    weights = compute_class_weight(
        class_weight="balanced", classes=training_classes, y=labels[train_indices]
    )
    class_weights = {
        int(class_id): float(weight)
        for class_id, weight in zip(training_classes, weights)
    }
    print(f"Training class weights: {class_weights}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not MODEL_PATH.is_file() and not BASELINE_MODEL_PATH.is_file():
        raise DatasetError(
            "Cannot compare with the original CNN: models/dr_model.keras is missing. "
            "Restore the original model checkpoint before starting this transfer-learning run."
        )
    if not BASELINE_MODEL_PATH.is_file():
        shutil.copy2(MODEL_PATH, BASELINE_MODEL_PATH)
        print(f"Preserved original CNN baseline at {BASELINE_MODEL_PATH}")
    baseline_model = tf.keras.models.load_model(BASELINE_MODEL_PATH)
    baseline_input_shape = baseline_model.input_shape
    if isinstance(baseline_input_shape, list):
        baseline_input_shape = baseline_input_shape[0]
    baseline_size = int(baseline_input_shape[1])
    baseline_test_data = make_dataset(
        paths, labels, test_indices, baseline_size, args.batch_size, False, args.seed
    )
    print("Evaluating original CNN baseline on the fixed held-out test split.")
    baseline_probabilities = baseline_model.predict(baseline_test_data, verbose=0)

    actual = labels[test_indices]
    baseline_metrics, baseline_predictions, baseline_report = calculate_metrics(
        actual, baseline_probabilities
    )
    print(
        "Original CNN test metrics: "
        f"accuracy={baseline_metrics['accuracy']:.4f}, "
        f"macro F1={baseline_metrics['macro_f1']:.4f}"
    )

    frozen_epochs = max(1, args.epochs * 2 // 3)
    fine_tune_epochs = args.epochs - frozen_epochs
    frozen_candidate = MODEL_PATH.parent / ".dr_model_frozen_candidate.keras"
    fine_tune_candidate = MODEL_PATH.parent / ".dr_model_finetuned_candidate.keras"
    histories: list[tf.keras.callbacks.History] = []

    print(
        f"Starting MobileNetV2 transfer learning: {frozen_epochs} frozen-backbone "
        f"epoch(s), up to {fine_tune_epochs} fine-tuning epoch(s)."
    )
    model = build_model(args.image_size)
    frozen_history = model.fit(
        train_data,
        validation_data=validation_data,
        epochs=frozen_epochs,
        class_weight=class_weights,
        callbacks=training_callbacks(frozen_candidate),
        shuffle=False,
        verbose=2,
    )
    histories.append(frozen_history)
    frozen_model = tf.keras.models.load_model(frozen_candidate)
    frozen_validation_loss = float(frozen_model.evaluate(validation_data, verbose=0)[0])
    selected_model = frozen_model
    selected_validation_loss = frozen_validation_loss

    if fine_tune_epochs > 0:
        fine_tune_model = tf.keras.models.load_model(frozen_candidate)
        backbone = fine_tune_model.get_layer("mobilenetv2_backbone")
        backbone.trainable = True
        fine_tune_from = max(0, len(backbone.layers) - 30)
        for layer in backbone.layers[:fine_tune_from]:
            layer.trainable = False
        for layer in backbone.layers[fine_tune_from:]:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False
        fine_tune_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        fine_history = fine_tune_model.fit(
            train_data,
            validation_data=validation_data,
            initial_epoch=len(frozen_history.epoch),
            epochs=len(frozen_history.epoch) + fine_tune_epochs,
            class_weight=class_weights,
            callbacks=training_callbacks(fine_tune_candidate),
            shuffle=False,
            verbose=2,
        )
        histories.append(fine_history)
        fine_model = tf.keras.models.load_model(fine_tune_candidate)
        fine_validation_loss = float(fine_model.evaluate(validation_data, verbose=0)[0])
        print(
            f"Validation loss - frozen checkpoint: {frozen_validation_loss:.4f}; "
            f"fine-tuned checkpoint: {fine_validation_loss:.4f}"
        )
        if fine_validation_loss < selected_validation_loss:
            selected_model = fine_model
            selected_validation_loss = fine_validation_loss
        del fine_tune_model, fine_model

    # Final test evaluation happens once, after checkpoint selection using validation only.
    probabilities = selected_model.predict(test_data, verbose=0)
    model_metrics, predicted, model_report = calculate_metrics(actual, probabilities)
    _, baseline_predictions, baseline_report = calculate_metrics(
        actual, baseline_probabilities
    )
    selected_model.save(MODEL_PATH)

    history = combine_histories(histories)
    comparison = {
        "evaluation": "same fixed held-out test indices; no test-based model selection",
        "test_image_count": int(len(test_indices)),
        "test_image_ids": [Path(paths[index]).stem for index in test_indices],
        "seed": int(args.seed),
        "test_indices": test_indices.tolist(),
        "class_names": CLASS_NAMES,
        "original_cnn": baseline_metrics,
        "mobilenetv2_transfer_learning": model_metrics,
        "best_checkpoint_validation_loss": selected_validation_loss,
    }
    metrics = {
        "test_accuracy": model_metrics["accuracy"],
        "test_loss": float(
            tf.keras.losses.sparse_categorical_crossentropy(
                actual, probabilities
            ).numpy().mean()
        ),
        "macro_precision": model_metrics["macro_precision"],
        "macro_recall": model_metrics["macro_recall"],
        "macro_f1": model_metrics["macro_f1"],
        "test_image_count": int(len(test_indices)),
        "image_size": int(args.image_size),
        "class_names": CLASS_NAMES,
        "model_architecture": "MobileNetV2 pretrained on ImageNet with fine-tuning",
        "normalization": "input images are scaled to [0, 1]; model rescales to [-1, 1]",
        "comparison_file": "model_comparison.json",
        "split_seed": int(args.seed),
    }
    save_training_plots(history)
    save_confusion_matrix(
        actual,
        predicted,
        OUTPUT_DIR / "confusion_matrix.png",
        "MobileNetV2 held-out test confusion matrix",
    )
    save_confusion_matrix(
        actual,
        baseline_predictions,
        OUTPUT_DIR / "baseline_confusion_matrix.png",
        "Original CNN held-out test confusion matrix",
    )
    (OUTPUT_DIR / "model_comparison.json").write_text(
        json.dumps(comparison, indent=2), encoding="utf-8"
    )
    (OUTPUT_DIR / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    (OUTPUT_DIR / "classification_report.txt").write_text(
        model_report, encoding="utf-8"
    )
    (OUTPUT_DIR / "baseline_classification_report.txt").write_text(
        baseline_report, encoding="utf-8"
    )
    for candidate in (frozen_candidate, fine_tune_candidate):
        if candidate.exists():
            candidate.unlink()

    print("\nSame held-out test split comparison (no test-set tuning):")
    for model_name, result in (
        ("Original CNN", baseline_metrics),
        ("MobileNetV2 transfer learning", model_metrics),
    ):
        print(
            f"  {model_name}: accuracy={result['accuracy']:.4f}, "
            f"macro precision={result['macro_precision']:.4f}, "
            f"macro recall={result['macro_recall']:.4f}, "
            f"macro F1={result['macro_f1']:.4f}"
        )
    print(f"\nClassification report saved to {OUTPUT_DIR / 'classification_report.txt'}")
    print(f"Model saved to {MODEL_PATH}")
    print(
        "Results are educational estimates from one small held-out split. IDRiD "
        "class counts are limited and imbalanced; metrics are not clinical validation."
    )


def main() -> int:
    args = parse_args()
    try:
        train(args)
    except DatasetError as error:
        print(f"Dataset/configuration error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
