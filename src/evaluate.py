"""Evaluate a trained model on the held-out test set.

Writes a classification report, metrics JSON, confusion matrix, ROC and
precision-recall curves to `reports/`.

Usage:
    python -m src.evaluate
    python -m src.evaluate --model models/resnet50_pneumonia.keras --threshold 0.5
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

from src import config
from src.data import load_test_dataset
from src.model import ResNetPreprocess  # noqa: F401  (registers the custom layer for loading)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, default=config.DEFAULT_MODEL_PATH)
    parser.add_argument("--data-dir", type=Path, default=config.DATA_DIR)
    parser.add_argument("--threshold", type=float, default=config.DECISION_THRESHOLD)
    return parser.parse_args()


def main():
    args = parse_args()
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    model = tf.keras.models.load_model(args.model)
    test_ds = load_test_dataset(args.data_dir)

    y_true = np.concatenate([y.numpy() for _, y in test_ds]).astype(int).ravel()
    y_prob = model.predict(test_ds).ravel()
    y_pred = (y_prob >= args.threshold).astype(int)

    report = classification_report(y_true, y_pred, target_names=config.CLASS_NAMES, digits=3)
    cm = confusion_matrix(y_true, y_pred)
    metrics = {
        "threshold": args.threshold,
        "accuracy": accuracy_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "average_precision": average_precision_score(y_true, y_prob),
        "confusion_matrix": cm.tolist(),
        "report": classification_report(
            y_true, y_pred, target_names=config.CLASS_NAMES, output_dict=True
        ),
    }

    print(report)
    print("Confusion matrix (rows = true, cols = predicted):")
    print(cm)
    print(f"ROC AUC: {metrics['roc_auc']:.4f}  |  Average precision: {metrics['average_precision']:.4f}")

    (config.REPORTS_DIR / "classification_report.txt").write_text(report)
    (config.REPORTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))

    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay(cm, display_labels=config.CLASS_NAMES).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion matrix (threshold {args.threshold})")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "confusion_matrix.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 5))
    RocCurveDisplay.from_predictions(y_true, y_prob, name="ResNet50", ax=ax)
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1)
    ax.set_title("ROC curve")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "roc_curve.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 5))
    PrecisionRecallDisplay.from_predictions(y_true, y_prob, name="ResNet50", ax=ax)
    ax.set_title("Precision-recall curve")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "pr_curve.png", dpi=120)
    plt.close(fig)

    print(f"Reports written to {config.REPORTS_DIR}")


if __name__ == "__main__":
    main()
