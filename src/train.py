"""Two-phase training: frozen-backbone head training, then fine-tuning.

Usage:
    python -m src.train
    python -m src.train --data-dir data/chest_xray --head-epochs 15 --fine-tune-epochs 10
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

from src import config
from src.data import compute_class_weights, load_datasets
from src.model import build_model, compile_model, unfreeze_top_layers


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=config.DATA_DIR)
    parser.add_argument("--output", type=Path, default=config.DEFAULT_MODEL_PATH)
    parser.add_argument("--head-epochs", type=int, default=config.HEAD_EPOCHS)
    parser.add_argument("--fine-tune-epochs", type=int, default=config.FINE_TUNE_EPOCHS)
    parser.add_argument("--no-class-weights", action="store_true", help="Disable inverse-frequency class weighting")
    return parser.parse_args()


def plot_history(history_df, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, metric in zip(axes, ["loss", "accuracy", "auc"]):
        ax.plot(history_df["epoch"], history_df[metric], label="train")
        ax.plot(history_df["epoch"], history_df[f"val_{metric}"], label="validation")
        ax.set_title(metric.capitalize())
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.3)
        ax.legend()
    fine_tune_start = history_df.loc[history_df["phase"] == "fine_tune", "epoch"].min()
    if pd.notna(fine_tune_start):
        for ax in axes:
            ax.axvline(fine_tune_start, color="grey", linestyle="--", linewidth=1)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main():
    args = parse_args()
    tf.keras.utils.set_random_seed(config.SEED)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds, _ = load_datasets(args.data_dir)
    class_weight = None if args.no_class_weights else compute_class_weights(args.data_dir)
    print(f"Class weights: {class_weight}")

    model = build_model()

    # One checkpoint instance across both phases keeps the best model overall.
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        args.output, monitor="val_auc", mode="max", save_best_only=True, verbose=1
    )

    def callbacks():
        return [
            checkpoint,
            tf.keras.callbacks.EarlyStopping(
                monitor="val_auc", mode="max", patience=5, restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1),
        ]

    print("\nPhase 1: training classification head (backbone frozen)")
    compile_model(model, config.HEAD_LEARNING_RATE)
    head = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.head_epochs,
        class_weight=class_weight,
        callbacks=callbacks(),
    )
    frames = [pd.DataFrame(head.history).assign(epoch=head.epoch, phase="head")]

    if args.fine_tune_epochs > 0:
        print(f"\nPhase 2: fine-tuning ResNet50 from layer {config.FINE_TUNE_FROM_LAYER}")
        unfreeze_top_layers(model)
        compile_model(model, config.FINE_TUNE_LEARNING_RATE)
        start = len(head.epoch)
        fine = model.fit(
            train_ds,
            validation_data=val_ds,
            initial_epoch=start,
            epochs=start + args.fine_tune_epochs,
            class_weight=class_weight,
            callbacks=callbacks(),
        )
        frames.append(pd.DataFrame(fine.history).assign(epoch=fine.epoch, phase="fine_tune"))

    history_df = pd.concat(frames, ignore_index=True)
    history_df.drop(columns=["learning_rate"], errors="ignore").to_csv(
        config.REPORTS_DIR / "training_history.csv", index=False
    )
    plot_history(history_df, config.FIGURES_DIR / "training_curves.png")
    print(f"\nBest model saved to {args.output}")


if __name__ == "__main__":
    main()
