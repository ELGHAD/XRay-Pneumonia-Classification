"""Dataset loading, augmentation and class weighting."""

from pathlib import Path

import tensorflow as tf

from src import config

AUTOTUNE = tf.data.AUTOTUNE


def _load_split(directory, subset=None, shuffle=True):
    return tf.keras.utils.image_dataset_from_directory(
        directory,
        labels="inferred",
        label_mode="binary",
        class_names=config.CLASS_NAMES,
        color_mode="rgb",
        image_size=config.IMAGE_SIZE,
        batch_size=config.BATCH_SIZE,
        shuffle=shuffle,
        seed=config.SEED,
        validation_split=config.VALIDATION_SPLIT if subset else None,
        subset=subset,
    )


def load_datasets(data_dir=config.DATA_DIR):
    """Return (train, validation, test) datasets.

    Validation is carved out of the training folder. The test folder is kept
    untouched and unshuffled so predictions line up with file order.
    """
    data_dir = Path(data_dir)
    train_ds = _load_split(data_dir / "train", subset="training")
    val_ds = _load_split(data_dir / "train", subset="validation")
    test_ds = _load_split(data_dir / "test", shuffle=False)

    train_ds = train_ds.prefetch(AUTOTUNE)
    val_ds = val_ds.cache().prefetch(AUTOTUNE)
    test_ds = test_ds.cache().prefetch(AUTOTUNE)
    return train_ds, val_ds, test_ds


def load_test_dataset(data_dir=config.DATA_DIR):
    return _load_split(Path(data_dir) / "test", shuffle=False).prefetch(AUTOTUNE)


def build_augmentation():
    """Light geometric augmentation, roughly matching the original notebook."""
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomRotation(0.05),
            tf.keras.layers.RandomTranslation(0.1, 0.1),
            tf.keras.layers.RandomZoom(0.2),
            tf.keras.layers.RandomContrast(0.1),
        ],
        name="augmentation",
    )


def compute_class_weights(data_dir=config.DATA_DIR):
    """Inverse-frequency weights so the minority NORMAL class is not ignored.

    The training folder holds about 1,341 NORMAL and 3,875 PNEUMONIA images.
    """
    train_dir = Path(data_dir) / "train"
    counts = [
        sum(1 for p in (train_dir / name).iterdir() if p.is_file())
        for name in config.CLASS_NAMES
    ]
    total = sum(counts)
    return {i: total / (len(counts) * n) for i, n in enumerate(counts)}
