"""Predict pneumonia probability for one or more chest X-ray images.

Usage:
    python -m src.predict path/to/xray.jpeg [more images ...]
    python -m src.predict images/*.jpeg --threshold 0.6
"""

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf

from src import config
from src.model import ResNetPreprocess  # noqa: F401  (registers the custom layer for loading)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("images", type=Path, nargs="+")
    parser.add_argument("--model", type=Path, default=config.DEFAULT_MODEL_PATH)
    parser.add_argument("--threshold", type=float, default=config.DECISION_THRESHOLD)
    return parser.parse_args()


def load_image(path):
    # Grayscale X-rays are expanded to 3 channels; pixels stay in [0, 255]
    # because ResNet preprocessing is part of the saved model.
    img = tf.keras.utils.load_img(path, color_mode="rgb", target_size=config.IMAGE_SIZE)
    return tf.keras.utils.img_to_array(img)


def main():
    args = parse_args()
    model = tf.keras.models.load_model(args.model)

    batch = np.stack([load_image(p) for p in args.images])
    probs = model.predict(batch, verbose=0).ravel()

    width = max(len(str(p)) for p in args.images)
    for path, prob in zip(args.images, probs):
        label = config.CLASS_NAMES[int(prob >= args.threshold)]
        print(f"{str(path):<{width}}  {label:<9}  p(pneumonia) = {prob:.3f}")


if __name__ == "__main__":
    main()
