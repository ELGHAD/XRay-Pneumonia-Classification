"""Central configuration for data paths and training hyperparameters."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "chest_xray"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

DEFAULT_MODEL_PATH = MODELS_DIR / "resnet50_pneumonia.keras"

# Class folders are read in alphabetical order, so NORMAL -> 0 and PNEUMONIA -> 1.
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42

# The official Kaggle validation folder holds only 16 images, which is too small
# to monitor training. A random, seeded slice of the training set is used instead.
VALIDATION_SPLIT = 0.15

# Phase 1: train the classification head on top of a frozen backbone.
HEAD_EPOCHS = 15
HEAD_LEARNING_RATE = 1e-4

# Phase 2: unfreeze the last ResNet50 blocks and fine-tune with a small learning rate.
FINE_TUNE_EPOCHS = 10
FINE_TUNE_LEARNING_RATE = 1e-5
FINE_TUNE_FROM_LAYER = 140  # layers before this index stay frozen

DECISION_THRESHOLD = 0.5
