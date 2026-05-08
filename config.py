"""
Central config for the multimodal crop disease prediction project.
All paths, hyperparameters, and class names live here so other modules
stay clean and there is one place to change things.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "plantvillage_raw"
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"
TEST_DIR = DATA_DIR / "test"
SENSOR_DIR = DATA_DIR / "sensor"
SENSOR_CSV = SENSOR_DIR / "sensor_dataset.csv"

MODELS_DIR = PROJECT_ROOT / "models"
IMAGE_MODEL_PATH = MODELS_DIR / "image" / "image_model.h5"
SENSOR_MODEL_PATH = MODELS_DIR / "sensor" / "sensor_model.h5"
SENSOR_SCALER_PATH = MODELS_DIR / "sensor" / "scaler.pkl"
FUSION_MODEL_PATH = MODELS_DIR / "fusion" / "fusion_model.h5"
TFLITE_MODEL_PATH = MODELS_DIR / "tflite" / "model.tflite"

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
RESULTS_DIR = REPORTS_DIR / "results"

GRADCAM_DIR = PROJECT_ROOT / "server" / "static" / "gradcam"

# ---------------------------------------------------------------------------
# Image model hyperparameters
# ---------------------------------------------------------------------------
IMG_SIZE = (224, 224)
IMG_CHANNELS = 3
IMG_BATCH_SIZE = 32
IMG_EPOCHS_HEAD = 10        # train classification head only
IMG_EPOCHS_FINETUNE = 10    # fine-tune top layers of EfficientNetB0
IMG_LR_HEAD = 1e-3
IMG_LR_FINETUNE = 1e-5
IMG_FINETUNE_LAYERS = 30    # number of top layers to unfreeze

# ---------------------------------------------------------------------------
# Sensor model hyperparameters
# ---------------------------------------------------------------------------
SENSOR_FEATURES = ["temperature", "humidity", "soil_moisture"]
SENSOR_BATCH_SIZE = 64
SENSOR_EPOCHS = 50
SENSOR_LR = 1e-3
SENSOR_HIDDEN = [32, 16]

# Realistic sensor ranges used by the synthetic data generator
SENSOR_RANGES = {
    "temperature": (5.0, 45.0),     # °C
    "humidity": (10.0, 100.0),      # %
    "soil_moisture": (0.0, 100.0),  # %
}

# ---------------------------------------------------------------------------
# Fusion model hyperparameters
# ---------------------------------------------------------------------------
FUSION_HIDDEN = 128
FUSION_DROPOUT = 0.3
FUSION_BATCH_SIZE = 64
FUSION_EPOCHS = 30
FUSION_LR = 1e-3

# ---------------------------------------------------------------------------
# Dataset split
# ---------------------------------------------------------------------------
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Classes
# ---------------------------------------------------------------------------
# The 38 PlantVillage classes (filled at runtime from the train folder).
# Kept here as None so other modules can import a single source of truth.
NUM_CLASSES = 38
CLASS_NAMES = None  # populated by utils.load_class_names()

# ---------------------------------------------------------------------------
# Server / UI
# ---------------------------------------------------------------------------
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5000
SENSOR_POST_ENDPOINT = "/sensor"
PREDICT_ENDPOINT = "/predict"
LATEST_SENSOR_ENDPOINT = "/latest_sensor"
