"""Central configuration: paths, model hyper-parameters and UI constants.

Everything that more than one module needs lives here so that training,
evaluation and the Streamlit app always agree (same input size, same
normalisation, same checkpoint path).
"""
from pathlib import Path

# ----------------------------------------------------------------- paths
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"                     # downloaded source photos
SYNTH_DIR = DATA_DIR / "synthetic"             # generated forgery dataset
CHECKPOINT_DIR = ROOT_DIR / "checkpoints"
CHECKPOINT_PATH = CHECKPOINT_DIR / "forensic_unet_best.pt"
REPORTS_DIR = ROOT_DIR / "reports"
HISTORY_PATH = REPORTS_DIR / "training_history.json"
METRICS_PATH = REPORTS_DIR / "test_metrics.json"
EVAL_EXAMPLES_DIR = REPORTS_DIR / "examples"
ASSETS_DIR = ROOT_DIR / "assets"
SAMPLES_DIR = ASSETS_DIR / "samples"

# ------------------------------------------------------- image validation
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
ALLOWED_PIL_FORMATS = {"JPEG", "PNG"}
MAX_FILE_SIZE_MB = 10
MIN_IMAGE_SIDE = 64
MAX_IMAGE_PIXELS = 40_000_000                  # guards against decompression bombs

# ------------------------------------------------------------------ model
# The U-Net is fully convolutional. Training uses 256x256 crops; at
# inference the image is down-scaled so its longer side is <= INFER_MAX_SIDE
# and padded to a multiple of 32 (the encoder's total stride).
TRAIN_CROP = 256
INFER_MAX_SIDE = 512
STRIDE_MULTIPLE = 32
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Default pixel threshold applied to the probability map -> binary mask.
DEFAULT_PIXEL_THRESHOLD = 0.5
# Fraction of highest-probability pixels averaged into the image-level score.
IMAGE_SCORE_TOP_FRACTION = 0.01
# Fallback image-level threshold, used only if the checkpoint does not carry
# one calibrated on the validation split (see training/train.py).
DEFAULT_IMAGE_THRESHOLD = 0.5

SEED = 42

# ------------------------------------------------------------- UI palette
PALETTE = {
    "cream": "#FAF7F0",
    "blush": "#F4E6DF",
    "ink": "#292929",
    "muted": "#6F6B65",
    "blue": "#DCEAF5",
    "sage": "#E4EDDF",
    "butter": "#FFF0C7",
    "pink": "#F7E0DD",
    "border": "#D8D0C7",
    # stronger tones used for chart lines / text on pastel backgrounds
    "blue_ink": "#4F7CA8",
    "sage_ink": "#5E7F55",
    "pink_ink": "#B5524A",
    "butter_ink": "#9A7B22",
}
