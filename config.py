from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RAW_CSV = DATA_DIR / "MetObjects.csv"
IMAGE_DIR = DATA_DIR / "images"
ARTIFACT_DIR = ROOT / "artifacts"
MODEL_DIR = ARTIFACT_DIR / "model"
INDEX_PATH = ARTIFACT_DIR / "met.index"
EMBEDDINGS_PATH = ARTIFACT_DIR / "embeddings.npy"
METADATA_PATH = ARTIFACT_DIR / "metadata.csv"
METRICS_PATH = ARTIFACT_DIR / "metrics.json"

MODEL_NAME = "ViT-B-32"
PRETRAINED = "laion2b_s34b_b79k"
SEED = 42

CSV_COLUMNS = [
    "Object ID",
    "Is Public Domain",
    "Primary Image",
    "Primary Image Small",
    "Title",
    "Artist Display Name",
    "Object Date",
    "Medium",
    "Culture",
    "Department",
    "Classification",
    "Tags",
    "Link Resource",
]
