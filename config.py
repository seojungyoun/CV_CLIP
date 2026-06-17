from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
IMAGE_DIR = DATA_DIR / "images"

ARTIFACT_DIR = BASE_DIR / "artifacts"

DATA_DIR.mkdir(exist_ok=True)
IMAGE_DIR.mkdir(exist_ok=True)
ARTIFACT_DIR.mkdir(exist_ok=True)

MET_CSV_URL = (
    "https://media.githubusercontent.com/media/"
    "metmuseum/openaccess/master/MetObjects.csv"
)

METADATA_PATH = DATA_DIR / "metadata.csv"

INDEX_PATH = ARTIFACT_DIR / "met.index"

MODEL_NAME = "ViT-B-32"
PRETRAINED = "laion2b_s34b_b79k"

NUM_OBJECTS = 3000

TOP_K = 9

METRICS_PATH = ARTIFACT_DIR / "metrics.json"