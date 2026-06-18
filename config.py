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
INDEXED_METADATA_PATH = DATA_DIR / "indexed_metadata.csv"
IMAGE_EMBED_WEIGHT = 0.8
SAMPLE_PER_DEPARTMENT = 1000
RERANK_CANDIDATES = 50
RERANK_CLIP_WEIGHT = 0.85
RERANK_KEYWORD_WEIGHT = 0.15

MODEL_NAME = "ViT-L-14"
PRETRAINED = "laion2b_s32b_b82k"

NUM_OBJECTS = 3000

TOP_K = 9

METRICS_PATH = ARTIFACT_DIR / "metrics.json"
