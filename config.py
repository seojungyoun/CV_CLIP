# config.py
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RAW_CSV = DATA_DIR / "MetObjects.csv"
IMAGE_DIR = DATA_DIR / "images"
ARTIFACT_DIR = ROOT / "artifacts"
MODEL_DIR = ARTIFACT_DIR / "model"
INDEX_PATH = ARTIFACT_DIR / "met.index"

# [💡 에러 해결 핵심] 누락되었던 임베딩 캐시 저장 경로 선언 추가
EMBEDDINGS_PATH = ARTIFACT_DIR / "embeddings.npy"

METADATA_PATH = ARTIFACT_DIR / "metadata.csv"
CLEAN_DATA_CSV = ARTIFACT_DIR / "dataset.csv"
LORA_WEIGHTS_PATH = MODEL_DIR / "best.pt"
METRICS_PATH = ARTIFACT_DIR / "metrics.json"

MODEL_NAME = "ViT-B-32"
PRETRAINED = "laion2b_s34b_b79k"
SEED = 42

CSV_COLUMNS = [
    "Object ID", "Is Public Domain", "Primary Image", "Primary Image Small",
    "Title", "Artist Display Name", "Object Date", "Medium",
    "Culture", "Department", "Classification", "Tags", "Link Resource"
]

TEXT_COLUMNS = [
    "Title", "Artist Display Name", "Object Date", "Medium", 
    "Culture", "Department", "Classification", "Tags"
]