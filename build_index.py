import argparse
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.amp import autocast
from tqdm import tqdm

from clip_utils import get_device, load_clip
from config import ARTIFACT_DIR, EMBEDDINGS_PATH, INDEX_PATH, METADATA_PATH


def main() -> None:
    parser = argparse.ArgumentParser(description="학습 모델의 이미지 FAISS 인덱스 생성")
    parser.add_argument("--dataset", type=Path, default=ARTIFACT_DIR / "dataset.csv")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    device = get_device()
    model, preprocess, _ = load_clip(device, trained=True)
    frame = pd.read_csv(args.dataset)
    frame = frame[frame["split"].isin(["train", "valid", "test"])].reset_index(drop=True)
    vectors: list[np.ndarray] = []
    batch: list[torch.Tensor] = []

    with torch.inference_mode():
        for path in tqdm(frame["image_path"], desc="index"):
            with Image.open(path) as image:
                batch.append(preprocess(image.convert("RGB")))
            if len(batch) == args.batch_size:
                images = torch.stack(batch).to(device)
                with autocast(device.type, enabled=device.type == "cuda"):
                    embedding = model.encode_image(images)
                embedding = torch.nn.functional.normalize(embedding.float(), dim=-1)
                vectors.append(embedding.cpu().numpy())
                batch.clear()
        if batch:
            images = torch.stack(batch).to(device)
            embedding = model.encode_image(images)
            embedding = torch.nn.functional.normalize(embedding.float(), dim=-1)
            vectors.append(embedding.cpu().numpy())

    matrix = np.concatenate(vectors).astype("float32")
    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_PATH))
    np.save(EMBEDDINGS_PATH, matrix)
    frame.to_csv(METADATA_PATH, index=False, encoding="utf-8")
    print(f"인덱스 저장 완료: {len(frame):,}개, {INDEX_PATH}")


if __name__ == "__main__":
    main()
