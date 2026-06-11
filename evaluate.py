import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.amp import autocast

from clip_utils import get_device, load_clip
from config import ARTIFACT_DIR, METRICS_PATH, MODEL_DIR


def encode_test(frame, model, preprocess, tokenizer, device, batch_size):
    image_vectors, text_vectors = [], []
    latencies = []
    with torch.inference_mode():
        for start in range(0, len(frame), batch_size):
            rows = frame.iloc[start : start + batch_size]
            images = []
            for path in rows["image_path"]:
                with Image.open(path) as image:
                    images.append(preprocess(image.convert("RGB")))
            image_tensor = torch.stack(images).to(device)
            tokens = tokenizer(rows["caption"].tolist()).to(device)
            tick = time.perf_counter()
            with autocast(device.type, enabled=device.type == "cuda"):
                image_embedding = model.encode_image(image_tensor)
                text_embedding = model.encode_text(tokens)
            if device.type == "cuda":
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - tick) * 1000 / len(rows))
            image_vectors.append(torch.nn.functional.normalize(image_embedding.float(), dim=-1).cpu())
            text_vectors.append(torch.nn.functional.normalize(text_embedding.float(), dim=-1).cpu())
    return torch.cat(image_vectors), torch.cat(text_vectors), float(np.mean(latencies))


def metrics(
    image_vectors,
    text_vectors,
    labels,
    latency_ms,
    model,
    tokenizer,
    device,
):
    similarity = text_vectors @ image_vectors.T
    order = similarity.argsort(dim=1, descending=True)
    target = torch.arange(len(similarity)).unsqueeze(1)
    r1 = (order[:, :1] == target).any(dim=1).float().mean().item()
    r5 = (order[:, :5] == target).any(dim=1).float().mean().item()

    unique_labels = sorted(set(labels))
    prompts = [f"a museum object from the {label} department" for label in unique_labels]
    with torch.inference_mode():
        prompt_tokens = tokenizer(prompts).to(device)
        prompt_vectors = model.encode_text(prompt_tokens).float()
        prompt_vectors = torch.nn.functional.normalize(prompt_vectors, dim=-1).cpu()
    predictions = (image_vectors @ prompt_vectors.T).argmax(dim=1)
    expected = torch.tensor([unique_labels.index(label) for label in labels])
    accuracy = (predictions == expected).float().mean().item()
    return {
        "zero_shot_accuracy": accuracy,
        "image_retrieval_r1": r1,
        "image_retrieval_r5": r5,
        "inference_latency_ms_per_item": latency_ms,
        "test_samples": len(labels),
        "zero_shot_definition": "department classification using class text prompts",
        "prompts": prompts,
    }


def run(dataset: Path, trained: bool, batch_size: int) -> dict:
    frame = pd.read_csv(dataset)
    frame = frame[frame["split"] == "test"].dropna(subset=["Department"]).reset_index(drop=True)
    device = get_device()
    model, preprocess, tokenizer = load_clip(device, trained=trained)
    image_vectors, text_vectors, latency = encode_test(
        frame, model, preprocess, tokenizer, device, batch_size
    )
    return metrics(
        image_vectors,
        text_vectors,
        frame["Department"].tolist(),
        latency,
        model,
        tokenizer,
        device,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline/고도화 CLIP 정량 평가")
    parser.add_argument("--dataset", type=Path, default=ARTIFACT_DIR / "dataset.csv")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    if not (MODEL_DIR / "best.pt").exists():
        raise FileNotFoundError("학습 가중치가 없습니다. 먼저 `python train.py`를 실행하세요.")
    result = {
        "baseline": run(args.dataset, trained=False, batch_size=args.batch_size),
        "fine_tuned": run(args.dataset, trained=True, batch_size=args.batch_size),
    }
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
