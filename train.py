import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

from clip_utils import get_device, load_clip, trainable_text_parameters
from config import ARTIFACT_DIR, MODEL_DIR, SEED


class MetImageTextDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, preprocess, tokenizer):
        self.frame = frame.reset_index(drop=True)
        self.preprocess = preprocess
        self.tokens = tokenizer(self.frame["caption"].tolist())

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        with Image.open(row["image_path"]) as image:
            image_tensor = self.preprocess(image.convert("RGB"))
        return image_tensor, self.tokens[index]


def clip_loss(image_features, text_features, logit_scale):
    image_features = F.normalize(image_features, dim=-1)
    text_features = F.normalize(text_features, dim=-1)
    logits = logit_scale.exp() * image_features @ text_features.T
    labels = torch.arange(len(logits), device=logits.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Met 공개 도메인 이미지로 CLIP 미세조정")
    parser.add_argument("--dataset", type=Path, default=ARTIFACT_DIR / "dataset.csv")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    args = parser.parse_args()

    set_seed(SEED)
    device = get_device()
    model, preprocess, tokenizer = load_clip(device, trained=False)
    params = trainable_text_parameters(model)
    frame = pd.read_csv(args.dataset)
    train_frame = frame[frame["split"] == "train"]
    valid_frame = frame[frame["split"] == "valid"]
    pin_memory = device.type == "cuda"
    loader_args = dict(
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=pin_memory,
        persistent_workers=args.workers > 0,
    )
    train_loader = DataLoader(
        MetImageTextDataset(train_frame, preprocess, tokenizer),
        shuffle=True,
        drop_last=True,
        **loader_args,
    )
    valid_loader = DataLoader(
        MetImageTextDataset(valid_frame, preprocess, tokenizer),
        shuffle=False,
        **loader_args,
    )
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    scaler = GradScaler("cuda", enabled=pin_memory)
    best_loss = float("inf")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        model.visual.eval()
        started = time.perf_counter()
        total = 0.0
        for images, tokens in train_loader:
            images = images.to(device, non_blocking=True)
            tokens = tokens.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(device.type, enabled=pin_memory):
                with torch.no_grad():
                    image_features = model.encode_image(images)
                text_features = model.encode_text(tokens)
                loss = clip_loss(image_features, text_features, model.logit_scale)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total += loss.item()

        model.eval()
        valid_total = 0.0
        with torch.inference_mode():
            for images, tokens in valid_loader:
                images = images.to(device, non_blocking=True)
                tokens = tokens.to(device, non_blocking=True)
                with autocast(device.type, enabled=pin_memory):
                    valid_total += clip_loss(
                        model.encode_image(images),
                        model.encode_text(tokens),
                        model.logit_scale,
                    ).item()
        train_loss = total / max(len(train_loader), 1)
        valid_loss = valid_total / max(len(valid_loader), 1)
        print(
            f"epoch={epoch + 1} train={train_loss:.4f} valid={valid_loss:.4f} "
            f"seconds={time.perf_counter() - started:.1f}"
        )
        if valid_loss < best_loss:
            best_loss = valid_loss
            trainable_names = {
                name for name, parameter in model.named_parameters() if parameter.requires_grad
            }
            compact_state = {
                name: tensor.cpu()
                for name, tensor in model.state_dict().items()
                if name in trainable_names
            }
            torch.save(
                {"model": compact_state, "valid_loss": best_loss},
                MODEL_DIR / "best.pt",
            )

    (MODEL_DIR / "train_summary.json").write_text(
        json.dumps({"best_valid_loss": best_loss, "device": str(device)}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
