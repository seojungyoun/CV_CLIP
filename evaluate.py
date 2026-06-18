import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json
import time

import open_clip
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

import config


def clean_text(value):

    if pd.isna(value):
        return ""

    text = str(value).strip()

    if not text or text.lower() == "nan":
        return ""

    return text


def build_caption(row):

    fields = [
        "title",
        "blip_caption",
        "classification",
        "medium",
        "department",
    ]

    return ". ".join(
        value
        for value in (
            clean_text(row.get(field))
            for field in fields
        )
        if value
    )


def load_metadata():

    blip_path = config.DATA_DIR / "valid_metadata_blip.csv"

    if blip_path.exists():
        return blip_path, pd.read_csv(blip_path)

    fallback_path = config.DATA_DIR / "valid_metadata.csv"

    return fallback_path, pd.read_csv(fallback_path)


def main():

    print("Evaluation Start")

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    metadata_path, meta = load_metadata()

    print(f"Metadata: {metadata_path}")
    print(f"Rows: {len(meta)}")

    model, _, preprocess = open_clip.create_model_and_transforms(
        config.MODEL_NAME,
        pretrained=config.PRETRAINED,
        device=device
    )

    tokenizer = open_clip.get_tokenizer(
        config.MODEL_NAME
    )

    model.eval()

    image_vectors = []
    text_vectors = []
    valid_departments = []
    latencies = []
    prompts = []

    print("Running evaluation...")

    with torch.no_grad():

        for _, row in tqdm(
            meta.iterrows(),
            total=len(meta)
        ):

            try:

                caption = build_caption(row)

                if not caption:
                    continue

                img = Image.open(
                    row["image_path"]
                ).convert("RGB")

                img_tensor = (
                    preprocess(img)
                    .unsqueeze(0)
                    .to(device)
                )

                tokens = tokenizer(
                    [caption]
                ).to(device)

                start = time.perf_counter()

                img_feat = model.encode_image(
                    img_tensor
                )

                txt_feat = model.encode_text(
                    tokens
                )

                if device.type == "cuda":
                    torch.cuda.synchronize()

                elapsed = (
                    time.perf_counter()
                    - start
                ) * 1000

                latencies.append(elapsed)

                img_feat = F.normalize(
                    img_feat.float(),
                    dim=-1
                )

                txt_feat = F.normalize(
                    txt_feat.float(),
                    dim=-1
                )

                image_vectors.append(
                    img_feat.cpu()
                )

                text_vectors.append(
                    txt_feat.cpu()
                )

                valid_departments.append(
                    row["department"]
                )

                prompts.append(caption)

            except Exception as exc:
                print("Skip:", row.get("image_path", ""), exc)

    img_mat = torch.cat(image_vectors)
    txt_mat = torch.cat(text_vectors)

    sim = txt_mat @ img_mat.T

    ranking = sim.argsort(
        dim=1,
        descending=True
    )

    target = torch.arange(
        len(sim)
    ).unsqueeze(1)

    r1 = (
        (ranking[:, :1] == target)
        .any(dim=1)
        .float()
        .mean()
        .item()
    )

    r5 = (
        (ranking[:, :5] == target)
        .any(dim=1)
        .float()
        .mean()
        .item()
    )

    departments = sorted(
        pd.Series(valid_departments)
        .dropna()
        .unique()
        .tolist()
    )

    dept_prompts = [
        f"a museum artwork from {department}"
        for department in departments
    ]

    dept_tokens = tokenizer(
        dept_prompts
    ).to(device)

    with torch.no_grad():

        dept_feats = model.encode_text(
            dept_tokens
        )

        dept_feats = F.normalize(
            dept_feats.float(),
            dim=-1
        ).cpu()

    pred = (
        img_mat @ dept_feats.T
    ).argmax(dim=1)

    gt = torch.tensor([
        departments.index(department)
        for department in valid_departments
    ])

    accuracy = (
        (pred == gt)
        .float()
        .mean()
        .item()
    )

    metrics = {
        "Evaluation Metadata":
            metadata_path.name,

        "Text Features":
            "title + blip_caption + classification + medium + department",

        "Evaluated Samples":
            len(valid_departments),

        "Zero-shot Accuracy":
            f"{accuracy * 100:.2f}%",

        "Image Retrieval R@1":
            f"{r1 * 100:.2f}%",

        "Image Retrieval R@5":
            f"{r5 * 100:.2f}%",

        "Inference Latency":
            f"{pd.Series(latencies).mean():.2f} ms"
    }

    metrics_path = (
        config.ARTIFACT_DIR /
        "metrics.json"
    )

    with open(
        metrics_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metrics,
            f,
            indent=4,
            ensure_ascii=False
        )

    print()
    print(
        json.dumps(
            metrics,
            indent=4,
            ensure_ascii=False
        )
    )

    print()
    print(f"Saved: {metrics_path}")


if __name__ == "__main__":
    main()
