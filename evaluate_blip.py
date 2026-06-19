import json
import time

import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
import open_clip

import config


def main():

    print("📊 Evaluation Start")

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    meta = pd.read_csv(
        config.DATA_DIR /
        "valid_metadata_blip.csv"
    )

    model, _, preprocess = (
        open_clip.create_model_and_transforms(
            config.MODEL_NAME,
            pretrained=config.PRETRAINED,
            device=device
        )
    )

    tokenizer = open_clip.get_tokenizer(
        config.MODEL_NAME
    )

    model.eval()

    image_vectors = []
    text_vectors = []

    latencies = []

    prompts = []

    print(
        "🏃 Running evaluation..."
    )

    with torch.no_grad():

        for _, row in tqdm(
            meta.iterrows(),
            total=len(meta)
        ):

            try:

                img = Image.open(
                    row["image_path"]
                ).convert("RGB")

                caption = (
                    f"{row['title']} "
                    f"{row['classification']} "
                    f"{row['medium']}"
                )

                img_tensor = (
                    preprocess(img)
                    .unsqueeze(0)
                    .to(device)
                )

                tokens = tokenizer(
                    [caption]
                ).to(device)

                start = (
                    time.perf_counter()
                )

                img_feat = (
                    model.encode_image(
                        img_tensor
                    )
                )

                txt_feat = (
                    model.encode_text(
                        tokens
                    )
                )

                if device.type == "cuda":
                    torch.cuda.synchronize()

                elapsed = (
                    time.perf_counter()
                    - start
                ) * 1000

                latencies.append(
                    elapsed
                )

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

                prompts.append(
                    caption
                )

            except Exception:

                continue

    img_mat = torch.cat(
        image_vectors
    )

    txt_mat = torch.cat(
        text_vectors
    )

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
        meta["department"]
        .unique()
        .tolist()
    )

    dept_prompts = [
        f"a museum artwork from {d}"
        for d in departments
    ]

    dept_tokens = tokenizer(
        dept_prompts
    ).to(device)

    with torch.no_grad():

        dept_feats = (
            model.encode_text(
                dept_tokens
            )
        )

        dept_feats = F.normalize(
            dept_feats.float(),
            dim=-1
        ).cpu()

    pred = (
        img_mat @ dept_feats.T
    ).argmax(dim=1)

    gt = torch.tensor([
        departments.index(d)
        for d in meta["department"]
    ])

    accuracy = (
        (pred == gt)
        .float()
        .mean()
        .item()
    )

    metrics = {
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
    print(
        f"Saved: {metrics_path}"
    )


if __name__ == "__main__":
    main()