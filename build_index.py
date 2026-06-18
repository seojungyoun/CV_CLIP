import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import faiss
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import open_clip

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


def metadata_text(row):

    fields = [
        "title",
        "blip_caption",
        "classification",
        "medium",
        "object_date",
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


def main():

    print("Loading metadata...")

    metadata_path = (
        config.DATA_DIR / "valid_metadata_blip.csv"
        if (config.DATA_DIR / "valid_metadata_blip.csv").exists()
        else config.DATA_DIR / "valid_metadata.csv"
    )

    df = pd.read_csv(
        metadata_path
    )

    print(
        f"Rows: {len(df)}"
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device: {device}"
    )

    print(
        "Loading OpenCLIP..."
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

    vectors = []
    indexed_rows = []

    print(
        "Building image embeddings..."
    )

    with torch.no_grad():

        for _, row in tqdm(
            df.iterrows(),
            total=len(df)
        ):

            try:

                image = (
                    Image.open(row["image_path"])
                    .convert("RGB")
                )

                image_tensor = (
                    preprocess(image)
                    .unsqueeze(0)
                    .to(device)
                )

                image_feat = model.encode_image(
                    image_tensor
                )

                image_feat = F.normalize(
                    image_feat.float(),
                    dim=-1
                )

                text = metadata_text(row)

                if text:

                    tokens = tokenizer(
                        [text]
                    ).to(device)

                    text_feat = model.encode_text(
                        tokens
                    )

                    text_feat = F.normalize(
                        text_feat.float(),
                        dim=-1
                    )

                    feat = (
                        config.IMAGE_EMBED_WEIGHT
                        * image_feat
                        + (1 - config.IMAGE_EMBED_WEIGHT)
                        * text_feat
                    )

                    feat = F.normalize(
                        feat.float(),
                        dim=-1
                    )

                else:

                    feat = image_feat

                vectors.append(
                    feat.cpu().numpy()
                )

                indexed_rows.append(
                    {
                        **row.to_dict(),
                        "search_text": text,
                    }
                )

            except Exception as exc:

                print(
                    "Skip:",
                    row.get("image_path", ""),
                    exc
                )

    if not vectors:
        raise RuntimeError(
            "No image embeddings were created."
        )

    vectors = np.concatenate(
        vectors,
        axis=0
    ).astype(
        "float32"
    )

    dim = vectors.shape[1]

    print(
        f"Embedding dim: {dim}"
    )

    index = faiss.IndexFlatIP(
        dim
    )

    index.add(
        vectors
    )

    faiss.write_index(
        index,
        str(config.INDEX_PATH)
    )

    indexed_df = pd.DataFrame(
        indexed_rows
    )

    indexed_df.to_csv(
        config.INDEXED_METADATA_PATH,
        index=False
    )

    print()
    print(
        f"Saved index: {config.INDEX_PATH}"
    )

    print(
        f"Saved indexed metadata: {config.INDEXED_METADATA_PATH}"
    )

    print(
        "Index mode: hybrid "
        f"image={config.IMAGE_EMBED_WEIGHT:.2f} "
        f"metadata={1 - config.IMAGE_EMBED_WEIGHT:.2f}"
    )

    print(
        f"Vectors: {index.ntotal}"
    )


if __name__ == "__main__":
    main()
