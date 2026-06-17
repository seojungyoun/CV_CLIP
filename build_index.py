import faiss
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
import open_clip

import config


def main():

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    meta = pd.read_csv(
        config.DATA_DIR /
        "valid_metadata.csv"
    )

    print(
        f"Indexing {len(meta)} images..."
    )

    model, _, preprocess = (
        open_clip.create_model_and_transforms(
            config.MODEL_NAME,
            pretrained=config.PRETRAINED,
            device=device
        )
    )

    model.eval()

    vectors = []

    with torch.no_grad():

        for _, row in tqdm(
            meta.iterrows(),
            total=len(meta)
        ):

            try:

                img = Image.open(
                    row["image_path"]
                ).convert("RGB")

                img_tensor = (
                    preprocess(img)
                    .unsqueeze(0)
                    .to(device)
                )

                feat = (
                    model.encode_image(
                        img_tensor
                    )
                )

                feat = F.normalize(
                    feat.float(),
                    dim=-1
                )

                vectors.append(
                    feat.cpu().numpy()[0]
                )

            except Exception:
                continue

    matrix = np.array(
        vectors,
        dtype="float32"
    )

    index = faiss.IndexFlatIP(
        matrix.shape[1]
    )

    index.add(matrix)

    faiss.write_index(
        index,
        str(config.INDEX_PATH)
    )

    np.save(
        config.ARTIFACT_DIR /
        "embeddings.npy",
        matrix
    )

    print(
        f"FAISS index saved "
        f"({len(matrix)} vectors)"
    )


if __name__ == "__main__":
    main()