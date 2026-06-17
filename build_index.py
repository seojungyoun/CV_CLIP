import faiss
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import open_clip

from tqdm import tqdm

import config


def main():

    print("Loading metadata...")

    df = pd.read_csv(
        "data/valid_metadata_blip.csv"
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

    model, _, _ = (
        open_clip.create_model_and_transforms(
            config.MODEL_NAME,
            pretrained=config.PRETRAINED,
            device=device
        )
    )

    tokenizer = (
        open_clip.get_tokenizer(
            config.MODEL_NAME
        )
    )

    model.eval()

    vectors = []

    print(
        "Building embeddings..."
    )

    with torch.no_grad():

        for _, row in tqdm(
            df.iterrows(),
            total=len(df)
        ):

            blip_caption = str(
                row.get(
                    "blip_caption",
                    ""
                )
            )

            title = str(
                row.get(
                    "title",
                    ""
                )
            )

            text = (
                blip_caption
                + ". "
                + title
            )

            tokens = tokenizer(
                [text]
            ).to(device)

            feat = (
                model.encode_text(
                    tokens
                )
            )

            feat = F.normalize(
                feat.float(),
                dim=-1
            )

            vectors.append(
                feat.cpu().numpy()
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

    print()
    print(
        f"Saved index: {config.INDEX_PATH}"
    )

    print(
        f"Vectors: {index.ntotal}"
    )


if __name__ == "__main__":
    main()