import pandas as pd
from PIL import Image
from tqdm import tqdm

from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration
)


INPUT_CSV = "data/valid_metadata.csv"
OUTPUT_CSV = "data/valid_metadata_blip.csv"


def main():

    print("Loading metadata...")

    meta = pd.read_csv(INPUT_CSV)

    print(
        f"Rows: {len(meta)}"
    )

    print("Loading BLIP model...")

    processor = (
        BlipProcessor.from_pretrained(
        "Salesforce/blip-image-captioning-base",
        local_files_only=True
    )
    )

    model = (
        BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base",
            local_files_only=True
        )
    )

    captions = []

    for _, row in tqdm(
        meta.iterrows(),
        total=len(meta)
    ):

        image_path = row["image_path"]

        try:

            img = (
                Image.open(image_path)
                .convert("RGB")
            )

            inputs = processor(
                img,
                return_tensors="pt"
            )

            output = model.generate(
                **inputs,
                max_new_tokens=30
            )

            blip_caption = (
                processor.decode(
                    output[0],
                    skip_special_tokens=True
                )
            )

            classification = str(
                row.get(
                    "classification",
                    ""
                )
            )

            department = str(
                row.get(
                    "department",
                    ""
                )
            )

            final_caption = (
                f"{blip_caption}. "
                f"{classification}. "
                f"{department}."
            )

            captions.append(
                final_caption
            )

        except Exception as e:

            print(
                "Skip:",
                image_path,
                e
            )

            captions.append("")

    meta["blip_caption"] = captions

    meta.to_csv(
        OUTPUT_CSV,
        index=False
    )

    print()
    print(
        f"Saved: {OUTPUT_CSV}"
    )


if __name__ == "__main__":
    main()