import io
import time

import pandas as pd
import requests
from PIL import Image
from tqdm import tqdm

import config


session = requests.Session()

session.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/137.0 Safari/537.36"
    )
})


def get_image_url(object_id):

    url = (
        "https://collectionapi.metmuseum.org/"
        f"public/collection/v1/objects/{object_id}"
    )

    try:

        r = session.get(
            url,
            timeout=20
        )

        if not r.ok:
            return None

        data = r.json()

        image_url = (
            data.get("primaryImageSmall")
            or data.get("primaryImage")
        )

        if not image_url:
            return None

        return image_url

    except Exception:
        return None


def download_image(url, save_path):

    try:

        r = session.get(
            url,
            timeout=30
        )

        if not r.ok:
            return False

        img = Image.open(
            io.BytesIO(r.content)
        ).convert("RGB")

        img.save(
            save_path,
            quality=95
        )

        return True

    except Exception:
        return False


def main():

    frame = pd.read_csv(
        config.METADATA_PATH
    )

    success = 0
    valid_rows = []

    print(
        f"Downloading images from "
        f"{len(frame)} artworks..."
    )

    for _, row in tqdm(
        frame.iterrows(),
        total=len(frame)
    ):

        object_id = int(
            row["object_id"]
        )

        save_path = (
            config.IMAGE_DIR /
            f"{object_id}.jpg"
        )

        if save_path.exists():

            success += 1

            row_dict = row.to_dict()

            row_dict["image_path"] = str(save_path)

            # row_dict["image_url"] = image_url

            row_dict["met_url"] = (
                f"https://www.metmuseum.org/art/collection/search/{object_id}"
            )

            valid_rows.append(
                row_dict
            )

            continue

        image_url = get_image_url(
            object_id
        )

        if image_url is None:
            continue

        ok = download_image(
            image_url,
            save_path
        )

        if ok:

            success += 1

            row_dict = row.to_dict()

            row_dict["image_path"] = str(
                save_path
            )

            valid_rows.append(
                row_dict
            )

        time.sleep(0.05)

    valid_df = pd.DataFrame(
        valid_rows
    )

    valid_path = (
        config.DATA_DIR /
        "valid_metadata.csv"
    )

    valid_df.to_csv(
        valid_path,
        index=False
    )

    print()
    print(
        f"Downloaded: "
        f"{success}/{len(frame)}"
    )

    print(
        f"Valid metadata saved: "
        f"{len(valid_df)}"
    )

    print(
        f"Saved to: {valid_path}"
    )


if __name__ == "__main__":
    main()