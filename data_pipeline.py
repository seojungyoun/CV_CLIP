import hashlib
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

import config
import requests
import time


def _split_for_id(object_id: str, seed: int = config.SEED) -> str:
    val = int(
        hashlib.sha1(
            f"{seed}:{object_id}".encode("utf-8")
        ).hexdigest()[:8],
        16
    ) % 100

    if val < 80:
        return "train"
    if val < 90:
        return "valid"
    return "test"


session = requests.Session()

session.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/137.0 Safari/537.36"
    ),
    "Accept": "application/json",
    "Referer": "https://www.metmuseum.org/"
})


def get_image_url(object_id):

    try:

        url = (
            "https://collectionapi.metmuseum.org/"
            f"public/collection/v1/objects/{object_id}"
        )

        r = session.get(
            url,
            timeout=20
        )

        if r.status_code == 403:
            time.sleep(1)

            r = session.get(
                url,
                timeout=20
            )

        if not r.ok:
            print(
                f"FAIL: {object_id} {r.status_code}"
            )
            return ""

        data = r.json()

        return (
            data.get("primaryImageSmall")
            or data.get("primaryImage")
            or ""
        )

    except Exception as e:

        print(
            f"ERROR: {object_id} {e}"
        )

        return ""

def generate_visual_caption(row):

    title = str(row.get("Title", "")).strip()
    title_lower = title.lower()

    medium = str(
        row.get("Medium", "")
    ).strip().lower()

    culture = str(
        row.get("Culture", "")
    ).strip()

    department = str(
        row.get("Department", "")
    ).strip()

    classification = str(
        row.get("Classification", "")
    ).strip().lower()

    date = str(
        row.get("Object Date", "")
    ).strip()

    adjectives = []

    # 재질 기반
    if any(x in medium for x in ["gold", "silver", "gilded"]):
        adjectives += [
            "luxurious",
            "ornate",
            "royal",
            "valuable"
        ]

    if any(x in medium for x in ["glass", "crystal"]):
        adjectives += [
            "elegant",
            "decorative",
            "delicate"
        ]

    if any(x in medium for x in ["porcelain", "ceramic"]):
        adjectives += [
            "colorful",
            "decorative",
            "artistic"
        ]

    if any(x in medium for x in ["wood"]):
        adjectives += [
            "traditional",
            "crafted",
            "historical"
        ]

    if any(x in medium for x in ["marble", "stone"]):
        adjectives += [
            "classical",
            "monumental",
            "historical"
        ]

        # 재질 기반 확장

        if "brass" in medium:
            adjectives += [
                "ornate",
                "decorative",
                "metalwork"
            ]

        if "iron" in medium:
            adjectives += [
                "historical",
                "crafted",
                "industrial"
            ]

        if "mahogany" in medium:
            adjectives += [
                "luxurious",
                "elegant",
                "crafted"
            ]

        if "paper" in medium:
            adjectives += [
                "printed",
                "historical"
            ]

        if "earthenware" in medium:
            adjectives += [
                "decorative",
                "ceramic",
                "crafted"
            ]

        # 제목 기반 확장

        if "clock" in title_lower:
            adjectives += [
                "decorative",
                "historical",
                "crafted"
            ]

        if "advertisement" in title_lower:
            adjectives += [
                "printed",
                "graphic",
                "historical"
            ]

        if "andiron" in title_lower:
            adjectives += [
                "ornate",
                "decorative",
                "metalwork"
            ]

        if "glass" in title_lower:
            adjectives += [
                "elegant",
                "decorative"
            ]

    # 분류 기반
    if "painting" in classification:
        adjectives += [
            "fine art",
            "painted"
        ]

    if "sculpture" in classification:
        adjectives += [
            "three-dimensional",
            "classical"
        ]

    if "furniture" in classification:
        adjectives += [
            "decorative",
            "crafted"
        ]

    if "textile" in classification:
        adjectives += [
            "ornamental",
            "patterned"
        ]

    # 제목 기반
    title_lower = title.lower()

    if any(x in title_lower for x in [
        "portrait",
        "woman",
        "man",
        "figure",
        "lady",
        "gentleman"
    ]):
        adjectives += [
            "portrait",
            "person",
            "human"
        ]

    if any(x in title_lower for x in [
        "vase",
        "bottle",
        "jar"
    ]):
        adjectives += [
            "decorative",
            "crafted"
        ]

    if any(x in title_lower for x in [
        "chair",
        "table",
        "cabinet"
    ]):
        adjectives += [
            "furniture",
            "historical"
        ]

    

    adjectives = list(dict.fromkeys(adjectives))

    adjective_text = ", ".join(adjectives)

    return (
        f"A museum artwork. "
        f"Visual style: {adjective_text}. "
        f"Title: {title}. "
        f"Culture: {culture}. "
        f"Department: {department}. "
        f"Material: {medium}. "
        f"Date: {date}."
    )

def load_public_domain_rows(
    csv_path: Path,
    limit: int | None = None
) -> pd.DataFrame:

    print("🔍 Loading Met Open Access dataset...")

    frame = pd.read_csv(
        csv_path,
        low_memory=False,
        dtype=str
    )

    frame = frame.fillna("")

    frame = frame[
        frame["Is Public Domain"]
        .astype(str)
        .str.lower()
        .isin(["true", "1", "yes"])
    ].copy()

    frame = frame[
        frame["Title"].astype(str).str.strip() != ""
    ].copy()

    if limit:
        frame = frame.sample(
            n=min(limit, len(frame)),
            random_state=config.SEED
        )

    print(f"📦 Public domain objects: {len(frame):,}")

    image_urls = []

    object_ids = (
        frame["Object ID"]
        .astype(str)
        .tolist()
    )

    for object_id in tqdm(
        object_ids,
        desc="Fetching image URLs"
    ):
        image_urls.append(
            get_image_url(object_id)
        )

    frame["image_url"] = image_urls

    before = len(frame)

    frame = frame[
        frame["image_url"].astype(str) != ""
    ].copy()

    print(
        f"🖼️ Valid images: "
        f"{len(frame):,} / {before:,}"
    )

    frame["caption"] = frame.apply(
        generate_visual_caption,
        axis=1
    )

    frame["Object ID"] = (
        frame["Object ID"]
        .astype(str)
        .str.strip()
    )

    frame["split"] = [
        _split_for_id(obj_id)
        for obj_id in frame["Object ID"]
    ]

    return frame.reset_index(drop=True)


def download_images(
    frame: pd.DataFrame,
    image_dir: Path = config.IMAGE_DIR,
    workers: int = 16,
    timeout: int = 15
) -> pd.DataFrame:

    result = frame.copy()

    result["image_path"] = (
        result["image_url"]
        .astype(str)
    )

    return result.reset_index(drop=True)