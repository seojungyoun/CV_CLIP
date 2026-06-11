import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests
from tqdm import tqdm

from config import CSV_COLUMNS, IMAGE_DIR, SEED

TEXT_COLUMNS = [
    "Title",
    "Artist Display Name",
    "Object Date",
    "Medium",
    "Culture",
    "Department",
    "Classification",
    "Tags",
]


def _as_true(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def _first_image_url(frame: pd.DataFrame) -> pd.Series:
    primary = frame["Primary Image"].fillna("").astype(str).str.strip()
    small = frame["Primary Image Small"].fillna("").astype(str).str.strip()
    return primary.where(primary.ne(""), small)


def _split_for_id(object_id: str, seed: int = SEED) -> str:
    value = int(hashlib.sha1(f"{seed}:{object_id}".encode()).hexdigest()[:8], 16) % 100
    if value < 80:
        return "train"
    if value < 90:
        return "valid"
    return "test"


def load_public_domain_rows(csv_path: Path, limit: int | None = None) -> pd.DataFrame:
    """Load only rows whose images are explicitly reusable and available."""
    chunks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        csv_path,
        usecols=lambda col: col in CSV_COLUMNS,
        chunksize=50_000,
        low_memory=False,
    ):
        required = {"Object ID", "Is Public Domain", "Title"}
        missing = required.difference(chunk.columns)
        if missing:
            raise ValueError(f"CSV 필수 열이 없습니다: {sorted(missing)}")
        for col in CSV_COLUMNS:
            if col not in chunk:
                chunk[col] = ""
        chunk["image_url"] = _first_image_url(chunk)
        mask = (
            _as_true(chunk["Is Public Domain"])
            & chunk["image_url"].str.startswith(("http://", "https://"))
            & chunk["Title"].fillna("").astype(str).str.strip().ne("")
        )
        chunks.append(chunk.loc[mask].copy())
        if limit and sum(len(part) for part in chunks) >= limit:
            break

    if not chunks:
        return pd.DataFrame(columns=CSV_COLUMNS + ["image_url", "caption", "split"])

    frame = pd.concat(chunks, ignore_index=True)
    if limit:
        frame = frame.head(limit).copy()
    for col in TEXT_COLUMNS:
        frame[col] = frame[col].fillna("").astype(str).str.strip()

    values = frame[TEXT_COLUMNS].astype(str)
    frame["caption"] = values.apply(
        lambda row: " | ".join(value for value in row if value), axis=1
    )
    frame["Object ID"] = frame["Object ID"].astype(str)
    frame["split"] = frame["Object ID"].map(_split_for_id)
    return frame.reset_index(drop=True)


def _download_one(row: tuple[str, str], image_dir: Path, timeout: int) -> bool:
    object_id, url = row
    target = image_dir / f"{object_id}.jpg"
    if target.exists() and target.stat().st_size > 1_024:
        return True
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        if not response.headers.get("content-type", "").startswith("image/"):
            return False
        target.write_bytes(response.content)
        return target.stat().st_size > 1_024
    except (requests.RequestException, OSError):
        target.unlink(missing_ok=True)
        return False


def download_images(
    frame: pd.DataFrame,
    image_dir: Path = IMAGE_DIR,
    workers: int = 16,
    timeout: int = 15,
) -> pd.DataFrame:
    image_dir.mkdir(parents=True, exist_ok=True)
    rows: Iterable[tuple[str, str]] = frame[["Object ID", "image_url"]].itertuples(
        index=False, name=None
    )
    ok_ids: set[str] = set()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_download_one, row, image_dir, timeout): row[0] for row in rows
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="images"):
            if future.result():
                ok_ids.add(str(futures[future]))
    result = frame[frame["Object ID"].isin(ok_ids)].copy()
    result["image_path"] = result["Object ID"].map(
        lambda value: str(image_dir / f"{value}.jpg")
    )
    return result.reset_index(drop=True)
