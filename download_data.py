from pathlib import Path

import requests
from tqdm import tqdm

from config import DATA_DIR, RAW_CSV

URL = "https://media.githubusercontent.com/media/metmuseum/openaccess/master/MetObjects.csv"


def download(destination: Path = RAW_CSV) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 100 * 1024 * 1024:
        print(f"이미 존재합니다: {destination}")
        return
    temporary = destination.with_suffix(".csv.part")
    with requests.get(URL, stream=True, timeout=60) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        with temporary.open("wb") as output, tqdm(
            total=total, unit="B", unit_scale=True, desc="MetObjects.csv"
        ) as bar:
            for chunk in response.iter_content(1024 * 1024):
                if chunk:
                    output.write(chunk)
                    bar.update(len(chunk))
    temporary.replace(destination)
    print(f"저장 완료: {destination}")


if __name__ == "__main__":
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    download()
