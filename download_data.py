"""
MetObjects.csv 다운로드 스크립트
출처: https://github.com/metmuseum/openaccess
크기: ~317 MB  /  약 470,000 오브젝트
"""
import os
import requests
from tqdm import tqdm

URL = "https://raw.githubusercontent.com/metmuseum/openaccess/master/MetObjects.csv"
DEST = "./MetObjects.csv"


def download():
    if os.path.exists(DEST):
        size_mb = os.path.getsize(DEST) / 1e6
        print(f"이미 존재합니다: {DEST} ({size_mb:.1f} MB) — 재다운로드하려면 파일을 삭제하세요.")
        return

    print(f"다운로드 중: {URL}")
    resp = requests.get(URL, stream=True, timeout=60)
    resp.raise_for_status()

    total = int(resp.headers.get("content-length", 0))
    with open(DEST, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, desc="MetObjects.csv"
    ) as bar:
        for chunk in resp.iter_content(chunk_size=1024 * 64):
            f.write(chunk)
            bar.update(len(chunk))

    print(f"완료: {DEST} ({os.path.getsize(DEST)/1e6:.1f} MB)")


if __name__ == "__main__":
    download()
