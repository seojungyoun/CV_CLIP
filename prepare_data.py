import argparse
from pathlib import Path

from config import ARTIFACT_DIR, IMAGE_DIR, RAW_CSV
from data_pipeline import download_images, load_public_domain_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="공개 도메인 Met 이미지 데이터 준비")
    parser.add_argument("--csv", type=Path, default=RAW_CSV)
    parser.add_argument("--limit", type=int, default=20_000)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    frame = load_public_domain_rows(args.csv, limit=args.limit)
    print(f"저작권/이미지 URL 필터 통과: {len(frame):,}개")
    frame = download_images(frame, IMAGE_DIR, workers=args.workers)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    output = ARTIFACT_DIR / "dataset.csv"
    frame.to_csv(output, index=False, encoding="utf-8")
    print(frame["split"].value_counts().to_string())
    print(f"유효 이미지 데이터 저장: {output} ({len(frame):,}개)")


if __name__ == "__main__":
    main()
