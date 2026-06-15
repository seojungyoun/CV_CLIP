# prepare_data.py
import argparse
from pathlib import Path
import config
from data_pipeline import download_images, load_public_domain_rows

def main() -> None:
    parser = argparse.ArgumentParser(description="공개 도메인 Met 이미지 데이터 준비 스크립트")
    parser.add_argument("--limit", type=int, default=20000)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    # 1. 원본 CSV 파일 존재 여부 선제 확인
    if not config.RAW_CSV.exists():
        import requests
        print("📥 원본 MetObjects.csv 파일이 없습니다. 자동 다운로드를 시작합니다...")
        config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        with requests.get(config.CSV_URL, stream=True) as r:
            r.raise_for_status()
            with open(config.RAW_CSV, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024*1024):
                    if chunk:
                        f.write(chunk)
        print("✅ 원본 CSV 다운로드 완료.")
                    
    # 2. 메타데이터 파싱 및 저작권 검증 스크리닝
    frame = load_public_domain_rows(config.RAW_CSV, limit=args.limit)
    print(f" 1차 저작권 필터 통과: {len(frame):,}개")
    
    # 3. 이미지 정합성 검증 및 로컬 수집
    frame = download_images(frame, config.IMAGE_DIR, workers=args.workers)
    config.ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    
    output = config.CLEAN_DATA_CSV
    frame.to_csv(output, index=False, encoding="utf-8")
    
    print("\n======= 📈 데이터 Split 분할 분포 =======")
    if not frame.empty and "split" in frame.columns:
        print(frame["split"].value_counts().to_string())
    else:
        print("데이터 분할 정보가 없습니다.")
    print(f"✅ 유효 이미지 데이터셋 빌드 성공: {output} ({len(frame):,}개)")

if __name__ == "__main__":
    main()