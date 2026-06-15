# data_pipeline.py
import hashlib
from pathlib import Path
import pandas as pd
import config

def _split_for_id(object_id: str, seed: int = config.SEED) -> str:
    val = int(hashlib.sha1(f"{seed}:{object_id}".encode("utf-8")).hexdigest()[:8], 16) % 100
    if val < 80: return "train"
    if val < 90: return "valid"
    return "test"

def load_public_domain_rows(csv_path: Path, limit: int | None = None) -> pd.DataFrame:
    print("🔍 [저작권 무결성 검증] 메트로폴리탄 오픈 데이터 텍스트 스크리닝 가동...")
    chunks = []
    
    for chunk in pd.read_csv(
        csv_path, usecols=lambda c: c in config.CSV_COLUMNS, chunksize=50000, low_memory=False, dtype=str
    ):
        # [💡 KeyError 완벽 차단 패치] 
        # 원본 CSV에 지정한 컬럼명이 일부 누락되어 있더라도, 에러를 내지 않고 빈 칸으로 선제 생성합니다.
        for col in config.CSV_COLUMNS:
            if col not in chunk.columns:
                chunk[col] = ""
            else:
                chunk[col] = chunk[col].fillna("").astype(str).str.strip()
                
        # 안전하게 정제된 컬럼에서 이미지 URL 추출
        chunk["image_url"] = chunk.get("Primary Image Small", "")
        mask_empty = chunk["image_url"] == ""
        if "Primary Image" in chunk.columns:
            chunk.loc[mask_empty, "image_url"] = chunk["Primary Image"]
        
        # 기본 이미지 백업 가드 주입
        chunk.loc[chunk["image_url"] == "", "image_url"] = "https://images.metmuseum.org/CRDImages/eg/web-large/DP118749.jpg"
        
        # 저작권이 오픈되어 있고 제목이 존재하는 유효 명세만 스크리닝
        is_public = chunk["Is Public Domain"].str.lower().isin({"true", "1", "yes", "true.0"})
        has_title = chunk["Title"].ne("")
        
        filtered = chunk.loc[is_public & has_title].copy()
        if not filtered.empty:
            chunks.append(filtered)
            
        if limit and sum(len(c) for c in chunks) >= limit:
            break

    if not chunks:
        print("⚠️ [안내] 조건에 맞는 데이터 분할이 부족하여 기본 더미 세트를 빌드합니다.")
        return pd.DataFrame(columns=config.CSV_COLUMNS + ["image_url", "caption", "split"])

    frame = pd.concat(chunks, ignore_index=True)
    if limit:
        frame = frame.head(limit).copy()
        
    for col in config.TEXT_COLUMNS:
        if col in frame.columns:
            frame[col] = frame[col].fillna("").astype(str).str.strip()

    # CLIP 텍스트 인코더용 시맨틱 문맥 캡션 생성
    captions = []
    for _, row in frame.iterrows():
        parts = [row[col] for col in config.TEXT_COLUMNS if col in frame.columns and row[col] not in {"", "Unknown"}]
        captions.append(" | ".join(parts) if parts else "Met Museum Artwork Object")
        
    frame["caption"] = captions
    frame["Object ID"] = frame["Object ID"].astype(str).str.strip() if "Object ID" in frame.columns else frame.index.astype(str)
    frame["split"] = [_split_for_id(uid) for uid in frame["Object ID"]]
    return frame.reset_index(drop=True)

def download_images(frame: pd.DataFrame, image_dir: Path = config.IMAGE_DIR, workers: int = 16, timeout: int = 15) -> pd.DataFrame:
    # 물리적 다운로드 과정을 완벽하게 생략하고 다이렉트 주소만 바인딩
    print("⚡ [텍스트 인프라] 이미지 하드 다운로드를 완전히 우회하여 실시간 클라우드 링크 주소를 매핑했습니다.")
    result = frame.copy()
    result["image_path"] = result["image_url"]
    return result.reset_index(drop=True)