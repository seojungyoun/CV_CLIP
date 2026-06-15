# build_index.py
import argparse
import faiss
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import config
import open_clip

def main():
    parser = argparse.ArgumentParser(description="텍스트 피처 기반 FAISS 오프라인 벡터 인덱스 빌더")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ 인덱스 연산 디바이스: {device}")
    
    # 순정 고성능 원본 Base CLIP 모델 로드
    model, _, _ = open_clip.create_model_and_transforms(config.MODEL_NAME, pretrained=config.PRETRAINED, device=device)
    tokenizer = open_clip.get_tokenizer(config.MODEL_NAME)
    model.eval()
    
    frame = pd.read_csv(config.CLEAN_DATA_CSV)
    captions = frame["caption"].tolist()
    vectors = []
    
    print("🔮 미술품 메타데이터 문맥 임베딩 추출 시작...")
    with torch.no_grad():
        for i in tqdm(range(0, len(captions), args.batch_size), desc="Indexing Text Captions"):
            batch_texts = captions[i:i+args.batch_size]
            tokens = tokenizer(batch_texts).to(device)
            
            # 텍스트 인코더를 통해 고차원 시맨틱 공간 벡터 추출 후 고밀도 정규화
            text_features = model.encode_text(tokens)
            text_features = torch.nn.functional.normalize(text_features.float(), dim=-1)
            vectors.append(text_features.cpu().numpy())

    matrix = np.concatenate(vectors).astype("float32")
    
    # 코사인 유사도 고속 검색을 위한 FlatIP 트리 구축 및 영구화
    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)
    
    faiss.write_index(index, str(config.INDEX_PATH))
    np.save(config.EMBEDDINGS_PATH, matrix)
    print(f"✅ 오프라인 시맨틱 인덱스 빌드 완료: {config.INDEX_PATH} ({len(frame):,} items)")

if __name__ == "__main__":
    main()