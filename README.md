## CLIP + LoRA 시맨틱 검색 엔진: 검색 정확도 수정 및 Met Museum 데이터셋 전환
## Summary

- **검색 정확도 3가지 근본 원인 수정**: 코사인 유사도 적용, LoRA 학습 루프 추가, LoRA 타겟 모듈 오류 수정
- **데이터셋 전환**: Pushkin Museum (7,562개, 러시아어) → Met Museum Open Access (470,000+개, 영어)
- **UI 전면 재설계**: 카드 레이아웃, 유사도 바, 부서 배지, Met API 썸네일

## 수정된 버그

### 1. IndexFlatL2 → IndexFlatIP + 정규화
CLIP은 코사인 유사도 기반인데 비정규화 벡터에 L2 거리 사용 → 순위 역전.
`emb / emb.norm()` + `IndexFlatIP`로 교체.

### 2. LoRA 학습 루프 누락
학습 루프 자체가 없었음. 대칭 InfoNCE loss로 파인튜닝 추가.

### 3. LoRA 타겟 모듈 오류
`out_proj`는 CLIP 내부에서 weight만 직접 추출 → LoRA forward 우회 → 그래디언트 없음.
MLP 레이어 `c_fc`, `c_proj`로 변경.

## 실행 방법

```bash
python download_data.py   # MetObjects.csv 다운로드
python train.py           # 파인튜닝
streamlit run app.py      # 앱 실행
