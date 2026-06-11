# 최종 보고서 체크리스트

## 데이터셋

- `artifacts/dataset.csv`의 전체 수량과 `split`, `Department` 분포를 표와 그래프로 제시
- 공개 도메인 이미지와 캡션 샘플 6개 이상 제시
- `Object ID` 기반 고정 해시 분할(Train 80 / Valid 10 / Test 10)과 누수 방지 설명
- `Is Public Domain == True` 및 유효 이미지 URL 조건을 명시

## 정량 평가

`python evaluate.py` 실행 후 `artifacts/metrics.json`의 Baseline/Fine-tuned 결과를 표로 작성한다.

| Model | Zero-shot Accuracy | Image Retrieval R@1 | Image Retrieval R@5 | Latency (ms/item) |
|---|---:|---:|---:|---:|
| Baseline CLIP | 실행 결과 | 실행 결과 | 실행 결과 | 실행 결과 |
| Fine-tuned CLIP | 실행 결과 | 실행 결과 | 실행 결과 | 실행 결과 |

측정하지 않은 수치를 임의로 작성하지 않는다.

## Failure Case

- 검색어와 무관한 작품이 상위 노출된 사례
- 비슷한 재질이나 부서에 과도하게 편향된 사례
- 작고 흐리거나 배경이 복잡한 이미지 사례
- 원인, 재현 조건, 개선 아이디어를 각 사례에 함께 기록

## 서비스 및 기술 구현

- 앱 시작 시 모델/FAISS/메타데이터 캐시
- 시작할 때 전체 데이터 재임베딩하지 않는 오프라인 인덱스 구조
- 잘못된 데이터와 누락된 산출물에 대한 안내
- `config.py`, `data_pipeline.py`, `train.py`, `build_index.py`, `evaluate.py`, `app.py` 분리
- 대용량 파일의 외부 저장소 링크 또는 자동 다운로드 절차
