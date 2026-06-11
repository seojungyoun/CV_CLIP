# Met Museum CLIP Semantic Search

Metropolitan Museum of Art Open Access 이미지와 메타데이터를 사용해 CLIP을 미세조정하고, 자연어로 작품을 검색하는 Streamlit 프로젝트입니다. CSV에서 `Is Public Domain == True`이며 이미지 URL이 존재하는 작품만 데이터셋에 포함합니다.

> 실행 화면 캡처는 최종 배포 후 `docs/images/app.png`에 추가하세요. 측정하지 않은 성능 수치는 README나 보고서에 임의로 기입하지 않습니다.

## 주요 기능

- 공개 도메인 이미지 자동 필터링 및 병렬 다운로드
- 실제 이미지-텍스트 쌍을 이용한 CLIP contrastive fine-tuning
- Object ID 해시 기반 Train/Valid/Test 고정 분할
- Baseline 대비 Zero-shot Accuracy, Image Retrieval R@1/R@5, Latency 평가
- FAISS 인덱스 사전 생성으로 웹 앱 시작 시간 단축
- Streamlit 모델 및 인덱스 싱글톤 캐시
- 로컬 절대 경로 없는 설정 구조

## 프로젝트 구조

```text
CV_CLIP/
├── app.py                         # Streamlit UI
├── build_index.py                 # 이미지 임베딩 및 FAISS 인덱스 생성
├── clip_utils.py                  # CLIP 로드 및 학습 범위 설정
├── config.py                      # 경로와 모델 설정
├── data_pipeline.py               # 저작권 필터, 분할, 이미지 다운로드
├── download_data.py               # MetObjects.csv 자동 다운로드
├── evaluate.py                    # Baseline/Fine-tuned 정량 평가
├── prepare_data.py                # 학습 데이터 준비 CLI
├── train.py                       # CLIP 미세조정
├── requirements.txt
└── docs/FINAL_REPORT_CHECKLIST.md
```

## 개발 환경

- Python 3.10 또는 3.11 권장
- CUDA GPU 권장, CPU 실행 가능
- PyTorch, OpenCLIP, FAISS, Streamlit, pandas

## 설치 및 실행

```bash
git clone <REPOSITORY_URL>
cd CV_CLIP
python -m venv .venv
```

Windows:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

macOS/Linux:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

데이터 준비부터 앱 실행까지:

```bash
python download_data.py
python prepare_data.py --limit 20000 --workers 16
python train.py --epochs 3 --batch-size 64
python evaluate.py
python build_index.py
streamlit run app.py
```

GPU 메모리가 부족하면 `--batch-size 16` 또는 `32`를 사용하세요. 빠른 기능 확인은 `prepare_data.py --limit 1000`으로 가능합니다.

학습 산출물이 없어도 `streamlit run app.py`는 중단되지 않고 필요한 준비 명령을 화면에 안내합니다.

## 데이터 파이프라인

1. Met Open Access의 `MetObjects.csv`를 `data/`에 다운로드합니다.
2. `Is Public Domain` 값이 참인 행만 남깁니다.
3. `Primary Image` 또는 `Primary Image Small` URL이 있는 행만 남깁니다.
4. 병렬 다운로드 후 손상되거나 이미지가 아닌 응답을 제외합니다.
5. `Object ID`의 SHA-1 해시로 Train 80%, Valid 10%, Test 10%를 고정 분할합니다.
6. 제목, 작가, 연도, 재질, 문화권, 부서, 분류, 태그를 캡션으로 구성합니다.

동일한 `Object ID`는 항상 한 split에만 배정되므로 실행 순서가 바뀌어도 데이터 누수가 발생하지 않습니다.

## 속도 최적화

- 47만 행 전체가 아니라 저작권 및 이미지 조건을 먼저 적용합니다.
- 이미지 다운로드에 `ThreadPoolExecutor`를 사용합니다.
- CLIP 전체가 아닌 마지막 text transformer block과 projection만 학습합니다.
- CUDA AMP, pinned memory, non-blocking transfer, persistent DataLoader worker를 사용합니다.
- 앱 시작 시 임베딩을 다시 계산하지 않고 저장된 FAISS 인덱스를 읽습니다.
- Streamlit `@st.cache_resource`와 `@st.cache_data`로 모델, 인덱스, 메타데이터를 재사용합니다.

## 평가

```bash
python evaluate.py
```

결과는 `artifacts/metrics.json`에 저장됩니다.

| 구분 | 필수 지표 |
|---|---|
| Zero-shot | Department prompt classification accuracy |
| Retrieval | Image Retrieval R@1, R@5 |
| Runtime | 이미지 1개당 추론 latency(ms) |

`baseline`은 사전학습 CLIP, `fine_tuned`는 프로젝트 학습 가중치를 사용합니다. 보고서에는 실제 실행 결과와 함께 데이터 수, GPU/CPU, batch size를 기록하세요.

## 대용량 파일

`data/`, `artifacts/`, `.pt`, `.onnx`, `.npy`, `.index`는 Git에서 제외됩니다. 최종 가중치와 인덱스는 Google Drive 같은 외부 저장소에 업로드하고 아래 링크를 실제 주소로 교체하세요.

- Fine-tuned weights: `<WEIGHTS_DOWNLOAD_URL>`
- FAISS index and metadata: `<INDEX_DOWNLOAD_URL>`

또는 이 README의 명령으로 새 PC에서 직접 재생성할 수 있습니다.

## 팀원 역할

제출 전에 실제 역할로 수정하세요.

| 팀원 | 역할 |
|---|---|
| 팀원 1 | 데이터 필터링, 학습 파이프라인, 정량 평가 |
| 팀원 2 | FAISS 검색, Streamlit UI, 배포 및 문서화 |

## 보고서 작성

데이터 분포, 샘플과 라벨 상태, Baseline 비교표, Failure Case, 서비스 구현 내용은 [`docs/FINAL_REPORT_CHECKLIST.md`](docs/FINAL_REPORT_CHECKLIST.md)를 기준으로 작성합니다.

## 데이터 및 라이선스

- Metadata: [The Met Open Access](https://github.com/metmuseum/openaccess)
- 본 프로젝트는 CSV에서 공개 도메인으로 표시된 이미지에 한해 학습 및 화면 표시를 수행합니다.
