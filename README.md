# MuseAI: BLIP와 OpenCLIP 기반 의미 중심 미술품 검색 시스템

## 프로젝트 소개

MuseAI는 Met Museum Open Access Dataset을 활용하여 자연어 기반 미술품 검색을 수행하는 Vision-Language 검색 시스템이다.

사용자가 입력한 자연어 질의를 OpenCLIP 임베딩으로 변환하고, 미리 구축된 FAISS 벡터 인덱스와 비교하여 의미적으로 유사한 작품을 검색한다.

또한 BLIP(Bootstrapping Language-Image Pre-training)를 활용하여 작품 이미지의 시각적 특징을 자연어 캡션으로 변환하고, 이를 OpenCLIP 입력에 활용하여 검색 성능 향상을 시도하였다.

---

## 프로젝트 목표

* OpenCLIP 기반 이미지-텍스트 검색 시스템 구현
* FAISS 기반 벡터 검색 인덱스 구축
* BLIP Caption 기반 텍스트 확장
* Streamlit 기반 검색 서비스 제공
* Zero-shot Retrieval 성능 향상

---

## 데이터셋

### 데이터 출처

* Met Museum Open Access Dataset
* MetObjects.csv
* https://github.com/metmuseum/openaccess/blob/master/MetObjects.csv

### 원본 데이터 규모

* 전체 작품 수: 484,956개

### 사용 Department

* Arms and Armor
* Asian Art
* Egyptian Art
* European Paintings
* European Sculpture and Decorative Arts
* Greek and Roman Art
* Islamic Art
* The American Wing

### 데이터 구축 과정

| 단계             |       개수 |
| -------------- | -------: |
| MetObjects 전체  |  484,956 |
| Department 샘플링 | 2,400 |
| 이미지 다운로드 성공    |      656 |
| 최종 사용 데이터      |      656 |

### 최종 데이터 컬럼

* object_id
* title
* department
* classification
* medium
* object_date
* image_path
* blip_caption

---

## 데이터 구축 파이프라인

```text
MetObjects.csv
        ↓
Department 필터링
        ↓
Department별 300개 샘플링
        ↓
이미지 다운로드
        ↓
유효 이미지 선별
        ↓
BLIP Caption 생성
        ↓
Classification + Department 결합
        ↓
valid_metadata_blip.csv 저장
        ↓
OpenCLIP Text Encoder
        ↓
텍스트 임베딩 생성
        ↓
FAISS Index 구축
```

---

## 검색 시스템 구조

```text
사용자 질의
        ↓
Query Expansion
        ↓
OpenCLIP Text Encoder
        ↓
텍스트 임베딩 생성
        ↓
FAISS 검색
        ↓
유사 작품 반환
        ↓
Streamlit UI 출력
```

---

## 프로젝트 구조

```text
CV_CLIP
│
├── artifacts
│   ├── image_cache/
│   ├── model/
│   ├── dataset.csv
│   ├── embeddings.npy
│   ├── met.index
│   ├── metadata.csv
│   ├── metrics.json
│   └── metrics_blip.json
│
├── data
│   ├── images/
│   ├── metadata.csv
│   ├── MetObjects.csv
│   ├── valid_metadata.csv
│   └── valid_metadata_blip.csv
│
├── docs/
├── tests/
│
├── app.py
├── build_index.py
├── config.py
├── download_images.py
├── evaluate.py
├── evaluate_blip.py
├── generate_captions.py
├── prepare_data.py
│
├── README.md
└── .gitignore
```

---

## 핵심 파일 설명

### prepare_data.py

Met Museum Open Access Dataset을 불러온 후 Public Domain 작품만 선택한다.

지정된 8개 Department를 필터링하고 Department별 300개 작품을 샘플링하여 metadata.csv를 생성한다.

### download_images.py

Met Museum Collection API를 통해 작품 이미지를 다운로드한다.

다운로드 성공 작품만 valid_metadata.csv에 저장한다.

### generate_captions.py

BLIP 모델을 사용하여 작품 이미지 캡션을 생성한다.

저장 형식:

```text
BLIP Caption.
Classification.
Department.
```

결과는 valid_metadata_blip.csv에 저장된다.

### build_index.py

OpenCLIP Text Encoder를 이용하여 텍스트 임베딩을 생성한다.


생성된 임베딩으로 FAISS IndexFlatIP 인덱스를 구축한다.

### evaluate.py

Baseline 성능 평가.

### evaluate_blip.py

BLIP 적용 성능 평가.

### app.py

Streamlit 기반 검색 서비스.

주요 기능:

* 자연어 검색
* 출력 개수 조절
* 작품 이미지 출력
* Met Museum 원본 링크 제공

---

## Query Expansion

검색 성능 향상을 위해 일부 한국어 키워드를 의미 기반 영어 표현으로 확장한다.

예시:

| 입력  | 확장                                                            |
| --- | ------------------------------------------------------------- |
| 화려한 | ornate luxurious decorative artwork gold jewelry royal object |
| 우아한 | elegant refined artwork decorative object                     |
| 왕실풍 | royal luxurious golden artwork                                |
| 초상화 | portrait of a person                                          |
| 유리  | glass decorative object                                       |
| 고대  | ancient artifact museum object                                |

---

## 성능 평가

### Baseline

| Metric              |      Score |
| ------------------- | ---------: |
| Zero-shot Accuracy  |     16.62% |
| Image Retrieval R@1 |     62.96% |
| Image Retrieval R@5 |     97.53% |
| Inference Latency   | 1020.50 ms |

### BLIP 적용

| Metric              |     Score |
| ------------------- | --------: |
| Zero-shot Accuracy  |    18.52% |
| Image Retrieval R@1 |    72.84% |
| Image Retrieval R@5 |   100.00% |
| Inference Latency   | 593.01 ms |

### 성능 비교

| Metric              | Baseline |    BLIP |
| ------------------- | -------: | ------: |
| Zero-shot Accuracy  |   16.62% |  18.52% |
| Image Retrieval R@1 |   62.96% |  72.84% |
| Image Retrieval R@5 |   97.53% | 100.00% |

BLIP Caption을 활용한 텍스트 확장을 통해 Retrieval 성능이 향상되었음을 확인할 수 있었다.

---

## 실패 사례

### 이미지 다운로드 실패

일부 작품은 API 접근 제한(403) 또는 이미지 URL 오류로 다운로드에 실패하였다.

### BLIP Caption 오류

일부 작품에서 실제 유물보다 배경을 설명하는 캡션이 생성되었다.

예시:

```text
a black background with a yellow border
```

### 의미 기반 검색 한계

"심심한", "감성적인" 등 추상적인 질의에서는 사용자의 의도와 다른 작품이 검색되는 경우가 존재하였다.

---

## 설치

```bash
pip install -r requirements.txt
```

---

## 실행 방법

### 1. 데이터 준비

```bash
python prepare_data.py
```

### 2. 이미지 다운로드

```bash
python download_images.py
```

### 3. BLIP 캡션 생성

```bash
python generate_captions.py
```

### 4. 인덱스 구축

```bash
python build_index.py
```

### 5. 성능 평가

```bash
python evaluate.py
```

```bash
python evaluate_blip.py
```

### 6. 검색 서비스 실행

```bash
streamlit run app.py
```

---

## 사용 기술

* Python
* OpenCLIP
* BLIP
* FAISS
* PyTorch
* Transformers
* Streamlit
* Pandas
* Pillow

---

## 개발 환경

* Python 3.13
* Windows 11
* VS Code
* CUDA GPU (선택)

---

## 참고 자료

* OpenCLIP
* BLIP
* FAISS
* Met Museum Open Access Dataset
* https://github.com/metmuseum/openaccess/blob/master/MetObjects.csv
