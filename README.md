# CV_CLIP: OpenCLIP 기반 미술품 의미 검색 시스템

## 프로젝트 소개

CV_CLIP은 Met Museum Open Access Dataset을 활용하여 자연어 기반 미술품 검색을 수행하는 Vision-Language 검색 시스템이다.

사용자가 입력한 텍스트 질의를 OpenCLIP 임베딩으로 변환하고, 미리 구축된 이미지 임베딩 데이터베이스와 비교하여 의미적으로 유사한 작품을 검색한다.

또한 BLIP(Bootstrapping Language-Image Pre-training)를 활용하여 작품 이미지로부터 자동 캡션을 생성하고, 이를 검색 인덱스 구축 과정에 활용하여 검색 성능을 향상시켰다.

---

## 프로젝트 목표

* OpenCLIP 기반 이미지-텍스트 검색 시스템 구현
* FAISS 기반 대규모 유사도 검색 구축
* BLIP Caption을 활용한 검색 성능 향상
* Streamlit 기반 검색 UI 제공

---

## 데이터셋

### Met Museum Open Access Dataset

* 원본 데이터: 484,956개 작품
* 데이터 출처:

  * Met Museum Open Access Dataset
  * MetObjects.csv

### 데이터 구축 과정

| 단계            | 개수      |
| ------------- | ------- |
| 원본 데이터        | 484,956 |
| 수집 메타데이터      | 2,400   |
| 최종 유효 이미지     | 656     |
| 검색 인덱스 사용 데이터 | 656     |

### 선택 Department

* Arms and Armor
* Asian Art
* Egyptian Art
* European Paintings
* European Sculpture and Decorative Arts
* Greek and Roman Art
* Islamic Art
* The American Wing

---

## 시스템 아키텍처

MetObjects.csv

↓

이미지 다운로드

↓

유효 이미지 선별

↓

BLIP Caption 생성

↓

BLIP Caption + Metadata 결합

↓

OpenCLIP Text Encoder

↓

텍스트 임베딩 생성

↓

FAISS Index 구축

↓

Streamlit 검색 서비스

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

## 파일 설명

### prepare_data.py

MetObjects.csv에서 프로젝트에 사용할 메타데이터를 추출한다.

### download_images.py

작품 이미지를 다운로드한다.

### generate_captions.py

BLIP 모델을 사용하여 작품 이미지의 캡션을 생성한다.

생성된 캡션은 Classification, Department 정보와 결합되어 valid_metadata_blip.csv에 저장된다.

### build_index.py

OpenCLIP을 이용하여 텍스트 임베딩을 생성하고 FAISS 검색 인덱스를 구축한다.

입력 텍스트:

```text
BLIP Caption
+ Classification
+ Department
+ Title
```

### evaluate.py

Baseline 성능 평가

입력 텍스트:

```text
Title
+ Classification
+ Medium
```

### evaluate_blip.py

BLIP 적용 성능 평가

입력 텍스트:

```text
BLIP Caption
+ Title
```

(BLIP Caption 내부에 Classification 및 Department 정보 포함)

### app.py

Streamlit 기반 검색 인터페이스

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

### 4. 검색 인덱스 구축

```bash
python build_index.py
```

### 5. 성능 평가

Baseline

```bash
python evaluate.py
```

BLIP

```bash
python evaluate_blip.py
```

### 6. 검색 서비스 실행

```bash
streamlit run app.py
```

---

## 성능 평가

### Baseline

| Metric              | Score  |
| ------------------- | ------ |
| Zero-shot Accuracy  | 16.62% |
| Image Retrieval R@1 | 62.96% |
| Image Retrieval R@5 | 97.53% |

### BLIP 적용

| Metric              | Score   |
| ------------------- | ------- |
| Zero-shot Accuracy  | 18.52%  |
| Image Retrieval R@1 | 72.84%  |
| Image Retrieval R@5 | 100.00% |

---

## 실패 사례

### 이미지 다운로드 실패

일부 작품은 이미지 URL이 존재하더라도 접근이 불가능하거나 다운로드 과정에서 오류가 발생하였다.

그 결과 초기 2,400개 데이터 중 최종적으로 656개의 작품만 활용할 수 있었다.

### 의미 기반 검색 한계

추상적인 질의 또는 작품 맥락에 대한 질의에서는 CLIP이 사용자의 의도를 정확히 이해하지 못하는 경우가 존재하였다.

---

## 사용 기술

* Python
* OpenCLIP
* BLIP
* FAISS
* PyTorch
* Streamlit
* Pandas
* Pillow

---

## 개발 환경

* Python 3.13
* Windows 11
* CUDA 지원 GPU (선택)
* VS Code

---

## 참고 자료

* OpenCLIP
* BLIP
* FAISS
* Met Museum Open Access Dataset
