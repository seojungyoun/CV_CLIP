# 🏛 MuseAI: CLIP 기반 의미 중심 미술품 검색 시스템

## 1. 프로젝트 소개

MuseAI는 OpenCLIP과 FAISS를 활용하여 사용자가 입력한 자연어의 의미를 기반으로 Met Museum 작품을 검색하는 웹 서비스입니다.

기존의 키워드 기반 검색과 달리 "화려한", "우아한", "고풍스러운", "왕실풍", "초상화"와 같은 추상적이고 형용사적인 표현을 이해하여 관련 미술품을 검색할 수 있습니다.

본 프로젝트는 Met Museum Open Access Dataset을 활용하였으며, OpenCLIP 임베딩과 벡터 검색 기술을 결합하여 의미 기반(Semantic Search) 검색 시스템을 구현하였습니다.

---

## 2. 주요 기능

### Semantic Artwork Search

* 자연어 기반 의미 검색
* 한국어 검색 지원
* OpenCLIP Text Encoder 활용
* FAISS 기반 고속 벡터 검색

### Performance Dashboard

* Zero-shot Accuracy
* Image Retrieval R@1
* Image Retrieval R@5
* Inference Latency

### Museum Navigation

* Met Museum 원본 작품 페이지 연결
* 작품 메타데이터 제공
* 실시간 검색 시간 측정

---

## 3. 시스템 아키텍처

### 데이터 구축

Met Museum Open Access Dataset

↓

Public Domain 작품 필터링

↓

이미지 다운로드

↓

OpenCLIP Image Encoder

↓

512차원 임베딩 생성

↓

FAISS Index 저장

---

### 검색 과정

사용자 검색어 입력

↓

OpenCLIP Text Encoder

↓

텍스트 임베딩 생성

↓

FAISS Similarity Search

↓

Top-K 작품 반환

↓

Streamlit UI 출력

---

## 4. 개발 환경

### Language

* Python 3.10+

### Framework

* Streamlit

### Deep Learning

* PyTorch
* OpenCLIP

### Vector Search

* FAISS

### Data Processing

* Pandas
* Pillow

---

## 5. 설치 방법

### 1. Repository Clone

```bash
git clone https://github.com/your-repository/MuseAI.git

cd MuseAI
```

### 2. 가상환경 생성

```bash
conda create -n museai python=3.10

conda activate museai
```

또는

```bash
python -m venv venv

venv\Scripts\activate
```

### 3. 의존성 설치

```bash
pip install -r requirements.txt
```

---

## 6. 데이터 파이프라인

### Step 1. 메타데이터 수집

```bash
python prepare_data.py --limit 3000
```

결과:

```text
data/metadata.csv
```

생성

---

### Step 2. 이미지 다운로드

```bash
python download_images.py
```

결과:

```text
data/images/
data/valid_metadata.csv
```

생성

---

### Step 3. 벡터 인덱스 생성

```bash
python build_index.py
```

결과:

```text
artifacts/met.index
```

생성

---

### Step 4. 성능 평가

```bash
python evaluate.py
```

결과:

```text
artifacts/metrics.json
```

생성

---

### Step 5. 웹 서비스 실행

```bash
streamlit run app.py
```

---

## 7. 프로젝트 구조

```text
MuseAI
│
├── app.py
├── config.py
├── prepare_data.py
├── download_images.py
├── build_index.py
├── evaluate.py
├── requirements.txt
│
├── data
│   ├── metadata.csv
│   ├── valid_metadata.csv
│   └── images
│
├── artifacts
│   ├── met.index
│   └── metrics.json
│
└── README.md
```

---

## 8. 데이터셋

### Source

Met Museum Open Access Collection

https://github.com/metmuseum/openaccess

### Original Dataset

* 약 480,000개 작품

### Experiment Dataset

* Public Domain 작품 필터링
* 이미지 다운로드 성공 작품만 사용

최종 데이터셋:

* 656 작품

---

## 9. 성능 평가

### Evaluation Metrics

CLIP 기반 Retrieval Task

| Metric              | Score    |
| ------------------- | -------- |
| Zero-shot Accuracy  | 36.59%   |
| Image Retrieval R@1 | 16.62%   |
| Image Retrieval R@5 | 30.64%   |
| Inference Latency   | 16.82 ms |

---

## 10. Failure Case 분석

### Case 1

검색어:

```text
초상화
```

결과:

```text
갑옷
장식품
조각상
```

일부 반환

### 원인

OpenCLIP은 대규모 이미지-텍스트 데이터로 사전학습되었으며, 박물관 작품에 특화된 모델이 아니다.

Met Museum 데이터셋에는 유사한 시각적 특징을 갖는 작품이 다수 존재하여 의미적으로 관련성이 높은 작품이 함께 검색되는 현상이 발생하였다.

---

### Case 2

검색어:

```text
화려한
```

결과:

금속 공예품, 갑옷, 장식품 등이 혼합되어 검색

### 원인

"화려한"이라는 추상적 표현은 다양한 시각적 특징으로 해석될 수 있으며, CLIP 임베딩 공간에서 여러 장식성 높은 작품이 유사하게 배치되기 때문이다.

---

## 11. 구현 과정에서 적용한 기술적 개선

### Query Expansion

예시

```text
화려한
```

↓

```text
ornate luxurious decorative artwork
```

확장

---

### FAISS Index 활용

* 실시간 벡터 검색
* 검색 속도 개선

---

### Streamlit Cache

```python
@st.cache_resource
@st.cache_data
```

사용

효과

* 모델 중복 로딩 방지
* 인덱스 재생성 방지
* 응답 속도 개선

---

### 코드 모듈화

* config.py
* prepare_data.py
* download_images.py
* build_index.py
* evaluate.py
* app.py

분리

하드코딩 제거 및 유지보수성 향상

---

## 12. 팀원 역할 분담

### 서정윤

* 프로젝트 기획
* 데이터 파이프라인 구축
* OpenCLIP 기반 검색 모델 구현
* FAISS 벡터 검색 구현
* Streamlit UI 개발
* 성능 평가 및 분석
* 최종 보고서 작성

---

## 13. 실행 화면

프로젝트 실행 화면 캡처 이미지를 아래에 첨부한다.

* 검색 메인 화면
* 성능 평가 화면
* 검색 결과 화면

---

## 14. 참고 문헌

OpenCLIP

https://github.com/mlfoundations/open_clip

FAISS

https://github.com/facebookresearch/faiss

Met Museum Open Access

https://github.com/metmuseum/openaccess
