# MuseAI: BLIP와 OpenCLIP 기반 의미 중심 미술품 검색 시스템 

## 프로젝트 소개

**MuseAI**는 Met Museum Open Access Dataset을 활용하여 자연어 기반 미술품 검색을 수행하는 Vision-Language 검색 시스템입니다.

사용자가 '우아한', '화려한'과 같은 추상적인 질의를 입력하더라도 , 이를 **OpenCLIP** 임베딩으로 변환하고 미리 구축된 이미지 임베딩 데이터베이스와 비교하여 의미적으로 가장 유사한 작품을 찾아냅니다.

특히 작품 제목(Title)만으로는 실제 시각적 특징을 모두 담기 어렵다는 한계(`Fragment`, `Vase` 등)를 극복하기 위해 , **BLIP** 모델을 활용하여 이미지의 시각적 특징을 자연어 문장(Caption)으로 자동 생성 및 확장하여 검색 인덱스에 결합함으로써 검색 성능을 대폭 향상시켰습니다.

---

## 프로젝트 목표

* 
**OpenCLIP 기반 이미지-텍스트 검색 시스템 구현**: 자연어와 작품 이미지를 동일한 의미 공간으로 정렬.


* 
**FAISS 기반 벡터 검색 인덱스 구축**: 대규모 벡터 공간에서 초고속 유사도 검색 알고리즘 적용.


* 
**BLIP Caption 기반 텍스트 확장**: 시각적 특징 기술을 통한 Zero-shot 검색 성능 고도화.


* 
**Streamlit 기반 웹 인터페이스 서비스 제공**: 직관적인 카드형 UI 및 메트로폴리탄 미술관 원본 웹페이지 연동.



---

## 데이터셋 및 구축 과정

### 1. 데이터 출처

* 
**Met Museum Open Access Dataset** (`MetObjects.csv`) 


* 원본 데이터 전체 규모: **484,956개** 작품 



### 2. 데이터 정제 및 샘플링 전략

* 원본 데이터의 극심한 부서(Department)별 불균형을 완화하기 위해 , Public Domain 작품 중 **8개 주요 부서**를 선정하여 각 300개씩 총 **2,400개**를 균등 샘플링하였습니다.


* 
**선정된 8개 부서 (Allowed Departments)** 


* Arms and Armor / Asian Art / Egyptian Art / European Paintings / European Sculpture and Decorative Arts / Greek and Roman Art / Islamic Art / The American Wing



### 3. 데이터 수집 단계별 규모

다운로드 과정에서 접근 권한 문제(HTTP 403 에러) 및 유효하지 않은 URL로 인해 최종 유효 이미지는 아래와 같이 압축되었습니다.

| 단계 | 데이터 개수 | 설명 |
| --- | --- | --- |
| **원본 데이터 (MetObjects)** | 484,956 | 메트로폴리탄 박물관 원본 전체 데이터셋 

 |
| **부서별 균등 샘플링** | 2,400 | 8개 부서에서 각 300개씩 무작위 추출 

 |
| **이미지 다운로드 성공** | 656 | 실제 서버 접근 및 유효성 검증을 통과한 이미지 

 |
| **최종 검색 및 평가 데이터** | **656** | 인덱스 구축 및 성능 평가에 사용된 최종 데이터셋 

 |

* 
**최종 데이터셋 부서별 편향 분포**: 이미지 확보 가능성에 따라 최종 데이터셋은 The American Wing(288점), Arms and Armor(210점), European Paintings(90점), Egyptian Art(68점)로 재구성되었습니다.



---

## 시스템 아키텍처 및 파이프라인

데이터 구축 및 임베딩 파이프라인 

1. 
`MetObjects.csv` 로드 및 조건 필터링 (`prepare_data.py`) 


2. 대상 작품 이미지 비동기 다운로드 및 예외 처리 (`download_images.py`) 


3. 
**BLIP** 기반 이미지 자동 캡션 생성 (`generate_captions.py`) 


4. **텍스트 확장 결합**: `[BLIP Caption]. [Classification]. [cite_start][Department].` ➔ 여기에 작품 제목(`Title`)을 더해 최종 인덱싱용 텍스트 구성.


5. 
**OpenCLIP Text Encoder**를 통한 고차원 벡터 변환 및 L2 정규화 


6. 
**FAISS 내적 인덱스(IndexFlatIP)** 구축 및 파일(`met.index`) 저장 (`build_index.py`) 


7. 
**Streamlit 웹 어플리케이션** 구동 및 실시간 시맨틱 검색 서비스 실행 (`app.py`) 



---

## 프로젝트 디렉토리 구조

```text
26-1_MuseAI
│
[cite_start]├── artifacts/                  # 전역 산출물 관리 폴더 [cite: 113]
│   ├── met.index               # FAISS 내적 검색 인덱스 파일
│   ├── metrics.json            # Baseline 성능 평가 결과 리포트
│   └── metrics_blip.json       # BLIP 적용 후 성능 평가 결과 리포트
│
[cite_start]├── data/                       # 데이터셋 저장 폴더 [cite: 113]
│   ├── images/                 # 다운로드 완료된 작품 이미지 (.jpg)
│   ├── metadata.csv            # 샘플링된 2,400개 행의 초기 메타데이터
│   ├── valid_metadata.csv      # 다운로드 성공한 656개 행의 유효 메타데이터
│   └── valid_metadata_blip.csv # BLIP 캡션이 추가된 최종 데이터셋
│
├── docs/                       # 프로젝트 문서 폴더
├── tests/                      # 테스트 스크립트 폴더
│
├── app.py                      # Streamlit 웹 서비스 구동 파일
├── build_index.py              # OpenCLIP 임베딩 생성 및 FAISS 인덱스 빌드 스크립트
├── config.py                   # 경로, 모델명, 하이퍼파라미터 등 전역 설정 관리
├── download_images.py          # 메타데이터 URL 기반 이미지 다운로드 스크립트
├── evaluate.py                 # Baseline 모델 성능 평가 스크립트
├── evaluate_blip.py            # BLIP 텍스트 확장 모델 성능 평가 스크립트
├── generate_captions.py        # Salesforce BLIP 기반 자동 캡션 생성 스크립트
└── prepare_data.py             # 원본 데이터 필터링 및 부서별 균등 샘플링 스크립트

```

---

## 핵심 파일별 실제 텍스트 인코딩 로직 정리

실제 시스템 내에서 각 모델이 인코딩하는 텍스트 결합 로직은 다음과 같이 철저히 분리되어 작동합니다.

1. **`generate_captions.py` (BLIP 모델 입력 및 저장)**
* 입력: `Image` ➔ BLIP 모델이 자연어 캡션 생성 


* 저장 텍스트 구성 (`valid_metadata_blip.csv`의 `blip_caption` 컬럼):
```text
[BLIP 생성 문장]. [Classification]. [Department].

```




2. **`build_index.py` (FAISS 인덱스 구축용 OpenCLIP 입력)**
* 입력: 생성된 `blip_caption` 컬럼 문장 뒤에 한 칸을 띄우고 고유 작품 제목(`title`)을 결합.


```text
[BLIP 생성 문장]. [Classification]. [Department]. [Title]

```




3. **`evaluate.py` (기존 Baseline 모델 성능 평가용 OpenCLIP 입력)**
* 입력: 이미지 고유의 순수 텍스트 메타데이터 3가지를 공백으로 결합.


```text
[Title] [Classification] [Medium]

```




4. **`evaluate_blip.py` (BLIP 고도화 모델 성능 평가용 OpenCLIP 입력)**
* 입력: `blip_caption` 컬럼 문장 뒤에 `". "`와 작품 제목(`title`)을 결합.


```text
[BLIP 생성 문장]. [Classification]. [Department].: [Title]

```





---

## 성능 평가 결과 (Quantitative Analysis)

별도의 Fine-tuning을 진행하지 않은 **Zero-shot 조건**에서 메타데이터 기반 Baseline 모델과 BLIP 고도화 모델의 검색(Retrieval) 성능 및 부서 분류(Classification) 성능을 비교 측정한 결과입니다.

| 평가 지표 (Metric) | Baseline 모델 (`evaluate.py`) | BLIP 고도화 모델 (`evaluate_blip.py`) | 성능 변화 폭 |
| --- | --- | --- | --- |
| **Zero-shot Accuracy** | 16.62% | **18.52%** | <br>**+1.90%p** (소폭 상승) 

 |
| **Image Retrieval R@1** | 62.96% | **72.84%** | <br>**+9.88%p** (약 15.7% 상대 향상) 

 |
| **Image Retrieval R@5** | 97.53% | **100.00%** | <br>**+2.47%p** (정답 완벽 포함) 

 |
| **Inference Latency** | 1,020.50 ms | **593.01 ms** | <br>**-427.49 ms** (성능 효율 개선) 

 |

### 결과 분석

* 
**Retrieval 성능의 유의미한 향상**: `R@1`이 62.96%에서 72.84%로 크게 증가하였고, `R@5`는 100.00%를 달성하여 정답 작품이 예외 없이 상위 5개 검색 결과 안에 진입하는 쾌거를 보였습니다. 이는 BLIP가 추출한 구체적인 시각적 묘사 문장들이 OpenCLIP 벡터 공간에서 정밀한 매핑을 도왔음을 입증합니다.


* 
**분류 Accuracy의 정체**: 검색 성능 향상 폭에 비해 `Zero-shot Accuracy`는 1.90%p 향상에 그쳤습니다. 이는 생성된 BLIP 캡션이 거시적인 박물관 부서(Department) 도메인 체계를 분류하기보다는, 개별 작품 간 세부적인 시각적 유사성을 찾아내는 검색 작업에 더욱 강력한 영향력을 행사한다는 것을 의미합니다.



---

## 주요 실패 사례 및 한계점 분석 (Failure Case)

1. 웹 이미지 데이터 접근 및 수집 한계 

* 원본 메타데이터가 정상 제공되더라도 Met Museum API 측의 접근 제한 정책 변화(HTTP 403) 혹은 이미지 유실로 인해 2,400개 중 656개만 수집에 성공하였습니다. 외부 Open Access 데이터 크롤링 시 실제 유효성 검증 단계의 필수성을 시사합니다.



2. 일반 도메인 기반 BLIP의 캡션 생성 오류 

* 일반 사물 위주의 데이터셋으로 사전 학습된 BLIP 모델의 특성상, 박물관 고유의 유물 자체 집중하기보다 무의미한 배경 정보를 과도하게 인지하는 현상이 확인되었습니다.


* 
*오류 예시*: 특정 유물을 삽입했을 때 고유 특징이 아닌 `"a black background with a yellow border(노란색 테두리가 있는 검은색 배경)"`로 캡션을 오용하여 생성하는 한계가 존재했습니다.



3. 고차원 추상적 질의 검색의 간극 

* OpenCLIP은 뛰어난 제로샷 성능을 보이지만 일반 이미지 도메인 기반으로 학습되었기 때문에, 사용자가 '심심한' 등 극도로 감성적이거나 추상적인 도메인 외 단어를 입력했을 경우 모델의 의미 공간과 사용자의 의도가 완전히 일치하지 않아 상이한 작품(총기류, 십자가 등)이 반환되는 한계가 존재합니다.



---

## 서비스 실행 방법

### 1. 의존성 패키지 설치

```bash
pip install -r requirements.txt

```

2. 엔드투엔드 파이프라인 가동 

```bash
# [Step 1] 데이터셋 로드 및 필터링 수행
python prepare_data.py

# [Step 2] 필터링된 작품 이미지 다운로드
python download_images.py

# [Step 3] 다운로드 완료 이미지 대상 BLIP 캡션 빌드
python generate_captions.py

# [Step 4] OpenCLIP 추출 및 FAISS 벡터 인덱싱 완료
python build_index.py

```

### 3. 정량적 성능 평가 스크립트 실행

```bash
# Baseline 결과 도출 (metrics.json 생성)
python evaluate.py

# BLIP 고도화 모델 결과 도출 (metrics_blip.json 생성)
python evaluate_blip.py

```

### 4. 웹 UI 검색 서비스 실행

```bash
streamlit run app.py

```

---

## 기술 스택 및 개발 환경

* **언어 및 프레임워크**: Python 3.13 / PyTorch / OpenCLIP / Transformers (Salesforce BLIP) / FAISS (IndexFlatIP)
* **데이터 및 UI 관리**: Pandas / Pillow / Streamlit