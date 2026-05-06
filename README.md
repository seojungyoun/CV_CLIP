# MetVision — CLIP-LoRA 기반 메트로폴리탄 박물관 시맨틱 검색 엔진

> 자연어 한 문장으로 470,000점의 Met Museum 컬렉션을 탐색하는 AI 검색 시스템

---

## 프로젝트 소개

**MetVision**은 OpenAI CLIP + LoRA 파인튜닝 + FAISS 벡터 DB를 결합한 시맨틱 검색 엔진입니다.  
기존 박물관 검색의 단순 키워드 매칭 방식을 대체하여, `"ancient Egyptian gold jewelry"`, `"19세기 인상주의 파리 풍경화"` 처럼 의미 기반 자연어 쿼리로 관련 작품을 즉시 찾아줍니다.

| | |
|---|---|
| **데이터셋** | Metropolitan Museum of Art Open Access (470,000+ 오브젝트) |
| **모델** | OpenAI CLIP ViT-B/32 + LoRA (MLP 레이어 파인튜닝) |
| **벡터 DB** | FAISS IndexFlatIP (코사인 유사도) |
| **UI** | Streamlit (부서 필터 · 유사도 바 · Met API 이미지 썸네일) |
| **학습 방식** | Self-supervised Contrastive Learning (라벨링 불필요) |

---

## 주요 기능

- **자연어 시맨틱 검색** — 영어 자연어 쿼리를 512차원 벡터로 임베딩 후 FAISS로 Top-K 탐색
- **부서 필터** — 17개 부서(Egyptian Art, European Paintings, Photographs 등) 별도 필터링
- **실시간 이미지 썸네일** — 공공도메인 작품은 Met API를 통해 실제 이미지 표시 (토글)
- **유사도 점수 표시** — 검색 결과마다 코사인 유사도를 바(bar) 형태로 시각화
- **LoRA 경량 파인튜닝** — 전체 파라미터의 0.32%(49만 개)만 학습, 도메인 특화 성능 향상

---

## 프로젝트 구조

```
CV_CLIP/
├── app.py              # Streamlit 웹 애플리케이션
├── train.py            # LoRA 파인튜닝 스크립트
├── download_data.py    # MetObjects.csv 다운로드 스크립트
├── MetObjects.csv      # Met Museum 데이터 (~303 MB, 직접 다운로드)
├── museum_lora_weights.pt  # 학습된 LoRA 가중치 (train.py 실행 후 생성)
└── Final_초안 (2).ipynb   # Colab 전체 파이프라인 노트북
```

---

## 기술 스택

```
CLIP ViT-B/32  ──►  LoRA (c_fc, c_proj)  ──►  L2 정규화
                                                    │
                                              FAISS IndexFlatIP
                                             (코사인 유사도 검색)
                                                    │
                                             Streamlit UI
                                           (검색창 · 카드 · 필터)
```

### LoRA 타겟 모듈 선택 이유

CLIP의 `MultiheadAttention`은 `out_proj.weight`를 직접 추출해 `F.multi_head_attention_forward()`에 전달하므로 LoRA 래퍼의 `forward()`가 **우회**됩니다.  
그래디언트가 흐르는 MLP 레이어(`c_fc`, `c_proj`)를 타겟으로 지정해야 정상 학습이 가능합니다.

### 검색 정확도 핵심 수정

| 항목 | 기존 | 수정 |
|---|---|---|
| 유사도 지표 | `IndexFlatL2` (비정규화) | `IndexFlatIP` + L2 정규화 |
| 학습 | 없음 | Self-supervised InfoNCE loss |
| LoRA 타겟 | `q_proj, v_proj, out_proj` | `c_fc, c_proj` |

---

## 실행 방법

### 1. 환경 설정

```bash
pip install ftfy regex tqdm requests
pip install git+https://github.com/openai/CLIP.git
pip install peft faiss-cpu streamlit
```

### 2. 데이터 다운로드

```bash
python download_data.py
```

> `MetObjects.csv` (~303 MB) 를 GitHub에서 자동 다운로드합니다.  
> 이미 파일이 있으면 건너뜁니다.

### 3. LoRA 파인튜닝 (선택)

```bash
python train.py
```

| 환경 | 설정 | 예상 시간 |
|---|---|---|
| CPU | 10,000쌍 / batch 32 / 3 에폭 | 약 20~30분 |
| GPU (CUDA) | 100,000쌍 / batch 64 / 5 에폭 | 약 20분 |

학습이 끝나면 `museum_lora_weights.pt` 가 생성됩니다.  
파일이 없어도 앱은 Base CLIP으로 동작합니다.

### 4. 앱 실행

```bash
streamlit run app.py
```

브라우저에서 `http://localhost:8501` 접속.

---

## 검색 예시

| 쿼리 | 기대 결과 |
|---|---|
| `ancient Egyptian mummy sarcophagus` | 이집트 유물 · 석관 |
| `Renaissance oil portrait nobleman` | 유럽 회화 초상화 |
| `black and white street photography` | 20세기 흑백 사진 |
| `medieval knight sword armor` | 무기 · 갑옷 컬렉션 |
| `abstract expressionism painting` | 현대미술 추상 회화 |
| `ancient Greek marble sculpture` | 그리스·로마 조각 |

---

## Colab 배포 (Google Colab)

`Final_초안 (2).ipynb` 를 Colab에서 순서대로 실행하면 cloudflared 터널을 통해 퍼블릭 URL이 생성됩니다.

```
셀 1: 라이브러리 설치
셀 2: 데이터 다운로드
셀 3: 데이터 전처리
셀 4: CLIP + LoRA 모델 로드
셀 5: FAISS 인덱스 구축
셀 6: 검색 함수 테스트
셀 7: LoRA 파인튜닝
셀 8: app.py 작성
셀 9: Streamlit + cloudflared 배포
```

---

## 데이터셋

**Metropolitan Museum of Art Open Access**  
- 출처: [github.com/metmuseum/openaccess](https://github.com/metmuseum/openaccess)  
- 규모: 약 470,000 오브젝트, 54개 컬럼  
- 라이선스: [CC0 1.0 Universal](https://creativecommons.org/publicdomain/zero/1.0/)

주요 컬럼:

| 컬럼 | 설명 |
|---|---|
| `Title` | 작품명 |
| `Artist Display Name` | 작가명 |
| `Object Date` | 제작 연도 |
| `Medium` | 재료/기법 |
| `Department` | 부서 (17개) |
| `Classification` | 분류 |
| `Culture` | 문화권 |
| `Tags` | 태그 |
| `Is Public Domain` | 공공도메인 여부 |
| `Link Resource` | Met Museum 공식 페이지 URL |

---

## 팀원

| 이름 | 학번 |
|---|---|
| 서정윤 | 20220689 |
| 김연우 | 20220858 |

---

## 참고 문헌

- [CLIP: Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020) — Radford et al., 2021
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) — Hu et al., 2021
- [FAISS: A Library for Efficient Similarity Search](https://github.com/facebookresearch/faiss) — Facebook AI Research
