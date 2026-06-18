import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json
import re
import time

import faiss
import open_clip
import pandas as pd
import streamlit as st
import torch
import torch.nn.functional as F

import config


QUERY_MAP = {
    "화려한": "ornate luxurious decorative gold jewelry royal object",
    "우아한": "elegant refined graceful decorative artwork",
    "고풍스러운": "historical antique classical old artifact",
    "왕실풍": "royal aristocratic palace luxury gold ornate decorative",
    "초상화": "portrait person face figure painting",
    "인물": "portrait person face figure painting",
    "꽃무늬": "floral flower pattern decorative artwork",
    "도자기": "ceramic porcelain pottery vase bowl vessel",
    "유리": "glass transparent vessel decorative object",
    "조각상": "sculpture statue carved figure artwork",
    "고대": "ancient archaeological historical artifact",
    "풍경": "landscape scenery river mountain city view painting",
    "그릇": "bowl plate dish vessel tableware",
    "은": "silver metal pewter decorative object",
    "무기": "weapon sword armor arms military historical object",
    "중세": "medieval gothic armor weapon historical artifact",
}


st.set_page_config(
    page_title="MuseAI",
    layout="wide"
)

st.markdown("""
<style>
.block-container {
    padding-top: 1rem;
    padding-bottom: 0rem;
}
div[data-testid="stVerticalBlock"] {
    gap: 0.35rem;
}
img {
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model, _, _ = open_clip.create_model_and_transforms(
        config.MODEL_NAME,
        pretrained=config.PRETRAINED,
        device=device
    )

    tokenizer = open_clip.get_tokenizer(
        config.MODEL_NAME
    )

    model.eval()

    return model, tokenizer, device


@st.cache_resource
def load_index():

    return faiss.read_index(
        str(config.INDEX_PATH)
    )


@st.cache_data
def load_metadata():

    metadata_candidates = [
        config.INDEXED_METADATA_PATH,
        config.DATA_DIR / "metadata_blip.csv",
        config.DATA_DIR / "valid_metadata_blip.csv",
        config.DATA_DIR / "valid_metadata.csv",
    ]

    metadata_path = next(
        path
        for path in metadata_candidates
        if path.exists()
    )

    return pd.read_csv(metadata_path)


@st.cache_data
def load_metrics():

    if not config.METRICS_PATH.exists():
        return None

    with open(
        config.METRICS_PATH,
        "r",
        encoding="utf-8"
    ) as f:
        return json.load(f)


def clean_value(value, fallback="-"):

    if pd.isna(value):
        return fallback

    text = str(value).strip()

    if not text or text.lower() == "nan":
        return fallback

    return text


def row_search_text(row):

    if "search_text" in row and clean_value(row.get("search_text")) != "-":
        return clean_value(row.get("search_text"), "")

    fields = [
        "title",
        "blip_caption",
        "classification",
        "medium",
        "object_date",
        "department",
    ]

    return " ".join(
        clean_value(row.get(field), "")
        for field in fields
        if clean_value(row.get(field), "")
    )


def query_expansions(query):

    expansions = [query]

    for key, value in QUERY_MAP.items():
        if key in query:
            expansions.append(value)

    return expansions


def expand_query(query):

    return " ".join(query_expansions(query))


def query_terms(query):

    expanded = expand_query(query).lower()
    terms = re.findall(r"[a-zA-Z가-힣0-9]+", expanded)

    return [
        term
        for term in terms
        if len(term) >= 2
    ]


def keyword_score(row, terms):

    if not terms:
        return 0.0

    text = row_search_text(row).lower()
    title = clean_value(row.get("title"), "").lower()

    matches = 0.0

    for term in terms:
        if term in text:
            matches += 1.0
        if title and term in title:
            matches += 0.5

    return min(
        matches / max(len(terms), 1),
        1.0
    )


def minmax(values):

    values = list(values)

    if not values:
        return []

    low = min(values)
    high = max(values)

    if high == low:
        return [1.0 for _ in values]

    return [
        (value - low) / (high - low)
        for value in values
    ]


def rerank(ids, scores, query, metadata):

    terms = query_terms(query)
    norm_scores = minmax(scores)
    ranked = []

    for idx, clip_score, norm_clip in zip(ids, scores, norm_scores):

        if idx < 0 or idx >= len(metadata):
            continue

        row = metadata.iloc[int(idx)]
        key_score = keyword_score(row, terms)
        final_score = (
            config.RERANK_CLIP_WEIGHT * norm_clip
            + config.RERANK_KEYWORD_WEIGHT * key_score
        )

        ranked.append(
            {
                "idx": int(idx),
                "clip_score": float(clip_score),
                "keyword_score": float(key_score),
                "final_score": float(final_score),
            }
        )

    ranked.sort(
        key=lambda item: item["final_score"],
        reverse=True
    )

    return ranked


def search(query, top_k=12):

    model, tokenizer, device = load_model()
    index = load_index()
    metadata = load_metadata()

    expanded = expand_query(query)

    with torch.no_grad():

        tokens = tokenizer(
            [expanded]
        ).to(device)

        feat = model.encode_text(tokens)

        feat = F.normalize(
            feat.float(),
            dim=-1
        )

        feat = (
            feat.cpu()
            .numpy()
            .astype("float32")
        )

    candidate_k = min(
        max(top_k, config.RERANK_CANDIDATES),
        index.ntotal
    )

    scores, ids = index.search(
        feat,
        candidate_k
    )

    return rerank(
        ids[0],
        scores[0],
        query,
        metadata
    )[:top_k]


meta = load_metadata()
metrics = load_metrics()

st.title("🏛 MuseAI Semantic Artwork Search")
st.caption(
    "OpenCLIP 이미지/캡션/메타데이터 하이브리드 임베딩 + FAISS 후보 검색 + Re-ranking"
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("작품 수", len(meta))

with col2:
    st.metric("CLIP 모델", config.MODEL_NAME)

with col3:
    st.metric("FAISS 벡터", len(meta))

with col4:
    st.metric("후보 재정렬", f"Top {config.RERANK_CANDIDATES}")

if metrics:

    st.markdown("---")

    metric_title = "성능 평가"

    if metrics.get("Evaluation Metadata"):
        metric_title += f" ({metrics['Evaluation Metadata']})"

    st.subheader(metric_title)

    feature_text = metrics.get("Text Features")

    if feature_text:
        st.caption(f"평가 입력: {feature_text}")

    m1, m2, m3, m4, m5 = st.columns(5)

    with m1:
        st.metric(
            "Zero-shot Accuracy",
            metrics.get("Zero-shot Accuracy", "-")
        )

    with m2:
        st.metric(
            "R@1",
            metrics.get("Image Retrieval R@1", "-")
        )

    with m3:
        st.metric(
            "R@5",
            metrics.get("Image Retrieval R@5", "-")
        )

    with m4:
        st.metric(
            "Latency",
            metrics.get("Inference Latency", "-")
        )

    with m5:
        st.metric(
            "Samples",
            metrics.get("Evaluated Samples", len(meta))
        )

st.markdown("---")

top_k = st.slider(
    "표시할 작품 수",
    min_value=4,
    max_value=24,
    value=12,
    step=4
)

query = st.text_input(
    "검색어 입력",
    placeholder="예: 화려한, 우아한 유리병, 초상화, 왕실풍 장식품"
)

st.caption(
    "예시 검색어: 화려한 | 우아한 | 초상화 | 왕실풍 | 고대 유물 | 은 공예품 | 풍경화"
)

if query:

    start = time.perf_counter()
    results = search(query, top_k=top_k)
    elapsed = (time.perf_counter() - start) * 1000

    st.success(f"검색 시간: {elapsed:.1f} ms")
    st.caption(f"확장 질의: {expand_query(query)}")
    st.subheader(f"'{query}' 검색 결과")

    num_cols = 4

    for row_start in range(0, len(results), num_cols):

        cols = st.columns(num_cols)

        for col_idx in range(num_cols):

            result_idx = row_start + col_idx

            if result_idx >= len(results):
                continue

            result = results[result_idx]
            row = meta.iloc[result["idx"]]

            with cols[col_idx]:

                try:
                    st.image(
                        row["image_path"],
                        use_container_width=True
                    )
                except Exception:
                    st.empty()

                st.markdown(f"**{clean_value(row.get('title'))}**")
                st.caption(clean_value(row.get("department")))

                classification = clean_value(row.get("classification"))
                medium = clean_value(row.get("medium"))

                if classification != "-":
                    st.caption(f"분류: {classification}")

                if medium != "-":
                    st.caption(f"재료: {medium}")

                st.caption(
                    "점수: "
                    f"{result['final_score']:.3f} "
                    f"(CLIP {result['clip_score']:.3f}, "
                    f"키워드 {result['keyword_score']:.2f})"
                )

                object_id = row["object_id"]
                met_url = (
                    "https://www.metmuseum.org/"
                    f"art/collection/search/{object_id}"
                )

                st.markdown(f"[🏛 작품 보기]({met_url})")
