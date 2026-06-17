import json
import time

import faiss
import pandas as pd
import streamlit as st
import torch
import torch.nn.functional as F
import open_clip

import config


QUERY_MAP = {
    "화려한": "ornate luxurious decorative artwork gold jewelry royal object",
    "우아한": "elegant refined artwork decorative object",
    "고풍스러운": "historical antique classical artwork",
    "왕실풍": "royal luxurious golden artwork",
    "초상화": "portrait of a person",
    "꽃무늬": "floral decorative pattern artwork",
    "도자기": "ceramic porcelain vessel",
    "유리": "glass decorative object",
    "조각상": "sculpture statue artwork",
    "고대": "ancient artifact museum object"
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
    gap: 0.3rem;
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

    model, _, preprocess = (
        open_clip.create_model_and_transforms(
            config.MODEL_NAME,
            pretrained=config.PRETRAINED,
            device=device
        )
    )

    tokenizer = open_clip.get_tokenizer(
        config.MODEL_NAME
    )

    model.eval()

    return (
        model,
        tokenizer,
        device
    )


@st.cache_resource
def load_index():

    return faiss.read_index(
        str(config.INDEX_PATH)
    )


@st.cache_data
def load_metadata():

    return pd.read_csv(
        config.DATA_DIR /
        "valid_metadata.csv"
    )


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


def expand_query(query):

    expanded = query

    for key, value in QUERY_MAP.items():

        if key in query:
            expanded += " " + value

    return expanded


def search(query, top_k=12):

    model, tokenizer, device = (
        load_model()
    )

    index = load_index()

    query = expand_query(query)

    with torch.no_grad():

        tokens = tokenizer(
            [query]
        ).to(device)

        feat = model.encode_text(
            tokens
        )

        feat = F.normalize(
            feat.float(),
            dim=-1
        )

        feat = (
            feat.cpu()
            .numpy()
            .astype("float32")
        )

    scores, ids = index.search(
        feat,
        top_k
    )

    return (
        ids[0],
        scores[0]
    )


meta = load_metadata()
metrics = load_metrics()

st.title(
    "🏛 MuseAI Semantic Artwork Search"
)

st.caption(
    "OpenCLIP + FAISS 기반 의미 중심 미술품 검색 시스템"
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "작품 수",
        len(meta)
    )

with col2:
    st.metric(
        "CLIP 모델",
        config.MODEL_NAME
    )

with col3:
    st.metric(
        "FAISS 벡터",
        len(meta)
    )

with col4:
    st.metric(
        "검색 결과",
        "Top-K"
    )

if metrics:

    st.markdown("---")

    m1, m2, m3, m4 = st.columns(4)

    with m1:
        st.metric(
            "Zero-shot Accuracy",
            metrics.get(
                "Zero-shot Accuracy",
                "-"
            )
        )

    with m2:
        st.metric(
            "R@1",
            metrics.get(
                "Image Retrieval R@1",
                "-"
            )
        )

    with m3:
        st.metric(
            "R@5",
            metrics.get(
                "Image Retrieval R@5",
                "-"
            )
        )

    with m4:
        st.metric(
            "Latency",
            metrics.get(
                "Inference Latency",
                "-"
            )
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
    "예시 검색어: 화려한 | 우아한 | 초상화 | 왕실풍 | 고대 유물"
)

if query:

    start = time.perf_counter()

    ids, scores = search(
        query,
        top_k=top_k
    )

    elapsed = (
        time.perf_counter()
        - start
    ) * 1000

    st.success(
        f"검색 시간: {elapsed:.1f} ms"
    )

    st.subheader(
        f"'{query}' 검색 결과"
    )

    num_cols = 4

    for row_start in range(
        0,
        len(ids),
        num_cols
    ):

        cols = st.columns(num_cols)

        for col_idx in range(num_cols):

            result_idx = (
                row_start +
                col_idx
            )

            if result_idx >= len(ids):
                continue

            idx = ids[result_idx]

            if idx >= len(meta):
                continue

            row = meta.iloc[idx]

            score = scores[result_idx]

            with cols[col_idx]:

                try:

                    st.image(
                        row["image_path"],
                        use_container_width=True
                    )

                except Exception:

                    st.empty()

                st.markdown(
                    f"**{row['title']}**"
                )

                st.caption(
                    row["department"]
                )

                st.caption(
                    f"🏺 {row['classification']}"
                )

                st.caption(
                    f"유사도 {score:.3f}"
                )

                object_id = row["object_id"]

                met_url = (
                    "https://www.metmuseum.org/"
                    f"art/collection/search/{object_id}"
                )

                st.markdown(
                    f"[🏛 작품 보기]({met_url})"
                )