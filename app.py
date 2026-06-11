import html
import time

import faiss
import pandas as pd
import streamlit as st
import torch

from clip_utils import get_device, load_clip
from config import INDEX_PATH, METADATA_PATH, METRICS_PATH

st.set_page_config(page_title="Met Museum CLIP Search", page_icon="M", layout="wide")


@st.cache_resource(show_spinner="CLIP 모델을 불러오는 중입니다...")
def cached_model():
    device = get_device()
    model, _, tokenizer = load_clip(device, trained=True)
    return model, tokenizer, device


@st.cache_resource(show_spinner="검색 인덱스를 불러오는 중입니다...")
def cached_index():
    return faiss.read_index(str(INDEX_PATH))


@st.cache_data
def cached_metadata():
    return pd.read_csv(METADATA_PATH, dtype={"Object ID": str})


def search(query: str, count: int, department: str):
    model, tokenizer, device = cached_model()
    index = cached_index()
    frame = cached_metadata()
    started = time.perf_counter()
    with torch.inference_mode():
        tokens = tokenizer([query]).to(device)
        vector = model.encode_text(tokens).float()
        vector = torch.nn.functional.normalize(vector, dim=-1).cpu().numpy()
    candidate_count = min(len(frame), max(count * 30, 300))
    scores, positions = index.search(vector, candidate_count)
    results = frame.iloc[positions[0]].copy()
    results["_score"] = scores[0]
    if department != "전체":
        results = results[results["Department"] == department]
    latency = (time.perf_counter() - started) * 1000
    return results.head(count), latency


st.title("Met Museum CLIP 시맨틱 검색")
st.caption("저작권 사용이 허용된 공개 도메인 이미지만 학습 및 검색에 사용합니다.")

if not INDEX_PATH.exists() or not METADATA_PATH.exists():
    st.warning("검색 산출물이 아직 없습니다. 아래 명령을 순서대로 실행하면 앱이 활성화됩니다.")
    st.code(
        "python download_data.py\n"
        "python prepare_data.py --limit 20000\n"
        "python train.py\n"
        "python build_index.py\n"
        "streamlit run app.py"
    )
    st.stop()

frame = cached_metadata()
departments = ["전체"] + sorted(frame["Department"].dropna().unique().tolist())
with st.sidebar:
    st.header("검색 설정")
    top_k = st.slider("결과 수", 4, 20, 8)
    department = st.selectbox("부서", departments)
    st.metric("검색 가능 작품", f"{len(frame):,}")
    if METRICS_PATH.exists():
        st.caption("정량 평가는 `artifacts/metrics.json`에 저장되어 있습니다.")

query = st.text_input(
    "자연어 검색",
    placeholder="예: a delicate golden ceremonial object",
    help="학습 캡션이 영어 중심이므로 영어 검색어의 정확도가 가장 높습니다.",
)

if query:
    results, latency = search(query, top_k, department)
    st.metric("검색 지연 시간", f"{latency:.1f} ms")
    if results.empty:
        st.info("조건에 맞는 결과가 없습니다. 부서 필터를 해제해 보세요.")
    for _, row in results.iterrows():
        with st.container(border=True):
            left, right = st.columns([1, 2])
            image_url = str(row.get("image_url", ""))
            if image_url.startswith("http"):
                left.image(image_url, use_container_width=True)
            title = html.escape(str(row.get("Title", "Untitled")))
            right.subheader(title)
            right.write(
                f"**Department:** {row.get('Department', '')}  \n"
                f"**Artist:** {row.get('Artist Display Name', '')}  \n"
                f"**Date:** {row.get('Object Date', '')}  \n"
                f"**Medium:** {row.get('Medium', '')}"
            )
            right.progress(max(0.0, min(1.0, (float(row["_score"]) + 1) / 2)))
            link = str(row.get("Link Resource", ""))
            if link.startswith("http"):
                right.link_button("Met Museum 원문", link)
