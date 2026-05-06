"""
CLIP-LoRA Museum Semantic Search — Streamlit App
핵심 수정사항:
  1. 임베딩 정규화 + IndexFlatIP (코사인 유사도) — L2 거리는 비정규화 벡터에 잘못된 결과를 냄
  2. 파인튜닝된 LoRA 가중치 자동 로드
  3. 모달리티 필터, 유사도 점수 표시, 이미지 썸네일
"""
import os
import time

import clip
import faiss
import numpy as np
import pandas as pd
import streamlit as st
import torch
from peft import LoraConfig, get_peft_model

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Museum Semantic Search",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
/* 전체 폰트 */
html, body, [class*="css"] { font-family: 'Inter', 'Noto Sans KR', sans-serif; }

/* 카드 */
div[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 10px !important;
    transition: box-shadow 0.2s;
}
div[data-testid="stVerticalBlockBorderWrapper"]:hover {
    box-shadow: 0 4px 16px rgba(0,0,0,0.12) !important;
}

/* 배지 공통 */
.badge {
    display: inline-block;
    padding: 2px 9px;
    border-radius: 20px;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.6px;
    text-transform: uppercase;
}
.badge-video   { background:#FF4B4B20; color:#FF4B4B; border:1px solid #FF4B4B50; }
.badge-audio   { background:#FF9F0A20; color:#FF9F0A; border:1px solid #FF9F0A50; }
.badge-image   { background:#34C75920; color:#2ea84f; border:1px solid #34C75950; }
.badge-article { background:#0A84FF20; color:#0A84FF; border:1px solid #0A84FF50; }

/* 유사도 바 */
.sim-bar-wrap {
    background:#e9ecef;
    border-radius:4px;
    height:5px;
    margin:5px 0 10px;
    overflow:hidden;
}
.sim-bar-fill {
    height:100%;
    border-radius:4px;
    background: linear-gradient(90deg, #0A84FF, #34C759);
}

/* 검색바 크게 */
div[data-testid="stTextInput"] input {
    font-size: 18px !important;
    padding: 14px 18px !important;
    border-radius: 10px !important;
}

/* 메트릭 카드 */
div[data-testid="metric-container"] {
    background: #f8f9fa;
    border-radius: 8px;
    padding: 8px 12px;
}

/* 사이드바 */
section[data-testid="stSidebar"] > div { padding-top: 1.5rem; }
</style>
""",
    unsafe_allow_html=True,
)

# ── Constants ─────────────────────────────────────────────────────────────────
MODALITY_ICON = {"video": "🎬", "audio": "🎵", "image": "🖼️", "article": "📰"}
MODALITY_COLOR = {"video": "badge-video", "audio": "badge-audio", "image": "badge-image", "article": "badge-article"}

SUGGESTED = [
    ("🇪🇬 Ancient Egypt",   "ancient Egypt ritual mummy"),
    ("🎨 Portrait",         "portrait painting nobleman"),
    ("🏺 Greek",            "ancient Greek vase sculpture"),
    ("⚔️ Medieval",         "medieval sword knight armor"),
    ("🖼️ Viking Poster",    "Viking poster 1920"),
]


# ── Loaders (cached) ──────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="모델 로딩 중...")
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_model, _ = clip.load("ViT-B/32", device=device)

    config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["c_fc", "c_proj"],
        lora_dropout=0.1,
    )
    model = get_peft_model(base_model, config)

    weights_path = "./museum_lora_weights.pt"
    if os.path.exists(weights_path):
        state = torch.load(weights_path, map_location=device)
        model.load_state_dict(state, strict=False)

    model.eval()
    return model, device


@st.cache_resource(show_spinner="데이터 로딩 중...")
def load_data():
    items = pd.read_csv("./items.csv")
    items["text_data"] = (
        items["title"].fillna("") + ". " + items["description"].fillna("")
    ).str.strip(". ")
    return items


@st.cache_resource(show_spinner="검색 인덱스 구축 중 (최초 1회)...")
def build_index(_model, _device, _items):
    """
    [핵심 수정] encode_text → L2 정규화 → IndexFlatIP (코사인 유사도)
    기존 IndexFlatL2는 비정규화 벡터에 부적합 — 유사도 순위가 뒤집힌 이유
    """
    batch_size = 128
    all_vecs = []

    with torch.no_grad():
        for i in range(0, len(_items), batch_size):
            batch = _items["text_data"].iloc[i : i + batch_size].tolist()
            tokens = clip.tokenize(batch, truncate=True).to(_device)
            embs = _model.model.encode_text(tokens).float()
            # 정규화: ||v|| = 1  →  dot product == cosine similarity
            embs = embs / embs.norm(dim=-1, keepdim=True)
            all_vecs.append(embs.cpu().numpy())

    vectors = np.vstack(all_vecs).astype("float32")
    dim = vectors.shape[1]
    idx = faiss.IndexFlatIP(dim)   # Inner Product on normalized → cosine
    idx.add(vectors)
    return idx, vectors


# ── Search ────────────────────────────────────────────────────────────────────
def search(query: str, k: int, modality: str, model, device, items, index, stored_vecs):
    t0 = time.time()

    with torch.no_grad():
        tokens = clip.tokenize([query], truncate=True).to(device)
        q_vec = model.model.encode_text(tokens).float()
        q_vec = q_vec / q_vec.norm(dim=-1, keepdim=True)
        q_np = q_vec.cpu().numpy().astype("float32")

    if modality != "All":
        mask = items["modality"] == modality.lower()
        filtered_idx = items.index[mask].tolist()
        sub_vecs = stored_vecs[filtered_idx]
        sub_index = faiss.IndexFlatIP(sub_vecs.shape[1])
        sub_index.add(sub_vecs.astype("float32"))
        scores, local_idx = sub_index.search(q_np, k)
        result_rows = items.iloc[[filtered_idx[i] for i in local_idx[0]]].copy()
    else:
        scores, raw_idx = index.search(q_np, k)
        result_rows = items.iloc[raw_idx[0]].copy()

    result_rows["similarity"] = scores[0][: len(result_rows)]
    fps = 1 / (time.time() - t0)
    return result_rows, fps


# ── Init ──────────────────────────────────────────────────────────────────────
model, device = load_model()
items = load_data()
index, stored_vecs = build_index(model, device, items)

lora_loaded = os.path.exists("./museum_lora_weights.pt")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    try:
        st.image(
            "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4b/Pushkin_Museum_logo.svg/1200px-Pushkin_Museum_logo.svg.png",
            width=130,
        )
    except Exception:
        pass

    st.markdown("## 🏛️ Museum Search")
    if lora_loaded:
        st.success("✅ Fine-tuned LoRA 적용됨")
    else:
        st.warning("⚠️ Base CLIP 사용 중\n\n`python train.py` 실행 후 재시작하면 검색 품질이 향상됩니다.")

    st.divider()

    st.markdown("### ⚙️ 검색 설정")
    top_k = st.slider("결과 수", 4, 20, 8, step=2)
    modality_filter = st.selectbox("유형 필터", ["All", "Image", "Video", "Audio", "Article"])

    st.divider()

    st.markdown("### 📊 컬렉션 현황")
    col1, col2 = st.columns(2)
    col1.metric("총 에셋", f"{len(items):,}")
    col2.metric("디바이스", device.upper())

    for mod, cnt in items["modality"].value_counts().items():
        icon = MODALITY_ICON.get(mod, "📦")
        pct = cnt / len(items) * 100
        st.markdown(f"`{icon} {mod.capitalize()}` — **{cnt:,}** ({pct:.0f}%)")

    st.divider()
    fps_slot = st.empty()

# ── Main Layout ───────────────────────────────────────────────────────────────
st.title("🏛️ Museum Digital Asset Search")
st.markdown(
    "CLIP + LoRA 기반 시맨틱 검색 — Pushkin Museum 7,500+ 에셋 (영상, 오디오, 이미지, 아티클)"
)
st.divider()

query_input = st.text_input(
    "search",
    placeholder="🔍  검색어 입력  (예: ancient Egypt ritual · 19세기 초상화 · Viking poster 1920)",
    label_visibility="collapsed",
)

# 추천 검색어 버튼
btn_cols = st.columns(len(SUGGESTED))
for i, (label, q_text) in enumerate(SUGGESTED):
    if btn_cols[i].button(label, use_container_width=True):
        query_input = q_text
        st.rerun()

st.markdown("")  # spacer

# ── Results ───────────────────────────────────────────────────────────────────
if query_input:
    with st.spinner("검색 중..."):
        results, fps = search(
            query_input, top_k, modality_filter,
            model, device, items, index, stored_vecs,
        )

    fps_slot.metric("⚡ 검색 속도", f"{fps:.1f} FPS")

    filter_label = f" ({modality_filter})" if modality_filter != "All" else ""
    st.subheader(f"`{query_input}` 검색 결과 {len(results)}건{filter_label}")

    # 모달리티 분포
    breakdown = results["modality"].value_counts()
    if len(breakdown):
        bc = st.columns(len(breakdown))
        for i, (mod, cnt) in enumerate(breakdown.items()):
            bc[i].metric(f"{MODALITY_ICON.get(mod,'📦')} {mod.capitalize()}", cnt)
    st.markdown("---")

    # 결과 카드 (2열)
    left, right = st.columns(2)
    for idx, (_, row) in enumerate(results.iterrows()):
        col = left if idx % 2 == 0 else right
        with col:
            sim = float(row["similarity"])
            sim_pct = int(max(0, min(1, (sim + 1) / 2)) * 100)  # [-1,1] → [0,100]%
            bar_w = max(4, sim_pct)
            modality = str(row["modality"])
            icon = MODALITY_ICON.get(modality, "📦")
            badge_cls = MODALITY_COLOR.get(modality, "badge-image")

            with st.container(border=True):
                # 배지 + 유사도
                c1, c2 = st.columns([3, 1])
                c1.markdown(
                    f'<span class="badge {badge_cls}">{icon} {modality.upper()}</span>',
                    unsafe_allow_html=True,
                )
                c2.markdown(f"**{sim_pct}%**")

                # 유사도 바
                st.markdown(
                    f'<div class="sim-bar-wrap">'
                    f'<div class="sim-bar-fill" style="width:{bar_w}%"></div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                # 이미지 썸네일 (image 유형이고 URL 있을 때)
                extra = row.get("extra")
                if modality == "image" and pd.notna(extra) and str(extra).startswith("http"):
                    try:
                        st.image(str(extra), use_container_width=True)
                    except Exception:
                        pass

                # 제목
                title = row["title"] if pd.notna(row["title"]) else "Untitled"
                st.markdown(f"**{title}**")

                # 설명 (펼침)
                desc = row["description"] if pd.notna(row["description"]) else ""
                if desc:
                    with st.expander("설명 보기"):
                        st.write(desc)

                # 하단 메타 + 링크
                fc1, fc2 = st.columns([1, 1])
                fc1.caption(f"`{row['item_id']}`")
                if pd.notna(extra) and modality != "image":
                    fc2.link_button("🔗 원문 보기", str(extra), use_container_width=True)
                elif pd.notna(extra) and modality == "image":
                    fc2.link_button("🔗 고화질", str(extra), use_container_width=True)

# ── Empty State ───────────────────────────────────────────────────────────────
else:
    st.markdown(
        """
<div style="text-align:center;padding:64px 20px 32px;color:#9ea7b3">
    <div style="font-size:72px;margin-bottom:16px">🏛️</div>
    <h2 style="color:#3d4452;margin-bottom:8px">Pushkin Museum Collection</h2>
    <p style="font-size:16px;max-width:500px;margin:0 auto">
        자연어로 검색하면 AI가 의미를 분석해 가장 관련 있는 영상, 오디오, 이미지, 아티클을 찾아드립니다.
    </p>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown("### 📈 컬렉션 구성")
    mc = st.columns(4)
    for i, (mod, cnt) in enumerate(items["modality"].value_counts().items()):
        icon = MODALITY_ICON.get(mod, "📦")
        pct = cnt / len(items) * 100
        mc[i % 4].metric(
            f"{icon} {mod.capitalize()}",
            f"{cnt:,}",
            f"{pct:.1f}%",
        )
