"""
CLIP-LoRA Met Museum Semantic Search — Streamlit App
데이터: Metropolitan Museum of Art Open Access (470,000+ objects)
이미지: Is Public Domain == True 인 작품은 Met API로 썸네일 로드
"""
import os
import time

import clip
import faiss
import numpy as np
import pandas as pd
import requests
import streamlit as st
import torch
from peft import LoraConfig, get_peft_model

CSV_PATH = "./MetObjects.csv"
WEIGHTS_PATH = "./museum_lora_weights.pt"
MET_API = "https://collectionapi.metmuseum.org/public/collection/v1/objects/{}"

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Met Museum Search",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

div[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 12px !important;
    transition: box-shadow 0.2s;
}
div[data-testid="stVerticalBlockBorderWrapper"]:hover {
    box-shadow: 0 6px 20px rgba(0,0,0,0.10) !important;
}

.dept-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    background: #f0f4ff;
    color: #3355cc;
    border: 1px solid #c0ccee;
}
.highlight-badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 20px;
    font-size: 10px;
    font-weight: 700;
    background: #fff3cd;
    color: #856404;
    border: 1px solid #ffc107;
    margin-left: 6px;
}
.sim-bar-wrap { background:#e9ecef; border-radius:4px; height:4px; margin:5px 0 10px; overflow:hidden; }
.sim-bar-fill  { height:100%; border-radius:4px; background: linear-gradient(90deg,#3355cc,#22c55e); }

div[data-testid="stTextInput"] input {
    font-size: 18px !important;
    padding: 14px 18px !important;
    border-radius: 10px !important;
}
div[data-testid="metric-container"] {
    background: #f8f9fa; border-radius: 8px; padding: 8px 12px;
}
section[data-testid="stSidebar"] > div { padding-top: 1.5rem; }
</style>
""", unsafe_allow_html=True)

# ── 부서 → 이모지 매핑 ─────────────────────────────────────────────────────
DEPT_ICON = {
    "Egyptian Art": "𓂀",
    "Greek and Roman Art": "🏺",
    "European Paintings": "🖼️",
    "Drawings and Prints": "✏️",
    "Photographs": "📷",
    "Asian Art": "🏯",
    "The American Wing": "🗽",
    "Modern and Contemporary Art": "🎨",
    "Arms and Armor": "⚔️",
    "Medieval Art": "⛪",
    "Islamic Art": "🕌",
    "The Cloisters": "🏰",
    "Musical Instruments": "🎵",
    "Arts of Africa, Oceania, and the Americas": "🌍",
    "The Costume Institute": "👗",
    "Robert Lehman Collection": "🖼️",
    "Libraries": "📚",
}

SUGGESTED = [
    ("𓂀 Egypt", "ancient Egyptian mummy sarcophagus gold"),
    ("🏺 Greece", "ancient Greek vase sculpture marble"),
    ("🖼️ Portrait", "Renaissance oil portrait nobleman"),
    ("📷 Photo", "black and white street photography 20th century"),
    ("⚔️ Armor", "medieval knight sword armor battle"),
    ("🎨 Modern", "abstract expressionism modern painting"),
]


# ── 모델 로드 ─────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="🔧 모델 로딩 중...")
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base, _ = clip.load("ViT-B/32", device=device)
    cfg = LoraConfig(r=16, lora_alpha=32, target_modules=["c_fc", "c_proj"], lora_dropout=0.1)
    model = get_peft_model(base, cfg)
    if os.path.exists(WEIGHTS_PATH):
        state = torch.load(WEIGHTS_PATH, map_location=device)
        model.load_state_dict(state, strict=False)
    model.eval()
    return model, device


# ── 데이터 로드 ───────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="📂 Met Museum 데이터 로딩 중 (~317MB)...")
def load_data():
    if not os.path.exists(CSV_PATH):
        st.error(f"`{CSV_PATH}` 파일이 없습니다. 먼저 `python download_data.py`를 실행하세요.")
        st.stop()

    df = pd.read_csv(CSV_PATH, low_memory=False)
    df = df[df["Title"].notna() & (df["Title"].str.strip() != "")].copy()
    df["Object ID"] = df["Object ID"].astype(str)

    def build_text(row):
        parts = [str(row["Title"]).strip()]
        for col, prefix in [
            ("Artist Display Name", "by "),
            ("Object Date", ""),
            ("Medium", ""),
            ("Culture", ""),
            ("Department", ""),
            ("Classification", ""),
            ("Tags", ""),
        ]:
            val = row.get(col)
            if pd.notna(val) and str(val).strip():
                parts.append(prefix + str(val).strip())
        return " | ".join(parts)

    df["text_data"] = df.apply(build_text, axis=1)
    return df.reset_index(drop=True)


# ── FAISS 인덱스 ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="⚙️ 검색 인덱스 구축 중 (최초 1회, 수분 소요)...")
def build_index(_model, _device, _df):
    batch_size = 256
    all_vecs = []
    n = len(_df)

    prog = st.progress(0, "벡터화 중...")
    with torch.no_grad():
        for i in range(0, n, batch_size):
            batch = _df["text_data"].iloc[i: i + batch_size].tolist()
            tok = clip.tokenize(batch, truncate=True).to(_device)
            emb = _model.model.encode_text(tok).float()
            emb = emb / emb.norm(dim=-1, keepdim=True)
            all_vecs.append(emb.cpu().numpy())
            prog.progress(min((i + batch_size) / n, 1.0))

    prog.empty()
    vecs = np.vstack(all_vecs).astype("float32")
    idx = faiss.IndexFlatIP(vecs.shape[1])
    idx.add(vecs)
    return idx, vecs


# ── Met API 이미지 (공공도메인만) ─────────────────────────────────────────────
@st.cache_data(ttl=86400, show_spinner=False)
def fetch_image_url(object_id: str) -> str:
    try:
        r = requests.get(MET_API.format(object_id), timeout=3)
        if r.ok:
            return r.json().get("primaryImageSmall", "")
    except Exception:
        pass
    return ""


# ── 검색 ──────────────────────────────────────────────────────────────────────
def search(query, k, dept_filter, model, device, df, index, vecs):
    t0 = time.time()
    with torch.no_grad():
        tok = clip.tokenize([query], truncate=True).to(device)
        qv = model.model.encode_text(tok).float()
        qv = qv / qv.norm(dim=-1, keepdim=True)
        qnp = qv.cpu().numpy().astype("float32")

    if dept_filter != "All":
        mask = df["Department"] == dept_filter
        idxs = df.index[mask].tolist()
        if not idxs:
            return pd.DataFrame(), 0.0
        sub = vecs[idxs].astype("float32")
        si = faiss.IndexFlatIP(sub.shape[1])
        si.add(sub)
        scores, local = si.search(qnp, k)
        rows = df.iloc[[idxs[i] for i in local[0]]].copy()
    else:
        scores, raw = index.search(qnp, k)
        rows = df.iloc[raw[0]].copy()

    rows["_score"] = scores[0][: len(rows)]
    fps = 1 / (time.time() - t0)
    return rows, fps


# ── 초기화 ────────────────────────────────────────────────────────────────────
model, device = load_model()
df = load_data()
index, stored_vecs = build_index(model, device, df)

lora_ok = os.path.exists(WEIGHTS_PATH)
departments = ["All"] + sorted(df["Department"].dropna().unique().tolist())

# ── 사이드바 ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏛️ Met Museum Search")
    if lora_ok:
        st.success("✅ Fine-tuned LoRA 적용됨")
    else:
        st.warning("⚠️ Base CLIP\n\n`python train.py` 후 재시작하면 품질 향상")

    st.divider()
    st.markdown("### ⚙️ 검색 설정")
    top_k = st.slider("결과 수", 4, 20, 8, step=2)
    dept_filter = st.selectbox("부서 필터", departments)
    show_images = st.toggle("이미지 로드 (공공도메인, 느림)", value=False)

    st.divider()
    st.markdown("### 📊 컬렉션")
    col1, col2 = st.columns(2)
    col1.metric("총 작품", f"{len(df):,}")
    col2.metric("디바이스", device.upper())

    pub = df["Is Public Domain"].sum() if "Is Public Domain" in df.columns else 0
    st.caption(f"공공도메인: {pub:,}개")

    fps_slot = st.empty()

# ── 메인 ──────────────────────────────────────────────────────────────────────
st.title("🏛️ Metropolitan Museum of Art — Semantic Search")
st.markdown("CLIP + LoRA 기반 자연어 검색 · **470,000+** 작품 · 이집트, 그리스, 유럽 회화, 사진, 현대미술 등")
st.divider()

query_input = st.text_input(
    "search",
    placeholder="🔍  예: ancient Egyptian gold jewelry · Impressionist Paris street · Japanese woodblock print",
    label_visibility="collapsed",
)

# 추천 검색어 버튼
btn_cols = st.columns(len(SUGGESTED))
for i, (label, q) in enumerate(SUGGESTED):
    if btn_cols[i].button(label, use_container_width=True):
        query_input = q
        st.rerun()

st.markdown("")

# ── 결과 ──────────────────────────────────────────────────────────────────────
if query_input:
    with st.spinner("검색 중..."):
        results, fps = search(query_input, top_k, dept_filter,
                              model, device, df, index, stored_vecs)

    if results.empty:
        st.warning("결과 없음 — 필터를 바꿔보세요.")
    else:
        fps_slot.metric("⚡ 검색 속도", f"{fps:.1f} FPS")

        dept_label = f" ({dept_filter})" if dept_filter != "All" else ""
        st.subheader(f"`{query_input}` — {len(results)}건{dept_label}")

        # 부서 분포
        breakdown = results["Department"].value_counts()
        if len(breakdown):
            bc = st.columns(min(len(breakdown), 4))
            for i, (dept, cnt) in enumerate(breakdown.items()):
                icon = DEPT_ICON.get(dept, "🏛️")
                bc[i % 4].metric(f"{icon} {dept[:20]}", cnt)
        st.markdown("---")

        left, right = st.columns(2)
        for idx, (_, row) in enumerate(results.iterrows()):
            col = left if idx % 2 == 0 else right
            with col:
                score = float(row["_score"])
                sim_pct = int(max(0, min(1, (score + 1) / 2)) * 100)
                dept = str(row.get("Department", "")) or "Unknown"
                icon = DEPT_ICON.get(dept, "🏛️")
                is_highlight = str(row.get("Is Highlight", "")).lower() == "true"
                is_public = str(row.get("Is Public Domain", "")).lower() == "true"

                with st.container(border=True):
                    # 배지 행
                    c1, c2 = st.columns([4, 1])
                    badge_html = f'<span class="dept-badge">{icon} {dept[:30]}</span>'
                    if is_highlight:
                        badge_html += '<span class="highlight-badge">⭐ Highlight</span>'
                    c1.markdown(badge_html, unsafe_allow_html=True)
                    c2.markdown(f"**{sim_pct}%**")

                    # 유사도 바
                    st.markdown(
                        f'<div class="sim-bar-wrap">'
                        f'<div class="sim-bar-fill" style="width:{max(4,sim_pct)}%"></div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                    # 이미지 (공공도메인 + 토글 ON)
                    obj_id = str(row.get("Object ID", ""))
                    if show_images and is_public and obj_id:
                        img_url = fetch_image_url(obj_id)
                        if img_url:
                            st.image(img_url, use_container_width=True)

                    # 제목
                    title = str(row.get("Title", "Untitled")).strip()
                    st.markdown(f"**{title}**")

                    # 메타 정보
                    meta_parts = []
                    for col_name, label in [
                        ("Artist Display Name", "Artist"),
                        ("Object Date", "Date"),
                        ("Medium", "Medium"),
                        ("Culture", "Culture"),
                        ("Classification", "Type"),
                    ]:
                        val = row.get(col_name)
                        if pd.notna(val) and str(val).strip():
                            meta_parts.append(f"**{label}:** {str(val).strip()}")
                    if meta_parts:
                        st.caption("  ·  ".join(meta_parts[:3]))
                        if len(meta_parts) > 3:
                            with st.expander("더 보기"):
                                for mp in meta_parts[3:]:
                                    st.caption(mp)

                    # Tags
                    tags = row.get("Tags")
                    if pd.notna(tags) and str(tags).strip():
                        tag_list = [t.strip() for t in str(tags).split("|") if t.strip()]
                        if tag_list:
                            st.caption("🏷️ " + "  ·  ".join(tag_list[:5]))

                    # 링크
                    link = row.get("Link Resource")
                    fc1, fc2 = st.columns([1, 1])
                    fc1.caption(f"`ID {obj_id}`")
                    if pd.notna(link) and str(link).startswith("http"):
                        fc2.link_button("🔗 Met Museum", str(link), use_container_width=True)

# ── Empty State ───────────────────────────────────────────────────────────────
else:
    st.markdown("""
<div style="text-align:center;padding:60px 20px 32px;color:#9ea7b3">
    <div style="font-size:72px;margin-bottom:16px">🏛️</div>
    <h2 style="color:#3d4452;margin-bottom:8px">Metropolitan Museum of Art</h2>
    <p style="font-size:16px;max-width:520px;margin:0 auto;line-height:1.6">
        자연어로 입력하면 AI가 의미를 분석해<br>
        470,000개 이상의 작품 중 가장 관련 있는 결과를 찾아드립니다.
    </p>
</div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📈 컬렉션 구성")
    dept_counts = df["Department"].value_counts().head(8)
    cols = st.columns(4)
    for i, (dept, cnt) in enumerate(dept_counts.items()):
        icon = DEPT_ICON.get(dept, "🏛️")
        cols[i % 4].metric(f"{icon} {dept[:22]}", f"{cnt:,}")
