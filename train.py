"""
Met Museum CLIP Fine-tuning — Self-Supervised Contrastive Learning
라벨 없이 각 작품의 두 가지 텍스트 뷰를 positive pair로 사용:
  View A (anchor)  : Title
  View B (positive): "Medium by Artist, Culture, Period, Department"
같은 작품을 가리키는 두 설명 → embedding space에서 가깝게 학습
"""
import os
import pandas as pd
import torch
import torch.nn.functional as F
import clip
import numpy as np
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import time

CSV_PATH = "./MetObjects.csv"
WEIGHTS_PATH = "./museum_lora_weights.pt"


# ── 데이터 로드 & 전처리 ──────────────────────────────────────────────────────
def load_met_data(csv_path: str = CSV_PATH) -> pd.DataFrame:
    print("CSV 로딩 중... (~317MB, 잠시 대기)")
    df = pd.read_csv(csv_path, low_memory=False)
    print(f"전체 오브젝트: {len(df):,}개")

    # 제목 없는 행 제거
    df = df[df["Title"].notna() & (df["Title"].str.strip() != "")].copy()

    # 풍부한 텍스트 생성
    def build_text(row):
        parts = [str(row["Title"]).strip()]
        if pd.notna(row.get("Artist Display Name")) and str(row["Artist Display Name"]).strip():
            parts.append(f"by {str(row['Artist Display Name']).strip()}")
        if pd.notna(row.get("Object Date")) and str(row["Object Date"]).strip():
            parts.append(str(row["Object Date"]).strip())
        if pd.notna(row.get("Medium")) and str(row["Medium"]).strip():
            parts.append(str(row["Medium"]).strip())
        if pd.notna(row.get("Culture")) and str(row["Culture"]).strip():
            parts.append(str(row["Culture"]).strip())
        if pd.notna(row.get("Department")) and str(row["Department"]).strip():
            parts.append(str(row["Department"]).strip())
        if pd.notna(row.get("Classification")) and str(row["Classification"]).strip():
            parts.append(str(row["Classification"]).strip())
        if pd.notna(row.get("Tags")) and str(row["Tags"]).strip():
            parts.append(str(row["Tags"]).strip())
        return " | ".join(parts)

    df["text_data"] = df.apply(build_text, axis=1)

    # Self-supervised pair용 메타데이터 뷰
    def build_meta(row):
        parts = []
        if pd.notna(row.get("Medium")) and str(row["Medium"]).strip():
            parts.append(str(row["Medium"]).strip())
        if pd.notna(row.get("Artist Display Name")) and str(row["Artist Display Name"]).strip():
            parts.append(f"by {str(row['Artist Display Name']).strip()}")
        if pd.notna(row.get("Culture")) and str(row["Culture"]).strip():
            parts.append(str(row["Culture"]).strip())
        if pd.notna(row.get("Period")) and str(row["Period"]).strip():
            parts.append(str(row["Period"]).strip())
        if pd.notna(row.get("Department")) and str(row["Department"]).strip():
            parts.append(str(row["Department"]).strip())
        return " | ".join(parts) if parts else str(row["Title"]).strip()

    df["meta_view"] = df.apply(build_meta, axis=1)
    print(f"유효 오브젝트: {len(df):,}개")
    return df


# ── Dataset ───────────────────────────────────────────────────────────────────
class MetPairDataset(Dataset):
    """
    Self-supervised: (Title) ↔ (Medium + Artist + Culture + Period + Department)
    같은 작품의 두 텍스트 뷰를 positive pair로 학습
    """
    def __init__(self, df: pd.DataFrame, max_pairs: int = 100_000):
        pairs = [
            (str(row["Title"]).strip(), str(row["meta_view"]).strip())
            for _, row in df.iterrows()
            if len(str(row["Title"]).strip()) > 3
            and len(str(row["meta_view"]).strip()) > 3
        ]
        self.pairs = pairs[:max_pairs]
        print(f"학습 쌍: {len(self.pairs):,}개")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]


# ── Loss ──────────────────────────────────────────────────────────────────────
def contrastive_loss(q: torch.Tensor, p: torch.Tensor, temperature: float = 0.07):
    """대칭 InfoNCE (CLIP 방식)"""
    q = F.normalize(q.float(), dim=-1)
    p = F.normalize(p.float(), dim=-1)
    logits = (q @ p.T) / temperature
    labels = torch.arange(len(q), device=q.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2


# ── Main ──────────────────────────────────────────────────────────────────────
def train():
    if not os.path.exists(CSV_PATH):
        print(f"{CSV_PATH} 없음 → 먼저 python download_data.py 실행")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    df = load_met_data()

    # 모델 로드 + LoRA (MLP 레이어 타겟 — attention은 weight 직접 추출로 LoRA 우회됨)
    base_model, _ = clip.load("ViT-B/32", device=device)
    config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["c_fc", "c_proj"],
        lora_dropout=0.1,
    )
    model = get_peft_model(base_model, config)
    model.print_trainable_parameters()

    # CPU/GPU 환경에 맞게 자동 조정
    if device == "cuda":
        max_pairs, batch_size, num_epochs = 100_000, 64, 5
    else:
        # CPU: 10,000쌍 × 3에폭 ≈ 20~30분
        max_pairs, batch_size, num_epochs = 10_000, 32, 3
        print("CPU 감지 → 경량 설정 적용 (10,000쌍 / batch 32 / 3 에폭, 약 20~30분)")

    dataset = MetPairDataset(df, max_pairs=max_pairs)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    lora_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(lora_params, lr=2e-4, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=len(loader) * num_epochs)

    best_loss = float("inf")

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        t0 = time.time()

        for i, (titles, metas) in enumerate(loader):
            q_tok = clip.tokenize(list(titles), truncate=True).to(device)
            p_tok = clip.tokenize(list(metas), truncate=True).to(device)

            q_emb = model.model.encode_text(q_tok)
            p_emb = model.model.encode_text(p_tok)

            loss = contrastive_loss(q_emb, p_emb)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

            if i % 30 == 0:
                elapsed = time.time() - t0
                print(f"Epoch {epoch+1}/5 | Batch {i}/{len(loader)} | Loss {loss.item():.4f} | {elapsed:.0f}s")

        avg = total_loss / len(loader)
        print(f"[Epoch {epoch+1}] Avg Loss: {avg:.4f}")

        if avg < best_loss:
            best_loss = avg
            torch.save(model.state_dict(), WEIGHTS_PATH)
            print(f"  Best model saved → {WEIGHTS_PATH}")

    print(f"학습 완료! Best Loss: {best_loss:.4f}")


if __name__ == "__main__":
    train()
