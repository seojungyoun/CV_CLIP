"""
Museum CLIP Fine-tuning with Contrastive Loss
train.csv의 query-item 쌍을 사용해 LoRA로 경량 파인튜닝
"""
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
import os


# ── Dataset ───────────────────────────────────────────────────────────────────
class MuseumPairDataset(Dataset):
    """train.csv의 (query_item, positive_item) 텍스트 쌍 데이터셋"""

    def __init__(self, train_df: pd.DataFrame, item_text: dict):
        self.pairs = []
        skipped = 0
        for _, row in train_df.iterrows():
            q = item_text.get(row["query_id"], "").strip()
            p = item_text.get(row["item_id"], "").strip()
            if len(q) > 5 and len(p) > 5:
                self.pairs.append((q, p))
            else:
                skipped += 1
        print(f"유효 쌍: {len(self.pairs):,}개 | 스킵: {skipped}개")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]


# ── Loss ──────────────────────────────────────────────────────────────────────
def contrastive_loss(q_embs: torch.Tensor, p_embs: torch.Tensor, temperature: float = 0.07):
    """대칭 InfoNCE loss (CLIP 원본 방식)"""
    q_embs = F.normalize(q_embs.float(), dim=-1)
    p_embs = F.normalize(p_embs.float(), dim=-1)
    logits = (q_embs @ p_embs.T) / temperature
    labels = torch.arange(len(q_embs), device=q_embs.device)
    loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
    return loss


# ── Main ──────────────────────────────────────────────────────────────────────
def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # 데이터 로드
    items = pd.read_csv("./items.csv")
    # title만 쓰면 영어 텍스트 비율이 더 높아짐(특히 video/article)
    # description은 러시아어가 많아 노이즈가 될 수 있으므로 title 우선
    items["text_data"] = items["title"].fillna("") + ". " + items["description"].fillna("")
    items["text_data"] = items["text_data"].str.strip(". ")
    item_text = dict(zip(items["item_id"], items["text_data"]))

    train_df = pd.read_csv("./train.csv")

    # 모델 로드 + LoRA 적용
    model, _ = clip.load("ViT-B/32", device=device)
    # CLIP의 MultiheadAttention은 out_proj.weight를 직접 추출해 F.multi_head_attention_forward에 넘기므로
    # LoRA 래퍼의 forward()가 호출되지 않아 그래디언트가 없음 → MLP 레이어(c_fc, c_proj)를 타겟으로 변경
    config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["c_fc", "c_proj"],
        lora_dropout=0.1,
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    # 데이터셋
    dataset = MuseumPairDataset(train_df, item_text)
    loader = DataLoader(dataset, batch_size=64, shuffle=True, drop_last=True)

    # 옵티마이저 (LoRA 파라미터만)
    lora_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(lora_params, lr=2e-4, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=len(loader) * 5)

    # 학습
    best_loss = float("inf")
    num_epochs = 5

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        t0 = time.time()

        for batch_idx, (queries, positives) in enumerate(loader):
            q_tokens = clip.tokenize(list(queries), truncate=True).to(device)
            p_tokens = clip.tokenize(list(positives), truncate=True).to(device)

            q_embs = model.model.encode_text(q_tokens)
            p_embs = model.model.encode_text(p_tokens)

            loss = contrastive_loss(q_embs, p_embs)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

            if batch_idx % 30 == 0:
                elapsed = time.time() - t0
                print(
                    f"Epoch {epoch+1}/{num_epochs} | "
                    f"Batch {batch_idx}/{len(loader)} | "
                    f"Loss: {loss.item():.4f} | "
                    f"Time: {elapsed:.1f}s"
                )

        avg_loss = total_loss / len(loader)
        print(f"\n[Epoch {epoch+1}] Avg Loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "./museum_lora_weights.pt")
            print("✅ Best model saved → museum_lora_weights.pt\n")

    print(f"학습 완료! 최종 best loss: {best_loss:.4f}")


if __name__ == "__main__":
    train()
