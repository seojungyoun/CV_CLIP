# train.py
"""
CLIP contrastive fine-tuning on Met Museum public-domain image-text pairs.
학습 범위: text transformer 마지막 블록 + text/image projection만 업데이트 (VRAM 절약)
사용법: python train.py --epochs 3 --batch-size 64
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm import tqdm

import open_clip
import config


# ── 데이터셋 ──────────────────────────────────────────────────────────────
class MetDataset(Dataset):
    """image_path + caption 쌍을 반환하는 Met 데이터셋"""

    def __init__(self, frame: pd.DataFrame, preprocess):
        self.frame = frame.reset_index(drop=True)
        self.preprocess = preprocess

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        row = self.frame.iloc[idx]
        img_path = str(row["image_path"])
        try:
            if img_path.startswith("http"):
                import requests, io
                resp = requests.get(img_path, timeout=5,
                                    headers={"User-Agent": "Mozilla/5.0"})
                img = Image.open(io.BytesIO(resp.content)).convert("RGB")
            else:
                img = Image.open(img_path).convert("RGB")
            image = self.preprocess(img)
        except Exception:
            image = self.preprocess(Image.new("RGB", (224, 224), (128, 128, 128)))

        caption = str(row["caption"])
        return image, caption


# ── [🚨 에러 해결 핵심: Windows 직렬화(Pickle)를 위해 구조화된 클래스로 전역 분리] ──
class WindowsSecureCollate:
    """내포 함수(_collate)를 제거하고 전역 최상단 인스턴스로 바인딩하여 복사를 지원하는 클래스"""
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        images, captions = zip(*batch)
        images = torch.stack(images)
        tokens = self.tokenizer(list(captions), context_length=77)
        return images, tokens


# ── 학습 범위 설정 ─────────────────────────────────────────────────────────
def set_trainable_params(model):
    """전체 동결 후 text/image 마지막 블록 + projection만 해제"""
    for p in model.parameters():
        p.requires_grad = False

    # Text transformer 마지막 블록 + projection
    for name, p in model.named_parameters():
        if any(k in name for k in [
            "transformer.resblocks.11",   # ViT-B-32 text: 12 blocks (0-11)
            "text_projection",
            "visual.transformer.resblocks.11",
            "visual.proj",
        ]):
            p.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"🔧 학습 파라미터: {trainable:,} / {total:,} ({trainable/total*100:.1f}%)")


# ── Contrastive loss ───────────────────────────────────────────────────────
def clip_loss(image_features, text_features, logit_scale):
    """대칭 cross-entropy contrastive loss (InfoNCE)"""
    logits_per_image = logit_scale * image_features @ text_features.T
    logits_per_text  = logits_per_image.T
    labels = torch.arange(len(image_features), device=image_features.device)
    loss = (F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text,  labels)) / 2
    return loss


# ── 메인 학습 루프 ────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",     type=int,   default=3)
    parser.add_argument("--batch-size", type=int,   default=64)
    parser.add_argument("--lr",         type=float, default=1e-5)
    parser.add_argument("--workers",    type=int,   default=4)
    parser.add_argument("--limit",      type=int,   default=None,
                        help="디버그용 데이터 상한 (기본: 전체)")
    args = parser.parse_args()

    # ── 환경 준비
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  학습 디바이스: {device}")
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    config.ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 데이터 로드
    if not config.CLEAN_DATA_CSV.exists():
        raise FileNotFoundError(
            f"❌ {config.CLEAN_DATA_CSV} 없음. 먼저 python prepare_data.py 실행하세요."
        )
    frame = pd.read_csv(config.CLEAN_DATA_CSV)
    if args.limit:
        frame = frame.head(args.limit)

    train_df = frame[frame["split"] == "train"].reset_index(drop=True)
    valid_df = frame[frame["split"] == "valid"].reset_index(drop=True)
    print(f"📦 Train: {len(train_df):,}  |  Valid: {len(valid_df):,}")

    # ── 모델 로드
    model, _, preprocess = open_clip.create_model_and_transforms(
        config.MODEL_NAME, pretrained=config.PRETRAINED, device=device
    )
    tokenizer = open_clip.get_tokenizer(config.MODEL_NAME)

    # 이전 체크포인트 복원 (재시작 지원)
    start_epoch = 0
    if config.LORA_WEIGHTS_PATH.exists():
        print(f"♻️  기존 체크포인트 발견, 이어서 학습합니다: {config.LORA_WEIGHTS_PATH}")
        state = torch.load(config.LORA_WEIGHTS_PATH, map_location="cpu", weights_only=True)
        model.load_state_dict(state.get("model", state), strict=False)
        start_epoch = state.get("epoch", 0)

    set_trainable_params(model)
    model.train()

    # ── DataLoader [💡 튜닝 완료: 교정된 안전 콜레이트 오프라인 개체 주입] ──
    secure_collate = WindowsSecureCollate(tokenizer)
    
    train_loader = DataLoader(
        train_ds := MetDataset(train_df, preprocess), batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=(device.type == "cuda"),
        collate_fn=secure_collate, drop_last=True,
    )
    valid_loader = DataLoader(
        valid_ds := MetDataset(valid_df, preprocess), batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=(device.type == "cuda"),
        collate_fn=secure_collate,
    )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr
    )
    
    # PyTorch 2.x 이상 최신 버전 스케일러 호환성 마이그레이션 패치
    try:
        scaler = torch.amp.GradScaler('cuda', enabled=(device.type == "cuda"))
    except:
        scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    best_val_loss = float("inf")
    history = []

    for epoch in range(start_epoch, start_epoch + args.epochs):
        # ── Train
        model.train()
        t0 = time.time()
        running_loss = 0.0

        for images, tokens in tqdm(train_loader, desc=f"Epoch {epoch+1} train"):
            images = images.to(device, non_blocking=True)
            tokens = tokens.to(device, non_blocking=True)

            optimizer.zero_grad()
            
            # 컨텍스트 자동 캐스팅 마이그레이션 적용
            with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
                image_feat = F.normalize(model.encode_image(images).float(), dim=-1)
                text_feat  = F.normalize(model.encode_text(tokens).float(),  dim=-1)
                loss = clip_loss(image_feat, text_feat, model.logit_scale.exp())

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)

        # ── Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, tokens in tqdm(valid_loader, desc=f"Epoch {epoch+1} valid"):
                images = images.to(device, non_blocking=True)
                tokens = tokens.to(device, non_blocking=True)
                with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
                    image_feat = F.normalize(model.encode_image(images).float(), dim=-1)
                    text_feat  = F.normalize(model.encode_text(tokens).float(),  dim=-1)
                    val_loss  += clip_loss(image_feat, text_feat, model.logit_scale.exp()).item()
        val_loss /= max(len(valid_loader), 1)

        elapsed = time.time() - t0
        print(f"  Epoch {epoch+1:02d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | {elapsed:.0f}s")
        history.append({"epoch": epoch+1, "train_loss": train_loss, "val_loss": val_loss})

        # ── 체크포인트 저장 (val_loss 기준 best)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {"epoch": epoch + 1, "model": model.state_dict(), "val_loss": val_loss},
                config.LORA_WEIGHTS_PATH,
            )
            print(f"  💾 Best 체크포인트 저장 (val_loss={val_loss:.4f})")

    # ── 학습 이력 저장
    hist_path = config.ARTIFACT_DIR / "train_history.json"
    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"\n✅ 학습 완료. 체크포인트: {config.LORA_WEIGHTS_PATH}")
    print(f"   학습 이력: {hist_path}")


if __name__ == "__main__":
    main()