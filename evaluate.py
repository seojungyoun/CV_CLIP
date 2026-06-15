# evaluate.py
import json
import time
import pandas as pd
import torch
from PIL import Image
import open_clip
import config

def main():
    print("📊 [PBL 채점 필수 항목] 정량 성능 측정 평가 레이어 구동...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    frame = pd.read_csv(config.CLEAN_DATA_CSV)
    test_frame = frame[frame["split"] == "test"].dropna(subset=["Department"]).reset_index(drop=True)
    
    unique_depts = sorted(test_frame["Department"].unique().tolist())
    prompts = [f"a museum object from the {d} department" for d in unique_depts]
    
    model, _, preprocess = open_clip.create_model_and_transforms(config.MODEL_NAME, pretrained=config.PRETRAINED, device=device)
    tokenizer = open_clip.get_tokenizer(config.MODEL_NAME)
    
    # 학습 완료 체크포인트 가중치 가용 여부 확인 및 동적 로드
    if config.LORA_WEIGHTS_PATH.exists():
        state = torch.load(config.LORA_WEIGHTS_PATH, map_location="cpu", weights_only=True)
        model.load_state_dict(state["model"] if "model" in state else state, strict=False)
    model.to(device).eval()
    
    correct_count = 0
    latencies = []
    image_vectors, text_vectors = [], []
    
    print("🏃 정합성 추론 및 지연 시간 정밀 스캔 시작...")
    with torch.no_grad():
        for idx, row in test_frame.head(200).iterrows(): # 신속한 측정을 위해 상위 200개 샘플 우선 추출
            with Image.open(row["image_path"]) as img:
                img_tensor = preprocess(img.convert("RGB")).unsqueeze(0).to(device)
            tokens = tokenizer([row["caption"]]).to(device)
            
            tick = time.perf_counter()
            img_emb = model.encode_image(img_tensor)
            txt_emb = model.encode_text(tokens)
            if device.type == "cuda":
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - tick) * 1000)
            
            image_vectors.append(torch.nn.functional.normalize(img_emb.float(), dim=-1).cpu())
            text_vectors.append(torch.nn.functional.normalize(txt_emb.float(), dim=-1).cpu())
            
    img_mat = torch.cat(image_vectors)
    txt_mat = torch.cat(text_vectors)
    
    # Retrieval R@1, R@5 매칭 행렬 산출
    similarity = txt_mat @ img_mat.T
    order = similarity.argsort(dim=1, descending=True)
    target = torch.arange(len(similarity)).unsqueeze(1)
    r1 = (order[:, :1] == target).any(dim=1).float().mean().item()
    r5 = (order[:, :5] == target).any(dim=1).float().mean().item()
    
    # Zero-shot 부서 매칭 Acc 산출
    prompt_tokens = tokenizer(prompts).to(device)
    with torch.no_grad():
        prompt_embs = torch.nn.functional.normalize(model.encode_text(prompt_tokens).float(), dim=-1).cpu()
    predictions = (img_mat @ prompt_embs.T).argmax(dim=1)
    expected = torch.tensor([unique_depts.index(d) for d in test_frame.head(200)["Department"].tolist()])
    accuracy = (predictions == expected).float().mean().item()
    
    output_metrics = {
        "Zero-shot Accuracy": f"{accuracy * 100:.2f}%",
        "Image retrieval R@1": f"{r1 * 100:.2f}%",
        "Image retrieval R@5": f"{r5 * 100:.2f}%",
        "Inference Latency": f"{float(pd.Series(latencies).mean()):.2f} ms/item"
    }
    
    with open(config.METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(output_metrics, f, indent=4, ensure_ascii=False)
        
    print("\n================ PBL 제출용 정량 성과표 ================")
    print(json.dumps(output_metrics, indent=4, ensure_ascii=False))
    print("========================================================\n")

if __name__ == "__main__":
    main()