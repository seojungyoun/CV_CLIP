# clip_utils.py
import torch
import open_clip
import config

def load_clip_model(device: torch.device, trained: bool = True):
    """안정적인 OpenCLIP 원본 모델 구조화 및 부분 가중치(LoRA) 가중치 복원 모듈"""
    model, _, preprocess = open_clip.create_model_and_transforms(
        config.MODEL_NAME, pretrained=config.PRETRAINED
    )
    tokenizer = open_clip.get_tokenizer(config.MODEL_NAME)
    
    checkpoint = config.LORA_WEIGHTS_PATH
    if trained and checkpoint.exists():
        print(f"🔄 파인튜닝된 도메인 특화 LoRA 가중치 주입 중: {checkpoint}")
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if "model" in state:
            model.load_state_dict(state["model"], strict=False)
        else:
            model.load_state_dict(state, strict=False)
    else:
        print("💡 사전 학습된 원본 Base CLIP 모델을 로드합니다.")
            
    model = model.to(device).eval()
    return model, preprocess, tokenizer