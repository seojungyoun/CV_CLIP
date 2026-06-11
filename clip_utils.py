from pathlib import Path

import open_clip
import torch

from config import MODEL_DIR, MODEL_NAME, PRETRAINED


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_clip(device: torch.device, trained: bool = True):
    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME, pretrained=PRETRAINED
    )
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)
    checkpoint = MODEL_DIR / "best.pt"
    if trained and checkpoint.exists():
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(state["model"], strict=False)
    model = model.to(device).eval()
    return model, preprocess, tokenizer


def trainable_text_parameters(model) -> list[torch.nn.Parameter]:
    for parameter in model.parameters():
        parameter.requires_grad = False
    # A small trainable surface is much faster and less prone to overfitting.
    for parameter in model.transformer.resblocks[-1].parameters():
        parameter.requires_grad = True
    model.text_projection.requires_grad = True
    model.logit_scale.requires_grad = True
    return [parameter for parameter in model.parameters() if parameter.requires_grad]
