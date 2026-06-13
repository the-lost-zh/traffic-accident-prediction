from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


DEFAULT_VISION_LANGUAGE_MODELS = {
    "clip": "openai/clip-vit-large-patch14",
    "siglip": "google/siglip-base-patch16-224",
}


def extract_vision_language_text_features(
    texts: Iterable[str],
    encoder: str = "siglip",
    model_name: str | None = None,
    batch_size: int = 32,
    device: str | torch.device | None = None,
    normalize: bool = True,
) -> np.ndarray:
    """Extract dense text embeddings from a CLIP/SigLIP-style model."""
    text_list = ["" if text is None else str(text) for text in texts]
    if not text_list:
        return np.empty((0, 0), dtype=np.float32)

    model_name = resolve_vision_language_model_name(encoder, model_name)
    resolved_device = _resolve_device(device)
    model, processor = _load_transformers_encoder(model_name, resolved_device)

    outputs: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in _batched(text_list, batch_size):
            inputs = processor(
                text=batch,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            features = model.get_text_features(**_move_to_device(inputs, resolved_device))
            outputs.append(_postprocess_features(features, normalize).cpu())

    return torch.cat(outputs, dim=0).numpy().astype(np.float32)


def extract_vision_language_image_features(
    image_paths: Iterable[str | Path],
    encoder: str = "siglip",
    model_name: str | None = None,
    image_root: str | Path | None = None,
    batch_size: int = 32,
    device: str | torch.device | None = None,
    normalize: bool = True,
) -> np.ndarray:
    """Extract dense image embeddings from a CLIP/SigLIP-style model."""
    paths = [_resolve_image_path(path, image_root) for path in image_paths]
    if not paths:
        return np.empty((0, 0), dtype=np.float32)

    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Image feature extraction requires Pillow.") from exc

    model_name = resolve_vision_language_model_name(encoder, model_name)
    resolved_device = _resolve_device(device)
    model, processor = _load_transformers_encoder(model_name, resolved_device)

    outputs: list[torch.Tensor] = []
    with torch.no_grad():
        for batch_paths in _batched(paths, batch_size):
            images = []
            for image_path in batch_paths:
                with Image.open(image_path) as image:
                    images.append(image.convert("RGB"))
            inputs = processor(images=images, return_tensors="pt")
            features = model.get_image_features(**_move_to_device(inputs, resolved_device))
            outputs.append(_postprocess_features(features, normalize).cpu())

    return torch.cat(outputs, dim=0).numpy().astype(np.float32)


def resolve_vision_language_model_name(encoder: str, model_name: str | None = None) -> str:
    normalized_encoder = encoder.lower()
    if model_name:
        return model_name
    if normalized_encoder not in DEFAULT_VISION_LANGUAGE_MODELS:
        choices = ", ".join(sorted(DEFAULT_VISION_LANGUAGE_MODELS))
        raise ValueError(f"Unsupported vision-language encoder: {encoder}. Expected one of: {choices}")
    return DEFAULT_VISION_LANGUAGE_MODELS[normalized_encoder]


def _load_transformers_encoder(model_name: str, device: torch.device) -> tuple[Any, Any]:
    try:
        from transformers import AutoModel, AutoProcessor
    except ImportError as exc:
        raise RuntimeError(
            "CLIP/SigLIP feature extraction requires transformers. "
            "Install project dependencies or run: pip install transformers"
        ) from exc

    model = AutoModel.from_pretrained(model_name)
    processor = AutoProcessor.from_pretrained(model_name)
    if not hasattr(model, "get_text_features") or not hasattr(model, "get_image_features"):
        raise TypeError(f"Model {model_name!r} does not expose CLIP-style text/image feature methods.")
    model.to(device)
    model.eval()
    return model, processor


def _postprocess_features(features: torch.Tensor, normalize: bool) -> torch.Tensor:
    features = features.float()
    if normalize:
        features = torch.nn.functional.normalize(features, dim=-1)
    return features


def _move_to_device(batch: Any, device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) if hasattr(value, "to") else value for key, value in dict(batch).items()}


def _resolve_device(device: str | torch.device | None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _resolve_image_path(image_path: str | Path, image_root: str | Path | None = None) -> Path:
    path = Path(image_path)
    if image_root is not None and not path.is_absolute():
        path = Path(image_root) / path
    return path


def _batched(items: list[Any], batch_size: int) -> Iterable[list[Any]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]
