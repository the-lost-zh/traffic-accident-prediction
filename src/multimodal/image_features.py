from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from multimodal.labels import encode_labels
from multimodal.vision_language import (
    extract_vision_language_image_features,
    resolve_vision_language_model_name,
)
from utils import ensure_dir, save_json


def extract_image_features(image_paths: list[str | Path], image_root: str | Path | None = None, bins: int = 16) -> np.ndarray:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Image feature extraction requires Pillow.") from exc

    root = Path(image_root) if image_root is not None else None
    features: list[np.ndarray] = []
    for image_path in image_paths:
        path = Path(image_path)
        if root is not None and not path.is_absolute():
            path = root / path
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            array = np.asarray(rgb, dtype=np.float32)
            height, width = array.shape[:2]
            histograms = [
                np.histogram(array[:, :, channel], bins=bins, range=(0, 255), density=True)[0]
                for channel in range(3)
            ]
            color_mean = array.reshape(-1, 3).mean(axis=0) / 255.0
            color_std = array.reshape(-1, 3).std(axis=0) / 255.0
            shape_stats = np.array([width, height, width / max(height, 1)], dtype=np.float32)
            features.append(np.concatenate([*histograms, color_mean, color_std, shape_stats]).astype(np.float32))
    if not features:
        return np.empty((0, bins * 3 + 9), dtype=np.float32)
    return np.vstack(features)


def save_image_feature_artifacts(
    df: pd.DataFrame,
    image_path_column: str,
    output_dir: str | Path,
    label_column: str | None = None,
    image_root: str | Path | None = None,
    bins: int = 16,
    encoder: str = "color",
    model_name: str | None = None,
    batch_size: int = 32,
    device: str | None = None,
    normalize: bool = True,
) -> dict[str, Any]:
    if image_path_column not in df.columns:
        raise ValueError(f"Missing image path column: {image_path_column}")

    output_path = ensure_dir(output_dir)
    normalized_encoder = encoder.lower()
    if normalized_encoder == "color":
        features = extract_image_features(df[image_path_column].astype(str).tolist(), image_root=image_root, bins=bins)
        resolved_model_name = None
    elif normalized_encoder in {"clip", "siglip"}:
        resolved_model_name = resolve_vision_language_model_name(normalized_encoder, model_name)
        features = extract_vision_language_image_features(
            df[image_path_column].astype(str).tolist(),
            encoder=normalized_encoder,
            model_name=resolved_model_name,
            image_root=image_root,
            batch_size=batch_size,
            device=device,
            normalize=normalize,
        )
    else:
        raise ValueError("Image encoder must be one of: color, clip, siglip")

    np.save(output_path / "features.npy", features)

    metadata: dict[str, Any] = {
        "modality": "image",
        "encoder": normalized_encoder,
        "model_name": resolved_model_name,
        "image_path_column": image_path_column,
        "image_root": str(image_root) if image_root is not None else None,
        "num_samples": int(features.shape[0]),
        "feature_dim": int(features.shape[1]),
        "bins": int(bins) if normalized_encoder == "color" else None,
        "batch_size": int(batch_size) if normalized_encoder != "color" else None,
        "normalized": bool(normalize) if normalized_encoder != "color" else None,
    }
    if label_column is not None:
        if label_column not in df.columns:
            raise ValueError(f"Missing label column: {label_column}")
        labels, label_metadata = encode_labels(df[label_column])
        np.save(output_path / "labels.npy", labels)
        metadata["label_column"] = label_column
        metadata["label_mapping"] = label_metadata

    save_json(metadata, output_path / "metadata.json")
    return metadata
