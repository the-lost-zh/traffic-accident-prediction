from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from traffic_accident.features.labels import encode_labels
from traffic_accident.features.vision_language import (
    extract_vision_language_text_features,
    resolve_vision_language_model_name,
)
from traffic_accident.utils.io import ensure_dir, save_json, save_pickle


def extract_text_features(
    texts: pd.Series | list[str],
    max_features: int = 768,
    ngram_range: tuple[int, int] = (1, 2),
) -> tuple[np.ndarray, TfidfVectorizer]:
    clean_texts = pd.Series(texts).fillna("").astype(str)
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
    features = vectorizer.fit_transform(clean_texts).astype(np.float32).toarray()
    return features, vectorizer


def save_text_feature_artifacts(
    df: pd.DataFrame,
    text_column: str,
    output_dir: str | Path,
    label_column: str | None = None,
    max_features: int = 768,
    encoder: str = "tfidf",
    model_name: str | None = None,
    batch_size: int = 32,
    device: str | None = None,
    normalize: bool = True,
) -> dict[str, Any]:
    if text_column not in df.columns:
        raise ValueError(f"Missing text column: {text_column}")

    output_path = ensure_dir(output_dir)
    normalized_encoder = encoder.lower()
    if normalized_encoder == "tfidf":
        features, vectorizer = extract_text_features(df[text_column], max_features=max_features)
        save_pickle(vectorizer, output_path / "vectorizer.pkl")
        resolved_model_name = None
    elif normalized_encoder in {"clip", "siglip"}:
        resolved_model_name = resolve_vision_language_model_name(normalized_encoder, model_name)
        features = extract_vision_language_text_features(
            df[text_column].fillna("").astype(str).tolist(),
            encoder=normalized_encoder,
            model_name=resolved_model_name,
            batch_size=batch_size,
            device=device,
            normalize=normalize,
        )
    else:
        raise ValueError("Text encoder must be one of: tfidf, clip, siglip")

    np.save(output_path / "features.npy", features)

    metadata: dict[str, Any] = {
        "modality": "text",
        "encoder": normalized_encoder,
        "model_name": resolved_model_name,
        "text_column": text_column,
        "num_samples": int(features.shape[0]),
        "feature_dim": int(features.shape[1]),
        "max_features": int(max_features) if normalized_encoder == "tfidf" else None,
        "batch_size": int(batch_size) if normalized_encoder != "tfidf" else None,
        "normalized": bool(normalize) if normalized_encoder != "tfidf" else None,
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
