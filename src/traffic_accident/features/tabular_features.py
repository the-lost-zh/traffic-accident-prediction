"""Save preprocessed tabular features as .npy for multimodal training."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from traffic_accident.preprocessing.tabular import load_tabular_data
from traffic_accident.utils.io import ensure_dir, save_json


def save_tabular_feature_artifacts(
    data_path: str | Path,
    output_dir: str | Path,
    target_col: str = "Severity",
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
    max_missing_ratio: float = 0.5,
    max_categorical_unique_ratio: float = 0.05,
) -> dict[str, Any]:
    """Preprocess tabular CSV and save features + labels as .npy for multimodal training.

    Saves:
        features.npy  — all train features (n_samples × n_features)
        labels.npy    — corresponding severity labels (0-indexed)
        preprocessor.pkl — fitted TabularPreprocessor for inference reuse
        metadata.json  — feature metadata
    """
    data_path = Path(data_path)
    output_path = ensure_dir(output_dir)

    bundle = load_tabular_data(
        data_path=data_path,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_state=random_state,
        target_col=target_col,
        artifact_dir=str(output_path),
        max_missing_ratio=max_missing_ratio,
        max_categorical_unique_ratio=max_categorical_unique_ratio,
    )

    # Save train features and labels for multimodal training
    np.save(output_path / "features.npy", bundle.X_train.astype(np.float32))
    np.save(output_path / "labels.npy", bundle.y_train.astype(np.int64))

    # Also save val/test for evaluation
    np.save(output_path / "features_val.npy", bundle.X_val.astype(np.float32))
    np.save(output_path / "labels_val.npy", bundle.y_val.astype(np.int64))
    np.save(output_path / "features_test.npy", bundle.X_test.astype(np.float32))
    np.save(output_path / "labels_test.npy", bundle.y_test.astype(np.int64))

    metadata: dict[str, Any] = {
        "modality": "tabular",
        "data_path": str(data_path),
        "feature_names": bundle.feature_names,
        "numeric_features": bundle.numeric_features,
        "categorical_features": bundle.categorical_features,
        "num_samples": int(bundle.X_train.shape[0]),
        "feature_dim": int(bundle.X_train.shape[1]),
        "num_classes": int(len(np.unique(bundle.y_train))),
        "label_mapping": {int(i): f"Severity {i + 1}" for i in range(4)},
    }
    save_json(metadata, output_path / "metadata.json")

    return metadata
