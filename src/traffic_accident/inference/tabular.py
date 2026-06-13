from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from traffic_accident.models import create_model
from traffic_accident.preprocessing.tabular import TabularPreprocessor
from traffic_accident.utils.io import load_json, load_pickle


class TabularRunPredictor:
    """Load a trained run directory and serve tabular predictions."""

    def __init__(self, run_dir: str | Path, device: str | None = None) -> None:
        self.run_dir = Path(run_dir)
        self.config = load_json(self.run_dir / "config.json")
        self.preprocessor: TabularPreprocessor = load_pickle(self.run_dir / "artifacts" / "preprocessor.pkl")
        self.device = torch.device(device) if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = create_model(
            model_type=self.config["model_type"],
            input_dim=len(self.config["feature_names"]),
            num_classes=int(self.config["num_classes"]),
            config=self.config["model_config"],
        )
        state_dict = torch.load(self.run_dir / "checkpoints" / "best.pt", map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    def predict_one(self, features: dict[str, Any]) -> dict[str, Any]:
        batch_result = self.predict_many([features])
        return {
            "severity": int(batch_result["severity"][0]),
            "label_index": int(batch_result["label_index"][0]),
            "probability": float(batch_result["probability"][0]),
            "probabilities": batch_result["probabilities"][0].tolist(),
        }

    def predict_many(self, records: list[dict[str, Any]]) -> dict[str, np.ndarray]:
        X = self.preprocessor.transform_features(records)
        batch_size = int(self.config["training_config"]["batch_size"])
        probabilities: list[torch.Tensor] = []

        with torch.no_grad():
            for start in range(0, len(X), batch_size):
                batch = torch.as_tensor(X[start : start + batch_size], dtype=torch.float32, device=self.device)
                logits = self.model(batch)
                probabilities.append(torch.softmax(logits, dim=1).cpu())

        proba = torch.cat(probabilities).numpy()
        label_index = proba.argmax(axis=1)
        severity = label_index + 1
        confidence = proba[np.arange(len(label_index)), label_index]
        return {
            "severity": severity,
            "label_index": label_index,
            "probability": confidence,
            "probabilities": proba,
        }

