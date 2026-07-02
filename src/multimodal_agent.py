"""
Multi-modal inference agent.

Loads a trained UnifiedMultimodalTransformerClassifier checkpoint along with
per-modality encoder artifacts, and exposes a single predict() entry point
that accepts raw inputs (tabular dict, text string, image path) and returns
predictions with per-modality logits.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

_src_dir = os.path.dirname(os.path.abspath(__file__))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from multimodal.data import ModalityFeatureDataset
from multimodal.model import UnifiedMultimodalTransformerClassifier
from utils import load_json


class MultimodalPredictiveAgent:
    """End-to-end multimodal inference agent."""

    def __init__(self, run_dir: str | Path, device: str | torch.device | None = None) -> None:
        self.run_dir = Path(run_dir)
        self.device = torch.device(device) if device else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model: UnifiedMultimodalTransformerClassifier | None = None
        self.modality_configs: dict[str, dict[str, Any]] = {}
        self._encoders: dict[str, Any] = {}  # cached encoders per modality
        self.is_loaded = False

    def load(self, checkpoint_name: str = "best.pt") -> bool:
        """Load model checkpoint, config, and all modality encoders."""
        try:
            config_path = self.run_dir / "config.json"
            if not config_path.exists():
                print(f"Config not found: {config_path}")
                return False

            config: dict[str, Any] = load_json(config_path)
            self.modality_configs = config.get("modalities", {})
            model_cfg = config.get("model_config", {})
            training_cfg = config.get("training_config", {})

            if not self.modality_configs:
                print("No modality configurations in config.json")
                return False

            modality_list = [
                {
                    "name": name,
                    "input_dim": mod["input_dim"],
                    "projector_type": mod.get("projector_type", "auto"),
                }
                for name, mod in self.modality_configs.items()
            ]

            self.model = UnifiedMultimodalTransformerClassifier(
                modalities=modality_list,
                num_classes=4,
                d_model=model_cfg.get("d_model", 128),
                nhead=model_cfg.get("nhead", 4),
                num_layers=model_cfg.get("num_layers", 2),
                dim_feedforward=model_cfg.get("dim_feedforward", 256),
                dropout=model_cfg.get("dropout", 0.1),
                tabular_projector_layers=model_cfg.get("tabular_projector_layers", 1),
            )

            checkpoint_path = self.run_dir / "checkpoints" / checkpoint_name
            if not checkpoint_path.exists():
                checkpoint_path = self.run_dir / "checkpoints" / "final_model.pth"
            if not checkpoint_path.exists():
                print(f"Checkpoint not found: {checkpoint_path}")
                return False

            self.model.load_state_dict(
                torch.load(checkpoint_path, map_location=self.device, weights_only=True)
            )
            self.model.to(self.device)
            self.model.eval()

            for name, mod_cfg in self.modality_configs.items():
                self._load_encoder(name, mod_cfg)

            self.is_loaded = True
            n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print(f"Multimodal agent loaded [{self.device}]: {len(self.modality_configs)} modalities, {n_params:,} params")
            return True

        except Exception as exc:
            print(f"Multimodal agent load failed: {exc}")
            return False

    def predict(
        self,
        tabular: dict[str, Any] | None = None,
        text: str | None = None,
        image_path: str | Path | None = None,
    ) -> dict[str, Any]:
        """Predict severity from one or multiple raw modalities."""
        if not self.is_loaded:
            raise RuntimeError("Agent not loaded. Call load() first.")

        results: dict[str, dict[str, Any]] = {}
        all_probs: list[np.ndarray] = []

        for modality, mod_cfg in self.modality_configs.items():
            features = self._encode(modality, tabular=tabular, text=text, image_path=image_path)
            if features is None:
                continue

            tensor = torch.as_tensor(features, dtype=torch.float32).to(self.device)
            if tensor.dim() == 1:
                tensor = tensor.unsqueeze(0)

            with torch.no_grad():
                logits = self.model(modality, tensor)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

            results[modality] = {
                "severity": int(probs.argmax()),
                "probability": float(probs.max()),
                "probabilities": probs.tolist(),
            }
            all_probs.append(probs)

        # Ensemble: average probabilities across modalities if multiple results
        if len(all_probs) > 1:
            ensemble = np.stack(all_probs).mean(axis=0)
            results["ensemble"] = {
                "severity": int(ensemble.argmax()),
                "probability": float(ensemble.max()),
                "probabilities": ensemble.tolist(),
            }

        return results

    def _encode(
        self,
        modality: str,
        tabular: dict[str, Any] | None = None,
        text: str | None = None,
        image_path: str | Path | None = None,
    ) -> np.ndarray | None:
        """Encode raw input for a given modality using cached encoder."""
        if modality == "tabular" and tabular is not None:
            return self._encode_tabular(tabular)
        if modality == "text" and text is not None:
            return self._encode_text(text)
        if modality == "image" and image_path is not None:
            return self._encode_image(image_path)
        return None

    def _encode_tabular(self, data: dict[str, Any]) -> np.ndarray | None:
        encoder = self._encoders.get("tabular")
        if encoder is None:
            print("  [tabular] encoder not loaded — skipping tabular modality")
            return None
        try:
            features, _ = encoder.preprocess_input(data)
            return features
        except Exception as exc:
            print(f"  [tabular] encoding failed: {exc} — skipping")
            return None

    def _encode_text(self, text: str) -> np.ndarray | None:
        encoder_info = self._encoders.get("text")
        if encoder_info is None:
            print("  [text] encoder not loaded — skipping text modality")
            return None

        encoder_type = encoder_info.get("encoder", "tfidf")
        if encoder_type == "tfidf":
            vectorizer = encoder_info["vectorizer"]
            features = vectorizer.transform([text]).astype(np.float32).toarray()
        elif encoder_type in ("clip", "siglip"):
            from multimodal.vision_language import extract_vision_language_text_features
            model_name = encoder_info.get("model_name")
            features = extract_vision_language_text_features(
                [text],
                encoder=encoder_type,
                model_name=model_name,
                device=str(self.device),
                normalize=True,
            )
        else:
            raise ValueError(f"Unknown text encoder: {encoder_type}")
        return features

    def _encode_image(self, image_path: str | Path) -> np.ndarray | None:
        encoder_info = self._encoders.get("image")
        if encoder_info is None:
            print("  [image] encoder not loaded — skipping image modality")
            return None

        encoder_type = encoder_info.get("encoder", "color")
        if encoder_type == "color":
            from multimodal.image_features import extract_image_features
            features = extract_image_features([image_path], image_root=encoder_info.get("image_root"))
        elif encoder_type in ("clip", "siglip"):
            from multimodal.vision_language import extract_vision_language_image_features
            model_name = encoder_info.get("model_name")
            features = extract_vision_language_image_features(
                [image_path],
                encoder=encoder_type,
                model_name=model_name,
                image_root=encoder_info.get("image_root"),
                device=str(self.device),
                normalize=True,
            )
        else:
            raise ValueError(f"Unknown image encoder: {encoder_type}")
        return features

    def _load_encoder(self, modality: str, mod_cfg: dict[str, Any]) -> None:
        """Load modality-specific encoder artifacts."""
        if modality == "tabular":
            self._load_tabular_encoder()
        elif modality == "text":
            self._load_text_encoder(mod_cfg)
        elif modality == "image":
            self._load_image_encoder(mod_cfg)
        else:
            print(f"  [{modality}] unknown modality, skipping encoder load")

    def _load_tabular_encoder(self) -> None:
        """Load the tabular DataPreprocessor from the standard models directory."""
        import joblib
        preprocessor_path = self.run_dir.parent / "models" / "preprocessor.pkl"
        candidate_paths = [
            self.run_dir / "artifacts" / "preprocessor.pkl",
            preprocessor_path,
            Path("models") / "preprocessor.pkl",
        ]
        for path in candidate_paths:
            if path.exists():
                from data_preprocessing import DataPreprocessor
                preprocessor = DataPreprocessor.load(str(path))
                self._encoders["tabular"] = preprocessor
                print(f"  [tabular] preprocessor loaded from {path}")
                return
        print(f"  [tabular] preprocessor not found; tabular encoding will be unavailable")

    def _load_text_encoder(self, mod_cfg: dict[str, Any]) -> None:
        encoder_type = mod_cfg.get("encoder", "tfidf")
        feature_path = mod_cfg.get("feature_path", "")
        feature_dir = Path(feature_path).parent if feature_path else self.run_dir

        if encoder_type == "tfidf":
            import joblib
            vectorizer_path = feature_dir / "vectorizer.pkl"
            if vectorizer_path.exists():
                vectorizer = joblib.load(vectorizer_path)
                self._encoders["text"] = {
                    "encoder": "tfidf",
                    "vectorizer": vectorizer,
                }
                print(f"  [text] TF-IDF vectorizer loaded from {vectorizer_path}")
            else:
                print(f"  [text] vectorizer.pkl not found at {vectorizer_path}")
        elif encoder_type in ("clip", "siglip"):
            self._encoders["text"] = {
                "encoder": encoder_type,
                "model_name": mod_cfg.get("model_name"),
            }
            print(f"  [text] will use {encoder_type} (model={mod_cfg.get('model_name', 'default')})")
        else:
            print(f"  [text] unknown encoder type: {encoder_type}")

    def _load_image_encoder(self, mod_cfg: dict[str, Any]) -> None:
        encoder_type = mod_cfg.get("encoder", "color")
        self._encoders["image"] = {
            "encoder": encoder_type,
            "model_name": mod_cfg.get("model_name"),
            "image_root": mod_cfg.get("image_root"),
        }
        print(f"  [image] encoder={encoder_type}")
