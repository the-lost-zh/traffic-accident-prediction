from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from multimodal.data import ModalityBatch, ModalityFeatureDataset, make_modality_loaders, round_robin_modality_batches
from utils import calculate_metrics, ensure_dir, save_json


@dataclass
class MultimodalTrainingConfig:
    epochs: int = 20
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    label_smoothing: float = 0.0
    gradient_clip_norm: float | None = 1.0
    steps_per_epoch: int | None = None
    num_workers: int = 0
    prototype_alignment_weight: float = 0.0
    contrastive_weight: float = 0.0
    contrastive_temperature: float = 0.2


@dataclass
class MultimodalTrainingResult:
    history: dict[str, list[float]]
    train_metrics_by_modality: dict[str, dict[str, float]]
    checkpoint_path: str | None = None
    config: dict = field(default_factory=dict)


class UnpairedMultimodalTrainer:
    """Train one shared classifier from non-paired modality batches."""

    def __init__(
        self,
        model: nn.Module,
        config: MultimodalTrainingConfig | None = None,
        device: torch.device | str | None = None,
        checkpoint_path: str | Path | None = None,
    ) -> None:
        self.model = model
        self.config = config or MultimodalTrainingConfig()
        self.device = torch.device(device) if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path is not None else None
        self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=self.config.label_smoothing)
        self.label_prototypes = self._make_label_prototypes()
        optimizer_params = list(self.model.parameters())
        if self.label_prototypes is not None:
            optimizer_params.append(self.label_prototypes)
        self.optimizer = torch.optim.AdamW(
            optimizer_params,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        self.history = {
            "loss": [],
            "classification_loss": [],
            "prototype_alignment_loss": [],
            "contrastive_loss": [],
            "accuracy": [],
        }

    def fit(self, datasets: dict[str, ModalityFeatureDataset]) -> MultimodalTrainingResult:
        loaders = make_modality_loaders(
            datasets=datasets,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
        )

        best_acc = -1.0
        for _epoch in range(self.config.epochs):
            batches = round_robin_modality_batches(loaders, steps_per_epoch=self.config.steps_per_epoch)
            epoch_metrics = self._train_batches(batches)
            for key, value in epoch_metrics.items():
                self.history[key].append(value)
            epoch_acc = epoch_metrics["accuracy"]
            if epoch_acc > best_acc:
                best_acc = epoch_acc
                self._save_checkpoint()

        self._load_best_checkpoint()
        metrics_by_modality = self.evaluate(datasets)
        return MultimodalTrainingResult(
            history=self.history,
            train_metrics_by_modality=metrics_by_modality,
            checkpoint_path=str(self.checkpoint_path) if self.checkpoint_path else None,
            config=asdict(self.config),
        )

    def evaluate(self, datasets: dict[str, ModalityFeatureDataset]) -> dict[str, dict[str, float]]:
        return {
            modality: calculate_metrics(dataset.labels.numpy(), self.predict(dataset))
            for modality, dataset in datasets.items()
        }

    def predict(self, dataset: ModalityFeatureDataset) -> np.ndarray:
        loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
        )
        predictions: list[torch.Tensor] = []
        self.model.eval()
        with torch.no_grad():
            for features, _labels in loader:
                logits = self.model(dataset.modality, features.to(self.device))
                predictions.append(logits.argmax(dim=1).cpu())
        return torch.cat(predictions).numpy()

    def save_result(self, result: MultimodalTrainingResult, output_path: str | Path) -> Path:
        return save_json(asdict(result), output_path)

    def _train_batches(self, batches: list[ModalityBatch]) -> dict[str, float]:
        self.model.train()
        total_loss = 0.0
        total_classification_loss = 0.0
        total_prototype_loss = 0.0
        total_contrastive_loss = 0.0
        total_correct = 0
        total_count = 0

        for batch in batches:
            features = batch.features.to(self.device)
            labels = batch.labels.to(self.device)
            self.optimizer.zero_grad()
            if self._uses_embedding_losses:
                logits, embeddings = self.model(batch.modality, features, return_embedding=True)
            else:
                logits = self.model(batch.modality, features)
                embeddings = None

            classification_loss = self.criterion(logits, labels)
            prototype_loss = self._prototype_alignment_loss(embeddings, labels)
            contrastive_loss = self._supervised_contrastive_loss(embeddings, labels)
            loss = (
                classification_loss
                + self.config.prototype_alignment_weight * prototype_loss
                + self.config.contrastive_weight * contrastive_loss
            )
            loss.backward()
            if self.config.gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)
            self.optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_classification_loss += classification_loss.item() * batch_size
            total_prototype_loss += prototype_loss.item() * batch_size
            total_contrastive_loss += contrastive_loss.item() * batch_size
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_count += batch_size

        if total_count == 0:
            return {
                "loss": 0.0,
                "classification_loss": 0.0,
                "prototype_alignment_loss": 0.0,
                "contrastive_loss": 0.0,
                "accuracy": 0.0,
            }
        return {
            "loss": total_loss / total_count,
            "classification_loss": total_classification_loss / total_count,
            "prototype_alignment_loss": total_prototype_loss / total_count,
            "contrastive_loss": total_contrastive_loss / total_count,
            "accuracy": total_correct / total_count,
        }

    @property
    def _uses_embedding_losses(self) -> bool:
        return self.config.prototype_alignment_weight > 0 or self.config.contrastive_weight > 0

    def _make_label_prototypes(self) -> nn.Parameter | None:
        if self.config.prototype_alignment_weight <= 0:
            return None

        embedding_dim = getattr(self.model, "embedding_dim", None)
        num_classes = getattr(self.model, "num_classes", None)
        if embedding_dim is None and hasattr(self.model, "classifier"):
            embedding_dim = getattr(self.model.classifier, "in_features", None)
        if num_classes is None and hasattr(self.model, "classifier"):
            num_classes = getattr(self.model.classifier, "out_features", None)
        if embedding_dim is None or num_classes is None:
            raise ValueError("Prototype alignment requires model.embedding_dim and model.num_classes metadata.")
        return nn.Parameter(torch.randn(int(num_classes), int(embedding_dim), device=self.device) * 0.02)

    def _prototype_alignment_loss(self, embeddings: torch.Tensor | None, labels: torch.Tensor) -> torch.Tensor:
        if self.label_prototypes is None or embeddings is None:
            return torch.zeros((), device=self.device)
        temperature = max(self.config.contrastive_temperature, 1e-6)
        normalized_embeddings = torch.nn.functional.normalize(embeddings, dim=1)
        normalized_prototypes = torch.nn.functional.normalize(self.label_prototypes, dim=1)
        logits = normalized_embeddings @ normalized_prototypes.t() / temperature
        return torch.nn.functional.cross_entropy(logits, labels)

    def _supervised_contrastive_loss(self, embeddings: torch.Tensor | None, labels: torch.Tensor) -> torch.Tensor:
        if self.config.contrastive_weight <= 0 or embeddings is None or embeddings.size(0) <= 1:
            return torch.zeros((), device=self.device)

        temperature = max(self.config.contrastive_temperature, 1e-6)
        normalized = torch.nn.functional.normalize(embeddings, dim=1)
        similarity = normalized @ normalized.t() / temperature
        similarity = similarity - similarity.max(dim=1, keepdim=True).values.detach()

        batch_size = labels.size(0)
        logits_mask = ~torch.eye(batch_size, dtype=torch.bool, device=self.device)
        positive_mask = labels.unsqueeze(0).eq(labels.unsqueeze(1)) & logits_mask
        positive_counts = positive_mask.sum(dim=1)
        valid_anchors = positive_counts > 0
        if not valid_anchors.any():
            return torch.zeros((), device=self.device)

        exp_similarity = torch.exp(similarity) * logits_mask.float()
        log_prob = similarity - torch.log(exp_similarity.sum(dim=1, keepdim=True).clamp_min(1e-12))
        mean_log_prob_pos = (positive_mask.float() * log_prob).sum(dim=1) / positive_counts.clamp_min(1)
        return -mean_log_prob_pos[valid_anchors].mean()

    def _save_checkpoint(self) -> None:
        if self.checkpoint_path is None:
            return
        ensure_dir(self.checkpoint_path.parent)
        torch.save(self.model.state_dict(), self.checkpoint_path)

    def _load_best_checkpoint(self) -> None:
        if self.checkpoint_path is None or not self.checkpoint_path.exists():
            return
        self.model.load_state_dict(torch.load(self.checkpoint_path, map_location=self.device))
