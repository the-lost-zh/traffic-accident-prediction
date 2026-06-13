from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from traffic_accident.evaluation import calculate_metrics
from traffic_accident.utils.io import ensure_dir, save_json


@dataclass
class TrainingConfig:
    epochs: int = 100
    batch_size: int = 512
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    early_stopping_patience: int = 15
    label_smoothing: float = 0.0
    gradient_clip_norm: float | None = 1.0
    num_workers: int = 0


@dataclass
class TrainingResult:
    history: dict[str, list[float]]
    best_val_metric: float
    train_metrics: dict[str, float]
    val_metrics: dict[str, float]
    checkpoint_path: str | None = None
    config: dict = field(default_factory=dict)


class SupervisedTrainer:
    """Small supervised trainer for tabular classification models."""

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig | None = None,
        device: torch.device | str | None = None,
        checkpoint_path: str | Path | None = None,
    ) -> None:
        self.model = model
        self.config = config or TrainingConfig()
        self.device = torch.device(device) if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path is not None else None

        self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=self.config.label_smoothing)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=5,
        )
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
        }

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> TrainingResult:
        train_loader = self._make_loader(X_train, y_train, shuffle=True)
        val_loader = self._make_loader(X_val, y_val, shuffle=False)

        best_val_acc = -1.0
        patience_counter = 0

        for _epoch in range(self.config.epochs):
            train_loss, train_acc = self._run_epoch(train_loader, train=True)
            val_loss, val_acc = self._run_epoch(val_loader, train=False)
            self.scheduler.step(val_loss)

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_acc"].append(val_acc)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                self._save_checkpoint()
            else:
                patience_counter += 1

            if patience_counter >= self.config.early_stopping_patience:
                break

        self._load_best_checkpoint()
        train_pred = self.predict(X_train)
        val_pred = self.predict(X_val)

        return TrainingResult(
            history=self.history,
            best_val_metric=best_val_acc,
            train_metrics=calculate_metrics(y_train, train_pred),
            val_metrics=calculate_metrics(y_val, val_pred),
            checkpoint_path=str(self.checkpoint_path) if self.checkpoint_path else None,
            config=asdict(self.config),
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        dataset = TensorDataset(torch.as_tensor(X, dtype=torch.float32))
        loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
        )
        predictions: list[torch.Tensor] = []
        with torch.no_grad():
            for (batch_X,) in loader:
                logits = self.model(batch_X.to(self.device))
                predictions.append(logits.argmax(dim=1).cpu())
        return torch.cat(predictions).numpy()

    def save_result(self, result: TrainingResult, output_path: str | Path) -> Path:
        return save_json(asdict(result), output_path)

    def _run_epoch(self, loader: DataLoader, train: bool) -> tuple[float, float]:
        self.model.train(train)
        total_loss = 0.0
        total_correct = 0
        total_count = 0

        for batch_X, batch_y in loader:
            batch_X = batch_X.to(self.device)
            batch_y = batch_y.to(self.device)

            if train:
                self.optimizer.zero_grad()

            with torch.set_grad_enabled(train):
                logits = self.model(batch_X)
                loss = self.criterion(logits, batch_y)

                if train:
                    loss.backward()
                    if self.config.gradient_clip_norm is not None:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)
                    self.optimizer.step()

            batch_size = batch_X.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (logits.argmax(dim=1) == batch_y).sum().item()
            total_count += batch_size

        if total_count == 0:
            return 0.0, 0.0
        return total_loss / total_count, total_correct / total_count

    def _make_loader(self, X: np.ndarray, y: np.ndarray, shuffle: bool) -> DataLoader:
        dataset = TensorDataset(
            torch.as_tensor(X, dtype=torch.float32),
            torch.as_tensor(y, dtype=torch.long),
        )
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=self.config.num_workers,
        )

    def _save_checkpoint(self) -> None:
        if self.checkpoint_path is None:
            return
        ensure_dir(self.checkpoint_path.parent)
        torch.save(self.model.state_dict(), self.checkpoint_path)

    def _load_best_checkpoint(self) -> None:
        if self.checkpoint_path is None or not self.checkpoint_path.exists():
            return
        state_dict = torch.load(self.checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state_dict)

