from __future__ import annotations

from dataclasses import dataclass
from itertools import cycle

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


@dataclass
class ModalityBatch:
    modality: str
    features: torch.Tensor
    labels: torch.Tensor


class ModalityFeatureDataset(Dataset):
    """Feature-vector dataset for one modality in a shared label space."""

    def __init__(self, modality: str, features: np.ndarray | torch.Tensor, labels: np.ndarray | torch.Tensor) -> None:
        if len(features) != len(labels):
            raise ValueError("features and labels must contain the same number of samples.")
        self.modality = modality
        self.features = torch.as_tensor(features, dtype=torch.float32)
        self.labels = torch.as_tensor(labels, dtype=torch.long)

        if self.features.dim() != 2:
            raise ValueError(f"features must be a 2D matrix, got shape {tuple(self.features.shape)}")

    def __len__(self) -> int:
        return self.features.size(0)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.labels[index]


def make_modality_loaders(
    datasets: dict[str, ModalityFeatureDataset],
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
) -> dict[str, DataLoader]:
    return {
        modality: DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        for modality, dataset in datasets.items()
    }


def round_robin_modality_batches(
    loaders: dict[str, DataLoader],
    steps_per_epoch: int | None = None,
) -> list[ModalityBatch]:
    """Return one epoch of non-paired modality batches in round-robin order."""
    if not loaders:
        raise ValueError("At least one modality loader is required.")

    max_steps = steps_per_epoch or sum(len(loader) for loader in loaders.values())
    iterators = {modality: cycle(loader) for modality, loader in loaders.items()}
    modalities = list(loaders.keys())
    batches: list[ModalityBatch] = []

    for step in range(max_steps):
        modality = modalities[step % len(modalities)]
        features, labels = next(iterators[modality])
        batches.append(ModalityBatch(modality=modality, features=features, labels=labels))

    return batches

