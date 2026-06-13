import torch
from torch import nn


class LinearClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int = 4) -> None:
        super().__init__()
        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)

