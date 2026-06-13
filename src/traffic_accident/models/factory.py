from typing import Any

from torch import nn

from .ft_transformer import FTTransformerClassifier
from .linear import LinearClassifier
from .mlp import MLPClassifier
from .transformer import TransformerClassifier


def create_model(model_type: str, input_dim: int, num_classes: int, config: dict[str, Any] | None = None) -> nn.Module:
    config = config or {}
    normalized_type = model_type.lower()

    if normalized_type == "linear":
        return LinearClassifier(input_dim=input_dim, num_classes=num_classes)

    if normalized_type == "mlp":
        return MLPClassifier(
            input_dim=input_dim,
            hidden_dims=config.get("hidden_dims", (128, 64)),
            num_classes=num_classes,
            dropout=config.get("dropout", 0.3),
        )

    if normalized_type == "transformer":
        return TransformerClassifier(
            input_dim=input_dim,
            num_classes=num_classes,
            d_model=config.get("d_model", 64),
            nhead=config.get("nhead", 4),
            num_layers=config.get("num_layers", 1),
            dim_feedforward=config.get("dim_feedforward", 128),
            dropout=config.get("dropout", 0.1),
        )

    if normalized_type == "fttransformer":
        return FTTransformerClassifier(
            input_dim=input_dim,
            num_classes=num_classes,
            d_model=config.get("d_model", 64),
            nhead=config.get("nhead", 4),
            num_layers=config.get("num_layers", 2),
            dim_feedforward=config.get("dim_feedforward", 256),
            dropout=config.get("dropout", 0.1),
        )

    raise ValueError(f"Unknown model type: {model_type}")

