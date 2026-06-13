from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class ModalityConfig:
    name: str
    input_dim: int
    projector_type: str = "auto"


class ModalityProjector(nn.Module):
    """Project one modality into the shared latent token space."""

    def __init__(self, input_dim: int, d_model: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).unsqueeze(1)


class TabularFTTransformerProjector(nn.Module):
    """FT-Transformer-style projector that keeps each tabular feature as a token."""

    def __init__(
        self,
        input_dim: int,
        d_model: int,
        nhead: int = 4,
        num_layers: int = 1,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        clamp_value: float = 10.0,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive for TabularFTTransformerProjector.")
        if d_model % nhead != 0:
            raise ValueError(f"d_model={d_model} must be divisible by nhead={nhead}")

        self.input_dim = input_dim
        self.d_model = d_model
        self.clamp_value = clamp_value
        self.feature_weights = nn.Parameter(torch.randn(1, input_dim, d_model) / d_model**0.5)
        self.feature_bias = nn.Parameter(torch.zeros(1, input_dim, d_model))
        self.feature_position = nn.Parameter(torch.randn(1, input_dim, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.feature_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.clamp(x, -self.clamp_value, self.clamp_value).unsqueeze(-1)
        tokens = x * self.feature_weights + self.feature_bias
        tokens = torch.nn.functional.layer_norm(tokens, tokens.shape[-1:])
        tokens = self.dropout(tokens + self.feature_position)
        tokens = self.feature_encoder(tokens)
        return self.norm(tokens)


class SharedTransformerBackbone(nn.Module):
    """Shared semantic encoder used by every modality."""

    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(f"d_model={d_model} must be divisible by nhead={nhead}")

        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        cls_tokens = self.cls_token.expand(tokens.size(0), -1, -1)
        sequence = torch.cat([cls_tokens, tokens], dim=1)
        encoded = self.encoder(sequence)
        return self.norm(encoded[:, 0, :])


class UnifiedMultimodalTransformerClassifier(nn.Module):
    """
    Non-paired multimodal classifier.

    Each batch contains one modality. Modality-specific projectors map heterogeneous
    inputs into the same latent token space, then a shared Transformer and shared
    classifier head learn the common label semantics.
    """

    def __init__(
        self,
        modalities: list[ModalityConfig] | dict[str, int | dict[str, Any]],
        num_classes: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        tabular_projector_layers: int = 1,
    ) -> None:
        super().__init__()
        modality_configs = _normalize_modalities(modalities)
        if not modality_configs:
            raise ValueError("At least one modality must be configured.")

        self.modality_dims = {item.name: item.input_dim for item in modality_configs}
        self.projector_types = {
            item.name: _resolve_projector_type(item.name, item.projector_type)
            for item in modality_configs
        }
        self.embedding_dim = d_model
        self.num_classes = num_classes
        self.projectors = nn.ModuleDict(
            {
                item.name: _create_projector(
                    input_dim=item.input_dim,
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    projector_type=self.projector_types[item.name],
                    tabular_projector_layers=tabular_projector_layers,
                )
                for item in modality_configs
            }
        )
        self.backbone = SharedTransformerBackbone(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, modality: str, x: torch.Tensor, return_embedding: bool = False):
        if modality not in self.projectors:
            raise KeyError(f"Unknown modality: {modality}")
        expected_dim = self.modality_dims[modality]
        if x.dim() != 2 or x.size(1) != expected_dim:
            raise ValueError(f"Expected {modality} input shape [batch, {expected_dim}], got {tuple(x.shape)}")

        tokens = self.projectors[modality](x)
        embedding = self.backbone(tokens)
        logits = self.classifier(embedding)
        if return_embedding:
            return logits, embedding
        return logits


def _normalize_modalities(modalities: list[ModalityConfig] | dict[str, int | dict[str, Any]]) -> list[ModalityConfig]:
    if isinstance(modalities, dict):
        configs: list[ModalityConfig] = []
        for name, value in modalities.items():
            if isinstance(value, dict):
                configs.append(
                    ModalityConfig(
                        name=name,
                        input_dim=int(value["input_dim"]),
                        projector_type=str(value.get("projector_type", value.get("projector", "auto"))),
                    )
                )
            else:
                configs.append(ModalityConfig(name=name, input_dim=int(value)))
        return configs
    return list(modalities)


def _resolve_projector_type(modality: str, projector_type: str) -> str:
    normalized_type = projector_type.lower()
    if normalized_type == "auto":
        return "fttransformer" if modality == "tabular" else "mlp"
    if normalized_type in {"dense", "mlp"}:
        return "mlp"
    if normalized_type in {"fttransformer", "ft_transformer", "tabular_fttransformer"}:
        return "fttransformer"
    raise ValueError(f"Unsupported projector type for {modality}: {projector_type}")


def _create_projector(
    input_dim: int,
    d_model: int,
    nhead: int,
    dim_feedforward: int,
    dropout: float,
    projector_type: str,
    tabular_projector_layers: int,
) -> nn.Module:
    if projector_type == "mlp":
        return ModalityProjector(input_dim, d_model=d_model, dropout=dropout)
    if projector_type == "fttransformer":
        return TabularFTTransformerProjector(
            input_dim=input_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=tabular_projector_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
    raise ValueError(f"Unsupported projector type: {projector_type}")
