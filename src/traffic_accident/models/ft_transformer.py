import torch
from torch import nn
from torch.nn import functional as F


class NumericalFeatureTokenizer(nn.Module):
    """Vectorized tokenizer that turns each numeric feature into one token."""

    def __init__(self, n_features: int, d_model: int) -> None:
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1, n_features, d_model) / d_model**0.5)
        self.bias = nn.Parameter(torch.zeros(1, n_features, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.weights + self.bias


class FTTransformerClassifier(nn.Module):
    """Feature Tokenizer Transformer for tabular classification."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int = 4,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(f"d_model={d_model} must be divisible by nhead={nhead}")

        self.input_dim = input_dim
        self.d_model = d_model
        self.feature_tokenizer = NumericalFeatureTokenizer(input_dim, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.positional_encoding = nn.Parameter(torch.randn(1, input_dim, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )
        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        x = torch.clamp(x, -10.0, 10.0).unsqueeze(-1)
        x = self.feature_tokenizer(x)
        x = F.layer_norm(x, x.shape[-1:])
        x = x + self.positional_encoding

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = self.transformer(x)
        return self.classifier(x[:, 0, :])

    def get_feature_importance(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            batch_size = x.size(0)
            x = torch.clamp(x, -10.0, 10.0).unsqueeze(-1)
            x = self.feature_tokenizer(x)
            x = F.layer_norm(x, x.shape[-1:])
            x = x + self.positional_encoding

            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)
            encoded = self.transformer(x)

            cls_output = encoded[:, 0, :]
            feature_outputs = encoded[:, 1:, :]
            cls_normalized = cls_output / (cls_output.norm(dim=1, keepdim=True) + 1e-8)
            feature_normalized = feature_outputs / (feature_outputs.norm(dim=2, keepdim=True) + 1e-8)
            return torch.sum(cls_normalized.unsqueeze(1) * feature_normalized, dim=2)

    def _init_weights(self) -> None:
        for name, parameter in self.named_parameters():
            if "weight" in name and parameter.dim() >= 2:
                gain = 0.5 if "tokenizer" in name else 1.0
                nn.init.xavier_uniform_(parameter, gain=gain)
            elif "bias" in name:
                nn.init.zeros_(parameter)
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.positional_encoding, std=0.02)

