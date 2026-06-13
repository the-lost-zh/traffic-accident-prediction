import sys
import unittest
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from traffic_accident.models import FTTransformerClassifier, create_model


class ModelForwardTest(unittest.TestCase):
    def test_factory_models_return_expected_logits_shape(self):
        config = {
            "hidden_dims": (16, 8),
            "dropout": 0.0,
            "d_model": 16,
            "nhead": 4,
            "num_layers": 1,
            "dim_feedforward": 32,
        }
        x = torch.randn(5, 7)

        for model_type in ["linear", "mlp", "transformer", "fttransformer"]:
            with self.subTest(model_type=model_type):
                model = create_model(model_type, input_dim=7, num_classes=4, config=config)
                model.eval()
                with torch.no_grad():
                    logits = model(x)
                self.assertEqual(tuple(logits.shape), (5, 4))

    def test_ft_transformer_feature_importance_shape(self):
        model = FTTransformerClassifier(
            input_dim=6,
            num_classes=4,
            d_model=16,
            nhead=4,
            num_layers=1,
            dim_feedforward=32,
            dropout=0.0,
        )
        scores = model.get_feature_importance(torch.randn(3, 6))
        self.assertEqual(tuple(scores.shape), (3, 6))

    def test_invalid_transformer_head_configuration_raises(self):
        with self.assertRaises(ValueError):
            create_model(
                "transformer",
                input_dim=5,
                num_classes=4,
                config={"d_model": 10, "nhead": 4},
            )


if __name__ == "__main__":
    unittest.main()

