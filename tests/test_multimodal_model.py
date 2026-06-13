import sys
import unittest
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from traffic_accident.models import TabularFTTransformerProjector, UnifiedMultimodalTransformerClassifier


class MultimodalModelTest(unittest.TestCase):
    def test_unpaired_modalities_share_classifier_shape(self):
        model = UnifiedMultimodalTransformerClassifier(
            modalities={
                "tabular": 12,
                "image": 32,
                "text": 24,
            },
            num_classes=4,
            d_model=16,
            nhead=4,
            num_layers=1,
            dim_feedforward=32,
            dropout=0.0,
        )
        model.eval()

        with torch.no_grad():
            tabular_logits, tabular_embedding = model("tabular", torch.randn(5, 12), return_embedding=True)
            image_logits = model("image", torch.randn(3, 32))
            text_logits = model("text", torch.randn(4, 24))

        self.assertEqual(tuple(tabular_logits.shape), (5, 4))
        self.assertEqual(tuple(image_logits.shape), (3, 4))
        self.assertEqual(tuple(text_logits.shape), (4, 4))
        self.assertEqual(tuple(tabular_embedding.shape), (5, 16))
        self.assertEqual(model.projector_types["tabular"], "fttransformer")
        self.assertEqual(model.projector_types["image"], "mlp")

    def test_unknown_modality_raises(self):
        model = UnifiedMultimodalTransformerClassifier({"tabular": 8}, num_classes=4, d_model=16, nhead=4)
        with self.assertRaises(KeyError):
            model("audio", torch.randn(2, 8))

    def test_wrong_input_dimension_raises(self):
        model = UnifiedMultimodalTransformerClassifier({"tabular": 8}, num_classes=4, d_model=16, nhead=4)
        with self.assertRaises(ValueError):
            model("tabular", torch.randn(2, 7))

    def test_tabular_projector_keeps_feature_tokens(self):
        model = UnifiedMultimodalTransformerClassifier(
            {"tabular": 8, "text": 12},
            num_classes=4,
            d_model=16,
            nhead=4,
            num_layers=1,
            dim_feedforward=32,
            dropout=0.0,
        )
        model.eval()

        with torch.no_grad():
            tabular_tokens = model.projectors["tabular"](torch.randn(2, 8))
            text_tokens = model.projectors["text"](torch.randn(2, 12))

        self.assertIsInstance(model.projectors["tabular"], TabularFTTransformerProjector)
        self.assertEqual(tuple(tabular_tokens.shape), (2, 8, 16))
        self.assertEqual(tuple(text_tokens.shape), (2, 1, 16))

    def test_projector_type_can_be_overridden(self):
        model = UnifiedMultimodalTransformerClassifier(
            {"tabular": {"input_dim": 8, "projector_type": "mlp"}},
            num_classes=4,
            d_model=16,
            nhead=4,
        )
        self.assertEqual(model.projector_types["tabular"], "mlp")
        with torch.no_grad():
            tokens = model.projectors["tabular"](torch.randn(2, 8))
        self.assertEqual(tuple(tokens.shape), (2, 1, 16))


if __name__ == "__main__":
    unittest.main()
