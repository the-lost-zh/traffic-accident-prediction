"""Tests for model architectures in task3_classifier.py."""
import sys
import unittest
from pathlib import Path

import torch

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from task3_classifier import (
    FocalLoss,
    LinearClassifier,
    MLPClassifier,
    TransformerClassifier,
    FTTransformerClassifier,
    NumericalFeatureTokenizer,
    create_model,
    get_default_config,
)


class TestNumericalFeatureTokenizer(unittest.TestCase):
    def test_output_shape(self):
        tok = NumericalFeatureTokenizer(n_features=5, d_model=16)
        x = torch.randn(3, 5, 1)
        out = tok(x)
        self.assertEqual(tuple(out.shape), (3, 5, 16))


class TestModelFactory(unittest.TestCase):
    def test_all_models_return_correct_logits_shape(self):
        config = get_default_config()
        config.update({"d_model": 16, "nhead": 4, "dim_feedforward": 32})
        x = torch.randn(5, 7)
        for model_type in ["linear", "mlp", "transformer", "fttransformer"]:
            with self.subTest(model_type=model_type):
                model = create_model(model_type, input_dim=7, num_classes=4, config=config)
                model.eval()
                with torch.no_grad():
                    logits = model(x)
                self.assertEqual(tuple(logits.shape), (5, 4))


class TestFTTransformer(unittest.TestCase):
    def test_feature_importance_shape(self):
        model = FTTransformerClassifier(
            input_dim=6, num_classes=4, d_model=16, nhead=4,
            num_layers=1, dim_feedforward=32, dropout=0.0,
        )
        scores = model.get_feature_importance(torch.randn(3, 6))
        self.assertEqual(tuple(scores.shape), (3, 6))

    def test_clamp_input(self):
        model = FTTransformerClassifier(input_dim=4, d_model=16, nhead=4)
        model.eval()
        x = torch.tensor([[100.0, -100.0, 5.0, -5.0]])
        with torch.no_grad():
            out = model(x)
        self.assertFalse(torch.isnan(out).any())


class TestTransformerValidator(unittest.TestCase):
    def test_incompatible_d_model_raises(self):
        with self.assertRaises(ValueError):
            create_model("transformer", input_dim=5, num_classes=4,
                         config={"d_model": 10, "nhead": 4})

    def test_auto_fix_nhead(self):
        model = create_model("fttransformer", input_dim=5, num_classes=4,
                             config={"d_model": 64, "nhead": 7})
        self.assertIsNotNone(model)


class TestFocalLoss(unittest.TestCase):
    def test_gamma_zero_equals_ce(self):
        fl = FocalLoss(gamma=0.0)
        ce = torch.nn.CrossEntropyLoss()
        inputs = torch.randn(4, 4)
        targets = torch.randint(0, 4, (4,))
        fl_val = fl(inputs, targets)
        ce_val = ce(inputs, targets)
        self.assertAlmostEqual(fl_val.item(), ce_val.item(), places=4)

    def test_gamma_positive_works(self):
        fl = FocalLoss(gamma=2.0)
        inputs = torch.randn(8, 4)
        targets = torch.randint(0, 4, (8,))
        loss = fl(inputs, targets)
        self.assertFalse(torch.isnan(loss))
        self.assertGreater(loss.item(), 0)


if __name__ == "__main__":
    unittest.main()
