import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from traffic_accident.data import ModalityFeatureDataset, make_modality_loaders, round_robin_modality_batches
from traffic_accident.models import UnifiedMultimodalTransformerClassifier
from traffic_accident.training import MultimodalTrainingConfig, UnpairedMultimodalTrainer


class MultimodalTrainingTest(unittest.TestCase):
    def test_round_robin_batches_cycle_modalities(self):
        datasets = {
            "tabular": ModalityFeatureDataset("tabular", np.zeros((6, 4), dtype=np.float32), np.array([0, 1, 0, 1, 0, 1])),
            "text": ModalityFeatureDataset("text", np.zeros((6, 5), dtype=np.float32), np.array([1, 0, 1, 0, 1, 0])),
        }
        loaders = make_modality_loaders(datasets, batch_size=2, shuffle=False)
        batches = round_robin_modality_batches(loaders, steps_per_epoch=4)

        self.assertEqual([batch.modality for batch in batches], ["tabular", "text", "tabular", "text"])
        self.assertEqual(tuple(batches[0].features.shape), (2, 4))
        self.assertEqual(tuple(batches[1].features.shape), (2, 5))

    def test_unpaired_multimodal_trainer_fit_predict_save(self):
        rng = np.random.default_rng(42)
        tabular_X = rng.normal(size=(24, 6)).astype(np.float32)
        text_X = rng.normal(size=(20, 8)).astype(np.float32)
        tabular_y = (tabular_X[:, 0] > 0).astype(np.int64)
        text_y = (text_X[:, 0] > 0).astype(np.int64)
        datasets = {
            "tabular": ModalityFeatureDataset("tabular", tabular_X, tabular_y),
            "text": ModalityFeatureDataset("text", text_X, text_y),
        }
        model = UnifiedMultimodalTransformerClassifier(
            modalities={"tabular": 6, "text": 8},
            num_classes=2,
            d_model=16,
            nhead=4,
            num_layers=1,
            dim_feedforward=32,
            dropout=0.0,
        )
        output_dir = PROJECT_ROOT / ".test_tmp" / "multimodal_training"
        trainer = UnpairedMultimodalTrainer(
            model=model,
            config=MultimodalTrainingConfig(epochs=1, batch_size=8, learning_rate=1e-2, steps_per_epoch=4),
            device="cpu",
            checkpoint_path=output_dir / "best.pt",
        )
        result = trainer.fit(datasets)
        saved_path = trainer.save_result(result, output_dir / "result.json")

        self.assertTrue((output_dir / "best.pt").exists())
        self.assertTrue(saved_path.exists())
        self.assertIn("tabular", result.train_metrics_by_modality)
        self.assertIn("text", result.train_metrics_by_modality)
        self.assertEqual(len(result.history["loss"]), 1)
        self.assertEqual(trainer.predict(datasets["tabular"]).shape, tabular_y.shape)

    def test_unpaired_multimodal_trainer_supports_alignment_losses(self):
        rng = np.random.default_rng(7)
        tabular_X = rng.normal(size=(16, 6)).astype(np.float32)
        text_X = rng.normal(size=(16, 8)).astype(np.float32)
        labels = np.array([0, 0, 1, 1] * 4, dtype=np.int64)
        datasets = {
            "tabular": ModalityFeatureDataset("tabular", tabular_X, labels),
            "text": ModalityFeatureDataset("text", text_X, labels),
        }
        model = UnifiedMultimodalTransformerClassifier(
            modalities={"tabular": 6, "text": 8},
            num_classes=2,
            d_model=16,
            nhead=4,
            num_layers=1,
            dim_feedforward=32,
            dropout=0.0,
        )
        trainer = UnpairedMultimodalTrainer(
            model=model,
            config=MultimodalTrainingConfig(
                epochs=1,
                batch_size=8,
                learning_rate=1e-2,
                steps_per_epoch=2,
                prototype_alignment_weight=0.1,
                contrastive_weight=0.1,
                contrastive_temperature=0.2,
            ),
            device="cpu",
        )

        result = trainer.fit(datasets)

        self.assertIn("prototype_alignment_loss", result.history)
        self.assertIn("contrastive_loss", result.history)
        self.assertGreaterEqual(result.history["prototype_alignment_loss"][0], 0.0)
        self.assertGreaterEqual(result.history["contrastive_loss"][0], 0.0)


if __name__ == "__main__":
    unittest.main()
