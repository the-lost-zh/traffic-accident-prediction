import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from traffic_accident.models import create_model
from traffic_accident.training import SupervisedTrainer, TrainingConfig


class SupervisedTrainerTest(unittest.TestCase):
    def test_fit_predict_and_save_result(self):
        rng = np.random.default_rng(42)
        X = rng.normal(size=(48, 4)).astype(np.float32)
        y = (X[:, 0] + X[:, 1] > 0).astype(np.int64)
        X_train, X_val = X[:36], X[36:]
        y_train, y_val = y[:36], y[36:]

        model = create_model("linear", input_dim=4, num_classes=2)
        output_dir = PROJECT_ROOT / ".test_tmp" / "training"
        checkpoint_path = output_dir / "best.pt"
        result_path = output_dir / "result.json"
        config = TrainingConfig(
            epochs=2,
            batch_size=8,
            learning_rate=1e-2,
            early_stopping_patience=3,
        )

        trainer = SupervisedTrainer(
            model=model,
            config=config,
            device="cpu",
            checkpoint_path=checkpoint_path,
        )
        result = trainer.fit(X_train, y_train, X_val, y_val)
        predictions = trainer.predict(X_val)
        saved_result_path = trainer.save_result(result, result_path)

        self.assertEqual(predictions.shape, y_val.shape)
        self.assertTrue(checkpoint_path.exists())
        self.assertTrue(saved_result_path.exists())
        self.assertIn("accuracy", result.train_metrics)
        self.assertEqual(len(result.history["train_loss"]), 2)


if __name__ == "__main__":
    unittest.main()

