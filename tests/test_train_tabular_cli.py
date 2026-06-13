import argparse
import sys
import unittest
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from traffic_accident.cli.train_tabular import run_training


class TrainTabularCliTest(unittest.TestCase):
    def test_run_training_creates_expected_run_artifacts(self):
        tmp_dir = PROJECT_ROOT / ".test_tmp" / "train_tabular_cli"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        csv_path = tmp_dir / "accidents.csv"
        output_root = tmp_dir / "runs"

        rows = []
        for i in range(80):
            severity = (i % 4) + 1
            rows.append(
                {
                    "Severity": severity,
                    "Distance": float(i % 10),
                    "Temperature": float(50 + i % 20),
                    "Junction": "yes" if i % 2 else "no",
                    "Weather": ["Clear", "Rain", "Fog", "Snow"][i % 4],
                    "Description": f"unique text {i}",
                }
            )
        pd.DataFrame(rows).to_csv(csv_path, index=False)

        args = argparse.Namespace(
            data_path=str(csv_path),
            output_root=str(output_root),
            run_name="unit_test_run",
            model_type="linear",
            epochs=1,
            batch_size=16,
            learning_rate=1e-2,
            weight_decay=1e-5,
            early_stopping_patience=3,
            dropout=0.0,
            hidden_dims="16,8",
            d_model=16,
            nhead=4,
            num_layers=1,
            dim_feedforward=32,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            max_missing_ratio=0.5,
            max_categorical_unique_ratio=0.5,
            seed=42,
            device="cpu",
        )

        result = run_training(args)
        run_dir = output_root / "unit_test_run"

        self.assertTrue((run_dir / "checkpoints" / "best.pt").exists())
        self.assertTrue((run_dir / "metrics" / "final_results.json").exists())
        self.assertTrue((run_dir / "artifacts" / "preprocessor.pkl").exists())
        self.assertTrue((run_dir / "config.json").exists())
        self.assertIn("test_metrics", result)
        self.assertIn("accuracy", result["test_metrics"])


if __name__ == "__main__":
    unittest.main()

