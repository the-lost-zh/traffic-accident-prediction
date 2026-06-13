import argparse
import sys
import unittest
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from traffic_accident.cli.evaluate_tabular import evaluate_run
from traffic_accident.cli.train_tabular import run_training


class EvaluateTabularCliTest(unittest.TestCase):
    def test_evaluate_run_loads_saved_artifacts(self):
        tmp_dir = PROJECT_ROOT / ".test_tmp" / "evaluate_tabular_cli"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        csv_path = tmp_dir / "accidents.csv"
        output_root = tmp_dir / "runs"

        rows = []
        for i in range(80):
            rows.append(
                {
                    "Severity": (i % 4) + 1,
                    "Distance": float(i % 8),
                    "Temperature": float(55 + i % 15),
                    "Junction": "yes" if i % 2 else "no",
                    "Weather": ["Clear", "Rain", "Fog", "Snow"][i % 4],
                }
            )
        pd.DataFrame(rows).to_csv(csv_path, index=False)

        args = argparse.Namespace(
            data_path=str(csv_path),
            output_root=str(output_root),
            run_name="eval_unit_test_run",
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
        run_training(args)

        run_dir = output_root / "eval_unit_test_run"
        result = evaluate_run(run_dir, device="cpu")

        self.assertTrue((run_dir / "metrics" / "evaluation.json").exists())
        self.assertEqual(result["num_test_samples"], 12)
        self.assertIn("accuracy", result["test_metrics"])


if __name__ == "__main__":
    unittest.main()

