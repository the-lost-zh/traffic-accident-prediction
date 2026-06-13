import argparse
import importlib
import os
import sys
import unittest
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from traffic_accident.cli.train_tabular import run_training
from traffic_accident.inference import TabularRunPredictor


class InferenceAndApiTest(unittest.TestCase):
    def test_predictor_and_api_load_from_run_dir(self):
        tmp_dir = PROJECT_ROOT / ".test_tmp" / "inference_api"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        csv_path = tmp_dir / "accidents.csv"
        output_root = tmp_dir / "runs"

        rows = []
        for i in range(80):
            rows.append(
                {
                    "Severity": (i % 4) + 1,
                    "Start_Lng": -120.0 + i * 0.01,
                    "Start_Lat": 35.0 + i * 0.01,
                    "Distance(mi)": float(i % 8),
                    "Temperature(F)": float(55 + i % 15),
                    "Junction": bool(i % 2),
                    "Weather_Condition": ["Clear", "Rain", "Fog", "Snow"][i % 4],
                }
            )
        pd.DataFrame(rows).to_csv(csv_path, index=False)

        args = argparse.Namespace(
            data_path=str(csv_path),
            output_root=str(output_root),
            run_name="predict_unit_test_run",
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
        run_dir = output_root / "predict_unit_test_run"

        predictor = TabularRunPredictor(run_dir, device="cpu")
        prediction = predictor.predict_one({"Start_Lng": -120.0, "Distance(mi)": 1.0, "Weather_Condition": "Clear"})
        self.assertIn(prediction["severity"], [1, 2, 3, 4])
        self.assertEqual(len(prediction["probabilities"]), 4)

        previous_run_dir = os.environ.get("TRAFFIC_ACCIDENT_RUN_DIR")
        previous_device = os.environ.get("TRAFFIC_ACCIDENT_DEVICE")
        os.environ["TRAFFIC_ACCIDENT_RUN_DIR"] = str(run_dir)
        os.environ["TRAFFIC_ACCIDENT_DEVICE"] = "cpu"
        try:
            api_app = importlib.import_module("api.app")
            api_app.load_predictor()
            client = api_app.app.test_client()
            response = client.post(
                "/api/predict",
                json={
                    "start_lng": -120.0,
                    "start_lat": 35.0,
                    "distance": 1.0,
                    "junction": 1,
                    "weather_condition": 0,
                },
            )
            self.assertEqual(response.status_code, 200)
            payload = response.get_json()
            self.assertEqual(payload["model_source"], "run")
            self.assertIn(payload["severity_level"], [1, 2, 3, 4])
        finally:
            if previous_run_dir is None:
                os.environ.pop("TRAFFIC_ACCIDENT_RUN_DIR", None)
            else:
                os.environ["TRAFFIC_ACCIDENT_RUN_DIR"] = previous_run_dir
            if previous_device is None:
                os.environ.pop("TRAFFIC_ACCIDENT_DEVICE", None)
            else:
                os.environ["TRAFFIC_ACCIDENT_DEVICE"] = previous_device


if __name__ == "__main__":
    unittest.main()

