import argparse
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from traffic_accident.cli.extract_features import run_extraction
from traffic_accident.features.vision_language import resolve_vision_language_model_name


class FeatureExtractionTest(unittest.TestCase):
    def test_text_feature_cli_writes_features_labels_and_metadata(self):
        tmp_dir = PROJECT_ROOT / ".test_tmp" / "features" / "text"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        csv_path = tmp_dir / "text.csv"
        pd.DataFrame(
            {
                "description": ["clear road crash", "rainy road collision", "fog low visibility"],
                "Severity": [1, 2, 4],
            }
        ).to_csv(csv_path, index=False)

        metadata = run_extraction(
            argparse.Namespace(
                modality="text",
                input_csv=str(csv_path),
                text_column="description",
                label_column="Severity",
                output_dir=str(tmp_dir / "out"),
                max_features=8,
            )
        )

        features = np.load(tmp_dir / "out" / "features.npy")
        labels = np.load(tmp_dir / "out" / "labels.npy")
        self.assertEqual(features.shape[0], 3)
        self.assertLessEqual(features.shape[1], 8)
        self.assertEqual(labels.tolist(), [0, 1, 3])
        self.assertEqual(metadata["modality"], "text")

    def test_image_feature_cli_writes_features_labels_and_metadata(self):
        from PIL import Image

        tmp_dir = PROJECT_ROOT / ".test_tmp" / "features" / "image"
        image_dir = tmp_dir / "images"
        image_dir.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (4, 3), color=(255, 0, 0)).save(image_dir / "red.png")
        Image.new("RGB", (4, 3), color=(0, 255, 0)).save(image_dir / "green.png")
        csv_path = tmp_dir / "images.csv"
        pd.DataFrame({"image_path": ["red.png", "green.png"], "Severity": [1, 2]}).to_csv(csv_path, index=False)

        metadata = run_extraction(
            argparse.Namespace(
                modality="image",
                input_csv=str(csv_path),
                image_path_column="image_path",
                label_column="Severity",
                image_root=str(image_dir),
                output_dir=str(tmp_dir / "out"),
                bins=4,
            )
        )

        features = np.load(tmp_dir / "out" / "features.npy")
        labels = np.load(tmp_dir / "out" / "labels.npy")
        self.assertEqual(features.shape, (2, 21))
        self.assertEqual(labels.tolist(), [0, 1])
        self.assertEqual(metadata["modality"], "image")
        self.assertEqual(metadata["encoder"], "color")

    def test_vision_language_model_name_resolution(self):
        self.assertEqual(
            resolve_vision_language_model_name("siglip"),
            "google/siglip-base-patch16-224",
        )
        self.assertEqual(
            resolve_vision_language_model_name("clip", "openai/clip-vit-base-patch32"),
            "openai/clip-vit-base-patch32",
        )
        with self.assertRaises(ValueError):
            resolve_vision_language_model_name("unknown")


if __name__ == "__main__":
    unittest.main()
