import sys
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from traffic_accident.cli import train_tabular
from traffic_accident.config import flatten_cli_config, load_config
import train as legacy_train


class ConfigAndLegacyWrapperTest(unittest.TestCase):
    def test_yaml_config_flattens_grouped_cli_sections(self):
        config_path = PROJECT_ROOT / ".test_tmp" / "config" / "tabular.yaml"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(
            "\n".join(
                [
                    "model_type: mlp",
                    "training:",
                    "  epochs: 3",
                    "  batch_size: 32",
                    "model:",
                    "  hidden_dims: [16, 8]",
                    "  dropout: 0.2",
                ]
            ),
            encoding="utf-8",
        )

        flattened = flatten_cli_config(load_config(config_path))

        self.assertEqual(flattened["model_type"], "mlp")
        self.assertEqual(flattened["epochs"], 3)
        self.assertEqual(flattened["batch_size"], 32)
        self.assertEqual(flattened["hidden_dims"], "16,8")

    def test_train_tabular_cli_config_defaults_can_be_overridden(self):
        config_path = PROJECT_ROOT / ".test_tmp" / "config" / "train_tabular.yaml"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text("model_type: mlp\nepochs: 9\nbatch_size: 64\n", encoding="utf-8")

        with patch.object(sys, "argv", ["prog", "--config", str(config_path), "--epochs", "2"]):
            args = train_tabular.parse_args()

        self.assertEqual(args.model_type, "mlp")
        self.assertEqual(args.epochs, 2)
        self.assertEqual(args.batch_size, 64)

    def test_legacy_train_py_maps_output_dir_and_ignores_legacy_flags(self):
        with patch.object(
            sys,
            "argv",
            [
                "train.py",
                "--data_path",
                "data/sample.csv",
                "--output_dir",
                "legacy_results",
                "--model_type",
                "mlp",
                "--skip_shap",
            ],
        ):
            args = legacy_train.parse_args()

        self.assertEqual(args.output_root, "legacy_results")
        self.assertEqual(args.data_path, "data/sample.csv")
        self.assertTrue(args.skip_shap)


if __name__ == "__main__":
    unittest.main()
