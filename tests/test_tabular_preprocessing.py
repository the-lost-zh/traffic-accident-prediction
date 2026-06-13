import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from traffic_accident.preprocessing import TabularPreprocessor, load_tabular_data


class TabularPreprocessorTest(unittest.TestCase):
    def test_fit_transform_uses_train_schema_and_handles_unknown_categories(self):
        train_df = pd.DataFrame(
            {
                "Severity": [1, 2, 3, 4, 1, 2, 3, 4],
                "Distance": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0],
                "Weather": ["Clear", "Rain", "Clear", "Rain", "Clear", "Rain", None, "Clear"],
                "HighCardinalityText": [f"id-{i}" for i in range(8)],
                "MostlyMissing": [None, None, None, None, None, "x", None, None],
            }
        )
        test_df = pd.DataFrame(
            {
                "Severity": [1, 4],
                "Distance": [10.0, np.nan],
                "Weather": ["Snow", None],
                "HighCardinalityText": ["new-1", "new-2"],
                "MostlyMissing": ["x", None],
            }
        )

        preprocessor = TabularPreprocessor(max_categorical_unique_ratio=0.5)
        X_train, y_train = preprocessor.fit_transform(train_df)
        X_test, y_test = preprocessor.transform(test_df)

        self.assertEqual(preprocessor.feature_names, ["Distance", "Weather"])
        self.assertEqual(preprocessor.drop_columns, ["HighCardinalityText", "MostlyMissing"])
        self.assertEqual(X_train.shape, (8, 2))
        self.assertEqual(X_test.shape, (2, 2))
        np.testing.assert_array_equal(y_train, np.array([0, 1, 2, 3, 0, 1, 2, 3]))
        np.testing.assert_array_equal(y_test, np.array([0, 3]))
        self.assertEqual(X_test[0, 1], -1.0)

        partial_X = preprocessor.transform_features({"Distance": 3.0})
        self.assertEqual(partial_X.shape, (1, 2))

    def test_load_tabular_data_saves_preprocessor_artifacts(self):
        rows = []
        labels = [1, 2, 3, 4] * 10
        for i, severity in enumerate(labels):
            rows.append(
                {
                    "Severity": severity,
                    "Distance": float(i),
                    "Junction": "yes" if i % 2 else "no",
                    "Description": f"unique free text {i}",
                }
            )
        df = pd.DataFrame(rows)

        tmp_path = PROJECT_ROOT / ".test_tmp"
        tmp_path.mkdir(exist_ok=True)
        csv_path = tmp_path / "accidents.csv"
        artifact_dir = tmp_path / "artifacts"
        df.to_csv(csv_path, index=False)

        bundle = load_tabular_data(
            csv_path,
            artifact_dir=artifact_dir,
            max_categorical_unique_ratio=0.5,
        )

        self.assertEqual(bundle.X_train.shape[1], 2)
        self.assertEqual(bundle.X_val.shape[1], 2)
        self.assertEqual(bundle.X_test.shape[1], 2)
        self.assertEqual(set(bundle.y_train), {0, 1, 2, 3})
        self.assertTrue((artifact_dir / "preprocessor.pkl").exists())
        self.assertTrue((artifact_dir / "preprocessor_metadata.json").exists())


if __name__ == "__main__":
    unittest.main()
