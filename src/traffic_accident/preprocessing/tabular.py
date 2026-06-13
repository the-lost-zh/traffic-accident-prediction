from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from traffic_accident.utils.io import ensure_dir, save_json, save_pickle


@dataclass
class TabularDataBundle:
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    numeric_features: list[str]
    categorical_features: list[str]
    preprocessor: "TabularPreprocessor"

    def as_dict(self) -> dict[str, Any]:
        return {
            "X_train": self.X_train,
            "X_val": self.X_val,
            "X_test": self.X_test,
            "y_train": self.y_train,
            "y_val": self.y_val,
            "y_test": self.y_test,
            "feature_names": self.feature_names,
            "numeric_features": self.numeric_features,
            "categorical_features": self.categorical_features,
            "preprocessor": self.preprocessor,
        }


class TabularPreprocessor:
    """Fit-on-train tabular preprocessing for the accident severity dataset."""

    def __init__(
        self,
        target_col: str = "Severity",
        max_missing_ratio: float = 0.5,
        max_categorical_unique_ratio: float = 0.05,
    ) -> None:
        self.target_col = target_col
        self.max_missing_ratio = max_missing_ratio
        self.max_categorical_unique_ratio = max_categorical_unique_ratio

        self.feature_names: list[str] = []
        self.numeric_features: list[str] = []
        self.categorical_features: list[str] = []
        self.drop_columns: list[str] = []
        self.numeric_fill_values: dict[str, float] = {}
        self.categorical_fill_values: dict[str, str] = {}
        self.scaler = StandardScaler()
        self.encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        self.is_fitted = False

    def fit(self, df: pd.DataFrame) -> "TabularPreprocessor":
        self._validate_input(df)
        features = df.drop(columns=[self.target_col]).copy()

        for col in list(features.columns):
            missing_ratio = features[col].isna().mean()
            if missing_ratio > self.max_missing_ratio:
                self.drop_columns.append(col)
                continue

            if pd.api.types.is_numeric_dtype(features[col]):
                self.numeric_features.append(col)
                self.numeric_fill_values[col] = float(features[col].median())
                continue

            unique_ratio = features[col].nunique(dropna=True) / max(len(features), 1)
            if unique_ratio <= self.max_categorical_unique_ratio:
                self.categorical_features.append(col)
                mode = features[col].mode(dropna=True)
                self.categorical_fill_values[col] = str(mode.iloc[0]) if not mode.empty else "Unknown"
            else:
                self.drop_columns.append(col)

        self.feature_names = self.numeric_features + self.categorical_features

        numeric_df = self._prepare_numeric(features)
        categorical_df = self._prepare_categorical(features)

        if self.numeric_features:
            self.scaler.fit(numeric_df)
        if self.categorical_features:
            self.encoder.fit(categorical_df)

        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        if not self.is_fitted:
            raise RuntimeError("TabularPreprocessor must be fitted before transform().")
        self._validate_input(df)

        X = self.transform_features(df.drop(columns=[self.target_col]))
        y = self._encode_target(df[self.target_col])
        return X, y

    def transform_features(self, features: pd.DataFrame | dict[str, Any] | list[dict[str, Any]]) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("TabularPreprocessor must be fitted before transform_features().")

        if isinstance(features, dict):
            features_df = pd.DataFrame([features])
        elif isinstance(features, list):
            features_df = pd.DataFrame(features)
        else:
            features_df = features.copy()

        parts: list[np.ndarray] = []

        if self.numeric_features:
            numeric_df = self._prepare_numeric(features_df)
            parts.append(self.scaler.transform(numeric_df))

        if self.categorical_features:
            categorical_df = self._prepare_categorical(features_df)
            parts.append(self.encoder.transform(categorical_df))

        if parts:
            X = np.concatenate(parts, axis=1).astype(np.float32)
        else:
            X = np.empty((len(features_df), 0), dtype=np.float32)
        return X

    def fit_transform(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        return self.fit(df).transform(df)

    def save(self, output_dir: str | Path) -> dict[str, Path]:
        output_path = ensure_dir(output_dir)
        artifact_path = save_pickle(self, output_path / "preprocessor.pkl")
        metadata_path = save_json(self.metadata(), output_path / "preprocessor_metadata.json")
        return {"artifact": artifact_path, "metadata": metadata_path}

    def metadata(self) -> dict[str, Any]:
        return {
            "target_col": self.target_col,
            "feature_names": self.feature_names,
            "numeric_features": self.numeric_features,
            "categorical_features": self.categorical_features,
            "drop_columns": self.drop_columns,
            "numeric_fill_values": self.numeric_fill_values,
            "categorical_fill_values": self.categorical_fill_values,
            "max_missing_ratio": self.max_missing_ratio,
            "max_categorical_unique_ratio": self.max_categorical_unique_ratio,
        }

    def _prepare_numeric(self, features: pd.DataFrame) -> pd.DataFrame:
        numeric_df = pd.DataFrame(index=features.index)
        for col in self.numeric_features:
            fill_value = self.numeric_fill_values[col]
            if col in features.columns:
                numeric_df[col] = pd.to_numeric(features[col], errors="coerce").fillna(fill_value)
            else:
                numeric_df[col] = fill_value
        return numeric_df

    def _prepare_categorical(self, features: pd.DataFrame) -> pd.DataFrame:
        categorical_df = pd.DataFrame(index=features.index)
        for col in self.categorical_features:
            fill_value = self.categorical_fill_values[col]
            if col in features.columns:
                categorical_df[col] = features[col].astype("object").where(features[col].notna(), fill_value).astype(str)
            else:
                categorical_df[col] = fill_value
        return categorical_df

    def _validate_input(self, df: pd.DataFrame) -> None:
        if self.target_col not in df.columns:
            raise ValueError(f"Missing target column: {self.target_col}")

    @staticmethod
    def _encode_target(target: pd.Series) -> np.ndarray:
        values = pd.to_numeric(target, errors="raise").to_numpy()
        return (values.astype(np.int64) - 1).astype(np.int64)


def load_tabular_data(
    data_path: str | Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
    target_col: str = "Severity",
    artifact_dir: str | Path | None = None,
    max_missing_ratio: float = 0.5,
    max_categorical_unique_ratio: float = 0.05,
) -> TabularDataBundle:
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    df = pd.read_csv(data_path)
    if target_col not in df.columns:
        raise ValueError(f"Missing target column: {target_col}")

    target = TabularPreprocessor._encode_target(df[target_col])
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_ratio,
        random_state=random_state,
        stratify=target,
    )
    train_val_target = TabularPreprocessor._encode_target(train_val_df[target_col])
    val_size = val_ratio / (train_ratio + val_ratio)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_size,
        random_state=random_state,
        stratify=train_val_target,
    )

    preprocessor = TabularPreprocessor(
        target_col=target_col,
        max_missing_ratio=max_missing_ratio,
        max_categorical_unique_ratio=max_categorical_unique_ratio,
    )
    X_train, y_train = preprocessor.fit_transform(train_df)
    X_val, y_val = preprocessor.transform(val_df)
    X_test, y_test = preprocessor.transform(test_df)

    if artifact_dir is not None:
        preprocessor.save(artifact_dir)

    return TabularDataBundle(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        feature_names=preprocessor.feature_names,
        numeric_features=preprocessor.numeric_features,
        categorical_features=preprocessor.categorical_features,
        preprocessor=preprocessor,
    )
