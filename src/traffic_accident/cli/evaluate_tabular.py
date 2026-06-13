from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from traffic_accident.evaluation import calculate_metrics
from traffic_accident.models import create_model
from traffic_accident.preprocessing.tabular import TabularPreprocessor
from traffic_accident.utils.io import load_json, load_pickle, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained tabular run on its saved test split.")
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--data_path", type=str, default=None, help="Optional override for the data path saved in config.json.")
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main() -> dict[str, Any]:
    args = parse_args()
    return evaluate_run(Path(args.run_dir), data_path=args.data_path, device=args.device)


def evaluate_run(run_dir: str | Path, data_path: str | Path | None = None, device: str | None = None) -> dict[str, Any]:
    run_path = Path(run_dir)
    config = load_json(run_path / "config.json")
    resolved_data_path = Path(data_path or config["data_path"])

    preprocessor: TabularPreprocessor = load_pickle(run_path / "artifacts" / "preprocessor.pkl")
    test_df = _load_saved_test_split(resolved_data_path, config)
    X_test, y_test = preprocessor.transform(test_df)

    model = create_model(
        model_type=config["model_type"],
        input_dim=X_test.shape[1],
        num_classes=int(config["num_classes"]),
        config=config["model_config"],
    )
    torch_device = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dict = torch.load(run_path / "checkpoints" / "best.pt", map_location=torch_device)
    model.load_state_dict(state_dict)
    model.to(torch_device)
    model.eval()

    predictions = _predict(model, X_test, torch_device, batch_size=config["training_config"]["batch_size"])
    metrics = calculate_metrics(y_test, predictions)
    result = {
        "run_dir": str(run_path),
        "data_path": str(resolved_data_path),
        "num_test_samples": int(len(y_test)),
        "test_metrics": metrics,
    }
    save_json(result, run_path / "metrics" / "evaluation.json")

    print(f"Run directory: {run_path}")
    print(f"Test samples: {len(y_test)}")
    print(f"Test accuracy: {metrics['accuracy']:.4f}")
    return result


def _load_saved_test_split(data_path: Path, config: dict[str, Any]) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    split_config = config["split"]
    target = TabularPreprocessor._encode_target(df["Severity"])
    train_val_df, test_df = train_test_split(
        df,
        test_size=split_config["test_ratio"],
        random_state=split_config["seed"],
        stratify=target,
    )
    return test_df


def _predict(model: torch.nn.Module, X: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    predictions: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            batch = torch.as_tensor(X[start : start + batch_size], dtype=torch.float32, device=device)
            predictions.append(model(batch).argmax(dim=1).cpu())
    return torch.cat(predictions).numpy()


if __name__ == "__main__":
    main()

