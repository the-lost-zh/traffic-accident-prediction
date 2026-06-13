from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from traffic_accident.config import flatten_cli_config, load_config
from traffic_accident.evaluation import calculate_metrics
from traffic_accident.models import create_model
from traffic_accident.preprocessing import load_tabular_data
from traffic_accident.training import SupervisedTrainer, TrainingConfig
from traffic_accident.utils.io import ensure_dir, save_json
from traffic_accident.utils.paths import make_run_paths
from traffic_accident.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a tabular traffic accident severity classifier.")
    parser.add_argument("--config", type=str, default=None, help="Optional JSON/YAML config file.")
    parser.add_argument("--data_path", type=str, default="data/US_Accidents_March23.csv")
    parser.add_argument("--output_root", type=str, default="outputs/runs")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--model_type", type=str, default="linear", choices=["linear", "mlp", "transformer", "fttransformer"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--early_stopping_patience", type=int, default=15)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--hidden_dims", type=str, default="128,64")
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dim_feedforward", type=int, default=256)
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    parser.add_argument("--max_missing_ratio", type=float, default=0.5)
    parser.add_argument("--max_categorical_unique_ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)

    config_args, _remaining = parser.parse_known_args()
    if config_args.config:
        parser.set_defaults(**flatten_cli_config(load_config(config_args.config)))
    return parser.parse_args()


def main() -> dict[str, Any]:
    args = parse_args()
    return run_training(args)


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    set_seed(args.seed)
    run_paths = make_run_paths(args.output_root, args.run_name)
    for directory in [run_paths.checkpoints, run_paths.metrics, run_paths.figures, run_paths.artifacts]:
        ensure_dir(directory)

    model_config = _model_config_from_args(args)
    training_config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        early_stopping_patience=args.early_stopping_patience,
    )

    bundle = load_tabular_data(
        data_path=args.data_path,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_state=args.seed,
        artifact_dir=run_paths.artifacts,
        max_missing_ratio=args.max_missing_ratio,
        max_categorical_unique_ratio=args.max_categorical_unique_ratio,
    )

    num_classes = int(len(np.unique(bundle.y_train)))
    model = create_model(
        model_type=args.model_type,
        input_dim=bundle.X_train.shape[1],
        num_classes=num_classes,
        config=model_config,
    )

    checkpoint_path = run_paths.checkpoints / "best.pt"
    trainer = SupervisedTrainer(
        model=model,
        config=training_config,
        device=args.device,
        checkpoint_path=checkpoint_path,
    )
    training_result = trainer.fit(bundle.X_train, bundle.y_train, bundle.X_val, bundle.y_val)

    test_pred = trainer.predict(bundle.X_test)
    test_metrics = calculate_metrics(bundle.y_test, test_pred)

    run_config = {
        "data_path": str(Path(args.data_path)),
        "model_type": args.model_type,
        "model_config": model_config,
        "training_config": asdict(training_config),
        "split": {
            "train_ratio": args.train_ratio,
            "val_ratio": args.val_ratio,
            "test_ratio": args.test_ratio,
            "seed": args.seed,
        },
        "preprocessing": {
            "max_missing_ratio": args.max_missing_ratio,
            "max_categorical_unique_ratio": args.max_categorical_unique_ratio,
        },
        "feature_names": bundle.feature_names,
        "numeric_features": bundle.numeric_features,
        "categorical_features": bundle.categorical_features,
        "num_classes": num_classes,
    }

    final_results = {
        "run_dir": str(run_paths.root),
        "checkpoint_path": str(checkpoint_path),
        "best_val_metric": training_result.best_val_metric,
        "train_metrics": training_result.train_metrics,
        "val_metrics": training_result.val_metrics,
        "test_metrics": test_metrics,
        "history": training_result.history,
        "config": run_config,
    }

    save_json(run_config, run_paths.root / "config.json")
    save_json(final_results, run_paths.metrics / "final_results.json")

    print(f"Run directory: {run_paths.root}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Validation accuracy: {training_result.val_metrics['accuracy']:.4f}")
    print(f"Test accuracy: {test_metrics['accuracy']:.4f}")

    return final_results


def _model_config_from_args(args: argparse.Namespace) -> dict[str, Any]:
    hidden_dims = tuple(int(value.strip()) for value in args.hidden_dims.split(",") if value.strip())
    return {
        "hidden_dims": hidden_dims,
        "dropout": args.dropout,
        "d_model": args.d_model,
        "nhead": args.nhead,
        "num_layers": args.num_layers,
        "dim_feedforward": args.dim_feedforward,
    }


if __name__ == "__main__":
    main()
