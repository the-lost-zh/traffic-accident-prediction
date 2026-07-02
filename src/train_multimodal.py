#!/usr/bin/env python3
"""
Unpaired multi-modal training script.

Pre-extract features first:
  python src/traffic_accident/cli/extract_features.py text \
      --input_csv data/captions.csv --text_column "Description" \
      --label_column Severity --output_dir features/text --encoder tfidf

  python src/traffic_accident/cli/extract_features.py image \
      --input_csv data/images.csv --image_path_column "path" \
      --label_column Severity --output_dir features/image --encoder siglip

Then train:
  python train_multimodal.py --config ../configs/multimodal_unpaired.yaml

Or run without a config file:
  python train_multimodal.py --tabular_features ../features/tabular.npy \
      --tabular_labels ../features/tabular_labels.npy \
      --image_features ../features/image.npy \
      --image_labels ../features/image_labels.npy
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

_src_dir = os.path.dirname(os.path.abspath(__file__))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from traffic_accident.config import flatten_cli_config, load_config
from traffic_accident.data.multimodal import ModalityFeatureDataset
from traffic_accident.models.multimodal import UnifiedMultimodalTransformerClassifier
from traffic_accident.training.multimodal import (
    MultimodalTrainingConfig,
    UnpairedMultimodalTrainer,
)
from traffic_accident.utils.io import ensure_dir, load_json, save_json
from traffic_accident.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a unified multi-modal classifier from pre-extracted features."
    )
    parser.add_argument("--config", type=str, default=None, help="YAML/JSON config file path")

    parser.add_argument("--tabular_features", type=str, default=None, help=".npy path for tabular features")
    parser.add_argument("--tabular_labels", type=str, default=None, help=".npy path for tabular labels")
    parser.add_argument("--tabular_input_dim", type=int, default=64)

    parser.add_argument("--image_features", type=str, default=None, help=".npy path for image features")
    parser.add_argument("--image_labels", type=str, default=None, help=".npy path for image labels")
    parser.add_argument("--image_input_dim", type=int, default=768)
    parser.add_argument("--image_projector", type=str, default="mlp")

    parser.add_argument("--text_features", type=str, default=None, help=".npy path for text features")
    parser.add_argument("--text_labels", type=str, default=None, help=".npy path for text labels")
    parser.add_argument("--text_input_dim", type=int, default=768)
    parser.add_argument("--text_projector", type=str, default="mlp")

    parser.add_argument("--output_root", type=str, default="outputs/multimodal_runs")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)

    # Training hyperparameters (also overridable from YAML config)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--gradient_clip_norm", type=float, default=1.0)
    parser.add_argument("--steps_per_epoch", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--prototype_alignment_weight", type=float, default=0.0)
    parser.add_argument("--contrastive_weight", type=float, default=0.0)
    parser.add_argument("--contrastive_temperature", type=float, default=0.2)

    # Model architecture
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dim_feedforward", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--tabular_projector_layers", type=int, default=1)

    config_args, _remaining = parser.parse_known_args()
    if config_args.config:
        parser.set_defaults(**flatten_cli_config(load_config(config_args.config)))
    return parser.parse_args()


def _extract_modality_configs(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    """Merge CLI args with YAML config to build modality configs."""
    if args.config:
        raw = load_config(args.config)
        modalities_cfg = raw.get("modalities", {})
        if modalities_cfg:
            return {
                name: {
                    "feature_path": mod.get("feature_path"),
                    "label_path": mod.get("label_path"),
                    "input_dim": mod.get("input_dim"),
                    "projector_type": mod.get("projector_type", "auto"),
                }
                for name, mod in modalities_cfg.items()
            }

    mods: dict[str, dict[str, Any]] = {}
    if args.tabular_features and args.tabular_labels:
        mods["tabular"] = {
            "feature_path": args.tabular_features,
            "label_path": args.tabular_labels,
            "input_dim": args.tabular_input_dim,
            "projector_type": "fttransformer",
        }
    if args.image_features and args.image_labels:
        mods["image"] = {
            "feature_path": args.image_features,
            "label_path": args.image_labels,
            "input_dim": args.image_input_dim,
            "projector_type": args.image_projector,
        }
    if args.text_features and args.text_labels:
        mods["text"] = {
            "feature_path": args.text_features,
            "label_path": args.text_labels,
            "input_dim": args.text_input_dim,
            "projector_type": args.text_projector,
        }

    if not mods:
        raise ValueError(
            "No modality configurations found. Provide --config or at least one "
            "set of --<modality>_features/--<modality>_labels arguments."
        )
    return mods


def _load_datasets(
    modality_configs: dict[str, dict[str, Any]]
) -> dict[str, ModalityFeatureDataset]:
    datasets: dict[str, ModalityFeatureDataset] = {}
    for name, cfg in modality_configs.items():
        feature_path = Path(cfg["feature_path"])
        label_path = Path(cfg["label_path"])
        if not feature_path.exists():
            print(f"Warning: skipping '{name}' — file not found: {feature_path}")
            continue
        if not label_path.exists():
            print(f"Warning: skipping '{name}' — file not found: {label_path}")
            continue

        features = np.load(feature_path).astype(np.float32)
        labels = np.load(label_path)
        datasets[name] = ModalityFeatureDataset(
            modality=name,
            features=features,
            labels=labels,
        )
        print(f"  [{name}] {len(features)} samples, dim={features.shape[1]}, "
              f"classes={len(np.unique(labels))}")

    if not datasets:
        raise RuntimeError("No modality datasets could be loaded. Check feature paths.")
    return datasets


def _model_config_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "d_model": args.d_model,
        "nhead": args.nhead,
        "num_layers": args.num_layers,
        "dim_feedforward": args.dim_feedforward,
        "dropout": args.dropout,
        "tabular_projector_layers": args.tabular_projector_layers,
    }


def main() -> dict[str, Any]:
    args = parse_args()
    set_seed(args.seed)

    modality_configs = _extract_modality_configs(args)
    model_cfg = _model_config_from_args(args)

    num_classes = 4  # severity levels 1-4 → 0-3

    print("=" * 70)
    print("多模态统一分类器训练")
    print("=" * 70)
    print(f"模态: {list(modality_configs.keys())}")
    print(f"特征维度: {', '.join(f'{name}: {cfg['input_dim']}' for name, cfg in modality_configs.items())}")
    print(f"设备: {args.device or ('cuda' if torch.cuda.is_available() else 'cpu')}")

    devices_per_modality = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if devices_per_modality > 0:
        print(f"可用GPU: {devices_per_modality}")

    print(f"\n模型配置: d_model={model_cfg['d_model']}, nhead={model_cfg['nhead']}, "
          f"layers={model_cfg['num_layers']}")
    print(f"训练: epochs={args.epochs}, batch_size={args.batch_size}, "
          f"lr={args.learning_rate}")

    from traffic_accident.utils.paths import make_run_paths
    run_paths = make_run_paths(args.output_root, args.run_name)
    for directory in [run_paths.checkpoints, run_paths.metrics, run_paths.artifacts]:
        ensure_dir(directory)

    print(f"\n输出目录: {run_paths.root}")

    print("\n[1/3] 加载模态数据")
    datasets = _load_datasets(modality_configs)

    print("\n[2/3] 构建统一多模态模型")
    modality_list = [
        {"name": name, "input_dim": cfg["input_dim"], "projector_type": cfg["projector_type"]}
        for name, cfg in modality_configs.items()
        if name in datasets
    ]
    model = UnifiedMultimodalTransformerClassifier(
        modalities=modality_list,
        num_classes=num_classes,
        d_model=model_cfg["d_model"],
        nhead=model_cfg["nhead"],
        num_layers=model_cfg["num_layers"],
        dim_feedforward=model_cfg["dim_feedforward"],
        dropout=model_cfg["dropout"],
        tabular_projector_layers=model_cfg.get("tabular_projector_layers", 1),
    )
    print(f"可训练参数: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    print("\n[3/3] 训练")
    training_config = MultimodalTrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        gradient_clip_norm=args.gradient_clip_norm,
        steps_per_epoch=args.steps_per_epoch,
        num_workers=args.num_workers,
        prototype_alignment_weight=args.prototype_alignment_weight,
        contrastive_weight=args.contrastive_weight,
        contrastive_temperature=args.contrastive_temperature,
    )

    checkpoint_path = run_paths.checkpoints / "best.pt"
    trainer = UnpairedMultimodalTrainer(
        model=model,
        config=training_config,
        device=args.device,
        checkpoint_path=checkpoint_path,
    )
    result = trainer.fit(datasets)

    run_config = {
        "modalities": modality_configs,
        "model_config": model_cfg,
        "training_config": asdict(training_config),
    }
    save_json(run_config, run_paths.root / "config.json")

    final_results = {
        "run_dir": str(run_paths.root),
        "checkpoint_path": str(checkpoint_path),
        "history": result.history,
        "metrics_by_modality": result.train_metrics_by_modality,
        "config": run_config,
    }
    save_json(final_results, run_paths.metrics / "final_results.json")

    # Save the model
    torch.save(model.state_dict(), run_paths.checkpoints / "final_model.pth")

    print("\n" + "=" * 70)
    print("训练完成!")
    print("=" * 70)
    for modality, metrics in result.train_metrics_by_modality.items():
        print(f"\n[{modality}]")
        print(f"  准确率: {metrics.get('accuracy', 0):.4f}")
        print(f"  F1 (macro): {metrics.get('f1_macro', 0):.4f}")
        print(f"  F1 (weighted): {metrics.get('f1_weighted', 0):.4f}")
    print(f"\n模型保存至: {run_paths.checkpoints / 'final_model.pth'}")
    print(f"结果保存至: {run_paths.metrics / 'final_results.json'}")

    return final_results


if __name__ == "__main__":
    main()
