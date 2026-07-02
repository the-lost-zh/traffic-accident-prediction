from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from traffic_accident.features import (
    save_image_feature_artifacts,
    save_tabular_feature_artifacts,
    save_text_feature_artifacts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract feature matrices for unpaired multimodal training.")
    subparsers = parser.add_subparsers(dest="modality", required=True)

    tabular_parser = subparsers.add_parser("tabular", help="Preprocess tabular CSV and save features as .npy.")
    tabular_parser.add_argument("--input_csv", type=str, required=True)
    tabular_parser.add_argument("--output_dir", type=str, required=True)
    tabular_parser.add_argument("--target_col", type=str, default="Severity")
    tabular_parser.add_argument("--train_ratio", type=float, default=0.7)
    tabular_parser.add_argument("--val_ratio", type=float, default=0.15)
    tabular_parser.add_argument("--test_ratio", type=float, default=0.15)
    tabular_parser.add_argument("--random_state", type=int, default=42)
    tabular_parser.add_argument("--max_missing_ratio", type=float, default=0.5)
    tabular_parser.add_argument("--max_categorical_unique_ratio", type=float, default=0.05)

    text_parser = subparsers.add_parser("text", help="Extract text features from a CSV column.")
    text_parser.add_argument("--input_csv", type=str, required=True)
    text_parser.add_argument("--text_column", type=str, required=True)
    text_parser.add_argument("--label_column", type=str, default=None)
    text_parser.add_argument("--output_dir", type=str, required=True)
    text_parser.add_argument("--max_features", type=int, default=768)
    text_parser.add_argument("--encoder", type=str, choices=["tfidf", "clip", "siglip"], default="tfidf")
    text_parser.add_argument("--model_name", type=str, default=None)
    text_parser.add_argument("--batch_size", type=int, default=32)
    text_parser.add_argument("--device", type=str, default=None)
    text_parser.add_argument("--no_normalize", action="store_true")

    image_parser = subparsers.add_parser("image", help="Extract lightweight RGB image features from CSV paths.")
    image_parser.add_argument("--input_csv", type=str, required=True)
    image_parser.add_argument("--image_path_column", type=str, required=True)
    image_parser.add_argument("--label_column", type=str, default=None)
    image_parser.add_argument("--image_root", type=str, default=None)
    image_parser.add_argument("--output_dir", type=str, required=True)
    image_parser.add_argument("--bins", type=int, default=16)
    image_parser.add_argument("--encoder", type=str, choices=["color", "clip", "siglip"], default="color")
    image_parser.add_argument("--model_name", type=str, default=None)
    image_parser.add_argument("--batch_size", type=int, default=32)
    image_parser.add_argument("--device", type=str, default=None)
    image_parser.add_argument("--no_normalize", action="store_true")

    return parser.parse_args()


def main() -> dict[str, Any]:
    args = parse_args()
    return run_extraction(args)


def run_extraction(args: argparse.Namespace) -> dict[str, Any]:
    if args.modality == "tabular":
        metadata = save_tabular_feature_artifacts(
            data_path=args.input_csv,
            output_dir=args.output_dir,
            target_col=getattr(args, "target_col", "Severity"),
            train_ratio=getattr(args, "train_ratio", 0.7),
            val_ratio=getattr(args, "val_ratio", 0.15),
            test_ratio=getattr(args, "test_ratio", 0.15),
            random_state=getattr(args, "random_state", 42),
            max_missing_ratio=getattr(args, "max_missing_ratio", 0.5),
            max_categorical_unique_ratio=getattr(args, "max_categorical_unique_ratio", 0.05),
        )
        print(f"Feature directory: {Path(args.output_dir)}")
        print(f"Feature dimension: {metadata['feature_dim']}")
        print(f"Samples: {metadata['num_samples']}")
        return metadata

    df = pd.read_csv(args.input_csv)
    if args.modality == "text":
        metadata = save_text_feature_artifacts(
            df=df,
            text_column=args.text_column,
            label_column=args.label_column,
            output_dir=args.output_dir,
            max_features=getattr(args, "max_features", 768),
            encoder=getattr(args, "encoder", "tfidf"),
            model_name=getattr(args, "model_name", None),
            batch_size=getattr(args, "batch_size", 32),
            device=getattr(args, "device", None),
            normalize=not getattr(args, "no_normalize", False),
        )
    elif args.modality == "image":
        metadata = save_image_feature_artifacts(
            df=df,
            image_path_column=args.image_path_column,
            label_column=args.label_column,
            image_root=args.image_root,
            output_dir=args.output_dir,
            bins=getattr(args, "bins", 16),
            encoder=getattr(args, "encoder", "color"),
            model_name=getattr(args, "model_name", None),
            batch_size=getattr(args, "batch_size", 32),
            device=getattr(args, "device", None),
            normalize=not getattr(args, "no_normalize", False),
        )
    else:
        raise ValueError(f"Unsupported modality: {args.modality}")

    print(f"Feature directory: {Path(args.output_dir)}")
    print(f"Feature dimension: {metadata['feature_dim']}")
    print(f"Samples: {metadata['num_samples']}")
    return metadata


if __name__ == "__main__":
    main()
