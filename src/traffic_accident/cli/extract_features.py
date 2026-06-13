from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from traffic_accident.features import save_image_feature_artifacts, save_text_feature_artifacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract feature matrices for unpaired multimodal training.")
    subparsers = parser.add_subparsers(dest="modality", required=True)

    text_parser = subparsers.add_parser("text", help="Extract TF-IDF text features from a CSV column.")
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
