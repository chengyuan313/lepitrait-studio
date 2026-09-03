"""Command-line interface for dataset preparation, training and inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from PIL import Image

from .manifest import assign_grouped_splits, validate_manifest


def _validate(args) -> None:
    frame = pd.read_csv(args.manifest)
    report = validate_manifest(
        frame,
        require_files=not args.skip_file_check,
        require_split=not args.before_split,
        minimum_train_images=args.minimum_train_images,
    )
    print(
        f"images={report.images} specimens={report.specimens} species={report.species} "
        f"valid={report.valid}"
    )
    for warning in report.warnings:
        print(f"WARNING: {warning}")
    if report.errors:
        raise SystemExit("Manifest invalid:\n- " + "\n- ".join(report.errors))


def _split(args) -> None:
    frame = pd.read_csv(args.manifest)
    result = assign_grouped_splits(
        frame,
        seed=args.seed,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
    )
    result.to_csv(args.output, index=False)
    print(result.groupby("split").size().to_string())
    print(f"Wrote {args.output}")


def _train(args) -> None:
    from .training import TrainingConfig, train

    checkpoint = train(TrainingConfig.from_yaml(args.config))
    print(f"Best checkpoint: {checkpoint}")


def _evaluate(args) -> None:
    from .training import evaluate

    print(json.dumps(evaluate(args.checkpoint, args.manifest, args.split), indent=2))


def _predict(args) -> None:
    from .inference import ButterflyIdentifier

    identifier = ButterflyIdentifier(args.checkpoint, threshold=args.threshold)
    with Image.open(args.image) as handle:
        result = identifier.predict(handle, top_k=args.top_k)
    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="eurolepi", description="European butterfly identification toolkit"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    validate = commands.add_parser("validate", help="Validate a dataset manifest")
    validate.add_argument("manifest", type=Path)
    validate.add_argument("--skip-file-check", action="store_true")
    validate.add_argument("--before-split", action="store_true")
    validate.add_argument("--minimum-train-images", type=int, default=50)
    validate.set_defaults(func=_validate)

    split = commands.add_parser("split", help="Create specimen-safe stratified splits")
    split.add_argument("manifest", type=Path)
    split.add_argument("--output", type=Path, default=Path("data/manifest.csv"))
    split.add_argument("--seed", type=int, default=42)
    split.add_argument("--validation-fraction", type=float, default=0.15)
    split.add_argument("--test-fraction", type=float, default=0.15)
    split.set_defaults(func=_split)

    train = commands.add_parser("train", help="Train MaxViT-T from a YAML config")
    train.add_argument("config", type=Path)
    train.set_defaults(func=_train)

    evaluate = commands.add_parser("evaluate", help="Evaluate a trained checkpoint")
    evaluate.add_argument("checkpoint", type=Path)
    evaluate.add_argument("manifest", type=Path)
    evaluate.add_argument("--split", choices=["validation", "test"], default="test")
    evaluate.set_defaults(func=_evaluate)

    predict = commands.add_parser("predict", help="Identify one image")
    predict.add_argument("checkpoint", type=Path)
    predict.add_argument("image", type=Path)
    predict.add_argument("--top-k", type=int, default=5)
    predict.add_argument("--threshold", type=float)
    predict.set_defaults(func=_predict)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

