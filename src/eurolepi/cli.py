"""Command-line access to the same services used by the GUI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eurolepi.batch import identify_batch
from eurolepi.dataset_package import inspect_dataset_zip
from eurolepi.prediction import ButterflyIdentifier, open_image
from eurolepi.registry import ModelRegistry
from eurolepi.training import TrainingOptions, train_from_zip


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="eurolepi")
    parser.add_argument("--workspace", type=Path, default=Path("workspace"))
    commands = parser.add_subparsers(dest="command", required=True)

    validate = commands.add_parser("validate", help="Validate a training dataset ZIP")
    validate.add_argument("dataset_zip", type=Path)

    train = commands.add_parser("train", help="Train a model from a dataset ZIP")
    train.add_argument("dataset_zip", type=Path)
    train.add_argument("--dataset-name", required=True)
    train.add_argument("--epochs", type=int, default=20)
    train.add_argument("--batch-size", type=int, default=16)

    predict = commands.add_parser("predict", help="Identify one image")
    predict.add_argument("model_id")
    predict.add_argument("image", type=Path)

    batch = commands.add_parser("batch", help="Identify an image folder")
    batch.add_argument("model_id")
    batch.add_argument("folder", type=Path)
    batch.add_argument("--output", type=Path, default=Path("identification_results.csv"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "validate":
        report = inspect_dataset_zip(args.dataset_zip.read_bytes())
        print(json.dumps({
            "valid": report.valid,
            "images": report.image_count,
            "specimens": report.specimen_count,
            "species": report.species_count,
            "errors": report.errors,
            "warnings": report.warnings,
        }, indent=2))
        return 0 if report.valid else 1

    if args.command == "train":
        record = train_from_zip(
            args.dataset_zip.read_bytes(),
            args.dataset_name,
            args.workspace,
            TrainingOptions(epochs=args.epochs, batch_size=args.batch_size),
        )
        print(record.model_id)
        return 0

    registry = ModelRegistry(args.workspace / "models")
    record = registry.get(args.model_id)
    identifier = ButterflyIdentifier(record)
    if args.command == "predict":
        result = identifier.predict(open_image(args.image), top_k=5)
        print(json.dumps([
            {"rank": item.rank, "scientific_name": item.scientific_name, "probability": item.probability}
            for item in result.candidates
        ], indent=2))
        return 0

    files = [
        (path.relative_to(args.folder).as_posix(), path.read_bytes())
        for path in sorted(args.folder.rglob("*"))
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    identify_batch(files, identifier).to_csv(args.output, index=False)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

