"""Command-line access to the same services used by the GUI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from eurolepi.batch import identify_batch
from eurolepi.dataset_package import inspect_dataset_zip
from eurolepi.lepy_adapter import (
    LepyAdapter,
    LepySettings,
    TraitSample,
    inspect_field_metadata,
    pair_trait_uploads,
)
from eurolepi.prediction import ButterflyIdentifier, open_image
from eurolepi.registry import ModelRegistry
from eurolepi.training import TrainingOptions, train_from_zip


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
TRAIT_SUFFIXES = IMAGE_SUFFIXES | {".tif", ".tiff"}


def _add_lepy_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--lepy-home", type=Path, required=True)
    parser.add_argument("--lepy-python", default=sys.executable)
    parser.add_argument("--lepy-config", type=Path)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=int, default=1800)


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

    trait_single = commands.add_parser("traits-single", help="Measure one image with LEPY")
    trait_single.add_argument("image", type=Path)
    trait_single.add_argument("--specimen-id", required=True)
    trait_single.add_argument("--uv", type=Path)
    trait_single.add_argument("--output", type=Path, default=Path("trait_results.csv"))
    trait_single.add_argument("--artifacts", type=Path, default=Path("lepy_outputs.zip"))
    _add_lepy_arguments(trait_single)

    trait_batch = commands.add_parser("traits-batch", help="Measure an image folder with LEPY")
    trait_batch.add_argument("folder", type=Path)
    trait_batch.add_argument("--metadata", type=Path)
    trait_batch.add_argument("--output", type=Path, default=Path("trait_results.csv"))
    trait_batch.add_argument("--artifacts", type=Path, default=Path("lepy_outputs.zip"))
    _add_lepy_arguments(trait_batch)
    return parser


def _lepy_adapter(args) -> LepyAdapter:
    return LepyAdapter(
        LepySettings(
            home=args.lepy_home,
            python_executable=args.lepy_python,
            config_path=args.lepy_config or args.lepy_home / "config.yml",
            n_jobs=args.n_jobs,
            timeout_seconds=args.timeout_seconds,
        )
    )


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

    if args.command == "traits-single":
        sample = TraitSample(
            specimen_id=args.specimen_id,
            rgb_name=args.image.name,
            rgb_bytes=args.image.read_bytes(),
            uv_name=args.uv.name if args.uv else None,
            uv_bytes=args.uv.read_bytes() if args.uv else None,
        )
        result = _lepy_adapter(args).run([sample])
        result.table.to_csv(args.output, index=False)
        args.artifacts.write_bytes(result.archive_bytes)
        print(args.output)
        return 0

    if args.command == "traits-batch":
        files = [
            (path.relative_to(args.folder).as_posix(), path.read_bytes())
            for path in sorted(args.folder.rglob("*"))
            if path.is_file() and path.suffix.lower() in TRAIT_SUFFIXES
        ]
        samples = pair_trait_uploads(files)
        inspection = inspect_field_metadata(
            args.metadata.read_bytes() if args.metadata else None,
            [sample.specimen_id for sample in samples],
        )
        if not inspection.valid:
            raise ValueError(" ".join(inspection.errors))
        result = _lepy_adapter(args).run(samples, inspection.table)
        result.table.to_csv(args.output, index=False)
        args.artifacts.write_bytes(result.archive_bytes)
        print(args.output)
        return 0

    registry = ModelRegistry(args.workspace / "models")
    record = registry.get(args.model_id)
    identifier = ButterflyIdentifier(record)
    if args.command == "predict":
        result = identifier.predict(open_image(args.image), top_k=5)
        print(json.dumps([
            {
                "rank": item.rank,
                "scientific_name": item.scientific_name,
                "probability": item.probability,
            }
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
