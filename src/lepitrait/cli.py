"""Small command-line entry point for reproducible batch scripts."""

from __future__ import annotations

import argparse
from pathlib import Path

from .image import load_rgb
from .pipeline import AnalysisPipeline
from .schema import ViewSide


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyse a standardized Lepidoptera specimen image.")
    parser.add_argument("image", type=Path)
    parser.add_argument("--specimen-id", required=True)
    parser.add_argument("--pixels-per-mm", type=float)
    parser.add_argument("--view-side", choices=[side.value for side in ViewSide], default="unknown")
    parser.add_argument("--colour-calibrated", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("specimen_record.json"))
    args = parser.parse_args()

    result = AnalysisPipeline().analyse(
        load_rgb(args.image),
        specimen_id=args.specimen_id,
        image_name=args.image.name,
        pixels_per_mm=args.pixels_per_mm,
        colour_calibrated=args.colour_calibrated,
        view_side=ViewSide(args.view_side),
    )
    args.output.write_text(result.record.model_dump_json(indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

