"""Flat exports for downstream R workflows."""

from __future__ import annotations

from typing import Any

from .schema import SpecimenRecord


def flatten_record(record: SpecimenRecord) -> dict[str, Any]:
    row: dict[str, Any] = {
        "schema_version": record.schema_version,
        "specimen_id": record.specimen_id,
        **{f"capture_{key}": value for key, value in record.capture.model_dump(mode="json").items()},
        **{f"metadata_{key}": value for key, value in record.metadata.model_dump(mode="json").items() if key != "field_confidence"},
        "reviewed": record.reviewed,
        "qc_codes": ";".join(flag.code for flag in record.quality_flags),
    }
    for measurement in record.measurements:
        row[f"trait_{measurement.name}_{measurement.unit}"] = measurement.value
    if record.colour:
        for key, value in record.colour[0].model_dump(mode="json").items():
            row[f"colour_{key}"] = value
    for index, prediction in enumerate(record.predictions, start=1):
        row[f"prediction_{index}_name"] = prediction.scientific_name
        row[f"prediction_{index}_probability"] = prediction.probability
    return row

