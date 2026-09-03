"""Versioned domain models shared by the GUI, pipelines and exports."""

from __future__ import annotations

from datetime import date, datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ViewSide(str, Enum):
    DORSAL = "dorsal"
    VENTRAL = "ventral"
    UNKNOWN = "unknown"


class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class CaptureMetadata(BaseModel):
    model_config = ConfigDict(extra="allow")

    image_name: str
    view_side: ViewSide = ViewSide.UNKNOWN
    camera_id: str | None = None
    lens_id: str | None = None
    imaging_batch: str | None = None
    pixels_per_mm: float | None = Field(default=None, gt=0)
    colour_calibrated: bool = False
    colour_profile: str | None = None
    captured_at: datetime | None = None


class SpecimenMetadata(BaseModel):
    catalog_number: str | None = None
    verbatim_scientific_name: str | None = None
    accepted_scientific_name: str | None = None
    event_date: date | None = None
    verbatim_event_date: str | None = None
    verbatim_locality: str | None = None
    decimal_latitude: float | None = Field(default=None, ge=-90, le=90)
    decimal_longitude: float | None = Field(default=None, ge=-180, le=180)
    recorded_by: str | None = None
    identified_by: str | None = None
    sex: str | None = None
    institution_code: str | None = None
    collection_code: str | None = None
    verbatim_label: str | None = None
    field_confidence: dict[str, float] = Field(default_factory=dict)


class TraitMeasurement(BaseModel):
    name: str
    value: float
    unit: str
    method: str
    region: str = "whole_specimen"
    confidence: float | None = Field(default=None, ge=0, le=1)


class ColourSummary(BaseModel):
    region: str = "whole_specimen"
    l_mean: float
    l_sd: float
    a_mean: float
    b_mean: float
    chroma_mean: float
    hue_degrees: float
    dark_fraction: float = Field(ge=0, le=1)
    calibrated: bool = False


class TaxonPrediction(BaseModel):
    scientific_name: str
    rank: str = "species"
    probability: float = Field(ge=0, le=1)
    model_name: str
    model_version: str


class QualityFlag(BaseModel):
    code: str
    severity: Severity
    message: str
    value: float | str | None = None


class Provenance(BaseModel):
    pipeline_version: str = "0.1.0"
    trait_engine: str = "baseline"
    trait_engine_version: str = "0.1.0"
    identification_model: str | None = None
    identification_model_version: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    parameters: dict[str, Any] = Field(default_factory=dict)


class SpecimenRecord(BaseModel):
    schema_version: str = "1.0.0"
    specimen_id: str
    capture: CaptureMetadata
    metadata: SpecimenMetadata = Field(default_factory=SpecimenMetadata)
    measurements: list[TraitMeasurement] = Field(default_factory=list)
    colour: list[ColourSummary] = Field(default_factory=list)
    predictions: list[TaxonPrediction] = Field(default_factory=list)
    quality_flags: list[QualityFlag] = Field(default_factory=list)
    provenance: Provenance = Field(default_factory=Provenance)
    reviewed: bool = False

