"""Application-level orchestration for one standardized specimen image."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .identification import SpeciesIdentifier, UnconfiguredIdentifier
from .qc import image_quality_flags
from .schema import CaptureMetadata, Provenance, SpecimenMetadata, SpecimenRecord, ViewSide
from .segmentation import SegmentationResult, segment_specimen
from .traits import colour_summary, morphology_measurements


@dataclass(frozen=True)
class PipelineConfig:
    background_threshold: int = 238
    min_saturation: int = 10
    trait_engine: str = "baseline"


@dataclass(frozen=True)
class AnalysisResult:
    record: SpecimenRecord
    segmentation: SegmentationResult


class AnalysisPipeline:
    def __init__(self, config: PipelineConfig | None = None, identifier: SpeciesIdentifier | None = None):
        self.config = config or PipelineConfig()
        self.identifier = identifier or UnconfiguredIdentifier()

    def analyse(
        self,
        rgb: np.ndarray,
        specimen_id: str,
        image_name: str,
        pixels_per_mm: float | None = None,
        colour_calibrated: bool = False,
        view_side: ViewSide = ViewSide.UNKNOWN,
        metadata: SpecimenMetadata | None = None,
    ) -> AnalysisResult:
        segmentation = segment_specimen(
            rgb,
            background_threshold=self.config.background_threshold,
            min_saturation=self.config.min_saturation,
        )
        measurements = morphology_measurements(segmentation.mask, pixels_per_mm)
        colour = colour_summary(rgb, segmentation.mask, colour_calibrated)
        predictions = self.identifier.predict(rgb, segmentation.mask, top_k=5)
        flags = image_quality_flags(rgb, segmentation.mask)
        if not colour_calibrated:
            from .schema import QualityFlag, Severity

            flags.append(QualityFlag(code="COLOUR_UNCALIBRATED", severity=Severity.WARNING, message="Colour values are relative and must not be compared across imaging batches."))
        if pixels_per_mm is None:
            from .schema import QualityFlag, Severity

            flags.append(QualityFlag(code="SCALE_MISSING", severity=Severity.WARNING, message="Millimetre measurements were not produced because scale is missing."))

        record = SpecimenRecord(
            specimen_id=specimen_id,
            capture=CaptureMetadata(
                image_name=image_name,
                view_side=view_side,
                pixels_per_mm=pixels_per_mm,
                colour_calibrated=colour_calibrated,
            ),
            metadata=metadata or SpecimenMetadata(),
            measurements=measurements,
            colour=[colour] if colour else [],
            predictions=predictions,
            quality_flags=flags,
            provenance=Provenance(
                trait_engine=self.config.trait_engine,
                identification_model=None if self.identifier.name == "not-configured" else self.identifier.name,
                identification_model_version=None if self.identifier.name == "not-configured" else self.identifier.version,
                parameters={
                    "background_threshold": self.config.background_threshold,
                    "min_saturation": self.config.min_saturation,
                },
            ),
        )
        return AnalysisResult(record=record, segmentation=segmentation)

