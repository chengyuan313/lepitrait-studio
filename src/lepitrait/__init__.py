"""LepiTrait Studio core package."""

from .pipeline import AnalysisPipeline, PipelineConfig
from .schema import SpecimenRecord

__all__ = ["AnalysisPipeline", "PipelineConfig", "SpecimenRecord"]
__version__ = "0.2.0"
