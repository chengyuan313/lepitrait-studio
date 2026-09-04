"""Safe subprocess integration with an independently installed LEPY engine.

LEPY deliberately exposes a command-line entry point and its ``main.py`` refuses
to be imported.  This module therefore stages controlled inputs, creates a
job-specific configuration, executes LEPY without a shell, and normalises its
tab-separated output for the rest of EuroLepi Studio.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Iterable, Sequence
import zipfile

import pandas as pd
from PIL import Image
import yaml


TRAIT_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
FIELD_METADATA_COLUMNS = (
    "specimen_id",
    "scientific_name",
    "site_id",
    "country",
    "latitude",
    "longitude",
    "collection_date",
    "temperature_c",
)

NORMALISED_TRAITS = {
    "contour_width_calibrated": "wing_span_mm",
    "contour_height_calibrated": "body_bbox_length_mm",
    "contour_area_calibrated": "specimen_area_mm2",
    "poi_dist_inner_outer_l": "left_forewing_length_mm",
    "poi_dist_inner_outer_r": "right_forewing_length_mm",
    "poi_dist_inner": "body_width_mm",
    "poi_dist_body": "body_length_mm",
    "poi_area_body": "body_area_mm2",
    "poi_area_wing_l": "left_wing_area_mm2",
    "poi_area_wing_r": "right_wing_area_mm2",
}


class LepyConfigurationError(ValueError):
    """Raised when the configured external LEPY installation is not usable."""


class LepyExecutionError(RuntimeError):
    """Raised when the external LEPY process cannot produce a result."""


@dataclass(frozen=True)
class TraitSample:
    """One biological specimen with a required RGB and optional UV image."""

    specimen_id: str
    rgb_name: str
    rgb_bytes: bytes
    uv_name: str | None = None
    uv_bytes: bytes | None = None


@dataclass(frozen=True)
class LepySettings:
    """Location and execution limits for an external LEPY installation."""

    home: Path
    python_executable: str
    config_path: Path
    n_jobs: int = 1
    timeout_seconds: int = 1800


@dataclass(frozen=True)
class LepyValidation:
    valid: bool
    errors: tuple[str, ...]
    engine_fingerprint: str = ""


@dataclass(frozen=True)
class MetadataInspection:
    valid: bool
    table: pd.DataFrame | None
    errors: tuple[str, ...]
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class LepyRunResult:
    table: pd.DataFrame
    archive_bytes: bytes
    visualisations: dict[str, bytes]
    stdout: str
    stderr: str
    engine_fingerprint: str
    config_hash: str


def _check_image(name: str, payload: bytes) -> None:
    if Path(name).suffix.lower() not in TRAIT_IMAGE_SUFFIXES:
        raise ValueError(f"Unsupported trait image type: {name}")
    try:
        with Image.open(BytesIO(payload)) as image:
            image.verify()
    except (OSError, ValueError) as exc:
        raise ValueError(f"Unreadable image {name}: {exc}") from exc


def pair_trait_uploads(files: Sequence[tuple[str, bytes]]) -> list[TraitSample]:
    """Pair batch files named ``ID_rgb.ext`` and optional ``ID_uv.ext``.

    A filename without a modality suffix is treated as RGB. Directory segments
    are intentionally ignored; specimen identifiers must be unique in a run.
    """

    paired: dict[str, dict[str, tuple[str, bytes]]] = {}
    display_ids: dict[str, str] = {}
    for original_name, payload in files:
        name = Path(original_name).name
        _check_image(name, payload)
        stem = Path(name).stem
        lowered = stem.lower()
        if lowered.endswith("_rgb"):
            specimen_id, modality = stem[:-4], "rgb"
        elif lowered.endswith("_uv"):
            specimen_id, modality = stem[:-3], "uv"
        else:
            specimen_id, modality = stem, "rgb"
        specimen_id = specimen_id.strip()
        if not specimen_id:
            raise ValueError(f"Filename does not contain a specimen ID: {name}")
        key = specimen_id.casefold()
        display_ids.setdefault(key, specimen_id)
        bucket = paired.setdefault(key, {})
        if modality in bucket:
            raise ValueError(
                f"Duplicate {modality.upper()} image for specimen {display_ids[key]}: "
                f"{bucket[modality][0]} and {name}"
            )
        bucket[modality] = (original_name, payload)

    samples: list[TraitSample] = []
    for key in sorted(paired):
        bucket = paired[key]
        specimen_id = display_ids[key]
        if "rgb" not in bucket:
            raise ValueError(f"UV image has no matching RGB image for specimen {specimen_id}.")
        rgb_name, rgb_bytes = bucket["rgb"]
        uv_name, uv_bytes = bucket.get("uv", (None, None))
        samples.append(TraitSample(specimen_id, rgb_name, rgb_bytes, uv_name, uv_bytes))
    if not samples:
        raise ValueError("No supported RGB images were supplied.")
    return samples


def inspect_field_metadata(
    csv_bytes: bytes | None, expected_specimen_ids: Iterable[str]
) -> MetadataInspection:
    """Validate an optional field metadata table used to enrich LEPY rows."""

    if not csv_bytes:
        return MetadataInspection(True, None, (), ())
    try:
        table = pd.read_csv(BytesIO(csv_bytes), dtype={"specimen_id": "string"})
    except (UnicodeDecodeError, pd.errors.ParserError, ValueError) as exc:
        return MetadataInspection(False, None, (f"Cannot read metadata CSV: {exc}",), ())

    errors: list[str] = []
    warnings: list[str] = []
    if "specimen_id" not in table.columns:
        errors.append("Metadata CSV must contain a specimen_id column.")
        return MetadataInspection(False, None, tuple(errors), ())
    unknown = sorted(set(table.columns) - set(FIELD_METADATA_COLUMNS))
    if unknown:
        errors.append("Unsupported metadata columns: " + ", ".join(unknown))
    table["specimen_id"] = table["specimen_id"].fillna("").str.strip()
    if (table["specimen_id"] == "").any():
        errors.append("Metadata specimen_id values cannot be blank.")
    duplicates = sorted(table.loc[table["specimen_id"].duplicated(), "specimen_id"].unique())
    if duplicates:
        errors.append("Metadata specimen_id values must be unique: " + ", ".join(duplicates[:10]))

    for coordinate, low, high in (("latitude", -90, 90), ("longitude", -180, 180)):
        if coordinate in table.columns:
            numeric = pd.to_numeric(table[coordinate], errors="coerce")
            invalid = table[coordinate].notna() & numeric.isna()
            invalid |= numeric.notna() & ~numeric.between(low, high)
            if invalid.any():
                errors.append(f"{coordinate} contains non-numeric or out-of-range values.")
            table[coordinate] = numeric
    if "temperature_c" in table.columns:
        numeric = pd.to_numeric(table["temperature_c"], errors="coerce")
        if (table["temperature_c"].notna() & numeric.isna()).any():
            errors.append("temperature_c must be numeric when provided.")
        table["temperature_c"] = numeric
    if "collection_date" in table.columns:
        parsed = pd.to_datetime(table["collection_date"], errors="coerce")
        if (table["collection_date"].notna() & parsed.isna()).any():
            errors.append("collection_date must use an ISO date such as 2027-06-14.")

    expected = {str(item).strip().casefold() for item in expected_specimen_ids}
    supplied = {str(item).strip().casefold() for item in table["specimen_id"]}
    missing = expected - supplied
    extra = supplied - expected
    if missing:
        warnings.append(f"Metadata is missing for {len(missing)} uploaded specimen(s).")
    if extra:
        warnings.append(f"Metadata contains {len(extra)} specimen(s) with no uploaded RGB image.")
    return MetadataInspection(not errors, table, tuple(errors), tuple(warnings))


def _resolve_executable(value: str) -> str | None:
    expanded = str(Path(value).expanduser())
    path = Path(expanded)
    if path.is_file():
        return str(path.resolve())
    return shutil.which(value)


class LepyAdapter:
    """Execute and normalise a pinned, external LEPY checkout."""

    def __init__(self, settings: LepySettings):
        self.settings = settings

    def validate(self) -> LepyValidation:
        errors: list[str] = []
        main_path = self.settings.home.expanduser() / "main.py"
        config_path = self.settings.config_path.expanduser()
        if not main_path.is_file():
            errors.append(f"LEPY main.py was not found in {self.settings.home}.")
        if not config_path.is_file():
            errors.append(f"LEPY configuration was not found: {config_path}")
        if _resolve_executable(self.settings.python_executable) is None:
            errors.append(
                f"LEPY Python executable was not found: {self.settings.python_executable}"
            )
        if self.settings.n_jobs == 0 or self.settings.n_jobs < -1:
            errors.append("LEPY n_jobs must be -1 or a positive integer.")
        if self.settings.timeout_seconds < 1:
            errors.append("LEPY timeout must be a positive number of seconds.")

        fingerprint = ""
        if main_path.is_file():
            fingerprint = sha256(main_path.read_bytes()).hexdigest()[:12]
        return LepyValidation(not errors, tuple(errors), fingerprint)

    def _job_config(self, destination: Path) -> tuple[Path, str]:
        source_config = self.settings.config_path.expanduser()
        lepy_home = self.settings.home.expanduser()
        try:
            config = yaml.safe_load(source_config.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
            raise LepyConfigurationError(f"Cannot read LEPY configuration: {exc}") from exc
        if not isinstance(config, dict):
            raise LepyConfigurationError("LEPY configuration must be a YAML mapping.")

        reading = config.setdefault("reading", {})
        if not isinstance(reading, dict):
            raise LepyConfigurationError("LEPY reading configuration must be a mapping.")
        reading.update(
            {
                "rgb_regex": r"(.+)_rgb\.(?:jpg|jpeg|png|tif|tiff)$",
                "uv_regex": r"(.+)_uv\.(?:jpg|jpeg|png|tif|tiff)$",
                "extensions": sorted(TRAIT_IMAGE_SUFFIXES),
                "ordered": False,
            }
        )
        execution = config.setdefault("execution", {})
        if not isinstance(execution, dict):
            raise LepyConfigurationError("LEPY execution configuration must be a mapping.")
        execution.update({"proceed": True, "force": True})

        calibration = config.get("calibration", {})
        if isinstance(calibration, dict) and calibration.get("enabled"):
            template = calibration.get("template_path")
            if not template:
                raise LepyConfigurationError(
                    "Calibration is enabled but calibration.template_path is missing."
                )
            template_path = Path(str(template)).expanduser()
            if not template_path.is_absolute():
                beside_config = source_config.parent / template_path
                beside_home = lepy_home / template_path
                template_path = beside_config if beside_config.is_file() else beside_home
            if not template_path.is_file():
                raise LepyConfigurationError(f"Scale-bar template was not found: {template_path}")
            calibration["template_path"] = str(template_path.resolve())

        rendered = yaml.safe_dump(config, sort_keys=False, allow_unicode=True)
        config_hash = sha256(rendered.encode("utf-8")).hexdigest()
        destination.write_text(rendered, encoding="utf-8")
        return destination, config_hash

    def run(
        self,
        samples: Sequence[TraitSample],
        metadata: pd.DataFrame | None = None,
    ) -> LepyRunResult:
        validation = self.validate()
        if not validation.valid:
            raise LepyConfigurationError(" ".join(validation.errors))
        if not samples:
            raise ValueError("At least one trait sample is required.")

        specimen_keys = [sample.specimen_id.casefold() for sample in samples]
        if len(specimen_keys) != len(set(specimen_keys)):
            raise ValueError("specimen_id values must be unique within a LEPY run.")
        for sample in samples:
            _check_image(sample.rgb_name, sample.rgb_bytes)
            if sample.uv_bytes is not None:
                _check_image(sample.uv_name or "specimen_uv.tif", sample.uv_bytes)

        with tempfile.TemporaryDirectory(prefix="eurolepi-lepy-") as temporary:
            root = Path(temporary)
            inputs = root / "input"
            outputs = root / "output"
            inputs.mkdir()
            outputs.mkdir()
            mapping: dict[str, TraitSample] = {}
            for index, sample in enumerate(samples, start=1):
                token = f"specimen_{index:06d}"
                rgb_suffix = Path(sample.rgb_name).suffix.lower()
                (inputs / f"{token}_rgb{rgb_suffix}").write_bytes(sample.rgb_bytes)
                if sample.uv_bytes is not None:
                    uv_suffix = Path(sample.uv_name or "uv.tif").suffix.lower()
                    (inputs / f"{token}_uv{uv_suffix}").write_bytes(sample.uv_bytes)
                mapping[f"{token}_rgb"] = sample

            config_path, config_hash = self._job_config(root / "bridge_config.yml")
            executable = _resolve_executable(self.settings.python_executable)
            assert executable is not None
            command = [
                executable,
                str((self.settings.home.expanduser() / "main.py").resolve()),
                str(inputs.resolve()),
                str(config_path.resolve()),
                "--output",
                str(outputs.resolve()),
                "--yes",
                "--force",
                "--n_jobs",
                str(self.settings.n_jobs),
            ]
            try:
                completed = subprocess.run(
                    command,
                    cwd=self.settings.home.expanduser(),
                    capture_output=True,
                    text=True,
                    errors="replace",
                    timeout=self.settings.timeout_seconds,
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                raise LepyExecutionError(
                    f"LEPY exceeded the {self.settings.timeout_seconds}-second limit."
                ) from exc
            except OSError as exc:
                raise LepyExecutionError(f"LEPY could not be started: {exc}") from exc
            if completed.returncode != 0:
                detail = (completed.stderr or completed.stdout).strip()[-2000:]
                raise LepyExecutionError(
                    f"LEPY exited with code {completed.returncode}. {detail}"
                )

            stats_path = outputs / "stats.csv"
            if not stats_path.is_file():
                raise LepyExecutionError("LEPY finished without creating output/stats.csv.")
            try:
                raw = pd.read_csv(stats_path, sep="\t")
            except (pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
                raise LepyExecutionError(f"Cannot read LEPY stats.csv: {exc}") from exc

            errors = self._read_failures(outputs / "errors.log")
            rows: list[dict[str, object]] = []
            raw_by_code = {
                str(row.get("Code", "")): row for row in raw.to_dict(orient="records")
            }
            processed_at = datetime.now(timezone.utc).isoformat()
            for code, sample in mapping.items():
                raw_row = raw_by_code.get(code, {})
                error = errors.get(f"{code}{Path(sample.rgb_name).suffix.lower()}", "")
                if not raw_row and not error:
                    error = "LEPY did not return a statistics row for this image."
                row: dict[str, object] = {
                    "specimen_id": sample.specimen_id,
                    "source_image": sample.rgb_name,
                    "uv_image": sample.uv_name or "",
                    "status": "error" if error else "completed",
                    "error": error,
                    "processed_at_utc": processed_at,
                    "lepy_engine_fingerprint": validation.engine_fingerprint,
                    "lepy_config_sha256": config_hash,
                }
                for source, normalised in NORMALISED_TRAITS.items():
                    row[normalised] = raw_row.get(source, "")
                for column, value in raw_row.items():
                    row["lepy_code" if column == "Code" else column] = value
                rows.append(row)
            table = pd.DataFrame(rows)
            table = self._merge_metadata(table, metadata)

            result_csv = table.to_csv(index=False).encode("utf-8-sig")
            (outputs / "trait_results.csv").write_bytes(result_csv)
            manifest = {
                "created_at_utc": processed_at,
                "engine_fingerprint": validation.engine_fingerprint,
                "config_sha256": config_hash,
                "specimens": len(samples),
                "completed": int((table["status"] == "completed").sum()),
                "failed": int((table["status"] == "error").sum()),
            }
            (outputs / "run_manifest.json").write_text(
                json.dumps(manifest, indent=2), encoding="utf-8"
            )

            visualisations: dict[str, bytes] = {}
            for code, sample in mapping.items():
                preview = outputs / "visualisations" / f"{code}.png"
                if preview.is_file():
                    visualisations[sample.specimen_id] = preview.read_bytes()
            archive = self._zip_output(outputs)
            return LepyRunResult(
                table=table,
                archive_bytes=archive,
                visualisations=visualisations,
                stdout=completed.stdout,
                stderr=completed.stderr,
                engine_fingerprint=validation.engine_fingerprint,
                config_hash=config_hash,
            )

    @staticmethod
    def _read_failures(path: Path) -> dict[str, str]:
        if not path.is_file():
            return {}
        failures: dict[str, str] = {}
        pattern = re.compile(r'Failed to process "([^"]+)"\. Reason \([^)]+\): (.*)$')
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            match = pattern.search(line)
            if match:
                failures[match.group(1)] = match.group(2).strip()
        return failures

    @staticmethod
    def _merge_metadata(table: pd.DataFrame, metadata: pd.DataFrame | None) -> pd.DataFrame:
        if metadata is None:
            return table
        metadata = metadata.copy()
        metadata["_join_id"] = metadata["specimen_id"].astype(str).str.casefold()
        metadata = metadata.drop(columns=["specimen_id"])
        table["_join_id"] = table["specimen_id"].astype(str).str.casefold()
        table = table.merge(metadata, on="_join_id", how="left", validate="one_to_one")
        return table.drop(columns=["_join_id"])

    @staticmethod
    def _zip_output(output: Path) -> bytes:
        buffer = BytesIO()
        with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(output.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(output).as_posix())
        return buffer.getvalue()
