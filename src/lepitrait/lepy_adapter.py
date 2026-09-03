"""Stable adapter boundary for an upstream LEPY installation."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any


class LepyNotConfiguredError(RuntimeError):
    pass


class LepyAdapter:
    """Wrap a pinned LEPY runner without coupling the GUI to upstream internals."""

    def __init__(self, runner: Callable[[Path, Path], dict[str, Any]] | None = None, version: str | None = None):
        self.runner = runner
        self.version = version or "unconfigured"

    @property
    def available(self) -> bool:
        return self.runner is not None

    def analyse(self, image_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
        if self.runner is None:
            raise LepyNotConfiguredError("No LEPY runner has been configured.")
        return self.runner(Path(image_path), Path(output_dir))

