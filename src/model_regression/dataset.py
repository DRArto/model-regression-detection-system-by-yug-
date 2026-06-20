from __future__ import annotations

from pathlib import Path
import json
from .schemas import GoldenCase


def load_dataset(path: str | Path) -> list[GoldenCase]:
    path = Path(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    cases = raw["cases"] if isinstance(raw, dict) and "cases" in raw else raw
    return [GoldenCase.model_validate(item) for item in cases]
