from __future__ import annotations

from pathlib import Path
import yaml
from .schemas import PromptConfig


def load_prompt(path: str | Path) -> PromptConfig:
    path = Path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))

    required = ["version_id", "timestamp", "system_prompt"]
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f"Prompt file {path} is missing: {', '.join(missing)}")

    return PromptConfig(
        version_id=str(data["version_id"]),
        timestamp=str(data["timestamp"]),
        system_prompt=str(data["system_prompt"]),
        few_shot_examples=list(data.get("few_shot_examples", [])),
    )
