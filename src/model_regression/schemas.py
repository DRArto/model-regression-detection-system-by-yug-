from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal
from pydantic import BaseModel, Field, ConfigDict

Category = Literal["billing", "technical", "account", "general"]
Difficulty = Literal["easy", "medium", "hard"]


class ClassificationOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    category: Category
    summary: str


class JudgeOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    score: int = Field(ge=1, le=5)
    rationale: str


class GoldenCase(BaseModel):
    id: str
    input: str
    expected_category: Category
    expected_summary: str
    expected_difficulty: Difficulty
    notes: str = ""


class EvalCaseResult(BaseModel):
    case_id: str
    expected_category: Category
    actual_category: Category
    expected_summary: str
    actual_summary: str
    category_match: bool
    summary_score: int
    summary_rationale: str
    passed: bool
    latency_ms: int
    input_tokens: int = 0
    output_tokens: int = 0
    difficulty: Difficulty
    notes: str = ""
    error: str | None = None


class EvalSummary(BaseModel):
    run_id: str
    timestamp: str
    prompt_version: str
    model: str
    total_cases: int
    accuracy: float
    avg_summary_score: float
    avg_latency_ms: float
    input_tokens: int
    output_tokens: int
    status: Literal["pass", "warn", "critical"]
    accuracy_delta: float | None = None
    regressions: list[str] = Field(default_factory=list)
    improvements: list[str] = Field(default_factory=list)
    drift_warning: bool = False


@dataclass(frozen=True)
class PromptConfig:
    version_id: str
    timestamp: str
    system_prompt: str
    few_shot_examples: list[dict[str, Any]]
