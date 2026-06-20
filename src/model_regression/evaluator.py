from __future__ import annotations

import asyncio
import statistics
import uuid
from datetime import datetime, timezone
from pathlib import Path
from openai import AsyncOpenAI

from .schemas import EvalCaseResult, EvalSummary, GoldenCase, PromptConfig
from .llm_feature import classify_email
from .judge import judge_summary
from .storage import get_previous_run, get_recent_accuracies


async def _evaluate_one_case(case: GoldenCase, prompt: PromptConfig, model: str, judge_model: str, client: AsyncOpenAI | None, semaphore: asyncio.Semaphore, fake: bool) -> EvalCaseResult:
    async with semaphore:
        try:
            output, latency_ms, input_tokens, output_tokens = await classify_email(case.input, prompt, model, client, fake)
            judge = await judge_summary(case.input, case.expected_summary, output.summary, judge_model, client, fake)
            category_match = output.category == case.expected_category
            passed = category_match and judge.score >= 4
            return EvalCaseResult(
                case_id=case.id,
                expected_category=case.expected_category,
                actual_category=output.category,
                expected_summary=case.expected_summary,
                actual_summary=output.summary,
                category_match=category_match,
                summary_score=judge.score,
                summary_rationale=judge.rationale,
                passed=passed,
                latency_ms=latency_ms,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                difficulty=case.expected_difficulty,
                notes=case.notes,
            )
        except Exception as exc:
            return EvalCaseResult(
                case_id=case.id,
                expected_category=case.expected_category,
                actual_category="general",
                expected_summary=case.expected_summary,
                actual_summary="",
                category_match=False,
                summary_score=1,
                summary_rationale="Call or parser failed.",
                passed=False,
                latency_ms=0,
                input_tokens=0,
                output_tokens=0,
                difficulty=case.expected_difficulty,
                notes=case.notes,
                error=str(exc),
            )


async def run_evaluation(prompt: PromptConfig, cases: list[GoldenCase], model: str, judge_model: str, db_path: str | Path, max_concurrency: int = 5, warning_drop: float = 0.03, critical_drop: float = 0.08, drift_avg_min: float = 0.85, fake: bool = False) -> tuple[EvalSummary, list[EvalCaseResult], EvalSummary | None, dict[str, EvalCaseResult]]:
    previous = get_previous_run(db_path)
    previous_summary = previous[0] if previous else None
    previous_cases = previous[1] if previous else {}
    client = None if fake else AsyncOpenAI()
    semaphore = asyncio.Semaphore(max_concurrency)

    results = await asyncio.gather(*[_evaluate_one_case(case, prompt, model, judge_model, client, semaphore, fake) for case in cases])
    total = len(results)
    accuracy = (sum(1 for r in results if r.passed) / total) if total else 0.0
    avg_summary_score = statistics.mean([r.summary_score for r in results]) if results else 0.0
    avg_latency_ms = statistics.mean([r.latency_ms for r in results]) if results else 0.0

    regressions = []
    improvements = []
    for result in results:
        old = previous_cases.get(result.case_id)
        if old and old.passed and not result.passed:
            regressions.append(result.case_id)
        if old and (not old.passed) and result.passed:
            improvements.append(result.case_id)

    accuracy_delta = None if previous_summary is None else accuracy - previous_summary.accuracy
    recent_accuracies = get_recent_accuracies(db_path, limit=6)
    rolling_values = [accuracy] + recent_accuracies
    drift_warning = len(rolling_values) >= 3 and (sum(rolling_values) / len(rolling_values)) < drift_avg_min

    status = "pass"
    if accuracy_delta is not None and accuracy_delta <= -critical_drop:
        status = "critical"
    elif accuracy_delta is not None and accuracy_delta <= -warning_drop:
        status = "warn"
    if drift_warning and status == "pass":
        status = "warn"

    summary = EvalSummary(
        run_id=str(uuid.uuid4())[:8],
        timestamp=datetime.now(timezone.utc).isoformat(),
        prompt_version=prompt.version_id,
        model=model,
        total_cases=total,
        accuracy=accuracy,
        avg_summary_score=avg_summary_score,
        avg_latency_ms=avg_latency_ms,
        input_tokens=sum(r.input_tokens for r in results),
        output_tokens=sum(r.output_tokens for r in results),
        status=status,
        accuracy_delta=accuracy_delta,
        regressions=regressions,
        improvements=improvements,
        drift_warning=drift_warning,
    )
    return summary, results, previous_summary, previous_cases
