from __future__ import annotations

from typing import Any
from openai import AsyncOpenAI

from .schemas import JudgeOutput


JUDGE_SCHEMA: dict[str, Any] = {
    "type": "json_schema",
    "name": "summary_judge",
    "strict": True,
    "schema": JudgeOutput.model_json_schema(),
}


def _fake_summary_score(expected: str, actual: str) -> JudgeOutput:
    if actual.strip():
        return JudgeOutput(score=4, rationale="Fake mode: non-empty summary accepted for local pipeline testing.")
    return JudgeOutput(score=1, rationale="Fake mode: empty summary failed.")


async def judge_summary(
    email: str,
    expected_summary: str,
    actual_summary: str,
    model: str,
    client: AsyncOpenAI | None = None,
    fake: bool = False,
) -> JudgeOutput:
    if fake:
        return _fake_summary_score(expected_summary, actual_summary)

    if client is None:
        client = AsyncOpenAI()

    prompt = f"""
You are judging a customer-support email summary.

Score the ACTUAL SUMMARY from 1 to 5.
5 = captures the same core meaning as the expected summary.
3 = partially correct but missing important details.
1 = wrong or misleading.

EMAIL:
{email}

EXPECTED SUMMARY:
{expected_summary}

ACTUAL SUMMARY:
{actual_summary}

Return only JSON that matches the schema.
""".strip()

    response = await client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": "You are a strict but fair evaluation judge."},
            {"role": "user", "content": prompt},
        ],
        text={"format": JUDGE_SCHEMA},
        temperature=0,
    )
    return JudgeOutput.model_validate_json(response.output_text)
