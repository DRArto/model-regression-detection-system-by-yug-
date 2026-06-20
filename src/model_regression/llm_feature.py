from __future__ import annotations

import asyncio
import json
import re
import time
from typing import Any
from openai import AsyncOpenAI

from .schemas import ClassificationOutput, PromptConfig


CLASSIFICATION_SCHEMA: dict[str, Any] = {
    "type": "json_schema",
    "name": "email_classification",
    "strict": True,
    "schema": ClassificationOutput.model_json_schema(),
}


def _usage_numbers(response: Any) -> tuple[int, int]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return 0, 0
    return int(getattr(usage, "input_tokens", 0) or 0), int(getattr(usage, "output_tokens", 0) or 0)


def _fake_classify(email: str) -> ClassificationOutput:
    text = email.lower()
    if any(word in text for word in ["invoice", "refund", "charged", "payment", "bill", "billing", "receipt", "paid"]):
        category = "billing"
    elif any(word in text for word in ["error", "bug", "crash", "not loading", "api", "technical", "freezes", "export button"]):
        category = "technical"
    elif any(word in text for word in ["password", "login", "account", "delete my account", "delete my workspace", "profile", "admin email", "sso", "access", "invited"]):
        category = "account"
    else:
        category = "general"

    clean = re.sub(r"\s+", " ", email).strip()
    summary = clean[:120] + ("..." if len(clean) > 120 else "")
    return ClassificationOutput(category=category, summary=summary)


async def classify_email(
    email: str,
    prompt: PromptConfig,
    model: str,
    client: AsyncOpenAI | None = None,
    fake: bool = False,
) -> tuple[ClassificationOutput, int, int, int]:
    start = time.perf_counter()

    if fake:
        await asyncio.sleep(0.01)
        output = _fake_classify(email)
        latency_ms = int((time.perf_counter() - start) * 1000)
        return output, latency_ms, 0, 0

    if client is None:
        client = AsyncOpenAI()

    messages: list[dict[str, str]] = [{"role": "system", "content": prompt.system_prompt}]

    for example in prompt.few_shot_examples:
        messages.append({"role": "user", "content": example["input"]})
        messages.append({"role": "assistant", "content": json.dumps(example["output"])})

    messages.append({"role": "user", "content": email})

    response = await client.responses.create(
        model=model,
        input=messages,
        text={"format": CLASSIFICATION_SCHEMA},
        temperature=0,
    )

    latency_ms = int((time.perf_counter() - start) * 1000)
    input_tokens, output_tokens = _usage_numbers(response)
    parsed = ClassificationOutput.model_validate_json(response.output_text)
    return parsed, latency_ms, input_tokens, output_tokens
