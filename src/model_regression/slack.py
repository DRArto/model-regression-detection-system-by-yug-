from __future__ import annotations

import requests
from .schemas import EvalSummary
from .reporting import pct, signed_pct


def send_slack_alert(webhook_url: str | None, summary: EvalSummary, report_path: str | None = None) -> None:
    if not webhook_url:
        return

    emoji = {"pass": "✅", "warn": "⚠️", "critical": "🚨"}.get(summary.status, "ℹ️")
    text = (
        f"{emoji} *LLM Regression Eval: {summary.status.upper()}*\n"
        f"Prompt: `{summary.prompt_version}` | Model: `{summary.model}`\n"
        f"Accuracy: *{pct(summary.accuracy)}* ({signed_pct(summary.accuracy_delta)})\n"
        f"Regressions: *{len(summary.regressions)}* | Improvements: *{len(summary.improvements)}*\n"
        f"Avg summary score: *{summary.avg_summary_score:.2f}/5*\n"
    )
    if summary.regressions:
        text += f"Regressed case IDs: `{', '.join(summary.regressions)}`\n"
    if report_path:
        text += f"Report path: `{report_path}`"

    response = requests.post(webhook_url, json={"text": text}, timeout=10)
    response.raise_for_status()
