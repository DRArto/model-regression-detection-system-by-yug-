from __future__ import annotations

from html import escape
from pathlib import Path
from .schemas import EvalCaseResult, EvalSummary


def pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.1f}%"


def signed_pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    sign = "+" if value >= 0 else ""
    return f"{sign}{value * 100:.1f}%"


def generate_markdown_summary(summary: EvalSummary, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = f"""## LLM Regression Eval: {summary.status.upper()}

| Metric | Value |
|---|---:|
| Prompt version | `{summary.prompt_version}` |
| Model | `{summary.model}` |
| Total cases | {summary.total_cases} |
| Accuracy | {pct(summary.accuracy)} |
| Accuracy delta | {signed_pct(summary.accuracy_delta)} |
| Avg summary score | {summary.avg_summary_score:.2f}/5 |
| Regressions | {len(summary.regressions)} |
| Improvements | {len(summary.improvements)} |
| Drift warning | {"Yes" if summary.drift_warning else "No"} |

Regressed case IDs: {", ".join(summary.regressions) if summary.regressions else "None"}
"""
    path.write_text(body, encoding="utf-8")


def _status_badge(status: str) -> str:
    if status == "critical":
        return "🚨 CRITICAL"
    if status == "warn":
        return "⚠️ WARNING"
    return "✅ PASS"


def generate_html_report(summary: EvalSummary, cases: list[EvalCaseResult], previous_summary: EvalSummary | None, previous_cases: dict[str, EvalCaseResult], output_dir: str | Path) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"eval_report_{summary.run_id}.html"

    rows = []
    for result in cases:
        marker = "✅" if result.passed else "❌"
        rows.append(f"""
        <tr>
          <td>{marker} <code>{escape(result.case_id)}</code></td>
          <td>{escape(result.difficulty)}</td>
          <td>{escape(result.expected_category)}</td>
          <td>{escape(result.actual_category)}</td>
          <td>{"yes" if result.category_match else "no"}</td>
          <td>{result.summary_score}/5</td>
          <td>{escape(result.actual_summary)}</td>
          <td>{escape(result.error or "")}</td>
        </tr>
        """)

    previous_accuracy = pct(previous_summary.accuracy) if previous_summary else "No previous run"

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>LLM Regression Eval Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; color: #172033; background: #f8fafc; }}
    .card {{ background: white; border: 1px solid #e2e8f0; border-radius: 16px; padding: 20px; margin: 16px 0; }}
    table {{ width: 100%; border-collapse: collapse; background: white; }}
    th, td {{ border-bottom: 1px solid #e2e8f0; text-align: left; vertical-align: top; padding: 10px; font-size: 14px; }}
    th {{ background: #f1f5f9; }}
    code {{ background:#eef2ff; padding:2px 6px; border-radius:6px; }}
    .grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; }}
    .metric {{ background:#f8fafc; border-radius:12px; padding:14px; border:1px solid #e2e8f0; }}
    .metric strong {{ display:block; font-size: 24px; margin-top:6px; }}
  </style>
</head>
<body>
  <h1>LLM Regression Eval Report {_status_badge(summary.status)}</h1>
  <p>Run <code>{escape(summary.run_id)}</code> · Prompt <code>{escape(summary.prompt_version)}</code> · Model <code>{escape(summary.model)}</code></p>

  <div class="card grid">
    <div class="metric">Accuracy<strong>{pct(summary.accuracy)}</strong></div>
    <div class="metric">Previous accuracy<strong>{previous_accuracy}</strong></div>
    <div class="metric">Delta<strong>{signed_pct(summary.accuracy_delta)}</strong></div>
    <div class="metric">Avg summary score<strong>{summary.avg_summary_score:.2f}/5</strong></div>
  </div>

  <div class="card grid">
    <div class="metric">Regressions<strong>{len(summary.regressions)}</strong></div>
    <div class="metric">Improvements<strong>{len(summary.improvements)}</strong></div>
    <div class="metric">Avg latency<strong>{summary.avg_latency_ms:.0f} ms</strong></div>
    <div class="metric">Tokens<strong>{summary.input_tokens + summary.output_tokens}</strong></div>
  </div>

  <div class="card">
    <h2>All case results</h2>
    <table>
      <thead><tr><th>Case</th><th>Difficulty</th><th>Expected</th><th>Actual</th><th>Category match</th><th>Summary score</th><th>Actual summary</th><th>Error</th></tr></thead>
      <tbody>{''.join(rows)}</tbody>
    </table>
  </div>
</body>
</html>"""
    report_path.write_text(html, encoding="utf-8")
    return report_path
