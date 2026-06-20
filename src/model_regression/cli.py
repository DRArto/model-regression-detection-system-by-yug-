from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path
from dotenv import load_dotenv

from .dataset import load_dataset
from .prompt_loader import load_prompt
from .evaluator import run_evaluation
from .reporting import generate_html_report, generate_markdown_summary, pct, signed_pct
from .slack import send_slack_alert
from .storage import save_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LLM prompt regression evaluation.")
    parser.add_argument("--prompt", default="prompts/v1.yaml", help="Path to prompt YAML file")
    parser.add_argument("--dataset", default="data/golden_dataset_v1.json", help="Path to golden dataset JSON")
    parser.add_argument("--model", default="gpt-4o-mini", help="Model used by the feature under test")
    parser.add_argument("--judge-model", default="gpt-4o-mini", help="Model used by the summary judge")
    parser.add_argument("--db", default="runs/evals.db", help="SQLite database path")
    parser.add_argument("--report-dir", default="reports", help="Directory for generated reports")
    parser.add_argument("--max-concurrency", type=int, default=5)
    parser.add_argument("--fake", action="store_true", help="Run without OpenAI API calls for local testing")
    return parser.parse_args()


async def async_main() -> int:
    load_dotenv()
    args = parse_args()

    prompt = load_prompt(args.prompt)
    cases = load_dataset(args.dataset)

    warning_drop = float(os.getenv("WARNING_DROP", "0.03"))
    critical_drop = float(os.getenv("CRITICAL_DROP", "0.08"))
    drift_avg_min = float(os.getenv("DRIFT_AVG_MIN", "0.85"))

    summary, results, previous_summary, previous_cases = await run_evaluation(
        prompt=prompt,
        cases=cases,
        model=args.model,
        judge_model=args.judge_model,
        db_path=args.db,
        max_concurrency=args.max_concurrency,
        warning_drop=warning_drop,
        critical_drop=critical_drop,
        drift_avg_min=drift_avg_min,
        fake=args.fake,
    )

    report_path = generate_html_report(summary, results, previous_summary, previous_cases, args.report_dir)
    summary_path = Path(args.report_dir) / "summary.md"
    generate_markdown_summary(summary, summary_path)
    save_run(args.db, summary, results, report_path=str(report_path))

    webhook = os.getenv("SLACK_WEBHOOK_URL")
    send_slack_alert(webhook, summary, report_path=str(report_path))

    print(f"Status: {summary.status.upper()}")
    print(f"Accuracy: {pct(summary.accuracy)} ({signed_pct(summary.accuracy_delta)})")
    print(f"Regressions: {len(summary.regressions)}")
    print(f"Improvements: {len(summary.improvements)}")
    print(f"Report: {report_path}")

    return 1 if summary.status == "critical" else 0


def main() -> None:
    raise SystemExit(asyncio.run(async_main()))


if __name__ == "__main__":
    main()
