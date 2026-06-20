# Model Regression Detection System

A production-style evaluation harness for catching LLM prompt regressions before they ship.

This project simulates the kind of quality gate a team would add to CI/CD when prompt behavior matters. It evaluates a customer-support classifier against a curated golden dataset, scores summary quality, compares results with previous runs, and publishes reports that can be reviewed in GitHub Actions.

## Why This Exists

LLM prompts can regress quietly. A wording change that improves one case can break another, and manual spot checks usually miss those failures. This repo turns prompt evaluation into a repeatable engineering workflow:

- run the same golden cases every time;
- measure category accuracy and summary quality;
- detect case-level regressions and improvements;
- store history in SQLite for previous-run comparison;
- publish Markdown and HTML reports for fast review;
- support local fake mode so the pipeline is testable without API keys.

## Features

| Feature | What it does |
|---|---|
| Golden dataset | 50 labeled support-ticket cases across billing, technical, account, and general intents |
| Prompt versioning | Evaluates prompts stored in `prompts/v1.yaml` |
| LLM classifier | Calls an OpenAI model for real classification runs |
| Judge model | Scores whether generated summaries match expected meaning |
| Fake mode | Runs the full pipeline locally without external API calls |
| Regression tracking | Flags accuracy drops, regressions, improvements, and rolling drift |
| Report generation | Writes Markdown summaries and detailed HTML reports |
| GitHub Actions | Runs validation and fake eval on PRs, with optional manual real eval |
| Slack hook | Sends run status to Slack when `SLACK_WEBHOOK_URL` is configured |

## Architecture

```text
Golden dataset + prompt
        |
        v
CLI runner
        |
        v
Classifier under test ----> Summary judge
        |                         |
        +-----------+-------------+
                    v
             Evaluation summary
                    |
        +-----------+-------------+
        v                         v
   SQLite history           Markdown/HTML reports
```

## Repository Layout

| Path | Purpose |
|---|---|
| `src/model_regression/cli.py` | Command-line entry point |
| `src/model_regression/evaluator.py` | Evaluation orchestration and regression logic |
| `src/model_regression/llm_feature.py` | Classifier implementation and fake-mode classifier |
| `src/model_regression/judge.py` | Summary quality judge |
| `src/model_regression/reporting.py` | Markdown and HTML report generation |
| `src/model_regression/storage.py` | SQLite run history |
| `data/golden_dataset_v1.json` | 50-case golden dataset |
| `prompts/v1.yaml` | Prompt configuration and few-shot examples |
| `.github/workflows/eval.yml` | Portfolio-ready CI workflow |
| `RUNBOOK.md` | Short operational runbook |

## Quickstart

Use Python 3.11 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

## Run The Fake Eval

Fake mode is the best local smoke test because it does not require OpenAI credentials:

```bash
python -m model_regression.cli --fake
```

Expected result for the current dataset:

```text
Status: PASS
Accuracy: 100.0%
Regressions: 0
Improvements: 0
```

## Run A Real Eval

Real mode requires `OPENAI_API_KEY`.

```bash
OPENAI_API_KEY=sk-... python -m model_regression.cli \
  --prompt prompts/v1.yaml \
  --dataset data/golden_dataset_v1.json \
  --model gpt-4o-mini \
  --judge-model gpt-4o-mini \
  --db runs/evals.db \
  --report-dir reports
```

Optional environment variables:

| Variable | Default | Purpose |
|---|---:|---|
| `WARNING_DROP` | `0.03` | Accuracy drop that marks a run as warning |
| `CRITICAL_DROP` | `0.08` | Accuracy drop that marks a run as critical |
| `DRIFT_AVG_MIN` | `0.85` | Minimum rolling average before drift warning |
| `SLACK_WEBHOOK_URL` | unset | Sends run status to Slack |

## Outputs

Each run creates:

- `reports/summary.md` for PR comments and quick review;
- `reports/eval_report_<run_id>.html` for detailed case-level inspection;
- `runs/evals.db` for historical comparison.

Generated reports and run databases are intentionally ignored by git.

## GitHub Actions

The workflow is designed to look good in a public portfolio while staying practical:

- validates that the package compiles;
- verifies the golden dataset has exactly 50 unique cases;
- runs fake mode on pull requests and pushes to `main`;
- uploads eval reports as artifacts;
- writes a GitHub step summary;
- updates a sticky PR comment instead of spamming duplicate comments;
- supports manual `real` mode when repository secrets are configured.

Manual real eval setup:

1. Add `OPENAI_API_KEY` as a repository secret.
2. Optionally add `SLACK_WEBHOOK_URL`.
3. Open the Actions tab.
4. Run **LLM Regression Evaluation** with `eval_mode=real`.

## Golden Dataset Design

The dataset includes easy, medium, and hard cases across:

- billing issues such as invoices, refunds, tax, coupons, charges, and renewals;
- technical issues such as crashes, API errors, upload failures, slow search, and browser bugs;
- account issues such as password recovery, SSO, MFA, role changes, invites, and workspace ownership;
- general questions such as roadmap, documentation, discounts, support policy, and product use cases.

Hard cases intentionally include ambiguity, short messages, sarcasm, multilingual wording, and overlapping categories.

## Portfolio Talking Points

This project demonstrates:

- evaluation-driven LLM engineering;
- CI/CD integration for prompt changes;
- golden dataset design;
- fake-mode testing for fast local development;
- regression and drift detection;
- practical reporting for pull request review.

## Extending The Project

Good next improvements would be adding pytest coverage, persisting report artifacts to cloud storage, comparing multiple prompt versions side by side, and adding per-category accuracy dashboards.
