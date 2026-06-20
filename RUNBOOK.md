# Runbook

Use Python 3.11 or newer.

Install:

```bash
python -m venv .venv
pip install -r requirements.txt
pip install -e .
```

Run local test mode:

```bash
python -m model_regression.cli --fake
```

Main folders:

- prompts: prompt versions
- data: golden test cases
- src: evaluation code
- reports: generated reports

For Codex, ask it to inspect the project, run fake mode, improve the README, expand the golden dataset, and make the workflow production-ready.
