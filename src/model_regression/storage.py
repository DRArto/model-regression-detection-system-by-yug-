from __future__ import annotations

from pathlib import Path
import json
import sqlite3

from .schemas import EvalCaseResult, EvalSummary


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS eval_runs (
    run_id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    model TEXT NOT NULL,
    total_cases INTEGER NOT NULL,
    accuracy REAL NOT NULL,
    avg_summary_score REAL NOT NULL,
    avg_latency_ms REAL NOT NULL,
    input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    status TEXT NOT NULL,
    accuracy_delta REAL,
    regressions_json TEXT NOT NULL,
    improvements_json TEXT NOT NULL,
    drift_warning INTEGER NOT NULL,
    report_path TEXT,
    summary_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS eval_case_results (
    run_id TEXT NOT NULL,
    case_id TEXT NOT NULL,
    expected_category TEXT NOT NULL,
    actual_category TEXT NOT NULL,
    expected_summary TEXT NOT NULL,
    actual_summary TEXT NOT NULL,
    category_match INTEGER NOT NULL,
    summary_score INTEGER NOT NULL,
    summary_rationale TEXT NOT NULL,
    passed INTEGER NOT NULL,
    latency_ms INTEGER NOT NULL,
    input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    difficulty TEXT NOT NULL,
    notes TEXT,
    error TEXT,
    result_json TEXT NOT NULL,
    PRIMARY KEY (run_id, case_id)
);
"""


def connect(db_path: str | Path) -> sqlite3.Connection:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA_SQL)
    return conn


def save_run(db_path: str | Path, summary: EvalSummary, cases: list[EvalCaseResult], report_path: str | None = None) -> None:
    conn = connect(db_path)
    try:
        with conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO eval_runs
                (run_id, timestamp, prompt_version, model, total_cases, accuracy,
                 avg_summary_score, avg_latency_ms, input_tokens, output_tokens,
                 status, accuracy_delta, regressions_json, improvements_json,
                 drift_warning, report_path, summary_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    summary.run_id,
                    summary.timestamp,
                    summary.prompt_version,
                    summary.model,
                    summary.total_cases,
                    summary.accuracy,
                    summary.avg_summary_score,
                    summary.avg_latency_ms,
                    summary.input_tokens,
                    summary.output_tokens,
                    summary.status,
                    summary.accuracy_delta,
                    json.dumps(summary.regressions),
                    json.dumps(summary.improvements),
                    int(summary.drift_warning),
                    report_path,
                    summary.model_dump_json(),
                ),
            )
            for result in cases:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO eval_case_results
                    (run_id, case_id, expected_category, actual_category,
                     expected_summary, actual_summary, category_match,
                     summary_score, summary_rationale, passed, latency_ms,
                     input_tokens, output_tokens, difficulty, notes, error, result_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        summary.run_id,
                        result.case_id,
                        result.expected_category,
                        result.actual_category,
                        result.expected_summary,
                        result.actual_summary,
                        int(result.category_match),
                        result.summary_score,
                        result.summary_rationale,
                        int(result.passed),
                        result.latency_ms,
                        result.input_tokens,
                        result.output_tokens,
                        result.difficulty,
                        result.notes,
                        result.error,
                        result.model_dump_json(),
                    ),
                )
    finally:
        conn.close()


def _summary_from_row(row: sqlite3.Row) -> EvalSummary:
    return EvalSummary.model_validate_json(row["summary_json"])


def _case_from_row(row: sqlite3.Row) -> EvalCaseResult:
    return EvalCaseResult.model_validate_json(row["result_json"])


def get_previous_run(db_path: str | Path) -> tuple[EvalSummary, dict[str, EvalCaseResult]] | None:
    db_path = Path(db_path)
    if not db_path.exists():
        return None
    conn = connect(db_path)
    try:
        run = conn.execute("SELECT * FROM eval_runs ORDER BY timestamp DESC LIMIT 1").fetchone()
        if not run:
            return None
        summary = _summary_from_row(run)
        rows = conn.execute("SELECT * FROM eval_case_results WHERE run_id = ?", (summary.run_id,)).fetchall()
        cases = {_case_from_row(row).case_id: _case_from_row(row) for row in rows}
        return summary, cases
    finally:
        conn.close()


def get_recent_accuracies(db_path: str | Path, limit: int = 6) -> list[float]:
    db_path = Path(db_path)
    if not db_path.exists():
        return []
    conn = connect(db_path)
    try:
        rows = conn.execute("SELECT accuracy FROM eval_runs ORDER BY timestamp DESC LIMIT ?", (limit,)).fetchall()
        return [float(row["accuracy"]) for row in rows]
    finally:
        conn.close()
