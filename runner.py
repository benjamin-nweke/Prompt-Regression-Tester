"""Repeated prompt evaluation and regression classification."""

from __future__ import annotations

import re
from typing import Any, Callable

from assertions import check_assertions
from llm_client import judge_output, run_prompt

_FENCE_RE = re.compile(r"^```(?:json)?\s*\n?(.*?)\n?```$", re.DOTALL | re.IGNORECASE)
_SOFT_DEFAULT_TYPES = {"contains", "not_contains", "regex_match"}


def _strip_code_fence(text: str) -> str:
    match = _FENCE_RE.match(text.strip())
    return match.group(1).strip() if match else text.strip()


def _stability(pass_count: int, total_runs: int) -> str:
    if pass_count == total_runs:
        return "stable_pass"
    if pass_count == 0:
        return "stable_fail"
    return "flaky"


def _assertion_severity(assertion: dict[str, Any]) -> str:
    explicit = assertion.get("severity")
    if explicit in {"hard", "soft"}:
        return explicit
    return "soft" if assertion.get("type") in _SOFT_DEFAULT_TYPES else "hard"


def run_case(
    system_prompt: str,
    case: dict[str, Any],
    runs: int = 3,
    model: str | None = None,
    judge_model: str | None = None,
    run_fn: Callable[..., str] = run_prompt,
    judge_fn: Callable[..., dict[str, Any]] = judge_output,
    progress_callback: Callable[[str, int, int], None] | None = None,
) -> dict[str, Any]:
    if runs < 1:
        raise ValueError("runs must be at least 1")

    attempts: list[dict[str, Any]] = []
    rubric = (case.get("rubric") or "").strip()
    assertion_specs = case.get("assertions", [])

    for run_number in range(1, runs + 1):
        prompt_kwargs: dict[str, Any] = {
            "system_prompt": system_prompt,
            "user_input": case["input"],
        }
        if model:
            prompt_kwargs["model"] = model

        output = _strip_code_fence(run_fn(**prompt_kwargs))
        deterministic = check_assertions(output, assertion_specs)

        assertion_results = []
        hard_results = []
        soft_results = []
        for spec, result in zip(assertion_specs, deterministic.assertion_results):
            severity = _assertion_severity(spec)
            item = {**result.__dict__, "severity": severity}
            assertion_results.append(item)
            (hard_results if severity == "hard" else soft_results).append(item)

        hard_passed = all(item["passed"] for item in hard_results)
        soft_passed = all(item["passed"] for item in soft_results)

        judge = None
        if rubric:
            judge_kwargs: dict[str, Any] = {
                "user_input": case["input"],
                "output": output,
                "rubric": rubric,
            }
            if judge_model:
                judge_kwargs["model"] = judge_model
            judge = judge_fn(**judge_kwargs)

        judge_passed = None if judge is None else bool(judge.get("passed"))

        # Hard checks always gate. When a semantic rubric exists, soft string
        # checks are signals rather than gates; the judge decides semantic fit.
        if judge is not None:
            passed = hard_passed and judge_passed
        else:
            passed = hard_passed and soft_passed

        evaluator_disagreement = bool(
            judge is not None
            and (
                hard_passed != judge_passed
                or (soft_results and soft_passed != judge_passed)
            )
        )

        attempts.append(
            {
                "run": run_number,
                "output": output,
                "passed": passed,
                "deterministic_passed": deterministic.passed,
                "hard_deterministic_passed": hard_passed,
                "soft_signal_passed": soft_passed,
                "assertion_results": assertion_results,
                "judge": judge,
                "evaluator_disagreement": evaluator_disagreement,
            }
        )

        if progress_callback is not None:
            progress_callback(case["id"], run_number, runs)

    pass_count = sum(1 for attempt in attempts if attempt["passed"])
    disagreement_count = sum(1 for attempt in attempts if attempt["evaluator_disagreement"])
    return {
        "id": case["id"],
        "runs": attempts,
        "pass_count": pass_count,
        "total_runs": runs,
        "pass_rate": pass_count / runs,
        "stability": _stability(pass_count, runs),
        "disagreement_count": disagreement_count,
    }


def run_test_suite(
    system_prompt: str,
    test_cases: list[dict[str, Any]],
    runs: int = 3,
    model: str | None = None,
    judge_model: str | None = None,
    run_fn: Callable[..., str] = run_prompt,
    judge_fn: Callable[..., dict[str, Any]] = judge_output,
    progress_callback: Callable[[str, int, int], None] | None = None,
) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for case in test_cases:
        results[case["id"]] = run_case(
            system_prompt=system_prompt,
            case=case,
            runs=runs,
            model=model,
            judge_model=judge_model,
            run_fn=run_fn,
            judge_fn=judge_fn,
            progress_callback=progress_callback,
        )
    return results


def classify_change(baseline: dict[str, Any], candidate: dict[str, Any]) -> str:
    """Separate stable regressions from stochastic drops.

    A real regression requires a stable baseline pass and a stable candidate
    failure. Mixed candidate outcomes stay classified as flaky even when the
    pass-rate drop is large; this avoids overstating three-run noise as a
    confirmed regression.
    """

    delta = candidate["pass_rate"] - baseline["pass_rate"]

    if baseline["stability"] == "stable_pass" and candidate["stability"] == "stable_fail":
        return "regression"
    if delta < 0:
        return "flaky_drop"
    if baseline["stability"] == "stable_fail" and candidate["stability"] == "stable_pass":
        return "improvement"
    if delta > 0:
        return "small_improvement"
    if candidate["stability"] == "flaky" or baseline["stability"] == "flaky":
        return "flaky_no_change"
    return "stable"


def compare_suites(
    baseline_results: dict[str, dict[str, Any]],
    candidate_results: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []

    for case_id, baseline in baseline_results.items():
        candidate = candidate_results[case_id]
        delta = candidate["pass_rate"] - baseline["pass_rate"]
        rows.append(
            {
                "id": case_id,
                "baseline_pass_rate": baseline["pass_rate"],
                "candidate_pass_rate": candidate["pass_rate"],
                "delta": delta,
                "baseline_stability": baseline["stability"],
                "candidate_stability": candidate["stability"],
                "baseline_disagreements": baseline.get("disagreement_count", 0),
                "candidate_disagreements": candidate.get("disagreement_count", 0),
                "classification": classify_change(baseline, candidate),
            }
        )

    def overall(results: dict[str, dict[str, Any]]) -> float:
        total_passes = sum(result["pass_count"] for result in results.values())
        total_runs = sum(result["total_runs"] for result in results.values())
        return total_passes / total_runs if total_runs else 0.0

    baseline_rate = overall(baseline_results)
    candidate_rate = overall(candidate_results)

    return {
        "rows": rows,
        "baseline_pass_rate": baseline_rate,
        "candidate_pass_rate": candidate_rate,
        "delta": candidate_rate - baseline_rate,
        "regressions": sum(row["classification"] == "regression" for row in rows),
        "flaky_drops": sum(row["classification"] == "flaky_drop" for row in rows),
        "improvements": sum(
            row["classification"] in {"improvement", "small_improvement"} for row in rows
        ),
        "candidate_disagreements": sum(
            result.get("disagreement_count", 0) for result in candidate_results.values()
        ),
    }
