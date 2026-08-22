"""Runs a test suite against a system prompt and checks assertions on each output."""

import re

from assertions import check_assertions
from llm_client import run_prompt

_FENCE_RE = re.compile(r"^```(?:json)?\s*\n?(.*?)\n?```$", re.DOTALL)


def _strip_code_fence(text: str) -> str:
    """Claude sometimes wraps JSON output in markdown code fences even when
    told not to. Strip that off before checking assertions, since the fence
    is a formatting choice, not something we're testing for here."""
    match = _FENCE_RE.match(text.strip())
    return match.group(1).strip() if match else text


def run_test_suite(system_prompt: str, test_cases: list, model: str | None = None) -> dict:
    results = {}
    for case in test_cases:
        kwargs = {"system_prompt": system_prompt, "user_input": case["input"]}
        if model:
            kwargs["model"] = model
        raw_output = run_prompt(**kwargs)
        output = _strip_code_fence(raw_output)
        result = check_assertions(output, case.get("assertions", []))
        results[case["id"]] = {
            "output": result.output,
            "passed": result.passed,
            "assertion_results": result.assertion_results,
        }
    return results
