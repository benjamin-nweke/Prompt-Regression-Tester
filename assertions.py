"""Deterministic checks for model outputs.

These checks are deliberately boring: they should be cheap, reproducible, and
explainable. Semantic checks live in the LLM judge instead of being hidden in
string heuristics here.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class AssertionResult:
    type: str
    value: Any
    passed: bool
    detail: str = ""


@dataclass
class TestResult:
    output: str
    assertion_results: list[AssertionResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(result.passed for result in self.assertion_results)


def _contains(output: str, value: Any) -> AssertionResult:
    needle = str(value)
    ok = needle.lower() in output.lower()
    return AssertionResult("contains", value, ok, "" if ok else f"missing '{needle}'")


def _not_contains(output: str, value: Any) -> AssertionResult:
    needle = str(value)
    ok = needle.lower() not in output.lower()
    return AssertionResult(
        "not_contains", value, ok, "" if ok else f"should not contain '{needle}'"
    )


def _max_length(output: str, value: Any) -> AssertionResult:
    limit = int(value)
    ok = len(output) <= limit
    return AssertionResult(
        "max_length",
        value,
        ok,
        "" if ok else f"{len(output)} chars, limit {limit}",
    )


def _min_length(output: str, value: Any) -> AssertionResult:
    limit = int(value)
    ok = len(output) >= limit
    return AssertionResult(
        "min_length",
        value,
        ok,
        "" if ok else f"{len(output)} chars, needs {limit}+",
    )


def _regex_match(output: str, value: Any) -> AssertionResult:
    pattern = str(value)
    ok = re.search(pattern, output) is not None
    return AssertionResult(
        "regex_match", value, ok, "" if ok else f"no match for /{pattern}/"
    )


def _json_valid(output: str, value: Any = None) -> AssertionResult:
    try:
        json.loads(output)
        return AssertionResult("json_valid", value, True)
    except json.JSONDecodeError as exc:
        return AssertionResult("json_valid", value, False, str(exc))


def _json_schema(output: str, value: Any) -> AssertionResult:
    """Check a tiny, intentionally limited JSON contract.

    Expected value shape:
    {
      "required": ["category", "priority"],
      "enum": {"priority": ["low", "medium", "high"]}
    }
    """

    try:
        data = json.loads(output)
    except json.JSONDecodeError as exc:
        return AssertionResult("json_schema", value, False, f"invalid JSON: {exc}")

    if not isinstance(data, dict):
        return AssertionResult("json_schema", value, False, "top-level JSON must be an object")

    contract = value if isinstance(value, dict) else {}
    required = contract.get("required", [])
    missing = [key for key in required if key not in data]
    if missing:
        return AssertionResult(
            "json_schema", value, False, f"missing required keys: {', '.join(missing)}"
        )

    for field_name, allowed_values in contract.get("enum", {}).items():
        if field_name in data and data[field_name] not in allowed_values:
            return AssertionResult(
                "json_schema",
                value,
                False,
                f"{field_name}={data[field_name]!r} not in {allowed_values!r}",
            )

    return AssertionResult("json_schema", value, True)


CHECKERS = {
    "contains": _contains,
    "not_contains": _not_contains,
    "max_length": _max_length,
    "min_length": _min_length,
    "regex_match": _regex_match,
    "json_valid": _json_valid,
    "json_schema": _json_schema,
}


def check_assertions(output: str, assertions: list[dict[str, Any]]) -> TestResult:
    results: list[AssertionResult] = []
    for assertion in assertions:
        assertion_type = assertion.get("type")
        checker = CHECKERS.get(assertion_type)
        if checker is None:
            results.append(
                AssertionResult(
                    assertion_type or "unknown",
                    assertion.get("value"),
                    False,
                    "unknown assertion type",
                )
            )
            continue
        results.append(checker(output, assertion.get("value")))

    return TestResult(output=output, assertion_results=results)
