"""Checks a model output against a list of assertions and reports pass/fail."""

import json
import re
from dataclasses import dataclass, field


@dataclass
class AssertionResult:
    type: str
    value: str | None
    passed: bool
    detail: str = ""


@dataclass
class TestResult:
    output: str
    assertion_results: list = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(r.passed for r in self.assertion_results)


def _contains(output, value):
    ok = value.lower() in output.lower()
    return AssertionResult("contains", value, ok, "" if ok else f"missing '{value}'")


def _not_contains(output, value):
    ok = value.lower() not in output.lower()
    return AssertionResult("not_contains", value, ok, "" if ok else f"should not contain '{value}'")


def _max_length(output, value):
    limit = int(value)
    ok = len(output) <= limit
    return AssertionResult("max_length", str(value), ok, "" if ok else f"{len(output)} chars, limit {limit}")


def _min_length(output, value):
    limit = int(value)
    ok = len(output) >= limit
    return AssertionResult("min_length", str(value), ok, "" if ok else f"{len(output)} chars, needs {limit}+")


def _regex_match(output, value):
    ok = re.search(value, output) is not None
    return AssertionResult("regex_match", value, ok, "" if ok else f"no match for /{value}/")


def _json_valid(output, value=None):
    try:
        json.loads(output)
        return AssertionResult("json_valid", None, True)
    except json.JSONDecodeError as e:
        return AssertionResult("json_valid", None, False, str(e))


CHECKERS = {
    "contains": _contains,
    "not_contains": _not_contains,
    "max_length": _max_length,
    "min_length": _min_length,
    "regex_match": _regex_match,
    "json_valid": _json_valid,
}


def check_assertions(output: str, assertions: list) -> TestResult:
    results = []
    for a in assertions:
        checker = CHECKERS.get(a.get("type"))
        if checker is None:
            results.append(AssertionResult(a.get("type"), a.get("value"), False, "unknown assertion type"))
            continue
        results.append(checker(output, a.get("value")))
    return TestResult(output=output, assertion_results=results)
