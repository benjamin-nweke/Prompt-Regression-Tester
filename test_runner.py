import json
import unittest

from assertions import check_assertions
from runner import classify_change, run_case


class AssertionTests(unittest.TestCase):
    def test_json_schema_checks_required_and_enum(self):
        output = json.dumps({"category": "billing", "priority": "high"})
        result = check_assertions(
            output,
            [{"type": "json_schema", "value": {"required": ["category", "priority"], "enum": {"priority": ["low", "medium", "high"]}}}],
        )
        self.assertTrue(result.passed)


class RunnerTests(unittest.TestCase):
    def test_mixed_attempts_are_flaky_without_judge(self):
        outputs = iter(["keep this", "drop it", "keep this"])
        def fake_run(**_kwargs): return next(outputs)
        case = {"id": "case_1", "input": "x", "assertions": [{"type": "contains", "value": "keep"}], "rubric": ""}
        result = run_case("prompt", case, runs=3, run_fn=fake_run)
        self.assertEqual(result["pass_count"], 2)
        self.assertEqual(result["stability"], "flaky")

    def test_judge_can_fail_output_that_soft_strings_accept(self):
        def fake_run(**_kwargs): return "schema semantic"
        def fake_judge(**_kwargs): return {"score": 2, "passed": False, "reason": "Lost the core distinction."}
        case = {"id": "case_2", "input": "x", "assertions": [{"type": "contains", "value": "schema"}, {"type": "contains", "value": "semantic"}], "rubric": "Preserve the distinction."}
        result = run_case("prompt", case, runs=1, run_fn=fake_run, judge_fn=fake_judge)
        self.assertEqual(result["pass_count"], 0)
        self.assertTrue(result["runs"][0]["soft_signal_passed"])
        self.assertTrue(result["runs"][0]["evaluator_disagreement"])

    def test_soft_string_false_negative_does_not_fail_when_judge_passes(self):
        def fake_run(**_kwargs): return "Schema validation checks structure, not meaning."
        def fake_judge(**_kwargs): return {"score": 5, "passed": True, "reason": "Meaning preserved."}
        case = {"id": "case_3", "input": "x", "assertions": [{"type": "contains", "value": "semantic", "severity": "soft"}], "rubric": "Preserve structural vs semantic distinction."}
        result = run_case("prompt", case, runs=1, run_fn=fake_run, judge_fn=fake_judge)
        self.assertEqual(result["pass_count"], 1)
        self.assertFalse(result["runs"][0]["soft_signal_passed"])
        self.assertTrue(result["runs"][0]["evaluator_disagreement"])

    def test_hard_exact_requirement_still_gates_even_when_judge_passes(self):
        def fake_run(**_kwargs): return "The function validates JSON."
        def fake_judge(**_kwargs): return {"score": 5, "passed": True, "reason": "Semantics fine."}
        case = {"id": "case_4", "input": "x", "assertions": [{"type": "contains", "value": "model_validate_json", "severity": "hard"}], "rubric": "Preserve meaning."}
        result = run_case("prompt", case, runs=1, run_fn=fake_run, judge_fn=fake_judge)
        self.assertEqual(result["pass_count"], 0)
        self.assertFalse(result["runs"][0]["hard_deterministic_passed"])
        self.assertTrue(result["runs"][0]["evaluator_disagreement"])

    def test_timing_regex_accepts_numeric_word_and_hyphenated_forms(self):
        pattern = r"(?i)\b(?:4|four)(?:\s+|-)(?:minutes?|mins?)\b"
        for output in [
            "The fix took 4 minutes.",
            "The fix took four minutes.",
            "It was a four-minute fix.",
            "It was a 4-minute fix.",
        ]:
            result = check_assertions(
                output,
                [{"type": "regex_match", "value": pattern}],
            )
            self.assertTrue(result.passed)

    def test_grpo_zero_variance_rule_accepts_equivalent_phrasing(self):
        pattern = r"(?i)(?:group_std\s*(?:==|=)\s*0|group_std\s+(?:is|equals?)\s+zero|zero(?:-|\s+)standard(?:-|\s+)deviation)"
        for output in [
            "when group_std == 0, every advantage should be zero",
            "group_std is zero, so every advantage should be zero",
            "a zero standard deviation means every advantage should be zero",
        ]:
            result = check_assertions(
                output,
                [{"type": "regex_match", "value": pattern}],
            )
            self.assertTrue(result.passed)


    def test_real_suite_timing_patterns_match_source_text(self):
        from pathlib import Path
        cases = json.loads(Path("sample_test_cases.json").read_text())
        case = next(c for c in cases if c["id"] == "native_architecture_tradeoff")
        result = check_assertions(case["input"], case["assertions"])
        self.assertTrue(result.passed)

    def test_progress_callback_fires_once_per_completed_run(self):
        seen = []
        outputs = iter(["ok", "ok", "ok"])

        def fake_run(**_kwargs):
            return next(outputs)

        def progress(case_id, run_number, total_runs):
            seen.append((case_id, run_number, total_runs))

        case = {"id": "case_progress", "input": "x", "assertions": [], "rubric": ""}
        result = run_case("prompt", case, runs=3, run_fn=fake_run, progress_callback=progress)

        self.assertEqual(result["pass_count"], 3)
        self.assertEqual(
            seen,
            [("case_progress", 1, 3), ("case_progress", 2, 3), ("case_progress", 3, 3)],
        )

    def test_only_stable_failure_is_confirmed_regression(self):
        self.assertEqual(
            classify_change(
                {"pass_rate": 1.0, "stability": "stable_pass"},
                {"pass_rate": 0.0, "stability": "stable_fail"},
            ),
            "regression",
        )
        self.assertEqual(
            classify_change(
                {"pass_rate": 1.0, "stability": "stable_pass"},
                {"pass_rate": 1 / 3, "stability": "flaky"},
            ),
            "flaky_drop",
        )
        self.assertEqual(
            classify_change(
                {"pass_rate": 1.0, "stability": "stable_pass"},
                {"pass_rate": 2 / 3, "stability": "flaky"},
            ),
            "flaky_drop",
        )


if __name__ == "__main__":
    unittest.main()
