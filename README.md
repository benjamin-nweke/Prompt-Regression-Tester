# Prompt Regression Tester

Prompt Regression Tester compares a trusted prompt with a proposed change across a reusable evaluation suite. It repeats each case, checks exact requirements with tolerant patterns where equivalent formatting is valid, applies optional semantic rubrics, and separates stable regressions from model variance and mixed evaluator signals.

The included default suite comes from a real technical-editing workflow used for Towards Data Science drafts. It tests whether prompt edits preserve code identifiers, numerical evidence, uncertainty, causal mechanisms, implementation caveats, and honest evidence boundaries. You can replace it in the UI or import any JSON suite.

## What it does

- **Repeated runs per case** — run each prompt 1, 3, or 5 times so one unlucky generation does not automatically become a regression.
- **Stability labels** — each case is classified as `stable_pass`, `flaky`, or `stable_fail`.
- **Regression classification** — only a stable baseline pass followed by a stable candidate failure is called a confirmed regression; mixed outcomes remain flaky drops.
- **LLM judge with a rubric** — deterministic checks handle exact contracts while optional rubrics evaluate semantic quality.
- **Hard checks and soft signals** — exact requirements can gate a run, while brittle string or regex checks can remain advisory.
- **Mixed evaluator signals** — when deterministic checks and the semantic judge reach different conclusions, the tool surfaces the conflict instead of quietly turning it into a false regression.
- **Baseline reuse** — if the trusted prompt, suite, models, and run count have not changed, the previous baseline evaluation is reused while you iterate on candidates.
- **Visible progress** — each completed run updates the progress bar so longer evaluations do not look frozen.
- **Create tests in the UI** — add cases without editing JSON by hand.
- **Import and export** — reuse test suites and export a full JSON report after evaluation.
- **JSON contract checks** — validate required keys and simple enums, not just whether JSON parses.

## The included real-world suite

The default baseline is a maintained technical-editing prompt. The proposed candidate is a plausible tightening change: it asks for more concise, confident prose and permits removal of hedging, examples, and implementation detail that appear secondary.

The six default cases test different failure modes:

1. **Structured output meaning** — preserve the distinction between schema correctness and semantic correctness, including uncertainty around a missing `customer_id`.
2. **Hidden multi-agent failure** — preserve how a later reviewer can hide an earlier handoff failure from final-output evaluation.
3. **Framework trade-off and timings** — preserve four concrete timings and a balanced first-person judgment about framework versus native architecture.
4. **GRPO equal-reward branch** — preserve `group_mean`, `group_std`, the zero-variance branch, and the ranking logic.
5. **Verifier constraints** — preserve exact accepted/rejected parser examples without confusing parser limitations with the learning algorithm.
6. **Honest claims about unrun training** — preserve the boundary between executed mechanics demonstrations and claims about model training or benchmark improvement.

These cases are deliberately mixed. Some requirements are exact and deterministic; others need semantic judgment. That mix is the point.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
export ANTHROPIC_API_KEY="sk-ant-..."
python3 -m streamlit run app.py
```

Optional model overrides:

```bash
export ANTHROPIC_MODEL="claude-haiku-4-5-20251001"
export ANTHROPIC_JUDGE_MODEL="claude-haiku-4-5-20251001"
```

## How evaluation works

Each test case can have two layers:

1. **Deterministic checks** such as `contains`, `not_contains`, `regex_match`, length bounds, JSON validity, and a small JSON contract check.
2. **A semantic rubric** evaluated by an LLM judge.

Hard checks always gate a run. When a rubric is present, soft string and regex checks are treated as signals rather than absolute proof of failure.

## Regression labels

The tool intentionally uses a simple, visible heuristic rather than pretending it has statistical certainty:

- **regression** — the baseline is a stable pass and the candidate is a stable fail across the repeated runs
- **flaky_drop** — the candidate is worse, but its outcomes are mixed rather than a stable failure
- **improvement** — the baseline is a stable fail and the candidate becomes a stable pass
- **small_improvement** — the candidate improves without moving from stable fail to stable pass
- **flaky_no_change** — the pass rate is unchanged, but at least one side is mixed across runs
- **stable** — no change and neither side is flaky

The classification logic is defined in `runner.py` so it can be inspected and changed.

## Hard checks, soft signals, and mixed evaluator signals

Hard checks are for things that truly must be exact: valid JSON, required fields, code identifiers, numbers, or other explicit contracts.

Soft checks are useful but more brittle. A keyword may disappear while the meaning remains intact. When a semantic rubric is present, those checks act as signals. If deterministic checks and the judge reach different conclusions, the run is marked with **mixed evaluator signals** for review.

## Cost warning

Repeated evaluation multiplies API calls. Start with 3 runs while iterating. After the first comparison, an unchanged baseline is reused so subsequent candidate iterations avoid repeating those baseline calls.

## Known limitations

- An LLM judge can itself be inconsistent or biased toward outputs from the same model family.
- Three repeated runs are a practical signal, not a statistical guarantee.
- Small suites can still give misleading confidence.
- The project currently uses a single model provider so the evaluation logic remains easy to inspect.

The tool is designed to make uncertainty visible, not to pretend prompt evaluation is deterministic.
