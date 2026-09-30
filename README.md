# Prompt Regression Tester

Prompt Regression Tester compares a trusted prompt with a proposed change across a reusable evaluation suite. It repeats each case, checks exact requirements with tolerant patterns where equivalent formatting is valid, applies optional semantic rubrics, and separates stable regressions from model variance and mixed evaluator signals.

The included default suite is based on an EvidenceDesk-style evidence review workflow: a claim, supplied source material, and a review card. It tests whether prompt changes preserve exact evidence, caveats, conflicting sources, causal limits, and the boundary between missing evidence and a truth judgment.

## What it does

- **Repeated runs per case** — run each prompt 1, 3, or 5 times so one unusual generation does not automatically become a regression.
- **Stability labels** — each case is classified as `stable_pass`, `flaky`, or `stable_fail`.
- **Regression classification** — only a stable baseline pass followed by a stable candidate failure is called a confirmed regression.
- **LLM judge with a rubric** — deterministic checks handle exact contracts while optional rubrics evaluate semantic quality.
- **Hard checks and soft signals** — exact requirements can gate a run, while brittle string or regex checks can remain advisory.
- **Mixed evaluator signals** — conflicts between deterministic checks and the semantic judge are surfaced for review.
- **Baseline reuse** — an unchanged trusted baseline can be reused while candidate prompts are iterated.
- **Import and export** — reuse test suites and export a full JSON run report.

## The included EvidenceDesk-style suite

The baseline prompt preserves evidence-card meaning exactly. The candidate is a plausible compact-interface change: it keeps the main evidentiary conclusion but removes source dates from card prose to reduce metadata clutter.

The six default cases test:

1. **Version and date sensitive evidence** — preserve two source dates, two context-window values, and a beta-only limitation.
2. **Percentage versus percentage points** — preserve the distinction between a 5-percentage-point increase and relative percent change.
3. **Causation overreach** — keep an observational association from being rewritten as a causal result.
4. **Unsupported metric substitution** — do not turn JSON-validity evidence into a hallucination-reduction claim.
5. **Conflicting source conditions** — preserve conflicting throughput results and their benchmark context.
6. **Missing evidence is not falsity** — keep EvidenceDesk's review limitation visible: missing support does not prove a claim false.

The first case is intentionally time-sensitive. A compact-card prompt that removes dates can still sound correct while losing the information needed to interpret a versioned capability claim. That is the kind of regression this project is designed to catch.

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

- **regression** — the baseline is a stable pass and the candidate is a stable fail across repeated runs
- **flaky_drop** — the candidate is worse, but outcomes are mixed
- **improvement** — the baseline is a stable fail and the candidate becomes a stable pass
- **small_improvement** — the candidate improves without moving from stable fail to stable pass
- **flaky_no_change** — pass rate is unchanged but at least one side is mixed
- **stable** — no change and neither side is flaky

## Hard checks, soft signals, and mixed evaluator signals

Hard checks are for information that truly must survive: review states, dates, numbers, percentages, identifiers, or other explicit contracts.

Soft checks are useful but more brittle. When a semantic rubric is present, they remain advisory signals. If deterministic checks and the judge reach different conclusions, the run is marked with **mixed evaluator signals** for human review.

## Cost warning

Repeated evaluation multiplies API calls. Start with 3 runs while iterating. After the first comparison, an unchanged baseline is reused so later candidate iterations avoid repeating those baseline calls.

## Known limitations

- An LLM judge can itself be inconsistent or biased.
- Three repeated runs are a practical signal, not a statistical guarantee.
- Small suites can still give misleading confidence.
- The project currently uses a single model provider so the evaluation logic remains easy to inspect.

The tool is designed to make uncertainty visible, not to pretend prompt evaluation is deterministic.
