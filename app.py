from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import streamlit as st

from llm_client import DEFAULT_JUDGE_MODEL, DEFAULT_MODEL
from runner import compare_suites, run_test_suite

ROOT = Path(__file__).parent
SAMPLE_SUITE = ROOT / "sample_test_cases.json"
SOFT_DEFAULT_TYPES = {"contains", "not_contains", "regex_match"}

st.set_page_config(
    page_title="Prompt Regression Tester",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Keep the local app focused on the evaluation workflow rather than Streamlit chrome.
st.markdown(
    """
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        [data-testid="stToolbar"] {visibility: hidden;}
        [data-testid="stHeader"] {background: transparent;}
        .block-container {
            max-width: 1280px;
            padding-top: 2rem;
            padding-bottom: 4rem;
        }
        [data-testid="stMetric"] {
            border: 1px solid rgba(128, 128, 128, 0.22);
            border-radius: 12px;
            padding: 0.9rem 1rem;
        }
        div[data-testid="stExpander"] {
            border-radius: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Prompt Regression Tester")
st.caption(
    "Compare a trusted prompt with a proposed change across repeated evaluations, then separate real regressions from flaky outputs and mixed evaluator signals."
)


def load_sample_suite() -> list[dict[str, Any]]:
    with SAMPLE_SUITE.open(encoding="utf-8") as handle:
        return json.load(handle)


def assertion_severity(assertion: dict[str, Any]) -> str:
    explicit = assertion.get("severity")
    if explicit in {"hard", "soft"}:
        return explicit
    return "soft" if assertion.get("type") in SOFT_DEFAULT_TYPES else "hard"


def humanize(value: str) -> str:
    return value.replace("_", " ").strip().title()


if "test_cases" not in st.session_state:
    st.session_state.test_cases = load_sample_suite()

if "last_results" not in st.session_state:
    st.session_state.last_results = None

if "baseline_cache" not in st.session_state:
    st.session_state.baseline_cache = None

with st.sidebar:
    st.header("Run settings")
    has_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
    if has_key:
        st.success("API key connected")
    else:
        st.error("ANTHROPIC_API_KEY is not set")

    runs_per_case = st.select_slider(
        "Runs per case",
        options=[1, 3, 5],
        value=3,
        help="Repeated runs help separate a stable failure from normal model variance.",
    )

    with st.expander("Advanced settings", expanded=False):
        model = st.text_input("Generation model", value=DEFAULT_MODEL)
        judge_model = st.text_input("Judge model", value=DEFAULT_JUDGE_MODEL)

    st.divider()
    st.caption(
        "Rubrics add a judge call for every model run. Start with 3 runs while iterating to keep API usage modest."
    )

st.subheader("Evaluation suite")
st.caption(
    f"{len(st.session_state.test_cases)} cases loaded. Hard checks gate a run; soft signals can disagree with the semantic judge without creating a false regression."
)

suite_controls = st.columns([1, 1, 1])
with suite_controls[0]:
    uploaded = st.file_uploader("Import suite", type=["json"], label_visibility="collapsed")
    if uploaded is not None:
        try:
            imported = json.load(uploaded)
            if not isinstance(imported, list):
                raise ValueError("suite must be a JSON array")
            st.session_state.test_cases = imported
            st.session_state.last_results = None
            st.session_state.baseline_cache = None
            st.success(f"Imported {len(imported)} cases")
        except (json.JSONDecodeError, ValueError) as exc:
            st.error(f"Could not import suite: {exc}")

with suite_controls[1]:
    st.download_button(
        "Export suite",
        data=json.dumps(st.session_state.test_cases, indent=2),
        file_name="prompt-regression-suite.json",
        mime="application/json",
        use_container_width=True,
    )

with suite_controls[2]:
    if st.button("Reset default suite", use_container_width=True):
        st.session_state.test_cases = load_sample_suite()
        st.session_state.last_results = None
        st.session_state.baseline_cache = None
        st.rerun()

with st.expander("Add a test case", expanded=False):
    with st.form("add_case"):
        case_id = st.text_input("Case ID", placeholder="preserve_metric_caveat")
        case_input = st.text_area("Input", height=130, placeholder="Paste a real input this prompt handles.")

        c1, c2 = st.columns(2)
        with c1:
            contains = st.text_input("Must contain", placeholder="optional exact term")
            not_contains = st.text_input("Must not contain", placeholder="optional forbidden term")
            regex = st.text_input("Regex", placeholder="optional regex")
            exact_mode = st.radio(
                "Exact/regex checks",
                options=["Soft signal", "Hard requirement"],
                horizontal=True,
                help="Use hard only when exact wording, an identifier, number, or contract truly must be preserved.",
            )
        with c2:
            max_length = st.number_input("Max characters (0 = off)", min_value=0, value=0, step=50)
            require_json = st.checkbox("Must be valid JSON")

        rubric = st.text_area(
            "Semantic rubric",
            height=120,
            placeholder="What should an LLM judge verify that string checks cannot?",
        )

        add_clicked = st.form_submit_button("Add test case", type="primary")
        if add_clicked:
            if not case_id.strip() or not case_input.strip():
                st.error("Case ID and input are required.")
            elif any(case["id"] == case_id.strip() for case in st.session_state.test_cases):
                st.error("That case ID already exists.")
            else:
                assertions: list[dict[str, Any]] = []
                severity = "hard" if exact_mode == "Hard requirement" else "soft"
                if contains.strip():
                    assertions.append({"type": "contains", "value": contains.strip(), "severity": severity})
                if not_contains.strip():
                    assertions.append({"type": "not_contains", "value": not_contains.strip(), "severity": severity})
                if regex.strip():
                    assertions.append({"type": "regex_match", "value": regex.strip(), "severity": severity})
                if max_length:
                    assertions.append({"type": "max_length", "value": int(max_length), "severity": "hard"})
                if require_json:
                    assertions.append({"type": "json_valid", "severity": "hard"})

                st.session_state.test_cases.append(
                    {
                        "id": case_id.strip(),
                        "input": case_input.strip(),
                        "assertions": assertions,
                        "rubric": rubric.strip(),
                    }
                )
                st.session_state.last_results = None
                st.session_state.baseline_cache = None
                st.success("Test case added.")
                st.rerun()

with st.expander(f"View suite details ({len(st.session_state.test_cases)} cases)", expanded=False):
    for index, case in enumerate(st.session_state.test_cases):
        cols = st.columns([5, 1])
        with cols[0]:
            assertions = case.get("assertions", [])
            hard_count = sum(assertion_severity(a) == "hard" for a in assertions)
            soft_count = sum(assertion_severity(a) == "soft" for a in assertions)
            rubric_label = " · rubric" if case.get("rubric") else ""
            display_name = case.get('name') or humanize(case['id'])
            st.markdown(f"**{display_name}**  ·  {hard_count} hard  ·  {soft_count} soft{rubric_label}")
            st.caption(f"ID: {case['id']}")
            st.caption(case["input"][:220] + ("…" if len(case["input"]) > 220 else ""))
        with cols[1]:
            if st.button("Remove", key=f"remove_{index}", use_container_width=True):
                st.session_state.test_cases.pop(index)
                st.session_state.last_results = None
                st.session_state.baseline_cache = None
                st.rerun()
        if index < len(st.session_state.test_cases) - 1:
            st.divider()

st.divider()
st.subheader("Prompts")
st.caption("Keep the prompt you currently trust on the left and the proposed change on the right.")

baseline_default = (
      "You are revising EvidenceDesk review cards generated from a claim and supplied source material. "
    "Improve clarity and readability without changing the evidentiary meaning. Preserve the review state exactly—"
    "Supported, Potential mismatch, or Insufficient evidence—and preserve every number, percentage, date, model or "
    "product identifier, experimental condition, qualifier, causal limitation, conflict, and explicit uncertainty that "
    "affects verification. Never strengthen a claim beyond the supplied source, never substitute one metric for another, "
    "and never treat missing evidence as proof that a claim is false. If sources conflict, keep both sides visible. "
    "Return only the revised evidence card."
)

candidate_default = (
    "You are revising EvidenceDesk review cards for faster scanning in a compact interface. Preserve the review state, "
    "important numbers and percentages, evidence boundaries, causal caveats, source conflicts, and unsupported findings. "
    "Make the card concise and direct. To reduce metadata clutter, omit source dates and publication dates from the prose "
    "and describe chronology as earlier or later instead. Condense repeated setup details when the evidentiary meaning is "
    "unchanged. Never invent support or upgrade an unsupported claim. Return only the revised evidence card."
)

left, right = st.columns(2)
with left:
    st.markdown("#### Baseline")
    baseline_prompt = st.text_area("baseline", value=baseline_default, height=220, label_visibility="collapsed")
with right:
    st.markdown("#### Candidate")
    candidate_prompt = st.text_area("candidate", value=candidate_default, height=220, label_visibility="collapsed")

run_clicked = st.button("Run comparison", type="primary", use_container_width=True)

if run_clicked:
    if not has_key:
        st.error("Set ANTHROPIC_API_KEY, then reload the app.")
        st.stop()
    if not st.session_state.test_cases:
        st.error("Add at least one test case.")
        st.stop()

    suite_snapshot = json.dumps(st.session_state.test_cases, sort_keys=True, separators=(",", ":"))
    cache_material = "\n".join(
        [baseline_prompt, suite_snapshot, str(runs_per_case), model, judge_model]
    )
    baseline_cache_key = hashlib.sha256(cache_material.encode("utf-8")).hexdigest()
    cached_baseline = st.session_state.baseline_cache

    total_attempts = len(st.session_state.test_cases) * runs_per_case
    case_names_for_progress = {case["id"]: case.get("name") or humanize(case["id"]) for case in st.session_state.test_cases}
    progress = st.progress(0, text="Preparing evaluation…")

    if cached_baseline and cached_baseline.get("key") == baseline_cache_key:
        baseline_results = cached_baseline["results"]
        progress.progress(0, text="Baseline unchanged — reusing the previous baseline evaluation.")
        baseline_reused = True
    else:
        baseline_reused = False
        completed_baseline = {"count": 0}

        def baseline_progress(case_id: str, run_number: int, case_runs: int) -> None:
            completed_baseline["count"] += 1
            count = completed_baseline["count"]
            fraction = count / max(total_attempts, 1)
            progress.progress(
                fraction * 0.5,
                text=f"Evaluating baseline · {count}/{total_attempts} runs · {case_names_for_progress.get(case_id, humanize(case_id))}",
            )

        baseline_results = run_test_suite(
            baseline_prompt,
            st.session_state.test_cases,
            runs=runs_per_case,
            model=model,
            judge_model=judge_model,
            progress_callback=baseline_progress,
        )
        st.session_state.baseline_cache = {
            "key": baseline_cache_key,
            "results": baseline_results,
        }

    completed_candidate = {"count": 0}

    def candidate_progress(case_id: str, run_number: int, case_runs: int) -> None:
        completed_candidate["count"] += 1
        count = completed_candidate["count"]
        if baseline_reused:
            fraction = count / max(total_attempts, 1)
        else:
            fraction = 0.5 + (count / max(total_attempts, 1)) * 0.5
        progress.progress(
            min(fraction, 1.0),
            text=f"Evaluating candidate · {count}/{total_attempts} runs · {case_names_for_progress.get(case_id, humanize(case_id))}",
        )

    candidate_results = run_test_suite(
        candidate_prompt,
        st.session_state.test_cases,
        runs=runs_per_case,
        model=model,
        judge_model=judge_model,
        progress_callback=candidate_progress,
    )

    progress.progress(1.0, text="Evaluation complete.")
    comparison = compare_suites(baseline_results, candidate_results)
    st.session_state.last_results = {
        "baseline": baseline_results,
        "candidate": candidate_results,
        "comparison": comparison,
        "baseline_reused": baseline_reused,
    }

results = st.session_state.last_results
if results:
    comparison = results["comparison"]
    baseline_results = results["baseline"]
    candidate_results = results["candidate"]

    st.divider()
    st.subheader("Results")
    if results.get("baseline_reused"):
        st.caption("Baseline evaluation was reused because the trusted prompt, suite, models, and run count were unchanged.")

    if comparison["regressions"]:
        st.error(
            f"{comparison['regressions']} regression(s) detected. Open the flagged cases below to inspect what changed."
        )
    elif comparison["flaky_drops"] or comparison.get("candidate_disagreements", 0):
        st.warning(
            "No stable regression was detected, but there are noisy or disputed cases worth reviewing before you ship the prompt change."
        )
    else:
        st.success("No regressions detected across this evaluation suite.")

    metrics = st.columns(4)
    metrics[0].metric("Baseline pass rate", f"{comparison['baseline_pass_rate']:.0%}")
    metrics[1].metric(
        "Candidate pass rate",
        f"{comparison['candidate_pass_rate']:.0%}",
        f"{comparison['delta']:+.0%}",
    )
    metrics[2].metric("Regressions", comparison["regressions"])
    metrics[3].metric("Flaky drops", comparison["flaky_drops"])

    st.caption(
        f"Other signals: {comparison['improvements']} improvement(s) · "
        f"{comparison.get('candidate_disagreements', 0)} judge disagreement(s)."
    )

    case_names = {case["id"]: case.get("name") or humanize(case["id"]) for case in st.session_state.test_cases}

    display_rows = []
    for row in comparison["rows"]:
        display_rows.append(
            {
                "Case": case_names.get(row["id"], humanize(row["id"])),
                "Baseline": f"{row['baseline_pass_rate']:.0%} ({humanize(row['baseline_stability'])})",
                "Candidate": f"{row['candidate_pass_rate']:.0%} ({humanize(row['candidate_stability'])})",
                "Change": f"{row['delta']:+.0%}",
                "Judge disagreements": f"{row.get('baseline_disagreements', 0)} → {row.get('candidate_disagreements', 0)}",
                "Verdict": humanize(row["classification"]),
            }
        )
    st.dataframe(display_rows, use_container_width=True, hide_index=True)

    st.caption(
        "🔴 Regression · 🟠 Flaky drop · 🟡 Mixed/noisy · 🟢 Improvement · ⚪ Stable"
    )

    for row in comparison["rows"]:
        case_id = row["id"]
        icon = {
            "regression": "🔴",
            "flaky_drop": "🟠",
            "improvement": "🟢",
            "small_improvement": "🟢",
            "flaky_no_change": "🟡",
            "stable": "⚪",
        }.get(row["classification"], "⚪")

        with st.expander(f"{icon} {case_names.get(case_id, humanize(case_id))} — {humanize(row['classification'])}"):
            bcol, ccol = st.columns(2)
            for label, col, case_results in [
                ("Baseline", bcol, baseline_results[case_id]),
                ("Candidate", ccol, candidate_results[case_id]),
            ]:
                with col:
                    st.markdown(
                        f"**{label}: {case_results['pass_count']}/{case_results['total_runs']} passed · "
                        f"{humanize(case_results['stability'])} · "
                        f"{case_results.get('disagreement_count', 0)} disagreement(s)**"
                    )
                    for attempt in case_results["runs"]:
                        state = "PASS" if attempt["passed"] else "FAIL"
                        disagreement = " · MIXED EVALUATOR SIGNALS" if attempt.get("evaluator_disagreement") else ""
                        st.markdown(f"**Run {attempt['run']} · {state}{disagreement}**")
                        st.code(attempt["output"], language=None)
                        failed_checks = [
                            check for check in attempt["assertion_results"] if not check["passed"]
                        ]
                        for check in failed_checks:
                            prefix = "Hard check failed" if check.get("severity") == "hard" else "Soft signal missed"
                            st.caption(f"{prefix}: {check['type']} — {check['detail']}")
                        if attempt.get("judge"):
                            judge = attempt["judge"]
                            st.caption(
                                f"Judge: {judge.get('score', 0)}/5 · "
                                f"{'pass' if judge.get('passed') else 'fail'} — {judge.get('reason', '')}"
                            )
                        if attempt.get("evaluator_disagreement"):
                            st.warning(
                                "Mixed evaluator signals: a deterministic check and the semantic judge reached different conclusions. Hard requirements still gate the run."
                            )

    export_payload = {
        "comparison": comparison,
        "baseline": baseline_results,
        "candidate": candidate_results,
    }
    st.download_button(
        "Export run report",
        data=json.dumps(export_payload, indent=2),
        file_name="prompt-regression-report.json",
        mime="application/json",
        use_container_width=True,
    )
else:
    st.divider()
    st.info("Run a comparison to see pass-rate changes, regressions, flaky cases, and mixed evaluator signals here.")
