import json
import os

import streamlit as st

from runner import run_test_suite

st.set_page_config(page_title="Prompt Regression Tester", layout="wide")

st.title("Prompt Regression Tester")
st.caption("Run two prompt versions against the same test cases and see what broke.")

with st.sidebar:
    st.header("Setup")
    has_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
    if has_key:
        st.success("API key found")
    else:
        st.error("Set ANTHROPIC_API_KEY before running")
    st.caption("Using claude-haiku-4-5-20251001")
    st.divider()
    st.caption("Test cases come from sample_test_cases.json — edit that file to test your own prompts.")

test_cases_path = os.path.join(os.path.dirname(__file__), "sample_test_cases.json")
with open(test_cases_path) as f:
    test_cases = json.load(f)

with st.expander(f"{len(test_cases)} test cases loaded"):
    for case in test_cases:
        st.markdown(f"**{case['id']}** — {case['input']}")
        st.caption(str(case["assertions"]))

left, right = st.columns(2)
with left:
    st.subheader("Prompt A (baseline)")
    prompt_a = st.text_area(
        "prompt_a",
        height=180,
        value=(
            "You triage incoming API support tickets for a developer platform. "
            "Given a ticket, respond with ONLY valid JSON: "
            "{\"category\": string, \"priority\": \"low\"|\"medium\"|\"high\", "
            "\"suggested_reply\": string}. If the ticket lacks enough detail to "
            "diagnose, set category to \"needs_more_info\" and ask a specific "
            "clarifying question in suggested_reply."
        ),
        label_visibility="collapsed",
    )
with right:
    st.subheader("Prompt B (candidate)")
    prompt_b = st.text_area(
        "prompt_b",
        height=180,
        value=(
            "You triage incoming API support tickets for a developer platform. "
            "Given a ticket, respond with JSON containing category, priority, "
            "and suggested_reply. Always give a direct suggested_reply even if "
            "the ticket is vague — do not ask clarifying questions."
        ),
        label_visibility="collapsed",
    )

run_clicked = st.button("Run both prompts", type="primary", use_container_width=True)

if run_clicked:
    if not has_key:
        st.error("No API key set — export ANTHROPIC_API_KEY and reload.")
        st.stop()

    with st.spinner("Running Prompt A..."):
        results_a = run_test_suite(prompt_a, test_cases)
    with st.spinner("Running Prompt B..."):
        results_b = run_test_suite(prompt_b, test_cases)

    st.divider()
    st.subheader("Results")

    # a regression is a case that used to pass and now doesn't — that's the
    # only thing worth flagging loudly, everything else is just normal output
    regressions = [
        case["id"] for case in test_cases
        if results_a[case["id"]]["passed"] and not results_b[case["id"]]["passed"]
    ]

    if regressions:
        st.error(f"{len(regressions)} regression(s): {', '.join(regressions)}")
    else:
        st.success("Nothing regressed — B passes everything A passed.")

    for case in test_cases:
        cid = case["id"]
        ra, rb = results_a[cid], results_b[cid]
        flagged = cid in regressions

        with st.expander(("[REGRESSION] " if flagged else "") + cid, expanded=flagged):
            st.caption(case["input"])
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"**A** — {'pass' if ra['passed'] else 'fail'}")
                st.write(ra["output"])
                for r in ra["assertion_results"]:
                    if not r.passed:
                        st.caption(f"failed: {r.type} — {r.detail}")
            with c2:
                st.markdown(f"**B** — {'pass' if rb['passed'] else 'fail'}")
                st.write(rb["output"])
                for r in rb["assertion_results"]:
                    if not r.passed:
                        st.caption(f"failed: {r.type} — {r.detail}")
