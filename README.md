# Prompt Regression Tester

Catches prompt regressions before they hit production. You give it a set
of test inputs with pass/fail rules, run two versions of a system prompt
against them, and it tells you what broke between A and B.

Why bother: it's easy to "fix" a prompt for one case and quietly break
another one nobody's watching. This surfaces that in seconds instead of
after a user complains.

## Setup

```bash
pip install -r requirements.txt --break-system-packages
export ANTHROPIC_API_KEY="sk-ant-..."
streamlit run app.py
```

Opens at `http://localhost:8501`.

## How it works

Test cases live in `sample_test_cases.json`. Each one has an `input`
(what gets sent to the model) and a list of `assertions` the output has
to satisfy:

- `contains` / `not_contains` — substring checks
- `max_length` / `min_length` — character bounds
- `regex_match` — pattern match
- `json_valid` — output has to parse as JSON

Paste a baseline prompt and a candidate prompt into the two boxes, hit
run, and it flags any test that passed under the baseline but fails
under the candidate.

## Files

- `assertions.py` — the checks, no API calls, easy to unit test on its own
- `llm_client.py` — thin wrapper around the Anthropic API
- `runner.py` — glues test cases + client + assertions together
- `app.py` — the Streamlit UI
- `sample_test_cases.json` — example suite (API support ticket triage)

## Adding your own tests

```json
{
  "id": "tc_005",
  "input": "the exact input you'd send the model",
  "assertions": [
    { "type": "contains", "value": "required phrase" },
    { "type": "max_length", "value": 250 }
  ]
}
```

## Cost

Running the sample suite (4 cases x 2 prompts) costs a fraction of a
cent on Haiku. Fine to run over and over while you iterate.
