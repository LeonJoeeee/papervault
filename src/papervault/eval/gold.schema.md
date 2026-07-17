# gold.jsonl — the fixed ground-truth for the downstream eval

One JSON object per line, **25 entries** authored over the FIXED 100-paper corpus in
`experiments/test100.txt`. This file is the ground-truth half of the ruler; the runner's
results and the judge are both scored against it. `#`-prefixed lines and blank lines are
ignored (comments allowed). The canonical 25-question file is `gold.jsonl`, authored +
validated by `write_gold.py` (source data in `_gold_source.py`); run
`uv run python experiments/eval/write_gold.py --check` to re-validate it against test100.

## Entry schema

```json
{
  "qid": "gold_s0_q1",
  "intent": "<the user's natural-language question, exactly as it would be asked of KS>",
  "gold_keys": ["Reames2023", "Jokipii1966"],
  "nuggets": [
    "<atomic must-mention point 1>",
    "<atomic must-mention point 2>"
  ],
  "per_paper_relevance": {
    "Reames2023": "directly-answering",
    "Jokipii1966": "context",
    "SomeOther2019": "off"
  },
  "expected_coverage_band": "strong",
  "rationale": "<optional free-text note on why these keys/nuggets — authoring aid, not scored>"
}
```

| field                    | type            | meaning |
|--------------------------|-----------------|---------|
| `qid`                    | string, unique  | stable id. Used to PAIR baseline-vs-variant per question. |
| `intent`                 | string          | the exact NL question. If it names a context ("I do Voyager XPINN inversion"), the answer is expected to speak to it (judge metric 5). |
| `gold_keys`              | list[str]       | the test100 citation keys that TRULY answer the intent. **Empty list `[]` ⇒ this is a TRAP** (nothing in the corpus answers it; the correct answer is the empty sentinel). recall/gold-citation-recall are undefined on a trap and excluded from those averages. |
| `nuggets`                | list[str]       | 3–7 atomic, independently-checkable must-mention points (non-trap). The judge scores each ternary covered/partial/missing → nugget-recall (the headline quality metric). A trap has `[]`. |
| `per_paper_relevance`    | obj{key: label} | per-key label, `directly-answering` \| `context` \| `off`. Context for the judge; not a scored quantity itself. Every `gold_key` is `directly-answering`. |
| `expected_coverage_band` | enum            | `empty` \| `thin` \| `strong`. The coverage a healthy system SHOULD report. On a trap it MUST be `empty` (the kb_coverage guardrail). A prediction of KS's graph-signal kb_coverage, NOT a count of gold papers. |
| `rationale`              | string, optional| free-text authoring note (why these keys/nuggets). Ignored by all scoring. |

## Authoring rules (so the ruler stays sound)

- `gold_keys` MUST be a subset of the 100 keys in `test100.txt` (write_gold.py asserts this).
  A key outside the corpus can never be retrieved, so it would silently cap recall below 1.0
  for a perfect system. Every `per_paper_relevance` key must also be in test100.
- Include **trap questions**: `gold_keys=[]`, `nuggets=[]`, `expected_coverage_band="empty"`,
  and no `per_paper_relevance` value is `directly-answering`. They exercise the over-confidence
  guardrail and the judge's "correct refusal scores well" path. (gold.jsonl ships 4 traps.)
- Non-trap: `gold_keys != []`, every `directly-answering` key ∈ `gold_keys`, every `gold_key`
  is labelled `directly-answering`, and 3..7 nuggets.
- Nuggets are atomic and phrasing-independent: one fact/mechanism/number each, judged on
  meaning not wording. Avoid compound nuggets ("X and Y") — split them.
- The 25 questions are FIXED once authored. Re-authoring changes the ruler and invalidates
  cross-run comparisons (and the noise floor). Treat edits as a new ruler version.

A minimal valid fixture (1 trap + 2 answerable) lives in `gold.sample.jsonl` for tests and
smoke runs. The canonical 25-question `gold.jsonl` is authored via `write_gold.py`.
