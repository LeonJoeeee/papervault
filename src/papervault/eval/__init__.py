"""Downstream (query/answer) evaluation 'ruler' for KS v3.

The fixed yardstick used to judge upstream graph variants (and later the downstream itself):
a 6-metric vector reported together, absolute per-question over the FIXED 25 gold questions,
compared PAIRED vs a baseline (never pairwise A/B win-rate).

Modules:
  backbone.py        — deterministic backbone (recall@12/@5/hit@12 + references-recall +
                       rerank-cut gap, citation-integrity hallucinated/phantom, gold-citation-
                       recall, kb_coverage guardrail). Pure set-math, zero noise. CANONICAL
                       deterministic scorer (tests/test_eval_backbone.py).
  run_eval.py        — drives the REAL query path over gold questions → results/<tag>.jsonl
                       (l0_probe ONLY; prod-safe). Saves the backbone result shape.
  judge_prompt.md    — the EXACT contract for the Claude judge subagent (metrics 3/4/5).
  judge_aggregate.py — validate judge JSONs, average over 3 seeds, emit per-question scalars.
  stats.py           — paired-difference: bootstrap BCa CI + sign-flip permutation p +
                       noise-floor-from-test-retest. Consumes backbone (det.) + judge_aggregate.
  gold.jsonl         — the REAL fixed 25-question ground truth (authored via write_gold.py).
  gold.schema.md     — the gold.jsonl entry schema.
  gold.sample.jsonl  — a tiny 3-question fixture (1 trap) for tests + smoke runs.
  write_gold.py / _gold_source.py — author + validate gold.jsonl against test100.
"""
