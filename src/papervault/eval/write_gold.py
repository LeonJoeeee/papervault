"""Validate `_gold_source.GOLD` against the FIXED test100 corpus, then write gold.jsonl.

Run:  python -m papervault.eval.write_gold
      (add --check to validate WITHOUT writing — useful in CI/pre-commit)

Validation (loud failure on any violation — a bad gold file silently poisons every
downstream eval run, so we fail closed):
  - qids are unique
  - every gold_key ∈ test100
  - every per_paper_relevance key ∈ test100
  - per_paper_relevance values ∈ {directly-answering, context, off}
  - expected_coverage_band ∈ {empty, thin, strong}
  - trap (band=='empty') ⇒ gold_keys == [] AND nuggets == [] (and no per_paper_relevance
    key is 'directly-answering')
  - non-trap ⇒ gold_keys != [] AND every directly-answering key ∈ gold_keys AND
    every gold_key has per_paper_relevance == 'directly-answering' AND 3..7 nuggets

NOTE we deliberately do NOT couple expected_coverage_band to the number of gold papers.
The band is a prediction of KS's graph-signal kb_coverage (total_entities_found bins), NOT
a count of directly-answering papers — a single-paper question can legitimately yield
'strong' coverage if that one paper produces many graph entities (e.g. q-mishev-gle66-method).
The only band rule is the trap rule: band=='empty' iff the question is a trap.

Output: src/papervault/eval/gold.jsonl — one compact JSON object per line, fields in a
stable order, qids written in GOLD order (already deduped). Re-running overwrites it.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
TEST100 = EXPERIMENTS / "test100.txt"
OUT = HERE / "gold.jsonl"

_FIELD_ORDER = (
    "qid",
    "intent",
    "gold_keys",
    "nuggets",
    "per_paper_relevance",
    "expected_coverage_band",
    "rationale",
)
_BANDS = {"empty", "thin", "strong"}
_RELEVANCE = {"directly-answering", "context", "off"}


def load_test100() -> set[str]:
    keys = [ln.strip() for ln in TEST100.read_text().splitlines() if ln.strip()]
    if len(keys) != 100:
        raise SystemExit(f"test100.txt has {len(keys)} keys, expected 100")
    if len(set(keys)) != len(keys):
        raise SystemExit("test100.txt has duplicate keys")
    return set(keys)


def validate(gold: list[dict], test100: set[str]) -> None:
    errors: list[str] = []
    seen_qids: set[str] = set()

    for i, g in enumerate(gold):
        qid = g.get("qid", f"<index {i}>")
        tag = f"[{qid}]"

        # required fields present
        missing = [f for f in _FIELD_ORDER if f not in g]
        if missing:
            errors.append(f"{tag} missing fields: {missing}")
            continue

        if qid in seen_qids:
            errors.append(f"{tag} duplicate qid")
        seen_qids.add(qid)

        band = g["expected_coverage_band"]
        if band not in _BANDS:
            errors.append(f"{tag} bad band {band!r}")

        gold_keys = g["gold_keys"]
        nuggets = g["nuggets"]
        ppr = g["per_paper_relevance"]

        # membership in test100
        for k in gold_keys:
            if k not in test100:
                errors.append(f"{tag} gold_key {k!r} not in test100")
        for k, v in ppr.items():
            if k not in test100:
                errors.append(f"{tag} per_paper_relevance key {k!r} not in test100")
            if v not in _RELEVANCE:
                errors.append(f"{tag} per_paper_relevance[{k!r}]={v!r} invalid")

        # no dup gold keys
        if len(set(gold_keys)) != len(gold_keys):
            errors.append(f"{tag} duplicate gold_keys")

        da_keys = {k for k, v in ppr.items() if v == "directly-answering"}

        if band == "empty":
            # trap
            if gold_keys:
                errors.append(f"{tag} trap (band=empty) must have gold_keys==[]")
            if nuggets:
                errors.append(f"{tag} trap (band=empty) must have nuggets==[]")
            if da_keys:
                errors.append(f"{tag} trap must have NO directly-answering paper")
        else:
            if not gold_keys:
                errors.append(f"{tag} non-trap must have non-empty gold_keys")
            if not (3 <= len(nuggets) <= 7):
                errors.append(f"{tag} non-trap must have 3..7 nuggets (has {len(nuggets)})")
            # every directly-answering paper is a gold key, and vice versa
            if da_keys != set(gold_keys):
                errors.append(
                    f"{tag} directly-answering set {sorted(da_keys)} != gold_keys {sorted(gold_keys)}"
                )
            # (no band-vs-gold-count coupling — see module docstring NOTE)

    if errors:
        raise SystemExit("GOLD VALIDATION FAILED:\n  " + "\n  ".join(errors))


def write(gold: list[dict]) -> None:
    lines = []
    for g in gold:
        ordered = {k: g[k] for k in _FIELD_ORDER}
        lines.append(json.dumps(ordered, ensure_ascii=False))
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    from _gold_source import GOLD  # noqa: PLC0415 (run from this dir)

    test100 = load_test100()
    validate(GOLD, test100)
    check_only = "--check" in sys.argv
    if check_only:
        print(f"OK: {len(GOLD)} gold entries valid against test100 (no write).")
        return
    write(GOLD)
    n_trap = sum(1 for g in GOLD if g["expected_coverage_band"] == "empty")
    print(f"OK: wrote {len(GOLD)} entries ({n_trap} traps) -> {OUT}")


if __name__ == "__main__":
    sys.path.insert(0, str(HERE))
    main()
