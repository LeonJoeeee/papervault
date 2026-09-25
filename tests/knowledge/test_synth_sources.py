"""Source labels must expose the paper keys already accepted by citation aggregation."""

import re

import pytest

from papervault.knowledge.query.synth import _SYNTH_SYSTEM, _build_prompt, _source_label

# Wording that told the model NOT to bracket an operator source (or never to bracket a colon),
# contradicting the one rule of issue #122: operator sources are bracketed with their full key.
_CONTRADICTING = (
    "inline in prose", "not in square brackets", "colon inside", "PAPER keys only",
    "do NOT put it in square brackets",
)


@pytest.mark.parametrize("path, expected", [
    ("paper/An2017", ("An2017", "empirical")),
    ("An2017", ("An2017", "empirical")),
    ("Suvorovand", ("Suvorovand", "empirical")),
    ("textbook:Griffiths", ("textbook:Griffiths", "established")),
    ("notebook:idea", ("notebook:idea", "preliminary")),
    ("web:nasa", ("web:nasa", "preliminary")),
    ("", ("unknown", "preliminary")),
    ("unknown_source", ("unknown", "preliminary")),
    ("unknown", ("unknown", "preliminary")),
    ("   ", ("unknown", "preliminary")),
    ("bogus:x", ("unknown", "preliminary")),
    ("paper/", ("unknown", "preliminary")),
])
def test_source_label_provenance_shapes(path, expected):
    assert _source_label(path) == expected


def test_prompt_exposes_bare_paper_key_and_marks_residue_unciteable():
    prompt = _build_prompt({"chunks": [
        {"file_path": "An2017", "content": "A measured storm response."},
        {"file_path": "", "content": "Unattributed material."},
    ]}, "Explain the storm response.")
    assert "(source key: An2017 | credibility: empirical)" in prompt
    assert "source key: unknown" not in prompt
    assert "Unattributed material." in prompt
    assert "no citeable source key" in prompt


def test_prompt_carries_one_bracketed_operator_source_rule():
    """#122: a textbook/notebook/web source is cited as [textbook:Key], and nothing says otherwise."""
    prompt = _build_prompt({"chunks": [
        {"file_path": "textbook:Griffiths#s3", "content": "Plasma frequency."},
        {"file_path": "Reames2023", "content": "SEP onset."},
    ]}, "Explain the plasma frequency.")
    assert "(source key: textbook:Griffiths | credibility: established)" in prompt
    operator_rules = re.findall(r"\[(?:textbook|notebook|web):[^\]\s]+\]", prompt)
    assert operator_rules == ["[textbook:Baumjohann2012]"]
    for text in (_SYNTH_SYSTEM, prompt):
        for phrase in _CONTRADICTING:
            assert phrase not in text, phrase


def test_system_prompt_cites_operator_sources_with_the_same_bracket_form():
    assert "[textbook:" in _SYNTH_SYSTEM
