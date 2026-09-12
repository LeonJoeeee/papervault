"""Source labels must expose the paper keys already accepted by citation aggregation."""

import pytest

from papervault.knowledge.query.synth import _build_prompt, _source_label


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
