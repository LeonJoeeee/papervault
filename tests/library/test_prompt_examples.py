"""Issue 320: model-facing examples render and obey their consumer's JSON contracts."""
import json
import re
import asyncio
import importlib.util
from dataclasses import replace

import pytest

from papervault import domain
from papervault.knowledge.query import multiquery
from papervault.library import download, extract
from papervault.library.services import intent_parser, judge


def _outputs(prompt):
    """Read complete worked outputs, excluding illustrative schema placeholders."""
    decoder = json.JSONDecoder()
    return [decoder.raw_decode(prompt[match.end():].lstrip())[0]
            for match in re.finditer(r"(?:^|\n)Output:\s*", prompt)]


def test_intent_worked_plans_all_supply_canonical_kg_concepts():
    prompt = intent_parser._SYSTEM_PROMPT
    outputs = _outputs(prompt)
    # Retain every existing worked intent, including the four added after the issue was drafted.
    assert len(outputs) == 13
    for output in outputs:
        assert output["kg_concepts"]
        assert all(isinstance(concept, str) and concept.strip() for concept in output["kg_concepts"])
        plan = intent_parser._parse_response("example research intent", json.dumps(output))
        assert plan["kg_concepts"] == output["kg_concepts"]
    assert any(output["kg_concepts"] != output["search_terms"] for output in outputs)


def test_space_computing_intent_example_preserves_anchor_facets_and_no_inferred_year():
    output = _outputs(intent_parser._SYSTEM_PROMPT)[-1]
    plan = intent_parser._parse_response("on-board AI inference", json.dumps(output))
    assert plan["search_terms"][0] == "on-board AI inference"
    assert "radiation-tolerant FPGA" in plan["search_terms"]
    assert "quantized neural inference" in plan["kg_concepts"]
    assert plan["filters"]["year_min"] is None
    assert plan["filters"]["year_max"] is None
    for axis in ("phenomenon", "method", "system", "regime"):
        assert re.search(rf"\* {axis} —", intent_parser._SYSTEM_PROMPT)


def test_ingest_examples_render_domain_and_cover_admission_boundaries():
    rendered = judge._INGEST_SYSTEM.format(domain_md="Configured domain rubric")
    assert "Configured domain rubric" in rendered
    outputs = _outputs(rendered)
    assert len(outputs) == 4
    verdicts = [item for output in outputs for item in output["items"]]
    assert any(item["tier"] == "1A" and item["is_paper"] is True for item in verdicts)
    assert any(item["tier"] == "1B" and item["is_paper"] is True for item in verdicts)
    assert sum(item["tier"] == "1C" and item["is_paper"] is True for item in verdicts) == 2
    assert any(item["tier"] == "3" for item in verdicts)
    assert any(item["is_paper"] is False for item in verdicts)
    for output in outputs:
        parsed = judge._parse_ingest(json.dumps(output), {item["i"] for item in output["items"]})
        assert len(parsed) == len(output["items"])


def test_return_calibration_examples_survive_score_parser():
    outputs = _outputs(judge._RETURN_SYSTEM)
    assert len(outputs) == 4
    scores = [judge._parse_score(json.dumps(output), {0})[0]["score"] for output in outputs]
    assert scores == [0.95, 0.76, 0.45, 0.10]


def test_download_examples_have_boolean_identity_verdicts():
    outputs = _outputs(download._VERIFY_PROMPT)
    assert [output["match"] for output in outputs] == [True, False, False]
    assert all(isinstance(output["match"], bool) and output["reason"] for output in outputs)


def test_extract_readability_examples_allow_chunk_cuts_and_latex():
    outputs = _outputs(extract._REVIEW_PROMPT)
    assert [output["ok"] for output in outputs] == [True, False, True]
    for output in outputs:
        assert isinstance(output["issues"], list)
        assert 0 <= output["confidence"] <= 1
        assert bool(output["issues"]) is not output["ok"]


def test_completeness_examples_distinguish_short_whole_from_stubs_and_cuts():
    outputs = _outputs(extract._GATE_PROMPT)
    assert [output["complete"] for output in outputs] == [True, False, False]
    assert all(output["reason"] for output in outputs)


def test_standard_decomposition_examples_render_parseable_keyword_arrays():
    rendered = multiquery._STD_DECOMPOSE_PROMPT.format(intent="SEP transport", n=5)
    outputs = _outputs(rendered)
    assert len(outputs) == 2
    assert len(outputs[0]) == 2  # Narrow intent must not become synonym shards.
    assert 3 <= len(outputs[1]) <= 5
    for output in outputs:
        assert multiquery._parse_facets(json.dumps(output), 5) == output
        for facet in output:
            assert isinstance(facet["query"], str) and facet["query"]
            assert 2 <= len(facet["hl"]) <= 4
            assert 2 <= len(facet["ll"]) <= 6


def test_fanout_calibration_includes_middle_source_budgets():
    rendered = multiquery._FANOUT_PROMPT.format(intent="SEP transport")
    estimates = [int(value) for value in re.findall(r'"\s*->\s*(\d+)', rendered)]
    assert {1, 3, 5, 6, 12} <= set(estimates)


@pytest.mark.parametrize("label", [
    "space physics, space computing, and AI4Science", "planetary robotics test domain",
])
def test_fanout_sends_configured_domain_label_to_model(monkeypatch, label):
    configured = replace(domain.get_domain(), label=label)
    monkeypatch.setattr(domain, "get_domain", lambda: configured)
    # Load a separate module instance: import-time configuration must work without
    # disturbing the production singleton used by the rest of this test session.
    spec = importlib.util.spec_from_file_location("configured_multiquery", multiquery.__file__)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    async def complete(prompt, *, system_prompt, **kwargs):
        assert label in system_prompt
        assert "{get_domain().label}" not in system_prompt
        assert "Compare orbital edge computing architectures" in prompt
        return "6"

    monkeypatch.setattr(module, "mimo_complete", complete)
    assert asyncio.run(module._estimate_n_sources("Compare orbital edge computing architectures")) == 6


@pytest.mark.parametrize("topic", [
    "orbital edge computing", "on-board AI", "radiation-tolerant", "satellite computing constellations",
])
def test_fanout_space_computing_examples_get_in_domain_budgets(topic):
    rendered = multiquery._FANOUT_PROMPT.format(intent="space computing")
    examples = re.findall(r'"([^"]+)"\s*->\s*(\d+)', rendered)
    assert any(topic in query and 2 <= int(estimate) <= 15 for query, estimate in examples)
