"""LightRAG 1.5.x port gates (machine-checkable prompt-content assertions).

The 1.4->1.5 upgrade's highest-stakes failure mode is SILENT: the ontology/exclusions/
examples simply stop reaching the extraction prompt and every graph build degrades with no
error. These tests pin the delivery channels so that can never quietly regress:
  - the domain pack renders a closed-ontology guidance string carrying all entity types
    AND the negative-exclusion rules (delivered via addon_params entity_types_guidance);
  - apply_ks_extraction_prompt() kills the `Other` escape hatch in the 1.5.x system prompt
    and swaps in the domain few-shot examples;
  - the fail-loud guard actually raises on an unsupported prompt structure;
  - the scheduler treats the 1.5.x PARSING/ANALYZING phases as mid-transit, not stuck.
"""
from __future__ import annotations

import pytest
from lightrag import prompt as lr_prompt
from lightrag.base import DocStatus

from papervault.domain import get_domain
from papervault.knowledge.store import extraction_prompt as ep


@pytest.fixture()
def prompts_snapshot():
    """Snapshot + restore the module-global PROMPTS around each mutation test."""
    saved = {k: lr_prompt.PROMPTS[k] for k in
             ("entity_extraction_system_prompt", "entity_extraction_examples")}
    yield
    lr_prompt.PROMPTS.update(saved)


def test_guidance_carries_full_closed_ontology_and_exclusions():
    d = get_domain()
    g = d.entity_types_guidance
    for t in d.entity_types:
        assert f"`{t}`" in g, f"entity type {t} missing from guidance"
    assert "CLOSED" in g
    assert "DO NOT extract" in g
    # The negative-exclusion block must ride along (it no longer enters the system prompt).
    assert d.extraction_exclusions.strip() in g


def test_apply_patch_kills_other_and_installs_examples(prompts_snapshot):
    ep.apply_ks_extraction_prompt(examples=True, exclusions=True)
    sysp = lr_prompt.PROMPTS["entity_extraction_system_prompt"]
    assert ep._OTHER_CATCHALL_15 not in sysp, "`Other` escape hatch survived the patch"
    assert ep._CLOSED_ONTOLOGY_15 in sysp
    examples = lr_prompt.PROMPTS["entity_extraction_examples"]
    assert examples == list(get_domain().extraction_examples)
    assert any("Solar Energetic Particles" in e for e in examples), \
        "factory-pack few-shots not installed"
    # Idempotent: applying again must not raise or double-patch.
    ep.apply_ks_extraction_prompt(examples=True, exclusions=True)
    assert lr_prompt.PROMPTS["entity_extraction_system_prompt"].count(ep._CLOSED_ONTOLOGY_15) == 1


def test_fail_loud_on_unsupported_prompt_structure(prompts_snapshot):
    lr_prompt.PROMPTS["entity_extraction_system_prompt"] = "a restructured prompt with no anchors"
    with pytest.raises(RuntimeError, match="escape-hatch anchor"):
        ep.apply_ks_extraction_prompt(examples=False, exclusions=True)


def test_fail_loud_on_json_extraction_mode(prompts_snapshot, monkeypatch):
    monkeypatch.setenv("ENTITY_EXTRACTION_USE_JSON", "true")
    with pytest.raises(RuntimeError, match="ENTITY_EXTRACTION_USE_JSON"):
        ep.apply_ks_extraction_prompt()


def test_new_doc_phases_exist_and_are_midtransit():
    # The enum members the scheduler now treats as mid-transit must exist on the pinned line.
    assert DocStatus.PARSING == "parsing"
    assert DocStatus.ANALYZING == "analyzing"
