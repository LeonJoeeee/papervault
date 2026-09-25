"""Offline unit tests for the operator CLI (SDD §6.8). No LightRAG / no DB / no LLM.

The CLI is the v1 operator surface (MCP exposes only query(intent)). These tests pin the
v3 wiring after the stale-v2 rewire:
  - `ks query`  → query.aquery.query (NOT the dead query.pipeline.answer)
  - `ks stats`  → ks_ledger by-status + lightrag_doc_status by-workspace, with the §6.5
                  step5 done+done_meta roll-up; workspace resolved via assert_safe_workspace
  - `ks get`    → rag.aget_docs_by_ids single-row by-id; exit 1 when absent

Everything the CLI would touch (query pipeline, ledger PG, doc_status PG, get_graph) is
monkeypatched, so nothing opens a real pool or spins up LightRAG.
"""
from __future__ import annotations

import json

from click.testing import CliRunner

import papervault.knowledge.cli as cli_mod


def _runner() -> CliRunner:
    return CliRunner()


# ---- query ------------------------------------------------------------------

def test_query_wires_to_aquery_and_renders(monkeypatch):
    captured = {}

    async def _fake_query(intent: str):
        captured["intent"] = intent
        return {
            "answer": "Prose with [Reames2023].",
            "cited_papers": ["Reames2023", "Jokipii1966"],
            "kb_coverage": "strong",
        }

    monkeypatch.setattr("papervault.knowledge.query.aquery.query", _fake_query)
    monkeypatch.setattr("papervault.knowledge.store.graph.close_graph", _noop)
    monkeypatch.setattr("papervault.knowledge.ledger.store.close_pool", _noop)

    res = _runner().invoke(cli_mod.cli, ["query", "what is SEP transport"])
    assert res.exit_code == 0, res.output
    assert captured["intent"] == "what is SEP transport"
    assert "strong" in res.output
    assert "Reames2023" in res.output
    assert "Jokipii1966" in res.output


def test_query_json_out(monkeypatch):
    async def _fake_query(intent: str):
        return {"answer": "a", "cited_papers": ["K1"], "kb_coverage": "thin"}

    monkeypatch.setattr("papervault.knowledge.query.aquery.query", _fake_query)
    monkeypatch.setattr("papervault.knowledge.store.graph.close_graph", _noop)
    monkeypatch.setattr("papervault.knowledge.ledger.store.close_pool", _noop)

    res = _runner().invoke(cli_mod.cli, ["query", "x", "--json"])
    assert res.exit_code == 0, res.output
    payload = json.loads(res.output)
    assert payload == {"answer": "a", "cited_papers": ["K1"], "kb_coverage": "thin"}


# ---- stats ------------------------------------------------------------------

def test_stats_aggregates_ledger_and_doc_status(monkeypatch):
    async def _fake_count_by_status(source: str):
        assert source == "paper"
        return {"done": 3000, "done_meta": 691, "done_abstract": 40, "processing": 5, "error": 2}

    async def _fake_count_doc_status(workspace: str):
        assert workspace == "l0_probe"
        return {"processed": 3000, "processing": 5, "failed": 2}

    monkeypatch.setattr(
        "papervault.knowledge.store.graph.assert_safe_workspace", lambda: "l0_probe"
    )
    monkeypatch.setattr("papervault.knowledge.ledger.store.count_by_status", _fake_count_by_status)
    monkeypatch.setattr("papervault.knowledge.ledger.store.close_pool", _noop)
    monkeypatch.setattr(cli_mod, "_count_doc_status", _fake_count_doc_status)

    res = _runner().invoke(cli_mod.cli, ["stats", "--json"])
    assert res.exit_code == 0, res.output
    payload = json.loads(res.output)
    assert payload["workspace"] == "l0_probe"
    assert payload["ledger"]["total"] == 3000 + 691 + 40 + 5 + 2
    assert payload["ledger"]["done_plus_done_meta"] == 3000 + 691
    # §6.5 step5: every terminal success class (#144 adds done_abstract) is the number vs |idx|.
    assert payload["ledger"]["done_terminal"] == 3000 + 691 + 40
    assert payload["doc_status"]["by_status"] == {
        "processed": 3000, "processing": 5, "failed": 2
    }


def test_stats_gate_refusal_surfaces(monkeypatch):
    """If assert_safe_workspace refuses (prod 'l0' w/o opt-in), the CLI fails loudly,
    never silently reading the wrong workspace."""
    from papervault.knowledge.store.graph import UnsafeWorkspaceError

    def _refuse():
        raise UnsafeWorkspaceError("refusing prod l0")

    monkeypatch.setattr("papervault.knowledge.store.graph.assert_safe_workspace", _refuse)
    monkeypatch.setattr("papervault.knowledge.ledger.store.close_pool", _noop)

    res = _runner().invoke(cli_mod.cli, ["stats"])
    assert res.exit_code != 0
    assert isinstance(res.exception, UnsafeWorkspaceError)


# ---- get --------------------------------------------------------------------

def _fake_status() -> dict:
    # Mirror the real LightRAG 1.4.16 RUNTIME shape: aget_docs_by_ids returns a plain dict per
    # doc (PGDocStatusStorage/JsonDocStatusStorage get_by_id both return dicts), `status` a bare
    # string — NOT a DocProcessingStatus object. An object-shaped fake here used to mask the
    # getattr-on-dict all-null bug in `ks get` (SDD §6.6 ★ / §6.8).
    return {
        "status": "processed",
        "content_summary": "summary...",
        "content_length": 12345,
        "chunks_count": 7,
        "file_path": "paper/Reames2023",
        "track_id": "tk-1",
        "error_msg": None,
        "created_at": "2026-06-01T00:00:00",
        "updated_at": "2026-06-01T00:01:00",
    }


class _FakeRag:
    def __init__(self, found: dict):
        self._found = found

    async def aget_docs_by_ids(self, ids):
        return {i: self._found[i] for i in ids if i in self._found}


def test_get_found(monkeypatch):
    rag = _FakeRag({"paper:Reames2023": _fake_status()})

    async def _fake_get_graph():
        return rag

    monkeypatch.setattr("papervault.knowledge.store.graph.get_graph", _fake_get_graph)
    monkeypatch.setattr("papervault.knowledge.store.graph.close_graph", _noop)
    monkeypatch.setattr("papervault.knowledge.ledger.store.close_pool", _noop)

    res = _runner().invoke(cli_mod.cli, ["get", "paper:Reames2023"])
    assert res.exit_code == 0, res.output
    payload = json.loads(res.output)
    assert payload["doc_id"] == "paper:Reames2023"
    assert payload["status"] == "processed"   # enum .value unwrapped
    assert payload["chunks_count"] == 7
    assert payload["file_path"] == "paper/Reames2023"


def test_get_not_found_exits_1(monkeypatch):
    rag = _FakeRag({})  # nothing

    async def _fake_get_graph():
        return rag

    monkeypatch.setattr("papervault.knowledge.store.graph.get_graph", _fake_get_graph)
    monkeypatch.setattr("papervault.knowledge.store.graph.close_graph", _noop)
    monkeypatch.setattr("papervault.knowledge.ledger.store.close_pool", _noop)

    res = _runner().invoke(cli_mod.cli, ["get", "paper:Nope"])
    assert res.exit_code == 1
    assert "Not found" in res.output


# ---- helper -----------------------------------------------------------------

async def _noop():
    return None
