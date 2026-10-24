"""Optional, read-only vector and graph candidate retrieval for library search.

Use the installed LightRAG 1.5.7 schema directly: constructing its storage adapters
would initialize schemas and indexes. Returned identities always belong to the
caller's library snapshot; database paths alone cannot introduce catalog records.
"""
from __future__ import annotations

import asyncio
from collections import defaultdict
import logging
import math
import os
import threading

from .registration import serving_excluded

log = logging.getLogger(__name__)
_ENCODE_LOCK = threading.Lock()
_TOP_K = 60
_EDGE_LIMIT = 1000
_TIMEOUT = 5


async def _embedding_permit():
    from papervault.knowledge.store.lightrag_init import _get_embed_sem

    semaphore = _get_embed_sem()
    await asyncio.wait_for(semaphore.acquire(), timeout=_TIMEOUT)
    return semaphore


def _configuration():
    enabled = os.getenv("PAPERVAULT_SEARCH_KG")
    if enabled != "1" and (enabled is not None
                           or os.getenv("PAPERVAULT_STORAGE_BACKEND") != "postgres"):
        return None
    workspace = os.getenv("POSTGRES_WORKSPACE", "").strip()
    database = os.getenv("POSTGRES_DATABASE") or os.getenv("POSTGRES_DB")
    neo_workspace = os.getenv("NEO4J_WORKSPACE", "").strip()
    required = (workspace, database, os.getenv("POSTGRES_USER"),
                os.getenv("POSTGRES_PASSWORD"), os.getenv("NEO4J_URI"),
                os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
    if not all(required) or (neo_workspace and neo_workspace != workspace):
        return None
    return workspace, {
        "host": os.getenv("POSTGRES_HOST", "localhost"),
        "port": int(os.getenv("POSTGRES_PORT", "5432")),
        "user": os.environ["POSTGRES_USER"], "password": os.environ["POSTGRES_PASSWORD"],
        "dbname": database, "connect_timeout": _TIMEOUT,
        "options": "-c default_transaction_read_only=on -c statement_timeout=5000",
    }


def _canonical_keys(library):
    papers = {paper.key: paper for paper in library.all_papers(include_quarantined=True)}
    canonical = {}
    for key, paper in papers.items():
        seen = set()
        while paper is not None and paper.acquisition_state.alias_key and paper.key not in seen:
            seen.add(paper.key)
            paper = papers.get(paper.acquisition_state.alias_key)
        if paper is not None and paper.key not in seen and not serving_excluded(paper):
            canonical[key] = paper.key
    return canonical


def _path_keys(path, canonical):
    keys = []
    for item in (path or "").split("<SEP>"):
        item = item.strip()
        key = canonical.get(item) or canonical.get(item.removeprefix("paper/"))
        if key and key not in keys:
            keys.append(key)
    return keys


def _paper_ranklist(rows, canonical):
    return list(dict.fromkeys(key for row in rows
                              for key in _path_keys(row.get("file_path"), canonical)))


def _vectors(connection, workspace, vectors, canonical):
    ranklists, concept_entities = [], []
    connection.execute("SET LOCAL hnsw.ef_search = 120")
    for table, columns in (("chunks", "file_path"),
                           ("entity", "file_path, entity_name"),
                           ("relation", "file_path")):
        for index, vector in enumerate(vectors[:1] if table == "chunks" else vectors):
            vector_text = "[" + ",".join(str(float(value)) for value in vector) + "]"
            # Table names are fixed literals; workspace and vectors are bound values.
            rows = connection.execute(
                f"SELECT {columns} FROM lightrag_vdb_{table} WHERE workspace = %s "
                "ORDER BY content_vector <=> %s::vector LIMIT %s",
                (workspace, vector_text, _TOP_K),
            ).fetchall()
            ranklists.append(_paper_ranklist(rows, canonical))
            if table == "entity" and index:
                concept_entities.append({row["entity_name"] for row in rows if row.get("entity_name")})
    return ranklists, concept_entities


def _topology(neo4j, workspace, concept_entities, canonical):
    all_entities = sorted(set().union(*concept_entities)) if concept_entities else []
    if not all_entities:
        return []
    # LightRAG uses the workspace as the actual node label, including punctuation.
    label = workspace.replace("`", "``")
    query = neo4j.Query(
        f"MATCH (u:`{label}`)-[r]-(v:`{label}`) "
        "WHERE u.entity_id IN $all_ents AND v.entity_id IN $all_ents "
        "AND elementId(u) < elementId(v) "
        "RETURN u.entity_id AS source, v.entity_id AS target, "
        "r.weight AS weight, r.file_path AS file_path LIMIT $limit",
        timeout=_TIMEOUT,
    )
    scores = defaultdict(float)
    with neo4j.GraphDatabase.driver(
        os.environ["NEO4J_URI"],
        auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]),
        connection_timeout=_TIMEOUT, connection_acquisition_timeout=_TIMEOUT,
        max_transaction_retry_time=0, max_connection_pool_size=1,
    ) as driver:
        # Match LightRAG's sanitized chunk_entity_relation namespace and its
        # catalog reader, without the upstream database-creation attempt.
        preferred = os.environ.get("NEO4J_DATABASE", "chunk-entity-relation")
        edges = []
        for database in (preferred, None):
            try:
                with driver.session(database=database, default_access_mode=neo4j.READ_ACCESS) as session:
                    edges = list(session.run(query, all_ents=all_entities, limit=_EDGE_LIMIT))
                break
            except neo4j.exceptions.ClientError as exc:
                if database is None or exc.code != "Neo.ClientError.Database.DatabaseNotFound":
                    raise
    for edge in edges:
        source, target = edge["source"], edge["target"]
        cross_facet = any(source in left and target in right
                          for i, left in enumerate(concept_entities)
                          for j, right in enumerate(concept_entities) if i != j)
        weight = float(edge["weight"] or 0)
        if not math.isfinite(weight):
            raise ValueError("Non-finite graph edge weight")
        score = float(cross_facet) + math.log1p(max(0.0, weight))
        if score > 0:
            for key in _path_keys(edge["file_path"], canonical):
                scores[key] += score
    return sorted(scores, key=lambda key: (-scores[key], key))


def kg_local_search(query: str, plan, library, *, embedding_loop=None) -> list[str]:
    """Return canonical paper keys ranked by vector consensus and graph topology.

    The caller runs this synchronous helper off the event loop. Dependencies and
    the shared BGE model are loaded only after explicit backend configuration.
    Any failure discards this optional arm, allowing ordinary search to continue.
    Production supplies its serving loop to share S2's GPU admission semaphore.
    Standalone synchronous callers without a loop serialize only with other KG
    calls and must not run alongside S2 embedding.
    """
    try:
        configuration = _configuration()
        if configuration is None or not query.strip():
            return []
        canonical = _canonical_keys(library)
        if not canonical:
            return []
        workspace, pg_options = configuration
        import neo4j
        import psycopg
        from psycopg.rows import dict_row

        from papervault.knowledge.store.lightrag_init import _bge_encode_sync

        concepts = list(getattr(plan, "kg_concepts", None) or plan.terms)[:8]
        # Bound concurrent KG model use and its queue. The shared model itself is
        # the existing process singleton, never a second GPU model instance.
        if not _ENCODE_LOCK.acquire(timeout=_TIMEOUT):
            return []
        try:
            semaphore = None
            if embedding_loop is not None:
                semaphore = asyncio.run_coroutine_threadsafe(
                    _embedding_permit(), embedding_loop).result()
            try:
                vectors = _bge_encode_sync([query, *concepts])
            finally:
                if semaphore is not None:
                    # The worker retains admission even when its async awaiter
                    # is cancelled. Release only after the real GPU call exits,
                    # before any database reads, on the semaphore's owning loop.
                    embedding_loop.call_soon_threadsafe(semaphore.release)
        finally:
            _ENCODE_LOCK.release()
        with psycopg.connect(**pg_options, row_factory=dict_row) as connection:
            ranklists, concept_entities = _vectors(connection, workspace, vectors, canonical)
        ranklists.append(_topology(neo4j, workspace, concept_entities, canonical))
        scores = defaultdict(float)
        for ranklist in ranklists:
            for rank, key in enumerate(ranklist):
                scores[key] += 1.0 / (60 + rank)
        return sorted(scores, key=lambda key: (-scores[key], key))
    except Exception:
        log.debug("Optional KG local search unavailable", exc_info=True)
        return []
