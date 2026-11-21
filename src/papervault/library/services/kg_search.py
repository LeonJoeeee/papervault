"""Optional, read-only vector and graph candidate retrieval for library search.

Use the installed LightRAG 1.5.7 schema directly: constructing its storage adapters
would initialize schemas and indexes. Returned identities always belong to the
caller's library snapshot; database paths alone cannot introduce catalog records.
"""
from __future__ import annotations

import asyncio
import atexit
from collections import defaultdict
import logging
import math
import os
import threading

from .registration import serving_excluded

log = logging.getLogger(__name__)
_ENCODE_LOCK = threading.Lock()
_POOL_LOCK = threading.Lock()
_STATS_LOCK = threading.Lock()
_TOP_K = 60
_EDGE_LIMIT = 1000
_DB_TIMEOUT = 5.0
_DEFAULT_TIMEOUT = 25.0
_MAX_PG_POOL = 4

_PG_POOLS: dict[tuple, list] = defaultdict(list)
_NEO_DRIVERS: dict[tuple, object] = {}
_STATS = {
    "calls": 0,
    "hits": 0,
    "encode_lock_timeouts": 0,
    "permit_timeouts": 0,
    "errors": 0,
    "last_error": None,
}


def _parse_timeout(raw: str | None, default: float = _DEFAULT_TIMEOUT) -> float:
    if raw is None or not raw.strip():
        return default
    try:
        value = float(raw)
    except (TypeError, ValueError):
        log.warning("Invalid PAPERVAULT_SEARCH_KG_TIMEOUT_S=%r; using %.1fs", raw, default)
        return default
    if not math.isfinite(value) or value <= 0:
        log.warning("Non-positive PAPERVAULT_SEARCH_KG_TIMEOUT_S=%r; using %.1fs", raw, default)
        return default
    return value


_TIMEOUT = _parse_timeout(os.getenv("PAPERVAULT_SEARCH_KG_TIMEOUT_S"), _DEFAULT_TIMEOUT)
_IMPORT_TIMEOUT = _TIMEOUT


def _search_timeout() -> float:
    if _TIMEOUT != _IMPORT_TIMEOUT:
        return float(_TIMEOUT)
    return _parse_timeout(os.getenv("PAPERVAULT_SEARCH_KG_TIMEOUT_S"), _DEFAULT_TIMEOUT)


def _db_timeout(timeout: float) -> float:
    return min(_DB_TIMEOUT, max(0.01, float(timeout)))


def _record_stat(field: str, *, error: str | None = None) -> None:
    with _STATS_LOCK:
        _STATS[field] += 1
        if error is not None:
            _STATS["last_error"] = error


def kg_search_stats() -> dict:
    """Return a thread-safe snapshot of optional KG search counters."""
    with _STATS_LOCK:
        return dict(_STATS)


def _safe_close(resource) -> None:
    close = getattr(resource, "close", None)
    if callable(close):
        try:
            close()
        except Exception:
            pass


def close_pools() -> None:
    """Close and clear all pooled read-only KG Postgres/Neo4j connections."""
    with _POOL_LOCK:
        pg_conns = [conn for conns in _PG_POOLS.values() for conn in conns]
        _PG_POOLS.clear()
        neo_drivers = list(_NEO_DRIVERS.values())
        _NEO_DRIVERS.clear()
    for conn in pg_conns:
        _safe_close(conn)
    for driver in neo_drivers:
        _safe_close(driver)


atexit.register(close_pools)


async def _embedding_permit(timeout: float | None = None):
    from papervault.knowledge.store.lightrag_init import _get_embed_sem

    semaphore = _get_embed_sem()
    wait_timeout = _search_timeout() if timeout is None else timeout
    await asyncio.wait_for(semaphore.acquire(), timeout=wait_timeout)
    return semaphore


def _configuration(db_timeout: float | None = None):
    enabled = os.getenv("PAPERVAULT_SEARCH_KG")
    pg_backend = (
        os.getenv("PAPERVAULT_CATALOG_AUTHORITY") == "postgres"
        or os.getenv("PAPERVAULT_STORAGE_BACKEND") == "postgres"
    )
    if enabled != "1" and (enabled is not None or not pg_backend):
        return None
    workspace = os.getenv("POSTGRES_WORKSPACE", "").strip()
    database = os.getenv("POSTGRES_DATABASE") or os.getenv("POSTGRES_DB")
    neo_workspace = os.getenv("NEO4J_WORKSPACE", "").strip()
    required = (workspace, database, os.getenv("POSTGRES_USER"),
                os.getenv("POSTGRES_PASSWORD"), os.getenv("NEO4J_URI"),
                os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
    if not all(required) or (neo_workspace and neo_workspace != workspace):
        return None
    effective_db_timeout = _db_timeout(_search_timeout() if db_timeout is None else db_timeout)
    connect_timeout = max(1, int(effective_db_timeout))
    statement_timeout_ms = max(100, int(effective_db_timeout * 1000))
    return workspace, {
        "host": os.getenv("POSTGRES_HOST", "localhost"),
        "port": int(os.getenv("POSTGRES_PORT", "5432")),
        "user": os.environ["POSTGRES_USER"], "password": os.environ["POSTGRES_PASSWORD"],
        "dbname": database, "connect_timeout": connect_timeout,
        "options": f"-c default_transaction_read_only=on -c statement_timeout={statement_timeout_ms}",
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


def _weighted_row_scores(rows, canonical) -> dict[str, float]:
    """Specificity-weighted RRF scores for one vector query's result rows.

    Each row's contribution scales as ``1 / ((60 + row_rank) * log2(1 + n))``,
    where ``n`` is the number of distinct canonical papers attached to that row
    (1.0x for a 1-paper row, ~7.65x discount for a 200-paper hub entity). Within
    one query each paper keeps only its strongest row score so duplicate paths
    never stack extra RRF votes.
    """
    scores: dict[str, float] = {}
    for row_rank, row in enumerate(rows):
        keys = _path_keys(row.get("file_path"), canonical)
        if not keys:
            continue
        weight = 1.0 / ((60 + row_rank) * math.log2(1 + len(keys)))
        for key in keys:
            prev = scores.get(key)
            if prev is None or weight > prev:
                scores[key] = weight
    return scores


def _vectors(connection, workspace, vectors, canonical):
    score_maps, concept_entities = [], []
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
            score_maps.append(_weighted_row_scores(rows, canonical))
            if table == "entity" and index:
                concept_entities.append({row["entity_name"] for row in rows if row.get("entity_name")})
    return score_maps, concept_entities


def _is_poolable_pg(connection) -> bool:
    return callable(getattr(connection, "close", None)) and callable(getattr(connection, "transaction", None))


def _is_poolable_neo(driver) -> bool:
    return callable(getattr(driver, "close", None))


def _checkout_pg(pool_key: tuple):
    with _POOL_LOCK:
        bucket = _PG_POOLS.get(pool_key)
        while bucket:
            conn = bucket.pop()
            if not getattr(conn, "closed", False):
                return conn
            _safe_close(conn)
    return None


def _checkin_pg(pool_key: tuple, conn) -> None:
    if getattr(conn, "closed", False):
        _safe_close(conn)
        return
    with _POOL_LOCK:
        bucket = _PG_POOLS[pool_key]
        if len(bucket) < _MAX_PG_POOL:
            bucket.append(conn)
            return
    _safe_close(conn)


def _run_vectors_pooled(psycopg, dict_row, pg_options, workspace, vectors, canonical):
    connect_fn = psycopg.connect
    pool_key = (tuple(sorted(pg_options.items())), id(connect_fn))
    pooled = _checkout_pg(pool_key)
    if pooled is not None:
        try:
            with pooled.transaction():
                result = _vectors(pooled, workspace, vectors, canonical)
            _checkin_pg(pool_key, pooled)
            return result
        except Exception as exc:
            _safe_close(pooled)
            log.debug("Reconnecting stale KG Postgres connection after error: %s", exc)
    connection = connect_fn(**pg_options, row_factory=dict_row)
    if not _is_poolable_pg(connection):
        with connection as active:
            return _vectors(active, workspace, vectors, canonical)
    try:
        with connection.transaction():
            result = _vectors(connection, workspace, vectors, canonical)
    except Exception:
        _safe_close(connection)
        raise
    _checkin_pg(pool_key, connection)
    return result


def _run_topology_query(neo4j, driver, query, all_entities):
    # Match LightRAG's sanitized chunk_entity_relation namespace and its
    # catalog reader, without the upstream database-creation attempt.
    preferred = os.environ.get("NEO4J_DATABASE", "chunk-entity-relation")
    for database in (preferred, None):
        try:
            with driver.session(database=database, default_access_mode=neo4j.READ_ACCESS) as session:
                return list(session.run(query, all_ents=all_entities, limit=_EDGE_LIMIT))
        except neo4j.exceptions.ClientError as exc:
            if database is None or exc.code != "Neo.ClientError.Database.DatabaseNotFound":
                raise
    return []


def _topology(neo4j, workspace, concept_entities, canonical, *, db_timeout: float | None = None):
    all_entities = sorted(set().union(*concept_entities)) if concept_entities else []
    if not all_entities:
        return []
    effective_timeout = _db_timeout(_search_timeout() if db_timeout is None else db_timeout)
    # LightRAG uses the workspace as the actual node label, including punctuation.
    label = workspace.replace("`", "``")
    query = neo4j.Query(
        f"MATCH (u:`{label}`)-[r]-(v:`{label}`) "
        "WHERE u.entity_id IN $all_ents AND v.entity_id IN $all_ents "
        "AND elementId(u) < elementId(v) "
        "RETURN u.entity_id AS source, v.entity_id AS target, "
        "r.weight AS weight, r.file_path AS file_path "
        "ORDER BY r.weight DESC LIMIT $limit",
        timeout=effective_timeout,
    )
    uri = os.environ["NEO4J_URI"]
    user = os.environ["NEO4J_USERNAME"]
    password = os.environ["NEO4J_PASSWORD"]
    driver_fn = neo4j.GraphDatabase.driver
    driver_key = (uri, user, password, effective_timeout, id(driver_fn))

    with _POOL_LOCK:
        cached_driver = _NEO_DRIVERS.get(driver_key)

    edges = None
    if cached_driver is not None:
        try:
            edges = _run_topology_query(neo4j, cached_driver, query, all_entities)
        except Exception as exc:
            with _POOL_LOCK:
                if _NEO_DRIVERS.get(driver_key) is cached_driver:
                    del _NEO_DRIVERS[driver_key]
            _safe_close(cached_driver)
            log.debug("Reconnecting stale KG Neo4j driver after error: %s", exc)

    if edges is None:
        driver = driver_fn(
            uri,
            auth=(user, password),
            connection_timeout=effective_timeout,
            connection_acquisition_timeout=effective_timeout,
            max_transaction_retry_time=0,
            max_connection_pool_size=4,
        )
        if not _is_poolable_neo(driver):
            with driver as active:
                edges = _run_topology_query(neo4j, active, query, all_entities)
        else:
            try:
                edges = _run_topology_query(neo4j, driver, query, all_entities)
            except Exception:
                _safe_close(driver)
                raise
            evicted = None
            with _POOL_LOCK:
                if driver_key not in _NEO_DRIVERS:
                    _NEO_DRIVERS[driver_key] = driver
                else:
                    evicted = driver
            if evicted is not None:
                _safe_close(evicted)

    scores = defaultdict(float)
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
            keys = _path_keys(edge["file_path"], canonical)
            if keys:
                per_key = score / math.log2(1 + len(keys))
                for key in keys:
                    scores[key] += per_key
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
        timeout = _search_timeout()
        db_timeout = _db_timeout(timeout)
        configuration = _configuration(db_timeout=db_timeout)
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
        if not _ENCODE_LOCK.acquire(timeout=timeout):
            _record_stat("encode_lock_timeouts", error=f"encode_lock_timeout:{timeout:.2f}s")
            log.warning("Optional KG local search skipped: encode lock timed out after %.1fs", timeout)
            return []
        try:
            semaphore = None
            if embedding_loop is not None:
                semaphore = asyncio.run_coroutine_threadsafe(
                    _embedding_permit(timeout), embedding_loop).result()
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
        score_maps, concept_entities = _run_vectors_pooled(
            psycopg, dict_row, pg_options, workspace, vectors, canonical
        )
        topo_ranks = _topology(neo4j, workspace, concept_entities, canonical, db_timeout=db_timeout)
        scores = defaultdict(float)
        for score_map in score_maps:
            for key, value in score_map.items():
                scores[key] += value
        for rank, key in enumerate(topo_ranks):
            scores[key] += 1.0 / (60 + rank)
        ranked = sorted(scores, key=lambda key: (-scores[key], key))
        _record_stat("calls")
        if ranked:
            _record_stat("hits")
        return ranked
    except (TimeoutError, asyncio.TimeoutError) as exc:
        _record_stat("permit_timeouts", error=f"{type(exc).__name__}: {exc}")
        log.warning("Optional KG local search timed out after %.1fs: %s", _search_timeout(), exc)
        return []
    except Exception as exc:
        _record_stat("errors", error=f"{type(exc).__name__}: {exc}")
        log.warning("Optional KG local search unavailable: %s", exc, exc_info=True)
        return []
