"""Idempotent, legacy-only repair of chunk tracking and graph source references.

Uses existing tables directly; never initializes LightRAG or creates a database.
The operator CLI checks service quiescence before calling with apply=True.
"""
from __future__ import annotations

import os

from lightrag.constants import (
    DEFAULT_MAX_SOURCE_IDS_PER_ENTITY,
    DEFAULT_MAX_SOURCE_IDS_PER_RELATION,
    GRAPH_FIELD_SEP,
)
from lightrag.utils import make_relation_chunk_key


def _rows(connection, table, workspace, columns):
    # Table and column names are internal literals, never operator input.
    exists = connection.execute('SELECT to_regclass(%s) AS name', (table,)).fetchone()
    if exists['name'] is None:
        return []
    return connection.execute(
        f'SELECT {columns} FROM {table} WHERE workspace = %s', (workspace,)).fetchall()


def _clean(ids, valid):
    return list(dict.fromkeys(cid for cid in ids if cid in valid))


def _tracking(connection, table, workspace, valid, ghosts, summary, *, apply):
    prefix = table.removeprefix('lightrag_')
    cleaned = {}
    for row in _rows(connection, table, workspace, 'id, chunk_ids, count'):
        old = row['chunk_ids']
        new = _clean(old, valid)
        cleaned[row['id']] = new
        ghosts.update(cid for cid in old if cid not in valid)
        if new == old:
            continue
        summary[f'{prefix}_rows_affected'] += 1
        summary[f'{prefix}_rows_emptied'] += int(not new)
        if apply:
            from psycopg.types.json import Jsonb

            # Retain empty tracking rows: they are authoritative in LightRAG 1.5.7
            # and prevent fallback reseeding from stale graph source_id values.
            connection.execute(
                f'UPDATE {table} SET chunk_ids = %s, count = %s, update_time = CURRENT_TIMESTAMP '
                'WHERE workspace = %s AND id = %s',
                (Jsonb(new), len(new), workspace, row['id']))
    return cleaned


def _graph(session, workspace, valid, tracking, ghosts, summary, *, apply):
    label = workspace.replace('`', '``')
    specs = (
        ('nodes', f'MATCH (n:`{label}`)', 'n',
         'n.entity_id AS entity_id', DEFAULT_MAX_SOURCE_IDS_PER_ENTITY),
        ('edges', f'MATCH (s:`{label}`)-[r]->(t:`{label}`)', 'r',
         's.entity_id AS src_id, t.entity_id AS tgt_id', DEFAULT_MAX_SOURCE_IDS_PER_RELATION),
    )
    # Finish both reads before writing so database selection errors cannot leave
    # a partially applied graph in the preferred database before fallback.
    snapshots = [list(session.run(
        f'{match} RETURN elementId({var}) AS element_id, {identity}, {var}.source_id AS source_id'))
        for _, match, var, identity, _ in specs]
    for (kind, match, var, _, cap), rows in zip(specs, snapshots):
        for row in rows:
            if not row['source_id']:
                continue
            old = row['source_id'].split(GRAPH_FIELD_SEP)
            missing = {cid for cid in old if cid not in valid}
            if not missing:
                continue
            ghosts.update(missing)
            new = _clean(old, valid)
            before = len(new)
            key = (row['entity_id'] if kind == 'nodes' else
                   make_relation_chunk_key(row['src_id'], row['tgt_id']))
            for cid in tracking[kind].get(key, []):
                if len(new) >= cap:
                    break
                if cid not in new:
                    new.append(cid)
            summary[f'neo4j_{kind}_affected'] += 1
            summary[f'neo4j_{kind}_backfilled'] += int(len(new) > before)
            summary[f'neo4j_{kind}_emptied'] += int(not new)
            if apply:
                session.run(
                    f'{match} WHERE elementId({var}) = $element_id SET {var}.source_id = $source_id',
                    element_id=row['element_id'], source_id=GRAPH_FIELD_SEP.join(new)).consume()


def scrub_ghost_chunks(*, apply: bool = False) -> dict:
    """Inspect or repair one guarded workspace and return counts of changed rows."""
    from papervault.knowledge.config import PostgresConfig
    from papervault.knowledge.store.graph import assert_safe_workspace

    if os.environ.get('PAPERVAULT_KNOWLEDGE_JOBS', 'legacy').strip().lower() != 'legacy':
        raise RuntimeError('scrub-ghost-chunks requires legacy knowledge mode; use isolated catalog rebuilds')
    workspace = assert_safe_workspace()
    summary = {'workspace': workspace, 'dry_run': not apply, 'valid_chunks': 0,
               'distinct_ghost_chunk_ids': 0}
    for kind in ('entity_chunks', 'relation_chunks'):
        for metric in ('affected', 'emptied'):
            summary[f'{kind}_rows_{metric}'] = 0
    for kind in ('nodes', 'edges'):
        for metric in ('affected', 'backfilled', 'emptied'):
            summary[f'neo4j_{kind}_{metric}'] = 0

    import neo4j
    import psycopg
    from psycopg.rows import dict_row

    config = PostgresConfig.from_env()
    options = {} if apply else {'options': '-c default_transaction_read_only=on'}
    ghosts = set()
    with psycopg.connect(
        host=config.host, port=config.port, user=config.user, password=config.password,
        dbname=os.environ.get('POSTGRES_DATABASE') or config.db,
        connect_timeout=5, row_factory=dict_row, **options,
    ) as connection:
        docs = {r['id'] for r in _rows(connection, 'lightrag_doc_chunks', workspace, 'id')}
        vectors = {r['id'] for r in _rows(connection, 'lightrag_vdb_chunks', workspace, 'id')}
        valid = docs & vectors
        summary['valid_chunks'] = len(valid)
        tracking = {
            kind: _tracking(connection, table, workspace, valid, ghosts, summary, apply=apply)
            for kind, table in (('nodes', 'lightrag_entity_chunks'), ('edges', 'lightrag_relation_chunks'))}
        with neo4j.GraphDatabase.driver(
            os.environ['NEO4J_URI'],
            auth=(os.environ['NEO4J_USERNAME'], os.environ['NEO4J_PASSWORD']),
            connection_timeout=5, connection_acquisition_timeout=5, max_transaction_retry_time=0,
        ) as driver:
            preferred = os.environ.get('NEO4J_DATABASE', 'chunk-entity-relation')
            for database in (preferred, None):
                try:
                    with driver.session(database=database, default_access_mode=(
                        neo4j.WRITE_ACCESS if apply else neo4j.READ_ACCESS)) as session:
                        _graph(session, workspace, valid, tracking, ghosts, summary, apply=apply)
                    break
                except neo4j.exceptions.ClientError as exc:
                    if database is None or exc.code != 'Neo.ClientError.Database.DatabaseNotFound':
                        raise
    summary['distinct_ghost_chunk_ids'] = len(ghosts)
    return summary
