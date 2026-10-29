"""Legacy scrub exercises the real CLI and SQL/Cypher boundaries, without services."""
from copy import deepcopy
from types import SimpleNamespace
import json

from click.testing import CliRunner
import neo4j
import psycopg
import pytest
from lightrag.utils import make_relation_chunk_key

from papervault.knowledge.cli import cli


class Result(list):
    def fetchone(self):
        return self[0] if self else None

    def fetchall(self):
        return list(self)

    def consume(self):
        return None


@pytest.fixture
def backends(monkeypatch):
    workspace = 'probe`344'
    for name in ('NEO4J_WORKSPACE', 'POSTGRES_WORKSPACE'):
        monkeypatch.setenv(name, workspace)
    for name, value in {'POSTGRES_DATABASE': 'test-db', 'POSTGRES_USER': 'user',
                        'POSTGRES_PASSWORD': 'secret', 'NEO4J_URI': 'bolt://test',
                        'NEO4J_USERNAME': 'neo', 'NEO4J_PASSWORD': 'secret'}.items():
        monkeypatch.setenv(name, value)
    valid = [f'c{i}' for i in range(205)]
    edge_key = make_relation_chunk_key('A', 'B')
    state = SimpleNamespace(workspace=workspace, connects=[], sessions=[], writes=[],
                            missing_database=False, driver_args=None)
    state.tables = {
        'lightrag_doc_chunks': [{'id': c} for c in valid + ['no-vector']],
        'lightrag_vdb_chunks': [{'id': c} for c in valid + ['no-text']],
        'lightrag_entity_chunks': [
            {'id': 'A', 'chunk_ids': ['gone', 'c2', 'c2', 'no-vector', 'no-text'] + valid, 'count': 210},
            {'id': 'Empty', 'chunk_ids': ['gone'], 'count': 1},
            {'id': 'Clean', 'chunk_ids': ['c3'], 'count': 1}],
        'lightrag_relation_chunks': [
            {'id': edge_key, 'chunk_ids': ['gone', 'c3', 'c3'] + valid, 'count': 208},
            {'id': make_relation_chunk_key('Empty', 'B'), 'chunk_ids': ['gone'], 'count': 1}],
    }
    state.nodes = [
        {'element_id': 'n1', 'entity_id': 'A', 'source_id': 'gone<SEP>c1<SEP>c1'},
        {'element_id': 'n2', 'entity_id': 'Empty', 'source_id': 'gone'},
        {'element_id': 'n3', 'entity_id': 'Clean', 'source_id': 'c3'},
        {'element_id': 'n4', 'entity_id': 'NoSource', 'source_id': ''}]
    state.edges = [
        {'element_id': 'e1', 'src_id': 'A', 'tgt_id': 'B', 'source_id': 'gone<SEP>c4'},
        {'element_id': 'e2', 'src_id': 'Empty', 'tgt_id': 'B', 'source_id': 'gone'}]

    class Connection:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def execute(self, query, params=()):
            if 'to_regclass' in query:
                return Result([{'name': params[0] if params[0] in state.tables else None}])
            if query.startswith('SELECT'):
                assert params == (workspace,)
                assert 'WHERE workspace = %s' in query
                table = query.split(' FROM ')[1].split()[0]
                return Result(deepcopy(state.tables[table]))
            assert query.startswith('UPDATE '), query
            assert 'WHERE workspace = %s AND id = %s' in query
            assert 'update_time' in query
            assert 'default_transaction_read_only=on' not in state.connects[-1].get('options', '')
            chunks, count, ws, row_id = params
            assert ws == workspace
            table = query.split()[1]
            row = next(r for r in state.tables[table] if r['id'] == row_id)
            row.update(chunk_ids=chunks.obj, count=count)
            state.writes.append(('pg', table, row_id))
            return Result()

    def connect(**kwargs):
        state.connects.append(kwargs)
        return Connection()

    class Session:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def run(self, query, **params):
            query = str(query)
            if state.missing_database and state.sessions[-1]['database'] is not None:
                raise neo4j.exceptions.ClientError._hydrate_neo4j(
                    code='Neo.ClientError.Database.DatabaseNotFound', message='missing')
            assert '`probe``344`' in query
            rows = state.edges if '-[r]->' in query else state.nodes
            if 'SET ' not in query:
                return Result(deepcopy(rows))
            assert state.sessions[-1]['default_access_mode'] == neo4j.WRITE_ACCESS
            row = next(r for r in rows if r['element_id'] == params['element_id'])
            row['source_id'] = params['source_id']
            state.writes.append(('neo', params['element_id']))
            return Result()

    class Driver:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def session(self, **kwargs):
            state.sessions.append(kwargs)
            return Session()

    def driver(uri, **kwargs):
        state.driver_args = (uri, kwargs)
        return Driver()

    monkeypatch.setattr(psycopg, 'connect', connect)
    monkeypatch.setattr(neo4j.GraphDatabase, 'driver', driver)
    monkeypatch.setattr('papervault.knowledge.cli._active_service', lambda: [])
    return state


def run_cli(*args):
    return CliRunner().invoke(cli, ['scrub-ghost-chunks', *args])


def test_scrub_dry_run_apply_and_idempotence(backends):
    before = deepcopy((backends.tables, backends.nodes, backends.edges))
    result = run_cli('--json')
    assert result.exit_code == 0, result.output
    metrics = json.loads(result.output)
    assert metrics == {
        'workspace': 'probe`344', 'dry_run': True, 'valid_chunks': 205,
        'distinct_ghost_chunk_ids': 3,
        'entity_chunks_rows_affected': 2, 'entity_chunks_rows_emptied': 1,
        'relation_chunks_rows_affected': 2, 'relation_chunks_rows_emptied': 1,
        'neo4j_nodes_affected': 2, 'neo4j_nodes_backfilled': 1, 'neo4j_nodes_emptied': 1,
        'neo4j_edges_affected': 2, 'neo4j_edges_backfilled': 1, 'neo4j_edges_emptied': 1}
    assert before == (backends.tables, backends.nodes, backends.edges)
    assert backends.writes == []
    assert 'default_transaction_read_only=on' in backends.connects[0]['options']
    assert backends.sessions[0]['default_access_mode'] == neo4j.READ_ACCESS
    assert backends.connects[0]['dbname'] == 'test-db'
    assert backends.driver_args[0] == 'bolt://test'
    assert backends.driver_args[1]['auth'] == ('neo', 'secret')

    applied = run_cli('--apply', '--json')
    assert applied.exit_code == 0, applied.output
    assert json.loads(applied.output) == {**metrics, 'dry_run': False}
    assert len(backends.writes) == 8
    entities = backends.tables['lightrag_entity_chunks']
    assert entities[0]['chunk_ids'][:4] == ['c2', 'c0', 'c1', 'c3']
    assert entities[0]['count'] == 205
    assert entities[1] == {'id': 'Empty', 'chunk_ids': [], 'count': 0}
    assert backends.tables['lightrag_relation_chunks'][1]['chunk_ids'] == []
    assert backends.tables['lightrag_relation_chunks'][1]['count'] == 0
    assert backends.nodes[0]['source_id'].split('<SEP>')[:4] == ['c1', 'c2', 'c0', 'c3']
    assert backends.edges[0]['source_id'].split('<SEP>')[:4] == ['c4', 'c3', 'c0', 'c1']
    for row in (backends.nodes[0], backends.edges[0]):
        assert len(row['source_id'].split('<SEP>')) == 200
    assert backends.nodes[1]['source_id'] == backends.edges[1]['source_id'] == ''
    backends.writes.clear()
    again = run_cli('--apply', '--json')
    assert again.exit_code == 0, again.output
    assert backends.writes == []
    assert all(value == 0 for key, value in json.loads(again.output).items()
               if key not in ('workspace', 'dry_run', 'valid_chunks'))


@pytest.mark.parametrize('mode', ['postgres', '', 'unknown'])
def test_catalog_mode_refuses_before_connections(backends, monkeypatch, mode):
    monkeypatch.setenv('PAPERVAULT_KNOWLEDGE_JOBS', mode)
    result = run_cli('--apply')
    assert result.exit_code != 0
    assert 'legacy' in result.output
    assert backends.connects == backends.sessions == []


@pytest.mark.parametrize('neo,pg', [('l0', 'l0'), ('probe', 'other'), ('', '')])
def test_workspace_guard_before_connections(backends, monkeypatch, neo, pg):
    monkeypatch.setenv('NEO4J_WORKSPACE', neo)
    monkeypatch.setenv('POSTGRES_WORKSPACE', pg)
    result = run_cli()
    assert result.exit_code != 0
    assert backends.connects == backends.sessions == []


def test_active_service_refuses_apply_before_connections(backends, monkeypatch):
    monkeypatch.setattr('papervault.knowledge.cli._active_service', lambda: ['papervault.service'])
    result = run_cli('--apply')
    assert result.exit_code == 2
    assert 'papervault.service' in result.output
    assert backends.connects == backends.sessions == []
    assert run_cli().exit_code == 0  # Read-only inspection is still allowed.


@pytest.mark.parametrize('database', [None, 'custom'])
def test_neo4j_home_database_fallback(backends, monkeypatch, database):
    if database:
        monkeypatch.setenv('NEO4J_DATABASE', database)
    backends.missing_database = True
    result = run_cli('--apply')
    assert result.exit_code == 0, result.output
    assert [s['database'] for s in backends.sessions] == [database or 'chunk-entity-relation', None]


@pytest.mark.parametrize('missing', ['lightrag_doc_chunks', 'lightrag_vdb_chunks', 'all'])
def test_uninitialized_tables_are_cleanly_empty(backends, missing):
    if missing == 'all':
        backends.tables.clear()
    else:
        del backends.tables[missing]
        backends.tables['lightrag_entity_chunks'] = []
        backends.tables['lightrag_relation_chunks'] = []
    backends.nodes.clear()
    backends.edges.clear()
    result = run_cli('--apply', '--json')
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)['valid_chunks'] == 0
    assert backends.writes == []


def test_duplicate_tracking_ids_are_scrubbed_without_ghosts(backends, monkeypatch):
    monkeypatch.setenv('PAPERVAULT_KNOWLEDGE_JOBS', ' LEGACY ')
    backends.tables['lightrag_entity_chunks'] = [{'id': 'Clean', 'chunk_ids': ['c3', 'c3'], 'count': 2}]
    backends.tables['lightrag_relation_chunks'] = []
    backends.nodes = [{'element_id': 'n3', 'entity_id': 'Clean', 'source_id': 'c3'}]
    backends.edges = []
    result = run_cli('--apply', '--json')
    assert result.exit_code == 0, result.output
    summary = json.loads(result.output)
    assert summary['distinct_ghost_chunk_ids'] == summary['neo4j_nodes_affected'] == 0
    assert summary['entity_chunks_rows_affected'] == 1
    assert backends.tables['lightrag_entity_chunks'] == [{'id': 'Clean', 'chunk_ids': ['c3'], 'count': 1}]
    assert backends.writes == [('pg', 'lightrag_entity_chunks', 'Clean')]
