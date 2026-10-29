"""Merge discovery indexes must still lead document deletion to canonical keys."""
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from papervault.knowledge.store.dedup import sweep


class Index:
    def __init__(self, rows):
        self.rows = deepcopy(rows)
        self.upsert = AsyncMock(side_effect=self.rows.update)
        self.index_done_callback = AsyncMock()

    async def get_by_ids(self, ids):
        return [deepcopy(self.rows.get(did)) for did in ids]


def rag_with_indexes(*, fail=False):
    names = {'Alpha Beta': {'entity_type': 'concept'}, 'alpha-beta': {'entity_type': 'concept'}}
    graph = SimpleNamespace(
        get_all_labels=AsyncMock(side_effect=lambda: list(names)),
        get_nodes_batch=AsyncMock(side_effect=lambda ids: {i: names[i] for i in ids}),
        node_degree=AsyncMock(side_effect=lambda name: 10 if name == 'Alpha Beta' else 1),
    )

    async def merge(**kwargs):
        if fail:
            raise RuntimeError('merge failed')
        for source in kwargs['source_entities']:
            names.pop(source)

    return SimpleNamespace(
        chunk_entity_relation_graph=graph,
        amerge_entities=AsyncMock(side_effect=merge),
        doc_status=SimpleNamespace(get_docs_by_statuses=AsyncMock(return_value={
            'affected': {}, 'unchanged': {}, 'missing-index': {}})),
        full_entities=Index({
            'affected': {'entity_names': ['Other', 'alpha-beta', 'Alpha Beta', 'alpha-beta'], 'count': 4, 'extra': 1},
            'unchanged': {'entity_names': ['Other'], 'count': 1}}),
        full_relations=Index({
            'affected': {'relation_pairs': [['Other', 'alpha-beta'], ['Other', 'Alpha Beta'],
                                             ['alpha-beta', 'Alpha Beta'], ['alpha-beta', 'Third']], 'count': 4},
            'unchanged': {'relation_pairs': [['Other', 'Third']], 'count': 1}}),
    )


async def test_merge_reconciles_document_discovery_indexes():
    rag = rag_with_indexes()
    stats = await sweep(rag, dry_run=False)
    assert stats.single_type_merged == 1
    assert rag.full_entities.rows['affected'] == {
        'entity_names': ['Other', 'Alpha Beta'], 'count': 2, 'extra': 1}
    assert rag.full_relations.rows['affected'] == {
        'relation_pairs': [['Other', 'Alpha Beta'], ['Alpha Beta', 'Third']], 'count': 2}
    for index in (rag.full_entities, rag.full_relations):
        assert set(index.upsert.call_args.args[0]) == {'affected'}
        index.index_done_callback.assert_awaited_once()
    # A later deletion reads these indexes, so neither lookup can use the retired name.
    assert 'alpha-beta' not in str(rag.full_entities.rows)
    assert 'alpha-beta' not in str(rag.full_relations.rows)


@pytest.mark.parametrize('dry_run,fail', [(True, False), (False, True)])
async def test_unmerged_sources_leave_indexes_untouched(dry_run, fail):
    rag = rag_with_indexes(fail=fail)
    await sweep(rag, dry_run=dry_run)
    for index in (rag.full_entities, rag.full_relations):
        index.upsert.assert_not_awaited()
        index.index_done_callback.assert_not_awaited()
    rag.doc_status.get_docs_by_statuses.assert_not_awaited()


async def test_minimal_rag_without_document_indexes_remains_supported():
    rag = rag_with_indexes()
    del rag.full_entities, rag.full_relations, rag.doc_status
    assert (await sweep(rag, dry_run=False)).single_type_merged == 1


async def test_merge_deduplicates_reversed_undirected_pairs_in_first_seen_order():
    rag = rag_with_indexes()
    rag.full_relations.rows['affected'] = {
        'relation_pairs': [['Alpha Beta', 'Other'], ['Other', 'alpha-beta'],
                           ['Third', 'alpha-beta']], 'count': 3}
    await sweep(rag, dry_run=False)
    assert rag.full_relations.rows['affected'] == {
        'relation_pairs': [['Alpha Beta', 'Other'], ['Third', 'Alpha Beta']], 'count': 2}
