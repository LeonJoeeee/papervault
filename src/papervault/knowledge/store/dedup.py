"""Surface-form entity de-duplication for the KS graph — the corrected, type-aware Route B sweep.

Background (2026-06-06 entity-dedup-sweep decision, §9-§11): LightRAG keys entities by
the case/whitespace/hyphen-preserving name, so surface variants ("Physics-Informed Neural Network" vs
"Physics Informed Neural Network" vs "PINN") become distinct nodes. A naive blind merge by normalized key
is UNSAFE — it fused distinct entities that collide by case (NUCLEON-experiment vs Nucleon-particle,
GARNET-mineral vs GarNet-ML-layer) and it mislabels the survivor's entity_type (LightRAG `keep_first`).

This module does the SAFE version:
  - cluster by normalize_key (lowercase + strip space/hyphen/underscore), key-len >= MIN_KEY_LEN, size>1;
  - SINGLE-entity_type clusters -> auto-merge (native amerge_entities) with the survivor type set EXPLICITLY;
  - MULTI-entity_type clusters -> NOT auto-merged; emitted for Claude adjudication (the remediate workflow),
    because type disagreement is a (noisy) signal that distinct referents may have collided.

Merge is native `rag.amerge_entities` (0 LLM tokens; local BGE-M3 re-embed only). Always backs up first.
The KS service MUST be down during a sweep (the merge lock is in-process only — no cross-process exclusion).
Restore of a bad merge: experiments/restore_wrongmerges.py (recreates members + edges from the backup).
"""
from __future__ import annotations

import collections
import json
import re
from dataclasses import dataclass, field

MIN_KEY_LEN = 6


def normalize_key(name: str) -> str:
    """Conservative surface-form key. NOT a semantic identity — see module docstring."""
    import unicodedata
    k = unicodedata.normalize("NFKC", name.casefold())
    return re.sub(r"[\s\-_]+", "", k)


@dataclass
class SweepStats:
    nodes_before: int = 0
    nodes_after: int = 0
    clusters_total: int = 0
    single_type_merged: int = 0
    multi_type_deferred: int = 0
    merge_errors: int = 0
    collision_before: float = 0.0
    collision_after: float = 0.0
    deferred_clusters: list = field(default_factory=list)  # [{key, members:[{name,type}]}]


async def _all_nodes(rag):
    """Return {name: entity_type} for all nodes on the active workspace (batched, via the graph storage)."""
    g = rag.chunk_entity_relation_graph
    labels = await g.get_all_labels()  # distinct entity_id strings
    out = {}
    for i in range(0, len(labels), 1000):
        batch = labels[i:i + 1000]
        nodes = await g.get_nodes_batch(batch)  # {name: props}
        for nm in batch:
            out[nm] = (nodes.get(nm) or {}).get("entity_type") or "UNKNOWN"
    return out


async def _node_degree(rag, name: str) -> int:
    try:
        return await rag.chunk_entity_relation_graph.node_degree(name)
    except Exception:
        return 0


def find_clusters(name_to_type: dict[str, str], min_key_len: int = MIN_KEY_LEN) -> dict[str, list[str]]:
    by = collections.defaultdict(list)
    for nm in name_to_type:
        if nm:
            by[normalize_key(nm)].append(nm)
    return {k: v for k, v in by.items() if len(k) >= min_key_len and len(v) > 1}


def _collision_rate(name_to_type: dict[str, str], min_key_len: int = MIN_KEY_LEN) -> float:
    clusters = find_clusters(name_to_type, min_key_len)
    n = len(name_to_type) or 1
    return sum(len(v) for v in clusters.values()) / n


async def backup_members(rag, names: list[str], path: str) -> int:
    """Dump node props + edges for `names` to JSONL (for restore). Returns count backed up.

    Uses the graph storage's get_node + get_node_edges so it is backend-agnostic.
    """
    g = rag.chunk_entity_relation_graph
    backed = 0
    with open(path, "w") as f:
        for nm in names:
            node = await g.get_node(nm)
            if node is None:
                continue
            edges = []
            try:
                for (a, b) in (await g.get_node_edges(nm)) or []:
                    ed = await g.get_edge(a, b)
                    edges.append({"src": a, "dst": b, "rp": ed or {}})
            except Exception:
                pass
            f.write(json.dumps({"name": nm, "props": node, "edges": edges}, ensure_ascii=False) + "\n")
            backed += 1
    return backed


async def sweep(rag, *, min_key_len: int = MIN_KEY_LEN, backup_path: str | None = None,
                deferred_path: str | None = None, dry_run: bool = True,
                only_names: set[str] | None = None,
                exclude_keys: set[str] | None = None) -> SweepStats:
    """Run one type-aware dedup pass.

    only_names: if given, restrict to clusters that contain at least one of these names
                (incremental dedup after a top-up — process only NEW collisions).
    exclude_keys: normalized keys to SKIP entirely (neither merge nor defer). Pass already-adjudicated
                  split keys here so a later run never re-merges a deliberately-split cluster (e.g. a split
                  whose two groups happen to share an entity_type would otherwise be re-merged as single-type).
    """
    name_to_type = await _all_nodes(rag)
    st = SweepStats(nodes_before=len(name_to_type))
    st.collision_before = _collision_rate(name_to_type, min_key_len)
    clusters = find_clusters(name_to_type, min_key_len)
    if exclude_keys:
        clusters = {k: v for k, v in clusters.items() if k not in exclude_keys}
    if only_names is not None:
        clusters = {k: v for k, v in clusters.items() if any(n in only_names for n in v)}
    st.clusters_total = len(clusters)

    single, multi = [], []
    for k, members in clusters.items():
        types = {name_to_type[m] for m in members if name_to_type[m] != "UNKNOWN"}
        (single if len(types) <= 1 else multi).append((k, members, types))

    # defer multi-type for adjudication (never auto-merge — the unsafe case)
    st.multi_type_deferred = len(multi)
    st.deferred_clusters = [{"key": k, "members": [{"name": m, "type": name_to_type[m]} for m in members]}
                            for (k, members, _t) in multi]
    if deferred_path and st.deferred_clusters:
        with open(deferred_path, "w") as f:
            for c in st.deferred_clusters:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")

    if dry_run:
        st.nodes_after = st.nodes_before
        st.collision_after = st.collision_before
        return st

    # backup every member we are about to touch (single-type only)
    touch = [m for (_k, members, _t) in single for m in members]
    if backup_path and touch:
        backed = await backup_members(rag, touch, backup_path)
        if backed != len(touch):
            raise RuntimeError(f"backup incomplete {backed}!={len(touch)} — aborting before any merge")

    for (_k, members, types) in single:
        target = max(members, key=lambda m: 0)  # placeholder; replaced below by degree
        # choose highest-degree member as canonical target
        degs = {m: await _node_degree(rag, m) for m in members}
        target = max(members, key=lambda m: degs[m])
        sources = [m for m in members if m != target]
        the_type = next(iter(types)) if types else (name_to_type[target] or "UNKNOWN")
        try:
            await rag.amerge_entities(source_entities=sources, target_entity=target,
                                      target_entity_data={"entity_type": the_type})
            st.single_type_merged += 1
        except Exception:
            st.merge_errors += 1

    after = await _all_nodes(rag)
    st.nodes_after = len(after)
    st.collision_after = _collision_rate(after, min_key_len)
    return st
