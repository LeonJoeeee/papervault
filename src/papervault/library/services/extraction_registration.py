"""Extraction completion hooks; delivery retries never execute transformations."""

from __future__ import annotations

import hashlib
import json
import os
from contextlib import nullcontext
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from uuid import uuid4

from filelock import FileLock

from papervault.store.extraction import ExtractionStore, fsync_directory, register_completion


_IDENTITY_FIELDS = {"key", "doi", "arxiv_id", "paper_id", "title", "authors", "year"}


class StaleExtraction(RuntimeError):
    """Retained output is valid evidence but cannot advance the current record."""


def _authority(library):
    from papervault.store.knowledge import KnowledgeRepository
    from papervault.store.knowledge_runtime import KnowledgeAuthorityError, knowledge_jobs_mode

    backend = library._catalog_backend
    if backend is None:
        return {"authority": "legacy", "authority_epoch": None}
    with backend._repository() as repository:
        runtime = KnowledgeRepository(repository).get_authority_state()
    expected = "catalog" if knowledge_jobs_mode() == "postgres" else "legacy"
    if runtime["authority"] != expected:
        raise KnowledgeAuthorityError("extraction selector differs from committed knowledge authority")
    return runtime


def _write_lock(library):
    return library._catalog_save_lock or nullcontext()


def effective_recipe(engine: str, *, request=None, llm=None) -> dict:
    from .. import extract, mineru_client, llm as llm_module
    from papervault import config
    from papervault.llm_routing import route

    packages = {}
    for name in ("papervault", "mineru", "pypdf"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = "unknown"
    root = Path(__file__).parents[1]
    sources = (("extract.py", "mineru_client.py") if engine == "mineru2.5-pro" else (
        "download_sources/firecrawl.py", "download_sources/_shared.py", "extract.py")) + ("llm.py",)
    return {"schema_version": 1, "name": "papervault-extraction", "engine": engine,
            "configured_model": "MinerU2.5-Pro-2605-1.2B" if engine == "mineru2.5-pro" else "unknown",
            "packages": packages,
            "code_sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in sources},
            "settings": {"whole_document": True, "formula_enable": True, "table_enable": True,
                         "image_analysis": False, "blank_floor": extract._BLANK_CHUNK_FLOOR,
                         "ceiling_seconds": extract._CEILING_SECONDS,
                         "http_timeout": mineru_client._MINERU_HTTP_TIMEOUT,
                         "native_retries": mineru_client._MINERU_INLINE_NATIVE_RETRIES,
                         "native_backoff_factor": mineru_client._MINERU_NATIVE_BACKOFF_FACTOR,
                         "inline_retries": mineru_client._INLINE_RETRIES,
                         "backoff_base": mineru_client._BACKOFF_BASE,
                         "backoff_cap": mineru_client._BACKOFF_CAP,
                         "client_limits": list(mineru_client.client_limits())},
            "gate": {"configured_model": getattr(llm, "model", "unknown") if llm else route("gate"),
                     "max_tokens": getattr(llm, "max_tokens", "unknown") if llm else llm_module._DEFAULT_MAX_TOKENS,
                     "first_chunk_seconds": llm_module._FIRST_CHUNK_S,
                     "retry_budget_seconds": llm_module._RETRY_BUDGET_S,
                     "client_timeout_seconds": llm_module._CLIENT_TIMEOUT_S,
                     "gateway_mode": config.USE_GATEWAY,
                     "actual_provider_model": "unknown"},
            "request": request or {},
            "output_contract": {"fulltext": {"required": ["markdown"], "not_applicable": []}}}


def begin_extraction(paper, library, source: bytes, *, engine="mineru2.5-pro",
                     source_role="pdf", request=None, llm=None):
    from ..models import Paper

    authority_library = getattr(library, "_extraction_live", library)
    backend = authority_library._catalog_backend
    with _write_lock(authority_library):
        authority = _authority(authority_library)
        if authority_library is library:
            # Acquisition can correct identity before its queue's final save.
            # Commit that state before binding evidence to a catalog revision;
            # the same durable boundary makes JSON crash recovery coherent.
            library.save(force=True, durable=True)
        record = backend.persisted.get(paper.key) if backend else None
        if backend:
            if record is None:
                raise StaleExtraction("extraction record is not persisted before capture")
            unit_paper = Paper.model_validate(record.payload)
        else:
            # JSON saves rotate the primary before replacing it. Read under the
            # same writer lock, after save() has released its own lock instance.
            with FileLock(str(authority_library.root / ".write.lock"), timeout=30):
                unit_paper = Paper.model_validate(json.loads(
                    authority_library.index_path.read_bytes())["papers"][paper.key])
        identity = unit_paper.model_dump(mode="json", include=_IDENTITY_FIELDS)
        # Detach all captured values from the mutable queue object. A mutation
        # during save cannot pair a newer identity with the acknowledged revision.
        paper = paper.model_copy(deep=True)
        if authority_library is library and paper.model_dump(mode="json", include=_IDENTITY_FIELDS) != identity:
            raise StaleExtraction("extraction identity changed across capture persistence")
        alias = library.md_path(paper.key)
        guard = {"previous_completion_id": paper.extraction_completion_id,
                 "previous_output_sha256": hashlib.sha256(alias.read_bytes()).hexdigest() if alias.is_file() else None,
                 "identity": paper.model_dump(mode="json", include=_IDENTITY_FIELDS)}
        if authority["authority"] == "catalog":
            from papervault.store.knowledge import KnowledgeRepository
            with backend._repository() as repository:
                knowledge = KnowledgeRepository(repository)
                unit = knowledge.get_unit_for_record(paper.key)
                head = knowledge.get_extraction_head(unit["unit_id"]) if unit else None
            # Merge adoption can select another unit's output without changing this
            # unit's catalog head. Capture both predecessors, including absent heads.
            guard["catalog_previous_completion_id"] = head["completion_id"] if head else None
        # Staged verified fills remain an explicit transition: consumed metadata
        # can differ from the persisted unit until guarded writeback accepts it.
        captured = ExtractionStore(library.root).capture(
            record_key=paper.key, metadata=paper.model_dump(mode="json"), source=source,
            source_role=source_role, recipe=effective_recipe(engine, request=request, llm=llm), guard=guard,
            unit_evidence={"kind": "paper", "record_key": paper.key, "identity": identity},
            source_catalog_revision=record.revision if record else None,
            authority_epoch=authority["authority_epoch"] if authority["authority"] == "catalog" else None)
        if authority_library is not library:
            captured["staged_claim"] = paper.reopen_claim.model_dump(mode="json")
        return captured


def _check_current(library, paper, completion, *, registered_head=False):
    if paper.model_dump(mode="json", include=_IDENTITY_FIELDS) != completion["guard"]["identity"]:
        raise StaleExtraction("extraction identity changed during execution")
    if not registered_head and paper.extraction_completion_id not in {
            completion["guard"]["previous_completion_id"], completion["completion_id"]}:
        raise StaleExtraction("extraction completion is stale")
    if paper.extraction_completion_id == completion["completion_id"]:
        return  # Already projected: a subsequent source revision cannot rewrite its history.
    alias = library.md_path(paper.key)
    current_hash = hashlib.sha256(alias.read_bytes()).hexdigest() if alias.is_file() else None
    if not registered_head and current_hash not in {
            completion["guard"].get("previous_output_sha256"), completion["output_inventory"][0]["sha256"]}:
        raise StaleExtraction("extraction output changed during execution")
    source = completion["input_manifest"]["artifacts"][0]
    if source["role"] == "pdf":
        path = library.pdf_path(paper.key)
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != source["sha256"]:
            raise StaleExtraction("extraction source changed during execution")


def deliver_completion(library, completion):
    """Legacy/standalone outputs have durable evidence but no publication credit."""
    from papervault.store.knowledge import KnowledgeRepository
    from papervault.store.knowledge_runtime import knowledge_jobs_mode

    store = ExtractionStore(library.root)
    store.verify(completion)
    if (library._catalog_backend is None or hasattr(library, "_extraction_live")
            or knowledge_jobs_mode() == "legacy"):
        if not hasattr(library, "_extraction_live"):
            _authority(library)
        return {"registration": "pending", "completion_id": completion["completion_id"]}
    with library._catalog_backend._repository() as repository:
        return register_completion(KnowledgeRepository(repository), library.root, completion)


def require_selected_registration(library, paper):
    """An existing pending alias cannot become catalog success through a shortcut."""
    if hasattr(library, "_extraction_live") or _authority(library)["authority"] != "catalog":
        return
    if paper.extraction_completion_id and paper.md_path:
        from papervault.store.knowledge import KnowledgeRepository
        completion = ExtractionStore(library.root).completion(paper.extraction_completion_id)
        deliver_completion(library, completion)
        with library._catalog_backend._repository() as repository:
            knowledge = KnowledgeRepository(repository)
            knowledge.extraction_origin(paper.key, knowledge.get_extraction_receipt(paper.extraction_completion_id))


def _project(library, paper, completion):
    output = completion["output_inventory"][0]
    data = ExtractionStore(library.root).read_artifact(output)
    # Always write the conventional compatibility projection, never a version.
    path = library.md_dir / (paper.key + ".md")
    temporary = path.parent / ("." + path.name + "." + uuid4().hex)
    try:
        with temporary.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)
    paper.md_path = output["path"]
    paper.extraction_completion_id = completion["completion_id"]
    paper.md_engine = completion["recipe"]["engine"]
    paper.md_engine_version = ("MinerU2.5-Pro-2605-1.2B" if paper.md_engine == "mineru2.5-pro" else "v1")
    paper.download_status = "ok"
    if paper.md_engine == "firecrawl":
        paper.download_source = "firecrawl"
        paper.firecrawl_pdf_hunt_exhausted = True


def finish_extraction(paper, library, request, text, *, provider_observations=None):
    completion = ExtractionStore(library.root).finalize(
        request, text.encode(), provider_observations=provider_observations)
    with _write_lock(library), FileLock(str(library.root / ".extraction-write.lock")):
        _check_current(library, paper, completion)
        deliver_completion(library, completion)
        _project(library, paper, completion)
    return completion


def adopt_staged_completion(paper, library, staged, completion):
    with _write_lock(library), FileLock(str(library.root / ".extraction-write.lock")):
        ExtractionStore(library.root).retain_from(ExtractionStore(staged.root), completion)
        _check_current(library, paper, completion)
        deliver_completion(library, completion)
        _project(library, paper, completion)


def _registered_head_status(library, completion) -> bool | None:
    """True for the current receipt, False for superseded, None for unregistered."""
    from papervault.store.knowledge import KnowledgeRepository

    if hasattr(library, "_extraction_live") or _authority(library)["authority"] != "catalog":
        return None
    with library._catalog_backend._repository() as repository:
        knowledge = KnowledgeRepository(repository)
        receipt = knowledge.get_extraction_receipt(completion["completion_id"])
        if receipt is None:
            return None
        if receipt["receipt"] != completion:
            raise ValueError("registered extraction differs from retained completion")
        head = knowledge.get_extraction_head(receipt["unit_id"])
        return bool(head and head["completion_id"] == completion["completion_id"])


def recover_completions(library, *, key=None):
    from papervault.store.knowledge import ClaimRejected

    recovered, blocked = 0, []
    for completion in ExtractionStore(library.root).completions(key=key):
        paper = library.get(completion["record_key"])
        if paper is None:
            continue
        try:
            with _write_lock(library), FileLock(str(library.root / ".extraction-write.lock")):
                registered_head = _registered_head_status(library, completion)
                # A committed head survives a lost reply and merge adoption.
                # Superseded receipts remain history even if their old local
                # predecessor still matches after another lost reply.
                if registered_head is False:
                    continue
                if registered_head is not True and paper.extraction_completion_id not in {
                        completion["guard"]["previous_completion_id"], completion["completion_id"]}:
                    continue
                ExtractionStore(library.root).verify(completion)
                claim = completion.get("staged_claim")
                if claim and (not paper.reopen_claim or paper.reopen_claim.phase != "completed"
                              or paper.reopen_claim.lease_id != claim["lease_id"]):
                    raise StaleExtraction("staged extraction requires accepted guarded writeback")
                _check_current(library, paper, completion, registered_head=registered_head is True)
                deliver_completion(library, completion)
                if paper.extraction_completion_id != completion["completion_id"]:
                    _project(library, paper, completion)
                    recovered += 1
        except (StaleExtraction, ClaimRejected) as exc:
            blocked.append({"completion_id": completion["completion_id"], "reason": str(exc)})
        except (OSError, ValueError):
            if key is not None:
                raise
            blocked.append({"completion_id": completion["completion_id"], "reason": "invalid retained evidence"})
    return {"recovered": recovered, "blocked": blocked}
