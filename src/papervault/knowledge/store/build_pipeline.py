"""Breaker deferral at LightRAG 1.5.4's document failure boundary (#148).

The stock pipeline consumes per-document exceptions before callers can catch them.
This adapter preserves its workers, task cleanup and ordinary failure handling, but
writes PENDING for a paused document and drains siblings without running extraction.
Private hooks require an explicit port whenever the installed/locked version changes.
"""
from __future__ import annotations

import asyncio
import importlib.metadata

from lightrag import LightRAG
from lightrag.base import DocStatus

from papervault.knowledge.store.build_breaker import (
    BuildPausedError,
    get_breaker,
    is_upstream_failure,
)

_PAUSED = "papervault_build_paused"
_SUPPORTED_VERSION = "1.5.4"


def is_build_paused(error: BaseException) -> bool:
    """LightRAG may prefix/wrap an extraction exception; follow its preserved cause."""
    seen = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        if isinstance(error, BuildPausedError):
            return True
        error = error.__cause__
    return False


class BuildAwareLightRAG(LightRAG):
    def __post_init__(self, addon_params=None):
        installed = importlib.metadata.version("lightrag-hku")
        if installed != _SUPPORTED_VERSION:
            raise RuntimeError(
                f"papervault build-pause adapter requires lightrag-hku=={_SUPPORTED_VERSION}; "
                f"found {installed}. Port the private pipeline hooks before upgrading.")
        super().__post_init__(addon_params)

    async def _run_pipeline_batch(self, to_process_docs, *, pipeline_status, pipeline_status_lock):
        async with pipeline_status_lock:
            pipeline_status[_PAUSED] = False
        await super()._run_pipeline_batch(
            to_process_docs, pipeline_status=pipeline_status,
            pipeline_status_lock=pipeline_status_lock)
        if pipeline_status.get(_PAUSED):
            # All workers have wound down and every queue join completed. Stop the outer
            # pipeline from consuming request_pending and starting another batch this round.
            raise BuildPausedError("build batch deferred until the next scheduler round")

    async def process_single_document(self, *, doc_id, status_doc, parsed_data, ctx):
        if ctx.pipeline_status.get(_PAUSED) or not get_breaker().allows():
            await self._defer_document(
                doc_id=doc_id, status_doc=status_doc, file_path=status_doc.file_path,
                pipeline_status=ctx.pipeline_status, pipeline_status_lock=ctx.pipeline_status_lock,
                failed_chunks_snapshot=(status_doc.chunks_list or [], status_doc.chunks_count or 0))
            return
        await super().process_single_document(
            doc_id=doc_id, status_doc=status_doc, parsed_data=parsed_data, ctx=ctx)

    async def _finalize_doc_failure(self, **kwargs):
        error = kwargs["error"]
        if not (is_build_paused(error) or
                (is_upstream_failure(error) and not get_breaker().allows())):
            await super()._finalize_doc_failure(**kwargs)
            return
        tasks = [t for t in kwargs["pending_tasks"] if t is not None]
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        await self._defer_document(
            doc_id=kwargs["doc_id"], status_doc=kwargs["status_doc"],
            file_path=kwargs["file_path"], pipeline_status=kwargs["pipeline_status"],
            pipeline_status_lock=kwargs["pipeline_status_lock"],
            failed_chunks_snapshot=kwargs["failed_chunks_snapshot"])

    async def _defer_document(self, *, doc_id, status_doc, file_path, pipeline_status,
                              pipeline_status_lock, failed_chunks_snapshot):
        async with pipeline_status_lock:
            pipeline_status[_PAUSED] = True
        chunks, count = failed_chunks_snapshot
        await self._upsert_doc_status_transition(
            doc_id=doc_id, status=DocStatus.PENDING, status_doc=status_doc, file_path=file_path,
            extra_fields={"error_msg": "", "chunks_list": chunks, "chunks_count": count})
