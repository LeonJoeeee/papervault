"""Boot-path tests for the unified library background boot (issue #34 remaining
scope: bind the port BEFORE the reconcile/backlog sweep).

``library/mcp/boot.py::start_background`` used to run the long backlog recovery
(a ``pdf_probe`` subprocess per EXTRACT paper + the first reconcile sweep)
SYNCHRONOUSLY, inside the ASGI lifespan startup — and an ASGI server does not
bind its socket until lifespan-startup returns, so the port stayed unbound for
the whole sweep. The fix moves that work to a detached background task so
``start_background`` returns promptly (the lifespan yields → uvicorn binds).

These are structural/unit checks: they assert the *ordering* (start_background
returns before the recovery scan runs) without needing a real uvicorn bind.
"""

from __future__ import annotations

import asyncio
import logging
import threading

from papervault.library.mcp.boot import start_background
from papervault.library.store import Library
from papervault.mcp.server import build_server

log = logging.getLogger("test.boot")


def _make_extract_paper(tmp_path) -> None:
    """Seed the vault at ``tmp_path`` with ONE EXTRACT-routed paper (status ok +
    a PDF on disk + no md), so the boot recovery scan enqueues it and a worker
    reaches the MinerU extraction path."""
    lib = Library(tmp_path)
    p, _ = lib.upsert({
        "title": "Boot resilience paper ok pdf no md for extract recovery",
        "authors": ["ROne"], "year": 2020,
    })
    p.download_status = "ok"
    p.download_source = "fake"
    pdf = lib.pdf_path(p.key)
    pdf.parent.mkdir(parents=True, exist_ok=True)
    pdf.write_bytes(b"%PDF-1.4\nfake")
    lib.save()


def test_start_background_returns_before_recovery_scan(tmp_path):
    """The recovery scan (extract_queue.start) must NOT run before
    start_background returns — proving the port can bind first (issue #34)."""

    async def scenario():
        server = build_server(library_path=str(tmp_path))
        extract_queue = server._paper_extract_queue

        gate = asyncio.Event()          # holds the recovery scan open
        scan_started = asyncio.Event()  # signals the scan actually began
        orig_start = extract_queue.start

        async def gated_start():
            scan_started.set()
            await gate.wait()           # block the "sweep" until the test releases it
            await orig_start()

        extract_queue.start = gated_start  # type: ignore[method-assign]

        # start_background must return WITHOUT waiting for the (blocked) scan.
        shutdown = await asyncio.wait_for(start_background(server, log), timeout=2.0)

        # The port would be bound here. The sweep has NOT completed:
        assert extract_queue._started is False
        boot_task = server._paper_boot_task
        assert not boot_task.done()

        # Release the gate → the background boot finishes the recovery scan.
        gate.set()
        await asyncio.wait_for(scan_started.wait(), timeout=2.0)
        await asyncio.wait_for(boot_task, timeout=5.0)
        assert extract_queue._started is True

        await shutdown()

    asyncio.run(scenario())


def test_boot_task_starts_queues_and_shutdown_is_clean(tmp_path):
    """The detached boot task starts both queues; shutdown tears everything
    down without error (including the still-running reconcile loop)."""

    async def scenario():
        server = build_server(library_path=str(tmp_path))
        dq = server._paper_download_queue
        eq = server._paper_extract_queue

        shutdown = await start_background(server, log)
        await asyncio.wait_for(server._paper_boot_task, timeout=5.0)

        assert eq._started is True
        assert dq._started is True

        await shutdown()
        # After shutdown the queues are stopped and the boot task is finished.
        assert eq._started is False
        assert dq._started is False

    asyncio.run(scenario())


def test_shutdown_before_boot_completes_is_clean(tmp_path):
    """A shutdown that races an in-flight boot (port bound, sweep still running)
    must cancel the boot cleanly and not raise — the zombie-boot guard.

    This ALSO pins the shutdown-robustness fix: the boot is cancelled while blocked
    BEFORE either queue's ``start()`` ran, so shutdown calls ``stop()`` on two
    NEVER-STARTED queues. The old code gated those stops behind a coarse
    ``queues_started`` flag; shutdown now calls them UNCONDITIONALLY, relying on
    ``stop()`` being idempotent-safe when unstarted (empty worker pool → no-op)."""

    async def scenario():
        server = build_server(library_path=str(tmp_path))
        eq = server._paper_extract_queue
        dq = server._paper_download_queue

        gate = asyncio.Event()
        orig_start = eq.start

        async def gated_start():
            await gate.wait()
            await orig_start()

        eq.start = gated_start  # type: ignore[method-assign]

        shutdown = await start_background(server, log)
        # Boot is blocked mid-sweep; shut down anyway (do not release the gate).
        await asyncio.wait_for(shutdown(), timeout=5.0)
        assert server._paper_boot_task.done()
        # Neither queue ever started, yet the unconditional stop() left both in a
        # clean, un-started state without raising.
        assert eq._started is False
        assert dq._started is False

    asyncio.run(scenario())


def test_bind_not_gated_on_mineru_reachability(tmp_path):
    """A down/hung MinerU (OCR at :30000 refused or the readiness probe blocking)
    must NOT gate the port bind (issue #34): ``start_background`` returns promptly
    and the detached boot COMPLETES even while the MinerU readiness probe blocks
    forever. Extraction stays degraded (the worker handles MinerU-down per-item,
    C1) but the bind/boot never wait on it.

    The readiness probe is replaced with one that HANGS (never returns), so if the
    boot/bind path awaited MinerU readiness anywhere this test would time out. It
    does not: MinerU is touched only on the worker critical path, never on the
    bind path. Seed one EXTRACT-routed paper so the recovery scan has real work.
    """
    from papervault.library.services import mineru_server

    _make_extract_paper(tmp_path)

    async def scenario():
        mineru_server.reset_server_controller_for_test()
        ctrl = mineru_server.get_server_controller()

        release = asyncio.Event()   # never set during the boot window → probe hangs

        async def hanging_ensure_ready() -> None:
            await release.wait()     # simulate :30000 unreachable / vLLM still loading

        ctrl.ensure_ready = hanging_ensure_ready  # type: ignore[method-assign]
        try:
            server = build_server(library_path=str(tmp_path))
            eq = server._paper_extract_queue

            # The bind point (lifespan yield) is reached the instant this returns —
            # it must NOT wait on the hung MinerU probe.
            shutdown = await asyncio.wait_for(start_background(server, log), timeout=2.0)

            # The DETACHED boot completes (queues up, reconcile up) despite MinerU
            # being unreachable — the boot never awaits the extraction backend.
            await asyncio.wait_for(server._paper_boot_task, timeout=5.0)
            assert eq._started is True
            # MinerU-down is NOT a boot failure — reads stay healthy, only
            # extraction degrades, so no service_degraded signal is raised.
            assert server._paper_boot_failed is None

            release.set()  # unpark any worker that reached the probe before teardown
            await asyncio.wait_for(shutdown(), timeout=5.0)
        finally:
            release.set()
            mineru_server.reset_server_controller_for_test()

    asyncio.run(scenario())


def test_recovery_scan_probes_pdf_off_the_event_loop(tmp_path, monkeypatch):
    """The ``pdf_probe`` recovery scan (a BLOCKING ``subprocess.run``) must run
    OFF the event loop so the detached boot does not freeze serving during the
    sweep (issue #34 serve-during-sweep): the port binds, but an in-line probe
    would still leave uvicorn unable to answer for the whole scan. Assert the
    probe executes on a worker thread, not the loop's main thread.
    """
    from papervault.library import extract as extract_mod

    _make_extract_paper(tmp_path)

    probe_threads: list[str] = []

    def spy_probe(path, **kw):
        probe_threads.append(threading.current_thread().name)
        return extract_mod.PDFProbe(bad=False, n_pages=1, reason="ok")

    async def fake_extract_mineru(*a, **kw):
        # Keep the test hermetic — never touch a real :30000 if a worker ticks.
        raise extract_mod.MineruTransportError("test: mineru unreachable")

    monkeypatch.setattr(extract_mod, "pdf_probe", spy_probe)
    monkeypatch.setattr(extract_mod, "extract_mineru", fake_extract_mineru)

    async def scenario():
        server = build_server(library_path=str(tmp_path))
        eq = server._paper_extract_queue
        await eq.start()
        assert probe_threads, "recovery scan never probed the EXTRACT paper"
        assert all(t != threading.main_thread().name for t in probe_threads), (
            f"pdf_probe ran on the event-loop thread (blocks serving): {probe_threads}")
        await eq.stop()

    asyncio.run(scenario())
