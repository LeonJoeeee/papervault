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

from papervault.library.mcp.boot import start_background
from papervault.mcp.server import build_server

log = logging.getLogger("test.boot")


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
    must cancel the boot cleanly and not raise — the zombie-boot guard."""

    async def scenario():
        server = build_server(library_path=str(tmp_path))
        eq = server._paper_extract_queue

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

    asyncio.run(scenario())
