"""Background facade tests removed in D13 (2026-05).

``materialize_paper`` (and its ``background`` module) was deleted along
with the unused ``add_and_wait`` / per-key completion-Future machinery on
the stage queues. Foreground ``get_paper`` is read-only and enqueues
download/extract work fire-and-forget via the queues' ``add()``. The
equivalent coverage now lives in ``tests/test_download_queue.py``,
``tests/test_extract_queue.py``, and ``tests/test_mcp_server.py`` (which
exercise the full flow via the MCP tools).
"""
