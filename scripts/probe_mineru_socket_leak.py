#!/usr/bin/env python3
"""Live socket-leak probe for the MinerU OCR path (issue #93).

Reproduces the outbound-HTTP socket leak and verifies the fix, by exercising the
REAL ``papervault.library.mineru_client.extract_mineru`` code path against a live
MinerU vLLM server and counting the process's sockets to that server via
``/proc/self/fd`` after each call.

Root cause (issue #93): ``extract_mineru`` delegates the HTTP to mineru's
``aio_do_parse``, which drives a PROCESS-GLOBAL singleton ``HttpVlmClient`` that
caches one ``httpx.AsyncClient`` PER EVENT LOOP and, on every loop switch, drops
the other loop's client WITHOUT ``aclose()``
(``mineru_vl_utils/vlm_client/http_client.py``: ``_aio_client_cache.clear()``).
Each dropped client keeps its pooled keepalive sockets to the server OPEN. So the
leak manifests only when OCR touches MORE THAN ONE event loop in a process — this
probe alternates two loops (a fresh ``asyncio.run`` turn per call, like a
transient loop beside the daemon's persistent one) that share the singleton.

Modes (same script — the fix lives in ``extract_mineru``):
  * default (PAPER_LIBRARY_MINERU_CLIENT_CLOSE=1): sockets stay FLAT (each loop's
    client is aclosed + evicted when its parse finishes).  -> AFTER
  * PAPER_LIBRARY_MINERU_CLIENT_CLOSE=0: the fix is disabled, reproducing the
    pre-#93 leak (sockets grow ~N per loop switch).        -> BEFORE

Usage (from the papervault repo root, with the runtime venv):
    PYTHONPATH=src python scripts/probe_mineru_socket_leak.py            # after (flat)
    PROBE_N=12 python scripts/probe_mineru_socket_leak.py
    PAPER_LIBRARY_MINERU_CLIENT_CLOSE=0 python scripts/probe_mineru_socket_leak.py  # before (leaks)

Env:
    MINERU_URL   MinerU endpoint (default http://127.0.0.1:30000; only :30000 is counted)
    PROBE_N      number of alternating-loop calls (default 12)

The probe sends a tiny synthetic 2-page PDF, so it is light on the GPU; it shares
the server with any live queue, so socket counts reflect THIS process only
(``/proc/self/fd`` is process-local). NEVER prints .env or secrets.
"""
from __future__ import annotations

import asyncio
import io
import os
import sys
import threading

_PORT = 30000  # only sockets to this remote port are counted (the MinerU server)
N = int(os.environ.get("PROBE_N", "12"))


def _rem_port_by_inode() -> dict[str, int]:
    """socket inode -> remote TCP port, parsed from /proc/net/tcp{,6}."""
    out: dict[str, int] = {}
    for path in ("/proc/net/tcp", "/proc/net/tcp6"):
        try:
            with open(path) as fh:
                lines = fh.read().splitlines()[1:]
        except OSError:
            continue
        for ln in lines:
            f = ln.split()
            if len(f) < 10:
                continue
            try:
                out[f[9]] = int(f[2].split(":")[1], 16)
            except (IndexError, ValueError):
                continue
    return out


def snapshot() -> tuple[int, int]:
    """(total open fds, sockets whose remote peer port == _PORT) for THIS process."""
    fd_dir = "/proc/self/fd"
    try:
        fds = os.listdir(fd_dir)
    except OSError:
        return (-1, -1)
    inode_port = _rem_port_by_inode()
    to_server = 0
    for fd in fds:
        try:
            tgt = os.readlink(os.path.join(fd_dir, fd))
        except OSError:
            continue
        if tgt.startswith("socket:[") and inode_port.get(tgt[len("socket:["):-1]) == _PORT:
            to_server += 1
    return (len(fds), to_server)


def make_pdf(n_pages: int = 2) -> bytes:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    for p in range(n_pages):
        c.setFont("Helvetica", 14)
        c.drawString(72, 720, f"Socket-leak probe page {p + 1}")
        c.drawString(72, 700, "The quick brown fox jumps over the lazy dog.")
        c.showPage()
    c.save()
    return buf.getvalue()


def _run_one_on_fresh_loop(pdf: bytes, stem: str) -> str:
    """Run ONE extract_mineru on a brand-new event loop, then close it — mirrors a
    transient ``asyncio.run`` turn sharing mineru's process-global singleton."""
    from papervault.library.mineru_client import (
        MineruExtractionError,
        MineruTransportError,
        endpoints_from_env,
        extract_mineru,
    )

    async def _go() -> str:
        try:
            md = await extract_mineru(pdf, endpoints_from_env(), stem=stem)
            return f"ok({len(md)}B)"
        except (MineruTransportError, MineruExtractionError) as e:
            return f"{type(e).__name__}"

    return asyncio.run(_go())


def main() -> int:
    pdf = make_pdf(2)
    mode = "BEFORE (leak)" if os.environ.get(
        "PAPER_LIBRARY_MINERU_CLIENT_CLOSE") == "0" else "AFTER (fixed)"
    print(f"probe_mineru_socket_leak: N={N} pdf={len(pdf)}B port={_PORT} mode={mode}",
          flush=True)
    base_fd, base_srv = snapshot()
    print(f"[baseline] fd_total={base_fd} sockets_to_server={base_srv}", flush=True)

    series: list[tuple[int, int]] = []
    for i in range(N):
        box: dict = {}
        stem = f"probe_{i}"
        # Alternate event loops: even i on THIS thread's fresh loop, odd i on a
        # SEPARATE thread's fresh loop. Both share mineru's global singleton, so
        # each step is a loop switch — the condition that triggers the leak.
        if i % 2 == 0:
            box["s"] = _run_one_on_fresh_loop(pdf, stem)
        else:
            t = threading.Thread(
                target=lambda: box.__setitem__("s", _run_one_on_fresh_loop(pdf, stem)))
            t.start()
            t.join()
        fd, srv = snapshot()
        series.append((fd, srv))
        print(f"[call {i:02d} {'main' if i % 2 == 0 else 'thrd'}] "
              f"fd_total={fd} sockets_to_server={srv}  {box.get('s')}", flush=True)

    if len(series) >= 3:
        d_fd = series[-1][0] - series[1][0]
        d_srv = series[-1][1] - series[1][1]
        span = len(series) - 2
        print(f"\n[slope over calls 1..{len(series) - 1}] "
              f"fd_total {d_fd:+d} ({d_fd / span:+.2f}/call), "
              f"sockets_to_server {d_srv:+d} ({d_srv / span:+.2f}/call)", flush=True)
        print("EXPECT: ~0.0/call FIXED (default); a clear positive slope with "
              "PAPER_LIBRARY_MINERU_CLIENT_CLOSE=0.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
