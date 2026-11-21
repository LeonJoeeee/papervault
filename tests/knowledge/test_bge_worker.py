"""Tests for the on-demand local BGE embed & rerank worker subprocess (Plan B).

Runs completely offline with no GPU and no Hugging Face downloads via the
test-only `KS_BGE_WORKER_TEST_FAKE=1` hook. Verifies:
  - Binary-framed JSON + raw float32 RPC round-trips (strictly zero-pickle)
  - OS fd 1 + sys.stdout redirection protects binary framing from noisy libraries
  - Parent process never imports torch / transformers / sentence_transformers / FlagEmbedding
  - Real subprocess embed (exact float32 values + empty batch) and rerank (top_n + pool_cap)
  - Worker error propagation and rerank_failure_count() accounting
  - Thread-safe concurrent embed & rerank serialization
  - Idle unload terminates and reaps the worker PID; next call respawns
  - Worker crash recovery (both already-dead and mid-pipe race), bounded without infinite retry
  - Short-lived spawning thread exit does NOT kill the worker
  - Parent stdin EOF exits the worker cleanly with code 0 (zero orphans)
  - Seamless compatibility with KS_BGE_WORKER_MODE=inprocess and monkeypatched models/getters
"""
from __future__ import annotations

import asyncio
import io
import logging
import os
import struct
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from papervault.knowledge.store import bge_worker, lightrag_init


@pytest.fixture
def isolated_worker(monkeypatch):
    """Provide a fresh, isolated _BGEWorkerClient wired into lightrag_init with fake models."""
    monkeypatch.setenv("KS_BGE_WORKER_MODE", "subprocess")
    monkeypatch.setenv("KS_BGE_WORKER_TEST_FAKE", "1")
    monkeypatch.delenv("KS_BGE_WORKER_TEST_NOISY_STDOUT", raising=False)

    worker = bge_worker._BGEWorkerClient()
    monkeypatch.setattr(lightrag_init, "_BGE_WORKER", worker)
    monkeypatch.setattr(lightrag_init, "_BGE_MODEL", None)
    monkeypatch.setattr(lightrag_init, "_BGE_RERANKER", None)
    monkeypatch.setattr(lightrag_init, "_IDLE_UNLOAD_ENABLED", False)
    monkeypatch.setattr(lightrag_init, "_IDLE_UNLOAD_TIMEOUT", 600.0)
    monkeypatch.setattr(lightrag_init, "_EMBED_SEM", None)
    monkeypatch.setattr(lightrag_init, "_RERANK_SEM", None)
    lightrag_init.reset_rerank_failures(reset_sem=True)

    yield worker

    timer = lightrag_init._IDLE_UNLOAD_TIMER
    if timer is not None:
        timer.cancel()
        lightrag_init._IDLE_UNLOAD_TIMER = None
    worker.shutdown(timeout=5.0, force_if_busy=True)


# ---------------------------------------------------------------------------
# 1. Binary-framed RPC protocol (zero-pickle)
# ---------------------------------------------------------------------------

def test_framing_roundtrip_json_and_float32_payload():
    arr = np.array([[1.25, -0.5, 3.0], [0.0, 42.5, -7.75]], dtype=np.float32)
    raw = np.ascontiguousarray(arr, dtype=np.float32).tobytes()
    buf = io.BytesIO()

    bge_worker.write_frame(
        buf,
        {"id": 7, "ok": True, "shape": [2, 3], "dtype": "float32", "payload_bytes": len(raw)},
        raw,
    )
    wire = buf.getvalue()
    # Verify exact wire framing: 4-byte big-endian uint32 header_len + UTF-8 JSON + raw bytes.
    (header_len,) = struct.unpack(">I", wire[:4])
    assert wire[4 : 4 + header_len].startswith(b"{")
    assert wire[4 + header_len :] == raw

    buf.seek(0)
    frame = bge_worker.read_frame(buf)
    assert frame is not None
    header, payload = frame
    assert header["id"] == 7
    assert header["ok"] is True
    assert header["shape"] == [2, 3]
    assert header["payload_bytes"] == len(raw)
    reconstructed = np.frombuffer(payload, dtype=np.float32).reshape(header["shape"]).copy()
    np.testing.assert_array_equal(reconstructed, arr)
    # Clean EOF at next frame boundary returns None.
    assert bge_worker.read_frame(buf) is None


def test_framing_empty_payload_and_truncation_errors():
    buf = io.BytesIO()
    bge_worker.write_frame(buf, {"id": 1, "op": "ping"})
    buf.seek(0)
    header, payload = bge_worker.read_frame(buf)
    assert header == {"id": 1, "op": "ping"}
    assert payload == b""

    # Truncated length prefix (2 of 4 bytes) -> EOFError.
    with pytest.raises(EOFError):
        bge_worker.read_frame(io.BytesIO(b"\x00\x01"))

    # Truncated JSON header -> EOFError.
    with pytest.raises(EOFError):
        bge_worker.read_frame(io.BytesIO(struct.pack(">I", 20) + b'{"id":1'))

    # Truncated binary payload -> EOFError.
    short_hdr = b'{"id":1,"payload_bytes":16}'
    with pytest.raises(EOFError):
        bge_worker.read_frame(io.BytesIO(struct.pack(">I", len(short_hdr)) + short_hdr + b"1234"))

    # Invalid header_len=0 -> ValueError.
    with pytest.raises(ValueError):
        bge_worker.read_frame(io.BytesIO(struct.pack(">I", 0)))

    # Mismatched payload_bytes with empty payload -> ValueError.
    with pytest.raises(ValueError):
        bge_worker.write_frame(io.BytesIO(), {"id": 1, "payload_bytes": 8}, b"")


# ---------------------------------------------------------------------------
# 2. Stdout-redirection guard (Python sys.stdout + raw OS fd 1)
# ---------------------------------------------------------------------------

def test_worker_stdout_redirection_protects_rpc_framing(isolated_worker, monkeypatch):
    monkeypatch.setenv("KS_BGE_WORKER_TEST_NOISY_STDOUT", "1")

    vecs = lightrag_init._bge_encode_sync(["alpha", "beta gamma"])
    assert vecs.shape == (2, 1024)
    assert vecs.dtype == np.float32
    assert vecs[0, 0] == pytest.approx(5.0)
    assert vecs[1, 0] == pytest.approx(10.0)

    ranked = lightrag_init._bge_rerank_sync(
        "solar energetic particles",
        ["unrelated chunk", "solar energetic particles transport", "solar wind"],
        top_n=2,
    )
    assert [r["index"] for r in ranked] == [1, 2]
    assert ranked[0]["relevance_score"] == pytest.approx(1.0)
    assert ranked[1]["relevance_score"] == pytest.approx(1.0 / 3.0)


# ---------------------------------------------------------------------------
# 3. Parent-process torch isolation (clean child interpreter proof)
# ---------------------------------------------------------------------------

def test_parent_process_never_imports_torch_in_subprocess_mode():
    """Prove in a clean Python process that subprocess mode never imports ML stacks."""
    script = """
import asyncio
import sys
import time
import numpy as np
from papervault.knowledge.store import lightrag_init as li

li._IDLE_UNLOAD_ENABLED = True
li._IDLE_UNLOAD_TIMEOUT = 0.05

# Sync + async embed and rerank via real worker subprocess.
v1 = li._bge_encode_sync(["hello world", "space physics"])
assert v1.shape == (2, 1024) and v1.dtype == np.float32
r1 = li._bge_rerank_sync("space physics", ["unrelated", "space physics notes"], top_n=1)
assert r1[0]["index"] == 1

v2 = asyncio.run(li._bge_embed(["async probe"]))
assert v2.shape == (1, 1024) and v2.dtype == np.float32
r2 = asyncio.run(li._bge_rerank("async probe", ["none", "async probe match"], top_n=1))
assert r2[0]["index"] == 1

# Wait for idle unload to terminate the worker without touching torch.
deadline = time.monotonic() + 5.0
while li._BGE_WORKER.is_running() and time.monotonic() < deadline:
    time.sleep(0.02)
assert not li._BGE_WORKER.is_running(), "idle unload did not shut down worker"

# Explicitly shut down after another call too.
v3 = li._bge_encode_sync(["post-reload"])
assert v3.shape == (1, 1024)
assert li.shutdown_bge_worker() is True

forbidden = ("torch", "transformers", "sentence_transformers", "FlagEmbedding")
loaded = [m for m in forbidden if m in sys.modules]
assert not loaded, f"Parent imported forbidden ML modules: {loaded}"
"""
    env = {
        **os.environ,
        "KS_BGE_WORKER_MODE": "subprocess",
        "KS_BGE_WORKER_TEST_FAKE": "1",
    }
    proc = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"


# ---------------------------------------------------------------------------
# 4. Real subprocess embed, rerank, error propagation, concurrency
# ---------------------------------------------------------------------------

def test_subprocess_embed_exact_values_and_empty_input(isolated_worker):
    texts = ["abc", "solar wind"]
    arr = lightrag_init._bge_encode_sync(texts)
    assert arr.shape == (2, 1024)
    assert arr.dtype == np.float32
    assert arr.flags.writeable  # reconstructed via .copy(), owns its buffer
    for i, text in enumerate(texts):
        assert arr[i, 0] == pytest.approx(float(len(text)))
        expected_c1 = float(sum(text.encode("utf-8")) % 1000) / 1000.0
        assert arr[i, 1] == pytest.approx(expected_c1)
        assert arr[i, 2] == pytest.approx(0.25)

    # Empty input round-trips cleanly to (0, 1024) float32.
    empty = lightrag_init._bge_encode_sync([])
    assert empty.shape == (0, 1024)
    assert empty.dtype == np.float32


def test_subprocess_rerank_respects_top_n_and_pool_cap(isolated_worker, monkeypatch):
    docs = [
        "score=0.20 d0",
        "score=0.95 d1",
        "score=0.50 d2",
        "score=0.99 d3",  # beyond pool_cap=3
        "score=0.88 d4",  # beyond pool_cap=3
    ]
    monkeypatch.setattr(lightrag_init, "_RERANK_POOL_CAP", 3)

    out = lightrag_init._bge_rerank_sync("q", docs, top_n=2)
    assert [r["index"] for r in out] == [1, 2]
    assert [r["relevance_score"] for r in out] == pytest.approx([0.95, 0.50])
    assert all(isinstance(r["index"], int) for r in out)
    assert all(isinstance(r["relevance_score"], float) for r in out)

    # Pool cap never truncates below top_n.
    out_four = asyncio.run(lightrag_init._bge_rerank("q", docs, top_n=4))
    assert [r["index"] for r in out_four] == [3, 1, 2, 0]
    assert len(out_four) == 4

    # top_n=None and top_n=0 keep all documents (within cap).
    monkeypatch.setattr(lightrag_init, "_RERANK_POOL_CAP", 0)
    assert len(lightrag_init._bge_rerank_sync("q", docs, top_n=None)) == 5
    assert len(lightrag_init._bge_rerank_sync("q", docs, top_n=0)) == 5

    # Empty documents short-circuits cleanly.
    assert lightrag_init._bge_rerank_sync("q", [], top_n=5) == []


def test_subprocess_error_propagation_and_rerank_failure_counter(isolated_worker, caplog):
    # Warm up worker to record its PID.
    lightrag_init._bge_encode_sync(["warmup"])
    pid = isolated_worker.pid
    assert pid is not None

    # Embed exceptions (including OSError subclasses like FileNotFoundError) propagate
    # as descriptive RuntimeErrors without killing or exiting the worker.
    with pytest.raises(RuntimeError, match="ValueError.*simulated worker embed failure"):
        lightrag_init._bge_encode_sync(["__TEST_RAISE__"])
    assert isolated_worker.pid == pid
    with pytest.raises(RuntimeError, match="FileNotFoundError.*simulated missing checkpoint"):
        lightrag_init._bge_encode_sync(["__TEST_OSERROR__"])
    assert isolated_worker.pid == pid
    assert lightrag_init._bge_encode_sync(["after error"]).shape == (1, 1024)

    # Async _bge_rerank catches worker failure, bumps rerank_failure_count, logs LOUD, returns [].
    assert lightrag_init.rerank_failure_count() == 0
    with caplog.at_level(logging.ERROR, logger="ks.store.lightrag_init"):
        out = asyncio.run(lightrag_init._bge_rerank("__TEST_RAISE__", ["d0", "d1"]))
    assert out == []
    assert lightrag_init.rerank_failure_count() == 1
    assert any("bge rerank FAILED" in r.getMessage() for r in caplog.records)
    # Worker remains healthy on the same PID.
    assert isolated_worker.pid == pid
    ok = asyncio.run(lightrag_init._bge_rerank("hello", ["world", "hello world"], top_n=1))
    assert ok[0]["index"] == 1


def test_concurrent_threads_serialize_cleanly_over_worker_pipes(isolated_worker):
    def task(i: int):
        if i % 2 == 0:
            text = f"embed-item-{i}"
            arr = lightrag_init._bge_encode_sync([text])
            assert arr.shape == (1, 1024)
            assert arr[0, 0] == pytest.approx(float(len(text)))
        else:
            res = lightrag_init._bge_rerank_sync(
                f"topic{i}",
                [f"other {i}", f"topic{i} match"],
                top_n=1,
            )
            assert res[0]["index"] == 1
        return i

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(task, range(24)))
    assert results == list(range(24))


# ---------------------------------------------------------------------------
# 5. Idle unload, explicit shutdown, server.stop_background, crash recovery
# ---------------------------------------------------------------------------

def test_idle_unload_terminates_and_reaps_worker_and_respawns(isolated_worker, monkeypatch):
    monkeypatch.setattr(lightrag_init, "_IDLE_UNLOAD_ENABLED", True)
    monkeypatch.setattr(lightrag_init, "_IDLE_UNLOAD_TIMEOUT", 0.05)

    lightrag_init._bge_encode_sync(["first"])
    proc1 = isolated_worker._proc
    pid1 = isolated_worker.pid
    assert proc1 is not None and pid1 is not None

    deadline = time.monotonic() + 5.0
    while isolated_worker.is_running() and time.monotonic() < deadline:
        time.sleep(0.02)

    assert not isolated_worker.is_running()
    assert proc1.poll() is not None
    assert proc1.returncode == 0

    # Next call respawns a fresh worker with a new PID.
    res = lightrag_init._bge_rerank_sync("q", ["score=0.4 a", "score=0.9 b"], top_n=1)
    assert res[0]["index"] == 1
    pid2 = isolated_worker.pid
    assert pid2 is not None and pid2 != pid1


def test_explicit_unload_and_server_stop_background_shut_down_worker(isolated_worker, monkeypatch):
    from papervault.knowledge.ledger import store as ledger_store
    from papervault.knowledge.mcp import server
    from papervault.knowledge.store import graph

    lightrag_init._bge_encode_sync(["a"])
    proc1 = isolated_worker._proc
    assert proc1 is not None and proc1.poll() is None

    assert lightrag_init.unload_bge_models() is True
    assert proc1.poll() is not None
    assert isolated_worker.is_running() is False
    # Idempotent when nothing is loaded.
    assert lightrag_init.unload_bge_models() is False

    # Respawn and verify server.stop_background() shuts down and reaps the worker.
    lightrag_init._bge_encode_sync(["b"])
    proc2 = isolated_worker._proc
    assert proc2 is not None and proc2.poll() is None

    async def noop():
        pass

    monkeypatch.setattr(graph, "close_graph", noop)
    monkeypatch.setattr(ledger_store, "close_pool", noop)
    monkeypatch.setattr(server, "_query_stopping", False)
    monkeypatch.setattr(server, "_bg_task", None)
    monkeypatch.setattr(server, "_startup_task", None)
    monkeypatch.setattr(server, "_embed_probe_task", None)

    asyncio.run(server.stop_background())
    assert proc2.poll() is not None
    assert isolated_worker.is_running() is False


def test_worker_crash_recovers_cleanly_on_next_call(isolated_worker):
    lightrag_init._bge_encode_sync(["before crash"])
    proc1 = isolated_worker._proc
    pid1 = isolated_worker.pid
    assert proc1 is not None and pid1 is not None

    # Hard kill while idle -> dead process detected and replaced on next call.
    proc1.kill()
    proc1.wait(timeout=5)
    assert proc1.poll() is not None

    arr = lightrag_init._bge_encode_sync(["after crash"])
    assert arr.shape == (1, 1024)
    pid2 = isolated_worker.pid
    assert pid2 is not None and pid2 != pid1

    # Also simulate a pre-existing worker dying right after poll() returned None
    # (BrokenPipeError / EOFError race before work runs) -> retries ONCE cleanly.
    proc2 = isolated_worker._proc
    assert proc2 is not None
    real_poll = proc2.poll
    faked_once = {"done": False}

    def stale_poll():
        if not faked_once["done"]:
            faked_once["done"] = True
            return None
        return real_poll()

    proc2.kill()
    proc2.wait(timeout=5)
    proc2.poll = stale_poll  # type: ignore[method-assign]

    arr2 = lightrag_init._bge_encode_sync(["after race"])
    assert arr2.shape == (1, 1024)
    pid3 = isolated_worker.pid
    assert pid3 is not None and pid3 != pid2


def test_fresh_worker_fatal_crash_does_not_retry_infinitely(monkeypatch):
    worker = bge_worker._BGEWorkerClient()
    spawns = {"n": 0}

    def spawn_crashing():
        spawns["n"] += 1
        proc = subprocess.Popen(
            [sys.executable, "-c", "import sys; sys.exit(42)"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        worker._proc = proc
        return proc

    monkeypatch.setattr(worker, "_ensure_started_locked", spawn_crashing)
    with pytest.raises(RuntimeError, match="BGE worker subprocess failed during embed"):
        worker.embed(["boom"])
    assert spawns["n"] == 1
    assert not worker.is_running()


def test_short_lived_spawning_thread_does_not_kill_worker(isolated_worker):
    """Spawning from a thread that subsequently exits must NOT trigger false PDEATHSIG death."""
    out: list[np.ndarray] = []

    def run_in_thread():
        out.append(lightrag_init._bge_encode_sync(["from thread"]))

    t = threading.Thread(target=run_in_thread)
    t.start()
    t.join(timeout=10)
    assert len(out) == 1
    pid1 = isolated_worker.pid
    assert pid1 is not None

    time.sleep(0.15)
    assert isolated_worker.is_running(), "worker died when spawning thread exited"
    assert isolated_worker.pid == pid1
    ping = isolated_worker.ping()
    assert ping["pid"] == pid1
    assert ping["embed_loaded"] is True


def test_eof_on_parent_stdin_exits_worker_cleanly():
    env = {
        **os.environ,
        "KS_BGE_WORKER_TEST_FAKE": "1",
    }
    proc = subprocess.Popen(
        [sys.executable, "-m", "papervault.knowledge.store.bge_worker"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )
    try:
        assert proc.stdin is not None and proc.stdout is not None
        bge_worker.write_frame(proc.stdin, {"id": 1, "op": "ping"})
        frame = bge_worker.read_frame(proc.stdout)
        assert frame is not None
        resp, _ = frame
        assert resp["ok"] is True
        assert resp["pid"] == proc.pid

        # Close stdin (EOF) -> worker must exit cleanly with code 0 (zero orphans).
        proc.stdin.close()
        rc = proc.wait(timeout=5.0)
        assert rc == 0
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=2.0)
        for pipe in (proc.stdin, proc.stdout, proc.stderr):
            if pipe is not None:
                try:
                    pipe.close()
                except OSError:
                    pass


# ---------------------------------------------------------------------------
# 6. Compatibility with in-process mode & monkeypatched singletons/getters
# ---------------------------------------------------------------------------

def test_inprocess_mode_uses_inprocess_path_without_spawning_worker(isolated_worker, monkeypatch):
    monkeypatch.setenv("KS_BGE_WORKER_MODE", "inprocess")
    assert lightrag_init._use_subprocess_embedder() is False
    assert lightrag_init._use_subprocess_reranker() is False

    class _FakeInProcessEmbed:
        def encode(self, texts, **kwargs):
            return {"dense_vecs": np.ones((len(texts), 4), dtype=np.float32)}

    class _FakeInProcessRerank:
        def predict(self, pairs, **kwargs):
            return np.array([0.1, 0.9][: len(pairs)], dtype=np.float32)

    monkeypatch.setattr(lightrag_init, "_BGE_MODEL", _FakeInProcessEmbed())
    monkeypatch.setattr(lightrag_init, "_BGE_RERANKER", _FakeInProcessRerank())

    vecs = lightrag_init._bge_encode_sync(["x"])
    assert vecs.shape == (1, 4)
    ranked = lightrag_init._bge_rerank_sync("q", ["d0", "d1"], top_n=1)
    assert ranked[0]["index"] == 1
    assert isolated_worker.is_running() is False


def test_monkeypatched_getter_or_singleton_bypasses_worker_in_subprocess_mode(
    isolated_worker, monkeypatch
):
    monkeypatch.setenv("KS_BGE_WORKER_MODE", "subprocess")

    class _FakeEmbed:
        def encode(self, texts, **kwargs):
            return {"dense_vecs": np.zeros((len(texts), 3), dtype=np.float32)}

    class _FakeRerank:
        def predict(self, pairs, **kwargs):
            return np.array([0.7] * len(pairs), dtype=np.float32)

    # Patching _get_bge_model / _get_bge_reranker bypasses worker transparently.
    monkeypatch.setattr(lightrag_init, "_get_bge_model", lambda: _FakeEmbed())
    monkeypatch.setattr(lightrag_init, "_get_bge_reranker", lambda: _FakeRerank())
    assert lightrag_init._bge_encode_sync(["a"]).shape == (1, 3)
    assert lightrag_init._bge_rerank_sync("q", ["d"])[0]["relevance_score"] == pytest.approx(0.7)
    assert isolated_worker.is_running() is False


def test_invalid_worker_mode_warns_and_defaults_to_subprocess(monkeypatch, caplog):
    monkeypatch.setenv("KS_BGE_WORKER_MODE", "invalid-mode")
    with caplog.at_level(logging.WARNING, logger="ks.store.lightrag_init"):
        mode = lightrag_init._worker_mode()
    assert mode == "subprocess"
    assert any("Invalid KS_BGE_WORKER_MODE" in r.getMessage() for r in caplog.records)
