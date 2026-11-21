"""On-demand local BGE embedder & reranker worker subprocess (Plan B).

In production (`KS_BGE_WORKER_MODE=subprocess`, the default), the long-lived
`papervault-mcp` server process never imports `torch`, `transformers`,
`sentence_transformers`, or `FlagEmbedding` and never initializes a CUDA
context. Instead, `_BGEWorkerClient` spawns this module as an isolated local
worker (`sys.executable -m papervault.knowledge.store.bge_worker`) on first
embedding or reranking demand and communicates over stdin/stdout using a
strictly non-pickle binary-framed RPC protocol:

    Frame layout:
      [4-byte big-endian uint32 header_len]
      [header_len bytes of UTF-8 JSON header]
      [optional payload_bytes raw bytes (float32 C-contiguous ndarray)]

When idle (`KS_BGE_IDLE_UNLOAD=1`), shutting down the worker process releases
100% of its PyTorch CPU RAM, glibc arenas, and CUDA context (0 MiB idle VRAM)
immediately back to the OS.

CRITICAL: Keep top-level imports in this module strictly lightweight stdlib so
importing `_BGEWorkerClient` in the parent process never loads PyTorch or CUDA,
and so `main()` can redirect fd 1 -> fd 2 before any ML library is imported.
"""
from __future__ import annotations

import atexit
import ctypes
import json
import logging
import os
import signal
import struct
import subprocess
import sys
import threading
import time
from typing import Any, BinaryIO

log = logging.getLogger("ks.store.bge_worker")

_HEADER_STRUCT = struct.Struct(">I")
_MAX_HEADER_BYTES = 256 * 1024 * 1024
_MAX_PAYLOAD_BYTES = 512 * 1024 * 1024
_EMBED_DIM = 1024
_PARENT_PID_ENV = "_KS_BGE_WORKER_PARENT_PID"


# ---------------------------------------------------------------------------
# Binary-framed JSON + raw-bytes RPC helpers (strictly zero-pickle)
# ---------------------------------------------------------------------------

def _write_all(stream: BinaryIO, data: bytes | memoryview) -> None:
    view = memoryview(data)
    total = view.nbytes
    sent = 0
    while sent < total:
        written = stream.write(view[sent:])
        if written is None:
            break
        if written <= 0:
            raise BrokenPipeError("BGE worker pipe closed during write")
        sent += written


def _read_exact(stream: BinaryIO, n: int, *, allow_empty_eof: bool = False) -> bytes:
    if n == 0:
        return b""
    buf = bytearray()
    while len(buf) < n:
        chunk = stream.read(n - len(buf))
        if not chunk:
            if allow_empty_eof and not buf:
                return b""
            raise EOFError(
                f"Unexpected EOF reading BGE worker frame ({len(buf)}/{n} bytes)"
            )
        buf.extend(chunk)
    return bytes(buf)


def write_frame(
    stream: BinaryIO,
    header: dict[str, Any],
    payload: bytes | bytearray | memoryview = b"",
) -> None:
    """Write one length-prefixed JSON header + optional raw byte payload."""
    payload_view = memoryview(payload) if payload else memoryview(b"")
    if payload_view.nbytes:
        if header.get("payload_bytes") != payload_view.nbytes:
            header = {**header, "payload_bytes": payload_view.nbytes}
    elif header.get("payload_bytes", 0) != 0:
        raise ValueError(
            f"Header payload_bytes={header.get('payload_bytes')!r} with empty payload"
        )
    header_bytes = json.dumps(
        header, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    if len(header_bytes) > _MAX_HEADER_BYTES:
        raise ValueError(
            f"BGE worker frame header too large: {len(header_bytes)} bytes"
        )
    if payload_view.nbytes > _MAX_PAYLOAD_BYTES:
        raise ValueError(
            f"BGE worker frame payload too large: {payload_view.nbytes} bytes"
        )
    _write_all(stream, _HEADER_STRUCT.pack(len(header_bytes)))
    _write_all(stream, header_bytes)
    if payload_view.nbytes:
        _write_all(stream, payload_view)
    stream.flush()


def read_frame(stream: BinaryIO) -> tuple[dict[str, Any], bytes] | None:
    """Read one frame. Returns None on clean EOF at a frame boundary."""
    len_bytes = _read_exact(stream, _HEADER_STRUCT.size, allow_empty_eof=True)
    if not len_bytes:
        return None
    (header_len,) = _HEADER_STRUCT.unpack(len_bytes)
    if header_len <= 0 or header_len > _MAX_HEADER_BYTES:
        raise ValueError(f"Invalid BGE worker frame header_len={header_len}")
    header_raw = _read_exact(stream, header_len)
    header = json.loads(header_raw.decode("utf-8"))
    if not isinstance(header, dict):
        raise ValueError(f"BGE worker frame header must be dict, got {type(header).__name__}")
    payload_bytes = int(header.get("payload_bytes", 0))
    if payload_bytes < 0 or payload_bytes > _MAX_PAYLOAD_BYTES:
        raise ValueError(f"Invalid BGE worker frame payload_bytes={payload_bytes}")
    payload = _read_exact(stream, payload_bytes) if payload_bytes else b""
    return header, payload


# ---------------------------------------------------------------------------
# Shared rerank OOM-retry helper (used by both worker and in-process fallback)
# ---------------------------------------------------------------------------

def predict_with_oom_retry(model: Any, pairs: list[tuple[str, str]], initial_batch_size: int) -> Any:
    """CrossEncoder.predict with CUDA-OOM survival (Fix B).

    Starts at `initial_batch_size` (clamped >= 1) and on a CUDA OOM RuntimeError
    empties the CUDA cache and steps down through smaller power-of-two batch sizes
    (32->16->8->4->2->1). Non-OOM errors propagate immediately. If even batch_size=1
    OOMs, re-raise the last OOM.
    """
    last = None
    initial = max(1, int(initial_batch_size))
    ladder = list(dict.fromkeys([initial, *(b for b in (32, 16, 8, 4, 2, 1) if b < initial)]))
    for bs in ladder:
        try:
            return model.predict(pairs, batch_size=bs)
        except RuntimeError as e:  # torch.cuda.OutOfMemoryError is a RuntimeError subclass
            is_oom = (
                "out of memory" in str(e).lower()
                or any(cls.__name__ == "OutOfMemoryError" for cls in type(e).__mro__)
            )
            if not is_oom:
                raise
            last = e
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001 — best-effort cache drop
                pass
    raise last


# ---------------------------------------------------------------------------
# Parent-side thread-safe worker client
# ---------------------------------------------------------------------------

class _BGEWorkerClient:
    """Thread-safe lifecycle & RPC manager for the local BGE worker subprocess."""

    def __init__(self, *, module_name: str = "papervault.knowledge.store.bge_worker") -> None:
        self._module_name = module_name
        self._io_lock = threading.Lock()
        self._proc: subprocess.Popen[bytes] | None = None
        self._next_id = 0
        self._shutdown_epoch = 0

    @property
    def pid(self) -> int | None:
        with self._io_lock:
            proc = self._proc
            if proc is not None and proc.poll() is None:
                return proc.pid
            return None

    def is_running(self) -> bool:
        proc = self._proc
        return proc is not None and proc.poll() is None

    def _build_env(self) -> dict[str, str]:
        return {
            **os.environ,
            "MALLOC_ARENA_MAX": "2",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "TOKENIZERS_PARALLELISM": "false",
            _PARENT_PID_ENV: str(os.getpid()),
        }

    def _ensure_started_locked(self) -> subprocess.Popen[bytes]:
        if self._proc is not None:
            if self._proc.poll() is None:
                return self._proc
            self._close_pipes(self._proc)
            self._proc = None
        cmd = [sys.executable, "-m", self._module_name]
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            env=self._build_env(),
            start_new_session=False,
            close_fds=True,
        )
        self._proc = proc
        log.info("spawned BGE worker subprocess (pid=%d)", proc.pid)
        return proc

    @staticmethod
    def _close_pipes(proc: subprocess.Popen[bytes]) -> None:
        for pipe in (proc.stdin, proc.stdout):
            if pipe is not None:
                try:
                    pipe.close()
                except OSError:
                    pass

    def _terminate_and_cleanup_locked(self) -> int | None:
        proc = self._proc
        if proc is None:
            return None
        self._proc = None
        if proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=2.0)
            except (OSError, subprocess.TimeoutExpired):
                try:
                    proc.kill()
                    proc.wait(timeout=2.0)
                except (OSError, subprocess.TimeoutExpired):
                    pass
        self._close_pipes(proc)
        return proc.returncode

    def _request(self, op: str, **params: Any) -> tuple[dict[str, Any], bytes]:
        with self._io_lock:
            epoch = self._shutdown_epoch
            reused = self._proc is not None and self._proc.poll() is None
            for attempt in range(2):
                proc = self._ensure_started_locked()
                self._next_id += 1
                req_id = self._next_id
                header = {"id": req_id, "op": op, **params}
                assert proc.stdin is not None and proc.stdout is not None
                try:
                    write_frame(proc.stdin, header)
                    frame = read_frame(proc.stdout)
                    if frame is None:
                        raise EOFError("BGE worker closed stdout unexpectedly")
                except (BrokenPipeError, EOFError, OSError) as exc:
                    rc = self._terminate_and_cleanup_locked()
                    if self._shutdown_epoch != epoch:
                        raise RuntimeError(
                            f"BGE worker shut down during {op} (rc={rc})"
                        ) from exc
                    if reused and attempt == 0:
                        log.warning(
                            "BGE worker died before completing %s (rc=%s, %s); "
                            "restarting worker once",
                            op, rc, exc,
                        )
                        reused = False
                        continue
                    raise RuntimeError(
                        f"BGE worker subprocess failed during {op} (rc={rc}): {exc}"
                    ) from exc

                resp, payload = frame
                if resp.get("id") != req_id:
                    rc = self._terminate_and_cleanup_locked()
                    raise RuntimeError(
                        f"BGE worker protocol desync on {op}: expected id={req_id}, "
                        f"got {resp.get('id')!r} (rc={rc})"
                    )
                if not resp.get("ok", False):
                    err_type = resp.get("error_type", "RuntimeError")
                    message = resp.get("message", "unknown error")
                    raise RuntimeError(
                        f"BGE worker {op} error ({err_type}): {message}"
                    )
                return resp, payload
            raise RuntimeError(f"BGE worker subprocess failed during {op}")

    def ping(self) -> dict[str, Any]:
        resp, _ = self._request("ping")
        return resp

    def embed(
        self,
        texts: list[str],
        *,
        batch_size: int = 32,
        max_length: int = 8192,
    ) -> Any:
        import numpy as np

        resp, payload = self._request(
            "embed",
            texts=list(texts),
            batch_size=int(batch_size),
            max_length=int(max_length),
        )
        shape = tuple(int(dim) for dim in resp["shape"])
        dtype_name = resp.get("dtype", "float32")
        if dtype_name != "float32":
            raise RuntimeError(f"Unexpected BGE worker embed dtype: {dtype_name!r}")
        expected_bytes = int(np.prod(shape, dtype=np.int64)) * np.dtype(np.float32).itemsize
        if len(payload) != expected_bytes:
            raise RuntimeError(
                f"BGE worker embed payload size mismatch: got {len(payload)} bytes, "
                f"expected {expected_bytes} for shape {shape}"
            )
        return np.frombuffer(payload, dtype=np.float32).reshape(shape).copy()

    def rerank(
        self,
        query: str,
        documents: list[str],
        *,
        top_n: int | None = None,
        batch_size: int = 32,
        max_length: int = 4096,
    ) -> list[dict[str, int | float]]:
        resp, _ = self._request(
            "rerank",
            query=query,
            documents=list(documents),
            top_n=top_n,
            batch_size=int(batch_size),
            max_length=int(max_length),
        )
        results = resp.get("results")
        if not isinstance(results, list):
            raise RuntimeError(f"Invalid BGE worker rerank results: {type(results).__name__}")
        return [
            {"index": int(item["index"]), "relevance_score": float(item["relevance_score"])}
            for item in results
        ]

    def _shutdown_locked(self, timeout: float = 5.0) -> bool:
        proc = self._proc
        if proc is None:
            return False
        self._proc = None
        was_alive = proc.poll() is None
        if was_alive:
            if proc.stdin is not None:
                try:
                    self._next_id += 1
                    write_frame(proc.stdin, {"id": self._next_id, "op": "shutdown"})
                except (BrokenPipeError, OSError, ValueError):
                    pass
                try:
                    proc.stdin.close()
                except OSError:
                    pass
            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                log.warning(
                    "BGE worker (pid=%d) did not exit within %.1fs; sending SIGTERM",
                    proc.pid, timeout,
                )
                try:
                    proc.terminate()
                except OSError:
                    pass
                try:
                    proc.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    log.warning(
                        "BGE worker (pid=%d) ignored SIGTERM; sending SIGKILL",
                        proc.pid,
                    )
                    try:
                        proc.kill()
                        proc.wait(timeout=2.0)
                    except (OSError, subprocess.TimeoutExpired):
                        pass
        self._close_pipes(proc)
        if proc.poll() is None:
            try:
                proc.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                pass
        if was_alive:
            log.info(
                "BGE worker subprocess shut down cleanly (pid=%d, rc=%s)",
                proc.pid, proc.returncode,
            )
        return was_alive

    def shutdown(self, timeout: float = 5.0, *, force_if_busy: bool = False) -> bool:
        """Cleanly terminate and reap the worker subprocess. Returns True if it was alive."""
        self._shutdown_epoch += 1
        acquired = (
            self._io_lock.acquire(timeout=timeout)
            if force_if_busy
            else self._io_lock.acquire()
        )
        if not acquired:
            proc = self._proc
            if proc is not None and proc.poll() is None:
                log.warning(
                    "BGE worker busy during forced shutdown (pid=%d); terminating",
                    proc.pid,
                )
                try:
                    proc.terminate()
                except OSError:
                    pass
                try:
                    proc.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    try:
                        proc.kill()
                        proc.wait(timeout=2.0)
                    except (OSError, subprocess.TimeoutExpired):
                        pass
                return True
            return False
        try:
            return self._shutdown_locked(timeout=timeout)
        finally:
            self._io_lock.release()


_DEFAULT_WORKER = _BGEWorkerClient()


def get_default_worker() -> _BGEWorkerClient:
    """Return the process-wide BGE worker client singleton."""
    return _DEFAULT_WORKER


def _atexit_shutdown() -> None:
    try:
        _DEFAULT_WORKER.shutdown(timeout=5.0, force_if_busy=True)
    except Exception:  # noqa: BLE001 — never fail interpreter exit
        pass


atexit.register(_atexit_shutdown)


# ---------------------------------------------------------------------------
# Worker subprocess implementation (runs ONLY when executed as __main__)
# ---------------------------------------------------------------------------

def _emit_test_stdout_noise(stage: str) -> None:
    """Test-only helper verifying fd-1 stdout redirection protects RPC framing."""
    if (
        os.getenv("KS_BGE_WORKER_TEST_FAKE") == "1"
        and os.getenv("KS_BGE_WORKER_TEST_NOISY_STDOUT") == "1"
    ):
        print(f"[bge_worker noisy python stdout] {stage}", flush=True)
        os.write(1, f"[bge_worker noisy raw fd1 stdout] {stage}\n".encode("utf-8"))


class _FakeEmbedModel:
    """Lightweight deterministic embedder enabled ONLY when KS_BGE_WORKER_TEST_FAKE=1."""

    def encode(
        self, texts: list[str], batch_size: int = 32, max_length: int = 8192
    ) -> dict[str, Any]:
        import numpy as np

        _emit_test_stdout_noise("embed")
        arr = np.full((len(texts), _EMBED_DIM), 0.25, dtype=np.float32)
        for i, text in enumerate(texts):
            if text == "__TEST_RAISE__":
                raise ValueError("simulated worker embed failure")
            if text == "__TEST_OSERROR__":
                raise FileNotFoundError("simulated missing checkpoint")
            arr[i, 0] = float(len(text))
            arr[i, 1] = float(sum(text.encode("utf-8")) % 1000) / 1000.0
        return {"dense_vecs": arr}


class _FakeRerankModel:
    """Lightweight deterministic reranker enabled ONLY when KS_BGE_WORKER_TEST_FAKE=1."""

    def predict(
        self, pairs: list[tuple[str, str]], batch_size: int = 32
    ) -> Any:
        import numpy as np

        _emit_test_stdout_noise("rerank")
        scores: list[float] = []
        for query, doc in pairs:
            if query == "__TEST_RAISE__" or doc == "__TEST_RAISE__":
                raise ValueError("simulated worker rerank failure")
            if doc.startswith("score="):
                scores.append(float(doc.split("=", 1)[1].split()[0]))
                continue
            q_words = set(query.lower().split())
            d_words = set(doc.lower().split())
            overlap = len(q_words & d_words) / max(1, len(q_words))
            scores.append(float(overlap))
        return np.array(scores, dtype=np.float32)


class _WorkerState:
    """Lazy model holder inside the worker subprocess."""

    def __init__(self) -> None:
        self.embed_model: Any = None
        self.rerank_model: Any = None
        self.use_fake = os.getenv("KS_BGE_WORKER_TEST_FAKE") == "1"

    def get_embedder(self) -> Any:
        if self.embed_model is not None:
            return self.embed_model
        if self.use_fake:
            _emit_test_stdout_noise("load_embedder")
            self.embed_model = _FakeEmbedModel()
            return self.embed_model
        from FlagEmbedding import BGEM3FlagModel
        from papervault.knowledge.config import CONFIG

        started = time.perf_counter()
        self.embed_model = BGEM3FlagModel(
            CONFIG.bge_m3.model_path,
            use_fp16=True,
            devices=[CONFIG.bge_m3.device],
        )
        log.info(
            "bge embedder loaded in %.3fs (worker pid=%d)",
            time.perf_counter() - started,
            os.getpid(),
        )
        return self.embed_model

    def get_reranker(self, max_length: int) -> Any:
        if self.rerank_model is not None:
            return self.rerank_model
        if self.use_fake:
            _emit_test_stdout_noise("load_reranker")
            self.rerank_model = _FakeRerankModel()
            return self.rerank_model
        import torch
        from sentence_transformers import CrossEncoder
        from papervault.knowledge.config import CONFIG

        started = time.perf_counter()
        dtype_name = os.getenv("KS_RERANK_DTYPE", "float32")
        model_kwargs = (
            {"torch_dtype": getattr(torch, dtype_name)} if dtype_name != "float32" else None
        )
        model = CrossEncoder(
            CONFIG.bge_reranker.model_path,
            device=CONFIG.bge_reranker.device,
            max_length=max_length,
            model_kwargs=model_kwargs,
        )
        score = model.predict([("preflight query", "preflight passage")])
        log.info(
            "bge reranker preflight OK (model=%s, max_length=%d, dtype=%s, sample_score=%.4f)",
            CONFIG.bge_reranker.model_path,
            max_length,
            next(model.model.parameters()).dtype,
            float(score[0]),
        )
        log.info(
            "bge reranker loaded in %.3fs (worker pid=%d)",
            time.perf_counter() - started,
            os.getpid(),
        )
        self.rerank_model = model
        return self.rerank_model


def _install_parent_death_guard() -> None:
    """Exit immediately if the parent process dies (including SIGKILL) without polling.

    Uses Linux `PR_SET_PDEATHSIG` with `SIGUSR1` + an `os.getppid()` check so that
    if a short-lived worker thread inside the parent spawned this subprocess and
    exits while the parent process stays alive, the worker keeps serving — whereas
    if the parent process itself terminates, `os.getppid()` changes and the worker
    exits immediately with zero orphan GPU processes.
    """
    if sys.platform != "linux":
        return
    raw_ppid = os.getenv(_PARENT_PID_ENV, "").strip()
    expected_ppid = int(raw_ppid) if raw_ppid.isdigit() else os.getppid()

    def _on_pdeath(_signum: int, _frame: Any) -> None:
        if os.getppid() != expected_ppid:
            os._exit(0)

    try:
        signal.signal(signal.SIGUSR1, _on_pdeath)
        libc = ctypes.CDLL("libc.so.6")
        pr_set_pdeathsig = 1
        libc.prctl(pr_set_pdeathsig, int(signal.SIGUSR1))
    except Exception:  # noqa: BLE001 — best-effort orphan guard
        return

    # Close the fork()-to-prctl() race window.
    if os.getppid() != expected_ppid:
        os._exit(0)


def _execute_request(
    state: _WorkerState,
    req: dict[str, Any],
) -> tuple[dict[str, Any], bytes, bool]:
    """Execute one RPC request and return `(response_header, payload_bytes, keep_running)`."""
    req_id = req.get("id")
    op = req.get("op")

    if op == "shutdown":
        return {"id": req_id, "ok": True}, b"", False

    if op == "ping":
        return (
            {
                "id": req_id,
                "ok": True,
                "pid": os.getpid(),
                "embed_loaded": state.embed_model is not None,
                "rerank_loaded": state.rerank_model is not None,
            },
            b"",
            True,
        )

    if op == "embed":
        import numpy as np

        texts = req.get("texts")
        if not isinstance(texts, list):
            raise TypeError(f"embed texts must be list[str], got {type(texts).__name__}")
        batch_size = int(req.get("batch_size", 32))
        max_length = int(req.get("max_length", 8192))
        if not texts:
            dense = np.empty((0, _EMBED_DIM), dtype=np.float32)
        else:
            model = state.get_embedder()
            result = model.encode(texts, batch_size=batch_size, max_length=max_length)
            dense = np.ascontiguousarray(result["dense_vecs"], dtype=np.float32)
            if dense.ndim == 1:
                dense = dense.reshape(1, -1)
        raw = dense.tobytes()
        return (
            {
                "id": req_id,
                "ok": True,
                "shape": list(dense.shape),
                "dtype": "float32",
                "payload_bytes": len(raw),
            },
            raw,
            True,
        )

    if op == "rerank":
        query = req.get("query")
        documents = req.get("documents")
        top_n = req.get("top_n")
        batch_size = int(req.get("batch_size", os.getenv("KS_RERANK_BATCH_SIZE", "32")))
        max_length = int(req.get("max_length", os.getenv("KS_RERANK_MAX_LENGTH", "4096")))
        if not isinstance(query, str) or not isinstance(documents, list):
            raise TypeError("rerank requires str query and list[str] documents")
        if not documents:
            return {"id": req_id, "ok": True, "results": []}, b"", True
        model = state.get_reranker(max_length=max_length)
        pairs = [(query, d) for d in documents]
        scores = predict_with_oom_retry(model, pairs, initial_batch_size=batch_size)
        ranked = sorted(range(len(documents)), key=lambda i: float(scores[i]), reverse=True)
        if top_n:
            ranked = ranked[: int(top_n)]
        results = [
            {"index": int(i), "relevance_score": float(scores[i])} for i in ranked
        ]
        return {"id": req_id, "ok": True, "results": results}, b"", True

    raise ValueError(f"Unknown BGE worker op: {op!r}")


def serve_loop(rpc_in: BinaryIO, rpc_out: BinaryIO) -> int:
    """Main stdin/stdout framing loop for the BGE worker."""
    state = _WorkerState()
    while True:
        try:
            frame = read_frame(rpc_in)
        except (EOFError, OSError):
            return 0
        if frame is None:
            return 0
        req, _payload = frame
        req_id = req.get("id")
        try:
            resp_header, resp_payload, keep_running = _execute_request(state, req)
        except Exception as exc:  # noqa: BLE001 — return structured error frame to client
            log.exception("BGE worker error handling op=%r", req.get("op"))
            resp_header = {
                "id": req_id,
                "ok": False,
                "error_type": type(exc).__name__,
                "message": str(exc),
            }
            resp_payload = b""
            keep_running = True
        try:
            write_frame(rpc_out, resp_header, resp_payload)
        except (BrokenPipeError, OSError):
            return 0
        if not keep_running:
            return 0


def main() -> None:
    """Worker entrypoint: isolate RPC fd 1 before importing any heavy libraries."""
    sys.stdout.flush()
    rpc_out_fd = os.dup(1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr

    logging.basicConfig(
        stream=sys.stderr,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    _install_parent_death_guard()
    _emit_test_stdout_noise("startup")

    rpc_out = os.fdopen(rpc_out_fd, "wb")
    rpc_in = sys.stdin.buffer
    try:
        rc = serve_loop(rpc_in, rpc_out)
    except KeyboardInterrupt:
        rc = 0
    finally:
        try:
            rpc_out.close()
        except OSError:
            pass
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
