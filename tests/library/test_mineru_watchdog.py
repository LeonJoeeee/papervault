"""Global vLLM watchdog evidence; metrics, clock, HTTP and systemd stay local fakes."""

import pytest

from papervault.library.services import mineru_server as ms


def metrics(*, running=0, waiting=0, success=12, tokens=50, extra=""):
    return f'''# HELP vllm:num_requests_running Number of requests running.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{{model_name="mineru"}} {running}
vllm:num_requests_waiting{{model_name="mineru"}} {waiting}
vllm:request_success_total{{model_name="mineru",finished_reason="stop"}} {success}
vllm:generation_tokens_total{{model_name="mineru"}} {tokens}
{extra}'''


def test_metrics_parse_all_series_and_ignore_unrelated_exposition():
    snapshot = ms._parse_metrics(metrics(extra='''
vllm:num_requests_running{model_name="other"} 1 1792050000000
vllm:request_success_total{finished_reason="length",model_name="mineru"} 3
unrelated_metric{label="arbitrary"} NaN
'''))
    assert snapshot.busy is True
    assert sorted(snapshot.counters.values()) == [3, 12, 50]


def test_metric_label_order_and_escaping_do_not_change_counter_identity():
    first = metrics().replace('model_name="mineru",finished_reason="stop"',
                              r'model_name="mineru\\one",finished_reason="a\"b"')
    second = first.replace(r'model_name="mineru\\one",finished_reason="a\"b"',
                           r'finished_reason="a\"b", model_name="mineru\\one"')
    assert ms._parse_metrics(first).counters == ms._parse_metrics(second).counters


@pytest.mark.parametrize("extra", [
    'vllm:num_requests_waiting_by_reason{reason="capacity"} 1',
    'vllm:num_requests_waiting{model_name="other"} 2',
])
def test_any_waiting_series_marks_metrics_busy(extra):
    assert ms._parse_metrics(metrics(extra=extra)).busy is True


def test_generation_histogram_counters_can_back_up_missing_success_counter():
    text = '''vllm:num_requests_running 0
vllm:num_requests_waiting 0
vllm:generation_tokens_sum 25
vllm:generation_tokens_count 2
vllm:generation_tokens_bucket{le="+Inf"} 2
vllm:generation_tokens_created 1792050000
'''
    snapshot = ms._parse_metrics(text)
    assert snapshot.busy is False
    assert sorted(snapshot.counters.values()) == [2, 25]


@pytest.mark.parametrize("text", [
    "", "<html>unavailable</html>",
    'vllm:num_requests_running 0\nvllm:num_requests_waiting 0',
    'vllm:num_requests_running 0\nvllm:request_success_total 12',
    'vllm:num_requests_waiting 0\nvllm:request_success_total 12',
    metrics(running="NaN"), metrics(waiting="+Inf"), metrics(success=-1),
    metrics(tokens="broken"), metrics(extra='vllm:num_requests_running 0 0 extra'),
    metrics(extra='vllm:num_requests_running{model_name="unterminated} 0'),
    metrics(extra='vllm:num_requests_running{model_name="mineru"} 0'),
])
def test_unusable_metrics_raise_instead_of_authorizing_idle(text):
    with pytest.raises(ValueError):
        ms._parse_metrics(text)


class Clock:
    now = 0.0

    def __call__(self):
        return self.now


class Feed:
    text = metrics()
    error = None
    ready = True

    async def __call__(self):
        if self.error:
            raise self.error
        return self.text


@pytest.fixture
def watchdog(monkeypatch, tmp_path):
    import asyncio

    monkeypatch.setenv("PAPER_LIBRARY_MINERU_LEASE_FILE", str(tmp_path / "lease"))
    monkeypatch.setattr("papervault.library.services.concurrency.in_flight_keys", lambda: [])
    clock, feed = Clock(), Feed()
    controller = ms.MineruServerController(
        ondemand=True, base_url="http://127.0.0.1:30000", clock=clock, metrics_source=feed)
    controller._bus_ok = True
    calls = []

    async def systemctl(verb):
        assert verb == "stop"
        assert controller._lock.locked() and controller._stopping
        calls.append(verb)
        return 0, ""

    async def ready():
        return feed.ready

    async def active_age():
        return True

    controller._ready = ready
    controller._unit_active_long_enough = active_age
    controller._run_systemctl = systemctl
    return controller, clock, feed, calls, asyncio.Queue()


async def test_sustained_global_idle_stops_once_at_900_seconds(watchdog):
    c, clock, _, calls, queue = watchdog
    await c._monitor_once(queue)
    clock.now = 899
    await c._monitor_once(queue)
    assert calls == []
    clock.now = 900
    await c._monitor_once(queue)
    clock.now = 990
    await c._monitor_once(queue)
    assert calls == ["stop"]
    assert c._stopping is False


@pytest.mark.parametrize("busy", [
    metrics(running=1), metrics(waiting=1),
    metrics(extra='vllm:num_requests_waiting_by_reason{reason="capacity"} 1'),
])
async def test_any_global_request_gauge_prevents_stop(watchdog, busy):
    c, clock, feed, calls, queue = watchdog
    feed.text = busy
    await c._monitor_once(queue)
    clock.now = 1000
    await c._monitor_once(queue)
    assert calls == []
    feed.text = metrics()
    clock.now = 1090
    await c._monitor_once(queue)
    clock.now = 1989
    await c._monitor_once(queue)
    assert calls == []
    clock.now = 1990
    await c._monitor_once(queue)
    assert calls == ["stop"]


async def test_ready_active_grace_is_900_even_with_shorter_idle_timeout(watchdog, monkeypatch):
    monkeypatch.setattr(ms, "_IDLE_TIMEOUT", 30)
    c, clock, _, calls, queue = watchdog
    await c._monitor_once(queue)
    clock.now = 899
    await c._monitor_once(queue)
    assert calls == []
    clock.now = 900
    await c._monitor_once(queue)
    assert calls == ["stop"]


async def test_not_ready_and_new_ready_instance_cannot_inherit_idle(watchdog):
    c, clock, feed, calls, queue = watchdog
    await c._monitor_once(queue)
    feed.ready = False
    clock.now = 900
    await c._monitor_once(queue)
    assert calls == []
    feed.ready = True
    clock.now = 901
    await c._monitor_once(queue)
    clock.now = 1800
    await c._monitor_once(queue)
    assert calls == []
    clock.now = 1801
    await c._monitor_once(queue)
    assert calls == ["stop"]


@pytest.mark.parametrize("fresh", [metrics(success=13), metrics(tokens=51)])
async def test_fresh_success_or_token_counter_defers_idle_until_900_after_increment(watchdog, fresh):
    c, clock, feed, calls, queue = watchdog
    await c._monitor_once(queue)
    clock.now = 800
    feed.text = fresh
    await c._monitor_once(queue)
    clock.now = 1699
    await c._monitor_once(queue)
    assert calls == []
    clock.now = 1700
    await c._monitor_once(queue)
    assert calls == ["stop"]


async def test_counter_reset_restarts_active_grace(watchdog, monkeypatch):
    monkeypatch.setattr(ms, "_IDLE_TIMEOUT", 30)
    c, clock, feed, calls, queue = watchdog
    await c._monitor_once(queue)
    clock.now = 800
    feed.text = metrics(success=0, tokens=0)
    await c._monitor_once(queue)
    clock.now = 900
    await c._monitor_once(queue)
    assert calls == []
    clock.now = 1700
    await c._monitor_once(queue)
    assert calls == ["stop"]


async def test_per_series_counter_changes_cannot_cancel_each_other(watchdog):
    c, clock, feed, calls, queue = watchdog
    feed.text = metrics(success=12, tokens=50)
    await c._monitor_once(queue)
    clock.now = 900
    feed.text = metrics(success=13, tokens=49)
    await c._monitor_once(queue)
    assert calls == []


@pytest.mark.parametrize("final", [metrics(running=1), metrics(success=13), metrics(tokens=51)])
async def test_pre_stop_metrics_recheck_cancels_new_global_activity(watchdog, final):
    c, clock, _, calls, queue = watchdog
    await c._monitor_once(queue)
    snapshots = iter([metrics(), final])

    async def source():
        return next(snapshots)

    c._metrics_source = source
    clock.now = 900
    await c._monitor_once(queue)
    assert calls == []


@pytest.mark.parametrize("failure", [OSError("metrics down"), "", metrics(waiting="NaN")])
async def test_bad_metrics_warn_once_and_require_new_observation_window(watchdog, caplog, failure):
    c, clock, feed, calls, queue = watchdog
    await c._monitor_once(queue)
    clock.now = 900
    if isinstance(failure, Exception):
        feed.error = failure
    else:
        feed.text = failure
    await c._monitor_once(queue)
    assert calls == []
    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert len(warnings) == 1 and "metrics" in warnings[0].getMessage()
    feed.error, feed.text = None, metrics()
    clock.now = 901
    await c._monitor_once(queue)
    clock.now = 1800
    await c._monitor_once(queue)
    assert calls == []
    clock.now = 1801
    await c._monitor_once(queue)
    assert calls == ["stop"]


async def test_metrics_failure_on_final_recheck_cancels_stop(watchdog, caplog):
    c, clock, _, calls, queue = watchdog
    await c._monitor_once(queue)
    reads = 0

    async def source():
        nonlocal reads
        reads += 1
        if reads == 2:
            raise OSError("lost metrics just before stop")
        return metrics()

    c._metrics_source = source
    clock.now = 900
    await c._monitor_once(queue)
    assert calls == []
    assert len([r for r in caplog.records if r.levelname == "WARNING"]) == 1


@pytest.mark.parametrize("new_work", ["queue", "ocr", "extract"])
async def test_local_work_arriving_during_final_metrics_read_cancels_stop(
        watchdog, monkeypatch, new_work):
    c, clock, _, calls, queue = watchdog
    await c._monitor_once(queue)
    reads = 0

    async def source():
        nonlocal reads
        reads += 1
        if reads == 2:
            if new_work == "queue":
                queue.put_nowait("paper")
            elif new_work == "ocr":
                c._active_ocr = 1
            else:
                monkeypatch.setattr("papervault.library.services.concurrency.in_flight_keys",
                                    lambda: ["ex:paper"])
        return metrics()

    c._metrics_source = source
    clock.now = 900
    await c._monitor_once(queue)
    assert calls == []


async def test_local_activity_timestamp_no_longer_sets_global_idle(watchdog):
    c, clock, _, calls, queue = watchdog
    await c._monitor_once(queue)
    clock.now = 899
    c.note_activity()
    clock.now = 900
    await c._monitor_once(queue)
    assert calls == ["stop"]


@pytest.mark.parametrize("base_url", [
    "http://remote.example:30000", "http://127.0.0.1:30001", "http://127.0.0.1:30000/proxy",
])
async def test_watchdog_never_stops_for_another_endpoint(watchdog, base_url):
    c, clock, _, calls, queue = watchdog
    c._base = base_url
    await c._monitor_once(queue)
    clock.now = 900
    await c._monitor_once(queue)
    assert calls == []


async def test_off_mode_never_reads_metrics(watchdog):
    c, clock, feed, calls, queue = watchdog
    c._ondemand = False
    feed.error = AssertionError("off mode must not poll")
    await c._monitor_once(queue)
    clock.now = 900
    await c._monitor_once(queue)
    assert calls == []


async def test_metrics_http_uses_only_loopback_port_30000_without_redirects(monkeypatch):
    import httpx

    requested = []

    async def handler(request):
        requested.append(str(request.url))
        return httpx.Response(200, text=metrics())

    client = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient",
                        lambda **kwargs: client(transport=httpx.MockTransport(handler), **kwargs))
    c = ms.MineruServerController(ondemand=True, base_url="http://127.0.0.1:30000")
    assert await c._fetch_metrics() == metrics()
    assert requested == ["http://127.0.0.1:30000/metrics"]


@pytest.mark.parametrize("labels", ['{ }', '{model_name="mineru", }'])
def test_prometheus_whitespace_and_trailing_comma_are_valid(labels):
    text = (f"vllm:num_requests_running{labels} 0\n"
            "vllm:num_requests_waiting 0\nvllm:request_success_total 1\n")
    assert ms._parse_metrics(text).busy is False


async def test_readiness_lost_at_final_locked_probe_cancels_stop(watchdog):
    c, clock, _, calls, queue = watchdog
    await c._monitor_once(queue)
    readiness = iter([True, False])

    async def ready():
        return next(readiness)

    c._ready = ready
    clock.now = 900
    await c._monitor_once(queue)
    assert calls == []


@pytest.mark.parametrize("status", [302, 503])
async def test_metrics_http_errors_and_redirects_are_not_followed(monkeypatch, status):
    import httpx

    requested = []

    async def handler(request):
        requested.append(str(request.url))
        return httpx.Response(status, headers={"location": "http://remote.example/metrics"})

    client = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient",
                        lambda **kwargs: client(transport=httpx.MockTransport(handler), **kwargs))
    c = ms.MineruServerController(ondemand=True, base_url="http://127.0.0.1:30000")
    with pytest.raises(httpx.HTTPStatusError):
        await c._fetch_metrics()
    assert requested == ["http://127.0.0.1:30000/metrics"]


@pytest.mark.parametrize("action", ["start", "restart"])
async def test_controller_transition_cannot_transfer_idle_grace_to_new_engine(
        watchdog, monkeypatch, action):
    c, clock, _, calls, queue = watchdog
    monkeypatch.setattr(ms, "_IMAGE_DECODE_RESTART_AFTER", 1)
    await c._monitor_once(queue)

    async def systemctl(verb):
        assert c._lock.locked()
        if verb in {"stop", "restart"}:
            assert c._stopping
        calls.append(verb)
        # An outer monitor observation can finish while systemctl is awaited.
        # Its identical zero counters cannot identify the new engine.
        if verb == action:
            await c._observe_metrics()
        return 0, ""

    c._run_systemctl = systemctl
    clock.now = 899
    if action == "restart":
        await c.note_image_decode_failure()
    else:
        readiness = iter([False, False, True])

        async def ready():
            return next(readiness, True)

        c._ready = ready
        await c.ensure_ready()
    clock.now = 900
    await c._monitor_once(queue)
    assert calls == [action]
    clock.now = 1799
    await c._monitor_once(queue)
    assert calls == [action]
    clock.now = 1800
    await c._monitor_once(queue)
    assert calls == [action, "stop"]


async def test_newly_active_unit_blocks_stop_even_with_identical_metrics(watchdog):
    c, clock, _, calls, queue = watchdog
    await c._monitor_once(queue)
    clock.now = 900

    async def properties(verb):
        assert verb == "show" and c._lock.locked()
        return 0, "ActiveState=active\nActiveEnterTimestampMonotonic=890000000\n"

    c._run_systemctl = properties
    # Exercise the real age guard, with systemd replaced at its subprocess seam.
    c._unit_active_long_enough = ms.MineruServerController._unit_active_long_enough.__get__(c)
    await c._monitor_once(queue)
    assert calls == []


@pytest.mark.parametrize("rc,properties,want", [
    (0, "ActiveState=active\nActiveEnterTimestampMonotonic=100000000\n", True),
    (0, "ActiveState=active\nActiveEnterTimestampMonotonic=100000001\n", False),
    (0, "ActiveState=active\nActiveEnterTimestampMonotonic=1001000000\n", False),
    (0, "ActiveState=active\nActiveEnterTimestampMonotonic=0\n", False),
    (0, "ActiveState=inactive\nActiveEnterTimestampMonotonic=100000000\n", False),
    (0, "ActiveState=active\nActiveEnterTimestampMonotonic=NaN\n", False),
    (0, "ActiveState=active\n", False), (0, "", False), (1, "bus failure", False),
])
async def test_unit_age_guard_requires_valid_active_monotonic_age(watchdog, rc, properties, want):
    c, clock, _, _, _ = watchdog
    clock.now = 1000

    async def systemctl(verb):
        assert verb == "show"
        return rc, properties

    c._run_systemctl = systemctl
    assert await ms.MineruServerController._unit_active_long_enough(c) is want


async def test_unit_property_probe_requests_named_properties_and_uses_stdout(watchdog, monkeypatch):
    c, _, _, _, _ = watchdog
    commands = []

    class Process:
        returncode = 0

        async def communicate(self):
            return (b"ActiveState=active\nActiveEnterTimestampMonotonic=100000000\n", b"ignored stderr")

    async def spawn(*args, **kwargs):
        commands.append(args)
        return Process()

    monkeypatch.setattr(ms.asyncio, "create_subprocess_exec", spawn)
    rc, output = await ms.MineruServerController._run_systemctl(c, "show")
    assert rc == 0
    assert output == "ActiveState=active\nActiveEnterTimestampMonotonic=100000000\n"
    assert commands == [("systemctl", "--user", "show", "papervault-mineru.service",
                         "--property=ActiveState", "--property=ActiveEnterTimestampMonotonic")]


async def test_aborted_request_requires_full_zero_gauge_window_after_observed_clearance(watchdog):
    c, clock, feed, calls, queue = watchdog
    feed.text = metrics(running=1)
    await c._monitor_once(queue)
    clock.now = 899
    feed.text = metrics()  # no successful completion or token increment
    await c._monitor_once(queue)
    clock.now = 900
    await c._monitor_once(queue)
    assert calls == []
    clock.now = 1799
    await c._monitor_once(queue)
    assert calls == ["stop"]
