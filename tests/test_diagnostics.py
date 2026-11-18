"""Unit and interleaving tests for query-round diagnostics."""
from __future__ import annotations

import asyncio
import json

import fakeredis
import httpx
import pytest
import pytest_asyncio
from fastapi.testclient import TestClient

from app.config import Settings
from app.diagnostics import MAX_ROUND_SLOTS, Diagnostics, DiagnosticsError
from app.main import create_app


def make_settings(**overrides) -> Settings:
    base = {
        "redis_url": "redis://localhost:6379/0",
        "origin_url": "http://origin",
        "origin_timeout_ms": 1000,
        "cache_wait_timeout_ms": 3000,
        "lock_ttl_ms": 5000,
        "publish_timeout_ms": 2000,
        "redis_socket_timeout_ms": 1000,
        "pending_ttl_ms": 5000,
        "unconfirmed_ttl_s": 600,
        "revalidate_ttl_s": 300,
        "fetch_state_ttl_s": 30,
        "publish_record_ttl_s": 1000,
        "diagnostics_ttl_s": 600,
        "diagnostics_default_deadline_ms": 2000,
        "diagnostics_reaper_interval_s": 1,
        "instance_name": "test",
        "public_instances": ("http://a", "http://b"),
    }
    base.update(overrides)
    return Settings(**base)


class FakeOrigin:
    def __init__(self) -> None:
        self.tags: dict[str, dict] = {}
        self.calls = 0
        self.delay = 0.0

    def seed(self, tag: str, digest: str) -> None:
        self.tags[tag] = {"digest": digest, "revision": 1}

    async def get(self, url: str, timeout=None):
        self.calls += 1
        if timeout is not None and self.delay > timeout:
            await asyncio.sleep(timeout)
            raise httpx.ReadTimeout("simulated origin timeout")
        await asyncio.sleep(self.delay)
        tag = url.rsplit("/", 1)[-1]
        entry = self.tags.get(tag)
        if entry is None:
            return httpx.Response(404, json={"code": "tag_not_found"})
        return httpx.Response(200, json={"tag": tag, **entry})

    async def post(self, url: str, json=None, timeout=None):
        return httpx.Response(200, json=json or {})

    async def aclose(self):
        return None


@pytest_asyncio.fixture
async def diag_env():
    server = fakeredis.FakeServer()
    redis = fakeredis.aioredis.FakeRedis(server=server, decode_responses=True)
    settings = make_settings()
    diag = Diagnostics(redis, settings)
    yield redis, diag
    await redis.aclose()


def _entries(n: int) -> list[dict[str, str]]:
    return [{"instance": f"query-{i % 2 + 1}"} for i in range(n)]


@pytest.mark.asyncio
async def test_create_round_fixes_slots_tokens_and_deadline(diag_env):
    _redis, diag = diag_env
    doc = await diag.create_round("v1", _entries(16))
    assert doc["status"] == "open"
    assert doc["planned_count"] == 16
    assert doc["max_slots"] == MAX_ROUND_SLOTS
    assert len(doc["plan"]) == 16
    assert [item["slot"] for item in doc["plan"]] == list(range(16))
    assert len(doc["instances"]) == 2
    assert doc["complete"] is False
    assert doc["gap_count"] == 0
    # Tokens exist for every planned slot and are distinct secrets.
    tokens = await diag.redis.hgetall(diag._token_key(doc["round_id"]))
    assert set(tokens) == {str(i) for i in range(16)}
    assert len(set(tokens.values())) == 16
    assert doc["deadline_ms"] > doc["created_at_ms"]


@pytest.mark.asyncio
async def test_round_seals_when_all_slots_reported(diag_env):
    _redis, diag = diag_env
    doc = await diag.create_round("v1", _entries(4))
    round_id = doc["round_id"]
    tokens = await diag.redis.hgetall(diag._token_key(round_id))
    for slot in range(4):
        outcome = await diag.report(
            round_id,
            slot,
            tokens[str(slot)],
            {"instance": f"q{slot % 2 + 1}", "role": "leader" if slot == 0 else "follower",
             "result": "ok", "observed_generation": 1, "digest": "sha256:x"},
        )
    assert outcome["accepted"] == "recorded"
    assert outcome["sealed"] is True
    sealed = await diag.get_round(round_id)
    assert sealed["status"] == "sealed"
    assert sealed["seal_reason"] == "all_done"
    assert sealed["complete"] is True
    assert sealed["gap_count"] == 0
    assert sealed["gaps"] == []
    conclusion = sealed["conclusion"]
    assert conclusion["complete"] is True
    assert conclusion["roles"] == {"leader": 1, "follower": 3}
    assert conclusion["origin_called"] == 0
    assert conclusion["observed_digests"] == ["sha256:x"]
    assert conclusion["digest_consistent"] is True


@pytest.mark.asyncio
async def test_bad_token_is_rejected_and_slot_stays_open(diag_env):
    _redis, diag = diag_env
    doc = await diag.create_round("v1", _entries(3))
    round_id = doc["round_id"]
    outcome = await diag.report(
        round_id, 0, "wrong-token", {"instance": "q1", "result": "ok"}
    )
    assert outcome["accepted"] == "rejected"
    still_open = await diag.get_round(round_id)
    assert still_open["status"] == "open"
    assert all(slot["status"] == "gap" for slot in still_open["slots"])

    outcome = await diag.report(
        round_id, 99, "whatever", {"instance": "q1", "result": "ok"}
    )
    assert outcome["accepted"] == "rejected"


@pytest.mark.asyncio
async def test_deadline_seals_with_gaps_and_late_report_kept_separately(diag_env):
    redis, diag = diag_env
    doc = await diag.create_round("v1", _entries(4))
    round_id = doc["round_id"]
    tokens = await redis.hgetall(diag._token_key(round_id))
    # Slot 0 reports before the deadline.
    first = await diag.report(
        round_id, 0, tokens["0"],
        {"instance": "q1", "role": "leader", "result": "ok",
         "observed_generation": 0, "digest": "sha256:old"},
    )
    assert first["sealed"] is False
    # Force the deadline into the past, then a report arrives late.
    past = doc["created_at_ms"]
    await redis.hset(diag._round_key(round_id), "deadline_ms", str(past - 1000))
    late = await diag.report(
        round_id, 1, tokens["1"],
        {"instance": "q2", "role": "follower", "result": "ok",
         "observed_generation": 2, "digest": "sha256:new"},
    )
    assert late["accepted"] == "late"
    assert late["sealed"] is True
    sealed = await diag.get_round(round_id)
    assert sealed["status"] == "sealed"
    assert sealed["seal_reason"] == "deadline"
    assert sealed["complete"] is False
    assert sealed["gap_count"] == 3
    gap_slots = [s["slot"] for s in sealed["gaps"]]
    assert gap_slots == [1, 2, 3]
    # The late report is viewable but never filled the gap or changed the
    # sealed conclusion.
    assert len(sealed["late_reports"]) == 1
    assert sealed["late_reports"][0]["slot"] == 1
    assert sealed["late_reports"][0]["digest"] == "sha256:new"
    assert sealed["slots"][1]["status"] == "gap"
    assert sealed["conclusion"]["gap_count"] == 3
    assert sealed["conclusion"]["observed_digests"] == ["sha256:old"]
    # A second late report for the same slot does not duplicate.
    again = await diag.report(
        round_id, 2, tokens["2"],
        {"instance": "q1", "result": "error", "error_code": "origin_timeout"},
    )
    assert again["accepted"] == "late"
    view = await diag.get_round(round_id)
    assert {item["slot"] for item in view["late_reports"]} == {1, 2}
    assert view["gap_count"] == 3  # conclusion unchanged


@pytest.mark.asyncio
async def test_retries_create_separate_rounds_with_isolated_records(diag_env):
    _redis, diag = diag_env
    first = await diag.create_round("v1", _entries(2))
    second = await diag.create_round("v1", _entries(2))
    assert first["round_id"] != second["round_id"]
    tokens1 = await diag.redis.hgetall(diag._token_key(first["round_id"]))
    tokens2 = await diag.redis.hgetall(diag._token_key(second["round_id"]))
    # Old round fails; the failure must not leak into the new round.
    await diag.report(
        first["round_id"], 0, tokens1["0"],
        {"instance": "q1", "result": "error", "error_code": "origin_timeout",
         "role": "leader"},
    )
    await diag.report(
        first["round_id"], 1, tokens1["1"],
        {"instance": "q2", "result": "error", "error_code": "origin_timeout",
         "role": "follower"},
    )
    for slot in range(2):
        await diag.report(
            second["round_id"], slot, tokens2[str(slot)],
            {"instance": f"q{slot + 1}", "result": "ok", "role": "leader" if slot == 0 else "follower",
             "digest": "sha256:recovered", "observed_generation": 1},
        )
    old_view = await diag.get_round(first["round_id"])
    new_view = await diag.get_round(second["round_id"])
    assert old_view["conclusion"]["results"] == {"error": 2}
    assert new_view["conclusion"]["results"] == {"ok": 2}
    assert new_view["conclusion"]["observed_digests"] == ["sha256:recovered"]
    listing = await diag.list_rounds("v1")
    assert [r["round_id"] for r in listing["rounds"]] == [
        second["round_id"],
        first["round_id"],
    ]


@pytest.mark.asyncio
async def test_reaper_seals_overdue_open_rounds(diag_env):
    redis, diag = diag_env
    doc = await diag.create_round("v1", _entries(3))
    # A just-expired deadline (an ancient timestamp would also be trimmed from
    # the open index by the retention cleanup, which is intended).
    from app.diagnostics import _now_ms

    overdue = _now_ms() - 50
    round_key = diag._round_key(doc["round_id"])
    await redis.hset(round_key, "deadline_ms", str(overdue))
    # The open index scores rounds by deadline; keep it consistent, mirroring
    # what a genuinely expired round looks like.
    await redis.zadd("diag:index:open", {doc["round_id"]: overdue})
    sealed_count = await diag.reap_once()
    assert sealed_count == 1
    view = await diag.get_round(doc["round_id"])
    assert view["status"] == "sealed"
    assert view["seal_reason"] == "deadline"
    assert view["gap_count"] == 3
    # Reaping again is a no-op.
    assert await diag.reap_once() == 0


@pytest.mark.asyncio
async def test_unknown_and_expired_rounds_are_404(diag_env):
    _redis, diag = diag_env
    with pytest.raises(DiagnosticsError) as excinfo:
        await diag.get_round("r-does-not-exist")
    assert excinfo.value.status_code == 404
    with pytest.raises(DiagnosticsError):
        await diag.get_round("bad id!!")


@pytest.mark.asyncio
async def test_round_caps_at_16_slots(diag_env):
    _redis, diag = diag_env
    with pytest.raises(DiagnosticsError) as excinfo:
        await diag.create_round("v1", _entries(17))
    assert excinfo.value.status_code == 422
    doc = await diag.create_round("v1", _entries(16))
    assert doc["planned_count"] == 16


@pytest.mark.asyncio
async def test_leader_origin_error_marks_origin_called():
    """A 504 from the origin is a leader fetch: the round must say so."""
    server = fakeredis.FakeServer()
    redis = fakeredis.aioredis.FakeRedis(server=server, decode_responses=True)
    origin = FakeOrigin()
    origin.seed("v1", "sha256:slow")
    origin.delay = 0.5
    # FakeOrigin raises once its simulated delay exceeds the client timeout,
    # which is compared as a fraction of a second.
    settings = make_settings(origin_timeout_ms=50, cache_wait_timeout_ms=3000)
    app = create_app(settings)
    app.state.redis = redis
    app.state.origin = origin
    from app.cache import ServiceError, TagService

    app.state.service = TagService(redis, origin, settings)
    app.state.diagnostics = Diagnostics(redis, settings)
    client = TestClient(app)
    created = client.post(
        "/api/diagnostics/rounds",
        json={"tag": "v1", "entries": [{"instance": "test"}]},
    ).json()
    token = created["credentials"][0]["token"]
    resp = client.get(
        "/api/tag/v1",
        headers={
            "X-Diag-Round": created["round_id"],
            "X-Diag-Slot": "0",
            "X-Diag-Token": token,
        },
    )
    assert resp.status_code == 504
    detail = client.get(
        f"/api/diagnostics/rounds/{created['round_id']}"
    ).json()
    slot = detail["slots"][0]
    assert slot["status"] == "error"
    assert slot["error_code"] == "origin_timeout"
    assert slot["origin_called"] is True
    assert slot["role"] == "leader"
    assert detail["conclusion"]["origin_called"] == 1
    assert detail["conclusion"]["results"] == {"error": 1}


@pytest.mark.asyncio
async def test_diagnostics_failure_does_not_change_query_response():
    server = fakeredis.FakeServer()
    redis = fakeredis.aioredis.FakeRedis(server=server, decode_responses=True)
    origin = FakeOrigin()
    origin.seed("v1", "sha256:1111")
    settings = make_settings(instance_name="query-1")
    app = create_app(settings)
    app.state.redis = redis
    app.state.origin = origin
    from app.cache import TagService

    app.state.service = TagService(redis, origin, settings)
    app.state.diagnostics = Diagnostics(redis, settings)
    client = TestClient(app)

    # Create a round, then corrupt the report path: the query must still return
    # its normal 200 response and no diagnostics field must leak into it.
    created = client.post(
        "/api/diagnostics/rounds",
        json={"tag": "v1", "entries": [{"instance": "query-1"}]},
    ).json()
    credential = {item["slot"]: item["token"] for item in created["credentials"]}
    token = credential[0]

    from redis.exceptions import ConnectionError as RedisConnectionError

    real_eval = redis.eval

    async def broken_eval(script, *args, **kwargs):
        # Only the report script is broken; other Lua keeps working.
        if "local stored" in script:
            raise RedisConnectionError("simulated diagnostics outage")
        return await real_eval(script, *args, **kwargs)

    redis.eval = broken_eval
    try:
        resp = client.get(
            "/api/tag/v1",
            headers={
                "X-Diag-Round": created["round_id"],
                "X-Diag-Slot": "0",
                "X-Diag-Token": token,
            },
        )
    finally:
        redis.eval = real_eval
    assert resp.status_code == 200
    body = resp.json()
    assert body["digest"] == "sha256:1111"
    assert "diag" not in json.dumps(body).lower().replace("diagnostics", "")
    # The round stays open with a gap because the report could not be stored.
    view = client.get(f"/api/diagnostics/rounds/{created['round_id']}").json()
    assert view["status"] == "open"
    assert view["slots"][0]["status"] == "gap"


@pytest.mark.asyncio
async def test_full_round_through_http_endpoints():
    server = fakeredis.FakeServer()
    redis = fakeredis.aioredis.FakeRedis(server=server, decode_responses=True)
    origin = FakeOrigin()
    origin.seed("v1", "sha256:1111")
    settings = make_settings(instance_name="query-1")
    app = create_app(settings)
    app.state.redis = redis
    app.state.origin = origin
    from app.cache import TagService

    app.state.service = TagService(redis, origin, settings)
    app.state.diagnostics = Diagnostics(redis, settings)
    client = TestClient(app)

    created = client.post(
        "/api/diagnostics/rounds",
        json={
            "tag": "v1",
            "entries": [
                {"instance": "query-1"},
                {"instance": "query-2"},
            ],
        },
    )
    assert created.status_code == 200
    round_doc = created.json()
    round_id = round_doc["round_id"]
    credential = {item["slot"]: item["token"] for item in round_doc["credentials"]}
    headers = [
        {
            "X-Diag-Round": round_id,
            "X-Diag-Slot": str(item["slot"]),
            "X-Diag-Token": credential[item["slot"]],
        }
        for item in round_doc["plan"]
    ]
    responses = [client.get("/api/tag/v1", headers=h) for h in headers]
    assert all(r.status_code == 200 for r in responses)
    detail = client.get(f"/api/diagnostics/rounds/{round_id}").json()
    assert detail["status"] == "sealed"
    assert detail["seal_reason"] == "all_done"
    assert detail["complete"] is True
    assert len(detail["slots"]) == 2
    roles = {slot["role"] for slot in detail["slots"]}
    # The TestClient issues requests sequentially, so the second one is a
    # cache hit ("cache") rather than a waiting follower.
    assert roles == {"leader", "cache"}
    # Both requests hit this single TestClient instance; the planned instance
    # names from the round are still preserved in the plan.
    assert {slot["instance"] for slot in detail["slots"]} == {"query-1"}
    assert {item["instance"] for item in detail["plan"]} == {"query-1", "query-2"}
    assert detail["conclusion"]["origin_called"] == 1
    assert detail["conclusion"]["results"] == {"ok": 2}
    listing = client.get("/api/diagnostics/rounds", params={"tag": "v1"}).json()
    assert listing["rounds"][0]["round_id"] == round_id
    assert listing["rounds"][0]["complete"] is True
    # Validation: oversize round rejected explicitly, never silently truncated.
    bad = client.post(
        "/api/diagnostics/rounds",
        json={"tag": "v1", "entries": _entries(17)},
    )
    assert bad.status_code == 422
