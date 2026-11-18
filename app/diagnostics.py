"""Query-round diagnostics for concurrent same-tag queries.

A *round* is a fixed batch of at most :data:`MAX_ROUND_SLOTS` query requests
for one tag, created up front by the web page.  At creation time every request
gets a permanent slot number (its fixed request id), a per-slot token and an
absolute deadline.  The two query instances then report what each request did
(role, result, observed publication generation) into its own slot.

Correctness rules implemented here:

* Slot numbers, tokens and the deadline are fixed at round creation and never
  change afterwards.  A report without the matching slot token is rejected.
* A round is sealed exactly once, either when every planned slot has reported
  (``all_done``) or when the deadline has passed (``deadline``).  Sealing is an
  atomic Redis operation; slots still empty at that moment become permanent
  *gaps*.
* Reports arriving after sealing (or after the deadline) are kept in a separate
  ``late`` bucket.  They remain viewable but can never modify the sealed
  conclusion or fill a recorded gap.
* Retrying a tag creates a brand new round; an old round's records (including
  its errors) can never overwrite or merge into the new one.
* Every diagnostics key carries the same TTL
  (``DIAGNOSTICS_TTL_S``, default 600 seconds / 10 minutes), so records remove
  themselves.
* All writes are best-effort from the query path: a diagnostics failure is
  logged and swallowed and must never change the original query response.  The
  management endpoints (create/list/get) instead surface explicit errors and
  never fabricate a complete-looking record when slots are missing.

Redis layout (all keys prefixed ``diag:``)::

    diag:round:{id}         hash   round metadata + sealed conclusion
    diag:round:{id}:slots   hash   slot index -> JSON report (frozen at seal)
    diag:round:{id}:late    hash   slot index -> JSON late report
    diag:round:{id}:tokens  hash   slot index -> secret token
    diag:index:tag:{tag}    zset   round id -> created-at score
    diag:index:global       zset   round id -> created-at score
    diag:index:open         zset   round id -> deadline score (sealed rounds removed)
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import uuid
from datetime import UTC, datetime
from typing import Any

from redis.asyncio import Redis
from redis.exceptions import RedisError, ResponseError

from .config import Settings

logger = logging.getLogger("diagnostics")

MAX_ROUND_SLOTS = 16
# Deadlines requested by clients must live inside this window.
MIN_DEADLINE_MS = 500
MAX_DEADLINE_MS = 120_000
ROUND_ID_RE = re.compile(r"^[A-Za-z0-9_-]{8,64}$")
INSTANCE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,119}$")

# KEYS[1]=round hash, 2=slots, 3=late, 4=tokens, 5=open index,
#      6=tag index, 7=global index
# ARGV: 1 round id, 2 tag, 3 created_at_ms, 4 deadline_ms, 5 planned_count,
#       6 instances json, 7 plan json, 8 ttl seconds, 9 now_ms (open trim),
#       then slot/token pairs.
_CREATE_ROUND_LUA = """
if redis.call('EXISTS', KEYS[1]) == 1 then
  return redis.error_reply('round already exists')
end
redis.call('HSET', KEYS[1],
  'round_id', ARGV[1],
  'tag', ARGV[2],
  'created_at_ms', ARGV[3],
  'deadline_ms', ARGV[4],
  'planned_count', ARGV[5],
  'instances', ARGV[6],
  'plan', ARGV[7],
  'status', 'open',
  'seal_reason', '',
  'sealed_at_ms', '0',
  'gap_count', '0')
local i = 10
while ARGV[i] do
  redis.call('HSET', KEYS[4], ARGV[i], ARGV[i + 1])
  i = i + 2
end
local ttl = tonumber(ARGV[8])
for _, k in ipairs({KEYS[1], KEYS[2], KEYS[3], KEYS[4]}) do
  redis.call('EXPIRE', k, ttl)
end
redis.call('ZADD', KEYS[5], tonumber(ARGV[4]), ARGV[1])
redis.call('ZADD', KEYS[6], tonumber(ARGV[3]), ARGV[1])
redis.call('ZADD', KEYS[7], tonumber(ARGV[3]), ARGV[1])
-- Drop open-index members whose deadline is older than one retention window;
-- their round keys have expired by then.
redis.call('ZREMRANGEBYSCORE', KEYS[5], 0, tonumber(ARGV[9]) - ttl * 1000)
return 1
"""

# KEYS[1]=round hash, 2=slots, 3=late, 4=tokens, 5=open index
# ARGV: 1 slot index, 2 token, 3 now_ms, 4 report json, 5 ttl seconds
_REPORT_LUA = """
local stored = redis.call('HGET', KEYS[4], ARGV[1])
if false == stored then return redis.error_reply('unknown slot') end
if stored ~= ARGV[2] then return redis.error_reply('bad slot token') end
local now = tonumber(ARGV[3])
local ttl = tonumber(ARGV[5])
local function refresh_ttls()
  for _, k in ipairs({KEYS[1], KEYS[2], KEYS[3], KEYS[4]}) do
    redis.call('EXPIRE', k, ttl)
  end
end
local status = redis.call('HGET', KEYS[1], 'status')
local deadline = tonumber(redis.call('HGET', KEYS[1], 'deadline_ms'))
local planned = tonumber(redis.call('HGET', KEYS[1], 'planned_count'))
local function seal(reason, gaps)
  redis.call('HSET', KEYS[1],
    'status', 'sealed',
    'seal_reason', reason,
    'sealed_at_ms', tostring(now),
    'gap_count', tostring(gaps))
  redis.call('ZREM', KEYS[5], redis.call('HGET', KEYS[1], 'round_id'))
end
if status == 'sealed' then
  if redis.call('HEXISTS', KEYS[3], ARGV[1]) == 0 then
    redis.call('HSET', KEYS[3], ARGV[1], ARGV[4])
  end
  refresh_ttls()
  return {'late', 0, redis.call('HLEN', KEYS[2])}
end
if now > deadline then
  -- The deadline has passed: freeze the conclusion first, then keep this
  -- report as a late one.  It must never fill a gap in the sealed record.
  local done = redis.call('HLEN', KEYS[2])
  seal('deadline', planned - done)
  if redis.call('HEXISTS', KEYS[3], ARGV[1]) == 0 then
    redis.call('HSET', KEYS[3], ARGV[1], ARGV[4])
  end
  refresh_ttls()
  return {'late', 1, done}
end
if redis.call('HEXISTS', KEYS[2], ARGV[1]) == 0 then
  redis.call('HSET', KEYS[2], ARGV[1], ARGV[4])
end
local done = redis.call('HLEN', KEYS[2])
if done >= planned then
  seal('all_done', 0)
  refresh_ttls()
  return {'recorded', 1, done}
end
refresh_ttls()
return {'recorded', 0, done}
"""

# KEYS[1]=round hash, 2=slots, 3=open index
# ARGV: 1 now_ms
_SEAL_IF_DUE_LUA = """
local status = redis.call('HGET', KEYS[1], 'status')
if status ~= 'open' then return {0, status} end
local now = tonumber(ARGV[1])
local deadline = tonumber(redis.call('HGET', KEYS[1], 'deadline_ms'))
if now <= deadline then return {0, 'open'} end
local planned = tonumber(redis.call('HGET', KEYS[1], 'planned_count'))
local done = redis.call('HLEN', KEYS[2])
local gaps = planned - done
redis.call('HSET', KEYS[1],
  'status', 'sealed',
  'seal_reason', 'deadline',
  'sealed_at_ms', tostring(now),
  'gap_count', tostring(gaps))
redis.call('ZREM', KEYS[3], redis.call('HGET', KEYS[1], 'round_id'))
return {1, 'sealed', gaps}
"""


def _now_ms() -> int:
    return int(time.time() * 1000)


def _iso(ms: int | str | None) -> str | None:
    if ms in (None, ""):
        return None
    try:
        value = int(ms)
    except (TypeError, ValueError):
        return None
    if value <= 0:
        return None
    return datetime.fromtimestamp(value / 1000, tz=UTC).isoformat(
        timespec="milliseconds"
    )


class DiagnosticsError(Exception):
    """Raised by the management API when diagnostics state cannot be served."""

    def __init__(self, code: str, message: str, status_code: int = 503) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


class Diagnostics:
    def __init__(self, redis: Redis, settings: Settings) -> None:
        self.redis = redis
        self.s = settings
        self._reaper_task: asyncio.Task[None] | None = None

    # ------------------------------------------------------------------ keys
    @staticmethod
    def _round_key(round_id: str) -> str:
        return f"diag:round:{round_id}"

    @staticmethod
    def _slot_key(round_id: str) -> str:
        return f"diag:round:{round_id}:slots"

    @staticmethod
    def _late_key(round_id: str) -> str:
        return f"diag:round:{round_id}:late"

    @staticmethod
    def _token_key(round_id: str) -> str:
        return f"diag:round:{round_id}:tokens"

    @staticmethod
    def _tag_index_key(tag: str) -> str:
        return f"diag:index:tag:{tag}"

    # ------------------------------------------------------------- validation
    @staticmethod
    def validate_round_id(round_id: str) -> None:
        if not round_id or not ROUND_ID_RE.match(round_id):
            raise DiagnosticsError("invalid_round_id", "轮次编号格式无效", 422)

    @staticmethod
    def validate_entries(entries: list[dict[str, str]]) -> list[dict[str, str]]:
        if not isinstance(entries, list) or not entries:
            raise DiagnosticsError(
                "invalid_round", "每轮至少需要 1 个请求", 422
            )
        if len(entries) > MAX_ROUND_SLOTS:
            raise DiagnosticsError(
                "round_too_large",
                f"每轮最多 {MAX_ROUND_SLOTS} 个请求，收到 {len(entries)} 个",
                422,
            )
        clean: list[dict[str, str]] = []
        for index, entry in enumerate(entries):
            instance = str((entry or {}).get("instance") or "").strip()
            if not INSTANCE_RE.match(instance):
                raise DiagnosticsError(
                    "invalid_instance",
                    f"第 {index} 个请求的实例名无效：{instance!r}",
                    422,
                )
            clean.append({"instance": instance})
        return clean

    # ------------------------------------------------------------ round life
    async def create_round(
        self,
        tag: str,
        entries: list[dict[str, str]],
        deadline_ms: int | None = None,
    ) -> dict[str, Any]:
        """Fix slot numbers, tokens and the deadline for a brand-new round."""
        clean = self.validate_entries(entries)
        now = _now_ms()
        if deadline_ms is None:
            deadline = now + self.s.diagnostics_default_deadline_ms
        else:
            deadline = int(deadline_ms)
            if not (now + MIN_DEADLINE_MS <= deadline <= now + MAX_DEADLINE_MS):
                raise DiagnosticsError(
                    "invalid_deadline",
                    f"截止时间需在当前时间之后 {MIN_DEADLINE_MS}-{MAX_DEADLINE_MS}ms 之间",
                    422,
                )
        round_id = "r" + uuid.uuid4().hex
        # The stored plan is token-less; credentials are returned to the caller
        # once, at creation time, and otherwise only live in the token hash.
        plan = [
            {"slot": index, "instance": entry["instance"]}
            for index, entry in enumerate(clean)
        ]
        credentials = [
            {"slot": index, "token": uuid.uuid4().hex}
            for index in range(len(clean))
        ]
        argv: list[str] = [
            round_id,
            tag,
            str(now),
            str(deadline),
            str(len(plan)),
            json.dumps(sorted({item["instance"] for item in plan})),
            json.dumps(plan),
            str(self.s.diagnostics_ttl_s),
            str(now),
        ]
        for item in credentials:
            argv.extend([str(item["slot"]), item["token"]])
        try:
            await self.redis.eval(
                _CREATE_ROUND_LUA,
                7,
                self._round_key(round_id),
                self._slot_key(round_id),
                self._late_key(round_id),
                self._token_key(round_id),
                "diag:index:open",
                self._tag_index_key(tag),
                "diag:index:global",
                *argv,
            )
        except RedisError as exc:
            raise DiagnosticsError(
                "diagnostics_unavailable",
                f"诊断记录写入失败（{exc.__class__.__name__}），请重试",
            ) from exc
        doc = await self.get_round(round_id)
        doc["credentials"] = credentials
        return doc

    async def report(
        self,
        round_id: str,
        slot: int,
        token: str,
        report: dict[str, Any],
    ) -> dict[str, Any]:
        """Store one request's record. Never raises for business-level rejects.

        The return value describes what happened (``recorded`` / ``late`` /
        ``rejected``) so the query path can log it; the query response itself
        is untouched either way.
        """
        if not ROUND_ID_RE.match(round_id or ""):
            return {"accepted": "rejected", "reason": "invalid_round_id"}
        if not isinstance(slot, int) or not 0 <= slot < MAX_ROUND_SLOTS:
            return {"accepted": "rejected", "reason": "invalid_slot"}
        if not token or len(token) > 128:
            return {"accepted": "rejected", "reason": "invalid_token"}
        payload = json.dumps(report, separators=(",", ":"), ensure_ascii=False)
        try:
            outcome = await self.redis.eval(
                _REPORT_LUA,
                5,
                self._round_key(round_id),
                self._slot_key(round_id),
                self._late_key(round_id),
                self._token_key(round_id),
                "diag:index:open",
                str(slot),
                token,
                str(_now_ms()),
                payload,
                str(self.s.diagnostics_ttl_s),
            )
        except ResponseError as exc:
            # Script-level business rejects (unknown slot, bad token).
            return {"accepted": "rejected", "reason": str(exc)}
        except RedisError as exc:
            # Swallow: diagnostics must never change the query response.
            logger.warning(
                "diagnostics report failed round=%s slot=%s: %r",
                round_id,
                slot,
                exc,
            )
            return {"accepted": "failed", "reason": exc.__class__.__name__}
        except Exception as exc:  # noqa: BLE001 - defensive boundary
            logger.warning(
                "diagnostics report rejected round=%s slot=%s: %r",
                round_id,
                slot,
                exc,
            )
            return {"accepted": "rejected", "reason": exc.__class__.__name__}
        accepted, sealed, done = outcome
        result = {
            "accepted": str(accepted),
            "sealed": bool(int(sealed)),
            "recorded_slots": int(done),
        }
        if accepted == "late":
            result["note"] = "轮次已封存或已过截止，记录仅作迟到结果查看"
        return result

    async def _seal_if_due(self, round_id: str) -> None:
        try:
            await self.redis.eval(
                _SEAL_IF_DUE_LUA,
                3,
                self._round_key(round_id),
                self._slot_key(round_id),
                "diag:index:open",
                str(_now_ms()),
            )
        except RedisError:
            # Read paths must still work with whatever state Redis currently
            # holds; the reaper will retry the seal later.
            pass

    # ---------------------------------------------------------------- reads
    async def get_round(self, round_id: str) -> dict[str, Any]:
        self.validate_round_id(round_id)
        await self._seal_if_due(round_id)
        try:
            pipe = self.redis.pipeline()
            pipe.hgetall(self._round_key(round_id))
            pipe.hgetall(self._slot_key(round_id))
            pipe.hgetall(self._late_key(round_id))
            meta_raw, slots_raw, late_raw = await pipe.execute()
        except RedisError as exc:
            raise DiagnosticsError(
                "diagnostics_unavailable",
                f"诊断记录读取失败（{exc.__class__.__name__}）",
            ) from exc
        if not meta_raw:
            raise DiagnosticsError("round_not_found", "轮次不存在或已过保留期", 404)
        doc = self._assemble(meta_raw, slots_raw, late_raw)
        if doc["status"] == "sealed" and not doc.get("conclusion"):
            conclusion = self._build_conclusion(doc)
            doc["conclusion"] = conclusion
            # Persist once; slots are immutable after sealing so the conclusion
            # is stable even if this best-effort write fails.
            try:
                await self.redis.hset(
                    self._round_key(round_id),
                    "conclusion",
                    json.dumps(conclusion, ensure_ascii=False),
                )
            except RedisError:
                pass
        return doc

    async def list_rounds(
        self, tag: str | None = None, limit: int = 20
    ) -> dict[str, Any]:
        limit = max(1, min(int(limit or 20), 100))
        now = _now_ms()
        index = self._tag_index_key(tag) if tag else "diag:index:global"
        try:
            # Members older than one retention window reference expired keys.
            await self.redis.zremrangebyscore(
                index, 0, now - self.s.diagnostics_ttl_s * 1000
            )
            ids = await self.redis.zrevrange(index, 0, limit - 1)
            if not ids:
                return {"tag": tag, "rounds": []}
            pipe = self.redis.pipeline()
            for round_id in ids:
                pipe.hgetall(self._round_key(round_id))
            metas = await pipe.execute()
        except RedisError as exc:
            raise DiagnosticsError(
                "diagnostics_unavailable",
                f"诊断索引读取失败（{exc.__class__.__name__}）",
            ) from exc
        rounds = []
        for round_id, meta in zip(ids, metas):
            if not meta:
                continue
            rounds.append(self._summary(round_id, meta))
        return {"tag": tag, "rounds": rounds}

    # ------------------------------------------------------------- assembly
    def _assemble(
        self,
        meta_raw: dict[str, str],
        slots_raw: dict[str, str],
        late_raw: dict[str, str],
    ) -> dict[str, Any]:
        planned = int(meta_raw.get("planned_count") or 0)
        try:
            plan = json.loads(meta_raw.get("plan") or "[]")
        except ValueError:
            plan = []
        plan_by_slot = {int(item["slot"]): item for item in plan if "slot" in item}
        slots: list[dict[str, Any]] = []
        for index in range(planned):
            raw = slots_raw.get(str(index))
            if raw is None:
                planned_here = plan_by_slot.get(index, {})
                slots.append(
                    {
                        "slot": index,
                        "status": "gap",
                        "instance": planned_here.get("instance"),
                        "request_id": None,
                        "note": "轮次封存时该请求未提交记录",
                    }
                )
                continue
            try:
                report = json.loads(raw)
            except ValueError:
                report = {"status": "corrupt", "raw": raw}
            report.setdefault("slot", index)
            slots.append(report)
        late: list[dict[str, Any]] = []
        for index in sorted(late_raw, key=lambda x: int(x)):
            try:
                report = json.loads(late_raw[index])
            except ValueError:
                report = {"status": "corrupt", "raw": late_raw[index]}
            report.setdefault("slot", int(index))
            report["late"] = True
            late.append(report)
        status = meta_raw.get("status", "open")
        gap_count = int(meta_raw.get("gap_count") or 0)
        doc: dict[str, Any] = {
            "round_id": meta_raw.get("round_id"),
            "tag": meta_raw.get("tag"),
            "status": status,
            "seal_reason": meta_raw.get("seal_reason") or None,
            "created_at": _iso(meta_raw.get("created_at_ms")),
            "deadline": _iso(meta_raw.get("deadline_ms")),
            "sealed_at": _iso(meta_raw.get("sealed_at_ms")),
            "deadline_ms": int(meta_raw.get("deadline_ms") or 0),
            "created_at_ms": int(meta_raw.get("created_at_ms") or 0),
            "sealed_at_ms": int(meta_raw.get("sealed_at_ms") or 0),
            "max_slots": MAX_ROUND_SLOTS,
            "planned_count": planned,
            "instances": json.loads(meta_raw.get("instances") or "[]"),
            "plan": plan,
            "slots": slots,
            "gaps": [item for item in slots if item.get("status") == "gap"],
            "late_reports": late,
            "gap_count": gap_count,
            "complete": status == "sealed" and gap_count == 0,
            "retention_ttl_s": self.s.diagnostics_ttl_s,
        }
        conclusion_raw = meta_raw.get("conclusion")
        if conclusion_raw:
            try:
                doc["conclusion"] = json.loads(conclusion_raw)
            except ValueError:
                doc["conclusion"] = None
        return doc

    @staticmethod
    def _summary(round_id: str, meta: dict[str, str]) -> dict[str, Any]:
        return {
            "round_id": meta.get("round_id", round_id),
            "tag": meta.get("tag"),
            "status": meta.get("status", "open"),
            "seal_reason": meta.get("seal_reason") or None,
            "created_at": _iso(meta.get("created_at_ms")),
            "deadline": _iso(meta.get("deadline_ms")),
            "sealed_at": _iso(meta.get("sealed_at_ms")),
            "planned_count": int(meta.get("planned_count") or 0),
            "gap_count": int(meta.get("gap_count") or 0),
            "complete": meta.get("status") == "sealed"
            and int(meta.get("gap_count") or 0) == 0,
        }

    def _build_conclusion(self, doc: dict[str, Any]) -> dict[str, Any]:
        """Immutable summary computed from the frozen slot records."""
        records = [s for s in doc["slots"] if s.get("status") != "gap"]
        by_role: dict[str, int] = {}
        by_instance: dict[str, int] = {}
        by_result: dict[str, int] = {}
        digest_set: set[str] = set()
        generation_set: set[int] = set()
        origin_called = 0
        for record in records:
            role = str(record.get("role") or "unknown")
            instance = str(record.get("instance") or "unknown")
            result_kind = "error" if record.get("result") == "error" else "ok"
            by_role[role] = by_role.get(role, 0) + 1
            by_instance[instance] = by_instance.get(instance, 0) + 1
            by_result[result_kind] = by_result.get(result_kind, 0) + 1
            if record.get("origin_called"):
                origin_called += 1
            if record.get("digest"):
                digest_set.add(str(record["digest"]))
            if record.get("observed_generation") is not None:
                try:
                    generation_set.add(int(record["observed_generation"]))
                except (TypeError, ValueError):
                    pass
        return {
            "sealed_reason": doc.get("seal_reason"),
            "planned": doc["planned_count"],
            "recorded": len(records),
            "gap_count": len(doc["gaps"]),
            "complete": not doc["gaps"],
            "results": by_result,
            "roles": by_role,
            "instances": by_instance,
            "origin_called": origin_called,
            "observed_digests": sorted(digest_set),
            "observed_generations": sorted(generation_set),
            "digest_consistent": len(digest_set) <= 1,
            "late_report_count": len(doc["late_reports"]),
        }

    # ---------------------------------------------------------------- reaper
    async def reaper_loop(self) -> None:
        """Seal open rounds whose deadline passed, even if nobody reads them."""
        while True:
            try:
                await asyncio.sleep(self.s.diagnostics_reaper_interval_s)
                await self.reap_once()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - keep the loop alive
                logger.warning("diagnostics reaper iteration failed: %r", exc)

    async def reap_once(self) -> int:
        now = _now_ms()
        try:
            await self.redis.zremrangebyscore(
                "diag:index:open", 0, now - self.s.diagnostics_ttl_s * 1000
            )
            due = await self.redis.zrangebyscore("diag:index:open", 0, now)
        except RedisError as exc:
            logger.warning("reaper could not read open index: %r", exc)
            return 0
        sealed = 0
        for round_id in due:
            try:
                outcome = await self.redis.eval(
                    _SEAL_IF_DUE_LUA,
                    3,
                    self._round_key(round_id),
                    self._slot_key(round_id),
                    "diag:index:open",
                    str(now),
                )
                if int(outcome[0]) == 1:
                    sealed += 1
                    await self.get_round(round_id)  # persist conclusion
            except (RedisError, DiagnosticsError) as exc:
                logger.warning("reaper seal failed for %s: %r", round_id, exc)
        return sealed

    def start_reaper(self) -> None:
        if self._reaper_task is None or self._reaper_task.done():
            self._reaper_task = asyncio.create_task(self.reaper_loop())

    async def stop_reaper(self) -> None:
        if self._reaper_task and not self._reaper_task.done():
            self._reaper_task.cancel()
            try:
                await self._reaper_task
            except asyncio.CancelledError:
                pass
        self._reaper_task = None
