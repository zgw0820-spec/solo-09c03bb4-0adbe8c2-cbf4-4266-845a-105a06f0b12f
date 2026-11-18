from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from time import perf_counter
from typing import Any

import httpx
from fastapi import FastAPI, Header, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from redis.asyncio import Redis

from .cache import ServiceError, TagService
from .config import Settings, settings
from .diagnostics import Diagnostics, DiagnosticsError, MAX_ROUND_SLOTS

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("gateway")

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")


class PublishRequest(BaseModel):
    tag: str = Field(min_length=1, max_length=200)
    digest: str = Field(min_length=1, max_length=200)
    request_id: str | None = Field(default=None, max_length=120)


class TagBody(BaseModel):
    tag: str = Field(min_length=1, max_length=200)


class OriginControlRequest(BaseModel):
    delay_ms: int | None = Field(default=None, ge=0, le=120_000)
    fail_next: int | None = Field(default=None, ge=0, le=100)
    slow_next_ms: int | None = Field(default=None, ge=0, le=120_000)
    fail_publish_next: int | None = Field(default=None, ge=0, le=100)
    publish_delay_ms: int | None = Field(default=None, ge=0, le=120_000)
    reset_counters: bool = False


class RoundEntry(BaseModel):
    instance: str = Field(min_length=1, max_length=120)


class RoundCreateRequest(BaseModel):
    tag: str = Field(min_length=1, max_length=200)
    entries: list[RoundEntry] = Field(min_length=1, max_length=MAX_ROUND_SLOTS)
    deadline_ms: int | None = Field(default=None, ge=0)


def _error_payload(exc: ServiceError, instance: str) -> dict[str, Any]:
    return {
        "error": {
            "code": exc.code,
            "message": exc.message,
            "retryable": exc.retryable,
            "status": exc.status_code,
            **exc.extra,
        },
        "instance": instance,
    }


def _install_error_handler(app: FastAPI) -> None:
    @app.exception_handler(ServiceError)
    async def _service_error(request: Request, exc: ServiceError) -> JSONResponse:
        headers = {"X-Instance": app.state.settings.instance_name}
        if exc.retryable:
            headers["Retry-After"] = "1"
        body = _error_payload(exc, app.state.settings.instance_name)
        body["requested_path"] = request.url.path
        return JSONResponse(body, status_code=exc.status_code, headers=headers)

    @app.exception_handler(DiagnosticsError)
    async def _diagnostics_error(
        request: Request, exc: DiagnosticsError
    ) -> JSONResponse:
        # Diagnostics endpoints report failures explicitly instead of
        # pretending an incomplete record is complete.
        return JSONResponse(
            {
                "error": {
                    "code": exc.code,
                    "message": exc.message,
                    "status": exc.status_code,
                },
                "instance": app.state.settings.instance_name,
                "requested_path": request.url.path,
            },
            status_code=exc.status_code,
            headers={"X-Instance": app.state.settings.instance_name},
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg: Settings = app.state.settings
    redis = Redis.from_url(
        cfg.redis_url,
        decode_responses=True,
        socket_timeout=cfg.redis_socket_timeout_ms / 1000,
        socket_connect_timeout=cfg.redis_socket_timeout_ms / 1000,
        health_check_interval=15,
        retry_on_timeout=False,
    )
    origin = httpx.AsyncClient(
        base_url=cfg.origin_url,
        timeout=cfg.origin_timeout_ms / 1000,
        limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
    )
    app.state.redis = redis
    app.state.origin = origin
    app.state.service = TagService(redis, origin, cfg)
    app.state.diagnostics = Diagnostics(redis, cfg)
    try:
        await redis.ping()
        logger.info(
            "instance %s connected to redis, origin=%s",
            cfg.instance_name,
            cfg.origin_url,
        )
    except Exception as exc:  # noqa: BLE001 - startup diagnostics only
        logger.warning("redis not reachable at startup: %r", exc)
    app.state.diagnostics.start_reaper()
    try:
        yield
    finally:
        await app.state.diagnostics.stop_reaper()
        await origin.aclose()
        await redis.aclose()


def _parse_diag_headers(
    round_id: str | None, slot_raw: str | None, token: str | None
) -> dict[str, Any] | None:
    """Validate the diagnostics headers; malformed ones disable diagnostics."""
    if not round_id:
        return None
    try:
        slot = int(slot_raw) if slot_raw is not None else -1
    except (TypeError, ValueError):
        slot = -1
    if slot < 0 or not token:
        logger.warning(
            "ignoring diagnostics request with bad slot/token round=%s", round_id
        )
        return None
    return {"round_id": round_id, "slot": slot, "token": token}


async def _report_diagnostics(
    diagnostics: Diagnostics,
    context: dict[str, Any],
    tag: str,
    instance: str,
    started: float,
    result: dict[str, Any] | None,
    error: ServiceError | None,
    refresh: bool,
) -> None:
    """Best-effort round report. Must never alter the query response."""
    elapsed_ms = max(0, round((perf_counter() - started) * 1000))
    # Errors raised from the leader path did reach the origin; waiter-side
    # coordination errors did not.  The cache layer marks the former via
    # ``extra["origin_called"]``.
    origin_called = bool((error.extra or {}).get("origin_called")) if error else False
    if error is not None:
        report = {
            "tag": tag,
            "instance": instance,
            "role": "leader" if origin_called else None,
            "result": "error",
            "status": "error",
            "error_code": error.code,
            "error_message": error.message,
            "http_status": error.status_code,
            "retryable": bool(error.retryable),
            "origin_called": origin_called,
            "cache_hit": False,
            "digest": None,
            "observed_generation": None,
            "refresh": refresh,
            "duration_ms": elapsed_ms,
            "request_id": None,
            "reported_at_ms": int(time.time() * 1000),
        }
    else:
        assert result is not None
        publication = result.get("publication") or {}
        report = {
            "tag": tag,
            "instance": instance,
            "role": result.get("single_flight_role"),
            "result": "ok",
            "status": "ok",
            "error_code": None,
            "error_message": None,
            "http_status": 200,
            "retryable": False,
            "origin_called": bool(result.get("origin_called")),
            "cache_hit": bool(result.get("cache_hit")),
            "digest": result.get("digest"),
            "observed_generation": result.get("generation"),
            "publication_generation": result.get("generation"),
            "publication_status": publication.get("status"),
            "fenced_by_publish": bool(result.get("fenced_by_publish")),
            "stale_due_to_pending": bool(result.get("stale_due_to_pending")),
            "coalesced": bool(result.get("coalesced")),
            "result_kind": result.get("result_kind"),
            "refresh": refresh,
            "duration_ms": elapsed_ms,
            "request_id": result.get("gateway_request_id"),
            "reported_at_ms": int(time.time() * 1000),
        }
    try:
        await diagnostics.report(
            context["round_id"], context["slot"], context["token"], report
        )
    except Exception as exc:  # noqa: BLE001 - diagnostics never break queries
        logger.warning("diagnostics report swallowed: %r", exc)


def create_app(cfg: Settings | None = None) -> FastAPI:
    cfg = cfg or settings
    app = FastAPI(title="制品标签查询", version="1.0.0", lifespan=lifespan)
    app.state.settings = cfg
    # The page served by one instance drives the other instance directly, so
    # the public instance origins must be allowed to call this API.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(
            dict.fromkeys(
                list(cfg.public_instances)
                + ["http://localhost:8001", "http://localhost:8002"]
            )
        ),
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "X-Request-Id"],
        allow_credentials=False,
    )
    _install_error_handler(app)
    _register_routes(app)
    return app


def _register_routes(app: FastAPI) -> None:
    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(os.path.join(STATIC_DIR, "index.html"))

    @app.get("/api/health")
    async def health() -> dict[str, Any]:
        service: TagService = app.state.service
        redis_ok = True
        redis_error = None
        try:
            await service.redis.ping()
        except Exception as exc:  # noqa: BLE001
            redis_ok = False
            redis_error = exc.__class__.__name__
        return {
            "status": "ok" if redis_ok else "degraded",
            "instance": app.state.settings.instance_name,
            "redis": {"ok": redis_ok, "error": redis_error},
            "public_instances": list(app.state.settings.public_instances),
        }

    @app.get("/api/config")
    async def config() -> dict[str, Any]:
        cfg: Settings = app.state.settings
        return {
            "instance": cfg.instance_name,
            "public_instances": list(cfg.public_instances),
            "origin_url": cfg.origin_url,
            "origin_timeout_ms": cfg.origin_timeout_ms,
            "cache_wait_timeout_ms": cfg.cache_wait_timeout_ms,
            "revalidate_ttl_s": cfg.revalidate_ttl_s,
            "diagnostics": {
                "max_slots": MAX_ROUND_SLOTS,
                "retention_ttl_s": cfg.diagnostics_ttl_s,
                "default_deadline_ms": cfg.diagnostics_default_deadline_ms,
            },
        }

    @app.get("/api/tag/{tag:path}")
    async def get_tag(
        tag: str,
        refresh: bool = Query(default=False),
        x_request_id: str | None = Header(default=None, alias="X-Request-Id"),
        x_diag_round: str | None = Header(default=None, alias="X-Diag-Round"),
        x_diag_slot: str | None = Header(default=None, alias="X-Diag-Slot"),
        x_diag_token: str | None = Header(default=None, alias="X-Diag-Token"),
    ) -> dict[str, Any]:
        service: TagService = app.state.service
        diagnostics: Diagnostics = app.state.diagnostics
        diag_context = _parse_diag_headers(x_diag_round, x_diag_slot, x_diag_token)
        started = perf_counter()
        result: dict[str, Any] | None = None
        error: ServiceError | None = None
        refresh_flag = refresh
        try:
            result = await service.get_tag(tag, refresh=refresh)
            if x_request_id:
                result["gateway_request_id"] = x_request_id
            return result
        except ServiceError as exc:
            error = exc
            raise
        finally:
            if diag_context is not None:
                await _report_diagnostics(
                    diagnostics,
                    diag_context,
                    tag,
                    app.state.settings.instance_name,
                    started,
                    result,
                    error,
                    refresh_flag,
                )

    @app.get("/api/status/{tag:path}")
    async def get_status(tag: str) -> dict[str, Any]:
        service: TagService = app.state.service
        return await service.status(tag)

    @app.post("/api/publish")
    async def publish(
        body: PublishRequest,
        x_request_id: str | None = Header(default=None, alias="X-Request-Id"),
    ) -> dict[str, Any]:
        service: TagService = app.state.service
        return await service.publish(
            body.tag,
            body.digest,
            body.request_id or x_request_id,
        )

    @app.post("/api/cache/invalidate")
    async def invalidate(body: TagBody) -> dict[str, Any]:
        service: TagService = app.state.service
        return await service.invalidate(body.tag)

    @app.get("/api/stats")
    async def stats() -> dict[str, Any]:
        service: TagService = app.state.service
        gateway = await service.gateway_stats()
        origin = await service.origin_stats()
        counts = origin.get("request_count")
        return {
            "instance": app.state.settings.instance_name,
            "gateway": gateway,
            "origin": origin,
            "origin_request_count": counts,
        }

    @app.post("/api/origin/control")
    async def origin_control(body: OriginControlRequest) -> dict[str, Any]:
        service: TagService = app.state.service
        payload = {k: v for k, v in body.model_dump().items() if v is not None}
        if body.reset_counters:
            payload["reset_counters"] = True
        return await service.origin_control(payload)

    @app.get("/api/origin/tags")
    async def origin_tags() -> dict[str, Any]:
        service: TagService = app.state.service
        return await service.origin_stats()

    # -------------------------------------------------- query-round diagnostics
    @app.post("/api/diagnostics/rounds")
    async def create_diagnostics_round(body: RoundCreateRequest) -> dict[str, Any]:
        diagnostics: Diagnostics = app.state.diagnostics
        TagService.validate_tag(body.tag)
        entries = [entry.model_dump() for entry in body.entries]
        return await diagnostics.create_round(
            body.tag, entries, deadline_ms=body.deadline_ms
        )

    @app.get("/api/diagnostics/rounds")
    async def list_diagnostics_rounds(
        tag: str | None = Query(default=None),
        limit: int = Query(default=20, ge=1, le=100),
    ) -> dict[str, Any]:
        diagnostics: Diagnostics = app.state.diagnostics
        if tag is not None:
            TagService.validate_tag(tag)
        return await diagnostics.list_rounds(tag=tag, limit=limit)

    @app.get("/api/diagnostics/rounds/{round_id}")
    async def get_diagnostics_round(round_id: str) -> dict[str, Any]:
        diagnostics: Diagnostics = app.state.diagnostics
        return await diagnostics.get_round(round_id)


app = create_app()
