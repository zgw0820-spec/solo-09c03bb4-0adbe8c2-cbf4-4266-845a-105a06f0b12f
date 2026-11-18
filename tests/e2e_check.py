"""End-to-end interleaving checks against real Redis + two uvicorn instances.

Run with the stack up:
    python3 tests/e2e_check.py --q1 http://127.0.0.1:8101 --q2 http://127.0.0.1:8102
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import uuid

import httpx

FAILS: list[str] = []


def check(cond: bool, message: str) -> None:
    print(("PASS  " if cond else "FAIL  ") + message)
    if not cond:
        FAILS.append(message)


async def _server_now(client: httpx.AsyncClient) -> int:
    """Wall-clock milliseconds; all processes run on one host in the demo."""
    import time

    return int(time.time() * 1000)


async def reset(origin: httpx.AsyncClient, tag: str, digest: str) -> None:
    await origin.post(
        "/control",
        json={
            "delay_ms": 0,
            "fail_next": 0,
            # null clears the one-shot override; 0 would mean "next request
            # with zero delay", which is not a reset.
            "slow_next_ms": None,
            "publish_delay_ms": 0,
            "fail_publish_next": 0,
        },
    )
    await origin.post("/publish", json={"tag": tag, "digest": digest})
    await origin.post("/control", json={"reset_counters": True})


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--q1", default="http://127.0.0.1:8101")
    parser.add_argument("--q2", default="http://127.0.0.1:8102")
    parser.add_argument("--origin", default="http://127.0.0.1:9100")
    # Kept for parity with the Docker instructions; unused by the checks.
    parser.add_argument("--redis-cli", default="/tmp/e2e/redis-cli")
    parser.add_argument("--redis-port", default="6390")
    args = parser.parse_args()

    # Unique tags per run: Redis state from a previous run must not be able to
    # masquerade as a fresh cache entry for this run.
    run = uuid.uuid4().hex[:8]

    def tag(name: str) -> str:
        return f"{name}-{run}"

    def request_id(name: str) -> str:
        return f"{name}-{run}"

    q1 = httpx.AsyncClient(base_url=args.q1, timeout=30)
    q2 = httpx.AsyncClient(base_url=args.q2, timeout=30)
    origin = httpx.AsyncClient(base_url=args.origin, timeout=30)

    # ---------------------------------------------------------- 1. single flight
    await reset(origin, tag("e2e-single"), "sha256:a1")
    # Invalidate so the round starts from a guaranteed miss without calling the
    # origin ourselves (a forced refresh would legitimately hit the origin).
    invalidated = await q1.post("/api/cache/invalidate", json={"tag": tag("e2e-single")})
    check(invalidated.status_code == 200 and invalidated.json()["invalidated"] is False,
          "失效接口对冷标签不报错")
    await origin.post("/control", json={"delay_ms": 700, "reset_counters": True})
    responses = await asyncio.gather(
        *[q1.get(f"/api/tag/{tag('e2e-single')}") for _ in range(8)],
        *[q2.get(f"/api/tag/{tag('e2e-single')}") for _ in range(8)],
    )
    bodies = [r.json() for r in responses]
    digits = {b.get("digest") for b in bodies}
    stats = (await origin.get("/control")).json()
    leaders = sum(1 for b in bodies if b.get("single_flight_role") == "leader")
    followers = sum(1 for b in bodies if b.get("single_flight_role") == "follower")
    check(all(r.status_code == 200 for r in responses), "16 个并发查询全部成功")
    check(digits == {"sha256:a1"}, f"并发结果一致: {digits}")
    check(stats["request_count"] == 1, f"源站本轮只被调用 1 次（实际 {stats['request_count']}）")
    check(leaders == 1, f"仅 1 个单飞领导者（实际 {leaders}）")
    check(followers == 15, f"其余 15 个为等待者（实际 {followers}）")

    # ------------------------------------------------------------ 2. cache hit
    before = stats["request_count"]
    hits = await asyncio.gather(q1.get(f"/api/tag/{tag('e2e-single')}"), q2.get(f"/api/tag/{tag('e2e-single')}"))
    check(all(r.json().get("cache_hit") for r in hits), "后续查询命中缓存")
    after = (await origin.get("/control")).json()["request_count"]
    check(after == before, "缓存命中不增加源站调用")

    # ------------------------------------- 3. old origin reply cannot overwrite publish
    await reset(origin, tag("e2e-fence"), "sha256:old")
    await q1.get(f"/api/tag/{tag('e2e-fence')}")
    invalidated = await q2.post("/api/cache/invalidate", json={"tag": tag("e2e-fence")})
    check(invalidated.json().get("invalidated") is True, "失效接口确认已有缓存被标记失效")
    await origin.post("/control", json={"delay_ms": 900})
    stale_fetch = asyncio.create_task(q1.get(f"/api/tag/{tag('e2e-fence')}"))
    await asyncio.sleep(0.25)
    published = await q2.post("/api/publish", json={"tag": tag("e2e-fence"), "digest": "sha256:new", "request_id": request_id("e2e-pub-1")})
    pub_body = published.json()
    check(published.status_code == 200 and pub_body.get("status") == "confirmed", f"发布已确认: {pub_body.get('status')}")
    # The stale fetch must still be in flight when confirmation lands, and the
    # confirmation must be visible before we assert on its result.
    check(not stale_fetch.done(), "断言时旧回源仍在途中")
    confirmed = (await q2.get(f"/api/status/{tag('e2e-fence')}")).json()
    check(confirmed["publication"]["status"] == "confirmed", "发布确认已写入共享状态")
    stale_result = (await stale_fetch).json()
    check(stale_result.get("digest") == "sha256:new", f"晚到的旧回源未覆盖新发布，返回 {stale_result}")
    check(stale_result.get("fenced_by_publish") is True, "旧回源被发布代号栅栏拦截")
    post_pub = await asyncio.gather(q1.get(f"/api/tag/{tag('e2e-fence')}"), q2.get(f"/api/tag/{tag('e2e-fence')}"))
    check(all(r.json().get("digest") == "sha256:new" for r in post_pub), "确认后两实例均返回最新摘要")
    check(all(r.json().get("cache_hit") for r in post_pub), "确认后的查询直接命中缓存")

    # ----------------------------------------------- 4. retry publish no rollback
    await reset(origin, tag("e2e-retry"), "sha256:base")
    one = (await q1.post("/api/publish", json={"tag": tag("e2e-retry"), "digest": "sha256:one", "request_id": request_id("e2e-r1")})).json()
    two = (await q2.post("/api/publish", json={"tag": tag("e2e-retry"), "digest": "sha256:two", "request_id": request_id("e2e-r2")})).json()
    retry = (await q1.post("/api/publish", json={"tag": tag("e2e-retry"), "digest": "sha256:one", "request_id": request_id("e2e-r1")})).json()
    current = (await q2.get(f"/api/tag/{tag('e2e-retry')}")).json()
    check(two["generation"] > one["generation"], "两次发布会话代号递增")
    check(retry.get("status") == "superseded" and retry.get("applied") is False, f"重试旧发布返回 superseded（实际 {retry.get('status')}）")
    check(current.get("digest") == "sha256:two", f"重试未回退较新摘要（实际 {current.get('digest')}）")

    # ------------------------------------------ 5. publish unconfirmed is explicit
    await reset(origin, tag("e2e-unconf"), "sha256:keep")
    await q1.get(f"/api/tag/{tag('e2e-unconf')}")
    await origin.post("/control", json={"fail_publish_next": 1})
    failed = await q1.post("/api/publish", json={"tag": tag("e2e-unconf"), "digest": "sha256:lost", "request_id": request_id("e2e-u1")})
    check(failed.status_code >= 400, f"发布失败返回错误（HTTP {failed.status_code}）")
    body = failed.json()
    check(body.get("error", {}).get("retryable") is True, "发布失败标记为可重试")
    during = (await q2.get(f"/api/tag/{tag('e2e-unconf')}")).json()
    check(during.get("digest") == "sha256:keep", "未确认发布不改变已确认摘要")
    status = (await q2.get(f"/api/status/{tag('e2e-unconf')}")).json()
    check(status["publication"]["status"] in {"confirmed", "cached"}, f"发布状态未被标成已确认: {status['publication']['status']}")

    # --------------------------------------------------- 6. origin timeout + retry
    await reset(origin, tag("e2e-timeout"), "sha256:slow")
    await origin.post("/control", json={"delay_ms": 2500})
    timed_out = await q1.get(f"/api/tag/{tag('e2e-timeout')}")
    check(timed_out.status_code == 504, f"源站超时返回 504（实际 {timed_out.status_code}）")
    if timed_out.status_code == 504:
        err = timed_out.json()["error"]
        check(err["code"] == "origin_timeout" and err["retryable"] is True, "超时错误码明确且可重试")
    await origin.post("/control", json={"delay_ms": 0})
    recovered = await q1.get(f"/api/tag/{tag('e2e-timeout')}")
    check(recovered.status_code == 200 and recovered.json()["digest"] == "sha256:slow", "超时后可再次查询成功")

    # ------------------------------------ 7. publish while queries in flight
    await reset(origin, tag("e2e-pending"), "sha256:v1")
    await q1.get(f"/api/tag/{tag('e2e-pending')}")
    await origin.post("/control", json={"publish_delay_ms": 700})
    slow_pub = asyncio.create_task(
        q1.post("/api/publish", json={"tag": tag("e2e-pending"), "digest": "sha256:v2", "request_id": request_id("e2e-p2")})
    )
    await asyncio.sleep(0.1)
    during_pub = (await q2.get(f"/api/tag/{tag('e2e-pending')}")).json()
    check(during_pub.get("digest") == "sha256:v1", "发布未确认时返回最后已确认摘要")
    check(during_pub.get("stale_due_to_pending") is True, "明确标记该摘要不是发布目标")
    pub_done = (await slow_pub).json()
    check(pub_done.get("status") == "confirmed", "发布随后确认")
    final = (await q2.get(f"/api/tag/{tag('e2e-pending')}")).json()
    check(final.get("digest") == "sha256:v2", "确认后返回新摘要")

    # ------------------------------------------------ 8. slash-containing tags
    slash_tag = f"ns/{tag('e2e-slash')}/sub"
    await origin.post("/publish", json={"tag": slash_tag, "digest": "sha256:slash"})
    slash = await q1.get(f"/api/tag/{slash_tag}")
    check(slash.status_code == 200 and slash.json().get("digest") == "sha256:slash",
          f"含 / 的标签可查询（HTTP {slash.status_code}）")
    slash_status = await q2.get(f"/api/status/{slash_tag}")
    check(slash_status.status_code == 200, "含 / 的标签可查询状态")

    # ------------------------------------------------------- 9. invalid inputs
    bad_tag = await q1.get("/api/tag/bad%20tag")
    check(bad_tag.status_code == 422, f"非法标签返回 422（实际 {bad_tag.status_code}）")
    bad_digest = await q1.post("/api/publish", json={"tag": "ok-tag", "digest": "bad digest"})
    check(bad_digest.status_code == 422, f"非法摘要返回 422（实际 {bad_digest.status_code}）")

    # --------------------------------------------- 10. query-round diagnostics
    diag_tag = tag("e2e-diag")
    await reset(origin, diag_tag, "sha256:diag1")
    await q1.post("/api/cache/invalidate", json={"tag": diag_tag})
    await origin.post("/control", json={"delay_ms": 600, "reset_counters": True})
    created = (await q1.post(
        "/api/diagnostics/rounds",
        json={"tag": diag_tag, "entries": [
            {"instance": "query-1"} for _ in range(8)
        ] + [
            {"instance": "query-2"} for _ in range(8)
        ]},
    )).json()
    check(created.get("planned_count") == 16, f"轮次固定 16 个请求编号（实际 {created.get('planned_count')}）")
    check(len(created.get("credentials", [])) == 16, "创建时一次性发放 16 个槽位令牌")
    check(created.get("status") == "open" and created.get("gap_count") == 0, "新轮次为 open 且无缺口")
    round_id = created["round_id"]
    creds = {c["slot"]: c["token"] for c in created["credentials"]}
    jobs = []
    for slot in range(16):
        base = q1 if slot < 8 else q2
        jobs.append(base.get(
            f"/api/tag/{diag_tag}",
            headers={
                "X-Diag-Round": round_id,
                "X-Diag-Slot": str(slot),
                "X-Diag-Token": creds[slot],
            },
        ))
    responses = await asyncio.gather(*jobs)
    check(all(r.status_code == 200 for r in responses), "诊断轮次 16 个查询全部成功")
    detail = (await q2.get(f"/api/diagnostics/rounds/{round_id}")).json()
    check(detail["status"] == "sealed" and detail["seal_reason"] == "all_done",
          f"全部上报后按 all_done 封存（实际 {detail['status']}/{detail['seal_reason']}）")
    check(detail["complete"] is True and detail["gap_count"] == 0, "完整轮次无缺口")
    check(len(detail["slots"]) == 16, "封存结论保留全部 16 个槽位")
    roles = [s.get("role") for s in detail["slots"]]
    check(roles.count("leader") == 1 and roles.count("follower") == 15,
          f"轮次记录 1 个回源 leader、15 个等待者（实际 {roles}）")
    check(sum(1 for s in detail["slots"] if s.get("origin_called")) == 1,
          "轮次中实际回源请求数为 1")
    check({s.get("instance") for s in detail["slots"]} == {"query-1", "query-2"},
          "两个实例的请求分别带实例名")
    check(detail["conclusion"]["digest_consistent"] is True, "封存结论显示摘要一致")
    check((await origin.get("/control")).json()["request_count"] == 1,
          "诊断轮次源站仍只被调用 1 次")
    # A wrong token cannot write into a sealed round.
    before_forge = (await q2.get(f"/api/diagnostics/rounds/{round_id}")).json()
    bogus = await q1.get(
        f"/api/tag/{diag_tag}",
        headers={"X-Diag-Round": round_id, "X-Diag-Slot": "0", "X-Diag-Token": "forged"},
    )
    check(bogus.status_code == 200, "伪造令牌不影响原查询响应")
    detail_after = (await q1.get(f"/api/diagnostics/rounds/{round_id}")).json()
    check(
        detail_after["slots"][0] == before_forge["slots"][0],
        "伪造令牌无法改写封存槽位",
    )
    check(detail_after["late_reports"] == [], "伪造令牌不会产生迟到记录")

    # 10b. deadline seals gaps; late reports stay viewable but immutable.
    gap_tag = tag("e2e-diag-gap")
    await reset(origin, gap_tag, "sha256:gap1")
    gap_round = (await q1.post(
        "/api/diagnostics/rounds",
        json={"tag": gap_tag, "entries": [
            {"instance": "query-1"},
            {"instance": "query-2"},
        ], "deadline_ms": (await _server_now(q1)) + 1200},
    )).json()
    gid = gap_round["round_id"]
    gcreds = {c["slot"]: c["token"] for c in gap_round["credentials"]}
    first = await q1.get(
        f"/api/tag/{gap_tag}",
        headers={"X-Diag-Round": gid, "X-Diag-Slot": "0", "X-Diag-Token": gcreds[0]},
    )
    check(first.status_code == 200, "缺口轮次首个请求正常")
    await asyncio.sleep(1.4)
    sealed_gap = (await q2.get(f"/api/diagnostics/rounds/{gid}")).json()
    check(sealed_gap["status"] == "sealed" and sealed_gap["seal_reason"] == "deadline",
          "截止到达后按 deadline 封存")
    check(sealed_gap["gap_count"] == 1 and sealed_gap["gaps"][0]["slot"] == 1,
          "未结束的请求标明缺口")
    late = await q2.get(
        f"/api/tag/{gap_tag}",
        headers={"X-Diag-Round": gid, "X-Diag-Slot": "1", "X-Diag-Token": gcreds[1]},
    )
    check(late.status_code == 200, "迟到请求的原查询响应不受影响")
    final_gap = (await q1.get(f"/api/diagnostics/rounds/{gid}")).json()
    check(len(final_gap["late_reports"]) == 1 and final_gap["late_reports"][0]["slot"] == 1,
          "迟到结果可查看")
    check(final_gap["gap_count"] == 1 and final_gap["slots"][1]["status"] == "gap",
          "迟到结果不能改写封存结论/填补缺口")
    check(final_gap["conclusion"]["recorded"] == 1, "封存结论只计封存时已到记录")

    # 10c. retry on the same tag is a separate round; old errors stay behind.
    retry_round = (await q1.post(
        "/api/diagnostics/rounds",
        json={"tag": gap_tag, "entries": [{"instance": "query-1"}]},
    )).json()
    check(retry_round["round_id"] != gid, "同标签重试另建轮次")
    listing = (await q2.get("/api/diagnostics/rounds", params={"tag": gap_tag})).json()
    ids = [r["round_id"] for r in listing["rounds"]]
    check(ids[0] == retry_round["round_id"] and gid in ids, "按轮次列出，新旧轮次互不覆盖")

    # 10d. round size cap and unknown round handling.
    too_big = await q1.post(
        "/api/diagnostics/rounds",
        json={"tag": gap_tag, "entries": [{"instance": "query-1"} for _ in range(17)]},
    )
    check(too_big.status_code == 422, f"每轮超过 16 请求被拒绝（HTTP {too_big.status_code}）")
    missing = await q1.get("/api/diagnostics/rounds/r00000000000000000000000000000000")
    check(missing.status_code == 404, f"不存在轮次返回 404（HTTP {missing.status_code}）")

    await q1.aclose()
    await q2.aclose()
    await origin.aclose()

    print()
    if FAILS:
        print(f"{len(FAILS)} check(s) failed")
        return 1
    print("all end-to-end checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
