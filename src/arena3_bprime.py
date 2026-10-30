"""ARENA3 B′ — M4를 기본으로, 선택 지점마다 SUMO 롤아웃으로 고르는 탐색 제어(강화학습 아님, 208절).

규칙(잠금 설정 bprime): 여섯 모드의 첫 녹색이 갈리는 순간 상태를 저장하고, 서로 다른 첫 녹색(후보)마다 표본 미래 8개로
300초를 굴린다(첫 5초 후보, 이후 M4). 비용 = 300초 지연 + 남은 정체 × 75초. 후보끼리 같은 미래 번호로 짝을 지어
M4가 아닌 후보 a의 이득 g = 평균(M4 비용 − a 비용)이 0보다 크고 g ≥ 1.0 × 표준오차이면 a(여럿이면 g 최대), 아니면 M4.
표본 미래는 실제 미래 차를 지우고 최근 900초 차량에서 다시 뽑는다(explore_rollout, 진입 대기 보완 explore_m4probe_pending).
본 실행의 성적은 끊기지 않은 원래 시뮬레이션에서 잰다. 롤아웃 복원이 조금 어긋나도 판정 지표는 오염되지 않고 B′의 판단만 흐려진다.
그래서 선택 지점마다 '실제 미래 5초 재생'이 원래 실행과 같은지(허용 1e-6) 확인해 비율을 기록하고, 5%를 넘으면 멈춘다.

명령(Colab):
  python src/arena3_bprime.py smoke --out <Drive>/arena3/bprime_data --work /content/b3_smoke --rollout-workers 8
  python src/arena3_bprime.py run --role data --out <Drive>/arena3/bprime_data --work /content/b3 --shard 0 --shards 2 --rollout-workers 4
  python src/arena3_bprime.py run --role eval --out <Drive>/arena3/referee/bprime --work /content/b3e --shard 0 --shards 2 --rollout-workers 4
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import multiprocessing
import os
import pickle
import shutil
import statistics
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import arena3_common as ac  # noqa: E402
import arena3_referee as ref  # noqa: E402
import explore_weight as w  # noqa: E402

POLICY = "BPRIME"
LEFTS = {"left_east": ("477706392_02", "477706883_02"), "left_south": ("477706883_01", "477707013_01")}


def bcfg() -> dict:
    return ac.config()["bprime"]


def future_seed(state_id: str, index: int) -> int:
    return ac.seed32("BPRIME_FUTURE", state_id, index)


def keep_state(state_id: str) -> bool:
    """상태 파일 보관(약 10%): 결과와 무관한 해시 규칙."""
    return int(hashlib.sha256(f"ARENA3_KEEP|{state_id}".encode()).hexdigest()[:8], 16) % 1000 < bcfg()["keep_state_fraction"] * 1000


# ----------------------------------------------------------------------------- 결정 규칙(순수 계산)
def choose(m4: int, costs: dict[int, list[float | None]], se_multiplier: float = 1.0) -> dict:
    """costs[a][j] = 미래 j에서 후보 a의 300초 비용(그 미래가 품질 문제로 빠지면 None, 모든 후보에서 같이 뺀다)."""
    if m4 not in costs:
        raise ValueError("M4 후보의 롤아웃이 없습니다.")
    n = len(costs[m4])
    if any(len(v) != n for v in costs.values()):
        raise ValueError("후보마다 미래 수가 다릅니다.")
    valid = [j for j in range(n) if all(costs[a][j] is not None for a in costs)]
    gains = {}
    for a in sorted(costs):
        if a == m4:
            continue
        diff = [costs[m4][j] - costs[a][j] for j in valid]
        if len(diff) < 4:
            gains[a] = {"mean": None, "se": None, "futures": len(diff)}
            continue
        mean = statistics.fmean(diff)
        se = statistics.stdev(diff) / math.sqrt(len(diff))
        gains[a] = {"mean": mean, "se": se, "futures": len(diff)}
    ok = [a for a, g in gains.items() if g["mean"] is not None and g["mean"] > 0 and g["mean"] >= se_multiplier * g["se"]]
    chosen = max(ok, key=lambda a: (gains[a]["mean"], -a)) if ok else m4
    return {"chosen": int(chosen), "m4": int(m4), "gains": {str(a): g for a, g in gains.items()}, "valid_futures": len(valid)}


# ----------------------------------------------------------------------------- 롤아웃(작업자 프로세스)
def _meter_class():
    import explore_rollout as er

    class MovementMeter(er.DelayMeter):
        """DelayMeter와 같은 정확 셈을 두 좌회전 이동에 대해서도 따로 낸다(Gemini 학습기의 좌회전 값)."""

        def __init__(self, env, t0, vehicles, injected=(), marks=(300,)):
            super().__init__(env, t0, vehicles, injected, marks)
            self.move = {v: er.movement_of_edges(edges) for v, (_, edges, _) in vehicles.items()}
            self.move.update({vid: er.movement_of_edges(list(edges)) for vid, edges, _, _ in injected})

        def finish(self, tripinfo: Path) -> dict:
            arrived = {}
            if Path(tripinfo).exists():
                import xml.etree.ElementTree as ET
                arrived = {t.get("id"): (float(t.get("arrival")), float(t.get("timeLoss")))
                           for t in ET.parse(tripinfo).getroot().iter("tripinfo")}
            at = super().finish(tripinfo)
            for h, m in sorted(self._marked.items()):
                M = self.t0 + h
                done = {v for v, (arr, _) in arrived.items() if 0 <= arr <= M}
                for name, movement in LEFTS.items():
                    own = lambda v: self.move.get(v) == movement   # noqa: E731
                    loss = sum(tl - self.tl0.get(v, 0.0) for v, tl in m["tl"].items() if own(v) and v not in done)
                    loss += sum(tl - self.tl0.get(v, 0.0) for v, (arr, tl) in arrived.items() if own(v) and 0 <= arr <= M)
                    wait = sum(dep - max(self.desired.get(v, self.t0), self.t0) for v, dep in m["departed"].items() if own(v))
                    wait += sum(M - max(self.desired.get(v, self.t0), self.t0) for v in m["pending"] if own(v))
                    at[h][name] = round(loss + wait, 3)
            return at

    return MovementMeter


def trace_table(sumo) -> dict:
    from focused_data import TARGET_TLS
    return {"time": float(sumo.simulation.getTime()), "tls": sumo.trafficlight.getRedYellowGreenState(TARGET_TLS),
            "vehicles": [(v, sumo.vehicle.getLaneID(v), float(sumo.vehicle.getLanePosition(v)), float(sumo.vehicle.getSpeed(v)))
                         for v in sorted(sumo.vehicle.getIDList())],
            "pending": sorted(sumo.simulation.getPendingVehicles())}


def same_table(a: dict, b: dict, tol: float) -> bool:
    if a["time"] != b["time"] or a["tls"] != b["tls"] or a["pending"] != b["pending"] or len(a["vehicles"]) != len(b["vehicles"]):
        return False
    return all(x[0] == y[0] and x[1] == y[1] and abs(x[2] - y[2]) <= tol and abs(x[3] - y[3]) <= tol
               for x, y in zip(a["vehicles"], b["vehicles"]))


def _restore_env(job, future: str, seed, horizon: int, trip: Path | None, collision_file: Path | None):
    import explore_m4probe_pending as pend
    import explore_rollout as er
    er.SUMO_SEED = int(job["sumo_seed"])
    original_env = er._env
    if collision_file is not None:
        er._env = lambda route, extra, duration: original_env(route, f"{extra} --collision-output {collision_file}", duration)
    try:
        return pend.restore(Path(job["stem"]), Path(job["route_dir"]), future, seed, horizon, tripinfo=trip)
    finally:
        er._env = original_env


def rollout_worker(job: dict) -> dict:
    """저장 상태에서 표본 미래 하나로 300초: 첫 5초 후보 녹색, 이후 M4. 결과는 메모리로 돌려준다."""
    import explore_rollout as er
    from explore_weight_core import ModePlanner
    from focused_eval import judge_collisions
    started = time.monotonic()
    env = None
    fd, trip_name = tempfile.mkstemp(prefix="a3_trip_", suffix=".xml")
    os.close(fd)
    fd, coll_name = tempfile.mkstemp(prefix="a3_coll_", suffix=".xml")
    os.close(fd)
    trip, coll = Path(trip_name), Path(coll_name)
    try:
        horizon = int(job["horizon_s"])
        env, cons, ts, ctl, snap, injected, vehicles = _restore_env(job, "sampled", int(job["future_seed"]), horizon, trip, coll)
        planner = ac._planner_class()(ctl.movement_of, [p.state for p in ts.green_phases], snap["handover"] + 3600)
        check = planner.arena_snapshot(env)
        if list(check.greens) != job["greens"] or list(check.allowed) != job["allowed"]:
            raise ValueError("표본 미래를 넣은 뒤 여섯 모드의 첫 녹색이 저장 순간과 다릅니다.")
        if np.max(np.abs(check.observation - np.asarray(job["observation"], np.float32))) > 1e-6:
            raise ValueError("표본 미래를 넣은 뒤 관측 119칸이 저장 순간과 다릅니다(미래가 관측에 샜거나 복원 오차).")
        if job["candidate"] not in check.candidates:
            raise ValueError("후보가 여섯 모드의 첫 녹색이 아닙니다.")
        meter = _meter_class()(env, snap["t0"], vehicles, injected, marks=(horizon,))

        def m4(e, allowed):
            a = int(ModePlanner.snapshot(planner, e).mode_actions[ac.M4])
            if a not in allowed:
                raise ValueError("M4 녹색이 실행 가능 목록 밖입니다.")
            return a
        _, changed = er.drive(env, cons, m4, int(job["candidate"]), snap["t0"], horizon, meter.step)
        env.close()
        env = None
        meter.finish(trip)
        if horizon not in meter.at:
            raise ValueError("300초 비용이 없습니다.")
        collision = judge_collisions(coll)
        r = dict(meter.at[horizon])
        r.update({"candidate": int(job["candidate"]), "future": int(job["index"]), "future_seed": int(job["future_seed"]),
                  "leaks": len(meter.leaks), "changed_by_rules": int(changed), "injected": len(injected),
                  "collisions_unexplained": len(collision["collisions_unexplained"]),
                  "collisions_at_204820": int(collision["collisions_at_204820"]), "seconds": time.monotonic() - started})
        return {"ok": True, **r}
    except BaseException as error:
        return {"ok": False, "candidate": job.get("candidate"), "future": job.get("index"),
                "error": f"{type(error).__name__}: {error}", "seconds": time.monotonic() - started}
    finally:
        if env is not None:
            env.close()
        trip.unlink(missing_ok=True)
        coll.unlink(missing_ok=True)


def verify_worker(job: dict) -> dict:
    """실제 미래 그대로 복원해 실행한 녹색으로 5초 → 원래 실행과 비교할 표(복원 정확도 기록용)."""
    started = time.monotonic()
    env = None
    try:
        env, cons, ts, ctl, snap, injected, vehicles = _restore_env(job, "real", None, 10, None, None)
        a = int(job["green"])
        corrected = int(cons(env, a))
        env.step(corrected)
        return {"ok": True, "table": trace_table(env.sumo), "corrected": corrected, "seconds": time.monotonic() - started}
    except BaseException as error:
        return {"ok": False, "error": f"{type(error).__name__}: {error}", "seconds": time.monotonic() - started}
    finally:
        if env is not None:
            env.close()


# ----------------------------------------------------------------------------- B′ 정책
class BPrimePolicy:
    name = POLICY

    def __init__(self, pool, situation: dict, state_dir: Path, keep_dir: Path | None, *, always_m4: bool = False,
                 max_rollout_points: int | None = None):
        cfg = bcfg()
        self.pool, self.situation, self.state_dir, self.keep_dir = pool, situation, Path(state_dir), keep_dir
        self.futures, self.horizon, self.se_mult = int(cfg["futures"]), int(cfg["horizon_s"]), float(cfg["se_multiplier"])
        self.tol = float(cfg["restore_check"]["tolerance"])
        self.always_m4, self.max_points = always_m4, max_rollout_points
        self.points: list[dict] = []
        self.verifies: list = []               # (점 번호, future)
        self.rollout_seconds = 0.0
        self.sid = ac.situation_id(situation) if situation["tag"] else f"{situation['date']}_{situation['period']}_s{situation['sumo_seed']}"

    def decide(self, snap, ctx) -> int:
        import explore_m4probe_pending as pend
        import explore_rollout as er
        ep = ctx.ep
        k = len(self.points)
        state_id = f"{self.sid}_c{k:03d}"
        m4 = int(snap.greens[ac.M4])
        point = {"state": state_id, "step": int(ctx.step), "now": float(snap.now), "allowed": list(snap.allowed),
                 "greens": list(snap.greens), "mask": list(snap.mask), "p0": int(snap.p0_action), "m4": m4,
                 "candidates": list(snap.candidates), "observation": snap.observation.tolist(),
                 "m4_costs": snap.m4_costs, "w4_cost": snap.w4_cost}
        if self.max_points is not None and k >= self.max_points:
            point.update({"chosen": m4, "rollouts": None, "decision": None})
            self.points.append(point)
            return m4
        stem = self.state_dir / state_id
        saved = er.Episode.snapshot(ep, stem, snap.allowed, snap.p0_action)
        saved["pending_snapshot"] = pend.capture_pending(ep.env.sumo)
        stem.with_suffix(".pkl").write_bytes(pickle.dumps(saved))
        er.types_only_file(ep.route)          # 작업자들이 동시에 만들지 않게 미리 만든다
        base = {"stem": str(stem), "route_dir": str(ep.route.parent), "sumo_seed": int(self.situation["sumo_seed"]),
                "greens": list(snap.greens), "allowed": list(snap.allowed), "horizon_s": self.horizon,
                "observation": snap.observation.tolist()}
        jobs = [{**base, "candidate": int(a), "index": j, "future_seed": future_seed(state_id, j)}
                for a in snap.candidates for j in range(self.futures)]
        started = time.monotonic()
        results = list(self.pool.map(rollout_worker, jobs))
        self.rollout_seconds += sum(float(r["seconds"]) for r in results)
        bad = [r for r in results if not r["ok"]]
        if bad:
            raise RuntimeError(f"롤아웃 실패 {state_id}: {bad[0]['error']} (총 {len(bad)}개)")
        if any(r["leaks"] or r["changed_by_rules"] for r in results):
            raise RuntimeError(f"롤아웃에서 미래 엿봄 또는 규칙 변경이 생겼습니다: {state_id}")
        costs = {int(a): [None] * self.futures for a in snap.candidates}
        for r in results:
            costs[r["candidate"]][r["future"]] = float(r["cost"])
        for r in results:                                   # 충돌이 난 미래는 모든 후보에서 함께 뺀다
            if r["collisions_unexplained"] or r["collisions_at_204820"]:
                for a in costs:
                    costs[a][r["future"]] = None
        decision = choose(m4, costs, self.se_mult)
        chosen = m4 if self.always_m4 else decision["chosen"]
        keep = ("delay", "time_loss", "depart_wait", "residual", "cost", "left_east", "left_south", "injected",
                "collisions_unexplained", "collisions_at_204820", "future_seed", "seconds")
        point.update({"chosen": int(chosen), "decision": decision, "wall_seconds": time.monotonic() - started,
                      "rollouts": {str(a): [None] * self.futures for a in snap.candidates}})
        for r in results:
            point["rollouts"][str(r["candidate"])][r["future"]] = {key: r.get(key) for key in keep}
        self.points.append(point)
        vjob = {**base, "green": int(chosen)}
        self.verifies.append((k, self.pool.submit(verify_worker, vjob), stem))
        return int(chosen)

    def after_step(self, ctx, snap, action, executed) -> None:
        """선택 지점 실행 직후 원래 실행의 표를 남긴다(복원 확인용)."""
        if snap.choice and self.points and self.points[-1]["step"] == ctx.step and "main_after" not in self.points[-1]:
            self.points[-1]["main_after"] = trace_table(ctx.ep.env.sumo)

    def finish_verification(self) -> dict:
        checked = mismatched = 0
        for k, future, stem in self.verifies:
            r = future.result()
            self.rollout_seconds += float(r["seconds"])
            point = self.points[k]
            main = point.pop("main_after", None)
            ok = bool(r["ok"]) and main is not None and same_table(r["table"], main, self.tol)
            point["restore_check"] = {"ok": ok, "error": r.get("error")}
            checked += 1
            mismatched += not ok
            for ext in (".xml", ".pkl"):
                src = stem.with_suffix(ext)
                if self.keep_dir is not None and keep_state(stem.name) and src.exists():
                    # 상태 XML은 크므로 gzip으로 보관한다(쓰려면 풀어서 explore_rollout.restore에 넘긴다).
                    self.keep_dir.mkdir(parents=True, exist_ok=True)
                    with src.open("rb") as fin, gzip.open(self.keep_dir / (src.name + ".gz"), "wb") as fout:
                        shutil.copyfileobj(fin, fout)
                src.unlink(missing_ok=True)
        for p in self.points:
            p.pop("main_after", None)
        return {"checked": checked, "mismatched": mismatched}


# ----------------------------------------------------------------------------- 에피소드
def rollout_pool(workers: int):
    return ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                               initializer=ref._worker_init)


def run_bprime_episode(situation: dict, work: Path, pool, keep_dir: Path | None, **kwargs):
    work = Path(work)
    shutil.rmtree(work, ignore_errors=True)
    policy = BPrimePolicy(pool, situation, work / "states", keep_dir, **kwargs)
    started = time.monotonic()
    try:
        result = ac.run_arena_episode(situation, work / "episode", policy, max_seconds=6 * 3600.0,
                                      on_step=policy.after_step)
    finally:
        verification = policy.finish_verification() if policy.verifies else {"checked": 0, "mismatched": 0}
    waited = sum(float(p.get("wall_seconds", 0.0)) for p in policy.points)
    result.metrics.update({"bprime_points": len(policy.points), "bprime_restore_check": verification,
                           "bprime_switched": sum(p["chosen"] != p["m4"] for p in policy.points),
                           "bprime_rollout_cpu_seconds": policy.rollout_seconds,
                           "bprime_wall_seconds": time.monotonic() - started,
                           # 본 실행이 롤아웃을 기다린 시간은 빼고 센다(그동안 CPU는 롤아웃 작업자가 쓴다).
                           "bprime_main_seconds": max(0.0, time.monotonic() - started - waited)})
    return result, policy.points


def write_points(target: Path, points: list[dict]) -> str:
    target.mkdir(parents=True, exist_ok=True)
    path = target / "points.jsonl.gz"
    with gzip.open(path, "wt") as f:
        for p in points:
            f.write(json.dumps(w.plain(p), ensure_ascii=False, allow_nan=False) + "\n")
    return w.file_hash(path)


def read_points(target: Path) -> list[dict]:
    m = w.read(Path(target) / "metrics.json")
    path = Path(target) / "points.jsonl.gz"
    if w.file_hash(path) != m.get("points_sha256"):
        raise ValueError(f"선택 지점 자료 해시가 다릅니다: {target}")
    with gzip.open(path, "rt") as f:
        return [json.loads(line) for line in f]


def situations_for(role: str) -> list[dict]:
    cfg = ac.config()
    return cfg["situations_bprime_data"] if role == "data" else cfg["situations_eval"]


def cmd_run(args) -> int:
    out = Path(args.out)
    fp = ref.register(out, extra_files=("src/arena3_bprime.py",))
    sits = situations_for(args.role)
    mine = [s for i, s in enumerate(sits) if i % args.shards == args.shard]
    ledger = ref.Ledger(out, f"{args.role}_shard{args.shard}")
    status_path = out / "status" / f"{args.role}_shard{args.shard}.json"
    cfg = bcfg()["restore_check"]
    checked = mismatched = 0
    failed = []
    with rollout_pool(args.rollout_workers) as pool:
        for s in mine:
            sid = ac.situation_id(s)
            target = out / "runs" / POLICY / sid
            if ac.read_run(target, fp) is not None:
                m = w.read(target / "metrics.json")
                read_points(target)
                checked += m["bprime_restore_check"]["checked"]
                mismatched += m["bprime_restore_check"]["mismatched"]
                continue
            began = time.monotonic()
            try:
                result, points = run_bprime_episode(s, Path(args.work) / sid, pool,
                                                    out / "kept_states" if args.role == "data" else None)
                if result.metrics["route_sha256"] != s["route_sha256"]:
                    raise ValueError("차량 파일 SHA가 잠금과 다릅니다.")
                sha = write_points(target, points)
                ac.write_run(target, result, {"policy": POLICY, "fingerprint": fp, "model_identity": None,
                                              "points_sha256": sha, "role": args.role})
                rc = result.metrics["bprime_restore_check"]
                checked += rc["checked"]
                mismatched += rc["mismatched"]
                cpu = result.metrics["bprime_main_seconds"] + result.metrics["bprime_rollout_cpu_seconds"]
                ledger.add(cpu)
                ref.stamp(f"B′ {args.role} {sid}: 지연 {result.metrics['mean_delay_s']:.3f}초, 선택 지점 "
                          f"{result.metrics['bprime_points']}개, M4에서 바꿈 {result.metrics['bprime_switched']}번, "
                          f"복원 불일치 {rc['mismatched']}/{rc['checked']}, {cpu / 3600:.2f} CPU-시간")
            except BaseException as error:
                ledger.add(time.monotonic() - began)
                failed.append({"situation": sid, "error": f"{type(error).__name__}: {error}"})
                ref.stamp(f"B′ 실패 {sid}: {failed[-1]['error']}")
                if isinstance(error, KeyboardInterrupt):
                    raise
            finally:
                shutil.rmtree(Path(args.work) / sid, ignore_errors=True)
            rate = mismatched / checked if checked else 0.0
            w.write(status_path, {"done": sum((out / "runs" / POLICY / ac.situation_id(x) / "cache.json").exists() for x in mine),
                                  "total": len(mine), "failed": failed, "restore_checked": checked,
                                  "restore_mismatched": mismatched, "restore_mismatch_rate": rate,
                                  "updated_unix": time.time()})
            if checked >= cfg["min_points"] and rate > cfg["max_mismatch_rate"]:
                ref.stamp(f"복원 불일치 비율 {rate:.1%}가 {cfg['max_mismatch_rate']:.0%}를 넘어 멈춥니다.")
                return 3
    return 1 if failed else 0


def cmd_smoke(args) -> int:
    """실행 전 점검: (1) B′ 장치로 저장·롤아웃을 해도 항상 M4를 고르면 일반 M4와 같은가, (2) 롤아웃 비용·시간."""
    out = Path(args.out)
    fp = ref.register(out, extra_files=("src/arena3_bprime.py",))
    s = ac.config()["situations_bprime_data"][0]
    work = Path(args.work)
    shutil.rmtree(work, ignore_errors=True)
    plain = ac.run_arena_episode(s, work / "plain", ac.fixed_policy("M4"))
    with rollout_pool(args.rollout_workers) as pool:
        same_result, points = run_bprime_episode(s, work / "same", pool, None, always_m4=True,
                                                 max_rollout_points=args.points)
    keys = ("mean_delay_s", "actual_total_delay", "executed_sha16", "route_sha256", "choice_points")
    same = all(plain.metrics[k] == same_result.metrics[k] for k in keys)
    rolled = [p for p in points if p["rollouts"]]
    secs = [r["seconds"] for p in rolled for v in p["rollouts"].values() for r in v if r]
    report = {"situation": ac.situation_id(s), "unchanged_by_saving": same,
              "plain": {k: plain.metrics[k] for k in keys}, "bprime_always_m4": {k: same_result.metrics[k] for k in keys},
              "choice_points_per_hour": plain.metrics["choice_points"], "rolled_points": len(rolled),
              "rollout_seconds_mean": statistics.fmean(secs) if secs else None,
              "restore_check": same_result.metrics["bprime_restore_check"],
              "decisions_if_not_forced": [p["decision"] for p in rolled],
              "estimate_cpu_hours_per_episode": (plain.metrics["choice_points"] * statistics.fmean(len(p["candidates"]) for p in rolled)
                                                 * bcfg()["futures"] * statistics.fmean(secs) + plain.metrics["seconds"]) / 3600
              if secs else None,
              "fingerprint": fp}
    w.write(out / "smoke.json", report)
    ref.stamp(f"저장·롤아웃 무변경: {'같음' if same else '다름'} | 선택 지점 {report['choice_points_per_hour']}개/시간 | "
              f"롤아웃 평균 {report['rollout_seconds_mean']}초 | 에피소드당 약 {report['estimate_cpu_hours_per_episode']} CPU-시간 | "
              f"복원 확인 {report['restore_check']}")
    ok = same and report["restore_check"]["mismatched"] == 0 and len(rolled) == min(args.points, plain.metrics["choice_points"])
    shutil.rmtree(work, ignore_errors=True)
    return 0 if ok else 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("smoke")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--rollout-workers", type=int, default=8)
    p.add_argument("--points", type=int, default=3)
    p = sub.add_parser("run")
    p.add_argument("--role", choices=("data", "eval"), required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--shards", type=int, default=1)
    p.add_argument("--rollout-workers", type=int, default=4)
    args = parser.parse_args()
    w.configure()
    if args.cmd == "run" and not 0 <= args.shard < args.shards:
        raise SystemExit("--shard는 0 이상 --shards 미만이어야 합니다.")
    return {"smoke": cmd_smoke, "run": cmd_run}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
