"""[추가 탐색] PBASE — 무리 도착 세계의 기준선 재기: 보정 고정신호(FIXC)와 P0 (학습 없음, 지문 밖, 진행 기록 247절).

PLATOON0·PLATOON1은 M4·TM4만 기준으로 썼고, 고정신호와 P0는 무리 도착 세계(P)에서 잰 적이 없다(241절 한계).
같은 84상황·같은 차·같은 seed에서 제어만 바꿔 잰다. 세계는 PLATOON0 그대로 쓰고, 잠긴 파일·이전 실험 코드는 고치지 않는다.

- FIXC = 그날 실제 운영 신호 + 전환 규칙 전적색(focused_eval의 fixedc와 같은 보정, focused_scenario.apply_all_red).
  준비 30분은 다른 제어와 같은 실제 신호(보정 전, 같은 넘겨받기)다. 넘겨받은 순간부터 제어 60분과 꼬리 15분 동안
  1초마다 보정한 실제 신호 상태를 그대로 켠다. ARENA3 제어는 꼬리를 P0가 맡지만 고정신호는 꼬리까지 고정신호다.
  비용 정산(5초 경계·차량 손실)은 ARENA3 실행기와 같은 함수(reconcile_costs·_exact_metrics)를 쓴다.
- P0 = ARENA3 P0Policy 그대로. 무작위 세계(R)의 P0는 ARENA3 심판 기준선에 이미 있어 다시 돌리지 않는다.
- 실행: P/FIXC, P/P0, R/FIXC 각 84판 = 252판. 판정 없음(기술 측정). 비교표는 report가 만든다.

사용(FOCUSED_RULESET=pet1 FOCUSED_NORTH_EXT=240):
  python src/explore_platoon_base.py smoke --out <dir> --work /content/pbs --workers 4 [--platoon0 <Drive>/explore_platoon0] \\
      [--arena3 <Drive>/arena3/referee/baseline]
  python src/explore_platoon_base.py run --out <Drive>/explore_platoon_base --work /content/pb --workers 8
  python src/explore_platoon_base.py report --out <Drive>/explore_platoon_base --platoon0 <Drive>/explore_platoon0 \\
      --platoon1 <Drive>/explore_platoon1 --arena3 <Drive>/arena3/referee/baseline
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import arena3_common as ac  # noqa: E402  (w.configure()로 pet1·북쪽 240 m를 먼저 정한다)
import arena3_referee as ref  # noqa: E402
import explore_platoon0 as p0  # noqa: E402
import explore_weight as w  # noqa: E402
import focused_scenario as fs  # noqa: E402
import focused_train as ft  # noqa: E402
from focused_data import TARGET_TLS  # noqa: E402

EXPERIMENT = "PBASE_V1"
PLAN = ROOT / "results/explore/platoon_base_20261101/측정계획.md"
RUNS = (("P", "FIXC"), ("P", "P0"), ("R", "FIXC"))
SMOKE_RUNS = (*RUNS, ("R", "P0"))     # 무작위 세계 P0는 작은 실행에서만: ARENA3 기준선 P0를 그대로 재현하는지 본다
MAX_EPISODE_S = 3600.0
EXTRA_FILES = ("src/explore_platoon_base.py", "results/explore/platoon_base_20261101/측정계획.md", *p0.EXTRA_FILES)
LABELS = {"FIXC": "보정 고정신호(실제 운영 신호)", "P0": "단순 예측 제어 P0", "M4": "M4(좌회전 2배 계획기)",
          "TM4": "TM4(M4 + 이웃 시간표)", "OM4": "OM4(정확한 도착, 참고)", "P1M4": "P1M4", "PT1": "PT1(정책 반복 학생, 세 모델 평균)"}


# ----------------------------------------------------------------------------- 고정신호(보정한 실제 운영 신호)
def fixed_states(date: str, period: str, length_s: int) -> tuple[list[str], dict]:
    """차량 파일 시각 0(준비 30분 앞)부터 length_s초 동안 1초마다의 보정 고정신호 상태와 보정 기록."""
    origin = fs.period_start_s(period) - ft.WARMUP_S
    phases = fs.aihub_signal_program(date, origin, length_s)
    corrected, log = fs.apply_all_red(phases)
    states = [s for d, s in corrected for _ in range(int(d))]
    if len(states) < length_s:
        raise ValueError(f"고정신호 프로그램이 짧습니다: {len(states)} < {length_s}")
    return states[:length_s], log


def run_fixed_episode(situation: dict, work: Path, *, max_seconds: float = MAX_EPISODE_S) -> ac.ArenaResult:
    """실제 신호 30분(다른 제어와 같음) → 보정 고정신호 60분 + 꼬리 15분. 정산은 ARENA3 실행기와 같다."""
    from explore_weight_episode import (CONTROL_S, STEP_S, TAIL_S, _Episode, _exact_metrics, cohort_fingerprint,
                                        cohort_vehicle_ids, fold_tail_rewards, read_tripinfo, reconcile_costs)
    from focused_eval import judge_collisions

    started = time.monotonic()
    work = Path(work).resolve()
    with ac.arrival_placement(situation["tag"]):
        ep = _Episode(situation["date"], situation["period"], int(situation["sumo_seed"]), work,
                      deadline=started + max_seconds)
    boundaries, snapshots, departures, decisions = [ep.handover], [], {}, []
    cohort = cohort_vehicle_ids(ep.vehicles, ep.handover)
    cohort_set = set(cohort)
    sumo = ep.env.sumo

    def capture():
        snapshots.append({v: float(sumo.vehicle.getTimeLoss(v)) for v in sumo.vehicle.getIDList() if v in cohort_set})
        for v in sumo.vehicle.getIDList():
            if v in cohort_set and v not in departures:
                departures[v] = float(sumo.vehicle.getDeparture(v))

    try:
        h = int(round(ep.handover))
        if abs(ep.handover - h) > 1e-9 or h != int(ep.start["handover_s"]):
            raise RuntimeError(f"넘겨받은 시각이 준비 신호 길이와 다릅니다: {ep.handover} ≠ {ep.start['handover_s']}")
        end = h + CONTROL_S + TAIL_S
        states, log = fixed_states(situation["date"], situation["period"], end)
        if states[h] != ep.start["handover_state"]:
            raise RuntimeError("보정 고정신호가 넘겨받는 순간 다른 제어와 같은 녹색이 아닙니다.")
        capture()
        for t in range(h, end):
            if time.monotonic() - started > max_seconds:
                raise TimeoutError("고정신호 에피소드 계산 예산을 초과했습니다.")
            if t < h + CONTROL_S and (t - h) % STEP_S == 0:
                decisions.append({"now": float(t), "state": states[t]})
            sumo.trafficlight.setRedYellowGreenState(TARGET_TLS, states[t])
            ep.env._sumo_step()
            for v in sumo.simulation.getDepartedIDList():
                if v in cohort_set:
                    departures[v] = float(sumo.vehicle.getDeparture(v))
            at = float(ep.env.sim_step)
            if abs(at - (t + 1)) > 1e-9:
                raise RuntimeError(f"1초 전이가 아닙니다: {t} → {at}")
            elapsed = at - ep.handover
            if elapsed > 0 and abs(elapsed / STEP_S - round(elapsed / STEP_S)) < 1e-9:
                boundaries.append(at)
                capture()
    finally:
        ep.close()

    rows = read_tripinfo(ep.tripinfo)
    costs = reconcile_costs(ep.vehicles, rows, boundaries, snapshots, handover=ep.handover, actual_departures=departures)
    rewards, reward_parts = fold_tail_rewards(costs["increments"])
    metrics = _exact_metrics(rows, ep.vehicles, ep.handover)
    collision = judge_collisions(ep.collisions)
    control = states[h:h + CONTROL_S]
    metrics.update({
        "experiment": ac.EXPERIMENT, "situation": ac.situation_id(situation), "date": situation["date"],
        "period": situation["period"], "tag": situation["tag"], "sumo_seed": int(situation["sumo_seed"]),
        "policy": "FIXC", "handover": ep.handover, "control_s": CONTROL_S, "tail_s": TAIL_S,
        "seconds": time.monotonic() - started, "actual_total_delay": costs["actual_total_delay"],
        "reconstructed_total_delay": costs["reconstructed_total_delay"], "cost_check": costs["cost_check"],
        "cost_check_passed": bool(costs["cost_check"]["passed"]), **reward_parts,
        "changed_by_rules": 0, "rule_decided": {}, "mode_action_errors": 0, "p0_mismatch": 0,
        "collisions_at_204820": int(collision["collisions_at_204820"]),
        "collisions_unexplained": len(collision["collisions_unexplained"]), "collision_details": collision,
        "route_sha256": hashlib.sha256(ep.route.read_bytes()).hexdigest(),
        "cohort_sha256": cohort_fingerprint(ep.vehicles, cohort), "choice_points": 0, "planning_seconds": 0.0,
        "executed_sha16": hashlib.sha256(json.dumps(control).encode()).hexdigest()[:16],
        "fixed_program": {"all_red_log": {k: v for k, v in log.items() if k != "by_transition"},
                          "tail": "고정신호(ARENA3 제어의 꼬리는 P0)"},
        "code": {name: w.file_hash(ROOT / name)[:12] for name in ("src/arena3_common.py", "src/explore_weight_episode.py",
                                                                  "src/explore_platoon_base.py")}})
    if len(decisions) != CONTROL_S // STEP_S:
        raise RuntimeError(f"제어 기록이 720개가 아닙니다: {len(decisions)}")
    return ac.ArenaResult(metrics, decisions, list(rewards), [], time.monotonic() - started)


# ----------------------------------------------------------------------------- 작업
def jobs_for(cfg: dict, out: Path, work: Path, fingerprint: str, pairs=RUNS, situations=None) -> list[dict]:
    jobs = []
    for s in (situations or cfg["situations"]):
        sid = ac.situation_id(s)
        for world, name in pairs:
            jobs.append({"world": world, "policy": name, "sid": sid, "situation": p0._situation(s),
                         "route_sha256": s["route_sha256"][world], "fingerprint": fingerprint,
                         "model_identity": f"{world}/{name}", "target": str(Path(out) / "runs" / world / name / sid),
                         "work": str(Path(work) / world / name / sid)})
    return jobs


def base_worker(job: dict) -> dict:
    started = time.monotonic()
    work = Path(job["work"])
    name = f"{job['world']}/{job['policy']}/{job['sid']}"
    try:
        w.configure()
        shutil.rmtree(work, ignore_errors=True)
        with p0.world_context(job["world"], job["situation"]["tag"]) as stats:
            if job["policy"] == "FIXC":
                result = run_fixed_episode(job["situation"], work)
            elif job["policy"] == "P0":
                result = ac.run_arena_episode(job["situation"], work, ac.fixed_policy("P0"), max_seconds=MAX_EPISODE_S)
            else:
                raise ValueError(f"PBASE 정책이 아닙니다: {job['policy']}")
        if result.metrics["route_sha256"] != job["route_sha256"]:
            raise ValueError(f"차량 파일 SHA가 고정값과 다릅니다: {result.metrics['route_sha256']} ≠ {job['route_sha256']}")
        ac.write_run(Path(job["target"]), result,
                     {"policy": job["policy"], "world": job["world"], "fingerprint": job["fingerprint"],
                      "model_identity": job["model_identity"], "job_seconds": time.monotonic() - started,
                      "world_changes": dict(stats), "information": {"kind": job["policy"]}})
        return {"ok": True, "seconds": time.monotonic() - started, "name": name}
    except BaseException as error:
        return {"ok": False, "seconds": time.monotonic() - started, "name": name, "error": f"{type(error).__name__}: {error}"}
    finally:
        shutil.rmtree(work, ignore_errors=True)


def run_all(out: Path, work: Path, workers: int, pairs=RUNS, situations=None, role: str = "run") -> dict:
    cfg = p0.config()
    fingerprint = ref.register(out, EXTRA_FILES)
    p0.prepare_caches()
    jobs = jobs_for(cfg, out, work, fingerprint, pairs, situations)
    ref.stamp(f"PBASE {role}: 작업 {len(jobs)}개, 작업자 {workers}개, 지문 {fingerprint[:12]}")
    result = ref.run_jobs(jobs, workers, ref.Ledger(out, role), worker=base_worker)
    done = sum(1 for j in jobs if ac.read_run(Path(j["target"]), fingerprint, j["model_identity"]) is not None)
    ref.stamp(f"PBASE {role}: 끝난 판 {done}/{len(jobs)}, 실패 {len(result['failed'])}")
    return {"jobs": len(jobs), "done": done, "failed": result["failed"], "fingerprint": fingerprint}


def smoke(out: Path, work: Path, workers: int, platoon0: Path | None, arena3: Path | None = None) -> dict:
    """작은 실행: 첫 상황 4판(P/FIXC·P/P0·R/FIXC·R/P0). 같은 차·같은 넘겨받기인지 PLATOON0의 같은 세계 M4와,
    R/P0가 ARENA3 기준선 P0와 같은지 대조한다."""
    cfg = p0.config()
    first = cfg["situations"][0]
    sid = ac.situation_id(first)
    runs = run_all(out, work, workers, pairs=SMOKE_RUNS, situations=[first], role="smoke")
    m = {f"{wd}/{n}": ac.read_run(Path(out) / "runs" / wd / n / sid, runs["fingerprint"], f"{wd}/{n}")
         for wd, n in SMOKE_RUNS}
    checks = {"all_runs": all(v is not None for v in m.values())}
    if checks["all_runs"]:
        checks["quality_clean"] = all(not ref.quality_errors(v) for v in m.values())
        checks["route_pinned"] = all(v["route_sha256"] == first["route_sha256"][k.split("/")[0]] for k, v in m.items())
        checks["fixed_program_untouched"] = all(v["fixed_program"]["all_red_log"].get("shortened_below_rule", 0) == 0
                                                for k, v in m.items() if k.endswith("FIXC"))
        checks["platoon_moved"] = m["P/FIXC"]["world_changes"].get("platoon", 0) > 0
        if platoon0 is not None:
            for world in ("P", "R"):
                base = Path(platoon0) / "runs" / world / "M4" / sid / "metrics.json"
                if base.exists():
                    b = w.read(base)
                    checks[f"{world}_same_cohort_as_platoon0_m4"] = all(
                        m[k]["cohort_sha256"] == b["cohort_sha256"] and m[k]["vehicles"] == b["vehicles"]
                        and m[k]["handover"] == b["handover"] for k in m if k.startswith(world + "/"))
        if arena3 is not None:
            base = Path(arena3) / "runs" / "P0" / sid / "metrics.json"
            if base.exists():
                b = w.read(base)
                checks["r_p0_equals_arena3"] = all(m["R/P0"][k] == b[k] for k in ("mean_delay_s", "executed_sha16",
                                                                                   "route_sha256", "cohort_sha256"))
    summary = {"ok": all(checks.values()), "checks": checks, "situation": sid,
               "delay_s": {k: (v or {}).get("mean_delay_s") for k, v in m.items()},
               "seconds": {k: (v or {}).get("seconds") for k, v in m.items()}}
    w.write(Path(out) / "smoke.json", summary)
    return summary


# ----------------------------------------------------------------------------- 비교표(SUMO 없음)
def read_folder(folder: Path, sids: list[str], route: dict[str, str], name: str) -> list[dict]:
    """한 정책 폴더의 84판을 읽는다: cache.json의 metrics 해시, 상황 번호, 그 세계의 차량 파일 SHA를 대조한다."""
    rows = []
    for sid in sids:
        d = Path(folder) / sid
        if not (d / "metrics.json").exists() or not (d / "cache.json").exists():
            raise FileNotFoundError(f"{name}: 실행이 없습니다: {d}")
        if w.read(d / "cache.json").get("metrics_sha256") != w.file_hash(d / "metrics.json"):
            raise ValueError(f"{name}: metrics 해시가 다릅니다: {d}")
        m = w.read(d / "metrics.json")
        if m.get("situation") != sid or m.get("route_sha256") != route[sid]:
            raise ValueError(f"{name}: 상황·차량 파일이 설정과 다릅니다: {sid}")
        rows.append(m)
    return rows


def sources(out: Path, platoon0: Path, platoon1: Path, arena3: Path) -> dict[str, dict[str, list[Path]]]:
    """세계 → 정책 → 모델 폴더 목록(PT1은 세 모델)."""
    pt1 = Path(platoon1) / "eval" / "PT1" / "runs"
    return {"P": {"FIXC": [Path(out) / "runs/P/FIXC"], "P0": [Path(out) / "runs/P/P0"],
                  "M4": [Path(platoon0) / "runs/P/M4"], "TM4": [Path(platoon0) / "runs/P/TM4"],
                  "OM4": [Path(platoon0) / "runs/P/OM4"], "P1M4": [Path(platoon0) / "runs/P/P1M4"],
                  "PT1": [pt1 / f"PT1_m{i}" for i in range(3)]},
            "R": {"FIXC": [Path(out) / "runs/R/FIXC"], "P0": [Path(arena3) / "runs/P0"],
                  "M4": [Path(platoon0) / "runs/R/M4"], "OM4": [Path(platoon0) / "runs/R/OM4"]}}


def policy_summary(models: list[list[dict]]) -> dict:
    n = sum(int(m["vehicles"]) for m in models[0])
    total = sum(statistics.fmean(float(rows[i]["actual_total_delay"]) for rows in models) for i in range(len(models[0])))
    lefts = {k: statistics.fmean(ref.movement_mean(rows, key) for rows in models) for k, key in ref.LEFT_KEYS.items()}
    return {"mean_delay_s": total / n, "vehicles": n, "left_turns_s": lefts,
            "long_waits": statistics.fmean(sum(int(m["long_waits"]) for m in rows) for rows in models),
            "collisions_unexplained": statistics.fmean(sum(int(m.get("collisions_unexplained", 0)) for m in rows)
                                                       for rows in models),
            "quality_errors": sum(bool(ref.quality_errors(m)) for rows in models for m in rows)}


def report(out: Path, platoon0: Path, platoon1: Path, arena3: Path) -> dict:
    cfg, lock = p0.config(), ac.config()
    sids = [ac.situation_id(s) for s in cfg["situations"]]
    result = {"experiment": EXPERIMENT, "worlds": {}, "lock_sha256": w.file_hash(ac.CONFIG),
              "evaluation_fingerprints": {}, "note": "반복 사용 날짜의 기술 측정(판정 없음). 무리 도착은 모형이다."}
    for folder, label in ((out, "PBASE"), (platoon0, "PLATOON0"), (Path(platoon1) / "eval" / "PT1", "PLATOON1"),
                          (arena3, "ARENA3")):
        result["evaluation_fingerprints"][label] = ref.evaluation_fingerprint(folder)
    if len(set(result["evaluation_fingerprints"].values())) != 1:
        raise ValueError(f"결과 폴더들의 평가 지문이 다릅니다: {result['evaluation_fingerprints']}")
    for world, policies in sources(out, platoon0, platoon1, arena3).items():
        sits = [{**p0._situation(s), "route_sha256": s["route_sha256"][world]} for s in cfg["situations"]]
        route = {ac.situation_id(s): s["route_sha256"] for s in sits}
        runs = {name: [read_folder(f, sids, route, f"{world}/{name}") for f in folders] for name, folders in policies.items()}
        table = {}
        for name, models in runs.items():
            row = policy_summary(models)
            for base in ("FIXC", "P0"):
                if name != base:
                    j = ref.judge(models, runs[base][0], runs["P0"][0], lock, sits)
                    row[f"vs_{base}"] = {"gain_date_mean_s": j["gain_date_mean_s"], "date_t95_s": j["date_t95_s"],
                                         "positive_dates": j["positive_dates"], "gain_vehicle_weighted_s": j["gain_vehicle_weighted_s"]}
            table[name] = row
        fixc = table["FIXC"]["mean_delay_s"]
        for row in table.values():
            row["reduction_vs_fixc_pct"] = 100.0 * (fixc - row["mean_delay_s"]) / fixc
        result["worlds"][world] = table
    w.write(Path(out) / "report.json", result)
    (Path(out) / "summary.md").write_text(render(result), encoding="utf-8")
    return result


def render(rep: dict) -> str:
    f = lambda x, nd=2: f"{x:+.{nd}f}"  # noqa: E731
    lines = ["# PBASE — 고정신호·P0·계획기·학생 비교(같은 84상황)", "",
             "차량당 지연(초/대, 차량 가중). 감소율은 보정 고정신호 대비 평균 지연으로 단순 계산했다. "
             "'고정신호 대비 이득'은 날짜 7개 평균과 t95(+면 고정신호보다 짧음).", ""]
    names = {"P": "무리 도착 세계(P, PLATOON 경기장)", "R": "무작위 도착 세계(R, ARENA3 경기장)"}
    for world, table in rep["worlds"].items():
        lines += [f"## {names[world]}", "",
                  "| 제어 | 차량당 지연 | 고정신호 대비 감소 | 고정신호 대비 이득 [t95] | 양수 날짜 | P0 대비 이득 | 동 좌회전 | 남 좌회전 | 300초 초과 |",
                  "|---|---:|---:|---|---:|---:|---:|---:|---:|"]
        for name, r in table.items():
            vf, vp = r.get("vs_FIXC"), r.get("vs_P0")
            lines.append(f"| {LABELS.get(name, name)} | {r['mean_delay_s']:.2f} | {r['reduction_vs_fixc_pct']:.1f}% | "
                         + (f"{f(vf['gain_date_mean_s'])} [{f(vf['date_t95_s'][0])}, {f(vf['date_t95_s'][1])}] | "
                            f"{vf['positive_dates']}/7 | " if vf else "기준 | - | ")
                         + (f"{f(vp['gain_date_mean_s'])} | " if vp else "기준 | ")
                         + f"{r['left_turns_s']['left_east']:.1f} | {r['left_turns_s']['left_south']:.1f} | {r['long_waits']:.1f} |")
        lines.append("")
    lines += ["- 고정신호는 꼬리 15분까지 고정신호다(ARENA3 제어의 꼬리는 P0). PT1은 세 모델 평균이다.", f"- {rep['note']}"]
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------------- 명령
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("smoke", "run"):
        p = sub.add_parser(name)
        p.add_argument("--out", type=Path, required=True)
        p.add_argument("--work", type=Path, required=True)
        p.add_argument("--workers", type=int, default=8)
        if name == "smoke":
            p.add_argument("--platoon0", type=Path, help="PLATOON0 결과 폴더(같은 차·넘겨받기 대조)")
            p.add_argument("--arena3", type=Path, help="ARENA3 심판 기준선 폴더(R/P0 재현 대조)")
    p = sub.add_parser("report")
    for name in ("out", "platoon0", "platoon1", "arena3"):
        p.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    w.configure()
    if args.cmd == "smoke":
        s = smoke(args.out, args.work, args.workers, args.platoon0, args.arena3)
        print(json.dumps(s, ensure_ascii=False, indent=1))
        return 0 if s["ok"] else 2
    if args.cmd == "run":
        r = run_all(args.out, args.work, args.workers)
        print(json.dumps({k: v for k, v in r.items() if k != "failed"} | {"failed": len(r["failed"])}, ensure_ascii=False))
        return 1 if r["failed"] else 0
    rep = report(args.out, args.platoon0, args.platoon1, args.arena3)
    print(render(rep))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
