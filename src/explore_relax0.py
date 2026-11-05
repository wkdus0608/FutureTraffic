"""[추가 탐색] RELAX0 — 신호 규칙을 줄이면 지연이 얼마나 주나 (학습 없음, 지문 밖, 진행 기록 252절).

같은 84상황·같은 차에서 규칙만 바꿔 계획기(R = M4, P = TM4)를 돌린다. 규칙은 잠긴 파일을 고치지 않고 실행 프로세스 안에서만 바꾼다.
황색·전적색과 차량 최대 누적 대기 165초는 그대로다. 판정 없음(기술 측정). 고정신호·현재 규칙 값은 PBASE·PLATOON0 결과를 읽는다.

사용(FOCUSED_RULESET=pet1 FOCUSED_NORTH_EXT=240):
  python src/explore_relax0.py smoke --out <dir> --work /content/rx_smoke --workers 4 --platoon0 <Drive>/explore_platoon0
  python src/explore_relax0.py run --out <Drive>/explore_relax0/R --work /content/rx --workers 8 --pairs R/A,R/B,R/C,R/D
  python src/explore_relax0.py report --out <Drive>/explore_relax0 --platoon0 <Drive>/explore_platoon0 \\
      --pbase <Drive>/explore_platoon_base --arena3 <Drive>/arena3/referee/baseline
"""
from __future__ import annotations

import argparse
import gzip
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
import explore_platoon_base as pb  # noqa: E402
import explore_weight as w  # noqa: E402

EXPERIMENT = "RELAX0_V1"
PLAN = "results/explore/rule_relax_20261101/측정계획.md"
EXTRA_FILES = ("src/explore_relax0.py", PLAN, *p0.EXTRA_FILES)
POLICY = {"R": "M4", "P": "TM4"}
WALK = {"길주로 횡단": 3, "석천로 횡단": 1}      # 횡단 → 함께 열리는 녹색(3 = 남북 직진, 1 = 길주로 직진)
PED_LONGER = {"길주로 횡단": 200, "석천로 횡단": 155}
RULES = {
    "base": {},
    "A": {"veh_min": 10},
    "B": {"ped_wait": PED_LONGER},
    "C": {"veh_min": 10, "ped_wait": PED_LONGER},
    "D": {"veh_min": 10, "ped_wait": PED_LONGER, "walk_min": {"길주로 횡단": 10, "석천로 횡단": 10}},
}
LABELS = {"A": "A 좌회전 최소 녹색 10초", "B": "B 보행 최대 대기 200·155초", "C": "C = A + B", "D": "D = C + 보행 최소 10초(현실 불가, 상한)"}
VARIANTS = ("A", "B", "C", "D")
FOLDERS = ("R", "P1", "P2")                      # 런타임(노트북)마다 결과 폴더를 따로 둔다
PAIRS = {"R": ("R/A", "R/B", "R/C", "R/D"), "P1": ("P/A", "P/B"), "P2": ("P/C", "P/D")}
SMOKE_PAIRS = ("R/base", "P/base", "R/A", "P/A")
MAX_EPISODE_S = 3 * 3600.0
_ORIGINAL: dict = {}


def apply_rule(rule: dict) -> None:
    """이 프로세스의 규칙 값만 바꾼다. 빠진 항목은 원래 값으로 돌린다(작업자가 판마다 부른다)."""
    import focused_rl as fr
    if not _ORIGINAL:
        _ORIGINAL.update(veh_min=fr.VEHICLE_MIN_GREEN, ped_wait={n: c["max_wait"] for n, c in fr.CROSSWALKS.items()},
                         walk_min=dict(fr.CROSSWALK_MIN_GREEN))
    fr.VEHICLE_MIN_GREEN = rule.get("veh_min", _ORIGINAL["veh_min"])
    for n, c in fr.CROSSWALKS.items():
        c["max_wait"] = rule.get("ped_wait", {}).get(n, _ORIGINAL["ped_wait"][n])
    for n in fr.CROSSWALK_MIN_GREEN:
        fr.CROSSWALK_MIN_GREEN[n] = rule.get("walk_min", {}).get(n, _ORIGINAL["walk_min"][n])


def green_stats(executed: list[int], step_s: int = 5) -> dict:
    """실행 녹색 기록으로 전환 수, 녹색별 길이, 횡단별 다시 열리는 간격과 보행자 평균 대기 어림(Σ간격² / 2T)을 잰다."""
    runs: list[list[int]] = []
    for g in executed:
        if runs and runs[-1][0] == g:
            runs[-1][1] += step_s
        else:
            runs.append([int(g), step_s])
    lengths: dict[int, list[int]] = {}
    for g, n in runs[1:-1]:                       # 처음과 끝은 잘린 녹색이라 뺀다
        lengths.setdefault(g, []).append(n)
    total = len(executed) * step_s
    gaps = {}
    for name, phase in WALK.items():
        out, gap = [], 0
        for g in executed:
            if g == phase:
                if gap:
                    out.append(gap)
                gap = 0
            else:
                gap += step_s
        if gap:
            out.append(gap)
        gaps[name] = out
    return {"switches": len(runs) - 1, "median_green": {str(g): statistics.median(v) for g, v in sorted(lengths.items())},
            "min_green": {str(g): min(v) for g, v in sorted(lengths.items())},
            "ped_gap_mean": {n: (statistics.fmean(v) if v else 0.0) for n, v in gaps.items()},
            "ped_gap_max": {n: max(v, default=0) for n, v in gaps.items()},
            "ped_wait_estimate": {n: sum(x * x for x in v) / (2.0 * total) for n, v in gaps.items()}}


def run_extra(result: ac.ArenaResult) -> dict:
    executed = [int(d["executed_green"]) for d in result.decisions]
    return {"free_steps": sum(len(d["allowed"]) >= 2 for d in result.decisions), "greens": green_stats(executed)}


# ----------------------------------------------------------------------------- 작업
def jobs_for(cfg: dict, out: Path, work: Path, fingerprint: str, pairs, situations=None) -> list[dict]:
    jobs = []
    for pair in pairs:                            # 변형 하나를 끝내고 다음으로(오래 걸리는 D가 앞 변형을 늦추지 않게)
        world, variant = pair.split("/")
        for s in (situations or cfg["situations"]):
            sid = ac.situation_id(s)
            jobs.append({"world": world, "variant": variant, "sid": sid, "situation": p0._situation(s),
                         "route_sha256": s["route_sha256"][world], "fingerprint": fingerprint, "model_identity": pair,
                         "target": str(Path(out) / "runs" / world / variant / sid), "work": str(Path(work) / world / variant / sid)})
    return jobs


def relax_worker(job: dict) -> dict:
    started = time.monotonic()
    work = Path(job["work"])
    name = f"{job['model_identity']}/{job['sid']}"
    try:
        w.configure()
        apply_rule(RULES[job["variant"]])
        shutil.rmtree(work, ignore_errors=True)
        policy = p0.build_policy(POLICY[job["world"]])
        with p0.world_context(job["world"], job["situation"]["tag"]) as stats:
            result = ac.run_arena_episode(job["situation"], work, policy, max_seconds=MAX_EPISODE_S)
        if result.metrics["route_sha256"] != job["route_sha256"]:
            raise ValueError(f"차량 파일 SHA가 고정값과 다릅니다: {result.metrics['route_sha256']} ≠ {job['route_sha256']}")
        ac.write_run(Path(job["target"]), result,
                     {"policy": POLICY[job["world"]], "world": job["world"], "variant": job["variant"], "rule": RULES[job["variant"]],
                      "fingerprint": job["fingerprint"], "model_identity": job["model_identity"],
                      "job_seconds": time.monotonic() - started, "world_changes": dict(stats), **run_extra(result)})
        return {"ok": True, "seconds": time.monotonic() - started, "name": name}
    except BaseException as error:
        return {"ok": False, "seconds": time.monotonic() - started, "name": name, "error": f"{type(error).__name__}: {error}"}
    finally:
        shutil.rmtree(work, ignore_errors=True)


def run_all(out: Path, work: Path, workers: int, pairs, situations=None, role: str = "run") -> dict:
    cfg = p0.config()
    fingerprint = ref.register(out, EXTRA_FILES)
    p0.prepare_caches()
    jobs = jobs_for(cfg, out, work, fingerprint, pairs, situations)
    ref.stamp(f"RELAX0 {role}: 작업 {len(jobs)}개, 작업자 {workers}개, 지문 {fingerprint[:12]}")
    result = ref.run_jobs(jobs, workers, ref.Ledger(out, role), worker=relax_worker)
    done = sum(1 for j in jobs if ac.read_run(Path(j["target"]), fingerprint, j["model_identity"]) is not None)
    ref.stamp(f"RELAX0 {role}: 끝난 판 {done}/{len(jobs)}, 실패 {len(result['failed'])}")
    return {"jobs": len(jobs), "done": done, "failed": result["failed"], "fingerprint": fingerprint}


def smoke(out: Path, work: Path, workers: int, platoon0: Path | None) -> dict:
    """작은 실행: 첫 상황 4판. 규칙을 안 바꾼 판은 PLATOON0와 신호까지 같아야 하고, A는 좌회전 녹색이 실제로 짧아져야 한다."""
    cfg = p0.config()
    first = cfg["situations"][0]
    sid = ac.situation_id(first)
    runs = run_all(out, work, workers, SMOKE_PAIRS, [first], "smoke")
    m = {pair: ac.read_run(Path(out) / "runs" / pair / sid, runs["fingerprint"], pair) for pair in SMOKE_PAIRS}
    checks = {"all_runs": all(v is not None for v in m.values())}
    if checks["all_runs"]:
        checks["quality_clean"] = all(not ref.quality_errors(v) for v in m.values())
        checks["route_pinned"] = all(v["route_sha256"] == first["route_sha256"][k.split("/")[0]] for k, v in m.items())
        checks["no_rule_breaks"] = all(v["mode_action_errors"] == 0 for v in m.values())
        for world in POLICY:
            a, b = m[f"{world}/A"]["greens"]["min_green"], m[f"{world}/base"]["greens"]["min_green"]
            checks[f"{world}_left_green_shorter"] = min(a.get("0", 999), a.get("2", 999)) < min(b.get("0", 999), b.get("2", 999))
            checks[f"{world}_signal_changed"] = m[f"{world}/A"]["executed_sha16"] != m[f"{world}/base"]["executed_sha16"]
            if platoon0 is not None:
                old = w.read(Path(platoon0) / "runs" / world / POLICY[world] / sid / "metrics.json")
                checks[f"{world}_base_equals_platoon0"] = all(m[f"{world}/base"][k] == old[k] for k in
                                                              ("mean_delay_s", "executed_sha16", "route_sha256", "cohort_sha256"))
    summary = {"ok": all(checks.values()), "checks": checks, "situation": sid,
               "delay_s": {k: (v or {}).get("mean_delay_s") for k, v in m.items()},
               "seconds": {k: (v or {}).get("seconds") for k, v in m.items()}}
    w.write(Path(out) / "smoke.json", summary)
    return summary


# ----------------------------------------------------------------------------- 비교표(SUMO 없음)
def decisions_of(folder: Path, sid: str) -> list[dict]:
    with gzip.open(Path(folder) / sid / "decisions.jsonl.gz", "rt") as f:
        return [json.loads(line) for line in f]


def variant_folder(out: Path, world: str, variant: str) -> Path | None:
    for name in FOLDERS:
        if f"{world}/{variant}" in PAIRS[name]:
            return Path(out) / name / "runs" / world / variant
    return None


def executed_of(folder: Path, sid: str) -> list[int]:
    """기준 실행의 실행 녹색. 고정신호 기록은 상태 문자열이라 보행 쪽 직진 신호로 바꾼다."""
    from focused_rl import CROSSWALKS
    out = []
    for row in decisions_of(folder, sid):
        if "executed_green" in row:
            out.append(int(row["executed_green"]))
            continue
        state = row["state"]
        walk = [p for n, p in WALK.items() if all(state[l] == "G" for l in CROSSWALKS[n]["links"])]
        out.append(walk[0] if walk else -1)
    return out


def ped_summary(rows: list[list[int]]) -> dict:
    stats = [green_stats(x) for x in rows]
    return {"gap_mean": {n: statistics.fmean(s["ped_gap_mean"][n] for s in stats) for n in WALK},
            "gap_max": {n: max(s["ped_gap_max"][n] for s in stats) for n in WALK},
            "wait_estimate": {n: statistics.fmean(s["ped_wait_estimate"][n] for s in stats) for n in WALK}}


def report(out: Path, platoon0: Path, pbase: Path, arena3: Path) -> dict:
    cfg, lock = p0.config(), ac.config()
    sids = [ac.situation_id(s) for s in cfg["situations"]]
    result = {"experiment": EXPERIMENT, "worlds": {}, "lock_sha256": w.file_hash(ac.CONFIG),
              "note": "반복 사용 날짜의 기술 측정(판정 없음). 이득은 규칙을 푼 몫이고 강화학습의 몫이 아니다. 시뮬레이션에 보행자는 없다."}
    p0_folder = {"P": Path(pbase) / "runs/P/P0", "R": Path(arena3) / "runs/P0"}
    for world, policy in POLICY.items():
        sits = [{**p0._situation(s), "route_sha256": s["route_sha256"][world]} for s in cfg["situations"]]
        route = {ac.situation_id(s): s["route_sha256"] for s in sits}
        current_folder = Path(platoon0) / "runs" / world / policy
        fixed_folder = Path(pbase) / "runs" / world / "FIXC"
        current = pb.read_folder(current_folder, sids, route, f"{world}/{policy}")
        fixed = pb.read_folder(fixed_folder, sids, route, f"{world}/FIXC")
        p0_runs = pb.read_folder(p0_folder[world], sids, route, f"{world}/P0")
        fix_mean = pb.policy_summary([fixed])["mean_delay_s"]
        table = {"FIXC": {**pb.policy_summary([fixed]), "pedestrian": ped_summary([executed_of(fixed_folder, s) for s in sids])},
                 "current": {**pb.policy_summary([current]), "pedestrian": ped_summary([executed_of(current_folder, s) for s in sids]),
                             "free_steps": statistics.fmean(sum(len(d["allowed"]) >= 2 for d in decisions_of(current_folder, s)) for s in sids),
                             "switches": statistics.fmean(green_stats(executed_of(current_folder, s))["switches"] for s in sids)}}
        missing = {}
        for variant in VARIANTS:
            folder = variant_folder(out, world, variant)
            done = [s for s in sids if (folder / s / "cache.json").exists()]
            if len(done) < len(sids):
                missing[variant] = f"{len(done)}/{len(sids)}"
                continue
            rows = pb.read_folder(folder, sids, route, f"{world}/{variant}")
            judged = ref.judge([rows], current, p0_runs, lock, sits)
            table[variant] = {**pb.policy_summary([rows]),
                              "vs_current": {"gain_date_mean_s": judged["gain_date_mean_s"], "date_t95_s": judged["date_t95_s"],
                                             "positive_dates": judged["positive_dates"]},
                              "pedestrian": ped_summary([executed_of(folder, s) for s in sids]),
                              "free_steps": statistics.fmean(m["free_steps"] for m in rows),
                              "switches": statistics.fmean(m["greens"]["switches"] for m in rows)}
        for row in table.values():
            row["reduction_vs_fixc_pct"] = 100.0 * (fix_mean - row["mean_delay_s"]) / fix_mean
        result["worlds"][world] = {"policy": policy, "table": table, "missing": missing}
    w.write(Path(out) / "report.json", result)
    (Path(out) / "summary.md").write_text(render(result), encoding="utf-8")
    return result


def render(rep: dict) -> str:
    f = lambda x: f"{x:+.2f}"  # noqa: E731
    lines = ["# RELAX0 — 신호 규칙을 줄이면 (같은 84상황, 학습 없음)", "",
             "차량당 지연(초/대, 차량 가중). 감소율은 보정 고정신호 대비. 현재 규칙 대비 이득은 날짜 7개 평균과 t95(+면 짧음).",
             "보행 대기는 무작위로 도착한 보행자 기준 어림(5초 기록, 길주로 횡단·석천로 횡단).", ""]
    names = {"P": "무리 도착 세계(P), 도구 TM4", "R": "무작위 도착 세계(R), 도구 M4"}
    for world, part in rep["worlds"].items():
        lines += [f"## {names[world]}", "",
                  "| 규칙 | 지연 | 고정 대비 | 현재 규칙 대비 [t95] | 양수 날짜 | 동 좌 | 남 좌 | 300초 초과 | 보행 대기 | 고를 수 있는 순간 | 전환 |",
                  "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
        for name, r in part["table"].items():
            v = r.get("vs_current")
            ped = r["pedestrian"]["wait_estimate"]
            lines.append(f"| {dict(FIXC='보정 고정신호', current='현재 규칙', **LABELS)[name]} | {r['mean_delay_s']:.2f} | {r['reduction_vs_fixc_pct']:.1f}% | "
                         + (f"{f(v['gain_date_mean_s'])} [{f(v['date_t95_s'][0])}, {f(v['date_t95_s'][1])}] | {v['positive_dates']}/7 | " if v else "- | - | ")
                         + f"{r['left_turns_s']['left_east']:.1f} | {r['left_turns_s']['left_south']:.1f} | {r['long_waits']:.0f} | "
                         + f"{ped['길주로 횡단']:.0f}·{ped['석천로 횡단']:.0f}초 | {r.get('free_steps', 0):.0f}/720 | {r.get('switches', 0):.0f} |")
        if part["missing"]:
            lines.append("")
            lines.append(f"아직 안 끝남: {part['missing']}")
        lines.append("")
    lines += [f"- {rep['note']}"]
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
            p.add_argument("--platoon0", type=Path, help="PLATOON0 결과 폴더(규칙을 안 바꾼 판의 재현 대조)")
        else:
            p.add_argument("--pairs", required=True, help="세계/변형 목록. 예: R/A,R/B")
    p = sub.add_parser("report")
    for name in ("out", "platoon0", "pbase", "arena3"):
        p.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    w.configure()
    if args.cmd == "smoke":
        s = smoke(args.out, args.work, args.workers, args.platoon0)
        print(json.dumps(s, ensure_ascii=False, indent=1))
        return 0 if s["ok"] else 2
    if args.cmd == "run":
        r = run_all(args.out, args.work, args.workers, tuple(args.pairs.split(",")))
        print(json.dumps({k: v for k, v in r.items() if k != "failed"} | {"failed": len(r["failed"])}, ensure_ascii=False))
        return 1 if r["failed"] else 0
    rep = report(args.out, args.platoon0, args.pbase, args.arena3)
    print(render(rep))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
