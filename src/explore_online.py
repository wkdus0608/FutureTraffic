"""[추가 탐색] 제한된 1단계 — 좌회전 정체 시점에서만 SUMO 탐색으로 고르고 나머지는 P0로 1시간 연속 제어(진행 기록 123절). 지문 밖, 학습 없음.

질문: 좋았던 선택(122절)을 여러 번 적용하면 이득이 쌓이는가, 서로 지워지는가?
- 에피소드: 학습 날짜 08-22 13:00, 09-06 18:00, 09-07 13:00 × SUMO seed 42·43(짝 6쌍). 실제 신호 30분 → 제어 60분 → 꼬리 900초(P0).
- 비교: P0만 대 탐색(제어 60분 동안 좌회전 정체 시점에서만 탐색, 나머지와 꼬리는 P0). 같은 차량 파일·seed·운영 규칙.
- 탐색: 허용 녹색 × 표본 미래 8개, 첫 선택 뒤 P0, 600초, 비용 = 600초 실제 지연 + 남은 정체 × 75초(122절 민감도 규칙).
  탐색은 0단계 코드(explore_rollout.py, 6cc31fa07d09)를 고치지 않고 부른다. 탐색 안에서만 미래를 표본으로 만들고 본 실행은 실제 차 그대로.
- 재는 것: 제어 60분 동안 출발 예정인 차(두 방식에서 같은 차들)의 차량당 지연, 좌회전 두 곳, 300초 초과, 끝나지 않은 차, 충돌.
- 함께 저장: 모든 선택 시점의 관측 64·72칸, 허용 녹색, P0 선택, 실행한 선택, 탐색했으면 후보별 비용(다음 학습 단계용).

명령(Colab 38번 칸):
  FOCUSED_RULESET=pet1 FOCUSED_NORTH_EXT=240 python src/explore_online.py run --date 20220822 --period 1300 --seed 42 --mode p0 \
      --work /content/online --out <Drive>/explore_online
  FOCUSED_RULESET=pet1 FOCUSED_NORTH_EXT=240 python src/explore_online.py run ... --mode search --workers 8
  python src/explore_online.py summary --out <Drive>/explore_online
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import statistics
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

import explore_rollout as er  # noqa: E402

ROLLOUT_CODE = "6cc31fa07d09"                 # 0단계(122절) 탐색 코드
EPISODES = [("20220822", "1300"), ("20220906", "1800"), ("20220907", "1300")]
BASE_SEEDS = (42, 43)
TAIL_S = 900
LEFT_CUT = {"left_east": 8, "left_south": 7}  # 122절 상태 고르기의 좌회전 상위 10% 기준(3대 이상)
N_SCENARIOS = 8
MARK = str(er.H_SENS)                         # 600초 비용으로 고른다(122절 민감도 규칙)
MODES = ("p0", "search")


def code_fingerprint() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]


# ------------------------------------------------------------------ 순수 계산
def is_left_point(features: dict) -> bool:
    """123절 좌회전 정체 조건: 동쪽 좌회전 ≥ 8대(지금 녹색 아님) 또는 남쪽 좌회전 ≥ 7대(지금 녹색 아님)."""
    return any(not features[f"{n}_served"] and features[n] >= max(cut, er.LEFT_MIN) for n, cut in LEFT_CUT.items())


def scenario_seeds(base_seed: int, t0: float, n: int = N_SCENARIOS) -> list[int]:
    """선택 시점마다 새 시나리오: (SUMO seed × 10,000 + 시각) × 10 + j. 0단계 seed(1001~2008)와 겹치지 않는다."""
    return [(base_seed * 10_000 + int(t0)) * 10 + j for j in range(n)]


def choose(costs: dict, p0_action: int) -> tuple[int, dict]:
    """costs = {후보: [시나리오별 비용]}. 평균 비용이 가장 작은 후보(같으면 P0의 선택, 그다음 작은 번호)."""
    means = {c: statistics.fmean(v) for c, v in costs.items()}
    low = min(means.values())
    best = p0_action if means[p0_action] <= low + 1e-9 else min(c for c, v in means.items() if v <= low + 1e-9)
    return best, means


def episode_metrics(rows: dict, vehicles: dict, handover: float, control_s: int) -> dict:
    """제어 시간 동안 출발 예정인 차(차량 파일 기준)의 지연. rows = 차량별 기록(tripinfo, 끝나지 않은 차 포함).
    지연 = 시간 손실 + 진입 대기, 300초 초과 = 멈춰 있던 시간 + 진입 대기 > 300초(평가와 같은 정의)."""
    want = [v for v, (dep, _, _) in vehicles.items() if handover <= dep < handover + control_s]
    delays, by_move, long_waits, unfinished, missing = {}, {}, 0, 0, 0
    for v in want:
        r = rows.get(v)
        if r is None:
            missing += 1
            continue
        d = float(r["timeLoss"]) + float(r["departDelay"])
        delays[v] = d
        long_waits += float(r["waitingTime"]) + float(r["departDelay"]) > 300
        unfinished += float(r["arrival"]) < 0
        m = er.movement_of_edges(vehicles[v][1])
        by_move.setdefault("->".join(m) if m else "none", []).append(d)
    out = {"vehicles": len(want), "measured": len(delays), "missing": missing, "unfinished": unfinished, "long_waits": long_waits,
           "mean_delay_s": round(statistics.fmean(delays.values()), 3) if delays else None,
           "total_delay": round(sum(delays.values()), 1),
           "by_movement": {k: {"vehicles": len(v), "mean_delay_s": round(statistics.fmean(v), 2)} for k, v in sorted(by_move.items())}}
    for name, m in er.LEFTS.items():
        out[name] = out["by_movement"].get("->".join(m), {}).get("mean_delay_s")
    return out


def read_tripinfo(path: Path) -> dict:
    return {t.get("id"): t.attrib for t in ET.parse(path).getroot().iter("tripinfo")}


def pair_reading(pairs: list[dict]) -> dict:
    """123절 판정. pairs: [{"delta": P0 − 탐색 차량당 지연, "left_east_p0", "left_east_search", ..., "long_p0", "long_search"}]"""
    deltas = [p["delta"] for p in pairs]
    pos = sum(d > 0 for d in deltas)
    mean = statistics.fmean(deltas) if deltas else 0.0
    lefts_ok = all(statistics.fmean(p[f"{n}_search"] for p in pairs) <= statistics.fmean(p[f"{n}_p0"] for p in pairs)
                   for n in er.LEFTS)
    long_ok = sum(p["long_search"] for p in pairs) <= sum(p["long_p0"] for p in pairs)
    if mean > 0 and pos >= 5 and lefts_ok and long_ok:
        word = "이득이 쌓인다"
    elif mean <= 0 or pos <= 3:
        word = "이득이 없거나 지워진다"
    else:
        word = "불분명"
    return {"word": word, "mean_delta_s": round(mean, 3), "positive_pairs": pos, "pairs": len(pairs),
            "lefts_not_worse": lefts_ok, "long_waits_not_worse": long_ok}


# ------------------------------------------------------------------ SUMO
class OnlineEpisode(er.Episode):
    """0단계 Episode와 같되, 꼬리(TAIL_S) 동안의 실제 차도 넣고, SUMO seed를 고르고, 차량별 기록·충돌 기록을 남긴다."""

    def __init__(self, date: str, period: str, workdir: Path, base_seed: int, tail_s: int = TAIL_S):
        import focused_train as ft
        from focused_data import COUNTS, TARGET_TLS, load_counts
        from focused_rl import SignalConstraints, ZoneObservation, zone_queue_reward
        from focused_scenario import dqn_template_network, period_start_s, window_route_file
        from focused_sumo import make_environment
        workdir.mkdir(parents=True, exist_ok=True)
        self.date, self.period, self.base_seed = date, period, base_seed
        self.route = workdir / f"{date}_{period}.rou.xml"
        counts = load_counts(COUNTS)
        self.start = ft.warm_start(date, period, counts, self.route)
        origin = period_start_s(period) - ft.WARMUP_S        # 같은 시간대의 차는 창 길이와 상관없이 같다(window_route_file)
        window_route_file(date, origin, origin + ft.WARMUP_S + ft.CONTROL_S + tail_s, self.route, origin_s=origin, counts=counts)
        self.tripinfo, self.collisions = workdir / "tripinfo.xml", workdir / "collisions.xml"
        for p in (self.route, self.tripinfo, self.collisions):
            if " " in str(p):
                raise SystemExit(f"SUMO에 넘기는 경로에 공백이 있으면 안 된다: {p}")
        extra = (f"{er.STATE_OPTS} --tripinfo-output {self.tripinfo} --tripinfo-output.write-unfinished true "
                 f"--collision-output {self.collisions}")
        self.env = make_environment(seed=base_seed, net_file=dqn_template_network(ft.CACHE), route_file=self.route,
                                    simulation_duration_sec=self.start["handover_s"] + ft.CONTROL_S + tail_s,
                                    observation_class=ZoneObservation, reward_fn=zone_queue_reward, min_green=15, max_green=120,
                                    yellow_time=ft.AIHUB_YELLOW_S, additional_sumo_cmd=extra)
        self.env.warmup_states, self.env.warmup_handover = self.start["warmup_states"], self.start["handover_state"]
        self.cons = SignalConstraints().bind(self.env)
        self.cons.reset(self.start["last_served"])
        self.env.reset(seed=base_seed)
        self.ts = self.env.traffic_signals[TARGET_TLS]
        self.vehicles, _ = er.read_routes(self.route)
        P0 = er._p0()
        self.movement_of = P0.movements(self.route)
        self.ctl = P0(self.movement_of, [p.state for p in self.ts.green_phases])
        self.handover = float(self.env.sim_step)
        self.done = False


def observation72(ep: OnlineEpisode) -> tuple[list[float], list[float]]:
    """학습용 관측: 지금 관측 64칸과 114절 이동별 8칸을 더한 72칸(explore_choice와 같은 셈)."""
    import explore_choice as ec
    obs = ep.ts.observation_fn()
    moves = ec.protected_movements([p.state for p in ep.ts.green_phases])
    extra = ec.movement_features(ec.waiting_by_movement(ep.env.sumo, ep.movement_of), moves, ep.ts.observation_fn._capacity)
    return [float(x) for x in obs], [float(x) for x in ec.insert_before_mask(obs, extra, ep.ts.num_green_phases)]


def run(args) -> None:
    import focused_train as ft
    from concurrent.futures import ProcessPoolExecutor
    from focused_eval import judge_collisions
    import explore_pred
    if er.code_fingerprint() != ROLLOUT_CODE:
        raise SystemExit(f"탐색 코드 {er.code_fingerprint()}가 0단계(122절)의 {ROLLOUT_CODE}와 다르다.")
    if explore_pred.code_fingerprint() != er.P0_CODE:
        raise SystemExit("예측 제어 코드가 100절과 다르다.")
    tag = f"{args.date}_{args.period}_s{args.seed}_{args.mode}"
    out = Path(args.out) / "runs" / tag
    if (out / "metrics.json").exists():
        print(f"이미 있음 → 건너뜀: {out}")
        return
    work = Path(args.work) / tag
    shutil.rmtree(work, ignore_errors=True)
    pool = None
    if args.mode == "search":
        # 본 실행의 SUMO 연결을 물려받지 않게, 하위 실행은 SUMO를 열기 전에 새 프로세스(spawn)로 띄워 둔다
        import multiprocessing
        pool = ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn"))
        list(pool.map(abs, range(args.workers)))
    ep = OnlineEpisode(args.date, args.period, work / "routes", args.seed)
    er.types_only_file(ep.route)                       # 탐색 하위 실행이 동시에 만들지 않게 먼저
    end_control = ep.handover + ft.CONTROL_S
    decisions, searched = [], 0
    started = time.time()
    try:
        while not ep.done:
            allowed, a_p0 = ep.decide()
            action = a_p0
            if len(allowed) > 1:
                t0 = float(ep.env.sim_step)
                feat = er._features(ep, allowed, a_p0)
                obs64, obs72 = observation72(ep)
                row = {"t0": t0, "in_control": t0 < end_control, "allowed": allowed, "p0_action": a_p0, "W": feat["W"],
                       **{k: feat[k] for k in (*er.LEFTS, *[f"{n}_served" for n in er.LEFTS])}, "left_point": is_left_point(feat),
                       "obs64": obs64, "obs72": obs72, "searched": False}
                if (args.mode == "search" and row["in_control"] and row["left_point"]
                        and (args.max_searches is None or searched < args.max_searches)):
                    s0 = time.time()
                    stem = work / "states" / f"{tag}_{int(t0)}"
                    ep.snapshot(stem, allowed, a_p0)
                    seeds = scenario_seeds(args.seed, t0, args.scenarios)
                    jobs = [(str(stem), str(work / "routes"), c, s) for c in allowed for s in seeds]
                    res = list(pool.map(er._rollout_job, jobs))
                    costs = {c: [r["marks"][MARK]["cost"] for r in res if r["candidate"] == c] for c in allowed}
                    action, means = choose(costs, a_p0)
                    row.update(searched=True, chosen=action, seeds=seeds, means={str(c): round(v, 2) for c, v in means.items()},
                               scores={str(c): [{k: r["marks"][k] for k in r["marks"]} | {"seed": r["seed"]}
                                                for r in res if r["candidate"] == c] for c in allowed},
                               expected_gain=round(means[a_p0] - means[action], 2), wall_s=round(time.time() - s0, 1),
                               cpu_s=round(sum(r["seconds"] for r in res), 1), leaks=sum(r["leaks"] for r in res),
                               changed_by_rules=sum(r["changed_by_rules"] for r in res))
                    for suffix in (".xml", ".pkl"):              # 상태 파일은 크다(수 MB), 결과만 남긴다
                        stem.with_suffix(suffix).unlink(missing_ok=True)
                    searched += 1
                row["executed"] = action
                decisions.append(row)
            ep.apply(action)
    finally:
        if pool:
            pool.shutdown(wait=True, cancel_futures=True)
        ep.close()
    rows = read_tripinfo(ep.tripinfo)
    metrics = episode_metrics(rows, ep.vehicles, ep.handover, ft.CONTROL_S)
    coll = judge_collisions(ep.collisions)
    s = [d for d in decisions if d["searched"]]
    metrics.update({
        "date": args.date, "period": args.period, "seed": args.seed, "mode": args.mode, "handover": ep.handover,
        "code": code_fingerprint(), "rollout_code": er.code_fingerprint(), "p0_code": er.P0_CODE, "scenarios": args.scenarios,
        "max_searches": args.max_searches, "seconds": round(time.time() - started, 1),
        "choice_points": len(decisions), "choice_points_in_control": sum(d["in_control"] for d in decisions),
        "left_points_in_control": sum(d["in_control"] and d["left_point"] for d in decisions),
        "searches": len(s), "searched_differs": sum(d["chosen"] != d["p0_action"] for d in s),
        "expected_gain_sum": round(sum(d["expected_gain"] for d in s), 1),
        "expected_gains": sorted((d["expected_gain"] for d in s if d["chosen"] != d["p0_action"]), reverse=True),
        "search_wall_s": round(sum(d["wall_s"] for d in s), 1), "search_cpu_s": round(sum(d["cpu_s"] for d in s), 1),
        "search_leaks": sum(d["leaks"] for d in s), "search_changed_by_rules": sum(d["changed_by_rules"] for d in s),
        "changed_by_rules": ep.cons.interventions, "collisions_at_204820": coll["collisions_at_204820"],
        "collisions_unexplained": len(coll["collisions_unexplained"])})
    out.mkdir(parents=True, exist_ok=True)
    with (out / "decisions.jsonl").open("w") as f:
        for d in decisions:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    delays = {v: float(r["timeLoss"]) + float(r["departDelay"]) for v, r in rows.items()}
    (out / "vehicle_delays.json").write_text(json.dumps(delays))
    tmp = out / "metrics.json.tmp"
    tmp.write_text(json.dumps(metrics, ensure_ascii=False, indent=1))
    tmp.replace(out / "metrics.json")
    print(json.dumps({k: v for k, v in metrics.items() if k not in ("by_movement", "expected_gains")}, ensure_ascii=False), flush=True)


# ------------------------------------------------------------------ 정리
def summary(args) -> None:
    out = Path(args.out)
    runs = {}
    for p in sorted((out / "runs").glob("*/metrics.json")):
        m = json.loads(p.read_text())
        runs[(m["date"], m["period"], m["seed"], m["mode"])] = (m, p.parent)
    codes = {m["code"] for m, _ in runs.values()}
    if len(codes) != 1:
        raise SystemExit(f"서로 다른 코드의 실행이 섞였다: {sorted(codes)}")
    pairs, missing = [], []
    for date, period in EPISODES:
        for seed in BASE_SEEDS:
            a, b = runs.get((date, period, seed, "p0")), runs.get((date, period, seed, "search"))
            if not (a and b):
                missing.append(f"{date}_{period}_s{seed}")
                continue
            (p0, p0dir), (se, sedir) = a, b
            d0 = json.loads((p0dir / "vehicle_delays.json").read_text())
            d1 = json.loads((sedir / "vehicle_delays.json").read_text())
            same = [v for v in d0 if v in d1]
            realized = sum(d0[v] for v in same) - sum(d1[v] for v in same)
            pairs.append({"episode": f"{date}_{period}", "seed": seed, "delta": round(p0["mean_delay_s"] - se["mean_delay_s"], 3),
                          "p0_mean": p0["mean_delay_s"], "search_mean": se["mean_delay_s"],
                          "left_east_p0": p0["left_east"], "left_east_search": se["left_east"],
                          "left_south_p0": p0["left_south"], "left_south_search": se["left_south"],
                          "long_p0": p0["long_waits"], "long_search": se["long_waits"],
                          "unfinished_p0": p0["unfinished"], "unfinished_search": se["unfinished"],
                          "realized_total_gain": round(p0["total_delay"] - se["total_delay"], 1),
                          "realized_all_vehicles": round(realized, 1),
                          "expected_gain_sum": se["expected_gain_sum"],
                          "realized_over_expected": (round((p0["total_delay"] - se["total_delay"]) / se["expected_gain_sum"], 3)
                                                     if se["expected_gain_sum"] else None),
                          "top2_share": (round(sum(se["expected_gains"][:2]) / sum(se["expected_gains"]), 3)
                                         if se["expected_gains"] and sum(se["expected_gains"]) > 0 else None),
                          "searches": se["searches"], "searched_differs": se["searched_differs"],
                          "search_wall_h": round(se["search_wall_s"] / 3600, 2), "search_cpu_h": round(se["search_cpu_s"] / 3600, 2),
                          "collisions_at_204820": [p0["collisions_at_204820"], se["collisions_at_204820"]],
                          "unexplained": [p0["collisions_unexplained"], se["collisions_unexplained"]],
                          "leaks": se["search_leaks"], "changed_by_rules": [p0["changed_by_rules"], se["changed_by_rules"],
                                                                             se["search_changed_by_rules"]]})
    result = {"code": codes.pop(), "pairs": pairs, "missing": missing,
              "reading": pair_reading(pairs) if len(pairs) == len(EPISODES) * len(BASE_SEEDS) else None}
    (out / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=1))
    lines = ["# 제한된 1단계 요약(진행 기록 123절)", "",
             f"- 판정: {result['reading']['word'] if result['reading'] else '쌍이 다 모이지 않음: ' + ', '.join(missing)}", "",
             "| 에피소드 | seed | P0 지연 | 탐색 지연 | Δ(초) | 동쪽 좌회전 P0 / 탐색 | 남쪽 좌회전 P0 / 탐색 | 300초 초과 P0 / 탐색 | "
             "탐색 수(다르게 고름) | 줄어든 총 지연 / 예상 이득 합 | 상위 2개 몫 | 탐색 시간(벽·CPU, 시간) |",
             "|---|---:|---:|---:|---:|---|---|---|---|---|---:|---|"]
    for p in pairs:
        lines.append(f"| {p['episode']} | {p['seed']} | {p['p0_mean']} | {p['search_mean']} | {p['delta']} | {p['left_east_p0']} / "
                     f"{p['left_east_search']} | {p['left_south_p0']} / {p['left_south_search']} | {p['long_p0']} / {p['long_search']} | "
                     f"{p['searches']}({p['searched_differs']}) | {p['realized_total_gain']} / {p['expected_gain_sum']} | {p['top2_share']} | "
                     f"{p['search_wall_h']} · {p['search_cpu_h']} |")
    if result["reading"]:
        r = result["reading"]
        lines += ["", f"- 평균 Δ {r['mean_delta_s']}초, Δ > 0인 쌍 {r['positive_pairs']}/{r['pairs']}, 좌회전 안 나빠짐 {r['lefts_not_worse']}, "
                      f"300초 초과 안 나빠짐 {r['long_waits_not_worse']}"]
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--date", required=True)
    r.add_argument("--period", required=True)
    r.add_argument("--seed", type=int, required=True)
    r.add_argument("--mode", choices=MODES, required=True)
    r.add_argument("--work", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--workers", type=int, default=4)
    r.add_argument("--scenarios", type=int, default=N_SCENARIOS)
    r.add_argument("--max-searches", type=int, default=None, help="맥 시험용(본 실행에서는 쓰지 않음)")
    s = sub.add_parser("summary")
    s.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.cmd == "run":
        if (args.date, args.period) not in EPISODES or args.seed not in BASE_SEEDS:
            if args.max_searches is None:
                raise SystemExit("123절에 정한 에피소드·seed만 돈다(맥 시험은 --max-searches를 준다).")
        if args.scenarios != N_SCENARIOS and args.max_searches is None:
            raise SystemExit(f"123절: 시나리오는 {N_SCENARIOS}개다.")
        run(args)
    else:
        summary(args)


if __name__ == "__main__":
    main()
