"""확인 안 된 충돌을 같은 조건으로 다시 돌려, 충돌 전후 차의 움직임을 기록한다(원인 확인용).

- 평가(focused_eval.run_day)를 그대로 다시 돌린다. 날짜·정책·도착 방식·평가 seed는 원래 결과의 metrics.json에서 읽는다.
- SUMO를 1초 진행할 때마다, 충돌 BEFORE_S초 전부터 AFTER_S초 뒤까지 충돌한 두 차와 그 교차로 안에 있던 차의
  위치·속도·차로와 석천사거리 신호 상태를 기록한다. 값을 묻기만 하고 바꾸지 않으므로 차의 움직임은 평가와 같다.
  기록 구간이 끝나면 멈춘다(하루 끝까지 돌리지 않음).
- 다시 돌린 충돌 기록에 같은 시각·차로·두 차의 충돌이 있으면 '재현됨'이다. 맥과 Colab은 차 몇십 대 단위로
  다를 수 있으므로(10-06 확인), 결과를 낸 곳(Colab)에서 돌린다.
- 잠근 환경 파일과 평가 코드를 고치지 않는다(두 지문 그대로). 결과는 평가 폴더가 아닌 --out에 쓴다.
- DQN 실행(10-22부터): 원래 metrics.json에 적힌 모델 폴더·모델 파일 SHA-256을 그대로 평가에 넘긴다. 모델 파일이
  그 뒤 바뀌었으면 평가(run_day)가 멈춘다. 모델 폴더 경로는 평가한 곳(Colab Drive) 경로이므로 Colab에서 돌린다.
- 두 차는 1초마다 경로상 다음 도로 AHEAD_EDGES개의 차 수·멈춘 차 수도 적는다(나가는 길이 막혔는지 보려고).
- 12개 지도 조각 평가의 충돌은 network_collision이 이 파일의 기록 장치를 그대로 쓴다.

사용(Colab):
  python src/focused_collision.py --eval-dir /content/drive/MyDrive/FutureTraffic/eval_v5/955fd7d7b85f \
      --out /content/drive/MyDrive/FutureTraffic/collision_check --workers 3
  python src/focused_collision.py --run <평가 실행 폴더(들)> --out <폴더>        (실행 폴더를 직접 고를 때)
결과: <out>/<이름>/ 아래 case.json(재현 여부·요약), trajectory.csv(매초 기록). 이름은 <날짜>_<정책>[_platoon]_<충돌 시각>,
  DQN은 <날짜>_<실행 폴더 이름>_<충돌 시각>(학습 seed·모델 지문이 들어감). case.json이 이미 있는 건은 건너뛴다.
"""
from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from focused_data import TARGET_TLS

BEFORE_S, AFTER_S = 40, 3
FOE_LOOKAHEAD_M = 50.0      # 이 거리 안의 교차로 상대 차(getJunctionFoes)를 기록한다
AHEAD_EDGES = 3             # 두 차의 경로에서 지금 도로 다음 몇 개 도로의 차 수·멈춘 차 수를 적는다
DQN_KEYS = ("model", "model_sha256", "train_seed", "model_timesteps")   # DQN 평가를 다시 돌리는 데 필요한 값(metrics.json)


class StopAfterWindow(Exception):
    """기록 구간이 끝나 실행을 멈춘다."""


def junction_prefix(lane: str) -> str | None:
    """교차로 안 차로 이름의 교차로 부분. ':204820_15_0' → ':204820_'. 교차로 안 차로가 아니면 None."""
    if not lane.startswith(":") or lane.count("_") < 2:
        return None
    return lane.rsplit("_", 2)[0] + "_"


def cases_in_run(run_dir: Path) -> list[dict]:
    """평가 실행 하나의 확인 안 된 충돌(건마다 하나). DQN이면 모델 신원을 함께 담는다."""
    run_dir = Path(run_dir)
    m = json.loads((run_dir / "metrics.json").read_text())
    arrivals = m.get("arrivals", "random")
    cases = []
    for event in m.get("collisions_unexplained", []):
        time_s = float(event["time"])
        case = {"date": m["date"], "policy": m["policy"], "arrivals": arrivals, "eval_seed": m["eval_seed"],
                "time": time_s, "lane": event["lane"], "collider": event["collider"],
                "victim": event["victim"], "event": event, "source_run": str(run_dir),
                "evaluation_fingerprint": m.get("evaluation_fingerprint"),
                "name": f"{m['date']}_{m['policy']}{'' if arrivals == 'random' else '_' + arrivals}_{int(time_s)}"}
        if m["policy"] == "dqn":
            missing = [k for k in DQN_KEYS if m.get(k) is None]      # 흉내만 낸 모델은 학습량 0
            if missing:
                raise SystemExit(f"{run_dir}: DQN 실행인데 metrics.json에 {missing}이 없습니다(같은 모델로 다시 돌릴 수 없음).")
            case.update({k: m[k] for k in DQN_KEYS})
            case["name"] = f"{m['date']}_{run_dir.name}_{int(time_s)}"
        cases.append(case)
    return cases


def find_cases(eval_dir: Path) -> list[dict]:
    """평가 폴더에서 확인 안 된 충돌을 모두 찾는다(실행 하나에 여러 건이면 건마다 하나)."""
    return [c for metrics in sorted(Path(eval_dir).glob("*/*/metrics.json")) for c in cases_in_run(metrics.parent)]


def raw_connection(sumo):
    """12개 지도는 SUMO 연결을 LocalView(network_rl)로 감싼다. 기록은 감싸지 않은 연결로 읽어 LocalView의 셈을 건드리지 않는다."""
    return sumo._conn if type(sumo).__name__ == "LocalView" else sumo


def route_ahead(sumo, v: str) -> list[dict]:
    """차의 경로에서 지금 도로 다음 AHEAD_EDGES개 도로: 차 수·멈춘 차 수·길이(1번 차로)."""
    route, k = sumo.vehicle.getRoute(v), sumo.vehicle.getRouteIndex(v)
    return [{"edge": e, "vehicles": sumo.edge.getLastStepVehicleNumber(e), "halting": sumo.edge.getLastStepHaltingNumber(e),
             "length_m": round(sumo.lane.getLength(f"{e}_0"), 1)} for e in route[k + 1:k + 1 + AHEAD_EDGES]]


def snapshot(sumo, t: float, case: dict, prefix: str) -> list[dict]:
    """1초분 기록: 충돌한 두 차 + 그 교차로 안 차로에 있는 모든 차."""
    present = set(sumo.vehicle.getIDList())
    lanes = {v: sumo.vehicle.getLaneID(v) for v in present}
    roles = {case["collider"]: "collider", case["victim"]: "victim"}
    ids = sorted(v for v in present if v in roles or lanes[v].startswith(prefix))
    tls_state = sumo.trafficlight.getRedYellowGreenState(TARGET_TLS)
    rows = []
    for v in ids:
        x, y = sumo.vehicle.getPosition(v)
        row = {"time": t, "vehicle": v, "role": roles.get(v, "other"), "type": sumo.vehicle.getTypeID(v),
               "lane": lanes[v], "pos": round(sumo.vehicle.getLanePosition(v), 2),
               "speed": round(sumo.vehicle.getSpeed(v), 2), "accel": round(sumo.vehicle.getAcceleration(v), 2),
               "x": round(x, 2), "y": round(y, 2), "angle": round(sumo.vehicle.getAngle(v), 1),
               "length": sumo.vehicle.getLength(v), "tls_state": tls_state, "next_tls": "", "junction_foes": "",
               "route_ahead": ""}
        if v in roles:      # 다음 신호(번호·거리·상태), 경로상 다음 도로의 막힘, 차가 보고 있던 교차로 상대 차
            row["next_tls"] = json.dumps([list(x) for x in sumo.vehicle.getNextTLS(v)][:1])
            row["route_ahead"] = json.dumps(route_ahead(sumo, v))
            try:
                foes = sumo.vehicle.getJunctionFoes(v, FOE_LOOKAHEAD_M)
                row["junction_foes"] = json.dumps([{"foe": f[0], "ego_dist": round(f[1], 2), "foe_dist": round(f[2], 2),
                                                    "ego_lane": f[5], "foe_lane": f[6], "ego_response": f[7],
                                                    "foe_response": f[8]} for f in foes])
            except Exception as exc:   # SUMO 판에 따라 없을 수 있다. 기록만 하고 계속한다.
                row["junction_foes"] = f"error: {exc}"
        rows.append(row)
    return rows


def signal_changes(rows: list[dict]) -> list[tuple[float, str]]:
    seen, out = None, []
    for t in sorted({r["time"] for r in rows}):
        state = next(r["tls_state"] for r in rows if r["time"] == t)
        if state != seen:
            out.append((t, state))
            seen = state
    return out


def summarize(rows: list[dict], case: dict, prefix: str) -> dict:
    """두 차가 교차로에 들어간 시각·그때 속도·그때 신호, 충돌까지 걸린 시간, 충돌 직전 멈춰 있던 시간과 나가는 길."""
    out = {}
    for role in ("collider", "victim"):
        mine = [r for r in rows if r["role"] == role]
        inside = [r for r in mine if r["lane"].startswith(prefix)]
        before = [r for r in mine if not r["lane"].startswith(prefix) and r["next_tls"] not in ("", "[]")]
        entered = inside[0] if inside else None
        upto = [r for r in mine if r["time"] <= case["time"]]
        stopped = 0
        for r in reversed(upto):           # 충돌 시각에서 거슬러 속도 0.1m/s 미만이 이어진 초
            if r["speed"] >= 0.1:
                break
            stopped += 1
        out[role] = {"vehicle": case[role], "type": mine[0]["type"] if mine else None,
                     "entered_junction_s": entered["time"] if entered else None,
                     "speed_at_entry": entered["speed"] if entered else None,
                     "tls_state_at_entry": entered["tls_state"] if entered else None,
                     "seconds_inside_before_collision": (case["time"] - entered["time"]) if entered else None,
                     "last_next_tls_before_entry": json.loads(before[-1]["next_tls"]) if before else None,
                     "stopped_s_before_collision": stopped,
                     "route_ahead_at_collision": json.loads(upto[-1]["route_ahead"]) if upto and upto[-1]["route_ahead"] else None,
                     "lanes": sorted({r["lane"] for r in mine})}
    out["signal_changes"] = [{"time": t, "state": s} for t, s in signal_changes(rows)]
    return out


def install_recorder(case: dict, prefix: str, rows: list[dict]):
    """focused_sumo.make_environment를 바꿔 끼운다. 그 뒤에 만들어지는 환경은 SUMO를 1초 진행할 때마다 기록 구간
    [충돌 − BEFORE_S, 충돌 + AFTER_S] 안이면 rows에 적고, 구간이 끝나면 StopAfterWindow로 멈춘다.
    평가 코드(focused_eval.run_day, network_rl.WarmStart12Env)는 실행할 때 이 이름을 가져다 쓴다. 되돌리는 함수를 돌려준다."""
    import focused_sumo
    lo, hi = case["time"] - BEFORE_S, case["time"] + AFTER_S
    original = focused_sumo.make_environment

    def recording_environment(**kwargs):
        env = original(**kwargs)
        step = env._sumo_step

        def recorded_step():
            step()
            t = float(env.sim_step)
            if t >= lo:
                rows.extend(snapshot(raw_connection(env.sumo), t, case, prefix))
            if t >= hi:
                raise StopAfterWindow

        env._sumo_step = recorded_step
        return env

    focused_sumo.make_environment = recording_environment

    def restore():
        focused_sumo.make_environment = original

    return restore


def finish(case: dict, case_dir: Path, rows: list[dict], prefix: str) -> dict:
    """다시 돌린 충돌 기록(case_dir/run/collisions.xml)에서 같은 시각·차로·두 차의 충돌을 찾고, 매초 기록과 case.json을 쓴다."""
    from focused_sumo import parse_collisions
    again = parse_collisions(case_dir / "run" / "collisions.xml")
    same = [e for e in again if float(e["time"]) == case["time"] and e["lane"] == case["lane"]
            and e["collider"] == case["collider"] and e["victim"] == case["victim"]]
    case_dir.mkdir(parents=True, exist_ok=True)
    with open(case_dir / "trajectory.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["time"])
        w.writeheader()
        w.writerows(rows)
    result = {**case, "reproduced": bool(same), "reproduced_event": same[0] if same else None,
              "collisions_until_window_end": len(again), "window_s": [case["time"] - BEFORE_S, case["time"] + AFTER_S],
              "junction_prefix": prefix, "summary": summarize(rows, case, prefix)}
    (case_dir / "case.json").write_text(json.dumps(result, ensure_ascii=False, indent=1))
    return result


def rerun(case: dict, out_root: str) -> dict:
    """단독 모형 평가(focused_eval.run_day)를 같은 날짜·정책·도착 방식·평가 seed(DQN은 같은 모델)로 다시 돌려 기록한다."""
    import focused_eval

    fp = focused_eval.evaluation_fingerprint()
    if case.get("evaluation_fingerprint") not in (None, fp):
        raise SystemExit(f"{case['name']}: 충돌이 난 실행의 평가 지문 {case['evaluation_fingerprint']}이 지금 코드 {fp}과 다릅니다.")
    case_dir = Path(out_root) / case["name"]
    prefix = junction_prefix(case["lane"]) or f":{TARGET_TLS}_"
    job = {"date": case["date"], "policy": case["policy"], "eval_seed": case["eval_seed"], "arrivals": case["arrivals"],
           "run_dir": str(case_dir / "run"), "fingerprint": fp}
    if case["policy"] == "dqn":
        job.update({k: case[k] for k in DQN_KEYS})
    rows: list[dict] = []
    restore = install_recorder(case, prefix, rows)
    try:
        focused_eval.run_day(job)
        raise RuntimeError(f"{case['name']}: 충돌 시각 전에 실행이 끝났습니다")
    except StopAfterWindow:
        pass
    finally:
        restore()
    return finish(case, case_dir, rows, prefix)


def report(r: dict) -> str:
    """다시 돌린 결과 한 건을 몇 줄로."""
    s = r["summary"]
    lines = [f"- {r['name']} ({r['lane']}): {'재현됨' if r['reproduced'] else '재현 안 됨'}, "
             f"기록 구간까지 충돌 {r['collisions_until_window_end']}건"]
    for role, label in (("victim", "부딪힌 차"), ("collider", "부딪친 차")):
        v = s[role]
        ahead = ", ".join(f"{a['edge']} {a['halting']}/{a['vehicles']}대 멈춤" for a in v["route_ahead_at_collision"] or [])
        lines.append(f"    {label} {v['vehicle']}({v['type']}): 교차로 진입 {v['entered_junction_s']}초, 진입 속도 "
                     f"{v['speed_at_entry']}m/s, 충돌까지 {v['seconds_inside_before_collision']}초, 충돌 전 멈춤 "
                     f"{v['stopped_s_before_collision']}초 | 다음 도로: {ahead or '없음'}")
    return "\n".join(lines)


def main() -> None:
    from focused_eval import CACHE, dqn_template_network, evaluation_fingerprint
    from focused_scenario import day_route_file

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eval-dir", help="평가 지문 폴더(예: .../eval_v5/955fd7d7b85f) — 그 안의 확인 안 된 충돌을 모두")
    p.add_argument("--run", nargs="+", help="평가 실행 폴더(들) — 그 실행들의 확인 안 된 충돌만")
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=3)
    args = p.parse_args()

    fp = evaluation_fingerprint()
    if bool(args.eval_dir) == bool(args.run):
        raise SystemExit("--eval-dir와 --run 가운데 하나만 주세요.")
    if args.eval_dir:
        eval_dir = Path(args.eval_dir)
        if eval_dir.name != fp:
            raise SystemExit(f"평가 폴더 지문 {eval_dir.name}이 지금 코드의 평가 지문 {fp}과 다릅니다. "
                             "같은 코드로 돌려야 같은 충돌이 다시 납니다.")
        cases = find_cases(eval_dir)
    else:
        cases = [c for r in args.run for c in cases_in_run(Path(r))]
        wrong = [c["name"] for c in cases if c["evaluation_fingerprint"] != fp]
        if wrong:
            raise SystemExit(f"평가 지문이 지금 코드 {fp}과 다른 실행의 충돌입니다: {wrong}")
    todo = [c for c in cases if not (Path(args.out) / c["name"] / "case.json").exists()]
    print(f"확인 안 된 충돌 {len(cases)}건(새로 {len(todo)}건): " + ", ".join(c["name"] for c in cases), flush=True)
    for c in todo:                      # 차량 파일은 먼저 순서대로 만든다(동시 생성 충돌 방지)
        day_route_file(c["date"], CACHE / "routes")
        if c["arrivals"] == "platoon":
            from focused_platoon import platoon_route_file
            platoon_route_file(c["date"], CACHE / "routes")
    dqn_template_network(CACHE)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(rerun, c, args.out) for c in todo]):
            print(report(future.result()), flush=True)
    print(f"저장: {args.out}")


if __name__ == "__main__":
    main()
