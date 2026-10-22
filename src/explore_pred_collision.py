"""[추가 탐색] 예측 제어(PRED) 실행에서 난 확인 안 된 충돌을 같은 조건으로 다시 돌려, 충돌 전후 움직임을 기록한다 — 원인 확인용(100절).

- focused_collision.rerun을 그대로 쓴다(평가 run_day를 다시 돌리며 충돌 BEFORE_S초 전부터 두 차와 그 교차로 안 차를 1초마다
  기록하고, 같은 시각·차로·두 차의 충돌이 다시 나면 '재현됨').
- 다른 것은 하나다. 원래 실행은 단순 제어 자리에 예측 제어를 끼워 돌렸으므로(explore_pred.install), 다시 돌리기 전에 같은
  변형·내다보는 길이로 끼운다. 원래 실행의 metrics.json에 남은 예측 제어 코드 지문이 지금 explore_pred.py와 다르면 멈춘다
  (같은 제어가 아니면 같은 충돌이 다시 날 이유가 없다). 평가 지문이 다르면 역시 멈춘다.
- 값을 묻기만 하고 바꾸지 않는다. 채점 규칙(config/collision_exceptions.json)·평가 코드·잠근 환경 파일은 고치지 않는다.
  결과는 평가 폴더가 아닌 --out에 쓴다. 결과를 낸 곳(Colab)에서 돌린다(맥과 Colab은 차 몇 대 단위로 다를 수 있다).

사용(Colab 31번 칸):
  FOCUSED_RULESET=pet1 FOCUSED_NORTH_EXT=240 python src/explore_pred_collision.py \
      --run <Drive>/explore_pred/single/2030250938d6/20220813/pred_p0_eval42_platoon --out <Drive>/explore_pred/collision_check
결과: <out>/<날짜>_pred_<변형>[_platoon]_<충돌 시각>/ 아래 case.json(재현 여부·요약), trajectory.csv(매초 기록)
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


def build_cases(run_dir: Path) -> list[dict]:
    """예측 제어 실행 하나의 확인 안 된 충돌을 focused_collision.rerun이 받는 모양으로(건마다 하나)."""
    from focused_eval import evaluation_fingerprint
    m = json.loads((Path(run_dir) / "metrics.json").read_text())
    pred = m.get("pred")
    if not pred:
        raise SystemExit(f"{run_dir}: 예측 제어 실행이 아닙니다(metrics.json에 pred가 없음).")
    if m.get("evaluation_fingerprint") != evaluation_fingerprint():
        raise SystemExit(f"{run_dir}: 평가 지문 {m.get('evaluation_fingerprint')}이 지금 코드 {evaluation_fingerprint()}과 다릅니다.")
    arrivals = m.get("arrivals", "random")
    cases = []
    for event in m.get("collisions_unexplained", []):
        t = float(event["time"])
        cases.append({"date": m["date"], "policy": m["policy"], "arrivals": arrivals, "eval_seed": m["eval_seed"],
                      "time": t, "lane": event["lane"], "collider": event["collider"], "victim": event["victim"],
                      "event": event, "source_run": str(run_dir),
                      "pred_variant": pred["variant"], "pred_horizon_s": pred["horizon_s"], "pred_code": pred["code"],
                      "name": f"{m['date']}_pred_{pred['variant']}{'' if arrivals == 'random' else '_' + arrivals}_{int(t)}"})
    return cases


def rerun_case(case: dict, out: str) -> dict:
    """하위 실행에서: 같은 예측 제어를 끼우고 focused_collision.rerun을 그대로 부른다."""
    import explore_pred as ep
    import focused_collision
    if ep.code_fingerprint() != case["pred_code"]:
        raise SystemExit(f"{case['name']}: 예측 제어 코드 {ep.code_fingerprint()}가 충돌이 난 실행의 코드 {case['pred_code']}와 "
                         "다릅니다. 같은 코드로 돌려야 같은 충돌이 다시 납니다.")
    ep.install(case["pred_variant"], int(case["pred_horizon_s"]))
    return focused_collision.rerun(case, out)


def main() -> None:
    from focused_eval import CACHE, dqn_template_network
    from focused_scenario import day_route_file
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--run", nargs="+", required=True, help="확인 안 된 충돌이 난 예측 제어 실행 폴더(들)")
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=2)
    args = p.parse_args()
    cases = [c for r in args.run for c in build_cases(Path(r))]
    if not cases:
        raise SystemExit("확인 안 된 충돌이 없습니다.")
    print(f"확인 안 된 충돌 {len(cases)}건: " + ", ".join(f"{c['name']}({c['lane']})" for c in cases), flush=True)
    for c in cases:                     # 차량 파일은 먼저 순서대로 만든다(동시 생성 충돌 방지)
        day_route_file(c["date"], CACHE / "routes")
        if c["arrivals"] == "platoon":
            from focused_platoon import platoon_route_file
            platoon_route_file(c["date"], CACHE / "routes")
    dqn_template_network(CACHE)
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for f in as_completed([pool.submit(rerun_case, c, args.out) for c in cases]):
            r = f.result()
            results.append(r)
            s = r["summary"]
            print(f"- {r['name']} ({r['lane']}): {'재현됨' if r['reproduced'] else '재현 안 됨'}, "
                  f"기록 구간까지 충돌 {r['collisions_until_window_end']}건", flush=True)
            for role, label in (("victim", "부딪힌 차"), ("collider", "부딪친 차")):
                v = s[role]
                print(f"    {label} {v['vehicle']}({v['type']}): 교차로 진입 {v['entered_junction_s']}초, 진입 속도 "
                      f"{v['speed_at_entry']}m/s, 충돌까지 {v['seconds_inside_before_collision']}초, 지난 차로 {v['lanes']}", flush=True)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "summary.json").write_text(json.dumps(
        [{k: r[k] for k in ("name", "lane", "reproduced", "collisions_until_window_end", "window_s", "summary")} for r in results],
        ensure_ascii=False, indent=1))
    print(f"저장: {args.out}")


if __name__ == "__main__":
    main()
