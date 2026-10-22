"""12개 지도 조각 평가에서 난 석천사거리 안 충돌을 같은 조건으로 다시 돌려, 충돌 전후 움직임을 기록한다 — 원인 확인용(105절).

- 조각 평가(network_eval.run_window)를 그대로 다시 돌린다. 날짜·시·정책·평가 seed(DQN은 같은 모델 폴더·모델 파일 SHA-256)는
  원래 실행의 metrics.json에서 읽는다. 모델 파일이 그 뒤 바뀌었으면 평가가 멈춘다.
- 기록은 단독 모형 재현(focused_collision)과 같은 장치를 쓴다. SUMO를 1초 진행할 때마다 충돌 BEFORE_S초 전부터 AFTER_S초
  뒤까지 두 차와 그 교차로 안 차의 차로·위치·속도·교차로 상대 차, 석천사거리 신호, 두 차 경로상 다음 도로의 막힘을 적는다.
  12개 지도는 SUMO 연결을 LocalView로 감싸므로 감싸지 않은 연결로 읽는다. 값을 묻기만 하므로 차의 움직임은 평가와 같다.
- 다시 돌린 충돌 기록에 같은 시각·차로·두 차의 충돌이 있으면 '재현됨'이다. 원래 실행의 평가 지문이 지금 코드와 다르면 멈춘다.
- 채점·평가 코드·12개 지도·잠근 환경 파일은 고치지 않는다. 결과는 평가 폴더가 아닌 --out에 쓴다. case.json이 이미 있는 건은
  건너뛴다. 결과를 낸 곳(Colab)에서, 12개 지도 묶음(Colab 15번 칸)을 푼 뒤 돌린다.

사용(Colab 34번 칸):
  FOCUSED_RULESET=pet1 FOCUSED_NORTH_EXT=240 python src/network_collision.py \
      --run <Drive>/explore_teacher/network12/9d3b7504064b/windows/20220928_08/dqn_seed42_52fefd83af_eval42 \
      --out <Drive>/explore_teacher/collision_check
결과: <out>/<날짜>_<시>_<실행 폴더 이름>_<충돌 시각>/ 아래 case.json(재현 여부·요약), trajectory.csv(매초 기록)
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import focused_collision as fc
from focused_data import TARGET_TLS

JOB_KEYS = ("date", "hour", "policy", "eval_seed") + fc.DQN_KEYS        # run_window가 받는 값(DQN_KEYS에 학습 seed 포함)


def build_cases(run_dir: Path) -> list[dict]:
    """조각 평가 실행 하나의 석천사거리 안 충돌을 건마다 하나씩(focused_collision과 같은 모양 + 시·모델)."""
    from network_eval import evaluation_fingerprint
    run_dir = Path(run_dir)
    m = json.loads((run_dir / "metrics.json").read_text())
    if m.get("hour") is None:
        raise SystemExit(f"{run_dir}: 조각 평가 실행이 아닙니다(metrics.json에 hour가 없음).")
    if m.get("evaluation_fingerprint") != evaluation_fingerprint():
        raise SystemExit(f"{run_dir}: 평가 지문 {m.get('evaluation_fingerprint')}이 지금 코드 {evaluation_fingerprint()}과 다릅니다. "
                         "같은 코드로 돌려야 같은 충돌이 다시 납니다.")
    if m["policy"] == "dqn":
        missing = [k for k in fc.DQN_KEYS if m.get(k) is None]   # 흉내만 낸 모델은 학습량 0
        if missing:
            raise SystemExit(f"{run_dir}: DQN 실행인데 metrics.json에 {missing}이 없습니다(같은 모델로 다시 돌릴 수 없음).")
    cases = []
    for event in m.get("collisions_in_204820_events", []):
        t = float(event["time"])
        cases.append({**{k: m.get(k) for k in JOB_KEYS}, "arrivals": "window", "time": t, "lane": event["lane"],
                      "collider": event["collider"], "victim": event["victim"], "event": event, "source_run": str(run_dir),
                      "handover_s": m.get("handover_s"), "evaluation_fingerprint": m["evaluation_fingerprint"],
                      "name": f"{m['date']}_{m['hour']:02d}_{run_dir.name}_{int(t)}"})
    return cases


def rerun_case(case: dict, out: str) -> dict:
    """하위 실행에서: 기록 장치를 끼우고 network_eval.run_window를 그대로 부른다."""
    import network_eval
    fp = network_eval.evaluation_fingerprint()
    if case["evaluation_fingerprint"] != fp:
        raise SystemExit(f"{case['name']}: 평가 지문 {case['evaluation_fingerprint']}이 지금 코드 {fp}과 다릅니다.")
    case_dir = Path(out) / case["name"]
    prefix = fc.junction_prefix(case["lane"]) or f":{TARGET_TLS}_"
    job = {k: case[k] for k in JOB_KEYS if case.get(k) is not None}
    job.update(run_dir=str(case_dir / "run"), fingerprint=fp)
    rows: list[dict] = []
    restore = fc.install_recorder(case, prefix, rows)
    try:
        network_eval.run_window(job)
        raise RuntimeError(f"{case['name']}: 충돌 시각 전에 조각이 끝났습니다")
    except fc.StopAfterWindow:
        pass
    finally:
        restore()
    return fc.finish(case, case_dir, rows, prefix)


def main() -> None:
    import network_rl as nr
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--run", nargs="+", required=True, help="석천사거리 안 충돌이 난 조각 평가 실행 폴더(들)")
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=2)
    args = p.parse_args()
    cases = [c for r in args.run for c in build_cases(Path(r))]
    if not cases:
        raise SystemExit("석천사거리 안 충돌이 없습니다.")
    todo = [c for c in cases if not (Path(args.out) / c["name"] / "case.json").exists()]
    print(f"석천사거리 안 충돌 {len(cases)}건(새로 {len(todo)}건): " + ", ".join(c["name"] for c in cases), flush=True)
    nr.template_network_12()            # 하위 실행이 동시에 만들지 않게 먼저
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for f in as_completed([pool.submit(rerun_case, c, args.out) for c in todo]):
            print(fc.report(f.result()), flush=True)
    print(f"저장: {args.out}")


if __name__ == "__main__":
    main()
