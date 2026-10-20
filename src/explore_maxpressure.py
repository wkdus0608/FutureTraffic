"""[추가 탐색] Max-Pressure 제어(Varaiya 2013 계열)를 학습 없는 비교 기준으로 더한다 — 지문 밖, 공식 결과(진행 기록 79절)와 섞지 않는다.

단순 제어(focused_eval.QueueFirst)와 같은 틀·같은 운영 규칙(SignalConstraints)이다. 다른 것은 점수 하나다.
- 단순 제어: 녹색이 통과시키는 이동마다 '기다리는 차' 수를 더한다.
- Max-Pressure: 이동마다 '기다리는 차 − 그 이동이 나가는 길에 멈춘 차'를 더한다(나가는 길이 막히면 그 녹색의 값이 줄어든다).
  나가는 길 = 이동의 진출 도로에서 곧게 이어지는 도로를 DOWNSTREAM_M(100m)까지. 멈춘 차 = SUMO 정지 기준(0.1m/s 미만).
- 정보 차이: Max-Pressure는 나가는 길의 줄을 본다. DQN과 단순 제어는 보지 못한다. 그래서 이 비교는 '같은 정보' 비교가 아니다
  (메모 baseline-same-information). 결과를 보고할 때 이 차이를 함께 적는다.
- 평가 코드(focused_eval·network_eval)를 고치지 않는다. 실행할 때만 focused_eval.QueueFirst를 이 클래스로 바꿔 끼우고
  '단순 제어' 실행을 돌린다(평가 지문 그대로, 결과 폴더만 따로). 시험 날짜는 쓰지 않는다(검증 2일만).

사용:
  FOCUSED_RULESET=pet1 FOCUSED_NORTH_EXT=240 .venv/bin/python src/explore_maxpressure.py single --out results/explore/maxpressure_single
  FOCUSED_RULESET=pet1 FOCUSED_NORTH_EXT=240 python src/explore_maxpressure.py network12 --out <Drive>/explore_12map   (Colab 20번 칸)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
DOWNSTREAM_M = 100.0
SEEDS = [42, 43, 44, 45, 46]

from focused_eval import QueueFirst  # noqa: E402


class MaxPressure(QueueFirst):
    """점수 = 녹색이 통과시키는 이동마다 (기다리는 차 − 나가는 길에 멈춘 차)의 합. 같은 점수면 지금 녹색, 그다음 작은 번호."""

    _downstream: dict[str, list[str]] | None = None

    def _chains(self, env) -> dict[str, list[str]]:
        import sumolib
        net = sumolib.net.readNet(env._net)
        out = {}
        for _, b in {m for served in self.served for m in served}:
            chain, length, edge = [b], net.getEdge(b).getLength(), net.getEdge(b)
            while length < DOWNSTREAM_M:
                nxt = [c.getTo() for conns in edge.getOutgoing().values() for c in conns if c.getDirection() == "s"]
                if not nxt:
                    break
                edge = nxt[0]
                chain.append(edge.getID())
                length += edge.getLength()
            out[b] = chain
        return out

    def __call__(self, env, allowed: list[int]) -> int:
        if len(allowed) == 1:
            return allowed[0]
        if self._downstream is None:
            self._downstream = self._chains(env)
        sumo = env.sumo
        current = env.traffic_signals["204820"].green_phase
        n = self.waiting(sumo)
        out = Counter({b: sum(sumo.edge.getLastStepHaltingNumber(e) for e in chain) for b, chain in self._downstream.items()})
        score = {p: sum(n[m] - out[m[1]] for m in self.served[p]) for p in allowed}
        return max(allowed, key=lambda p: (score[p], p == current, -p))


def _single_job(job):
    import focused_eval
    focused_eval.QueueFirst = MaxPressure
    out = Path(job["run_dir"]) / "metrics.json"
    return json.loads(out.read_text()) if out.exists() else focused_eval.run_day(job)


def _window_job(job):
    import focused_eval
    import network_eval
    focused_eval.QueueFirst = MaxPressure
    out = Path(job["run_dir"]) / "metrics.json"
    return json.loads(out.read_text()) if out.exists() else network_eval.run_window(job)


def single(args) -> None:
    from focused_eval import CACHE, dqn_template_network, evaluation_fingerprint, planned_runs
    from focused_scenario import day_route_file
    dates = json.loads((PROJECT / "config/focused_split.json").read_text())["validation"]
    fp = evaluation_fingerprint()
    root = Path(args.out) / fp
    jobs = []
    for arrivals in ("random", "platoon"):
        for j in planned_runs(root, dates, ["greedy"], [], 42, arrivals):
            j["run_dir"] = j["run_dir"].replace("greedy_", "maxpressure_")
            jobs.append({**j, "fingerprint": fp})
    for d in dates:
        day_route_file(d, CACHE / "routes")
        from focused_platoon import platoon_route_file
        platoon_route_file(d, CACHE / "routes")
    dqn_template_network(CACHE)
    with ProcessPoolExecutor(args.workers) as ex:
        for f in as_completed([ex.submit(_single_job, j) for j in jobs]):
            r = f.result()
            print(f"  {r['date']} {r.get('arrivals', 'random')}: 차량당 지연 {r['all']['mean_delay_s']:.2f}초, "
                  f"300초 초과 {r['all']['long_waits']}대", flush=True)


def network12(args) -> None:
    import network_eval as ne
    import network_windows as nw
    import network_rl as nr
    dates = json.loads((PROJECT / "config/focused_split.json").read_text())["validation"]
    fp = ne.evaluation_fingerprint()
    root = Path(args.out) / fp
    windows = nw.evaluation_windows(dates)
    nr.template_network_12()
    jobs = []
    for s in SEEDS:
        for j in ne.window_runs(root, windows, ["greedy"], [], s):
            j["run_dir"] = j["run_dir"].replace(f"/greedy_eval{s}", f"/maxpressure_eval{s}")
            jobs.append({**j, "fingerprint": fp})
    todo = [j for j in jobs if not (Path(j["run_dir"]) / "metrics.json").exists()]
    print(f"Max-Pressure 12개 지도: 실행 {len(jobs)}개 중 새로 {len(todo)}개 → {root}", flush=True)
    with ProcessPoolExecutor(args.workers) as ex:
        for n, f in enumerate(as_completed([ex.submit(_window_job, j) for j in todo]), 1):
            f.result()
            if n % 25 == 0 or n == len(todo):
                print(f"  {n}/{len(todo)}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=("single", "network12"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 1)))
    args = ap.parse_args()
    (single if args.cmd == "single" else network12)(args)


if __name__ == "__main__":
    main()
