"""[추가 탐색] 천장 재기(CEIL) — 예측 제어가 더 멀리 보거나(L) 앞으로 올 차를 정확히 알면(O) 얼마나 줄어드나.
진행 기록 109절 계획 그대로, 지문 밖. 학습 없음.

예측 제어(explore_pred, 코드 PRED_CODE)는 고치지 않고 그 위에 얹는다. 줄 모형·운영 규칙·끝값(남은 차 × 75초)·동점·상태 합치기는
그대로다.
- L(long): P0와 같은 정보(멈춘 차 + 진입 대기)로 계산 길이만 늘린다(240초, 점검에서 느리면 180초). 단독 모형·12개 지도.
- O(oracle): 단독 모형만. P1처럼 감지 구간 안 달리는 차는 제한속도로 정지선에 오고, 구간 밖에서 올 차는 최근 비율 대신
  차량 파일의 실제 출발 시각·경로로 넣는다(출발 시각 + 출발 도로에서 정지선까지 빈 도로 시간에 그 이동의 줄 끝에 섬).
  단독 모형은 모든 차가 구간 입구 도로에서 출발하므로 정할 수 있다. 계산 길이는 P1과 같은 120초.
- L의 계산 길이는 check가 판단 시간만 보고 정한다(109.1): 240초에서 판단 한 번 평균이 TIME_LIMIT_S 이하면 240초, 아니면
  180초로 다시 재고, 그것도 넘으면 L을 하지 않는다. 판단 ABORT_AFTER번 뒤 평균이 ABORT_MEAN_S를 넘으면 그 자리에서 '느림'.

평가 코드(focused_eval·network_eval·focused_conflicts)는 고치지 않는다. 실행할 때만 focused_eval.QueueFirst를 이 파일의
제어로 바꿔 끼우고 '단순 제어' 실행을 돌린다(평가 지문 그대로, 결과 폴더 이름만 ceil_<변형>). 판단 통계는 metrics.json의
"ceil"에 적는다.

사용(Colab 35번 칸):
  FOCUSED_RULESET=pet1 FOCUSED_NORTH_EXT=240 python src/explore_ceiling.py check --out <Drive>/explore_ceiling/check.json
  FOCUSED_RULESET=pet1 FOCUSED_NORTH_EXT=240 python src/explore_ceiling.py single --check <check.json> --out <Drive>/explore_ceiling/single
  FOCUSED_RULESET=pet1 FOCUSED_NORTH_EXT=240 python src/explore_ceiling.py network12 --check <check.json> --out <Drive>/explore_ceiling/network12
"""
from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import os
import sys
import tempfile
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

import explore_pred as ep  # noqa: E402

PRED_CODE = "9440fe1d3869"            # 100절 결과를 낸 예측 제어 코드
LONG_HORIZONS_S = (240, 180)          # 109.1: L 계산 길이 후보(앞에서부터)
ORACLE_HORIZON_S = 120                # 109.1: O는 P1과 같은 길이
TIME_LIMIT_S = 1.0                    # 109.1: 판단 한 번 평균
ABORT_AFTER, ABORT_MEAN_S = 40, 2.0   # 109.1: 점검을 일찍 멈추는 기준
VARIANTS = ("long", "oracle")
SINGLE_VARIANTS = ("long", "oracle")
NETWORK_VARIANTS = ("long",)


def code_fingerprint() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]


def _check_pred_code() -> None:
    if ep.code_fingerprint() != PRED_CODE:
        raise SystemExit(f"예측 제어 코드 {ep.code_fingerprint()}가 100절 결과를 낸 코드 {PRED_CODE}와 다릅니다. "
                         "같은 예측 제어 위에 얹어야 P0·P1과 비교할 수 있습니다.")


class TooSlow(Exception):
    """점검 중 판단이 너무 느려 그 계산 길이를 쓰지 않는다."""


class Movements(dict):
    """차 → 이동(단순 제어 QueueFirst.movements와 같음). O는 앞으로 출발할 차 목록(departures)도 함께 갖는다."""
    departures: list = []


def route_departures(route_file: Path, movement_of: dict) -> list[tuple[float, str, tuple]]:
    """차량 파일의 차마다 (출발 시각, 출발 도로, 이동), 출발 시각 순. 이동이 없는 차(신호 이동을 지나지 않음)는 뺀다."""
    root = ET.parse(route_file).getroot()
    first = {r.get("id"): r.get("edges").split()[0] for r in root.iter("route")}
    out = [(float(v.get("depart")), first[v.get("route")], movement_of.get(v.get("id"))) for v in root.iter("vehicle")]
    return sorted((d for d in out if d[2] is not None), key=lambda d: d[0])


def future_arrivals(departures: list, times: list[float], travel_from: dict[str, float], index_of: dict,
                    now: float, step: float, K: int) -> tuple[list[tuple[int, int]], int]:
    """출발 시각이 (now, now + K·step)인 차가 정지선에 닿는 단계와 이동 번호 목록, 출발 도로를 모르는 차 수.
    닿는 시각 = 출발 시각 + 출발 도로 입구에서 정지선까지 빈 도로 시간. 이미 출발 시각이 지났는데 못 들어온 차는 진입 대기(줄)다."""
    lo, hi = bisect.bisect_right(times, now), bisect.bisect_left(times, now + K * step)
    out, unknown = [], 0
    for depart, edge, movement in departures[lo:hi]:
        i = index_of.get(movement)
        if i is None:
            continue
        if edge not in travel_from:
            unknown += 1
            continue
        k = int((depart - now + travel_from[edge]) // step)
        if k < K:
            out.append((k, i))
    return out, unknown


def choose_long_horizon(measure) -> tuple[int | None, list[dict]]:
    """109.1: LONG_HORIZONS_S를 앞에서부터 재서(measure(h) → 판단 평균 초, 느려서 멈췄으면 None) 처음으로 TIME_LIMIT_S 이하인
    길이. 모두 넘으면 None(L을 하지 않음)."""
    tried = []
    for h in LONG_HORIZONS_S:
        mean = measure(h)
        tried.append({"horizon_s": h, "mean_seconds": mean, "ok": mean is not None and mean <= TIME_LIMIT_S})
        if tried[-1]["ok"]:
            return h, tried
    return None, tried


# ------------------------------------------------------------------ SUMO에 붙는 제어
def controller_class(variant: str, horizon_s: int, abort: bool = False):
    if variant not in VARIANTS:
        raise SystemExit(f"변형은 {VARIANTS} 가운데 하나: {variant}")
    _check_pred_code()
    base = ep.controller_class("p0" if variant == "long" else "p1", horizon_s)

    class CeilingControl(base):
        def __init__(self, movement_of, greens):
            super().__init__(movement_of, greens)
            type(self).stats.update(variant=variant, horizon_s=horizon_s, code=code_fingerprint(), pred_code=ep.code_fingerprint(),
                                    oracle_unknown_start=0)

        @staticmethod
        def movements(route_file: Path) -> Movements:
            m = Movements(base.movements(route_file))
            if variant == "oracle":
                m.departures = route_departures(route_file, m)
            return m

        def _setup(self, env) -> None:
            super()._setup(env)
            if variant != "oracle":
                return
            departures = getattr(self.movement_of, "departures", None)
            if not departures:
                raise SystemExit("O에는 차량 파일의 출발 기록이 필요합니다(이 제어의 movements로 만든 목록이 아님).")
            import sumolib
            net = sumolib.net.readNet(env._net)
            geo = self.geo
            self._travel_from = {e: sum(net.getEdge(x).getLength() / net.getEdge(x).getSpeed()
                                        for x in ep._path_to_stop(net, e, a, geo["offset"])) for e, a in geo["zone_of"].items()}
            self._departures, self._times = departures, [d[0] for d in departures]

        def _observe(self, env) -> None:
            if variant != "oracle":            # O는 최근 진입 비율을 쓰지 않는다
                super()._observe(env)

        def _outside(self, arrivals, sumo, now: float, step: float) -> None:
            found, unknown = future_arrivals(self._departures, self._times, self._travel_from, self.index, now, step, len(arrivals))
            for k, i in found:
                arrivals[k][i] += 1
            type(self).stats["oracle_unknown_start"] += unknown

        def __call__(self, env, allowed: list[int]) -> int:
            action = super().__call__(env, allowed)
            s = type(self).stats
            if abort and s["planned"] >= ABORT_AFTER and s["seconds"] / s["planned"] > ABORT_MEAN_S:
                raise TooSlow(f"판단 {s['planned']}번 평균 {s['seconds'] / s['planned']:.2f}초 > {ABORT_MEAN_S}초")
            return action

    CeilingControl.__name__ = f"CeilingControl_{variant}"
    return CeilingControl


def install(variant: str, horizon_s: int, abort: bool = False):
    """이 프로세스의 단순 제어 자리에 끼운다(평가 코드는 실행할 때 focused_eval.QueueFirst를 가져다 쓴다)."""
    import focused_eval
    cls = controller_class(variant, horizon_s, abort)
    focused_eval.QueueFirst = cls
    return cls


def _add_stats(metrics_file: Path, stats: dict) -> None:
    m = json.loads(metrics_file.read_text())
    m["ceil"] = {**stats, "mean_seconds": stats["seconds"] / max(1, stats["planned"])}
    tmp = metrics_file.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(m, ensure_ascii=False, indent=1))
    tmp.replace(metrics_file)


# ------------------------------------------------------------------ 실행
def tag(variant: str) -> str:
    return f"ceil_{variant}"


def _horizons(check_file: str) -> dict[str, int]:
    """통과한 점검에서 변형별 계산 길이. L을 잴 수 없었으면 L은 빠진다."""
    c = json.loads(Path(check_file).read_text())
    if not c.get("ok") or c.get("code") != code_fingerprint():
        raise SystemExit(f"{check_file}: 통과한 점검이 아니거나 지금 코드({code_fingerprint()})의 점검이 아닙니다({c.get('code')}).")
    out = {"oracle": ORACLE_HORIZON_S}
    if c.get("long_horizon_s"):
        out["long"] = int(c["long_horizon_s"])
    return out


def _single_job(job: dict, target: str, variant: str, horizon_s: int) -> str:
    """단독 모형 하루(PET 포함, focused_conflicts.measure_job 그대로)."""
    import focused_conflicts
    cls = install(variant, horizon_s)
    focused_conflicts.measure_job(job, target)
    _add_stats(Path(target) / "metrics.json", cls.stats)
    return f"{job['date']} {Path(target).name}"


def single_jobs(out: Path, tmp: Path, horizons: dict[str, int]) -> list[tuple[dict, str, str]]:
    """(job, Drive 폴더, 변형) — 검증 2일(무작위·무리)을 먼저, 그다음 학습 13일(무작위)."""
    from focused_eval import planned_runs
    split = json.loads((PROJECT / "config/focused_split.json").read_text())
    jobs = []
    for dates, arrivals_list in ((split["validation"], ("random", "platoon")), (split["train"], ("random",))):
        for arrivals in arrivals_list:
            for v in SINGLE_VARIANTS:
                if v not in horizons:
                    continue
                for j in planned_runs(tmp, dates, ["greedy"], [], 42, arrivals):
                    j["run_dir"] = j["run_dir"].replace("/greedy_eval", f"/{tag(v)}_eval")
                    jobs.append((j, str(out / Path(j["run_dir"]).relative_to(tmp)), v))
    return jobs


def single(args) -> None:
    import focused_conflicts as fc
    from focused_eval import CACHE, dqn_template_network, evaluation_fingerprint
    from focused_platoon import platoon_route_file
    from focused_scenario import day_route_file
    horizons = _horizons(args.check)
    fp = evaluation_fingerprint()
    drive, tmp = Path(args.out) / fp, Path(args.tmp) / fp
    planned = single_jobs(drive, tmp, horizons)
    jobs = [({**j, "fingerprint": fp}, t, v) for j, t, v in planned if not (Path(t) / "events.json.gz").exists()]
    print(f"천장 재기 단독 모형(계산 길이 {horizons}): 실행 {len(planned)}개 중 새로 {len(jobs)}개 → {drive}", flush=True)
    split = json.loads((PROJECT / "config/focused_split.json").read_text())
    for date in sorted({j["date"] for j, _, _ in jobs}):      # 차량 파일은 먼저 순서대로 만든다(동시 생성 충돌 방지)
        day_route_file(date, CACHE / "routes")
        if date in split["validation"]:
            platoon_route_file(date, CACHE / "routes")
    dqn_template_network(CACHE)
    with ProcessPoolExecutor(max_workers=args.workers or fc.auto_workers()) as pool:
        futures = [pool.submit(_single_job, j, t, v, horizons[v]) for j, t, v in jobs]
        for n, f in enumerate(as_completed(futures), 1):
            print(f"  {n}/{len(futures)} {f.result()}", flush=True)
    fc.summary_all(drive)


def _window_job(job: dict, variant: str, horizon_s: int) -> str:
    import network_eval
    cls = install(variant, horizon_s)
    network_eval.run_window(job)
    _add_stats(Path(job["run_dir"]) / "metrics.json", cls.stats)
    return f"{job['date']} {job['hour']:02d}시 {Path(job['run_dir']).name}"


def network12(args) -> None:
    import network_eval as ne
    import network_rl as nr
    import network_windows as nw
    horizons = _horizons(args.check)
    if "long" not in horizons:
        print("L을 잴 수 없었습니다(점검에서 계산이 너무 느림). 12개 지도는 돌리지 않습니다.", flush=True)
        return
    dates = json.loads((PROJECT / "config/focused_split.json").read_text())["validation"]
    fp = ne.evaluation_fingerprint()
    root = Path(args.out) / fp
    nr.template_network_12()
    jobs = []
    for v in NETWORK_VARIANTS:
        for s in ep.SEEDS:
            for j in ne.window_runs(root, nw.evaluation_windows(dates), ["greedy"], [], s):
                j["run_dir"] = j["run_dir"].replace(f"/greedy_eval{s}", f"/{tag(v)}_eval{s}")
                jobs.append(({**j, "fingerprint": fp}, v))
    todo = [(j, v) for j, v in jobs if not (Path(j["run_dir"]) / "metrics.json").exists()]
    print(f"천장 재기 12개 지도(계산 길이 {horizons['long']}초): 실행 {len(jobs)}개 중 새로 {len(todo)}개 → {root}", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_window_job, j, v, horizons[v]) for j, v in todo]
        for n, f in enumerate(as_completed(futures), 1):
            r = f.result()
            if n % 10 == 0 or n == len(futures):
                print(f"  {n}/{len(futures)} {r}", flush=True)


# ------------------------------------------------------------------ 점검
def _check_window(horizon_s: int, workdir: str) -> dict:
    """12개 지도 학습 날짜 조각 하나에서 L의 판단 시간(느리면 일찍 멈춤)."""
    import network_eval as ne
    import network_rl as nr
    fp = ne.evaluation_fingerprint()
    job = ne.window_runs(Path(workdir) / f"long{horizon_s}" / fp, [(ep.CHECK_DATE, ep.CHECK_HOUR)], ["greedy"], [], 42)[0]
    cls = install("long", horizon_s, abort=True)
    nr.template_network_12()
    out = {"map": "network12", "variant": "long", "horizon_s": horizon_s, "too_slow": False}
    try:
        r = ne.run_window({**job, "fingerprint": fp})
        out.update(changed_by_rules=r["guard"]["changed_by_rules"], delay_s=round(r["window_204820"]["mean_delay_s"], 2))
    except TooSlow as exc:
        out.update(too_slow=True, reason=str(exc))
    s = cls.stats
    out.update(planned=s["planned"], decisions=s["decisions"], mean_seconds=round(s["seconds"] / max(1, s["planned"]), 4),
               max_seconds=round(s["max_seconds"], 3), mean_nodes=round(s["nodes"] / max(1, s["planned"])),
               outside_allowed=s["outside_allowed"])
    return out


def _check_single(variant: str, horizon_s: int, workdir: str) -> dict:
    """단독 모형 코드 길 확인(학습 날짜 첫 2시간)."""
    import focused_conflicts as fc
    from focused_eval import CACHE, dqn_template_network, evaluation_fingerprint, planned_runs
    from focused_scenario import day_route_file
    day_route_file(ep.CHECK_DATE, CACHE / "routes")
    dqn_template_network(CACHE)
    job = planned_runs(Path(workdir) / f"single_{variant}", [ep.CHECK_DATE], ["greedy"], [], 42, "random")[0]
    cls = install(variant, horizon_s)
    fc.record_run({**job, "fingerprint": evaluation_fingerprint()}, until=7200)
    s = cls.stats
    return {"map": "single", "variant": variant, "horizon_s": horizon_s, "until_s": 7200, "planned": s["planned"],
            "decisions": s["decisions"], "mean_seconds": round(s["seconds"] / max(1, s["planned"]), 4),
            "outside_allowed": s["outside_allowed"], "oracle_unknown_start": s["oracle_unknown_start"]}


def check(args) -> None:
    """평가 전 점검(Colab, SUMO, 109.1). L 계산 길이는 판단 시간으로만 정하고, 단독 모형에서 O·L 코드 길을 확인한다."""
    workdir = tempfile.mkdtemp(prefix="ceil_check_")
    windows = []

    def measure(h):
        with ProcessPoolExecutor(max_workers=1) as pool:
            r = pool.submit(_check_window, h, workdir).result()
        windows.append(r)
        print(f"  L {h}초: 판단 {r['planned']}번 평균 {r['mean_seconds']}초{' (느려서 멈춤)' if r['too_slow'] else ''}", flush=True)
        return None if r["too_slow"] else r["mean_seconds"]

    long_h, tried = choose_long_horizon(measure)
    singles = [("oracle", ORACLE_HORIZON_S)] + ([("long", long_h)] if long_h else [])
    with ProcessPoolExecutor(max_workers=len(singles)) as pool:
        single_runs = list(pool.map(_check_single, [v for v, _ in singles], [h for _, h in singles], [workdir] * len(singles)))
    chosen = [w for w in windows if w["horizon_s"] == long_h]
    ok = all(r["planned"] > 0 and r["outside_allowed"] == 0 for r in single_runs + chosen) and \
        all(sum(w["changed_by_rules"].values()) == 0 for w in chosen) and \
        all(r["oracle_unknown_start"] == 0 for r in single_runs if r["variant"] == "oracle")
    report = {"code": code_fingerprint(), "pred_code": ep.code_fingerprint(), "long_horizon_s": long_h, "tried": tried,
              "oracle_horizon_s": ORACLE_HORIZON_S, "time_limit_s": TIME_LIMIT_S, "ok": ok,
              "rule": "109.1: L은 240초에서 판단 한 번 평균이 1.0초 이하면 240초, 아니면 180초, 그것도 넘으면 L을 하지 않는다(시간으로만 정함)",
              "check_scenario": f"{ep.CHECK_DATE} {ep.CHECK_HOUR:02d}시(학습 날짜)", "windows": windows, "single": single_runs}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(json.dumps(report, ensure_ascii=False, indent=1))
    if not ok:
        raise SystemExit("점검 실패: 평가하지 않는다.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("--out", required=True)
    s = sub.add_parser("single")
    s.add_argument("--check", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--tmp", default="/content/ceil_tmp", help="기록을 잠시 둘 곳(분석 뒤 지움)")
    s.add_argument("--workers", type=int, default=0, help="0이면 CPU·메모리에 맞춰 자동")
    n = sub.add_parser("network12")
    n.add_argument("--check", required=True)
    n.add_argument("--out", required=True)
    n.add_argument("--workers", type=int, default=max(1, min(16, (os.cpu_count() or 2) - 1)))
    args = ap.parse_args()
    {"check": check, "single": single, "network12": network12}[args.cmd](args)


if __name__ == "__main__":
    main()
