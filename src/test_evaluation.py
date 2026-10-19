"""시험 4일 평가 — 설정을 모두 정한 뒤 한 번만 (10-18 사용자 확정: 주 지표 = 석천사거리 구간 지연·300초 넘게 늦은 차).

시험 날짜(config/focused_split.json의 test: 08-10 평일, 10-28 평일, 08-21 주말, 09-10 추석)는 학습·검증·규칙 고르기에 한 번도
쓰지 않았다. 다른 도구(focused_conflicts·network_windows·network_eval)는 시험 날짜를 막아 두었고, 이 파일만 확정한 설정을 확인한 뒤
같은 함수로 시험 날짜를 돌린다. 평가 코드를 고치지 않으므로 평가 지문이 검증과 같다(같은 채점).

- single(석천사거리 단독 모형, 검증 14번 칸과 같은 방법): 전환 규칙 하나(FOCUSED_RULESET)마다 고정신호(원자료)·보정 고정신호
  (규칙 none은 뺌)·단순 제어·DQN 5개, 무작위 도착(본 결과)과 동쪽 무리 도착(민감도), 평가 seed 42, 하루 연속.
  결과: <out>/<평가 지문>/<날짜>/<실행>/metrics.json, 표는 focused_v6_report.py --root <out> --dates <시험 4일>.
- network12(12개 지도, 검증 17·18번 칸과 같은 방법): 수요 만들기 → 고정신호 90분 조각 점검(무너진 조각 뺌) → 정책마다 조각당
  5번(고정·보정·단순 제어는 평가 seed 42~46, DQN은 학습 seed s → 평가 seed s) → network_eval_seeds.py 표.
  결과: <out>/<평가 지문>/windows/<날짜>_<시>/…, 조각 점검 <out>/<평가 지문>/screen_test/, 표 report_windows_seeds_test.md.
- 확정한 설정(CONFIRMED)과 코드·모델이 하나라도 다르면 멈춘다. 끊기면 같은 명령을 다시 돌리면 끝난 실행은 건너뛴다.
  설정을 바꿔 다시 돌리는 것은 안 된다(시험 날짜가 더는 '처음 보는 날'이 아니게 된다).

사용(Colab 8번·19번 칸):
  FOCUSED_RULESET=pet1 FOCUSED_NORTH_EXT=240 python src/test_evaluation.py single --models <seed 폴더 5개> --out <Drive>/v6_test --confirm-once
  FOCUSED_RULESET=pet1 FOCUSED_NORTH_EXT=240 python src/test_evaluation.py network12 --models <seed 폴더 5개> --out <Drive>/eval_12map --confirm-once
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from argparse import Namespace
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
SEEDS = [42, 43, 44, 45, 46]
# 10-18 확정(진행 기록 67·74·76·77절, docs/실험설정_v6.md). 모델 지문 = dqn_204820.zip sha256 앞 10자리.
CONFIRMED = {
    "single": {
        "pet1": {"environment": "b20888514ea3", "evaluation": "2030250938d6",
                 "models": ["8c399acc87", "2d7dd7ccdf", "8e9f284c7f", "955e9f589b", "e154c0be64"]},
        "pet1_5": {"environment": "f31f84aa9924", "evaluation": "6cf26052e21f",
                   "models": ["84f15dcd91", "508c3a9428", "f16d0c5742", "46e0858e61", "745b48316b"]},
        "none": {"environment": "84ba298fb9ba", "evaluation": "c0d443fb57ed",
                 "models": ["dcc5437749", "4c90b08a26", "80ec98e76d", "9ef6b9cc03", "dbcb1a8d5a"]},
    },
    "network12": {
        "pet1": {"environment": "b20888514ea3", "network12": "711875f6c939", "evaluation": "9d3b7504064b",
                 "models": ["0ebf324357", "5c3445496f", "f249eef136", "e2d838c222", "737f944b8b"]},
    },
    "timesteps": 100_000,
    "target_update_interval": 5_000,
}


def test_dates() -> list[str]:
    return json.loads((PROJECT / "config/focused_split.json").read_text())["test"]


def check_models(models: list[str], want_sha: list[str], extra: dict) -> None:
    """모델 5개가 확정한 것(학습 seed 42~46, 모델 지문, 학습 단계, 목표망 간격, 환경 지문)과 같은지."""
    from focused_eval import model_identity
    got = {}
    for m in models:
        ident = model_identity(m)
        cfg = json.loads((Path(m) / "training_config.json").read_text())
        bad = {k: (cfg.get(k), v) for k, v in extra.items() if cfg.get(k) != v}
        if cfg["timesteps"] != CONFIRMED["timesteps"]:
            bad["timesteps"] = (cfg["timesteps"], CONFIRMED["timesteps"])
        if cfg["dqn"]["target_update_interval"] != CONFIRMED["target_update_interval"]:
            bad["target_update_interval"] = (cfg["dqn"]["target_update_interval"], CONFIRMED["target_update_interval"])
        if bad:
            raise SystemExit(f"{m}: 확정한 설정과 다름 {bad}")
        got[ident["train_seed"]] = ident["model_sha256"][:10]
    if list(got) != SEEDS or [got[s] for s in SEEDS] != want_sha:
        raise SystemExit(f"모델(순서대로) {got}이 확정한 {dict(zip(SEEDS, want_sha))}와 다릅니다.")


def single(args) -> None:
    from focused_conflicts import measure
    from focused_data import environment_fingerprint, variant
    from focused_eval import evaluation_fingerprint
    rule = variant()["rule_set"]
    want = CONFIRMED["single"].get(rule)
    if want is None or os.environ.get("FOCUSED_NORTH_EXT") != "240":
        raise SystemExit(f"전환 규칙 {rule}·북쪽 연장 {os.environ.get('FOCUSED_NORTH_EXT')}는 확정한 설정이 아닙니다.")
    got = {"environment": environment_fingerprint(), "evaluation": evaluation_fingerprint()}
    if got != {k: want[k] for k in got}:
        raise SystemExit(f"{rule}: 코드 지문 {got}이 확정한 값과 다릅니다.")
    check_models(args.models, want["models"], {"environment_fingerprint": want["environment"]})
    policies = ["fixed", "greedy", "dqn"] if rule == "none" else ["fixed", "fixedc", "greedy", "dqn"]
    print(f"[단독 모형 {rule}] 코드와 모델 5개가 확정한 설정과 같다 → 시험 4일 {test_dates()} 평가", flush=True)
    if args.check_only:
        return
    measure(Namespace(dates=test_dates(), policies=policies, models=args.models, arrivals=["random", "platoon"],
                      out=args.out, tmp=args.tmp, workers=args.workers, until=None))


def _screen(job):
    import network_windows as nw
    date, hour, uturn, background, root = job
    return nw.screen_one((date, hour, uturn, background), root=Path(root))


def _run(job):
    import network_eval as ne
    out = Path(job["run_dir"]) / "metrics.json"
    return json.loads(out.read_text()) if out.exists() else ne.run_window(job)


def network12(args) -> None:
    import network_day
    import network_eval as ne
    import network_rl as nr
    import network_windows as nw
    from focused_data import environment_fingerprint, variant
    want = CONFIRMED["network12"].get(variant()["rule_set"])
    if want is None or os.environ.get("FOCUSED_NORTH_EXT") != "240":
        raise SystemExit("12개 지도 시험 평가는 pet1·북쪽 연장 240m로만 확정했습니다.")
    got = {"environment": environment_fingerprint(), "network12": nr._code_fingerprint(), "evaluation": ne.evaluation_fingerprint()}
    if got != {k: want[k] for k in got}:
        raise SystemExit(f"코드 지문 {got}이 확정한 값과 다릅니다.")
    check_models(args.models, want["models"], {"environment_fingerprint": want["environment"],
                                               "network12_fingerprint": want["network12"], "map": "network12"})
    dates = test_dates()
    final = nr.final_environment()
    root = Path(args.out) / got["evaluation"]
    workers = args.workers or max(1, min(16, os.cpu_count() or 1, int(_memory() / 3e9)))
    print(f"[12개 지도] 코드와 모델 5개가 확정한 설정과 같다 → 시험 4일 {dates}, 동시에 {workers}개", flush=True)
    if args.check_only:
        return
    for d in dates:                                   # 1) 수요(하루씩 차례로)
        t0 = time.time()
        demand = network_day.demand_for(d, final["uturn"], -1, final["background"], final["depart_fix"])
        print(f"  수요 {d}: {json.loads((demand / 'ledger.json').read_text())['meta']['vehicles']:,}대 ({time.time() - t0:.0f}초)", flush=True)
    nr.template_network_12()
    screen_root = root / "screen_test"                # 2) 고정신호 조각 점검(검증과 같은 기준)
    jobs = [(d, h, final["uturn"], final["background"], str(screen_root)) for d in dates for h in nw.HOURS]
    with ProcessPoolExecutor(workers) as ex:
        rows = [f.result() for f in as_completed([ex.submit(_screen, j) for j in jobs])]
    failed = sorted((r["date"], r["hour"]) for r in rows if nw.is_failed(r["end_pending"], r["jammed_at_end"]))
    windows = [(d, h) for d in dates for h in nw.HOURS if (d, h) not in failed]
    (root / "test_windows.json").write_text(json.dumps({"dates": dates, "screened": len(rows), "failed": failed,
                                                        "windows": windows}, indent=1))
    print(f"  조각 점검 {len(rows)}개, 고정신호로 무너져 뺀 조각 {failed or '없음'}", flush=True)
    runs = []                                         # 3) 정책마다 조각당 5번
    for s, model in zip(SEEDS, args.models):
        runs += ne.window_runs(root, windows, ["fixed", "fixedc", "greedy"], [], s)
        runs += ne.window_runs(root, windows, ["dqn"], [model], s)
    runs = [{**j, "fingerprint": got["evaluation"]} for j in runs]
    todo = [j for j in runs if not (Path(j["run_dir"]) / "metrics.json").exists()]
    print(f"  실행 {len(runs)}개 중 새로 {len(todo)}개", flush=True)
    with ProcessPoolExecutor(workers) as ex:
        for n, f in enumerate(as_completed([ex.submit(_run, j) for j in todo]), 1):
            r = f.result()
            if n % 50 == 0 or n == len(todo):
                print(f"  {n}/{len(todo)}", flush=True)
    subprocess.run([sys.executable, str(PROJECT / "src/network_eval_seeds.py"), "--root", str(root), "--dates", *dates,
                    "--models", *args.models, "--out", str(root / "report_windows_seeds_test.md")], check=True)


def _memory() -> float:
    try:
        import psutil
        return psutil.virtual_memory().total
    except ImportError:
        return 16e9


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("single", "network12"):
        p = sub.add_parser(name)
        p.add_argument("--models", nargs=5, required=True, help="학습 seed 42~46 모델 폴더(순서대로)")
        p.add_argument("--out", required=True)
        p.add_argument("--workers", type=int, default=0, help="0이면 CPU·메모리에 맞춰 자동")
        p.add_argument("--confirm-once", action="store_true", help="설정을 모두 정했고 시험 날짜를 한 번만 돌린다는 확인")
        p.add_argument("--check-only", action="store_true", help="설정·모델 확인만 하고 아무것도 돌리지 않는다")
        if name == "single":
            p.add_argument("--tmp", default="/content/test_tmp", help="기록을 잠시 둘 곳(분석 뒤 지움)")
    args = ap.parse_args()
    if not (args.confirm_once or args.check_only):
        raise SystemExit("시험 날짜는 설정을 모두 정한 뒤 한 번만 돌립니다. 그럴 때만 --confirm-once를 붙이세요.")
    sys.path.insert(0, str(PROJECT / "src"))
    (single if args.cmd == "single" else network12)(args)


if __name__ == "__main__":
    main()
