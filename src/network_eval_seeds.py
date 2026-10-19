"""12개 지도 조각 평가를 평가 seed 여러 개로 모아 보고한다 — 모든 정책을 조각마다 같은 횟수로 돌린 비교 (10-18, 진행 기록 75절).

왜: 09-28 08시처럼 무너지기 직전인 조각은 아주 작은 차이(평가 seed, 맥·Colab 계산 차이)로 결과가 뒤집힌다. 그런데
network_eval 보고는 고정신호·단순 제어를 평가 seed 42로 한 번, DQN을 학습 seed 5개로 다섯 번 돌린다. 그러면 우연이 DQN 쪽에만
더 많이 들어간다. 같은 채점을 위해 모든 정책을 조각마다 5번 돌린다.
- 고정신호(원자료·보정)·단순 제어: 평가 seed 42~46.
- DQN: 학습 seed s 모델을 평가 seed s로(42→42, …, 46→46). 모델마다 한 번이라 정책마다 실행 수가 같다.
- network_eval.py를 고치지 않는다(평가 지문 그대로, 이미 돌린 평가 seed 42 결과를 다시 쓴다). 결과 폴더
  <out>/<평가 지문>/windows/<날짜>_<시>/<정책>_eval<seed>/metrics.json을 읽기만 한다.
- 흔들리는 조각(민감도 표에서 뺌): 고정신호 원자료 5번의 구간 지연 최댓값 − 최솟값 > 20초, 또는 한 번이라도 끝날 때
  석천사거리 앞에서 기다리는 차 > 100대. 고정신호만으로 정하므로 DQN 결과는 쓰지 않는다(10-18 실행 전에 정함).

사용(Colab 18번 칸이 돌린 뒤):
  .venv/bin/python src/network_eval_seeds.py --root <eval_12map/평가 지문> --dates 20220928 20220813 --models <seed 폴더 5개>
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

SEEDS = [42, 43, 44, 45, 46]
BASE = ("fixed", "fixedc", "greedy")
LABELS = {"fixed": "고정신호(원자료 그대로)", "fixedc": "고정신호(전적색 보정)", "greedy": "단순 제어(학습 없음)", "dqn": "DQN"}
UNSTABLE_RANGE_S, UNSTABLE_PENDING = 20.0, 100


def load(root: Path, dates: list[str], models: list[str]) -> tuple[dict, list[str]]:
    """(정책, 조각) → [metrics 5개]. DQN은 학습 seed s를 평가 seed s로."""
    from focused_eval import model_identity
    dqn = {}
    for m in models:
        ident = model_identity(m)
        dqn[ident["train_seed"]] = f"dqn_seed{ident['train_seed']}_{ident['model_sha256'][:10]}"
    if models and sorted(dqn) != SEEDS:
        raise SystemExit(f"DQN 모델의 학습 seed {sorted(dqn)}가 {SEEDS}와 다릅니다.")
    runs, missing = defaultdict(list), []
    for wdir in sorted((root / "windows").iterdir()):
        if wdir.name[:8] not in dates:
            continue
        for policy in BASE + (("dqn",) if models else ()):
            for s in SEEDS:
                tag = dqn[s] if policy == "dqn" else policy
                p = wdir / f"{tag}_eval{s}" / "metrics.json"
                if p.exists():
                    runs[(policy, wdir.name)].append(json.loads(p.read_text()))
                else:
                    missing.append(str(p.relative_to(root)))
    return runs, missing


def unstable_windows(runs: dict) -> dict:
    out = {}
    for (policy, w), rs in runs.items():
        if policy != "fixed" or len(rs) < len(SEEDS):
            continue
        d = [r["window_204820"]["mean_delay_s"] for r in rs]
        p = max(r["window_204820"]["pending_at_end"] for r in rs)
        if max(d) - min(d) > UNSTABLE_RANGE_S or p > UNSTABLE_PENDING:
            out[w] = {"delay_min": round(min(d), 1), "delay_max": round(max(d), 1), "pending_max": p}
    return out


def table(runs: dict, windows: list[str]) -> list[str]:
    lines = ["| 정책 | 구간 지연(초, 5번 평균 ± 표준편차) | 차(1번 평균) | 300초 넘게 늦은 차(1번 평균) | 못 지난 차(1번 평균) | "
             "석천사거리 안 충돌(5번 합) | 충돌 전체(5번 합) | 보행 한도 초과(5번 합) |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for policy in BASE + ("dqn",):
        per_seed = []
        for i in range(len(SEEDS)):
            rs = [runs[(policy, w)][i] for w in windows if len(runs.get((policy, w), [])) == len(SEEDS)]
            if len(rs) != len(windows):
                break
            n = sum(r["window_204820"]["vehicles"] for r in rs)
            per_seed.append({"delay": sum(r["window_204820"]["mean_delay_s"] * r["window_204820"]["vehicles"] for r in rs) / n,
                             "vehicles": n, "long": sum(r["window_204820"]["long_delays"] for r in rs),
                             "unfinished": sum(r["window_204820"]["unfinished"] for r in rs),
                             "col_in": sum(r["collisions_in_204820"] for r in rs), "col_all": sum(r["collisions_total"] for r in rs),
                             "ped": sum(c["waits_over_limit"] for r in rs for c in r["pedestrian"].values())})
        if len(per_seed) != len(SEEDS):
            continue
        m = {k: statistics.fmean(p[k] for p in per_seed) for k in per_seed[0]}
        s = {k: sum(p[k] for p in per_seed) for k in ("col_in", "col_all", "ped")}
        lines.append(f"| {LABELS[policy]} | {m['delay']:.1f} ± {statistics.stdev(p['delay'] for p in per_seed):.1f} | "
                     f"{m['vehicles']:,.0f} | {m['long']:.0f} | {m['unfinished']:.1f} | {s['col_in']} | {s['col_all']} | {s['ped']} |")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", required=True, help="eval_12map/<평가 지문>")
    ap.add_argument("--dates", nargs="+", required=True)
    ap.add_argument("--models", nargs="*", default=[])
    ap.add_argument("--out")
    args = ap.parse_args()
    root = Path(args.root)
    runs, missing = load(root, args.dates, args.models)
    if missing:
        print(f"빠진 실행 {len(missing)}개(앞 10개): {missing[:10]}")
        raise SystemExit("모든 정책 × 조각 × 평가 seed 5개가 있어야 보고합니다. 평가 칸을 다시 돌려 빠진 것만 채우세요.")
    windows = sorted({w for _, w in runs})
    bad = unstable_windows(runs)
    stable = [w for w in windows if w not in bad]
    lines = [f"# 12개 지도 조각 평가 — 평가 seed 5개로 같은 횟수 비교 (날짜 {', '.join(args.dates)}, 평가 지문 {root.name})", "",
             "모든 정책을 조각마다 5번 돌렸다. 고정신호·보정·단순 제어는 평가 seed 42~46, DQN은 학습 seed s 모델을 평가 seed s로. "
             "표의 지연은 5번 각각 모든 조각의 차를 합친 값의 평균 ± 표준편차, 차·300초 초과·못 지난 차는 1번 평균이다.", "",
             f"## 모든 조각({len(windows)}개, 주 결과)", ""] + table(runs, windows)
    lines += ["", f"## 흔들리는 조각을 뺀 표({len(stable)}개, 민감도)", "",
              f"흔들리는 조각 = 고정신호 원자료 5번의 지연 최댓값 − 최솟값 > {UNSTABLE_RANGE_S:g}초, 또는 끝날 때 석천사거리 앞 대기 > "
              f"{UNSTABLE_PENDING}대(고정신호만으로 정함). " + ("; ".join(f"{w} {v['delay_min']}~{v['delay_max']}초, 대기 최대 {v['pending_max']}"
                                                         for w, v in bad.items()) or "없음"), ""] + table(runs, stable)
    text = "\n".join(lines) + "\n"
    print(text)
    target = Path(args.out) if args.out else root / ("report_windows_seeds_" + "_".join(args.dates) + ".md")
    target.write_text(text, encoding="utf-8")
    print("저장:", target)


if __name__ == "__main__":
    main()
