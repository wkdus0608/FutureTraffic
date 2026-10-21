"""'얼마나 멀리' 고르기와 새 12일 점검·마지막 시험 — 지문 밖(진행 기록 89~91절).

- select: 91절 규칙으로 H1~H5 가운데 하나를 고른다(검증 2일, 무작위 도착, seed 5개). 새 날은 쓰지 않는다.
- daycheck: 새 12일마다 고정신호 하루 점검(focused_day.check_day, 19일 때와 같은 합격 기준). 떨어진 날은 뺀다(89절 규칙 5).
- final: 새 12일 가운데 점검을 통과한 날로 한 번만 비교한다(89절 규칙 7). --confirm-once가 있어야 돈다.
  고정신호(원자료·보정)·단순 제어·기존 DQN(v6 pet1)·고른 후보, 무작위 도착(주)·무리 도착(민감도), 평가 seed 42.
  평가는 검증·시험 때와 같은 focused_conflicts.measure(PET 포함)이고 평가 지문도 같다. 새 날짜 자료는 extra_env가 끼운다.

사용(Colab 25~27번 칸):
  python src/extra_eval.py select --eval-dir <explore_single/eval/평가 지문> --base-dir <v6_check/평가 지문> \
      --base-models <v6 pet1 5개> --cand H1 <모델 5개> --cand H2 … --out <selection.json>
  python src/extra_eval.py daycheck --out <Drive>/extra_days/daycheck
  python src/extra_eval.py final --selection <selection.json> --v6-models <5개> --out <Drive>/extra_days/final --confirm-once
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from argparse import Namespace
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
SEEDS = [42, 43, 44, 45, 46]
DELAY_S = 1.3
CANDIDATES = {"H1": (10, 0.99), "H2": (20, 0.99), "H3": (10, 0.995), "H4": (20, 0.995), "H5": (20, 0.999)}
V6_PET1 = ["8c399acc87", "2d7dd7ccdf", "8e9f284c7f", "955e9f589b", "e154c0be64"]


def _check_candidate(name: str, models: list[str]) -> None:
    from focused_eval import model_identity
    n_want, g_want = CANDIDATES[name]
    seeds = []
    for m in models:
        c = json.loads((Path(m) / "training_config.json").read_text())
        got = (c["dqn"].get("n_steps", 1), c["dqn"].get("gamma", 0.99), c.get("explore", {}).get("double_dqn"),
               c["timesteps"], c["dqn"]["target_update_interval"], c["environment_fingerprint"])
        want = (n_want, g_want, True, 100_000, 5_000, "b20888514ea3")
        if got != want:
            raise SystemExit(f"{name} 모델 {m}: 설정 {got}이 91절 {want}와 다릅니다.")
        seeds.append(model_identity(m)["train_seed"])
    if seeds != SEEDS:
        raise SystemExit(f"{name}: 모델은 학습 seed 42~46 순서여야 합니다({seeds}).")


def select(args) -> None:
    from explore_decide import runs_for, summary
    base = summary(runs_for(Path(args.base_dir), "dqn", args.base_models))
    rows = {}
    for name, *models in args.cand:
        _check_candidate(name, models)
        rows[name] = summary(runs_for(Path(args.eval_dir), "dqn", models))
    eligible = {k: v for k, v in rows.items()
                if v["long_waits"] <= base["long_waits"] and v["left_east"] <= base["left_east"] and v["left_south"] <= base["left_south"]}
    pool = eligible or rows
    best = min(pool, key=lambda k: pool[k]["mean_delay_s"])
    near = [k for k in pool if pool[k]["mean_delay_s"] - pool[best]["mean_delay_s"] < DELAY_S]
    chosen = max(near, key=lambda k: (CANDIDATES[k][1], CANDIDATES[k][0]))      # 할인율 큰 것, 같으면 n 큰 것
    result = {"rule": "91절", "baseline_v6": base, "candidates": rows, "eligible": sorted(eligible),
              "best_delay": best, "within_1.3s": sorted(near), "selected": chosen,
              "selected_setting": {"n_steps": CANDIDATES[chosen][0], "gamma": CANDIDATES[chosen][1], "double_dqn": True},
              "note": "자격 후보 없음 → 지연이 가장 짧은 후보(조건 어김 표시)" if not eligible else ""}
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=1))
    print(json.dumps(result, ensure_ascii=False, indent=1))
    print(f"고른 후보: {chosen} (n={CANDIDATES[chosen][0]}, 할인율 {CANDIDATES[chosen][1]}) → {args.out}")


def _daycheck_one(date: str, out: str) -> dict:
    import extra_env
    extra_env.activate()
    import focused_day
    return focused_day.check_day(date, Path(out))


def daycheck(args) -> None:
    import extra_env
    extra_env.activate()
    out = Path(args.out)
    workers = args.workers or max(1, min(len(extra_env.NEW_DATES), os.cpu_count() or 1))
    results = {}
    with ProcessPoolExecutor(workers) as ex:
        futs = {ex.submit(_daycheck_one, d, str(out)): d for d in extra_env.NEW_DATES}
        for f in as_completed(futs):
            r = f.result()
            results[futs[f]] = r
            print(f"  {futs[f]}: {'합격' if r.get('passed') else '불합격'}", flush=True)
    accepted = sorted(d for d, r in results.items() if r.get("passed"))
    summary = {"rule": "89절 규칙 5(19일 때와 같은 고정신호 하루 점검 기준)", "dates": extra_env.NEW_DATES,
               "accepted": accepted, "rejected": sorted(set(extra_env.NEW_DATES) - set(accepted)),
               "failed_checks": {d: sorted(k for k, v in r["acceptance"].items() if not v) for d, r in results.items() if not r.get("passed")}}
    out.mkdir(parents=True, exist_ok=True)
    (out / "accepted_days.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    print(json.dumps(summary, ensure_ascii=False, indent=1))


def final(args) -> None:
    if not args.confirm_once:
        raise SystemExit("새 12일 마지막 시험은 설정을 모두 정한 뒤 한 번만 한다. 그때만 --confirm-once를 붙인다.")
    import multiprocessing
    if multiprocessing.get_start_method() != "fork":
        raise SystemExit("새 날짜 자료를 하위 실행에 넘기려면 fork 방식이 필요합니다(Colab·리눅스에서 실행).")
    import extra_env
    extra_env.activate()
    from focused_conflicts import measure
    from focused_data import environment_fingerprint
    from focused_eval import evaluation_fingerprint, model_identity
    if (environment_fingerprint(), evaluation_fingerprint()) != ("b20888514ea3", "2030250938d6"):
        raise SystemExit("환경·평가 지문이 확정한 값(pet1)과 다릅니다.")
    sel = json.loads(Path(args.selection).read_text())
    chosen = sel["selected"]
    _check_candidate(chosen, args.selected_models)
    if [model_identity(m)["model_sha256"][:10] for m in args.v6_models] != V6_PET1:
        raise SystemExit("기존 DQN 모델이 v6 pet1 공식 모델과 다릅니다.")
    accepted = json.loads(Path(args.accepted).read_text())["accepted"]
    if not accepted:
        raise SystemExit("점검을 통과한 새 날이 없습니다.")
    print(f"[새 날 마지막 시험] 날짜 {len(accepted)}일 {accepted}, 고른 후보 {chosen}", flush=True)
    common = dict(dates=accepted, arrivals=["random", "platoon"], out=args.out, tmp=args.tmp, workers=args.workers, until=None)
    measure(Namespace(policies=["fixed", "fixedc", "greedy", "dqn"], models=args.v6_models, **common))
    measure(Namespace(policies=["dqn"], models=args.selected_models, **common))
    report(Path(args.out) / "2030250938d6", accepted, args.v6_models, args.selected_models, chosen, Path(args.out))


def report(folder: Path, dates: list[str], v6_models: list[str], sel_models: list[str], chosen: str, out: Path) -> None:
    from focused_eval import model_identity
    from focused_v6_report import load_runs, pooled
    runs = [r for r in load_runs(folder) if r["date"] in dates]
    v6 = {model_identity(m)["model_sha256"] for m in v6_models}
    sel = {model_identity(m)["model_sha256"] for m in sel_models}
    groups = [("고정신호(원자료)", lambda r: r["policy"] == "fixed"), ("고정신호(보정)", lambda r: r["policy"] == "fixedc"),
              ("단순 제어", lambda r: r["policy"] == "greedy"), ("기존 DQN(v6 pet1)", lambda r: r.get("model_sha256") in v6),
              (f"멀리 보는 DQN({chosen})", lambda r: r.get("model_sha256") in sel)]
    lines = [f"# 새 날 마지막 시험(단독 모형) — {len(dates)}일: {', '.join(dates)}", "",
             "| 도착 | 정책 | 차량당 지연(초) | 300초 초과 | 길주로 동쪽 좌회전 | 석천로 남쪽 좌회전 | 보행 한도 초과 | 확인 안 된 충돌 |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    table = {}
    for arr in ("random", "platoon"):
        for name, pick in groups:
            rs = [r for r in runs if pick(r) and r.get("arrivals", "random") == arr]
            p = pooled(rs)
            table[f"{arr}/{name}"] = p
            sd = f" ± {p['mean_delay_s_std']:.2f}" if "mean_delay_s_std" in p else ""
            lines.append(f"| {arr} | {name} | {p['mean_delay_s']:.2f}{sd} | {p['long_waits']:.1f} | {p['left_east']:.1f} | "
                         f"{p['left_south']:.1f} | {p['ped_over']:.1f} | {p['unexplained']:.1f} |")
    text = "\n".join(lines) + "\n"
    (out / "final_report.md").write_text(text, encoding="utf-8")
    (out / "final_report.json").write_text(json.dumps(table, ensure_ascii=False, indent=1))
    print(text)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select")
    s.add_argument("--eval-dir", required=True)
    s.add_argument("--base-dir", required=True)
    s.add_argument("--base-models", nargs=5, required=True)
    s.add_argument("--cand", nargs=6, action="append", required=True, metavar=("이름", "M1", "M2", "M3", "M4", "M5"))
    s.add_argument("--out", required=True)
    d = sub.add_parser("daycheck")
    d.add_argument("--out", required=True)
    d.add_argument("--workers", type=int, default=0)
    f = sub.add_parser("final")
    f.add_argument("--selection", required=True)
    f.add_argument("--accepted", required=True)
    f.add_argument("--selected-models", nargs=5, required=True)
    f.add_argument("--v6-models", nargs=5, required=True)
    f.add_argument("--out", required=True)
    f.add_argument("--tmp", default="/content/extra_tmp")
    f.add_argument("--workers", type=int, default=0)
    f.add_argument("--confirm-once", action="store_true")
    args = ap.parse_args()
    {"select": select, "daycheck": daycheck, "final": final}[args.cmd](args)


if __name__ == "__main__":
    main()
