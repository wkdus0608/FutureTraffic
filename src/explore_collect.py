"""[추가 탐색] 새 날짜 3개 수집과 학생 연속 제어 평가(진행 기록 141절). 지문 밖.

질문: 경험하는 교통 상황을 늘리면(학습 묶음 3 → 6), 학생의 실제 제어가 좋아지는가?
- 수집: 새 묶음 3개(seed 42)를 125절과 같은 탐색 제어로 돈다(explore_online.run을 고치지 않고 부름). 탐색 시점의 SUMO 상태는
  기록된 행동으로 다시 돌려 만들고(모든 선택 시점이 기록과 같은지 확인, explore_remeasure.replay) 압축해 보존한다.
- 학생: 134절 고정 설정(10,000번 갱신, 문턱 0, 다섯 모델 평균). S3 = 기존 3개 묶음, S6 = 기존 + 새 3개, S6′(참고) = 6개 묶음의 seed 42만.
- 평가: 학습에 넣지 않은 08-11·08-23 × 13:00·18:00 × seed 42~45(16개)에서 P0·S3·S6·S6′ 연속 제어. 학생은 제어 60분 동안
  좌회전 정체 시점에서만 고르고 나머지와 꼬리는 P0(P0+신경망 혼합 제어).

명령(Colab 40번 칸)
  collect  --date --period --mode p0|search --work --out --workers   (새 묶음 하나)
  states   --work --out                                               (새 탐색 실행 3개의 상태 보존)
  train    --old <explore_online/runs> --out                          (S3·S6·S6′)
  evaluate --work --out --workers                                     (64번)
  summary  --out
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
import statistics
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import explore_distill as ed
import explore_online as eo
import explore_remeasure as rm
import explore_rollout as er
import explore_seedsplit as ess
import explore_student as es

PROJECT = ed.PROJECT
NEW_GROUPS = [("20220812", "1800"), ("20220819", "1300"), ("20220926", "1800")]
NEW_SEED = 42
EVAL_GROUPS = [("20220811", "1300"), ("20220811", "1800"), ("20220823", "1300"), ("20220823", "1800")]
EVAL_SEEDS = (42, 43, 44, 45)
STUDENTS = ("s3", "s6", "s6h")
EVAL_MODES = ("p0", *STUDENTS)
CHANGE_MIN = 0.5                               # S6의 변경 횟수 ≥ S3의 절반(P0로 돌아가서만 좋아진 것이 아님)
GROUPS_MIN = 3                                 # 평가 묶음 4개 중 3개 이상
ONLINE_CODE = rm.ONLINE_CODE


def code_fingerprint() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]


# ------------------------------------------------------------------ 학생 자료
def student_samples(name: str, old: list[dict], new: list[dict]) -> list[dict]:
    """S3 = 기존 묶음 전부, S6 = 기존 + 새 묶음 전부, S6′ = 6개 묶음의 seed 42만. 학습 비중은 탐색 시점 하나 = 같은 무게."""
    if name == "s3":
        return list(old)
    if name == "s6":
        return list(old) + list(new)
    if name == "s6h":
        return [s for s in list(old) + list(new) if s["seed"] == NEW_SEED]
    raise ValueError(name)


def data_hashes(runs: Path) -> dict:
    return {f.parent.name: rm.file_hash(f) for f in sorted(Path(runs).glob("*_search/decisions.jsonl"))}


def train(args) -> None:
    import torch as th
    out = Path(args.out)
    old, new = ed.load(Path(args.old)), ed.load(out / "runs")
    if sorted({s["group"] for s in new}) != sorted(f"{d}_{p}" for d, p in NEW_GROUPS):
        raise SystemExit(f"새 묶음 탐색 자료가 다 모이지 않았다: {sorted({s['group'] for s in new})}")
    for name in STUDENTS:
        d = out / "students" / name
        if (d / "policy.json").exists():
            print(f"이미 있음 → 건너뜀: {d}")
            continue
        samples = student_samples(name, old, new)
        states = es.fit_ensemble(samples, (ess.STEPS,), es.SEEDS)[ess.STEPS]
        d.mkdir(parents=True, exist_ok=True)
        for seed, state in zip(es.SEEDS, states):
            th.save(state, d / f"seed{seed}.pt")
        policy = {"name": name, "samples": len(samples), "groups": sorted({s["group"] for s in samples}),
                  "seeds_in_data": sorted({s["seed"] for s in samples}), "steps": ess.STEPS, "tau": ess.TAU, "model_seeds": list(es.SEEDS),
                  "weights": {f"seed{s}": rm.file_hash(d / f"seed{s}.pt") for s in es.SEEDS},
                  "data": {"old": data_hashes(Path(args.old)), "new": data_hashes(out / "runs")},
                  "code": code_fingerprint(), "student_code": es.code_fingerprint(), "distill_code": ed.code_fingerprint()}
        (d / "policy.json").write_text(json.dumps(policy, ensure_ascii=False, indent=1))
        print(f"학생 {name}: 탐색 시점 {len(samples)}번, 묶음 {len(policy['groups'])}개", flush=True)


def load_student(out: Path, name: str) -> tuple[list, dict]:
    import torch as th
    d = Path(out) / "students" / name
    policy = json.loads((d / "policy.json").read_text())
    for s in policy["model_seeds"]:
        if rm.file_hash(d / f"seed{s}.pt") != policy["weights"][f"seed{s}"]:
            raise SystemExit(f"학생 {name}의 가중치가 저장 때와 다르다")
    return [th.load(d / f"seed{s}.pt") for s in policy["model_seeds"]], policy


# ------------------------------------------------------------------ 수집(Colab)
def collect(args) -> None:
    """125절과 같은 실행(explore_online.run, df71623b23d4)으로 새 묶음 하나를 돈다."""
    if (args.date, args.period) not in NEW_GROUPS:
        raise SystemExit("141절에 정한 새 묶음만 돈다")
    if eo.code_fingerprint() != ONLINE_CODE:
        raise SystemExit(f"1단계 실행 코드 {eo.code_fingerprint()}가 125절의 {ONLINE_CODE}와 다르다")
    eo.run(SimpleNamespace(date=args.date, period=args.period, seed=NEW_SEED, mode=args.mode, work=args.work, out=args.out,
                           workers=args.workers, scenarios=eo.N_SCENARIOS, max_searches=None))


def _gzip(src: Path, dst: Path) -> None:
    with src.open("rb") as f, gzip.open(dst, "wb", compresslevel=6) as g:
        shutil.copyfileobj(f, g)


def states(args) -> None:
    """새 탐색 실행마다 기록된 행동으로 다시 돌려 탐색 시점의 SUMO 상태(.xml·.pkl)와 차량 파일을 압축 보존한다."""
    out = Path(args.out)
    for date, period in NEW_GROUPS:
        group, tag = f"{date}_{period}", f"{date}_{period}_s{NEW_SEED}_search"
        dst = out / "states" / tag
        if (dst / "done.json").exists():
            print(f"이미 있음 → 건너뜀: {dst}")
            continue
        rows = [json.loads(line) for line in (out / "runs" / tag / "decisions.jsonl").read_text().splitlines()]
        t0s = [r["t0"] for r in rows if r["searched"]]
        r = rm.replay(out / "runs", group, NEW_SEED, t0s, Path(args.work) / "states" / tag)
        dst.mkdir(parents=True, exist_ok=True)
        for t0, stem in sorted(r["stems"].items()):
            for suffix in (".xml", ".pkl"):
                _gzip(stem.with_suffix(suffix), dst / f"{int(t0)}{suffix}.gz")
        for f in sorted(r["routes"].glob("*.rou.xml")):
            _gzip(f, dst / f"{f.name}.gz")
        done = {"states": len(r["stems"]), "decisions_checked": r["decisions_checked"], "seconds": r["seconds"],
                "code": code_fingerprint(), "replay_code": rm.code_fingerprint(), "online_code": eo.code_fingerprint(),
                "rollout_code": er.code_fingerprint(), "decisions": rm.file_hash(out / "runs" / tag / "decisions.jsonl")}
        (dst / "done.json").write_text(json.dumps(done, ensure_ascii=False, indent=1))
        shutil.rmtree(Path(args.work) / "states" / tag, ignore_errors=True)
        print(f"상태 보존 {tag}: {len(r['stems'])}개, 선택 시점 {r['decisions_checked']}번 기록과 같음, {r['seconds']}초", flush=True)


# ------------------------------------------------------------------ 평가(Colab)
def eval_jobs(work: str, out: str) -> list[tuple]:
    return [(d, p, s, m, work, out) for d, p in EVAL_GROUPS for s in EVAL_SEEDS for m in EVAL_MODES]


def run_episode(job: tuple) -> str:
    """평가 에피소드 하나: P0만, 또는 제어 60분 동안 좌회전 정체 시점에서만 학생(나머지·꼬리는 P0). 125절 실행과 같은 셈."""
    import focused_train as ft
    from focused_eval import judge_collisions
    date, period, seed, mode, work, out = job
    tag = f"{date}_{period}_s{seed}_{mode}"
    d = Path(out) / "eval" / tag
    if (d / "metrics.json").exists():
        return f"{tag}: 이미 있음"
    models, policy = load_student(Path(out), mode) if mode != "p0" else (None, None)
    w = Path(work) / "eval" / tag
    shutil.rmtree(w, ignore_errors=True)
    started = time.time()
    ep = eo.OnlineEpisode(date, period, w / "routes", seed)
    end_control = ep.handover + ft.CONTROL_S
    decisions = []
    try:
        while not ep.done:
            allowed, a_p0 = ep.decide()
            action = a_p0
            if len(allowed) > 1:
                t0 = float(ep.env.sim_step)
                feat = er._features(ep, allowed, a_p0)
                obs64, obs72 = eo.observation72(ep)
                row = {"t0": t0, "in_control": t0 < end_control, "allowed": allowed, "p0_action": a_p0, "W": feat["W"],
                       **{k: feat[k] for k in (*er.LEFTS, *[f"{n}_served" for n in er.LEFTS])}, "left_point": eo.is_left_point(feat),
                       "obs64": obs64, "obs72": obs72, "student": False}
                if models is not None and row["in_control"] and row["left_point"]:
                    pred = es.predict(models, np.asarray([obs72], dtype=np.float32))[0]
                    action = es.hybrid(pred, {"allowed": allowed, "p0": a_p0}, ess.TAU)
                    row.update(student=True, pred=[round(float(x), 6) for x in pred])
                row["executed"] = action
                decisions.append(row)
            ep.apply(action)
    finally:
        ep.close()
    rows = eo.read_tripinfo(ep.tripinfo)
    metrics = eo.episode_metrics(rows, ep.vehicles, ep.handover, ft.CONTROL_S)
    coll = judge_collisions(ep.collisions)
    st = [x for x in decisions if x["student"]]
    metrics.update({"date": date, "period": period, "seed": seed, "mode": mode, "handover": ep.handover, "code": code_fingerprint(),
                    "online_code": eo.code_fingerprint(), "p0_code": er.P0_CODE, "student_weights": policy["weights"] if policy else None,
                    "seconds": round(time.time() - started, 1), "choice_points": len(decisions),
                    "left_points_in_control": sum(x["in_control"] and x["left_point"] for x in decisions),
                    "student_decisions": len(st), "changes": sum(x["executed"] != x["p0_action"] for x in st),
                    "changed_by_rules": ep.cons.interventions, "collisions_at_204820": coll["collisions_at_204820"],
                    "collisions_unexplained": len(coll["collisions_unexplained"])})
    d.mkdir(parents=True, exist_ok=True)
    with (d / "decisions.jsonl").open("w") as f:
        for x in decisions:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    (d / "vehicle_delays.json").write_text(json.dumps({v: float(r["timeLoss"]) + float(r["departDelay"]) for v, r in rows.items()}))
    tmp = d / "metrics.json.tmp"
    tmp.write_text(json.dumps(metrics, ensure_ascii=False, indent=1))
    tmp.replace(d / "metrics.json")
    shutil.rmtree(w, ignore_errors=True)
    return (f"{tag}: 차량당 지연 {metrics['mean_delay_s']}초, 좌회전 동/남 {metrics['left_east']}/{metrics['left_south']}초, "
            f"300초 초과 {metrics['long_waits']}, 학생 판단 {len(st)}번(변경 {metrics['changes']}), {metrics['seconds']}초")


def evaluate(args) -> None:
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor, as_completed
    for name in STUDENTS:
        load_student(Path(args.out), name)                             # 가중치가 다 있고 저장 때와 같은지 먼저
    jobs = [j for j in eval_jobs(args.work, args.out) if not (Path(args.out) / "eval" / f"{j[0]}_{j[1]}_s{j[2]}_{j[3]}" / "metrics.json").exists()]
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        for f in as_completed([pool.submit(run_episode, j) for j in jobs]):
            print(f.result(), flush=True)


# ------------------------------------------------------------------ 판정
def episode_table(out: Path) -> tuple[list[dict], list[str]]:
    rows, missing = [], []
    for d, p in EVAL_GROUPS:
        for s in EVAL_SEEDS:
            got = {}
            for m in EVAL_MODES:
                f = Path(out) / "eval" / f"{d}_{p}_s{s}_{m}" / "metrics.json"
                if f.exists():
                    got[m] = json.loads(f.read_text())
                else:
                    missing.append(f.parent.name)
            if len(got) == len(EVAL_MODES):
                rows.append({"group": f"{d}_{p}", "seed": s, **got})
    return rows, missing


def _group_means(rows: list[dict], value) -> dict:
    groups = sorted({r["group"] for r in rows})
    return {g: statistics.fmean(value(r) for r in rows if r["group"] == g) for g in groups}


def _worse(rows: list[dict], a: str, b: str) -> dict:
    """b가 a보다 나쁜지: 좌회전 두 곳(에피소드 평균)과 300초 초과(합)."""
    lefts = {n: statistics.fmean(r[b][n] for r in rows) > statistics.fmean(r[a][n] for r in rows) for n in er.LEFTS}
    return {"lefts_worse": any(lefts.values()), "long_worse": sum(r[b]["long_waits"] for r in rows) > sum(r[a]["long_waits"] for r in rows)}


def compare(rows: list[dict], base: str, mode: str) -> dict:
    """에피소드마다 base 지연 − mode 지연(양수면 mode가 나음)의 평균, 95% t 구간, 묶음별 평균, 좌회전·300초 초과가 나빠졌는지."""
    diffs = [r[base]["mean_delay_s"] - r[mode]["mean_delay_s"] for r in rows]
    mean, lo, hi = er.t_interval(diffs)
    groups = _group_means(rows, lambda r: r[base]["mean_delay_s"] - r[mode]["mean_delay_s"])
    return {"mean": round(mean, 3), "interval": [round(lo, 3), round(hi, 3)], "episodes": len(diffs),
            "positive_episodes": sum(x > 0 for x in diffs), "groups": {g: round(v, 3) for g, v in groups.items()},
            "positive_groups": sum(v > 0 for v in groups.values()), **_worse(rows, base, mode),
            "changes": {m: sum(r[m]["changes"] for r in rows) for m in (base, mode) if m != "p0"}}


def improve_reading(rows: list[dict]) -> dict:
    """141절 주 판정: S6가 S3보다 나은가(D = S3 지연 − S6 지연)."""
    c = compare(rows, "s3", "s6")
    enough_changes = c["changes"]["s6"] >= CHANGE_MIN * c["changes"]["s3"]
    if (c["mean"] > 0 and c["interval"][0] > 0 and c["positive_groups"] >= GROUPS_MIN and not c["lefts_worse"] and not c["long_worse"]
            and enough_changes):
        word = "새 날짜 자료로 학생 제어가 좋아졌다"
    elif c["mean"] <= 0:
        word = "변화 없음 또는 나빠짐"
    else:
        word = "불분명"
    return {"word": word, "enough_changes": enough_changes, **c}


def beats_p0(rows: list[dict], mode: str) -> dict:
    """141절 별도 판정: P0+신경망 혼합 제어가 P0를 넘었나(Δ = P0 지연 − 학생 지연)."""
    c = compare(rows, "p0", mode)
    ok = c["mean"] > 0 and c["interval"][0] > 0 and c["positive_groups"] >= GROUPS_MIN and not c["lefts_worse"] and not c["long_worse"]
    return {"word": "P0를 넘었다" if ok else "P0를 넘지 못했다", **c}


def offline_reference(out: Path) -> dict:
    """참고: S3를 새 3개 묶음의 탐색 자료로 저장 장면 평가(완전히 새 날짜, 판정에 안 씀)."""
    new = ed.load(Path(out) / "runs")
    models, _ = load_student(Path(out), "s3")
    picks = es.picks_for(models, new, ess.TAU)
    return {**ess.measure(new, picks), "per_episode": ess.per_episode(new, picks)}


def teacher_reference(out: Path) -> list[dict]:
    """참고: 새 묶음에서 탐색 제어 대 P0(seed 42 하나씩)."""
    res = []
    for d, p in NEW_GROUPS:
        m = {mode: json.loads((Path(out) / "runs" / f"{d}_{p}_s{NEW_SEED}_{mode}" / "metrics.json").read_text()) for mode in ("p0", "search")}
        res.append({"group": f"{d}_{p}", "delta": round(m["p0"]["mean_delay_s"] - m["search"]["mean_delay_s"], 3),
                    "p0": m["p0"]["mean_delay_s"], "search": m["search"]["mean_delay_s"], "searches": m["search"]["searches"],
                    "searched_differs": m["search"]["searched_differs"], "search_cpu_s": m["search"]["search_cpu_s"]})
    return res


def summary(args) -> None:
    out = Path(args.out)
    rows, missing = episode_table(out)
    codes = {r[m]["code"] for r in rows for m in EVAL_MODES}
    if len(codes) > 1:
        raise SystemExit(f"서로 다른 코드의 평가가 섞였다: {sorted(codes)}")
    complete = not missing and len(rows) == len(EVAL_GROUPS) * len(EVAL_SEEDS)
    result = {"code": code_fingerprint(), "eval_code": codes.pop() if codes else None, "missing": missing, "episodes": len(rows),
              "improve": improve_reading(rows) if complete else None,
              "beats_p0": {m: beats_p0(rows, m) for m in STUDENTS} if complete else None,
              "reference_s6h_vs_s3": compare(rows, "s3", "s6h") if complete else None,
              "reference_offline_s3_new_groups": offline_reference(out), "reference_teacher_new_groups": teacher_reference(out),
              "students": {m: json.loads((out / "students" / m / "policy.json").read_text())["samples"] for m in STUDENTS},
              "per_episode": [{"group": r["group"], "seed": r["seed"],
                               **{m: {k: r[m][k] for k in ("mean_delay_s", "left_east", "left_south", "long_waits", "unfinished", "changes",
                                                          "student_decisions", "changed_by_rules", "collisions_unexplained")}
                                  for m in EVAL_MODES}} for r in rows]}
    (out / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=1))
    lines = ["# 새 날짜 3개 수집과 학생 연속 제어 평가(진행 기록 141절)", ""]
    if not complete:
        lines.append(f"- 평가가 다 모이지 않음: {', '.join(missing)}")
    else:
        i = result["improve"]
        lines += [f"- 주 판정(S6 대 S3): **{i['word']}** — D 평균 {i['mean']}초 [{i['interval'][0]}, {i['interval'][1]}], "
                  f"묶음 {i['positive_groups']}/4, 좌회전 나빠짐 {i['lefts_worse']}, 300초 초과 나빠짐 {i['long_worse']}, "
                  f"변경 S3 {i['changes']['s3']}번 → S6 {i['changes']['s6']}번"]
        for m in STUDENTS:
            b = result["beats_p0"][m]
            lines.append(f"- {m} 대 P0: **{b['word']}** — Δ 평균 {b['mean']}초 [{b['interval'][0]}, {b['interval'][1]}], 에피소드 {b['positive_episodes']}/16, "
                         f"묶음 {b['positive_groups']}/4, 좌회전 나빠짐 {b['lefts_worse']}, 300초 초과 나빠짐 {b['long_worse']}, 변경 {b['changes'][m]}번")
        lines += ["", "| 묶음 | seed | P0 | S3 | S6 | S6′ | 변경 S3 / S6 / S6′ | 300초 초과 P0 / S3 / S6 |", "|---|---:|---:|---:|---:|---:|---|---|"]
        for r in result["per_episode"]:
            lines.append(f"| {r['group']} | {r['seed']} | {r['p0']['mean_delay_s']} | {r['s3']['mean_delay_s']} | {r['s6']['mean_delay_s']} | "
                         f"{r['s6h']['mean_delay_s']} | {r['s3']['changes']} / {r['s6']['changes']} / {r['s6h']['changes']} | "
                         f"{r['p0']['long_waits']} / {r['s3']['long_waits']} / {r['s6']['long_waits']} |")
    o = result["reference_offline_s3_new_groups"]
    lines += ["", f"- 참고(판정 안 씀): S3를 새 3개 묶음 탐색 자료로 저장 장면 평가 — 이득 {o['gain']}대·초(몫 {o['share']}), 변경 {o['changes']}번"
                  f"(이득 {o['good_changes']}·손해 {o['bad_changes']})"]
    for t in result["reference_teacher_new_groups"]:
        lines.append(f"- 참고: 새 묶음 {t['group']} 탐색 제어 대 P0 Δ {t['delta']}초(탐색 {t['searches']}번, P0와 다름 {t['searched_differs']})")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--date", required=True)
    c.add_argument("--period", required=True)
    c.add_argument("--mode", choices=eo.MODES, required=True)
    c.add_argument("--work", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--workers", type=int, default=4)
    s = sub.add_parser("states")
    s.add_argument("--work", required=True)
    s.add_argument("--out", required=True)
    t = sub.add_parser("train")
    t.add_argument("--old", required=True)
    t.add_argument("--out", required=True)
    e = sub.add_parser("evaluate")
    e.add_argument("--work", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--workers", type=int, default=4)
    m = sub.add_parser("summary")
    m.add_argument("--out", required=True)
    args = ap.parse_args()
    {"collect": collect, "states": states, "train": train, "evaluate": evaluate, "summary": summary}[args.cmd](args)


if __name__ == "__main__":
    main()
