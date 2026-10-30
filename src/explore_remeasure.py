"""[추가 탐색] 학생의 큰 손익 20곳 독립 재측정(진행 기록 137절). 지문 밖, 학습 없음.

목적: 135절 고정 학생(10,000번 갱신, 문턱 0, 다섯 모델 평균)의 큰 손해 10곳·큰 이득 10곳이 새 시나리오에서도 다시 나타나는가.
학생 전체 성능의 추정이 아니고, 원래 탐색 선택이 가장 좋았는지도 검증하지 않는다(P0와 학생 선택 두 후보만).

명령
  targets (맥, SUMO 없음): 학생 선택을 다시 계산해(135절 합계와 같은지 확인) 대상 20곳을 고르고 모델과 함께 저장한다.
  check   (Colab): 대상 2곳(손해·이득 묶음 1번째)의 상태를 기록된 행동으로 다시 만들고, 원래 8개 시나리오 비용이 기록과 같은지 본다.
  measure (Colab): 20곳에서 P0와 학생 선택을 새 시나리오 16개로 짝지어 잰다(곳마다 원래 시나리오 하나로 복구도 확인).
  summary : 137절 읽는 법.

  .venv/bin/python src/explore_remeasure.py targets --runs results/explore/online_20261023/runs --out results/explore/remeasure_20261024
  FOCUSED_RULESET=pet1 FOCUSED_NORTH_EXT=240 python src/explore_remeasure.py check --runs <Drive>/explore_online/runs \
      --out <Drive>/explore_remeasure --work /content/remeasure --workers 8
  (measure도 같은 인자) / python src/explore_remeasure.py summary --out <Drive>/explore_remeasure
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import statistics
import time
from pathlib import Path

import explore_distill as ed
import explore_online as eo
import explore_rollout as er
import explore_seedsplit as ess
import explore_student as es

PROJECT = ed.PROJECT
N_SIDE = 10
N_FRESH = 16
FRESH_BASE = 60_000_000
CLEAR_SE = 2.0
RETAIN = 0.5
Z95 = 1.96
SEEDSPLIT_GAIN = -15297.3                      # 135절 합계
ONLINE_CODE = "df71623b23d4"                   # 125절 실행 코드


def code_fingerprint() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]


def file_hash(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]


# ------------------------------------------------------------------ 순수 계산
def fresh_seeds(base_seed: int, t0: float, n: int = N_FRESH) -> list[int]:
    """새 시나리오: 60,000,000 + (SUMO seed × 10,000 + 시각) × 20 + j. 1단계((…) × 10 + j, 약 4.4백만 아래)·0단계(1001~2008)와 겹치지 않는다."""
    return [FRESH_BASE + (base_seed * 10_000 + int(t0)) * 20 + j for j in range(n)]


def paired(a: list[float], b: list[float]) -> tuple[float, float]:
    """짝 차이 a − b의 평균과 표준오차."""
    d = [x - y for x, y in zip(a, b)]
    return statistics.fmean(d), statistics.stdev(d) / math.sqrt(len(d))


def choose(changes: list[dict], n: int = N_SIDE) -> list[dict]:
    """137절: 분명한(|g| ≥ 표준오차 2배) 손해 중 g가 가장 작은 n곳, 분명한 이득 중 g가 가장 큰 n곳. 같으면 날짜·시간대·seed·시각 순."""
    key = lambda c: (c["group"], c["seed"], c["t0"])
    clear = [c for c in changes if abs(c["g"]) >= CLEAR_SE * c["se"]]
    loss = sorted((c for c in clear if c["g"] < 0), key=lambda c: (c["g"], key(c)))[:n]
    gain = sorted((c for c in clear if c["g"] > 0), key=lambda c: (-c["g"], key(c)))[:n]
    return [{**c, "set": "loss", "rank": i} for i, c in enumerate(loss)] + [{**c, "set": "gain", "rank": i} for i, c in enumerate(gain)]


def mismatches(row: dict, t0: float, allowed: list[int], p0_action: int, obs64: list[float], obs72: list[float]) -> list[str]:
    """다시 돌린 선택 시점이 기록과 다른 항목."""
    now = {"t0": t0, "allowed": list(allowed), "p0_action": int(p0_action), "obs64": obs64, "obs72": obs72}
    return [k for k, v in now.items() if row[k] != v]


def set_reading(rows: list[dict]) -> dict:
    """137절 읽는 법(묶음 하나). rows: [{"g": 원래 8개 평균, "fresh_mean", "fresh_se"}]"""
    n = len(rows)
    if n == 0:
        return {"n": 0, "word": "불분명"}
    orig = statistics.fmean(r["g"] for r in rows)
    mean = statistics.fmean(r["fresh_mean"] for r in rows)
    se = math.sqrt(sum(r["fresh_se"] ** 2 for r in rows)) / n
    lo, hi = sorted(((mean - Z95 * se) / orig, (mean + Z95 * se) / orig))
    if lo >= RETAIN:
        word = "대부분 다시 나타난다"
    elif 0 < lo and hi < RETAIN:
        word = "일부만 다시 나타난다"
    elif lo <= 0 and hi < RETAIN:
        word = "사라진다"
    else:
        word = "불분명"
    return {"n": n, "original_mean": round(orig, 1), "fresh_mean": round(mean, 1), "fresh_se": round(se, 1),
            "fresh_interval": [round(mean - Z95 * se, 1), round(mean + Z95 * se, 1)],
            "retained": round(mean / orig, 3), "retained_interval": [round(lo, 3), round(hi, 3)], "word": word}


# ------------------------------------------------------------------ 대상(맥)
def decision_row(runs: Path, group: str, seed: int, t0: float) -> dict:
    date, period = group.split("_")
    for line in (Path(runs) / f"{date}_{period}_s{seed}_search" / "decisions.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r["t0"] == t0:
            return r
    raise SystemExit(f"기록에 없는 시점: {group} s{seed} {t0}")


def targets(args) -> None:
    import torch as th
    out = Path(args.out)
    if (out / "targets.json").exists():
        raise SystemExit(f"이미 있음: {out / 'targets.json'} (137절: 대상은 한 번만 정한다)")
    samples = ed.load(Path(args.runs))
    rows, models = [], {}
    for a, b in ess.DIRECTIONS:                                        # 135절 run_direction과 같은 계산(모델을 남기려고 풀어 씀)
        train = [s for s in samples if s["seed"] == a]
        test = [s for s in samples if s["seed"] == b]
        ens = es.fit_ensemble(train, (ess.STEPS,), es.SEEDS)
        models[a] = ens[ess.STEPS]
        rows += zip(test, es.picks_for(ens[ess.STEPS], test, ess.TAU))
    total = round(es.gain([s for s, _ in rows], [p for _, p in rows]), 1)
    if abs(total - SEEDSPLIT_GAIN) > 0.05:
        raise SystemExit(f"학생 선택의 합계 {total}가 135절 {SEEDSPLIT_GAIN}와 다르다")
    changes = []
    for s, p in rows:
        if p == s["p0"]:
            continue
        g, se = paired(s["costs"][s["p0"]], s["costs"][p])
        rec = decision_row(Path(args.runs), s["group"], s["seed"], s["t0"])
        seeds = eo.scenario_seeds(s["seed"], s["t0"])
        if rec["seeds"] != seeds or [x["seed"] for x in rec["scores"][str(p)]] != seeds:
            raise SystemExit(f"기록된 시나리오 seed가 다르다: {s['group']} s{s['seed']} {s['t0']}")
        changes.append({"group": s["group"], "seed": s["seed"], "t0": s["t0"], "allowed": s["allowed"], "p0": s["p0"], "student": p,
                        "search": s["search"], "g": round(g, 3), "se": round(se, 3), "original_seeds": seeds,
                        "original_p0_costs": s["costs"][s["p0"]], "original_student_costs": s["costs"][p],
                        "fresh_seeds": fresh_seeds(s["seed"], s["t0"])})
    chosen = choose(changes)
    out.mkdir(parents=True, exist_ok=True)
    for a, states in models.items():
        th.save(states, out / f"student_train_s{a}.pt")
    record = {"code": code_fingerprint(), "seedsplit_code": ess.code_fingerprint(), "student_code": es.code_fingerprint(),
              "distill_code": ed.code_fingerprint(), "student": {"steps": ess.STEPS, "tau": ess.TAU, "seeds": list(es.SEEDS)},
              "student_total_gain": total, "changes": len(changes), "clear_losses": sum(c["g"] < 0 and abs(c["g"]) >= CLEAR_SE * c["se"] for c in changes),
              "clear_gains": sum(c["g"] > 0 and abs(c["g"]) >= CLEAR_SE * c["se"] for c in changes),
              "models": {str(a): file_hash(out / f"student_train_s{a}.pt") for a in models}, "targets": chosen}
    (out / "targets.json").write_text(json.dumps(record, ensure_ascii=False, indent=1))
    print(json.dumps({k: record[k] for k in ("student_total_gain", "changes", "clear_losses", "clear_gains", "models")}, ensure_ascii=False))
    for c in chosen:
        print(f"{c['set']} {c['rank']}: {c['group']} s{c['seed']} t0={c['t0']} P0 {c['p0']} → 학생 {c['student']} (탐색 {c['search']}), "
              f"g = {c['g']} ± {c['se']}")
    print("targets.json 해시", file_hash(out / "targets.json"))


# ------------------------------------------------------------------ 상태 복구·굴리기(Colab)
def replay(runs: Path, group: str, seed: int, t0s: list[float], work: Path) -> dict:
    """125절 탐색 실행을 기록된 '실행한 행동'으로 다시 돌려 t0s 시각의 상태를 저장한다. 원래처럼 탐색 시점마다 상태를 저장하고
    (대상이 아니면 바로 지움), 모든 선택 시점에서 시각·허용·P0·관측이 기록과 같은지 본다(다르면 멈춤)."""
    date, period = group.split("_")
    rows = [json.loads(line) for line in (Path(runs) / f"{date}_{period}_s{seed}_search" / "decisions.jsonl").read_text().splitlines()]
    wanted = set(t0s)
    started = time.time()
    shutil.rmtree(work, ignore_errors=True)
    ep = eo.OnlineEpisode(date, period, work / "routes", seed)
    er.types_only_file(ep.route)                                       # 원래 실행과 같은 순서
    stems, k = {}, 0
    try:
        while not ep.done and len(stems) < len(wanted):
            allowed, a_p0 = ep.decide()
            action = a_p0
            if len(allowed) > 1:
                t0 = float(ep.env.sim_step)
                if k >= len(rows):
                    raise SystemExit(f"{group} s{seed}: 기록보다 선택 시점이 많다")
                row = rows[k]
                k += 1
                er._features(ep, allowed, a_p0)
                obs64, obs72 = eo.observation72(ep)
                bad = mismatches(row, t0, allowed, a_p0, obs64, obs72)
                if bad:
                    raise SystemExit(f"{group} s{seed} 시각 {t0}: 기록과 다름 {bad}")
                if row["searched"]:
                    keep = t0 in wanted
                    stem = work / "states" / f"{group}_s{seed}_{int(t0)}" if keep else work / "scratch" / "snap"
                    ep.snapshot(stem, allowed, a_p0)
                    if keep:
                        stems[t0] = stem
                    else:
                        for suffix in (".xml", ".pkl"):
                            stem.with_suffix(suffix).unlink(missing_ok=True)
                action = row["executed"]
            ep.apply(action)
    finally:
        ep.close()
    if set(stems) != wanted:
        raise SystemExit(f"{group} s{seed}: 대상 시각을 다 만나지 못했다 {sorted(wanted - set(stems))}")
    return {"stems": stems, "routes": work / "routes", "decisions_checked": k, "seconds": round(time.time() - started, 1)}


def _pool(workers: int):
    """하위 실행이 본 실행의 SUMO 연결을 물려받지 않게, SUMO를 열기 전에 새 프로세스(spawn)로 띄워 둔다(125절과 같음)."""
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor
    pool = ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"))
    list(pool.map(abs, range(workers)))
    return pool


def _load_targets(out: Path, expect: str | None) -> dict:
    path = out / "targets.json"
    if not path.exists():
        raise SystemExit(f"대상 파일이 없다: {path}")
    if expect and file_hash(path) != expect:
        raise SystemExit(f"대상 파일 해시 {file_hash(path)}가 137절 기록 {expect}와 다르다")
    return json.loads(path.read_text())


def _guard() -> None:
    if er.code_fingerprint() != eo.ROLLOUT_CODE:
        raise SystemExit(f"탐색 코드 {er.code_fingerprint()}가 0단계의 {eo.ROLLOUT_CODE}와 다르다")
    if eo.code_fingerprint() != ONLINE_CODE:
        raise SystemExit(f"1단계 실행 코드 {eo.code_fingerprint()}가 125절의 {ONLINE_CODE}와 다르다")


def _restore_all(runs: Path, chosen: list[dict], work: Path) -> tuple[dict, dict]:
    by_episode: dict = {}
    for c in chosen:
        by_episode.setdefault((c["group"], c["seed"]), []).append(c["t0"])
    stems, info = {}, {}
    for (group, seed), t0s in sorted(by_episode.items()):
        r = replay(runs, group, seed, t0s, work / f"{group}_s{seed}")
        for t0, stem in r["stems"].items():
            stems[(group, seed, t0)] = (stem, r["routes"])
        info[f"{group}_s{seed}"] = {"targets": len(t0s), "decisions_checked": r["decisions_checked"], "seconds": r["seconds"]}
        print(f"복구 {group} s{seed}: 대상 {len(t0s)}곳, 선택 시점 {r['decisions_checked']}번 기록과 같음, {r['seconds']}초", flush=True)
    return stems, info


def _run_jobs(pool, jobs: list[tuple]) -> list[dict]:
    return list(pool.map(er._rollout_job, jobs))


def check(args) -> None:
    _guard()
    out = Path(args.out)
    rec = _load_targets(out, args.expect)
    chosen = [c for c in rec["targets"] if c["rank"] == 0]
    started = time.time()
    pool = _pool(args.workers)
    try:
        stems, info = _restore_all(Path(args.runs), chosen, Path(args.work) / "check")
        jobs, want = [], []
        for c in chosen:
            stem, routes = stems[(c["group"], c["seed"], c["t0"])]
            for who, cand, costs in (("p0", c["p0"], c["original_p0_costs"]), ("student", c["student"], c["original_student_costs"])):
                for s, cost in zip(c["original_seeds"], costs):
                    jobs.append((str(stem), str(routes), cand, s))
                    want.append((c["set"], who, s, cost))
        res = _run_jobs(pool, jobs)
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
    rows = [{"set": st, "who": who, "seed": s, "recorded": cost, "again": r["marks"][eo.MARK]["cost"]} for (st, who, s, cost), r in zip(want, res)]
    problems = [r for r in rows if r["recorded"] != r["again"]]
    result = {"ok": not problems, "code": code_fingerprint(), "targets_hash": file_hash(out / "targets.json"), "replay": info,
              "rollouts": len(rows), "same": len(rows) - len(problems), "problems": problems, "rows": rows,
              "cpu_s": round(sum(r["seconds"] for r in res), 1), "seconds": round(time.time() - started, 1)}
    (out / "check.json").write_text(json.dumps(result, ensure_ascii=False, indent=1))
    print(f"재현 확인: 원래 8개 비용 {result['same']}/{result['rollouts']}개 같음 → {'통과' if result['ok'] else '실패'}", flush=True)
    if problems:
        raise SystemExit(f"재현 실패 {len(problems)}개: {problems[:3]}")


def measure(args) -> None:
    _guard()
    out = Path(args.out)
    if (out / "measure.json").exists():
        print(f"이미 있음 → 건너뜀: {out / 'measure.json'}")
        return
    rec = _load_targets(out, args.expect)
    chk = json.loads((out / "check.json").read_text()) if (out / "check.json").exists() else {}
    if not (chk.get("ok") and chk.get("code") == code_fingerprint() and chk.get("targets_hash") == file_hash(out / "targets.json")):
        raise SystemExit("재현 확인(check)이 같은 코드·대상으로 통과하지 않았다(137절: 통과해야 본 재측정을 한다)")
    started = time.time()
    pool = _pool(args.workers)
    try:
        stems, info = _restore_all(Path(args.runs), rec["targets"], Path(args.work) / "measure")
        jobs, keys = [], []
        for i, c in enumerate(rec["targets"]):
            stem, routes = stems[(c["group"], c["seed"], c["t0"])]
            for who, cand in (("p0", c["p0"]), ("student", c["student"])):
                for kind, s in [("fresh", s) for s in c["fresh_seeds"]] + [("original", c["original_seeds"][0])]:
                    jobs.append((str(stem), str(routes), cand, s))
                    keys.append((i, who, kind))
        res = _run_jobs(pool, jobs)
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
    rows = []
    for i, c in enumerate(rec["targets"]):
        got = {(who, kind): [r for (j, w, k), r in zip(keys, res) if j == i and w == who and k == kind] for who in ("p0", "student")
               for kind in ("fresh", "original")}
        restored = (got[("p0", "original")][0]["marks"][eo.MARK]["cost"] == c["original_p0_costs"][0]
                    and got[("student", "original")][0]["marks"][eo.MARK]["cost"] == c["original_student_costs"][0])
        p0 = [r["marks"][eo.MARK]["cost"] for r in got[("p0", "fresh")]]
        st = [r["marks"][eo.MARK]["cost"] for r in got[("student", "fresh")]]
        mean, se = paired(p0, st)
        rows.append({**{k: c[k] for k in ("set", "rank", "group", "seed", "t0", "p0", "student", "search", "g", "se")}, "restored": restored,
                     "fresh_p0_costs": p0, "fresh_student_costs": st, "fresh_mean": round(mean, 3), "fresh_se": round(se, 3),
                     "fresh_marks": {who: [r["marks"] for r in got[(who, "fresh")]] for who in ("p0", "student")},
                     "leaks": sum(r["leaks"] for (j, *_), r in zip(keys, res) if j == i)})
    result = {"code": code_fingerprint(), "targets_hash": file_hash(out / "targets.json"), "replay": info, "rollouts": len(res),
              "cpu_s": round(sum(r["seconds"] for r in res), 1), "seconds": round(time.time() - started, 1), "rows": rows}
    tmp = out / "measure.json.tmp"
    tmp.write_text(json.dumps(result, ensure_ascii=False, indent=1))
    tmp.replace(out / "measure.json")
    print(f"본 재측정: 굴리기 {len(res)}번, CPU {result['cpu_s']}초, 벽 {result['seconds']}초, 복구 확인 실패 {sum(not r['restored'] for r in rows)}곳",
          flush=True)


def summary(args) -> None:
    out = Path(args.out)
    m = json.loads((out / "measure.json").read_text())
    rows = m["rows"]
    sets = {name: set_reading([r for r in rows if r["set"] == name and r["restored"]]) for name in ("loss", "gain")}
    result = {"code": m["code"], "targets_hash": m["targets_hash"], "sets": sets, "restore_failed": [r for r in rows if not r["restored"]],
              "leaks": sum(r["leaks"] for r in rows), "rollouts": m["rollouts"], "cpu_s": m["cpu_s"], "seconds": m["seconds"], "replay": m["replay"]}
    (out / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=1))
    names = {"loss": "손해 묶음", "gain": "이득 묶음"}
    lines = ["# 학생의 큰 손익 20곳 독립 재측정(진행 기록 137절)", ""]
    for name, s in sets.items():
        if not s["n"]:
            lines.append(f"- {names[name]}: 잴 수 있는 곳 없음 → 불분명")
            continue
        lines.append(f"- {names[name]}({s['n']}곳): **{s['word']}** — 원래 8개 평균 {s['original_mean']}대·초 → 새 16개 평균 {s['fresh_mean']} "
                     f"[{s['fresh_interval'][0]}, {s['fresh_interval'][1]}], 남은 비율 {s['retained']} [{s['retained_interval'][0]}, {s['retained_interval'][1]}]")
    lines += ["", f"- 복구 확인 실패 {len(result['restore_failed'])}곳, 실제 미래 차 엿봄 {result['leaks']}대, 굴리기 {m['rollouts']}번, "
                  f"CPU {m['cpu_s']}초, 벽 {m['seconds']}초", "",
              "| 묶음 | 순위 | 에피소드 | seed | 시각 | P0 → 학생 (탐색) | 원래 g ± 표준오차 | 새 평균 ± 표준오차 | 복구 |", "|---|---:|---|---:|---:|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {names[r['set']]} | {r['rank'] + 1} | {r['group']} | {r['seed']} | {r['t0']} | {r['p0']} → {r['student']} ({r['search']}) | "
                     f"{r['g']} ± {r['se']} | {r['fresh_mean']} ± {r['fresh_se']} | {'같음' if r['restored'] else '다름'} |")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("targets")
    t.add_argument("--runs", default=str(PROJECT / "results/explore/online_20261023/runs"))
    t.add_argument("--out", default=str(PROJECT / "results/explore/remeasure_20261024"))
    for name in ("check", "measure"):
        c = sub.add_parser(name)
        c.add_argument("--runs", required=True)
        c.add_argument("--out", required=True)
        c.add_argument("--work", required=True)
        c.add_argument("--workers", type=int, default=4)
        c.add_argument("--expect", default=None, help="137절에 적은 targets.json 해시")
    s = sub.add_parser("summary")
    s.add_argument("--out", required=True)
    args = ap.parse_args()
    {"targets": targets, "check": check, "measure": measure, "summary": summary}[args.cmd](args)


if __name__ == "__main__":
    main()
