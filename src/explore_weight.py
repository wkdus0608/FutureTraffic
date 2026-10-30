"""WEIGHT1: 고정 P0 가중치 → 작은 학습 → 조건부 본학습·확인 평가.

기존 결과와 코드는 보존한다. 모든 명령은 같은 --out을 재개할 수 있지만 코드·자료·설정
지문이 다르면 재사용하지 않는다. 총 48시간, 초기 6시간(병렬도 합산) 상한이다.
사용: python src/explore_weight.py run --out results/explore/weight_v1 --work /tmp/weight_v1
장시간 실행은 Colab에서도 동일한 명령을 사용한다. summary는 실행 없이 저장 결과만 읽는다.
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import gzip
import hashlib
import importlib.metadata
import json
import math
import os
import statistics
import sys
import time
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
CONFIG = PROJECT / "config/explore_weight_v1.json"
SOURCES = ("src/explore_weight.py", "src/explore_weight_core.py", "src/explore_weight_episode.py",
           "src/explore_weight_train.py", "src/explore_pred.py", "src/explore_online.py",
           "src/explore_rollout.py", "src/explore_choice.py", "src/focused_train.py", "src/focused_eval.py",
           "config/collision_exceptions.json")


def plain(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, dict):
        return {str(k): plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [plain(v) for v in x]
    return x


def digest(x) -> str:
    return hashlib.sha256(json.dumps(plain(x), sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def file_hash(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(plain(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def config():
    return read(CONFIG)


def configure():
    os.environ.setdefault("FOCUSED_RULESET", "pet1")
    os.environ.setdefault("FOCUSED_NORTH_EXT", "240")
    if os.environ["FOCUSED_RULESET"] != "pet1" or float(os.environ["FOCUSED_NORTH_EXT"]) != 240:
        raise ValueError("WEIGHT1은 pet1·북쪽 연장 240m 환경만 사용합니다")


def validate_implementation(cfg):
    from explore_weight_core import MODE_FACTORS, OBSERVATION_SIZE, MODE_MASK_SLICE
    from explore_weight_episode import CONTROL_S, STEP_S, TAIL_S, GAMMA, REWARD_SCALE
    from explore_weight_train import SETTINGS, PILOT_EPISODES, FULL_EPISODES
    actual = {"modes": [list(x) for x in MODE_FACTORS], "size": OBSERVATION_SIZE,
              "mask": [MODE_MASK_SLICE.start, MODE_MASK_SLICE.stop],
              "episode": [CONTROL_S, STEP_S, TAIL_S], "gamma": GAMMA, "scale": REWARD_SCALE,
              "episodes": [PILOT_EPISODES, FULL_EPISODES]}
    expected = {"modes": cfg["modes"], "size": cfg["observation"]["size"],
                "mask": cfg["observation"]["mode_mask_slice"],
                "episode": [cfg["episode"][k] for k in ("control_s", "step_s", "tail_s")],
                "gamma": cfg["learning"]["gamma"], "scale": cfg["learning"]["reward_scale"],
                "episodes": [cfg["learning"][k] for k in ("pilot_episodes", "full_episodes")]}
    mapping = {"network": "net_arch", "exploration_steps": "epsilon_decay_steps",
               "exploration_initial_eps": "epsilon_start", "exploration_final_eps": "epsilon_end"}
    for field in ("learning_rate", "buffer_size", "learning_starts", "batch_size", "train_freq",
                  "target_update_interval", "n_steps", "gamma", *mapping):
        if SETTINGS[field] != cfg["learning"][mapping.get(field, field)]:
            raise ValueError(f"설정 파일과 학습기 구현 불일치: {field}")
    if actual != expected:
        raise ValueError("설정 파일과 가중치/에피소드/관측 구현이 다릅니다")


def register(out: Path) -> str:
    configure()
    from focused_data import ENV_FILES, environment_fingerprint
    cfg = config()
    validate_implementation(cfg)
    env = environment_fingerprint()
    if env != cfg["environment"]["fingerprint"]:
        raise ValueError(f"잠근 환경 지문 불일치: {env}")
    if file_hash(PROJECT / "src/explore_pred.py")[:12] != cfg["environment"]["p0_code"]:
        raise ValueError("기존 P0 코드가 잠근 코드와 다릅니다")
    frozen = {"config": cfg, "environment_fingerprint": env,
              "files": {s: file_hash(PROJECT / s) for s in sorted(set(SOURCES) | set(ENV_FILES))},
              "packages": {p: importlib.metadata.version(p) for p in
                           ("numpy", "torch", "stable-baselines3", "sumo-rl", "eclipse-sumo")}}
    fp = digest(frozen)
    path = out / "experiment.json"
    if path.exists() and read(path) != {"fingerprint": fp, **frozen}:
        raise ValueError("코드·설정·자료가 기존 실행과 다릅니다. 기존 결과를 보존하고 새 출력 폴더를 쓰세요")
    write(path, {"fingerprint": fp, **frozen})
    return fp


class BudgetExceeded(RuntimeError):
    pass


class Budget:
    """실행 경과시간 합계. 실패한 작업도 비용에 포함하고 재개 때 초기화하지 않는다."""
    def __init__(self, out: Path, initial=False):
        self.path = Path(out) / "budget.json"
        self.cfg = config()["budget"]
        self.initial = initial
        saved = read(self.path) if self.path.exists() else {}
        self.entries = saved.get("entries", [])
        self.pending = saved.get("pending", {})
        self.reserved = sum(p["cap_s"] for p in self.pending.values())
        # An abrupt disconnect can occur after a cache/checkpoint is durable but
        # before its timing is committed. Never reuse that work for free.
        if self.pending:
            for token, p in self.pending.items():
                self.entries.append({"job": p["job"], "seconds": p["cap_s"], "initial": p["initial"],
                                     "status": "interrupted_conservative_cap", "token": token})
            self.pending, self.reserved = {}, 0.0
            self._persist()

    @property
    def used(self):
        return sum(e["seconds"] for e in self.entries)

    @property
    def initial_used(self):
        return sum(e["seconds"] for e in self.entries if e["initial"])

    @property
    def remaining(self):
        total = self.cfg["total_s"] - self.used - self.reserved
        return min(total, self.cfg["initial_s"] - self.initial_used - self.reserved) if self.initial else total

    def check(self, reserve=0.0):
        if reserve > self.remaining or self.remaining <= 0:
            raise BudgetExceeded(f"WEIGHT1 계산 예산 부족: 사용 {self.used:.1f}초, 남은 {max(0,self.remaining):.1f}초")

    def charge(self, name, seconds, status="complete"):
        self.entries.append({"job": name, "seconds": max(0.0, float(seconds)), "initial": self.initial,
                             "status": status})
        self._persist()

    def _persist(self):
        write(self.path, {"limit_s": self.cfg["total_s"], "initial_limit_s": self.cfg["initial_s"],
                          "used_s": self.used, "initial_used_s": self.initial_used,
                          "accounting": self.cfg["accounting"], "entries": self.entries, "pending": self.pending})

    def begin(self, name, cap):
        cap = float(cap)
        self.check(cap)
        token = uuid.uuid4().hex
        self.pending[token] = {"job": name, "cap_s": cap, "initial": self.initial, "started_unix": time.time()}
        self.reserved = sum(p["cap_s"] for p in self.pending.values())
        self._persist()
        return token

    def finish(self, token, seconds, status="complete"):
        pending = self.pending.pop(token)
        self.reserved = sum(p["cap_s"] for p in self.pending.values())
        self.entries.append({"job": pending["job"], "seconds": max(0.0, float(seconds)),
                             "initial": pending["initial"], "status": status, "token": token})
        self._persist()


def groups(part):
    p = config()[part]
    return [(d, t) for d in p["dates"] for t in p["periods"]]


def training_scenarios():
    from focused_data import COUNTS, load_counts
    from focused_train import training_scenarios as available
    excluded = set(config()["train_exclude_dates"])
    rows = [(d, p) for d, p in available(load_counts(COUNTS)) if d not in excluded]
    if len({d for d, _ in rows}) != 11:
        raise ValueError("고정한 학습 날짜 11일을 구성할 수 없습니다")
    return rows


def valid_metrics(m):
    errors = [k for k in ("missing", "unfinished", "changed_by_rules", "mode_action_errors",
                          "collisions_at_204820", "collisions_unexplained") if m.get(k, 0)]
    check = m.get("cost_check", False)
    ok = check.get("passed", False) if isinstance(check, dict) else bool(check)
    if not ok:
        errors.append("cost_check")
    if int(m.get("vehicles", 0)) <= 0 or not math.isfinite(float(m.get("mean_delay_s", math.nan))):
        errors.append("vehicles/delay")
    return errors


def aggregate(rows):
    n = sum(int(m["vehicles"]) for m in rows)
    if not n:
        raise ValueError("평가 대상 차량이 없습니다")
    total = sum(float(m.get("actual_total_delay", m["total_delay"])) for m in rows)
    out = {"vehicles": n, "mean_delay_s": total / n,
           "long_wait_rate": sum(m["long_waits"] for m in rows) / n,
           "invalid": [i for i, m in enumerate(rows) if valid_metrics(m)]}
    from explore_weight_core import LEFTS
    for name, movement in LEFTS.items():
        key = "->".join(movement)
        counts = [m.get("by_movement", {}).get(key, {}) for m in rows]
        count = sum(r.get("vehicles", 0) for r in counts)
        out[name] = (sum(r.get("vehicles", 0) * r.get("mean_delay_s", 0) for r in counts) / count
                     if count else None)
    return out


def guards(candidate, reference):
    checks = {"complete_valid": not candidate["invalid"] and not reference["invalid"],
              "long_waits_not_increased": candidate["long_wait_rate"] <= reference["long_wait_rate"] + 1e-12}
    for name in ("left_east", "left_south"):
        a, b = candidate[name], reference[name]
        checks[name] = a is not None and b is not None and a - b <= config()["gates"]["left_tolerance_s"] + 1e-9
    return checks


def verify_pairs(a, b):
    ka = {(m["date"], m["period"], m["seed"]): m for m in a}
    kb = {(m["date"], m["period"], m["seed"]): m for m in b}
    if len(ka) != len(a) or len(kb) != len(b) or ka.keys() != kb.keys():
        raise ValueError("짝 비교의 날짜·시각·평가 난수 또는 중복 실행이 다릅니다")
    for key in ka:
        for field in ("vehicles", "route_sha256", "cohort_sha256"):
            if field not in ka[key] or ka[key][field] != kb[key].get(field):
                raise ValueError(f"짝 비교의 차량/경로 불일치: {key} {field}")


def pilot_reading(candidate_rows, references):
    c = aggregate(candidate_rows)
    comparisons = {}
    for name, rows in references.items():
        verify_pairs(candidate_rows, rows)
        r = aggregate(rows)
        deltas = {f"{d}_{p}": aggregate([m for m in rows if (m["date"], m["period"]) == (d, p)])["mean_delay_s"]
                  - aggregate([m for m in candidate_rows if (m["date"], m["period"]) == (d, p)])["mean_delay_s"]
                  for d, p in groups("development")}
        checks = {**guards(c, r), "mean_improves": r["mean_delay_s"] > c["mean_delay_s"],
                  "positive_groups": sum(x > 0 for x in deltas.values()) >= config()["gates"]["pilot_positive_groups"]}
        comparisons[name] = {"gain_s": r["mean_delay_s"] - c["mean_delay_s"], "groups": deltas,
                             "checks": checks, "passed": all(checks.values())}
    return {"passed": all(v["passed"] for v in comparisons.values()), "comparisons": comparisons,
            "candidate": c, "interpretation": "개발 평가의 확대 판단용 단서; 최종 성능 증거 아님"}


def confirmation_reading(candidates, references, repeats=None):
    """고정 수요 조건에 조건부로 학습 seed와 driver seed를 독립 재표집한다."""
    cfg = config()
    seeds = cfg["learning"]["train_seeds"]
    drivers = cfg["confirmation"]["eval_seeds"]
    gs = groups("confirmation")
    if set(candidates) != set(seeds):
        raise ValueError("학습 모델 5개가 모두 있어야 합니다")
    wanted = {(d, p, s) for d, p in gs for s in drivers}
    for rows in [*candidates.values(), *references.values()]:
        if {(m["date"], m["period"], m["seed"]) for m in rows} != wanted or len(rows) != len(wanted):
            raise ValueError("확인 평가 실행이 누락되거나 중복됐습니다")
    for rows in candidates.values():
        for ref in references.values():
            verify_pairs(rows, ref)
    lookup = lambda rs: {(m["date"], m["period"], m["seed"]): m for m in rs}
    def matrices(rs):
        idx = lookup(rs)
        return (np.array([[idx[d,p,s].get("actual_total_delay", idx[d,p,s]["total_delay"]) for d,p in gs] for s in drivers]),
                np.array([[idx[d,p,s]["vehicles"] for d,p in gs] for s in drivers]))
    costs = np.stack([matrices(candidates[s])[0] for s in seeds])
    counts = matrices(candidates[seeds[0]])[1]
    rng = np.random.default_rng(cfg["gates"]["bootstrap_seed"])
    nrep = cfg["gates"]["bootstrap_repeats"] if repeats is None else int(repeats)
    if nrep <= 0:
        raise ValueError("재표집 횟수는 양수여야 합니다")
    model_draw = rng.integers(0, len(seeds), (nrep, len(seeds)))
    driver_draw = rng.integers(0, len(drivers), (nrep, len(drivers)))
    comparisons = {}
    all_candidate = aggregate([m for s in seeds for m in candidates[s]])
    for name, rows in references.items():
        ref_cost, _ = matrices(rows)
        gains = []
        for mi, di in zip(model_draw, driver_draw):
            den = float(counts[di].sum())
            gains.append(float((ref_cost[di].sum() - costs[mi][:, di].sum(axis=(1,2)).mean()) / den))
        ci = [float(x) for x in np.percentile(gains, cfg["gates"]["ci_percentiles"])]
        r = aggregate(rows)
        by_model = {s: r["mean_delay_s"] - aggregate(candidates[s])["mean_delay_s"] for s in seeds}
        by_group = {}
        for d, p in gs:
            ref = aggregate([m for m in rows if (m["date"],m["period"]) == (d,p)])
            cand = aggregate([m for s in seeds for m in candidates[s] if (m["date"],m["period"]) == (d,p)])
            by_group[f"{d}_{p}"] = ref["mean_delay_s"] - cand["mean_delay_s"]
        checks = {**guards(all_candidate, r), "mean_improves": r["mean_delay_s"] > all_candidate["mean_delay_s"],
                  "ci_lower_positive": ci[0] > 0,
                  "four_models_positive": sum(v > 0 for v in by_model.values()) >= cfg["gates"]["confirmation_positive_models"],
                  "six_groups_positive": sum(v > 0 for v in by_group.values()) >= cfg["gates"]["confirmation_positive_groups"]}
        comparisons[name] = {"gain_s": r["mean_delay_s"] - all_candidate["mean_delay_s"], "ci95_s": ci,
                             "by_training_seed": by_model, "by_group": by_group, "checks": checks,
                             "passed": all(checks.values())}
    return {"passed": all(v["passed"] for v in comparisons.values()), "candidate": all_candidate,
            "comparisons": comparisons, "bootstrap_repeats": nrep,
            "interpretation": "고정 8수요조건의 추가 개발 확인 평가. 학습 seed5개 한계가 있으며 새 날짜/실제 도로 일반화 아님"}


def job_path(out, part, policy, date, period, seed):
    return Path(out) / "runs" / part / f"{policy}_{date}_{period}_s{seed}"


def _eval_worker(job):
    started = time.monotonic()
    configure()
    from explore_weight_episode import run_episode
    policy = job["mode"]
    if job.get("model_dir"):
        from explore_weight_train import load_model, predict_mode
        model = load_model(Path(job["model_dir"]), fingerprint=job["fingerprint"])
        policy = lambda obs, snapshot: predict_mode(model, obs)
    try:
        r = run_episode(job["date"], job["period"], job["seed"], Path(job["work"]), policy,
                        collect=False, trace=True,
                        max_seconds=max(1, job["max_seconds"] - (time.monotonic()-started) - 5))
        metrics = {**r.metrics, "date": job["date"], "period": job["period"], "seed": job["seed"],
                   "policy": job["policy"], "experiment_fingerprint": job["fingerprint"],
                   "model_identity": job.get("model_identity"), "job_seconds": time.monotonic() - started}
        target = Path(job["target"])
        with gzip.open(target / "decisions.jsonl.gz", "wt") as f:
            for row in r.decisions:
                f.write(json.dumps(plain(row), ensure_ascii=False, allow_nan=False) + "\n")
        write(target / "metrics.json", metrics)
        write(target / "cache.json", {"fingerprint": job["fingerprint"], "model_identity": job.get("model_identity"),
                                       "metrics_sha256": file_hash(target / "metrics.json"),
                                       "decisions_sha256": file_hash(target / "decisions.jsonl.gz")})
        return {"ok": True, "seconds": time.monotonic() - started, "metrics": metrics}
    except Exception as e:
        return {"ok": False, "seconds": time.monotonic() - started, "error": f"{type(e).__name__}: {e}"}


def model_identity(folder):
    from explore_weight_train import checkpoint_identity
    return checkpoint_identity(Path(folder))


def evaluations(out, work, fp, budget, part, policies, workers=2):
    """policies=[(tag,mode|None,modeldir|None)]. 미완료 작업만 실행하고 모든 캐시를 검증한다."""
    import multiprocessing
    cfg = config()[part]
    jobs, results = [], {name: [] for name, _, _ in policies}
    for name, mode, model_dir in policies:
        identity = model_identity(model_dir) if model_dir else None
        for date, period in groups(part):
            for seed in cfg["eval_seeds"]:
                target = job_path(out, part, name, date, period, seed)
                cache = target / "cache.json"
                if cache.exists():
                    saved = read(cache)
                    if (saved["fingerprint"] != fp or saved.get("model_identity") != identity
                            or saved["metrics_sha256"] != file_hash(target / "metrics.json")
                            or saved["decisions_sha256"] != file_hash(target / "decisions.jsonl.gz")):
                        raise ValueError(f"재사용할 실행의 지문·해시 불일치: {target}")
                    results[name].append(read(target / "metrics.json"))
                    continue
                target.mkdir(parents=True, exist_ok=True)
                jobs.append({"date": date, "period": period, "seed": seed, "policy": name, "mode": mode,
                             "model_dir": str(model_dir) if model_dir else None, "model_identity": identity,
                             "work": str(Path(work) / part / target.name), "target": str(target), "fingerprint": fp})
    # Bounded batches: budget reservations include simultaneous jobs; do not submit the entire matrix at once.
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        while jobs:
            remaining = budget.remaining
            estimate = max([r.get("job_seconds", 0) for rs in results.values() for r in rs] or [180.0])
            reserve = max(60.0, 1.5 * estimate)
            batch = jobs[:workers]
            budget.check(len(batch) * reserve)
            jobs = jobs[len(batch):]
            caps = min(1800.0, remaining / len(batch))
            for j in batch:
                j["max_seconds"] = caps
                j["budget_token"] = budget.begin(f"{part}/{Path(j['target']).name}", caps)
            tasks = {pool.submit(_eval_worker, j): j for j in batch}
            errors = []
            for future in as_completed(tasks):
                j = tasks[future]
                try:
                    r = future.result()
                except Exception as e:
                    r = {"ok": False, "seconds": caps, "error": str(e)}
                budget.finish(j["budget_token"], r["seconds"], "complete" if r["ok"] else "failed")
                if not r["ok"]:
                    write(Path(j["target"]) / "failed.json", r)
                    errors.append(r["error"])
                else:
                    results[j["policy"]].append(r["metrics"])
                    print(f"{part}: {j['policy']} {j['date']} {j['period']} seed{j['seed']} "
                          f"지연 {r['metrics']['mean_delay_s']:.4f}초 / 계산 누적 {budget.used/3600:.2f}시간", flush=True)
            if errors:
                raise RuntimeError("; ".join(errors))
            budget.check()
    return results


def check(out, work, fp, budget):
    # Full original P0 vs M0 on the same real scenario, including all cohort vehicles and safety.
    rows = {}
    for name, mode in (("original", None), ("M0", 0)):
        target = Path(out) / "checks" / name
        target.mkdir(parents=True, exist_ok=True)
        budget.check(300)
        job = {"date": "20220811", "period": "1300", "seed": 42, "policy": name, "mode": mode,
               "fingerprint": fp, "target": str(target), "work": str(Path(work) / "checks" / name),
               "max_seconds": min(1800, budget.remaining / 2)}
        token = budget.begin(f"check/{name}", job["max_seconds"])
        r = _eval_worker(job)
        budget.finish(token, r["seconds"], "complete" if r["ok"] else "failed")
        if not r["ok"]:
            raise RuntimeError(r["error"])
        rows[name] = r["metrics"]
    verify_pairs([rows["original"]], [rows["M0"]])
    traces = []
    for name in rows:
        with gzip.open(Path(out) / "checks" / name / "decisions.jsonl.gz", "rt") as f:
            traces.append([json.loads(s)["executed_green"] for s in f])
    checks = {"identical_actions": traces[0] == traces[1],
              "identical_cost": abs(rows["original"]["total_delay"] - rows["M0"]["total_delay"]) < 1e-6,
              "valid_original": not valid_metrics(rows["original"]), "valid_M0": not valid_metrics(rows["M0"])}
    result = {"passed": all(checks.values()), "checks": checks, "rows": rows}
    write(Path(out) / "check.json", result)
    return result


def fixed(out, work, fp, budget, workers):
    rows = evaluations(out, work, fp, budget, "development", [(f"M{i}", i, None) for i in range(5)], workers)
    summaries = {k: aggregate(v) for k, v in rows.items()}
    for rs in rows.values():
        verify_pairs(rows["M0"], rs)
    best = min(range(5), key=lambda i: (round(summaries[f"M{i}"]["mean_delay_s"], 9), i))
    points, active = 0, set()
    for d, p in groups("development"):
        for seed in config()["development"]["eval_seeds"]:
            path = job_path(out, "development", "M0", d, p, seed) / "decisions.jsonl.gz"
            with gzip.open(path, "rt") as f:
                n = sum(bool(json.loads(s).get("mode_disagrees")) for s in f)
            points += n
            if n:
                active.add((d, p))
    checks = {"activity_points": points >= config()["gates"]["activity_points"],
              "activity_groups": len(active) >= config()["gates"]["activity_groups"],
              "valid": all(not v["invalid"] for v in summaries.values())}
    result = {"passed": all(checks.values()), "checks": checks, "best_mode": best,
              "summaries": summaries, "activity_points": points, "activity_groups": sorted(active)}
    write(Path(out) / "fixed_summary.json", result)
    return result


def saved_rows(out, part, policy):
    return [read(job_path(out, part, policy, d, p, s) / "metrics.json")
            for d, p in groups(part) for s in config()[part]["eval_seeds"]]


def train(out, work, fp, budget, seed, target_episodes):
    from explore_weight_episode import run_episode
    from explore_weight_train import train_seed
    cap = min(1800.0, budget.remaining)
    token = budget.begin(f"train{seed}/setup", cap)
    active_at, episode_count = time.monotonic(), 0
    def before():
        nonlocal token, active_at, cap, episode_count
        budget.finish(token, time.monotonic() - active_at)
        token = None
        episode_times = [e["seconds"] for e in budget.entries if "/episode" in e["job"]]
        budget.check(max(180, max(episode_times or [0]) * 1.5))
        cap = min(1800.0, budget.remaining)
        episode_count += 1
        token = budget.begin(f"train{seed}/episode{episode_count}", cap)
        active_at = time.monotonic()
    def episode(*args, **kwargs):
        kwargs["max_seconds"] = max(1, cap - min(90.0, cap * .05))
        args = list(args)
        episode_name = Path(args[3]).name
        args[3] = Path(work) / "training" / f"seed_{seed}" / episode_name
        r = run_episode(*args, **kwargs)
        evidence = Path(out) / "training" / f"seed_{seed}" / episode_name
        evidence.mkdir(parents=True, exist_ok=True)
        write(evidence / "metrics.json", r.metrics)
        with gzip.open(evidence / "decisions.jsonl.gz", "wt") as f:
            for row in r.decisions:
                f.write(json.dumps(plain(row), ensure_ascii=False) + "\n")
        for name in ("tripinfo.xml", "collisions.xml", "cost_ledger.json"):
            with (Path(args[3]) / name).open("rb") as src, gzip.open(evidence / f"{name}.gz", "wb") as dst:
                import shutil
                shutil.copyfileobj(src, dst)
        return r
    status = "failed"
    try:
        result = train_seed(Path(out) / "models" / f"seed_{seed}", training_scenarios(), seed, target_episodes,
                            episode, fingerprint=fp, budget_check=before)
        status = "complete"
        return result
    finally:
        if token is not None:
            budget.finish(token, time.monotonic() - active_at, status)


def pilot(out, work, fp, budget, workers):
    if not read(Path(out) / "check.json")["passed"] or not read(Path(out) / "fixed_summary.json")["passed"]:
        raise ValueError("동등성·고정 모드 단계가 통과되지 않았습니다")
    train(out, work, fp, budget, 42, config()["learning"]["pilot_episodes"])
    stage = Path(out) / "models/seed_42/stages/episode_000042"
    rows = evaluations(out, work, fp, budget, "development", [("pilot42", None, stage)], workers)
    best = read(Path(out) / "fixed_summary.json")["best_mode"]
    result = pilot_reading(rows["pilot42"], {"P0": saved_rows(out, "development", "M0"),
                                           "F": saved_rows(out, "development", f"M{best}")})
    write(Path(out) / "pilot_summary.json", result)
    return result


def forecast(out, budget):
    # Each episode ledger interval includes simulation, gradients, evidence and checkpoint writes.
    times = [e["seconds"] for e in budget.entries if "/episode" in e["job"] and e["status"] == "complete"]
    evals = [e["seconds"] for e in budget.entries if e["job"].startswith("development/")]
    if not times or not evals:
        raise ValueError("실측 학습·평가 시간이 없어 본실험 비용을 예측할 수 없습니다")
    count = config()["learning"]["full_episodes"]
    remaining_episodes = count * 5 - config()["learning"]["pilot_episodes"]
    seconds = config()["budget"]["forecast_margin"] * (remaining_episodes * statistics.fmean(times)
                                                       + 448 * max(evals))
    result = {"passed": seconds <= budget.remaining, "remaining_estimate_s": seconds,
              "remaining_budget_s": budget.remaining, "training_episodes": remaining_episodes,
              "max_confirmation_runs": 448, "margin": config()["budget"]["forecast_margin"]}
    write(Path(out) / "forecast.json", result)
    return result


def full(out, work, fp, budget):
    if not read(Path(out) / "pilot_summary.json")["passed"]:
        raise ValueError("짧은 학습의 확대 기준이 통과되지 않았습니다")
    # A saved accepted forecast freezes the expansion decision, but remaining budget is enforced throughout.
    p = Path(out) / "forecast.json"
    if not p.exists():
        forecast(out, budget)
    if not read(p)["passed"]:
        raise BudgetExceeded("실측 예측상 본실험 전체가 48시간 예산에 들어가지 않습니다")
    records = {}
    for seed in config()["learning"]["train_seeds"]:
        records[seed] = train(out, work, fp, budget, seed, config()["learning"]["full_episodes"])
    write(Path(out) / "full_training.json", records)
    return records


def confirm(out, work, fp, budget, workers):
    from explore_weight_train import validate_checkpoint
    record = read(Path(out) / "full_training.json")
    if set(record) != {str(s) for s in config()["learning"]["train_seeds"]}:
        raise ValueError("본학습 모델 5개가 모두 완료되지 않았습니다")
    for seed in config()["learning"]["train_seeds"]:
        identity = validate_checkpoint(Path(out) / "models" / f"seed_{seed}", fp)
        if identity["episodes"] != config()["learning"]["full_episodes"]:
            raise ValueError("확인 평가에는 마지막 139에피소드 모델만 사용합니다")
    identities = {s: model_identity(Path(out) / "models" / f"seed_{s}") for s in config()["learning"]["train_seeds"]}
    freeze = Path(out) / "confirmation_models.json"
    if freeze.exists() and read(freeze) != plain(identities):
        raise ValueError("확인 평가를 시작한 뒤 모델이 바뀌었습니다")
    write(freeze, identities)
    best = read(Path(out) / "fixed_summary.json")["best_mode"]
    policies = [("P0", None, None)]
    if best:
        policies.append(("F", best, None))
    policies += [(f"RL{s}", None, Path(out) / "models" / f"seed_{s}") for s in identities]
    rows = evaluations(out, work, fp, budget, "confirmation", policies, workers)
    result = confirmation_reading({s: rows[f"RL{s}"] for s in identities},
                                  {"P0": rows["P0"], "F": rows["F"] if best else rows["P0"]})
    write(Path(out) / "confirmation_summary.json", result)
    return result


def summary(out):
    files = ("check", "fixed_summary", "pilot_summary", "forecast", "confirmation_summary")
    state = {name: read(Path(out) / f"{name}.json") for name in files if (Path(out) / f"{name}.json").exists()}
    ledger = Path(out) / "budget.json"
    state["budget"] = read(ledger) if ledger.exists() else None
    write(Path(out) / "summary.json", state)
    lines = ["# WEIGHT1 결과", "", "반복 사용된 날짜의 추가 개발 확인 평가. 실제 도로/새 날짜 일반화 결과가 아니다.", ""]
    for name in files:
        if name in state:
            lines.append(f"- {name}: {'통과' if state[name]['passed'] else '미통과'}")
    if state["budget"]:
        lines.append(f"- 누적 계산: {state['budget']['used_s']/3600:.3f} / 48시간")
    if "fixed_summary" in state:
        lines += ["", "| 고정 모드 | 차량당 지연(초) |", "|---|---:|"]
        lines += [f"| {name} | {v['mean_delay_s']:.6f} |" for name, v in state["fixed_summary"]["summaries"].items()]
    for name in ("pilot_summary", "confirmation_summary"):
        if name in state:
            lines += ["", f"## {name}", ""]
            for ref, c in state[name]["comparisons"].items():
                lines.append(f"- {ref} 대비 줄인 지연: {c['gain_s']:.6f}초; 기준 {'통과' if c['passed'] else '미통과'}")
                if "ci95_s" in c:
                    lines.append(f"  - 95% 재표집 구간: {c['ci95_s']}")
    (Path(out) / "summary.md").write_text("\n".join(lines) + "\n")
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "check", "fixed", "pilot", "forecast", "full", "confirm", "summary", "run"))
    parser.add_argument("--out", required=True)
    parser.add_argument("--work", default="/tmp/weight_v1")
    parser.add_argument("--workers", type=int, default=2, choices=(1, 2))
    args = parser.parse_args()
    out, work = Path(args.out).resolve(), Path(args.work).resolve()
    if " " in str(work):
        raise ValueError("SUMO 작업 경로에는 공백이 없어야 합니다")
    out.mkdir(parents=True, exist_ok=True)
    with (out / ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as e:
            raise SystemExit("같은 출력 폴더의 WEIGHT1 실행이 이미 진행 중입니다") from e
        if args.command == "summary":
            summary(out)
            return
        fp = register(out)
        if args.command == "prepare":
            print(json.dumps({"fingerprint": fp, "train_scenarios": len(training_scenarios())}), flush=True)
            return
        initial = args.command in ("check", "fixed", "pilot", "run")
        budget = Budget(out, initial=initial)
        try:
            if args.command == "run":
                for name, operation in (("check", check), ("fixed_summary", fixed), ("pilot_summary", pilot)):
                    p = out / f"{name}.json"
                    if p.exists():
                        result = read(p)
                    else:
                        result = operation(out, work, fp, budget, *([] if name == "check" else [args.workers]))
                    if not result["passed"]:
                        print(f"{name} 미통과: 이번 경로의 자동 확대를 종료합니다", flush=True)
                        return
                budget.initial = False
                forecast_path = out / "forecast.json"
                cost = read(forecast_path) if forecast_path.exists() else forecast(out, budget)
                if not cost["passed"]:
                    print("예산 부족: 확대를 보류합니다", flush=True)
                    return
                full(out, work, fp, budget)
                confirm(out, work, fp, budget, args.workers)
            elif args.command == "forecast":
                forecast(out, budget)
            elif args.command == "full":
                full(out, work, fp, budget)
            else:
                function = {"check": check, "fixed": fixed, "pilot": pilot, "confirm": confirm}[args.command]
                function(out, work, fp, budget, *([] if args.command == "check" else [args.workers]))
        except BudgetExceeded as e:
            write(out / "stopped.json", {"reason": "budget", "message": str(e)})
            print(str(e), flush=True)
        finally:
            summary(out)


if __name__ == "__main__":
    main()
