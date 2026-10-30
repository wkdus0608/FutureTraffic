"""WEIGHT1_NIGHT: 밤샘 Colab 묶음 (탐색 6쌍 비교 · M4 주변 가중치 · M4 출발 듀얼링 DQN).

명령: prepare · smoke · checks · train · evaluate · summary.  모든 명령은 같은 --out을 재개할 수 있다.
기존 WEIGHT1 모듈은 고치지 않고 가져다 쓴다.  Mac에서는 실제 SUMO를 실행하지 않는다(테스트는 가짜 에피소드).
결과는 모두 '반복 사용한 날짜의 추가 개발 평가'다.  판정 규칙은 config/explore_night_v1.json에 실행 전에 고정했다.
"""
from __future__ import annotations

import argparse
import fcntl
import gzip
import importlib.metadata
import json
import math
import multiprocessing
import os
import shutil
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import explore_weight as ew

PROJECT = Path(__file__).resolve().parents[1]
CONFIG = PROJECT / "config/explore_night_v1.json"
SOURCES = ("src/explore_night.py", "src/explore_night_train.py", "src/explore_weight.py",
           "src/explore_weight_core.py", "src/explore_weight_episode.py", "src/explore_weight_train.py",
           "src/explore_pred.py", "src/explore_online.py", "src/explore_rollout.py", "src/explore_choice.py",
           "src/focused_train.py", "src/focused_eval.py", "config/collision_exceptions.json",
           "config/explore_weight_v1.json")
PACKAGES = ("numpy", "torch", "stable-baselines3", "sumo-rl", "eclipse-sumo")
EXIT_LIMIT = 3


def config() -> dict:
    return ew.read(CONFIG)


def stamp(*parts) -> None:
    print(time.strftime("%H:%M:%S"), *parts, flush=True)


def register(out: Path) -> str:
    """Freeze code, data, packages and the night's rules; refuse any silent change."""
    ew.configure()
    from focused_data import ENV_FILES, environment_fingerprint

    cfg = config()
    ew.validate_implementation(ew.config())
    env = environment_fingerprint()
    if env != cfg["environment"]["fingerprint"]:
        raise ValueError(f"잠근 환경 지문 불일치: {env}")
    if ew.file_hash(PROJECT / "src/explore_pred.py")[:12] != cfg["environment"]["p0_code"]:
        raise ValueError("기존 P0 코드가 잠근 코드와 다릅니다")
    frozen = {"config": cfg, "environment_fingerprint": env,
              "files": {s: ew.file_hash(PROJECT / s) for s in sorted(set(SOURCES) | set(ENV_FILES))},
              "packages": {p: importlib.metadata.version(p) for p in PACKAGES}}
    fingerprint = ew.digest(frozen)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / ".register.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = out / "experiment.json"
        record = {"fingerprint": fingerprint, **ew.plain(frozen)}
        if path.exists() and ew.read(path) != record:
            if (out / "runs").exists() or (out / "models").exists():
                raise ValueError("코드·설정·자료가 기존 실행과 다릅니다. 결과가 있는 폴더는 보존하고 새 출력 폴더를 쓰세요")
            stamp("아직 결과가 없는 이전 등록(점검·스모크만 실행됨)을 새 코드의 등록으로 바꿉니다.")
            path.unlink()
        if not path.exists():
            ew.write(path, record)
    return fingerprint


class Ledger:
    """역할별 경과시간 장부. 병렬 작업 시간은 합산하고, 이전 실험의 누적 사용량을 함께 본다."""

    def __init__(self, out: Path, role: str):
        self.dir = Path(out) / "ledger"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.role, self.path = role, self.dir / f"{role}.json"
        self.cfg = config()["budget"]
        self.seconds = float(ew.read(self.path)["seconds"]) if self.path.exists() else 0.0
        prior = self.dir / "prior.json"
        if not prior.exists():
            raise FileNotFoundError("prepare를 먼저 실행해야 합니다(이전 실험의 누적 사용량 기록 없음)")
        self.prior_s = float(ew.read(prior)["prior_used_s"])

    def add(self, seconds: float) -> None:
        self.seconds += max(0.0, float(seconds))
        ew.write(self.path, {"role": self.role, "seconds": self.seconds, "updated_unix": time.time()})

    def night_s(self) -> float:
        return sum(float(ew.read(p)["seconds"]) for p in self.dir.glob("*.json") if p.name != "prior.json")

    def check(self, reserve: float = 0.0) -> None:
        night = self.night_s()
        if night + reserve > self.cfg["night_cap_s"]:
            raise ew.BudgetExceeded(f"이번 밤샘 상한 초과 직전: 사용 {night:.0f}초 / {self.cfg['night_cap_s']}초")
        if self.prior_s + night + reserve > self.cfg["total_cap_s"] - self.cfg["stop_margin_s"]:
            raise ew.BudgetExceeded(f"전체 96시간 한도 직전: 이전 {self.prior_s:.0f}초 + 이번 {night:.0f}초")


def deadline_unix(args) -> float:
    return time.time() + float(args.wall_hours) * 3600


# ----------------------------------------------------------------------------- 평가 작업

def target_dir(out: Path, part: str, tag: str, date: str, period: str, seed: int) -> Path:
    return Path(out) / "runs" / part / f"{tag}_{date}_{period}_s{seed}"


def eval_jobs(out: Path, work: Path, fp: str, part: str, tag: str, conditions, *, mode=None,
              factors=None, model_dir=None, model_identity=None) -> list[dict]:
    jobs = []
    for date, period, seed in conditions:
        target = target_dir(out, part, tag, date, period, seed)
        jobs.append({"part": part, "tag": tag, "date": date, "period": period, "seed": int(seed), "mode": mode,
                     "factors": list(factors) if factors else None, "model_dir": str(model_dir) if model_dir else None,
                     "model_identity": model_identity, "target": str(target),
                     "work": str(Path(work) / part / target.name), "fingerprint": fp})
    return jobs


def cached(job: dict) -> dict | None:
    target = Path(job["target"])
    cache = target / "cache.json"
    if not cache.exists() or not (target / "metrics.json").exists():
        return None
    saved = ew.read(cache)
    if (saved.get("fingerprint") != job["fingerprint"] or saved.get("model_identity") != job.get("model_identity")
            or saved.get("metrics_sha256") != ew.file_hash(target / "metrics.json")):
        raise ValueError(f"재사용할 실행의 지문·해시 불일치: {target}")
    return ew.read(target / "metrics.json")


def _eval_worker(job: dict) -> dict:
    started = time.monotonic()
    try:
        ew.configure()
        import explore_weight_core as core
        from explore_weight_episode import run_episode

        policy = job["mode"]
        if job.get("factors"):
            # 고정 정책은 모드 4의 가중치만 바꿔 실행한다. 모드 0~3과 P0 코드는 그대로다.
            core.MODE_FACTORS = tuple(core.MODE_FACTORS[:4]) + ((float(job["factors"][0]), float(job["factors"][1])),)
            policy = 4
        if job.get("model_dir"):
            import explore_night_train as nt
            model = nt.load_model(Path(job["model_dir"]), fingerprint=job["fingerprint"])
            policy = lambda observation, snapshot: nt.predict_mode(model, observation)
        work = Path(job["work"])
        result = run_episode(job["date"], job["period"], job["seed"], work, policy, collect=False, trace=True,
                             max_seconds=float(job.get("max_seconds", 3600)))
        metrics = {**result.metrics, "part": job["part"], "policy": job["tag"], "factors": job.get("factors"),
                   "experiment_fingerprint": job["fingerprint"], "model_identity": job.get("model_identity"),
                   "job_seconds": time.monotonic() - started}
        target = Path(job["target"])
        target.mkdir(parents=True, exist_ok=True)
        with gzip.open(target / "decisions.jsonl.gz", "wt") as stream:
            for row in result.decisions:
                stream.write(json.dumps(ew.plain(row), ensure_ascii=False, allow_nan=False) + "\n")
        ew.write(target / "metrics.json", metrics)
        ew.write(target / "cache.json", {"fingerprint": job["fingerprint"], "model_identity": job.get("model_identity"),
                                         "metrics_sha256": ew.file_hash(target / "metrics.json"),
                                         "decisions_sha256": ew.file_hash(target / "decisions.jsonl.gz")})
        shutil.rmtree(work, ignore_errors=True)
        return {"ok": True, "seconds": time.monotonic() - started, "name": target.name}
    except BaseException as error:               # a failed run must not stop the others
        shutil.rmtree(job.get("work", ""), ignore_errors=True) if job.get("work") else None
        return {"ok": False, "seconds": time.monotonic() - started, "name": Path(job["target"]).name,
                "error": f"{type(error).__name__}: {error}"}


def _process_pool(workers: int):
    return ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"))


def run_jobs(jobs: list[dict], workers: int, ledger: Ledger, deadline: float, *,
             executor_factory=_process_pool, worker=None) -> dict:
    worker = worker or _eval_worker
    pending = [j for j in jobs if cached(j) is None]
    stamp(f"평가 작업 {len(jobs)}개 중 {len(pending)}개 실행 (재사용 {len(jobs) - len(pending)}개), 작업자 {workers}개")
    done, failed, stopped = 0, [], None
    if not pending:
        return {"done": 0, "failed": [], "stopped": None, "unsubmitted": 0}
    with executor_factory(workers) as pool:
        running = {}
        while pending or running:
            while pending and len(running) < workers and stopped is None:
                if time.time() >= deadline:
                    stopped = "wall_time"
                    break
                try:
                    ledger.check(reserve=600.0 * (len(running) + 1))
                except ew.BudgetExceeded as error:
                    stopped = f"budget: {error}"
                    break
                job = pending.pop(0)
                running[pool.submit(worker, job)] = job
            if not running:
                break
            finished, _ = wait(list(running), return_when=FIRST_COMPLETED)
            for future in finished:
                job = running.pop(future)
                try:
                    result = future.result()
                except Exception as error:                      # worker process died
                    result = {"ok": False, "seconds": 0.0, "name": Path(job["target"]).name,
                              "error": f"{type(error).__name__}: {error}"}
                ledger.add(result["seconds"])
                if result["ok"]:
                    done += 1
                    stamp(f"완료 {result['name']} ({result['seconds']:.0f}초)")
                else:
                    failed.append({"name": result["name"], "error": result["error"]})
                    stamp(f"실패 {result['name']}: {result['error']}")
    return {"done": done, "failed": failed, "stopped": stopped, "unsubmitted": len(pending)}


def dev_conditions(cfg: dict | None = None):
    d = (cfg or config())["n2_static_sweep"]["development"]
    return [(date, period, seed) for date in d["dates"] for period in d["periods"] for seed in d["eval_seeds"]]


def check_jobs(out: Path, work: Path, fp: str) -> list[dict]:
    cfg = config()
    jobs = []
    pairs = [tuple(p) for p in cfg["n1_search_pairs"]["pairs"]]
    jobs += eval_jobs(out, work, fp, "n1", "p0", pairs, mode=None)
    jobs += eval_jobs(out, work, fp, "n1", "m4", pairs, mode=4)
    repro = [tuple(c) for c in cfg["n2_static_sweep"]["reproduction_checks"]]
    jobs += eval_jobs(out, work, fp, "repro", "p0", repro, mode=None)
    jobs += eval_jobs(out, work, fp, "repro", "m4", repro, mode=4)
    for east, south in cfg["n2_static_sweep"]["factors"]:
        jobs += eval_jobs(out, work, fp, "n2", f"w{east:g}_{south:g}", dev_conditions(cfg), factors=(east, south))
    return jobs


# ----------------------------------------------------------------------------- 명령

def parent_folder(args) -> Path:
    return Path(args.parent) if args.parent else Path(args.out).parent / config()["parent"]["folder"]


def cmd_prepare(args) -> int:
    out = Path(args.out)
    fp = register(out)
    parent = parent_folder(args)
    budget_file = parent / config()["parent"]["budget_file"]
    prior = ew.read(budget_file)
    ledger_dir = out / "ledger"
    ledger_dir.mkdir(parents=True, exist_ok=True)
    path = ledger_dir / "prior.json"
    if path.exists():
        previous = ew.read(path)["prior_used_s"]
        if abs(previous - float(prior["used_s"])) > 1e-6:
            stamp(f"주의: 이전 실험 장부가 처음 기록({previous:.0f}초)과 다릅니다. 처음 기록을 유지합니다.")
    else:
        ew.write(path, {"prior_used_s": float(prior["used_s"]), "source": str(budget_file), "limit_s": prior.get("limit_s")})
    prior_s = float(ew.read(path)["prior_used_s"])
    refs = sorted((parent / config()["parent"]["reference_runs"]).glob("*/metrics.json"))
    stamp(f"등록 완료. 지문 {fp[:12]}…, 이전 사용 {prior_s / 3600:.2f}시간, 기준 실행 {len(refs)}개")
    if len(refs) != 32:
        raise ValueError("부모 개발 기준 실행(P0·M4 각 16개)이 32개가 아닙니다")
    return 0


def cmd_checks(args) -> int:
    out, work = Path(args.out), Path(args.work)
    fp = register(out)
    ledger = Ledger(out, "checks")
    jobs = check_jobs(out, work, fp)
    result = run_jobs(jobs, int(args.workers), ledger, deadline_unix(args))
    ew.write(out / "status" / "checks.json", {**result, "jobs": len(jobs), "updated_unix": time.time()})
    stamp("checks 종료:", result)
    return EXIT_LIMIT if result["stopped"] else (1 if result["failed"] else 0)


def make_runner(seed: int, work: Path, ledger: Ledger, deadline: float):
    from explore_weight_episode import run_episode

    def episode(date, period, eval_seed, episode_work, policy, **kwargs):
        local = Path(work) / f"seed_{seed}" / Path(episode_work).name
        began = time.monotonic()
        try:
            result = run_episode(date, period, eval_seed, local, policy, max_seconds=3600.0, **kwargs)
        finally:
            ledger.add(time.monotonic() - began)
            shutil.rmtree(local, ignore_errors=True)
        return result
    return episode


def cmd_train(args) -> int:
    import explore_night_train as nt

    out, work, seed = Path(args.out), Path(args.work), int(args.seed)
    cfg = config()["n3_learner"]
    if seed not in cfg["seeds"]:
        raise ValueError(f"학습 seed는 {cfg['seeds']} 중 하나여야 합니다")
    fp = register(out)
    ledger = Ledger(out, f"train_{seed}")
    deadline = deadline_unix(args)
    status = out / "status" / f"train_{seed}.json"

    def publish(state, **extra):
        ew.write(status, {"seed": seed, "state": state, "updated_unix": time.time(), **extra})

    def budget_check():
        if time.time() >= deadline:
            raise ew.BudgetExceeded("wall time limit")
        ledger.check(reserve=900.0)

    scenarios = ew.training_scenarios()
    runner = make_runner(seed, work, ledger, deadline)
    model_out = out / "models" / f"seed_{seed}"
    try:
        for stage in cfg["stages"]:
            stamp(f"seed {seed}: {stage}에피소드까지 학습")
            publish("training", target=stage)
            nt.train_seed(model_out, scenarios, seed, int(stage), runner, fingerprint=fp, budget_check=budget_check)
            publish("stage_done", completed=stage)
        publish("complete", completed=cfg["stages"][-1])
        return 0
    except ew.BudgetExceeded as error:
        publish("limit", error=str(error))
        stamp(f"seed {seed}: 한도로 멈춤 — {error}")
        return EXIT_LIMIT
    except BaseException as error:
        publish("failed", error=f"{type(error).__name__}: {error}")
        raise


def stage_dirs(out: Path):
    cfg = config()["n3_learner"]
    for seed in cfg["seeds"]:
        for stage in cfg["stages"]:
            folder = Path(out) / "models" / f"seed_{seed}" / "stages" / f"episode_{stage:06d}"
            if (folder / "manifest.json").exists():
                yield seed, stage, folder


def cmd_evaluate(args) -> int:
    import explore_night_train as nt

    out, work = Path(args.out), Path(args.work)
    fp = register(out)
    ledger = Ledger(out, "evaluate")
    jobs = []
    for seed, stage, folder in stage_dirs(out):
        identity = nt.checkpoint_identity(folder, fp)
        jobs += eval_jobs(out, work, fp, "n3", f"w2l_s{seed}_e{stage}", dev_conditions(), model_dir=folder,
                          model_identity=identity)
    if not jobs:
        stamp("평가할 학습 단계 모델이 없습니다.")
        return 1
    result = run_jobs(jobs, int(args.workers), ledger, deadline_unix(args))
    ew.write(out / "status" / "evaluate.json", {**result, "jobs": len(jobs), "updated_unix": time.time()})
    stamp("evaluate 종료:", result)
    return EXIT_LIMIT if result["stopped"] else (1 if result["failed"] else 0)


def cmd_smoke(args) -> int:
    """실제 SUMO로 3개 에피소드를 돌려 각 경로(고정 가중치·학습·학습 모델 평가)를 미리 확인한다."""
    import explore_night_train as nt
    from explore_weight_episode import run_episode

    out, work = Path(args.out), Path(args.work)
    fp = register(out)
    parent = parent_folder(args)
    cfg = config()
    smoke = out / "smoke"
    shutil.rmtree(smoke, ignore_errors=True)
    smoke.mkdir(parents=True)
    report = {"started_unix": time.time(), "fingerprint": fp, "steps": {}}
    date, period, seed = cfg["n2_static_sweep"]["reproduction_checks"][0]

    def reference(tag):
        path = parent / cfg["parent"]["reference_runs"] / f"{tag}_{date}_{period}_s{seed}" / "metrics.json"
        return ew.read(path)

    began = time.monotonic()
    stamp(f"1/3 고정 M4 에피소드({date} {period} seed {seed}) — 기존 기준값과 같아야 합니다")
    result = run_episode(date, period, seed, Path(work) / "smoke" / "m4", 4, collect=False, trace=True, max_seconds=1800)
    expected = reference("M4")["mean_delay_s"]
    report["steps"]["m4_reproduction"] = {"new": result.metrics["mean_delay_s"], "reference": expected,
                                          "difference": result.metrics["mean_delay_s"] - expected,
                                          "valid_errors": ew.valid_metrics(result.metrics), "seconds": time.monotonic() - began}
    stamp("  결과:", report["steps"]["m4_reproduction"])
    shutil.rmtree(Path(work) / "smoke" / "m4", ignore_errors=True)

    began = time.monotonic()
    stamp("2/3 학습 1에피소드(저장·불러오기 포함)")
    scenarios = ew.training_scenarios()
    class Quiet:                                         # 장부에 쓰지 않는 스모크용
        def add(self, seconds): pass
    runner = make_runner(999, Path(work) / "smoke", Quiet(), time.time() + 3600)
    summary = nt.train_seed(smoke / "model", scenarios, 999, 1, runner, fingerprint=fp)
    report["steps"]["train_one_episode"] = {k: summary[k] for k in ("episodes", "control_steps", "updates", "discarded", "seconds")}
    stamp("  결과:", report["steps"]["train_one_episode"])

    began = time.monotonic()
    stamp("3/3 학습된 모델로 평가 에피소드 1개")
    model = nt.load_model(smoke / "model" / "stages" / "episode_000001", fp)
    result = run_episode(date, period, seed, Path(work) / "smoke" / "model", lambda o, s: nt.predict_mode(model, o),
                         collect=False, trace=True, max_seconds=1800)
    report["steps"]["model_evaluation"] = {"mean_delay_s": result.metrics["mean_delay_s"],
                                           "valid_errors": ew.valid_metrics(result.metrics), "seconds": time.monotonic() - began}
    stamp("  결과:", report["steps"]["model_evaluation"])
    shutil.rmtree(Path(work) / "smoke", ignore_errors=True)
    problems = []
    if abs(report["steps"]["m4_reproduction"]["difference"]) > 1e-6:
        problems.append("고정 M4가 기존 기준값을 재현하지 못함(SUMO·환경 차이 가능)")
    for name, step in report["steps"].items():
        if step.get("valid_errors"):
            problems.append(f"{name}: {step['valid_errors']}")
    report["problems"], report["passed"] = problems, not problems
    ew.write(smoke / "report.json", report)
    stamp("스모크 통과" if not problems else f"스모크 문제: {problems}")
    return 0 if not problems else 1


# ----------------------------------------------------------------------------- 요약

def load_runs(folder: Path) -> list[dict]:
    rows = []
    for path in sorted(Path(folder).glob("*/metrics.json")):
        rows.append(ew.read(path))
    return rows


def wmean(rows) -> float | None:
    rows = list(rows)
    if not rows:
        return None
    return sum(float(m.get("actual_total_delay", m["total_delay"])) for m in rows) / sum(int(m["vehicles"]) for m in rows)


def key(m) -> tuple:
    return (m["date"], m["period"], int(m["seed"]))


def summary_data(out: Path, parent: Path) -> dict:
    cfg = config()
    refs = {"P0": {}, "M4": {}}
    for tag in refs:
        for path in sorted((parent / cfg["parent"]["reference_runs"]).glob(f"{tag}_*/metrics.json")):
            m = ew.read(path)
            refs[tag][key(m)] = m
    runs = {part: load_runs(Path(out) / "runs" / part) if (Path(out) / "runs" / part).exists() else []
            for part in ("n1", "repro", "n2", "n3")}
    data = {"fingerprint": ew.read(Path(out) / "experiment.json")["fingerprint"] if (Path(out) / "experiment.json").exists() else None}

    # 재현 확인 -------------------------------------------------------------------
    reproduction = []
    for m in runs["repro"]:
        ref = refs["P0" if m["policy"] == "p0" else "M4"].get(key(m))
        reproduction.append({"policy": m["policy"], "condition": list(key(m)), "new": m["mean_delay_s"],
                             "reference": ref["mean_delay_s"] if ref else None,
                             "difference": m["mean_delay_s"] - ref["mean_delay_s"] if ref else None})
    data["reproduction"] = reproduction
    data["reproduction_ok"] = bool(reproduction) and all(r["difference"] is not None and abs(r["difference"]) < 1e-6
                                                         for r in reproduction)

    # N1 -------------------------------------------------------------------------
    n1 = cfg["n1_search_pairs"]
    rows, p0 = [], {key(m): m for m in runs["n1"] if m["policy"] == "p0"}
    m4 = {key(m): m for m in runs["n1"] if m["policy"] == "m4"}
    for date, period, seed in map(tuple, n1["pairs"]):
        name = f"{date}_{period}_s{seed}"
        k = (date, period, int(seed))
        stored_p0, stored_search = n1["stored"]["p0"][name], n1["stored"]["search"][name]
        row = {"pair": name, "stored_p0": stored_p0, "stored_search": stored_search,
               "search_gain": stored_p0 - stored_search}
        if k in p0:
            row["new_p0"] = p0[k]["mean_delay_s"]
            row["p0_reproduced"] = abs(row["new_p0"] - stored_p0) <= n1["reproduction_tolerance_s"]
        if k in p0 and k in m4:
            row["m4_gain"] = p0[k]["mean_delay_s"] - m4[k]["mean_delay_s"]
            row["new_m4"] = m4[k]["mean_delay_s"]
            row["search_minus_m4"] = row["search_gain"] - row["m4_gain"]
        rows.append(row)
    complete = [r for r in rows if "search_minus_m4" in r]
    data["n1"] = {"rows": rows, "complete": len(complete)}
    if complete:
        data["n1"].update({"mean_search_gain": sum(r["search_gain"] for r in complete) / len(complete),
                           "mean_m4_gain": sum(r["m4_gain"] for r in complete) / len(complete),
                           "mean_search_minus_m4": sum(r["search_minus_m4"] for r in complete) / len(complete),
                           "p0_reproduced_all": all(r.get("p0_reproduced") for r in complete)})
        if len(complete) < len(rows):
            verdict = "6쌍 중 일부만 끝나 판정하지 않음"
        elif data["n1"]["mean_search_minus_m4"] >= n1["headroom_min_s"]:
            verdict = "이 틀에서 M4 위의 여지 단서(탐색 이득이 M4 이득보다 0.5초 이상 큼)"
        else:
            verdict = "M4 위의 여지 근거를 못 얻음(탐색 이득 − M4 이득 < 0.5초)"
        data["n1"]["verdict"] = verdict

    # N2 -------------------------------------------------------------------------
    n2 = cfg["n2_static_sweep"]
    dev_keys = set(dev_conditions(cfg))
    ref_p0 = wmean(refs["P0"].get(k) for k in dev_keys if refs["P0"].get(k)) if len(refs["P0"]) == 16 else None
    ref_m4 = wmean(refs["M4"].get(k) for k in dev_keys if refs["M4"].get(k)) if len(refs["M4"]) == 16 else None
    data["n2"] = {"p0_16": ref_p0, "m4_16": ref_m4, "configs": []}
    for east, south in n2["factors"]:
        tag = f"w{east:g}_{south:g}"
        mine = [m for m in runs["n2"] if m["policy"] == tag]
        row = {"factors": [east, south], "runs": len(mine)}
        if mine:
            agg = ew.aggregate(mine)
            row.update({"mean_delay_s": wmean(mine), "left_east": agg["left_east"], "left_south": agg["left_south"],
                        "invalid": len(agg["invalid"]), "long_wait_rate": agg["long_wait_rate"]})
            if len(mine) == 16 and ref_m4 is not None:
                row["gain_vs_p0"] = ref_p0 - row["mean_delay_s"]
                row["gain_vs_m4"] = ref_m4 - row["mean_delay_s"]
                row["verdict"] = ("확인 평가를 해 볼 만한 단서" if row["gain_vs_m4"] >= n2["worth_confirming_s"]
                                  else "M4와 비슷하거나 나쁨")
        data["n2"]["configs"].append(row)

    # N3 -------------------------------------------------------------------------
    n3 = cfg["n3_learner"]
    data["n3"] = {"baseline": n3["baseline_dev_means"], "stages": {}}
    matched = lambda ms: [m for m in ms if int(m["seed"]) in (42, 43)]
    for stage in n3["stages"]:
        per_seed = {}
        for seed in n3["seeds"]:
            tag = f"w2l_s{seed}_e{stage}"
            mine = [m for m in runs["n3"] if m["policy"] == tag]
            if mine:
                per_seed[seed] = {"runs": len(mine), "mean_delay_s": wmean(mine),
                                  "matched_8": wmean(matched(mine)) if len(matched(mine)) == 8 else None,
                                  "invalid": len(ew.aggregate(mine)["invalid"]),
                                  "choice_points": sum(m.get("mode_disagreement_points", 0) for m in mine) / len(mine)}
        entry = {"per_seed": per_seed}
        full = {s: v for s, v in per_seed.items() if v["runs"] == 16}
        if full and ref_m4 is not None:
            allrows = [m for s in full for m in runs["n3"] if m["policy"] == f"w2l_s{s}_e{stage}"]
            entry["mean_delay_s"] = wmean(allrows)
            entry["gain_vs_p0"] = ref_p0 - entry["mean_delay_s"]
            entry["gain_vs_m4"] = ref_m4 - entry["mean_delay_s"]
            entry["seeds_better_than_m4"] = sum(1 for v in full.values() if ref_m4 - v["mean_delay_s"] > 0)
            entry["seeds"] = len(full)
            if stage == n3["stages"][-1]:
                if len(full) < len(n3["seeds"]):
                    entry["verdict"] = "학습 seed 3개가 모두 끝나지 않아 판정하지 않음"
                elif entry["gain_vs_m4"] >= n3["verdict_margin_s"] and entry["seeds_better_than_m4"] >= n3["min_seeds_better"]:
                    entry["verdict"] = "M4 초과 단서"
                elif entry["gain_vs_m4"] <= -n3["verdict_margin_s"]:
                    entry["verdict"] = "M4보다 나쁨"
                else:
                    entry["verdict"] = f"M4 수준(±{n3['verdict_margin_s']:g}초 안)"
        data["n3"]["stages"][str(stage)] = entry

    # 학습 진행·장부 ---------------------------------------------------------------
    training = {}
    for seed in n3["seeds"]:
        run = Path(out) / "models" / f"seed_{seed}" / "run.json"
        if run.exists():
            r = ew.read(run)
            training[seed] = {"episodes": r["episodes"], "control_steps": r["control_steps"], "updates": r["updates"],
                              "discarded": r.get("discarded", 0)}
    data["training"] = training
    ledger_dir = Path(out) / "ledger"
    night = sum(float(ew.read(p)["seconds"]) for p in ledger_dir.glob("*.json") if p.name != "prior.json") if ledger_dir.exists() else 0.0
    prior = float(ew.read(ledger_dir / "prior.json")["prior_used_s"]) if (ledger_dir / "prior.json").exists() else None
    data["budget"] = {"night_s": night, "prior_s": prior, "total_cap_s": cfg["budget"]["total_cap_s"]}
    invalid = {part: sum(1 for m in ms if ew.valid_metrics(m)) for part, ms in runs.items()}
    data["integrity"] = {"runs": {p: len(ms) for p, ms in runs.items()}, "runs_with_quality_flags": invalid}
    return data


def fmt(x, digits=3, sign=False) -> str:
    if x is None:
        return "-"
    return f"{x:+.{digits}f}" if sign else f"{x:.{digits}f}"


def render(data: dict) -> str:
    lines = ["# WEIGHT1_NIGHT 결과", "",
             "반복 사용한 날짜의 추가 개발 평가다. 새 날짜·실제 도로 일반화나 확인 평가 결과가 아니다. 판정 규칙은 실행 전에 고정했다.", ""]
    integ = data["integrity"]
    lines += [f"- 실행 수: {integ['runs']}, 품질 플래그가 있는 실행: {integ['runs_with_quality_flags']}",
              f"- 이번 밤샘 사용 {data['budget']['night_s'] / 3600:.2f}작업시간, 이전 실험 누적 "
              f"{(data['budget']['prior_s'] or 0) / 3600:.2f} / {data['budget']['total_cap_s'] / 3600:.0f}시간", ""]
    lines += ["## 재현 확인 (이번 Colab이 이전 기준값을 그대로 재현하는가)", ""]
    if data["reproduction"]:
        for r in data["reproduction"]:
            lines.append(f"- {r['policy']} {r['condition']}: 새 {fmt(r['new'], 6)} / 기준 {fmt(r['reference'], 6)} / 차이 {fmt(r['difference'], 9, True)}")
        lines.append(f"- 판정: {'재현됨' if data['reproduction_ok'] else '재현 불일치 — 아래 비교의 신뢰가 낮음'}")
    else:
        lines.append("- 아직 없음")
    n1 = data["n1"]
    lines += ["", "## N1. 탐색 제어 6쌍에서 M4와 비교", "",
              "| 쌍 | P0(기존) | P0(새) | 탐색 이득 | M4 이득 | 탐색−M4 |", "|---|---:|---:|---:|---:|---:|"]
    for r in n1["rows"]:
        lines.append(f"| {r['pair']} | {fmt(r['stored_p0'])} | {fmt(r.get('new_p0'))} | {fmt(r['search_gain'], 3, True)} | "
                     f"{fmt(r.get('m4_gain'), 3, True)} | {fmt(r.get('search_minus_m4'), 3, True)} |")
    if n1["complete"]:
        lines += ["", f"- 평균: 탐색 이득 {fmt(n1['mean_search_gain'], 3, True)}, M4 이득 {fmt(n1['mean_m4_gain'], 3, True)}, "
                      f"탐색−M4 {fmt(n1['mean_search_minus_m4'], 3, True)}초 (쌍 {n1['complete']}개, P0 재현 "
                      f"{'모두 확인' if n1['p0_reproduced_all'] else '일부 불일치'})",
                  f"- 판정: **{n1.get('verdict')}**"]
    n2 = data["n2"]
    lines += ["", "## N2. M4 주변 고정 가중치 (개발 16조건, 차량당 지연 초)", "",
              f"- 기준: P0 {fmt(n2['p0_16'], 6)}, M4(2,2) {fmt(n2['m4_16'], 6)}", "",
              "| 가중치(동,남) | 실행 | 평균 지연 | M4 대비 | 동쪽 좌회전 | 남쪽 좌회전 | 판정 |", "|---|---:|---:|---:|---:|---:|---|"]
    for r in n2["configs"]:
        lines.append(f"| {r['factors'][0]:g}, {r['factors'][1]:g} | {r['runs']}/16 | {fmt(r.get('mean_delay_s'), 4)} | "
                     f"{fmt(r.get('gain_vs_m4'), 3, True)} | {fmt(r.get('left_east'), 1)} | {fmt(r.get('left_south'), 1)} | {r.get('verdict', '-')} |")
    n3 = data["n3"]
    base = n3["baseline"]
    lines += ["", "## N3. M4에서 출발하는 듀얼링 DQN (개발 16조건)", "",
              f"- 비교값: P0 {base['p0_16']}, M4 {base['m4_16']}; 기존 WEIGHT1(평가 seed 42·43 8회) 42에피소드 {base['weight1_ep42_seeds_42_43']}, "
              f"139에피소드 {base['weight1_ep139_seeds_42_43']}, 417에피소드 {base['weight1_ep417_seeds_42_43']} (M4 같은 8회: {base['m4_seeds_42_43']})", ""]
    for stage, entry in n3["stages"].items():
        lines.append(f"### {stage}에피소드")
        for seed, v in entry["per_seed"].items():
            lines.append(f"- seed {seed}: {v['runs']}/16회 평균 {fmt(v['mean_delay_s'], 4)}, 같은 8회(평가 seed 42·43) {fmt(v['matched_8'], 4)}, "
                         f"선택 지점 평균 {v['choice_points']:.1f}, 품질 플래그 {v['invalid']}")
        if "mean_delay_s" in entry:
            lines.append(f"- 합산 평균 {fmt(entry['mean_delay_s'], 4)} → P0 대비 {fmt(entry['gain_vs_p0'], 3, True)}, M4 대비 "
                         f"{fmt(entry['gain_vs_m4'], 3, True)}초, M4보다 짧은 seed {entry['seeds_better_than_m4']}/{entry['seeds']}")
        if "verdict" in entry:
            lines.append(f"- 판정: **{entry['verdict']}**")
    lines += ["", f"- 학습 진행: {data['training']}", "",
              "해석 주의: 학습 seed 3개의 개발 평가이며 확인 전에는 성능 주장에 쓰지 않는다. 판정 규칙·조건·seed는 결과를 보고 바꾸지 않는다."]
    return "\n".join(lines) + "\n"


def cmd_summary(args) -> int:
    out = Path(args.out)
    parent = parent_folder(args)
    data = summary_data(out, parent)
    ew.write(out / "summary.json", data)
    (out / "summary.md").write_text(render(data), encoding="utf-8")
    print(render(data))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "smoke", "checks", "train", "evaluate", "summary"])
    parser.add_argument("--out", required=True)
    parser.add_argument("--work", default="/content/night_work")
    parser.add_argument("--parent", default=None)
    parser.add_argument("--workers", default="1")
    parser.add_argument("--seed", default=None)
    parser.add_argument("--wall-hours", default=str(config()["budget"]["wall_hours"]))
    args = parser.parse_args()
    if args.workers == "auto":
        args.workers = str(max(1, (os.cpu_count() or 2) - 3 if args.command == "checks" else (os.cpu_count() or 2)))
    if args.command in ("smoke", "checks", "train", "evaluate") and sys.platform == "darwin":
        raise SystemExit("WEIGHT1_NIGHT 실제 실행은 Mac에서 하지 않습니다. Colab에서 실행하세요.")
    return {"prepare": cmd_prepare, "smoke": cmd_smoke, "checks": cmd_checks, "train": cmd_train,
            "evaluate": cmd_evaluate, "summary": cmd_summary}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
