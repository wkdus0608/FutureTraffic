"""ARENA3 심판(런타임 A) — 같은 84상황·같은 코드·같은 판정으로 모든 정책을 잰다(208절).

명령(Colab, FOCUSED_RULESET=pet1 FOCUSED_NORTH_EXT=240):
  python src/arena3_referee.py check    --out <Drive>/arena3/referee/baseline --work /content/a3_check
      원래 배치(이전 실험과 같은 차량)에서 P0·M4·W4가 FIXED24의 값과 소수점까지 같은지 본다. 다르면 멈춘다.
  python src/arena3_referee.py baseline --out <Drive>/arena3/referee/baseline --work /content/a3 --workers 8
      평가 84상황 × P0·M4·W4 = 252회.
  python src/arena3_referee.py learner  --name G1 --models <모델 폴더 3개> --out <Drive>/arena3/referee/G1 ...
      학습기 모델 3개 × 84상황(학습이 끝난 뒤).
  python src/arena3_referee.py judge    --baseline <...>/baseline --candidate BPRIME=<...>/referee/bprime ...
      잠근 판정 규칙으로 M4 대비 통과 여부를 계산해 judge.json·summary.md를 쓴다.

판정(잠금 설정 judge): 날짜별 (M4 − 후보) 차량당 지연의 평균 ≥ 0.3초, 날짜 7개 t95 하한 > 0, 양수 날짜 ≥ 5,
좌회전 동·남 각각 M4 + 5초 이내, 300초 초과 차량 수 ≤ M4, 정책 고유 미확인 충돌 0, 품질 오류 0.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import multiprocessing
import os
import shutil
import statistics
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import arena3_common as ac  # noqa: E402
import explore_weight as w  # noqa: E402

# 평가 결과를 정하는 코드만 평가 지문에 넣는다. 학습기·B′ 코드는 각 결과 폴더에 따로 적는다(extra_files).
SOURCES = ("src/arena3_common.py", "src/arena3_referee.py",
           "config/arena3_v1.json", "src/explore_weight.py", "src/explore_weight_core.py",
           "src/explore_weight_episode.py", "src/explore_pred.py", "src/explore_rollout.py",
           "src/explore_m4probe.py", "src/explore_m4probe_pending.py", "src/explore_online.py",
           "src/focused_train.py", "src/focused_eval.py", "config/collision_exceptions.json")
PACKAGES = ("numpy", "torch", "sumo-rl", "eclipse-sumo")
LEFT_KEYS = {"left_east": "477706392_02->477706883_02", "left_south": "477706883_01->477707013_01"}


def stamp(*parts) -> None:
    print(time.strftime("%H:%M:%S"), *parts, flush=True)


# ----------------------------------------------------------------------------- 등록
def evaluation_record() -> dict:
    """평가 지문의 내용: 잠금 설정·평가 코드·환경 파일·패키지. 모든 결과 폴더에서 같아야 서로 비교할 수 있다."""
    w.configure()
    from focused_data import ENV_FILES, environment_fingerprint
    cfg = ac.validate_lock()
    env = environment_fingerprint()
    if env != cfg["environment"]["fingerprint"]:
        raise ValueError(f"잠근 환경 지문 {cfg['environment']['fingerprint']}과 지금 {env}이 다릅니다.")
    if w.file_hash(ROOT / "src/explore_pred.py")[:12] != cfg["environment"]["p0_code"]:
        raise ValueError("P0 코드가 잠근 코드와 다릅니다.")
    files = {s: w.file_hash(ROOT / s) for s in sorted(set(SOURCES) | set(ENV_FILES))}
    return {"kind": ac.EXPERIMENT, "lock_sha256": w.file_hash(ac.CONFIG), "environment_fingerprint": env,
            "files": files, "packages": {p: importlib.metadata.version(p) for p in PACKAGES}}


def register(out: Path, extra_files: tuple[str, ...] = ()) -> str:
    """결과 폴더의 지문 = 평가 지문 + 이 폴더만의 코드(B′·학습기). 같은 폴더를 다른 코드로 이어 쓰지 않는다."""
    evaluation = evaluation_record()
    frozen = {"evaluation_fingerprint": w.digest(evaluation), "evaluation": evaluation,
              "extra_files": {s: w.file_hash(ROOT / s) for s in sorted(extra_files)}}
    fp = w.digest(frozen)
    path = Path(out) / "experiment.json"
    if not path.exists():
        # 같은 폴더를 여러 프로세스(B′ 두 조각, G1 난수 세 개)가 동시에 처음 쓸 수 있어 임시 파일 이름을 따로 둔다.
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".experiment.{os.getpid()}.{time.time_ns()}.tmp")
        tmp.write_text(json.dumps(w.plain({"fingerprint": fp, **frozen}), ensure_ascii=False, indent=2) + "\n")
        os.replace(tmp, path)
    if w.read(path).get("fingerprint") != fp:
        raise ValueError("코드·설정·패키지가 이 결과 폴더를 만든 때와 다릅니다. 기존 결과를 보존하고 새 폴더를 쓰세요.")
    return fp


def evaluation_fingerprint(folder: Path) -> str:
    return w.read(Path(folder) / "experiment.json")["evaluation_fingerprint"]


class Ledger:
    """역할별 계산 시간 장부(실패한 작업도 센다). 계산 한도는 해제됐으므로 상한 없이 기록만 한다."""

    def __init__(self, out: Path, role: str):
        self.path = Path(out) / "ledger" / f"{role}.json"
        saved = w.read(self.path) if self.path.exists() else {}
        self.seconds = float(saved.get("seconds", 0.0))
        self.jobs = int(saved.get("jobs", 0))

    def add(self, seconds: float) -> None:
        self.seconds += max(0.0, float(seconds))
        self.jobs += 1
        w.write(self.path, {"seconds": self.seconds, "jobs": self.jobs, "cpu_hours": self.seconds / 3600})


# ----------------------------------------------------------------------------- 작업
def target_dir(out: Path, policy: str, sid: str) -> Path:
    return Path(out) / "runs" / policy / sid


def jobs_for(out: Path, work: Path, fp: str, policy: str, situations: list[dict], *, model_dir=None,
             model_identity=None, kind="fixed") -> list[dict]:
    jobs = []
    for s in situations:
        sid = ac.situation_id(s) if s["tag"] else f"{s['date']}_{s['period']}_s{s['sumo_seed']}"
        jobs.append({"policy": policy, "kind": kind, "situation": s, "sid": sid, "fingerprint": fp,
                     "model_dir": str(model_dir) if model_dir else None, "model_identity": model_identity,
                     "target": str(target_dir(out, policy, sid)), "work": str(Path(work) / policy / sid)})
    return jobs


def build_policy(job: dict):
    if job["kind"] == "fixed":
        return ac.fixed_policy(job["policy"])
    if job["kind"] == "g1":
        import arena3_g1 as g1
        return g1.GreedyPolicy(g1.load_model(Path(job["model_dir"])), name=job["policy"])
    raise ValueError(f"알 수 없는 정책 종류: {job['kind']}")


def _worker_init():
    try:
        os.setsid()
    except OSError:
        pass
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"


def eval_worker(job: dict) -> dict:
    started = time.monotonic()
    work = Path(job["work"])
    try:
        w.configure()
        shutil.rmtree(work, ignore_errors=True)
        result = ac.run_arena_episode(job["situation"], work, build_policy(job), max_seconds=1800.0)
        expected = job["situation"].get("route_sha256")
        if expected and result.metrics["route_sha256"] != expected:
            raise ValueError(f"차량 파일 SHA가 잠금과 다릅니다: {result.metrics['route_sha256']} ≠ {expected}")
        ac.write_run(Path(job["target"]), result, {"policy": job["policy"], "fingerprint": job["fingerprint"],
                                                   "model_identity": job.get("model_identity"),
                                                   "job_seconds": time.monotonic() - started})
        return {"ok": True, "seconds": time.monotonic() - started, "name": f"{job['policy']}/{job['sid']}"}
    except BaseException as error:
        return {"ok": False, "seconds": time.monotonic() - started, "name": f"{job['policy']}/{job['sid']}",
                "error": f"{type(error).__name__}: {error}"}
    finally:
        shutil.rmtree(work, ignore_errors=True)


def cached(job: dict) -> dict | None:
    return ac.read_run(Path(job["target"]), job["fingerprint"], job.get("model_identity"))


def run_jobs(jobs: list[dict], workers: int, ledger: Ledger, *, worker=eval_worker, executor=None) -> dict:
    """실패한 작업이 있어도 나머지를 끝까지 돌리고, 실패 목록을 돌려준다(한 번 왕복으로 모두 보이게)."""
    pending = [j for j in jobs if cached(j) is None]
    stamp(f"작업 {len(jobs)}개 중 {len(pending)}개 실행(재사용 {len(jobs) - len(pending)}개), 작업자 {workers}개")
    failed, done = [], 0
    if not pending:
        return {"done": 0, "failed": []}
    pool = executor or ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                                           initializer=_worker_init)
    with pool:
        running = {}
        while pending or running:
            while pending and len(running) < workers:
                job = pending.pop(0)
                running[pool.submit(worker, job)] = job
            finished, _ = wait(list(running), return_when=FIRST_COMPLETED)
            for future in finished:
                job = running.pop(future)
                try:
                    r = future.result()
                except Exception as error:
                    r = {"ok": False, "seconds": 0.0, "name": job["sid"], "error": f"{type(error).__name__}: {error}"}
                ledger.add(r["seconds"])
                if r["ok"]:
                    done += 1
                    stamp(f"완료 {r['name']} ({r['seconds']:.0f}초) | 누적 {ledger.seconds / 3600:.2f} CPU-시간")
                else:
                    failed.append({"name": r["name"], "error": r["error"]})
                    w.write(Path(job["target"]).parent / f"{job['sid']}.failed.json", r)
                    stamp(f"실패 {r['name']}: {r['error']}")
    return {"done": done, "failed": failed}


# ----------------------------------------------------------------------------- 판정
def t_interval(values: list[float], t975: float) -> tuple[float, float, float]:
    mean = statistics.fmean(values)
    se = statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else math.inf
    return mean, mean - t975 * se, mean + t975 * se


def quality_errors(m: dict) -> list[str]:
    errors = [k for k in ("missing", "unfinished", "changed_by_rules", "mode_action_errors", "p0_mismatch")
              if m.get(k, 0)]
    check = m.get("cost_check")
    if not (isinstance(check, dict) and check.get("passed")):
        errors.append("cost_check")
    if int(m.get("vehicles", 0)) <= 0 or not math.isfinite(float(m.get("mean_delay_s") or math.nan)):
        errors.append("vehicles/delay")
    return errors


def collision_events(m: dict) -> set:
    details = m.get("collision_details") or {}
    events = details.get("collisions_unexplained") or []
    return {(str(e.get("time")), e.get("lane"), e.get("collider"), e.get("victim")) for e in events}


def movement_mean(rows: list[dict], key: str) -> float | None:
    n = total = 0.0
    for m in rows:
        r = (m.get("by_movement") or {}).get(key)
        if r:
            n += r["vehicles"]
            total += r["vehicles"] * r["mean_delay_s"]
    return total / n if n else None


def index_runs(rows: list[dict], sids: list[str], name: str) -> dict:
    idx = {m["situation"]: m for m in rows}
    if len(idx) != len(rows) or set(idx) != set(sids):
        missing = sorted(set(sids) - set(idx))[:5]
        raise ValueError(f"{name}: 84상황이 모두 있지 않거나 중복입니다(빠진 예: {missing}).")
    return idx


def judge(candidate_models: list[list[dict]], reference: list[dict], p0: list[dict] | None, cfg: dict,
          situations: list[dict]) -> dict:
    """candidate_models: 모델별 실행 목록(고정·B′는 1개, 학습기는 3개). reference = M4."""
    j = cfg["judge"]
    sids = [ac.situation_id(s) for s in situations]
    date_of = {ac.situation_id(s): s["date"] for s in situations}
    ref = index_runs(reference, sids, "M4")
    cands = [index_runs(rows, sids, f"후보 모델 {i}") for i, rows in enumerate(candidate_models)]
    p0i = index_runs(p0, sids, "P0") if p0 is not None else {}
    for i, c in enumerate(cands):
        for sid in sids:
            for field in ("vehicles", "route_sha256", "cohort_sha256"):
                if c[sid][field] != ref[sid][field]:
                    raise ValueError(f"짝 불일치 {sid} {field} (후보 모델 {i})")
    expected_route = {ac.situation_id(s): s["route_sha256"] for s in situations}
    for sid in sids:
        if ref[sid]["route_sha256"] != expected_route[sid]:
            raise ValueError(f"M4 실행의 차량 파일이 잠금과 다릅니다: {sid}")
    dates = sorted({date_of[s] for s in sids})
    per_date, per_situation = {}, {}
    for sid in sids:
        n = ref[sid]["vehicles"]
        cand_total = statistics.fmean(float(c[sid]["actual_total_delay"]) for c in cands)
        per_situation[sid] = (float(ref[sid]["actual_total_delay"]) - cand_total) / n
    for d in dates:
        own = [s for s in sids if date_of[s] == d]
        n = sum(ref[s]["vehicles"] for s in own)
        ref_total = sum(float(ref[s]["actual_total_delay"]) for s in own)
        cand_total = sum(statistics.fmean(float(c[s]["actual_total_delay"]) for c in cands) for s in own)
        per_date[d] = (ref_total - cand_total) / n
    mean, low, high = t_interval(list(per_date.values()), j["t975"])
    n_all = sum(ref[s]["vehicles"] for s in sids)
    weighted = (sum(float(ref[s]["actual_total_delay"]) for s in sids)
                - sum(statistics.fmean(float(c[s]["actual_total_delay"]) for c in cands) for s in sids)) / n_all
    lefts = {}
    for name, key in LEFT_KEYS.items():
        r = movement_mean(list(ref.values()), key)
        c = statistics.fmean(movement_mean(list(ci.values()), key) for ci in cands)
        lefts[name] = {"candidate_s": c, "m4_s": r, "difference_s": c - r}
    long_c = statistics.fmean(sum(int(ci[s]["long_waits"]) for s in sids) for ci in cands)
    long_r = sum(int(ref[s]["long_waits"]) for s in sids)
    quality = {f"model{i}/{sid}": quality_errors(c[sid]) for i, c in enumerate(cands) for sid in sids
               if quality_errors(c[sid])}
    shared, specific = [], []
    for i, c in enumerate(cands):
        for sid in sids:
            events = collision_events(c[sid])
            inside = int(c[sid].get("collisions_at_204820", 0))
            if not events and not inside:
                continue
            same = collision_events(ref[sid]) | (collision_events(p0i[sid]) if sid in p0i else set())
            same_inside = max(int(ref[sid].get("collisions_at_204820", 0)),
                              int(p0i[sid].get("collisions_at_204820", 0)) if sid in p0i else 0)
            row = {"model": i, "situation": sid, "events": sorted(map(list, events)), "at_204820": inside}
            (shared if events <= same and inside <= same_inside else specific).append(row)
    checks = {
        "gain_at_least_min": mean >= j["min_gain_s"],
        "date_t95_lower_positive": low > 0,
        "positive_dates": sum(v > 0 for v in per_date.values()) >= j["min_positive_dates"],
        "left_east_within_tolerance": lefts["left_east"]["difference_s"] <= j["left_tolerance_s"] + 1e-9,
        "left_south_within_tolerance": lefts["left_south"]["difference_s"] <= j["left_tolerance_s"] + 1e-9,
        "long_waits_not_more_than_m4": long_c <= long_r + 1e-9,
        "no_policy_specific_collision": not specific,
        "quality_clean": not quality,
    }
    status = "pass" if all(checks.values()) and not shared else ("hold_shared_collision" if all(checks.values()) else "fail")
    return {"status": status, "checks": checks, "gain_date_mean_s": mean, "date_t95_s": [low, high],
            "gain_vehicle_weighted_s": weighted, "per_date_gain_s": per_date,
            "positive_dates": sum(v > 0 for v in per_date.values()), "per_situation_gain_s": per_situation,
            "left_turns": lefts, "long_waits": {"candidate": long_c, "m4": long_r},
            "collisions": {"shared_with_m4_or_p0": shared, "policy_specific": specific},
            "quality_errors": quality, "models": len(cands),
            "note": "반복 사용 날짜의 개발 비교; 새 날짜·실제 도로 일반화가 아님"}


def load_policy_runs(folder: Path, policy: str, fp: str | None = None) -> list[dict]:
    base = Path(folder) / "runs" / policy
    rows = []
    for d in sorted(base.iterdir()) if base.is_dir() else []:
        if d.is_dir() and (d / "metrics.json").exists():
            m = ac.read_run(d, fp if fp is not None else w.read(d / "cache.json").get("fingerprint"),
                            w.read(d / "cache.json").get("model_identity"))
            rows.append(m)
    return rows


def render(results: dict, sensitivity: dict | None) -> str:
    def f(x, nd=3):
        return "-" if x is None else f"{x:+.{nd}f}"
    lines = ["# ARENA3 심판 결과", "",
             "M4 대비 이득(초/대, +면 후보가 짧음). 날짜 7개 평균과 t95 구간, 반복 사용 날짜의 개발 비교.", "",
             "| 후보 | 판정 | 날짜 평균 | t95 구간 | 차량 가중 | 양수 날짜 | 동 좌회전 차이 | 남 좌회전 차이 | 300초 초과(후보/M4) |",
             "|---|---|---|---|---|---|---|---|---|"]
    for name, r in results.items():
        lines.append(f"| {name} | {r['status']} | {f(r['gain_date_mean_s'])} | [{f(r['date_t95_s'][0])}, {f(r['date_t95_s'][1])}] "
                     f"| {f(r['gain_vehicle_weighted_s'])} | {r['positive_dates']}/7 | {f(r['left_turns']['left_east']['difference_s'], 2)} "
                     f"| {f(r['left_turns']['left_south']['difference_s'], 2)} | {r['long_waits']['candidate']:.1f}/{r['long_waits']['m4']} |")
    if sensitivity is not None:
        lines += ["", f"심판 감도 확인(M4 대 P0, 같은 계산): 날짜 평균 {f(-sensitivity['gain_date_mean_s'])}초, "
                      f"t95 [{f(-sensitivity['date_t95_s'][1])}, {f(-sensitivity['date_t95_s'][0])}] — "
                      + ("감도 있음" if sensitivity["date_t95_s"][1] < 0 else "**심판 감도 부족** 표시")]
    lines += ["", "실패한 조건은 judge.json의 checks를 보세요. 'hold_shared_collision'은 M4·P0에도 같은 미확인 충돌이 있어 재현 진단 전 보류라는 뜻입니다."]
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------------- 명령
def cmd_check(args) -> int:
    cfg = ac.config()
    out = Path(args.out)
    fp = register(out)
    ledger = Ledger(out, "check")
    report, ok = [], True
    for case in cfg["reproduction"]["cases"]:
        s = ac.make_situation(case["date"], case["period"], None, case["sumo_seed"])
        job = jobs_for(out / "reproduction", Path(args.work), fp, case["policy"], [s])[0]
        r = cached(job)
        if r is None:
            res = eval_worker(job)
            ledger.add(res["seconds"])
            if not res["ok"]:
                report.append({**case, "ok": False, "error": res["error"]})
                ok = False
                continue
            r = cached(job)
        same = (r["mean_delay_s"] == case["mean_delay_s"] and abs(r["actual_total_delay"] - case["actual_total_delay"]) < 1e-6
                and r["executed_sha16"] == case["executed_sha16"] and r["route_sha256"] == case["route_sha256"])
        report.append({**case, "ok": same, "got_mean_delay_s": r["mean_delay_s"], "got_executed_sha16": r["executed_sha16"],
                       "got_route_sha256": r["route_sha256"], "choice_points_6mode": r["choice_points"]})
        ok &= same
        stamp(f"재현 {case['policy']}: {'같음' if same else '다름'} ({r['mean_delay_s']} 대 {case['mean_delay_s']})")
    w.write(out / "reproduction.json", {"passed": ok, "cases": report})
    if not ok:
        stamp("재현 점검 실패 — 기준선을 시작하지 않습니다. reproduction.json을 확인하세요.")
        return 2
    stamp("재현 점검 통과: P0·M4·W4가 이전 실행과 같습니다.")
    return 0


def cmd_baseline(args) -> int:
    cfg = ac.config()
    out = Path(args.out)
    fp = register(out)
    repro = out / "reproduction.json"
    if not repro.exists() or not w.read(repro).get("passed"):
        raise SystemExit("재현 점검(check)이 통과하지 않았습니다. 먼저 check를 실행해 통과시키세요.")
    jobs = []
    for policy in cfg["evaluation"]["baselines"]:
        jobs += jobs_for(out, Path(args.work), fp, policy, cfg["situations_eval"])
    status = run_jobs(jobs, args.workers, Ledger(out, "baseline"))
    w.write(out / "status" / "baseline.json", {**status, "total": len(jobs), "updated_unix": time.time()})
    if status["failed"]:
        stamp(f"실패 {len(status['failed'])}개 — status/baseline.json과 *.failed.json을 확인하세요.")
        return 1
    return cmd_judge(argparse.Namespace(baseline=out, candidate=[f"W4={out}"], out=out))


def cmd_learner(args) -> int:
    cfg = ac.config()
    out = Path(args.out)
    fp = register(out, extra_files=("src/arena3_g1.py",))
    if len(args.models) != cfg["judge"]["learner_models"]:
        raise SystemExit(f"학습기 모델은 {cfg['judge']['learner_models']}개여야 합니다.")
    import arena3_g1 as g1
    jobs = []
    for i, folder in enumerate(args.models):
        identity = g1.model_identity(Path(folder))
        jobs += jobs_for(out, Path(args.work), fp, f"{args.name}_m{i}", cfg["situations_eval"], model_dir=folder,
                         model_identity=identity, kind="g1")
    status = run_jobs(jobs, args.workers, Ledger(out, f"learner_{args.name}"))
    w.write(out / "status" / f"learner_{args.name}.json", {**status, "total": len(jobs), "updated_unix": time.time()})
    return 1 if status["failed"] else 0


def collect_candidate(spec: str) -> tuple[str, list[list[dict]], int]:
    """'이름=폴더'. 폴더의 runs/ 아래 이름(모델 1개) 또는 이름_m0·_m1·_m2(학습기 모델 3개) 정책을 모은다.
    돌려줌: (이름, 모델별 실행 목록, 있어야 할 모델 수). 아직 없으면 빈 목록."""
    name, folder = spec.split("=", 1)
    runs = Path(folder) / "runs"
    direct = load_policy_runs(Path(folder), name)
    if direct:
        return name, [direct], 1
    models = sorted(p.name for p in runs.iterdir() if p.is_dir() and p.name.startswith(f"{name}_m")) if runs.is_dir() else []
    return name, [load_policy_runs(Path(folder), m) for m in models], ac.config()["judge"]["learner_models"]


def cmd_judge(args) -> int:
    cfg = ac.config()
    base = Path(args.baseline)
    m4 = load_policy_runs(base, "M4")
    p0 = load_policy_runs(base, "P0")
    situations = cfg["situations_eval"]
    if len(m4) != 84:
        stamp(f"기준선 M4가 84상황 중 {len(m4)}개만 있어 판정하지 않습니다(런타임 A의 기준선을 먼저 끝내세요).")
        return 0
    results = {}
    sensitivity = judge([p0], m4, None, cfg, situations) if len(p0) == 84 and len(m4) == 84 else None
    base_fp = evaluation_fingerprint(base)
    for spec in args.candidate or []:
        name, models, want = collect_candidate(spec)
        folder = Path(spec.split("=", 1)[1])
        if not models or not (folder / "experiment.json").exists():
            stamp(f"{name}: 아직 실행 결과가 없어 판정하지 않습니다.")
            continue
        if evaluation_fingerprint(folder) != base_fp:
            stamp(f"{name}: 평가 지문이 기준선과 달라(평가 코드·잠금·패키지가 다름) 판정하지 않습니다.")
            continue
        if len(models) != want or any(len(rows) != 84 for rows in models):
            stamp(f"{name}: 84상황 × 모델 {want}개가 다 끝나지 않아 판정하지 않습니다({[len(r) for r in models]}).")
            continue
        results[name] = judge(models, m4, p0, cfg, situations)
        stamp(f"{name}: {results[name]['status']} | 날짜 평균 {results[name]['gain_date_mean_s']:+.3f}초 "
              f"t95 [{results[name]['date_t95_s'][0]:+.3f}, {results[name]['date_t95_s'][1]:+.3f}]")
    out = Path(args.out)
    w.write(out / "judge.json", {"results": results, "sensitivity_m4_vs_p0": sensitivity,
                                 "lock_sha256": w.file_hash(ac.CONFIG), "updated_unix": time.time()})
    (out / "summary.md").write_text(render(results, sensitivity), encoding="utf-8")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("check", "baseline", "learner"):
        p = sub.add_parser(name)
        p.add_argument("--out", type=Path, required=True)
        p.add_argument("--work", type=Path, required=True)
        p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2)))
        if name == "learner":
            p.add_argument("--name", required=True)
            p.add_argument("--models", nargs="+", required=True)
    p = sub.add_parser("judge")
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--candidate", action="append", help="이름=폴더 (여러 번)")
    p.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    w.configure()
    return {"check": cmd_check, "baseline": cmd_baseline, "learner": cmd_learner, "judge": cmd_judge}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
