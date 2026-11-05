"""[추가 탐색] PLATOON1C — 같은 학생 PT1에 '확실할 때만 바꾸기' 문턱을 둔다(사전 등록 results/explore/platoon1c_20261101/사전등록.md).

학생 3개는 PLATOON1 그대로이고 새 수집·재학습은 없다. 예측 이득이 문턱 τ 이상일 때만 TM4와 다르게 고른다.
세계·평가 84상황·심판 함수는 PLATOON1과 같고, 기준 실행(PLATOON0·PLATOON1·PBASE)은 다시 돌리지 않고 읽는다.
실제 SUMO 평가는 Colab에서만 한다(맥은 작은 실행과 judge만).

사용(FOCUSED_RULESET=pet1 FOCUSED_NORTH_EXT=240):
  python src/explore_platoon1c.py smoke --candidate t09 --students <Drive>/explore_platoon1/students --out <dir> --work /content/pcs \\
      --platoon0 <Drive>/explore_platoon0 --platoon1 <Drive>/explore_platoon1
  python src/explore_platoon1c.py evaluate --candidate t09 --students <Drive>/explore_platoon1/students \\
      --out <Drive>/explore_platoon1c/eval/t09 --work /content/pc --workers 8
  python src/explore_platoon1c.py judge --eval <Drive>/explore_platoon1c/eval --platoon0 <Drive>/explore_platoon0 \\
      --platoon1 <Drive>/explore_platoon1 --pbase <Drive>/explore_platoon_base --out <Drive>/explore_platoon1c/judge
  python src/explore_platoon1c.py confirm --judge <Drive>/explore_platoon1c/judge --students <Drive>/explore_platoon1/students \\
      --out <Drive>/explore_platoon1c/judge/confirm --work /content/pcc --workers 8
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import platform
import shutil
import statistics
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import arena3_common as ac  # noqa: E402
import arena3_referee as ref  # noqa: E402
import arena3_stage2_eval as s2e  # noqa: E402
import explore_platoon1 as pt  # noqa: E402
import explore_platoon1_learn as pl  # noqa: E402
import explore_platoon_base as pb  # noqa: E402
import explore_weight as w  # noqa: E402

EXPERIMENT = "PLATOON1C_V1"
CONFIG = ROOT / "config/explore_platoon1c_v1.json"
PREREG = ROOT / "results/explore/platoon1c_20261101/사전등록.md"
SOURCES = ("src/explore_platoon1c.py", "src/explore_platoon_base.py", "config/explore_platoon1c_v1.json",
           "results/explore/platoon1c_20261101/사전등록.md", *pl.eval_sources())
MAX_EPISODE_S = 3600.0
COUNT_AT = (0.0, 0.9, 2.0, 3.0)              # 지나간 결정마다 예측 이득이 이 값 이상인 횟수(바꾸는 비율 읽기용)
PT1_BASE = ("M4", "TM4", "OM4", "P1M4")


# ----------------------------------------------------------------------------- 설정
def config() -> dict:
    cfg = w.read(CONFIG)
    if cfg.get("experiment") != EXPERIMENT:
        raise ValueError("PLATOON1C 설정이 아닙니다.")
    for key, path in (("prereg_sha256", PREREG), ("lock_sha256", ac.CONFIG), ("platoon1_config_sha256", pt.CONFIG)):
        if cfg.get(key) != w.file_hash(path):
            raise ValueError(f"PLATOON1C 설정을 만든 때와 파일이 다릅니다: {path.relative_to(ROOT)}")
    return cfg


def identity(model_sha: str, tau: float) -> str:
    return f"{model_sha}|tau={tau:g}"


# ----------------------------------------------------------------------------- 정책
def sure_green(net, obs, tau: float) -> tuple[int, float]:
    """TM4가 아닌 허용 녹색 중 예측 이득이 가장 큰 것을, 그 이득이 τ 이상일 때만 고른다. (녹색, 가장 큰 예측 이득)."""
    obs = np.asarray(obs, np.float32)
    mask = pt.allowed_mask(obs)
    anchor = int(pt.anchor_green(obs))
    if not mask[anchor] or mask.sum() < 2:
        raise ValueError("TM4 녹색이 허용 밖이거나 고를 녹색이 없습니다.")
    with torch.no_grad():
        rel = net(torch.from_numpy(obs).view(1, -1))[0].numpy().astype(np.float64)
    if not np.all(np.isfinite(rel)):
        raise ValueError("녹색 값이 유한하지 않습니다.")
    rel = np.where(mask, rel, -np.inf)
    rel[anchor] = -np.inf
    best = int(rel.argmax())
    gain = float(rel[best])
    return (best if gain >= tau else anchor), gain


class SureStudentPolicy(pt.StudentPolicy):
    """PT1 평가 정책과 같은 관측·TM4 기록. 고르는 규칙만 문턱을 둔다."""

    def __init__(self, net, tau: float, name: str):
        super().__init__(net, name)
        self.tau = float(tau)
        self.pairs = Counter()
        self.above = Counter()

    def decide(self, snap, ctx) -> int:
        core = pt.episode_core(self, ctx)
        env = ctx.ep.env
        core.observe(env)
        allowed = [int(a) for a in ctx.allowed]
        if len(allowed) == 1:
            return allowed[0]
        view = core.view(env, allowed)
        action, gain = sure_green(self.net, pt.observation167(snap.observation, view["extra"]), self.tau)
        self.stats["planned"] += 1
        for level in COUNT_AT:
            self.above[f"{level:g}"] += int(gain >= level)
        if action != view["green"]:
            self.stats["differs_from_tm4"] += 1
            self.pairs[f"{view['green']}->{action}"] += 1
        return action

    def info(self) -> dict:
        return {"kind": "sure_student", "tau": self.tau, **self.stats, "pairs": dict(self.pairs), "above": dict(self.above)}


# ----------------------------------------------------------------------------- 실행
def check_models(folders: list[Path], cfg: dict) -> dict:
    """PLATOON1 학생 3개인지(라운드·자료·순서는 PLATOON1 선언 함수, 파일 해시는 설정에 고정한 값)."""
    decl = pl.declaration([Path(f) for f in folders], "eval", pt.config())
    if decl["model_identities"] != cfg["students"]["model_sha256"] or decl["dataset_fingerprint"] != cfg["students"]["dataset_fingerprint"]:
        raise ValueError("학생 모델이 설정에 고정한 PLATOON1 학생과 다릅니다.")
    return decl


def policy_names(candidate: str) -> list[str]:
    return [f"{candidate}_m{i}" for i in range(3)]


def jobs_for(out: Path, work: Path, fp_: str, candidate: str, tau: float, sits: list[dict], folders: list[Path], cfg: dict) -> list[dict]:
    jobs = []
    for name, folder, sha in zip(policy_names(candidate), folders, cfg["students"]["model_sha256"]):
        for job in ref.jobs_for(out, Path(work), fp_, name, sits, model_dir=folder, model_identity=identity(sha, tau), kind="sure"):
            job.update(tau=float(tau), model_sha256=sha)
            jobs.append(job)
    return jobs


def run_one(job: dict) -> dict:
    started = time.monotonic()
    work = Path(job["work"])
    name = f"{job['policy']}/{job['sid']}"
    try:
        w.configure()
        torch.set_num_threads(1)
        shutil.rmtree(work, ignore_errors=True)
        if pl.student_identity(Path(job["model_dir"])) != job["model_sha256"]:
            raise ValueError("작업 시작 후 모델이 바뀌었습니다.")
        policy = SureStudentPolicy(pl.load_student(Path(job["model_dir"])), job["tau"], job["policy"])
        result = pt.run_episode(job["situation"], work, policy, max_seconds=MAX_EPISODE_S)
        if result.metrics["route_sha256"] != job["situation"].get("route_sha256"):
            raise ValueError(f"평가 차량 파일의 SHA가 설정과 다릅니다: {result.metrics['route_sha256']}")
        ac.write_run(Path(job["target"]), result,
                     {"policy": job["policy"], "fingerprint": job["fingerprint"], "model_identity": job["model_identity"],
                      "job_seconds": time.monotonic() - started, "information": policy.info()})
        return {"ok": True, "seconds": time.monotonic() - started, "name": name}
    except BaseException as error:
        return {"ok": False, "seconds": time.monotonic() - started, "name": name, "error": f"{type(error).__name__}: {error}"}
    finally:
        shutil.rmtree(work, ignore_errors=True)


def eval_worker(job: dict) -> dict:
    if platform.system() == "Darwin":
        return {"ok": False, "seconds": 0.0, "name": job["sid"], "error": "실제 SUMO 평가는 Colab에서만 실행합니다."}
    return run_one(job)


def student_dirs(students: Path, cfg: dict) -> list[Path]:
    return [Path(students) / f"seed_{s}" for s in cfg["students"]["seeds"]]


def evaluate(candidate: str, students: Path, out: Path, work: Path, workers: int, cfg: dict, part: str = "eval", *,
             executor=None, worker=eval_worker) -> dict:
    tau = float(cfg["candidates"][candidate]["tau"])
    sits = pl.situations(pt.config(), part)
    out = Path(out)
    fp_ = ref.register(out, extra_files=SOURCES)
    folders = student_dirs(students, cfg)
    decl = check_models(folders, cfg)
    s2e._freeze(out / "models.json", {"candidate": candidate, "tau": tau, "part": part, "seeds": decl["seeds"],
                                      "model_identities": decl["model_identities"],
                                      "dataset_fingerprint": decl["dataset_fingerprint"]})
    jobs = jobs_for(out, work, fp_, candidate, tau, sits, folders, cfg)
    if worker is eval_worker and executor is None:
        pl._prepare_map_cache()
    status = ref.run_jobs(jobs, max(1, workers), ref.Ledger(out, candidate), worker=worker, executor=executor)
    s2e._write(out / "status" / f"{candidate}_{part}.json", {**status, "total": len(jobs), "part": part, "updated_unix": time.time()})
    return status


# ----------------------------------------------------------------------------- 작은 실행
def smoke(candidate: str, students: Path, out: Path, work: Path, workers: int, platoon0: Path, platoon1: Path, cfg: dict) -> dict:
    """첫 평가 상황 3판(난수 42 모델): τ 매우 큼 = PLATOON0 TM4, τ≈0 = PLATOON1 PT1_m0, 후보 τ는 PT1_m0보다 덜 바꾼다."""
    out, work = Path(out), Path(work)
    fp_ = ref.register(out, extra_files=SOURCES)
    pl._prepare_map_cache()
    folders = student_dirs(students, cfg)
    check_models(folders, cfg)
    s = pl.situations(pt.config(), "eval")[0]
    sid = ac.situation_id(s)
    sha = cfg["students"]["model_sha256"][0]
    taus = {"never": float(cfg["smoke"]["never_tau"]), "always": float(cfg["smoke"]["always_tau"]),
            candidate: float(cfg["candidates"][candidate]["tau"])}
    jobs = [{"policy": name, "situation": s, "sid": sid, "fingerprint": fp_, "model_dir": str(folders[0]), "model_sha256": sha,
             "tau": tau, "model_identity": identity(sha, tau), "target": str(out / "runs" / name / sid), "work": str(work / name)}
            for name, tau in taus.items()]
    with ProcessPoolExecutor(max_workers=max(1, min(workers, len(jobs))), mp_context=multiprocessing.get_context("spawn"),
                             initializer=ref._worker_init) as pool:
        done = list(pool.map(run_one, [j for j in jobs if ref.cached(j) is None]))
    bad = [d for d in done if not d["ok"]]
    if bad:
        raise RuntimeError(f"작은 실행이 실패했습니다: {bad}")
    runs = {j["policy"]: ac.read_run(Path(j["target"]), fp_, j["model_identity"]) for j in jobs}
    fp0 = w.read(Path(platoon0) / "experiment.json")["fingerprint"]
    tm4 = ac.read_run(Path(platoon0) / "runs" / "P" / "TM4" / sid, fp0, "P/TM4")
    pt1_folder = Path(platoon1) / "eval" / "PT1"
    pt1 = ac.read_run(pt1_folder / "runs" / "PT1_m0" / sid, w.read(pt1_folder / "experiment.json")["fingerprint"], sha)
    same = ("mean_delay_s", "actual_total_delay", "executed_sha16", "route_sha256", "cohort_sha256")
    deviate = {k: r["information"].get("differs_from_tm4", 0) for k, r in runs.items()}
    checks = {
        "never_equals_platoon0_tm4": all(runs["never"][k] == tm4[k] for k in same),
        "never_never_deviates": deviate["never"] == 0,
        "always_equals_platoon1_pt1": all(runs["always"][k] == pt1[k] for k in same),
        "always_same_deviations_as_pt1": deviate["always"] == pt1["information"]["differs_from_tm4"],
        "candidate_deviates_less": deviate[candidate] < deviate["always"],
        "quality_clean": not any(ref.quality_errors(r) for r in runs.values()),
    }
    if platform.system() != "Darwin":                # 맥은 패키지 버전이 달라 평가 지문이 다르다(Colab에서만 대조)
        checks["same_evaluation"] = (ref.evaluation_fingerprint(out) == ref.evaluation_fingerprint(platoon0)
                                     == ref.evaluation_fingerprint(pt1_folder))
    report = {"ok": all(checks.values()), "checks": checks, "situation": sid, "candidate": candidate, "taus": taus,
              "deviations": deviate, "planned": {k: r["information"].get("planned", 0) for k, r in runs.items()},
              "above": {k: r["information"].get("above") for k, r in runs.items()},
              "pairs": {k: r["information"].get("pairs") for k, r in runs.items()},
              "seconds": {k: r["job_seconds"] for k, r in runs.items()}, "fingerprint": fp_}
    w.write(out / "smoke.json", report)
    ref.stamp(f"PLATOON1C 작은 실행: {'통과' if report['ok'] else '실패'} {checks} | 바꾼 횟수 {deviate}")
    shutil.rmtree(work, ignore_errors=True)
    return report


# ----------------------------------------------------------------------------- 판정
def _candidate_rows(folder: Path, candidate: str, tau: float, sits: list[dict], cfg: dict) -> tuple[list[list[dict]], dict]:
    record = s2e.folder_record(folder)
    models = []
    for name, sha in zip(policy_names(candidate), cfg["students"]["model_sha256"]):
        models.append(pl._rows(folder, name, identity(sha, tau), sits, record))
    return models, record


def deviation_summary(models: list[list[dict]]) -> dict:
    per_model, pairs = [], Counter()
    for rows in models:
        planned = sum(int(m["information"].get("planned", 0)) for m in rows)
        differs = sum(int(m["information"].get("differs_from_tm4", 0)) for m in rows)
        per_model.append({"planned": planned, "differs": differs, "share": differs / planned if planned else 0.0})
        for m in rows:
            pairs.update(m["information"].get("pairs", {}))
    total = sum(pairs.values())
    return {"share": statistics.fmean(p["share"] for p in per_model), "per_model": per_model,
            "pairs": {k: {"count": v, "share": v / total} for k, v in pairs.most_common()}}


def reference_rows(platoon0: Path, platoon1: Path, sits: list[dict], cfg: dict) -> dict:
    """평가 배치의 기준 실행: PLATOON0 P 세계 M4·TM4·OM4·P1M4와 PLATOON1 PT1(τ = 0에 해당)."""
    record = s2e.folder_record(platoon0)
    base = {n: pl._rows(platoon0, f"P/{n}", f"P/{n}", sits, record) for n in PT1_BASE}
    folder = Path(platoon1) / "eval" / "PT1"
    decl = w.read(folder / "models.json")
    if decl.get("model_identities") != cfg["students"]["model_sha256"]:
        raise ValueError("PLATOON1 평가의 학생이 설정에 고정한 학생과 다릅니다.")
    pt1_record = s2e.folder_record(folder)
    pt1 = [pl._rows(folder, f"PT1_m{i}", sha, sits, pt1_record) for i, sha in enumerate(decl["model_identities"])]
    return {"base": base, "PT1": pt1, "fingerprints": [record["evaluation_fingerprint"], pt1_record["evaluation_fingerprint"]]}


def select(verdicts: dict, paired: dict) -> str | None:
    passed = [c for c, v in verdicts.items() if v["passed"]]
    return max(passed, key=lambda c: paired[f"{c}-TM4"]["mean_s"]) if passed else None


def collect_report(eval_root: Path, platoon0: Path, platoon1: Path, pbase: Path | None, cfg: dict) -> dict:
    """개발 평가: 후보 세 개 + 기준 실행. 모델·실행은 설정에 고정한 해시로 대조한다(맥에서 원시 실행으로 다시 계산할 수 있다)."""
    lock, pcfg = ac.config(), pt.config()
    sits = pl.situations(pcfg, "eval")
    t975 = float(lock["judge"]["t975"])
    ref_rows = reference_rows(platoon0, platoon1, sits, cfg)
    fingerprints = set(ref_rows["fingerprints"])
    candidates, missing = {}, {}
    for name, c in cfg["candidates"].items():
        folder = Path(eval_root) / name
        if not (folder / "experiment.json").exists():
            missing[name] = "평가 폴더 없음"
            continue
        models, record = _candidate_rows(folder, name, float(c["tau"]), sits, cfg)
        fingerprints.add(record["evaluation_fingerprint"])
        lacking = [len(sits) - len(rows) for rows in models]
        if any(lacking):
            missing[name] = lacking
            continue
        candidates[name] = models
    if missing:
        return {"status": "incomplete", "missing": missing, "part": "eval"}
    m4 = ref_rows["base"]["M4"]
    results = {n: ref.judge([rows], m4, None, lock, sits) for n, rows in ref_rows["base"].items() if n != "M4"}
    results["PT1"] = ref.judge(ref_rows["PT1"], m4, None, lock, sits)
    paired = {"PT1-TM4": pl.paired_difference(results["PT1"], results["TM4"], t975),
              "OM4-TM4": pl.paired_difference(results["OM4"], results["TM4"], t975)}
    verdicts, deviations, per_model = {}, {"PT1": {"share": _pt1_share(ref_rows["PT1"])}}, {}
    for name, models in candidates.items():
        results[name] = ref.judge(models, m4, None, lock, sits)
        paired[f"{name}-TM4"] = pl.paired_difference(results[name], results["TM4"], t975)
        paired[f"{name}-OM4"] = pl.paired_difference(results[name], results["OM4"], t975)
        verdicts[name] = pl.main_verdict(results[name], paired[f"{name}-TM4"], cfg["rules"])
        deviations[name] = deviation_summary(models)
        per_model[name] = [pl.paired_difference(ref.judge([rows], m4, None, lock, sits), results["TM4"], t975) for rows in models]
    baseline = None
    if pbase is not None:
        baseline = pbase_table(Path(pbase), sits, {**{n: [rows] for n, rows in ref_rows["base"].items()}, "PT1": ref_rows["PT1"], **candidates}, lock)
        fingerprints.add(baseline.pop("evaluation_fingerprint"))
    if len(fingerprints) != 1:
        raise ValueError(f"결과 폴더들의 평가 지문이 다릅니다: {sorted(fingerprints)}")
    return {"status": "done", "part": "eval", "candidates": {n: float(c["tau"]) for n, c in cfg["candidates"].items()},
            "selected": select(verdicts, paired), "verdicts": verdicts, "results": results, "paired": paired,
            "per_model_learning": per_model, "deviations": deviations, "baseline_table": baseline,
            "evaluation_fingerprint": fingerprints.pop(), "lock_sha256": w.file_hash(ac.CONFIG), "config_sha256": w.file_hash(CONFIG),
            "updated_unix": time.time(), "note": "반복 사용 날짜의 SUMO 개발 비교. 무리 도착은 모형이고 시간표는 무리를 만든 것과 같다."}


def _pt1_share(models: list[list[dict]]) -> float:
    shares = []
    for rows in models:
        planned = sum(int(m["information"].get("planned", 0)) for m in rows)
        shares.append(sum(int(m["information"].get("differs_from_tm4", 0)) for m in rows) / planned)
    return statistics.fmean(shares)


def pbase_table(folder: Path, sits: list[dict], runs: dict[str, list[list[dict]]], lock: dict) -> dict:
    """고정신호·P0 대비(PBASE와 같은 계산): 평균 지연, 고정신호 대비 감소율(단순 계산), P0 대비 이득."""
    sids = [ac.situation_id(s) for s in sits]
    route = {ac.situation_id(s): s["route_sha256"] for s in sits}
    fixc = pb.read_folder(folder / "runs/P/FIXC", sids, route, "P/FIXC")
    p0_rows = pb.read_folder(folder / "runs/P/P0", sids, route, "P/P0")
    table = {"FIXC": pb.policy_summary([fixc]), "P0": pb.policy_summary([p0_rows])}
    for name, models in runs.items():
        table[name] = pb.policy_summary(models)
    for name, models in {"P0": [p0_rows], **runs}.items():
        j = ref.judge(models, fixc, p0_rows, lock, sits)
        table[name]["vs_FIXC"] = {"gain_date_mean_s": j["gain_date_mean_s"], "date_t95_s": j["date_t95_s"], "positive_dates": j["positive_dates"]}
    for name, models in {"FIXC": [fixc], **runs}.items():
        j = ref.judge(models, p0_rows, p0_rows, lock, sits)
        table[name]["vs_P0"] = {"gain_date_mean_s": j["gain_date_mean_s"], "date_t95_s": j["date_t95_s"], "positive_dates": j["positive_dates"]}
    base = table["FIXC"]["mean_delay_s"]
    for row in table.values():
        row["reduction_vs_fixc_pct"] = 100.0 * (base - row["mean_delay_s"]) / base
    return {"table": table, "evaluation_fingerprint": ref.evaluation_fingerprint(folder)}


def render(report: dict, title: str) -> str:
    if report["status"] != "done":
        return f"# PLATOON1C — {title}\n\n아직 끝나지 않은 실행이 있습니다: {json.dumps(report['missing'], ensure_ascii=False)}\n"
    f = lambda x, nd=3: "-" if x is None else f"{x:+.{nd}f}"  # noqa: E731
    lines = [f"# PLATOON1C — {title}", "",
             "무리 도착 세계 P의 M4 대비 이득(초/대, +면 후보가 짧음). 날짜 7개 평균과 t95, ARENA3 심판 함수 그대로.", "",
             "| 후보 | 문턱 | 바꾼 비율 | ARENA3 판정 | M4 대비 | t95 | 양수 날짜 | 동 좌회전 차이 | 남 좌회전 차이 | 300초 초과(후보/M4) | 정책 고유 충돌 |",
             "|---|---:|---:|---|---:|---|---:|---:|---:|---:|---:|"]
    taus = {**report.get("candidates", {}), "PT1": 0.0}
    for name, r in report["results"].items():
        share = report["deviations"].get(name, {}).get("share")
        lines.append(f"| {name} | {taus.get(name, '-')} | {'-' if share is None else f'{100 * share:.1f}%'} | {r['status']} "
                     f"| {f(r['gain_date_mean_s'])} | [{f(r['date_t95_s'][0])}, {f(r['date_t95_s'][1])}] | {r['positive_dates']}/7 "
                     f"| {f(r['left_turns']['left_east']['difference_s'], 2)} | {f(r['left_turns']['left_south']['difference_s'], 2)} "
                     f"| {r['long_waits']['candidate']:.1f}/{r['long_waits']['m4']} | {len(r['collisions']['policy_specific'])} |")
    lines += ["", "## 학습의 몫 L = 후보 − TM4 (같은 상황, 날짜 7개, 초/대, +면 후보가 짧음)", "",
              "| 비교 | 평균 | t95 | 앞쪽이 나은 날짜 |", "|---|---:|---|---:|"]
    for key, p in report["paired"].items():
        lines.append(f"| {key.replace('-', ' − ')} | {f(p['mean_s'])} | [{f(p['t95_s'][0])}, {f(p['t95_s'][1])}] | {p['positive_dates']}/{p['dates']} |")
    if report.get("per_model_learning"):
        lines += ["", "모델별 L(난수 42·43·44): " + "; ".join(
            f"{n} " + " / ".join(f(p["mean_s"]) for p in rows) for n, rows in report["per_model_learning"].items())]
    pairs = {n: d["pairs"] for n, d in report["deviations"].items() if d.get("pairs")}
    if pairs:
        lines += ["", "바꿈 종류(TM4 녹색 → 고른 녹색, 많은 순 3개): " + "; ".join(
            f"{n} " + ", ".join(f"{k} {100 * v['share']:.0f}%" for k, v in list(rows.items())[:3]) for n, rows in pairs.items())]
    if report.get("baseline_table"):
        lines += ["", "## 고정신호·P0와 비교(같은 84상황, 차량당 지연 초/대)", "",
                  "| 제어 | 차량당 지연 | 고정신호 대비 감소 | P0 대비 이득 [t95] |", "|---|---:|---:|---|"]
        for name, row in report["baseline_table"]["table"].items():
            v = row.get("vs_P0")
            cell = "기준" if v is None else f"{f(v['gain_date_mean_s'], 2)} [{f(v['date_t95_s'][0], 2)}, {f(v['date_t95_s'][1], 2)}]"
            lines.append(f"| {name} | {row['mean_delay_s']:.2f} | {row['reduction_vs_fixc_pct']:.1f}% | {cell} |")
    lines.append("")
    for name, v in report.get("verdicts", {}).items():
        lines.append(f"- {name}: {'**통과**' if v['passed'] else '통과 못 함'} — ARENA3 {v['arena3_status']}, L: {v['learning_reading']}")
    if "selected" in report:
        lines.append(f"- **확인할 후보: {report['selected'] or '없음(확인을 돌리지 않음)'}**")
    lines.append("- " + report["note"])
    return "\n".join(lines) + "\n"


def save_report(out: Path, report: dict, title: str) -> None:
    out = Path(out)
    s2e._write(out / "judge.json", report)
    (out / "summary.md").write_text(render(report, title), encoding="utf-8")


def confirmation_report(cand_folder: Path, base_folder: Path, candidate: str, cfg: dict) -> dict:
    lock, pcfg = ac.config(), pt.config()
    sits = pl.situations(pcfg, "confirmation")
    models, record = _candidate_rows(cand_folder, candidate, float(cfg["candidates"][candidate]["tau"]), sits, cfg)
    base_record = s2e.folder_record(base_folder)
    if base_record["evaluation_fingerprint"] != record["evaluation_fingerprint"]:
        raise ValueError("확인 기준선과 후보의 평가 지문이 다릅니다.")
    base = {n: pl._rows(base_folder, n, f"P/{n}", sits, base_record) for n in ("M4", "TM4")}
    missing = {n: len(sits) - len(r) for n, r in base.items() if len(r) != len(sits)}
    if any(len(r) != len(sits) for r in models):
        missing[candidate] = [len(sits) - len(r) for r in models]
    if missing:
        return {"status": "incomplete", "missing": missing, "part": "confirmation"}
    results = {"TM4": ref.judge([base["TM4"]], base["M4"], None, lock, sits), candidate: ref.judge(models, base["M4"], None, lock, sits)}
    t975 = float(lock["judge"]["t975"])
    paired = {f"{candidate}-TM4": pl.paired_difference(results[candidate], results["TM4"], t975)}
    return {"status": "done", "part": "confirmation", "candidates": {candidate: float(cfg["candidates"][candidate]["tau"])},
            "results": results, "paired": paired, "deviations": {candidate: deviation_summary(models)},
            "evaluation_fingerprint": record["evaluation_fingerprint"], "updated_unix": time.time(),
            "note": "아직 쓰지 않은 확인 84상황(C1·C2). 같은 7일의 새 도착 배치이고 무리 도착은 모형이다."}


def confirm(judge_dir: Path, students: Path, out: Path, work: Path, workers: int, cfg: dict, *, executor=None,
            worker=eval_worker, base_worker=pl.eval_worker) -> dict:
    """개발 판정이 고른 후보만 확인 84상황(P)에서 M4·TM4와 함께 한 번 잰다. 고른 후보가 없으면 SUMO를 돌리지 않는다."""
    out = Path(out)
    report = w.read(Path(judge_dir) / "judge.json")
    if report.get("status") != "done" or report.get("part") != "eval":
        raise ValueError("개발 판정이 끝나기 전에는 확인하지 않습니다.")
    candidate = report.get("selected")
    if not candidate:
        result = {"status": "not_run_development_failed", "verdicts": report["verdicts"]}
        s2e._write(out / "confirmation.json", result)
        return result
    status = pl.baseline(out / "baseline", Path(work) / "baseline", workers, pt.config(), "confirmation", executor=executor, worker=base_worker)
    status_c = evaluate(candidate, students, out / candidate, Path(work) / candidate, workers, cfg, "confirmation",
                        executor=executor, worker=worker)
    if status["failed"] or status_c["failed"]:
        return {"status": "runs_failed", "failed": status["failed"] + status_c["failed"]}
    rep = confirmation_report(out / candidate, out / "baseline", candidate, cfg)
    save_report(out, rep, f"확인(C1·C2), 후보 {candidate}")
    passed = rep["status"] == "done" and rep["results"][candidate]["status"] == "pass" \
        and rep["paired"][f"{candidate}-TM4"]["t95_s"][0] > 0
    result = {"status": "pass" if passed else "fail", "candidate": candidate, "learning": rep.get("paired", {}).get(f"{candidate}-TM4")}
    s2e._write(out / "confirmation.json", result)
    return result


# ----------------------------------------------------------------------------- 명령
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("smoke")
    p.add_argument("--candidate", required=True)
    p.add_argument("--students", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--platoon0", type=Path, required=True)
    p.add_argument("--platoon1", type=Path, required=True)
    p = sub.add_parser("evaluate")
    p.add_argument("--candidate", required=True)
    p.add_argument("--students", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1))
    p = sub.add_parser("judge")
    p.add_argument("--eval", type=Path, required=True, help="후보별 평가 폴더의 부모")
    p.add_argument("--platoon0", type=Path, required=True)
    p.add_argument("--platoon1", type=Path, required=True)
    p.add_argument("--pbase", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("confirm")
    p.add_argument("--judge", type=Path, required=True)
    p.add_argument("--students", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1))
    args = parser.parse_args()
    w.configure()
    torch.set_num_threads(1)
    cfg = config()
    if args.cmd in ("evaluate", "confirm") and platform.system() == "Darwin":
        raise SystemExit("맥에서는 본 평가를 실행하지 않습니다. Colab 노트북에서 실행하세요(맥은 smoke·judge만).")
    if args.cmd == "smoke":
        report = smoke(args.candidate, args.students, args.out, args.work, args.workers, args.platoon0, args.platoon1, cfg)
        return 0 if report["ok"] else 2
    if args.cmd == "evaluate":
        return 1 if evaluate(args.candidate, args.students, args.out, args.work, args.workers, cfg)["failed"] else 0
    if args.cmd == "judge":
        title = "개발 평가(평가 84상황, 무리 도착 세계)"
        report = collect_report(args.eval, args.platoon0, args.platoon1, args.pbase, cfg)
        save_report(args.out, report, title)
        print(render(report, title))
        return 0 if report["status"] == "done" else 2
    result = confirm(args.judge, args.students, args.out, args.work, args.workers, cfg)
    print(json.dumps(w.plain(result), ensure_ascii=False, indent=2))
    return 1 if result["status"] == "runs_failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
