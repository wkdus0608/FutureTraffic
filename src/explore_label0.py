"""[추가 탐색] LABEL0 — 5분 점수의 끝 순간 항 무게만 바꿔 학생을 다시 학습한다(사전 등록 results/explore/label0_20261102/사전등록.md).

비용 = 300초 지연 + w × 300초째 멈춘 차 수. 자료에 저장된 값으로 다시 계산하므로 새 굴림은 없다. 자료·신경망·학습 설정·심판은 그대로다.
무리 세계(P)는 PLATOON1 자료·PT1과 같은 평가, 무작위 세계(R)는 PI R1 자료·R1A와 같은 평가를 쓴다. 이전 코드·결과는 고치지 않는다.
평가는 Colab 또는 맥에서 돈다(맥은 11/02 사용자 요청). 맥은 numpy·torch 버전만 달라 평가 지문이 다르다. 판정은 그 차이만 있는지 확인하고 기록한다.

사용(FOCUSED_RULESET=pet1 FOCUSED_NORTH_EXT=240):
  python src/explore_label0.py train --variant P_w0 --data <Drive>/explore_platoon1/data --out <Drive>/explore_label0/P_w0/students
  python src/explore_label0.py smoke --variant P_w0 --data <...> --students <...> --reference-students <Drive>/explore_platoon1/students \\
      --reference-eval <Drive>/explore_platoon1/eval/PT1 --out <Drive>/explore_label0/P_w0/smoke --work /content/ls --evaluation <Drive>/explore_platoon0
  python src/explore_label0.py evaluate --variant P_w0 --students <...> --out <Drive>/explore_label0/P_w0/eval --work /content/l --workers 8
  python src/explore_label0.py judge --root <Drive>/explore_label0 --platoon0 <Drive>/explore_platoon0 --platoon1 <Drive>/explore_platoon1 \\
      --pi <Drive>/arena3/pi/R1 --stage2 <Drive>/arena3/stage2 --baseline <Drive>/arena3/referee/baseline \\
      --pbase <Drive>/explore_platoon_base --out <Drive>/explore_label0/judge
"""
from __future__ import annotations

import argparse
import contextlib
import gzip
import json
import platform
import shutil
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import arena3_common as ac  # noqa: E402
import arena3_pi as pi  # noqa: E402
import arena3_pi_learn as pil  # noqa: E402
import arena3_referee as ref  # noqa: E402
import arena3_stage2_eval as s2e  # noqa: E402
import explore_platoon1 as pt  # noqa: E402
import explore_platoon1_learn as pl  # noqa: E402
import explore_platoon1c as pc  # noqa: E402
import explore_platoon_base as pb  # noqa: E402
import explore_weight as w  # noqa: E402

EXPERIMENT = "LABEL0_V1"
CONFIG = ROOT / "config/explore_label0_v1.json"
PREREG = ROOT / "results/explore/label0_20261102/사전등록.md"
SOURCES = ("src/explore_label0.py", "config/explore_label0_v1.json", "results/explore/label0_20261102/사전등록.md")
STORED_WEIGHT = 75.0
ROUNDING = 0.01                               # 저장값이 소수 셋째 자리로 반올림돼 있다
GREEN_NAMES = {"0": "석천로 좌회전", "1": "길주로 직진", "2": "길주로 좌회전", "3": "남북 직진"}


def config() -> dict:
    cfg = w.read(CONFIG)
    if cfg.get("experiment") != EXPERIMENT:
        raise ValueError("LABEL0 설정이 아닙니다.")
    for key, path in (("prereg_sha256", PREREG), ("lock_sha256", ac.CONFIG), ("platoon1_config_sha256", pt.CONFIG),
                      ("pi_config_sha256", pi.config_path("R1"))):
        if cfg.get(key) != w.file_hash(path):
            raise ValueError(f"LABEL0 설정을 만든 때와 파일이 다릅니다: {Path(path).name}")
    return cfg


def world_parts(world: str) -> tuple:
    """세계 → (학습 모듈, 그 설정, 평가 정책 이름 앞부분)."""
    if world == "P":
        return pl, pt.config(), "PT1"
    if world == "R":
        return pil, pi.config("R1"), "R1A"
    raise ValueError(f"세계는 P 또는 R: {world}")


# ----------------------------------------------------------------------------- 라벨
def relabel_point(point: dict, weight: float) -> dict:
    """지점 하나의 비용을 '지연 + weight × 끝 순간 멈춘 차'로 바꾼 사본. 저장된 비용이 공식과 다르면 멈춘다."""
    rollouts = {}
    for green, runs in point["rollouts"].items():
        out = []
        for s in runs:
            if s is None:
                out.append(None)
                continue
            delay, residual, cost = float(s["delay"]), float(s["residual"]), float(s["cost"])
            if abs(cost - (delay + STORED_WEIGHT * residual)) > ROUNDING:
                raise ValueError(f"저장된 비용이 지연 + 75 × 멈춘 차와 다릅니다: {point.get('state')} {green}")
            out.append(s if weight == STORED_WEIGHT else {**s, "cost": delay + float(weight) * residual})
        rollouts[green] = out
    return {**point, "rollouts": rollouts}


@contextlib.contextmanager
def labelled(module, weight: float):
    """이 with 안에서만 학습 모듈이 읽는 지점의 비용을 바꾼다. 이전 실험 파일은 고치지 않는다."""
    original = module.point_to_row

    def point_to_row(point, date, situation, cfg):
        return original(relabel_point(point, weight), date, situation, cfg)

    module.point_to_row = point_to_row
    try:
        yield
    finally:
        module.point_to_row = original


def load_dataset(world: str, data: Path, weight: float) -> dict:
    module, cfg, _ = world_parts(world)
    with labelled(module, weight):
        dataset = module.load_dataset(Path(data), cfg)
    dataset["manifest"] = {**dataset["manifest"], "label_end_weight": float(weight),
                           "labels": f"paired 300-second delay + {weight:g} × end-moment stopped vehicles, base policy continuation"}
    dataset["fingerprint"] = w.digest(dataset["manifest"])
    return dataset


# ----------------------------------------------------------------------------- 학생
def _meta(world: str, cfg: dict, seed: int, dataset: dict, stats: dict, weight: float, module) -> dict:
    common = {"seed": int(seed), "stage": "final", "dataset_fingerprint": dataset["fingerprint"], "network": list(cfg["student"]["network"]),
              "target": cfg["student"]["target"], "training": stats, "label_end_weight": float(weight)}
    if world == "P":
        return {"round": pt.ROUND, **common, "observation_size": pt.OBS, "config_sha256": w.file_hash(pt.CONFIG),
                "source_sha256": {p: w.file_hash(ROOT / p) for p in module.eval_sources()}}
    return {"round": cfg["round"], **common, "observation_size": pi.OBS, "config_sha256": w.file_hash(pi.config_path(cfg["round"])),
            "source_sha256": {p: w.file_hash(ROOT / p) for p in module.eval_sources(cfg["round"])}}


def train(variant: str, data: Path, out: Path, cfg: dict) -> dict:
    spec = cfg["variants"][variant]
    world, weight = spec["world"], float(spec["weight"])
    module, mcfg, _ = world_parts(world)
    torch.set_num_threads(1)
    dataset = load_dataset(world, data, weight)
    out = Path(out)
    s2e._write(out / "dataset_manifest.json", {"fingerprint": dataset["fingerprint"], **{k: v for k, v in dataset["manifest"].items() if k != "future_partitions"}})
    summary = {}
    for seed, folder in zip(mcfg["student"]["seeds"], module.student_dirs(out, mcfg)):
        if (folder / "manifest.json").exists():
            m = w.read(folder / "manifest.json")
            if m.get("dataset_fingerprint") != dataset["fingerprint"] or m.get("seed") != seed or m.get("label_end_weight") != weight:
                raise ValueError(f"이미 있는 학생이 다른 자료·무게·난수로 학습됐습니다: {folder}")
            summary[seed] = {"model_sha256": module.student_identity(folder), "reused": True}
            continue
        net, stats = module.train_student(dataset, mcfg, seed)
        summary[seed] = {"model_sha256": module.save_student(folder, net, _meta(world, mcfg, seed, dataset, stats, weight, module)), "reused": False, **stats}
        print(f"{variant} 학생 난수 {seed}: 지점 {stats['points']}, 손실 {stats['initial_full_loss']:.4f} → {stats['final_full_loss']:.4f} ({stats['seconds']:.0f}초)", flush=True)
    s2e._write(out / "students.json", {"variant": variant, "label_end_weight": weight, "dataset_fingerprint": dataset["fingerprint"], "students": summary,
                                       "updated_unix": time.time()})
    return summary


def same_model(a, b) -> bool:
    x, y = a.state_dict(), b.state_dict()
    return list(x) == list(y) and all(torch.equal(x[k], y[k]) for k in x)


# ----------------------------------------------------------------------------- 평가
def mac_worker(job: dict) -> dict:
    """맥 평가 한 판(사용자 요청 11/02). 정책·실행기·기록은 pl.eval_worker·pil.eval_worker와 같고, Colab 전용 검사만 없다."""
    started = time.monotonic()
    work = Path(job["work"])
    try:
        w.configure()
        torch.set_num_threads(1)
        shutil.rmtree(work, ignore_errors=True)
        if job["kind"] == "student":
            if pl.student_identity(Path(job["model_dir"])) != job["model_identity"]:
                raise ValueError("작업 시작 후 모델이 바뀌었습니다.")
            policy = pt.StudentPolicy(pl.load_student(Path(job["model_dir"])), name=job["policy"])
            result = pt.run_episode(job["situation"], work, policy, max_seconds=3600.0)
            extra = {"information": policy.info() if hasattr(policy, "info") else {"kind": job["kind"]}}
        elif job["kind"] == "pi":
            if pil.student_identity(Path(job["model_dir"])) != job["model_identity"]:
                raise ValueError("작업 시작 후 모델이 바뀌었습니다.")
            policy = pi.GreenPolicy(pil.load_student(Path(job["model_dir"])), job["variant"], name=job["policy"])
            result = ac.run_arena_episode(job["situation"], work, policy, max_seconds=1800.0)
            extra = {"variant": job["variant"]}
        else:
            raise ValueError(f"알 수 없는 작업 종류: {job['kind']}")
        if result.metrics["route_sha256"] != job["situation"].get("route_sha256"):
            raise ValueError(f"평가 차량 파일의 SHA가 설정과 다릅니다: {result.metrics['route_sha256']}")
        ac.write_run(Path(job["target"]), result, {"policy": job["policy"], "fingerprint": job["fingerprint"], "model_identity": job.get("model_identity"),
                                                   "job_seconds": time.monotonic() - started, **extra})
        return {"ok": True, "seconds": time.monotonic() - started, "name": f"{job['policy']}/{job['sid']}"}
    except BaseException as error:
        return {"ok": False, "seconds": time.monotonic() - started, "name": f"{job['policy']}/{job['sid']}", "error": f"{type(error).__name__}: {error}"}
    finally:
        shutil.rmtree(work, ignore_errors=True)


def evaluate(variant: str, students: Path, out: Path, work: Path, workers: int, cfg: dict) -> dict:
    on_mac = platform.system() == "Darwin"
    spec = cfg["variants"][variant]
    module, mcfg, prefix = world_parts(spec["world"])
    folders = module.student_dirs(Path(students), mcfg)
    for folder in folders:
        if w.read(folder / "manifest.json").get("label_end_weight") != float(spec["weight"]):
            raise ValueError(f"학생의 라벨 무게가 후보와 다릅니다: {folder}")
    s2e._freeze(Path(out) / "label.json", {"variant": variant, "world": spec["world"], "label_end_weight": float(spec["weight"]),
                                           "machine": "mac" if on_mac else "colab"})
    kwargs = {"worker": mac_worker} if on_mac else {}
    if on_mac:
        pl._prepare_map_cache()
    if spec["world"] == "P":
        return module.evaluate(folders, Path(out), Path(work), workers, mcfg, "eval", **kwargs)
    return module.evaluate(prefix, folders, Path(out), Path(work), workers, mcfg, "eval", **kwargs)


def smoke_run(job: dict) -> dict:
    """작은 실행 한 판(맥에서도 돈다). 정책과 실행기는 본 평가와 같다."""
    started = time.monotonic()
    work = Path(job["work"])
    try:
        w.configure()
        torch.set_num_threads(1)
        shutil.rmtree(work, ignore_errors=True)
        if job["world"] == "P":
            policy = pt.StudentPolicy(pl.load_student(Path(job["model_dir"])), name=job["policy"])
            result = pt.run_episode(job["situation"], work, policy, max_seconds=3600.0)
        else:
            policy = pi.GreenPolicy(pil.load_student(Path(job["model_dir"])), "A", name=job["policy"])
            result = ac.run_arena_episode(job["situation"], work, policy, max_seconds=1800.0)
        if result.metrics["route_sha256"] != job["situation"]["route_sha256"]:
            raise ValueError("차량 파일 SHA가 설정과 다릅니다.")
        ac.write_run(Path(job["target"]), result, {"policy": job["policy"], "fingerprint": job["fingerprint"], "model_identity": job["model_identity"],
                                                   "job_seconds": time.monotonic() - started})
        return {"ok": True}
    except BaseException as error:
        return {"ok": False, "error": f"{type(error).__name__}: {error}"}
    finally:
        shutil.rmtree(work, ignore_errors=True)


def smoke(variant: str, data: Path, students: Path, reference_students: Path, reference_eval: Path, out: Path, work: Path, evaluation: Path | None, cfg: dict) -> dict:
    """무게 75 라벨이 원래 자료와 같은지, 후보 라벨이 달라졌는지, 학생 한 판이 온전한지 본다."""
    spec = cfg["variants"][variant]
    world, weight = spec["world"], float(spec["weight"])
    module, mcfg, prefix = world_parts(world)
    torch.set_num_threads(1)
    original = module.load_dataset(Path(data), mcfg)
    same_75 = load_dataset(world, data, STORED_WEIGHT)
    changed = load_dataset(world, data, weight)
    keys = [k for k in module.ARRAY_KEYS if k in original]
    checks = {"label75_equals_original": all(np.array_equal(original[k], same_75[k]) for k in keys),
              "label_changed": not np.array_equal(original["gains"], changed["gains"]),
              "same_observations": np.array_equal(original["observations"], changed["observations"]),
              "weight_in_fingerprint": changed["fingerprint"] != same_75["fingerprint"]}
    # 무게 75로 다시 학습하면 기존 학생과 같은가(기록만: 기기·패키지가 다르면 달라질 수 있다)
    net75, _ = module.train_student(same_75, mcfg, mcfg["student"]["seeds"][0], epochs=None)
    reference_folder = module.student_dirs(Path(reference_students), mcfg)[0]
    retrain_same = same_model(net75, module.load_student(reference_folder))
    folder = module.student_dirs(Path(students), mcfg)[0]
    new_net = module.load_student(folder)
    checks["student_differs_from_w75"] = not same_model(new_net, module.load_student(reference_folder))
    fp_ = ref.register(Path(out), extra_files=SOURCES)
    pl._prepare_map_cache()
    sits = pl.situations(mcfg, "eval") if world == "P" else s2e.situations_for("eval")
    s = sits[0]
    sid = ac.situation_id(s)
    identity = module.student_identity(folder)
    name = f"{prefix}_m0"
    job = {"world": world, "policy": name, "situation": s, "sid": sid, "fingerprint": fp_, "model_dir": str(folder), "model_identity": identity,
           "target": str(Path(out) / "runs" / name / sid), "work": str(Path(work) / name)}
    if ref.cached(job) is None:
        done = smoke_run(job)
        if not done["ok"]:
            raise RuntimeError(f"작은 실행이 실패했습니다: {done}")
    run = ac.read_run(Path(job["target"]), fp_, identity)
    old = w.read(Path(reference_eval) / "runs" / name / sid / "metrics.json")
    checks.update(quality_clean=not ref.quality_errors(run), route_pinned=run["route_sha256"] == s["route_sha256"],
                  same_cars_as_w75=run["cohort_sha256"] == old["cohort_sha256"] and run["vehicles"] == old["vehicles"],
                  signal_differs_from_w75=run["executed_sha16"] != old["executed_sha16"])
    if platform.system() != "Darwin" and evaluation is not None:      # 맥은 패키지 버전이 달라 평가 지문이 다르다
        checks["same_evaluation"] = ref.evaluation_fingerprint(Path(out)) == ref.evaluation_fingerprint(Path(evaluation))
    report = {"ok": all(checks.values()), "checks": checks, "variant": variant, "situation": sid, "delay_s": run["mean_delay_s"], "delay_w75_s": old["mean_delay_s"],
              "retrain_w75_same_model": retrain_same, "seconds": run["job_seconds"], "fingerprint": fp_}
    w.write(Path(out) / "smoke.json", report)
    ref.stamp(f"LABEL0 작은 실행 {variant}: {'통과' if report['ok'] else '실패'} {checks} | 지연 {run['mean_delay_s']:.2f} (무게 75: {old['mean_delay_s']:.2f}) | 무게 75 재학습 일치 {retrain_same}")
    shutil.rmtree(work, ignore_errors=True)
    return report


# ----------------------------------------------------------------------------- 판정
def effect_verdict(diff: dict, rules: dict) -> bool:
    return bool(diff["mean_s"] >= float(rules["effect_min_gain_s"]) and diff["t95_s"][0] > 0 and diff["positive_dates"] >= int(rules["effect_min_positive_dates"]))


def read_learning(diff: dict) -> str:
    low, high = diff["t95_s"]
    return "출발 정책보다 짧다" if low > 0 else ("출발 정책보다 길다" if high < 0 else "구분 못 함")


def overall_reading(effects: dict, deltas: dict, weights: list[str]) -> dict:
    """effects[세계][무게] = 효과 있음. 사전 등록의 A·B·C."""
    both = [k for k in weights if effects["P"].get(k) and effects["R"].get(k)]
    if both:
        best = max(both, key=lambda k: (deltas["P"][k] + deltas["R"][k]) / 2)
        return {"reading": "A", "weight": best, "text": "두 세계 모두 효과 있음: 끝 항이 큰 원인이다. 이 무게로 신경망만 학습을 설계한다."}
    one = [(world, k) for world in ("P", "R") for k in weights if effects[world].get(k)]
    if one:
        return {"reading": "B", "weight": None, "where": one, "text": "한 세계에서만 효과 있음: 세계 차이를 먼저 본다. 다음 단계는 사용자와 정한다."}
    return {"reading": "C", "weight": None, "text": "효과 없음: 끝 항이 주원인이 아니다. 굴림 길이 점검을 먼저 한다."}


def phase_share(folders: list[Path], sids: list[str]) -> dict:
    """녹색별 시간 비중(%)과 전환 수. 없는 판은 건너뛴다."""
    counts, total, switches, runs = {str(g): 0 for g in range(4)}, 0, 0, 0
    for folder in folders:
        for sid in sids:
            path = Path(folder) / sid / "decisions.jsonl.gz"
            if not path.exists():
                continue
            with gzip.open(path, "rt") as f:
                executed = [int(json.loads(x)["executed_green"]) for x in f]
            for g in executed:
                counts[str(g)] += 1
            total += len(executed)
            switches += sum(a != b for a, b in zip(executed, executed[1:]))
            runs += 1
    return {"share_pct": {g: 100.0 * n / total for g, n in counts.items()} if total else {}, "switches": switches / runs if runs else None}


MAC_ALLOWED_PACKAGES = {"numpy", "torch"}


def differs_only_in_packages(a: dict, b: dict) -> bool:
    """두 평가 기록이 numpy·torch 버전 말고는 같은가(잠금·환경·코드·SUMO 버전). 맥과 Colab의 결과를 함께 읽어도 되는 조건이다."""
    if {k: v for k, v in a.items() if k != "packages"} != {k: v for k, v in b.items() if k != "packages"}:
        return False
    pa, pb_ = a["packages"], b["packages"]
    return set(pa) == set(pb_) and {k for k in pa if pa[k] != pb_[k]} <= MAC_ALLOWED_PACKAGES


def _reduction(m4_mean: float, gain_vehicle: float, fixed: float) -> float:
    return 100.0 * (fixed - (m4_mean - gain_vehicle)) / fixed


def world_report(world: str, root: Path, refs: dict, cfg: dict, lock: dict) -> dict:
    """한 세계의 후보를 무게 75 학생·출발 정책·고정신호와 비교한다."""
    t975 = float(lock["judge"]["t975"])
    rules, sits, m4, p0_rows, prefix = cfg["rules"], refs["situations"], refs["m4"], refs["p0"], refs["prefix"]
    sids = [ac.situation_id(s) for s in sits]
    route = {ac.situation_id(s): s["route_sha256"] for s in sits}
    fixed = pb.read_folder(refs["pbase"] / "runs" / world / "FIXC", sids, route, f"{world}/FIXC")
    fixed_mean, m4_mean = pb.policy_summary([fixed])["mean_delay_s"], pb.policy_summary([m4])["mean_delay_s"]
    out = {"fixed_mean_s": fixed_mean, "m4_mean_s": m4_mean, "fixed_left": pb.policy_summary([fixed])["left_turns_s"], "candidates": {}, "missing": {},
           "w75": {"gain_vs_m4_s": refs["w75"]["gain_date_mean_s"], "reduction_vs_fixed_pct": _reduction(m4_mean, refs["w75"]["gain_vehicle_weighted_s"], fixed_mean),
                   "left": {k: v["difference_s"] for k, v in refs["w75"]["left_turns"].items()}, "signal": phase_share(refs["w75_folders"], sids)},
           "anchor": {"name": refs["anchor_name"], "gain_vs_m4_s": refs["anchor"]["gain_date_mean_s"],
                      "reduction_vs_fixed_pct": _reduction(m4_mean, refs["anchor"]["gain_vehicle_weighted_s"], fixed_mean),
                      "signal": phase_share(refs["anchor_folders"], sids)}}
    fingerprints = set(refs["fingerprints"])
    reference_record = refs["reference_record"]
    for key, weight in cfg["weights"].items():
        variant = f"{world}_{key}"
        folder = Path(root) / variant / "eval"
        if not (folder / "models.json").exists():
            out["missing"][variant] = "평가 폴더 없음"
            continue
        if w.read(folder / "label.json").get("label_end_weight") != float(weight):
            raise ValueError(f"평가 폴더의 라벨 무게가 후보와 다릅니다: {variant}")
        decl = w.read(folder / "models.json")
        models = [s2e.verified_runs(folder, f"{prefix}_m{i}", sits, model_identity=x, complete=False) for i, x in enumerate(decl["model_identities"])]
        if any(len(r) != len(sits) for r in models):
            out["missing"][variant] = [len(r) for r in models]
            continue
        record = s2e.folder_record(folder)
        if record["evaluation_fingerprint"] not in fingerprints:     # 맥 실행: numpy·torch 버전만 다른 것을 확인하고 기록한다
            if not differs_only_in_packages(record["evaluation"], reference_record["evaluation"]):
                raise ValueError(f"평가 기록이 패키지 버전 말고도 다릅니다: {variant}")
            out.setdefault("package_only_differences", {})[variant] = record["evaluation"]["packages"]
        result = ref.judge(models, m4, p0_rows, lock, sits)
        delta = pl.paired_difference(result, refs["w75"], t975)
        learning = pl.paired_difference(result, refs["anchor"], t975)
        run_folders = [folder / "runs" / f"{prefix}_m{i}" for i in range(len(models))]
        out["candidates"][key] = {
            "variant": variant, "weight": float(weight), "effect": effect_verdict(delta, rules), "delta_vs_w75": delta, "learning": learning,
            "learning_reading": read_learning(learning), "arena3_status": result["status"], "failed_checks": [k for k, v in result["checks"].items() if not v],
            "gain_vs_m4_s": result["gain_date_mean_s"], "gain_vs_m4_t95": result["date_t95_s"], "gain_vehicle_weighted_s": result["gain_vehicle_weighted_s"],
            "reduction_vs_fixed_pct": _reduction(m4_mean, result["gain_vehicle_weighted_s"], fixed_mean),
            "left": {k: v["difference_s"] for k, v in result["left_turns"].items()}, "left_s": {k: v["candidate_s"] for k, v in result["left_turns"].items()},
            "long_waits": result["long_waits"], "collisions": result["collisions"], "quality_errors": result["quality_errors"],
            "signal": phase_share(run_folders, sids),
            "per_model_delta": [pl.paired_difference(ref.judge([rows], m4, p0_rows, lock, sits), refs["w75"], t975) for rows in models]}
    out["evaluation_fingerprints"] = sorted(fingerprints)
    if len(fingerprints) != 1:
        raise ValueError(f"결과 폴더들의 평가 지문이 다릅니다: {sorted(fingerprints)}")
    return out


def collect_report(root: Path, platoon0: Path, platoon1: Path, pi_root: Path, stage2: Path, baseline: Path, pbase: Path, cfg: dict) -> dict:
    lock = ac.config()
    pcfg = pt.config()
    sits_p = pl.situations(pcfg, "eval")
    pc_cfg = pc.config()
    rows = pc.reference_rows(Path(platoon0), Path(platoon1), sits_p, pc_cfg)
    m4_p = rows["base"]["M4"]
    pt1_eval = Path(platoon1) / "eval" / "PT1"
    refs_p = {"situations": sits_p, "m4": m4_p, "p0": None, "prefix": "PT1", "pbase": Path(pbase), "anchor_name": "TM4",
              "w75": ref.judge(rows["PT1"], m4_p, None, lock, sits_p), "anchor": ref.judge([rows["base"]["TM4"]], m4_p, None, lock, sits_p),
              "w75_folders": [pt1_eval / "runs" / f"PT1_m{i}" for i in range(3)], "anchor_folders": [Path(platoon0) / "runs" / "P" / "TM4"],
              "fingerprints": rows["fingerprints"]}
    sits_r = s2e.situations_for("eval")
    base_record = s2e.verify_baseline(Path(baseline))
    m4_r = s2e.verified_runs(Path(baseline), "M4", sits_r)
    p0_r = s2e.verified_runs(Path(baseline), "P0", sits_r)
    old = w.read(Path(pi_root) / "judge" / "judge.json")
    if old["evaluation_fingerprint"] != base_record["evaluation_fingerprint"]:
        raise ValueError("PI R1 판정과 기준선의 평가 지문이 다릅니다.")
    refs_r = {"situations": sits_r, "m4": m4_r, "p0": p0_r, "prefix": "R1A", "pbase": Path(pbase), "anchor_name": "S2A_pre",
              "w75": old["results"]["R1A"], "anchor": old["results"]["S2A_pre"],
              "w75_folders": [Path(pi_root) / "eval" / "R1A" / "runs" / f"R1A_m{i}" for i in range(3)],
              "anchor_folders": [Path(stage2) / "eval" / "S2A_pre" / "runs" / f"S2A_pre_m{i}" for i in range(3)],
              "fingerprints": [base_record["evaluation_fingerprint"]]}
    refs_p["reference_record"] = s2e.folder_record(Path(platoon0))
    refs_r["reference_record"] = base_record
    worlds = {"P": world_report("P", root, refs_p, cfg, lock), "R": world_report("R", root, refs_r, cfg, lock)}
    missing = {**worlds["P"]["missing"], **worlds["R"]["missing"]}
    report = {"experiment": EXPERIMENT, "status": "incomplete" if missing else "done", "missing": missing, "worlds": worlds, "lock_sha256": w.file_hash(ac.CONFIG),
              "config_sha256": w.file_hash(CONFIG), "updated_unix": time.time(),
              "note": "반복 사용 날짜의 SUMO 개발 비교(확인 상황 안 씀). 학생 입력에 계획기 출력이 있어 시연 후보가 아니다. 무리 도착은 모형이다."}
    if not missing:
        weights = list(cfg["weights"])
        effects = {x: {k: worlds[x]["candidates"][k]["effect"] for k in weights} for x in worlds}
        deltas = {x: {k: worlds[x]["candidates"][k]["delta_vs_w75"]["mean_s"] for k in weights} for x in worlds}
        report["reading"] = overall_reading(effects, deltas, weights)
    return report


def render(report: dict) -> str:
    f = lambda x: f"{x:+.3f}"  # noqa: E731
    lines = ["# LABEL0 — 끝 순간 항 무게만 바꾼 학생 (같은 84상황)", "",
             "Δ = 같은 세계 무게 75 학생 대비(날짜 7개 평균, +면 짧음). 몫 = 출발 정책 대비. 감소율은 보정 고정신호 대비.", ""]
    if report.get("reading"):
        lines += [f"**읽기 {report['reading']['reading']}:** {report['reading']['text']}" + (f" (무게 {report['reading']['weight']})" if report['reading'].get("weight") else ""), ""]
    names = {"P": "무리 도착 세계(무게 75 = PT1, 출발 = TM4)", "R": "무작위 도착 세계(무게 75 = R1A, 출발 = S2A_pre)"}
    for world, part in report["worlds"].items():
        lines += [f"## {names[world]}", "", f"고정신호 {part['fixed_mean_s']:.2f}초. 고정신호 좌회전 동 {part['fixed_left']['left_east']:.1f}·남 {part['fixed_left']['left_south']:.1f}초.", "",
                  "| 후보 | 효과 | Δ [t95] | 양수 날짜 | 몫 [t95] | 고정신호 대비 | 남북 직진 비중 | 동 좌(M4 대비) | 남 좌(M4 대비) | ARENA3 |",
                  "|---|---|---|---:|---|---:|---:|---:|---:|---|"]
        a = part["anchor"]
        lines.append(f"| 출발 {a['name']} | - | - | - | - | {a['reduction_vs_fixed_pct']:.1f}% | {a['signal']['share_pct'].get('3', 0):.1f}% | - | - | - |")
        x = part["w75"]
        lines.append(f"| 무게 75 | 기준 | 기준 | - | - | {x['reduction_vs_fixed_pct']:.1f}% | {x['signal']['share_pct'].get('3', 0):.1f}% | {x['left']['left_east']:+.2f} | {x['left']['left_south']:+.2f} | - |")
        for key, c in part["candidates"].items():
            d, l = c["delta_vs_w75"], c["learning"]
            lines.append(f"| 무게 {c['weight']:g} | {'있음' if c['effect'] else '없음'} | {f(d['mean_s'])} [{f(d['t95_s'][0])}, {f(d['t95_s'][1])}] | {d['positive_dates']}/7 | "
                         f"{f(l['mean_s'])} [{f(l['t95_s'][0])}, {f(l['t95_s'][1])}] | {c['reduction_vs_fixed_pct']:.1f}% | {c['signal']['share_pct'].get('3', 0):.1f}% | "
                         f"{c['left']['left_east']:+.2f} | {c['left']['left_south']:+.2f} | {c['arena3_status']} |")
        if part["missing"]:
            lines += ["", f"아직 안 끝남: {part['missing']}"]
        lines.append("")
    lines += [f"- {report['note']}"]
    return "\n".join(lines) + "\n"


def save_report(out: Path, report: dict) -> None:
    out = Path(out)
    s2e._write(out / "judge.json", report)
    (out / "summary.md").write_text(render(report), encoding="utf-8")


# ----------------------------------------------------------------------------- 명령
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("train")
    for name in ("variant", "data", "out"):
        p.add_argument(f"--{name}", required=True)
    p = sub.add_parser("smoke")
    for name in ("variant", "data", "students", "reference-students", "reference-eval", "out", "work"):
        p.add_argument(f"--{name}", required=True)
    p.add_argument("--evaluation", help="평가 지문을 대조할 이전 결과 폴더(Colab)")
    p = sub.add_parser("evaluate")
    for name in ("variant", "students", "out", "work"):
        p.add_argument(f"--{name}", required=True)
    p.add_argument("--workers", type=int, default=8)
    p = sub.add_parser("judge")
    for name in ("root", "platoon0", "platoon1", "pi", "stage2", "baseline", "pbase", "out"):
        p.add_argument(f"--{name}", required=True)
    args = parser.parse_args()
    w.configure()
    cfg = config()
    if args.cmd == "train":
        print(json.dumps(train(args.variant, Path(args.data), Path(args.out), cfg), ensure_ascii=False)[:600])
        return 0
    if args.cmd == "smoke":
        r = smoke(args.variant, Path(args.data), Path(args.students), Path(args.reference_students), Path(args.reference_eval), Path(args.out), Path(args.work),
                  Path(args.evaluation) if args.evaluation else None, cfg)
        print(json.dumps(r, ensure_ascii=False, indent=1))
        return 0 if r["ok"] else 2
    if args.cmd == "evaluate":
        s = evaluate(args.variant, Path(args.students), Path(args.out), Path(args.work), args.workers, cfg)
        print(json.dumps({"done": s["done"], "failed": len(s["failed"])}, ensure_ascii=False))
        return 1 if s["failed"] else 0
    report = collect_report(Path(args.root), Path(args.platoon0), Path(args.platoon1), Path(args.pi), Path(args.stage2), Path(args.baseline), Path(args.pbase), cfg)
    save_report(Path(args.out), report)
    print(render(report))
    return 0 if report["status"] == "done" else 3


if __name__ == "__main__":
    raise SystemExit(main())
