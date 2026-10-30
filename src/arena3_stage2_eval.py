"""Stage 2 evaluation on the unchanged ARENA3 referee, plus a frozen C1/C2 confirmation.

This module adds a model loader, not a new scoring rule. Old reference, environment,
and referee files are imported unchanged. A confirmation winner is selected from
complete development results before confirmation traffic is read or simulated.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import arena3_common as ac
import arena3_g1 as g1
import arena3_referee as ref
import explore_weight as w

EXTRA_FILES = ("src/arena3_g1.py", "src/arena3_stage2_data.py",
               "src/arena3_stage2_train.py", "src/arena3_stage2_eval.py",
               "config/arena3_stage2_v1.json")
FINAL_NAMES = ("S2A", "S2B")
VARIANTS = {"S2A": "advantage", "S2B": "policy"}


def training():
    import arena3_stage2_train
    return arena3_stage2_train


def _method_name(name: str) -> str:
    return name[:-4] if name.lower().endswith("_pre") else name


def situations_for(part: str) -> list[dict]:
    if part not in ("eval", "confirmation"):
        raise ValueError(f"알 수 없는 평가 배치: {part}")
    return ac.config()["situations_eval" if part == "eval" else "situations_confirmation"]


def _write(path: Path, value: dict) -> None:
    """A unique temporary name also works when independent notebooks report together."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    temp.write_text(json.dumps(w.plain(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    os.replace(temp, path)


def _freeze(path: Path, value: dict) -> dict:
    """Create immutable metadata; reusing an output folder never changes its models."""
    path = Path(path)
    if path.exists():
        previous = w.read(path)
        if previous != value:
            raise ValueError(f"고정한 모델/배치와 새 요청이 다릅니다: {path}")
        return previous
    _write(path, value)
    return value


def folder_record(folder: Path) -> dict:
    saved = w.read(Path(folder) / "experiment.json")
    if w.digest(saved["evaluation"]) != saved["evaluation_fingerprint"]:
        raise ValueError(f"평가 지문 기록이 손상됐습니다: {folder}")
    if w.digest({k: v for k, v in saved.items() if k != "fingerprint"}) != saved["fingerprint"]:
        raise ValueError(f"폴더 지문 기록이 손상됐습니다: {folder}")
    if saved["evaluation"]["lock_sha256"] != w.file_hash(ac.CONFIG):
        raise ValueError(f"원래 ARENA3 잠금과 다른 결과입니다: {folder}")
    return saved


def verify_baseline(baseline: Path, expected_evaluation: str | None = None) -> dict:
    record = folder_record(baseline)
    if expected_evaluation is not None and record["evaluation_fingerprint"] != expected_evaluation:
        raise ValueError("기준선과 학습기 평가 장치의 지문이 다릅니다.")
    reproduction = Path(baseline) / "reproduction.json"
    if not reproduction.exists() or not w.read(reproduction).get("passed"):
        raise ValueError("원래 기준선의 재현 점검이 통과하지 않았습니다.")
    return record


def verified_runs(folder: Path, policy: str, situations: list[dict], *,
                  model_identity: str | None = None, complete: bool = True) -> list[dict]:
    """Bind each cache to its containing experiment and the declared model identity."""
    folder = Path(folder)
    record = folder_record(folder)
    sids = {ac.situation_id(s) for s in situations}
    base = folder / "runs" / policy
    found = {p.name for p in base.iterdir() if p.is_dir() and (p / "metrics.json").exists()} if base.is_dir() else set()
    if found - sids:
        raise ValueError(f"다른 배치/상황의 결과가 섞였습니다: {policy}: {sorted(found - sids)[:3]}")
    rows = []
    for sid in sorted(found):
        m = ac.read_run(base / sid, record["fingerprint"], model_identity)
        if m is None:
            raise ValueError(f"완료 지표에 캐시가 없습니다: {policy}/{sid}")
        if m.get("fingerprint") != record["fingerprint"] or m.get("model_identity") != model_identity:
            raise ValueError(f"지표의 지문/모델이 선언과 다릅니다: {policy}/{sid}")
        if m.get("situation") != sid:
            raise ValueError(f"상황 식별자가 폴더와 다릅니다: {policy}/{sid}")
        rows.append(m)
    if complete and found != sids:
        raise ValueError(f"{policy}: {len(rows)}/{len(sids)}상황만 완료됐습니다.")
    return rows


def eval_worker(job: dict) -> dict:
    started = time.monotonic()
    work = Path(job["work"])
    try:
        if platform.system() == "Darwin":
            raise RuntimeError("실제 SUMO 평가는 Colab에서만 실행합니다. 맥에서는 저장 결과 판정만 가능합니다.")
        w.configure()
        g1.torch.set_num_threads(1)
        shutil.rmtree(work, ignore_errors=True)
        if job["kind"] == "fixed":
            policy = ac.fixed_policy(job["policy"])
        elif job["kind"] == "stage2":
            loader = training()
            if loader.model_identity(Path(job["model_dir"])) != job["model_identity"]:
                raise ValueError("작업 시작 후 모델이 바뀌었습니다.")
            policy = g1.GreedyPolicy(loader.load_model(Path(job["model_dir"])), name=job["policy"])
        else:
            raise ValueError(f"알 수 없는 작업 종류: {job['kind']}")
        result = ac.run_arena_episode(job["situation"], work, policy, max_seconds=1800.0)
        expected = job["situation"].get("route_sha256")
        if expected and result.metrics["route_sha256"] != expected:
            raise ValueError("평가 차량 파일의 SHA가 잠금과 다릅니다.")
        ac.write_run(Path(job["target"]), result,
                     {"policy": job["policy"], "fingerprint": job["fingerprint"],
                      "model_identity": job.get("model_identity"), "job_seconds": time.monotonic() - started})
        return {"ok": True, "seconds": time.monotonic() - started,
                "name": f"{job['policy']}/{job['sid']}"}
    except BaseException as error:
        return {"ok": False, "seconds": time.monotonic() - started,
                "name": f"{job['policy']}/{job['sid']}", "error": f"{type(error).__name__}: {error}"}
    finally:
        shutil.rmtree(work, ignore_errors=True)


def model_declaration(name: str, folders: list[Path], part: str) -> dict:
    method = _method_name(name)
    if method not in FINAL_NAMES or len(folders) != ac.config()["judge"]["learner_models"]:
        raise ValueError("학습기는 S2A/S2B의 세 모델이어야 합니다.")
    identities, seeds, datasets = [], [], []
    for folder in folders:
        identities.append(training().model_identity(Path(folder)))
        manifest = w.read(Path(folder) / "manifest.json")
        seeds.append(manifest.get("seed"))
        datasets.append(manifest.get("dataset_fingerprint"))
        if manifest.get("variant") != VARIANTS[method]:
            raise ValueError("평가 이름과 학습 방식이 다릅니다.")
        expected_stage = "pretrained" if name.lower().endswith("_pre") else "final"
        if manifest.get("stage") != expected_stage:
            raise ValueError("사전학습/최종 학습 단계와 평가 이름이 다릅니다.")
    if seeds != ac.config()["g1"]["seeds"]:
        raise ValueError("평가 모델의 순서는 난수 42·43·44여야 합니다.")
    if len(set(datasets)) != 1 or not datasets[0]:
        raise ValueError("세 모델이 같은 교사 자료를 사용하지 않았습니다.")
    return {"name": name, "part": part, "model_dirs": [str(Path(p).resolve()) for p in folders],
            "model_identities": identities, "seeds": seeds, "dataset_fingerprint": datasets[0]}


def evaluate(name: str, model_dirs: list[Path], out: Path, work: Path, workers: int,
             part: str = "eval", *, executor=None, worker=eval_worker) -> dict:
    situations = situations_for(part)
    fp = ref.register(Path(out), extra_files=EXTRA_FILES)
    declaration = model_declaration(name, model_dirs, part)
    _freeze(Path(out) / "models.json", declaration)
    jobs = []
    for i, (folder, identity) in enumerate(zip(model_dirs, declaration["model_identities"])):
        jobs += ref.jobs_for(Path(out), Path(work), fp, f"{name}_m{i}", situations,
                             model_dir=folder, model_identity=identity, kind="stage2")
    status = ref.run_jobs(jobs, max(1, workers), ref.Ledger(Path(out), name), worker=worker, executor=executor)
    _write(Path(out) / "status" / f"{name}.json",
           {**status, "total": len(jobs), "part": part, "updated_unix": time.time()})
    return status


def candidate_models(spec: str, situations: list[dict], part: str = "eval") -> tuple[str, Path, list[list[dict]]]:
    name, raw = spec.split("=", 1)
    folder = Path(raw)
    if not (folder / "experiment.json").exists():
        return name, folder, []
    declared = folder / "models.json"
    if declared.exists():
        model_record = w.read(declared)
        if model_record.get("name") != name or model_record.get("part") != part:
            raise ValueError("모델 선언의 이름/배치가 요청과 다릅니다.")
        folders = [Path(x) for x in model_record["model_dirs"]]
        if model_record != model_declaration(name, folders, part):
            raise ValueError("평가에 사용한 최종 모델이 바뀌었습니다.")
        return name, folder, [verified_runs(folder, f"{name}_m{i}", situations,
                                             model_identity=identity, complete=False)
                              for i, identity in enumerate(model_record["model_identities"])]
    if name in ("W4", "BPRIME"):
        return name, folder, [verified_runs(folder, name, situations, complete=False)]
    if name == "G1":
        old_train = folder.parent.parent / "G1_train"
        identities = [g1.model_identity(old_train / f"seed_{seed}" / "final") for seed in ac.config()["g1"]["seeds"]]
        return name, folder, [verified_runs(folder, f"G1_m{i}", situations, model_identity=identity,
                                             complete=False) for i, identity in enumerate(identities)]
    raise ValueError(f"학습기 평가의 models.json이 없습니다: {folder}")


def collect_judgement(baseline: Path, candidates: list[str], *, part: str = "eval",
                      require_reproduction: bool = True) -> dict:
    cfg = ac.config()
    situations = situations_for(part)
    base_record = verify_baseline(baseline) if require_reproduction else folder_record(baseline)
    m4 = verified_runs(baseline, "M4", situations)
    p0 = verified_runs(baseline, "P0", situations)
    for policy, rows in (("M4", m4), ("P0", p0)):
        problems = {m["situation"]: ref.quality_errors(m) for m in rows if ref.quality_errors(m)}
        if problems:
            raise ValueError(f"기준선 {policy}의 품질 문제가 남아 있습니다: {problems}")
    results, pending, identities = {}, {}, {}
    for spec in candidates:
        name, folder, models = candidate_models(spec, situations, part)
        if name in results or name in pending:
            raise ValueError(f"중복 후보 이름: {name}")
        if not models:
            pending[name] = "아직 평가 결과 없음"
            continue
        if folder_record(folder)["evaluation_fingerprint"] != base_record["evaluation_fingerprint"]:
            raise ValueError(f"{name} 평가 지문이 기준선과 다릅니다.")
        want = 1 if name in ("W4", "BPRIME") else cfg["judge"]["learner_models"]
        if len(models) != want or any(len(m) != len(situations) for m in models):
            pending[name] = {"counts": [len(m) for m in models], "expected": len(situations), "models": want}
            continue
        results[name] = ref.judge(models, m4, p0, cfg, situations)
        if (folder / "models.json").exists():
            identities[name] = w.read(folder / "models.json")
    teacher_fingerprints = {m["dataset_fingerprint"] for m in identities.values()}
    if len(teacher_fingerprints) > 1:
        raise ValueError("비교할 Stage 2 모델이 서로 다른 교사 자료를 사용했습니다.")
    sensitivity = ref.judge([p0], m4, None, cfg, situations)
    return {"results": results, "pending": pending, "models": identities,
            "part": part, "evaluation_fingerprint": base_record["evaluation_fingerprint"],
            "sensitivity_m4_vs_p0": sensitivity, "lock_sha256": w.file_hash(ac.CONFIG),
            "updated_unix": time.time()}


def save_judgement(out: Path, report: dict) -> bool:
    out = Path(out)
    if report["pending"]:
        # A reconnect with incomplete candidates cannot replace a complete judgement.
        _write(out / "pending.json", report)
        text = "# ARENA3 Stage 2 — 평가 진행 중\n\n" + json.dumps(report["pending"], ensure_ascii=False, indent=2) + "\n"
        (out / "pending_summary.md").write_text(text)
        return False
    _write(out / "judge.json", report)
    text = ref.render(report["results"], report["sensitivity_m4_vs_p0"])
    text = text.replace("# ARENA3 심판 결과", f"# ARENA3 Stage 2 — {'C1·C2 확인' if report['part'] == 'confirmation' else '개발 평가'}")
    text += "\n사전학습 모델(_pre)은 강화학습 성과나 최종 승자 선정에 포함하지 않습니다.\n"
    (out / "summary.md").write_text(text)
    return True


def choose_winner(results: dict) -> str | None:
    eligible = [(name, r) for name, r in results.items() if name in FINAL_NAMES and r.get("status") == "pass"]
    return min(eligible, key=lambda x: (-float(x[1]["gain_date_mean_s"]), x[0]))[0] if eligible else None


def freeze_selection(path: Path, report: dict, train_roots: dict[str, Path]) -> dict:
    """Once written, selection is read as-is, even if later development results improve."""
    path = Path(path)
    if path.exists():
        saved = w.read(path)
        if saved["lock_sha256"] != w.file_hash(ac.CONFIG):
            raise ValueError("확인 실험의 원래 잠금이 바뀌었습니다.")
        winner = saved["winner"]
        if winner:
            current = model_declaration(winner, [Path(p) for p in saved["model_dirs"]], "eval")
            if current["model_identities"] != saved["model_identities"]:
                raise ValueError("확인 전에 동결한 모델이 바뀌었습니다.")
        return saved
    if report.get("pending"):
        raise ValueError("두 최종 학습기의 개발 평가가 끝나기 전에 승자를 고를 수 없습니다.")
    missing = set(FINAL_NAMES) - set(report["results"])
    if missing:
        raise ValueError(f"최종 학습기 평가가 빠졌습니다: {sorted(missing)}")
    winner = choose_winner(report["results"])
    selected = {"winner": winner, "lock_sha256": w.file_hash(ac.CONFIG),
                "evaluation_fingerprint": report["evaluation_fingerprint"],
                "development_report_sha256": w.digest(report),
                "model_dirs": [], "model_identities": [], "selection_unix": time.time()}
    if winner:
        if winner not in train_roots:
            raise ValueError(f"승자의 학습 폴더가 없습니다: {winner}")
        dirs = [Path(train_roots[winner]) / f"seed_{seed}" / "final" for seed in ac.config()["g1"]["seeds"]]
        declaration = model_declaration(winner, dirs, "eval")
        declared = report.get("models", {}).get(winner)
        if declared is None or declared["model_identities"] != declaration["model_identities"]:
            raise ValueError("승자 평가 모델과 최종 학습 모델이 다릅니다.")
        selected.update(model_dirs=declaration["model_dirs"], model_identities=declaration["model_identities"])
    _write(path, selected)
    return selected


def confirm(baseline: Path, candidates: list[str], train_roots: dict[str, Path], out: Path,
            work: Path, workers: int, *, executor=None, worker=eval_worker) -> dict:
    out, work = Path(out), Path(work)
    selection_path = out / "selection.json"
    if selection_path.exists():
        # Do not re-open development results to reselect a winner on a resume.
        selected = freeze_selection(selection_path, {}, train_roots)
        base_record = verify_baseline(baseline, selected["evaluation_fingerprint"])
    else:
        report = collect_judgement(baseline, candidates)
        selected = freeze_selection(selection_path, report, train_roots)
        base_record = verify_baseline(baseline, selected["evaluation_fingerprint"])
    if selected["winner"] is None:
        result = {"status": "no_passing_final_learner", "selection": selected}
        _write(out / "confirmation.json", result)
        return result
    # The selection file exists before the confirmation list is accessed.
    situations = situations_for("confirmation")
    fixed_out = out / "baseline"
    fp = ref.register(fixed_out, extra_files=EXTRA_FILES)
    if ref.evaluation_fingerprint(fixed_out) != base_record["evaluation_fingerprint"]:
        raise ValueError("확인 평가의 장치가 개발 평가와 다릅니다.")
    _write(fixed_out / "reproduction.json", {"passed": True, "inherited_from": str(Path(baseline).resolve()),
                                            "evaluation_fingerprint": base_record["evaluation_fingerprint"]})
    jobs = []
    for name in ac.config()["evaluation"]["baselines"]:
        jobs += ref.jobs_for(fixed_out, work / "baseline", fp, name, situations, kind="fixed")
    fixed_status = ref.run_jobs(jobs, max(1, workers), ref.Ledger(fixed_out, "confirmation_baseline"),
                               worker=worker, executor=executor)
    _write(fixed_out / "status" / "confirmation.json", {**fixed_status, "total": len(jobs), "part": "confirmation"})
    if fixed_status["failed"]:
        return {"status": "baseline_failed", "failed": fixed_status["failed"]}
    name = selected["winner"]
    candidate_out = out / name
    cand_status = evaluate(name, [Path(p) for p in selected["model_dirs"]], candidate_out,
                           work / name, workers, "confirmation", executor=executor, worker=worker)
    if cand_status["failed"]:
        return {"status": "candidate_failed", "failed": cand_status["failed"]}
    report = collect_judgement(fixed_out, [f"{name}={candidate_out}"], part="confirmation")
    save_judgement(out, report)
    result = {"status": report["results"][name]["status"], "selection": selected, "result": report["results"][name]}
    _write(out / "confirmation.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("evaluate")
    p.add_argument("--name", required=True)
    p.add_argument("--models", type=Path, nargs="+", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--part", choices=("eval", "confirmation"), default="eval")
    for command in ("judge", "confirm"):
        p = sub.add_parser(command)
        p.add_argument("--baseline", type=Path, required=True)
        p.add_argument("--candidate", action="append", required=True)
        p.add_argument("--out", type=Path, required=True)
        if command == "confirm":
            p.add_argument("--train", action="append", required=True)
            p.add_argument("--work", type=Path, required=True)
            p.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    w.configure()
    if args.cmd in ("evaluate", "confirm") and platform.system() == "Darwin":
        parser.error("실제 SUMO 평가는 Colab에서만 실행합니다. 맥에서는 judge 명령만 사용하세요.")
    if args.cmd == "evaluate":
        result = evaluate(args.name, args.models, args.out, args.work, args.workers, args.part)
        return 1 if result["failed"] else 0
    if args.cmd == "judge":
        report = collect_judgement(args.baseline, args.candidate)
        complete = save_judgement(args.out, report)
        print(ref.render(report["results"], report["sensitivity_m4_vs_p0"]))
        if report["pending"]:
            print("아직 평가 중:", report["pending"])
        return 0 if complete else 2
    roots = {name: Path(raw) for name, raw in (s.split("=", 1) for s in args.train)}
    result = confirm(args.baseline, args.candidate, roots, args.out, args.work, args.workers)
    print(json.dumps(w.plain(result), ensure_ascii=False, indent=2))
    return 1 if result["status"] in ("baseline_failed", "candidate_failed") else 0


if __name__ == "__main__":
    raise SystemExit(main())
