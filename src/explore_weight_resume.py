"""Versioned Colab diagnosis and continuation; original WEIGHT1 files stay frozen."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import json
from pathlib import Path
import signal
import sys
import time
import uuid

import explore_weight as w
import explore_weight_long as old

ROOT = Path(__file__).resolve().parents[1]
APPROVAL_PENDING = 78
CODE = ("src/explore_weight_resume.py", "src/explore_weight_resume_train.py",
        "src/explore_weight_collision_probe.py")


def register(source, study, out, wall_hours=24, compute_hours=96, *, check_runtime=True):
    """Bind the unchanged physics/learner and the immutable stopped study."""
    from explore_weight_long_train import validate_parent
    w.configure()
    parent = validate_parent(source, check_runtime=check_runtime)
    original = w.read(study / "experiment.json")
    if original["fingerprint"] != w.digest({k:v for k,v in original.items() if k != "fingerprint"}):
        raise ValueError("원본 긴 학습의 실험 지문이 다릅니다")
    if parent["fingerprint"] != original["parent_fingerprint"]:
        raise ValueError("원본 WEIGHT1과 긴 학습의 부모가 다릅니다")
    for name, sha in original["code"].items():
        if w.file_hash(ROOT / name) != sha:
            raise ValueError(f"보존해야 할 원본 코드가 달라졌습니다: {name}")
    budget = w.read(study / "budget.json")
    if budget["pending"]:
        raise ValueError("원본 실행의 미정산 작업부터 확인해야 합니다")
    if not (0 < wall_hours <= 24 and compute_hours == original["compute_hours"] == 96):
        raise ValueError("실행당24시간·누적96작업시간을 유지해야 합니다")
    snapshots = {}
    for seed in original["config"]["train_seeds"]:
        folder = study / "models" / f"seed_{seed}"
        pointer = w.read(folder / "latest.json")
        checkpoint = (folder / pointer["checkpoint"]).resolve()
        if checkpoint.parent != (folder / "checkpoints").resolve():
            raise ValueError("원본 체크포인트 경로 오류")
        if w.file_hash(checkpoint / "manifest.json") != pointer["manifest_sha256"]:
            raise ValueError("원본 체크포인트 manifest 지문 오류")
        snapshots[str(seed)] = {"pointer":pointer, "manifest":w.read(checkpoint / "manifest.json")}
    frozen = {"kind":"WEIGHT1_RESUME_V1", "config":original["config"], "profile":original["profile"],
              "parent_fingerprint":parent["fingerprint"], "original_study_fingerprint":original["fingerprint"],
              "original_experiment_sha256":w.file_hash(study / "experiment.json"),
              "original_budget_sha256":w.file_hash(study / "budget.json"), "original_checkpoints":snapshots,
              "code":{name:w.file_hash(ROOT / name) for name in CODE},
              "wall_hours":wall_hours, "compute_hours":compute_hours, "prior_compute_s":budget["used_s"],
              "protocol":"same physics/learner and final endpoint; only independently diagnosed exact precontrol case may be classified; all policy comparisons rerun"}
    record = {"fingerprint":w.digest(frozen), **frozen}
    remaining_episodes = sum(max(0, original["profile"]["primary_episodes"]-v["manifest"]["episodes"])
                             for v in snapshots.values())
    cfg = original["config"]
    development_runs = len(cfg["train_seeds"])*len(old.groups("development"))*(
        (len(original["profile"]["milestones"])-1)*len(cfg["development"]["interim_seeds"])
        + len(cfg["development"]["primary_seeds"]))
    reference_runs = 2*len(old.groups("development"))*len(cfg["development"]["primary_seeds"])
    final_runs = (len(cfg["train_seeds"])+2)*len(old.groups("confirmation"))*len(cfg["confirmation"]["eval_seeds"])
    new_seconds = (remaining_episodes*cfg["budget"]["episode_seconds_estimate"]
                   + (development_runs+reference_runs+final_runs)*cfg["budget"]["evaluation_seconds_estimate"] + 2400)
    forecast = {"remaining_training_episodes":remaining_episodes,"development_runs":development_runs,
                "new_reference_runs":reference_runs,"confirmation_runs_if_gate_passes":final_runs,
                "nominal_new_s":new_seconds,"conservative_total_s":budget["used_s"]+new_seconds*1.25,
                "includes_prior_s":budget["used_s"],"note":"진단·복사600+1800초 여유 포함, 실측 기반 예상이며 보장 아님"}
    if forecast["conservative_total_s"] > compute_hours*3600:
        raise old.StudyStopped("남은 전체 학습·평가 예상 비용이 누적96시간을 넘어 확대하지 않습니다")
    target = out / "experiment.json"
    if target.exists() and w.read(target) != record:
        raise ValueError("재개 실험의 코드·부모·저장점이 달라졌습니다")
    out.mkdir(parents=True, exist_ok=True)
    w.write(target, record)
    w.write(out/"forecast.json",forecast)
    return record


@contextmanager
def charged(ledger, name, cap):
    token = ledger.begin(name, min(cap, ledger.remaining))
    started, status = time.monotonic(), "failed"
    try:
        yield
        status = "complete"
    finally:
        ledger.finish(token, time.monotonic()-started, status)


def diagnose(source, study, out, work, record, ledger, deadline):
    from explore_weight_collision_probe import run_probe
    cap = min(600.0, ledger.remaining, deadline-time.time()-10)
    if cap < 60:
        raise old.StudyStopped("진단을 실행할 계산 시간이 부족합니다")
    output, scratch = out / "diagnostic", work / "diagnostic"
    previous = out / "diagnostic_result.json"
    if previous.exists():
        saved = w.read(previous)
        if saved.get("status") in ("reproduced", "not_reproduced"):
            output = Path(saved["report_path"]).parent
        else:
            attempt = "diagnostic_retry_" + uuid.uuid4().hex
            output, scratch = out / attempt, work / attempt
    elif (output / "report.json").exists() and w.read(output / "report.json").get("status") not in ("reproduced", "not_reproduced"):
        attempt = "diagnostic_retry_" + uuid.uuid4().hex
        output, scratch = out / attempt, work / attempt
    with charged(ledger, "diagnostic/warmup_collision", cap):
        result = run_probe(source, study, output, scratch, cap_s=cap)
    w.write(out / "diagnostic_result.json", result)
    if result["status"] not in ("reproduced", "not_reproduced"):
        raise RuntimeError(f"충돌 진단을 완료하지 못했습니다: {result}")
    print("진단 자료 저장 완료. 재현 일치만으로 충돌 예외를 자동 승인하지 않습니다.", flush=True)
    return result


def require_approval(out, fingerprint):
    from explore_weight_resume_train import load_approval
    if not (out / "approval.json").exists():
        return None
    approval = load_approval(out, fingerprint)
    receipt = {"study_fingerprint":fingerprint, "approval_sha256":w.file_hash(out / "approval.json"),
               "comparison":"P0 and M4 development references and every RL observation/final evaluation rerun",
               "final_episodes":w.read(out / "experiment.json")["profile"]["primary_episodes"]}
    activation = out / "protocol_activation.json"
    if activation.exists() and w.read(activation) != receipt:
        raise ValueError("이미 시작한 재개의 판정 승인이 바뀌었습니다")
    w.write(activation, receipt)
    return approval


@contextmanager
def versioned_workers():
    """Only replace pool callables; immutable original source bytes stay intact."""
    from explore_weight_resume_train import training_worker, eval_worker
    train, evaluate = old._training_worker, old._eval_worker
    old._training_worker, old._eval_worker = training_worker, eval_worker
    try:
        yield
    finally:
        old._training_worker, old._eval_worker = train, evaluate


def evaluate(source, out, work, record, workers, ledger, deadline):
    cfg = record["config"]
    primary = record["profile"]["primary_episodes"]
    # A new collision adjudication is a new evaluation version. Do not reuse
    # the old, selected P0/M4 development caches under the changed rule.
    references = old.evaluations(out, work, record, workers, ledger, deadline,
                                "development_references", [("P0",None,None),("M4",4,None)],
                                old.groups("development"), cfg["development"]["primary_seeds"])
    for episode in record["profile"]["milestones"]:
        seeds = cfg["development"]["primary_seeds"] if episode == primary else cfg["development"]["interim_seeds"]
        policies = [(f"RL{s}",None,out/"models"/f"seed_{s}"/"stages"/f"episode_{episode:06d}") for s in cfg["train_seeds"]]
        rows = old.evaluations(out, work, record, workers, ledger, deadline, f"dev_ep{episode}", policies,
                               old.groups("development"), seeds)
        refs = {name:[r for r in rs if r["seed"] in seeds] for name,rs in references.items()}
        result = old.development_reading({s:rows[f"RL{s}"] for s in cfg["train_seeds"]},refs)
        result.update(episodes=episode,primary=episode==primary)
        w.write(out/f"development_{episode}.json",result)
    main = w.read(out/f"development_{primary}.json")
    if not main["passed"]:
        w.write(out/"outcome.json",{"status":"primary_development_not_passed","primary_episodes":primary})
        return
    from explore_weight_train import checkpoint_identity
    policies = [("P0",None,None),("M4",4,None)]+[(f"RL{s}",None,out/"models"/f"seed_{s}"/"stages"/f"episode_{primary:06d}") for s in cfg["train_seeds"]]
    identities = {name:checkpoint_identity(model,record["parent_fingerprint"]) for name,_,model in policies if model}
    freeze=out/"confirmation_models.json"
    if freeze.exists() and w.read(freeze)!=identities:
        raise ValueError("확인 평가 모델이 바뀌었습니다")
    w.write(freeze,identities)
    rows = old.evaluations(out,work,record,workers,ledger,deadline,"confirmation",policies,
                           old.groups("confirmation"),cfg["confirmation"]["eval_seeds"])
    result = old.confirmation_reading({s:rows[f"RL{s}"] for s in cfg["train_seeds"]},
                                     {"P0":rows["P0"],"M4":rows["M4"]},cfg)
    result["interpretation"] = "새 충돌 판정 버전의 추가 개발 확인 평가; 기존 WEIGHT1 성공이나 실제 도로 일반화 아님"
    w.write(out/"confirmation_summary.json",result)
    w.write(out/"outcome.json",{"status":"confirmation_passed" if result["passed"] else "confirmation_not_passed", "primary_episodes":primary})


def summary(out):
    if not (out / "experiment.json").exists():
        return {}
    result=old.summary(out)
    for name in ("diagnostic_result","protocol_activation"):
        if (out/f"{name}.json").exists():
            result[name]=w.read(out/f"{name}.json")
    approved=(out/"protocol_activation.json").exists()
    result["resume_state"]="approved_versioned_continuation" if approved else "diagnostic_review_pending_no_training"
    w.write(out/"summary.json",result)
    base=(out/"summary.md").read_text()
    note = "진단 근거를 검토한 새 판정 버전으로 재개합니다." if approved else "진단 검토 대기 중입니다. 재현 일치만으로 학습을 자동 재개하지 않습니다."
    (out/"summary.md").write_text("# WEIGHT1 진단·재개\n\n"+note+"\n\n"+base)
    return result


def safe_paths(source,study,out,work):
    paths=[source.resolve(),study.resolve(),out.resolve(),work.resolve()]
    for i,a in enumerate(paths):
        for b in paths[i+1:]:
            if a==b or a in b.parents or b in a.parents:
                raise ValueError("원본·긴 학습·새 결과·임시 작업 폴더는 서로 겹칠 수 없습니다")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("command",choices=("diagnose","run","summary"))
    for name in ("source","study","out","work"):
        p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--workers",default="auto")
    p.add_argument("--wall-hours",type=float,default=24)
    p.add_argument("--compute-hours",type=float,default=96)
    args=p.parse_args()
    source,study,out,work=[getattr(args,n).resolve() for n in ("source","study","out","work")]
    safe_paths(source,study,out,work)
    if args.command=="summary":
        summary(out);return 0
    if sys.platform=="darwin":
        raise SystemExit("실제 진단·학습은 Colab에서만 실행합니다. 맥에서는 시작하지 않습니다")
    out.mkdir(parents=True,exist_ok=True)
    with (out/".lock").open("a") as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise SystemExit("같은 진단·재개가 이미 실행 중입니다")
        started=time.monotonic()
        record=register(source,study,out,args.wall_hours,args.compute_hours)
        ledger=old.Ledger(out,record)
        seconds=time.monotonic()-started
        token=ledger.begin("setup/validation",max(.001,seconds));ledger.finish(token,seconds,"complete")
        deadline=time.time()+args.wall_hours*3600-seconds
        def terminate(signum, frame):raise old.StudyStopped("실행 중단 요청을 받아 저장된 상태를 보존합니다")
        signal.signal(signal.SIGTERM,terminate)
        try:
            if args.command=="diagnose":
                diagnose(source,study,out,work,record,ledger,deadline)
                return 0
            if require_approval(out,record["fingerprint"]) is None:
                print("충돌 진단 검토 대기: 학습은 시작하지 않았습니다.",flush=True)
                return APPROVAL_PENDING
            if old.available_cpus()<2:raise ValueError("재개에는 CPU2개 이상이 필요합니다")
            workers=min(3,old.available_cpus()) if args.workers=="auto" else min(3,old.available_cpus(),int(args.workers))
            if workers<1:raise ValueError("작업 수 오류")
            from explore_weight_resume_train import clone_models
            with charged(ledger,"setup/clone_models",min(1800,deadline-time.time()-10)):
                clone_models(study,out,record["fingerprint"],record["parent_fingerprint"])
            (out/"stopped.json").unlink(missing_ok=True)
            with versioned_workers():
                old.train_all(source,out,work,record,workers,ledger,deadline)
                evaluate(source,out,work,record,workers,ledger,deadline)
            return 0
        except (old.StudyStopped,KeyboardInterrupt) as e:
            w.write(out/"stopped.json",{"reason":"time_budget_or_interrupt","message":str(e),"models_preserved":True})
            return 0
        except BaseException as e:
            w.write(out/"stopped.json",{"reason":"execution_error","message":f"{type(e).__name__}: {e}","models_preserved":True})
            raise
        finally:
            summary(out)


if __name__=="__main__":
    raise SystemExit(main())
