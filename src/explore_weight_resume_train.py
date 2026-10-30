"""Versioned WEIGHT1_LONG continuation adapters; original files stay frozen.

This module does not approve a collision. A separately reviewed, hash-bound
approval is required before any simulation, and only the one audited warmup
event can be reclassified. The original learner and physical environment are
unchanged. Original event counts and XML remain available for every result.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
import uuid

import explore_weight_long as original_long
import explore_weight_long_train as original_train

_ORIGINAL_EVAL_WORKER = original_long._eval_worker
APPROVAL_KIND = "WEIGHT1_RESUME_COLLISION_APPROVAL_V1"
CLASSIFICATION_KIND = "WEIGHT1_RESUME_COLLISION_CLASSIFICATION_V1"
CLONE_KIND = "WEIGHT1_RESUME_MODEL_CLONE_V1"
RULE_ID = "confirmed_precontrol_20220906_1500_seed338"
SEEDS = (42, 43, 44)
MILESTONES = (42, 139, 278, 417)
EXPECTED_EVENT = {
    "time": "1584.000", "type": "junction", "lane": ":SN20220920232934C_2_2",
    "pos": "10.304930000000", "collider": "1445_t1297", "victim": "1445_t1275",
    "colliderType": "truck", "victimType": "truck", "colliderSpeed": "13.764990000000",
    "victimSpeed": "0.000000000000", "colliderFront": "2012.671257664159,4979.499189062927",
    "colliderBack": "2013.376025521929,4972.387445391992",
    "victimFront": "2016.595834131327,4983.108207250342",
    "victimBack": "2015.292929226737,4976.135422018348",
}
EXPECTED_CASE = {
    "date": "20220906", "period": "1500", "seed": 338,
    "route_sha256": "92648bbfd8ccb1a8a8323f5cb938283ecbc16e602bf350a0e2f673ed9f7ef684",
    "event": EXPECTED_EVENT,
}


class ApprovalError(ValueError):
    """The reviewed collision evidence is missing, changed, or mismatched."""


class TrainingJobFailed(RuntimeError):
    """A learning job failed; it must not be presented as a time limit."""


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp_" + uuid.uuid4().hex)
    try:
        temporary.write_text(json.dumps(data, ensure_ascii=False, sort_keys=True,
                                        indent=2, allow_nan=False) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _runtime_guard():
    if sys.platform == "darwin":
        raise RuntimeError("재개 학습·평가는 Colab에서 실행합니다. 맥에서는 시작하지 않습니다.")


def validate_approval(approval, study_fingerprint=None):
    """Validate a human/agent reviewed receipt; never infer approval from a probe."""
    if not isinstance(approval, dict) or approval.get("kind") != APPROVAL_KIND or approval.get("approved") is not True:
        raise ApprovalError("확인된 충돌 원인에 대한 별도 승인 기록이 필요합니다")
    if not approval.get("study_fingerprint"):
        raise ApprovalError("승인에 새 실험 지문이 없습니다")
    if study_fingerprint is not None and approval["study_fingerprint"] != study_fingerprint:
        raise ApprovalError("충돌 승인과 재개 실험 지문이 다릅니다")
    if approval.get("case") != EXPECTED_CASE:
        raise ApprovalError("승인 사례가 감사한 준비 구간 사건과 정확히 같지 않습니다")
    report = Path(approval.get("diagnostic_report_path", ""))
    expected = approval.get("diagnostic_report_sha256")
    if not report.is_absolute() or not report.is_file() or not isinstance(expected, str) or len(expected) != 64:
        raise ApprovalError("충돌 진단 보고서 경로·해시가 없습니다")
    if _sha(report) != expected:
        raise ApprovalError("승인에 묶인 충돌 진단 보고서가 바뀌었습니다")
    if not isinstance(_read(report), dict):
        raise ApprovalError("충돌 진단 보고서는 JSON 객체여야 합니다")
    bindings = approval.get("diagnosis_evidence_hashes", {})
    if not isinstance(bindings, dict):
        raise ApprovalError("진단 증거 해시는 경로와 SHA의 대응이어야 합니다")
    for name, digest in bindings.items():
        path = Path(name)
        if not path.is_absolute() or not path.is_file() or _sha(path) != digest:
            raise ApprovalError(f"승인에 묶인 충돌 증거가 바뀌었습니다: {name}")
    return approval


def load_approval(out, study_fingerprint=None):
    path = Path(out) / "approval.json"
    if not path.is_file():
        raise ApprovalError("충돌 원인 승인 전에는 학습·평가를 재개하지 않습니다")
    return validate_approval(_read(path), study_fingerprint)


def apply_classification(result, approval):
    """Return a metrics-only copy; costs, transitions and raw events are retained."""
    approval = validate_approval(approval)
    metrics = copy.deepcopy(result.metrics)
    if "collision_classification" in metrics:
        raise ApprovalError("같은 에피소드에 충돌 분류를 중복 적용할 수 없습니다")
    details = metrics.get("collision_details")
    if not isinstance(details, dict) or not isinstance(details.get("collisions_unexplained"), list):
        raise ApprovalError("원시 충돌 상세 기록이 없어 검증할 수 없습니다")
    events = details["collisions_unexplained"]
    if metrics.get("collisions_unexplained") != len(events):
        raise ApprovalError("원시 충돌 상세와 총 미확인 건수가 다릅니다")
    known_before = details.get("collisions_known_map_defect", {})
    if (not isinstance(known_before, dict)
            or any(not isinstance(n, int) or n < 0 for n in known_before.values())
            or details.get("collisions_total") != len(events) + sum(known_before.values())):
        raise ApprovalError("원시 충돌 총 건수와 알려진 원인·미확인 집계가 다릅니다")
    metrics["raw_collision_details"] = copy.deepcopy(details)
    metrics["raw_collisions_unexplained"] = len(events)
    case = approval["case"]
    same_case = all(metrics.get(k) == case[k] for k in ("date", "period", "seed", "route_sha256"))
    if "eval_seed" in metrics:
        same_case = same_case and metrics["eval_seed"] == case["seed"]
    handover = float(metrics.get("handover", math.nan))
    if not math.isfinite(handover):
        raise ApprovalError("실제 RL 제어 시작 시각이 없습니다")
    same_case = same_case and handover == 1803.0
    matched, remaining = [], []
    for event in events:
        if same_case and event == case["event"] and float(event["time"]) < handover:
            matched.append(copy.deepcopy(event))
        else:
            remaining.append(copy.deepcopy(event))
    # The approved raw episode contained exactly one such event. A duplicated
    # or otherwise changed collision list is a fresh incident, not this receipt.
    if len(matched) > 1:
        matched, remaining = [], copy.deepcopy(events)
    if matched:
        known = details.setdefault("collisions_known_map_defect", {})
        if not isinstance(known, dict) or RULE_ID in known:
            raise ApprovalError("충돌 알려진 원인 집계가 이미 바뀌어 있습니다")
        known[RULE_ID] = len(matched)
        details["collisions_unexplained"] = remaining
        metrics["collisions_unexplained"] = len(remaining)
        nested = metrics.get("collisions")
        if isinstance(nested, dict) and "unexplained" in nested:
            if nested["unexplained"] != len(events):
                raise ApprovalError("중첩 충돌 건수가 원시 기록과 다릅니다")
            nested["unexplained"] = len(remaining)
    metrics["collision_classification"] = {
        "kind": CLASSIFICATION_KIND, "study_fingerprint": approval["study_fingerprint"],
        "diagnostic_report_sha256": approval["diagnostic_report_sha256"],
        "rule_id": RULE_ID, "confirmed_events": matched,
        "raw_unexplained_count": len(events), "remaining_unexplained_count": len(remaining),
        "total_collision_count_unchanged": details.get("collisions_total"),
    }
    copied = copy.copy(result)
    copied.metrics = metrics
    return copied


def _lineage_for_resume(lineage, new_fingerprint):
    result = copy.deepcopy(lineage)
    if not isinstance(result, dict) or not result.get("study_fingerprint"):
        raise ValueError("원본 모델의 실험 계보가 없습니다")
    result["study_fingerprint"] = new_fingerprint
    return result


def clone_models(study_root, out, new_fingerprint, parent_fingerprint):
    """Copy committed generations and official stages; never hard-link sources."""
    from explore_weight_train import validate_checkpoint

    started = time.monotonic()
    source, destination = Path(study_root).resolve(), Path(out).resolve()
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError("재개 결과는 원본 실험 폴더와 겹칠 수 없습니다")
    if not new_fingerprint or not parent_fingerprint:
        raise ValueError("새 실험과 부모 지문이 필요합니다")
    experiment = _read(source / "experiment.json")
    if experiment.get("parent_fingerprint") != parent_fingerprint:
        raise ValueError("모델 복사의 부모 지문이 다릅니다")
    audit_path = destination / "clone_audit.json"
    if audit_path.exists():
        previous = _read(audit_path)
        if (previous.get("kind"), previous.get("study_fingerprint"), previous.get("parent_fingerprint"),
                previous.get("source_experiment_sha256"), previous.get("source_root")) != (
                CLONE_KIND, new_fingerprint, parent_fingerprint, _sha(source / "experiment.json"), str(source)):
            raise ValueError("기존 복제 계보가 다른 실험을 가리킵니다")
        for seed in SEEDS:
            original_identity = validate_checkpoint(source / "models" / f"seed_{seed}", parent_fingerprint)
            target = destination / "models" / f"seed_{seed}"
            identity = validate_checkpoint(target, parent_fingerprint)
            initial = previous["models"][str(seed)]
            if initial["episodes"] != original_identity["episodes"] or initial["files"] != original_identity["files"]:
                raise ValueError("복제 감사 기록이 실제 원본 체크포인트와 다릅니다")
            if identity["seed"] != seed or identity["episodes"] < initial["episodes"]:
                raise ValueError("복제한 모델이 최초 재개 상태보다 이전입니다")
            if _read(target / "long_lineage.json")["study_fingerprint"] != new_fingerprint:
                raise ValueError("복제 모델의 새 실험 계보가 다릅니다")
            if identity["episodes"] == initial["episodes"] and identity["files"] != initial["files"]:
                raise ValueError("최초 재개 회차의 모델 내용이 원본과 다릅니다")
            for episode, expected in initial["stages"].items():
                stage = validate_checkpoint(target / "stages" / f"episode_{int(episode):06d}", parent_fingerprint)
                if stage["files"] != expected["files"]:
                    raise ValueError("보존한 공식 중간 모델이 바뀌었습니다")
        return {**previous, "reused": True, "seconds": time.monotonic()-started}
    destination.mkdir(parents=True, exist_ok=True)
    audit = {"kind": CLONE_KIND, "study_fingerprint": new_fingerprint,
             "parent_fingerprint": parent_fingerprint, "source_root": str(source),
             "source_study_fingerprint": experiment["fingerprint"],
             "source_experiment_sha256": _sha(source / "experiment.json"),
             "source_budget_sha256": _sha(source / "budget.json"),
             "originalsha": {}, "models": {}, "created_unix": time.time()}
    for seed in SEEDS:
        origin = source / "models" / f"seed_{seed}"
        target = destination / "models" / f"seed_{seed}"
        identity = validate_checkpoint(origin, parent_fingerprint)
        if identity["seed"] != seed:
            raise ValueError("원본 모델 난수가 다릅니다")
        pointer = _read(origin / "latest.json")
        origin_generation = Path(identity["path"])
        generation = Path(pointer["checkpoint"])
        lineage = _read(origin / "long_lineage.json")
        if lineage.get("study_fingerprint") != experiment["fingerprint"] or lineage.get("parent_fingerprint") != parent_fingerprint:
            raise ValueError("원본 모델과 실험 계보가 다릅니다")
        new_lineage = _lineage_for_resume(lineage, new_fingerprint)
        snapshots = {}
        relative_paths = [Path("latest.json"), Path("long_lineage.json")]
        relative_paths.extend(generation / name for name in ("model.pt", "replay.pkl", "manifest.json"))
        if (origin / "run.json").is_file():
            relative_paths.append(Path("run.json"))
        for episode in MILESTONES:
            stage = origin / "stages" / f"episode_{episode:06d}"
            if episode > identity["episodes"]:
                continue
            if not stage.is_dir():
                raise ValueError("원본 모델의 공식 중간 저장이 없습니다")
            stage_identity = validate_checkpoint(stage, parent_fingerprint)
            if stage_identity["seed"] != seed or stage_identity["episodes"] != episode:
                raise ValueError("원본 중간 모델의 난수·회차가 다릅니다")
            snapshots[str(episode)] = {"files": stage_identity["files"], "manifest_sha256": _sha(stage / "manifest.json")}
            relative_paths.extend(Path("stages") / stage.name / name for name in ("model.pt", "replay.pkl", "manifest.json"))
        original_hashes = {str(p): _sha(origin / p) for p in relative_paths}
        audit["originalsha"].update({f"models/seed_{seed}/{name}": digest for name, digest in original_hashes.items()})
        if target.exists():
            # Recovery from a copy interrupted before the outer audit committed.
            restored = validate_checkpoint(target, parent_fingerprint)
            if restored["episodes"] != identity["episodes"] or restored["files"] != identity["files"] or _read(target / "long_lineage.json") != new_lineage:
                raise ValueError("미완료 복제 폴더가 원본 모델과 다릅니다")
            for episode, expected in snapshots.items():
                if validate_checkpoint(target / "stages" / f"episode_{int(episode):06d}", parent_fingerprint)["files"] != expected["files"]:
                    raise ValueError("미완료 복제의 중간 모델이 다릅니다")
        else:
            temporary = target.with_name(".clone_" + target.name + "_" + uuid.uuid4().hex)
            temporary.mkdir(parents=True)
            try:
                for relative in relative_paths:
                    copied = temporary / relative
                    copied.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(origin / relative, copied)
                    if _sha(copied) != original_hashes[str(relative)]:
                        raise ValueError("모델 복사 중 바이트 해시가 달라졌습니다")
                _write(temporary / "long_lineage.json", new_lineage)
                if validate_checkpoint(temporary, parent_fingerprint)["files"] != identity["files"]:
                    raise ValueError("새 복제 모델의 파일 해시가 다릅니다")
                os.replace(temporary, target)
            finally:
                if temporary.exists():
                    shutil.rmtree(temporary)
        if _sha(origin / "latest.json") != original_hashes["latest.json"] or _sha(origin / "long_lineage.json") != original_hashes["long_lineage.json"]:
            raise ValueError("복사 중 원본 모델이 바뀌었습니다")
        audit["models"][str(seed)] = {"episodes": identity["episodes"], "control_steps": identity["control_steps"],
                                      "checkpoint": str(generation), "files": identity["files"], "stages": snapshots}
        source_status = source / "training_status" / f"seed_{seed}.json"
        if source_status.is_file():
            raw_status = source_status.read_bytes()
            audit["originalsha"][f"training_status/seed_{seed}.json"] = hashlib.sha256(raw_status).hexdigest()
            status_out = destination / "training_status"
            status_out.mkdir(parents=True, exist_ok=True)
            preserved = status_out / f"source_seed_{seed}.json"
            if preserved.exists() and preserved.read_bytes() != raw_status:
                raise ValueError("보존한 원래 작업 상태가 바뀌었습니다")
            preserved.write_bytes(raw_status)
            status = json.loads(raw_status)
            status["study_fingerprint"] = new_fingerprint
            status["status"] = "cloned_for_resume"
            status["completed_episodes"] = identity["episodes"]
            status["last_completed_episodes"] = identity["episodes"]
            status["control_steps"] = identity["control_steps"]
            status["latest_checkpoint"] = str(target / generation)
            for episode, item in status.get("milestones", {}).items():
                item["stage"] = str(target / "stages" / f"episode_{int(episode):06d}")
            _write(status_out / f"seed_{seed}.json", status)
    audit["seconds"] = time.monotonic()-started
    _write(audit_path, audit)
    return {**audit, "reused": False}


def training_worker(job):
    """Use the unchanged learner; return only successful or genuine time stops."""
    _runtime_guard()
    approval = load_approval(job["out"], job["study_fingerprint"])
    original_factory = original_train._episode_runner
    runner = original_factory()

    def classified_runner(*args, **kwargs):
        current = load_approval(job["out"], job["study_fingerprint"])
        if current != approval:
            raise ApprovalError("진행 중 충돌 승인이 바뀌었습니다")
        return apply_classification(runner(*args, **kwargs), current)

    original_train._episode_runner = lambda: classified_runner
    try:
        result = original_train.run_training_job(job)
    finally:
        original_train._episode_runner = original_factory
    status = result.get("status")
    if status == "failed":
        raise TrainingJobFailed(f"seed{job['seed']} 실제 학습 오류: {result.get('error', '원인 미기록')}")
    if status not in ("complete", "time_limit"):
        raise TrainingJobFailed(f"seed{job['seed']} 알 수 없는 학습 종료 상태: {status}")
    if status == "complete" and int(result.get("completed_episodes", -1)) < max(job["episodes"]):
        raise TrainingJobFailed(f"seed{job['seed']} 완료 상태지만 목표 회차에 도달하지 않았습니다")
    result["resume_approval_sha256"] = _sha(Path(job["out"]) / "approval.json")
    return result


def eval_worker(job):
    """Same approved classification for every policy, without recursive patches."""
    _runtime_guard()
    approval = load_approval(Path(job["target"]).parents[2], job["study_fingerprint"])
    import explore_weight_episode as episode
    runner = episode.run_episode

    def classified_runner(*args, **kwargs):
        return apply_classification(runner(*args, **kwargs), approval)

    episode.run_episode = classified_runner
    try:
        return _ORIGINAL_EVAL_WORKER(job)
    finally:
        episode.run_episode = runner
