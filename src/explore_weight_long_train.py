"""WEIGHT1-L: isolated Colab training-length jobs, with unchanged WEIGHT1 TD.

The parent experiment is read-only. Seed 42 starts from its exact episode-42
checkpoint; other seeds start fresh. Model lineage keeps the parent fingerprint
because learner, observation, reward and exploration schedules are unchanged.
The separate study fingerprint belongs to long_lineage/training_status records.
No policy-quality gate stops this learning-curve experiment early.
"""
from __future__ import annotations

import fcntl
import gzip
import hashlib
import importlib.metadata
import json
import math
import os
import shutil
import sys
import time
import uuid
from pathlib import Path

OFFICIAL_MILESTONES = (42, 139, 278, 417)


class TrainingTimeLimit(RuntimeError):
    """A job stopped before its next episode; its latest checkpoint is valid."""


def _read(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _plain(value):
    import numpy as np
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "detach") and hasattr(value, "tolist"):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    return value


def _write(path: Path, data: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp_" + uuid.uuid4().hex)
    temporary.write_text(json.dumps(_plain(data), ensure_ascii=False, sort_keys=True,
                                    indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _sha(path: Path) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(_plain(value), sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def _current_parent_record(check_runtime: bool = True) -> dict:
    """Equivalent to the original register calculation, without calling/writing it."""
    import explore_weight as parent
    from focused_data import ENV_FILES, environment_fingerprint

    parent.configure()
    cfg = parent.config()
    parent.validate_implementation(cfg)
    environment = environment_fingerprint()
    if environment != cfg["environment"]["fingerprint"]:
        raise ValueError("WEIGHT1-L parent environment fingerprint differs")
    if parent.file_hash(parent.PROJECT / "src/explore_pred.py")[:12] != cfg["environment"]["p0_code"]:
        raise ValueError("WEIGHT1-L parent P0 source differs")
    current = {"config": cfg, "environment_fingerprint": environment,
               "files": {s: parent.file_hash(parent.PROJECT / s)
                         for s in sorted(set(parent.SOURCES) | set(ENV_FILES))}}
    if check_runtime:
        current["packages"] = {name: importlib.metadata.version(name) for name in
                               ("numpy", "torch", "stable-baselines3", "sumo-rl", "eclipse-sumo")}
    return current


def verify_parent(source_root: Path, parent_fingerprint: str, *, check_runtime: bool = True) -> dict:
    """Verify code, data/environment and package identity without writing parent files."""
    source_root = Path(source_root)
    path = source_root / "experiment.json"
    source = _read(path)
    if source.get("fingerprint") != parent_fingerprint:
        raise ValueError("WEIGHT1-L requested parent fingerprint differs from experiment.json")
    frozen = {k: v for k, v in source.items() if k != "fingerprint"}
    if _digest(frozen) != parent_fingerprint:
        raise ValueError("WEIGHT1-L parent experiment record digest differs")
    current = _current_parent_record(check_runtime=check_runtime)
    if not check_runtime:
        # Only package versions are exempted for read-only preparation on a
        # different host. Source/config/environment hashes remain mandatory.
        current = {**current, "packages": frozen.get("packages")}
    if frozen != current or _digest(current) != parent_fingerprint:
        raise ValueError("WEIGHT1-L original source/config/environment/package versions do not match parent")
    return {"parent_fingerprint": parent_fingerprint, "experiment_sha256": _sha(path),
            "source_root": str(source_root.resolve())}


def validate_parent(source_root: Path, check_runtime: bool = True) -> dict:
    """Read-only parent identity used by the long-study coordinator.

    ``checkpoint_sha256`` deliberately identifies manifest.json, whose hashes
    cover model.pt and replay.pkl. It is not the model-file hash.
    """
    source_root = Path(source_root).resolve()
    saved = _read(source_root / "experiment.json")
    fingerprint = saved.get("fingerprint")
    if not isinstance(fingerprint, str) or not fingerprint:
        raise ValueError("Parent experiment fingerprint is missing")
    verify_parent(source_root, fingerprint, check_runtime=check_runtime)
    audit = _source_checkpoint_audit(source_root / "models" / "seed_42", fingerprint)
    return {"fingerprint": fingerprint, "checkpoint_sha256": audit["manifest_sha256"]}


def _source_checkpoint_audit(source_model: Path, parent_fingerprint: str) -> dict:
    import torch
    from explore_weight_train import validate_checkpoint

    identity = validate_checkpoint(source_model, parent_fingerprint)
    if (identity["seed"], identity["episodes"], identity["control_steps"],
            identity["updates"], identity["target_syncs"]) != (42, 42, 30_240, 7_310, 6):
        raise ValueError("WEIGHT1-L seed 42 source must be the exact completed episode-42 parent")
    path = Path(identity["path"])
    payload = torch.load(path / "model.pt", map_location="cpu", weights_only=False)
    for key in ("seed", "episodes", "control_steps", "updates", "target_syncs", "fingerprint"):
        if payload[key] != identity[key]:
            raise ValueError(f"WEIGHT1-L source model/manifest mismatch: {key}")
    state = payload["trainer_state"]
    # These bytes are copied, not synthesized or reseeded. The digest records
    # all stochastic streams required by the original exact-resume learner.
    rng_state = {name: payload[name] for name in ("sample_rng", "torch_rng", "numpy_rng", "python_rng")}
    rng_state.update({name: state[name] for name in ("scenario_rng", "exploration_rng")})
    rng_sha = _digest(rng_state)
    return {"model_dir": str(Path(source_model).resolve()), "checkpoint": str(path.resolve()),
            "identity": identity, "rng_sha256": rng_sha,
            "latest_sha256": _sha(Path(source_model) / "latest.json"),
            "manifest_sha256": _sha(path / "manifest.json")}


def clone_parent_checkpoint(source_root: Path, destination: Path, *,
                            parent_fingerprint: str, study_fingerprint: str) -> dict:
    """Copy the parent's committed generation; never hard-link into the parent."""
    from explore_weight_train import validate_checkpoint

    source_model = Path(source_root).resolve() / "models" / "seed_42"
    destination = Path(destination).resolve()
    if destination == source_model or source_model in destination.parents or destination in source_model.parents:
        raise ValueError("WEIGHT1-L model destination overlaps its read-only parent")
    audit = _source_checkpoint_audit(source_model, parent_fingerprint)
    lineage = {"kind": "WEIGHT1-learning-length-v1", "seed": 42,
               "parent_fingerprint": parent_fingerprint, "study_fingerprint": study_fingerprint,
               "source": audit}
    if destination.exists():
        existing = destination / "long_lineage.json"
        if not existing.exists() or _read(existing) != _plain(lineage):
            raise ValueError("Existing seed 42 destination has a different or missing long-study lineage")
        checkpoint = validate_checkpoint(destination, parent_fingerprint)
        if checkpoint["seed"] != 42 or checkpoint["episodes"] < 42:
            raise ValueError("Existing seed 42 destination is not a valid continuation")
        return lineage
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(".clone_" + destination.name + "_" + uuid.uuid4().hex)
    generation = Path(audit["checkpoint"]).name
    checkpoint = temporary / "checkpoints" / generation
    try:
        checkpoint.mkdir(parents=True)
        for name in ("model.pt", "replay.pkl", "manifest.json"):
            shutil.copy2(Path(audit["checkpoint"]) / name, checkpoint / name)
        _write(temporary / "latest.json", {"checkpoint": str(checkpoint.relative_to(temporary)),
                                           "manifest_sha256": _sha(checkpoint / "manifest.json")})
        _write(temporary / "long_lineage.json", lineage)
        identity = validate_checkpoint(temporary, parent_fingerprint)
        if identity["files"] != audit["identity"]["files"]:
            raise ValueError("Copied parent model/replay hashes differ")
        # A separate local copy makes the initial stage immutable even if the
        # continuation learner later cleans its rotating generations.
        baseline = temporary / "stages" / "episode_000042"
        shutil.copytree(checkpoint, baseline)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    if _source_checkpoint_audit(source_model, parent_fingerprint) != audit:
        raise ValueError("Parent checkpoint changed while it was being copied")
    return lineage


def _runtime_guard() -> None:
    if sys.platform == "darwin":
        raise RuntimeError("WEIGHT1-L 실제 학습은 Mac에서 실행하지 않습니다. Colab에서 실행하세요.")


def _training_scenarios():
    from explore_weight import training_scenarios
    return training_scenarios()


def _episode_runner():
    from explore_weight_episode import run_episode
    return run_episode


def _job_parameters(job: dict) -> dict:
    required = ("seed", "source_root", "out", "work", "episodes", "study_fingerprint",
                "global_deadline_unix", "job_cap_s", "parent_fingerprint")
    if any(name not in job for name in required):
        raise ValueError("WEIGHT1-L training job has missing parameters")
    result = dict(job)
    result["seed"] = int(result["seed"])
    if result["seed"] not in (42, 43, 44):
        raise ValueError("WEIGHT1-L training seeds are frozen at 42,43,44")
    milestones = [int(n) for n in result["episodes"]]
    if (not milestones or milestones != sorted(set(milestones))
            or any(not 42 <= n <= 417 for n in milestones)):
        raise ValueError("WEIGHT1-L chunk targets must be increasing episode counts in 42..417")
    result["episodes"] = milestones
    for key in ("global_deadline_unix", "job_cap_s"):
        result[key] = float(result[key])
        if not math.isfinite(result[key]) or result[key] <= 0:
            raise ValueError("WEIGHT1-L deadline and job cap must be finite and positive")
    for key in ("source_root", "out", "work"):
        result[key] = Path(result[key]).resolve()
    source, out, work = (result[k] for k in ("source_root", "out", "work"))
    if out == source or source in out.parents or out in source.parents:
        raise ValueError("WEIGHT1-L output must be separate from its read-only parent")
    if work == source or source in work.parents or work == out or out in work.parents:
        raise ValueError("WEIGHT1-L scratch must be separate from parent and durable output")
    if " " in str(work):
        raise ValueError("WEIGHT1-L SUMO scratch paths must not contain spaces")
    if not result["study_fingerprint"] or not result["parent_fingerprint"]:
        raise ValueError("WEIGHT1-L study and parent fingerprints are required")
    return result


def _copy_evidence(result, work: Path, destination: Path, *, study_fingerprint: str,
                   check_time) -> str:
    destination.mkdir(parents=True, exist_ok=True)
    attempt = "attempt_" + uuid.uuid4().hex
    temporary = destination / (".tmp_" + attempt)
    final = destination / attempt
    temporary.mkdir()
    try:
        _write(temporary / "metrics.json", result.metrics)
        with gzip.open(temporary / "decisions.jsonl.gz", "wt", encoding="utf-8") as stream:
            for decision in result.decisions:
                stream.write(json.dumps(_plain(decision), ensure_ascii=False, allow_nan=False) + "\n")
        originals = {}
        for name in ("tripinfo.xml", "collisions.xml", "cost_ledger.json"):
            check_time()
            originals[name] = _sha(work / name)
            with (work / name).open("rb") as source, gzip.open(temporary / (name + ".gz"), "wb") as target:
                shutil.copyfileobj(source, target)
        _write(temporary / "evidence.json", {"study_fingerprint": study_fingerprint,
                                            "source_sha256": originals,
                                            "files": {p.name: _sha(p) for p in temporary.iterdir() if p.is_file()}})
        os.replace(temporary, final)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return str(final)


def run_training_job(job: dict) -> dict:
    """Run a single bounded Colab seed job; root owns cross-job reservations.

    Completed milestones and the last committed episode are published after
    each milestone and on a handled stop. A time limit does not erase partial
    progress; resume uses the original learner's episode-boundary checkpoint.
    """
    job = _job_parameters(job)
    _runtime_guard()
    import explore_weight_train as learner

    started = time.monotonic()
    out, work, source = (job[k] for k in ("out", "work", "source_root"))
    seed = job["seed"]
    status_path = out / "training_status" / f"seed_{seed}.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = status_path.with_suffix(".lock")
    lock = lock_path.open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        raise RuntimeError(f"WEIGHT1-L seed {seed} already has an active training job")
    model_out = out / "models" / f"seed_{seed}"
    payload = {"kind": "WEIGHT1-learning-length-v1", "study_fingerprint": job["study_fingerprint"],
               "parent_fingerprint": job["parent_fingerprint"], "seed": seed,
               "requested_milestones": job["episodes"], "completed_milestones": [], "milestones": {},
               "status": "preparing", "completed_episodes": 0, "last_completed_episodes": 0,
               "control_steps": 0, "job_started_unix": time.time(), "job_cap_s": job["job_cap_s"],
               "global_deadline_unix": job["global_deadline_unix"], "episode_seconds": 0.0,
               "evidence_seconds": 0.0, "training_calls": [], "job_id": uuid.uuid4().hex}
    previous = _read(status_path) if status_path.exists() else None
    if previous:
        for key in ("kind", "study_fingerprint", "parent_fingerprint", "seed"):
            if previous.get(key) != _plain(payload[key]):
                lock.close()
                raise ValueError(f"WEIGHT1-L existing status differs: {key}")
        payload["completed_milestones"] = list(previous.get("completed_milestones", []))
        payload["milestones"] = dict(previous.get("milestones", {}))
        payload["previous_job_id"] = previous.get("job_id")

    def remaining():
        return min(job["job_cap_s"] - (time.monotonic() - started),
                   job["global_deadline_unix"] - time.time())

    def check_time(reserve=0.0):
        if remaining() <= reserve:
            raise TrainingTimeLimit("WEIGHT1-L wall/job budget reached; resume from last completed episode")

    def checkpoint_state():
        if (model_out / "latest.json").exists():
            identity = learner.validate_checkpoint(model_out, job["parent_fingerprint"])
            payload["last_completed_episodes"] = identity["episodes"]
            payload["completed_episodes"] = identity["episodes"]
            payload["control_steps"] = identity["control_steps"]
            payload["latest_model_sha256"] = identity["files"]["model.pt"]
            payload["latest_replay_sha256"] = identity["files"]["replay.pkl"]
            payload["latest_checkpoint"] = identity["path"]

    def publish():
        payload["seconds"] = time.monotonic() - started
        payload["updated_unix"] = time.time()
        payload["remaining_job_s"] = max(0.0, remaining())
        payload["timings"] = {"seconds": payload["seconds"], "episode_seconds": payload["episode_seconds"],
                              "evidence_seconds": payload["evidence_seconds"],
                              "overrun_s": max(0.0, -remaining())}
        checkpoint_state()
        _write(status_path, payload)

    source_audit = None
    episode_times = []
    try:
        check_time()
        parent = verify_parent(source, job["parent_fingerprint"])
        payload["verified_parent"] = parent
        if seed == 42:
            lineage = clone_parent_checkpoint(source, model_out, parent_fingerprint=job["parent_fingerprint"],
                                               study_fingerprint=job["study_fingerprint"])
            source_audit = lineage["source"]
        else:
            if model_out.exists() and not (model_out / "long_lineage.json").exists() and any(model_out.iterdir()):
                raise ValueError("Existing fresh-seed directory has no owned long-study lineage")
            model_out.mkdir(parents=True, exist_ok=True)
            lineage = {"kind": payload["kind"], "seed": seed, "parent_fingerprint": job["parent_fingerprint"],
                       "study_fingerprint": job["study_fingerprint"], "initialization": "fresh unchanged WEIGHT1"}
            lineage_path = model_out / "long_lineage.json"
            if lineage_path.exists() and _read(lineage_path) != lineage:
                raise ValueError("WEIGHT1-L fresh seed lineage differs")
            _write(lineage_path, lineage)
        scenarios = _training_scenarios()
        if (model_out / "latest.json").exists():
            _, old_state, _ = learner._load_checkpoint(model_out, job["parent_fingerprint"])
            episode_times = [float(r["episode_seconds"]) for r in old_state["history"][-10:]]
        for completed in OFFICIAL_MILESTONES:
            existing_stage = model_out / "stages" / f"episode_{completed:06d}"
            if existing_stage.exists():
                identity = learner.validate_checkpoint(existing_stage, job["parent_fingerprint"])
                if identity["seed"] != seed or identity["episodes"] != completed:
                    raise ValueError("Existing official stage has an invalid seed or episode count")
                payload["milestones"][str(completed)] = {
                    "episodes": completed, "control_steps": identity["control_steps"], "stage": str(existing_stage),
                    "model_sha256": identity["files"]["model.pt"], "replay_sha256": identity["files"]["replay.pkl"],
                    "reused": True}
                payload["completed_milestones"] = sorted(set(payload["completed_milestones"] + [completed]))
        payload["status"] = "training"
        publish()
        runner = _episode_runner()

        def before_episode():
            # Reserve time for reconciliation, gradients and the atomic save.
            # The outer coordinator can also kill an overlong process safely:
            # the last pointer always refers to a complete prior generation.
            predicted = max(180.0, max(episode_times[-10:] or [0.0]) * 1.25)
            check_time(predicted + 120.0)

        def episode(*args, **kwargs):
            local_args = list(args)
            episode_name = Path(local_args[3]).name
            local_work = work / f"seed_{seed}" / payload["job_id"] / episode_name
            local_args[3] = local_work
            check_time(120.0)
            kwargs["max_seconds"] = max(0.001, remaining() - 120.0)
            began = time.monotonic()
            result = runner(*local_args, **kwargs)
            elapsed = time.monotonic() - began
            payload["episode_seconds"] += elapsed
            episode_times.append(elapsed)
            evidence_started = time.monotonic()
            evidence = _copy_evidence(result, local_work, out / "training" / f"seed_{seed}" / episode_name,
                                      study_fingerprint=job["study_fingerprint"], check_time=check_time)
            payload["evidence_seconds"] += time.monotonic() - evidence_started
            result.metrics["long_study_fingerprint"] = job["study_fingerprint"]
            result.metrics["long_study_evidence"] = evidence
            check_time(120.0)
            return result

        for milestone in job["episodes"]:
            official = milestone in OFFICIAL_MILESTONES and bool(job.get("preserve_milestone", True))
            stage = model_out / "stages" / f"episode_{milestone:06d}"
            if official and stage.exists():
                identity = learner.validate_checkpoint(stage, job["parent_fingerprint"])
                if identity["seed"] != seed or identity["episodes"] != milestone:
                    raise ValueError("WEIGHT1-L existing stage identity differs")
                record = {"episodes": milestone, "control_steps": identity["control_steps"],
                          "stage": str(stage), "model_sha256": identity["files"]["model.pt"],
                          "replay_sha256": identity["files"]["replay.pkl"], "reused": True}
            else:
                latest = learner.validate_checkpoint(model_out, job["parent_fingerprint"]) if (model_out / "latest.json").exists() else None
                if official and latest and latest["episodes"] > milestone:
                    raise ValueError("WEIGHT1-L passed a milestone without preserving its immutable stage")
                if latest and latest["episodes"] >= milestone:
                    record = {"episodes": latest["episodes"], "control_steps": latest["control_steps"],
                              "stage": None, "model_sha256": latest["files"]["model.pt"],
                              "replay_sha256": latest["files"]["replay.pkl"], "reused": True}
                    if official:
                        # Crash recovery may have committed the final episode
                        # before train_seed created its immutable stage.
                        snapshot = learner._milestone(model_out, Path(latest["path"]), milestone)
                        record["stage"] = str(snapshot)
                    payload["last_chunk"] = record
                    if official:
                        payload["milestones"][str(milestone)] = record
                        payload["completed_milestones"] = sorted(set(payload["completed_milestones"] + [milestone]))
                    publish()
                    continue
                check_time()
                call_started = time.monotonic()
                original_milestone = learner._milestone
                def stage_for_chunk(folder, checkpoint, completed_episode):
                    if completed_episode in OFFICIAL_MILESTONES and bool(job.get("preserve_milestone", True)):
                        return original_milestone(folder, checkpoint, completed_episode)
                    return checkpoint
                learner._milestone = stage_for_chunk
                try:
                    trained = learner.train_seed(model_out, scenarios, seed, milestone, episode,
                                                 fingerprint=job["parent_fingerprint"], budget_check=before_episode)
                finally:
                    learner._milestone = original_milestone
                payload["training_calls"].append({**trained, "call_wall_seconds": time.monotonic() - call_started})
                identity = learner.validate_checkpoint(Path(trained["stage"]), job["parent_fingerprint"])
                record = {"episodes": milestone, "control_steps": identity["control_steps"],
                          "stage": trained["stage"] if official else None, "model_sha256": identity["files"]["model.pt"],
                          "replay_sha256": identity["files"]["replay.pkl"], "reused": False}
            payload["last_chunk"] = record
            if official:
                payload["milestones"][str(milestone)] = record
                payload["completed_milestones"] = sorted(set(payload["completed_milestones"] + [milestone]))
            publish()
        payload["status"] = "complete"
    except (TrainingTimeLimit, TimeoutError) as error:
        payload["status"], payload["error"] = "time_limit", str(error)
    except Exception as error:
        payload["status"], payload["error"] = "failed", f"{type(error).__name__}: {error}"
    finally:
        try:
            if source_audit is not None:
                unchanged = _source_checkpoint_audit(source / "models" / "seed_42", job["parent_fingerprint"]) == source_audit
                payload["parent_checkpoint_unchanged"] = unchanged
                if not unchanged:
                    payload["status"], payload["error"] = "failed", "Read-only parent checkpoint changed"
            publish()
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
            lock.close()
    return _plain(payload)
