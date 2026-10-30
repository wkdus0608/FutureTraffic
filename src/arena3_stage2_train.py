"""ARENA3 second stage: B′ pretraining followed by the unchanged G1 SMDP task.

The two variants have the same observations, network, simulator, seeds, and
per-seed job-time allowance. Only the teacher objective differs. Existing ARENA3
files and the referee's evaluation fingerprint are never changed.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import io
import json
import math
import multiprocessing
import os
import random
import shutil
import sys
import threading
import time
import uuid
from concurrent.futures import ProcessPoolExecutor, FIRST_COMPLETED, wait
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import arena3_common as ac
import arena3_g1 as g1
import explore_weight as w

CONFIG = ROOT / "config/arena3_stage2_v1.json"
KIND = "arena3-stage2-teacher-smdp-v1"
VARIANTS = {"advantage": "S2A", "policy": "S2B"}
SOURCES = ("src/arena3_stage2_data.py", "src/arena3_stage2_train.py", "src/arena3_g1.py")


def config() -> dict:
    return json.loads(CONFIG.read_text())


def settings(variant: str, dataset_fingerprint: str) -> dict:
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant: {variant}")
    return {"kind": KIND, "variant": variant, "stage2": config(), "g1": g1.settings(),
            "arena3_lock_sha256": w.file_hash(ac.CONFIG), "dataset_fingerprint": dataset_fingerprint,
            "source_sha256": {p: w.file_hash(ROOT / p) for p in SOURCES}}


def atomic_json(path: Path, value: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    tmp.write_text(json.dumps(w.plain(value), ensure_ascii=False, allow_nan=False, indent=2) + "\n")
    os.replace(tmp, path)


class SpendLog:
    """Persist job time and conservatively account for interrupted operations.

    A five-second heartbeat reserves the next five seconds for active workers.
    Recovery charges that reservation, never the downtime between runtimes.
    Completed jobs use their measured worker elapsed time, not the reservation.
    """
    def __init__(self, folder: Path):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.path = self.folder / "spend.jsonl"
        self.pending = self.folder / "pending_operation.json"

    def entries(self):
        return [json.loads(s) for s in self.path.read_text().splitlines() if s.strip()] if self.path.exists() else []

    def total(self) -> float:
        return math.fsum(float(r["seconds"]) for r in self.entries())

    def add(self, seconds: float, note: str, operation_id: str) -> None:
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError("Invalid job-time accounting")
        if operation_id in {r.get("operation_id") for r in self.entries()}:
            return
        with self.path.open("a") as f:
            f.write(json.dumps({"seconds": seconds, "note": note, "operation_id": operation_id,
                                "unix": time.time()}) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def recover(self) -> dict | None:
        if not self.pending.exists():
            return None
        record = json.loads(self.pending.read_text())
        self.add(float(record["reserved_seconds"]), "recovered interrupted " + record["note"], record["operation_id"])
        self.pending.unlink()
        return record


class Operation:
    """Heartbeat for one pretrain or collection/update batch."""
    def __init__(self, spend: SpendLog, note: str, interval: float = 5.0):
        self.spend, self.note, self.interval = spend, note, interval
        self.id = uuid.uuid4().hex
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.completed = 0.0
        self.active: dict[str, float] = {}

    def start(self, key: str):
        with self.lock:
            self.active[key] = time.monotonic()
        self.heartbeat()

    def finish(self, key: str, seconds: float | None = None):
        with self.lock:
            began = self.active.pop(key)
            self.completed += max(0.0, time.monotonic() - began if seconds is None else seconds)
        self.heartbeat()

    def elapsed(self, reserve: bool = False):
        with self.lock:
            return self.completed + sum(max(0.0, time.monotonic() - t) + (self.interval if reserve else 0.0)
                                        for t in self.active.values())

    def heartbeat(self):
        atomic_json(self.spend.pending, {"operation_id": self.id, "note": self.note,
                    "reserved_seconds": self.elapsed(True), "heartbeat_unix": time.time(),
                    "reservation_interval_s": self.interval})

    def _heartbeat_loop(self):
        while not self.stop.wait(self.interval):
            self.heartbeat()

    def __enter__(self):
        self.heartbeat()
        self.thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop.set()
        self.thread.join()
        self.spend.add(self.elapsed(), self.note + (" failed" if exc_type else ""), self.id)
        self.spend.pending.unlink(missing_ok=True)


class Learner(g1.Learner):
    def __init__(self, seed: int, dataset: dict, variant: str):
        super().__init__(seed)
        self.dataset, self.variant = dataset, variant
        self.teacher_rng = np.random.default_rng(seed + 200_000)
        self.choices = 0
        self.finetune = config()["finetune"]
        self.teacher = {k: torch.as_tensor(dataset[k]) for k in ("observations", "gains", "gain_se", "mask")}

    def teacher_weight(self) -> float:
        """Constant when teacher_decay_choices is null (DQfD-style); otherwise linear decay to zero."""
        decay = self.finetune["teacher_decay_choices"]
        scale = 1.0 if decay is None else max(0.0, 1.0 - self.choices / float(decay))
        return float(self.finetune["teacher_weight"]) * scale

    def gradient_step(self) -> float:
        from arena3_stage2_data import supervised_loss
        cfg = g1.gcfg()
        d = self.replay.sample(cfg["batch_size"], self.sample_rng)
        with torch.no_grad():
            nxt = d["next_obs"]
            mask = torch.from_numpy(g1.mode_mask(nxt.numpy()))
            chosen = self.online(nxt).masked_fill(~mask, -torch.inf).argmax(dim=1, keepdim=True)
            value = self.target(nxt).gather(1, chosen).squeeze(1)
            value = torch.where(d["done"], torch.zeros_like(value), value)
            target = d["reward"] + d["discount"] * value
        pred = self.online(d["obs"]).gather(1, d["mode"].view(-1, 1)).squeeze(1)
        td = F.smooth_l1_loss(pred, target)
        weight = self.teacher_weight()
        loss = td
        if weight:
            ix = self.teacher_rng.integers(0, len(self.dataset["observations"]), size=cfg["batch_size"])
            teacher = supervised_loss(self.online, self.teacher["observations"][ix], self.teacher["gains"][ix],
                                      self.teacher["gain_se"][ix], self.teacher["mask"][ix], self.variant)
            loss = td + weight * teacher
        if not bool(torch.isfinite(loss)):
            raise ValueError("Non-finite RL/teacher loss")
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.optimizer.step()
        self.updates += 1
        if self.updates % cfg["target_every_updates"] == 0:
            self.target.load_state_dict(self.online.state_dict())
            self.syncs += 1
        return float(loss.detach())


def save_model(folder: Path, learner: Learner, state: dict, stage: str) -> str:
    folder = Path(folder)
    folder.parent.mkdir(parents=True, exist_ok=True)
    tmp = folder.with_name(f".{folder.name}.{uuid.uuid4().hex}.tmp")
    tmp.mkdir()
    st = settings(learner.variant, learner.dataset["fingerprint"])
    meta = {"kind": KIND, "settings": st, "seed": learner.seed, "variant": learner.variant,
            "dataset_fingerprint": learner.dataset["fingerprint"], "stage": stage,
            "episodes": state["episodes"], "control_steps": state["control_steps"], "choices": state["choices"]}
    torch.save({**meta, "online": learner.online.state_dict()}, tmp / "model.pt")
    sha = w.file_hash(tmp / "model.pt")
    atomic_json(tmp / "manifest.json", {**meta, "model_sha256": sha})
    if folder.exists():
        shutil.rmtree(folder)
    os.replace(tmp, folder)
    return sha


def model_identity(folder: Path) -> str:
    folder = Path(folder)
    m = json.loads((folder / "manifest.json").read_text())
    if m.get("kind") != KIND or m.get("settings") != settings(m.get("variant"), m.get("dataset_fingerprint")):
        raise ValueError(f"Stage2 model settings/source mismatch: {folder}")
    sha = w.file_hash(folder / "model.pt")
    if sha != m["model_sha256"]:
        raise ValueError(f"Stage2 model SHA mismatch: {folder}")
    return sha


def load_model(folder: Path):
    model_identity(folder)
    m = json.loads((Path(folder) / "manifest.json").read_text())
    p = torch.load(Path(folder) / "model.pt", map_location="cpu", weights_only=False)
    for key in ("kind", "settings", "seed", "variant", "dataset_fingerprint", "stage"):
        if p.get(key) != m.get(key):
            raise ValueError(f"Payload/manifest mismatch: {key}")
    net = g1.DuelingNet(tuple(g1.gcfg()["network"]))
    net.load_state_dict(p["online"])
    net.eval()
    return net


def save_checkpoint(seed_dir: Path, learner: Learner, state: dict):
    payload = {"kind": KIND, "settings": settings(learner.variant, learner.dataset["fingerprint"]),
        "seed": learner.seed, "variant": learner.variant, "state": state,
        "online": learner.online.state_dict(), "target": learner.target.state_dict(),
        "optimizer": learner.optimizer.state_dict(), "replay": learner.replay.state(),
        "sample_rng": learner.sample_rng.bit_generator.state, "teacher_rng": learner.teacher_rng.bit_generator.state,
        "torch_rng": torch.get_rng_state(), "updates": learner.updates, "syncs": learner.syncs}
    tmp = seed_dir / "checkpoint.pt.tmp"
    torch.save(payload, tmp)
    sha = w.file_hash(tmp)
    os.replace(tmp, seed_dir / "checkpoint.pt")
    atomic_json(seed_dir / "checkpoint.json", {"sha256": sha, "episodes": state["episodes"],
                "choices": state["choices"], "settings": payload["settings"], "updated_unix": time.time()})


def load_checkpoint(seed_dir: Path, dataset: dict, variant: str, seed: int):
    path = seed_dir / "checkpoint.pt"
    if not path.exists():
        return None
    meta = json.loads((seed_dir / "checkpoint.json").read_text())
    if w.file_hash(path) != meta["sha256"]:
        raise ValueError("Checkpoint SHA mismatch")
    p = torch.load(path, map_location="cpu", weights_only=False)
    if p.get("kind") != KIND or p.get("settings") != settings(variant, dataset["fingerprint"]) or p.get("seed") != seed:
        raise ValueError("Checkpoint variant, data, seed, source, or configuration mismatch")
    if p["state"]["episodes"] != meta["episodes"]:
        raise ValueError("Checkpoint metadata mismatch")
    learner = Learner(seed, dataset, variant)
    learner.online.load_state_dict(p["online"])
    learner.target.load_state_dict(p["target"])
    learner.optimizer.load_state_dict(p["optimizer"])
    learner.replay = g1.Replay.from_state(p["replay"])
    learner.sample_rng.bit_generator.state = p["sample_rng"]
    learner.teacher_rng.bit_generator.state = p["teacher_rng"]
    torch.set_rng_state(p["torch_rng"])
    learner.updates, learner.syncs, learner.choices = int(p["updates"]), int(p["syncs"]), int(p["state"]["choices"])
    return learner, p["state"]


def scenario_for(seed: int, episode: int) -> dict:
    """Same arrival placement and SUMO seed schedule for both variants."""
    rng = random.Random(ac.seed32("STAGE2_SCENARIO", seed, episode))
    date = rng.choice(ac.config()["dates"]["train"])
    slot = rng.choice(ac.TRAIN_SLOTS)
    return ac.make_situation(date, slot, f"S2s{seed}e{episode}")


def _collect(pool, worker, jobs, operation):
    pending = {}
    for i, job in enumerate(jobs):
        key = f"episode_{i}"
        operation.start(key)
        pending[pool.submit(worker, job)] = (i, key)
    results = [None] * len(jobs)
    while pending:
        done, _ = wait(pending, timeout=5, return_when=FIRST_COMPLETED)
        for future in done:
            i, key = pending.pop(future)
            try:
                result = future.result()
            except BaseException as e:
                result = {"ok": False, "status": "error", "errors": [repr(e)], "seconds": None}
            operation.finish(key, result.get("seconds"))
            results[i] = result
    return results


def train(seed: int, out: Path, work: Path, dataset: dict, variant: str, *, executor=None,
          worker=g1.episode_worker, max_batches: int | None = None) -> dict:
    from arena3_stage2_data import pretrain
    if sys.platform == "darwin" and worker is g1.episode_worker:
        raise RuntimeError("맥에서는 SUMO를 실행하지 않습니다. 실제 학습은 Colab에서 실행하세요.")
    cfg = g1.gcfg()
    if variant not in VARIANTS or seed not in cfg["seeds"]:
        raise ValueError("Unknown stage2 variant or seed")
    if len(dataset["observations"]) == 0:
        raise ValueError("No clean teacher samples")
    if not set(map(str, dataset["dates"])) <= set(ac.config()["dates"]["train"]):
        raise ValueError("Teacher data contains evaluation dates")
    seed_dir = Path(out) / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    spend = SpendLog(seed_dir)
    recovered = spend.recover()
    budget = float(cfg["cpu_hours_per_seed"]) * 3600
    loaded = load_checkpoint(seed_dir, dataset, variant, seed)
    if loaded is None:
        if spend.total() >= budget:
            raise RuntimeError("Budget spent before a valid initial checkpoint; preserve output for audit")
        learner = Learner(seed, dataset, variant)
        state = {"episodes": 0, "choices": 0, "control_steps": 0, "discarded": [], "history": [],
                 "milestones": [], "pretraining": None}
        with Operation(spend, "pretraining") as op:
            op.start("pretrain")
            pre = pretrain(learner.online, dataset, variant, seed)
            op.finish("pretrain")
            state["pretraining"] = pre
            learner.target.load_state_dict(learner.online.state_dict())
            learner.optimizer = torch.optim.Adam(learner.online.parameters(), lr=cfg["learning_rate"])
            save_model(seed_dir / "pretrained", learner, state, "pretrained")
        save_checkpoint(seed_dir, learner, state)
    else:
        learner, state = loaded
        if not (seed_dir / "pretrained" / "manifest.json").exists():
            raise ValueError("Missing original pretrained model; cannot reproduce pretrain/RL comparison")
        model_identity(seed_dir / "pretrained")
    train_dates = set(ac.config()["dates"]["train"])
    batches = 0

    def status(finished=False):
        result = {"variant": variant, "name": VARIANTS[variant], "seed": seed, "finished": finished,
                  "episodes": state["episodes"], "choices": state["choices"], "control_steps": state["control_steps"],
                  "updates": learner.updates, "discarded": len(state["discarded"]),
                  "cpu_hours": spend.total() / 3600, "budget_cpu_hours": budget / 3600,
                  "dataset_fingerprint": dataset["fingerprint"], "pretraining": state["pretraining"],
                  "recovered_interruption": bool(recovered), "updated_unix": time.time()}
        atomic_json(seed_dir / "status.json", result)
        return result

    status()
    if spend.total() < budget and (max_batches is None or max_batches > 0):
        pool = executor or ProcessPoolExecutor(max_workers=cfg["parallel_episodes"],
                mp_context=multiprocessing.get_context("spawn"), initializer=g1._worker_init)
        with pool:
            while spend.total() < budget and (max_batches is None or batches < max_batches):
                jobs = []
                for b in range(cfg["parallel_episodes"]):
                    idx = state["episodes"] + b
                    s = scenario_for(seed, idx)
                    if s["date"] not in train_dates:
                        raise ValueError("Nontraining scenario selected")
                    jobs.append({"situation": s, "work": str(Path(work) / f"seed_{seed}" / f"e{idx}"),
                        "net": g1._state_bytes(learner.online), "base_choices": state["choices"],
                        "rng_seed": ac.seed32("STAGE2_EPSILON", seed, idx)})
                with Operation(spend, f"batch e{state['episodes']}") as op:
                    results = _collect(pool, worker, jobs, op)
                    bad = [r for r in results if not r["ok"]]
                    if bad:
                        raise RuntimeError(f"{VARIANTS[variant]} seed {seed}: episode failed: {bad[0]['errors']}")
                    op.start("gradient")
                    for job, r in zip(jobs, results):
                        idx = state["episodes"]
                        state["episodes"] += 1
                        state["control_steps"] += 720
                        if r["status"] == "discard":
                            state["discarded"].append({"episode": idx, "situation": ac.situation_id(job["situation"]),
                                                       "collision": r.get("collision_details")})
                            if len(state["discarded"]) > cfg["max_discarded"]:
                                raise RuntimeError("Discarded episode limit exceeded")
                            continue
                        transitions = g1.smdp_transitions(r["choices"], r["rewards"], cfg["gamma_per_5s"], cfg["n_choice_steps"])
                        state["choices"] += len(r["choices"])
                        learner.choices = state["choices"]
                        info = learner.learn(transitions)
                        state["history"].append({"episode": idx, "situation": ac.situation_id(job["situation"]),
                            **r["metrics"], "transitions": len(transitions), **info, "updates_total": learner.updates})
                    op.finish("gradient")
                for steps in cfg["milestone_control_steps"]:
                    if state["control_steps"] >= steps and steps not in state["milestones"]:
                        save_model(seed_dir / "milestones" / f"steps_{steps}", learner, state, "milestone")
                        state["milestones"].append(steps)
                save_checkpoint(seed_dir, learner, state)
                batches += 1
                result = status()
                print(f"{VARIANTS[variant]} seed {seed}: episodes={state['episodes']}, choices={state['choices']}, "
                      f"updates={learner.updates}, job-hours={result['cpu_hours']:.3f}/{budget/3600:.0f}", flush=True)
    finished = spend.total() >= budget
    if finished:
        save_model(seed_dir / "final", learner, state, "final")
    return status(finished)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["train", "smoke"])
    parser.add_argument("--variant", choices=list(VARIANTS), required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--max-batches", type=int)
    args = parser.parse_args()
    if sys.platform == "darwin":
        raise SystemExit("맥에서는 SUMO를 실행하지 않습니다. 이 명령은 Colab 노트북에서 실행하세요.")
    w.configure()
    torch.set_num_threads(1)
    import arena3_referee as ref
    from arena3_stage2_data import load_dataset
    ref.register(args.out, extra_files=(*SOURCES, "config/arena3_stage2_v1.json"))
    dataset = load_dataset(args.data)
    result = train(args.seed, args.out, args.work, dataset, args.variant,
                   max_batches=1 if args.command == "smoke" else args.max_batches)
    if args.command == "smoke":
        if result["episodes"] < 2 or result["choices"] <= 0 or result["discarded"]:
            raise RuntimeError("SUMO smoke did not complete two clean, informative episodes")
        load_model(args.out / f"seed_{args.seed}" / "pretrained")
        atomic_json(args.out / "smoke.json", {"passed": True, **result})
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
