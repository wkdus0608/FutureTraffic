"""PI1: 작은 수요 정보 검사와 고정 학생의 미래 비용으로 한 차례 정책 개선(156절).

Colab 42번: run --old <online/runs> --new <collect/runs> --combo <combo>
  --out <새 PI1 폴더> --work /content/policy_iter --workers 4 --cpu-budget-s 36000
기존 실행 코드와 결과는 보존한다. 확인용 미래는 학습에 넣지 않는다.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import multiprocessing
import resource
import shutil
import statistics
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import explore_collect as ecl
import explore_combo as cb
import explore_distill as ed
import explore_history as eh
import explore_online as eo
import explore_policy_rollout as pr
import explore_remeasure as rm
import explore_rollout as er

PROJECT = ed.PROJECT
TRAIN_GROUPS = tuple(ecl.NEW_GROUPS)
BASE_SEED = 42
STATES_PER_GROUP = 16
SELECT_SEEDS = tuple(range(3001, 3009))
CONFIRM_SEEDS = tuple(range(4001, 4009))
HORIZON_S = 600
UPDATE_STEPS, UPDATE_LR, ANCHOR_WEIGHT = 2000, 1e-4, 0.1
MAX_CPU_S = 36_000
EVAL_MODES = ("p0", "parent", "child")


def _plain(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, dict):
        return {str(k): _plain(v) for k, v in x.items()}
    if isinstance(x, (tuple, list)):
        return [_plain(v) for v in x]
    return x


def digest(x) -> str:
    return hashlib.sha256(json.dumps(_plain(x), sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(_plain(value), ensure_ascii=False, indent=1, allow_nan=False))
    tmp.replace(path)


def read_json(path: Path):
    return json.loads(path.read_text())


def cpu_now() -> float:
    a, b = resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN)
    return a.ru_utime + a.ru_stime + b.ru_utime + b.ru_stime


def plan() -> dict:
    from focused_data import environment_fingerprint
    files = (Path(__file__), Path(eh.__file__), Path(pr.__file__), Path(cb.__file__), Path(eo.__file__), Path(er.__file__))
    return {"experiment": "PI1", "section": 156, "environment": environment_fingerprint(),
            "code": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
            "train_groups": TRAIN_GROUPS, "base_seed": BASE_SEED, "states_per_group": STATES_PER_GROUP,
            "select_seeds": SELECT_SEEDS, "confirm_seeds": CONFIRM_SEEDS, "horizon_s": HORIZON_S,
            "terminal_s": er.TERMINAL_S, "update_steps": UPDATE_STEPS, "update_lr": UPDATE_LR,
            "anchor_weight": ANCHOR_WEIGHT, "eval_groups": ecl.EVAL_GROUPS, "eval_seeds": ecl.EVAL_SEEDS}


def register(args) -> str:
    from focused_data import environment_fingerprint
    rm._guard()
    if environment_fingerprint() != "b20888514ea3":
        raise SystemExit("PI1은 잠근 pet1·북쪽 연장 240 환경만 쓴다")
    out = Path(args.out)
    record = _plain({"plan": plan(), "cpu_budget_s": min(float(args.cpu_budget_s), MAX_CPU_S)})
    key = digest(record)
    target = out / "experiment.json"
    if target.exists() and read_json(target) != record:
        raise SystemExit("PI1 코드·설정이 이전 실행과 다르다. 기존 결과를 보존하고 새 출력 폴더를 쓴다")
    write_json(target, record)
    return key


class BudgetExceeded(RuntimeError):
    pass


class Budget:
    """완료 작업의 실제 프로세스·SUMO CPU 합. 새 묶음 시작 전 여유를 확보한다."""
    def __init__(self, out: Path, limit: float):
        self.path = Path(out) / "cpu_ledger.json"
        self.limit = min(float(limit), MAX_CPU_S)
        self.entries = read_json(self.path) if self.path.exists() else {}

    @property
    def used(self):
        return sum(float(x) for x in self.entries.values())

    def check(self, reserve=0.0):
        if self.used + reserve > self.limit:
            raise BudgetExceeded(f"PI1 CPU 예산: 사용 {self.used:.1f}초, 상한 {self.limit:.1f}초. 자동 확대하지 않는다")

    def add(self, key: str, seconds: float):
        if key not in self.entries:
            self.entries[key] = max(0.0, float(seconds))
            write_json(self.path, self.entries)


def spaced_indices(n: int, maximum: int = STATES_PER_GROUP) -> list[int]:
    if n <= 0:
        return []
    return sorted(set(int(x) for x in np.linspace(0, n - 1, min(n, maximum))))


def choose(costs: dict[int, list[float]], parent: int, p0: int) -> int:
    means = {c: statistics.fmean(v) for c, v in costs.items()}
    low = min(means.values())
    for c in (parent, p0, *sorted(means)):
        if c in means and means[c] <= low + 1e-9:
            return int(c)
    raise ValueError("후보가 없다")


def _pack(src: Path, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    with src.open("rb") as a, gzip.open(dest, "wb", compresslevel=6) as b:
        shutil.copyfileobj(a, b)


def _unpack(src: Path, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(src, "rb") as a, dest.open("wb") as b:
        shutil.copyfileobj(a, b)


def collect_group(date: str, period: str, out: Path, work: Path, experiment: str) -> dict:
    """부모 운용 1회 후 대상 시점만 행동 재생해 저장. 부모는 탐색 결과로 바꾸지 않는다."""
    if (date, period) not in TRAIN_GROUPS:
        raise ValueError("학습 날짜만 수집한다")
    parent = pr.FrozenPolicy(out / "parent")
    group = f"{date}_{period}"
    dst = out / "states" / group
    done = dst / "done.json"
    if done.exists():
        old = read_json(done)
        if old["experiment"] != experiment or old["parent_hash"] != parent.hash:
            raise RuntimeError("수집 캐시의 코드·부모가 다르다")
        for name, sha in old["files"].items():
            if hashlib.sha256((dst / name).read_bytes()).hexdigest() != sha:
                raise RuntimeError(f"보존 상태가 바뀌었다: {name}")
        return old
    t_cpu = cpu_now()
    w = work / "collect" / group
    ep = eo.OnlineEpisode(date, period, w / "first" / "routes", BASE_SEED)
    trace = []
    try:
        while not ep.done:
            allowed, p0 = ep.decide()
            action = p0
            if len(allowed) > 1:
                row = parent.sample_current(ep, allowed, p0)
                action = int(row["parent"])
                trace.append(row)
            ep.apply(action)
    finally:
        ep.close()
    from focused_eval import judge_collisions
    source_metrics = eo.episode_metrics(eo.read_tripinfo(ep.tripinfo), ep.vehicles, ep.handover, 3600)
    collisions = judge_collisions(ep.collisions)
    if (source_metrics["missing"] or source_metrics["unfinished"] or ep.cons.interventions
            or collisions["collisions_at_204820"] or collisions["collisions_unexplained"]):
        raise RuntimeError(f"{group}: 부모 상태 수집 실행의 누락·규칙·충돌 검사 실패")
    eligible = [r for r in trace if r["left_point"] and r["t0"] < ep.handover + 3600]
    selected = [eligible[i] for i in spaced_indices(len(eligible))]
    if not selected:
        raise RuntimeError(f"{group}: 학생 개입 시점 없음")
    wanted = {float(r["t0"]): r for r in selected}
    dst.mkdir(parents=True, exist_ok=True)
    write_json(dst / "trace.json", trace)
    write_json(dst / "source_metrics.json", source_metrics)
    replay = eo.OnlineEpisode(date, period, w / "replay" / "routes", BASE_SEED)
    saved, cursor = [], 0
    try:
        while not replay.done and replay.env.sim_step <= max(wanted):
            allowed, p0 = replay.decide()
            action = p0
            if len(allowed) > 1:
                now = parent.sample_current(replay, allowed, p0)
                if cursor >= len(trace) or digest(now) != digest(trace[cursor]):
                    raise RuntimeError(f"{group}: 부모 행동 재생 불일치, {replay.env.sim_step}")
                cursor += 1
                action = int(now["parent"])
                if float(now["t0"]) in wanted:
                    stem = w / "save" / str(int(now["t0"]))
                    replay.snapshot(stem, allowed, p0)
                    for suffix in (".xml", ".pkl"):
                        _pack(stem.with_suffix(suffix), dst / f"{stem.name}{suffix}.gz")
                    saved.append({**now, "group": group, "seed": BASE_SEED, "stem": stem.name})
            replay.apply(action)
    finally:
        replay.close()
    if len(saved) != len(wanted):
        raise RuntimeError("저장 대상으로 고른 상태가 모두 복구되지 않았다")
    _pack(replay.route, dst / (replay.route.name + ".gz"))
    write_json(dst / "states.json", saved)
    files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(dst.iterdir()) if p.is_file() and p != done}
    result = {"experiment": experiment, "parent_hash": parent.hash, "group": group, "states": len(saved),
              "eligible": len(eligible), "decisions_checked": cursor, "files": files, "route": replay.route.name,
              "cpu_s": cpu_now() - t_cpu}
    write_json(done, result)
    return result


def unpack_states(out: Path, work: Path) -> list[dict]:
    parent = pr.FrozenPolicy(out / "parent")
    rows = []
    for d, p in TRAIN_GROUPS:
        group = f"{d}_{p}"
        src, dst = out / "states" / group, work / "states" / group
        done = read_json(src / "done.json")
        if done["parent_hash"] != parent.hash:
            raise RuntimeError("상태 수집 부모와 탐색 부모가 다르다")
        for name, sha in done["files"].items():
            if hashlib.sha256((src / name).read_bytes()).hexdigest() != sha:
                raise RuntimeError(f"보존 상태 검사값 불일치: {src / name}")
            if name.endswith(".gz"):
                _unpack(src / name, dst / name[:-3])
        rows += [{**r, "state_stem": str(dst / r["stem"]), "route_dir": str(dst)} for r in read_json(src / "states.json")]
    return rows


def rollout_key(row, candidate, seed, continuation):
    return f"{row['group']}_{int(row['t0'])}_{continuation}_c{candidate}_s{seed}"


def _rollout_job(job):
    row, candidate, seed, continuation, parent_dir = job
    t_cpu = cpu_now()
    r = pr.measured(Path(row["state_stem"]), Path(row["route_dir"]), candidate, seed, continuation, Path(parent_dir), HORIZON_S)
    r["cpu_s"] = cpu_now() - t_cpu
    if r["leaks"] or r["changed_by_rules"]:
        raise RuntimeError("후보 탐색에서 실제 미래 유출 또는 규칙 개입")
    return r


def run_jobs(jobs: list, out: Path, workers: int, budget: Budget, experiment: str):
    """파일마다 코드·부모·시나리오 검사. CPU 여유가 있는 묶음만 제출한다."""
    parent_hash = pr.FrozenPolicy(out / "parent").hash
    results, missing = {}, []
    for j in jobs:
        row, c, seed, cont, _ = j
        key = rollout_key(row, c, seed, cont)
        path = out / "rollouts" / f"{key}.json"
        if path.exists():
            r = read_json(path)
            if (r.get("experiment") != experiment or r.get("parent_hash") != parent_hash
                    or r.get("candidate") != c or r.get("seed") != seed or r.get("continuation") != cont):
                raise RuntimeError(f"탐색 캐시 불일치: {key}")
            results[key] = r
            budget.add(key, r["cpu_s"])
        else:
            missing.append(j)
    if not missing:
        return results
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        for start in range(0, len(missing), workers):
            batch = missing[start:start + workers]
            budget.check(reserve=60.0 * len(batch))
            for j, r in zip(batch, pool.map(_rollout_job, batch)):
                row, c, seed, cont, _ = j
                key = rollout_key(row, c, seed, cont)
                r["experiment"] = experiment
                write_json(out / "rollouts" / f"{key}.json", r)
                budget.add(key, r["cpu_s"])
                results[key] = r
            budget.check()
            print(f"탐색 {min(start + workers, len(missing))}/{len(missing)}, CPU 누적 {budget.used:.0f}초", flush=True)
    return results


def smoke_check(rows: list[dict], out: Path, experiment: str, budget: Budget):
    """처음 두 상태: 부모 복원 관측·행동, 같은 미래 재실행, P0 후속 기존 측정과 대조."""
    target = out / "check.json"
    parent = pr.FrozenPolicy(out / "parent")
    if target.exists():
        old = read_json(target)
        if old.get("experiment") != experiment or old.get("parent_hash") != parent.hash or not old.get("passed"):
            raise RuntimeError("작은 복원 검사의 코드·부모가 다르다")
        budget.add("smoke_check", old["cpu_s"])
        return old
    budget.check(reserve=1200)
    t_cpu = cpu_now()
    checked = []
    for row in rows[:2]:
        restored = pr.verify_snapshot(Path(row["state_stem"]), Path(row["route_dir"]), out / "parent", row)
        if not restored["ok"]:
            raise RuntimeError(f"부모 상태 복원 불일치: {restored['problems']}")
        for c in row["allowed"][:2]:
            for seed in SELECT_SEEDS[:2]:
                a = pr.measured(Path(row["state_stem"]), Path(row["route_dir"]), c, seed, "parent", out / "parent", HORIZON_S)
                b = pr.measured(Path(row["state_stem"]), Path(row["route_dir"]), c, seed, "parent", out / "parent", HORIZON_S)
                if a["marks"] != b["marks"] or a["leaks"] or b["leaks"] or a["changed_by_rules"] or b["changed_by_rules"]:
                    raise RuntimeError("부모 후속 탐색이 같은 미래에서 재현되지 않는다")
                original = er.rollout(Path(row["state_stem"]), Path(row["route_dir"]), c, seed, HORIZON_S)
                p0 = pr.measured(Path(row["state_stem"]), Path(row["route_dir"]), c, seed, "p0", out / "parent", HORIZON_S)
                if original["marks"] != p0["marks"] or p0["leaks"] or p0["changed_by_rules"]:
                    raise RuntimeError("P0 후속 지연 셈이 기존 탐색과 다르다")
                checked.append({"group": row["group"], "t0": row["t0"], "candidate": c, "seed": seed})
    result = {"experiment": experiment, "parent_hash": parent.hash, "passed": True, "checked": checked,
              "cpu_s": cpu_now() - t_cpu}
    write_json(target, result)
    budget.add("smoke_check", result["cpu_s"])
    return result


def pilot_reading(rows: list[dict]) -> dict:
    """선택에 쓰지 않은 확인 미래만 가능성 판정에 쓴다. 학습 목표와 분리."""
    groups = {f"{d}_{p}": [] for d, p in TRAIN_GROUPS}
    for row in rows:
        groups[row["group"]].append(float(row["confirmed_gain"]))
    group_gain = {g: statistics.fmean(v) if v else 0.0 for g, v in groups.items()}
    changes = sum(r["chosen"] != r["parent"] for r in rows)
    positive = sum(x > 0 for x in group_gain.values())
    total = sum(r["confirmed_gain"] for r in rows)
    passed = total > 0 and positive >= 2 and changes >= 6
    return {"passed": passed, "confirmed_gain": total, "group_gain": group_gain,
            "positive_groups": positive, "changes": changes, "states": len(rows),
            "word": "한 번 개선할 후보 확인(소규모 검사)" if passed else "개선 후보 확인 기준 미통과",
            "note": "독립 미래를 썼으나 같은 학습 날짜의 제한된 상태 표본이다"}


def compare(rows: list[dict], out: Path, workers: int, budget: Budget, experiment: str) -> dict:
    parent_dir = str(out / "parent")
    jobs = [(row, c, seed, cont, parent_dir) for row in rows for c in row["allowed"]
            for seed in SELECT_SEEDS for cont in ("p0", "parent")]
    select_results = run_jobs(jobs, out, workers, budget, experiment)
    selected, confirm_jobs = [], []
    for row in rows:
        q = {c: [select_results[rollout_key(row, c, s, "parent")]["marks"]["600"]["cost"] for s in SELECT_SEEDS]
             for c in row["allowed"]}
        q0 = {c: [select_results[rollout_key(row, c, s, "p0")]["marks"]["600"]["cost"] for s in SELECT_SEEDS]
              for c in row["allowed"]}
        chosen = choose(q, row["parent"], row["p0"])
        selected.append({**row, "chosen": chosen, "p0_follow_chosen": choose(q0, row["parent"], row["p0"]),
                         "costs": q, "p0_follow_costs": q0})
        confirm_jobs += [(row, c, seed, "parent", parent_dir) for c in sorted({chosen, row["parent"]}) for seed in CONFIRM_SEEDS]
    confirms = run_jobs(confirm_jobs, out, workers, budget, experiment)
    for row in selected:
        old = [confirms[rollout_key(row, row["parent"], s, "parent")]["marks"]["600"] for s in CONFIRM_SEEDS]
        new = [confirms[rollout_key(row, row["chosen"], s, "parent")]["marks"]["600"] for s in CONFIRM_SEEDS]
        row["confirmed_gain"] = statistics.fmean(a["cost"] - b["cost"] for a, b in zip(old, new))
        row["confirmed_delay_gain"] = statistics.fmean(a["delay"] - b["delay"] for a, b in zip(old, new))
        # 확인 표본의 비용은 targets()가 읽는 costs에 넣지 않는다.
        row["confirm_seeds"] = list(CONFIRM_SEEDS)
    result = {"experiment": experiment, "parent_hash": pr.FrozenPolicy(out / "parent").hash,
              "reading": pilot_reading(selected), "rows": selected,
              "continuation_choice_changes": sum(r["chosen"] != r["p0_follow_chosen"] for r in selected)}
    write_json(out / "pilot.json", result)
    return result


def improve(states: list[dict], samples: list[dict], anchor_x: np.ndarray, steps: int = UPDATE_STEPS) -> list[dict]:
    """새 Q부모 회귀 + 과거 입력의 부모 출력 유지 벌점. 옛 QP0는 정답으로 읽지 않는다."""
    import torch as th
    th.set_num_threads(1)
    x = th.as_tensor(np.stack([r["obs"] for r in samples]), dtype=th.float32)
    normalized = [{**r, "costs": {int(c): v for c, v in r["costs"].items()}} for r in samples]
    y0 = np.stack([ed.targets(r) for r in normalized])
    mask, y = th.as_tensor(~np.isnan(y0)), th.as_tensor(np.nan_to_num(y0))
    anchor = th.as_tensor(anchor_x, dtype=th.float32)
    updated = []
    for initial in states:
        net = cb.make_net(x.shape[1])
        net.load_state_dict(initial)
        with th.no_grad():
            original_output = net(anchor).detach()
        opt = th.optim.Adam(net.parameters(), lr=UPDATE_LR, weight_decay=ed.MLP_DECAY)
        for _ in range(steps):
            opt.zero_grad()
            value_loss = (((net(x) - y) ** 2) * mask).sum() / mask.sum()
            # 원래 입력의 행동 가능 마스크로 정규화해 불가능한 출력은 묶지 않는다.
            anchor_mask = anchor[:, -ed.N_GREENS:] > 0.5
            keep = (((net(anchor) - original_output) ** 2) * anchor_mask).sum() / anchor_mask.sum()
            loss = value_loss + ANCHOR_WEIGHT * keep
            if not th.isfinite(loss):
                raise RuntimeError("새 학생 학습 손실이 유한하지 않다")
            loss.backward()
            opt.step()
        updated.append({k: v.detach().clone() for k, v in net.state_dict().items()})
    return updated


def train(args, experiment: str):
    import torch as th
    out = Path(args.out)
    pilot = read_json(out / "pilot.json")
    parent = pr.FrozenPolicy(out / "parent")
    if pilot["experiment"] != experiment or pilot["parent_hash"] != parent.hash or not pilot["reading"]["passed"]:
        raise RuntimeError("같은 부모·코드의 작은 검사 통과 전에는 학습하지 않는다")
    target = out / "child" / "policy.json"
    if target.exists():
        policy = read_json(target)
        if (policy.get("experiment") != experiment or policy.get("parent_hash") != parent.hash
                or policy.get("pilot_sha256") != hashlib.sha256((out / "pilot.json").read_bytes()).hexdigest()):
            raise RuntimeError("학생 학습 캐시가 다른 코드·부모의 결과다")
        pr.FrozenPolicy(target.parent)
        return policy
    samples = eh.load_samples(Path(args.old), Path(args.new), Path(args.combo), Path(args.work) / "history")
    key = "obs125" if parent.width == 125 else "obs96"
    anchors = np.stack([s[key] for s in samples])
    initials = [th.load(out / "parent" / f"seed{s}.pt", map_location="cpu", weights_only=True) for s in range(5)]
    states = improve(initials, pilot["rows"], anchors)
    target.parent.mkdir(parents=True, exist_ok=True)
    for i, state in enumerate(states):
        th.save(state, target.parent / f"seed{i}.pt")
    policy = {"width": parent.width, "method": "l0", "uses_history": parent.width == 125,
              "model_seeds": list(range(5)), "weights": {f"seed{i}": rm.file_hash(target.parent / f"seed{i}.pt") for i in range(5)},
              "experiment": experiment, "parent_hash": parent.hash, "samples": len(pilot["rows"]),
              "targets": "first_action_then_frozen_parent_600s_plus_terminal75", "select_seeds": SELECT_SEEDS,
              "confirm_seeds_not_in_training": CONFIRM_SEEDS, "steps": UPDATE_STEPS,
              "lr": UPDATE_LR, "anchor_weight": ANCHOR_WEIGHT, "anchor_role": "parent_output_regularization_not_QP0_target",
              "pilot_sha256": hashlib.sha256((out / "pilot.json").read_bytes()).hexdigest()}
    write_json(target, policy)
    return policy


def _eval_job(job):
    from focused_eval import judge_collisions
    import focused_train as ft
    date, period, seed, mode, out_string, work_string, experiment = job
    out, work = Path(out_string), Path(work_string)
    directory = out / "eval" / f"{date}_{period}_s{seed}_{mode}"
    policy = pr.FrozenPolicy(out / ("parent" if mode == "p0" else mode))
    target = directory / "metrics.json"
    if target.exists():
        saved = read_json(target)
        if saved.get("experiment") != experiment or saved.get("policy_hash") != policy.hash:
            raise RuntimeError("연속 평가 캐시가 현재 코드·학생과 다르다")
        return saved
    started, t_cpu = time.monotonic(), cpu_now()
    ep = eo.OnlineEpisode(date, period, work / "eval" / directory.name / "routes", seed)
    decisions = []
    try:
        while not ep.done:
            allowed, p0 = ep.decide()
            action = p0
            if len(allowed) > 1:
                row = policy.sample_current(ep, allowed, p0)
                action = p0 if mode == "p0" else int(row["parent"])
                row["executed"] = action
                decisions.append(row)
            ep.apply(action)
    finally:
        ep.close()
    trips = eo.read_tripinfo(ep.tripinfo)
    metrics = eo.episode_metrics(trips, ep.vehicles, ep.handover, ft.CONTROL_S)
    collisions = judge_collisions(ep.collisions)
    active = [r for r in decisions if r["in_control"] and r["left_point"]]
    cohort = sorted(v for v, (dep, _, _) in ep.vehicles.items() if ep.handover <= dep < ep.handover + ft.CONTROL_S)
    metrics.update({"date": date, "period": period, "seed": seed, "mode": mode, "experiment": experiment,
                    "policy_hash": policy.hash, "cohort_sha256": digest(cohort), "handover": ep.handover,
                    "changes": sum(r["executed"] != r["p0"] for r in active), "student_decisions": len(active) if mode != "p0" else 0,
                    "changed_by_rules": ep.cons.interventions, "collisions_at_204820": collisions["collisions_at_204820"],
                    "collisions_unexplained": len(collisions["collisions_unexplained"]),
                    "cpu_s": cpu_now() - t_cpu, "seconds": time.monotonic() - started})
    write_json(directory / "decisions.json", decisions)
    write_json(target, metrics)
    return metrics


def evaluate(args, experiment: str, budget: Budget):
    jobs = [(d, p, s, m, args.out, args.work, experiment) for d, p in ecl.EVAL_GROUPS for s in ecl.EVAL_SEEDS for m in EVAL_MODES]
    pending = []
    for job in jobs:
        d, p, s, m = job[:4]
        cache = Path(args.out) / "eval" / f"{d}_{p}_s{s}_{m}" / "metrics.json"
        if cache.exists():
            metrics = _eval_job(job)   # 새 SUMO 실행 없이 코드·가중치 검증
            budget.add(f"eval_{d}_{p}_{s}_{m}", metrics["cpu_s"])
        else:
            pending.append(job)
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        for start in range(0, len(pending), args.workers):
            batch = pending[start:start + args.workers]
            budget.check(reserve=180 * len(batch))
            for job, metrics in zip(batch, pool.map(_eval_job, batch)):
                budget.add(f"eval_{job[0]}_{job[1]}_{job[2]}_{job[3]}", metrics["cpu_s"])
                print(f"평가 {job[0]} {job[1]} s{job[2]} {job[3]} 완료", flush=True)
            budget.check()


def final_comparison(rows: list[dict], reference: str, minimum: float) -> dict:
    gains, groups = [], {}
    for row in rows:
        gain = row[reference]["mean_delay_s"] - row["child"]["mean_delay_s"]
        gains.append(gain)
        groups.setdefault(row["group"], []).append(gain)
    group_means = {g: statistics.fmean(v) for g, v in groups.items()}
    mean, lo, hi = er.t_interval(list(group_means.values()))
    lefts = {n: statistics.fmean(r["child"][n] - r[reference][n] for r in rows) for n in er.LEFTS}
    long_ok = sum(r["child"]["long_waits"] for r in rows) <= sum(r[reference]["long_waits"] for r in rows)
    positive = sum(g > 0 for g in group_means.values())
    passed = mean >= minimum and lo > 0 and positive >= 3 and all(x <= 5.0 for x in lefts.values()) and long_ok
    return {"passed": passed, "reference": reference, "mean_gain_s": mean, "minimum_gain_s": minimum,
            "group_interval": [lo, hi], "group_means": group_means, "episode_interval_reference_only": er.t_interval(gains),
            "positive_groups": positive, "left_delay_increase_s": lefts, "long_waits_not_worse": long_ok}


def summary(args, experiment: str, budget: Budget):
    out = Path(args.out)
    pilot_file = out / "pilot.json"
    result = {"experiment": experiment, "cpu_s": budget.used, "development_evaluation": True}
    if not pilot_file.exists():
        result["word"] = "준비 또는 탐색 진행 중"
    else:
        pilot = read_json(pilot_file)
        result["pilot"] = pilot["reading"]
        if not pilot["reading"]["passed"]:
            result["word"] = "작은 검사 기준 미통과, 학생 개선·확대 중단"
        else:
            rows, missing = [], []
            for d, p in ecl.EVAL_GROUPS:
                for seed in ecl.EVAL_SEEDS:
                    row = {"group": f"{d}_{p}", "seed": seed}
                    for mode in EVAL_MODES:
                        f = out / "eval" / f"{d}_{p}_s{seed}_{mode}" / "metrics.json"
                        if not f.exists():
                            missing.append(str(f))
                        else:
                            row[mode] = read_json(f)
                    if all(m in row for m in EVAL_MODES):
                        rows.append(row)
            if missing:
                result.update(word="정책 개선·연속 평가 미완료", missing=missing)
            else:
                invalid = []
                for row in rows:
                    for mode in EVAL_MODES:
                        m = row[mode]
                        for key in ("missing", "unfinished", "changed_by_rules", "collisions_at_204820", "collisions_unexplained"):
                            if m[key] != 0:
                                invalid.append((row["group"], row["seed"], mode, key, m[key]))
                        if m["experiment"] != experiment or m["cohort_sha256"] != row["p0"]["cohort_sha256"]:
                            invalid.append((row["group"], mode, "다른 코드 또는 차량 코호트"))
                    original = Path(args.combo) / "eval" / f"{row['group']}_s{row['seed']}_p0" / "metrics.json"
                    if abs(read_json(original)["mean_delay_s"] - row["p0"]["mean_delay_s"]) > cb.SAME_DELAY:
                        invalid.append((row["group"], row["seed"], "P0 재현 불일치"))
                if invalid:
                    raise RuntimeError(f"연속 평가 실행 확인 실패: {invalid[:6]}")
                if pr.FrozenPolicy(out / "parent").width == 96:
                    for row in rows:
                        original = Path(args.combo) / "eval" / f"{row['group']}_s{row['seed']}_b" / "metrics.json"
                        if abs(read_json(original)["mean_delay_s"] - row["parent"]["mean_delay_s"]) > cb.SAME_DELAY:
                            raise RuntimeError(f"기존 B 부모의 연속 제어가 재현되지 않았다: {row['group']} s{row['seed']}")
                parent_gain = final_comparison(rows, "parent", 0.5)
                p0_gain = final_comparison(rows, "p0", 1.3)
                word = ("P0 초과 채택 기준 통과(개발 평가)" if p0_gain["passed"] else
                        "한 차례 정책 개선 단서(개발 평가)" if parent_gain["passed"] else "정책 개선 미확인")
                result.update(word=word, parent_comparison=parent_gain, p0_comparison=p0_gain,
                              means={m: statistics.fmean(r[m]["mean_delay_s"] for r in rows) for m in EVAL_MODES}, episodes=rows)
    write_json(out / "summary.json", result)
    text = ["# PI1 — 한 차례 정책 개선", "", f"판정: **{result['word']}**", "",
            f"누적 CPU: {budget.used:.1f}초 / 상한 {budget.limit:.0f}초", "",
            "현재 자료는 AI Hub 수요의 SUMO 개발 평가다. 실제 도로 성능 또는 새 날짜 최종 시험이 아니다."]
    if "means" in result:
        text += ["", "| 정책 | 평균 지연(초) |", "|---|---:|"] + [f"| {m} | {v:.3f} |" for m, v in result["means"].items()]
    (out / "summary.md").write_text("\n".join(text) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("episodes", "missing")}, ensure_ascii=False, indent=1), flush=True)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=("run", "history", "collect", "check", "compare", "train", "evaluate", "summary"))
    ap.add_argument("--old", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--combo", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--cpu-budget-s", type=float, default=MAX_CPU_S)
    args = ap.parse_args()
    if args.workers < 1 or args.workers > 4 or not 0 < args.cpu_budget_s <= MAX_CPU_S:
        ap.error("workers는 1~4, CPU 예산은 0 초과 36000 이하")
    experiment = register(args)
    out, work = Path(args.out), Path(args.work)
    budget = Budget(out, args.cpu_budget_s)
    command = args.command
    try:
        if command in ("run", "history"):
            cached_history = (out / "diagnostic.json").exists() and (out / "parent" / "policy.json").exists()
            budget.check(reserve=0 if cached_history else 3600)
            start_cpu = cpu_now()
            eh.diagnostic_and_parent(Path(args.old), Path(args.new), Path(args.combo), out, work / "history")
            receipt = out / "history_cpu.json"
            if not receipt.exists():
                write_json(receipt, {"cpu_s": cpu_now() - start_cpu})
            budget.add("history", read_json(receipt)["cpu_s"])
            budget.check()
        if command in ("run", "collect"):
            for date, period in TRAIN_GROUPS:
                done = out / "states" / f"{date}_{period}" / "done.json"
                budget.check(reserve=0 if done.exists() else 300)
                r = collect_group(date, period, out, work, experiment)
                budget.add(f"collect_{date}_{period}", r["cpu_s"])
                budget.check()
        if command in ("run", "check", "compare"):
            rows = unpack_states(out, work)
            smoke_check(rows, out, experiment, budget)
        if command in ("run", "compare"):
            pilot = compare(rows, out, args.workers, budget, experiment)
            if not pilot["reading"]["passed"]:
                summary(args, experiment, budget)
                return
        if command in ("run", "train"):
            budget.check(reserve=0 if (out / "child" / "policy.json").exists() else 600)
            start_cpu = cpu_now()
            train(args, experiment)
            receipt = out / "train_cpu.json"
            if not receipt.exists():
                write_json(receipt, {"cpu_s": cpu_now() - start_cpu})
            budget.add("train", read_json(receipt)["cpu_s"])
            budget.check()
        if command in ("run", "evaluate"):
            evaluate(args, experiment, budget)
        if command in ("run", "summary"):
            summary(args, experiment, budget)
    except BudgetExceeded as error:
        write_json(out / "stopped.json", {"experiment": experiment, "reason": str(error), "cpu_s": budget.used})
        summary(args, experiment, budget)
        raise SystemExit(str(error))


if __name__ == "__main__":
    main()
