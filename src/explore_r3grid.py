"""[추가 탐색] R3GRID — 라운드 3 학생의 은닉층·학습률·배치·에포크(사전 등록 results/explore/r3grid_20261105/사전등록.md).

라운드 3 recent 자료(9,199지점)로 마지막 학생만 54조합 다시 학습하고, 앞 정책(N5 + 학생 2개)에 쌓아 한 시간 SUMO로 비교한다. 새 굴림은 없다.
잠긴 파일·이전 실험 코드와 결과는 고치지 않고 가져다 쓴다. 실행 때 계획기를 부르면 멈춘다.

사용(FOCUSED_RULESET=pet1 FOCUSED_NORTH_EXT=240):
  python src/explore_r3grid.py make-config --n3 <Drive>/explore_nn3                                 (맥, SUMO 없음)
  python src/explore_r3grid.py check --n3 <…> --out <dir> --work /content/rc
  python src/explore_r3grid.py run --n3 <…> --out <Drive>/explore_tune/r3grid --work /content/rg --shard 0 --shards 4 --workers 8
  python src/explore_r3grid.py judge --n3 <…> --root <Drive>/explore_tune/r3grid --out <…/judge>
"""
from __future__ import annotations

import argparse
import functools
import itertools
import json
import multiprocessing
import os
import shutil
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
from torch import nn as tnn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import arena3_common as ac  # noqa: E402
import arena3_referee as ref  # noqa: E402
import arena3_stage2_eval as s2e  # noqa: E402
import explore_nn1 as nn  # noqa: E402
import explore_nn3 as n3  # noqa: E402
import explore_nn3r2 as r2  # noqa: E402
import explore_nn3r3 as r3  # noqa: E402
import explore_platoon1 as pt  # noqa: E402
import explore_platoon_base as pb  # noqa: E402
import explore_weight as w  # noqa: E402

EXPERIMENT = "R3GRID_V1"
CONFIG = ROOT / "config/explore_r3grid_v1.json"
PREREG = ROOT / "results/explore/r3grid_20261105/사전등록.md"
TUNE1_CONFIG = ROOT / "config/explore_tune1_v1.json"
SOURCES = ("src/explore_r3grid.py", "config/explore_r3grid_v1.json", "results/explore/r3grid_20261105/사전등록.md", "config/explore_tune1_v1.json", *r3.SOURCES)
WORLD = "P"
BASE, R3, FIXC = "BASE", "recent_s10", "FIXC"
SHARE = 0.10
NETWORKS = {"64": [64, 64], "256": [256, 256], "64x3": [64, 64, 64]}
GRID = {"network": ["64", "256", "64x3"], "learning_rate": ["3e-4", "1e-3", "3e-3"], "batch_size": [64, 256], "epochs": [10, 30, 100]}
DEFAULT = {"network": "64", "learning_rate": "1e-3", "batch_size": 64, "epochs": 30}
RULES = {"finalists": 3, "adopt_gain_s": 0.2, "models_positive": 2, "factor_tie_s": 0.3, "target_reduction_pct": 35.0}


# ----------------------------------------------------------------------------- 설정
def combo_name(c: dict) -> str:
    return f"h{c['network']}_lr{c['learning_rate']}_b{c['batch_size']}_e{c['epochs']}"


def combos(grid: dict = GRID) -> list[dict]:
    return [dict(zip(grid, values)) for values in itertools.product(*grid.values())]


def candidates(cfg: dict) -> list[str]:
    """거르기 후보: 기본값 조합을 뺀 나머지(기본값은 저장된 recent_s10을 쓴다)."""
    return [combo_name(c) for c in combos(cfg["grid"]) if combo_name(c) != combo_name(cfg["default"])]


def settings_of(c: dict, base: dict) -> dict:
    return {**base, "network": NETWORKS[c["network"]], "learning_rate": float(c["learning_rate"]), "batch_size": int(c["batch_size"]), "epochs_student": int(c["epochs"])}


def config() -> dict:
    cfg = w.read(CONFIG)
    if cfg.get("experiment") != EXPERIMENT:
        raise ValueError("R3GRID 설정이 아닙니다.")
    for key, path in (("prereg_sha256", PREREG), ("lock_sha256", ac.CONFIG), ("nn3r3_config_sha256", r3.CONFIG), ("tune1_config_sha256", TUNE1_CONFIG)):
        if cfg.get(key) != w.file_hash(path):
            raise ValueError(f"R3GRID 설정을 만든 때와 파일이 다릅니다: {Path(path).relative_to(ROOT)}")
    return cfg


def make_config(n3_root: Path) -> dict:
    """맥, SUMO 없음. recent_s10 학생의 해시·문턱, 자료 지문과 두 상황 묶음을 고정한다."""
    r3cfg = r3.config()
    n3_root = Path(n3_root)
    done = w.read(n3_root / "r3" / "students" / "done.json")
    frozen = {}
    for seed in r3cfg["training"]["seeds"]:
        folder = n3_root / "r3" / "students" / "recent" / f"seed_{seed}"
        frozen[str(seed)] = {"student_sha256": nn.model_identity(folder), "tau": float(w.read(folder / "manifest.json")["training"]["taus"][f"{SHARE:.2f}"])}
    tune1 = w.read(TUNE1_CONFIG)
    cfg = {"experiment": EXPERIMENT, "world": WORLD, "design": "PROJECT_PROGRESS.md 283·284절, results/explore/r3grid_20261105/사전등록.md", "prereg": PREREG.relative_to(ROOT).as_posix(), "prereg_sha256": w.file_hash(PREREG), "lock_sha256": w.file_hash(ac.CONFIG), "nn3r3_config_sha256": w.file_hash(r3.CONFIG), "tune1_config_sha256": w.file_hash(TUNE1_CONFIG),
           "grid": GRID, "default": DEFAULT, "networks": NETWORKS, "share": SHARE, "label": r3.LABEL, "training": r3cfg["training"], "rules": RULES, "recent": {"fingerprint": done["dataset_fingerprints"]["recent"], "points": done["points"]["recent"], "frozen": frozen}, "input": r3cfg["input"],
           "reporting": "검증 날짜의 개발 비교다. 라운드 3 학생만 바꾼다. 무리 도착은 모형이다.", "situations_screen": r3cfg["situations_screen"], "situations_final": tune1["situations_screen"]}
    CONFIG.write_text(json.dumps(w.plain(cfg), ensure_ascii=False, indent=1) + "\n")
    return {"config": str(CONFIG), "sha256": w.file_hash(CONFIG), "candidates": len(candidates(cfg)), "recent": cfg["recent"]["fingerprint"]}


# ----------------------------------------------------------------------------- 학생 신경망
class StudentNet(tnn.Module):
    """explore_nn1.GainNet과 같은 구조·같은 파라미터 이름에 은닉층 수만 자유롭다. 2층이면 같은 난수에서 같은 초기값이다."""

    def __init__(self, size: int, mean=None, std=None, hidden=(64, 64)):
        super().__init__()
        self.register_buffer("mean", torch.zeros(size) if mean is None else torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.ones(size) if std is None else torch.as_tensor(std, dtype=torch.float32))
        layers, last = [], size + nn.GREENS
        for width in hidden:
            layers += [tnn.Linear(last, int(width)), tnn.ReLU()]
            last = int(width)
        self.trunk = tnn.Sequential(*layers)
        self.advantage = tnn.Linear(last, nn.GREENS)
        tnn.init.zeros_(self.advantage.weight)
        tnn.init.zeros_(self.advantage.bias)

    def forward(self, x: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
        z = (x - self.mean) / self.std
        adv = self.advantage(self.trunk(torch.cat([z, F.one_hot(base, nn.GREENS).to(z.dtype)], dim=1)))
        return adv - adv.gather(1, base[:, None])


def load_student(folder: Path) -> StudentNet:
    """이전 라운드 학생(GainNet)도 같은 파라미터 이름이라 그대로 읽는다."""
    nn.model_identity(Path(folder))
    manifest = w.read(Path(folder) / "manifest.json")
    payload = torch.load(Path(folder) / "model.pt", map_location="cpu", weights_only=False)
    if manifest["kind"] != nn.KIND_GAIN or payload.get("seed") != manifest.get("seed") or payload.get("dataset_fingerprint") != manifest.get("dataset_fingerprint"):
        raise ValueError(f"학생 모델이 아니거나 파일과 기록이 다릅니다: {folder}")
    net = StudentNet(manifest["input_size"], hidden=tuple(manifest["network"]))
    net.load_state_dict(payload["state"])
    net.eval()
    return net


def fit_student(x: np.ndarray, base: np.ndarray, gains: np.ndarray, se: np.ndarray, allowed: np.ndarray, seed: int, st: dict) -> StudentNet:
    """explore_nn1.fit_gain과 같은 학습 절차. 신경망 모양·학습률·배치·에포크만 st에서 받는다."""
    torch.manual_seed(int(seed))
    mean, std = nn.scale_of(x)
    net = StudentNet(x.shape[1], mean, std, tuple(st["network"]))
    opt = torch.optim.Adam(net.parameters(), lr=float(st["learning_rate"]), weight_decay=float(st["weight_decay"]))
    rng = np.random.default_rng(int(seed))
    X, B = torch.from_numpy(np.asarray(x, np.float32)), torch.from_numpy(np.asarray(base, np.int64))
    G, S, M = torch.from_numpy(np.asarray(gains, np.float32)), torch.from_numpy(np.asarray(se, np.float32)), torch.from_numpy(np.asarray(allowed, bool))
    size = int(st["batch_size"])
    net.train()
    for _ in range(int(st["epochs_student"])):
        order = rng.permutation(len(X))
        for k in range(0, len(order), size):
            i = order[k:k + size]
            loss = nn.gain_loss(net, X[i], B[i], G[i], S[i], M[i], st)
            if not bool(torch.isfinite(loss)):
                raise ValueError("학생 손실이 유한하지 않습니다.")
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), float(st["grad_clip"]))
            opt.step()
    return net.eval()


# ----------------------------------------------------------------------------- 학습
@functools.lru_cache(maxsize=1)
def _dataset(path: str) -> dict:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def student_task(args: tuple) -> dict:
    """조합 하나 × 난수 하나: 날짜 보류 예측으로 문턱을 정하고, 모든 날짜로 배포 학생을 만든다(explore_nn3._student_task와 같은 절차)."""
    path, fingerprint, combo, seed, folder, base_st, label, share = args
    torch.set_num_threads(1)
    started = time.monotonic()
    d = _dataset(path)
    st = settings_of(combo, base_st)
    x, base, allowed, dates = d["x"], d["base"], d["allowed"], d["dates"]
    costs = n3.label_costs(d["delay"], d["residual"], 0, float(label["weight"]))
    front, back = nn.half_costs(costs)
    gains_f, se_f = nn.gains_against(front, base, allowed)
    predicted, best = np.zeros(len(x)), np.zeros(len(x), np.int64)
    for date in sorted(set(dates.tolist())):
        held = dates == date
        fold = fit_student(x[~held], base[~held], gains_f[~held], se_f[~held], allowed[~held], seed, st)
        best[held], predicted[held] = nn.predicted_best(fold, x[held], base[held], allowed[held])
    tau = float(np.quantile(predicted, 1.0 - float(share)))
    gains, se = nn.gains_against(costs, base, allowed)
    net = fit_student(x, base, gains, se, allowed, seed, st)
    full = nn.predicted_best(net, x, base, allowed)[1]
    info = {"mark": int(label["mark"]), "weight": float(label["weight"]), "taus": {f"{share:.2f}": tau}, "score": n3.back_half_score(best, predicted, base, back, tau), "share_in_sample": float((full >= tau).mean()), "points": int(len(x)), "seconds": time.monotonic() - started}
    name = combo_name(combo)
    meta = n3._meta(nn.KIND_GAIN, name, seed, fingerprint, st, tau=tau, label_mark=int(label["mark"]), label_end_weight=float(label["weight"]), training=info, combo=combo)
    sha = nn.save_model(Path(folder), net, meta)
    return {"combo": name, "seed": int(seed), "model_sha256": sha, **info}


def weight_difference(a: Path, b: Path) -> float:
    """두 학생의 가중치 최대 차이. 모델 파일에는 학습 시간 같은 기록도 들어 있어 파일 해시로는 비교할 수 없다."""
    sa = torch.load(Path(a) / "model.pt", map_location="cpu", weights_only=False)["state"]
    sb = torch.load(Path(b) / "model.pt", map_location="cpu", weights_only=False)["state"]
    if list(sa) != list(sb):
        raise ValueError("두 학생의 구조가 다릅니다.")
    return max(float((sa[k].double() - sb[k].double()).abs().max()) for k in sa)


def recent_dataset(n3_root: Path, models: dict, stacks: dict, r3cfg: dict, r2cfg: dict) -> dict:
    """라운드 3 recent와 같은 자료: 라운드 3 + 라운드 2(기준 녹색은 BASE로 다시 계산)."""
    new = r2.load_points(n3_root / "r3" / "data", r3cfg["situations_round"], 3, r3cfg)
    old = r2.load_points(n3_root / "r2" / "data", r2cfg["situations_round"], 2, r3cfg)
    return r2.merge_points(new, {**old, "base": r2.rebased_points(old, r2.policy_base_of(models, stacks))})


def learn(n3_root: Path, models: dict, stacks: dict, out: Path, work: Path, cfg: dict, workers: int) -> dict:
    """모든 조합·난수의 학생을 만든다. 이미 있으면 같은 자료인지 보고 건너뛴다."""
    out, work = Path(out), Path(work)
    if (out / "done.json").exists():
        return w.read(out / "done.json")
    d = recent_dataset(Path(n3_root), models, stacks, r3.config(), r2.config())
    if d["fingerprint"] != cfg["recent"]["fingerprint"] or len(d["x"]) != int(cfg["recent"]["points"]):
        raise ValueError(f"자료가 라운드 3 recent와 다릅니다: {d['fingerprint']} / {len(d['x'])}지점")
    work.mkdir(parents=True, exist_ok=True)
    path = work / "recent.npz"
    np.savez(path, **{k: d[k] for k in ("x", "base", "allowed", "dates", "delay", "residual")})
    tasks, rows = [], []
    for combo in combos(cfg["grid"]):
        for seed in cfg["training"]["seeds"]:
            folder = out / combo_name(combo) / f"seed_{seed}"
            if (folder / "manifest.json").exists():
                manifest = w.read(folder / "manifest.json")
                if manifest.get("dataset_fingerprint") != d["fingerprint"]:
                    raise ValueError(f"이미 있는 학생이 다른 자료로 학습됐습니다: {folder}")
                rows.append({"combo": combo_name(combo), "seed": int(seed), "model_sha256": nn.model_identity(folder), **manifest["training"]})
            else:
                tasks.append((str(path), d["fingerprint"], combo, seed, str(folder), cfg["training"], cfg["label"], float(cfg["share"])))
    ref.stamp(f"학생 {len(rows) + len(tasks)}개 중 {len(tasks)}개 학습")
    if tasks:
        with ProcessPoolExecutor(max_workers=max(1, workers), mp_context=multiprocessing.get_context("spawn"), initializer=ref._worker_init) as pool:
            for k, got in enumerate(pool.map(student_task, tasks, chunksize=1), 1):
                rows.append(got)
                if k % 10 == 0 or k == len(tasks):
                    ref.stamp(f"학생 {k}/{len(tasks)}개 끝")
    default = combo_name(cfg["default"])
    reproduce = {str(seed): weight_difference(out / default / f"seed_{seed}", Path(n3_root) / "r3" / "students" / "recent" / f"seed_{seed}") for seed in cfg["training"]["seeds"]}
    summary = {"points": len(d["x"]), "fingerprint": d["fingerprint"], "students": sorted(rows, key=lambda r: (r["combo"], r["seed"])), "default_weight_difference": reproduce, "cpu_hours": sum(float(r["seconds"]) for r in rows) / 3600}
    s2e._write(out / "done.json", summary)
    ref.stamp(f"학생 학습 끝. 다시 학습한 기본값과 저장 학생의 가중치 최대 차이: {reproduce}")
    return summary


# ----------------------------------------------------------------------------- 평가
def job_worker(job: dict) -> dict:
    if job.get("mode") == "fixed":
        return pb.base_worker(job)
    started = time.monotonic()
    work = Path(job["work"])
    name = f"{job['policy']}/{job['sid']}"
    try:
        w.configure()
        torch.set_num_threads(1)
        shutil.rmtree(work, ignore_errors=True)
        net = nn.load_model(Path(job["model_dir"]))[0]
        policy = n3.StackPolicy(net, job["policy"], [(load_student(Path(d)), float(t)) for d, t in job["stack"]])
        with nn.planner_free(policy, "NN") as guard:
            result = pt.run_episode(job["situation"], work, policy, max_seconds=nn.MAX_EPISODE_S)
        if result.metrics["route_sha256"] != job["situation"]["route_sha256"]:
            raise ValueError(f"차량 파일의 SHA가 설정과 다릅니다: {result.metrics['route_sha256']}")
        ac.write_run(Path(job["target"]), result, {"policy": job["policy"], "fingerprint": job["fingerprint"], "model_identity": job.get("model_identity"), "job_seconds": time.monotonic() - started, "information": policy.info(), "planner_free": True, "tail": "NN", "tail_planner_calls": guard["tail_calls"]})
        return {"ok": True, "seconds": time.monotonic() - started, "name": name}
    except BaseException as error:
        return {"ok": False, "seconds": time.monotonic() - started, "name": name, "error": f"{type(error).__name__}: {error}"}
    finally:
        shutil.rmtree(work, ignore_errors=True)


def grid_stacks(students: Path, name: str, stacks: dict, tau_key: str) -> dict:
    out = {}
    for seed, stack in stacks.items():
        folder = Path(students) / name / f"seed_{seed}"
        out[int(seed)] = [*stack, [str(folder), float(w.read(folder / "manifest.json")["training"]["taus"][tau_key])]]
    return out


def default_stacks(n3_root: Path, stacks: dict, cfg: dict) -> dict:
    """저장된 recent_s10: 해시·문턱을 설정과 대조한다."""
    out = {}
    for seed, stack in stacks.items():
        folder = Path(n3_root) / "r3" / "students" / "recent" / f"seed_{seed}"
        frozen = cfg["recent"]["frozen"][str(seed)]
        if nn.model_identity(folder) != frozen["student_sha256"]:
            raise ValueError(f"recent_s10 학생이 설정에 고정한 것과 다릅니다: 난수 {seed}")
        out[int(seed)] = [*stack, [str(folder), float(frozen["tau"])]]
    return out


def run_stage(stage: str, jobs: list[dict], out: Path, shard: int, shards: int, workers: int, ledger, worker=job_worker) -> bool:
    mine = n3.shard_jobs(jobs, shard, shards)
    path = n3._stage_status(out, stage, shard)
    if path.exists() and w.read(path).get("jobs") == len(mine):
        return True
    status = ref.run_jobs(mine, max(1, workers), ledger, worker=worker)
    if status["failed"]:
        ref.stamp(f"{stage}: 실패 {len(status['failed'])}판. 다시 실행하면 끝난 판은 건너뜁니다.")
        return False
    n3.write_status(path, jobs=len(mine), rows=n3.job_rows(mine))
    return True


def finalists(means: dict[str, float], count: int) -> list[str]:
    """짧은 지연 → 이름순."""
    return sorted(means, key=lambda n: (means[n], n))[:int(count)]


def selection(out: Path, shards: int, cfg: dict) -> dict:
    sids = [ac.situation_id(s) for s in cfg["situations_screen"]]
    means = n3.candidate_means(n3.stage_rows(out, "screen", shards), candidates(cfg), sids)
    return {"means": means, "finalists": finalists(means, cfg["rules"]["finalists"])}


def run(n3_root: Path, out: Path, work: Path, shard: int, shards: int, workers: int, cfg: dict, *, every_s: float = 60.0) -> int:
    """학습 → 거르기 → 확인. 런타임 사이는 Drive의 상태 파일로 맞춘다. 다시 실행하면 끝난 부분은 건너뛴다."""
    n3_root, out, work = Path(n3_root), Path(out), Path(work)
    r3cfg = r3.config()
    models, stacks = r3.base_policy(n3_root / "prep", n3_root / "r1" / "students", n3_root / "r2" / "students", r3cfg)
    students, evaluation = out / "students", out / "eval"
    nn.prepare_caches(WORLD)
    if shard == 0:
        learn(n3_root, models, stacks, students, work / "learn", cfg, workers)
    n3.wait_for([students / "done.json"], "학생", every_s)
    if shard != 0:
        n3.wait_for([evaluation / "experiment.json"], "평가 지문", every_s)
    fp_ = ref.register(evaluation, extra_files=SOURCES)
    ledger = ref.Ledger(evaluation, f"shard{shard}")
    key = f"{float(cfg['share']):.2f}"
    screen, final = cfg["situations_screen"], cfg["situations_final"]
    jobs = []
    for name in candidates(cfg):
        jobs += n3.nn_jobs(evaluation, work / "eval", fp_, name, models, screen, stacks=grid_stacks(students, name, stacks, key))
    if not run_stage("screen", jobs, out, shard, shards, workers, ledger):
        return 1
    n3.wait_for([n3._stage_status(out, "screen", s) for s in range(shards)], "거르기", every_s)
    chosen = selection(out, shards, cfg)
    ref.stamp(f"거르기 끝. 확인 후보: {chosen['finalists']}")
    jobs = [*n3.nn_jobs(evaluation, work / "eval", fp_, BASE, models, final, stacks=stacks), *n3.nn_jobs(evaluation, work / "eval", fp_, R3, models, final, stacks=default_stacks(n3_root, stacks, cfg)), *r2.fixed_jobs(evaluation, work / "eval", fp_, final)]
    for name in chosen["finalists"]:
        jobs += n3.nn_jobs(evaluation, work / "eval", fp_, name, models, final, stacks=grid_stacks(students, name, stacks, key))
    if not run_stage("final", jobs, out, shard, shards, workers, ledger):
        return 1
    ref.stamp("이 런타임의 확인 끝. 모든 런타임이 끝나면 R3GRID_judge를 실행하세요.")
    return 0


# ----------------------------------------------------------------------------- 판정
def factor_means(means: dict[str, float], cfg: dict) -> dict:
    """변수 값별 평균 지연. means는 54칸 전부(기본값 칸은 저장된 recent_s10 값)라 값마다 칸 수가 같다."""
    rows = combos(cfg["grid"])
    out = {}
    for factor, values in cfg["grid"].items():
        got = {str(value): statistics.fmean(means[combo_name(c)] for c in rows if c[factor] == value) for value in values}
        spread = max(got.values()) - min(got.values())
        out[factor] = {"means": got, "spread_s": spread, "distinguished": spread >= float(cfg["rules"]["factor_tie_s"])}
    return out


def adoption(gain: dict, models: list[float], rules: dict) -> bool:
    return gain["t95"][0] is not None and gain["gain_s"] >= float(rules["adopt_gain_s"]) and gain["t95"][0] > 0.0 and sum(x > 0 for x in models) >= int(rules["models_positive"])


def model_gains(base: dict, cand: dict) -> list[float]:
    models = len(next(iter(base.values())))
    return [n3.weighted_mean([v[i] for v in base.values()]) - n3.weighted_mean([v[i] for v in cand.values()]) for i in range(models)]


def collect_report(n3_root: Path, root: Path, cfg: dict) -> dict:
    root, n3_root = Path(root), Path(n3_root)
    report = {"experiment": EXPERIMENT}
    if not (root / "students" / "done.json").exists():
        return {**report, "status": "before_students"}
    done = w.read(root / "students" / "done.json")
    report["students"] = {"points": done["points"], "cpu_hours": done["cpu_hours"], "default_weight_difference": done["default_weight_difference"], "scores": {f"{r['combo']}|{r['seed']}": r["score"] for r in done["students"]}}
    screen, final = cfg["situations_screen"], cfg["situations_final"]
    screen_ids, final_ids = [ac.situation_id(s) for s in screen], [ac.situation_id(s) for s in final]
    names = candidates(cfg)
    try:
        rows = n3.read_stage_runs(root / "eval", names, screen)
        stored = n3.read_stage_runs(n3_root / "r3" / "eval", [BASE, R3], screen)
        means = n3.candidate_means(rows, names, screen_ids)
        best = finalists(means, cfg["rules"]["finalists"])
        all_final = [BASE, R3, *best]
        last = n3.read_stage_runs(root / "eval", all_final, final)
        fixed = r2.read_fixed(root / "eval", final)[FIXC]
    except FileNotFoundError as error:
        return {**report, "status": "incomplete", "missing": str(error)}
    stored_r3 = n3._by_situation(stored, R3, screen_ids)
    report["screen"] = {n: {"mean_delay_s": v, "gain_vs_r3": n3.paired_gain(stored_r3, n3._by_situation(rows, n, screen_ids)), "change_share": n3.policy_summary(n3._by_situation(rows, n, screen_ids))["change_share"]} for n, v in sorted(means.items(), key=lambda kv: (kv[1], kv[0]))}
    report["screen_r3_s"] = n3.candidate_means(stored, [R3], screen_ids)[R3]
    report["screen_base_s"] = n3.candidate_means(stored, [BASE], screen_ids)[BASE]
    report["factors"] = factor_means({**means, combo_name(cfg["default"]): report["screen_r3_s"]}, cfg)
    report["finalists"] = best
    by = {n: n3._by_situation(last, n, final_ids) for n in all_final}
    fixed_rows = [fixed[s] for s in final_ids]
    fixed_mean = n3.weighted_mean(fixed_rows)
    report["final"] = {"fixed": {"mean_delay_s": fixed_mean, "left_east_s": statistics.fmean(r["left_east"] for r in fixed_rows if r["left_east"] is not None), "left_south_s": statistics.fmean(r["left_south"] for r in fixed_rows if r["left_south"] is not None), "long_per_run": sum(r["long"] for r in fixed_rows) / len(fixed_rows)}, "policies": {}}
    for n in all_final:
        for s in final_ids:
            if {r["vehicles"] for r in by[n][s]} != {fixed[s]["vehicles"]}:
                raise ValueError(f"고정신호와 차 수가 다릅니다: {n}/{s}")
        summary = n3.policy_summary(by[n])
        row = {"summary": summary, "percent_vs_fixed": r2.percent(summary["mean_delay_s"], fixed_mean)}
        if n != BASE:
            row["gain_vs_base"] = n3.paired_gain(by[BASE], by[n])
        if n not in (BASE, R3):
            row.update(gain_vs_r3=n3.paired_gain(by[R3], by[n]), models_vs_r3=model_gains(by[R3], by[n]), per_date=n3.part_gains(by[R3], by[n], [(d, lambda k, x=d: k.startswith(x)) for d in n3.VALID_DATES]), per_batch=n3.part_gains(by[R3], by[n], [(b, lambda k, x=b: k.endswith(x)) for b in ("VU1", "VU2")]),
                       per_band=n3.part_gains(by[R3], by[n], [(a, lambda k, lo=lo, hi=hi: lo <= int(k.split("_")[1]) < hi) for a, lo, hi in (("아침 06~08", 600, 900), ("낮 09~15", 900, 1600), ("저녁 16~18", 1600, 1900), ("밤 19~21", 1900, 2400))]))
        report["final"]["policies"][n] = row
    winner = max(best, key=lambda n: (report["final"]["policies"][n]["gain_vs_r3"]["gain_s"], n))
    w_row = report["final"]["policies"][winner]
    report["reading"] = {"winner": winner, "gain": w_row["gain_vs_r3"], "models": w_row["models_vs_r3"], "adopted": adoption(w_row["gain_vs_r3"], w_row["models_vs_r3"], cfg["rules"]), "percent_vs_fixed": {n: report["final"]["policies"][n]["percent_vs_fixed"] for n in (BASE, R3, winner)}, "reaches_target_here": -w_row["percent_vs_fixed"] >= float(cfg["rules"]["target_reduction_pct"])}
    return {**report, "status": "done"}


gain_text = n3.gain_text


def render(report: dict) -> str:
    lines = ["# R3GRID 결과 — 라운드 3 학생 54조합(무리 세계, 검증 날짜 개발 비교)", ""]
    if report["status"] != "done":
        return "\n".join([*lines, f"아직 끝나지 않았습니다: {report.get('missing', report['status'])}"]) + "\n"
    st = report["students"]
    lines += [f"자료 {st['points']}지점, 학생 학습 {st['cpu_hours']:.1f} CPU-시간. 다시 학습한 기본값과 저장 학생의 가중치 최대 차이: {st['default_weight_difference']}.", "", f"## 거르기(VT 24상황 × 모델 3개, 상위 10개). recent_s10 {report['screen_r3_s']:.2f}초, BASE {report['screen_base_s']:.2f}초", "", "| 조합 | 지연 | recent_s10 대비 [t95] | N5와 다른 결정 |", "|---|---:|---|---:|"]
    for name, row in list(report["screen"].items())[:10]:
        lines.append(f"| {name} | {row['mean_delay_s']:.2f} | {gain_text(row['gain_vs_r3'])} | {row['change_share']:.1%} |")
    lines += ["", "## 변수 값별 평균 지연(거르기 54칸, 기본값은 저장 결과)", "", "| 변수 | 값별 평균 | 폭 | 구분 |", "|---|---|---:|---|"]
    for factor, row in report["factors"].items():
        values = ", ".join(f"{k} {v:.2f}" for k, v in row["means"].items())
        lines.append(f"| {factor} | {values} | {row['spread_s']:.2f} | {'폭 0.3초 이상' if row['distinguished'] else '구분 안 됨'} |")
    fx = report["final"]["fixed"]
    lines += ["", f"## 확인(VU 84상황 × 모델 3개). 고정신호 {fx['mean_delay_s']:.2f}초, 동 좌회전 {fx['left_east_s']:.1f}, 남 좌회전 {fx['left_south_s']:.1f}, 300초 초과 판당 {fx['long_per_run']:.1f}", "", "| 정책 | 지연 | 고정신호 대비 | recent_s10 대비 [t95] | BASE 대비 [t95] | 모델별 | 앞선 상황 | 동 좌회전 | 남 좌회전 | 300초 초과(판당) | 미확인 충돌 |", "|---|---:|---:|---|---|---|---:|---:|---:|---:|---:|"]
    for name, row in report["final"]["policies"].items():
        m = row["summary"]
        g = row.get("gain_vs_r3")
        models = ", ".join(f"{x:+.2f}" for x in row["models_vs_r3"]) if g else ", ".join(f"{x:.2f}" for x in m["per_model_s"])
        lines.append(f"| {name} | {m['mean_delay_s']:.2f} | {row['percent_vs_fixed']:+.1f}% | {gain_text(g) if g else '-'} | {gain_text(row['gain_vs_base']) if 'gain_vs_base' in row else '-'} | {models} | {str(g['positive']) + '/' + str(g['n']) if g else '-'} | {m['left_east_s']:.1f} | {m['left_south_s']:.1f} | {m['long_per_run']:.2f} | {m['collisions_unexplained']} |")
    r = report["reading"]
    lines += ["", f"**읽기:** {r['winner']} − recent_s10 {gain_text(r['gain'])}초, 모델 {', '.join(f'{x:+.2f}' for x in r['models'])} → **{'채택' if r['adopted'] else '미채택'}** (기준: +0.2초 이상, t95 하한 > 0, 모델 2개 이상 양수).", "",
              f"- 고정신호 대비: BASE {r['percent_vs_fixed'][BASE]:+.1f}%, recent_s10 {r['percent_vs_fixed'][R3]:+.1f}%, {r['winner']} {r['percent_vs_fixed'][r['winner']]:+.1f}%. 이 상황에서 −35% 도달: {r['reaches_target_here']}.", "- 검증 날짜의 개발 비교다. 라운드 3 학생만 바꿨다. 54조합에서 골라 낙관이 들어 있을 수 있다. 무리 도착은 모형이다."]
    return "\n".join(lines) + "\n"


def save_report(out: Path, report: dict) -> None:
    s2e._write(Path(out) / "judge.json", w.plain(report))
    (Path(out) / "summary.md").write_text(render(report), encoding="utf-8")


# ----------------------------------------------------------------------------- 점검(맥·Colab에서 같은 코드)
SAME_KEYS = ("executed_sha16", "route_sha256", "actual_total_delay")


def check(n3_root: Path, out: Path, work: Path, cfg: dict) -> dict:
    """(1) BASE·recent_s10이 라운드 3 저장 결과와 같다 (2) 이득 0인 3층 학생은 신호를 안 바꾼다 (3) 값이 있는 3층 학생은 바꾼다 (4) 계획기 0, 품질 오류 0 (5) 고정신호 한 판."""
    n3_root, out, work = Path(n3_root), Path(out), Path(work)
    shutil.rmtree(work, ignore_errors=True)
    nn.prepare_caches(WORLD)
    r3cfg = r3.config()
    models, stacks = r3.base_policy(n3_root / "prep", n3_root / "r1" / "students", n3_root / "r2" / "students", r3cfg)
    one = {42: models[42]}
    s = cfg["situations_screen"][0]
    sid = ac.situation_id(s)
    stored = {name: w.read(n3_root / "r3" / "eval" / "runs" / f"{name}_m0" / sid / "metrics.json") for name in (BASE, R3)}

    def play(name: str, stack: list) -> dict:
        job = n3.nn_jobs(work / "out", work / "run", "check", name, one, [s], stacks={42: stack})[0]
        got = job_worker(job)
        if not got["ok"]:
            raise RuntimeError(f"{name}: {got['error']}")
        return w.read(Path(job["target"]) / "metrics.json")

    base = play(BASE, stacks[42])
    default = play(R3, default_stacks(n3_root, {42: stacks[42]}, cfg)[42])
    size = n3.INPUT_SIZE
    zero, active = StudentNet(size, hidden=(64, 64, 64)), StudentNet(size, hidden=(64, 64, 64))
    torch.manual_seed(7)
    tnn.init.normal_(active.advantage.weight, std=1.0)
    meta = {"world": WORLD, "seed": 42, "input_size": size, "network": [64, 64, 64], "dataset_fingerprint": "check", "training": {"taus": {"0.10": 0.0}}}
    zero_dir, active_dir = work / "students" / "zero", work / "students" / "active"
    nn.save_model(zero_dir, zero, {**meta, "kind": nn.KIND_GAIN})
    nn.save_model(active_dir, active, {**meta, "kind": nn.KIND_GAIN})
    with_zero = play("zero", [*stacks[42], [str(zero_dir), 0.5]])
    with_active = play("active", [*stacks[42], [str(active_dir), 0.0]])
    fixed = job_worker(r2.fixed_jobs(work / "fixed_out", work / "fixed", "check", cfg["situations_final"][:1])[0])
    runs = (base, default, with_zero, with_active)
    report = {"same_as_stored": {name: {k: stored[name][k] == m[k] for k in SAME_KEYS} for name, m in ((BASE, base), (R3, default))}, "zero_same_as_base": all(with_zero[k] == base[k] for k in SAME_KEYS), "active_changes": with_active["executed_sha16"] != base["executed_sha16"], "active_changed_decisions": (with_active.get("information") or {}).get("student_changed"),
              "planner_calls": sum(int(m.get("tail_planner_calls") or 0) for m in runs), "quality_errors": [e for m in runs for e in ref.quality_errors(m)], "seconds": [m.get("job_seconds") for m in runs], "fixed_ok": fixed["ok"], "fixed_error": fixed.get("error"), "delays": {"BASE": base["mean_delay_s"], R3: default["mean_delay_s"], "zero": with_zero["mean_delay_s"], "active": with_active["mean_delay_s"]}}
    report["ok"] = bool(all(all(v.values()) for v in report["same_as_stored"].values()) and report["zero_same_as_base"] and report["active_changes"] and report["planner_calls"] == 0 and not report["quality_errors"] and fixed["ok"])
    s2e._write(out / "check.json", w.plain(report))
    shutil.rmtree(work, ignore_errors=True)
    return report


# ----------------------------------------------------------------------------- 명령
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("make-config")
    p.add_argument("--n3", required=True)
    p = sub.add_parser("check")
    p.add_argument("--n3", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--work", required=True)
    p = sub.add_parser("run")
    p.add_argument("--n3", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--work", required=True)
    p.add_argument("--shard", type=int, required=True)
    p.add_argument("--shards", type=int, default=4)
    p.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1))
    p = sub.add_parser("judge")
    p.add_argument("--n3", required=True)
    p.add_argument("--root", required=True)
    p.add_argument("--out", required=True)
    args = parser.parse_args()
    w.configure()
    if args.cmd == "make-config":
        print(json.dumps(make_config(Path(args.n3)), ensure_ascii=False, indent=1))
        return 0
    cfg = config()
    if args.cmd == "check":
        report = check(Path(args.n3), Path(args.out), Path(args.work), cfg)
        print(json.dumps(w.plain(report), ensure_ascii=False, indent=1))
        return 0 if report["ok"] else 1
    if args.cmd == "run":
        return run(Path(args.n3), Path(args.out), Path(args.work), args.shard, args.shards, args.workers, cfg)
    report = collect_report(Path(args.n3), Path(args.root), cfg)
    save_report(Path(args.out), report)
    print(render(report))
    return 0 if report["status"] == "done" else 3


if __name__ == "__main__":
    raise SystemExit(main())
