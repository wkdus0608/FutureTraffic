"""[추가 탐색] NN3R2 — 신경망 강화학습 라운드 2(계획서 3A단계, 사전 등록 results/explore/nn3r2_20261104/사전등록.md).

라운드 1 정책(N5 + 학생)이 직접 운전한 판의 25% 지점에서 허용 녹색을 모두 같은 표본 미래로 300초 굴린다(뒤도 그 정책이 잇는다)
  → 자료 세 가지(half 60판, new 120판, pool 180판)로 학생을 학습 → 한 시간 SUMO로 거르고 확인한다. 예측 이득이 문턱 이상일 때만 바꾼다.
같은 84상황에서 N5와 고정신호도 함께 재서 라운드 1 재현, 누적 몫, 고정신호 대비 %를 직접 낸다.
잠긴 파일·이전 실험 코드와 결과는 고치지 않고 가져다 쓴다. 실행 때 계획기를 부르면 멈춘다.

사용(FOCUSED_RULESET=pet1 FOCUSED_NORTH_EXT=240):
  python src/explore_nn3r2.py make-config --prep <N3>/prep --students <N3>/r1/students --workers 8   (맥, SUMO 없음)
  python src/explore_nn3r2.py check --prep <N3>/prep --r1-students <N3>/r1/students --out <dir> --work /content/n3c --workers 8
  python src/explore_nn3r2.py round --prep <N3>/prep --r1 <N3>/r1 --out <N3>/r2 --work /content/n3r --shard 0 --shards 4 --workers 8
  python src/explore_nn3r2.py judge --r2 <N3>/r2 --out <N3>/r2/judge
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import shutil
import statistics
import sys
import tempfile
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, wait
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import arena3_common as ac  # noqa: E402
import arena3_referee as ref  # noqa: E402
import arena3_stage2_eval as s2e  # noqa: E402
import explore_nn1 as nn  # noqa: E402
import explore_nn3 as n3  # noqa: E402
import explore_platoon0 as p0  # noqa: E402
import explore_platoon1 as pt  # noqa: E402
import explore_platoon_base as pb  # noqa: E402
import explore_weight as w  # noqa: E402

EXPERIMENT = "NN3R2_V1"
ROUND = 2
CONFIG = ROOT / "config/explore_nn3r2_v1.json"
PREREG = ROOT / "results/explore/nn3r2_20261104/사전등록.md"
REFERENCE = ROOT / "results/explore/nn3r2_20261104/smoke_reference/check.json"
SOURCES = ("src/explore_nn3r2.py", "config/explore_nn3r2_v1.json", "results/explore/nn3r2_20261104/사전등록.md", *n3.SOURCES)
WORLD = n3.WORLD
LABEL = {"mark": 300, "weight": 25.0, "name": "h300w25"}
ROUND1 = {"label": "h300w25", "share": 0.05, "winner": "h300w25s05"}
BAND_COUNTS = {"am_peak": 4, "pm_peak": 4, "morning": 2, "noon": 2, "afternoon": 2, "evening": 2, "dawn": 2, "late": 2}
ROLLOUT = {**n3.ROLLOUT, "horizon_s": 300, "marks": [30, 60, 150, 300], "long_fraction": 0.0, "future_salt": "NN3R2_FUTURE_SEED", "long_salt": "NN3R2_LONG_POINTS", "max_backlog_jobs": 1500}
POINTS = {"fraction": 0.25, "salt": "NN3_R2_POINTS"}
LEARN = {"label_marks": [LABEL["mark"]], "label_weights": [LABEL["weight"]], "shares": [0.03, 0.05, 0.10], "data": ["half", "new", "pool"], "finalists": 2, "curve_share": 0.05}
RULES = {"adopt_gain_s": 0.2}
BASE, N5, FIXC = "BASE", "N5", "FIXC"
DATA_SIZE = {"half": 0, "new": 1, "pool": 2}


# ----------------------------------------------------------------------------- 설정
def config() -> dict:
    cfg = w.read(CONFIG)
    if cfg.get("experiment") != EXPERIMENT:
        raise ValueError("NN3R2 설정이 아닙니다.")
    for key, path in (("prereg_sha256", PREREG), ("lock_sha256", ac.CONFIG), ("nn3_config_sha256", n3.CONFIG)):
        if cfg.get(key) != w.file_hash(path):
            raise ValueError(f"NN3R2 설정을 만든 때와 파일이 다릅니다: {Path(path).relative_to(ROOT)}")
    return cfg


def round_situations() -> list[dict]:
    """학습 6일 × 날짜당 20칸(라운드 1 묶음의 두 배), 새 도착 배치. 칸은 결과와 무관한 해시 규칙."""
    slots = {**{k: slot for k, (_, slot) in nn.AUG_BANDS.items()}, **n3.LOW_SLOTS}
    if set(slots) != set(BAND_COUNTS) or sum(BAND_COUNTS.values()) != 20:
        raise ValueError("날짜당 칸이 20개가 아닙니다.")
    out, index = [], 0
    for date in n3.TRAIN_DATES:
        for band, count in BAND_COUNTS.items():
            for period in n3._picked(slots[band], f"r{ROUND}|{date}|{band}", count):
                index += 1
                out.append({**ac.make_situation(date, period, f"N3R{ROUND}x{index}"), "round": ROUND, "band": band})
    return out


def screen_situations(n3cfg: dict) -> list[dict]:
    """NN1 검증 칸과 같은 시간대, 새 배치."""
    periods = sorted({s["period"] for s in n3.day_validation(n3cfg)})
    return [ac.make_situation(d, p, f"VS{j}") for d in n3.VALID_DATES for p in periods for j in (1, 2)]


def final_situations() -> list[dict]:
    return [ac.make_situation(d, p, f"VG{j}") for d in n3.VALID_DATES for p in n3.FINAL_PERIODS for j in (1, 2)]


def low_situations() -> list[dict]:
    return [ac.make_situation(d, p, f"VH{j}") for d in n3.VALID_DATES for p in n3.LOW_VALID_PERIODS for j in (1, 2)]


def half_ids(cfg: dict) -> list[str]:
    """half 자료 = 새 자료 중 판 번호가 짝수인 판. 묶음마다 칸 수가 짝수라 날짜·묶음·운전자가 고르게 절반이 된다."""
    return [ac.situation_id(s) for i, s in enumerate(cfg["situations_round"]) if i % 2 == 0]


def make_config(prep: Path, students: Path, workers: int) -> dict:
    """맥, SUMO 없음. 앞 정책의 해시·문턱과 새 상황의 무리 세계 차량 파일 SHA를 고정한다."""
    n3cfg = n3.config()
    start = w.read(Path(prep) / "start.json")
    if start["start"] != "N5":
        raise ValueError("라운드 1 출발 모델이 N5가 아닙니다.")
    frozen = {}
    for seed in n3cfg["training"]["seeds"]:
        folder = Path(students) / ROUND1["label"] / f"seed_{seed}"
        frozen[str(seed)] = {"student_sha256": nn.model_identity(folder), "tau": float(w.read(folder / "manifest.json")["training"]["taus"][f"{ROUND1['share']:.2f}"]), "n5_sha256": start["identities"][str(seed)]}
    data, screen, final, low = round_situations(), screen_situations(n3cfg), final_situations(), low_situations()
    rows = [*data, *screen, *final, *low]
    if len({ac.situation_id(s) for s in rows}) != len(rows):
        raise ValueError("새 상황 이름이 겹칩니다.")
    previous = w.read(CONFIG) if CONFIG.exists() else {}
    keep = {ac.situation_id(s): s["route_sha256"] for part in ("situations_round", "situations_screen", "situations_final", "situations_low") for s in previous.get(part, [])}
    need = [s for s in rows if ac.situation_id(s) not in keep]
    if need:
        with tempfile.TemporaryDirectory() as d, p0._pool(workers) as pool:
            for sid, _, sha in pool.map(p0._sha_job, [(s, WORLD, d) for s in need]):
                keep[sid] = sha
    for s in rows:
        s["route_sha256"] = keep[ac.situation_id(s)]
    lock, nn1, pcfg = ac.config(), nn.config(), pt.config()
    used = {x["route_sha256"] for part in ("eval", "confirmation", "bprime_data") for x in lock[f"situations_{part}"]}
    used |= {s["route_sha256"] for part in ("situations_data", "situations_eval", "situations_confirmation") for s in pcfg[part]}
    used |= {s["route_sha256"]["P"] for part in ("situations_aug", "situations_validation") for s in nn1[part]}
    used |= {s["route_sha256"] for part in ("situations_low_logs", "situations_low_validation", "situations_round", "situations_final") for s in n3cfg[part]}
    new = [s["route_sha256"] for s in rows]
    if len(set(new)) != len(new) or set(new) & used:
        raise ValueError("새 차량 파일이 서로 같거나 기존 차량 파일과 같습니다.")
    cfg = {"experiment": EXPERIMENT, "round": ROUND, "world": WORLD, "design": "docs/신경망_강화학습_계획서_20261102.md 3A단계, PROJECT_PROGRESS.md 275절, results/explore/nn3r2_20261104/사전등록.md", "prereg": PREREG.relative_to(ROOT).as_posix(),
           "prereg_sha256": w.file_hash(PREREG), "lock_sha256": w.file_hash(ac.CONFIG), "nn3_config_sha256": w.file_hash(n3.CONFIG), "base": {"round1": ROUND1, "frozen": frozen}, "training": n3cfg["training"], "points": POINTS, "rollout": ROLLOUT, "learn": LEARN, "rules": RULES, "input": n3cfg["input"],
           "reporting": "검증 날짜의 개발 비교다. 강화학습의 몫은 라운드 정책 − 앞 정책으로만 말한다. 무리 도착은 모형이다.", "situations_round": data, "situations_screen": screen, "situations_final": final, "situations_low": low}
    CONFIG.write_text(json.dumps(w.plain(cfg), ensure_ascii=False, indent=1) + "\n")
    return {"config": str(CONFIG), "sha256": w.file_hash(CONFIG), "situations": len(rows), "new_sha": len(need), "frozen": frozen}


def base_policy(prep: Path, students: Path, cfg: dict) -> tuple[dict[int, Path], dict[int, list]]:
    """N5 폴더와 라운드 1 학생 [(폴더, 문턱)]을 해시로 대조해 돌려준다."""
    models, stacks = {}, {}
    for key, frozen in cfg["base"]["frozen"].items():
        seed = int(key)
        model, student = Path(prep) / "start" / f"seed_{seed}", Path(students) / ROUND1["label"] / f"seed_{seed}"
        if nn.model_identity(model) != frozen["n5_sha256"] or nn.model_identity(student) != frozen["student_sha256"]:
            raise ValueError(f"앞 정책 모델이 설정에 고정한 것과 다릅니다: 난수 {seed}")
        models[seed], stacks[seed] = model, [[str(student), float(frozen["tau"])]]
    return models, stacks


# ----------------------------------------------------------------------------- 후보(순수 계산)
def candidate_name(data: str, share: float) -> str:
    return f"{data}_s{int(round(share * 100)):02d}"


def parse_candidate(name: str) -> tuple[str, float]:
    data, share = name.split("_s")
    return data, int(share) / 100.0


def screen_candidates(cfg: dict) -> list[str]:
    learn = cfg["learn"]
    return [*(candidate_name(d, s) for d in ("new", "pool") for s in learn["shares"]), candidate_name("half", learn["curve_share"])]


def curve_candidates(cfg: dict) -> list[str]:
    return [candidate_name(d, cfg["learn"]["curve_share"]) for d in cfg["learn"]["data"]]


def rank_key(name: str, delay: float) -> tuple:
    """짧은 지연 → 작은 비율 → 많은 자료."""
    data, share = parse_candidate(name)
    return (delay, share, -DATA_SIZE[data])


def finalists(means: dict[str, float], cfg: dict) -> list[str]:
    usable = {k: v for k, v in means.items() if parse_candidate(k)[0] != "half"}
    return sorted(usable, key=lambda n: rank_key(n, usable[n]))[:int(cfg["learn"]["finalists"])]


def final_names(best: list[str], cfg: dict) -> list[str]:
    """확인에 올릴 정책. 순서는 런타임마다 같아야 한다."""
    return [BASE, N5, *sorted({*best, *curve_candidates(cfg)})]


def rebased_points(d: dict, base_of) -> np.ndarray:
    """라운드 1 지점의 기준 녹색을 그 판 운전 난수의 앞 정책이 고를 녹색으로 바꾼다. base_of(난수, x, 허용 목록) → 녹색."""
    out = np.zeros(len(d["x"]), np.int64)
    for i in range(len(out)):
        allowed = np.flatnonzero(d["allowed"][i]).astype(int).tolist()
        green = int(base_of(int(d["drivers"][i]), d["x"][i], allowed))
        if green not in allowed:
            raise ValueError("앞 정책이 허용 밖 녹색을 골랐습니다.")
        out[i] = green
    return out


def merge_points(a: dict, b: dict) -> dict:
    out = {k: np.concatenate([a[k], b[k]]) for k in ("x", "allowed", "base", "dates", "delay", "residual", "drivers")}
    out["fingerprint"] = w.digest({"a": a["fingerprint"], "b": b["fingerprint"]})
    return out


def subset_points(d: dict, keep: np.ndarray, tag: str) -> dict:
    out = {k: d[k][keep] for k in ("x", "allowed", "base", "dates", "delay", "residual", "drivers")}
    out["fingerprint"] = w.digest({"a": d["fingerprint"], "subset": tag})
    return out


# ----------------------------------------------------------------------------- 자료
def data_identity(model: Path, stack: list) -> str:
    return "|".join([nn.model_identity(model), *(f"{nn.model_identity(Path(d))}|tau={float(t):.6f}" for d, t in stack)])


def finalize(out: Path, fp_: str, item: dict) -> dict:
    policy, result = item["policy"], item["result"]
    rc = policy.finish()
    target = Path(out) / "runs" / "DATA" / item["sid"]
    sha = pt.write_points(target, policy.points)
    result.metrics.update({"nn3_round": ROUND, "nn3_driver_seed": item["seed"], "nn3_points": len(policy.points), "nn3_counts": dict(policy.counts), "nn3_restore_check": {k: rc[k] for k in ("checked", "mismatched")},
                           "nn3_rollouts": rc["rollouts"], "nn3_rollout_failures": rc["rollout_failures"], "nn3_rollout_cpu_seconds": rc["rollout_cpu_seconds"], "nn3_main_seconds": item["main_seconds"], "nn3_east_future": rc["east_future"], "information": policy.info()})
    ac.write_run(target, result, {"policy": "DATA", "fingerprint": fp_, "model_identity": item["identity"], "points_sha256": sha, "role": "data"})
    return {**rc, "cpu_seconds": rc["rollout_cpu_seconds"] + item["main_seconds"], "points": len(policy.points)}


def collect(models: dict[int, Path], stacks: dict[int, list], cfg: dict, out: Path, work: Path, shard: int, shards: int, workers: int, fp_: str, *, executor=None) -> int:
    out, work = Path(out), Path(work)
    r = cfg["rollout"]
    seeds = sorted(models)
    ledger = ref.Ledger(out, f"collect_shard{shard}")
    status_path = out / "status" / f"collect_shard{shard}.json"
    rc_cfg = r["restore_check"]
    mine = [(i, s) for i, s in enumerate(cfg["situations_round"]) if i % shards == shard]
    checked = mismatched = 0
    failed, inflight = [], []
    stopped = False

    def status():
        done = sum((out / "runs" / "DATA" / ac.situation_id(s) / "cache.json").exists() for _, s in mine)
        n3.write_status(status_path, done=done, total=len(mine), failed=failed, restore_checked=checked, restore_mismatched=mismatched, inflight=[i["sid"] for i in inflight], stopped_by_restore_rule=stopped)

    def harvest():
        nonlocal checked, mismatched
        for item in [i for i in inflight if i["policy"].outstanding() == 0]:
            inflight.remove(item)
            try:
                got = finalize(out, fp_, item)
                checked += got["checked"]
                mismatched += got["mismatched"]
                ledger.add(got["cpu_seconds"])
                ref.stamp(f"자료 {item['sid']}: 지연 {item['result'].metrics['mean_delay_s']:.3f}초, 지점 {got['points']}개, 복원 불일치 {got['mismatched']}/{got['checked']}, 굴림 실패 {got['rollout_failures']}, {got['cpu_seconds'] / 3600:.2f} CPU-시간")
            except BaseException as error:
                failed.append({"situation": item["sid"], "error": f"{type(error).__name__}: {error}"})
                ref.stamp(f"자료 실패 {item['sid']}: {failed[-1]['error']}")
                ledger.add(item["main_seconds"] + item["policy"].spent())
                if isinstance(error, KeyboardInterrupt):
                    raise
            finally:
                shutil.rmtree(work / item["sid"], ignore_errors=True)
            status()

    def backlog() -> int:
        return sum(i["policy"].outstanding() for i in inflight)

    def wait_some():
        pending = [f for i in inflight for f in i["policy"].futures_list() if not f.done()]
        if pending:
            wait(pending, timeout=30, return_when=FIRST_COMPLETED)

    pool = executor or pt.rollout_pool(workers)
    with pool:
        for i, s in mine:
            sid = ac.situation_id(s)
            seed = seeds[i % len(seeds)]
            model, stack = models[seed], stacks[seed]
            identity = data_identity(model, stack)
            target = out / "runs" / "DATA" / sid
            if ac.read_run(target, fp_, identity) is not None:
                m = w.read(target / "metrics.json")
                pt.read_points(target)
                checked += m["nn3_restore_check"]["checked"]
                mismatched += m["nn3_restore_check"]["mismatched"]
                continue
            while backlog() > int(r["max_backlog_jobs"]):
                wait_some()
                harvest()
            began = time.monotonic()
            shutil.rmtree(work / sid, ignore_errors=True)
            net = nn.load_model(model)[0]
            built = [(nn.load_model(Path(d))[0], float(t)) for d, t in stack]
            policy = n3.Collector(pool, cfg, s, work / sid / "states", net, built, n3.policy_spec(model, stack))
            try:
                with nn.planner_free(policy, "NN"):
                    result = pt.run_episode(s, work / sid / "episode", policy, max_seconds=float(r["main_episode_max_seconds"]), on_step=policy.after_step)
                if result.metrics["route_sha256"] != s["route_sha256"]:
                    raise ValueError("차량 파일 SHA가 설정과 다릅니다.")
                dirty = ref.quality_errors(result.metrics)
                if dirty:
                    raise RuntimeError(f"수집 본 판 품질 오류: {dirty}")
                inflight.append({"situation": s, "sid": sid, "seed": seed, "identity": identity, "policy": policy, "result": result, "main_seconds": time.monotonic() - began})
            except BaseException as error:
                failed.append({"situation": sid, "error": f"{type(error).__name__}: {error}"})
                ref.stamp(f"본 판 실패 {sid}: {failed[-1]['error']}")
                for future in policy.futures_list():
                    future.cancel()
                wait([f for f in policy.futures_list() if not f.cancelled()])
                policy.cleanup()
                shutil.rmtree(work / sid, ignore_errors=True)
                ledger.add(time.monotonic() - began + policy.spent())
                if isinstance(error, KeyboardInterrupt):
                    raise
            harvest()
            status()
            total_checked, total_mismatched = pt.restore_totals(out)
            rate = total_mismatched / total_checked if total_checked else 0.0
            if total_checked >= int(rc_cfg["min_points"]) and rate > float(rc_cfg["max_mismatch_rate"]):
                stopped = True
                ref.stamp(f"모든 조각 합 복원 불일치 {rate:.1%}({total_mismatched}/{total_checked})가 기준을 넘어 새 판을 시작하지 않습니다.")
                break
        while inflight:
            wait_some()
            harvest()
    status()
    if stopped:
        return 3
    return 1 if failed else 0


# ----------------------------------------------------------------------------- 학생
def load_points(data: Path, situations: list[dict], round_: int, cfg: dict) -> dict:
    """끝난 자료를 표로 읽는다. 라벨 칸은 cfg의 300초 하나다. 라운드 1 자료도 같은 함수로 읽는다."""
    data = Path(data)
    record = s2e.folder_record(data)
    rows, skipped, runs = [], Counter(), []
    for s in situations:
        sid = ac.situation_id(s)
        target = data / "runs" / "DATA" / sid
        m = w.read(target / "metrics.json")
        if m.get("fingerprint") != record["fingerprint"] or m.get("route_sha256") != s["route_sha256"] or m.get("nn3_round") != round_:
            raise ValueError(f"자료의 지문·차량 파일·라운드가 설정과 다릅니다: {sid}")
        dirty = ref.quality_errors(m)
        if dirty:
            raise ValueError(f"자료 본 판 품질 오류 {sid}: {dirty}")
        runs.append({"sid": sid, "band": s["band"], "driver": m["nn3_driver_seed"], "points": m["nn3_points"], "cpu_seconds": m["nn3_rollout_cpu_seconds"], "main_seconds": m["nn3_main_seconds"], "points_sha256": m["points_sha256"], "mean_delay_s": m["mean_delay_s"]})
        if m.get("collisions_unexplained", 0) or m.get("collisions_at_204820", 0):
            skipped["collision_runs"] += 1
            continue
        for point in pt.read_points(target):
            row, reason = n3.point_row(point, cfg)
            if row is None:
                skipped[reason] += 1
                continue
            rows.append({**row, "date": s["date"], "sid": sid, "driver": m["nn3_driver_seed"]})
    if not rows:
        raise ValueError("쓸 지점이 없습니다.")
    return {"x": np.stack([r["x"] for r in rows]).astype(np.float32), "allowed": np.stack([r["allowed"] for r in rows]), "base": np.asarray([r["base"] for r in rows], np.int64), "dates": np.asarray([r["date"] for r in rows]),
            "delay": np.stack([r["delay"] for r in rows]), "residual": np.stack([r["residual"] for r in rows]), "drivers": np.asarray([r["driver"] for r in rows]), "sids": np.asarray([r["sid"] for r in rows]),
            "skipped": dict(skipped), "runs": runs, "fingerprint": w.digest({"runs": [[r["sid"], r["points_sha256"]] for r in runs]})}


def policy_base_of(models: dict[int, Path], stacks: dict[int, list]):
    nets = {seed: nn.load_model(model)[0] for seed, model in models.items()}
    built = {seed: [(nn.load_model(Path(d))[0], float(t)) for d, t in stacks[seed]] for seed in models}

    def base_of(seed: int, x: np.ndarray, allowed: list[int]) -> int:
        return n3.choose(nets[seed], built[seed], x, allowed)[0]

    return base_of


def build_datasets(new: dict, old: dict, base_of, cfg: dict) -> dict[str, dict]:
    """half(짝수 판), new, pool(new + 라운드 1 기준 녹색을 앞 정책으로 다시 계산)."""
    keep = np.isin(new["sids"], half_ids(cfg))
    rebased = {**old, "base": rebased_points(old, base_of)}
    return {"half": subset_points(new, keep, "half"), "new": subset_points(new, np.ones(len(new["x"]), bool), "new"), "pool": merge_points(new, rebased)}


def learn(data: Path, r1_data: Path, n3cfg: dict, models: dict[int, Path], stacks: dict[int, list], out: Path, cfg: dict, workers: int) -> dict:
    """자료 세 가지 × 난수 3개의 학생을 만든다. 이미 있으면 같은 자료인지 보고 건너뛴다."""
    new = load_points(data, cfg["situations_round"], ROUND, cfg)
    old = load_points(r1_data, n3cfg["situations_round"], 1, cfg)
    sets = build_datasets(new, old, policy_base_of(models, stacks), cfg)
    st, le = cfg["training"], cfg["learn"]
    out = Path(out)
    tasks, done = [], []
    for name in le["data"]:
        d = sets[name]
        payload = {k: d[k] for k in ("x", "base", "allowed", "dates", "delay", "residual", "fingerprint")}
        for seed in st["seeds"]:
            folder = out / name / f"seed_{seed}"
            if (folder / "manifest.json").exists():
                manifest = w.read(folder / "manifest.json")
                if manifest.get("dataset_fingerprint") != d["fingerprint"]:
                    raise ValueError(f"이미 있는 학생이 다른 자료로 학습됐습니다: {folder}")
                done.append({"data": name, "seed": int(seed), "model_sha256": nn.model_identity(folder), **manifest["training"]})
            else:
                tasks.append((name, (payload, 0, LABEL["mark"], LABEL["weight"], seed, str(folder), st, le["shares"])))
    if tasks:
        with ProcessPoolExecutor(max_workers=max(1, workers), mp_context=multiprocessing.get_context("spawn"), initializer=ref._worker_init) as pool:
            for (name, _), got in zip(tasks, pool.map(n3._student_task, [t for _, t in tasks])):
                done.append({"data": name, **got})
    changed = float((sets["pool"]["base"][len(new["x"]):] != old["base"]).mean())
    summary = {"round": ROUND, "points": {k: int(len(v["x"])) for k, v in sets.items()}, "skipped_new": new["skipped"], "skipped_old": old["skipped"], "runs": new["runs"], "old_base_changed_share": changed, "dataset_fingerprints": {k: v["fingerprint"] for k, v in sets.items()},
               "students": sorted(done, key=lambda r: (r["data"], r["seed"]))}
    s2e._write(out / "done.json", summary)
    ref.stamp(f"학생 학습 끝: 지점 {summary['points']}, 라운드 1 기준 녹색이 바뀐 지점 {changed:.1%}, 학생 {len(done)}개")
    return summary


def candidate_stacks(students: Path, name: str, stacks: dict[int, list]) -> dict[int, list]:
    """후보 이름 → 난수별 [라운드 1 학생, 이번 학생]."""
    data, share = parse_candidate(name)
    out = {}
    for seed, stack in stacks.items():
        folder = Path(students) / data / f"seed_{seed}"
        tau = w.read(folder / "manifest.json")["training"]["taus"][f"{share:.2f}"]
        out[int(seed)] = [*stack, [str(folder), float(tau)]]
    return out


# ----------------------------------------------------------------------------- 평가
def job_worker(job: dict) -> dict:
    if job.get("mode") == "fixed":
        return pb.base_worker(job)
    return n3.free_job(job)


def fixed_jobs(folder: Path, work: Path, fp_: str, sits: list[dict]) -> list[dict]:
    jobs = []
    for s in sits:
        job = ref.jobs_for(Path(folder), Path(work), fp_, FIXC, [p0._situation(s)], model_identity=f"{WORLD}/{FIXC}", kind="fixed")[0]
        job.update(mode="fixed", world=WORLD, route_sha256=s["route_sha256"])
        jobs.append(job)
    return jobs


def run_stage(stage: str, jobs: list[dict], out: Path, shard: int, shards: int, workers: int, ledger) -> bool:
    mine = n3.shard_jobs(jobs, shard, shards)
    path = n3._stage_status(out, stage, shard)
    if path.exists() and w.read(path).get("jobs") == len(mine):
        return True
    status = ref.run_jobs(mine, max(1, workers), ledger, worker=job_worker)
    if status["failed"]:
        ref.stamp(f"{stage}: 실패 {len(status['failed'])}판. 다시 실행하면 끝난 판은 건너뜁니다.")
        return False
    n3.write_status(path, jobs=len(mine), rows=n3.job_rows(mine))
    return True


def selection(out: Path, shards: int, cfg: dict) -> dict:
    sids = [ac.situation_id(s) for s in cfg["situations_screen"]]
    screen = n3.stage_rows(out, "screen", shards)
    means = n3.candidate_means(screen, screen_candidates(cfg), sids)
    return {"screen_means": means, "finalists": finalists(means, cfg)}


def run_round(prep: Path, r1: Path, out: Path, work: Path, shard: int, shards: int, workers: int, cfg: dict, *, every_s: float = 60.0) -> int:
    """자료 → 학생 → 거르기 → 확인. 런타임 사이는 Drive의 상태 파일로 맞춘다. 다시 실행하면 끝난 부분은 건너뛴다."""
    prep, r1, out, work = Path(prep), Path(r1), Path(out), Path(work)
    n3cfg = n3.config()
    models, stacks = base_policy(prep, r1 / "students", cfg)
    data, students, evaluation = out / "data", out / "students", out / "eval"
    nn.prepare_caches(WORLD)
    # 지문 파일은 조각 0이 먼저 쓰고 다른 조각은 기다린다(Drive에 같은 이름 파일이 둘 생기지 않게).
    if shard != 0:
        n3.wait_for([data / "experiment.json"], "자료 지문", every_s)
    fp_data = ref.register(data, extra_files=SOURCES)
    code = collect(models, stacks, cfg, data, work / "data", shard, shards, workers, fp_data)
    if code:
        return code
    n3.write_status(n3._stage_status(out, "data", shard), done=True)
    n3.wait_for([n3._stage_status(out, "data", s) for s in range(shards)], "자료", every_s)
    if shard == 0:
        learn(data, r1 / "data", n3cfg, models, stacks, students, cfg, workers)
    n3.wait_for([students / "done.json"], "학생", every_s)
    if shard != 0:
        n3.wait_for([evaluation / "experiment.json"], "평가 지문", every_s)
    fp_ = ref.register(evaluation, extra_files=SOURCES)
    ledger = ref.Ledger(evaluation, f"shard{shard}")
    screen, final, low = cfg["situations_screen"], cfg["situations_final"], cfg["situations_low"]

    def stack_of(name: str) -> dict | None:
        if name == N5:
            return None
        return stacks if name == BASE else candidate_stacks(students, name, stacks)

    def jobs_of(names: list[str], sits: list[dict]) -> list[dict]:
        rows = []
        for name in names:
            rows += n3.nn_jobs(evaluation, work / "eval", fp_, name, models, sits, stacks=stack_of(name))
        return rows

    if not run_stage("screen", jobs_of([BASE, *screen_candidates(cfg)], screen), out, shard, shards, workers, ledger):
        return 1
    n3.wait_for([n3._stage_status(out, "screen", s) for s in range(shards)], "거르기", every_s)
    chosen = selection(out, shards, cfg)
    ref.stamp(f"거르기 끝. 확인 후보: {chosen['finalists']}")
    if not run_stage("final", [*jobs_of(final_names(chosen["finalists"], cfg), [*final, *low]), *fixed_jobs(evaluation, work / "eval", fp_, [*final, *low])], out, shard, shards, workers, ledger):
        return 1
    ref.stamp("이 런타임의 확인 끝. 모든 런타임이 끝나면 NN3_P_R2_judge를 실행하세요.")
    return 0


# ----------------------------------------------------------------------------- 판정
def read_fixed(evaluation: Path, sits: list[dict]) -> dict:
    fingerprint = w.read(Path(evaluation) / "experiment.json")["fingerprint"]

    def one(s):
        sid = ac.situation_id(s)
        path = Path(evaluation) / "runs" / FIXC / sid / "metrics.json"
        if not path.exists():
            raise FileNotFoundError(f"{FIXC}/{sid}")
        m = w.read(path)
        if m.get("fingerprint") != fingerprint:
            raise ValueError(f"결과의 지문이 폴더와 다릅니다: {path}")
        return sid, n3.run_row(m)

    with ThreadPoolExecutor(16) as pool:
        return {FIXC: dict(pool.map(one, sits))}


def percent(delay: float, fixed: float) -> float:
    return 100.0 * (delay - fixed) / fixed


def parts_of(base: dict, cand: dict) -> dict:
    return {"per_date": n3.part_gains(base, cand, [(d, lambda k, x=d: k.startswith(x)) for d in n3.VALID_DATES]), "per_batch": n3.part_gains(base, cand, [(b, lambda k, x=b: k.endswith(x)) for b in ("VG1", "VG2")]),
            "per_model": [n3.weighted_mean([v[i] for v in base.values()]) - n3.weighted_mean([v[i] for v in cand.values()]) for i in range(3)],
            "per_band": n3.part_gains(base, cand, [(n, lambda k, a=lo, z=hi: a <= int(k.split("_")[1]) < z) for n, lo, hi in (("아침 06~08", 600, 900), ("낮 09~15", 900, 1600), ("저녁 16~18", 1600, 1900), ("밤 19~21", 1900, 2400))])}


def collect_report(r2: Path, cfg: dict) -> dict:
    r2 = Path(r2)
    done = r2 / "students" / "done.json"
    report = {"experiment": EXPERIMENT}
    if not done.exists():
        return {**report, "status": "before_students"}
    summary = w.read(done)
    scores = {}
    for s in summary["students"]:
        scores.setdefault(s["data"], {})[str(s["seed"])] = s["scores"]
    report["students"] = {"points": summary["points"], "skipped_new": summary["skipped_new"], "skipped_old": summary["skipped_old"], "old_base_changed_share": summary["old_base_changed_share"], "runs": len(summary["runs"]), "cpu_hours_rollouts": sum(r["cpu_seconds"] for r in summary["runs"]) / 3600, "scores": scores}
    evaluation = r2 / "eval"
    screen_sits, final, low = cfg["situations_screen"], cfg["situations_final"], cfg["situations_low"]
    day_ids, final_ids, low_ids = [ac.situation_id(s) for s in screen_sits], [ac.situation_id(s) for s in final], [ac.situation_id(s) for s in low]
    try:
        names = screen_candidates(cfg)
        screen = n3.read_stage_runs(evaluation, [BASE, *names], screen_sits)
        means = n3.candidate_means(screen, names, day_ids)
        best = finalists(means, cfg)
        all_final = final_names(best, cfg)
        last = n3.read_stage_runs(evaluation, all_final, [*final, *low])
        fixed = read_fixed(evaluation, [*final, *low])[FIXC]
    except FileNotFoundError as error:
        return {**report, "status": "incomplete", "missing": str(error)}
    base_day = n3._by_situation(screen, BASE, day_ids)
    report["screen"] = {n: {"mean_delay_s": v, "gain_vs_base": n3.paired_gain(base_day, n3._by_situation(screen, n, day_ids)), "summary": n3.policy_summary(n3._by_situation(screen, n, day_ids))} for n, v in sorted(means.items(), key=lambda kv: kv[1])}
    report["base_day_s"] = n3.candidate_means(screen, [BASE], day_ids)[BASE]
    report["finalists"] = best
    by = {n: n3._by_situation(last, n, final_ids) for n in all_final}
    by_low = {n: n3._by_situation(last, n, low_ids) for n in all_final}
    fixed_mean = n3.weighted_mean([fixed[s] for s in final_ids])
    for n in all_final:
        for s in final_ids:
            if {r["vehicles"] for r in by[n][s]} != {fixed[s]["vehicles"]}:
                raise ValueError(f"고정신호와 차 수가 다릅니다: {n}/{s}")
    fixed_rows = [fixed[s] for s in final_ids]
    report["final"] = {"fixed": {"mean_delay_s": fixed_mean, "left_east_s": statistics.fmean(r["left_east"] for r in fixed_rows if r["left_east"] is not None), "left_south_s": statistics.fmean(r["left_south"] for r in fixed_rows if r["left_south"] is not None), "long_per_run": sum(r["long"] for r in fixed_rows) / len(fixed_rows)}, "policies": {}}
    for n in all_final:
        summary_row = n3.policy_summary(by[n])
        row = {"summary": summary_row, "percent_vs_fixed": percent(summary_row["mean_delay_s"], fixed_mean), "low_mean_delay_s": n3.weighted_mean([r for v in by_low[n].values() for r in v])}
        if n != BASE:
            row.update(gain_vs_base=n3.paired_gain(by[BASE], by[n]), low_gain_vs_base=n3.paired_gain(by_low[BASE], by_low[n]), **parts_of(by[BASE], by[n]))
        if n != N5:
            row.update(gain_vs_n5=n3.paired_gain(by[N5], by[n]))
        report["final"]["policies"][n] = row
    winner = max(best, key=lambda n: report["final"]["policies"][n]["gain_vs_base"]["gain_s"])
    gain = report["final"]["policies"][winner]["gain_vs_base"]
    adopted = gain["t95"][0] is not None and gain["gain_s"] >= float(cfg["rules"]["adopt_gain_s"]) and gain["t95"][0] > 0.0
    report["reading"] = {"winner": winner, "gain": gain, "adopted": adopted, "round1_replication": report["final"]["policies"][BASE]["gain_vs_n5"], "cumulative": report["final"]["policies"][winner]["gain_vs_n5"], "percent_vs_fixed": {n: report["final"]["policies"][n]["percent_vs_fixed"] for n in (N5, BASE, winner)}}
    c = {parse_candidate(n)[0]: by[n] for n in curve_candidates(cfg)}
    report["data_curve"] = {"new_minus_half": n3.paired_gain(c["half"], c["new"]), "pool_minus_new": n3.paired_gain(c["new"], c["pool"]), "means": {k: n3.weighted_mean([r for v in x.values() for r in v]) for k, x in c.items()}}
    return {**report, "status": "done"}


gain_text = n3.gain_text


def render(report: dict) -> str:
    lines = ["# NN3R2 라운드 2 결과(무리 세계, 검증 날짜 개발 비교)", ""]
    if report["status"] != "done":
        return "\n".join([*lines, f"아직 끝나지 않았습니다: {report.get('missing', report['status'])}"]) + "\n"
    st = report["students"]
    lines += [f"자료: 새 {st['runs']}판, 지점 {st['points']}, 제외 새 자료 {st['skipped_new']}·라운드 1 {st['skipped_old']}, 라운드 1 지점 중 기준 녹색이 바뀐 비율 {st['old_base_changed_share']:.1%}, 굴림 {st['cpu_hours_rollouts']:.0f} CPU-시간.", "",
              "## 거르기(새 배치 24상황 × 모델 3개)", "", f"앞 정책(BASE) {report['base_day_s']:.2f}초", "", "| 후보 | 지연 | BASE 대비 [t95] | N5와 다른 결정 | 동 좌회전 | 남 좌회전 | 미확인 충돌 |", "|---|---:|---|---:|---:|---:|---:|"]
    for name, row in report["screen"].items():
        g, m = row["gain_vs_base"], row["summary"]
        lines.append(f"| {name} | {row['mean_delay_s']:.2f} | {gain_text(g)} | {m['change_share']:.1%} | {m['left_east_s']:.1f} | {m['left_south_s']:.1f} | {m['collisions_unexplained']} |")
    fx = report["final"]["fixed"]
    lines += ["", f"확인 후보: {', '.join(report['finalists'])}", "", "## 확인(새 배치 84상황 × 모델 3개)", "", f"고정신호 {fx['mean_delay_s']:.2f}초, 동 좌회전 {fx['left_east_s']:.1f}, 남 좌회전 {fx['left_south_s']:.1f}, 300초 초과 판당 {fx['long_per_run']:.1f}", "",
              "| 정책 | 지연 | 고정신호 대비 | BASE 대비 [t95] | N5 대비 [t95] | 앞선 상황 | 날짜별 | 모델별 | 새벽·심야(BASE 대비) | N5와 다른 결정 | 동 좌회전 | 남 좌회전 | 300초 초과(판당) | 미확인 충돌 |", "|---|---:|---:|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|"]
    for name, row in report["final"]["policies"].items():
        m = row["summary"]
        g = row.get("gain_vs_base")
        vs_n5 = gain_text(row["gain_vs_n5"]) if "gain_vs_n5" in row else "기준"
        ahead = f"{g['positive']}/{g['n']}" if g else "-"
        dates = ", ".join(f"{v:+.2f}" for v in row["per_date"].values()) if g else "-"
        by_model = ", ".join(f"{v:+.2f}" for v in row["per_model"]) if g else ", ".join(f"{v:.2f}" for v in m["per_model_s"])
        low_cell = f"{row['low_gain_vs_base']['gain_s']:+.2f}" if g else f"{row['low_mean_delay_s']:.2f}초"
        lines.append(f"| {name} | {m['mean_delay_s']:.2f} | {row['percent_vs_fixed']:+.1f}% | {gain_text(g) if g else '기준'} | {vs_n5} | {ahead} | {dates} | {by_model} | {low_cell} | {m['change_share']:.1%} | {m['left_east_s']:.1f} | {m['left_south_s']:.1f} | {m['long_per_run']:.2f} | {m['collisions_unexplained']} |")
    r, c = report["reading"], report["data_curve"]
    lines += ["", f"**읽기:** {r['winner']} − BASE {gain_text(r['gain'])}초 → **{'채택' if r['adopted'] else '미채택'}** (기준: 평균 +0.2초 이상, t95 하한 > 0).", "",
              f"- 라운드 1 재현(BASE − N5, 새 배치): {gain_text(r['round1_replication'])}초 (라운드 1 값 +0.69초).", f"- 누적 몫({r['winner']} − N5): {gain_text(r['cumulative'])}초.",
              f"- 고정신호 대비: N5 {r['percent_vs_fixed'][N5]:+.1f}%, BASE {r['percent_vs_fixed'][BASE]:+.1f}%, {r['winner']} {r['percent_vs_fixed'][r['winner']]:+.1f}%.",
              f"- 자료 곡선(문턱 5%, 84상황): half {c['means']['half']:.2f} · new {c['means']['new']:.2f} · pool {c['means']['pool']:.2f}초. new − half {gain_text(c['new_minus_half'])}초, pool − new {gain_text(c['pool_minus_new'])}초.",
              "- 검증 날짜의 개발 비교다. 무리 도착은 모형이다. 12개 지도는 재지 않았다."]
    return "\n".join(lines) + "\n"


def save_report(out: Path, report: dict) -> None:
    s2e._write(Path(out) / "judge.json", w.plain(report))
    (Path(out) / "summary.md").write_text(render(report), encoding="utf-8")


# ----------------------------------------------------------------------------- 점검(맥·Colab에서 같은 코드)
def check(prep: Path, r1_students: Path, out: Path, work: Path, workers: int, cfg: dict, points: int = 3) -> dict:
    """(1) 상태 저장·굴림을 해도 앞 정책 신호가 같다 (2) 학습 전 학생을 쌓아도 같다 (3) 복원·입력 대조·엿봄 없음 (4) 300초 라벨 칸이 있다 (5) 고정신호 한 판 (6) 맥 기록과 같은 신호."""
    out, work = Path(out), Path(work)
    shutil.rmtree(work, ignore_errors=True)
    nn.prepare_caches(WORLD)
    models, stacks = base_policy(prep, r1_students, cfg)
    model, stack = models[42], stacks[42]
    net = nn.load_model(model)[0]
    built = [(nn.load_model(Path(d))[0], float(t)) for d, t in stack]
    s = cfg["situations_round"][0]

    def plain_run(policy, tag, situation=s):
        with nn.planner_free(policy, "NN") as guard:
            result = pt.run_episode(situation, work / tag, policy, max_seconds=nn.MAX_EPISODE_S)
        if result.metrics["route_sha256"] != situation["route_sha256"]:
            raise ValueError("차량 파일 SHA가 설정과 다릅니다.")
        return result, guard, policy

    plain, guard, plain_policy = plain_run(n3.StackPolicy(net, BASE, built), "plain")
    untrained = nn.GainNet(n3.INPUT_SIZE)
    stacked, _, _ = plain_run(n3.StackPolicy(net, "stacked", [*built, (untrained.eval(), 0.5)]), "stacked")
    n5_only, _, _ = plain_run(n3.StackPolicy(net, N5), "n5")
    with pt.rollout_pool(workers) as pool:
        policy = n3.Collector(pool, cfg, s, work / "same" / "states", net, built, n3.policy_spec(model, stack), limit=points)
        with nn.planner_free(policy, "NN"):
            same = pt.run_episode(s, work / "same" / "episode", policy, max_seconds=float(cfg["rollout"]["main_episode_max_seconds"]), on_step=policy.after_step)
        wait(policy.futures_list())
        rc = policy.finish()
    rows = [n3.point_row(p, cfg) for p in policy.points]
    low_s = next(x for x in cfg["situations_round"] if x["band"] == "dawn")
    low, low_guard, _ = plain_run(n3.StackPolicy(net, "low", built), "low", low_s)
    fixed = job_worker(fixed_jobs(work / "fixed_out", work / "fixed", "check", cfg["situations_final"][:1])[0])
    fixed_row = None
    if fixed["ok"]:
        fixed_row = n3.run_row(w.read(ref.target_dir(work / "fixed_out", FIXC, ac.situation_id(cfg["situations_final"][0])) / "metrics.json"))
    seconds = [float(x.get("seconds") or 0.0) for *_, f in policy.rollouts for x in [f.result()]]
    keys = ("executed_sha16", "route_sha256", "actual_total_delay")
    report = {"same_as_plain": all(same.metrics[k] == plain.metrics[k] for k in keys), "stacked_same": all(stacked.metrics[k] == plain.metrics[k] for k in keys), "student_changes_signal": n5_only.metrics["executed_sha16"] != plain.metrics["executed_sha16"], "planner_calls": guard["tail_calls"], "low_planner_calls": low_guard["tail_calls"],
              "points": len(policy.points), "restore": {k: rc[k] for k in ("checked", "mismatched")}, "rollouts": rc["rollouts"], "rollout_failures": rc["rollout_failures"], "usable_points": sum(r[0] is not None for r in rows), "skip_reasons": [r[1] for r in rows if r[0] is None], "east_future": rc["east_future"],
              "rollout_seconds": {"mean": statistics.fmean(seconds), "max": max(seconds)}, "main_seconds": same.seconds, "plain_seconds": plain.seconds, "plain": {k: plain.metrics[k] for k in keys}, "plain_delay_s": plain.metrics["mean_delay_s"], "n5_delay_s": n5_only.metrics["mean_delay_s"], "low": {k: low.metrics[k] for k in keys},
              "change_share": plain_policy.stats["student_changed"] / max(plain_policy.stats["planned"], 1), "quality_errors": ref.quality_errors(plain.metrics) + ref.quality_errors(low.metrics), "fixed": {"ok": fixed["ok"], "error": fixed.get("error"), "row": fixed_row}}
    if REFERENCE.exists():
        saved = w.read(REFERENCE)
        report["same_as_mac"] = all(saved["plain"][k] == report["plain"][k] for k in keys) and all(saved["low"][k] == report["low"][k] for k in keys)
    report["ok"] = bool(report["same_as_plain"] and report["stacked_same"] and report["student_changes_signal"] and report["usable_points"] > 0 and not report["quality_errors"] and rc["mismatched"] == 0 and not rc["rollout_failures"] and fixed["ok"] and report.get("same_as_mac", True))
    s2e._write(out / "check.json", w.plain(report))
    shutil.rmtree(work, ignore_errors=True)
    return report


# ----------------------------------------------------------------------------- 명령
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("make-config")
    p.add_argument("--prep", required=True)
    p.add_argument("--students", required=True)
    p.add_argument("--workers", type=int, default=8)
    p = sub.add_parser("check")
    p.add_argument("--prep", required=True)
    p.add_argument("--r1-students", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--work", required=True)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--points", type=int, default=3)
    p = sub.add_parser("round")
    p.add_argument("--prep", required=True)
    p.add_argument("--r1", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--work", required=True)
    p.add_argument("--shard", type=int, required=True)
    p.add_argument("--shards", type=int, default=4)
    p.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1))
    p = sub.add_parser("judge")
    p.add_argument("--r2", required=True)
    p.add_argument("--out", required=True)
    args = parser.parse_args()
    w.configure()
    if args.cmd == "make-config":
        print(json.dumps(make_config(Path(args.prep), Path(args.students), args.workers), ensure_ascii=False, indent=1))
        return 0
    cfg = config()
    if args.cmd == "check":
        report = check(Path(args.prep), Path(args.r1_students), Path(args.out), Path(args.work), args.workers, cfg, args.points)
        print(json.dumps(w.plain(report), ensure_ascii=False, indent=1))
        return 0 if report["ok"] else 1
    if args.cmd == "round":
        return run_round(Path(args.prep), Path(args.r1), Path(args.out), Path(args.work), args.shard, args.shards, args.workers, cfg)
    report = collect_report(Path(args.r2), cfg)
    save_report(Path(args.out), report)
    print(render(report))
    return 0 if report["status"] == "done" else 1


if __name__ == "__main__":
    sys.exit(main())
