"""[추가 탐색] NN12 — 신경망만 모델을 12개 지도에서 잰다(계획서 2단계, 사전 등록 results/explore/nn12_20261103/사전등록.md).

단독 지도에서 배운 N5(무리 세계)·Rn(무작위 세계)과 짝 계획기 TM4·M4를 12개 지도 검증 2일 46조각 × seed 5개에서 돌린다. 학습은 없다.
평가 코드(network_eval)는 고치지 않고 실행할 때만 '단순 제어' 자리(focused_eval.QueueFirst)에 끼운다(P2·JOINT2와 같은 방식).
입력은 NN1과 같은 코드로 만들고, 신경망 실행은 계획기를 부르면 멈춘다. 보정 고정신호·P2·P0·B12는 다시 돌리지 않고 읽는다.

사용(FOCUSED_RULESET=pet1 FOCUSED_NORTH_EXT=240):
  python src/explore_nn12.py make-config --models-N5 <…/r1> --models-Rn <…/Rn>     (SUMO 없음: 모델 해시·신호 대응 고정)
  python src/explore_nn12.py check --policy N5 --models <…/r1> --reference-p2 <explore_joint2> --out <dir> --work /content/n12c
  python src/explore_nn12.py run --policy N5 --models <…/r1> --out <Drive>/explore_nn12/N5 --work /content/n12 --workers 8
  python src/explore_nn12.py recheck --policy N5 --models <…/r1> --out <Drive>/explore_nn12/N5 --work /content/n12c     (사전 등록 정정 기록: 끝난 판 하나가 소수점까지 같은지)
  python src/explore_nn12.py judge --root <Drive>/explore_nn12 --joint2 <explore_joint2> --pred <explore_pred/network12> --b12 <explore_12map> --out <…/judge>
"""
from __future__ import annotations

import argparse
import contextlib
import gzip
import hashlib
import json
import math
import multiprocessing
import os
import shutil
import statistics
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import arena3_common as ac  # noqa: E402
import arena3_referee as ref  # noqa: E402
import arena3_stage2_eval as s2e  # noqa: E402
import explore_coord12 as ec  # noqa: E402
import explore_joint2 as ej  # noqa: E402
import explore_nn1 as nn  # noqa: E402
import explore_platoon0 as p0  # noqa: E402
import explore_platoon1 as pt  # noqa: E402
import explore_pred as ep  # noqa: E402
import explore_weight as w  # noqa: E402
import explore_weight_core as wc  # noqa: E402
import focused_eval  # noqa: E402
import network_eval as ne  # noqa: E402
import network_rl as nr  # noqa: E402
import network_signals as ns  # noqa: E402
import network_windows as nw  # noqa: E402
from focused_data import TARGET_TLS, environment_fingerprint  # noqa: E402
from focused_scenario import aihub_states  # noqa: E402

EXPERIMENT = "NN12_V1"
CONFIG = ROOT / "config/explore_nn12_v1.json"
PREREG = ROOT / "results/explore/nn12_20261103/사전등록.md"
SOURCES = ("src/explore_nn12.py", "src/explore_nn1.py", "src/explore_joint2.py", "src/explore_platoon0.py", "src/explore_platoon1.py", "src/explore_pred.py",
           "src/explore_weight_core.py", "src/arena3_common.py", "src/explore_coord12.py", "config/explore_nn12_v1.json", "results/explore/nn12_20261103/사전등록.md")
MODELS = ("N5", "Rn")
PLANNERS = ("TM4", "M4")
POLICIES = (*MODELS, *PLANNERS)
PARTNER = {"N5": "TM4", "Rn": "M4"}
SEEDS = (42, 43, 44, 45, 46)
EVALUATION_FP, NETWORK_FP, ENVIRONMENT_FP = "9d3b7504064b", "711875f6c939", "b20888514ea3"
DATES = ("20220928", "20220813")
MODEL_SPEC = {"N5": {"world": "P", "seeds": [42, 43, 44], "source": "explore_nn1/P/aug/models/r1(자료 강화 라운드 1, NN1 selected.json)"},
              "Rn": {"world": "R", "seeds": [42, 43, 44], "source": "explore_nn1/R/base/models/Rn"}}
B12_MODELS = {"42": "b4dbff0863", "43": "f99c3749f2", "44": "fdf0af3fb4", "45": "e49f8bee48", "46": "336be6559e"}
RULES = {"p2_gap_s": 2.2, "transfer_gap_s": 2.2, "unstable_window": "20220928_08"}
CHECK = {"window": ["20220811", 18], "seed": 42, "reproduce": {"window": ["20220813", 12], "seed": 42}}
BANDS = {"새벽 1~6시": range(1, 7), "아침 7~9시": range(7, 10), "낮 10~15시": range(10, 16), "저녁 16~18시": range(16, 19), "밤 19~23시": range(19, 24)}
REFERENCES = ("FIXC", "B12", "P0", "P2")
ORIGINAL_SEAT = focused_eval.QueueFirst
SAME = (("window_204820", "mean_delay_s"), ("network", "mean_delay_s"), ("via_204822", "mean_delay_s"))
TABLE_READ_LAG_S = 1                 # SUMO가 t초에 보여 주는 상태는 원자료의 t − 1초 칸이다(맥 720판단 불일치 0, 다른 어긋남은 39~90)
CORRECTION = {"earlier_code": "e9a1c2307a9652d08268814c23314692e07c99ea641a86857dfd96cf6d9899a5", "rerun_policies": ["N5", "TM4"], "rerun_hour": 23, "recheck": {"window": ["20220813", 22], "seed": 42},
              "why": "기록용 시간표 대조가 자정 뒤 첫 판단에서 표 밖을 읽어 23시 조각 20판이 실패했다. 판단·입력·평가 코드는 그대로이고 끝난 900판은 그대로 쓴다(사전 등록 정정 기록)."}


# ----------------------------------------------------------------------------- 설정
def neighbor_labels(dates=DATES) -> dict[str, str]:
    """12개 지도 204822 상태 → 시간표(원본 이름) 상태. 날짜마다 초 단위로 일대일인지 확인한다(SUMO 없음)."""
    labels: dict[str, str] = {}
    for date in dates:
        programs, _ = ns.day_programs(date)
        states = [s for d, s in programs[ej.NEIGHBOR] for _ in range(d)]
        table = p0.timetable(date)
        if len(states) != len(table):
            raise ValueError(f"{date}: 12개 지도 신호와 시간표의 길이가 다릅니다.")
        for a, b in Counter(zip(states, table)):
            if labels.setdefault(a, b) != b:
                raise ValueError(f"{date}: 12개 지도 상태 {a}가 시간표 상태 둘에 대응합니다.")
    if len(set(labels.values())) != len(labels):
        raise ValueError("시간표 상태 하나에 12개 지도 상태 둘이 대응합니다.")
    return labels


def make_config(models: dict[str, Path]) -> dict:
    """모델 해시·조각·신호 대응을 고정한다(본 실행 전 맥에서 한 번)."""
    if ne.evaluation_fingerprint() != EVALUATION_FP or nr._code_fingerprint() != NETWORK_FP or environment_fingerprint() != ENVIRONMENT_FP:
        raise ValueError("12개 지도·평가·환경 지문이 기존 기준 실행과 다릅니다.")
    spec = {}
    for name, row in MODEL_SPEC.items():
        folder = Path(models[name])
        spec[name] = {**row, "identities": {str(s): nn.model_identity(folder / f"seed_{s}") for s in row["seeds"]}}
    labels = neighbor_labels((*DATES, CHECK["window"][0]))
    cfg = {"experiment": EXPERIMENT, "prereg_sha256": w.file_hash(PREREG), "nn1_config_sha256": w.file_hash(nn.CONFIG), "evaluation_fingerprint": EVALUATION_FP,
           "network12_fingerprint": NETWORK_FP, "environment_fingerprint": ENVIRONMENT_FP, "dates": list(DATES), "windows": [list(x) for x in nw.evaluation_windows(list(DATES))],
           "seeds": list(SEEDS), "input_seed": SEEDS[0], "models": spec, "planners": {"TM4": {"world": "P"}, "M4": {"world": "R"}}, "neighbor_labels": labels, "check": CHECK, "correction": CORRECTION, "rules": RULES,
           "bands": {k: [v.start, v.stop] for k, v in BANDS.items()}, "b12_models": B12_MODELS,
           "reporting": "검증 2일 개발 비교. 흉내 값은 강화학습 값으로 쓰지 않는다. 단독 지도 감소율과 직접 비교하지 않는다."}
    w.write(CONFIG, cfg)
    return cfg


def config() -> dict:
    cfg = w.read(CONFIG)
    if cfg.get("experiment") != EXPERIMENT:
        raise ValueError("NN12 설정이 아닙니다.")
    if cfg.get("prereg_sha256") != w.file_hash(PREREG) or cfg.get("nn1_config_sha256") != w.file_hash(nn.CONFIG):
        raise ValueError("NN12 설정을 만든 때와 사전 등록·NN1 설정이 다릅니다.")
    return cfg


def code_record() -> dict:
    return {"experiment": EXPERIMENT, "sources": {name: w.file_hash(ROOT / name) for name in SOURCES}, "evaluation_fingerprint": ne.evaluation_fingerprint(),
            "network12_fingerprint": nr._code_fingerprint(), "environment_fingerprint": environment_fingerprint()}


def code_fingerprint() -> str:
    return w.digest(code_record())


def check_policy(name: str) -> str:
    if name not in POLICIES:
        raise ValueError(f"정책은 {POLICIES} 중 하나: {name}")
    return name


# ----------------------------------------------------------------------------- 실행 계획(순수 계산)
def model_slot(window_index: int, seed_index: int, models: int = 3) -> int:
    return (window_index + seed_index) % models


def run_tag(date: str, hour: int, seed: int) -> str:
    return f"{date}_{int(hour):02d}_eval{seed}"


def plan_jobs(name: str, cfg: dict, out: Path, work: Path, models: Path | None = None) -> list[dict]:
    jobs = []
    for i, (date, hour) in enumerate(cfg["windows"]):
        for j, seed in enumerate(cfg["seeds"]):
            tag = run_tag(date, hour, seed)
            job = {"name": name, "tag": tag, "date": date, "hour": int(hour), "eval_seed": int(seed), "work": str(Path(work) / tag), "target": str(Path(out) / "runs" / tag), "save_inputs": False}
            if name in MODELS:
                spec = cfg["models"][name]
                chosen = spec["seeds"][model_slot(i, j, len(spec["seeds"]))]
                job.update(model_seed=chosen, model_identity=spec["identities"][str(chosen)], model_dir=str(Path(models) / f"seed_{chosen}"), save_inputs=seed == cfg["input_seed"])
            jobs.append(job)
    return jobs


def finished(job: dict, code: str, earlier: tuple[str, ...] = ()) -> bool:
    """끝난 판인가. 다른 코드·모델의 결과가 있으면 멈춘다(섞이지 않게). earlier = 정정 기록이 그대로 쓰기로 한 이전 코드."""
    path = Path(job["target"]) / "metrics.json"
    if not path.exists():
        return False
    record = w.read(path).get("nn12", {})
    if record.get("code") not in (code, *earlier) or record.get("model_identity") != job.get("model_identity"):
        raise ValueError(f"이미 있는 결과가 지금 코드·모델과 다릅니다: {path}")
    return True


# ----------------------------------------------------------------------------- '단순 제어' 자리에 앉는 제어
@contextlib.contextmanager
def seated(cls):
    """run_window이 도는 동안 이 제어를 단순 제어 자리에 앉힌다. 끝나면 원래대로 돌려놓는다."""
    focused_eval.QueueFirst = cls
    try:
        yield cls
    finally:
        focused_eval.QueueFirst = ORIGINAL_SEAT


class Seat:
    """run_window이 만들면 자리를 곧바로 돌려놓는다(계획기·입력 클래스가 단순 제어를 상속해 만들어지기 때문)."""
    date: str = ""
    origin_s: float = 0.0
    labels: dict = {}
    record: Counter = Counter()
    check_timetable: bool = False
    movements = staticmethod(ORIGINAL_SEAT.movements)

    def __init__(self, movement_of, greens):
        focused_eval.QueueFirst = ORIGINAL_SEAT
        if list(greens) != list(aihub_states()["greens"]):
            raise ValueError("12개 지도의 녹색 순서가 학습 때와 다릅니다.")
        self.movement_of, self.greens = movement_of, list(greens)
        self.ready = False
        self._table = p0.timetable(self.date) if self.check_timetable else None

    def compare_timetable(self, env) -> None:
        """입력이 보는 시간표와 12개 지도의 실제 이웃 신호가 같은 시각표인지 판단마다 대조한다. 23시 조각은 제어가 자정을 30초 넘겨, 자정 뒤 판단은 읽을 칸이 없어 따로 센다."""
        if self._table is None:
            return
        second = int(self.origin_s + float(env.sim_step)) - TABLE_READ_LAG_S
        if second >= len(self._table):
            self.record["timetable_past_midnight"] += 1
            return
        state = env.sumo.trafficlight.getRedYellowGreenState(ej.NEIGHBOR)
        self.record["timetable_checked"] += 1
        self.record["timetable_mismatch"] += int(self.labels.get(state) != self._table[second])


class NetSeat(Seat):
    """신경망만. 입력·선택은 NN1의 FeaturePlanner·NNPolicy 그대로다."""
    policy: nn.NNPolicy | None = None

    def _setup(self, env) -> None:
        ts = env.traffic_signals[TARGET_TLS]
        self.feature = nn.FeaturePlanner(SimpleNamespace(env=env, movement_of=self.movement_of, ts=ts), float("inf"))
        if self.policy.world == "P":
            self.policy.core = pt.TM4Core().build(self.movement_of, self.greens, self.date, self.origin_s, env, movements=self.feature.p0.movements)
        self.ready = True

    def __call__(self, env, allowed: list[int]) -> int:
        if not self.ready:
            self._setup(env)
        self.compare_timetable(env)
        return self.policy.act(env, self.feature.sensed(env), [int(a) for a in allowed])


class PlanSeat(Seat):
    """짝 계획기. TM4 = 이웃 시간표 정보 + 좌회전 4배(PLATOON1 TM4Core 그대로), M4 = P0 정보 + 좌회전 4배."""
    kind: str = ""

    def _setup(self, env) -> None:
        core = pt.TM4Core()
        if self.kind == "TM4":
            core.build(self.movement_of, self.greens, self.date, self.origin_s, env)
        else:
            ctl = ep.controller_class("p0", wc.HORIZON_S)(self.movement_of, self.greens)
            ctl._setup(env)
            core.ctl, core.weights = ctl, ac.lefts_weights(ac.MODES6[ac.M4], ctl.movements)
        self.core = core
        self.ready = True

    def __call__(self, env, allowed: list[int]) -> int:
        if not self.ready:
            self._setup(env)
        self.compare_timetable(env)
        self.core.observe(env)
        allowed = [int(a) for a in allowed]
        if len(allowed) == 1:
            self.record["forced"] += 1
            return allowed[0]
        self.record["planned"] += 1
        return self.core.decide(env, allowed)


def seat_class(base, job: dict, labels: dict, record: Counter, **more):
    world_p = job["name"] in ("N5", "TM4")
    return type(base.__name__, (base,), {"date": job["date"], "origin_s": float(job["hour"] * 3600 - nr.WARMUP_S), "labels": labels, "record": record, "check_timetable": world_p, **more})


# ----------------------------------------------------------------------------- 한 판
def execute(job: dict, run_dir: Path, labels: dict) -> dict:
    """조각 하나를 돌리고 metrics.json에 더할 기록을 돌려준다."""
    window = {"date": job["date"], "hour": job["hour"], "policy": "greedy", "eval_seed": job["eval_seed"], "run_dir": str(run_dir), "fingerprint": EVALUATION_FP}
    ej.install_metrics()
    record: Counter = Counter()
    if job["name"] in MODELS:
        net, manifest = nn.load_model(Path(job["model_dir"]))
        if nn.model_identity(Path(job["model_dir"])) != job["model_identity"] or manifest["world"] != MODEL_SPEC[job["name"]]["world"]:
            raise ValueError("모델이 설정에 고정한 것과 다릅니다.")
        policy = nn.NNPolicy(manifest["world"], net, job["name"], keep_inputs=bool(job["save_inputs"]))
        with seated(seat_class(NetSeat, job, labels, record, policy=policy)), nn.planner_free(None):
            ne.run_window(window)
        if job["save_inputs"]:
            policy.save_inputs(run_dir / "inputs.npz")
        return {"stats": {**policy.info(), **record}, "model_seed": job["model_seed"], "model_identity": job["model_identity"], "planner_free": True}
    if job["name"] in PLANNERS:
        with seated(seat_class(PlanSeat, job, labels, record, kind=job["name"])):
            ne.run_window(window)
        return {"stats": {"kind": job["name"], **record}, "model_seed": None, "model_identity": None, "planner_free": False}
    if job["name"] == "P2":
        focused_eval.QueueFirst = ORIGINAL_SEAT
        cls = ep.controller_class("p2", ep.HORIZON_S)
        with seated(cls):
            ne.run_window(window)
        return {"stats": {"kind": "P2", "planned": cls.stats["planned"]}, "model_seed": None, "model_identity": None, "planner_free": False}
    raise ValueError(f"알 수 없는 정책: {job['name']}")


def run_job(job: dict) -> dict:
    """한 판을 작업 폴더에서 돌리고, 끝나면 결과만 대상 폴더로 옮긴다. metrics.json을 마지막에 써서 있으면 끝난 판이다."""
    started = time.monotonic()
    work, target = Path(job["work"]), Path(job["target"])
    try:
        w.configure()
        torch.set_num_threads(1)
        shutil.rmtree(work, ignore_errors=True)
        work.mkdir(parents=True)
        extra = execute(job, work, job["labels"])
        metrics = w.read(work / "metrics.json")
        metrics["nn12"] = {"name": job["name"], "code": job["code"], "job_seconds": time.monotonic() - started, **extra}
        target.mkdir(parents=True, exist_ok=True)
        for name in ("vehicles.csv.gz", "inputs.npz"):
            if (work / name).exists():
                shutil.copyfile(work / name, target / name)
        w.write(target / "metrics.json", metrics)
        return {"ok": True, "tag": job["tag"], "seconds": time.monotonic() - started}
    except BaseException as error:
        return {"ok": False, "tag": job["tag"], "seconds": time.monotonic() - started, "error": f"{type(error).__name__}: {error}"}
    finally:
        shutil.rmtree(work, ignore_errors=True)
        focused_eval.QueueFirst = ORIGINAL_SEAT


def prepare_caches() -> None:
    """작업자들이 동시에 지도 캐시를 처음 만들다 부딪히지 않게 부모 프로세스에서 한 번 만든다."""
    nr.template_network_12()
    ej.neighbor_layout()
    ec.east_layout()
    p0.east_geometry()


def pool_of(workers: int) -> ProcessPoolExecutor:
    return ProcessPoolExecutor(max_workers=max(1, workers), mp_context=multiprocessing.get_context("spawn"), initializer=ref._worker_init)


def run_policy(name: str, out: Path, work: Path, workers: int, cfg: dict, models: Path | None = None, *, worker=run_job, executor=None) -> dict:
    out = Path(out)
    code = code_fingerprint()
    (out / "runs").mkdir(parents=True, exist_ok=True)                  # 작업자들이 동시에 처음 만들지 않게
    s2e._freeze(out / "experiment_fix1.json", code_record())         # experiment.json은 이전 코드의 기록이라 그대로 둔다
    jobs = plan_jobs(check_policy(name), cfg, out, work, models)
    todo = [{**j, "labels": cfg["neighbor_labels"], "code": code} for j in jobs if not finished(j, code, (cfg["correction"]["earlier_code"],))]
    ref.stamp(f"NN12 {name}: {len(jobs)}판 중 새로 {len(todo)}판 → {out}")
    if worker is run_job and executor is None and todo:
        prepare_caches()
    failed, done = [], 0
    if todo:
        pool = executor or pool_of(workers)
        try:
            futures = {pool.submit(worker, j): j for j in todo}
            for n, future in enumerate(as_completed(futures), 1):
                r = future.result()
                if r["ok"]:
                    done += 1
                else:
                    failed.append(r)
                    w.write(Path(futures[future]["target"]).with_name(r["tag"] + ".failed.json"), r)
                if n % 10 == 0 or n == len(todo):
                    ref.stamp(f"  {n}/{len(todo)} 끝, 실패 {len(failed)}, 마지막 {r['tag']} {r['seconds']:.0f}초")
        finally:
            if executor is None:
                pool.shutdown()
    status = {"total": len(jobs), "done_now": done, "failed": failed, "updated_unix": time.time()}
    s2e._write(out / "status.json", status)
    return status


# ----------------------------------------------------------------------------- 실행 전 점검
def check_job(name: str, cfg: dict, out: Path, work: Path, models: Path | None) -> dict:
    window, seed = cfg["check"]["window"], cfg["check"]["seed"]
    tag = run_tag(window[0], window[1], seed)
    job = {"name": name, "tag": tag, "date": window[0], "hour": int(window[1]), "eval_seed": seed, "work": str(Path(work) / name / tag), "target": str(Path(out) / name / tag), "save_inputs": False}
    if name in MODELS:
        spec = cfg["models"][name]
        chosen = spec["seeds"][0]
        job.update(model_seed=chosen, model_identity=spec["identities"][str(chosen)], model_dir=str(Path(models) / f"seed_{chosen}"))
    return job


def reproduce_job(cfg: dict, out: Path, work: Path) -> dict:
    window, seed = cfg["check"]["reproduce"]["window"], cfg["check"]["reproduce"]["seed"]
    tag = run_tag(window[0], window[1], seed)
    return {"name": "P2", "tag": tag, "date": window[0], "hour": int(window[1]), "eval_seed": seed, "work": str(Path(work) / "P2" / tag), "target": str(Path(out) / "P2" / tag), "save_inputs": False}


def values_of(m: dict) -> dict:
    return {**{f"{a}.{b}": m[a][b] for a, b in SAME}, "action_counts": {str(k): int(v) for k, v in sorted(m["action_counts"].items())}}


def stored_p2(reference: Path, cfg: dict) -> dict:
    window, seed = cfg["check"]["reproduce"]["window"], cfg["check"]["reproduce"]["seed"]
    return w.read(Path(reference) / EVALUATION_FP / "windows" / f"{window[0]}_{int(window[1]):02d}" / f"p2_eval{seed}" / "metrics.json")


def check_rows(name: str, m: dict) -> dict:
    """코드 확인용 판의 건강 점검(값의 좋고 나쁨은 보지 않는다)."""
    stats = m["nn12"]["stats"]
    rules = m["guard"]["changed_by_rules"]
    out = {"finished": m["evaluation_fingerprint"] == EVALUATION_FP and m["network12_fingerprint"] == NETWORK_FP, "planned": stats.get("planned", 0) > 0, "rules_not_changed": sum(rules.values()) == 0}
    if name in MODELS:
        out["planner_free"] = m["nn12"]["planner_free"] is True
    if name in ("N5", "TM4"):
        out["timetable_same"] = stats.get("timetable_checked", 0) > 0 and stats.get("timetable_mismatch", 0) == 0
    return out


def check(name: str, models: Path | None, reference_p2: Path | None, out: Path, work: Path, workers: int, cfg: dict) -> dict:
    """정책 한 판 + P2 재현 한 판. 12개 지도는 맥(arm64)과 Colab(x86-64)의 값이 달라 저장된 P2와의 대조는 Colab에서만 한다(reference_p2)."""
    out, work = Path(out), Path(work)
    if ne.evaluation_fingerprint() != EVALUATION_FP or nr._code_fingerprint() != NETWORK_FP or environment_fingerprint() != ENVIRONMENT_FP:
        raise ValueError("12개 지도·평가·환경 지문이 기존 기준 실행과 다릅니다.")
    out.mkdir(parents=True, exist_ok=True)
    code = code_fingerprint()
    jobs = [check_job(check_policy(name), cfg, out, work, models), reproduce_job(cfg, out, work)]
    todo = [{**j, "labels": cfg["neighbor_labels"], "code": code} for j in jobs if not finished(j, code)]
    prepare_caches()
    if todo:
        with pool_of(min(workers, len(todo))) as pool:
            done = list(pool.map(run_job, todo))
        bad = [d for d in done if not d["ok"]]
        if bad:
            raise RuntimeError(f"점검 실행이 실패했습니다: {bad}")
    own = w.read(Path(jobs[0]["target"]) / "metrics.json")
    p2 = w.read(Path(jobs[1]["target"]) / "metrics.json")
    checks = check_rows(name, own)
    if reference_p2 is not None:
        stored = stored_p2(reference_p2, cfg)
        checks["p2_reproduced"] = all(p2[a][b] == stored[a][b] for a, b in SAME) and p2["action_counts"] == stored["action_counts"]
    report = {"policy": name, "values": values_of(own), "p2_values": values_of(p2), "seconds": {"policy": own["nn12"]["job_seconds"], "p2": p2["nn12"]["job_seconds"]}, "stats": own["nn12"]["stats"], "code": code,
              "checks": checks, "ok": all(checks.values())}
    s2e._write(out / f"check_{name}.json", report)
    ref.stamp(f"NN12 점검 {name}: {'통과' if report['ok'] else '실패'} {checks} | 판당 {report['seconds']}")
    return report


# ----------------------------------------------------------------------------- 고친 코드 재확인(사전 등록 정정 기록)
def unzipped_digest(path: Path) -> str:
    with gzip.open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def arrays_equal(a: Path, b: Path) -> bool:
    with np.load(a) as x, np.load(b) as y:
        return set(x.files) == set(y.files) and all(np.array_equal(x[k], y[k]) for k in x.files)


def same_as_stored(new_dir: Path, stored_dir: Path) -> dict:
    """같은 판의 두 결과를 비교한다. 걸린 시간만 빼고 전부 같아야 한다. 값은 돌려주지 않는다(판정 전이라 읽지 않는다)."""
    new, old = w.read(Path(new_dir) / "metrics.json"), w.read(Path(stored_dir) / "metrics.json")
    body = lambda m: {k: v for k, v in m.items() if k not in ("seconds", "nn12")}  # noqa: E731
    record = lambda m: {k: m["nn12"].get(k) for k in ("name", "model_seed", "model_identity", "planner_free", "stats")}  # noqa: E731
    same = {"metrics": body(new) == body(old), "record": record(new) == record(old), "vehicles": unzipped_digest(Path(new_dir) / "vehicles.csv.gz") == unzipped_digest(Path(stored_dir) / "vehicles.csv.gz")}
    if (Path(stored_dir) / "inputs.npz").exists():
        same["inputs"] = (Path(new_dir) / "inputs.npz").exists() and arrays_equal(Path(new_dir) / "inputs.npz", Path(stored_dir) / "inputs.npz")
    return same


def recheck(name: str, models: Path | None, out: Path, work: Path, cfg: dict) -> dict:
    """끝난 판 하나를 지금 코드로 다시 돌려 저장된 결과와 소수점까지 같은지 본다. 다르면 나머지 판을 돌리지 않는다."""
    fix = cfg["correction"]
    if name not in fix["rerun_policies"]:
        raise ValueError(f"다시 돌릴 정책이 아닙니다: {name}")
    if ne.evaluation_fingerprint() != EVALUATION_FP or nr._code_fingerprint() != NETWORK_FP or environment_fingerprint() != ENVIRONMENT_FP:
        raise ValueError("12개 지도·평가·환경 지문이 기존 기준 실행과 다릅니다.")
    out, work = Path(out), Path(work)
    window, seed = fix["recheck"]["window"], fix["recheck"]["seed"]
    tag = run_tag(window[0], window[1], seed)
    planned = next(j for j in plan_jobs(name, cfg, out, work, models) if j["tag"] == tag)
    stored = Path(planned["target"])
    if w.read(stored / "metrics.json")["nn12"]["code"] != fix["earlier_code"]:
        raise ValueError(f"비교할 판이 이전 코드의 결과가 아닙니다: {stored}")
    code = code_fingerprint()
    job = {**planned, "target": str(out / "check_fix1" / tag), "labels": cfg["neighbor_labels"], "code": code}
    if not finished(job, code):
        prepare_caches()
        with pool_of(1) as pool:
            done = list(pool.map(run_job, [job]))
        if not done[0]["ok"]:
            raise RuntimeError(f"점검 실행이 실패했습니다: {done}")
    same = same_as_stored(Path(job["target"]), stored)
    report = {"policy": name, "tag": tag, "code": code, "earlier_code": fix["earlier_code"], "same": same, "seconds": w.read(Path(job["target"]) / "metrics.json")["nn12"]["job_seconds"], "ok": all(same.values())}
    s2e._write(out / "check_fix1" / f"recheck_{name}.json", report)
    ref.stamp(f"NN12 재확인 {name}: {'같음' if report['ok'] else '다름'} {same}")
    return report


# ----------------------------------------------------------------------------- 판정(SUMO 없음)
def read_many(paths: dict) -> dict:
    """작은 파일 여러 개를 스레드로 읽는다(Drive에서 하나씩 읽으면 느리다). 없는 파일은 빠진다."""
    def one(item):
        key, path = item
        try:
            return key, w.read(path)
        except FileNotFoundError:
            return key, None
    with ThreadPoolExecutor(16) as pool:
        got = list(pool.map(one, paths.items()))
    return {k: v for k, v in got if v is not None}


def reference_paths(name: str, cfg: dict, joint2: Path, pred: Path, b12: Path) -> dict:
    out = {}
    for date, hour in cfg["windows"]:
        window = f"{date}_{int(hour):02d}"
        for seed in cfg["seeds"]:
            if name == "FIXC":
                base = Path(joint2) / EVALUATION_FP / "windows" / window / f"fixedc_eval{seed}"
            elif name == "P2":
                base = Path(joint2) / EVALUATION_FP / "windows" / window / f"p2_eval{seed}"
            elif name == "P0":
                base = Path(pred) / EVALUATION_FP / "windows" / window / f"pred_p0_eval{seed}"
            elif name == "B12":
                base = Path(b12) / EVALUATION_FP / "windows" / window / f"dqn_seed{seed}_{cfg['b12_models'][str(seed)]}_eval{seed}"
            else:
                raise ValueError(name)
            out[(window, seed)] = base / "metrics.json"
    return out


def policy_paths(name: str, cfg: dict, root: Path) -> dict:
    return {(f"{date}_{int(hour):02d}", seed): Path(root) / name / "runs" / run_tag(date, hour, seed) / "metrics.json" for date, hour in cfg["windows"] for seed in cfg["seeds"]}


def weighted_delay(rows) -> float:
    total = math.fsum(m["window_204820"]["mean_delay_s"] * m["window_204820"]["vehicles"] for m in rows)
    return total / sum(m["window_204820"]["vehicles"] for m in rows)


def movement_delay(rows, move: tuple[str, str]) -> float | None:
    key = f"{move[0]}->{move[1]}"
    pairs = [m["window_204820"]["by_movement"][key] for m in rows if key in m["window_204820"]["by_movement"]]
    n = sum(p["vehicles"] for p in pairs)
    return math.fsum(p["vehicles"] * p["mean_delay_s"] for p in pairs) / n if n else None


def per_window(runs: dict, getter) -> dict[str, float]:
    acc: dict[str, list[float]] = {}
    for (window, _), m in runs.items():
        acc.setdefault(window, []).append(float(getter(m)))
    return {k: statistics.fmean(v) for k, v in acc.items()}


def segment(m: dict) -> float:
    return m["window_204820"]["mean_delay_s"]


def network(m: dict) -> float:
    return m["network"]["mean_delay_s"]


def has_network(runs: dict) -> bool:
    return all("network" in m for m in runs.values())


def summary_row(name: str, runs: dict, fixed: float, cfg: dict) -> dict:
    rows = list(runs.values())
    weighted = weighted_delay(rows)
    row = {"weighted_delay_s": weighted, "reduction_vs_fixc_pct": 100.0 * (fixed - weighted) / fixed, "window_mean_s": statistics.fmean(segment(m) for m in rows),
           "runs": len(rows), "long_delays_per_run": statistics.fmean(m["window_204820"]["long_delays"] for m in rows), "unfinished_per_run": statistics.fmean(m["window_204820"]["unfinished"] for m in rows),
           "left_east_s": movement_delay(rows, wc.LEFTS["left_east"]), "left_south_s": movement_delay(rows, wc.LEFTS["left_south"]), "collisions_204820": sum(m.get("collisions_in_204820", 0) for m in rows),
           "collisions_total": sum(m.get("collisions_total", 0) for m in rows)}
    evening = [m for (window, _), m in runs.items() if int(window.split("_")[1]) in BANDS["저녁 16~18시"]]
    row["evening_east_west_s"] = movement_delay(evening, ej.EAST_TO_WEST)
    if has_network(runs):
        row.update(network_mean_s=statistics.fmean(network(m) for m in rows), via_204822_mean_s=statistics.fmean(m["via_204822"]["mean_delay_s"] for m in rows),
                   network_long_per_run=statistics.fmean(m["network"]["long_delays"] for m in rows), not_inserted_per_run=statistics.fmean(m["network"]["not_inserted"] for m in rows),
                   collisions_204822=sum(m.get("collisions_204822", 0) for m in rows))
    if name in POLICIES:
        stats = [m["nn12"]["stats"] for m in rows]
        planned = sum(s.get("planned", 0) for s in stats)
        row.update(decisions_per_run=planned / len(rows), rules_changed=sum(sum(m["guard"]["changed_by_rules"].values()) for m in rows), seconds_per_run=statistics.fmean(m["nn12"]["job_seconds"] for m in rows))
        if name in MODELS:
            row["z5_cells_per_decision"] = sum(s.get("z5_cells", 0) for s in stats) / max(planned, 1)
            seeds = sorted({m["nn12"]["model_seed"] for m in rows})
            row["per_model"] = {str(s): {"runs": sum(m["nn12"]["model_seed"] == s for m in rows), "weighted_delay_s": weighted_delay([m for m in rows if m["nn12"]["model_seed"] == s])} for s in seeds}
        if name in ("N5", "TM4"):
            row["timetable_mismatch"] = sum(s.get("timetable_mismatch", 0) for s in stats)
            row["timetable_past_midnight"] = sum(s.get("timetable_past_midnight", 0) for s in stats)
    return row


def paired(a: dict, b: dict, drop: str | None = None) -> dict:
    """조각 짝 비교. 양수면 a가 짧다."""
    keep = {k: v for k, v in a.items() if k != drop}, {k: v for k, v in b.items() if k != drop}
    if set(keep[0]) != set(keep[1]):
        raise ValueError("짝 비교할 두 정책의 조각이 다릅니다.")
    return ej._paired(*keep)


def reading(table: dict, rules: dict) -> dict:
    """사전 등록의 읽기. 후보 = N5·Rn 중 구간 지연이 짧은 쪽."""
    best = min(MODELS, key=lambda c: table[c]["weighted_delay_s"])
    gap = table[best]["weighted_delay_s"] - table["P2"]["weighted_delay_s"]
    if gap <= float(rules["p2_gap_s"]):
        action = "continue_single"
    elif table[best]["weighted_delay_s"] < table["B12"]["weighted_delay_s"]:
        action = "labels_on_12map"
    else:
        action = "stop"
    transfer = {}
    for model, partner in PARTNER.items():
        behind = table[model]["weighted_delay_s"] - table[partner]["weighted_delay_s"]
        transfer[model] = {"partner": partner, "behind_s": behind, "transferred": behind <= float(rules["transfer_gap_s"]),
                           "partner_longer_than_p0": table[partner]["weighted_delay_s"] > table["P0"]["weighted_delay_s"]}
    return {"best": best, "gap_to_p2_s": gap, "action": action, "transfer": transfer}


def wrong_code_runs(data: dict, code: str, fix: dict) -> list[str]:
    """정정 기록대로인가: N5·TM4의 23시 판만 고친 코드이고 나머지는 이전 코드다."""
    wrong = []
    for name in POLICIES:
        for (window, seed), m in data[name].items():
            rerun = name in fix["rerun_policies"] and int(window.split("_")[1]) == fix["rerun_hour"]
            if m["nn12"]["code"] != (code if rerun else fix["earlier_code"]):
                wrong.append(f"{name} {window} seed {seed}")
    return wrong


def collect_report(root: Path, joint2: Path, pred: Path, b12: Path, cfg: dict) -> dict:
    expected = len(cfg["windows"]) * len(cfg["seeds"])
    data = {name: read_many(policy_paths(name, cfg, root)) for name in POLICIES}
    missing = {k: expected - len(v) for k, v in data.items() if len(v) != expected}
    if missing:
        return {"experiment": EXPERIMENT, "status": "incomplete", "missing": missing}
    for name in REFERENCES:
        data[name] = read_many(reference_paths(name, cfg, joint2, pred, b12))
        if len(data[name]) != expected:
            raise ValueError(f"기준 실행 {name}이 {expected}판이 아닙니다: {len(data[name])}")
    code = code_fingerprint()
    wrong = wrong_code_runs(data, code, cfg["correction"])
    if wrong:
        raise ValueError(f"정정 기록과 다른 코드로 돈 판이 {len(wrong)}개 있습니다: {wrong[:5]}")
    for name, runs in data.items():
        for key, m in runs.items():
            if m.get("evaluation_fingerprint") != EVALUATION_FP or m.get("network12_fingerprint") != NETWORK_FP:
                raise ValueError(f"평가 지문이 다릅니다: {name} {key}")
    for name in POLICIES:
        for key, m in data[name].items():
            if m["network"]["vehicles"] != data["FIXC"][key]["network"]["vehicles"]:
                raise ValueError(f"같은 조각·seed의 도로망 차 수가 다릅니다: {name} {key}")
    fixed = weighted_delay(list(data["FIXC"].values()))
    order = ("FIXC", "B12", "P0", "P2", "M4", "TM4", "Rn", "N5")
    table = {name: summary_row(name, data[name], fixed, cfg) for name in order}
    seg = {name: per_window(data[name], segment) for name in order}
    net = {name: per_window(data[name], network) for name in order if has_network(data[name])}
    via = {name: per_window(data[name], lambda m: m["via_204822"]["mean_delay_s"]) for name in net}
    unstable = cfg["rules"]["unstable_window"]
    comparisons = [("N5", "FIXC"), ("Rn", "FIXC"), ("TM4", "FIXC"), ("M4", "FIXC"), ("N5", "TM4"), ("Rn", "M4"), ("N5", "Rn"), ("N5", "P2"), ("Rn", "P2"), ("N5", "B12"), ("Rn", "B12"), ("TM4", "P2"), ("M4", "P0")]
    pairs = {}
    for a, b in comparisons:
        row = {"segment": paired(seg[a], seg[b]), "segment_without_unstable": paired(seg[a], seg[b], unstable)}
        if a in net and b in net:
            row.update(network=paired(net[a], net[b]), via_204822=paired(via[a], via[b]), network_without_unstable=paired(net[a], net[b], unstable))
        pairs[f"{a}-{b}"] = row
    bands = {}
    for label, hours in BANDS.items():
        keys = [k for k in seg["FIXC"] if int(k.split("_")[1]) in hours]
        bands[label] = {name: {"segment_vs_fixc_s": statistics.fmean(seg["FIXC"][k] - seg[name][k] for k in keys), "network_vs_fixc_s": statistics.fmean(net["FIXC"][k] - net[name][k] for k in keys) if name in net else None}
                        for name in order if name != "FIXC"}
    return {"experiment": EXPERIMENT, "status": "done", "table": table, "pairs": pairs, "bands": bands, "reading": reading(table, cfg["rules"]), "code": {"rerun": code, "earlier": cfg["correction"]["earlier_code"]}, "config_sha256": w.file_hash(CONFIG),
            "evaluation_fingerprint": EVALUATION_FP, "updated_unix": time.time(), "note": cfg["reporting"]}


ACTION_TEXT = {"continue_single": "후보가 P2와 2.2초 안: 단독 지도 학습을 계속하고 12개 지도는 평가로 쓴다", "labels_on_12map": "후보가 P2보다 2.2초 넘게 길지만 B12보다 짧다: 3단계는 하되 12개 지도 상태에서 라벨 라운드를 더한다",
               "stop": "후보가 B12와 같거나 길다: 3단계를 멈추고 학습 지도를 사용자·교수님과 다시 정한다"}
NAMES = {"FIXC": "보정 고정신호", "B12": "DQN B12", "P0": "P0", "P2": "P2", "M4": "M4(무작위 세계 계획기)", "TM4": "TM4(무리 세계 계획기)", "Rn": "Rn(무작위 세계 흉내)", "N5": "N5(무리 세계 흉내)"}


def render(report: dict) -> str:
    if report["status"] != "done":
        return f"# NN12 — 아직 끝나지 않았습니다\n\n남은 판: {json.dumps(report['missing'], ensure_ascii=False)}\n"
    f = lambda x, nd=2: "-" if x is None else f"{x:+.{nd}f}"  # noqa: E731
    v = lambda x, nd=2: "-" if x is None else f"{x:.{nd}f}"  # noqa: E731
    lines = ["# NN12 — 신경망만 모델을 12개 지도에서 잰 결과(46조각 × seed 5개)", "", "주 지표는 석천사거리 구간 지연(차 가중)이다. 흉내 값은 강화학습 값이 아니다.", "",
             "| 제어 | 구간 지연 | 고정신호 대비 | 도로망 지연 | 이웃 지나는 차 | 300초 초과(구간, 판당) | 동 좌회전 | 남 좌회전 | 저녁 동→서 직진 | 충돌(석천사거리/이웃/전체) |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, r in report["table"].items():
        lines.append(f"| {NAMES[name]} | {r['weighted_delay_s']:.2f} | {-r['reduction_vs_fixc_pct']:.1f}% | {v(r.get('network_mean_s'))} | {v(r.get('via_204822_mean_s'))} | {r['long_delays_per_run']:.1f} | {v(r['left_east_s'], 1)} | {v(r['left_south_s'], 1)} | "
                     f"{v(r['evening_east_west_s'], 1)} | {r['collisions_204820']}/{r.get('collisions_204822', '-')}/{r['collisions_total']} |")
    lines += ["", "조각 짝 비교(46조각, +면 앞쪽이 짧음)", "", "| 비교 | 구간 | t95 | 앞쪽이 나은 조각 | 흔들리는 조각 뺌 | 도로망 | t95 | 앞쪽이 나은 조각 |", "|---|---:|---|---:|---:|---:|---|---:|"]
    for key, row in report["pairs"].items():
        s, n, x = row["segment"], row.get("network"), row["segment_without_unstable"]
        net_cells = f"{f(n['mean'])} | [{f(n['t95'][0])}, {f(n['t95'][1])}] | {n['positive']}/{n['n']}" if n else "- | - | -"
        lines.append(f"| {key.replace('-', ' − ')} | {f(s['mean'])} | [{f(s['t95'][0])}, {f(s['t95'][1])}] | {s['positive']}/{s['n']} | {f(x['mean'])} | {net_cells} |")
    labels = list(report["bands"])
    lines += ["", "시간대별 보정 고정신호 대비 구간 지연(초, +면 짧음)", "", "| 제어 | " + " | ".join(labels) + " |", "|---|" + "---:|" * len(labels)]
    for name in report["table"]:
        if name != "FIXC":
            lines.append(f"| {name} | " + " | ".join(f(report["bands"][b][name]["segment_vs_fixc_s"]) for b in labels) + " |")
    lines += ["", "시간대별 보정 고정신호 대비 도로망 지연(초, +면 짧음)", "", "| 제어 | " + " | ".join(labels) + " |", "|---|" + "---:|" * len(labels)]
    for name in report["table"]:
        if name != "FIXC" and report["bands"][labels[0]][name]["network_vs_fixc_s"] is not None:
            lines.append(f"| {name} | " + " | ".join(f(report["bands"][b][name]["network_vs_fixc_s"]) for b in labels) + " |")
    lines += ["", "| 정책 | 판당 결정 | 규칙이 바꾼 판단 | 시간표 불일치 | 학습 범위 밖 입력(결정당) | 판당 초 | 모델별 구간 지연 |", "|---|---:|---:|---:|---:|---:|---|"]
    for name in POLICIES:
        r = report["table"][name]
        each = " / ".join(f"{x['weighted_delay_s']:.2f}({x['runs']}판)" for x in r["per_model"].values()) if "per_model" in r else "-"
        lines.append(f"| {name} | {r['decisions_per_run']:.1f} | {r['rules_changed']} | {r.get('timetable_mismatch', '-')} | {v(r.get('z5_cells_per_decision'), 3)} | {r['seconds_per_run']:.0f} | {each} |")
    g = report["reading"]
    lines += ["", f"**읽기:** 후보 {g['best']}, P2보다 {f(g['gap_to_p2_s'])}초(+면 김). {ACTION_TEXT[g['action']]}."]
    for model, t in g["transfer"].items():
        state = "옮겨짐" if t["transferred"] else "옮겨지지 않음"
        extra = ", 짝 계획기 자체가 P0보다 김" if t["partner_longer_than_p0"] else ""
        lines.append(f"- {model}: {t['partner']}보다 {f(t['behind_s'])}초(+면 김) → {state}{extra}.")
    lines += ["", f"- {report['note']}", "- 900판은 이전 코드, N5·TM4의 23시 20판은 기록용 대조를 고친 코드로 돌렸다(사전 등록 정정 기록). 판단·입력·평가 코드는 같다."]
    return "\n".join(lines) + "\n"


def save_report(out: Path, report: dict) -> None:
    out = Path(out)
    s2e._write(out / "judge.json", report)
    (out / "summary.md").write_text(render(report), encoding="utf-8")


# ----------------------------------------------------------------------------- 명령
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("make-config")
    p.add_argument("--models-N5", required=True)
    p.add_argument("--models-Rn", required=True)
    p = sub.add_parser("check")
    p.add_argument("--policy", required=True)
    p.add_argument("--models")
    p.add_argument("--reference-p2", help="explore_joint2 폴더(저장된 P2). Colab에서만 준다(맥은 소수점 차이로 값이 다르다)")
    p.add_argument("--out", required=True)
    p.add_argument("--work", required=True)
    p.add_argument("--workers", type=int, default=2)
    p = sub.add_parser("run")
    p.add_argument("--policy", required=True)
    p.add_argument("--models")
    p.add_argument("--out", required=True)
    p.add_argument("--work", required=True)
    p.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1))
    p = sub.add_parser("recheck")
    p.add_argument("--policy", required=True)
    p.add_argument("--models")
    p.add_argument("--out", required=True)
    p.add_argument("--work", required=True)
    p = sub.add_parser("judge")
    for name in ("root", "joint2", "pred", "b12", "out"):
        p.add_argument(f"--{name}", required=True)
    args = parser.parse_args()
    w.configure()
    torch.set_num_threads(1)
    if args.cmd == "make-config":
        print(json.dumps(make_config({"N5": Path(args.models_N5), "Rn": Path(args.models_Rn)}), ensure_ascii=False)[:1200])
        return 0
    cfg = config()
    if args.cmd == "check":
        r = check(args.policy, Path(args.models) if args.models else None, Path(args.reference_p2) if args.reference_p2 else None, Path(args.out), Path(args.work), args.workers, cfg)
        return 0 if r["ok"] else 2
    if args.cmd == "recheck":
        r = recheck(args.policy, Path(args.models) if args.models else None, Path(args.out), Path(args.work), cfg)
        return 0 if r["ok"] else 2
    if args.cmd == "run":
        status = run_policy(args.policy, Path(args.out), Path(args.work), args.workers, cfg, Path(args.models) if args.models else None)
        return 1 if status["failed"] else 0
    report = collect_report(Path(args.root), Path(args.joint2), Path(args.pred), Path(args.b12), cfg)
    save_report(Path(args.out), report)
    print(render(report))
    return 0 if report["status"] == "done" else 3


if __name__ == "__main__":
    raise SystemExit(main())
