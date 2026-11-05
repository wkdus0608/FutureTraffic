"""[추가 탐색] NN1 — 신경망만 모델의 0·1단계(계획서 docs/신경망_강화학습_계획서_20261102.md, 사전 등록 results/explore/nn1_20261102/사전등록.md).

0단계: 실행 경로에서 계획기를 뺀 실행기. 계획기를 부르면 멈춘다. 입력은 감지 80칸, 무리 세계는 여기에 동쪽 도착 추정 16칸 + 이웃 시간표 24칸.
1단계: 출발 모델 7종을 같은 84상황에서 잰다. 모두 학습 6일로 만들고 입력 표준화·학습 횟수는 같다. 제어 60분과 꼬리 15분을 모두 신경망이 맡는다.
  무리 세계 N1(TM4 흉내)·N2(t20 흉내)·N4(N1 + 계획기 없는 학생)·N5(t20 흉내 + 자료 강화)
  무작위 세계 Rn(S2A_pre 흉내)·RnS(Rn + 계획기 없는 학생)·Rn2(S2A_pre 흉내 + 자료 강화)
자료 강화: 선생님이 운전한 판의 모든 결정을 기록하고, 신경망이 운전하며 선생님이 라벨을 붙이는 라운드를 3번 돈다. 어느 라운드를 쓸지는 검증 날짜의 한 시간 결과로 정한다.
잠긴 파일·이전 실험 코드와 결과는 고치지 않는다. 계획기를 막는 일과 입력 계산은 실행하는 동안 이 파일 안에서만 끼운다.

사용(FOCUSED_RULESET=pet1 FOCUSED_NORTH_EXT=240):
  python src/explore_nn1.py make-config --workers 8                   (맥, SUMO 없음: 새 상황의 차량 파일 SHA 고정)
  python src/explore_nn1.py train --world P --data <Drive>/explore_platoon1/data --teacher <Drive>/explore_platoon1/students/seed_42 --out <Drive>/explore_nn1/P/base/models
  python src/explore_nn1.py check-paths --world P --model <모델 폴더> --out <dir> --work /content/n1c --situations 1
  python src/explore_nn1.py check-logs --world P --teacher <…> --teacher-run <…/t20_m0> --model <모델 폴더> --out <dir> --work /content/n1l
  python src/explore_nn1.py evaluate --world P --candidates N1,N2,N4 --models <…/models> --out <…/eval> --work /content/n1 --workers 8
  python src/explore_nn1.py augment --world P --teacher <…> --out <Drive>/explore_nn1/P/aug --work /content/n1a --workers 8
  python src/explore_nn1.py judge --root <Drive>/explore_nn1 --platoon0 <…> --platoon1 <…> --platoon1c <…> --pbase <…> --baseline <…> --stage2 <…> --out <…/judge>
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import multiprocessing
import os
import platform
import shutil
import statistics
import sys
import tempfile
import time
import uuid
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import arena3_common as ac  # noqa: E402
import arena3_pi as pi  # noqa: E402
import arena3_referee as ref  # noqa: E402
import arena3_stage2_eval as s2e  # noqa: E402
import explore_label0 as lb  # noqa: E402
import explore_platoon0 as p0  # noqa: E402
import explore_platoon1 as pt  # noqa: E402
import explore_platoon1_learn as pl  # noqa: E402
import explore_platoon1c as pc  # noqa: E402
import explore_platoon_base as pb  # noqa: E402
import explore_pred as ep  # noqa: E402
import explore_weight as w  # noqa: E402
import explore_weight_core as wc  # noqa: E402
import focused_platoon as fp  # noqa: E402
from focused_data import TARGET_TLS  # noqa: E402

EXPERIMENT = "NN1_V1"
CONFIG = ROOT / "config/explore_nn1_v1.json"
PREREG = ROOT / "results/explore/nn1_20261102/사전등록.md"
KIND_GREEN, KIND_GAIN = "nn1-green-v1", "nn1-gain-v1"
GREENS = 4
SENSED = 80
COLUMNS = {"P": np.r_[np.arange(0, SENSED), np.arange(127, 167)], "R": np.arange(0, SENSED)}
SCALE = 1000.0
MAX_EPISODE_S = 3600.0
TRAIN_DATES = ("20220812", "20220822", "20220907", "20220926", "20221030", "20220909")
VALID_DATES = ("20220928", "20220814")
EVAL_PERIODS = ("0730", "1000", "1230", "1500", "1730", "2100")
AUG_BANDS = {"am_peak": (2, tuple(f"{h:02d}{m:02d}" for h in (7, 8) for m in (0, 15, 30, 45))),
             "pm_peak": (2, tuple(f"{h:02d}{m:02d}" for h in (17, 18) for m in (0, 15, 30, 45))),
             "morning": (1, tuple(f"{h:02d}{m:02d}" for h in (6, 9, 10, 11) for m in (0, 15, 30, 45))),
             "noon": (1, tuple(f"{h:02d}{m:02d}" for h in (12, 13, 14) for m in (0, 15, 30, 45))),
             "afternoon": (1, tuple(f"{h:02d}{m:02d}" for h in (15, 16) for m in (0, 15, 30, 45))),
             "evening": (1, tuple(f"{h:02d}{m:02d}" for h in (19, 20, 21) for m in (0, 15, 30, 45)))}
AUG_SALT = "NN1_AUG_SLOTS"
SOURCES = ("src/explore_nn1.py", "src/explore_label0.py", "src/explore_platoon1c.py", "src/explore_platoon_base.py", "src/arena3_pi_learn.py",
           "src/arena3_stage2_eval.py", "config/explore_nn1_v1.json", "results/explore/nn1_20261102/사전등록.md",
           *pl.eval_sources(), *pi.sources("R1"))


TRAINING = {"seeds": [42, 43, 44], "network": [64, 64], "batch_size": 64, "learning_rate": 0.001, "weight_decay": 0.0001,
            "epochs_imitation": 30, "epochs_student": 30, "student_label_end_weight": 25.0, "student_change_share": 0.10,
            "confidence_floor": 0.1, "confidence_clip": [0.25, 4.0], "grad_clip": 10.0, "rounds": 3, "margin_clip": [0.25, 4.0],
            "select_tolerance_s": 0.2}
CANDIDATES = {"P": {"N1": {"kind": "imitate", "teacher": "TM4", "parts": 1}, "N2": {"kind": "imitate", "teacher": "t20", "parts": 2},
                    "N4": {"kind": "student", "base": "N1", "parts": 4}, "N5": {"kind": "aug", "teacher": "t20", "parts": 3}},
              "R": {"Rn": {"kind": "imitate", "teacher": "S2A_pre", "parts": 1}, "RnS": {"kind": "student", "base": "Rn", "parts": 4},
                    "Rn2": {"kind": "aug", "teacher": "S2A_pre", "parts": 3}}}
RULES = {"tie_s": 0.3, "planner_gap_s": 0.5, "stop_reduction_pct": 22.0, "basic_reduction_pct": 23.9,
         "pairs": {"P": [["N2", "N1"], ["N4", "N1"], ["N5", "N2"], ["N1", "TM4"], ["N2", "TM4"], ["N4", "TM4"], ["N5", "TM4"], ["N2", "t20"], ["N5", "t20"]],
                   "R": [["RnS", "Rn"], ["Rn2", "Rn"], ["Rn", "M4"], ["RnS", "M4"], ["Rn2", "M4"], ["Rn", "S2A_pre"], ["Rn2", "S2A_pre"]]}}
CHECK_SITUATIONS = ["20220811_0730_E1", "20220820_0730_E2", "20220813_1000_E1", "20220823_1000_E2", "20220819_1230_E1", "20220906_1230_E2",
                    "20220820_1500_E1", "20220911_1500_E2", "20220823_1730_E1", "20220811_1730_E2", "20220906_2100_E1", "20220813_2100_E2"]


# ----------------------------------------------------------------------------- 설정
def config() -> dict:
    cfg = w.read(CONFIG)
    if cfg.get("experiment") != EXPERIMENT:
        raise ValueError("NN1 설정이 아닙니다.")
    for key, path in (("prereg_sha256", PREREG), ("lock_sha256", ac.CONFIG), ("platoon0_config_sha256", p0.CONFIG), ("platoon1_config_sha256", pt.CONFIG),
                      ("platoon1c_config_sha256", pc.CONFIG), ("pi_config_sha256", pi.config_path("R1"))):
        if cfg.get(key) != w.file_hash(path):
            raise ValueError(f"NN1 설정을 만든 때와 파일이 다릅니다: {Path(path).relative_to(ROOT)}")
    return cfg


def check_world(world: str) -> str:
    if world not in COLUMNS:
        raise ValueError(f"세계는 P 또는 R: {world}")
    return world


def eval_situations(world: str) -> list[dict]:
    return pl.situations(pt.config(), "eval") if check_world(world) == "P" else s2e.situations_for("eval")


def named_situations(cfg: dict, key: str, world: str, round_: int | None = None) -> list[dict]:
    """설정의 새 상황을 그 세계의 차량 파일 SHA와 함께 돌려준다."""
    rows = cfg[key] if round_ is None else [s for s in cfg[key] if s["round"] == round_]
    return [{**{k: s[k] for k in ("date", "period", "tag", "sumo_seed")}, "route_sha256": s["route_sha256"][check_world(world)]} for s in rows]


def aug_situations(rounds: int) -> list[dict]:
    """라운드·날짜마다 첨두 2칸씩(07·17시대) + 나머지 네 묶음 1칸씩, 새 도착 배치. 칸은 결과와 무관한 해시 규칙으로 고른다."""
    if sum(n for n, _ in AUG_BANDS.values()) != 8 or set().union(*(b for _, b in AUG_BANDS.values())) != set(ac.TRAIN_SLOTS):
        raise ValueError("시간 묶음이 학습 칸 전체를 나누지 않습니다.")
    out, index = [], 0
    for r in range(rounds + 1):
        for date in TRAIN_DATES:
            for band, (count, slots) in AUG_BANDS.items():
                ordered = sorted(slots, key=lambda p: hashlib.sha256(f"{AUG_SALT}|{r}|{date}|{band}|{p}".encode()).hexdigest())
                for period in sorted(ordered[:count]):
                    index += 1
                    out.append({**ac.make_situation(date, period, f"NA{r}x{index}"), "round": r, "band": band})
    return out


def validation_situations() -> list[dict]:
    return [ac.make_situation(d, p, f"V{j}") for d in VALID_DATES for p in EVAL_PERIODS for j in (1, 2)]


def make_config(workers: int) -> dict:
    """맥, SUMO 없음. 새 상황의 차량 파일 SHA를 두 세계 모두 고정한다."""
    lock, pcfg, pccfg, picfg = ac.config(), pt.config(), pc.config(), pi.config("R1")
    if set(TRAIN_DATES) | set(VALID_DATES) != set(lock["dates"]["train"]) or set(TRAIN_DATES) & set(VALID_DATES):
        raise ValueError("학습 6일·검증 2일이 ARENA3 학습 8일을 나누지 않습니다.")
    previous = w.read(CONFIG) if CONFIG.exists() else {}
    aug, valid = aug_situations(int(TRAINING["rounds"])), validation_situations()
    rows = [*aug, *valid]
    if previous.get("situations_aug") and previous.get("situations_validation"):
        keep = {ac.situation_id(s): s["route_sha256"] for s in [*previous["situations_aug"], *previous["situations_validation"]]}
    else:
        keep = {}
    if len({ac.situation_id(s) for s in rows}) != len(rows):
        raise ValueError("새 상황 이름이 겹칩니다.")
    need = [s for s in rows if ac.situation_id(s) not in keep]
    shas: dict[str, dict] = {k: dict(v) for k, v in keep.items()}
    if need:
        with tempfile.TemporaryDirectory() as d, p0._pool(workers) as pool:
            for x in ("P", "R"):                     # 같은 상황의 두 세계가 같은 파일 이름을 쓰지 않게 나눈다
                Path(d, x).mkdir()
            for sid, world, sha in pool.map(p0._sha_job, [(s, x, str(Path(d, x))) for s in need for x in ("P", "R")]):
                shas.setdefault(sid, {})[world] = sha
    for s in rows:
        s["route_sha256"] = shas[ac.situation_id(s)]
    used = {x["route_sha256"] for part in ("eval", "confirmation", "bprime_data") for x in lock[f"situations_{part}"]}
    used |= {s["route_sha256"] for part in ("situations_data", "situations_eval", "situations_confirmation") for s in pcfg[part]}
    used |= {s["route_sha256"] for s in picfg["situations_data"]}
    new = [s["route_sha256"][x] for s in rows for x in ("P", "R")]
    if len(set(new)) != len(new) or set(new) & used:
        raise ValueError("새 차량 파일이 서로 같거나 기존 차량 파일과 같습니다.")
    cfg = {
        "experiment": EXPERIMENT,
        "design": "docs/신경망_강화학습_계획서_20261102.md 0·1단계, PROJECT_PROGRESS.md 263절, results/explore/nn1_20261102/사전등록.md",
        "prereg": PREREG.relative_to(ROOT).as_posix(), "prereg_sha256": w.file_hash(PREREG), "lock_sha256": w.file_hash(ac.CONFIG),
        "platoon0_config_sha256": w.file_hash(p0.CONFIG), "platoon1_config_sha256": w.file_hash(pt.CONFIG),
        "platoon1c_config_sha256": w.file_hash(pc.CONFIG), "pi_config_sha256": w.file_hash(pi.config_path("R1")),
        "dates": {"train": list(TRAIN_DATES), "validation": list(VALID_DATES), "eval": lock["dates"]["eval"],
                  "note": "모델을 만들기 전에 고정. 선생님 PT1·S2A_pre는 8일로 학습된 그대로라 검증 날짜가 선생님을 통해 조금 샌다."},
        "teachers": {"P": {"name": "t20", "student_seed": pccfg["students"]["seeds"][0], "model_sha256": pccfg["students"]["model_sha256"][0],
                           "tau": float(pccfg["candidates"]["t20"]["tau"]), "folder": "explore_platoon1/students/seed_42",
                           "stored_run": "t20_m0", "planner_for_costs": "TM4"},
                     "R": {"name": "S2A_pre", "seed": picfg["base_policy"]["seed"], "model_sha256": picfg["base_policy"]["model_sha256"],
                           "folder": "arena3/stage2/S2A_train/seed_42/pretrained", "stored_run": "S2A_pre_m0", "planner_for_costs": "M4"}},
        "candidates": CANDIDATES, "reference_planner": {"P": "TM4", "R": "M4"},
        "input": {"P": {"size": 120, "columns": "감지 0:80 + 동쪽 도착 추정 127:143 + 이웃 시간표 143:167"}, "R": {"size": 80, "columns": "감지 0:80"},
                  "not_included": "계획기 출력(80:127), 남은 제어시간 칸(109)", "scale": "학습 자료의 평균·표준편차로 맞추고 값은 모델 파일에 저장"},
        "training": TRAINING,
        "aug": {"rounds": TRAINING["rounds"], "per_date": 8, "salt": AUG_SALT, "bands": {k: {"count": n, "slots": list(b)} for k, (n, b) in AUG_BANDS.items()},
                "driver": "라운드 k의 판 i는 라운드 k−1 신경망 난수 [42, 43, 44][i % 3]이 운전(라운드 0은 선생님). 꼬리는 P0.",
                "loss_weight": "결정마다 clip(1·2위 계획 비용 차 ÷ 그 자료의 중앙값, 0.25, 4)", "select": "검증 24상황 평균 지연이 가장 짧은 라운드. 더 뒤 라운드가 그 값의 select_tolerance_s 안이면 가장 뒤 라운드"},
        "evaluation": {"models": 3, "situations": 84, "tail": "NN", "planner": "막음(부르면 멈춤)", "input_log": "모델 0의 check_situations 12판"},
        "rules": RULES, "check_situations": CHECK_SITUATIONS,
        "reporting": "반복 사용 날짜의 개발 비교. 흉내 값은 강화학습 값으로 쓰지 않는다. 무리 도착은 모형이다. 성과로 말하는 숫자는 4단계의 새 배치에서 낸다.",
        "situations_aug": aug, "situations_validation": valid,
    }
    CONFIG.write_text(json.dumps(w.plain(cfg), ensure_ascii=False, indent=1) + "\n")
    return {"config": str(CONFIG), "sha256": w.file_hash(CONFIG), "aug": len(aug), "validation": len(valid)}


# ----------------------------------------------------------------------------- 신경망
def scale_of(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, np.float64)
    mean, std = x.mean(axis=0), x.std(axis=0)
    std[std < 1e-6] = 1.0
    return mean, std


class GreenNet(nn.Module):
    """입력(저장된 평균·표준편차로 맞춤) → 64-64 → 녹색 4개의 점수."""

    def __init__(self, size: int, mean=None, std=None, hidden=(64, 64)):
        super().__init__()
        self.register_buffer("mean", torch.zeros(size) if mean is None else torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.ones(size) if std is None else torch.as_tensor(std, dtype=torch.float32))
        self.net = nn.Sequential(nn.Linear(size, hidden[0]), nn.ReLU(), nn.Linear(hidden[0], hidden[1]), nn.ReLU(), nn.Linear(hidden[1], GREENS))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net((x - self.mean) / self.std)


class GainNet(nn.Module):
    """입력(표준화) + 기준 녹색 표시 4칸 → 녹색 4개의 기준 녹색 대비 이득(천 차량·초). 이점 머리를 0으로 시작해 학습 전에는 바꾸지 않는다."""

    def __init__(self, size: int, mean=None, std=None, hidden=(64, 64)):
        super().__init__()
        self.register_buffer("mean", torch.zeros(size) if mean is None else torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.ones(size) if std is None else torch.as_tensor(std, dtype=torch.float32))
        self.trunk = nn.Sequential(nn.Linear(size + GREENS, hidden[0]), nn.ReLU(), nn.Linear(hidden[0], hidden[1]), nn.ReLU())
        self.advantage = nn.Linear(hidden[1], GREENS)
        nn.init.zeros_(self.advantage.weight)
        nn.init.zeros_(self.advantage.bias)

    def forward(self, x: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
        z = (x - self.mean) / self.std
        adv = self.advantage(self.trunk(torch.cat([z, F.one_hot(base, GREENS).to(z.dtype)], dim=1)))
        return adv - adv.gather(1, base[:, None])


def save_model(folder: Path, net: nn.Module, meta: dict) -> str:
    folder = Path(folder)
    folder.parent.mkdir(parents=True, exist_ok=True)
    tmp = folder.with_name(f".{folder.name}.{uuid.uuid4().hex}.tmp")
    tmp.mkdir()
    torch.save({**meta, "state": net.state_dict()}, tmp / "model.pt")
    sha = w.file_hash(tmp / "model.pt")
    s2e._write(tmp / "manifest.json", {**meta, "model_sha256": sha})
    if folder.exists():
        shutil.rmtree(folder)
    os.replace(tmp, folder)
    return sha


def model_identity(folder: Path) -> str:
    folder = Path(folder)
    manifest = w.read(folder / "manifest.json")
    sha = w.file_hash(folder / "model.pt")
    if manifest.get("kind") not in (KIND_GREEN, KIND_GAIN) or sha != manifest["model_sha256"]:
        raise ValueError(f"NN1 모델이 아니거나 파일 해시가 다릅니다: {folder}")
    return sha


def load_model(folder: Path) -> tuple[nn.Module, dict]:
    model_identity(folder)
    manifest = w.read(Path(folder) / "manifest.json")
    payload = torch.load(Path(folder) / "model.pt", map_location="cpu", weights_only=False)
    for key in ("kind", "world", "seed", "input_size", "dataset_fingerprint"):
        if payload.get(key) != manifest.get(key):
            raise ValueError(f"모델 파일과 기록이 다릅니다: {key}")
    if manifest["input_size"] != len(COLUMNS[manifest["world"]]):
        raise ValueError("모델의 입력 크기가 세계의 입력과 다릅니다.")
    cls = GreenNet if manifest["kind"] == KIND_GREEN else GainNet
    net = cls(manifest["input_size"], hidden=tuple(manifest["network"]))
    net.load_state_dict(payload["state"])
    net.eval()
    return net, manifest


# ----------------------------------------------------------------------------- 고르기(순수 계산)
def mask_of(allowed: list[int]) -> np.ndarray:
    mask = np.zeros(GREENS, bool)
    mask[list(allowed)] = True
    return mask


def green_choice(net: GreenNet, x: np.ndarray, allowed: list[int]) -> int:
    with torch.no_grad():
        scores = net(torch.from_numpy(np.asarray(x, np.float32)).view(1, -1))[0].numpy().astype(np.float64)
    if not np.all(np.isfinite(scores)):
        raise ValueError("녹색 점수가 유한하지 않습니다.")
    return int(np.where(mask_of(allowed), scores, -np.inf).argmax())


def student_choice(student: GainNet, x: np.ndarray, base: int, allowed: list[int], tau: float) -> tuple[int, float]:
    """기준 녹색이 아닌 허용 녹색 중 예측 이득이 가장 큰 것을, 그 이득이 문턱 이상일 때만 고른다. (녹색, 가장 큰 예측 이득)."""
    if base not in allowed or len(allowed) < 2:
        raise ValueError("기준 녹색이 허용 밖이거나 고를 녹색이 없습니다.")
    with torch.no_grad():
        rel = student(torch.from_numpy(np.asarray(x, np.float32)).view(1, -1), torch.tensor([base]))[0].numpy().astype(np.float64)
    if not np.all(np.isfinite(rel)):
        raise ValueError("예측 이득이 유한하지 않습니다.")
    rel = np.where(mask_of(allowed), rel, -np.inf)
    rel[base] = -np.inf
    best = int(rel.argmax())
    gain = float(rel[best])
    return (best if gain >= tau else int(base)), gain


# ----------------------------------------------------------------------------- 계획기 없는 실행기
class FeaturePlanner:
    """감지 80칸과 실행 가능 녹색만 만든다. 칸 값은 WEIGHT1 관측 앞 80칸과 같은 계산이고, 계획은 하지 않는다."""

    def __init__(self, episode, control_end: float, tail_policy=None):
        self.env = episode.env
        self.movement_of = episode.movement_of
        self.p0 = ep.controller_class("p0", wc.HORIZON_S)(episode.movement_of, [p.state for p in episode.ts.green_phases])
        self.p0._setup(episode.env)
        self.control_end = float(control_end)
        if tail_policy is not None:
            episode.ctl = lambda env, allowed: tail_policy.act(env, self.sensed(env), [int(a) for a in allowed])

    def sensed(self, env) -> np.ndarray:
        from focused_rl import ALL_ZONE_EDGES

        ts = env.traffic_signals[TARGET_TLS]
        base = np.asarray(ts.observation_fn(), dtype=np.float32)
        q0, _ = self.p0.queues_and_arrivals(env)
        moves, capacity = self.p0.movements, ts.observation_fn._capacity
        queues = [min(1.0, q / max(1.0, capacity[m[0]])) for m, q in zip(moves, q0)]
        waits = wc.longest_wait_features(env.sumo, self.movement_of, moves, ALL_ZONE_EDGES, wc.WAIT_SCALE_S)
        return np.asarray([*base, *queues, *waits], dtype=np.float32)

    def arena_snapshot(self, env) -> SimpleNamespace:
        allowed = list(env.signal_constraints.allowed(env))
        return SimpleNamespace(observation=self.sensed(env), allowed=allowed, greens=[allowed[0]] * ac.N_MODES, mask=[True] + [False] * (ac.N_MODES - 1),
                               choice=False, p0_action=allowed[0], planning_seconds=0.0, now=float(env.sim_step))


def _blocked(*args, **kwargs):
    raise RuntimeError("계획기를 불렀습니다(신경망만 실행).")


@contextlib.contextmanager
def planner_free(policy, tail: str = "NN"):
    """with 안에서는 계획기를 부르면 멈춘다. tail = "P0"이면 제어가 끝난 뒤의 P0 꼬리만 허용하고 그 횟수를 센다(점검용)."""
    if tail not in ("NN", "P0"):
        raise ValueError(f"꼬리는 NN 또는 P0: {tail}")
    saved = ep.plan, wc.weighted_plan, ac.make_planner
    real_plan = ep.plan
    guard = {"tail_calls": 0, "env": None, "control_end": None}

    def guarded(*args, **kwargs):
        env = guard["env"]
        if tail == "P0" and env is not None and float(env.sim_step) >= guard["control_end"] - 1e-9:
            guard["tail_calls"] += 1
            return real_plan(*args, **kwargs)
        return _blocked()

    def make_planner(episode, control_end):
        guard.update(env=episode.env, control_end=float(control_end))
        return FeaturePlanner(episode, control_end, policy if tail == "NN" else None)

    ep.plan, wc.weighted_plan, ac.make_planner = guarded, _blocked, make_planner
    try:
        yield guard
    finally:
        ep.plan, wc.weighted_plan, ac.make_planner = saved


def run_world(world: str, situation: dict, work: Path, policy, **kwargs):
    if check_world(world) == "P":
        return pt.run_episode(situation, work, policy, **kwargs)
    result = ac.run_arena_episode(situation, work, policy, **kwargs)
    result.metrics["world"] = "R"
    return result


def east_cells(core, env) -> list[float]:
    """동쪽 도착 추정 16칸 + 이웃 시간표 24칸. 계획은 하지 않는다(TM4 관측과 같은 계산)."""
    _, arrivals = core.ctl.queues_and_arrivals(env)
    forecast = pt.east_forecast(arrivals, core.ctl.index)
    table = pt.timetable_view(core.ctl._state_at, float(env.sim_step), core.ctl.geo["travel"][fp.EAST_APPROACH])
    return [*forecast, *table]


class NNPolicy(ac.P0Policy):
    """신경망(+ 있으면 학생)만으로 녹색을 고른다. P0 종류라 실행기가 모든 제어 단계에서 부른다. 꼬리는 FeaturePlanner가 act로 잇는다."""

    def __init__(self, world: str, net: GreenNet, name: str, student: GainNet | None = None, tau: float | None = None, keep_inputs: bool = False):
        if student is not None and tau is None:
            raise ValueError("학생에는 문턱이 필요합니다.")
        self.world, self.net, self.name, self.student, self.tau = check_world(world), net, name, student, tau
        self.core, self.ctx = None, None
        self.stats, self.pairs = Counter(), Counter()
        self.rows = [] if keep_inputs else None
        self._mean = net.mean.numpy().astype(np.float64)
        self._std = net.std.numpy().astype(np.float64)

    def features(self, env, sensed: np.ndarray, core) -> np.ndarray:
        if self.world == "P":
            return np.asarray([*sensed, *east_cells(core, env)], dtype=np.float32)
        return np.asarray(sensed, dtype=np.float32)

    def act(self, env, sensed: np.ndarray, allowed: list[int]) -> int:
        core = None
        if self.world == "P":
            core = pt.episode_core(self, self.ctx)
            core.observe(env)
        if len(allowed) == 1:
            self.stats["forced"] += 1
            return allowed[0]
        x = self.features(env, sensed, core)
        action = green_choice(self.net, x, allowed)
        self.stats["planned"] += 1
        self.stats["z5_cells"] += int((np.abs((x - self._mean) / self._std) > 5.0).sum())
        if self.student is not None:
            changed, gain = student_choice(self.student, x, action, allowed, float(self.tau))
            if changed != action:
                self.stats["student_changed"] += 1
                self.pairs[f"{action}->{changed}"] += 1
                action = changed
        if self.rows is not None:
            self.rows.append((float(env.sim_step), x, action))
        return action

    def decide(self, snap, ctx) -> int:
        self.ctx = ctx
        return self.act(ctx.ep.env, snap.observation, [int(a) for a in ctx.allowed])

    def info(self) -> dict:
        return {"kind": "nn", "student": self.student is not None, "tau": self.tau, **self.stats, "pairs": dict(self.pairs)}

    def save_inputs(self, path: Path) -> None:
        if self.rows:
            np.savez_compressed(path, now=np.asarray([r[0] for r in self.rows]), x=np.stack([r[1] for r in self.rows]),
                                action=np.asarray([r[2] for r in self.rows], np.int64))


class PlannerPathPolicy(ac.P0Policy):
    """같은 신경망을 기존 경로(계획기가 함께 돔)로 돌리며, 결정마다 계획기 없이 만든 입력과 칸별로 대조한다(점검용)."""

    def __init__(self, world: str, net: GreenNet, name: str):
        self.world, self.net, self.name = check_world(world), net, name
        self.core, self.feature = None, None
        self.stats = Counter()
        self.max_diff = 0.0

    def decide(self, snap, ctx) -> int:
        env = ctx.ep.env
        allowed = [int(a) for a in ctx.allowed]
        if self.feature is None:
            self.feature = FeaturePlanner(ctx.ep, ctx.planner.control_end)
        core = None
        if self.world == "P":
            core = pt.episode_core(self, ctx)
            core.observe(env)
        if len(allowed) == 1:
            return allowed[0]
        if self.world == "P":
            x = pt.observation167(snap.observation, core.view(env, allowed)["extra"])[COLUMNS["P"]]
            free = np.asarray([*self.feature.sensed(env), *east_cells(core, env)], dtype=np.float32)
        else:
            x = np.asarray(snap.observation, np.float32)[COLUMNS["R"]]
            free = self.feature.sensed(env)
        self.max_diff = max(self.max_diff, float(np.abs(x - free).max()))
        self.stats["compared"] += 1
        return green_choice(self.net, x, allowed)

    def info(self) -> dict:
        return {"kind": "planner_path", **self.stats, "max_input_diff": self.max_diff}


def m4_costs(planner, env, allowed: list[int]) -> list[float]:
    """허용 녹색마다 '그 녹색만 허용한 M4 계획'의 비용(ArenaPlanner가 갈림 지점에서 하는 계산과 같다)."""
    ts, guard, now = env.traffic_signals[TARGET_TLS], env.signal_constraints, float(env.sim_step)
    q0, arrivals = planner.p0.queues_and_arrivals(env)
    served = guard._served_now(ts, now)
    state = ep.SignalState(ts.green_phase, now - ts.time_since_last_phase_change, float(ts.yellow_time), tuple(served[c] for c in planner.p0._model.crosswalks))
    weights = ac.lefts_weights(ac.MODES6[ac.M4], planner.p0.movements)
    costs = [float("nan")] * GREENS
    for a in allowed:
        _, info = wc.weighted_plan(planner.p0._model, state, now, q0, arrivals, [int(a)], weights)
        costs[int(a)] = float(info["cost"])
    return costs


class LogPolicy(ac.P0Policy):
    """자료 기록: 고를 수 있는 모든 결정에서 입력·선생님 녹색·계획 비용을 남긴다. driver가 없으면 선생님이, 있으면 신경망이 운전한다. 꼬리는 P0."""

    def __init__(self, world: str, teacher, tau: float | None, driver: GreenNet | None, name: str):
        self.world, self.teacher, self.tau, self.driver, self.name = check_world(world), teacher, tau, driver, name
        self.core = None
        self.rows: list[tuple] = []
        self.stats = Counter()

    def decide(self, snap, ctx) -> int:
        env = ctx.ep.env
        allowed = [int(a) for a in ctx.allowed]
        core = None
        if self.world == "P":
            core = pt.episode_core(self, ctx)
            core.observe(env)
        if len(allowed) == 1:
            self.stats["forced"] += 1
            return allowed[0]
        if self.world == "P":
            view = core.view(env, allowed)
            obs = pt.observation167(snap.observation, view["extra"])
            teacher, _ = pc.sure_green(self.teacher, obs, float(self.tau))
            costs = [float("nan") if c is None else float(c) for c in view["costs"]]
            x = obs[COLUMNS["P"]]
            planner_green = int(view["green"])
        else:
            teacher = int(self.teacher.act(snap))
            costs = m4_costs(ctx.planner, env, allowed)
            if snap.m4_costs is not None and any(c is not None and abs(c - costs[g]) > 1e-9 for g, c in enumerate(snap.m4_costs)):
                raise RuntimeError("M4 녹색별 비용이 계획기의 값과 다릅니다.")
            x = np.asarray(snap.observation, np.float32)[COLUMNS["R"]]
            planner_green = int(snap.greens[ac.M4])
        action = teacher if self.driver is None else green_choice(self.driver, x, allowed)
        self.stats["planned"] += 1
        self.stats["differs_from_teacher"] += int(action != teacher)
        self.rows.append((int(ctx.step), float(env.sim_step), x.copy(), mask_of(allowed), teacher, action, planner_green, costs))
        return action

    def info(self) -> dict:
        return {"kind": "log", "driver": self.driver is not None, **self.stats}

    def save(self, path: Path) -> None:
        r = self.rows
        np.savez_compressed(path, step=np.asarray([x[0] for x in r], np.int32), now=np.asarray([x[1] for x in r]), x=np.stack([x[2] for x in r]),
                            allowed=np.stack([x[3] for x in r]), teacher=np.asarray([x[4] for x in r], np.int64), executed=np.asarray([x[5] for x in r], np.int64),
                            planner=np.asarray([x[6] for x in r], np.int64), costs=np.asarray([x[7] for x in r], np.float64))


def load_teacher(world: str, folder: Path, cfg: dict):
    """(선생님, 문턱). 설정에 고정한 모델인지 해시로 대조한다."""
    spec = cfg["teachers"][check_world(world)]
    if world == "P":
        if pl.student_identity(Path(folder)) != spec["model_sha256"]:
            raise ValueError("선생님 학생 모델이 설정에 고정한 PT1과 다릅니다.")
        return pl.load_student(Path(folder)), float(spec["tau"])
    import arena3_stage2_train as s2
    if s2.model_identity(Path(folder)) != spec["model_sha256"]:
        raise ValueError("선생님 모델이 설정에 고정한 S2A_pre와 다릅니다.")
    return pi.ModeBase(s2.load_model(Path(folder))), None


def build_policy(job: dict):
    net, _ = load_model(Path(job["model_dir"]))
    student, tau = None, None
    if job.get("student_dir"):
        student, manifest = load_model(Path(job["student_dir"]))
        tau = float(manifest["tau"])
    return NNPolicy(job["world"], net, job["policy"], student, tau, keep_inputs=bool(job.get("save_inputs")))


def run_job(job: dict) -> dict:
    """실행 한 판(평가·점검·자료 기록 공통). 결과는 ARENA3 실행 형식으로 쓴다."""
    started = time.monotonic()
    work = Path(job["work"])
    name = f"{job['policy']}/{job['sid']}"
    try:
        w.configure()
        torch.set_num_threads(1)
        shutil.rmtree(work, ignore_errors=True)
        target = Path(job["target"])
        extra = {}
        if job["mode"] == "free":
            policy = build_policy(job)
            with planner_free(policy, job.get("tail", "NN")) as guard:
                result = run_world(job["world"], job["situation"], work, policy, max_seconds=MAX_EPISODE_S)
            extra = {"planner_free": True, "tail": job.get("tail", "NN"), "tail_planner_calls": guard["tail_calls"]}
        elif job["mode"] == "planner":
            net, _ = load_model(Path(job["model_dir"]))
            policy = PlannerPathPolicy(job["world"], net, job["policy"])
            result = run_world(job["world"], job["situation"], work, policy, max_seconds=MAX_EPISODE_S)
        elif job["mode"] == "log":
            teacher, tau = load_teacher(job["world"], Path(job["teacher_dir"]), job["cfg"])
            driver = load_model(Path(job["driver_dir"]))[0] if job.get("driver_dir") else None
            policy = LogPolicy(job["world"], teacher, tau, driver, job["policy"])
            result = run_world(job["world"], job["situation"], work, policy, max_seconds=MAX_EPISODE_S)
        else:
            raise ValueError(f"알 수 없는 실행 종류: {job['mode']}")
        if result.metrics["route_sha256"] != job["situation"]["route_sha256"]:
            raise ValueError(f"차량 파일의 SHA가 설정과 다릅니다: {result.metrics['route_sha256']}")
        target.mkdir(parents=True, exist_ok=True)
        if job["mode"] == "log":
            policy.save(target / "log.npz")
            extra["log_sha256"] = w.file_hash(target / "log.npz")
        elif job.get("save_inputs"):
            policy.save_inputs(target / "inputs.npz")
        ac.write_run(target, result, {"policy": job["policy"], "fingerprint": job["fingerprint"], "model_identity": job.get("model_identity"),
                                      "job_seconds": time.monotonic() - started, "information": policy.info(), **extra})
        return {"ok": True, "seconds": time.monotonic() - started, "name": name}
    except BaseException as error:
        return {"ok": False, "seconds": time.monotonic() - started, "name": name, "error": f"{type(error).__name__}: {error}"}
    finally:
        shutil.rmtree(work, ignore_errors=True)


def prepare_caches(world: str) -> None:
    pl._prepare_map_cache()
    if world == "P":
        p0.prepare_caches()


# ----------------------------------------------------------------------------- 되돌려 굴린 자료
def cost_matrix(point: dict, row: dict, futures: int) -> np.ndarray:
    """(녹색 4, 미래 수) 천 차량·초. 허용 녹색이 아니거나 쓰지 않는 미래는 nan."""
    out = np.full((GREENS, futures), np.nan)
    valid = [*row["fit_future_indices"], *row["score_future_indices"]]
    for green, runs in point["rollouts"].items():
        for j in valid:
            out[int(green), j] = float(runs[j]["cost"]) / SCALE
    return out


def load_points(world: str, data: Path, weight: float) -> dict:
    """기존 자료 읽기(검사는 기존 함수 그대로) + 지점별 미래별 비용. 비용은 '300초 지연 + weight × 끝 순간 멈춘 차'."""
    module, mcfg, _ = lb.world_parts(world)
    costs: dict[str, np.ndarray] = {}
    original = module.point_to_row

    def spy(point, date, situation, cfg):
        relabeled = lb.relabel_point(point, weight)
        row, reason = original(relabeled, date, situation, cfg)
        if row is not None:
            costs[str(point["state"])] = cost_matrix(relabeled, row, int(cfg["rollout"]["futures"]))
        return row, reason

    module.point_to_row = spy
    try:
        d = module.load_dataset(Path(data), mcfg)
    finally:
        module.point_to_row = original
    d["costs"] = np.stack([costs[s] for s in d["state_ids"].tolist()])
    d["manifest"] = {**{k: v for k, v in d["manifest"].items() if k != "future_partitions"}, "label_end_weight": float(weight)}
    d["fingerprint"] = w.digest(d["manifest"])
    return d


def restrict(d: dict, dates) -> dict:
    keep = np.isin(d["dates"], list(dates))
    if not keep.any():
        raise ValueError("고른 날짜의 자료가 없습니다.")
    out = {k: (v[keep] if isinstance(v, np.ndarray) and len(v) == len(keep) else v) for k, v in d.items()}
    out["fingerprint"] = w.digest({"source": d["fingerprint"], "dates": sorted(dates)})
    return out


def gains_against(costs: np.ndarray, base: np.ndarray, allowed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """같은 미래 짝 이득(기준 녹색 비용 − 후보 비용)의 평균과 표준오차. 기준 녹색을 바꾸면 표준오차도 그 기준으로 다시 잰다."""
    rows = np.arange(len(base))
    diffs = costs[rows, base][:, None, :] - costs
    count = np.isfinite(diffs).sum(axis=2)
    filled = np.nan_to_num(diffs)
    gains = filled.sum(axis=2) / np.maximum(count, 1)
    variance = np.where(np.isfinite(diffs), (filled - gains[:, :, None]) ** 2, 0.0).sum(axis=2) / np.maximum(count - 1, 1)
    se = np.sqrt(variance / np.maximum(count, 1))
    return gains * allowed, se * allowed


def half_costs(costs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """유효 미래의 앞 절반(학습)과 뒤 절반(채점)으로 나눈다. 기존 자료 읽기의 fit·score 나누기와 같다."""
    front, back = np.full(costs.shape, np.nan), np.full(costs.shape, np.nan)
    for i, row in enumerate(costs):
        valid = np.flatnonzero(np.isfinite(row).any(axis=0))
        cut = len(valid) // 2
        front[i][:, valid[:cut]] = row[:, valid[:cut]]
        back[i][:, valid[cut:]] = row[:, valid[cut:]]
    return front, back


def t20_labels(net, observations: np.ndarray, tau: float) -> np.ndarray:
    """PT1 학생 + 문턱: pc.sure_green과 같은 규칙을 묶음으로(같으면 TM4)."""
    obs = np.asarray(observations, np.float32)
    with torch.no_grad():
        rel = net(torch.from_numpy(obs)).numpy().astype(np.float64)
    rows, anchor = np.arange(len(obs)), pt.anchor_green(obs)
    rel = np.where(pt.allowed_mask(obs), rel, -np.inf)
    rel[rows, anchor] = -np.inf
    best = rel.argmax(axis=1)
    return np.where(rel[rows, best] >= float(tau), best, anchor)


# ----------------------------------------------------------------------------- 학습
def fit_green(x: np.ndarray, y: np.ndarray, allowed: np.ndarray, seed: int, epochs: int, st: dict, weights: np.ndarray | None = None, on_epoch=None) -> GreenNet:
    """선생님 녹색 맞히기(허용 녹색 안에서). weights를 주면 결정마다 무게를 둔다."""
    torch.manual_seed(int(seed))
    mean, std = scale_of(x)
    net = GreenNet(x.shape[1], mean, std, tuple(st["network"]))
    opt = torch.optim.Adam(net.parameters(), lr=float(st["learning_rate"]), weight_decay=float(st["weight_decay"]))
    rng = np.random.default_rng(int(seed))
    X, Y, M = torch.from_numpy(np.asarray(x, np.float32)), torch.from_numpy(np.asarray(y, np.int64)), torch.from_numpy(np.asarray(allowed, bool))
    W = None if weights is None else torch.from_numpy(np.asarray(weights, np.float32))
    for e in range(1, int(epochs) + 1):
        order = rng.permutation(len(X))
        for k in range(0, len(order), int(st["batch_size"])):
            i = order[k:k + int(st["batch_size"])]
            loss = F.cross_entropy(net(X[i]).masked_fill(~M[i], -1e9), Y[i], reduction="none")
            loss = loss.mean() if W is None else (loss * W[i]).sum() / W[i].sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
        if on_epoch is not None:
            on_epoch(e, net)
    return net.eval()


def greens_of(net: GreenNet, x: np.ndarray, allowed: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return net(torch.from_numpy(np.asarray(x, np.float32))).masked_fill(~torch.from_numpy(np.asarray(allowed, bool)), -1e9).argmax(dim=1).numpy()


def heldout_greens(x: np.ndarray, y: np.ndarray, allowed: np.ndarray, dates: np.ndarray, seed: int, epochs: int, st: dict) -> np.ndarray:
    """날짜 하나씩 보류한 흉내 신경망의 녹색. 학생이 '신경망이 틀리는 곳'을 보게 한다."""
    out = np.zeros(len(x), np.int64)
    for d in sorted(set(dates.tolist())):
        held = dates == d
        out[held] = greens_of(fit_green(x[~held], y[~held], allowed[~held], seed, epochs, st), x[held], allowed[held])
    return out


def gain_loss(net: GainNet, x, base, gains, se, allowed, st: dict) -> torch.Tensor:
    """PT1과 같은 신뢰 가중 Huber. 허용 녹색 중 기준 녹색이 아닌 것만 맞춘다."""
    rel = net(x, base)
    valid = allowed.clone()
    valid.scatter_(1, base[:, None], False)
    low, high = st["confidence_clip"]
    weights = (gains.abs() / (se + float(st["confidence_floor"]))).clamp(float(low), float(high)) * valid
    return (F.smooth_l1_loss(rel, gains, reduction="none") * weights).sum() / weights.sum().clamp_min(1e-8)


def fit_gain(x: np.ndarray, base: np.ndarray, gains: np.ndarray, se: np.ndarray, allowed: np.ndarray, seed: int, epochs: int, st: dict, on_epoch=None) -> GainNet:
    torch.manual_seed(int(seed))
    mean, std = scale_of(x)
    net = GainNet(x.shape[1], mean, std, tuple(st["network"]))
    opt = torch.optim.Adam(net.parameters(), lr=float(st["learning_rate"]), weight_decay=float(st["weight_decay"]))
    rng = np.random.default_rng(int(seed))
    X, B = torch.from_numpy(np.asarray(x, np.float32)), torch.from_numpy(np.asarray(base, np.int64))
    G, S, M = torch.from_numpy(np.asarray(gains, np.float32)), torch.from_numpy(np.asarray(se, np.float32)), torch.from_numpy(np.asarray(allowed, bool))
    net.train()
    for e in range(1, int(epochs) + 1):
        order = rng.permutation(len(X))
        for k in range(0, len(order), int(st["batch_size"])):
            i = order[k:k + int(st["batch_size"])]
            loss = gain_loss(net, X[i], B[i], G[i], S[i], M[i], st)
            if not bool(torch.isfinite(loss)):
                raise ValueError("학생 손실이 유한하지 않습니다.")
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), float(st["grad_clip"]))
            opt.step()
        if on_epoch is not None:
            on_epoch(e, net)
    return net.eval()


def predicted_best(net: GainNet, x: np.ndarray, base: np.ndarray, allowed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """기준 녹색이 아닌 허용 녹색 중 예측 이득이 가장 큰 녹색과 그 이득."""
    with torch.no_grad():
        rel = net(torch.from_numpy(np.asarray(x, np.float32)), torch.from_numpy(np.asarray(base, np.int64))).numpy().astype(np.float64)
    rows = np.arange(len(x))
    rel = np.where(allowed, rel, -np.inf)
    rel[rows, base] = -np.inf
    best = rel.argmax(axis=1)
    return best, rel[rows, best]


def build_student(x: np.ndarray, y: np.ndarray, costs: np.ndarray, allowed: np.ndarray, dates: np.ndarray, seed: int, st: dict) -> tuple[GainNet, dict]:
    """계획기 없는 학생: 기준 녹색은 날짜 보류 흉내의 녹색, 이득·표준오차는 그 기준으로 다시 잰다. 문턱은 날짜 보류 예측의 상위 비율로 정한다."""
    base = heldout_greens(x, y, allowed, dates, seed, int(st["epochs_imitation"]), st)
    gains, se = gains_against(costs, base, allowed)
    predicted = np.zeros(len(x))
    for d in sorted(set(dates.tolist())):
        held = dates == d
        fold = fit_gain(x[~held], base[~held], gains[~held], se[~held], allowed[~held], seed, int(st["epochs_student"]), st)
        predicted[held] = predicted_best(fold, x[held], base[held], allowed[held])[1]
    tau = float(np.quantile(predicted, 1.0 - float(st["student_change_share"])))
    net = fit_gain(x, base, gains, se, allowed, seed, int(st["epochs_student"]), st)
    best, full = predicted_best(net, x, base, allowed)
    return net, {"tau": tau, "base_agreement_heldout": float((base == y).mean()), "change_share_heldout": float((predicted >= tau).mean()),
                 "change_share_in_sample": float((full >= tau).mean()), "points": int(len(x))}


def imitation_labels(world: str, d: dict, teacher_dir: Path | None, cfg: dict) -> dict[str, np.ndarray]:
    """후보 이름 → 선생님 녹색(되돌려 굴린 자료 위)."""
    if world == "R":
        return {"Rn": np.asarray(d["executed"], np.int64)}
    labels = {"N1": np.asarray(d["anchor"], np.int64)}
    net, tau = load_teacher("P", Path(teacher_dir), cfg)
    labels["N2"] = t20_labels(net, d["observations"], tau)
    return labels


def model_meta(world: str, kind: str, candidate: str, seed: int, dataset: str, st: dict, **more) -> dict:
    return {"kind": kind, "world": world, "candidate": candidate, "seed": int(seed), "input_size": int(len(COLUMNS[world])), "network": list(st["network"]),
            "dataset_fingerprint": dataset, "config_sha256": w.file_hash(CONFIG), "source_sha256": w.file_hash(ROOT / "src/explore_nn1.py"), **more}


def train_base(world: str, data: Path, teacher: Path | None, out: Path, cfg: dict) -> dict:
    """되돌려 굴린 기존 자료(학습 6일)로 흉내 후보와 학생 후보를 만든다. 이미 있으면 같은 자료인지 보고 건너뛴다."""
    torch.set_num_threads(1)
    st = cfg["training"]
    d = restrict(load_points(world, Path(data), float(st["student_label_end_weight"])), cfg["dates"]["train"])
    x, allowed, dates = d["observations"][:, COLUMNS[world]], d["allowed"], d["dates"]
    labels = imitation_labels(world, d, teacher, cfg)
    out = Path(out)
    summary = {"points": int(len(x)), "dates": sorted(set(dates.tolist())), "dataset_fingerprint": d["fingerprint"], "models": {}}
    if not np.isfinite(x).all():
        raise ValueError("입력에 유한하지 않은 값이 있습니다.")
    for seed in st["seeds"]:
        for name, spec in cfg["candidates"][world].items():
            folder = out / name / f"seed_{seed}"
            if (folder / "manifest.json").exists():
                m = w.read(folder / "manifest.json")
                if m.get("dataset_fingerprint") != d["fingerprint"] or m.get("seed") != seed:
                    raise ValueError(f"이미 있는 모델이 다른 자료·난수로 학습됐습니다: {folder}")
                summary["models"].setdefault(name, {})[seed] = {"model_sha256": model_identity(folder), "reused": True}
                continue
            if spec["kind"] == "imitate":
                y = labels[name]
                started = time.monotonic()
                net = fit_green(x, y, allowed, seed, int(st["epochs_imitation"]), st)
                held = heldout_greens(x, y, allowed, dates, seed, int(st["epochs_imitation"]), st)
                info = {"agreement_in_sample": float((greens_of(net, x, allowed) == y).mean()), "agreement_heldout": float((held == y).mean()),
                        "label_differs_from_planner": float((y != (d["anchor"] if world == "P" else d["m4"])).mean()), "seconds": time.monotonic() - started}
                sha = save_model(folder, net, model_meta(world, KIND_GREEN, name, seed, d["fingerprint"], st, teacher=spec["teacher"], training=info))
            elif spec["kind"] == "student":
                started = time.monotonic()
                net, info = build_student(x, labels[spec["base"]], d["costs"], allowed, dates, seed, st)
                info["seconds"] = time.monotonic() - started
                sha = save_model(folder, net, model_meta(world, KIND_GAIN, name, seed, d["fingerprint"], st, base=spec["base"], tau=info["tau"],
                                                         label_end_weight=float(st["student_label_end_weight"]), training=info))
            else:
                continue
            summary["models"].setdefault(name, {})[seed] = {"model_sha256": sha, "reused": False, **info}
            ref.stamp(f"{name} 난수 {seed}: {json.dumps(info, ensure_ascii=False)[:220]}")
    s2e._write(out / "training.json", summary)
    return summary


# ----------------------------------------------------------------------------- 자료 강화
def log_rows(out: Path, round_: int, sits: list[dict], record: dict) -> tuple[list[dict], dict]:
    """라운드 하나의 기록을 읽는다. 해시를 대조하고, 미확인·204820 충돌이 난 판은 모두 뺀다(PLATOON1과 같은 규칙)."""
    rows, skipped = [], Counter()
    for s in sits:
        sid = ac.situation_id(s)
        folder = Path(out) / "runs" / f"r{round_}" / sid
        m = w.read(folder / "metrics.json")
        cache = w.read(folder / "cache.json")
        if cache.get("fingerprint") != record["fingerprint"] or m.get("route_sha256") != s["route_sha256"] or w.file_hash(folder / "log.npz") != m["log_sha256"]:
            raise ValueError(f"기록의 지문·차량 파일·해시가 다릅니다: {sid}")
        if m.get("collisions_unexplained", 0) or m.get("collisions_at_204820", 0):
            skipped["collision_runs"] += 1
            continue
        z = np.load(folder / "log.npz")
        rows.append({"sid": sid, "date": s["date"], "round": round_, "log_sha256": m["log_sha256"], **{k: z[k] for k in z.files}})
    return rows, dict(skipped)


def margin_weights(costs: np.ndarray, clip: list[float]) -> tuple[np.ndarray, float]:
    """틀리면 비용이 큰 결정에 무게를 더 준다: clip(1·2위 계획 비용 차 ÷ 중앙값). 고를 수 있는 결정만 들어온다."""
    filled = np.where(np.isfinite(costs), costs, np.inf)
    ordered = np.sort(filled, axis=1)
    margin = ordered[:, 1] - ordered[:, 0]
    if not np.isfinite(margin).all():
        raise ValueError("허용 녹색이 둘 미만인 결정이 기록에 있습니다.")
    median = float(np.median(margin))
    return np.clip(margin / max(median, 1e-6), float(clip[0]), float(clip[1])), median


def round_jobs(world: str, round_: int, out: Path, work: Path, fp_: str, teacher: Path, cfg: dict) -> list[dict]:
    sits = named_situations(cfg, "situations_aug", world, round_)
    seeds = cfg["training"]["seeds"]
    jobs = []
    for i, s in enumerate(sits):
        driver = None if round_ == 0 else Path(out) / "models" / f"r{round_ - 1}" / f"seed_{seeds[i % len(seeds)]}"
        identity = "teacher" if driver is None else model_identity(driver)
        job = ref.jobs_for(Path(out), Path(work), fp_, f"r{round_}", [s], model_identity=identity, kind="log")[0]
        job.update(world=world, mode="log", teacher_dir=str(teacher), driver_dir=str(driver) if driver else None, cfg={"teachers": cfg["teachers"]})
        jobs.append(job)
    return jobs


def train_round(world: str, round_: int, out: Path, record: dict, cfg: dict) -> dict:
    """라운드 0~k의 기록을 모두 모아 모델 3개를 학습한다. 표준화·학습 횟수는 다른 후보와 같고, 결정별 무게만 더한다."""
    torch.set_num_threads(1)
    st, models = cfg["training"], Path(out) / "models" / f"r{round_}"
    rows, skipped = [], {}
    for k in range(round_ + 1):
        got, skip = log_rows(out, k, named_situations(cfg, "situations_aug", world, k), record)
        rows += got
        skipped[f"r{k}"] = skip
    x = np.concatenate([r["x"] for r in rows]).astype(np.float32)
    y = np.concatenate([r["teacher"] for r in rows])
    allowed = np.concatenate([r["allowed"] for r in rows])
    costs = np.concatenate([r["costs"] for r in rows])
    if not np.isfinite(x).all() or not allowed[np.arange(len(y)), y].all():
        raise ValueError("기록의 입력이 유한하지 않거나 선생님 녹색이 허용 밖입니다.")
    weights, median = margin_weights(costs, st["margin_clip"])
    fingerprint = w.digest({"round": round_, "logs": [[r["sid"], r["log_sha256"]] for r in rows]})
    summary = {"round": round_, "runs": len(rows), "decisions": int(len(x)), "skipped": skipped, "margin_median": median, "dataset_fingerprint": fingerprint, "models": {}}
    for seed in st["seeds"]:
        folder = models / f"seed_{seed}"
        if (folder / "manifest.json").exists():
            if w.read(folder / "manifest.json").get("dataset_fingerprint") != fingerprint:
                raise ValueError(f"이미 있는 라운드 모델이 다른 기록으로 학습됐습니다: {folder}")
            summary["models"][seed] = {"model_sha256": model_identity(folder), "reused": True}
            continue
        started = time.monotonic()
        net = fit_green(x, y, allowed, seed, int(st["epochs_imitation"]), st, weights=weights)
        info = {"agreement_in_sample": float((greens_of(net, x, allowed) == y).mean()), "decisions": int(len(x)), "seconds": time.monotonic() - started}
        sha = save_model(folder, net, model_meta(world, KIND_GREEN, cfg["teachers"][world]["name"], seed, fingerprint, st, round=round_, training=info))
        summary["models"][seed] = {"model_sha256": sha, "reused": False, **info}
    s2e._write(models / "training.json", summary)
    ref.stamp(f"라운드 {round_} 학습: 판 {len(rows)}, 결정 {len(x)}, 제외 {skipped}")
    return summary


def select_round(means: dict[int, float], tolerance: float) -> int:
    """검증 평균 지연이 가장 짧은 라운드. 더 뒤 라운드가 그 값의 tolerance 안이면 가장 뒤 라운드(자료가 더 많다)."""
    best = min(means.values())
    return max(k for k, v in means.items() if v <= best + float(tolerance))


def validate_rounds(world: str, name: str, out: Path, work: Path, workers: int, cfg: dict) -> dict:
    """라운드마다 모델 3개 × 검증 24상황(계획기 없는 경로, 꼬리까지 신경망)."""
    st, sits = cfg["training"], named_situations(cfg, "situations_validation", world)
    folder = Path(out) / "valid"
    fp_ = ref.register(folder, extra_files=SOURCES)
    jobs = []
    for k in range(int(st["rounds"]) + 1):
        for i, seed in enumerate(st["seeds"]):
            model = Path(out) / "models" / f"r{k}" / f"seed_{seed}"
            for job in ref.jobs_for(folder, Path(work) / "valid", fp_, f"{name}_r{k}_m{i}", sits, model_identity=model_identity(model), kind="nn"):
                job.update(world=world, mode="free", model_dir=str(model), tail="NN")
                jobs.append(job)
    status = ref.run_jobs(jobs, max(1, workers), ref.Ledger(folder, "valid"), worker=run_job)
    if status["failed"]:
        return {"status": "runs_failed", "failed": status["failed"]}
    means, per_model = {}, {}
    n = None
    for k in range(int(st["rounds"]) + 1):
        totals = []
        for i, seed in enumerate(st["seeds"]):
            model = Path(out) / "models" / f"r{k}" / f"seed_{seed}"
            rows = [ac.read_run(folder / "runs" / f"{name}_r{k}_m{i}" / ac.situation_id(s), fp_, model_identity(model)) for s in sits]
            n = sum(int(r["vehicles"]) for r in rows)
            totals.append(sum(float(r["actual_total_delay"]) for r in rows) / n)
        means[k], per_model[k] = statistics.fmean(totals), totals
    chosen = select_round(means, float(st["select_tolerance_s"]))
    return {"status": "done", "selected_round": chosen, "means": {str(k): v for k, v in means.items()}, "per_model": {str(k): v for k, v in per_model.items()},
            "vehicles": n, "note": "검증 24상황(0928·0814), 한 시간 결과로 라운드를 고른다."}


def augment(world: str, teacher: Path, out: Path, work: Path, workers: int, cfg: dict) -> int:
    """자료 강화 후보 하나: 라운드 0~3 기록·학습 → 검증으로 라운드 고르기 → 84상황 평가. 끊겨도 다시 실행하면 이어서 돈다."""
    out, work = Path(out), Path(work)
    name = next(k for k, v in cfg["candidates"][world].items() if v["kind"] == "aug")
    fp_ = ref.register(out, extra_files=SOURCES)
    record = s2e.folder_record(out)
    load_teacher(world, teacher, cfg)
    prepare_caches(world)
    ledger = ref.Ledger(out, "logs")
    for k in range(int(cfg["training"]["rounds"]) + 1):
        status = ref.run_jobs(round_jobs(world, k, out, work / f"r{k}", fp_, teacher, cfg), max(1, workers), ledger, worker=run_job)
        if status["failed"]:
            ref.stamp(f"라운드 {k} 실패 {len(status['failed'])}판. 다시 실행하면 끝난 판은 건너뜁니다.")
            return 1
        train_round(world, k, out, record, cfg)
    selected = validate_rounds(world, name, out, work, workers, cfg)
    if selected["status"] != "done":
        return 1
    s2e._freeze(out / "selected.json", selected)
    ref.stamp(f"{name}: 라운드 {selected['selected_round']} 선택, 검증 평균 {selected['means']}")
    status = evaluate(world, name, out / "models" / f"r{selected['selected_round']}", out / "eval" / name, work / "eval", workers, cfg)
    return 1 if status["failed"] else 0


# ----------------------------------------------------------------------------- 평가
def declaration(world: str, candidate: str, models: Path, cfg: dict) -> dict:
    """후보의 모델 3개. 학생 후보는 기준 신경망(같은 난수의 흉내 모델)과 함께 묶는다."""
    spec = cfg["candidates"][check_world(world)].get(candidate)
    if spec is None:
        raise ValueError(f"{world} 세계의 후보가 아닙니다: {candidate}")
    rows = []
    for seed in cfg["training"]["seeds"]:
        if spec["kind"] == "student":
            base, student = Path(models) / spec["base"] / f"seed_{seed}", Path(models) / candidate / f"seed_{seed}"
            manifest = w.read(student / "manifest.json")
            if w.read(base / "manifest.json").get("dataset_fingerprint") != manifest.get("dataset_fingerprint"):
                raise ValueError("학생과 기준 신경망의 자료가 다릅니다.")
            rows.append({"seed": seed, "base_dir": str(base.resolve()), "student_dir": str(student.resolve()), "tau": float(manifest["tau"]),
                         "identity": f"{model_identity(base)}|{model_identity(student)}|tau={float(manifest['tau']):.6f}"})
        else:
            folder = Path(models) / candidate / f"seed_{seed}" if spec["kind"] == "imitate" else Path(models) / f"seed_{seed}"
            rows.append({"seed": seed, "base_dir": str(folder.resolve()), "student_dir": None, "tau": None, "identity": model_identity(folder)})
    return {"candidate": candidate, "world": world, "kind": spec["kind"], "models": rows}


def eval_jobs(world: str, candidate: str, decl: dict, out: Path, work: Path, fp_: str, sits: list[dict], cfg: dict) -> list[dict]:
    jobs = []
    for i, row in enumerate(decl["models"]):
        for job in ref.jobs_for(Path(out), Path(work), fp_, f"{candidate}_m{i}", sits, model_identity=row["identity"], kind="nn"):
            job.update(world=world, mode="free", tail="NN", model_dir=row["base_dir"], student_dir=row["student_dir"],
                       save_inputs=i == 0 and job["sid"] in cfg["check_situations"])
            jobs.append(job)
    return jobs


def evaluate(world: str, candidate: str, models: Path, out: Path, work: Path, workers: int, cfg: dict, *, executor=None, worker=run_job) -> dict:
    sits = eval_situations(world)
    out = Path(out)
    fp_ = ref.register(out, extra_files=SOURCES)
    decl = declaration(world, candidate, Path(models), cfg)
    s2e._freeze(out / "models.json", {**decl, "models": [{k: v for k, v in r.items() if k not in ("base_dir", "student_dir")} for r in decl["models"]]})
    if worker is run_job and executor is None:
        prepare_caches(world)
    jobs = eval_jobs(world, candidate, decl, out, work, fp_, sits, cfg)
    status = ref.run_jobs(jobs, max(1, workers), ref.Ledger(out, candidate), worker=worker, executor=executor)
    s2e._write(out / "status" / f"{candidate}.json", {**status, "total": len(jobs), "updated_unix": time.time()})
    return status


# ----------------------------------------------------------------------------- 점검(맥·Colab에서 같은 코드)
SAME = ("mean_delay_s", "actual_total_delay", "executed_sha16", "route_sha256", "cohort_sha256")


def _run_all(jobs: list[dict], workers: int) -> None:
    pending = [j for j in jobs if ref.cached(j) is None]
    if pending:
        with ProcessPoolExecutor(max_workers=max(1, min(workers, len(pending))), mp_context=multiprocessing.get_context("spawn"), initializer=ref._worker_init) as pool:
            done = list(pool.map(run_job, pending))
        bad = [d for d in done if not d["ok"]]
        if bad:
            raise RuntimeError(f"점검 실행이 실패했습니다: {bad[:3]}")


def check_paths(world: str, model: Path, out: Path, work: Path, workers: int, situations: int, cfg: dict, evaluation: Path | None = None) -> dict:
    """같은 모델을 기존 경로(계획기 있음)와 계획기 없는 경로로 돌려 신호·입력을 대조한다. 첫 상황은 꼬리 P0로도 돌려 총지연까지 본다."""
    out, work = Path(out), Path(work)
    fp_ = ref.register(out, extra_files=SOURCES)
    prepare_caches(world)
    by_id = {ac.situation_id(s): s for s in eval_situations(world)}
    sits = [by_id[s] for s in cfg["check_situations"][:int(situations)]]
    identity = model_identity(model)

    def jobs(name, mode, chosen, **more):
        rows = ref.jobs_for(out, work, fp_, name, chosen, model_identity=identity, kind="nn")
        for j in rows:
            j.update(world=world, mode=mode, model_dir=str(model), **more)
        return rows

    _run_all([*jobs("planner", "planner", sits), *jobs("free", "free", sits, tail="NN"), *jobs("free_p0tail", "free", sits[:1], tail="P0")], workers)
    read = lambda name, s: ac.read_run(out / "runs" / name / ac.situation_id(s), fp_, identity)  # noqa: E731
    planner, free = {ac.situation_id(s): read("planner", s) for s in sits}, {ac.situation_id(s): read("free", s) for s in sits}
    first = ac.situation_id(sits[0])
    p0tail = read("free_p0tail", sits[0])
    checks = {"same_signals": sum(planner[s]["executed_sha16"] == free[s]["executed_sha16"] for s in planner) == len(sits),
              "same_cars": sum(planner[s]["cohort_sha256"] == free[s]["cohort_sha256"] for s in planner) == len(sits),
              "inputs_identical": all(m["information"]["max_input_diff"] == 0.0 and m["information"]["compared"] > 0 for m in planner.values()),
              "p0_tail_equals_planner_path": all(p0tail[k] == planner[first][k] for k in SAME),
              "tail_guard_active": p0tail["tail_planner_calls"] > 0 and all(m["tail_planner_calls"] == 0 for m in free.values()),
              "quality_clean": not any(ref.quality_errors(m) for m in [*planner.values(), *free.values(), p0tail])}
    if platform.system() != "Darwin" and evaluation is not None:
        checks["same_evaluation"] = ref.evaluation_fingerprint(out) == ref.evaluation_fingerprint(Path(evaluation))
    reference = Path(model) / "free_run.json"
    expected = w.read(reference) if reference.exists() else None
    if expected is not None and first in expected:
        checks["same_as_mac"] = all(free[first][k] == expected[first][k] for k in SAME)
    report = {"ok": all(checks.values()), "checks": checks, "world": world, "model_sha256": identity, "situations": list(planner),
              "free": {s: {k: free[s][k] for k in SAME} for s in free}, "seconds": {"planner": statistics.fmean(m["job_seconds"] for m in planner.values()),
                                                                                   "free": statistics.fmean(m["job_seconds"] for m in free.values())},
              "tail_nn_minus_p0_s": free[first]["mean_delay_s"] - p0tail["mean_delay_s"], "compared_decisions": sum(m["information"]["compared"] for m in planner.values()),
              "fingerprint": fp_}
    w.write(out / "check_paths.json", report)
    ref.stamp(f"NN1 경로 점검 {world}: {'통과' if report['ok'] else '실패'} {checks} | 판당 {report['seconds']}")
    shutil.rmtree(work, ignore_errors=True)
    return report


def check_logs(world: str, teacher: Path, teacher_run: Path, model: Path, out: Path, work: Path, workers: int, cfg: dict, evaluation: Path | None = None) -> dict:
    """선생님이 운전한 기록이 저장된 선생님 실행과 같은가, 신경망이 운전한 기록의 신호가 계획기 없는 실행과 같은가."""
    out, work = Path(out), Path(work)
    fp_ = ref.register(out, extra_files=SOURCES)
    prepare_caches(world)
    s = eval_situations(world)[0]
    sid = ac.situation_id(s)
    identity = model_identity(model)
    base = {"world": world, "teacher_dir": str(teacher), "cfg": {"teachers": cfg["teachers"]}, "mode": "log"}
    jobs = [{**ref.jobs_for(out, work, fp_, "log_teacher", [s], model_identity="teacher", kind="log")[0], **base, "driver_dir": None},
            {**ref.jobs_for(out, work, fp_, "log_driver", [s], model_identity=identity, kind="log")[0], **base, "driver_dir": str(model)},
            {**ref.jobs_for(out, work, fp_, "free", [s], model_identity=identity, kind="nn")[0], "world": world, "mode": "free", "tail": "NN", "model_dir": str(model)}]
    _run_all(jobs, workers)
    read = lambda name, i: ac.read_run(out / "runs" / name / sid, fp_, i)  # noqa: E731
    log_t, log_d, free = read("log_teacher", "teacher"), read("log_driver", identity), read("free", identity)
    stored = w.read(Path(teacher_run) / sid / "metrics.json")
    zt, zd = np.load(out / "runs" / "log_teacher" / sid / "log.npz"), np.load(out / "runs" / "log_driver" / sid / "log.npz")
    rows_t, rows_d = np.arange(len(zt["x"])), np.arange(len(zd["x"]))
    checks = {"teacher_log_equals_stored_run": all(log_t[k] == stored[k] for k in SAME),
              "teacher_labels_executed": bool((zt["teacher"] == zt["executed"]).all()),
              "driver_signals_equal_free_path": log_d["executed_sha16"] == free["executed_sha16"] and log_d["cohort_sha256"] == free["cohort_sha256"],
              "labels_allowed": bool(zt["allowed"][rows_t, zt["teacher"]].all() and zd["allowed"][rows_d, zd["teacher"]].all()),
              "costs_on_allowed_only": bool((np.isfinite(zt["costs"]) == zt["allowed"]).all() and (np.isfinite(zd["costs"]) == zd["allowed"]).all()),
              "input_size": zt["x"].shape[1] == len(COLUMNS[world]) == zd["x"].shape[1],
              "driver_differs_from_teacher_somewhere": int((zd["teacher"] != zd["executed"]).sum()) > 0,
              "quality_clean": not any(ref.quality_errors(m) for m in (log_t, log_d, free))}
    if platform.system() != "Darwin" and evaluation is not None:
        checks["same_evaluation"] = ref.evaluation_fingerprint(out) == ref.evaluation_fingerprint(Path(evaluation))
    report = {"ok": all(checks.values()), "checks": checks, "world": world, "situation": sid, "decisions": {"teacher": int(len(rows_t)), "driver": int(len(rows_d))},
              "driver_teacher_agreement": float((zd["teacher"] == zd["executed"]).mean()), "seconds": {"log_teacher": log_t["job_seconds"], "log_driver": log_d["job_seconds"], "free": free["job_seconds"]},
              "delay": {"log_teacher": log_t["mean_delay_s"], "log_driver": log_d["mean_delay_s"], "free": free["mean_delay_s"], "stored_teacher": stored["mean_delay_s"]}, "fingerprint": fp_}
    w.write(out / "check_logs.json", report)
    ref.stamp(f"NN1 기록 점검 {world}: {'통과' if report['ok'] else '실패'} {checks} | 판당 {report['seconds']}")
    shutil.rmtree(work, ignore_errors=True)
    return report


# ----------------------------------------------------------------------------- 판정(읽기)
def world_reading(table: dict, planner: dict, world: str, cfg: dict) -> dict:
    """사전 등록의 읽기. table[후보] = {mean_delay_s, reduction_vs_fixc_pct}, planner = 같은 세계 계획기."""
    rules, spec = cfg["rules"], cfg["candidates"][world]
    low = min(table[c]["mean_delay_s"] for c in spec)
    near = [c for c in spec if table[c]["mean_delay_s"] <= low + float(rules["tie_s"])]
    best = min(near, key=lambda c: (spec[c]["parts"], table[c]["mean_delay_s"]))
    top = min(spec, key=lambda c: table[c]["mean_delay_s"])
    behind = table[best]["mean_delay_s"] - planner["mean_delay_s"]
    if table[best]["reduction_vs_fixc_pct"] < float(rules["stop_reduction_pct"]):
        action = "stop"
    elif behind > float(rules["planner_gap_s"]):
        action = "more_labels"
    else:
        action = "go"
    return {"start_model": best, "shortest": top, "behind_planner_s": behind, "start_reduction_pct": table[best]["reduction_vs_fixc_pct"], "action": action,
            "basic_met": table[best]["reduction_vs_fixc_pct"] > float(rules["basic_reduction_pct"]), "target_met": behind <= 0.0}


def _nn_rows(folder: Path, candidate: str, sits: list[dict]) -> tuple[list[list[dict]] | None, dict | None, list]:
    """후보 평가의 실행을 지문·모델 표시·차량 파일까지 대조해 읽는다. 아직 끝나지 않았으면 (None, None, 남은 수)."""
    if not (Path(folder) / "models.json").exists():
        return None, None, ["평가 폴더 없음"]
    record = s2e.folder_record(folder)
    decl = w.read(Path(folder) / "models.json")
    if decl.get("candidate") != candidate:
        raise ValueError(f"평가 폴더의 후보가 다릅니다: {folder}")
    models = [pl._rows(folder, f"{candidate}_m{i}", row["identity"], sits, record) for i, row in enumerate(decl["models"])]
    lacking = [len(sits) - len(r) for r in models]
    return (None, None, lacking) if any(lacking) else (models, record, [])


def _evaluation_ok(record: dict, reference: dict, packages: dict, name: str) -> None:
    if record["evaluation_fingerprint"] == reference["evaluation_fingerprint"]:
        return
    if not lb.differs_only_in_packages(record["evaluation"], reference["evaluation"]):
        raise ValueError(f"평가 기록이 패키지 버전 말고도 다릅니다: {name}")
    packages[name] = record["evaluation"]["packages"]


def reference_table(world: str, sits: list[dict], paths: dict, cfg: dict) -> dict:
    """기존 기준 실행을 다시 돌리지 않고 읽는다. {이름: 모델별 실행 목록}, 평가 지문 기준 폴더 기록."""
    sids = [ac.situation_id(s) for s in sits]
    route = {ac.situation_id(s): s["route_sha256"] for s in sits}
    pbase = Path(paths["pbase"])
    out = {"FIXC": [pb.read_folder(pbase / "runs" / world / "FIXC", sids, route, f"{world}/FIXC")]}
    if world == "P":
        pccfg = pc.config()
        rows = pc.reference_rows(Path(paths["platoon0"]), Path(paths["platoon1"]), sits, pccfg)
        out.update({"M4": [rows["base"]["M4"]], "TM4": [rows["base"]["TM4"]], "OM4": [rows["base"]["OM4"]],
                    "P0": [pb.read_folder(pbase / "runs/P/P0", sids, route, "P/P0")]})
        t20, _ = pc._candidate_rows(Path(paths["platoon1c"]) / "eval" / "t20", "t20", float(pccfg["candidates"]["t20"]["tau"]), sits, pccfg)
        if any(len(r) != len(sits) for r in t20):
            raise ValueError("t20 평가가 84상황을 모두 갖지 않았습니다.")
        out["t20"] = t20
        record = s2e.folder_record(Path(paths["platoon0"]))
    else:
        record = s2e.verify_baseline(Path(paths["baseline"]))
        out.update({"M4": [s2e.verified_runs(Path(paths["baseline"]), "M4", sits)], "P0": [s2e.verified_runs(Path(paths["baseline"]), "P0", sits)]})
        for name in ("S2A_pre", "S2B"):
            folder = Path(paths["stage2"]) / "eval" / name
            decl = w.read(folder / "models.json")
            out[name] = [s2e.verified_runs(folder, f"{name}_m{i}", sits, model_identity=x) for i, x in enumerate(decl["model_identities"])]
    return {"runs": out, "record": record}


def world_report(world: str, root: Path, paths: dict, cfg: dict, lock: dict) -> dict:
    sits, t975 = eval_situations(world), float(lock["judge"]["t975"])
    refs = reference_table(world, sits, paths, cfg)
    runs, packages, missing = dict(refs["runs"]), {}, {}
    for name, spec in cfg["candidates"][world].items():
        folder = Path(root) / world / ("aug" if spec["kind"] == "aug" else "base") / "eval" / name
        models, record, lacking = _nn_rows(folder, name, sits)
        if models is None:
            missing[name] = lacking
            continue
        _evaluation_ok(record, refs["record"], packages, name)
        runs[name] = models
    if missing:
        return {"status": "incomplete", "missing": missing}
    m4, p0_rows = runs["M4"][0], runs["P0"][0]
    results, table = {}, {}
    fixed = pb.policy_summary(runs["FIXC"])["mean_delay_s"]
    for name, models in runs.items():
        row = pb.policy_summary(models)
        row["reduction_vs_fixc_pct"] = 100.0 * (fixed - row["mean_delay_s"]) / fixed
        if len(models) > 1:
            each = [pb.policy_summary([m])["mean_delay_s"] for m in models]
            row["per_model_delay_s"] = each
            row["per_model_reduction_pct"] = [100.0 * (fixed - x) / fixed for x in each]
        results[name] = ref.judge(models, m4, p0_rows, lock, sits)
        table[name] = row
        if name in cfg["candidates"][world]:
            info = [m["information"] for rows in models for m in rows]
            planned = sum(int(x.get("planned", 0)) for x in info)
            row["decisions_per_run"] = planned / len(info)
            row["student_change_share"] = sum(int(x.get("student_changed", 0)) for x in info) / max(planned, 1)
            row["z5_cells_per_decision"] = sum(int(x.get("z5_cells", 0)) for x in info) / max(planned, 1)
            row["tail_planner_calls"] = sum(int(m.get("tail_planner_calls", 0)) for rows in models for m in rows)
            row["seconds"] = statistics.fmean(float(m["job_seconds"]) for rows in models for m in rows)
    paired = {f"{a}-{b}": pl.paired_difference(results[a], results[b], t975) for a, b in cfg["rules"]["pairs"][world]}
    planner_name = cfg["reference_planner"][world]
    planner = table[planner_name]
    periods = {n: per_period_gap(results[n], results[planner_name], sits) for n in cfg["candidates"][world]}
    return {"status": "done", "table": table, "paired": paired, "reading": world_reading(table, planner, world, cfg), "planner": planner_name, "per_period_vs_planner": periods,
            "arena3": {n: {"status": r["status"], "gain_vs_m4_s": r["gain_date_mean_s"], "date_t95_s": r["date_t95_s"], "positive_dates": r["positive_dates"],
                           "collisions_specific": len(r["collisions"]["policy_specific"]), "collisions_shared": len(r["collisions"]["shared_with_m4_or_p0"]),
                           "quality_errors": len(r["quality_errors"])} for n, r in results.items()},
            "package_only_differences": packages, "evaluation_fingerprint": refs["record"]["evaluation_fingerprint"]}


def per_period_gap(result: dict, planner: dict, sits: list[dict]) -> dict:
    """시간대별 (계획기 − 후보) 차량당 지연의 상황 평균(+면 후보가 짧음)."""
    out: dict[str, list[float]] = {}
    for s in sits:
        sid = ac.situation_id(s)
        out.setdefault(s["period"], []).append(float(result["per_situation_gain_s"][sid]) - float(planner["per_situation_gain_s"][sid]))
    return {p: statistics.fmean(v) for p, v in sorted(out.items())}


def collect_report(root: Path, paths: dict, cfg: dict) -> dict:
    lock = ac.config()
    worlds = {x: world_report(x, root, paths, cfg, lock) for x in ("P", "R")}
    return {"experiment": EXPERIMENT, "status": "done" if all(v["status"] == "done" for v in worlds.values()) else "incomplete", "worlds": worlds,
            "lock_sha256": w.file_hash(ac.CONFIG), "config_sha256": w.file_hash(CONFIG), "updated_unix": time.time(), "note": cfg["reporting"]}


ACTION_TEXT = {"go": "계획기와 0.5초 안: 그대로 2단계", "more_labels": "계획기보다 0.5초 넘게 뒤짐: 라벨 라운드를 한 번 더 돌고 2단계",
               "stop": "−22%보다 못함: 3단계로 가지 않고 사용자와 다시 정함"}
WORLD_NAME = {"P": "무리 도착 세계(모형)", "R": "무작위 도착 세계"}


def render(report: dict) -> str:
    f = lambda x, nd=2: f"{x:+.{nd}f}"  # noqa: E731
    lines = ["# NN1 — 출발 모델 7종을 84상황에서 잰 결과(계획기 없는 실행, 꼬리까지 신경망)", "", "고정신호 대비 감소율을 먼저 본다. 짝 비교는 같은 상황·날짜 7개, +면 앞쪽이 짧다. 흉내 값은 강화학습 값이 아니다.", ""]
    for world, part in report["worlds"].items():
        lines += [f"## {WORLD_NAME[world]}", ""]
        if part["status"] != "done":
            lines += [f"아직 끝나지 않았습니다: {json.dumps(part['missing'], ensure_ascii=False)}", ""]
            continue
        lines += ["| 제어 | 차량당 지연 | 고정신호 대비 | 모델별 | 동 좌회전 | 남 좌회전 | 300초 초과 | 미확인 충돌 | 품질 오류 | 학생이 바꾼 비율 |", "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|"]
        for name, r in part["table"].items():
            each = " / ".join(f"{-x:.1f}" for x in r["per_model_reduction_pct"]) if "per_model_reduction_pct" in r else "-"
            change = "-" if r.get("student_change_share") in (None, 0) else f"{100 * r['student_change_share']:.1f}%"
            lines.append(f"| {name} | {r['mean_delay_s']:.2f} | {-r['reduction_vs_fixc_pct']:.1f}% | {each} | {r['left_turns_s']['left_east']:.1f} | {r['left_turns_s']['left_south']:.1f} | "
                         f"{r['long_waits']:.1f} | {r['collisions_unexplained']:.1f} | {r['quality_errors']} | {change} |")
        lines += ["", "| 비교 | 평균(초) | t95 | 앞쪽이 나은 날짜 |", "|---|---:|---|---:|"]
        for key, p in part["paired"].items():
            lines.append(f"| {key.replace('-', ' − ')} | {f(p['mean_s'], 3)} | [{f(p['t95_s'][0], 3)}, {f(p['t95_s'][1], 3)}] | {p['positive_dates']}/{p['dates']} |")
        names = list(next(iter(part["per_period_vs_planner"].values())))
        lines += ["", f"시간대별 {part['planner']} 대비(초, +면 후보가 짧음)", "", "| 후보 | " + " | ".join(names) + " |", "|---|" + "---:|" * len(names)]
        for name, row in part["per_period_vs_planner"].items():
            lines.append(f"| {name} | " + " | ".join(f(row[p]) for p in names) + " |")
        v = part["reading"]
        lines += ["", f"**읽기:** 출발 모델 {v['start_model']}(가장 짧은 모델 {v['shortest']}), 고정신호 대비 {-v['start_reduction_pct']:.1f}%, 같은 세계 계획기보다 {f(-v['behind_planner_s'])}초. {ACTION_TEXT[v['action']]}.", ""]
    lines.append(f"- {report['note']}")
    return "\n".join(lines) + "\n"


def save_report(out: Path, report: dict) -> None:
    out = Path(out)
    s2e._write(out / "judge.json", report)
    (out / "summary.md").write_text(render(report), encoding="utf-8")


# ----------------------------------------------------------------------------- 명령
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("make-config")
    p.add_argument("--workers", type=int, default=8)
    p = sub.add_parser("train")
    for name in ("world", "data", "out"):
        p.add_argument(f"--{name}", required=True)
    p.add_argument("--teacher", help="무리 세계 선생님 학생 폴더(PT1 난수 42)")
    p = sub.add_parser("check-paths")
    for name in ("world", "model", "out", "work"):
        p.add_argument(f"--{name}", required=True)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--situations", type=int, default=1)
    p.add_argument("--evaluation", help="평가 지문을 대조할 이전 결과 폴더(Colab)")
    p = sub.add_parser("check-logs")
    for name in ("world", "teacher", "teacher-run", "model", "out", "work"):
        p.add_argument(f"--{name}", required=True)
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--evaluation", help="평가 지문을 대조할 이전 결과 폴더(Colab)")
    p = sub.add_parser("evaluate")
    for name in ("world", "candidates", "models", "out", "work"):
        p.add_argument(f"--{name}", required=True)
    p.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1))
    p = sub.add_parser("augment")
    for name in ("world", "teacher", "out", "work"):
        p.add_argument(f"--{name}", required=True)
    p.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1))
    p = sub.add_parser("judge")
    for name in ("root", "platoon0", "platoon1", "platoon1c", "pbase", "baseline", "stage2", "out"):
        p.add_argument(f"--{name}", required=True)
    args = parser.parse_args()
    w.configure()
    torch.set_num_threads(1)
    if args.cmd == "make-config":
        print(json.dumps(make_config(args.workers), ensure_ascii=False, indent=1))
        return 0
    cfg = config()
    if args.cmd == "train":
        summary = train_base(args.world, Path(args.data), Path(args.teacher) if args.teacher else None, Path(args.out), cfg)
        print(json.dumps(w.plain(summary), ensure_ascii=False)[:1500])
        return 0
    if args.cmd == "check-paths":
        r = check_paths(args.world, Path(args.model), Path(args.out), Path(args.work), args.workers, args.situations, cfg, Path(args.evaluation) if args.evaluation else None)
        return 0 if r["ok"] else 2
    if args.cmd == "check-logs":
        r = check_logs(args.world, Path(args.teacher), Path(args.teacher_run), Path(args.model), Path(args.out), Path(args.work), args.workers, cfg, Path(args.evaluation) if args.evaluation else None)
        return 0 if r["ok"] else 2
    if args.cmd == "evaluate":
        failed = 0
        for candidate in args.candidates.split(","):
            s = evaluate(args.world, candidate, Path(args.models), Path(args.out) / candidate, Path(args.work) / candidate, args.workers, cfg)
            print(json.dumps({"candidate": candidate, "done": s["done"], "failed": len(s["failed"])}, ensure_ascii=False), flush=True)
            failed += len(s["failed"])
        return 1 if failed else 0
    if args.cmd == "augment":
        return augment(args.world, Path(args.teacher), Path(args.out), Path(args.work), args.workers, cfg)
    paths = {k: Path(getattr(args, k)) for k in ("platoon0", "platoon1", "platoon1c", "pbase", "baseline", "stage2")}
    report = collect_report(Path(args.root), paths, cfg)
    save_report(Path(args.out), report)
    print(render(report))
    return 0 if report["status"] == "done" else 3


if __name__ == "__main__":
    raise SystemExit(main())
