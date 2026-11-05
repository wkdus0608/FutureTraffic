"""ARENA3 정책 반복(PI) — 지금 정책이 운전하는 학습 상황에서 허용 녹색을 모두 같은 표본 미래로 굴려 '더 나은 녹색'을 배운다(235·236절).

방법(롤아웃 분류 정책 반복, Lagoudakis·Parr 2003 RCPI · Lazaric 외 DPI 계열의 강화학습):
  1. 지금 정책 π(1라운드는 S2A_pre 난수 42)가 학습 상황 64판을 운전한다. 행동은 언제나 π의 선택이고 롤아웃 결과를 기다리지 않는다.
  2. 결정 지점마다 상태를 저장한다. 갈림 지점(여섯 모드의 첫 녹색이 다름, 시간당 약 29곳)은 모두,
     합의 지점(허용 녹색이 2개 이상인데 모드는 같음, 시간당 약 59곳)은 해시 규칙으로 25%만. 허용 녹색이 하나인 곳(약 632곳)은 고를 것이 없다.
  3. 그 순간 허용된 녹색마다 표본 미래 8개로 300초를 굴린다. 첫 5초는 그 녹색, 이후는 π, 제어 60분이 끝나면 P0(본 에피소드와 같음).
     비용은 B′와 같다(300초 지연 + 남은 정체 × 75초). 같은 번호의 미래끼리 짝을 지어 비교한다.
  4. 관측 119칸만 보는 신경망(녹색 4개의 M4 녹색 대비 값)이 짝 평균 이득을 배운다(arena3_pi_learn.py). 다음 라운드에서는 이 신경망이 π다.
행동 세 가지(같은 신경망, 고를 수 있는 녹색만 다름): M = 갈림 지점에서 모드가 내는 녹색만(지금 행동 범위),
G = 갈림 지점에서 허용 녹색 전부, A = 갈림·합의 지점 모두에서 허용 녹색 전부.
에피소드 실행기·심판·잠금 파일은 고치지 않는다. 실행기는 P0 종류의 정책만 모든 제어 단계에서 부르므로,
모든 단계에서 봐야 하는 수집기와 평가 정책은 그 종류로 둔다(P0 일치 검사 값은 늘리지 않는다).

명령(Colab; lock만 맥):
  python src/arena3_pi.py lock     --work /tmp/pi_lock                      (설정 파일을 만들거나 같은지 확인, SUMO 없음)
  python src/arena3_pi.py smoke    --out <Drive>/arena3/pi/R1/smoke/shard0 --work /content/pi_smoke --base-root <Drive>/arena3/stage2/S2A_train
  python src/arena3_pi.py collect  --out <Drive>/arena3/pi/R1/data --work /content/pi --shard 0 --shards 3 --base-root <…>/S2A_train
  python src/arena3_pi.py offline  --data <Drive>/arena3/pi/R1/data --out <Drive>/arena3/pi/R1/offline
  python src/arena3_pi.py train    --data <Drive>/arena3/pi/R1/data --out <Drive>/arena3/pi/R1/students
  python src/arena3_pi.py evaluate --variant G --students <…>/R1/students --out <Drive>/arena3/pi/R1/eval/R1G --work /content/pi_eval
  python src/arena3_pi.py judge    --baseline <Drive>/arena3/referee/baseline --pi R1M=<…> --pi R1G=<…> --pi R1A=<…> --reference S2A_pre=<…> … --out <…>/R1/judge
  python src/arena3_pi.py confirm  --judge <…>/R1/judge --baseline <Drive>/arena3/stage2/confirm/baseline --out <…>/R1/judge/confirm --work /content/pi_confirm
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import multiprocessing
import os
import pickle
import platform
import shutil
import statistics
import sys
import tempfile
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import arena3_bprime as bp  # noqa: E402
import arena3_common as ac  # noqa: E402
import arena3_g1 as g1  # noqa: E402
import arena3_referee as ref  # noqa: E402
import explore_weight as w  # noqa: E402

ROUNDS = ("R1",)
STAGE2_CONFIG = ROOT / "config/arena3_stage2_v1.json"
# 수집 결과를 정하는 코드(자료 폴더 지문). 학생·평가 코드(arena3_pi_learn.py)는 평가 폴더에만 더한다.
COLLECT_SOURCES = ("src/arena3_pi.py", "src/arena3_bprime.py", "src/arena3_g1.py", "src/arena3_stage2_train.py",
                   "src/arena3_stage2_data.py", "config/arena3_stage2_v1.json")
POLICY = "PI_DATA"
OBS, GREENS = ac.OBSERVATION_SIZE, 4
ALLOWED = slice(60, 64)                     # WEIGHT1 원래 64칸의 끝 4칸 = 지금 실행할 수 있는 녹색(explore_weight_core.ModePlanner)
M4_ONEHOT = slice(84 + 4 * ac.M4, 88 + 4 * ac.M4)
VARIANTS = ("M", "G", "A")
CONTROL_S = 3600
KEEP = ("delay", "time_loss", "depart_wait", "residual", "cost", "left_east", "left_south", "injected",
        "collisions_unexplained", "collisions_at_204820", "future_seed", "seconds")


# ----------------------------------------------------------------------------- 설정
def config_path(round_name: str) -> Path:
    if round_name not in ROUNDS:
        raise ValueError(f"준비된 라운드는 {ROUNDS}입니다: {round_name}")
    return ROOT / f"config/arena3_pi_{round_name.lower()}.json"


def sources(round_name: str) -> tuple[str, ...]:
    return (*COLLECT_SOURCES, f"config/arena3_pi_{round_name.lower()}.json")


def config(round_name: str = "R1") -> dict:
    cfg = w.read(config_path(round_name))
    if cfg.get("experiment") != f"ARENA3_PI_{round_name}" or cfg.get("round") != round_name:
        raise ValueError("정책 반복 설정 파일이 아닙니다.")
    if cfg["base_lock_sha256"] != w.file_hash(ac.CONFIG) or cfg["stage2_config_sha256"] != w.file_hash(STAGE2_CONFIG):
        raise ValueError("정책 반복 설정이 가리키는 ARENA3 잠금·2단계 설정이 지금 파일과 다릅니다.")
    check_constants(cfg)
    return cfg


def check_constants(cfg: dict) -> None:
    """설정에 적은 롤아웃 값이 실제로 쓰는 코드의 상수와 같은지(설정만 바꿔서는 바뀌지 않는 값들)."""
    import explore_rollout as er
    from explore_weight_episode import CONTROL_S as control, STEP_S
    r = cfg["rollout"]
    if (r["terminal_s"], r["history_s"], r["first_action_s"]) != (er.TERMINAL_S, er.HISTORY_S, STEP_S) or control != CONTROL_S:
        raise ValueError("설정의 롤아웃 값이 explore_rollout·실행기 상수와 다릅니다.")


def data_situations(lock: dict, spec: dict) -> list[dict]:
    """학습 8일마다 15분 칸 8개(해시 순서 앞쪽), 새 도착 배치 이름(B′ 자료 T1~T64와 다름)."""
    out, n = [], 0
    for d in lock["dates"]["train"]:
        slots = sorted(ac.TRAIN_SLOTS, key=lambda p: hashlib.sha256(f"{spec['slot_salt']}|{d}|{p}".encode()).hexdigest())
        for p in sorted(slots[:spec["per_date"]]):
            n += 1
            out.append(ac.make_situation(d, p, f"{spec['tag_prefix']}{n}"))
    return out


def future_seed(cfg: dict, state_id: str, index: int) -> int:
    return ac.seed32(cfg["rollout"]["future_salt"], state_id, index)


def agree_sampled(cfg: dict, sid: str, step: int) -> bool:
    """합의 지점 표본(결과와 무관한 해시 규칙): 같은 상황·같은 단계는 언제나 같은 답."""
    a = cfg["agree_sample"]
    value = int(hashlib.sha256(f"{a['salt']}|{sid}|{int(step)}".encode()).hexdigest()[:8], 16) % 10000
    return value < round(float(a["fraction"]) * 10000)


def keep_state(cfg: dict, state_id: str) -> bool:
    value = int(hashlib.sha256(f"ARENA3_PI_KEEP|{state_id}".encode()).hexdigest()[:8], 16) % 1000
    return value < round(float(cfg["rollout"]["keep_state_fraction"]) * 1000)


# ----------------------------------------------------------------------------- 관측에서 녹색 정보(순수 계산)
def allowed_mask(obs) -> np.ndarray:
    return np.asarray(obs, np.float32)[..., ALLOWED] > 0.5


def mode_greens(obs) -> np.ndarray:
    """여섯 모드의 첫 녹색(M0~M4는 84:104, W4는 110:114)."""
    obs = np.asarray(obs, np.float32)
    rows = np.concatenate((obs[..., 84:104].reshape(*obs.shape[:-1], 5, 4), obs[..., 110:114].reshape(*obs.shape[:-1], 1, 4)),
                          axis=-2)
    return rows.argmax(axis=-1)


def action_mask(obs, variant: str) -> np.ndarray:
    """변형마다 고를 수 있는 녹색(…, 4). M4 녹색은 언제나 들어 있고, 고를 수 없는 곳은 M4 녹색 하나다."""
    if variant not in VARIANTS:
        raise ValueError(f"알 수 없는 변형: {variant}")
    obs = np.asarray(obs, np.float32)
    allowed = allowed_mask(obs)
    greens = mode_greens(obs)
    eye = np.eye(GREENS, dtype=bool)
    m4 = eye[greens[..., ac.M4]]
    proposed = eye[greens].any(axis=-2)
    choice = proposed.sum(axis=-1) > 1
    if np.any(proposed & ~allowed) or np.any(m4 & ~allowed):
        raise ValueError("모드가 실행할 수 없는 녹색을 냈다는 관측입니다.")
    if variant == "M":
        mask = np.where(choice[..., None], proposed, m4)
    elif variant == "G":
        mask = np.where(choice[..., None], allowed, m4)
    else:
        mask = np.where((allowed.sum(axis=-1) >= 2)[..., None], allowed, m4)
    return mask


class GreenNet(nn.Module):
    """관측 119칸 → 녹색 4개의 값(M4 녹색 = 0). 이점 머리를 0으로 시작해 학습 전 탐욕 정책은 M4다(G1·S2와 같은 64-64)."""

    def __init__(self, hidden=(64, 64)):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(OBS, hidden[0]), nn.ReLU(), nn.Linear(hidden[0], hidden[1]), nn.ReLU())
        self.advantage = nn.Linear(hidden[1], GREENS)
        nn.init.zeros_(self.advantage.weight)
        nn.init.zeros_(self.advantage.bias)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        obs = obs.reshape(-1, OBS)
        adv = self.advantage(self.trunk(obs))
        return adv - adv.gather(1, obs[:, M4_ONEHOT].argmax(dim=1, keepdim=True))


def greedy_green(net: GreenNet, obs, variant: str) -> int:
    """고를 수 있는 녹색 중 가장 큰 값(같으면 M4 녹색). 고를 것이 없으면 신경망을 부르지 않는다."""
    obs = np.asarray(obs, np.float32)
    mask = action_mask(obs, variant)
    m4 = int(mode_greens(obs)[ac.M4])
    if mask.sum() == 1:
        return m4
    with torch.no_grad():
        rel = net(torch.from_numpy(obs).view(1, -1))[0].numpy().astype(np.float64)
    if not np.all(np.isfinite(rel)):
        raise ValueError("녹색 값이 유한하지 않습니다.")
    rel = np.where(mask, rel, -np.inf)
    rel[m4] += 1e-6
    return int(rel.argmax())


def point_kind(snap) -> str | None:
    if len(snap.allowed) < 2:
        return None
    return "choice" if snap.choice else "agree"


# ----------------------------------------------------------------------------- 정책
class ModeBase:
    """2단계 6모드 모델(S2A_pre 등): 갈림 지점에서만 고르고 나머지는 모든 모드가 같은 녹색(g1.GreedyPolicy와 같은 선택)."""
    kind = "mode"

    def __init__(self, net: g1.DuelingNet):
        self.net = net

    def act(self, snap) -> int:
        return snap.green(g1.greedy_mode(self.net, snap.observation)) if snap.choice else int(snap.greens[0])


class GreenBase:
    """정책 반복 학생(녹색 직접 선택): 변형이 허락하는 곳에서 고른다."""
    kind = "green"

    def __init__(self, net: GreenNet, variant: str):
        self.net, self.variant = net, variant

    def act(self, snap) -> int:
        return greedy_green(self.net, snap.observation, self.variant)


class GreenPolicy(ac.P0Policy):
    """심판 평가용. 합의 지점에서도 고르는 변형(A)이 있어 실행기가 모든 제어 단계에서 부르는 P0 종류로 둔다(M·G도 같은 길)."""

    def __init__(self, net: GreenNet, variant: str, name: str):
        self.base, self.name = GreenBase(net, variant), name

    def decide(self, snap, ctx) -> int:
        return self.base.act(snap)


def referee_policy(base, name: str):
    """수집 점검에서 '평가 때와 같은 정책'으로 쓸 것: 6모드 모델은 2단계 평가와 같은 g1.GreedyPolicy."""
    if base.kind == "mode":
        return g1.GreedyPolicy(base.net, name=name)
    return GreenPolicy(base.net, base.variant, name)


def _state_bytes(net: nn.Module) -> bytes:
    buf = io.BytesIO()
    torch.save(net.state_dict(), buf)
    return buf.getvalue()


def base_payload(spec: dict, root: Path) -> dict:
    """설정의 지금 정책을 읽어 해시를 대조하고, 작업자에 넘길 바이트로 만든다."""
    folder = Path(root) / spec["folder"]
    if spec["kind"] == "mode":
        import arena3_stage2_train as s2
        manifest = w.read(folder / "manifest.json")
        if (manifest.get("variant"), manifest.get("stage"), manifest.get("seed")) != (spec["stage2_variant"], spec["stage"], spec["seed"]):
            raise ValueError(f"지금 정책 폴더가 설정({spec['name']})과 다릅니다: {folder}")
        sha, net = s2.model_identity(folder), s2.load_model(folder)
    elif spec["kind"] == "green":
        import arena3_pi_learn as pl
        sha, net = pl.student_identity(folder), pl.load_student(folder)
    else:
        raise ValueError(f"알 수 없는 정책 종류: {spec['kind']}")
    if sha != spec["model_sha256"]:
        raise ValueError(f"지금 정책 모델이 설정에 적은 모델과 다릅니다: {sha} ≠ {spec['model_sha256']}")
    return {"kind": spec["kind"], "variant": spec.get("variant"), "name": spec["name"], "sha256": sha,
            "state": _state_bytes(net)}


def base_from_payload(payload: dict):
    if payload["kind"] == "mode":
        net = g1.DuelingNet(tuple(g1.gcfg()["network"]))
    else:
        net = GreenNet()
    net.load_state_dict(torch.load(io.BytesIO(payload["state"]), map_location="cpu", weights_only=True))
    net.eval()
    return ModeBase(net) if payload["kind"] == "mode" else GreenBase(net, payload["variant"])


# ----------------------------------------------------------------------------- 롤아웃(작업자 프로세스)
_BASE = None


def _pool_init(payload: dict) -> None:
    global _BASE
    ref._worker_init()
    torch.set_num_threads(1)
    _BASE = base_from_payload(payload)


def rollout_pool(workers: int, payload: dict):
    return ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                               initializer=_pool_init, initargs=(payload,))


def rollout_worker(job: dict) -> dict:
    """저장 상태에서 표본 미래 하나로 300초: 첫 5초는 후보 녹색, 이후 지금 정책(제어가 끝나면 P0). 결과는 메모리로 돌려준다."""
    import explore_rollout as er
    from focused_eval import judge_collisions
    started = time.monotonic()
    env = None
    fd, trip_name = tempfile.mkstemp(prefix="pi_trip_", suffix=".xml")
    os.close(fd)
    fd, coll_name = tempfile.mkstemp(prefix="pi_coll_", suffix=".xml")
    os.close(fd)
    trip, coll = Path(trip_name), Path(coll_name)
    try:
        if _BASE is None:
            raise RuntimeError("작업자에 지금 정책이 없습니다(rollout_pool로 만든 작업자만 씁니다).")
        horizon = int(job["horizon_s"])
        env, cons, ts, ctl, snap, injected, vehicles = bp._restore_env(job, "sampled", int(job["future_seed"]), horizon,
                                                                         trip, coll)
        control_end = float(snap["handover"]) + CONTROL_S
        planner = ac._planner_class()(ctl.movement_of, [p.state for p in ts.green_phases], control_end)
        check = planner.arena_snapshot(env)
        if list(check.greens) != job["greens"] or list(check.allowed) != job["allowed"]:
            raise ValueError("표본 미래를 넣은 뒤 여섯 모드의 첫 녹색 또는 허용 녹색이 저장 순간과 다릅니다.")
        if np.max(np.abs(check.observation - np.asarray(job["observation"], np.float32))) > 1e-6:
            raise ValueError("표본 미래를 넣은 뒤 관측 119칸이 저장 순간과 다릅니다(미래가 관측에 샜거나 복원 오차).")
        if int(job["candidate"]) not in check.allowed:
            raise ValueError("후보가 허용 녹색이 아닙니다.")
        meter = bp._meter_class()(env, snap["t0"], vehicles, injected, marks=(horizon,))

        def follow(e, allowed):
            if float(e.sim_step) >= control_end - 1e-9:
                return int(ctl(e, allowed))                  # 본 에피소드처럼 제어 60분 뒤는 P0
            s = planner.arena_snapshot(e)
            if list(s.allowed) != list(allowed):
                raise ValueError("계획기와 운영 규칙의 실행 가능 녹색이 다릅니다.")
            return int(_BASE.act(s))

        _, changed = er.drive(env, cons, follow, int(job["candidate"]), snap["t0"], horizon, meter.step)
        env.close()
        env = None
        meter.finish(trip)
        if horizon not in meter.at:
            raise ValueError("300초 비용이 없습니다.")
        collision = judge_collisions(coll)
        r = dict(meter.at[horizon])
        r.update({"candidate": int(job["candidate"]), "future": int(job["index"]), "future_seed": int(job["future_seed"]),
                  "leaks": len(meter.leaks), "changed_by_rules": int(changed), "injected": len(injected),
                  "collisions_unexplained": len(collision["collisions_unexplained"]),
                  "collisions_at_204820": int(collision["collisions_at_204820"]), "seconds": time.monotonic() - started})
        return {"ok": True, **r}
    except BaseException as error:
        return {"ok": False, "candidate": job.get("candidate"), "future": job.get("index"),
                "error": f"{type(error).__name__}: {error}", "seconds": time.monotonic() - started}
    finally:
        if env is not None:
            env.close()
        trip.unlink(missing_ok=True)
        coll.unlink(missing_ok=True)


# ----------------------------------------------------------------------------- 수집기(본 에피소드)
class Collector(ac.P0Policy):
    """본 에피소드: 행동은 언제나 지금 정책의 선택. 결정 지점에서는 상태를 저장하고 롤아웃을 작업자에 맡긴 뒤 기다리지 않고 진행한다.

    실행기가 모든 제어 단계에서 부르도록 P0 종류로 둔다. 갈림 지점이 아닌 곳의 행동은 모든 모드의 같은 녹색이라 평소 실행과 같다.
    limit(점검용) = 종류별 최대 지점 수 {"choice": n, "agree": n}.
    """

    def __init__(self, base, pool, cfg: dict, situation: dict, state_dir: Path, keep_dir: Path | None = None, *,
                 limit: dict | None = None):
        self.base, self.pool, self.cfg = base, pool, cfg
        self.situation, self.state_dir, self.keep_dir = situation, Path(state_dir), keep_dir
        self.sid = ac.situation_id(situation)
        self.futures, self.horizon = int(cfg["rollout"]["futures"]), int(cfg["rollout"]["horizon_s"])
        self.limit = limit
        self.name = f"PI_COLLECT_{cfg['round']}"
        self.points: list[dict] = []
        self.rollouts: list[tuple] = []          # (지점 번호, 후보 녹색, 미래 번호, Future)
        self.verifies: list[tuple] = []          # (지점 번호, Future)
        self.counts = {"forced": 0, "choice": 0, "agree": 0, "agree_sampled": 0}

    def decide(self, snap, ctx) -> int:
        action = int(self.base.act(snap))
        kind = point_kind(snap)
        self.counts[kind or "forced"] += 1
        if kind is None:
            return action
        if kind == "agree":
            if not agree_sampled(self.cfg, self.sid, ctx.step):
                return action
            self.counts["agree_sampled"] += 1
        if self.limit is not None and sum(p["kind"] == kind for p in self.points) >= self.limit.get(kind, 0):
            return action
        self._submit(snap, ctx, kind, action)
        return action

    def _submit(self, snap, ctx, kind: str, action: int) -> None:
        import explore_m4probe_pending as pend
        import explore_rollout as er
        ep = ctx.ep
        k = len(self.points)
        state_id = f"{self.sid}_p{k:03d}"
        stem = self.state_dir / state_id
        saved = er.Episode.snapshot(ep, stem, snap.allowed, snap.p0_action)
        saved["pending_snapshot"] = pend.capture_pending(ep.env.sumo)
        stem.with_suffix(".pkl").write_bytes(pickle.dumps(saved))
        er.types_only_file(ep.route)          # 작업자들이 동시에 만들지 않게 미리 만든다
        base = {"stem": str(stem), "route_dir": str(ep.route.parent), "sumo_seed": int(self.situation["sumo_seed"]),
                "greens": list(snap.greens), "allowed": list(snap.allowed), "horizon_s": self.horizon,
                "observation": snap.observation.tolist()}
        self.points.append({"state": state_id, "kind": kind, "step": int(ctx.step), "now": float(snap.now),
                            "allowed": list(snap.allowed), "greens": list(snap.greens), "mask": list(snap.mask),
                            "p0": int(snap.p0_action), "m4": int(snap.greens[ac.M4]), "executed": int(action),
                            "observation": snap.observation.tolist(), "m4_costs": snap.m4_costs, "w4_cost": snap.w4_cost})
        for a in snap.allowed:
            for j in range(self.futures):
                job = {**base, "candidate": int(a), "index": j, "future_seed": future_seed(self.cfg, state_id, j)}
                self.rollouts.append((k, int(a), j, self.pool.submit(rollout_worker, job)))
        self.verifies.append((k, self.pool.submit(bp.verify_worker, {**base, "green": int(action)})))

    def after_step(self, ctx, snap, action, executed) -> None:
        """결정 지점 실행 직후 원래 실행의 표를 남긴다(복원 확인용)."""
        if self.points and self.points[-1]["step"] == ctx.step and "main_after" not in self.points[-1]:
            self.points[-1]["main_after"] = bp.trace_table(ctx.ep.env.sumo)

    def outstanding(self) -> int:
        return sum(not f.done() for *_, f in self.rollouts) + sum(not f.done() for _, f in self.verifies)

    def futures_list(self) -> list:
        return [f for *_, f in self.rollouts] + [f for _, f in self.verifies]

    def spent(self) -> float:
        """끝난 작업들의 계산 시간 합(실패한 판도 장부에 남긴다)."""
        done = [f for f in self.futures_list() if f.done() and not f.cancelled() and f.exception() is None]
        return sum(float(f.result().get("seconds") or 0.0) for f in done)

    def finish(self) -> dict:
        """모든 롤아웃이 끝난 뒤: 비용을 지점에 붙이고 복원을 확인한다.

        미래 엿봄·규칙 변경은 하나라도 있으면 이 판 전체를 실패로 둔다(정확성 문제). 실패한 롤아웃은 설정 비율 이하이면
        그 칸을 비워 두고(자료를 읽을 때 그 미래를 모든 후보에서 뺀다) 오류를 지점에 적고, 넘으면 판 전체를 실패로 둔다.
        """
        tol = float(self.cfg["rollout"]["restore_check"]["tolerance"])
        cpu, errors, failures = 0.0, [], []
        for p in self.points:
            p["rollouts"] = {str(a): [None] * self.futures for a in p["allowed"]}
        for k, a, j, future in self.rollouts:
            r = future.result()
            cpu += float(r.get("seconds") or 0.0)
            if not r["ok"]:
                failures.append(f"{self.points[k]['state']} 녹색 {a} 미래 {j}: {r['error']}")
                self.points[k].setdefault("rollout_errors", []).append({"green": a, "future": j, "error": r["error"][:300]})
            elif r["leaks"] or r["changed_by_rules"]:
                errors.append(f"{self.points[k]['state']} 녹색 {a} 미래 {j}: 미래 엿봄 또는 규칙 변경")
            else:
                self.points[k]["rollouts"][str(a)][j] = {key: r.get(key) for key in KEEP}
        allowed_failures = max(3, int(float(self.cfg["rollout"]["max_failed_rollout_fraction"]) * len(self.rollouts)))
        if len(failures) > allowed_failures:
            errors.insert(0, f"롤아웃 실패 {len(failures)}/{len(self.rollouts)}개(허용 {allowed_failures}), 첫 번째: {failures[0]}")
        checked = mismatched = 0
        for k, future in self.verifies:
            r = future.result()
            cpu += float(r.get("seconds") or 0.0)
            point = self.points[k]
            main = point.pop("main_after", None)
            ok = bool(r["ok"]) and main is not None and bp.same_table(r["table"], main, tol)
            point["restore_check"] = {"ok": ok, "error": r.get("error")}
            checked += 1
            mismatched += not ok
        for p in self.points:
            p.pop("main_after", None)
            p["valid_futures"] = len(valid_futures(p))
        self.cleanup()
        if errors:
            raise RuntimeError(f"수집 판 실패({len(errors)}건): {errors[0]}")
        return {"checked": checked, "mismatched": mismatched, "rollout_cpu_seconds": cpu,
                "rollout_failures": len(failures), "rollouts": len(self.rollouts)}

    def cleanup(self) -> None:
        for p in self.points:
            stem = self.state_dir / p["state"]
            for ext in (".xml", ".pkl"):
                src = stem.with_suffix(ext)
                if self.keep_dir is not None and keep_state(self.cfg, p["state"]) and src.exists():
                    # 상태 XML은 크므로 gzip으로 보관한다(쓰려면 풀어서 explore_rollout.restore에 넘긴다).
                    self.keep_dir.mkdir(parents=True, exist_ok=True)
                    with src.open("rb") as fin, gzip.open(self.keep_dir / (src.name + ".gz"), "wb") as fout:
                        shutil.copyfileobj(fin, fout)
                src.unlink(missing_ok=True)


def valid_futures(point: dict) -> list[int]:
    """모든 후보에서 비용이 있고, 어느 후보에서도 미확인 충돌·204820 충돌이 없는 미래 번호(B′와 같은 규칙)."""
    rows = list(point["rollouts"].values())
    out = []
    for j in range(len(rows[0])):
        samples = [r[j] for r in rows]
        if any(s is None or s.get("collisions_unexplained") or s.get("collisions_at_204820") for s in samples):
            continue
        out.append(j)
    return out


# ----------------------------------------------------------------------------- 수집 명령
def write_points(target: Path, points: list[dict]) -> str:
    target.mkdir(parents=True, exist_ok=True)
    path = target / "points.jsonl.gz"
    with gzip.open(path, "wt") as f:
        for p in points:
            f.write(json.dumps(w.plain(p), ensure_ascii=False, allow_nan=False) + "\n")
    return w.file_hash(path)


def read_points(target: Path) -> list[dict]:
    m = w.read(Path(target) / "metrics.json")
    path = Path(target) / "points.jsonl.gz"
    if w.file_hash(path) != m.get("points_sha256"):
        raise ValueError(f"결정 지점 자료 해시가 다릅니다: {target}")
    with gzip.open(path, "rt") as f:
        return [json.loads(line) for line in f]


def restore_totals(out: Path) -> tuple[int, int]:
    """모든 조각의 복원 확인 수를 합한다(조각마다 따로 세면 한 조각이 우연히 넘을 수 있다, 210절)."""
    checked = mismatched = 0
    for path in sorted((Path(out) / "status").glob("collect_shard*.json")):
        s = w.read(path)
        checked += int(s.get("restore_checked", 0))
        mismatched += int(s.get("restore_mismatched", 0))
    return checked, mismatched


def finalize(cfg: dict, out: Path, fp: str, payload: dict, item: dict) -> dict:
    """끝난 수집 판 하나를 쓴다: points.jsonl.gz + metrics.json(+ decisions·cache, ARENA3 실행 형식)."""
    s, policy, result = item["situation"], item["policy"], item["result"]
    rc = policy.finish()
    target = out / "runs" / POLICY / item["sid"]
    sha = write_points(target, policy.points)
    result.metrics.update({"pi_round": cfg["round"], "pi_base_policy": payload["name"], "pi_points": len(policy.points),
                           "pi_counts": dict(policy.counts), "pi_restore_check": {k: rc[k] for k in ("checked", "mismatched")},
                           "pi_rollouts": rc["rollouts"], "pi_rollout_failures": rc["rollout_failures"],
                           "pi_rollout_cpu_seconds": rc["rollout_cpu_seconds"], "pi_main_seconds": item["main_seconds"]})
    ac.write_run(target, result, {"policy": POLICY, "fingerprint": fp, "model_identity": payload["sha256"],
                                  "points_sha256": sha, "role": "data"})
    return {**rc, "cpu_seconds": rc["rollout_cpu_seconds"] + item["main_seconds"], "points": len(policy.points)}


def collect(cfg: dict, out: Path, work: Path, shard: int, shards: int, base_root: Path, workers: int, *,
            executor=None, run_episode=None) -> int:
    run_episode = run_episode or ac.run_arena_episode
    out, work = Path(out), Path(work)
    fp = ref.register(out, extra_files=sources(cfg["round"]))
    r = cfg["rollout"]
    mine = [s for i, s in enumerate(cfg["situations_data"]) if i % shards == shard]
    payload = base_payload(cfg["base_policy"], base_root)
    base = base_from_payload(payload)
    ledger = ref.Ledger(out, f"collect_shard{shard}")
    status_path = out / "status" / f"collect_shard{shard}.json"
    rc_cfg = r["restore_check"]
    checked = mismatched = 0
    failed, inflight = [], []
    stopped = False

    def write_status():
        done = sum((out / "runs" / POLICY / ac.situation_id(x) / "cache.json").exists() for x in mine)
        w.write(status_path, {"done": done, "total": len(mine), "failed": failed, "restore_checked": checked,
                              "restore_mismatched": mismatched, "inflight": [i["sid"] for i in inflight],
                              "stopped_by_restore_rule": stopped, "updated_unix": time.time()})

    def harvest():
        nonlocal checked, mismatched
        for item in [i for i in inflight if i["policy"].outstanding() == 0]:
            inflight.remove(item)
            try:
                got = finalize(cfg, out, fp, payload, item)
                checked += got["checked"]
                mismatched += got["mismatched"]
                ledger.add(got["cpu_seconds"])
                c = item["policy"].counts
                ref.stamp(f"PI {cfg['round']} 자료 {item['sid']}: 지연 {item['result'].metrics['mean_delay_s']:.3f}초, "
                          f"지점 {got['points']}개(갈림 {c['choice']}·합의 {c['agree_sampled']}/{c['agree']}), 복원 불일치 "
                          f"{got['mismatched']}/{got['checked']}, {got['cpu_seconds'] / 3600:.2f} CPU-시간")
            except BaseException as error:
                failed.append({"situation": item["sid"], "error": f"{type(error).__name__}: {error}"})
                ref.stamp(f"PI 자료 실패 {item['sid']}: {failed[-1]['error']}")
                ledger.add(item["main_seconds"] + item["policy"].spent())
                if isinstance(error, KeyboardInterrupt):
                    raise
            finally:
                shutil.rmtree(work / item["sid"], ignore_errors=True)
            write_status()

    def backlog() -> int:
        return sum(i["policy"].outstanding() for i in inflight)

    def wait_some():
        pending = [f for i in inflight for f in i["policy"].futures_list() if not f.done()]
        if pending:
            wait(pending, timeout=30, return_when=FIRST_COMPLETED)

    pool = executor or rollout_pool(workers, payload)
    with pool:
        for s in mine:
            sid = ac.situation_id(s)
            target = out / "runs" / POLICY / sid
            if ac.read_run(target, fp, payload["sha256"]) is not None:
                m = w.read(target / "metrics.json")
                read_points(target)
                checked += m["pi_restore_check"]["checked"]
                mismatched += m["pi_restore_check"]["mismatched"]
                continue
            while backlog() > int(r["max_backlog_jobs"]):
                wait_some()
                harvest()
            began = time.monotonic()
            shutil.rmtree(work / sid, ignore_errors=True)
            policy = Collector(base, pool, cfg, s, work / sid / "states", out / "kept_states")
            try:
                result = run_episode(s, work / sid / "episode", policy, max_seconds=float(r["main_episode_max_seconds"]),
                                     on_step=policy.after_step)
                if result.metrics["route_sha256"] != s["route_sha256"]:
                    raise ValueError("차량 파일 SHA가 설정과 다릅니다.")
                dirty = ref.quality_errors(result.metrics)            # 심판과 같은 품질 검사(충돌은 자료를 읽을 때 뺀다)
                if dirty:
                    raise RuntimeError(f"수집 본 에피소드 품질 오류: {dirty}")
                inflight.append({"situation": s, "sid": sid, "policy": policy, "result": result,
                                 "main_seconds": time.monotonic() - began})
            except BaseException as error:
                failed.append({"situation": sid, "error": f"{type(error).__name__}: {error}"})
                ref.stamp(f"PI 본 에피소드 실패 {sid}: {failed[-1]['error']}")
                for future in policy.futures_list():
                    future.cancel()
                wait([f for f in policy.futures_list() if not f.cancelled()])
                policy.cleanup()
                shutil.rmtree(work / sid, ignore_errors=True)
                ledger.add(time.monotonic() - began + policy.spent())
                if isinstance(error, KeyboardInterrupt):
                    raise
            harvest()
            write_status()
            total_checked, total_mismatched = restore_totals(out)          # 이 조각의 방금 쓴 상태 포함
            rate = total_mismatched / total_checked if total_checked else 0.0
            if total_checked >= int(rc_cfg["min_points"]) and rate > float(rc_cfg["max_mismatch_rate"]):
                stopped = True
                ref.stamp(f"모든 조각 합 복원 불일치 {rate:.1%}({total_mismatched}/{total_checked})가 "
                          f"{float(rc_cfg['max_mismatch_rate']):.0%}를 넘어 새 판을 시작하지 않습니다.")
                break
        while inflight:
            wait_some()
            harvest()
    write_status()
    if stopped:
        return 3
    return 1 if failed else 0


def smoke(cfg: dict, out: Path, work: Path, base_root: Path, workers: int, points: int = 2) -> int:
    """실행 전 점검: (1) 상태 저장·롤아웃을 해도 본 에피소드가 지금 정책의 평가 실행과 같은가, (2) 복원 확인, (3) 시간 어림."""
    out, work = Path(out), Path(work)
    fp = ref.register(out, extra_files=sources(cfg["round"]))
    s = cfg["situations_data"][0]
    payload = base_payload(cfg["base_policy"], base_root)
    base = base_from_payload(payload)
    shutil.rmtree(work, ignore_errors=True)
    plain = ac.run_arena_episode(s, work / "plain", referee_policy(base, payload["name"]))
    limit = {"choice": points, "agree": points}
    with rollout_pool(workers, payload) as pool:
        policy = Collector(base, pool, cfg, s, work / "same" / "states", None, limit=limit)
        same = ac.run_arena_episode(s, work / "same" / "episode", policy, max_seconds=float(cfg["rollout"]["main_episode_max_seconds"]),
                                    on_step=policy.after_step)
        wait(policy.futures_list())
        rc = policy.finish()
    keys = ("mean_delay_s", "actual_total_delay", "executed_sha16", "route_sha256", "choice_points")
    unchanged = all(plain.metrics[k] == same.metrics[k] for k in keys) and same.metrics["p0_mismatch"] == 0
    secs = [x["seconds"] for p in policy.points for v in p["rollouts"].values() for x in v if x]
    cands = {kind: [len(p["allowed"]) for p in policy.points if p["kind"] == kind] for kind in ("choice", "agree")}
    c = policy.counts
    per_episode = None
    if secs and cands["choice"] and cands["agree"]:
        rollouts = (c["choice"] * statistics.fmean(cands["choice"]) + c["agree_sampled"] * statistics.fmean(cands["agree"])) \
            * policy.futures
        per_episode = (rollouts * statistics.fmean(secs) + same.metrics["seconds"]) / 3600
    ok = (unchanged and rc["mismatched"] == 0 and rc["rollout_failures"] == 0 and not ref.quality_errors(same.metrics)
          and len(cands["choice"]) == min(points, c["choice"]) and len(cands["agree"]) == min(points, c["agree_sampled"])
          and per_episode is not None)
    report = {"passed": ok, "situation": ac.situation_id(s), "base_policy": payload["name"], "base_sha256": payload["sha256"],
              "unchanged_by_saving": unchanged, "plain": {k: plain.metrics[k] for k in keys},
              "collector": {k: same.metrics[k] for k in keys}, "counts_per_hour": c,
              "rolled_points": {kind: len(v) for kind, v in cands.items()}, "candidates": cands,
              "rollout_seconds_mean": statistics.fmean(secs) if secs else None, "restore_check": rc,
              "estimate_cpu_hours_per_episode": per_episode,
              "estimate_cpu_hours_all": per_episode * len(cfg["situations_data"]) if per_episode else None,
              "fingerprint": fp}
    w.write(out / "smoke.json", report)
    ref.stamp(f"저장·롤아웃 무변경: {'같음' if unchanged else '다름'} | 시간당 갈림 {c['choice']}·합의 {c['agree']}·강제 {c['forced']} | "
              f"롤아웃 평균 {report['rollout_seconds_mean']}초 | 판당 약 {per_episode} CPU-시간, 64판 약 "
              f"{report['estimate_cpu_hours_all']} CPU-시간 | 복원 확인 {rc['mismatched']}/{rc['checked']} 불일치 | "
              f"{'통과' if ok else '실패'}")
    shutil.rmtree(work, ignore_errors=True)
    return 0 if ok else 2


# ----------------------------------------------------------------------------- 설정 만들기(맥, SUMO 없음)
def build_config(round_name: str, work: Path) -> dict:
    if round_name != "R1":
        raise ValueError("이 함수는 1라운드 설정만 만든다. 다음 라운드는 1라운드 판정을 본 뒤 정한다.")
    lock = ac.validate_lock()
    w.configure()
    from focused_data import COUNTS, load_counts
    spec = {"per_date": 8, "slot_salt": "ARENA3_PI_R1_SLOTS", "tag_prefix": "R1T"}
    cfg = {
        "experiment": "ARENA3_PI_R1", "round": "R1",
        "base_lock_sha256": w.file_hash(ac.CONFIG), "stage2_config_sha256": w.file_hash(STAGE2_CONFIG),
        "design": "PROJECT_PROGRESS.md 235·236절, docs/ARENA3_PI_실행안내.md",
        "method": ("롤아웃 분류 정책 반복(Lagoudakis·Parr 2003 RCPI, Lazaric 외 DPI 계열 강화학습). 지금 정책이 운전하는 학습 상황에서 "
                   "결정 지점마다 허용 녹색을 같은 표본 미래로 굴려 비교하고(정책 평가), 관측만 보는 신경망이 더 나은 녹색을 배운다"
                   "(정책 개선). 다음 라운드는 그 신경망이 지금 정책이다."),
        "base_policy": {"name": "S2A_pre_42", "kind": "mode", "variant": None, "stage2_variant": "advantage",
                        "stage": "pretrained", "seed": 42, "root": "arena3/stage2/S2A_train", "folder": "seed_42/pretrained",
                        "model_sha256": "1150ce509c9f9baed0318733f524169b08920a8a654f6aa69bfac259b0f256ad",
                        "why": ("개발 판정을 통과한 학습 모델 중 날짜 평균이 가장 큰 S2A_pre(+0.599초)의 첫 난수. 난수는 성적이 아닌 순서 규칙으로 "
                                "골랐다(난수별 +0.538·+0.680·+0.577초).")},
        "situations": {**spec, "count": 64, "note": "학습 8일 × 날짜마다 15분 칸 8개(해시 순서), 새 도착 배치(B′ 자료 T1~T64와 다름)"},
        "decision_points": {
            "choice": "여섯 모드의 첫 녹색이 갈리는 곳(시간당 약 29곳, M4·S2A_pre 84판 실측): 모두 굴린다",
            "agree": "허용 녹색이 2개 이상이고 여섯 모드가 같은 곳(시간당 약 59곳): 해시 규칙으로 일부만 굴린다",
            "forced": "허용 녹색이 하나인 곳(시간당 약 632곳): 고를 것이 없다",
            "candidates": "그 순간 허용된 녹색 전부(갈림 평균 3.9개, 합의 평균 3.4개)"},
        "agree_sample": {"fraction": 0.25, "salt": "ARENA3_PI_R1_AGREE"},
        "rollout": {"horizon_s": 300, "first_action_s": 5, "terminal_s": 75.0, "history_s": 900, "futures": 8,
                    "future_salt": "PI_R1_FUTURE", "min_valid_futures": 4,
                    "continuation": "첫 5초 후보 녹색 → 지금 정책(평가 때와 같은 선택) → 제어 60분이 끝나면 P0(본 에피소드와 같음)",
                    "cost": "300초 지연(시간 손실 + 진입 대기) + 남은 정체 × 75초(B′와 같음)",
                    "drop_future": "어느 후보에서든 미확인 충돌·204820 충돌이 난 미래는 모든 후보에서 뺀다",
                    "restore_check": {"tolerance": 1e-6, "max_mismatch_rate": 0.08, "min_points": 200,
                                      "scope": "모든 조각을 합해서(조각마다 세면 우연히 넘는다, 210절)",
                                      "use": "불일치 지점은 학습에서 뺀다"},
                    "max_failed_rollout_fraction": 0.02,
                    "failed_rollout": "실패한 롤아웃이 판의 2%(최소 3개) 이하이면 그 칸을 비우고(그 미래를 모든 후보에서 뺀다) 오류를 적는다. 넘으면 판 실패",
                    "keep_state_fraction": 0.05, "max_backlog_jobs": 1500, "main_episode_max_seconds": 10800},
        "data_quality": ("본 에피소드에 미확인 충돌·204820 충돌이 있으면 그 판의 지점은 모두 학습에서 뺀다(기록). "
                         "규칙 변경·행동 오류·P0 불일치·지연 재구성 실패는 멈춘다."),
        "student": {"seeds": [42, 43, 44], "network": [64, 64], "outputs": "녹색 4개의 M4 녹색 대비 값(M4 = 0)",
                    "target": "mean", "target_note": "짝 평균 이득(차량·초 ÷ 1000), 교사 규칙 c=0(225절: c=1보다 대리 비용 +22%, 8/8일)",
                    "loss": "S2A와 같은 신뢰 가중 Huber, 가중 = clamp(|이득| / (표준오차 + 0.1), 0.25, 4), 허용 녹색 중 M4가 아닌 것",
                    "confidence_floor": 0.1, "confidence_clip": [0.25, 4.0],
                    "epochs": 200, "batch_size": 64, "learning_rate": 0.001, "weight_decay": 0.0001, "grad_clip": 10.0,
                    "init": "이점 머리 0(학습 전 탐욕 = M4)", "greedy": "가장 큰 값, 같으면 M4 녹색"},
        "variants": {"M": {"name": "R1M", "means": "갈림 지점에서 모드가 내는 녹색만(지금 행동 범위 안의 정책 반복)"},
                     "G": {"name": "R1G", "means": "갈림 지점에서 허용 녹색 전부(새 행동), 합의 지점은 M4"},
                     "A": {"name": "R1A", "means": "갈림·합의 지점 모두에서 허용 녹색 전부"}},
        "offline": {"status": "참고 진단(판정·선택에 쓰지 않음)", "folds": "날짜 하나씩 보류, 미래 앞 절반으로 학습·뒤 절반으로 채점",
                    "targets": ["mean", "risk"], "risk": "이득 − 표준오차(S2A 방식), 진단용"},
        "evaluation": {"part": "eval", "situations": 84, "models": 3, "judge": "config/arena3_v1.json:judge (그대로)",
                       "baseline": "arena3/referee/baseline", "confirmation_baseline": "arena3/stage2/confirm/baseline"},
        "rules": {
            "main": "M4 대비 ARENA3 판정 그대로(날짜 평균 ≥ 0.3초, t95 하한 > 0, 양수 날짜 ≥ 5, 좌회전 +5초 이내, 300초 초과 ≤ M4, 정책 고유 충돌 0, 품질 오류 0).",
            "paired": "같은 84상황에서 날짜별 M4 대비 이득(차량 가중)의 차, 날짜 7개 평균과 t95(자유도 6).",
            "pi_gain": "R1M − S2A_pre의 t95 하한 > 0 이면 '지금 행동 범위에서 정책 반복 1라운드가 출발 정책보다 짧다'.",
            "new_action": "R1G − R1M의 t95 하한 > 0 이면 '갈림 지점에서 모드 밖 녹색까지 고르면 더 짧다', 상한 < 0 이면 '더 길다', 아니면 '구분 못 함'.",
            "agree_action": "R1A − R1G에 같은 읽기(합의 지점에서도 고르는 몫).",
            "winner": "R1M·R1G·R1A 가운데 main을 통과한 것 중 날짜 평균이 큰 하나만 C1·C2 84상황에서 한 번 확인(기준선은 2단계 확인 기준선 재사용). 통과한 것이 없으면 확인하지 않는다.",
            "next_round": "다음 라운드의 지금 정책은 확인까지 통과한 승자의 난수 42 모델(같은 변형). 통과가 없으면 다음 라운드 설계는 결과를 본 뒤 정한다.",
            "paired_comparisons": [["R1M", "S2A_pre"], ["R1G", "S2A_pre"], ["R1A", "S2A_pre"], ["R1G", "R1M"], ["R1A", "R1G"],
                                   ["R1M", "S2B"], ["R1G", "S2B"], ["R1A", "S2B"],
                                   ["R1M", "BPRIME"], ["R1G", "BPRIME"], ["R1A", "BPRIME"]],
            "confirmation_paired": [["winner", "S2B"]]},
        "reporting": ("반복 사용 날짜의 SUMO 개발 비교. C1·C2는 같은 7일의 새 도착 배치이며 새 날짜·실제 도로 일반화는 미검증. "
                      "최종 제어기는 신경망만(실행 때 탐색 없음). 롤아웃으로 직접 고르는 제어는 '탐색 제어'로 따로 부른다."),
    }
    sits = data_situations(lock, spec)
    counts = load_counts(COUNTS)
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    for s in sits:
        if (s["date"], s["period"]) not in counts:
            raise ValueError(f"수요 자료에 없는 상황: {s}")
        s["route_sha256"] = ac.route_file_sha(s, work)
    old = {x["route_sha256"] for part in ("eval", "confirmation", "bprime_data") for x in lock[f"situations_{part}"]}
    shas = [s["route_sha256"] for s in sits]
    if len(set(shas)) != len(shas) or set(shas) & old:
        raise ValueError("정책 반복 자료의 차량 파일이 서로 같거나 기존 상황과 같습니다.")
    cfg["situations_data"] = sits
    return cfg


# ----------------------------------------------------------------------------- 명령
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--round", default="R1", choices=ROUNDS)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("lock")
    p.add_argument("--work", type=Path, required=True)
    for name in ("smoke", "collect"):
        p = sub.add_parser(name)
        p.add_argument("--out", type=Path, required=True)
        p.add_argument("--work", type=Path, required=True)
        p.add_argument("--base-root", type=Path, required=True, help="지금 정책 모델의 뿌리 폴더(1라운드: arena3/stage2/S2A_train)")
        p.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1))
        if name == "collect":
            p.add_argument("--shard", type=int, default=0)
            p.add_argument("--shards", type=int, default=1)
    import arena3_pi_learn as pl
    pl.add_commands(sub)
    args = parser.parse_args()
    w.configure()
    torch.set_num_threads(1)
    if args.cmd == "lock":
        built = w.plain(build_config(args.round, args.work))
        path = config_path(args.round)
        if path.exists():
            if w.read(path) != json.loads(json.dumps(built, ensure_ascii=False)):
                raise SystemExit("이미 만든 정책 반복 설정과 다릅니다. 설정은 바꾸지 않습니다.")
            print(f"설정이 이미 있고 같습니다: {path} sha256={w.file_hash(path)}")
        else:
            w.write(path, built)
            print(f"설정을 만들었습니다: {path} sha256={w.file_hash(path)}")
        return 0
    if args.cmd in ("smoke", "collect") and platform.system() == "Darwin":
        parser.error("맥에서는 SUMO를 실행하지 않습니다. 이 명령은 Colab 노트북에서 실행하세요.")
    cfg = config(args.round)
    if args.cmd == "smoke":
        return smoke(cfg, args.out, args.work, args.base_root, args.workers)
    if args.cmd == "collect":
        if not 0 <= args.shard < args.shards:
            raise SystemExit("--shard는 0 이상 --shards 미만이어야 합니다.")
        return collect(cfg, args.out, args.work, args.shard, args.shards, args.base_root, args.workers)
    return pl.run_command(args, cfg)


if __name__ == "__main__":
    raise SystemExit(main())
