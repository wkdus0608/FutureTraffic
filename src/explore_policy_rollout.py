"""PI1의 고정 부모 혼합 정책과 후보 후속 운영(진행 기록 156절).

기존 탐색 코드는 변경하지 않는다. 첫 후보는 er.drive가 실행하고, 이후 매
선택에서 P0를 먼저 계산한 뒤 개입 조건 안에서만 고정 부모가 대신 고른다.
표본 미래에서는 관측 이력에도 실제 미래 차량을 남기지 않는다.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace

import numpy as np

import explore_combo as cb
import explore_online as eo
import explore_rollout as er


def code_fingerprint() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _end_control(ep) -> float:
    import focused_train as ft
    return float(ep.handover) + ft.CONTROL_S


def read_observation(ep, width: int = 96) -> list[float]:
    """기존 96칸 또는 같은 마스크 앞에 예정 수요 29칸을 더한 125칸."""
    if width not in (96, 125):
        raise ValueError(f"PI1 관측 폭은 96 또는 125여야 한다: {width}")
    _, obs72 = eo.observation72(ep)
    obs96 = cb.observation96(obs72, cb.arrivals24(ep))
    if width == 96:
        return obs96
    from explore_history import history_features, observation_with_history
    vehicles = getattr(ep, "history_vehicles", ep.vehicles)
    return [float(x) for x in observation_with_history(obs96, history_features(vehicles, float(ep.env.sim_step)))]


def _validate_allowed(allowed, p0_action: int) -> list[int]:
    values = sorted(set(int(c) for c in allowed))
    if not values or any(c not in range(4) for c in values) or int(p0_action) not in values:
        raise ValueError(f"허용 후보와 P0 행동이 일치하지 않는다: {values}, P0={p0_action}")
    return values


class FrozenPolicy:
    """가중치를 한 번 읽어 검증하고, 추론만 하는 비용 회귀 앙상블.

    모델마다 상대 비용을 출력한다. 평균 비용의 최소 허용 후보를 고르며
    1e-9 이내 동률은 P0, 그다음 작은 번호로 해결한다(기존 B와 동일).
    """

    def __init__(self, parent_dir: Path):
        import torch as th
        th.set_num_threads(1)
        self.directory = Path(parent_dir)
        source = self.directory / "policy.json"
        policy = json.loads(source.read_text())
        self.width = int(policy["width"])
        if self.width not in (96, 125) or policy.get("method", "l0") != "l0":
            raise ValueError("PI1 부모는 96/125칸 비용 회귀 정책이어야 한다")
        seeds = list(policy["model_seeds"])
        if seeds != [0, 1, 2, 3, 4]:
            raise ValueError("PI1 부모는 seed 0~4 다섯 모델이어야 한다")
        self.models, verified = [], {}
        for seed in seeds:
            name = f"seed{seed}"
            file = self.directory / f"{name}.pt"
            sha = _sha(file)
            declared = str(policy["weights"][name])
            if not (12 <= len(declared) <= 64 and sha.startswith(declared)):
                raise ValueError(f"부모 가중치 SHA가 저장 기록과 다르다: {name}")
            state = th.load(file, map_location="cpu", weights_only=True)
            model = cb.make_net(self.width)
            model.load_state_dict(state, strict=True)
            model.eval()
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            self.models.append(model)
            verified[name] = sha
        self.manifest = {"width": self.width, "method": "l0", "model_seeds": seeds,
                         "weights": verified, "policy_sha256": _sha(source),
                         "intervention": "left_point_and_multiple_allowed_before_original_control_end",
                         "tie_break": "p0_then_smallest", "threshold": 0.0}
        self.hash = hashlib.sha256(json.dumps(self.manifest, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        self.manifest["parent_hash"] = self.hash

    def predict(self, obs) -> np.ndarray:
        import torch as th
        x = np.asarray(obs, dtype=np.float32)
        if x.shape != (self.width,) or not np.isfinite(x).all():
            raise ValueError(f"부모 입력은 유한한 {self.width}칸 벡터여야 한다: {x.shape}")
        with th.inference_mode():
            tensor = th.as_tensor(x).unsqueeze(0)
            pred = np.mean([model(tensor).numpy()[0] for model in self.models], axis=0)
        if not np.isfinite(pred).all():
            raise ValueError("부모가 유한하지 않은 비용을 출력했다")
        return pred

    def choose(self, obs, allowed, p0_action: int) -> int:
        allowed = _validate_allowed(allowed, p0_action)
        pred = self.predict(obs)
        low = min(float(pred[c]) for c in allowed)
        if float(pred[p0_action]) <= low + 1e-9:
            return int(p0_action)
        return min(c for c in allowed if float(pred[c]) <= low + 1e-9)

    def read_observation(self, ep) -> list[float]:
        return read_observation(ep, self.width)

    def action(self, ep, allowed, p0_action: int, end_control: float | None = None) -> int:
        """P0는 호출자가 먼저 계산한다. 강제 구간·꼬리에서는 P0 그대로."""
        allowed = _validate_allowed(allowed, p0_action)
        end = _end_control(ep) if end_control is None else float(end_control)
        if len(allowed) == 1 or float(ep.env.sim_step) >= end:
            return int(p0_action)
        feat = er._features(ep, allowed, p0_action)
        if not eo.is_left_point(feat):
            return int(p0_action)
        return self.choose(self.read_observation(ep), allowed, p0_action)

    def sample_current(self, ep, allowed, p0_action: int) -> dict:
        return sample_current(ep, allowed, p0_action, self)


def sample_current(ep, allowed, p0_action: int, policy: FrozenPolicy) -> dict:
    """수집·재생에서 같은 순서로 읽는, JSON에 그대로 쓸 수 있는 현재 장면."""
    allowed = _validate_allowed(allowed, p0_action)
    feat = er._features(ep, allowed, p0_action)
    obs96 = read_observation(ep, 96)
    if policy.width == 125:
        from explore_history import history_features, observation_with_history
        vehicles = getattr(ep, "history_vehicles", ep.vehicles)
        obs = [float(x) for x in observation_with_history(obs96, history_features(vehicles, float(ep.env.sim_step)))]
    else:
        obs = obs96
    left = eo.is_left_point(feat)
    active = len(allowed) > 1 and float(ep.env.sim_step) < _end_control(ep) and left
    action = policy.choose(obs, allowed, p0_action) if active else int(p0_action)
    leftfeatures = {k: feat[k] for k in ("W", "green", *er.LEFTS, *[f"{n}_served" for n in er.LEFTS])}
    return {**feat, "t0": float(ep.env.sim_step), "obs96": obs96, "obs": obs, "width": policy.width,
            "allowed": allowed, "p0": int(p0_action), "p0_action": int(p0_action),
            "parent": int(action), "parent_action": int(action), "parent_hash": policy.hash,
            "leftfeatures": leftfeatures, "left_point": bool(left), "in_control": float(ep.env.sim_step) < _end_control(ep)}


def verify_snapshot(stem: Path, route_dir: Path, parent_dir: Path, row: dict) -> dict:
    """실제 미래를 보존한 복원 직후 현재 관측·정책이 수집 기록과 같은지 확인.

    시간을 진행하지 않으므로 미래 차량의 실제 도착은 보지 않는다. 예정
    이력 또한 현재 시각 이전만 읽는다. 불일치는 ok=False와 항목에 남긴다.
    """
    parent = FrozenPolicy(Path(parent_dir))
    env, cons, ts, ctl, snap, _, vehicles = er.restore(Path(stem), Path(route_dir), "real", horizon=1)
    try:
        ep = SimpleNamespace(env=env, cons=cons, ts=ts, ctl=ctl, date=snap["date"], period=snap["period"],
                             handover=snap["handover"], vehicles=vehicles, movement_of=ctl.movement_of)
        allowed = cons.allowed(env)
        p0 = int(ctl(env, allowed))
        actual = parent.sample_current(ep, allowed, p0)
        problems = []
        for key in ("t0", "allowed", "p0", "parent", "width", "left_point", "in_control"):
            if key in row and actual[key] != row[key]:
                problems.append(key)
        for key in ("obs", "obs96"):
            if key in row and not np.array_equal(np.asarray(actual[key]), np.asarray(row[key])):
                problems.append(key)
        if "parent_hash" in row and row["parent_hash"] != parent.hash:
            problems.append("parent_hash")
        if actual["allowed"] != sorted(snap["allowed"]) or actual["p0"] != int(snap["p0_action"]):
            problems.append("snapshot_allowed_or_p0")
        return {"ok": not problems, "problems": problems, "state": Path(stem).name,
                "t0": actual["t0"], "allowed": actual["allowed"], "p0": actual["p0"],
                "parent": actual["parent"], "width": parent.width, "parent_hash": parent.hash}
    finally:
        env.close()


def measured(stem: Path, route_dir: Path, candidate: int, seed: int, continuation: str,
             parent_dir: Path, horizon: int = 600) -> dict:
    """첫 후보 뒤 P0 또는 고정 부모로 운용하고 기존 방식의 지연을 잰다.

    cpu_s는 이 Python 프로세스와 종료된 SUMO 자식의 CPU 시간을 합한다.
    seconds는 벽시계 시간이며 병렬 작업의 CPU 예산과 혼동하지 않는다.
    """
    if continuation not in ("p0", "parent"):
        raise ValueError(f"후속 정책은 p0 또는 parent여야 한다: {continuation}")
    if horizon < 1:
        raise ValueError("탐색 길이는 양수여야 한다")
    started, cpu_start = time.perf_counter(), os.times()
    parent = FrozenPolicy(Path(parent_dir))
    fd, filename = tempfile.mkstemp(prefix="pi1_trip_", suffix=".xml")
    os.close(fd)
    trip = Path(filename)
    env = None
    follow_actions, follow_p0 = [], []
    try:
        env, cons, ts, ctl, snap, injected, vehicles = er.restore(
            Path(stem), Path(route_dir), "sampled", seed, horizon, tripinfo=trip)
        from explore_history import history_vehicles
        safe_history = history_vehicles(vehicles, snap["t0"], injected)
        ep = SimpleNamespace(env=env, cons=cons, ts=ts, ctl=ctl, date=snap["date"], period=snap["period"],
                             handover=snap["handover"], vehicles=safe_history, history_vehicles=safe_history,
                             movement_of=ctl.movement_of)
        allowed_initial = _validate_allowed(cons.allowed(env), snap["p0_action"])
        if int(candidate) not in allowed_initial:
            raise ValueError(f"첫 후보 {candidate}가 저장 상태의 허용 후보 {allowed_initial}에 없다")
        setup = time.perf_counter() - started
        meter = er.DelayMeter(env, snap["t0"], vehicles, injected,
                              marks=tuple(x for x in (300, 600) if x <= horizon) + ((horizon,) if horizon not in (300, 600) else ()))

        def continuation_ctl(current_env, allowed):
            # P0의 관측/내부 상태 갱신 순서를 기존 운영과 같게 유지한다.
            p0_action = int(ctl(current_env, allowed))
            action = parent.action(ep, allowed, p0_action) if continuation == "parent" else p0_action
            follow_p0.append(p0_action)
            follow_actions.append(action)
            return action

        actions, changed = er.drive(env, cons, continuation_ctl, int(candidate), snap["t0"], horizon, meter.step)
        env.close()
        env = None
        meter.finish(trip)
        cpu_end = os.times()
        cpu_s = ((cpu_end.user + cpu_end.system + cpu_end.children_user + cpu_end.children_system)
                 - (cpu_start.user + cpu_start.system + cpu_start.children_user + cpu_start.children_system))
        return {"state": Path(stem).name, "candidate": int(candidate), "seed": int(seed), "continuation": continuation,
                "marks": {str(k): v for k, v in sorted(meter.at.items())}, "injected": len(injected),
                "leaks": len(meter.leaks), "changed_by_rules": int(changed), "code": code_fingerprint(),
                "rollout_code": er.code_fingerprint(), "parent_hash": parent.hash, "parent_manifest": parent.manifest,
                "setup_s": round(setup, 6), "seconds": round(time.perf_counter() - started, 6),
                "cpu_s": round(cpu_s, 6), "actions": [int(a) for a in actions],
                "follow_actions": follow_actions, "follow_p0": follow_p0,
                "end_control": _end_control(ep), "history_actual_future_removed": sum(dep > snap["t0"] for dep, _, _ in vehicles.values())}
    finally:
        if env is not None:
            env.close()
        trip.unlink(missing_ok=True)
