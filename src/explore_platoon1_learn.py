"""PLATOON1 — 자료 검증, 날짜 보류 진단, 학생 학습, 무리 도착 세계 심판 평가·판정·확인(설정 config/explore_platoon1_v1.json).

학생 입력은 관측 167칸뿐이다(ARENA3 119칸 + 시간표 관측 48칸). 날짜·실제 미래·롤아웃 비용은 묶음 구분과 감독 라벨에만 쓴다.
오프라인 점수는 300초 대리 비용(차량·초/지점, TM4 녹색 대비, 뒤는 TM4가 운전)이며 초/대나 한 시간 성능으로 바꾸지 않는다.
평가 84상황의 기준 실행(P 세계 M4·TM4·OM4·P1M4)은 PLATOON0 결과를 다시 돌리지 않고 읽는다(폴더 지문·평가 지문·차량 파일 대조).
판정은 ARENA3 심판 함수(arena3_referee.judge)를 그대로 쓰고, 실제 SUMO 평가는 Colab에서만 한다(맥은 judge만).
"""
from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import platform
import shutil
import statistics
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import arena3_common as ac  # noqa: E402
import arena3_referee as ref  # noqa: E402
import arena3_stage2_eval as s2e  # noqa: E402
import explore_platoon1 as pt  # noqa: E402
import explore_weight as w  # noqa: E402

KIND = "platoon1-green-v1"
SCALE = 1000.0
STUDENT = "PT1"
LEARN_SOURCES = ("src/explore_platoon1_learn.py",)
DIRTY = ("missing", "unfinished", "changed_by_rules", "mode_action_errors", "p0_mismatch")
T95 = {2: 12.706204736, 3: 4.302652730, 4: 3.182446305, 5: 2.776445105, 6: 2.570581836, 7: 2.446911851, 8: 2.364624252}
PARTS = {"eval": "situations_eval", "confirmation": "situations_confirmation"}


def eval_sources() -> tuple[str, ...]:
    return (*pt.SOURCES, *LEARN_SOURCES)


def situations(cfg: dict, part: str) -> list[dict]:
    if part not in PARTS:
        raise ValueError(f"알 수 없는 평가 배치: {part}")
    return cfg[PARTS[part]]


def mode_greens(obs) -> np.ndarray:
    """119칸 안의 여섯 모드 첫 녹색(M0~M4는 84:104, W4는 110:114; arena3_pi.mode_greens와 같은 읽기)."""
    obs = np.asarray(obs, np.float32)
    rows = np.concatenate((obs[..., 84:104].reshape(*obs.shape[:-1], 5, 4), obs[..., 110:114].reshape(*obs.shape[:-1], 1, 4)),
                          axis=-2)
    return rows.argmax(axis=-1)


# ----------------------------------------------------------------------------- 자료
def validate_folder(folder: Path) -> dict:
    """자료 폴더의 지문 기록이 온전하고, 그 자료를 만든 코드·환경·잠금이 지금과 같은지."""
    e = s2e.folder_record(folder)
    if (e["evaluation"].get("kind") != ac.EXPERIMENT or
            e["evaluation"].get("environment_fingerprint") != ac.config()["environment"]["fingerprint"]):
        raise ValueError("자료의 실험·환경 지문이 잠금과 다릅니다.")
    if set(e["extra_files"]) != set(pt.SOURCES):
        raise ValueError("자료의 수집 소스 목록이 PLATOON1과 다릅니다.")
    for rel, expected in {**e["evaluation"]["files"], **e["extra_files"]}.items():
        p = ROOT / rel
        if not p.is_file() or w.file_hash(p) != expected:
            raise ValueError(f"자료를 만든 코드·환경 파일이 달라졌습니다: {rel}")
    return e


def _gain_rows(costs: dict[int, list], indices: list[int], allowed: list[int], anchor: int):
    gain = np.zeros(pt.GREENS, np.float32)
    se = np.zeros(pt.GREENS, np.float32)
    for g in allowed:
        diffs = [(float(costs[anchor][j]["cost"]) - float(costs[g][j]["cost"])) / SCALE for j in indices]
        gain[g] = statistics.fmean(diffs)
        se[g] = statistics.stdev(diffs) / math.sqrt(len(diffs)) if len(diffs) > 1 else 0.0
    return gain, se


def point_to_row(point: dict, date: str, situation: str, cfg: dict) -> tuple[dict | None, str | None]:
    """순수 계산. 복원 불일치·유효 미래 부족은 명시적으로 빼고, 관측·후보·비용 형식이 틀리면 멈춘다."""
    if not point.get("restore_check", {}).get("ok"):
        return None, "restore_mismatch"
    obs = np.asarray(point["observation"], np.float32)
    if obs.shape != (pt.OBS,) or not np.isfinite(obs).all():
        raise ValueError(f"167칸 유한 관측이 아닙니다: {point.get('state')}")
    allowed = sorted(int(a) for a in point["allowed"])
    if np.flatnonzero(pt.allowed_mask(obs)).tolist() != allowed or len(allowed) < 2:
        raise ValueError(f"관측의 허용 녹색이 기록과 다르거나 고를 것이 없습니다: {point.get('state')}")
    if mode_greens(obs).tolist() != point["greens"] or int(point["m4"]) != int(point["greens"][ac.M4]):
        raise ValueError("관측의 모드 녹색·M4 녹색이 기록과 다릅니다.")
    anchor = int(pt.anchor_green(obs))
    if anchor != int(point["tm4"]) or anchor != int(point["executed"]) or anchor not in allowed:
        raise ValueError("관측의 TM4 녹색이 기록·실행과 다르거나 허용 녹색이 아닙니다.")
    costs = {int(k): v for k, v in point["rollouts"].items()}
    if sorted(costs) != allowed:
        raise ValueError("롤아웃 후보가 허용 녹색과 다릅니다.")
    lengths = {len(v) for v in costs.values()}
    if lengths != {int(cfg["rollout"]["futures"])}:
        raise ValueError("후보마다 미래 수가 다르거나 설정과 다릅니다.")
    valid = []
    for j in range(next(iter(lengths))):
        samples = [costs[a][j] for a in allowed]
        if any(s is None or s.get("collisions_unexplained", 0) or s.get("collisions_at_204820", 0) for s in samples):
            continue
        if any(not math.isfinite(float(s["cost"])) for s in samples):
            raise ValueError("롤아웃 비용이 유한하지 않습니다.")
        if len({s["future_seed"] for s in samples}) != 1:
            raise ValueError("후보끼리 같은 미래를 쓰지 않았습니다.")
        valid.append(j)
    if len(valid) < int(cfg["rollout"]["min_valid_futures"]):
        return None, "insufficient_valid_futures"
    cut = len(valid) // 2
    gain, se = _gain_rows(costs, valid, allowed, anchor)
    fit_gain, fit_se = _gain_rows(costs, valid[:cut], allowed, anchor)
    score_gain, _ = _gain_rows(costs, valid[cut:], allowed, anchor)
    return {"observations": obs, "allowed": pt.allowed_mask(obs), "anchor": anchor, "m4": int(point["m4"]),
            "gains": gain, "gain_se": se, "fit_gains": fit_gain, "fit_gain_se": fit_se, "score_gains": score_gain,
            "dates": str(date), "state_ids": str(point["state"]), "situations": situation, "valid_futures": len(valid),
            "fit_future_indices": valid[:cut], "score_future_indices": valid[cut:]}, None


ARRAY_KEYS = ("observations", "allowed", "anchor", "m4", "gains", "gain_se", "fit_gains", "fit_gain_se", "score_gains",
              "dates", "state_ids", "situations", "valid_futures")


def load_dataset(folder: Path, cfg: dict, *, check_provenance: bool = True) -> dict:
    """끝난 자료 64판을 모두 읽는다. 평가·확인 날짜는 읽지 않는다."""
    folder = Path(folder)
    e = validate_folder(folder) if check_provenance else w.read(folder / "experiment.json")
    lock = ac.config()
    expected = {ac.situation_id(s): s for s in cfg["situations_data"]}
    runs = folder / "runs" / pt.POLICY
    present = {p.name for p in runs.iterdir() if p.is_dir() and (p / "cache.json").exists()} if runs.is_dir() else set()
    if present != set(expected):
        raise ValueError(f"자료 상황이 설정과 다릅니다: 누락 {sorted(set(expected) - present)[:5]} 등 "
                         f"{len(set(expected) - present)}개, 추가 {sorted(present - set(expected))[:5]}")
    if set(lock["dates"]["train"]) & set(lock["dates"]["eval"]):
        raise ValueError("학습과 평가 날짜가 겹칩니다.")

    def read_one(sid):
        p = runs / sid
        metric_raw, point_raw = (p / "metrics.json").read_bytes(), (p / "points.jsonl.gz").read_bytes()
        m, cache = json.loads(metric_raw), json.loads((p / "cache.json").read_text())
        if hashlib.sha256(metric_raw).hexdigest() != cache["metrics_sha256"] \
                or hashlib.sha256(point_raw).hexdigest() != m["points_sha256"]:
            raise ValueError(f"자료 파일 해시 불일치: {sid}")
        if cache.get("fingerprint") != e["fingerprint"] or m.get("fingerprint") != e["fingerprint"]:
            raise ValueError(f"자료 실행 지문 불일치: {sid}")
        if cache.get("model_identity") != "TM4" or m.get("model_identity") != "TM4":
            raise ValueError(f"자료를 운전한 정책이 TM4가 아닙니다: {sid}")
        s = expected[sid]
        if (s["date"] not in lock["dates"]["train"] or m.get("date") != s["date"] or m.get("situation") != sid
                or m.get("route_sha256") != s["route_sha256"] or m.get("role") != "data" or m.get("policy") != pt.POLICY
                or m.get("pt_round") != pt.ROUND or m.get("world") != "P"):
            raise ValueError(f"학습 상황·차량 설정 불일치: {sid}")
        dirty = [k for k in DIRTY if m.get(k, 0)]
        if dirty or not m.get("cost_check", {}).get("passed"):
            raise ValueError(f"자료 본 에피소드 품질 오류 {sid}: {dirty}")
        points = [json.loads(x) for x in gzip.decompress(point_raw).splitlines() if x]
        if len(points) != m["pt_points"]:
            raise ValueError(f"지점 수 불일치: {sid}")
        collided = bool(m.get("collisions_at_204820", 0) or m.get("collisions_unexplained", 0))
        return sid, s, points, collided, {"situation": sid, "metrics_sha256": cache["metrics_sha256"],
                                          "points_sha256": m["points_sha256"], "points": len(points),
                                          "mean_delay_s": m["mean_delay_s"], "collided": collided,
                                          "east_future": m.get("pt_east_future")}

    rows, files, exclusions, seen = [], [], {}, set()
    with ThreadPoolExecutor(max_workers=8) as pool:
        for sid, s, points, collided, info in pool.map(read_one, sorted(expected)):
            files.append(info)
            for point in points:
                if point["state"] in seen:
                    raise ValueError(f"중복 상태: {point['state']}")
                seen.add(point["state"])
                if collided:                              # 설정 data_quality: 충돌이 난 판의 지점은 모두 뺀다
                    exclusions["episode_collision"] = exclusions.get("episode_collision", 0) + 1
                    continue
                row, reason = point_to_row(point, s["date"], sid, cfg)
                if row is None:
                    exclusions[reason] = exclusions.get(reason, 0) + 1
                else:
                    rows.append(row)
    if not rows:
        raise ValueError("쓸 수 있는 자료가 없습니다.")
    dataset = {k: np.asarray([r[k] for r in rows]) for k in ARRAY_KEYS}
    manifest = {"kind": "platoon1-data-v1", "round": pt.ROUND, "source_fingerprint": e["fingerprint"],
                "evaluation_fingerprint": e["evaluation_fingerprint"], "lock_sha256": e["evaluation"]["lock_sha256"],
                "config_sha256": w.file_hash(pt.CONFIG), "base_policy": "TM4",
                "files": files, "raw_points": len(seen), "used_points": len(rows), "excluded": exclusions,
                "dates": sorted(set(dataset["dates"].tolist())), "reward_scale": SCALE,
                "input": "observation167 only (ARENA3 119 + timetable 48)",
                "labels": "paired 300-second surrogate cost under TM4 continuation, platoon-rule sampled futures, privileged",
                "future_split": "first floor(valid/2) fit; remaining score; no overlap",
                "future_partitions": {r["state_ids"]: {"fit": r["fit_future_indices"], "score": r["score_future_indices"]}
                                      for r in rows}}
    dataset["fingerprint"] = w.digest(manifest)
    dataset["manifest"] = manifest
    return dataset


# ----------------------------------------------------------------------------- 학생
def student_loss(net, obs, gains, gain_se, allowed, target: str, st: dict):
    """허용 녹색 중 TM4가 아닌 것의 상대값을 짝 이득(target='mean') 또는 이득 − 표준오차('risk')로 맞춘다(PI R1과 같은 가중)."""
    if target not in ("mean", "risk"):
        raise ValueError(f"알 수 없는 학습 목표: {target}")
    rel = net(obs)
    anchor = obs[:, pt.TM4_ONEHOT].argmax(dim=1)
    valid = allowed.bool().clone()
    valid.scatter_(1, anchor[:, None], False)
    if not bool(valid.any()):
        return rel.sum() * 0.0
    goal = gains if target == "mean" else gains - gain_se
    low, high = st["confidence_clip"]
    weights = (gains.abs() / (gain_se + float(st["confidence_floor"]))).clamp(float(low), float(high)) * valid
    loss = F.smooth_l1_loss(rel, goal, reduction="none")
    return (loss * weights).sum() / weights.sum().clamp_min(1e-8)


def _tensors(dataset: dict, ids, gains_key: str, se_key: str) -> dict:
    ids = np.asarray(ids, np.int64)
    return {"obs": torch.from_numpy(dataset["observations"][ids].astype(np.float32)),
            "gains": torch.from_numpy(dataset[gains_key][ids].astype(np.float32)),
            "gain_se": torch.from_numpy(dataset[se_key][ids].astype(np.float32)),
            "allowed": torch.from_numpy(dataset["allowed"][ids].astype(np.bool_))}


def train_student(dataset: dict, cfg: dict, seed: int, *, indices=None, target: str | None = None,
                  gains_key: str = "gains", se_key: str = "gain_se", epochs: int | None = None):
    st = cfg["student"]
    target = st["target"] if target is None else target
    epochs = int(st["epochs"] if epochs is None else epochs)
    ids = np.arange(len(dataset["observations"])) if indices is None else np.asarray(indices, np.int64)
    if epochs < 0 or not len(ids):
        raise ValueError("학습 횟수·자료 수가 올바르지 않습니다.")
    torch.manual_seed(int(seed))
    net = pt.PlatoonNet(tuple(st["network"]))
    rng = np.random.default_rng(int(seed))
    opt = torch.optim.Adam(net.parameters(), lr=float(st["learning_rate"]), weight_decay=float(st["weight_decay"]))
    started, updates, last = time.monotonic(), 0, None
    full = _tensors(dataset, ids, gains_key, se_key)
    with torch.no_grad():
        initial = float(student_loss(net, full["obs"], full["gains"], full["gain_se"], full["allowed"], target, st))
    net.train()
    for _ in range(epochs):
        order = rng.permutation(ids)
        for start in range(0, len(order), int(st["batch_size"])):
            d = _tensors(dataset, order[start:start + int(st["batch_size"])], gains_key, se_key)
            loss = student_loss(net, d["obs"], d["gains"], d["gain_se"], d["allowed"], target, st)
            if not bool(torch.isfinite(loss)):
                raise ValueError("학생 손실이 유한하지 않습니다.")
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), float(st["grad_clip"]))
            opt.step()
            updates += 1
            last = float(loss.detach())
    net.eval()
    with torch.no_grad():
        final = float(student_loss(net, full["obs"], full["gains"], full["gain_se"], full["allowed"], target, st))
    return net, {"seed": int(seed), "target": target, "epochs": epochs, "points": int(len(ids)), "updates": updates,
                 "seconds": time.monotonic() - started, "last_loss": last, "initial_full_loss": initial,
                 "final_full_loss": final, "dataset_fingerprint": dataset["fingerprint"], "gains_key": gains_key}


def predict(net, observations: np.ndarray) -> np.ndarray:
    """허용 녹색 중 가장 큰 값(같으면 TM4) — pt.greedy_green과 같은 선택을 묶음으로."""
    with torch.no_grad():
        rel = net(torch.from_numpy(observations.astype(np.float32))).numpy().astype(np.float64)
    anchor = pt.anchor_green(observations)
    rel = np.where(pt.allowed_mask(observations), rel, -np.inf)
    rel[np.arange(len(rel)), anchor] += 1e-6
    return rel.argmax(axis=1)


def teacher_greens(dataset: dict, gains_key: str = "fit_gains") -> np.ndarray:
    """교사 c=0: 허용 녹색 중 짝 평균 이득이 가장 큰 것, 0 이하이면 TM4(특권 정보, 진단용)."""
    g = np.where(dataset["allowed"], dataset[gains_key].astype(np.float64), -np.inf)
    rows = np.arange(len(g))
    g[rows, dataset["anchor"]] = np.maximum(g[rows, dataset["anchor"]], 0.0) + 1e-9
    return g.argmax(axis=1)


def score(dataset: dict, greens: np.ndarray, cfg: dict) -> dict:
    """뒤 절반 미래로 잰 TM4 녹색 대비 이득(차량·초/지점). 날짜마다 평균 → 날짜 평균과 t95. 시간당 합은 어림."""
    greens = np.asarray(greens, np.int64)
    rows = np.arange(len(greens))
    gains = dataset["score_gains"][rows, greens].astype(np.float64) * SCALE
    switched = greens != dataset["anchor"]
    dates = sorted(set(dataset["dates"].tolist()))
    per_date = {d: float(gains[dataset["dates"] == d].mean()) for d in dates}
    values = list(per_date.values())
    mean = statistics.fmean(values)
    half = T95.get(len(values), 1.96) * statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else None
    fraction = float(cfg["points"]["fraction"])
    hourly = [gains[dataset["dates"] == d].sum() / fraction / len(set(dataset["situations"][dataset["dates"] == d].tolist()))
              for d in dates]
    return {"mean": mean, "t95": [mean - half, mean + half] if half is not None else [None, None],
            "positive_dates": sum(v > 0 for v in values), "points": int(len(greens)), "switched": int(switched.sum()),
            "harmful": int((switched & (gains < 0)).sum()), "per_date": per_date, "per_hour_rough": statistics.fmean(hourly)}


def offline_report(dataset: dict, cfg: dict, out: Path) -> dict:
    """날짜 하나씩 보류: 다른 날짜의 앞 절반 미래로 학습하고 보류 날짜를 뒤 절반 미래로 채점한다(참고 진단, 판정에 쓰지 않음)."""
    torch.set_num_threads(1)
    out = Path(out)
    s2e._write(out / "dataset_manifest.json", {"fingerprint": dataset["fingerprint"], **dataset["manifest"]})
    dates = sorted(set(dataset["dates"].tolist()))
    if len(dates) < 2:
        raise ValueError("날짜 보류 진단에는 두 날짜 이상이 필요합니다.")
    n = len(dataset["observations"])
    methods = {"TM4(지금 정책)": score(dataset, dataset["anchor"], cfg),
               "M4 녹색(정보 없음)": score(dataset, dataset["m4"], cfg),
               "교사 c=0(앞 절반, 특권)": score(dataset, teacher_greens(dataset), cfg)}
    predictions, folds = np.zeros(n, np.int64), []
    for date in dates:
        train = np.flatnonzero(dataset["dates"] != date)
        held = np.flatnonzero(dataset["dates"] == date)
        net, stats = train_student(dataset, cfg, 42, indices=train, gains_key="fit_gains", se_key="fit_gain_se")
        predictions[held] = predict(net, dataset["observations"][held])
        folds.append({"heldout_date": date, "train_points": len(train), "heldout_points": len(held),
                      **{k: stats[k] for k in ("updates", "seconds", "initial_full_loss", "final_full_loss")}})
        print(f"날짜 보류 진단: {date} ({len(train)}학습/{len(held)}보류)", flush=True)
    methods["학생(날짜 보류)"] = score(dataset, predictions, cfg)
    report = {"kind": "platoon1-offline-v1", "round": pt.ROUND, "dataset_fingerprint": dataset["fingerprint"],
              "status": "advisory_only", "methods": methods, "folds": folds,
              "units": "vehicle-seconds of 300s surrogate cost per decision point, relative to the TM4 green, TM4 continues",
              "per_hour_rough": "sum over sampled points ÷ fraction, per episode; overlapping horizons, ranking only",
              "student_input": "observation167 only", "future_split": "fit first half, score second half"}
    s2e._write(out / "offline_report.json", report)
    lines = [f"# PLATOON1 — 날짜 보류 진단(참고)", "",
             "300초 대리 비용 차이(차량·초/지점, TM4 녹색 대비, 뒤는 TM4). 초/대나 한 시간 성능이 아닙니다.", "",
             "| 방식 | 지점당 [t95] | 양수 날짜 | 시간당 합(어림) | 바꾼 지점 | 손해 바꿈 |", "|---|---|---|---|---|---|"]
    for name, r in methods.items():
        low, high = r["t95"]
        cell = f"{r['mean']:+.1f} [{low:+.1f}, {high:+.1f}]" if low is not None else f"{r['mean']:+.1f}"
        lines.append(f"| {name} | {cell} | {r['positive_dates']}/{len(dates)} | {r['per_hour_rough']:+.0f} | "
                     f"{r['switched']}/{r['points']} | {r['harmful']} |")
    lines += ["", "교사는 표본 미래를 보는 진단용입니다. 이 표는 판정·모델 선택에 쓰지 않습니다(사전 등록).",
              "학생 본 학습은 모든 유효 미래와 설정의 목표(mean)로 합니다."]
    (out / "offline_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def save_student(folder: Path, net, meta: dict) -> str:
    folder = Path(folder)
    folder.parent.mkdir(parents=True, exist_ok=True)
    tmp = folder.with_name(f".{folder.name}.{uuid.uuid4().hex}.tmp")
    tmp.mkdir()
    torch.save({"kind": KIND, **meta, "state": net.state_dict()}, tmp / "model.pt")
    sha = w.file_hash(tmp / "model.pt")
    s2e._write(tmp / "manifest.json", {"kind": KIND, **meta, "model_sha256": sha})
    if folder.exists():
        shutil.rmtree(folder)
    os.replace(tmp, folder)
    return sha


def student_identity(folder: Path) -> str:
    folder = Path(folder)
    manifest = w.read(folder / "manifest.json")
    if manifest.get("kind") != KIND:
        raise ValueError(f"PLATOON1 학생 모델이 아닙니다: {folder}")
    sha = w.file_hash(folder / "model.pt")
    if sha != manifest["model_sha256"]:
        raise ValueError(f"학생 모델 파일 해시가 다릅니다: {folder}")
    return sha


def load_student(folder: Path):
    student_identity(folder)
    manifest = w.read(Path(folder) / "manifest.json")
    payload = torch.load(Path(folder) / "model.pt", map_location="cpu", weights_only=False)
    for key in ("round", "seed", "dataset_fingerprint", "network", "target"):
        if payload.get(key) != manifest.get(key):
            raise ValueError(f"학생 모델 파일과 기록이 다릅니다: {key}")
    net = pt.PlatoonNet(tuple(manifest["network"]))
    net.load_state_dict(payload["state"])
    net.eval()
    return net


def student_dirs(root: Path, cfg: dict) -> list[Path]:
    return [Path(root) / f"seed_{s}" for s in cfg["student"]["seeds"]]


def train_students(data: Path, out: Path, cfg: dict) -> dict:
    torch.set_num_threads(1)
    dataset = load_dataset(data, cfg)
    out = Path(out)
    s2e._write(out / "dataset_manifest.json", {"fingerprint": dataset["fingerprint"], **dataset["manifest"]})
    summary = {}
    for seed, folder in zip(cfg["student"]["seeds"], student_dirs(out, cfg)):
        if (folder / "manifest.json").exists():
            m = w.read(folder / "manifest.json")
            if m.get("dataset_fingerprint") != dataset["fingerprint"] or m.get("seed") != seed:
                raise ValueError(f"이미 있는 학생이 다른 자료·난수로 학습됐습니다: {folder}")
            summary[seed] = {"model_sha256": student_identity(folder), "reused": True}
            continue
        net, stats = train_student(dataset, cfg, seed)
        meta = {"round": pt.ROUND, "seed": int(seed), "stage": "final", "dataset_fingerprint": dataset["fingerprint"],
                "network": list(cfg["student"]["network"]), "target": cfg["student"]["target"],
                "observation_size": pt.OBS, "config_sha256": w.file_hash(pt.CONFIG),
                "source_sha256": {p: w.file_hash(ROOT / p) for p in eval_sources()}, "training": stats}
        summary[seed] = {"model_sha256": save_student(folder, net, meta), "reused": False, **stats}
        print(f"학생 난수 {seed}: 지점 {stats['points']}, 갱신 {stats['updates']}, 손실 {stats['initial_full_loss']:.4f} → "
              f"{stats['final_full_loss']:.4f} ({stats['seconds']:.0f}초)", flush=True)
    s2e._write(out / "students.json", {"dataset_fingerprint": dataset["fingerprint"], "students": summary,
                                       "updated_unix": time.time()})
    return summary


# ----------------------------------------------------------------------------- 평가(무리 도착 세계)
def declaration(folders: list[Path], part: str, cfg: dict) -> dict:
    if len(folders) != len(cfg["student"]["seeds"]):
        raise ValueError("PT1은 세 모델이어야 합니다.")
    identities, seeds, datasets = [], [], []
    for folder in folders:
        identities.append(student_identity(Path(folder)))
        m = w.read(Path(folder) / "manifest.json")
        if m.get("round") != pt.ROUND or m.get("stage") != "final" or m.get("target") != cfg["student"]["target"] \
                or m.get("network") != list(cfg["student"]["network"]) or m.get("observation_size") != pt.OBS:
            raise ValueError(f"학생 모델의 라운드·단계·목표·신경망·관측이 설정과 다릅니다: {folder}")
        seeds.append(m.get("seed"))
        datasets.append(m.get("dataset_fingerprint"))
    if seeds != list(cfg["student"]["seeds"]):
        raise ValueError("평가 모델의 순서는 난수 42·43·44여야 합니다.")
    if len(set(datasets)) != 1 or not datasets[0]:
        raise ValueError("세 모델이 같은 자료로 학습되지 않았습니다.")
    return {"name": STUDENT, "part": part, "model_dirs": [str(Path(p).resolve()) for p in folders],
            "model_identities": identities, "seeds": seeds, "dataset_fingerprint": datasets[0]}


def build_policy(job: dict):
    if job["kind"] == "student":
        if student_identity(Path(job["model_dir"])) != job["model_identity"]:
            raise ValueError("작업 시작 후 모델이 바뀌었습니다.")
        return pt.StudentPolicy(load_student(Path(job["model_dir"])), name=job["policy"])
    if job["kind"] == "tm4":
        return pt.TM4Policy(job["policy"])
    if job["kind"] == "fixed":
        return ac.fixed_policy(job["policy"])
    raise ValueError(f"알 수 없는 작업 종류: {job['kind']}")


def eval_worker(job: dict) -> dict:
    started = time.monotonic()
    work = Path(job["work"])
    try:
        if platform.system() == "Darwin":
            raise RuntimeError("실제 SUMO 평가는 Colab에서만 실행합니다. 맥에서는 judge 명령만 사용하세요.")
        w.configure()
        torch.set_num_threads(1)
        shutil.rmtree(work, ignore_errors=True)
        policy = build_policy(job)
        result = pt.run_episode(job["situation"], work, policy, max_seconds=3600.0)
        expected = job["situation"].get("route_sha256")
        if result.metrics["route_sha256"] != expected:
            raise ValueError(f"평가 차량 파일의 SHA가 설정과 다릅니다: {result.metrics['route_sha256']} ≠ {expected}")
        ac.write_run(Path(job["target"]), result,
                     {"policy": job["policy"], "fingerprint": job["fingerprint"], "model_identity": job.get("model_identity"),
                      "job_seconds": time.monotonic() - started,
                      "information": policy.info() if hasattr(policy, "info") else {"kind": job["kind"]}})
        return {"ok": True, "seconds": time.monotonic() - started, "name": f"{job['policy']}/{job['sid']}"}
    except BaseException as error:
        return {"ok": False, "seconds": time.monotonic() - started, "name": f"{job['policy']}/{job['sid']}",
                "error": f"{type(error).__name__}: {error}"}
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _prepare_map_cache() -> None:
    """작업자 여러 개가 동시에 지도 캐시를 처음 만들다 부딪히지 않게(218절) 부모 프로세스에서 한 번 만든다."""
    from focused_eval import CACHE, dqn_template_network
    dqn_template_network(CACHE)


def evaluate(folders: list[Path], out: Path, work: Path, workers: int, cfg: dict, part: str = "eval", *,
             executor=None, worker=eval_worker) -> dict:
    sits = situations(cfg, part)
    out = Path(out)
    fp_ = ref.register(out, extra_files=eval_sources())
    decl = declaration([Path(p) for p in folders], part, cfg)
    s2e._freeze(out / "models.json", decl)
    jobs = []
    for i, (folder, identity) in enumerate(zip(folders, decl["model_identities"])):
        jobs += ref.jobs_for(out, Path(work), fp_, f"{STUDENT}_m{i}", sits, model_dir=folder, model_identity=identity,
                             kind="student")
    if worker is eval_worker and executor is None:
        _prepare_map_cache()
    status = ref.run_jobs(jobs, max(1, workers), ref.Ledger(out, STUDENT), worker=worker, executor=executor)
    s2e._write(out / "status" / f"{STUDENT}_{part}.json", {**status, "total": len(jobs), "part": part,
                                                           "updated_unix": time.time()})
    return status


def baseline(out: Path, work: Path, workers: int, cfg: dict, part: str = "confirmation", *, executor=None,
             worker=eval_worker) -> dict:
    """확인 배치의 P 세계 M4·TM4(평가 배치는 PLATOON0 결과를 쓰므로 돌리지 않는다)."""
    sits = situations(cfg, part)
    out = Path(out)
    fp_ = ref.register(out, extra_files=eval_sources())
    jobs = (ref.jobs_for(out, Path(work), fp_, "M4", sits, model_identity="P/M4", kind="fixed")
            + ref.jobs_for(out, Path(work), fp_, "TM4", sits, model_identity="P/TM4", kind="tm4"))
    if worker is eval_worker and executor is None:
        _prepare_map_cache()
    status = ref.run_jobs(jobs, max(1, workers), ref.Ledger(out, "baseline"), worker=worker, executor=executor)
    s2e._write(out / "status" / f"baseline_{part}.json", {**status, "total": len(jobs), "part": part,
                                                          "updated_unix": time.time()})
    return status


# ----------------------------------------------------------------------------- 판정
def _rows(folder: Path, policy: str, identity: str, sits: list[dict], record: dict) -> list[dict]:
    """한 정책의 실행을 폴더 지문·모델 표시·차량 파일 SHA까지 대조해 읽는다(빠진 상황은 빼고 돌려준다)."""
    rows = []
    for s in sits:
        m = ac.read_run(Path(folder) / "runs" / policy / ac.situation_id(s), record["fingerprint"], identity)
        if m is None:
            continue
        if m.get("route_sha256") != s["route_sha256"]:
            raise ValueError(f"{policy} 실행의 차량 파일이 설정과 다릅니다: {ac.situation_id(s)}")
        rows.append(m)
    return rows


def paired_difference(a: dict, b: dict, t975: float) -> dict:
    """같은 상황·같은 차량에서 날짜별 M4 대비 이득(판정과 같은 차량 가중 정의)의 차 a − b."""
    dates = sorted(a["per_date_gain_s"])
    if dates != sorted(b["per_date_gain_s"]):
        raise ValueError("짝 비교할 두 후보의 날짜가 다릅니다.")
    diffs = {d: float(a["per_date_gain_s"][d]) - float(b["per_date_gain_s"][d]) for d in dates}
    mean, low, high = ref.t_interval(list(diffs.values()), t975)
    return {"mean_s": mean, "t95_s": [low, high], "positive_dates": sum(v > 0 for v in diffs.values()),
            "dates": len(dates), "per_date_s": diffs}


def main_verdict(student: dict, learning: dict, rules: dict) -> dict:
    """사전 등록 주 판정: ARENA3 통과(M4 대비) + L 평균 ≥ 0.3초·하한 > 0·양수 날짜 ≥ 5."""
    low, high = learning["t95_s"]
    reading = "TM4보다 짧다(학습의 몫)" if low > 0 else ("TM4보다 길다" if high < 0 else "구분 못 함")
    passed = (student["status"] == "pass" and learning["mean_s"] >= float(rules["learning_min_gain_s"]) and low > 0
              and learning["positive_dates"] >= int(rules["learning_min_positive_dates"]))
    return {"passed": bool(passed), "arena3_status": student["status"], "learning_reading": reading}


def collect_report(student_folder: Path, baseline_folder: Path, cfg: dict, part: str = "eval") -> dict:
    """eval: 기준 = PLATOON0 폴더(runs/P/<정책>), confirmation: 기준 = PLATOON1 확인 기준선 폴더(runs/<정책>)."""
    lock = ac.config()
    sits = situations(cfg, part)
    student_record = s2e.folder_record(student_folder)
    base_record = s2e.folder_record(baseline_folder)
    if base_record["evaluation_fingerprint"] != student_record["evaluation_fingerprint"]:
        raise ValueError("기준 실행과 학생 평가의 평가 지문(잠금·평가 코드·패키지)이 다릅니다.")
    if part == "eval":
        prefix = cfg["evaluation"]["baseline"]["fingerprint_prefix"]
        if not base_record["fingerprint"].startswith(prefix):
            raise ValueError(f"PLATOON0 결과 폴더가 아닙니다(지문 {base_record['fingerprint'][:12]} ≠ {prefix}).")
        names = {n: (f"P/{n}", f"P/{n}") for n in cfg["evaluation"]["baseline"]["policies"]}
    else:
        names = {n: (n, f"P/{n}") for n in ("M4", "TM4")}
    base = {n: _rows(baseline_folder, policy, identity, sits, base_record) for n, (policy, identity) in names.items()}
    missing = {n: len(sits) - len(r) for n, r in base.items() if len(r) != len(sits)}
    decl = w.read(Path(student_folder) / "models.json")
    if decl.get("name") != STUDENT or decl.get("part") != part:
        raise ValueError("모델 선언의 이름·배치가 요청과 다릅니다.")
    if decl != declaration([Path(p) for p in decl["model_dirs"]], part, cfg):
        raise ValueError("평가에 사용한 학생 모델이 바뀌었습니다.")
    models = [_rows(student_folder, f"{STUDENT}_m{i}", x, sits, student_record) for i, x in enumerate(decl["model_identities"])]
    if any(len(r) != len(sits) for r in models):
        missing[STUDENT] = [len(sits) - len(r) for r in models]
    if missing:
        return {"status": "incomplete", "missing": missing, "part": part}
    for name, rows in (*base.items(), *((f"{STUDENT}_m{i}", r) for i, r in enumerate(models))):
        problems = {m["situation"]: ref.quality_errors(m) for m in rows if ref.quality_errors(m)}
        if problems and name in ("M4", "TM4"):
            raise ValueError(f"기준 {name}의 품질 문제가 남아 있습니다: {problems}")
    m4 = base["M4"]
    results = {STUDENT: ref.judge(models, m4, None, lock, sits)}
    for n, rows in base.items():
        if n != "M4":
            results[n] = ref.judge([rows], m4, None, lock, sits)
    t975 = float(lock["judge"]["t975"])
    pairs = [p for p in cfg["rules"]["paired_comparisons"] if p[0] in results and p[1] in results]
    paired = {f"{a}-{b}": paired_difference(results[a], results[b], t975) for a, b in pairs}
    verdict = main_verdict(results[STUDENT], paired[f"{STUDENT}-TM4"], cfg["rules"])
    return {"status": "done", "part": part, "verdict": verdict, "results": results, "paired": paired, "models": decl,
            "evaluation_fingerprint": student_record["evaluation_fingerprint"],
            "baseline_fingerprint": base_record["fingerprint"], "student_fingerprint": student_record["fingerprint"],
            "lock_sha256": w.file_hash(ac.CONFIG), "config_sha256": w.file_hash(pt.CONFIG), "updated_unix": time.time(),
            "note": "반복 사용 날짜의 SUMO 개발 비교. 무리 도착은 모형이고 시간표는 무리를 만든 것과 같다."}


def render(report: dict, title: str) -> str:
    if report["status"] != "done":
        return f"# PLATOON1 — {title}\n\n아직 끝나지 않은 실행이 있습니다: {json.dumps(report['missing'], ensure_ascii=False)}\n"
    f = lambda x, nd=3: "-" if x is None else f"{x:+.{nd}f}"  # noqa: E731
    lines = [f"# PLATOON1 — {title}", "",
             "무리 도착 세계 P의 M4 대비 이득(초/대, +면 후보가 짧음). 날짜 7개 평균과 t95, ARENA3 심판 함수 그대로.", "",
             "| 후보 | ARENA3 판정 | 날짜 평균 | t95 | 양수 날짜 | 동 좌회전 차이 | 남 좌회전 차이 | 300초 초과(후보/M4) | 정책 고유 충돌 |",
             "|---|---|---|---|---|---|---|---|---|"]
    for name, r in report["results"].items():
        lines.append(f"| {name} | {r['status']} | {f(r['gain_date_mean_s'])} | [{f(r['date_t95_s'][0])}, {f(r['date_t95_s'][1])}] "
                     f"| {r['positive_dates']}/7 | {f(r['left_turns']['left_east']['difference_s'], 2)} "
                     f"| {f(r['left_turns']['left_south']['difference_s'], 2)} | {r['long_waits']['candidate']:.1f}/{r['long_waits']['m4']} "
                     f"| {len(r['collisions']['policy_specific'])} |")
    lines += ["", "## 짝 비교(같은 상황, 날짜 7개, 초/대, +면 앞쪽이 짧음)", "", "| 비교 | 평균 | t95 | 앞쪽이 나은 날짜 |",
              "|---|---|---|---|"]
    for key, p in report["paired"].items():
        lines.append(f"| {key.replace('-', ' − ')} | {f(p['mean_s'])} | [{f(p['t95_s'][0])}, {f(p['t95_s'][1])}] "
                     f"| {p['positive_dates']}/{p['dates']} |")
    v = report["verdict"]
    lines += ["", f"- **주 판정(사전 등록): {'통과' if v['passed'] else '통과 못 함'}** — PT1 ARENA3 판정 {v['arena3_status']}, "
                  f"학습의 몫 L = PT1 − TM4: {v['learning_reading']}",
              "- PT1 = 무리 도착 세계 정책 반복 1라운드 학생(시간표 관측 167칸, 세 모델 평균). TM4 = 같은 정보의 계획기(지금 정책). "
              "OM4 = 정확한 도착을 아는 계획기(참고). " + report["note"]]
    return "\n".join(lines) + "\n"


def save_report(out: Path, report: dict, title: str) -> None:
    out = Path(out)
    s2e._write(out / "judge.json", report)
    (out / "summary.md").write_text(render(report, title), encoding="utf-8")


def confirm(judge_dir: Path, students: Path, out: Path, work: Path, workers: int, cfg: dict, *, executor=None,
            worker=eval_worker) -> dict:
    """주 판정을 통과했을 때만 확인 84상황(P)에서 M4·TM4·PT1을 한 번 잰다."""
    out = Path(out)
    report = w.read(Path(judge_dir) / "judge.json")
    if report.get("status") != "done" or report.get("part") != "eval":
        raise ValueError("개발 판정이 끝나기 전에는 확인하지 않습니다.")
    if not report["verdict"]["passed"]:
        result = {"status": "not_run_development_failed", "development_verdict": report["verdict"]}
        s2e._write(out / "confirmation.json", result)
        return result
    folders = student_dirs(students, cfg)
    if declaration(folders, "eval", cfg)["model_identities"] != report["models"]["model_identities"]:
        raise ValueError("개발 평가 모델과 지금 모델이 다릅니다.")
    status = baseline(out / "baseline", Path(work) / "baseline", workers, cfg, "confirmation", executor=executor, worker=worker)
    status_s = evaluate(folders, out / STUDENT, Path(work) / STUDENT, workers, cfg, "confirmation", executor=executor,
                        worker=worker)
    if status["failed"] or status_s["failed"]:
        return {"status": "runs_failed", "failed": status["failed"] + status_s["failed"]}
    rep = collect_report(out / STUDENT, out / "baseline", cfg, part="confirmation")
    save_report(out, rep, "확인(C1·C2)")
    passed = rep["status"] == "done" and rep["results"][STUDENT]["status"] == "pass" \
        and rep["paired"][f"{STUDENT}-TM4"]["t95_s"][0] > 0
    result = {"status": "pass" if passed else "fail", "result": rep.get("verdict"),
              "learning": rep.get("paired", {}).get(f"{STUDENT}-TM4")}
    s2e._write(out / "confirmation.json", result)
    return result


# ----------------------------------------------------------------------------- 명령(explore_platoon1.py가 부른다)
def add_commands(sub) -> None:
    for name in ("offline", "train"):
        p = sub.add_parser(name)
        p.add_argument("--data", type=Path, required=True)
        p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("evaluate")
    p.add_argument("--students", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1))
    p.add_argument("--part", choices=tuple(PARTS), default="eval")
    p = sub.add_parser("judge")
    p.add_argument("--eval", type=Path, required=True, help="PT1 평가 폴더")
    p.add_argument("--platoon0", type=Path, required=True, help="PLATOON0 결과 폴더(P 세계 M4·TM4·OM4·P1M4)")
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("confirm")
    p.add_argument("--judge", type=Path, required=True)
    p.add_argument("--students", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1))


def run_command(args, cfg: dict) -> int:
    if args.cmd in ("offline", "train", "evaluate", "confirm") and platform.system() == "Darwin":
        raise SystemExit("맥에서는 학습·SUMO를 실행하지 않습니다. 이 명령은 Colab 노트북에서 실행하세요(맥은 judge만).")
    if args.cmd == "offline":
        report = offline_report(load_dataset(args.data, cfg), cfg, args.out)
        print((Path(args.out) / "offline_report.md").read_text())
        return 0 if report else 1
    if args.cmd == "train":
        print(json.dumps(w.plain(train_students(args.data, args.out, cfg)), ensure_ascii=False, indent=1))
        return 0
    if args.cmd == "evaluate":
        status = evaluate(student_dirs(args.students, cfg), args.out, args.work, args.workers, cfg, args.part)
        return 1 if status["failed"] else 0
    if args.cmd == "judge":
        report = collect_report(args.eval, args.platoon0, cfg, "eval")
        save_report(args.out, report, "개발 평가(평가 84상황, 무리 도착 세계)")
        print(render(report, "개발 평가(평가 84상황, 무리 도착 세계)"))
        return 0 if report["status"] == "done" else 2
    result = confirm(args.judge, args.students, args.out, args.work, args.workers, cfg)
    print(json.dumps(w.plain(result), ensure_ascii=False, indent=2))
    return 1 if result["status"] == "runs_failed" else 0
