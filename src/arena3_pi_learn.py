"""ARENA3 정책 반복 — 자료 검증, 날짜 보류 진단, 학생(녹색 4개 값) 학습, 심판 평가·판정·확인(설정 config/arena3_pi_r1.json).

학생 입력은 관측 119칸뿐이다. 날짜·실제 미래·롤아웃 비용은 묶음 구분과 감독 라벨에만 쓴다.
오프라인 점수는 300초 대리 비용(차량·초/지점, M4 녹색 대비, 뒤는 지금 정책이 운전)이며 초/대나 한 시간 성능으로 바꾸지 않는다.
평가는 1단계 심판(arena3_referee.judge)을 그대로 쓰고, 실제 SUMO 평가는 Colab에서만 한다.
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
import arena3_pi as pi  # noqa: E402
import arena3_referee as ref  # noqa: E402
import arena3_stage2_eval as s2e  # noqa: E402
import explore_weight as w  # noqa: E402

KIND = "arena3-pi-green-v1"
SCALE = 1000.0
LEARN_SOURCES = ("src/arena3_pi_learn.py", "src/arena3_stage2_eval.py")
DIRTY = ("missing", "unfinished", "changed_by_rules", "mode_action_errors", "p0_mismatch")
T95 = {2: 12.706204736, 3: 4.302652730, 4: 3.182446305, 5: 2.776445105, 6: 2.570581836, 7: 2.446911851, 8: 2.364624252}


def eval_sources(round_name: str) -> tuple[str, ...]:
    return (*pi.sources(round_name), *LEARN_SOURCES)


def variant_names(cfg: dict) -> dict[str, str]:
    """{'R1M': 'M', 'R1G': 'G', 'R1A': 'A'}"""
    return {cfg["variants"][v]["name"]: v for v in pi.VARIANTS}


def _write(path: Path, value: dict) -> None:
    s2e._write(Path(path), value)


# ----------------------------------------------------------------------------- 자료
def validate_folder(folder: Path, cfg: dict) -> dict:
    """자료 폴더의 지문 기록이 온전하고, 그 자료를 만든 코드·환경·잠금이 지금과 같은지."""
    e = w.read(Path(folder) / "experiment.json")
    if w.digest(e["evaluation"]) != e["evaluation_fingerprint"]:
        raise ValueError("자료 폴더의 평가 지문이 내용과 다릅니다.")
    if w.digest({k: v for k, v in e.items() if k != "fingerprint"}) != e["fingerprint"]:
        raise ValueError("자료 폴더 지문이 내용과 다릅니다.")
    if e["evaluation"]["lock_sha256"] != w.file_hash(ac.CONFIG):
        raise ValueError("자료와 ARENA3 잠금이 다릅니다.")
    if (e["evaluation"].get("kind") != ac.EXPERIMENT or
            e["evaluation"].get("environment_fingerprint") != ac.config()["environment"]["fingerprint"]):
        raise ValueError("자료의 실험·환경 지문이 잠금과 다릅니다.")
    if set(e["extra_files"]) != set(pi.sources(cfg["round"])):
        raise ValueError("자료의 수집 소스 목록이 이 라운드와 다릅니다.")
    for rel, expected in {**e["evaluation"]["files"], **e["extra_files"]}.items():
        p = ROOT / rel
        if not p.is_file() or w.file_hash(p) != expected:
            raise ValueError(f"자료를 만든 코드·환경 파일이 달라졌습니다: {rel}")
    return e


def _gain_rows(costs: dict[int, list], indices: list[int], allowed: list[int], m4: int):
    gain = np.zeros(pi.GREENS, np.float32)
    se = np.zeros(pi.GREENS, np.float32)
    for g in allowed:
        diffs = [(float(costs[m4][j]["cost"]) - float(costs[g][j]["cost"])) / SCALE for j in indices]
        gain[g] = statistics.fmean(diffs)
        se[g] = statistics.stdev(diffs) / math.sqrt(len(diffs)) if len(diffs) > 1 else 0.0
    return gain, se


def point_to_row(point: dict, date: str, situation: str, cfg: dict) -> tuple[dict | None, str | None]:
    """순수 계산. 복원 불일치·유효 미래 부족은 명시적으로 빼고, 관측·후보·비용 형식이 틀리면 멈춘다."""
    if not point.get("restore_check", {}).get("ok"):
        return None, "restore_mismatch"
    obs = np.asarray(point["observation"], np.float32)
    if obs.shape != (pi.OBS,) or not np.isfinite(obs).all():
        raise ValueError(f"119칸 유한 관측이 아닙니다: {point.get('state')}")
    allowed = sorted(int(a) for a in point["allowed"])
    if np.flatnonzero(pi.allowed_mask(obs)).tolist() != allowed or len(allowed) < 2:
        raise ValueError(f"관측의 허용 녹색이 기록과 다르거나 고를 것이 없습니다: {point.get('state')}")
    greens = pi.mode_greens(obs)
    if greens.tolist() != point["greens"]:
        raise ValueError("관측의 모드 녹색이 기록과 다릅니다.")
    m4 = int(greens[ac.M4])
    if m4 != int(point["m4"]):
        raise ValueError("M4 녹색이 관측과 다릅니다.")
    kind = "choice" if len(set(greens.tolist())) > 1 else "agree"
    if kind != point["kind"]:
        raise ValueError("갈림·합의 구분이 관측과 다릅니다.")
    executed = int(point["executed"])
    if executed not in allowed:
        raise ValueError("본 에피소드의 행동이 허용 녹색이 아닙니다.")
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
    gain, se = _gain_rows(costs, valid, allowed, m4)
    fit_gain, fit_se = _gain_rows(costs, valid[:cut], allowed, m4)
    score_gain, _ = _gain_rows(costs, valid[cut:], allowed, m4)
    return {"observations": obs, "allowed": pi.allowed_mask(obs), "proposed": np.eye(pi.GREENS, dtype=bool)[greens].any(axis=0),
            "m4": m4, "executed": executed, "kinds": kind, "gains": gain, "gain_se": se, "fit_gains": fit_gain,
            "fit_gain_se": fit_se, "score_gains": score_gain, "dates": str(date), "state_ids": str(point["state"]),
            "situations": situation, "valid_futures": len(valid),
            "fit_future_indices": valid[:cut], "score_future_indices": valid[cut:]}, None


ARRAY_KEYS = ("observations", "allowed", "proposed", "m4", "executed", "kinds", "gains", "gain_se", "fit_gains",
              "fit_gain_se", "score_gains", "dates", "state_ids", "situations", "valid_futures")


def load_dataset(folder: Path, cfg: dict, *, check_provenance: bool = True) -> dict:
    """끝난 자료 64판을 모두 읽는다. 평가·확인 날짜 폴더는 읽지 않는다."""
    folder = Path(folder)
    e = validate_folder(folder, cfg) if check_provenance else w.read(folder / "experiment.json")
    lock = ac.config()
    expected = {ac.situation_id(s): s for s in cfg["situations_data"]}
    runs = folder / "runs" / pi.POLICY
    present = {p.name for p in runs.iterdir() if p.is_dir() and (p / "cache.json").exists()} if runs.is_dir() else set()
    if present != set(expected):
        raise ValueError(f"자료 상황이 설정과 다릅니다: 누락 {sorted(set(expected) - present)[:5]} 등 "
                         f"{len(set(expected) - present)}개, 추가 {sorted(present - set(expected))[:5]}")
    if set(lock["dates"]["train"]) & set(lock["dates"]["eval"]):
        raise ValueError("학습과 평가 날짜가 겹칩니다.")
    base_sha = cfg["base_policy"]["model_sha256"]

    def read_one(sid):
        p = runs / sid
        metric_raw, point_raw = (p / "metrics.json").read_bytes(), (p / "points.jsonl.gz").read_bytes()
        m, cache = json.loads(metric_raw), json.loads((p / "cache.json").read_text())
        if hashlib.sha256(metric_raw).hexdigest() != cache["metrics_sha256"] \
                or hashlib.sha256(point_raw).hexdigest() != m["points_sha256"]:
            raise ValueError(f"자료 파일 해시 불일치: {sid}")
        if cache.get("fingerprint") != e["fingerprint"] or m.get("fingerprint") != e["fingerprint"]:
            raise ValueError(f"자료 실행 지문 불일치: {sid}")
        if cache.get("model_identity") != base_sha or m.get("model_identity") != base_sha:
            raise ValueError(f"자료를 운전한 정책이 설정의 지금 정책과 다릅니다: {sid}")
        s = expected[sid]
        if (s["date"] not in lock["dates"]["train"] or m.get("date") != s["date"] or m.get("situation") != sid
                or m.get("route_sha256") != s["route_sha256"] or m.get("role") != "data" or m.get("policy") != pi.POLICY
                or m.get("pi_round") != cfg["round"]):
            raise ValueError(f"학습 상황·차량 설정 불일치: {sid}")
        dirty = [k for k in DIRTY if m.get(k, 0)]
        if dirty or not m.get("cost_check", {}).get("passed"):
            raise ValueError(f"자료 본 에피소드 품질 오류 {sid}: {dirty}")
        points = [json.loads(x) for x in gzip.decompress(point_raw).splitlines() if x]
        if len(points) != m["pi_points"]:
            raise ValueError(f"지점 수 불일치: {sid}")
        collided = bool(m.get("collisions_at_204820", 0) or m.get("collisions_unexplained", 0))
        return sid, s, points, collided, {"situation": sid, "metrics_sha256": cache["metrics_sha256"],
                                          "points_sha256": m["points_sha256"], "points": len(points),
                                          "mean_delay_s": m["mean_delay_s"], "collided": collided}

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
    dataset["masks"] = {v: pi.action_mask(dataset["observations"], v) for v in pi.VARIANTS}
    manifest = {"kind": "arena3-pi-data-v1", "round": cfg["round"], "source_fingerprint": e["fingerprint"],
                "evaluation_fingerprint": e["evaluation_fingerprint"], "lock_sha256": e["evaluation"]["lock_sha256"],
                "config_sha256": w.file_hash(pi.config_path(cfg["round"])), "base_policy": cfg["base_policy"]["name"],
                "files": files, "raw_points": len(seen), "used_points": len(rows), "excluded": exclusions,
                "kinds": {k: int((dataset["kinds"] == k).sum()) for k in ("choice", "agree")},
                "dates": sorted(set(dataset["dates"].tolist())), "reward_scale": SCALE,
                "input": "observation119 only", "labels": "paired 300-second surrogate cost under the base policy, privileged",
                "future_split": "first floor(valid/2) fit; remaining score; no overlap",
                "future_partitions": {r["state_ids"]: {"fit": r["fit_future_indices"], "score": r["score_future_indices"]}
                                      for r in rows}}
    dataset["fingerprint"] = w.digest(manifest)
    dataset["manifest"] = manifest
    return dataset


# ----------------------------------------------------------------------------- 학생
def student_loss(net, obs, gains, gain_se, allowed, target: str, st: dict):
    """허용 녹색 중 M4가 아닌 것의 상대값을 짝 이득(target='mean') 또는 이득 − 표준오차('risk')로 맞춘다(S2A와 같은 가중)."""
    if target not in ("mean", "risk"):
        raise ValueError(f"알 수 없는 학습 목표: {target}")
    rel = net(obs)
    anchor = obs[:, pi.M4_ONEHOT].argmax(dim=1)
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
    net = pi.GreenNet(tuple(st["network"]))
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


def predict(net, observations: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """변형의 고를 수 있는 녹색 중 가장 큰 값(같으면 M4) — pi.greedy_green과 같은 선택을 묶음으로."""
    with torch.no_grad():
        rel = net(torch.from_numpy(observations.astype(np.float32))).numpy().astype(np.float64)
    m4 = pi.mode_greens(observations)[:, ac.M4]
    rel = np.where(masks, rel, -np.inf)
    rel[np.arange(len(rel)), m4] += 1e-6
    return rel.argmax(axis=1)


def teacher_greens(dataset: dict, masks: np.ndarray, gains_key: str = "fit_gains") -> np.ndarray:
    """교사 c=0: 고를 수 있는 녹색 중 짝 평균 이득이 가장 큰 것, 0 이하이면 M4(특권 정보, 진단용)."""
    g = np.where(masks, dataset[gains_key].astype(np.float64), -np.inf)
    rows = np.arange(len(g))
    g[rows, dataset["m4"]] = np.maximum(g[rows, dataset["m4"]], 0.0) + 1e-9
    return g.argmax(axis=1)


def score(dataset: dict, greens: np.ndarray, cfg: dict) -> dict:
    """뒤 절반 미래로 잰 M4 녹색 대비 이득(차량·초/지점). 날짜마다 평균 → 날짜 평균과 t95. 갈림·합의 따로, 시간당 합은 어림."""
    greens = np.asarray(greens, np.int64)
    rows = np.arange(len(greens))
    gains = dataset["score_gains"][rows, greens].astype(np.float64) * SCALE
    switched = greens != dataset["m4"]
    out = {}
    dates = sorted(set(dataset["dates"].tolist()))
    for kind in ("choice", "agree", "all"):
        sel = np.ones(len(greens), bool) if kind == "all" else dataset["kinds"] == kind
        per_date = {d: float(gains[sel & (dataset["dates"] == d)].mean()) for d in dates if np.any(sel & (dataset["dates"] == d))}
        values = list(per_date.values())
        mean = statistics.fmean(values) if values else None
        half = T95.get(len(values), 1.96) * statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else None
        out[kind] = {"mean": mean, "t95": [mean - half, mean + half] if half is not None else [None, None],
                     "positive_dates": sum(v > 0 for v in values), "points": int(sel.sum()),
                     "switched": int((switched & sel).sum()), "harmful": int((switched & sel & (gains < 0)).sum()),
                     "per_date": per_date}
    fraction = float(cfg["agree_sample"]["fraction"])
    hourly = []
    for d in dates:
        own = dataset["dates"] == d
        episodes = len(set(dataset["situations"][own].tolist()))
        total = gains[own & (dataset["kinds"] == "choice")].sum() + gains[own & (dataset["kinds"] == "agree")].sum() / fraction
        hourly.append(total / episodes)
    out["per_hour_rough"] = statistics.fmean(hourly)
    return out


def offline_report(dataset: dict, cfg: dict, out: Path) -> dict:
    """날짜 하나씩 보류: 다른 날짜의 앞 절반 미래로 학습하고 보류 날짜를 뒤 절반 미래로 채점한다(참고 진단, 판정에 쓰지 않음)."""
    torch.set_num_threads(1)
    out = Path(out)
    _write(out / "dataset_manifest.json", {"fingerprint": dataset["fingerprint"], **dataset["manifest"]})
    dates = sorted(set(dataset["dates"].tolist()))
    if len(dates) < 2:
        raise ValueError("날짜 보류 진단에는 두 날짜 이상이 필요합니다.")
    n = len(dataset["observations"])
    methods = {"M4": score(dataset, dataset["m4"], cfg),
               f"{cfg['base_policy']['name']}(본 에피소드 행동)": score(dataset, dataset["executed"], cfg)}
    for v in pi.VARIANTS:
        methods[f"교사 {v}(앞 절반, 특권)"] = score(dataset, teacher_greens(dataset, dataset["masks"][v]), cfg)
    predictions = {(t, v): np.zeros(n, np.int64) for t in cfg["offline"]["targets"] for v in pi.VARIANTS}
    folds = []
    for date in dates:
        train = np.flatnonzero(dataset["dates"] != date)
        held = np.flatnonzero(dataset["dates"] == date)
        for target in cfg["offline"]["targets"]:
            net, stats = train_student(dataset, cfg, 42, indices=train, target=target, gains_key="fit_gains",
                                       se_key="fit_gain_se")
            for v in pi.VARIANTS:
                predictions[(target, v)][held] = predict(net, dataset["observations"][held], dataset["masks"][v][held])
            folds.append({"heldout_date": date, "target": target, "train_points": len(train), "heldout_points": len(held),
                          **{k: stats[k] for k in ("updates", "seconds", "initial_full_loss", "final_full_loss")}})
            print(f"날짜 보류 진단 {target}: {date} ({len(train)}학습/{len(held)}보류)", flush=True)
    for (target, v), greens in predictions.items():
        methods[f"학생 {v}({target})"] = score(dataset, greens, cfg)
    executed = dataset["executed"]
    improvement = {}
    for (target, v), greens in predictions.items():
        rows = np.arange(n)
        diff = (dataset["score_gains"][rows, greens] - dataset["score_gains"][rows, executed]).astype(np.float64) * SCALE
        per_date = [float(diff[dataset["dates"] == d].mean()) for d in dates]
        mean = statistics.fmean(per_date)
        half = T95.get(len(per_date), 1.96) * statistics.stdev(per_date) / math.sqrt(len(per_date))
        improvement[f"학생 {v}({target})"] = {"mean": mean, "t95": [mean - half, mean + half],
                                              "positive_dates": sum(x > 0 for x in per_date)}
    report = {"kind": "arena3-pi-offline-v1", "round": cfg["round"], "dataset_fingerprint": dataset["fingerprint"],
              "status": "advisory_only", "methods": methods, "improvement_over_base_all_points": improvement, "folds": folds,
              "units": "vehicle-seconds of 300s surrogate cost per decision point, relative to the M4 green, base policy continues",
              "per_hour_rough": "sum over choice points + sampled agree points ÷ fraction, per episode; overlapping horizons, ranking only",
              "student_input": "observation119 only", "future_split": "fit first half, score second half"}
    _write(out / "offline_report.json", report)
    lines = [f"# ARENA3 정책 반복 {cfg['round']} — 날짜 보류 진단(참고)", "",
             "300초 대리 비용 차이(차량·초/지점, M4 녹색 대비, 뒤는 지금 정책). 초/대나 한 시간 성능이 아닙니다.", "",
             "| 방식 | 갈림 지점 [t95] | 합의 지점 [t95] | 시간당 합(어림) | 바꿈 갈림/합의 | 손해 바꿈 |", "|---|---|---|---|---|---|"]

    def cell(r):
        if r["mean"] is None:
            return "-"
        low, high = r["t95"]
        return f"{r['mean']:+.1f} [{low:+.1f}, {high:+.1f}]" if low is not None else f"{r['mean']:+.1f}"
    for name, r in methods.items():
        lines.append(f"| {name} | {cell(r['choice'])} | {cell(r['agree'])} | {r['per_hour_rough']:+.0f} | "
                     f"{r['choice']['switched']}/{r['agree']['switched']} | {r['all']['harmful']} |")
    lines += ["", "본 에피소드 행동 대비 개선(모든 지점, 날짜 평균 [t95]):", ""]
    for name, r in improvement.items():
        lines.append(f"- {name}: {r['mean']:+.1f} [{r['t95'][0]:+.1f}, {r['t95'][1]:+.1f}], 양수 날짜 {r['positive_dates']}/{len(dates)}")
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
    _write(tmp / "manifest.json", {"kind": KIND, **meta, "model_sha256": sha})
    if folder.exists():
        shutil.rmtree(folder)
    os.replace(tmp, folder)
    return sha


def student_identity(folder: Path) -> str:
    """모델 파일 해시와 기록만 본다(코드 해시는 기록으로만 남겨 다음 라운드에서도 지금 정책으로 읽을 수 있게)."""
    folder = Path(folder)
    manifest = w.read(folder / "manifest.json")
    if manifest.get("kind") != KIND:
        raise ValueError(f"정책 반복 학생 모델이 아닙니다: {folder}")
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
    net = pi.GreenNet(tuple(manifest["network"]))
    net.load_state_dict(payload["state"])
    net.eval()
    return net


def student_dirs(root: Path, cfg: dict) -> list[Path]:
    return [Path(root) / f"seed_{s}" for s in cfg["student"]["seeds"]]


def train_students(data: Path, out: Path, cfg: dict) -> dict:
    torch.set_num_threads(1)
    dataset = load_dataset(data, cfg)
    out = Path(out)
    _write(out / "dataset_manifest.json", {"fingerprint": dataset["fingerprint"], **dataset["manifest"]})
    summary = {}
    for seed, folder in zip(cfg["student"]["seeds"], student_dirs(out, cfg)):
        if (folder / "manifest.json").exists():
            m = w.read(folder / "manifest.json")
            if m.get("dataset_fingerprint") != dataset["fingerprint"] or m.get("seed") != seed:
                raise ValueError(f"이미 있는 학생이 다른 자료·난수로 학습됐습니다: {folder}")
            summary[seed] = {"model_sha256": student_identity(folder), "reused": True}
            continue
        net, stats = train_student(dataset, cfg, seed)
        meta = {"round": cfg["round"], "seed": int(seed), "stage": "final", "dataset_fingerprint": dataset["fingerprint"],
                "network": list(cfg["student"]["network"]), "target": cfg["student"]["target"],
                "observation_size": pi.OBS, "config_sha256": w.file_hash(pi.config_path(cfg["round"])),
                "source_sha256": {p: w.file_hash(ROOT / p) for p in eval_sources(cfg["round"])}, "training": stats}
        summary[seed] = {"model_sha256": save_student(folder, net, meta), "reused": False, **stats}
        print(f"학생 난수 {seed}: 지점 {stats['points']}, 갱신 {stats['updates']}, 손실 {stats['initial_full_loss']:.4f} → "
              f"{stats['final_full_loss']:.4f} ({stats['seconds']:.0f}초)", flush=True)
    _write(out / "students.json", {"dataset_fingerprint": dataset["fingerprint"], "students": summary,
                                   "updated_unix": time.time()})
    return summary


# ----------------------------------------------------------------------------- 평가
def declaration(name: str, folders: list[Path], part: str, cfg: dict) -> dict:
    names = variant_names(cfg)
    if name not in names or len(folders) != len(cfg["student"]["seeds"]):
        raise ValueError(f"정책 반복 학습기는 {sorted(names)} 중 하나의 세 모델이어야 합니다.")
    identities, seeds, datasets = [], [], []
    for folder in folders:
        identities.append(student_identity(Path(folder)))
        m = w.read(Path(folder) / "manifest.json")
        if m.get("round") != cfg["round"] or m.get("stage") != "final" or m.get("target") != cfg["student"]["target"] \
                or m.get("network") != list(cfg["student"]["network"]):
            raise ValueError(f"학생 모델의 라운드·단계·목표·신경망이 설정과 다릅니다: {folder}")
        seeds.append(m.get("seed"))
        datasets.append(m.get("dataset_fingerprint"))
    if seeds != list(cfg["student"]["seeds"]):
        raise ValueError("평가 모델의 순서는 난수 42·43·44여야 합니다.")
    if len(set(datasets)) != 1 or not datasets[0]:
        raise ValueError("세 모델이 같은 자료로 학습되지 않았습니다.")
    return {"name": name, "variant": names[name], "part": part, "model_dirs": [str(Path(p).resolve()) for p in folders],
            "model_identities": identities, "seeds": seeds, "dataset_fingerprint": datasets[0]}


def _prepare_map_cache() -> None:
    """작업자 여러 개가 동시에 지도 캐시를 처음 만들다 부딪히지 않게(218절) 부모 프로세스에서 한 번 만든다."""
    from focused_eval import CACHE, dqn_template_network
    dqn_template_network(CACHE)


def eval_worker(job: dict) -> dict:
    started = time.monotonic()
    work = Path(job["work"])
    try:
        if platform.system() == "Darwin":
            raise RuntimeError("실제 SUMO 평가는 Colab에서만 실행합니다. 맥에서는 judge 명령만 사용하세요.")
        w.configure()
        torch.set_num_threads(1)
        shutil.rmtree(work, ignore_errors=True)
        if job["kind"] != "pi":
            raise ValueError(f"알 수 없는 작업 종류: {job['kind']}")
        if student_identity(Path(job["model_dir"])) != job["model_identity"]:
            raise ValueError("작업 시작 후 모델이 바뀌었습니다.")
        policy = pi.GreenPolicy(load_student(Path(job["model_dir"])), job["variant"], name=job["policy"])
        result = ac.run_arena_episode(job["situation"], work, policy, max_seconds=1800.0)
        expected = job["situation"].get("route_sha256")
        if expected and result.metrics["route_sha256"] != expected:
            raise ValueError("평가 차량 파일의 SHA가 잠금과 다릅니다.")
        ac.write_run(Path(job["target"]), result,
                     {"policy": job["policy"], "fingerprint": job["fingerprint"], "model_identity": job.get("model_identity"),
                      "variant": job["variant"], "job_seconds": time.monotonic() - started})
        return {"ok": True, "seconds": time.monotonic() - started, "name": f"{job['policy']}/{job['sid']}"}
    except BaseException as error:
        return {"ok": False, "seconds": time.monotonic() - started, "name": f"{job['policy']}/{job['sid']}",
                "error": f"{type(error).__name__}: {error}"}
    finally:
        shutil.rmtree(work, ignore_errors=True)


def evaluate(name: str, folders: list[Path], out: Path, work: Path, workers: int, cfg: dict, part: str = "eval", *,
             executor=None, worker=eval_worker) -> dict:
    situations = s2e.situations_for(part)
    out = Path(out)
    fp = ref.register(out, extra_files=eval_sources(cfg["round"]))
    decl = declaration(name, [Path(p) for p in folders], part, cfg)
    s2e._freeze(out / "models.json", decl)
    jobs = []
    for i, (folder, identity) in enumerate(zip(folders, decl["model_identities"])):
        for job in ref.jobs_for(out, Path(work), fp, f"{name}_m{i}", situations, model_dir=folder,
                                model_identity=identity, kind="pi"):
            jobs.append({**job, "variant": decl["variant"]})
    if worker is eval_worker and executor is None:
        _prepare_map_cache()
    status = ref.run_jobs(jobs, max(1, workers), ref.Ledger(out, name), worker=worker, executor=executor)
    _write(out / "status" / f"{name}_{part}.json", {**status, "total": len(jobs), "part": part, "updated_unix": time.time()})
    return status


# ----------------------------------------------------------------------------- 판정
def paired_difference(a: dict, b: dict, t975: float) -> dict:
    """같은 상황·같은 차량에서 날짜별 M4 대비 이득(판정과 같은 차량 가중 정의)의 차 a − b."""
    dates = sorted(a["per_date_gain_s"])
    if dates != sorted(b["per_date_gain_s"]):
        raise ValueError("짝 비교할 두 후보의 날짜가 다릅니다.")
    diffs = {d: float(a["per_date_gain_s"][d]) - float(b["per_date_gain_s"][d]) for d in dates}
    mean, low, high = ref.t_interval(list(diffs.values()), t975)
    return {"mean_s": mean, "t95_s": [low, high], "positive_dates": sum(v > 0 for v in diffs.values()),
            "dates": len(dates), "per_date_s": diffs}


def _pi_runs(spec: str, situations: list[dict], part: str, base_fp: str, cfg: dict):
    name, raw = spec.split("=", 1)
    folder = Path(raw)
    if name not in variant_names(cfg):
        raise ValueError(f"정책 반복 후보 이름은 {sorted(variant_names(cfg))} 중 하나여야 합니다: {name}")
    if not (folder / "models.json").exists():
        return name, None, None
    decl = w.read(folder / "models.json")
    if decl.get("name") != name or decl.get("part") != part:
        raise ValueError("모델 선언의 이름·배치가 요청과 다릅니다.")
    if decl != declaration(name, [Path(p) for p in decl["model_dirs"]], part, cfg):
        raise ValueError("평가에 사용한 학생 모델이 바뀌었습니다.")
    if s2e.folder_record(folder)["evaluation_fingerprint"] != base_fp:
        raise ValueError(f"{name} 평가 지문이 기준선과 다릅니다.")
    runs = [s2e.verified_runs(folder, f"{name}_m{i}", situations, model_identity=x, complete=False)
            for i, x in enumerate(decl["model_identities"])]
    return name, decl, runs


def collect_report(baseline: Path, pi_specs: list[str], reference_specs: list[str], cfg: dict, part: str = "eval",
                   pairs: list | None = None) -> dict:
    lock = ac.config()
    situations = s2e.situations_for(part)
    base_record = s2e.verify_baseline(baseline)
    m4 = s2e.verified_runs(baseline, "M4", situations)
    p0 = s2e.verified_runs(baseline, "P0", situations)
    for policy, rows in (("M4", m4), ("P0", p0)):
        problems = {m["situation"]: ref.quality_errors(m) for m in rows if ref.quality_errors(m)}
        if problems:
            raise ValueError(f"기준선 {policy}의 품질 문제가 남아 있습니다: {problems}")
    results, pending, models = {}, {}, {}
    for spec in pi_specs:
        name, decl, runs = _pi_runs(spec, situations, part, base_record["evaluation_fingerprint"], cfg)
        if decl is None:
            pending[name] = "아직 평가 결과 없음"
        elif any(len(r) != len(situations) for r in runs):
            pending[name] = {"counts": [len(r) for r in runs], "expected": len(situations)}
        else:
            results[name] = ref.judge(runs, m4, p0, lock, situations)
            models[name] = decl
    for spec in reference_specs:                     # S2A_pre·S2B·BPRIME 등 이미 끝난 후보(같은 심판)
        name, folder, runs = s2e.candidate_models(spec, situations, part)
        want = 1 if name in ("W4", "BPRIME") else lock["judge"]["learner_models"]
        if not runs or len(runs) != want or any(len(r) != len(situations) for r in runs):
            pending[name] = {"counts": [len(r) for r in runs] if runs else [], "expected": len(situations)}
            continue
        if s2e.folder_record(folder)["evaluation_fingerprint"] != base_record["evaluation_fingerprint"]:
            raise ValueError(f"{name} 평가 지문이 기준선과 다릅니다.")
        results[name] = ref.judge(runs, m4, p0, lock, situations)
    t975 = float(lock["judge"]["t975"])
    pairs = cfg["rules"]["paired_comparisons"] if pairs is None else pairs
    paired = {f"{a}-{b}": paired_difference(results[a], results[b], t975) for a, b in pairs if a in results and b in results}
    return {"results": results, "pending": pending, "models": models, "paired": paired, "part": part,
            "evaluation_fingerprint": base_record["evaluation_fingerprint"],
            "sensitivity_m4_vs_p0": ref.judge([p0], m4, None, lock, situations),
            "lock_sha256": w.file_hash(ac.CONFIG), "pi_config_sha256": w.file_hash(pi.config_path(cfg["round"])),
            "updated_unix": time.time()}


def render(report: dict, title: str, cfg: dict) -> str:
    text = ref.render(report["results"], report["sensitivity_m4_vs_p0"]).replace(
        "# ARENA3 심판 결과", f"# ARENA3 정책 반복 {cfg['round']} — {title}")
    if report["paired"]:
        text += ("\n## 짝 비교(같은 상황, 날짜 7개, 초/대, +면 앞쪽이 짧음)\n\n| 비교 | 평균 | t95 | 앞쪽이 나은 날짜 |\n"
                 "|---|---|---|---|\n")
        for key, p in report["paired"].items():
            text += (f"| {key.replace('-', ' − ')} | {p['mean_s']:+.3f} | [{p['t95_s'][0]:+.3f}, {p['t95_s'][1]:+.3f}] "
                     f"| {p['positive_dates']}/{p['dates']} |\n")
    if report["pending"]:
        text += "\n아직 평가 중: " + json.dumps(report["pending"], ensure_ascii=False) + "\n"
    m, g, a = (cfg["variants"][v]["name"] for v in pi.VARIANTS)
    text += (f"\n{m} = 지금 행동 범위(모드 녹색) 안의 정책 반복, {g} = 갈림 지점에서 허용 녹색 전부, {a} = 합의 지점까지. "
             f"{m} − S2A_pre는 정책 반복의 몫, {g} − {m}은 모드 밖 녹색의 몫, {a} − {g}는 합의 지점에서 고르는 몫이다. "
             "반복 사용 날짜의 개발 비교.\n")
    return text


def save_report(out: Path, report: dict, title: str, cfg: dict) -> None:
    out = Path(out)
    _write(out / "judge.json", report)
    (out / "summary.md").write_text(render(report, title, cfg), encoding="utf-8")


def choose_winner(results: dict, cfg: dict) -> str | None:
    names = variant_names(cfg)
    eligible = [(n, r) for n, r in results.items() if n in names and r.get("status") == "pass"]
    return min(eligible, key=lambda x: (-float(x[1]["gain_date_mean_s"]), x[0]))[0] if eligible else None


def confirm(judge_dir: Path, baseline: Path, out: Path, work: Path, workers: int, cfg: dict, reference_specs=(), *,
            executor=None, worker=eval_worker) -> dict:
    """개발 판정을 통과한 변형 중 날짜 평균이 큰 하나만 C1·C2에서 한 번 잰다. 고른 뒤에는 다시 고르지 않는다."""
    out = Path(out)
    names = variant_names(cfg)
    selection_path = out / "selection.json"
    if selection_path.exists():
        selected = w.read(selection_path)
        if selected["winner"]:
            current = declaration(selected["winner"], [Path(p) for p in selected["model_dirs"]], "eval", cfg)
            if current["model_identities"] != selected["model_identities"]:
                raise ValueError("확인 전에 고정한 모델이 바뀌었습니다.")
    else:
        report = w.read(Path(judge_dir) / "judge.json")
        if any(n in names for n in report["pending"]) or not set(names) <= set(report["results"]):
            raise ValueError("세 변형의 개발 평가가 끝나기 전에 승자를 고를 수 없습니다.")
        winner = choose_winner(report["results"], cfg)
        selected = {"winner": winner, "lock_sha256": w.file_hash(ac.CONFIG),
                    "evaluation_fingerprint": report["evaluation_fingerprint"],
                    "development_report_sha256": w.digest(report), "model_dirs": [], "model_identities": [],
                    "selection_unix": time.time()}
        if winner:
            decl = report["models"][winner]
            current = declaration(winner, [Path(p) for p in decl["model_dirs"]], "eval", cfg)
            if current["model_identities"] != decl["model_identities"]:
                raise ValueError("승자 평가 모델과 지금 모델이 다릅니다.")
            selected.update(model_dirs=decl["model_dirs"], model_identities=decl["model_identities"])
        _write(selection_path, selected)
    if not selected["winner"]:
        result = {"status": "no_passing_variant", "selection": selected}
        _write(out / "confirmation.json", result)
        return result
    s2e.verify_baseline(baseline, selected["evaluation_fingerprint"])
    name = selected["winner"]
    status = evaluate(name, [Path(p) for p in selected["model_dirs"]], out / name, Path(work) / name, workers, cfg,
                      "confirmation", executor=executor, worker=worker)
    if status["failed"]:
        return {"status": "candidate_failed", "failed": status["failed"]}
    pairs = [[name if a == "winner" else a, b] for a, b in cfg["rules"]["confirmation_paired"]]
    report = collect_report(baseline, [f"{name}={out / name}"], list(reference_specs), cfg, part="confirmation", pairs=pairs)
    save_report(out, report, "C1·C2 확인", cfg)
    result = {"status": report["results"][name]["status"], "selection": selected, "result": report["results"][name],
              "paired": report["paired"]}
    _write(out / "confirmation.json", result)
    return result


# ----------------------------------------------------------------------------- 명령(arena3_pi.py가 부른다)
def add_commands(sub) -> None:
    for name in ("offline", "train"):
        p = sub.add_parser(name)
        p.add_argument("--data", type=Path, required=True)
        p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("evaluate")
    p.add_argument("--variant", choices=pi.VARIANTS, required=True)
    p.add_argument("--students", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1))
    p = sub.add_parser("judge")
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--pi", action="append", default=[], help="R1M=<평가 폴더> 등")
    p.add_argument("--reference", action="append", default=[], help="S2A_pre=…, S2B=…, BPRIME=… 등 이미 끝난 후보")
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("confirm")
    p.add_argument("--judge", type=Path, required=True)
    p.add_argument("--baseline", type=Path, required=True, help="C1·C2 기준선(2단계 confirm/baseline)")
    p.add_argument("--reference", action="append", default=[], help="C1·C2에서 이미 잰 후보(S2B=<stage2/confirm/S2B>)")
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
        name = cfg["variants"][args.variant]["name"]
        status = evaluate(name, student_dirs(args.students, cfg), args.out, args.work, args.workers, cfg)
        return 1 if status["failed"] else 0
    if args.cmd == "judge":
        report = collect_report(args.baseline, args.pi, args.reference, cfg)
        save_report(args.out, report, "개발 평가", cfg)
        print(render(report, "개발 평가", cfg))
        return 0 if not any(n in variant_names(cfg) for n in report["pending"]) else 2
    result = confirm(args.judge, args.baseline, args.out, args.work, args.workers, cfg, args.reference)
    print(json.dumps(w.plain(result), ensure_ascii=False, indent=2))
    return 1 if result["status"] == "candidate_failed" else 0
