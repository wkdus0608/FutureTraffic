"""ARENA3 교사 자료 검증·날짜 보류 진단·두 방식의 사전학습(SUMO 실행 없음).

신경망 입력은 관측 119칸뿐이다. 날짜·실제 미래·롤아웃 비용은 입력이 아닌
묶음 구분/감독 라벨에만 쓰인다. 오프라인 비용은 300초 대리 비용이며,
초/차량 또는 60분 운용 성능으로 환산하지 않는다.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import arena3_common as ac
import arena3_g1 as g1

VARIANTS = ("advantage", "policy")
SCALE = 1000.0
DEFAULT_PRETRAIN = {"epochs": 200, "batch_size": 64, "learning_rate": 0.001,
                    "weight_decay": 0.0001, "temperature": 0.25}


def stage2_settings() -> dict:
    path = ROOT / "config/arena3_stage2_v1.json"
    cfg = json.loads(path.read_text()) if path.exists() else {}
    return {**DEFAULT_PRETRAIN, **cfg.get("pretrain", {})}


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def digest(value) -> str:
    return sha(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode())


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{time.time_ns()}.tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def _validate_provenance(folder: Path) -> dict:
    from arena3_referee import SOURCES
    e = json.loads((folder / "experiment.json").read_text())
    if digest(e["evaluation"]) != e["evaluation_fingerprint"]:
        raise ValueError("교사 평가 지문이 내용과 다릅니다.")
    if digest({k: v for k, v in e.items() if k != "fingerprint"}) != e["fingerprint"]:
        raise ValueError("교사 폴더 지문이 내용과 다릅니다.")
    if e["evaluation"]["lock_sha256"] != sha(ac.CONFIG.read_bytes()):
        raise ValueError("교사 자료와 ARENA3 잠금이 다릅니다.")
    if (e["evaluation"].get("kind") != ac.EXPERIMENT or
        e["evaluation"].get("environment_fingerprint") != ac.config()["environment"]["fingerprint"]):
        raise ValueError("교사 실험/환경 지문이 잠금과 다릅니다.")
    if not set(SOURCES).issubset(e["evaluation"]["files"]) or set(e["extra_files"]) != {"src/arena3_bprime.py"}:
        raise ValueError("교사 자료의 평가/수집 소스 목록이 불완전합니다.")
    for rel, expected in {**e["evaluation"]["files"], **e["extra_files"]}.items():
        p = ROOT / rel
        if not p.is_file() or sha(p.read_bytes()) != expected:
            raise ValueError(f"교사 소스/환경 파일이 달라졌습니다: {rel}")
    return e


def _mode_arrays(obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    greens = np.concatenate((obs[84:104].reshape(5, 4), obs[110:114].reshape(1, 4))).argmax(axis=1)
    mask = np.concatenate((obs[104:109], obs[114:115])) > 0.5
    return greens, mask


def _canonical_for_green(green: int, greens: np.ndarray, mask: np.ndarray) -> int:
    options = np.flatnonzero((greens == green) & mask)
    if len(options) != 1:
        raise ValueError("실행 녹색의 대표 모드가 하나가 아닙니다.")
    return int(options[0])


def _gain_arrays(costs: dict[int, list], indices: list[int], greens, mask, m4: int):
    gain = np.zeros(6, np.float32)
    se = np.zeros(6, np.float32)
    for mode in np.flatnonzero(mask):
        diffs = [(float(costs[m4][j]["cost"]) - float(costs[int(greens[mode])][j]["cost"])) / SCALE for j in indices]
        gain[mode] = statistics.fmean(diffs)
        se[mode] = statistics.stdev(diffs) / math.sqrt(len(diffs)) if len(diffs) > 1 else 0.0
    return gain, se


def _risk_choice(gains, se, mask, anchor: int, greens) -> int:
    viable = [int(m) for m in np.flatnonzero(mask) if m != anchor and gains[m] > 0 and gains[m] >= se[m]]
    return max(viable, key=lambda m: (float(gains[m]), -int(greens[m]))) if viable else anchor


def point_to_row(point: dict, date: str, situation: str) -> tuple[dict | None, str | None]:
    """순수 계산. 불일치/표본 부족은 명시적 제외, 잘못된 관측/모드/비용 형식은 실패."""
    if not point.get("restore_check", {}).get("ok"):
        return None, "restore_mismatch"
    obs = np.asarray(point["observation"], np.float32)
    if obs.shape != (119,) or not np.isfinite(obs).all():
        raise ValueError(f"119칸 유한 관측이 아닙니다: {point.get('state')}")
    greens, mask = _mode_arrays(obs)
    if greens.tolist() != point["greens"] or mask.tolist() != point["mask"]:
        raise ValueError("관측과 선택 모드의 녹색/마스크가 다릅니다.")
    if mask.tolist() != ac.canonical_mask6(greens.tolist()):
        raise ValueError("대표 모드 마스크가 아닙니다.")
    if not set(map(int, greens)).issubset(set(point["allowed"])):
        raise ValueError("실행할 수 없는 녹색이 교사 후보에 있습니다.")
    onehots = np.concatenate((obs[84:104].reshape(5, 4), obs[110:114].reshape(1, 4)))
    if not np.all((onehots == 0) | (onehots == 1)) or not np.all(onehots.sum(axis=1) == 1):
        raise ValueError("후보 녹색 표시가 one-hot이 아닙니다.")
    m4 = int(point["m4"])
    if int(greens[4]) != m4:
        raise ValueError("M4 녹색과 관측이 다릅니다.")
    anchor = _canonical_for_green(m4, greens, mask)
    chosen = _canonical_for_green(int(point["chosen"]), greens, mask)
    costs = {int(k): v for k, v in point["rollouts"].items()}
    if set(costs) != set(map(int, greens[mask])) or set(costs) != set(point["candidates"]):
        raise ValueError("실행 후보와 롤아웃 후보가 다릅니다.")
    lengths = {len(v) for v in costs.values()}
    if len(lengths) != 1:
        raise ValueError("후보마다 미래 수가 다릅니다.")
    valid = []
    for j in range(next(iter(lengths))):
        samples = [v[j] for v in costs.values()]
        if any(s is None or s.get("collisions_unexplained", 0) or s.get("collisions_at_204820", 0) for s in samples):
            continue
        if any(not math.isfinite(float(s["cost"])) for s in samples):
            raise ValueError("롤아웃 비용이 유한하지 않습니다.")
        seeds = {s["future_seed"] for s in samples if "future_seed" in s}
        if len(seeds) > 1:
            raise ValueError("후보끼리 같은 미래를 쓰지 않았습니다.")
        valid.append(j)
    if len(valid) < 4:
        return None, "insufficient_valid_futures"
    cut = len(valid) // 2
    fit_indices, score_indices = valid[:cut], valid[cut:]
    gain, se = _gain_arrays(costs, valid, greens, mask, m4)
    fit_gain, fit_se = _gain_arrays(costs, fit_indices, greens, mask, m4)
    score_gain, _ = _gain_arrays(costs, score_indices, greens, mask, m4)
    expected = _risk_choice(gain, se, mask, anchor, greens)
    if int(greens[expected]) != int(point["chosen"]):
        raise ValueError("교사 선택이 보관된 짝 비용의 선택 규칙과 다릅니다.")
    return {"observations": obs, "mask": mask, "gains": gain, "gain_se": se,
            "fit_gains": fit_gain, "fit_gain_se": fit_se, "score_gains": score_gain,
            "teacher_mode": chosen, "fit_teacher_mode": _risk_choice(fit_gain, fit_se, mask, anchor, greens),
            "anchor_mode": anchor, "w4_mode": _canonical_for_green(int(greens[5]), greens, mask),
            "rule_mode": min(np.flatnonzero(mask), key=lambda m: (float(obs[115 + greens[m]]), m != anchor, int(m))),
            "dates": str(date), "state_ids": str(point["state"]), "situations": situation,
            "valid_futures": len(valid), "fit_future_indices": fit_indices, "score_future_indices": score_indices}, None


def load_dataset(data_root: Path) -> dict:
    """완료된 B 자료만 읽는다. 평가/확인 날짜 폴더는 접근하지 않는다."""
    folder = Path(data_root)
    e = _validate_provenance(folder)
    cfg = ac.config()
    expected = {ac.situation_id(s): s for s in cfg["situations_bprime_data"]}
    runs = folder / "runs/BPRIME"
    present = {p.name for p in runs.iterdir() if p.is_dir()}
    if present != set(expected):
        raise ValueError(f"B 자료 상황이 잠금과 다릅니다: 누락 {sorted(set(expected)-present)}, 추가 {sorted(present-set(expected))}")
    if set(cfg["dates"]["train"]) & set(cfg["dates"]["eval"]):
        raise ValueError("학습과 평가 날짜가 겹칩니다.")

    def read_one(sid):
        p = runs / sid
        metric_raw, point_raw = (p / "metrics.json").read_bytes(), (p / "points.jsonl.gz").read_bytes()
        m, cache = json.loads(metric_raw), json.loads((p / "cache.json").read_text())
        if sha(metric_raw) != cache["metrics_sha256"] or sha(point_raw) != m["points_sha256"]:
            raise ValueError(f"교사 파일 해시 불일치: {sid}")
        if cache.get("fingerprint") != e["fingerprint"] or m.get("fingerprint") != e["fingerprint"]:
            raise ValueError(f"교사 실행 지문 불일치: {sid}")
        if cache.get("model_identity") is not None or m.get("model_identity") is not None:
            raise ValueError(f"B′ 자료에 학습 모델 식별자가 있습니다: {sid}")
        s = expected[sid]
        if (s["date"] not in cfg["dates"]["train"] or m.get("date") != s["date"] or
            m.get("situation") != sid or m.get("route_sha256") != s["route_sha256"] or
            m.get("role") != "data" or m.get("policy") != "BPRIME"):
            raise ValueError(f"학습 상황/차량 잠금 불일치: {sid}")
        dirty = [k for k in ("missing", "unfinished", "changed_by_rules", "mode_action_errors", "p0_mismatch",
                             "collisions_at_204820", "collisions_unexplained") if m.get(k, 0)]
        if dirty or not m.get("cost_check", {}).get("passed"):
            raise ValueError(f"교사 실행 품질 오류 {sid}: {dirty}")
        points = [json.loads(s) for s in gzip.decompress(point_raw).splitlines() if s]
        if len(points) != m["bprime_points"]:
            raise ValueError(f"포인트 수 불일치: {sid}")
        return sid, s, points, {"situation": sid, "metrics_sha256": sha(metric_raw), "points_sha256": sha(point_raw),
                                "points": len(points)}

    rows, files, exclusions, seen = [], [], {}, set()
    with ThreadPoolExecutor(max_workers=8) as pool:
        for sid, s, points, info in pool.map(read_one, sorted(expected)):
            files.append(info)
            for point in points:
                state = point["state"]
                if state in seen:
                    raise ValueError(f"중복 교사 상태: {state}")
                seen.add(state)
                row, reason = point_to_row(point, s["date"], sid)
                if row is None:
                    exclusions[reason] = exclusions.get(reason, 0) + 1
                else:
                    rows.append(row)
    if not rows:
        raise ValueError("사용 가능한 교사 자료가 없습니다.")
    keys = ["observations", "mask", "gains", "gain_se", "fit_gains", "fit_gain_se", "score_gains",
            "teacher_mode", "fit_teacher_mode", "anchor_mode", "w4_mode", "rule_mode", "dates", "state_ids", "situations"]
    dataset = {k: np.asarray([r[k] for r in rows]) for k in keys}
    manifest = {"kind": "arena3-stage2-teacher-data-v1", "source_fingerprint": e["fingerprint"],
                "evaluation_fingerprint": e["evaluation_fingerprint"], "lock_sha256": e["evaluation"]["lock_sha256"],
                "files": files, "raw_points": len(seen), "used_points": len(rows), "excluded": exclusions,
                "dates": sorted(set(dataset["dates"].tolist())), "reward_scale": SCALE,
                "input": "119 current observation values only", "labels": "paired 300-second surrogate cost, privileged",
                "future_split": "first floor(valid/2) fit; remaining score; no overlap",
                "future_partitions": {r["state_ids"]: {"fit": r["fit_future_indices"], "score": r["score_future_indices"]} for r in rows},
                "decisions_sha_rechecked": False}
    dataset["fingerprint"] = digest(manifest)
    dataset["manifest"] = manifest
    return dataset


def tensor_batch(dataset: dict, indices) -> dict:
    ids = np.asarray(indices, dtype=np.int64)
    return {"obs": torch.from_numpy(dataset["observations"][ids].astype(np.float32)),
            "gains": torch.from_numpy(dataset["gains"][ids].astype(np.float32)),
            "gain_se": torch.from_numpy(dataset["gain_se"][ids].astype(np.float32)),
            "mask": torch.from_numpy(dataset["mask"][ids].astype(np.bool_))}


def supervised_loss(net, obs, gains, gain_se, mask, variant: str):
    if variant not in VARIANTS:
        raise ValueError(f"알 수 없는 사전학습 방식: {variant}")
    q = net(obs)
    anchors = g1.anchor_mode(obs)
    relative = q - q.gather(1, anchors[:, None])
    risk_gain = gains - gain_se
    risk_gain = risk_gain.clone()
    risk_gain.scatter_(1, anchors[:, None], 0.0)
    valid = mask.bool().clone()
    valid.scatter_(1, anchors[:, None], False)
    if not bool(valid.any()):
        return q.sum() * 0.0
    if variant == "advantage":
        confidence = (gains.abs() / (gain_se + 0.1)).clamp(0.25, 4.0)
        weights = confidence * valid
        loss = F.smooth_l1_loss(relative, risk_gain, reduction="none")
        return (loss * weights).sum() / weights.sum().clamp_min(1e-8)
    temp = float(stage2_settings()["temperature"])
    target = torch.softmax((risk_gain / temp).masked_fill(~mask.bool(), -1e9), dim=1)
    log_prob = torch.log_softmax((relative / temp).masked_fill(~mask.bool(), -1e9), dim=1)
    impact = (gains.abs() * mask).max(dim=1).values.clamp(0.1, 5.0)
    return (-(target * log_prob).sum(dim=1) * impact).sum() / impact.sum()


def pretrain(net, dataset: dict, variant: str, seed: int, epochs=None, indices=None) -> dict:
    cfg = stage2_settings()
    epochs = int(cfg["epochs"] if epochs is None else epochs)
    ids = np.arange(len(dataset["observations"])) if indices is None else np.asarray(indices, np.int64)
    if epochs < 0 or not len(ids):
        raise ValueError("사전학습 횟수/자료 수가 올바르지 않습니다.")
    rng = np.random.default_rng(int(seed))
    opt = torch.optim.Adam(net.parameters(), lr=float(cfg.get("learning_rate", cfg.get("lr", 0.001))),
                           weight_decay=float(cfg["weight_decay"]))
    started, updates, last = time.monotonic(), 0, None
    complete = tensor_batch(dataset, ids)
    with torch.no_grad():
        initial_loss = float(supervised_loss(net, complete["obs"], complete["gains"], complete["gain_se"],
                                             complete["mask"], variant))
    net.train()
    for _ in range(epochs):
        order = rng.permutation(ids)
        for start in range(0, len(order), int(cfg["batch_size"])):
            d = tensor_batch(dataset, order[start:start + int(cfg["batch_size"])])
            loss = supervised_loss(net, d["obs"], d["gains"], d["gain_se"], d["mask"], variant)
            if not bool(torch.isfinite(loss)):
                raise ValueError("교사 사전학습 손실이 유한하지 않습니다.")
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)
            opt.step()
            updates += 1
            last = float(loss.detach())
    net.eval()
    with torch.no_grad():
        final_loss = float(supervised_loss(net, complete["obs"], complete["gains"], complete["gain_se"],
                                           complete["mask"], variant))
    return {"variant": variant, "seed": int(seed), "epochs": epochs, "points": len(ids), "updates": updates,
            "seconds": time.monotonic() - started, "last_loss": last, "initial_full_loss": initial_loss,
            "final_full_loss": final_loss, "dataset_fingerprint": dataset["fingerprint"],
            "target": "M4-relative paired mean gain minus one standard error; anchor zero",
            "gain_unit": "1000 vehicle-seconds of surrogate cost", "settings": cfg,
            "objective": "confidence-weighted Huber of relative values" if variant == "advantage" else
                         "impact-weighted soft-target cross entropy"}


def predict_modes(net, observations: np.ndarray, mask: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        t = torch.from_numpy(observations.astype(np.float32))
        q = net(t).numpy().astype(np.float64)
        anchors = g1.anchor_mode(t).numpy()
    q[~mask] = -np.inf
    q[np.arange(len(q)), anchors] += 1e-6
    return q.argmax(axis=1)


def _score(dataset, modes) -> dict:
    modes = np.asarray(modes, np.int64)
    gains = dataset["score_gains"][np.arange(len(modes)), modes] * SCALE
    switched = modes != dataset["anchor_mode"]
    dates = sorted(set(dataset["dates"].tolist()))
    per_date = {d: float(gains[dataset["dates"] == d].mean()) for d in dates}
    values = list(per_date.values())
    # Student coefficients for the actual number of date groups; eight in the locked experiment.
    t95 = {2: 12.706204736, 3: 4.302652730, 4: 3.182446305, 5: 2.776445105,
           6: 2.570581836, 7: 2.446911851, 8: 2.364624252}.get(len(values), 1.96)
    mean = statistics.fmean(values)
    margin = t95 * statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else None
    return {"date_mean_gain_vehicle_seconds_per_decision": mean,
            "date_t95": [mean - margin, mean + margin] if margin is not None else [None, None],
            "per_date_gain": per_date, "positive_dates": sum(x > 0 for x in values),
            "points": len(modes), "switched": int(switched.sum()), "switch_fraction": float(switched.mean()),
            "harmful_switches_on_score_futures": int((switched & (gains < 0)).sum()),
            "mean_harmful_switch_cost_vehicle_seconds": float(-gains[switched & (gains < 0)].mean()) if np.any(switched & (gains < 0)) else 0.0,
            "teacher_choice_agreement_descriptive": float((modes == dataset["teacher_mode"]).mean())}


def offline_report(dataset: dict, out: Path) -> dict:
    """Outer leave-one-date-out; no heldout costs determine a student's action."""
    torch.set_num_threads(1)
    out = Path(out)
    write_json(out / "dataset_manifest.json", {"fingerprint": dataset["fingerprint"], **dataset["manifest"]})
    dates = sorted(set(dataset["dates"].tolist()))
    if len(dates) < 2:
        raise ValueError("날짜 보류 진단에는 두 날짜 이상이 필요합니다.")
    cvdata = {**dataset, "gains": dataset["fit_gains"], "gain_se": dataset["fit_gain_se"]}
    reports = {"M4": _score(dataset, dataset["anchor_mode"]), "W4": _score(dataset, dataset["w4_mode"]),
               "M4_cost_rule": _score(dataset, dataset["rule_mode"]),
               "teacher_fit_futures_privileged_diagnostic": _score(dataset, dataset["fit_teacher_mode"])}
    folds = []
    predictions = {v: np.zeros(len(dataset["observations"]), np.int64) for v in VARIANTS}
    for date in dates:
        train = np.flatnonzero(dataset["dates"] != date)
        held = np.flatnonzero(dataset["dates"] == date)
        for variant in VARIANTS:
            torch.manual_seed(42)
            net = g1.DuelingNet()
            stats = pretrain(net, cvdata, variant, seed=42, indices=train)
            predictions[variant][held] = predict_modes(net, dataset["observations"][held], dataset["mask"][held])
            folds.append({"heldout_date": date, "variant": variant, "train_dates": sorted(set(dataset["dates"][train].tolist())),
                          "train_points": len(train), "heldout_points": len(held), **stats})
            print(f"날짜 보류 진단 {variant}: {date} ({len(train)}학습/{len(held)}보류)", flush=True)
    for variant in VARIANTS:
        reports[variant] = _score(dataset, predictions[variant])
    report = {"kind": "arena3-stage2-offline-v1", "dataset_fingerprint": dataset["fingerprint"],
              "status": "advisory_only", "blocks_rl": False, "methods": reports, "folds": folds,
              "units": "vehicle-seconds of 300s surrogate cost per decision, not delay seconds per vehicle",
              "aggregation": "equal weight across train dates; mean of decisions within each date",
              "future_split": "fit first half, independently score second half; no overlapping future index",
              "student_input": "observation119 only; no date, state id, future or rollout result",
              "interpretation": "offline diagnostic only; failure does not prove RL impossible; full 60-minute control must be evaluated"}
    write_json(out / "offline_report.json", report)
    lines = ["# ARENA3 2단계 관측 학습 가능성 진단", "", "날짜를 통째로 보류해 예측했습니다. 아래는 300초 대리 비용 차이이며 초/차량이나 한 시간 성능이 아닙니다.", "",
             "| 방식 | 날짜 평균 이득(차량·초/선택) | 날짜 t95 | 변경/전체 | 손해 난 변경 |", "|---|---:|---|---:|---:|"]
    for name, r in reports.items():
        ci = r["date_t95"]
        interval = f"[{ci[0]:+.1f}, {ci[1]:+.1f}]" if ci[0] is not None else "-"
        lines.append(f"| {name} | {r['date_mean_gain_vehicle_seconds_per_decision']:+.1f} | {interval} | {r['switched']}/{r['points']} | {r['harmful_switches_on_score_futures']} |")
    lines += ["", "teacher_fit_futures는 상세 상태와 표본 미래를 이용하는 진단용 교사입니다. 학생과 같은 정보의 기준선이 아닙니다.",
              "이 검사는 참고 진단이며 결과가 나쁘다는 이유만으로 RL 실행을 자동 중단하지 않습니다.",
              "두 방식의 본 사전학습은 깨끗한 자료의 전체 유효 미래를 사용하고, 이 표의 보류 미래로 모델이나 문턱을 고르지 않습니다."]
    (out / "offline_report.md").write_text("\n".join(lines) + "\n")
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "offline":
        argv.pop(0)
    args = parser.parse_args(argv)
    dataset = load_dataset(args.data)
    offline_report(dataset, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
