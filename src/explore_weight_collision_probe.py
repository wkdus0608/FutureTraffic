"""Colab-only, unchanged-physics replay of WEIGHT1_LONG's pre-control collision.

Only read-only TraCI getters are added around the original warmup step. No FCD
device, controller action, exception rule, route, or network is changed. Actual
simulation refuses macOS; the pure geometry/report helpers are testable there.
The caller owns calculation-budget reservations and charges returned seconds.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
import uuid
from unittest.mock import patch
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
CASE = {"date": "20220906", "period": "1500", "seed": 338,
        "training_seed": 44, "episode": 295,
        "time": 1584.0, "lane": ":SN20220920232934C_2_2",
        "collider": "1445_t1297", "victim": "1445_t1275"}
ORIGINAL_ATTEMPT = ("training/seed_44/episode_000295/"
                    "attempt_c1adbb88f336439fa9c9616b2375c833")
FCD_REFERENCE = "https://sumo.dlr.de/docs/Simulation/Output/FCDOutput.html"


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp_" + uuid.uuid4().hex)
    temporary.write_text(json.dumps(data, ensure_ascii=False, sort_keys=True,
                                    indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def _digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     allow_nan=False).encode()).hexdigest()


def _runtime_guard():
    if sys.platform == "darwin":
        raise RuntimeError("충돌 재현은 Colab에서만 실행합니다. 맥 SUMO는 시작하지 않습니다.")


def _separate(paths):
    """Avoid writes inside either read-only source or the stopped study."""
    source, study, output, work = [Path(p).resolve() for p in paths]
    for written in (output, work):
        for protected in (source, study):
            if written == protected or protected in written.parents or written in protected.parents:
                raise ValueError("진단 출력/작업 폴더는 원본 실험과 별도여야 합니다")
    if output == work or output in work.parents or work in output.parents:
        raise ValueError("임시 작업과 영속 진단 출력도 별도 폴더여야 합니다")
    if " " in str(work):
        raise ValueError("SUMO 작업 경로에는 공백을 사용할 수 없습니다")
    return source, study, output, work


def _verify_inputs(source, study):
    """Validate frozen inputs without registering or modifying either study."""
    from explore_weight_long_train import verify_parent

    experiment = _read(study / "experiment.json")
    frozen = {k: v for k, v in experiment.items() if k != "fingerprint"}
    if _digest(frozen) != experiment["fingerprint"]:
        raise ValueError("중단 실험의 등록 지문이 다릅니다")
    verify_parent(source, experiment["parent_fingerprint"])
    for name, expected in experiment["code"].items():
        if _sha(ROOT / name) != expected:
            raise ValueError(f"중단 실험의 고정 코드를 바꿀 수 없습니다: {name}")
    attempt = study / ORIGINAL_ATTEMPT
    evidence = _read(attempt / "evidence.json")
    if evidence["study_fingerprint"] != experiment["fingerprint"]:
        raise ValueError("충돌 근거의 실험 지문이 다릅니다")
    for name in ("metrics.json", "collisions.xml.gz"):
        if _sha(attempt / name) != evidence["files"][name]:
            raise ValueError(f"원래 충돌 근거 해시가 다릅니다: {name}")
    metrics = _read(attempt / "metrics.json")
    for key in ("date", "period", "seed", "train_seed"):
        expected = CASE["training_seed"] if key == "train_seed" else CASE[key]
        if metrics[key] != expected:
            raise ValueError(f"원래 충돌 시나리오가 다릅니다: {key}")
    original = metrics["collision_details"]["collisions_unexplained"]
    if len(original) != 1 or not event_matches(original[0], CASE):
        raise ValueError("예상한 중단 충돌 1건과 일치하지 않습니다")
    with gzip.open(attempt / "collisions.xml.gz", "rb") as stream:
        xml = stream.read()
    if hashlib.sha256(xml).hexdigest() != evidence["source_sha256"]["collisions.xml"]:
        raise ValueError("원래 충돌 XML의 압축 해제 해시가 다릅니다")
    raw_events = [dict(e.attrib) for e in ET.fromstring(xml).iter("collision")]
    if len(raw_events) != 1 or not event_matches(raw_events[0], original[0]):
        raise ValueError("원래 충돌 XML과 집계가 다릅니다")
    return experiment, metrics, raw_events


def event_matches(event, expected, *, tolerance=1e-8):
    """Compare every supplied collision field; header paths are irrelevant."""
    for key, value in expected.items():
        if key in ("date", "period", "seed", "training_seed", "episode"):
            continue
        if key not in event:
            return False
        if key in ("time", "pos", "colliderSpeed", "victimSpeed"):
            try:
                left, right = float(event[key]), float(value)
                if not math.isfinite(left) or not math.isfinite(right) or abs(left-right) > tolerance:
                    return False
            except (TypeError, ValueError):
                return False
        elif key in ("colliderFront", "colliderBack", "victimFront", "victimBack"):
            try:
                left, right = str(event[key]).split(","), str(value).split(",")
                if len(left) != 2 or len(right) != 2 or any(
                        not math.isfinite(float(a)) or not math.isfinite(float(b)) or abs(float(a)-float(b)) > tolerance
                        for a, b in zip(left, right)):
                    return False
            except (TypeError, ValueError):
                return False
        elif str(event[key]) != str(value):
            return False
    return True


def rectangle(front, back, width):
    """Straight SUMO front/back-centre rectangle, not a curved swept body."""
    fx, fy = map(float, front)
    bx, by = map(float, back)
    length = math.hypot(fx-bx, fy-by)
    if length <= 0 or not math.isfinite(width) or width <= 0:
        raise ValueError("차량 직사각형의 길이와 너비가 양수여야 합니다")
    nx, ny = -(fy-by)/length*width/2, (fx-bx)/length*width/2
    return [[fx+nx, fy+ny], [fx-nx, fy-ny], [bx-nx, by-ny], [bx+nx, by+ny]]


def rectangle_overlap(left, right):
    """Separating-axis check and minimum projection overlap in metres."""
    if len(left) != 4 or len(right) != 4:
        raise ValueError("직사각형 꼭짓점 4개가 필요합니다")
    overlaps = []
    for polygon in (left, right):
        for i, (x, y) in enumerate(polygon):
            nx, ny = polygon[(i+1) % len(polygon)]
            dx, dy = nx-x, ny-y
            norm = math.hypot(dx, dy)
            if norm <= 0:
                raise ValueError("길이 0인 직사각형 변은 허용하지 않습니다")
            ax, ay = -dy/norm, dx/norm
            a = [px*ax+py*ay for px, py in left]
            b = [px*ax+py*ay for px, py in right]
            overlaps.append(min(max(a), max(b)) - max(min(a), min(b)))
    minimum = min(overlaps)
    return {"overlap": minimum > 1e-9, "minimum_projection_overlap_m": minimum}


def network_context(path, lane_id):
    """Original shapes and right-of-way data around this exact internal lane."""
    net = ET.parse(path).getroot()
    junction = next((j for j in net.findall("junction") if lane_id in j.get("intLanes", "").split()), None)
    if junction is None:
        raise ValueError("충돌 내부 차로를 포함하는 분기점을 찾지 못했습니다")
    internals = set(junction.get("intLanes", "").split())
    connections = [dict(c.attrib) for c in net.findall("connection")
                   if c.get("via") in internals or c.get("from", "").startswith(":" + junction.get("id") + "_")]
    external_edges = {c[k] for c in connections for k in ("from", "to") if k in c}
    edges = [{**dict(e.attrib), "lanes": [dict(l.attrib) for l in e.findall("lane")]}
             for e in net.findall("edge") if e.get("id") in external_edges
             or any(l.get("id") in internals for l in e.findall("lane"))]
    return {"junction": dict(junction.attrib),
            "requests": [dict(r.attrib) for r in junction.findall("request")],
            "connections": connections, "edges": edges,
            "interpretation": "원본 지도와 양보 관계의 관측값이며 지도 오류 확정이나 예외 승격이 아니다"}


def analyse_trajectories(path, event, *, before_s=30, after_s=30):
    """Keep both full paths plus a readable, bounded collision-time window."""
    at = float(event["time"])
    targets = (event["collider"], event["victim"])
    window = []
    before = {v: [] for v in targets}
    exact = None
    metadata = {}
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            time_s = float(row["time"])
            for vid, vehicle in row["vehicles"].items():
                if vid in targets:
                    metadata[vid] = vehicle["geometry"]
                    if at-before_s <= time_s < at:
                        before[vid].append((time_s, vehicle))
            if at-before_s <= time_s <= at+after_s:
                window.append(row)
            if abs(time_s-at) <= 1e-9:
                exact = row
    stopped_before = {}
    for vid in targets:
        stopped = 0
        later = at
        for when, vehicle in reversed(before[vid]):
            if abs(later-when-1) > 1e-9 or vehicle["speed"] > .1:
                break
            stopped += 1
            later = when
        stopped_before[vid] = stopped
    overlap = None
    if all(v in metadata for v in targets):
        points = lambda key: [float(x) for x in event[key].split(",")]
        left = rectangle(points("colliderFront"), points("colliderBack"), metadata[targets[0]]["width"])
        right = rectangle(points("victimFront"), points("victimBack"), metadata[targets[1]]["width"])
        overlap = {"collider_polygon": left, "victim_polygon": right, **rectangle_overlap(left, right)}
    return {"window_s": [at-before_s, at+after_s], "window": window, "at_collision": exact,
            "vehicle_metadata": metadata, "consecutive_stopped_seconds_before": stopped_before,
            "straight_rectangle_approximation": overlap,
            "interpretation": "직사각형 겹침은 SUMO 접촉 기록의 기하학 설명이다. 실제 도로 안전 판정이 아니다."}


def _capture_step(environment, stream):
    """Wrap only getters around the original physical simulation step."""
    original = environment._sumo_step
    geometry = {}
    targets = (CASE["collider"], CASE["victim"])

    def capture():
        result = original()
        vehicle = environment.sumo.vehicle
        ids = set(vehicle.getIDList())
        selected = [vid for vid in targets if vid in ids]
        leaders = {}
        for vid in list(selected):
            leader = vehicle.getLeader(vid, 100)
            leaders[vid] = list(leader) if leader else None
            if leader and leader[0] in ids and leader[0] not in selected:
                selected.append(leader[0])
        states = {}
        for vid in selected:
            if vid not in geometry:
                geometry[vid] = {"type": vehicle.getTypeID(vid), "length": float(vehicle.getLength(vid)),
                                 "width": float(vehicle.getWidth(vid)), "route": list(vehicle.getRoute(vid))}
            x, y = vehicle.getPosition(vid)
            states[vid] = {"x": float(x), "y": float(y), "angle": float(vehicle.getAngle(vid)),
                           "speed": float(vehicle.getSpeed(vid)), "lane": vehicle.getLaneID(vid),
                           "lane_position": float(vehicle.getLanePosition(vid)), "road": vehicle.getRoadID(vid),
                           "leader": leaders.get(vid), "geometry": geometry[vid]}
        stream.write(json.dumps({"time": float(environment.sim_step), "vehicles": states},
                                ensure_ascii=False, allow_nan=False) + "\n")
        return result

    return capture


def _run_warmup(work, cap_s):
    """Use original _Episode/reset; never call step/apply after handover."""
    import focused_sumo
    from explore_weight_episode import _Episode

    work = Path(work)
    trajectory = work / "trajectories.jsonl"
    original_factory = focused_sumo.make_environment
    seen = {}
    began = time.monotonic()
    ep = None
    with trajectory.open("w", encoding="utf-8") as stream:
        def recorded_environment(**kwargs):
            seen["net_file"] = str(Path(kwargs["net_file"]).resolve())
            seen["original_environment_arguments"] = {
                k: str(v) if isinstance(v, Path) else v for k, v in kwargs.items()
                if k not in ("observation_class", "reward_fn")}
            environment = original_factory(**kwargs)
            environment._sumo_step = _capture_step(environment, stream)
            return environment
        try:
            with patch.object(focused_sumo, "make_environment", recorded_environment):
                ep = _Episode(CASE["date"], CASE["period"], CASE["seed"], work,
                              deadline=began + cap_s)
            seen["handover_s"] = ep.handover
            seen["warmup_start"] = ep.start
            seen["route_file"] = str(ep.route)
            seen["collisions_file"] = str(ep.collisions)
        finally:
            if ep is not None:
                ep.close()
    seen["trajectory_file"] = str(trajectory)
    seen["rl_actions_executed"] = 0
    return seen


def _compress(source, target):
    with Path(source).open("rb") as src, gzip.open(target, "wb", compresslevel=6) as dst:
        shutil.copyfileobj(src, dst)


def run_probe(source_root, study_root, output, work, cap_s=600):
    """Replay warmup on Colab, preserving the old study and collision rules.

    The root runner must reserve/charge ``cap_s`` in the shared budget before
    calling. A failed or unmatched replay never grants a training exception.
    """
    _runtime_guard()
    started = time.monotonic()
    source, study, output, work = _separate((source_root, study_root, output, work))
    if not math.isfinite(cap_s) or cap_s <= 30:
        raise ValueError("충돌 재현 시간 한도는30초보다 커야 합니다")
    experiment, original_metrics, original_events = _verify_inputs(source, study)
    identity = {"kind": "WEIGHT1_PRECONTROL_PROBE_V1", "case": CASE,
                "parent_study_fingerprint": experiment["fingerprint"],
                "probe_sha256": _sha(Path(__file__))}
    if (output / "report.json").exists():
        saved = _read(output / "report.json")
        if saved.get("identity") != identity:
            raise ValueError("기존 재현 기록의 입력/코드 계보가 다릅니다")
        for name, digest in saved.get("artifacts", {}).items():
            if Path(name).name != name or _sha(output / name) != digest:
                raise ValueError("기존 재현 근거가 변경됐습니다")
        return {**saved, "reused": True, "seconds": time.monotonic()-started,
                "report_path": str(output / "report.json")}
    if output.exists() and any(output.iterdir()):
        raise ValueError("미완료 진단 폴더를 덮어쓰지 않습니다. 새 시도 폴더를 사용하세요")
    output.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    if any(work.iterdir()):
        raise ValueError("새 진단 작업 폴더가 필요합니다")
    report = {"identity": identity, "status": "preparing", "matches": False,
              "started_unix": time.time(), "artifacts": {},
              "original_failed_metrics_sha256": _sha(study / ORIGINAL_ATTEMPT / "metrics.json"),
              "rule": "원본지도/수요/운영규칙/충돌예외변경없음. RL행동없음. 예외승격또는채택판정아님."}
    _write(output / "report.json", report)
    try:
        remaining = cap_s - (time.monotonic()-started) - 15
        if remaining <= 10:
            raise TimeoutError("검증 후 준비구간 재현 시간이 부족합니다")
        replay = _run_warmup(work, remaining)
        events = [dict(e.attrib) for e in ET.parse(replay["collisions_file"]).getroot().iter("collision")]
        match = [e for e in events if event_matches(e, original_events[0])]
        route_matches = _sha(replay["route_file"]) == original_metrics["route_sha256"]
        handover_matches = abs(replay["handover_s"]-original_metrics["handover"]) <= 1e-9
        report.update({"replay": replay, "events": events, "matched_events": match,
                       "route_sha256": _sha(replay["route_file"]), "route_matches_original": route_matches,
                       "net_sha256": _sha(replay["net_file"]), "handover_matches_original": handover_matches,
                       "collision_before_control": bool(match) and float(match[0]["time"]) < replay["handover_s"],
                       "network": network_context(replay["net_file"], CASE["lane"])})
        report["matches"] = len(events) == len(match) == 1 and route_matches and handover_matches
        if match:
            report["trajectory_analysis"] = analyse_trajectories(replay["trajectory_file"], match[0])
        report["status"] = "reproduced" if report["matches"] else "not_reproduced"
        for name, path in (("trajectories.jsonl.gz", replay["trajectory_file"]),
                           ("collisions.xml.gz", replay["collisions_file"]),
                           ("network.net.xml.gz", replay["net_file"]),
                           ("route.rou.xml.gz", replay["route_file"])):
            _compress(path, output / name)
            report["artifacts"][name] = _sha(output / name)
        _write(output / "network_context.json", report["network"])
        report["artifacts"]["network_context.json"] = _sha(output / "network_context.json")
    except TimeoutError as error:
        report["status"], report["error"] = "time_limit", str(error)
    except Exception as error:
        report["status"], report["error"] = "failed", f"{type(error).__name__}: {error}"
    finally:
        report["seconds"] = time.monotonic()-started
        report["finished_unix"] = time.time()
        report["report_path"] = str(output / "report.json")
        _write(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--study", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--work", required=True)
    parser.add_argument("--cap-seconds", type=float, default=600)
    args = parser.parse_args()
    result = run_probe(args.source, args.study, args.out, args.work, args.cap_seconds)
    print(json.dumps({k: result.get(k) for k in ("status", "matches", "seconds", "report_path")}, ensure_ascii=False))
    if result["status"] not in ("reproduced",):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
