"""[추가 탐색] PI1의 예정 수요 이력 검사와 고정 부모 정책(156절).

SUMO 없이 기존 314개 장면에 최근 900초의 이동 15칸·차종 14칸을 붙인다.
96/125칸을 묶음 제외 6분할로 한 번 비교하고, 미리 정한 기준으로 부모 하나를
고정한다. 이 모듈을 가져오는 것만으로 학습하거나 파일을 만들지 않는다.
"""
from __future__ import annotations

import functools
import gzip
import hashlib
import json
import math
import shutil
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

import explore_combo as cb
import explore_distill as ed
import explore_online as eo
import explore_remeasure as rm
import explore_rollout as er
import explore_student as es

PROJECT = ed.PROJECT
HISTORY_S = 900
HISTORY_WIDTH = 29
BASE_WIDTH, WIDTH = 96, 125
LOG_SCALE = 1000
SAMPLES, GROUPS = 314, 6
STEPS = 10_000
SEEDS = (0, 1, 2, 3, 4)
EXPECTED_ENV = "b20888514ea3"
EXPECTED_COMBO = "77ab8a69f7dd"
OTHER = "__other__"


def code_fingerprint() -> str:
    """이 모듈의 내용 지문(기존 탐색 코드의 지문은 바꾸지 않는다)."""
    return rm.file_hash(Path(__file__))


@functools.lru_cache(maxsize=1)
def schema() -> dict:
    """지도 이동 14개와 원래 차량 정의의 차종 13개를 사전순으로 고정한다.

    두 종류의 마지막 칸은 각각 그 밖이며, 이동과 차종의 합계는 중복 집계다.
    최근 총 대수를 두 종류로 각각 요약하는 것이지 전체 29칸의 합이 대수는 아니다.
    """
    from focused_data import VTYPES
    from focused_rl import MOVEMENT_LINKS
    movements = [list(x) for x in sorted(MOVEMENT_LINKS)]
    types = sorted({v.get("id") for v in ET.parse(VTYPES).getroot().iter("vType")})
    if len(movements) != 14 or len(types) != 13:
        raise SystemExit(f"156절 이력 정의와 다름: 이동 {len(movements)}, 차종 {len(types)}")
    return {"movements": movements, "type_ids": types, "other": OTHER,
            "width": HISTORY_WIDTH, "window_s": HISTORY_S,
            "window": "[t-900,t)", "scale": "min(1,log1p(count)/log1p(1000))"}


def history_features(vehicles: dict, t0: float) -> list[float]:
    """최근 [t0−900,t0)의 예정 출발 대수를 이동·차종별로 센다.

    vehicles는 er.read_routes와 같은 {ID: (예정 출발 시각, 경로, 차종)}다.
    실제 도로 진입 시각 또는 유입 증가·감소 추세를 나타내지는 않는다.
    """
    spec = schema()
    movement_index = {tuple(x): i for i, x in enumerate(spec["movements"])}
    type_index = {x: i for i, x in enumerate(spec["type_ids"])}
    offset = len(movement_index) + 1
    counts = [0] * HISTORY_WIDTH
    for dep, edges, vtype in vehicles.values():
        if t0 - HISTORY_S <= float(dep) < t0:
            movement = next(((a, b) for a, b in zip(edges, edges[1:]) if (a, b) in movement_index), None)
            counts[movement_index.get(movement, len(movement_index))] += 1
            counts[offset + type_index.get(vtype, len(type_index))] += 1
    return [min(1.0, math.log1p(n) / math.log1p(LOG_SCALE)) for n in counts]


def observation_with_history(base96, features) -> list[float]:
    """96칸의 마지막 행동 마스크 4칸 앞에 이력 29칸을 넣는다."""
    if len(base96) != BASE_WIDTH or len(features) != HISTORY_WIDTH:
        raise ValueError(f"관측 폭이 다름: 기존 {len(base96)}, 이력 {len(features)}")
    values = np.asarray([*base96[:-ed.N_GREENS], *features, *base96[-ed.N_GREENS:]], dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("관측에 유한하지 않은 값이 있다")
    return [float(x) for x in values]


def history_vehicles(vehicles: dict, snapshot_t0: float, injected) -> dict:
    """롤아웃 이력용 목록: 원래 미래를 제거하고 실제 주입한 표본 미래만 넣는다.

    저장 시각과 같은 출발은 제거하지 않는다(er.future_ids와 같은 > 기준).
    입력 목록은 수정하지 않는다. injected는 er.sample_future의 4항 튜플
    (ID, 경로, 차종, 예정 시각) 또는 read_routes와 같은 사전이다.
    """
    out = {vid: (float(dep), list(edges), vt) for vid, (dep, edges, vt) in vehicles.items()
           if float(dep) <= snapshot_t0}
    rows = ([(vid, edges, vt, dep) for vid, (dep, edges, vt) in injected.items()]
            if isinstance(injected, dict) else injected)
    for vid, edges, vt, dep in rows:
        if float(dep) <= snapshot_t0:
            raise ValueError("표본 미래의 출발은 저장 시각 이후여야 한다")
        if vid in out:
            raise ValueError(f"표본 차량 ID 중복: {vid}")
        out[vid] = (float(dep), list(edges), vt)
    return out


def adoption(gain96: dict[str, float], gain125: dict[str, float]) -> dict:
    """156절의 한 번뿐인 입력 채택 기준. 같은 묶음 여섯 개만 비교한다."""
    if set(gain96) != set(gain125) or len(gain96) != GROUPS:
        raise ValueError("같은 6개 묶음의 제외 평가가 필요하다")
    if not all(math.isfinite(float(v)) for v in (*gain96.values(), *gain125.values())):
        raise ValueError("평가 이득은 유한해야 한다")
    total96, total125 = sum(gain96.values()), sum(gain125.values())
    positive = sum(gain125[g] > gain96[g] for g in gain96)
    adopted = total125 > 0 and total125 > total96 and positive >= 4
    return {"uses_history": adopted, "width": WIDTH if adopted else BASE_WIDTH,
            "gain96": total96, "gain125": total125, "better_groups": positive,
            "criterion": "125칸 총 이득>0, 96칸보다 큼, 6묶음 중 4개 이상 개선"}


def _write_json(path: Path, value: dict) -> None:
    """중간 쓰기 파일이 완료 결과로 읽히지 않게 교체한다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=1))
    tmp.replace(path)


def guard_cache(path: Path, identity: dict) -> dict | None:
    """캐시가 있으면 코드·원자료·설정이 정확히 같은 경우에만 다시 쓴다."""
    if not path.exists():
        return None
    saved = json.loads(path.read_text())
    if saved.get("identity") != identity:
        raise SystemExit(f"코드·원자료·설정이 달라 기존 캐시를 쓸 수 없다: {path}; 새 출력 폴더 필요")
    return saved


def _source_signature() -> dict:
    """경로 생성에 영향을 주는 원본·설정·코드와 실행 환경을 남긴다."""
    from focused_data import COUNTS, VTYPES, BRANCH_INFLOW, NETWORK, environment_fingerprint
    import focused_train as ft
    env = environment_fingerprint()
    if env != EXPECTED_ENV:
        raise SystemExit(f"환경 {env}가 원래 pet1/240 환경 {EXPECTED_ENV}와 다르다")
    paths = [COUNTS, VTYPES, BRANCH_INFLOW, NETWORK,
             *[PROJECT / "src" / f for f in ("focused_data.py", "focused_scenario.py", "focused_train.py", "focused_rl.py")],
             *[PROJECT / "config" / f for f in ("driver_model.json", "environment_v6.json", "transition_rules.json")]]
    return {"environment": env, "files": {str(p.relative_to(PROJECT)): rm.file_hash(p) for p in paths},
            "warmup_s": ft.WARMUP_S, "control_s": ft.CONTROL_S, "tail_s": eo.TAIL_S}


def _verify_originals(old: Path, new: Path, combo: Path) -> dict:
    """B의 원자료·가중치와 9개 재생성 특징 파일, 비용의 시나리오를 대조한다."""
    rm._guard()
    if cb.code_fingerprint() != EXPECTED_COMBO:
        raise SystemExit("COMBO 코드가 153절과 다르다")
    policy = json.loads((combo / "students" / "b" / "policy.json").read_text())
    required = {"cell": "b", "width": BASE_WIDTH, "method": "l0", "steps": STEPS,
                "samples": SAMPLES, "model_seeds": list(SEEDS), "code": EXPECTED_COMBO,
                "student_code": es.code_fingerprint(), "distill_code": ed.code_fingerprint()}
    if any(policy.get(k) != v for k, v in required.items()):
        raise SystemExit("기준 B 정책 설정·코드가 153절과 다르다")
    data = cb._data_hashes(old, new, combo)
    if policy.get("data") != data:
        raise SystemExit("B가 학습했던 원자료·24칸 해시와 현재 파일이 다르다")
    for seed in SEEDS:
        if rm.file_hash(combo / "students" / "b" / f"seed{seed}.pt") != policy["weights"][f"seed{seed}"]:
            raise SystemExit("기준 B 가중치가 기록과 다르다")
    for runs, episodes in ((old, cb.OLD_RUNS), (new, cb.NEW_RUNS)):
        for date, period, seed in episodes:
            tag = f"{date}_{period}_s{seed}"
            folder = runs / f"{tag}_search"
            done = json.loads((combo / "features" / f"{tag}.done.json").read_text())
            checks = {"code": EXPECTED_COMBO, "online_code": eo.code_fingerprint(),
                      "rollout_code": er.code_fingerprint(), "replay_code": rm.code_fingerprint(),
                      "features": rm.file_hash(combo / "features" / f"{tag}.jsonl"),
                      "decisions": rm.file_hash(folder / "decisions.jsonl")}
            if any(done.get(k) != v for k, v in checks.items()):
                raise SystemExit(f"원래 재실행 확인 파일이 다르다: {tag}")
            metrics = json.loads((folder / "metrics.json").read_text())
            if metrics.get("code") != eo.code_fingerprint() or metrics.get("rollout_code") != er.code_fingerprint():
                raise SystemExit(f"원래 탐색 실행 코드가 다르다: {tag}")
            for line in (folder / "decisions.jsonl").read_text().splitlines():
                row = json.loads(line)
                if not row["searched"]:
                    continue
                expected = eo.scenario_seeds(seed, row["t0"])
                if (row.get("seeds") != expected or set(map(int, row["scores"])) != set(row["allowed"])
                        or any([v["seed"] for v in costs] != expected for costs in row["scores"].values())):
                    raise SystemExit(f"후보·시나리오가 원래 8개와 다르다: {tag} {row['t0']}")
    return {"data": data, "b_policy": rm.file_hash(combo / "students" / "b" / "policy.json"),
            "b_weights": policy["weights"], "source": _source_signature(), "schema": schema(),
            "code": code_fingerprint()}


def _same_routes(generated: Path, original: Path) -> bool:
    """보존된 원래 차량 파일은 XML 배치 대신 차량·경로 내용으로 대조한다."""
    if original.suffix == ".gz":
        root = ET.fromstring(gzip.decompress(original.read_bytes()))
        routes = {r.get("id"): r.get("edges").split() for r in root.iter("route")}
        expected = {v.get("id"): (float(v.get("depart")), routes[v.get("route")], v.get("type"))
                    for v in root.iter("vehicle")}
    else:
        expected, _ = er.read_routes(original)
    actual, _ = er.read_routes(generated)
    return actual == expected


def _regenerate(date: str, period: str, route: Path) -> dict:
    """OnlineEpisode와 같은 두 번의 차량 창 생성. SUMO 인스턴스는 만들지 않는다."""
    import focused_train as ft
    from focused_data import COUNTS, load_counts
    from focused_scenario import period_start_s, window_route_file
    route.parent.mkdir(parents=True, exist_ok=True)
    counts = load_counts(COUNTS)
    start = ft.warm_start(date, period, counts, route)
    origin = period_start_s(period) - ft.WARMUP_S
    end = origin + ft.WARMUP_S + ft.CONTROL_S + eo.TAIL_S
    window_route_file(date, origin, end, route, origin_s=origin, counts=counts)
    return {"origin_s": origin, "end_s": end, "handover": float(start["handover_s"]),
            "hash": rm.file_hash(route), "generator": "warm_start → window_route_file, original route_seed(date,period)"}


def load_samples(old_runs: Path, new_runs: Path, combo_dir: Path, work: Path) -> list[dict]:
    """314개에 obs96/obs125를 붙이고 재생성 경로의 설정·해시·코호트를 보존한다.

    원래 차량 파일 해시가 원자료에 없는 경우 직접 동일하다고 주장하지 않는다.
    동일 코드·난수·창 설정에 의한 재생성 및 기록 대조라는 근거를 manifest에 적는다.
    """
    old, new, combo, work = map(Path, (old_runs, new_runs, combo_dir, work))
    identity = _verify_originals(old, new, combo)
    manifest_path = work / "history_manifest.json"
    manifest = guard_cache(manifest_path, identity)
    samples = cb.load_samples(old, new, combo)
    groups = sorted({s["group"] for s in samples})
    if len(samples) != SAMPLES or len(groups) != GROUPS:
        raise SystemExit("156절의 314개·6묶음 자료가 아니다")
    generated, vehicles_of = {}, {}
    for group in groups:
        date, period = group.split("_")
        route = work / "history_routes" / f"{group}.rou.xml"
        if manifest is not None:
            info = manifest["routes"].get(group)
            if not info or not route.exists() or rm.file_hash(route) != info["hash"]:
                raise SystemExit(f"재생성 차량 파일이 바뀌거나 빠졌다: {group}")
        else:
            info = _regenerate(date, period, route)
        vehicles, _ = er.read_routes(route)
        comparisons = []
        for runs, episodes in ((old, cb.OLD_RUNS), (new, cb.NEW_RUNS)):
            for d, p, seed in episodes:
                if (d, p) != (date, period):
                    continue
                folder = runs / f"{group}_s{seed}_search"
                metrics = json.loads((folder / "metrics.json").read_text())
                cohort = {v for v, (dep, _, _) in vehicles.items()
                          if info["handover"] <= dep < info["handover"] + identity["source"]["control_s"]}
                if info["handover"] != metrics["handover"] or len(cohort) != metrics["vehicles"]:
                    raise SystemExit(f"원래 인계 시각·평가 차량 대수와 다르다: {folder}")
                delays = folder / "vehicle_delays.json"
                if delays.exists() and not cohort <= set(json.loads(delays.read_text())):
                    raise SystemExit(f"원래 차량 기록에 없는 평가 차량이 있다: {folder}")
                # collect/states의 보존 파일은 선택 사항. 다른 실행 경로를 임의로 찾지 않는다.
                candidates = [runs.parent / "states" / folder.name / f"{date}_{period}.rou.xml.gz",
                              folder / f"{date}_{period}.rou.xml"]
                preserved = next((p for p in candidates if p.exists()), None)
                if preserved is not None and not _same_routes(route, preserved):
                    raise SystemExit(f"보존 원래 차량 파일과 재생성 내용이 다르다: {preserved}")
                comparisons.append({"seed": seed, "cohort_size": len(cohort), "cohort_ids_present": delays.exists(),
                                    "preserved_route_compared": preserved is not None})
        generated[group] = {**info, "comparisons": comparisons}
        vehicles_of[group] = vehicles
    current = {"identity": identity, "routes": generated,
               "evidence": "동일 코드·난수·차량 창 재생성; 원자료 인계/차량 수/있을 때 ID 및 보존 경로 대조"}
    if manifest is None:
        _write_json(manifest_path, current)
    elif current != manifest:
        raise SystemExit("보존 경로·코호트 대조 근거가 이전 캐시와 달라졌다")
    return [{**s, "history": np.asarray(history_features(vehicles_of[s["group"]], s["t0"]), dtype=np.float32),
             "obs125": np.asarray(observation_with_history(s["obs96"], history_features(vehicles_of[s["group"]], s["t0"])), dtype=np.float32)}
            for s in samples]


def fit(samples: list[dict], width: int, steps: int = STEPS, seeds=SEEDS) -> list[dict]:
    """같은 허용 후보 상대 비용 회귀. 폭 외에는 COMBO B 설정을 그대로 쓴다."""
    if width not in (BASE_WIDTH, WIDTH):
        raise ValueError(width)
    x = np.stack([s[f"obs{width}"] for s in samples])
    y = np.stack([ed.targets(s) for s in samples])
    return [cb.train_value(x, y, seed, steps) for seed in seeds]


def picks(states: list[dict], samples: list[dict], width: int) -> list[int]:
    """다섯 모델 상대 비용 평균, 허용 후보만, P0 동률 우선·문턱 0."""
    x = np.stack([s[f"obs{width}"] for s in samples])
    predictions = np.mean(cb._outputs(states, x), axis=0)
    return [es.hybrid(pred, sample, 0.0) for pred, sample in zip(predictions, samples)]


def _validate_parent(parent: Path, identity: dict) -> dict:
    """완료된 부모를 다시 쓸 때 메타데이터와 가중치를 모두 확인한다."""
    policy = guard_cache(parent / "policy.json", identity)
    if policy is None:
        raise SystemExit("부모 정책 파일이 없다")
    if policy["model_seeds"] != list(SEEDS) or policy["width"] not in (BASE_WIDTH, WIDTH):
        raise SystemExit("부모 정책 설정이 다르다")
    for seed in SEEDS:
        if rm.file_hash(parent / f"seed{seed}.pt") != policy["weights"][f"seed{seed}"]:
            raise SystemExit("부모 가중치가 고정 시점과 다르다")
    return policy


def diagnostic_and_parent(old_runs: Path, new_runs: Path, combo_dir: Path, out: Path, work: Path) -> dict:
    """개발 진단 한 번과 부모 고정. 결과 확인 후 설정 후보를 늘리지 않는다.

    성공 기준은 저장 장면 비용에 대한 기준이며 연속 제어 개선의 증거가 아니다.
    부분 완료 시 이미 끝낸 제외 묶음은 지문·입력이 같은 경우에만 재사용한다.
    """
    import torch as th
    th.set_num_threads(1)
    old, new, combo, out, work = map(Path, (old_runs, new_runs, combo_dir, out, work))
    started, cpu_started = time.time(), time.process_time()
    samples = load_samples(old, new, combo, work)
    identity = {"sources": json.loads((work / "history_manifest.json").read_text())["identity"],
                "route_manifest": rm.file_hash(work / "history_manifest.json"), "code": code_fingerprint(),
                "steps": STEPS, "model_seeds": list(SEEDS), "widths": [BASE_WIDTH, WIDTH]}
    complete = guard_cache(out / "diagnostic.json", identity)
    if complete is not None:
        policy = _validate_parent(out / "parent", identity)
        return {"diagnostic": complete, "policy": policy}
    group_ids = sorted({s["group"] for s in samples})
    folds = {}
    for group in group_ids:
        cache = out / "history_folds" / f"{group}.json"
        fold_id = {**identity, "excluded_group": group}
        rec = guard_cache(cache, fold_id)
        if rec is None:
            train = [s for s in samples if s["group"] != group]
            test = [s for s in samples if s["group"] == group]
            rec = {"identity": fold_id, "n_train": len(train), "n_test": len(test), "results": {}}
            for width in (BASE_WIDTH, WIDTH):
                chosen = picks(fit(train, width), test, width)
                rec["results"][str(width)] = {"gain": es.gain(test, chosen), "chosen": chosen,
                                               "changes": sum(p != s["p0"] for p, s in zip(chosen, test))}
                print(f"수요 이력 검사: 제외 {group}, 입력 {width}칸 완료", flush=True)
            _write_json(cache, rec)
        folds[group] = rec
    decision = adoption({g: r["results"]["96"]["gain"] for g, r in folds.items()},
                        {g: r["results"]["125"]["gain"] for g, r in folds.items()})
    print(f"부모 입력 고정: {decision['width']}칸(사전 기준에 따른 개발 진단)", flush=True)
    parent = out / "parent"
    parent.mkdir(parents=True, exist_ok=True)
    parent_meta = guard_cache(parent / "policy.json", identity)
    if parent_meta is None:
        if decision["uses_history"]:
            for seed, state in zip(SEEDS, fit(samples, WIDTH)):
                th.save(state, parent / f"seed{seed}.pt")
            source = "156절: 이력 채택 후 314개 전체 재학습(125칸)"
        else:
            for seed in SEEDS:
                shutil.copy2(combo / "students" / "b" / f"seed{seed}.pt", parent / f"seed{seed}.pt")
            source = "153절 COMBO B 가중치 그대로(96칸)"
        policy = {"identity": identity, "width": decision["width"], "uses_history": decision["uses_history"],
                  "source": source, "model_seeds": list(SEEDS), "samples": len(samples), "steps": STEPS,
                  "method": "l0", "tau": 0.0, "history_schema": schema(),
                  "weights": {f"seed{s}": rm.file_hash(parent / f"seed{s}.pt") for s in SEEDS},
                  "data": identity["sources"]["data"], "code": code_fingerprint()}
        _write_json(parent / "policy.json", policy)
    policy = _validate_parent(parent, identity)
    if policy["uses_history"] != decision["uses_history"] or policy["width"] != decision["width"]:
        raise SystemExit("부모 정책과 수요 이력 채택 결과가 다르다")
    diagnostic = {"identity": identity, "samples": len(samples), "folds": folds, "decision": decision,
                  "seconds": time.time() - started, "cpu_s": time.process_time() - cpu_started,
                  "interpretation": "기존 장면을 재사용한 개발 진단. 실제 연속 제어 성능의 증거가 아님"}
    _write_json(out / "diagnostic.json", diagnostic)
    return {"diagnostic": diagnostic, "policy": policy}
