"""[추가 탐색] 진행 기록 81절에 실행 전에 적은 기준으로 B(n-step·Double DQN)와 C(에이징 보상)를 판정한다 — 지문 밖, 계산만.

입력은 focused_eval run(또는 focused_conflicts measure) 결과 폴더(<out>/<평가 지문>)다. 검증 2일(09-28, 08-13) 무작위 도착 실행만 쓰고,
DQN은 모델 파일 지문으로 골라 seed 5개를 묶는다. 계산은 표를 만드는 focused_v6_report.pooled와 같다(차량 가중 지연,
300초 초과는 seed 평균, 좌회전 두 곳은 차량 가중).

- B(채택 기준, 76절과 같음): 후보 지연이 기준 DQN보다 1.3초 이상 짧고, 300초 초과(seed 평균)가 늘지 않고,
  좌회전 두 곳(길주로 동쪽·석천로 남쪽)이 모두 늘지 않으면 '채택'.
- C(공평성 변형 기준): 좌회전 두 곳이 기준 DQN보다 각각 10초 이상 짧고, 지연 증가가 1.3초 이하이고,
  지연이 보정 고정신호보다 짧으면 '쓸 수 있음'. 기준 DQN = C가 올라탄 모델(B를 채택하면 B, 아니면 v6 pet1).
- ABL(시야 확인, 86절): 기존 DQN·Double만·B(n=5+Double)·n=10+Double을 한 표로 놓고, 실행 전에 적은 해석을 그대로 적용한다
  (차이 1.3초 이상일 때만 '다름'). 설정을 고르지 않는다(채택 판정 없음).
- B12(12개 지도, 85절): 검증 2일 46조각, 학습 seed s 모델을 평가 seed s로. 모든 조각 기준 구간 지연이 기준 DQN(75절 모델)보다
  2.2초 이상 짧고, 300초 초과(1번 평균)가 늘지 않고, 좌회전 두 곳의 구간 지연이 늘지 않으면 '채택'. 흔들리는 조각(09-28 08시)을
  뺀 값과 단순 제어(평가 seed 42~46)는 함께 적되 판정에는 쓰지 않는다.
- COORD(12개 지도 연동 정보 관측, 96절): 기준 DQN은 B12 모델, 후보는 src/explore_coord12.py 모델. 채택 기준은 B12와 같은 네 가지
  (구간 지연 2.2초 이상 감소, 300초 초과·좌회전 두 곳이 늘지 않음). 원인 확인은 따로: 저녁 16~19시 길주로 동 → 서 직진 지연이
  B12보다 10초 이상 짧으면 '노린 곳에서 작동'. 고정신호(원자료·보정)·단순 제어·기존 DQN(75절, --ref-dir·--ref-models)은 함께 적는다.
- PRED1·PRED12(예측 제어, 98절): 고르지 않고 비교만 한다. 두 정책 X·Y에서 X의 지연이 문턱(단독 1.3초, 12개 지도 2.2초) 이상 짧고
  300초 초과·좌회전 두 곳이 늘지 않으면 'X가 낫다', 문턱 이상 길면 'X가 못하다', 그 밖은 '차이 없음'. 98.4의 해석 1~6을 그대로 붙인다.
  PRED1은 단독 모형(검증 2일 무작위가 주, 무리 도착은 민감도, 학습 13일은 학습 없는 정책만), PRED12는 12개 지도 검증 46조각 × 5번.
- TEACH1·TEACH12(선생님 DQN, 103절): 질문 1 선생님 DQN 대 같은 설정 DQN(H1·B12, 주 판정), 2 대 단순 제어, 3 대 선생님 P0
  (선생님을 넘었다·선생님 수준·선생님에 못 미친다), 4 대 흉내만 낸 DQN. 말과 문턱은 PRED와 같다. 확인용 시범 일치율을 함께 적는다.

사용:
  python src/explore_decide.py B --base-dir <v6_check/평가 지문> --base-models <v6 pet1 seed 5개> \
      --cand-dir <explore_single/eval/평가 지문> --cand-models <B seed 5개> --out <decision_B.json>
  python src/explore_decide.py C --base-dir … --base-models … --cand-dir … --cand-models … --fixedc-dir <v6_check/평가 지문> --out <decision_C.json>
  python src/explore_decide.py COORD --base-dir <explore_12map/평가 지문> --base-models <B12 seed 5개> \
      --cand-dir <explore_12map_coord/평가 지문> --cand-models <연동 모델 seed 5개> \
      --ref-dir <eval_12map/평가 지문> --ref-models <75절 모델 seed 5개> --out <decision_COORD.json>
  python src/explore_decide.py PRED1 --pred-dir <explore_pred/single/평가 지문> --v6-dir <v6_check/평가 지문> --v6-models <v6 seed 5개> \
      --h1-dir <explore_single/eval/평가 지문> --h1-models <H1 seed 5개> --out <decision_PRED1.json>
  python src/explore_decide.py PRED12 --pred-dir <explore_pred/network12/평가 지문> --ref-dir <eval_12map/평가 지문> --ref-models <75절 5개> \
      --base-dir <explore_12map/평가 지문> --base-models <B12 5개> --cand-dir <explore_12map_coord/평가 지문> --cand-models <COORD 5개> \
      --out <decision_PRED12.json>
  python src/explore_decide.py TEACH1 --teach-dir <explore_teacher/single/eval/평가 지문> --teach-models <선생님 DQN 5개> \
      --imitation-models <흉내만 낸 DQN 5개> --h1-dir … --h1-models … --v6-dir … --pred-dir <explore_pred/single/평가 지문> --out …
  python src/explore_decide.py TEACH12 --teach-dir <explore_teacher/network12/평가 지문> --teach-models … --imitation-models … \
      --base-dir <explore_12map/평가 지문> --base-models <B12 5개> --ref-dir <eval_12map/평가 지문> --pred-dir <explore_pred/network12/평가 지문> --out …
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

VALIDATION = ["20220928", "20220813"]
DELAY_S, LEFT_C_S = 1.3, 10.0
DELAY_12_S = 2.2                     # 12개 지도: 기준 DQN seed 표준편차 1.1초의 2배(85절)
UNSTABLE_12 = "20220928_08"
LEFT_12 = {"left_east": "477706392_02->477706883_02", "left_south": "477706883_01->477707013_01"}
EAST_STRAIGHT_12 = "477706392_02->477707013_01"   # 길주로 동 → 서 직진(95절 저녁 손실)
EVENING_12 = range(16, 20)
MECH_12_S = 10.0                                   # 96절 원인 확인 기준
SEEDS = [42, 43, 44, 45, 46]


def _read(p: Path) -> dict:
    import time
    for _ in range(8):
        try:
            return json.loads(p.read_text())
        except (TimeoutError, OSError):
            time.sleep(10)
    raise SystemExit(f"읽기 실패: {p}")


def windows12(root: Path, tag_of_seed) -> dict:
    """{seed: [조각별 window_204820]} (검증 2일 조각 전부, 빠지면 멈춤)."""
    wins = sorted(w for w in (root / "windows").iterdir() if w.name[:8] in VALIDATION)
    out = {}
    for s in SEEDS:
        rows = []
        for w in wins:
            p = w / f"{tag_of_seed(s)}_eval{s}" / "metrics.json"
            if not p.exists():
                raise SystemExit(f"빠진 실행: {p}")
            rows.append({**_read(p)["window_204820"], "_w": w.name})
        out[s] = rows
    return out


def summary12(per_seed: dict, skip: str | None = None) -> dict:
    import statistics
    vals = []
    for s, rows in per_seed.items():
        rows = [r for r in rows if r["_w"] != skip]
        n = sum(r["vehicles"] for r in rows)
        v = {"delay": sum(r["vehicles"] * r["mean_delay_s"] for r in rows) / n, "long": sum(r["long_delays"] for r in rows)}
        for name, key in LEFT_12.items():
            mv = [r["by_movement"][key] for r in rows if key in r["by_movement"]]
            v[name] = sum(m["vehicles"] * m["mean_delay_s"] for m in mv) / max(1, sum(m["vehicles"] for m in mv))
        vals.append(v)
    out = {k: round(statistics.fmean(v[k] for v in vals), 2) for k in vals[0]}
    out["delay_std"] = round(statistics.stdev(v["delay"] for v in vals), 2)
    out["windows"] = len([r for r in next(iter(per_seed.values())) if r["_w"] != skip])
    return out


def dqn_tag(models):
    from focused_eval import model_identity
    ids = {model_identity(m)["train_seed"]: model_identity(m)["model_sha256"][:10] for m in models}
    return lambda s: f"dqn_seed{s}_{ids[s]}"


def movement12(per_seed: dict, key: str, hours: range) -> float:
    """그 시간대 조각에서 한 이동의 구간 지연(차 가중)을 seed마다 구해 평균한다."""
    import statistics
    vals = []
    for rows in per_seed.values():
        mv = [r["by_movement"][key] for r in rows if int(r["_w"][9:]) in hours and key in r["by_movement"]]
        vals.append(sum(m["vehicles"] * m["mean_delay_s"] for m in mv) / max(1, sum(m["vehicles"] for m in mv)))
    return round(statistics.fmean(vals), 2)


def decide_coord(args) -> dict:
    """96절: 기준 DQN = B12 모델(88절), 후보 = 연동 정보 관측 모델. 채택은 B12와 같은 네 가지, 원인 확인은 저녁 동 → 서 직진."""
    for m in args.cand_models:
        rec = json.loads((Path(m) / "explore.json").read_text())
        if rec.get("observation") != "coord" or rec.get("n_steps") != 5:
            raise SystemExit(f"{m}: 연동 정보 관측(n = 5 + Double) 모델이 아니다: {rec}")
    for m in args.base_models:
        rec = json.loads((Path(m) / "explore.json").read_text())
        if rec.get("observation", "zone") != "zone" or rec.get("n_steps") != 5 or not rec.get("double_dqn"):
            raise SystemExit(f"{m}: B12(n = 5 + Double, 기존 관측) 모델이 아니다: {rec}")
    base = windows12(Path(args.base_dir), dqn_tag(args.base_models))
    cand = windows12(Path(args.cand_dir), dqn_tag(args.cand_models))
    ref = {"fixed": windows12(Path(args.ref_dir), lambda s: "fixed"), "fixedc": windows12(Path(args.ref_dir), lambda s: "fixedc"),
           "greedy": windows12(Path(args.ref_dir), lambda s: "greedy"),
           "dqn_75": windows12(Path(args.ref_dir), dqn_tag(args.ref_models))}
    groups = {"baseline_b12": base, "candidate_coord": cand, **ref}
    result = {"rule": "COORD", "dates": VALIDATION,
              "all_windows": {k: summary12(v) for k, v in groups.items()},
              "without_" + UNSTABLE_12: {k: summary12(v, UNSTABLE_12) for k, v in groups.items()},
              "evening_east_straight_s": {k: movement12(v, EAST_STRAIGHT_12, EVENING_12) for k, v in groups.items()}}
    b, c = result["all_windows"]["baseline_b12"], result["all_windows"]["candidate_coord"]
    checks = {"구간 지연 2.2초 이상 감소": b["delay"] - c["delay"] >= DELAY_12_S,
              "300초 초과가 늘지 않음": c["long"] <= b["long"],
              "길주로 동쪽 좌회전이 늘지 않음": c["left_east"] <= b["left_east"],
              "석천로 남쪽 좌회전이 늘지 않음": c["left_south"] <= b["left_south"]}
    result["checks"] = checks
    result["adopted"] = all(checks.values())
    e = result["evening_east_straight_s"]
    gain = round(e["baseline_b12"] - e["candidate_coord"], 2)
    result["mechanism"] = {"evening_east_straight_gain_s": gain, "threshold_s": MECH_12_S,
                           "reading": ("연동 정보가 노린 곳에서 작동했다." if gain >= MECH_12_S
                                       else "노린 곳에서 작동했다고 말할 수 없다.")}
    result["note"] = "이웃 신호 정보는 DQN만 본다. 결론은 '이웃 신호 정보를 가진 DQN'으로 좁혀 말한다(96절)."
    return result



# ------------------------------------------------------------------ 예측 제어(98절)
PRED_TAGS = {"p0": "pred_p0", "p1": "pred_p1", "p2": "pred_p2"}


def verdict(x: dict, y: dict, threshold: float, delay: str, long: str) -> dict:
    """98.4: X가 Y보다 낫다 / 못하다 / 차이 없음. 주고받은 것(300초 초과·좌회전)도 적는다."""
    d = x[delay] - y[delay]
    no_worse = x[long] <= y[long] and x["left_east"] <= y["left_east"] and x["left_south"] <= y["left_south"]
    word = "낫다" if -d >= threshold and no_worse else "못하다" if d >= threshold else "차이 없음"
    return {"word": word, "delay_diff_s": round(d, 2), "long_diff": round(x[long] - y[long], 2),
            "left_east_diff_s": round(x["left_east"] - y["left_east"], 2),
            "left_south_diff_s": round(x["left_south"] - y["left_south"], 2), "threshold_s": threshold}


def versus_fixedc(x: dict, fixedc: dict, delay: str) -> dict:
    return {"delay_diff_s": round(x[delay] - fixedc[delay], 2), "percent": round(100 * (x[delay] - fixedc[delay]) / fixedc[delay], 1)}


def interpret(c: dict, network: bool) -> list[str]:
    """98.4 미리 정한 해석. c = 비교 이름 → verdict."""
    out = []
    w = c["P0_vs_greedy"]["word"]
    out.append("1. 같은 정보에서 내다보는 계산이 단순 규칙보다 낫다." if w == "낫다" else
               "1. 120초를 내다봐도 단순 규칙과 같다." if w == "차이 없음" else "1. 같은 정보에서 내다보는 계산이 단순 규칙보다 못하다.")
    out.append("2. 감지 구간 안 도착 예측이 몫이 있다." if c["P1_vs_P0"]["word"] == "낫다" else
               f"2. 감지 구간 안 도착 예측의 몫은 보이지 않는다(P1이 P0보다 {c['P1_vs_P0']['word']}).")
    if network:
        m = c["mechanism"]
        ok = c["P2_vs_P1"]["word"] == "낫다" and m["evening_east_straight_gain_s"] >= MECH_12_S
        out.append("3. 이웃 신호 정보가 계산형 제어에서는 작동한다." if ok else
                   f"3. 이웃 신호 정보가 계산형 제어에서 작동했다고 말할 수 없다(P2가 P1보다 {c['P2_vs_P1']['word']}, "
                   f"저녁 동 → 서 직진 {m['evening_east_straight_gain_s']:+.2f}초 줄어듦, 기준 {MECH_12_S:.0f}초).")
    for name, v in c.items():
        if not name.startswith("4_"):
            continue
        p, dqn = name[2:].split("_vs_")
        if v["word"] == "낫다":
            out.append(f"4. {p} 대 {dqn}: 이 환경에서 이 DQN은 그 정보를 가진 계산형 제어를 넘지 못한다.")
        elif v["word"] == "못하다":
            out.append(f"4. {p} 대 {dqn}: 학습이 계산형 제어가 못 잡는 것을 배웠다.")
        else:
            out.append(f"4. {p} 대 {dqn}: 차이 없음.")
    return out


def _single_rows(folder: Path, dates: list[str], tag: str, arrivals: str) -> list[dict]:
    import xml.etree.ElementTree as ET
    rows = []
    for d in dates:
        run = folder / d / f"{tag}_eval42{'' if arrivals == 'random' else '_' + arrivals}"
        if not (run / "metrics.json").exists():
            raise SystemExit(f"빠진 실행: {run}")
        row = _read(run / "metrics.json")
        row["_trip"], row["_run"] = None, run
        if (run / "statistics.xml").exists():
            t = ET.parse(run / "statistics.xml").getroot().find("vehicleTripStatistics")
            if t is not None:
                row["_trip"] = (int(t.get("count")), float(t.get("duration")))
        rows.append(row)
    return rows


def _dqn_rows(folder: Path, dates: list[str], models: list[str], arrivals: str) -> list[dict]:
    tag = dqn_tag(models)
    from focused_eval import model_identity
    return [r for m in models for r in _single_rows(folder, dates, tag(model_identity(m)["train_seed"]), arrivals)]


def pet_counts(rows: list[dict]) -> dict:
    """실행마다 events.json.gz의 전환별 통과 후 간격(끝나는 녹색 차가 먼저 지난 경우)으로 PET < 0·< 1초 수."""
    import gzip
    below0 = below1 = transitions = 0
    for r in rows:
        path = r["_run"] / "events.json.gz"
        if not path.exists():
            raise SystemExit(f"PET 기록이 없습니다: {path}")
        with gzip.open(path, "rt") as f:
            for e in json.load(f):
                transitions += 1
                p = (e.get("pet") or {}).get("clearing_first")
                if p is not None:
                    below0 += p["pet"] < 0
                    below1 += p["pet"] < 1
    return {"transitions": transitions, "pet_below_0": below0, "pet_below_1": below1}


def pred_stats(rows: list[dict]) -> dict:
    stats = [r["pred"] for r in rows]
    if len({(s["horizon_s"], s["code"]) for s in stats}) != 1:
        raise SystemExit(f"예측 제어 실행의 내다보는 길이·코드가 섞였습니다: {sorted({(s['horizon_s'], s['code']) for s in stats})}")
    planned = sum(s["planned"] for s in stats)
    return {"horizon_s": stats[0]["horizon_s"], "code": stats[0]["code"], "runs": len(stats), "planned": planned,
            "mean_seconds": round(sum(s["seconds"] for s in stats) / max(1, planned), 4),
            "max_seconds": round(max(s["max_seconds"] for s in stats), 3), "outside_allowed": sum(s["outside_allowed"] for s in stats),
            "changed_by_rules": sum(sum(r["guard"]["changed_by_rules"].values()) for r in rows)}


def decide_pred_single(args) -> dict:
    split = json.loads((Path(__file__).resolve().parents[1] / "config/focused_split.json").read_text())
    pred, v6, h1 = Path(args.pred_dir), Path(args.v6_dir), Path(args.h1_dir)
    out = {"rule": "PRED1", "threshold_s": DELAY_S}

    def groups(dates, arrivals, training=False):
        g = {"fixed": _single_rows(pred if training else v6, dates, "fixed", arrivals),
             "fixedc": _single_rows(v6, dates, "fixedc", arrivals), "greedy": _single_rows(v6, dates, "greedy", arrivals),
             "P0": _single_rows(pred, dates, PRED_TAGS["p0"], arrivals), "P1": _single_rows(pred, dates, PRED_TAGS["p1"], arrivals)}
        if not training:
            g["dqn_v6"] = _dqn_rows(v6, dates, args.v6_models, arrivals)
            g["dqn_h1"] = _dqn_rows(h1 if arrivals == "random" else pred, dates, args.h1_models, arrivals)
        return g

    def compare(g):
        s = {k: summary(v) for k, v in g.items()}
        c = {"P0_vs_greedy": verdict(s["P0"], s["greedy"], DELAY_S, "mean_delay_s", "long_waits"),
             "P1_vs_P0": verdict(s["P1"], s["P0"], DELAY_S, "mean_delay_s", "long_waits")}
        for dqn in ("dqn_v6", "dqn_h1"):
            if dqn in s:
                c[f"4_P1_vs_{dqn}"] = verdict(s["P1"], s[dqn], DELAY_S, "mean_delay_s", "long_waits")
        c["5_vs_fixedc"] = {p: versus_fixedc(s[p], s["fixedc"], "mean_delay_s") for p in ("P0", "P1")}
        return s, c

    for name, dates, arrivals, training in (("validation_random", VALIDATION, "random", False),
                                            ("validation_platoon", VALIDATION, "platoon", False),
                                            ("training13_random", split["train"], "random", True)):
        g = groups(dates, arrivals, training)
        s, c = compare(g)
        out[name] = {"dates": dates, "summary": s, "comparisons": c,
                     "safety": {p: pet_counts(g[p]) for p in ("P0", "P1")}, "pred": {p: pred_stats(g[p]) for p in ("P0", "P1")}}
    main = out["validation_random"]["comparisons"]
    out["interpretation"] = interpret({k: v for k, v in main.items() if k != "5_vs_fixedc"}, network=False)
    t = out["training13_random"]["comparisons"]
    same = all(t[k]["word"] == main[k]["word"] for k in ("P0_vs_greedy", "P1_vs_P0"))
    out["interpretation"].append("6. 날짜를 넓혀도 같다(학습 13일, 학습 없는 정책끼리)." if same else
                                 "6. 검증 2일 결과는 그 2일에 기댄 것일 수 있다(학습 13일에서 1·2번 판정이 다름).")
    unsafe = [f"{n}:{p}" for n in ("validation_random", "validation_platoon", "training13_random")
              for p, v in out[n]["safety"].items() if v["pet_below_0"] > 0]
    out["safety_note"] = ("PET < 0 없음" if not unsafe else f"PET < 0 있음 → 한계로 적고 재현으로 원인 확인: {unsafe}")
    return out


def decide_pred_12(args) -> dict:
    pred, ref = Path(args.pred_dir), Path(args.ref_dir)
    g = {"fixed": windows12(ref, lambda s: "fixed"), "fixedc": windows12(ref, lambda s: "fixedc"),
         "greedy": windows12(ref, lambda s: "greedy"), "dqn_75": windows12(ref, dqn_tag(args.ref_models)),
         "dqn_b12": windows12(Path(args.base_dir), dqn_tag(args.base_models)),
         "dqn_coord": windows12(Path(args.cand_dir), dqn_tag(args.cand_models))}
    for v, tag in PRED_TAGS.items():
        g[v.upper()] = windows12(pred, lambda s, tag=tag: tag)
    s = {k: summary12(v) for k, v in g.items()}
    v12 = lambda a, b: verdict(s[a], s[b], DELAY_12_S, "delay", "long")
    east = {k: movement12(v, EAST_STRAIGHT_12, EVENING_12) for k, v in g.items()}
    c = {"P0_vs_greedy": v12("P0", "greedy"), "P1_vs_P0": v12("P1", "P0"), "P2_vs_P1": v12("P2", "P1"),
         "mechanism": {"evening_east_straight_gain_s": round(east["P1"] - east["P2"], 2), "threshold_s": MECH_12_S},
         "4_P1_vs_dqn_75": v12("P1", "dqn_75"), "4_P1_vs_dqn_b12": v12("P1", "dqn_b12"), "4_P2_vs_dqn_coord": v12("P2", "dqn_coord")}
    out = {"rule": "PRED12", "threshold_s": DELAY_12_S, "dates": VALIDATION, "all_windows": s,
           "without_" + UNSTABLE_12: {k: summary12(v, UNSTABLE_12) for k, v in g.items()},
           "evening_east_straight_s": east, "comparisons": c,
           "5_vs_fixedc": {p: versus_fixedc(s[p], s["fixedc"], "delay") for p in ("P0", "P1", "P2")}}
    out["interpretation"] = interpret(c, network=True)
    safety, stats = {}, {}
    for v, tag in PRED_TAGS.items():
        runs = [_read(w / f"{tag}_eval{sd}" / "metrics.json") for w in sorted((pred / "windows").iterdir())
                if w.name[:8] in VALIDATION for sd in SEEDS]
        safety[v.upper()] = {"collisions_in_204820": sum(r["collisions_in_204820"] for r in runs)}
        stats[v.upper()] = pred_stats(runs)
    out["safety"], out["pred"] = safety, stats
    unsafe = [p for p, x in safety.items() if x["collisions_in_204820"] > 0]
    out["safety_note"] = "석천사거리 안 충돌 없음" if not unsafe else f"석천사거리 안 충돌 있음 → 한계로 적고 재현으로 원인 확인: {unsafe}"
    return out


# ------------------------------------------------------------------ 선생님 DQN(103절)
Q1 = {"낫다": "1. 선생님이 DQN을 개선했다.", "차이 없음": "1. 선생님의 효과가 보이지 않는다.", "못하다": "1. 선생님이 오히려 해가 됐다."}
Q2 = {"낫다": "2. DQN이 단순 규칙을 넘었다.", "차이 없음": "2. 단순 규칙과 차이 없음.", "못하다": "2. 단순 규칙보다 못하다."}
Q4 = {"낫다": "4. 강화학습 단계가 더했다.", "차이 없음": "4. 흉내 내기가 대부분이다.", "못하다": "4. 강화학습이 흉내 낸 것을 망쳤다."}


def versus_teacher(x: dict, p0: dict, threshold: float, delay: str, long: str) -> dict:
    """103.2 질문 3: 문턱 이상 짧고 나머지가 늘지 않음 → 넘었다 / 지연 차이가 문턱 안 → 수준 / 문턱 이상 긺 → 못 미친다."""
    v = verdict(x, p0, threshold, delay, long)
    d = v["delay_diff_s"]
    v["teacher_word"] = ("선생님을 넘었다" if v["word"] == "낫다" else "선생님에 못 미친다" if d >= threshold else
                         "선생님 수준" if abs(d) < threshold else "지연은 문턱 이상 짧지만 다른 지표가 늘어 넘었다고 말할 수 없다")
    return v


def teach_interpret(c: dict) -> list[str]:
    return [Q1[c["1_vs_same_dqn"]["word"]], Q2[c["2_vs_greedy"]["word"]], f"3. {c['3_vs_teacher']['teacher_word']}.",
            Q4[c["4_vs_imitation"]["word"]]]


def agreement_of(models: list[str]) -> list[dict]:
    out = []
    for m in models:
        rec = json.loads((Path(m) / "teacher.json").read_text())
        if rec.get("stage") != "teacher":
            raise SystemExit(f"{m}: 선생님 DQN이 아니다.")
        out.append({"model": Path(m).name, **rec.get("imitation", {}).get("holdout", {})})
    return out


def _check_imitation(models: list[str], teach_models: list[str]) -> None:
    for m, t in zip(models, teach_models):
        cfg = json.loads((Path(m) / "training_config.json").read_text())
        if cfg.get("stage") != "imitation_only" or Path(m).parent != Path(t):
            raise SystemExit(f"{m}: {t}의 '흉내만 낸 DQN'이 아니다.")


def decide_teach_single(args) -> dict:
    teach, v6, h1, pred = Path(args.teach_dir), Path(args.v6_dir), Path(args.h1_dir), Path(args.pred_dir)
    _check_imitation(args.imitation_models, args.teach_models)
    out = {"rule": "TEACH1", "threshold_s": DELAY_S, "agreement": agreement_of(args.teach_models)}
    for name, arrivals in (("validation_random", "random"), ("validation_platoon", "platoon")):
        g = {"teacher_dqn": _dqn_rows(teach, VALIDATION, args.teach_models, arrivals),
             "imitation_dqn": _dqn_rows(teach, VALIDATION, args.imitation_models, arrivals),
             "dqn_h1": _dqn_rows(h1 if arrivals == "random" else pred, VALIDATION, args.h1_models, arrivals),
             "greedy": _single_rows(v6, VALIDATION, "greedy", arrivals), "fixedc": _single_rows(v6, VALIDATION, "fixedc", arrivals),
             "P0": _single_rows(pred, VALIDATION, PRED_TAGS["p0"], arrivals)}
        s = {k: summary(v) for k, v in g.items()}
        sv = lambda a, b: verdict(s[a], s[b], DELAY_S, "mean_delay_s", "long_waits")
        c = {"1_vs_same_dqn": sv("teacher_dqn", "dqn_h1"), "2_vs_greedy": sv("teacher_dqn", "greedy"),
             "3_vs_teacher": versus_teacher(s["teacher_dqn"], s["P0"], DELAY_S, "mean_delay_s", "long_waits"),
             "4_vs_imitation": sv("teacher_dqn", "imitation_dqn"),
             "extra_imitation_vs_h1": sv("imitation_dqn", "dqn_h1"), "extra_imitation_vs_greedy": sv("imitation_dqn", "greedy"),
             "vs_fixedc": {k: versus_fixedc(s[k], s["fixedc"], "mean_delay_s") for k in ("teacher_dqn", "imitation_dqn")}}
        out[name] = {"summary": s, "comparisons": c, "safety": {k: pet_counts(g[k]) for k in ("teacher_dqn", "imitation_dqn")}}
    out["interpretation"] = teach_interpret(out["validation_random"]["comparisons"])
    unsafe = [f"{n}:{k}" for n in ("validation_random", "validation_platoon") for k, v in out[n]["safety"].items() if v["pet_below_0"] > 0]
    out["safety_note"] = "PET < 0 없음" if not unsafe else f"PET < 0 있음 → 한계로 적고 재현으로 원인 확인: {unsafe}"
    return out


def decide_teach_12(args) -> dict:
    teach, ref, pred = Path(args.teach_dir), Path(args.ref_dir), Path(args.pred_dir)
    _check_imitation(args.imitation_models, args.teach_models)
    g = {"teacher_dqn": windows12(teach, dqn_tag(args.teach_models)), "imitation_dqn": windows12(teach, dqn_tag(args.imitation_models)),
         "dqn_b12": windows12(Path(args.base_dir), dqn_tag(args.base_models)), "greedy": windows12(ref, lambda s: "greedy"),
         "fixedc": windows12(ref, lambda s: "fixedc"), "P0": windows12(pred, lambda s: PRED_TAGS["p0"])}
    s = {k: summary12(v) for k, v in g.items()}
    sv = lambda a, b: verdict(s[a], s[b], DELAY_12_S, "delay", "long")
    c = {"1_vs_same_dqn": sv("teacher_dqn", "dqn_b12"), "2_vs_greedy": sv("teacher_dqn", "greedy"),
         "3_vs_teacher": versus_teacher(s["teacher_dqn"], s["P0"], DELAY_12_S, "delay", "long"),
         "4_vs_imitation": sv("teacher_dqn", "imitation_dqn"),
         "extra_imitation_vs_b12": sv("imitation_dqn", "dqn_b12"), "extra_imitation_vs_greedy": sv("imitation_dqn", "greedy")}
    out = {"rule": "TEACH12", "threshold_s": DELAY_12_S, "dates": VALIDATION, "agreement": agreement_of(args.teach_models),
           "all_windows": s, "without_" + UNSTABLE_12: {k: summary12(v, UNSTABLE_12) for k, v in g.items()}, "comparisons": c,
           "vs_fixedc": {k: versus_fixedc(s[k], s["fixedc"], "delay") for k in ("teacher_dqn", "imitation_dqn")}}
    out["interpretation"] = teach_interpret(c)
    safety = {}
    for k, models in (("teacher_dqn", args.teach_models), ("imitation_dqn", args.imitation_models)):
        tag = dqn_tag(models)
        runs = [_read(w / f"{tag(sd)}_eval{sd}" / "metrics.json") for w in sorted((teach / "windows").iterdir())
                if w.name[:8] in VALIDATION for sd in SEEDS]
        safety[k] = {"collisions_in_204820": sum(r["collisions_in_204820"] for r in runs)}
    out["safety"] = safety
    unsafe = [k for k, v in safety.items() if v["collisions_in_204820"] > 0]
    out["safety_note"] = "석천사거리 안 충돌 없음" if not unsafe else f"석천사거리 안 충돌 있음 → 한계로 적고 재현으로 원인 확인: {unsafe}"
    return out

def decide_b12(args) -> dict:
    base = windows12(Path(args.base_dir), dqn_tag(args.base_models))
    cand = windows12(Path(args.cand_dir), dqn_tag(args.cand_models))
    greedy = windows12(Path(args.base_dir), lambda s: "greedy")
    result = {"rule": "B12", "dates": VALIDATION,
              "all_windows": {"baseline_dqn": summary12(base), "candidate_dqn": summary12(cand), "greedy": summary12(greedy)},
              "without_" + UNSTABLE_12: {"baseline_dqn": summary12(base, UNSTABLE_12), "candidate_dqn": summary12(cand, UNSTABLE_12),
                                         "greedy": summary12(greedy, UNSTABLE_12)}}
    b, c = result["all_windows"]["baseline_dqn"], result["all_windows"]["candidate_dqn"]
    checks = {"구간 지연 2.2초 이상 감소": b["delay"] - c["delay"] >= DELAY_12_S,
              "300초 초과가 늘지 않음": c["long"] <= b["long"],
              "길주로 동쪽 좌회전이 늘지 않음": c["left_east"] <= b["left_east"],
              "석천로 남쪽 좌회전이 늘지 않음": c["left_south"] <= b["left_south"]}
    result["checks"] = checks
    result["adopted"] = all(checks.values())
    return result


def runs_for(folder: Path, policy: str, models: list[str] | None) -> list[dict]:
    from focused_eval import model_identity
    want = {model_identity(m)["model_sha256"] for m in models} if models else None
    rows = []
    for r in _load_dates(folder):
        if r["policy"] != policy or r.get("arrivals", "random") != "random" or r["date"] not in VALIDATION:
            continue
        if want is not None and r.get("model_sha256") not in want:
            continue
        rows.append(r)
    expected = len(VALIDATION) * (len(want) if want is not None else 1)
    if len(rows) != expected:
        raise SystemExit(f"{folder}의 {policy} 실행이 {len(rows)}개입니다(검증 2일 × 모델 = {expected}개여야 함).")
    return rows


def _load_dates(folder: Path) -> list[dict]:
    """검증 날짜 폴더만 읽는다(다른 날짜 폴더는 건드리지 않음)."""
    import xml.etree.ElementTree as ET
    out = []
    for d in VALIDATION:
        for m in sorted((folder / d).glob("*/metrics.json")):
            row = json.loads(m.read_text())
            stats = m.parent / "statistics.xml"
            row["_trip"] = None
            if stats.exists():
                t = ET.parse(stats).getroot().find("vehicleTripStatistics")
                if t is not None:
                    row["_trip"] = (int(t.get("count")), float(t.get("duration")))
            out.append(row)
    return out


def summary(rows: list[dict]) -> dict:
    from focused_v6_report import pooled
    p = pooled(rows)
    keys = ("mean_delay_s", "mean_delay_s_std", "long_waits", "left_east", "left_south", "ped_over", "unexplained")
    return {k: (round(p[k], 2) if isinstance(p.get(k), float) else p.get(k)) for k in keys}


def ablation(args) -> dict:
    """86절: 이름 → (폴더, 모델 5개). 기존(base)·Double만(double)·B(n5)·n10을 같은 계산으로 요약하고 미리 적은 해석을 붙인다."""
    sets = {name: summary(runs_for(Path(folder), "dqn", models)) for name, folder, *models in args.set}
    need = {"base", "double", "n5", "n10"}
    if set(sets) != need:
        raise SystemExit(f"--set 이름은 {sorted(need)}이어야 합니다(받은 것 {sorted(sets)}).")
    d = {k: v["mean_delay_s"] for k, v in sets.items()}
    same = lambda a, b: abs(d[a] - d[b]) < DELAY_S
    if same("double", "base") and same("double", "n5"):
        cause = ("모호함: Double만이 기존과도 B와도 1.3초 안이다(효과가 단계마다 조금씩 쌓이는 경우, 86절 해석이 가르지 못함). "
                 "표와 n=10 대 Double만의 차이를 함께 보고한다(87절).")
    elif same("double", "base") and d["double"] - d["n5"] >= DELAY_S:
        cause = "B의 개선은 n-step(시야) 덕이다(Double만은 기존과 같고, n=5가 Double만보다 1.3초 이상 짧음)."
    elif same("double", "n5"):
        cause = "B의 개선은 Double 덕이고 시야 때문이 아니다(Double만이 B와 같음)."
    else:
        cause = "미리 정한 두 경우 어디에도 들지 않는다(표를 그대로 보고하고 원인은 단정하지 않음)."
    if same("n10", "n5"):
        reach = "n=10이 n=5와 같다: n=5에서 시야가 이미 찼다(할인율 0.99 한계와 맞음)."
    elif d["n5"] - d["n10"] >= DELAY_S:
        reach = "n=10이 n=5보다 1.3초 이상 짧다: 더 멀리 보는 것이 더 낫다(설정은 바꾸지 않음)."
    else:
        reach = "n=10이 n=5보다 1.3초 이상 길다: 너무 멀리 묶으면 학습이 나빠진다."
    return {"rule": "ABL", "dates": VALIDATION, "arrivals": "random", "sets": sets,
            "differences_s": {"double-base": round(d["double"] - d["base"], 2), "n5-double": round(d["n5"] - d["double"], 2),
                              "n10-n5": round(d["n10"] - d["n5"], 2), "n10-double": round(d["n10"] - d["double"], 2)},
            "interpretation": {"cause": cause, "how_far": reach}, "note": "설정을 고르지 않는다(86절)."}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("rule", choices=("B", "C", "B12", "ABL", "COORD", "PRED1", "PRED12", "TEACH1", "TEACH12"))
    ap.add_argument("--base-dir")
    ap.add_argument("--base-models", nargs=5)
    ap.add_argument("--cand-dir")
    ap.add_argument("--cand-models", nargs=5)
    ap.add_argument("--set", nargs=7, action="append", default=[], metavar=("이름", "폴더", "M1", "M2", "M3", "M4", "M5"),
                    help="ABL: 이름(base·double·n5·n10) 폴더 모델 5개")
    ap.add_argument("--fixedc-dir", help="C: 보정 고정신호 실행이 있는 폴더(v6_check/평가 지문)")
    ap.add_argument("--ref-dir", help="COORD: 고정신호·단순 제어·기존 DQN 실행이 있는 폴더(eval_12map/평가 지문)")
    ap.add_argument("--ref-models", nargs=5, help="COORD: 기존 12개 지도 DQN(75절) seed 5개")
    ap.add_argument("--pred-dir", help="PRED1·PRED12: 예측 제어 결과 폴더(explore_pred/…/평가 지문)")
    ap.add_argument("--v6-dir", help="PRED1: 고정신호·단순 제어·v6 DQN 결과(v6_check/평가 지문)")
    ap.add_argument("--v6-models", nargs=5, help="PRED1: v6 pet1 DQN seed 5개")
    ap.add_argument("--h1-dir", help="PRED1: H1 무작위 도착 결과(explore_single/eval/평가 지문)")
    ap.add_argument("--h1-models", nargs=5, help="PRED1: H1(n = 10 + Double) seed 5개")
    ap.add_argument("--teach-dir", help="TEACH1·TEACH12: 선생님 DQN 평가 결과(explore_teacher/…/평가 지문)")
    ap.add_argument("--teach-models", nargs=5, help="TEACH: 선생님 DQN seed 5개")
    ap.add_argument("--imitation-models", nargs=5, help="TEACH: 흉내만 낸 DQN seed 5개(선생님 DQN 폴더의 imitation)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.rule == "ABL":
        result = ablation(args)
        Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=1))
        print(json.dumps(result, ensure_ascii=False, indent=1))
        return
    if args.rule in ("TEACH1", "TEACH12"):
        need = (["teach_dir", "teach_models", "imitation_models", "h1_dir", "h1_models", "v6_dir", "pred_dir"] if args.rule == "TEACH1"
                else ["teach_dir", "teach_models", "imitation_models", "base_dir", "base_models", "ref_dir", "pred_dir"])
        missing = [n for n in need if not getattr(args, n)]
        if missing:
            ap.error(f"{args.rule}에 필요한 인자: {missing}")
        result = (decide_teach_single if args.rule == "TEACH1" else decide_teach_12)(args)
        Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=1))
        print(json.dumps({k: result[k] for k in ("interpretation", "safety_note", "agreement")}, ensure_ascii=False, indent=1))
        return
    if args.rule == "PRED1":
        if not (args.pred_dir and args.v6_dir and args.v6_models and args.h1_dir and args.h1_models):
            ap.error("PRED1에는 --pred-dir·--v6-dir·--v6-models·--h1-dir·--h1-models가 필요합니다.")
        result = decide_pred_single(args)
        Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=1))
        print(json.dumps({k: result[k] for k in ("interpretation", "safety_note")}, ensure_ascii=False, indent=1))
        return
    if args.rule == "PRED12":
        if not (args.pred_dir and args.ref_dir and args.ref_models and args.base_dir and args.base_models
                and args.cand_dir and args.cand_models):
            ap.error("PRED12에는 --pred-dir·--ref-dir·--ref-models·--base-dir·--base-models·--cand-dir·--cand-models가 필요합니다.")
        result = decide_pred_12(args)
        Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=1))
        print(json.dumps({k: result[k] for k in ("interpretation", "safety_note")}, ensure_ascii=False, indent=1))
        return
    if not (args.base_dir and args.base_models and args.cand_dir and args.cand_models):
        ap.error("B·C·B12·COORD에는 --base-dir·--base-models·--cand-dir·--cand-models가 필요합니다.")
    if args.rule == "COORD":
        if not (args.ref_dir and args.ref_models):
            ap.error("COORD에는 --ref-dir·--ref-models가 필요합니다.")
        result = decide_coord(args)
        Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=1))
        print(json.dumps(result, ensure_ascii=False, indent=1))
        print(f"판정(COORD): {'기준을 모두 만족' if result['adopted'] else '기준을 만족하지 못함'}; 원인 확인: "
              f"{result['mechanism']['reading']} → {args.out}")
        return
    if args.rule == "B12":
        result = decide_b12(args)
        Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=1))
        print(json.dumps(result, ensure_ascii=False, indent=1))
        print(f"판정(B12): {'기준을 모두 만족' if result['adopted'] else '기준을 만족하지 못함'} → {args.out}")
        return
    base = summary(runs_for(Path(args.base_dir), "dqn", args.base_models))
    cand = summary(runs_for(Path(args.cand_dir), "dqn", args.cand_models))
    result = {"rule": args.rule, "dates": VALIDATION, "arrivals": "random", "baseline_dqn": base, "candidate_dqn": cand}
    if args.rule == "B":
        checks = {"지연 1.3초 이상 감소": base["mean_delay_s"] - cand["mean_delay_s"] >= DELAY_S,
                  "300초 초과가 늘지 않음": cand["long_waits"] <= base["long_waits"],
                  "길주로 동쪽 좌회전이 늘지 않음": cand["left_east"] <= base["left_east"],
                  "석천로 남쪽 좌회전이 늘지 않음": cand["left_south"] <= base["left_south"]}
        result["adopted"] = all(checks.values())
    else:
        fixedc = summary(runs_for(Path(args.fixedc_dir), "fixedc", None))
        result["fixedc"] = fixedc
        checks = {"길주로 동쪽 좌회전 10초 이상 감소": base["left_east"] - cand["left_east"] >= LEFT_C_S,
                  "석천로 남쪽 좌회전 10초 이상 감소": base["left_south"] - cand["left_south"] >= LEFT_C_S,
                  "지연 증가 1.3초 이하": cand["mean_delay_s"] - base["mean_delay_s"] <= DELAY_S,
                  "보정 고정신호보다 짧음": cand["mean_delay_s"] < fixedc["mean_delay_s"]}
        result["usable"] = all(checks.values())
    result["checks"] = checks
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=1))
    print(json.dumps(result, ensure_ascii=False, indent=1))
    verdict = result.get("adopted", result.get("usable"))
    print(f"판정({args.rule}): {'기준을 모두 만족' if verdict else '기준을 만족하지 못함'} → {args.out}")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
