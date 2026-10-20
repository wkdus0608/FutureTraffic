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

사용:
  python src/explore_decide.py B --base-dir <v6_check/평가 지문> --base-models <v6 pet1 seed 5개> \
      --cand-dir <explore_single/eval/평가 지문> --cand-models <B seed 5개> --out <decision_B.json>
  python src/explore_decide.py C --base-dir … --base-models … --cand-dir … --cand-models … --fixedc-dir <v6_check/평가 지문> --out <decision_C.json>
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


def decide_b12(args) -> dict:
    from focused_eval import model_identity
    def dqn_tag(models):
        ids = {model_identity(m)["train_seed"]: model_identity(m)["model_sha256"][:10] for m in models}
        return lambda s: f"dqn_seed{s}_{ids[s]}"
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
    ap.add_argument("rule", choices=("B", "C", "B12", "ABL"))
    ap.add_argument("--base-dir")
    ap.add_argument("--base-models", nargs=5)
    ap.add_argument("--cand-dir")
    ap.add_argument("--cand-models", nargs=5)
    ap.add_argument("--set", nargs=7, action="append", default=[], metavar=("이름", "폴더", "M1", "M2", "M3", "M4", "M5"),
                    help="ABL: 이름(base·double·n5·n10) 폴더 모델 5개")
    ap.add_argument("--fixedc-dir", help="C: 보정 고정신호 실행이 있는 폴더(v6_check/평가 지문)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.rule == "ABL":
        result = ablation(args)
        Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=1))
        print(json.dumps(result, ensure_ascii=False, indent=1))
        return
    if not (args.base_dir and args.base_models and args.cand_dir and args.cand_models):
        ap.error("B·C·B12에는 --base-dir·--base-models·--cand-dir·--cand-models가 필요합니다.")
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
