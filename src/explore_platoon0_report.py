"""[추가 탐색] PLATOON0 판정 — 사전 등록(results/explore/platoon0_20261031/사전등록.md) 그대로. SUMO 없음.

결과 폴더의 실행(metrics.json)만 읽고, 비교는 ARENA3 심판 함수(arena3_referee.judge)로 한다: 날짜별 차량 가중 (기준 − 후보)
지연, 날짜 7개 평균, t95(자유도 6). 판정 문턱은 config/explore_platoon0_v1.json의 verdict.

  python src/explore_platoon0_report.py --out <Drive>/explore_platoon0 --reference <Drive>/arena3/referee/baseline
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import arena3_common as ac  # noqa: E402
import arena3_referee as ref  # noqa: E402
import explore_platoon0 as p0  # noqa: E402
import explore_weight as w  # noqa: E402

VERDICT_KO = {"room": "여지 있음", "no_room": "여지 없음", "unclear": "불분명"}
COMPARISONS = (("P", "P1M4", "M4"), ("P", "TM4", "M4"), ("P", "OM4", "M4"), ("R", "OM4", "M4"), ("P", "OM4", "TM4"))


def situations_for(cfg: dict, world: str) -> list[dict]:
    """심판 함수가 받는 상황 목록(차량 파일 SHA는 그 세계의 고정값)."""
    return [{**{k: s[k] for k in ("date", "period", "tag", "sumo_seed")}, "route_sha256": s["route_sha256"][world]}
            for s in cfg["situations"]]


def compare(candidate: dict, baseline: dict, lock: dict, situations: list[dict]) -> dict:
    return ref.judge([list(candidate.values())], list(baseline.values()), None, lock, situations)


def paired_dates(a: dict[str, float], b: dict[str, float], t975: float) -> dict:
    if set(a) != set(b):
        raise ValueError("두 비교의 날짜가 다릅니다.")
    diffs = {d: a[d] - b[d] for d in sorted(a)}
    mean, low, high = ref.t_interval(list(diffs.values()), t975)
    return {"mean_s": mean, "t95_s": [low, high], "per_date_s": diffs, "positive_dates": sum(v > 0 for v in diffs.values())}


def main_verdict(gain: dict, rule: dict) -> str:
    mean, low = gain["gain_date_mean_s"], gain["date_t95_s"][0]
    if mean >= rule["room_min_gain_s"] and low > 0:
        return "room"
    if mean < rule["no_room_below_s"]:
        return "no_room"
    return "unclear"


def reproduction(r_m4: dict[str, dict], reference: Path | None) -> dict:
    """R 세계 M4가 ARENA3 심판 기준선 M4와 차량당 지연·실행 녹색·차량 파일까지 같은가(사전 등록 점검)."""
    if reference is None:
        return {"checked": 0, "ok": None, "note": "기준선 폴더를 주지 않아 확인하지 않음"}
    bad, checked = [], 0
    for sid, m in sorted(r_m4.items()):
        path = Path(reference) / "runs" / "M4" / sid / "metrics.json"
        if not path.exists():
            bad.append({"situation": sid, "why": "기준선 없음"})
            continue
        r = w.read(path)
        checked += 1
        for field in ("mean_delay_s", "executed_sha16", "route_sha256", "cohort_sha256"):
            if r.get(field) != m.get(field):
                bad.append({"situation": sid, "field": field, "arena3": r.get(field), "platoon0": m.get(field)})
    return {"checked": checked, "ok": not bad, "mismatches": bad[:10], "mismatch_count": len(bad)}


def world_mean(rows: dict[str, dict]) -> float:
    return sum(float(m["actual_total_delay"]) for m in rows.values()) / sum(int(m["vehicles"]) for m in rows.values())


def build_report(out: Path, reference: Path | None) -> dict:
    cfg, lock = p0.config(), ac.config()
    fingerprint = w.read(Path(out) / "experiment.json")["fingerprint"]
    runs = {(wd, n): p0.read_runs(out, wd, n, cfg, fingerprint) for wd, names in p0.RUNS.items() for n in names}
    expected = len(cfg["situations"])
    missing = {f"{wd}/{n}": expected - len(r) for (wd, n), r in runs.items() if len(r) != expected}
    if missing:
        return {"status": "incomplete", "missing": missing}
    repro = reproduction(runs[("R", "M4")], reference)
    sits = {wd: situations_for(cfg, wd) for wd in p0.RUNS}
    results = {f"{wd}/{c}-{b}": compare(runs[(wd, c)], runs[(wd, b)], lock, sits[wd]) for wd, c, b in COMPARISONS}
    g_o, g_t, g_or = results["P/OM4-M4"], results["P/TM4-M4"], results["R/OM4-M4"]
    rule = cfg["verdict"]
    verdict = main_verdict(g_o, rule)
    cause = paired_dates(g_o["per_date_gain_s"], g_or["per_date_gain_s"], lock["judge"]["t975"])
    share = g_t["gain_date_mean_s"] / g_o["gain_date_mean_s"] if g_o["gain_date_mean_s"] > 0 else None
    return {
        "status": "done" if repro["ok"] is not False else "reproduction_failed",
        "verdict": verdict, "verdict_ko": VERDICT_KO[verdict],
        "main_gain_OM4_vs_M4_P": {k: g_o[k] for k in ("gain_date_mean_s", "date_t95_s", "positive_dates", "gain_vehicle_weighted_s")},
        "cause_platoon_minus_random": {**cause, "platoon_caused": cause["t95_s"][0] > 0},
        "timetable_share": share, "timetable_share_at_least_half": share is not None and share >= rule["timetable_share"],
        "reproduction_R_M4_vs_ARENA3": repro,
        "world_mean_delay_s": {f"{wd}/{n}": world_mean(r) for (wd, n), r in runs.items()},
        "comparisons": results,
        "rule": rule, "fingerprint": fingerprint,
        "note": "반복 사용 날짜의 개발 진단(학습 없음). 무리 도착은 모형이고, OM4·TM4는 줄 모형 계획기로 잰 값이라 진짜 천장이 아니다.",
    }


def summary_md(rep: dict) -> str:
    if rep["status"] == "incomplete":
        return "# PLATOON0 판정\n\n아직 끝나지 않은 실행이 있습니다: " + json.dumps(rep["missing"], ensure_ascii=False) + "\n"
    f = lambda x: "—" if x is None else f"{x:+.3f}"  # noqa: E731
    lines = ["# PLATOON0 판정 — 동쪽 무리 도착 경기장의 여지", "",
             "기준 대비 이득(초/대, +면 후보가 짧음). 날짜 7개 평균과 t95, ARENA3 심판 함수 그대로. 학습 없는 개발 진단.", "",
             "| 세계 | 후보 − 기준 | 날짜 평균 | t95 | 양수 날짜 | 동 좌회전 차이 | 남 좌회전 차이 | 300초 초과(후보/기준) | 품질 오류 | 정책 고유 충돌 |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for key, r in rep["comparisons"].items():
        world, pair = key.split("/")
        lefts = r["left_turns"]
        lines.append(f"| {world} | {pair} | {f(r['gain_date_mean_s'])} | [{f(r['date_t95_s'][0])}, {f(r['date_t95_s'][1])}] "
                     f"| {r['positive_dates']}/7 | {f(lefts['left_east']['difference_s'])} | {f(lefts['left_south']['difference_s'])} "
                     f"| {r['long_waits']['candidate']:.0f}/{r['long_waits']['m4']:.0f} | {len(r['quality_errors'])} "
                     f"| {len(r['collisions']['policy_specific'])} |")
    g = rep["main_gain_OM4_vs_M4_P"]
    c = rep["cause_platoon_minus_random"]
    share = rep["timetable_share"]
    repro = rep["reproduction_R_M4_vs_ARENA3"]
    lines += ["", f"- **주 판정: {rep['verdict_ko']}** (P 세계 OM4 − M4 = {f(g['gain_date_mean_s'])}초, "
                  f"t95 [{f(g['date_t95_s'][0])}, {f(g['date_t95_s'][1])}]; 문턱: ≥ {rep['rule']['room_min_gain_s']}초이고 하한 > 0이면 "
                  f"여지 있음, < {rep['rule']['no_room_below_s']}초면 없음)",
              f"- 원인 확인: G_O(P) − G_O(R) = {f(c['mean_s'])}초, t95 [{f(c['t95_s'][0])}, {f(c['t95_s'][1])}] → "
              + ("무리 도착 때문에 생긴 몫" if c["platoon_caused"] else "무리 도착 때문이라고 말할 수 없음"),
              f"- 시간표 몫 S = G_T ÷ G_O = " + ("—" if share is None else f"{share:.2f}") + " → "
              + ("이웃 시간표로 절반 넘게 잡힌다" if rep["timetable_share_at_least_half"] else "시간표 예측만으로는 절반에 못 미친다"),
              f"- 재현 점검(R 세계 M4 = ARENA3 기준선): " + ("확인 안 함" if repro["ok"] is None else
                                                       ("통과" if repro["ok"] else f"실패 {repro['mismatch_count']}건 — 결과를 읽지 말 것")),
              "- 세계별 M4 차량당 지연: " + ", ".join(f"{k} {v:.2f}초" for k, v in rep["world_mean_delay_s"].items()),
              "", "좌회전·300초 초과·충돌은 함께 적는 값이며 판정에 쓰지 않는다(사전 등록). " + rep["note"]]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reference", type=Path, help="ARENA3 심판 기준선 폴더(arena3/referee/baseline)")
    args = parser.parse_args()
    rep = build_report(args.out, args.reference)
    w.write(args.out / "report.json", rep)
    (args.out / "summary.md").write_text(summary_md(rep))
    print(summary_md(rep))
    return 0 if rep["status"] == "done" else 1


if __name__ == "__main__":
    raise SystemExit(main())
