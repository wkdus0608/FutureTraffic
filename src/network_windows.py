"""12개 지도 90분 조각 점검 — 고정신호로 돌려 막히는 조각을 찾는다(학습·평가에서 뺄 조각) — 지문 밖.

12개 지도는 평일 저녁에 하루 연속으로 돌리면 학습 날짜 몇 날이 교착한다(docs/12개지도_수요_20261015.md 6절). 그래서 10-16
사용자 결정으로 학습·평가를 90분 조각(준비 30분 + 제어 60분, network_rl 학습과 같은 틀) 단위로 하고, 고정신호에서
환경이 무너지는 조각은 뺀다.

- 점검 조각: 날짜마다 제어 시작 01:00 ~ 23:00(1시간 간격, 23개). 준비 30분은 그 앞이다.
- 돌리는 법: 학습 환경과 같은 조각 차량(network_rl.window_routes), 이웃 11곳 실제 신호(network_rl.neighbor_window),
  석천사거리 실제 운영 신호(원자료), 빈 도로에서 시작, 순간이동 끔, seed 42(하루 점검과 같은 SUMO 설정).
- 막힌(무너진) 조각 = 마지막 15분에 굳은 도로(점유 70% 넘고 속도 0.3m/s 미만, 하루 점검의 '굳은 도로'와 같은 기준)가
  하나라도 있거나, 조각이 끝날 때 지도에 못 들어간 차가 MAX_PENDING대를 넘는 조각. 붐벼서 못 들어간 차가 조금 남는
  것(08-11 18시: 128대, 굳은 도로 없음)은 무너짐이 아니다. 그런 조각까지 빼면 첨두가 덜 들어가 실험이 쉬운 쪽으로 치우친다.
- 학습: 제어 60분이 막힌 조각의 제어 시간과 겹치는 시나리오는 뺀다(excluded).
- 평가: 검증 날짜의 막히지 않은 조각(evaluation_windows)에서 정책을 비교한다(network_eval --windows).

사용:
  .venv/bin/python src/network_windows.py screen --dates train+validation --workers 4
  .venv/bin/python src/network_windows.py config          # 점검 결과로 config/network12_windows.json을 쓴다
결과: results/network_demand/windows/<배경>_<유턴>/<날짜>_<시>.json, screen_*.md
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import focused_data as fd
import network_demand as nd

HOURS = list(range(1, 24))                 # 제어 시작 시각(시)
MAX_PENDING = 500                          # 처음 100으로 시작했다가 08-11 18시(128대, 굳은 도로 없음)를 보고 올림(학습 날짜)
JAM_OCCUPANCY, JAM_SPEED = 70.0, 0.3
CONFIG = fd.PROJECT / "config/network12_windows.json"
ROOT = fd.PROJECT / "results/network_demand/windows"


def tag(uturn: str, background: float) -> str:
    return f"bg{round(background * 100)}_{uturn}"


def screen_one(job: tuple[str, int, str, float], root: Path | None = None) -> dict:
    """조각 하나를 고정신호로 돌려 막혔는지 본다. root를 주면 그 아래에 새로 돌린다(재현성 비교용, 저장된 결과를 쓰지 않음)."""
    import network_rl as nr
    import network_signals as ns
    from focused_scenario import aihub_signal_program
    from focused_sumo import configure_sumo
    from network_map import network_12
    date, hour, uturn, background = job
    out = (root or ROOT) / tag(uturn, background)
    out.mkdir(parents=True, exist_ok=True)
    result_path = out / f"{date}_{hour:02d}.json"
    if result_path.exists():
        return json.loads(result_path.read_text())
    work = out / f"{date}_{hour:02d}"
    work.mkdir(parents=True, exist_ok=True)
    origin, length = hour * 3600 - nr.WARMUP_S, nr.WARMUP_S + nr.CONTROL_S
    demand = nd.demand_dir(date, uturn, background=background)
    vehicles, _ = nr.window_routes(demand / "routes.rou.xml", origin, length, work / "routes.rou.xml")
    programs, _ = ns.day_programs(date)
    neighbors = nr.neighbor_window(programs, origin, length + 900, work / "neighbors.add.xml")
    target = [s for d, s in aihub_signal_program(date, origin, length + 900) for _ in range(d)]
    ns.write_additional({nr.TARGET_TLS: ns.compress(target)}, work / "target.add.xml")
    (work / "meandata.add.xml").write_text(f'<additional><edgeData id="ed" period="900" file="{work / "edgedata.xml"}" '
                                           'excludeEmpty="true"/></additional>')
    configure_sumo()
    import sumolib
    cmd = [sumolib.checkBinary("sumo"), "-n", str(network_12()), "-r", str(work / "routes.rou.xml"),
           "-a", f"{neighbors},{work / 'target.add.xml'},{work / 'meandata.add.xml'}", "--seed", "42", "--end", str(length),
           "--no-step-log", "--no-warnings", "--time-to-teleport", "-1", "--collision.check-junctions", "true",
           "--collision.action", "warn", "--collision-output", str(work / "collisions.xml"),
           "--summary-output", str(work / "summary.xml")]
    t0 = time.time()
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    steps = re.findall(r'time="([0-9.]+)"[^>]*?running="(\d+)"[^>]*?waiting="(\d+)"[^>]*?halting="(\d+)"',
                       (work / "summary.xml").read_text())
    end_pending = int(steps[-1][2])
    max_pending = max(int(w) for _, _, w, _ in steps)
    last = max(float(iv.get("begin")) for iv in ET.parse(work / "edgedata.xml").getroot().iter("interval"))
    jammed = sorted(e.get("id") for iv in ET.parse(work / "edgedata.xml").getroot().iter("interval")
                    if float(iv.get("begin")) == last for e in iv.iter("edge")
                    if not e.get("id").startswith(":") and float(e.get("occupancy", 0)) > JAM_OCCUPANCY
                    and float(e.get("speed", 99)) < JAM_SPEED)
    collisions = [c for c in ET.parse(work / "collisions.xml").getroot() if float(c.get("time")) >= nr.WARMUP_S]
    inside = sum(c.get("lane", "").startswith(f":{nr.TARGET_TLS}_") for c in collisions)
    result = {"date": date, "hour": hour, "vehicles": vehicles, "end_pending": end_pending, "max_pending": max_pending,
              "end_running": int(steps[-1][1]), "jammed_at_end": jammed, "collisions_control": len(collisions),
              "collisions_in_204820_control": inside, "failed": is_failed(end_pending, jammed),
              "seconds": round(time.time() - t0, 1)}
    result_path.write_text(json.dumps(result, ensure_ascii=False))
    shutil.rmtree(work)                      # 큰 기록은 지운다(판정은 result에)
    return result


def is_failed(end_pending: int, jammed: list[str]) -> bool:
    return bool(jammed) or end_pending > MAX_PENDING


def screen(dates: list[str], uturn: str, background: float, workers: int) -> str:
    jobs = [(d, h, uturn, background) for d in dates for h in HOURS]
    out = ROOT / tag(uturn, background)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    with ProcessPoolExecutor(workers) as ex:
        futures = [ex.submit(screen_one, j) for j in jobs]
        for n, f in enumerate(as_completed(futures), 1):
            r = f.result()
            r["failed"] = is_failed(r["end_pending"], r["jammed_at_end"])
            rows.append(r)
            if r["failed"] or n % 25 == 0:
                print(f"  {n}/{len(jobs)} {r['date']} {r['hour']:02d}시: {'막힘' if r['failed'] else '정상'} "
                      f"(끝날 때 못 들어간 차 {r['end_pending']}, 굳은 도로 {len(r['jammed_at_end'])})", flush=True)
    return summarize(rows, uturn, background)


def summarize(rows: list[dict], uturn: str, background: float) -> str:
    """조각별 결과를 지금 판정 기준(is_failed)으로 다시 판정해 표로 정리한다(screen과 config가 같이 씀)."""
    out = ROOT / tag(uturn, background)
    rows = sorted(rows, key=lambda r: (r["date"], r["hour"]))
    for r in rows:
        r["failed"] = is_failed(r["end_pending"], r["jammed_at_end"])
    failed = [r for r in rows if r["failed"]]
    dates = sorted({r["date"] for r in rows})
    split = json.loads(nd.SPLIT.read_text())
    lines = [f"# 12개 지도 90분 조각 점검({tag(uturn, background)}, 고정신호 원자료, 날짜 {len(dates)}일 × 조각 {len(HOURS)}개)", "",
             f"막힌(무너진) 조각 = 마지막 15분에 굳은 도로(점유 > {JAM_OCCUPANCY:.0f}%, 속도 < {JAM_SPEED}m/s) 있음, 또는 끝날 때 "
             f"못 들어간 차 > {MAX_PENDING}대.", "", f"- 막힌 조각 {len(failed)}개 / {len(rows)}개", "",
             "| 날짜 | 묶음 | 막힌 제어 시작 시각 | 끝날 때 못 들어간 차가 가장 많은 조각 |", "|---|---|---|---|"]
    for d in dates:
        kind = "학습" if d in split["train"] else ("검증" if d in split["validation"] else "?")
        hours = [r["hour"] for r in failed if r["date"] == d]
        worst = max((r for r in rows if r["date"] == d), key=lambda r: r["end_pending"])
        lines.append(f"| {d} | {kind} | {', '.join(f'{h:02d}시' for h in hours) or '없음'} | "
                     f"{worst['hour']:02d}시 {worst['end_pending']}대 |")
    lines += ["", "## 막힌 조각 자세히", "", "| 날짜 | 시 | 끝날 때 못 들어간 차 | 가장 많을 때 | 굳은 도로(끝) | 제어 중 충돌(석천사거리 안) |",
              "|---|---:|---:|---:|---|---:|"]
    for r in failed:
        lines.append(f"| {r['date']} | {r['hour']:02d} | {r['end_pending']} | {r['max_pending']} | "
                     f"{', '.join(r['jammed_at_end'][:4]) or '없음'} | {r['collisions_control']}({r['collisions_in_204820_control']}) |")
    text = "\n".join(lines) + "\n"
    path = out / f"screen_{time.strftime('%Y%m%d_%H%M')}.md"
    path.write_text(text)
    path.with_suffix(".json").write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    return text + f"\n저장: {path}\n"


def write_config(uturn: str, background: float) -> Path:
    """점검 결과(조각별 json)로 config/network12_windows.json을 쓴다. 학습·평가가 읽는다(12개 지도 지문에 들어감)."""
    out = ROOT / tag(uturn, background)
    rows = [json.loads(p.read_text()) for p in sorted(out.glob("2*_[0-9][0-9].json"))]
    failed = {}
    for r in rows:
        if is_failed(r["end_pending"], r["jammed_at_end"]):
            failed.setdefault(r["date"], []).append(r["hour"])
    cfg = {"version": f"network12_windows_{time.strftime('%Y%m%d')}",
           "description": "12개 지도 90분 조각 점검 결과(network_windows.py). 고정신호(원자료)로 막힌 조각의 제어 시작 시각. "
                          "학습은 제어 60분이 막힌 조각의 제어 시간과 겹치는 시나리오를 빼고, 평가는 검증 날짜의 막히지 않은 조각만 쓴다.",
           "background": background, "uturn_201293": uturn, "hours": HOURS,
           "criteria": {"max_pending_at_end": MAX_PENDING, "jam_occupancy_pct": JAM_OCCUPANCY, "jam_speed_mps": JAM_SPEED,
                        "signal": "석천사거리 실제 운영 고정신호(원자료), 이웃 실제 신호, 순간이동 끔, seed 42"},
           "screened_dates": sorted({r["date"] for r in rows}), "windows_screened": len(rows),
           "failed": {d: sorted(h) for d, h in sorted(failed.items())}}
    CONFIG.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n")
    return CONFIG


def load_config() -> dict | None:
    return json.loads(CONFIG.read_text()) if CONFIG.exists() else None


def excluded(date: str, control_start_s: int, cfg: dict | None = None) -> bool:
    """제어 [start, start + 60분)이 막힌 조각의 제어 시간(1시간)과 겹치면 True."""
    cfg = cfg if cfg is not None else load_config()
    if not cfg:
        return False
    bad = cfg["failed"].get(date, [])
    end = control_start_s + 3600
    return any(h * 3600 < end and control_start_s < (h + 1) * 3600 for h in bad)


def evaluation_windows(dates: list[str], cfg: dict | None = None) -> list[tuple[str, int]]:
    """평가 조각: 날짜마다 점검한 제어 시작 시각 중 막히지 않은 것."""
    cfg = cfg if cfg is not None else load_config()
    if not cfg:
        raise SystemExit(f"{CONFIG}가 없습니다. 먼저 network_windows.py screen과 config를 돌리세요.")
    missing = sorted(set(dates) - set(cfg["screened_dates"]))
    if missing:
        raise SystemExit(f"점검하지 않은 날짜 {missing}는 평가할 수 없습니다.")
    return [(d, h) for d in dates for h in cfg["hours"] if h not in cfg["failed"].get(d, [])]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("screen")
    s.add_argument("--dates", default="train+validation")
    s.add_argument("--workers", type=int, default=4)
    c = sub.add_parser("config")
    for p in (s, c):
        p.add_argument("--uturn", default=None, help="기본: 최종 환경 설정")
        p.add_argument("--background", type=float, default=None, help="기본: 최종 환경 설정")
    args = ap.parse_args()
    import network_rl as nr
    final = nr.final_environment()
    uturn = args.uturn or final["uturn"]
    background = final["background"] if args.background is None else args.background
    if args.cmd == "config":
        rows = [json.loads(p.read_text()) for p in sorted((ROOT / tag(uturn, background)).glob("2*_[0-9][0-9].json"))]
        print(summarize(rows, uturn, background))
        print("저장:", write_config(uturn, background))
        return
    split = json.loads(nd.SPLIT.read_text())
    names = {"train": split["train"], "validation": split["validation"],
             "train+validation": sorted(split["train"] + split["validation"])}
    dates = names.get(args.dates) or args.dates.split(",")
    if set(dates) & set(split["test"]):
        raise SystemExit("시험 날짜는 점검하지 않습니다(설정을 모두 정한 뒤 한 번만).")
    for d in dates:                            # 하루 수요가 없으면 먼저 만든다(차례로, 같은 날짜를 동시에 만들지 않게)
        demand = nd.demand_dir(d, uturn, background=background)
        if not (demand / "routes.rou.xml").exists():
            ledger = nd.build_day(d, uturn, demand, background=background)
            (demand / "audit.md").write_text(nd.audit(ledger))
    print(screen(sorted(dates), uturn, background, args.workers))


if __name__ == "__main__":
    main()
