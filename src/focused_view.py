"""석천사거리 장면을 SUMO 화면(sumo-gui)으로 본다.

실험과 같은 함수(write_routes, aihub_signal_program, write_actuated_additional)로 고른 날짜·시각의
차량과 신호를 만들고 SUMO 화면을 연다. 화면에 보이는 것이 실험 조건 그대로다.

- 화면 시계는 실제 시각이다(예: 08:00:00). 시작 시각 10분 전부터 차를 넣어 대기열이 있는 상태로 보여 준다.
- 시작 시각에서 한 번 멈춘다. ▶(Run)을 누르면 이어진다.
- DQN 장면은 학습한 모델이 생기면 추가한다.

- 지도와 신호는 지금 환경 변형(FOCUSED_RULESET, FOCUSED_NORTH_EXT)을 따른다. 기본은 북쪽 도로 연장(240m)이다.
  --policy fixedc는 전환 규칙의 전적색을 넣은 고정신호다(끝나는 녹색의 마지막 몇 초를 황색 뒤 전적색으로).

사용:
  .venv/bin/python src/focused_view.py --date 20220811 --start 0800
  .venv/bin/python src/focused_view.py --date 20220811 --start 1800 --policy actuated
  FOCUSED_RULESET=pet1 .venv/bin/python src/focused_view.py --date 20220928 --start 0800 --policy fixedc
  FOCUSED_NORTH_EXT=0 .venv/bin/python src/focused_view.py --date 20220928 --start 0800      (v5 지도, 북쪽 58m)
"""
from __future__ import annotations

import argparse
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

from focused_data import PROJECT, TARGET_TLS, find_network, variant
from focused_scenario import (AIHUB_YELLOW_S, aihub_signal_program, apply_all_red, window_route_file,
                              write_actuated_additional, write_tls_additional)

OUT = PROJECT / "results/focused_204820/view"
LEAD_S = 60   # 차량은 정지선 도착보다 최대 약 50초 먼저 출발하므로 화면 시작을 1분 당긴다


def seconds(hhmm: str) -> int:
    return int(hhmm[:2]) * 3600 + int(hhmm[2:]) * 60


def clock(s: int) -> str:
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}"


def write_view_settings(output: Path) -> Path:
    """화면 위치(석천사거리 중심, 접근로가 다 보이는 배율)와 재생 속도(0.1초에 1초 = 10배속)."""
    junction = ET.parse(find_network()).getroot().find(f"junction[@id='{TARGET_TLS}']")
    output.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n<viewsettings>\n'
        f'  <viewport zoom="350" x="{junction.get("x")}" y="{junction.get("y")}" angle="0"/>\n'
        '  <delay value="100"/>\n</viewsettings>\n', encoding="utf-8")
    return output


def prepare(date: str, start: str, minutes: int, warmup: int, policy: str) -> tuple[Path, dict]:
    t0 = seconds(start)
    first = max(0, t0 - warmup * 60) // 900 * 900             # 앞 시간대부터(15분 단위로 맞춤)
    end = t0 + minutes * 60
    last = (end + 899) // 900 * 900
    begin = max(0, first - LEAD_S)
    v = variant()
    run = OUT / f"{date}_{start}_{policy}_{v['rule_set']}_ext{v['north_extension_m']:g}"
    run.mkdir(parents=True, exist_ok=True)
    vehicles = len(window_route_file(date, first, last, run / "routes.rou.xml", origin_s=0))   # 실제 시각 그대로
    if policy in ("fixed", "fixedc"):
        phases = aihub_signal_program(date, 0, end + 600)       # 자정부터의 실제 신호. SUMO가 시작 시각에 맞춰 이어 간다
        if "y" not in phases[-1][1]:                              # 프로그램 끝→처음 되돌이의 황색(경고 방지, 실행 중에는 안 일어남)
            a, b = phases[-1][1], phases[0][1]
            phases.append((AIHUB_YELLOW_S, "".join("y" if x in "Gg" and y == "r" else x for x, y in zip(a, b))))
        if policy == "fixedc":
            phases, _ = apply_all_red(phases)                   # 평가의 보정 고정신호와 같은 방법
        tls = write_tls_additional(phases, run / "tls.add.xml", "aihub_corrected")
    else:
        from focused_eval import actuated_day_plan    # 평가와 같은 감응식 하루 계획
        greens, mins, maxes = actuated_day_plan(date)
        tls = write_actuated_additional(greens, run / "tls.add.xml", "actuated", mins, maxes, AIHUB_YELLOW_S)
    view = write_view_settings(run / "view.xml")
    from focused_sumo import configure_sumo
    configure_sumo()
    import sumolib
    config = run / "scene.sumocfg"
    subprocess.run([sumolib.checkBinary("sumo"), "-n", str(find_network()), "-r", str(run / "routes.rou.xml"),
                    "-a", str(tls), "--begin", str(begin), "--end", str(end), "--time-to-teleport", "-1",
                    "--seed", "42", "--human-readable-time", "true", "--no-step-log", "true",
                    "--save-configuration", str(config)], check=True)
    info = {"vehicles": vehicles, "begin": clock(begin), "pause": clock(t0), "pause_s": t0, "end": clock(end), "view": view}
    return config, info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--date", default="20220811", help="날짜(YYYYMMDD). 기본은 학습 날짜 2022-08-11")
    parser.add_argument("--start", default="0800", help="보고 싶은 시각(HHMM, 15분 단위)")
    parser.add_argument("--minutes", type=int, default=30, help="시작 시각부터 볼 길이(분)")
    parser.add_argument("--warmup", type=int, default=10, help="시작 전에 미리 돌릴 시간(분). 대기열이 쌓인 상태로 보기 위해")
    parser.add_argument("--policy", choices=("fixed", "fixedc", "actuated"), default="fixed",
                        help="fixed = 실제 운영 고정신호(원자료), fixedc = 전적색 보정 고정신호, actuated = 감응식")
    parser.add_argument("--prepare-only", action="store_true", help="파일만 만들고 화면은 열지 않는다")
    args = parser.parse_args()
    config, info = prepare(args.date, args.start, args.minutes, args.warmup, args.policy)
    print(f"차량 {info['vehicles']}대, 화면 {info['begin']}부터 {info['end']}까지, {info['pause']}에서 한 번 멈춤")
    print(f"설정 파일: {config}")
    if not args.prepare_only:
        import sumolib
        subprocess.Popen([sumolib.checkBinary("sumo-gui"), "-c", str(config), "--gui-settings-file", str(info["view"]),
                          "--start", "--breakpoints", str(info["pause_s"])])


if __name__ == "__main__":
    main()
