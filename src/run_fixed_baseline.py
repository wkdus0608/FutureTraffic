from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

from evaluate_policies import run_policy
from rl_environment import PROJECT_ROOT, SCENARIO_DIR


def configure_sumo_python() -> Path:
    """현재 Python 환경에 설치된 SUMO의 위치를 찾는다."""
    configured = os.environ.get("SUMO_HOME")
    sumo_home = (
        Path(configured).expanduser().resolve()
        if configured
        else (Path(sysconfig.get_paths()["purelib"]) / "sumo").resolve()
    )
    if not (sumo_home / "tools").exists():
        raise RuntimeError("SUMO 도구를 찾지 못했습니다. 프로젝트 가상환경에서 실행하세요.")
    return sumo_home


def find_sumo_binary(sumo_home: Path) -> str:
    """화면이 있는 sumo-gui 실행 파일을 찾는다."""
    candidates = [Path(sys.prefix) / "bin" / "sumo-gui", sumo_home / "bin" / "sumo-gui"]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    found = shutil.which("sumo-gui")
    if found:
        return found
    raise FileNotFoundError("sumo-gui 실행 파일을 찾지 못했습니다.")


def open_gui(config: Path) -> None:
    """측정 파일을 만들지 않고 SUMO 화면에서 고정신호를 확인한다."""
    environment = os.environ.copy()
    if sys.platform == "darwin":
        xquartz_socket = Path("/tmp/.X11-unix/X0")
        if not xquartz_socket.exists():
            raise RuntimeError(
                "XQuartz가 실행 중이 아닙니다. 먼저 다음 명령을 실행하세요:\n"
                "open /Applications/Utilities/XQuartz.app"
            )
        environment.setdefault("DISPLAY", ":0")
    subprocess.Popen(
        [find_sumo_binary(configure_sumo_python()), "-c", str(config)],
        env=environment,
        start_new_session=True,
    )
    print("SUMO-GUI를 열었습니다. 상단의 실행 버튼을 눌러 차량을 확인하세요.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="내장 고정신호를 공통 평가 코드로 실행합니다."
    )
    parser.add_argument("--scenario-dir", type=Path, default=SCENARIO_DIR)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "fixed_20220810_0500",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gui", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scenario_dir = args.scenario_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    config = scenario_dir / "scenario.sumocfg"
    if not config.exists():
        raise FileNotFoundError(
            f"설정 파일이 없습니다: {config}\n먼저 prepare_sumo_scenario.py를 실행하세요."
        )
    if args.gui:
        open_gui(config)
        return

    # 고정신호 전용 계산식을 따로 두지 않고 최종 평가와 정확히 같은 함수를 쓴다.
    result = run_policy(
        policy="fixed",
        evaluation_seed=args.seed,
        output_dir=output_dir,
        scenario_dir=scenario_dir,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "baseline_metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "baseline_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result))
        writer.writeheader()
        writer.writerow(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"결과 폴더: {output_dir}")


if __name__ == "__main__":
    main()
