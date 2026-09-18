from __future__ import annotations

import os
import sys
import sysconfig
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCENARIO_DIR = PROJECT_ROOT / "data" / "processed" / "20220810_0500"
TARGET_TLS = "204820"


def configure_sumo() -> None:
    """Configure the pip-installed SUMO before importing SUMO-RL."""
    sumo_home = Path(sysconfig.get_paths()["purelib"]) / "sumo"
    if not (sumo_home / "tools").exists():
        raise RuntimeError("SUMO를 찾지 못했습니다. 프로젝트 가상환경에서 실행하세요.")
    os.environ.setdefault("SUMO_HOME", str(sumo_home))
    bin_dir = str(Path(sys.prefix) / "bin")
    os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")


def make_environment(
    *,
    seed: int = 42,
    use_gui: bool = False,
    output_prefix: Path | None = None,
    fixed_ts: bool = False,
    additional_sumo_cmd: str | None = None,
):
    configure_sumo()
    from sumo_rl import SumoEnvironment

    net_file = SCENARIO_DIR / "network.net.xml"
    route_file = SCENARIO_DIR / "traffic_20220810_0500.rou.xml"
    for path in (net_file, route_file):
        if not path.exists():
            raise FileNotFoundError(
                f"필요한 파일이 없습니다: {path}\n"
                "먼저 prepare_sumo_scenario.py를 실행하세요."
            )

    return SumoEnvironment(
        net_file=str(net_file),
        route_file=str(route_file),
        out_csv_name=str(output_prefix) if output_prefix else None,
        single_agent=True,
        ts_ids=[TARGET_TLS],
        use_gui=use_gui,
        num_seconds=1800,
        delta_time=5,
        yellow_time=2,
        min_green=5,
        max_green=50,
        enforce_max_green=True,
        reward_fn="diff-waiting-time",
        sumo_seed=seed,
        fixed_ts=fixed_ts,
        sumo_warnings=False,
        additional_sumo_cmd=additional_sumo_cmd,
    )
