"""[추가 탐색] 12개 지도 DQN을 B 설정(n-step + Double DQN)으로 학습한다 — 지문 밖, network_rl.py를 고치지 않는다(진행 기록 85절).

network_rl.py는 12개 지도 지문에 들어가므로 고치면 기존 모델·평가를 쓸 수 없다. 그래서 실행할 때만
focused_rl.hold_aware_dqn_class를 Double 판(explore_dqn)으로 바꿔 끼우고 network_rl의 학습 명령을 그대로 부른다
(network_rl.train은 이 이름을 함수 안에서 가져온다). n-step은 FOCUSED_DQN_NSTEPS로 넣는다(focused_train.DQN_SETTINGS).
학습 기록(training_config.json)의 dqn에 n_steps가 남고, 이 파일이 seed 폴더에 explore.json(Double 여부)을 더 쓴다.

사용(Colab 23번 칸):
  FOCUSED_RULESET=pet1 FOCUSED_NORTH_EXT=240 FOCUSED_DQN_NSTEPS=5 python src/explore_train12.py --seed 42 --out <Drive>/explore_12map/models_nstep5_double/pet1 --resume
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main() -> None:
    import focused_rl
    import network_rl
    from explore_dqn import double_hold_aware_dqn_class
    if not os.environ.get("FOCUSED_DQN_NSTEPS"):
        raise SystemExit("B 설정은 FOCUSED_DQN_NSTEPS=5와 함께 돌린다.")
    focused_rl.hold_aware_dqn_class = double_hold_aware_dqn_class
    args = sys.argv[1:]
    out = Path(args[args.index("--out") + 1]) if "--out" in args else None
    seed = args[args.index("--seed") + 1] if "--seed" in args else "42"
    sys.argv = [sys.argv[0], "train", *args]
    network_rl.main()
    if out is not None and (out / f"seed_{seed}" / "training_config.json").exists():
        (out / f"seed_{seed}" / "explore.json").write_text(json.dumps(
            {"double_dqn": True, "n_steps": int(os.environ["FOCUSED_DQN_NSTEPS"]), "reward": "queue",
             "note": "진행 기록 85절, src/explore_train12.py"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
