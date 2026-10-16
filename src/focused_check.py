"""설명·확인용 도구. 계산해서 보여 주기만 하고 결과 파일을 바꾸지 않는다(환경·평가 지문 밖).

north   석천로 북쪽 진입로 길이: v5 지도(AI Hub 원본)와 지금 변형(연장)
rules   전환 규칙 묶음(none·pet1·pet1_5)별 전적색 표
signal  그날 실제 신호(원자료)와 전적색 보정 신호에서, 고른 시각 뒤 전환 몇 개를 초 단위로 나란히 보여 준다
mlp     학습한 DQN 모델의 신경망 구조(층, 활성화 함수, 가중치 수, 본 망·목표망)

사용:
  .venv/bin/python src/focused_check.py north
  .venv/bin/python src/focused_check.py rules
  FOCUSED_RULESET=pet1 .venv/bin/python src/focused_check.py signal --date 20220928 --time 0800
  .venv/bin/python src/focused_check.py mlp
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

NORTH_APPROACH = "477706911_01"      # 석천로 북쪽 진입로(정지선 앞 도로)
VEHICLE_SPACING_M = 7.5             # 줄 선 차 한 대가 차지하는 길이(차 + 간격, focused_rl과 같은 값)


def north(_args) -> None:
    from focused_data import NETWORK, find_network, variant
    from focused_scenario import UPSTREAM_LIMIT_M, FocusedNetwork
    v = variant()
    for label, net in (("v5 지도(AI Hub 원본)", NETWORK), (f"지금 변형(북쪽 연장 {v['north_extension_m']:g}m)", find_network())):
        fn = FocusedNetwork(net)
        chain = fn._chain(NORTH_APPROACH, True, UPSTREAM_LIMIT_M) + [NORTH_APPROACH]
        total = sum(fn.length[e] for e in chain)
        lanes = fn.lanes(chain[0])
        print(f"{label}: 차가 들어오는 곳 {chain[0]} → 정지선까지 {total:.1f}m, {lanes}차로, "
              f"줄 설 수 있는 차 약 {total * lanes / VEHICLE_SPACING_M:.0f}대 (지도 파일 {Path(net).name})")
    print("v5 지도는 정지선 58m 위에서 끝난다(그 위는 AI Hub 지도에 없음). 연장은 같은 방향 곧은 3차로로 가정했다.")


def rules(_args) -> None:
    from focused_data import TRANSITION_RULES
    from focused_intergreen import GREEN_NAMES, is_real_order
    cfg = json.loads(TRANSITION_RULES.read_text())
    sets = cfg["rule_sets"]
    pet = cfg["evidence"]["smallest_pet_s"]
    print("전환(녹색 → 녹색) | 실제 신호에 있나 | v5 15일 가장 작은 간격 | " + " | ".join(f"{n} 전적색" for n in sets))
    for i in range(4):
        for j in range(4):
            if i == j:
                continue
            key = f"{i}>{j}"
            cells = " | ".join(f"{sets[n].get(key, 0)}초" for n in sets)
            print(f"{GREEN_NAMES[i]} → {GREEN_NAMES[j]} | {'○' if is_real_order(i, j) else ' '} | {pet[key]:+.1f}초 | {cells}")
    print("간격 = 끝나는 녹색 마지막 차가 빠져나간 뒤 시작하는 녹색 첫 차가 그 자리에 닿기까지의 시간. "
          "pet1은 1초, pet1_5는 1.5초 이상이 되도록 전적색을 정했다.")


def _label(state: str, greens: list[str]) -> str:
    from focused_intergreen import GREEN_NAMES
    if state in greens:
        return f"녹색({GREEN_NAMES[greens.index(state)]})"
    return "황색" if "y" in state else "전적색"


def _timeline(program: list[tuple[int, str]], start_s: int, count: int, greens: list[str]) -> list[str]:
    """start_s 뒤 녹색 count개와 그 사이 전환을 '시각 이름 n초' 줄로."""
    rows, t, shown = [], 0, 0
    for dur, state in program:
        if t + dur > start_s and shown <= count:
            rows.append(f"{t // 3600:02d}:{t % 3600 // 60:02d}:{t % 60:02d}  {_label(state, greens)} {dur}초")
            if state in greens:
                shown += 1
        t += dur
        if shown > count:
            break
    return rows


def signal(args) -> None:
    from focused_data import variant_signature
    from focused_eval import fixed_day_program
    from focused_scenario import signal_greens
    greens = signal_greens()
    start = int(args.time[:2]) * 3600 + int(args.time[2:]) * 60
    raw = _timeline(fixed_day_program(args.date), start, args.count, greens)
    fixed = _timeline(fixed_day_program(args.date, corrected=True), start, args.count, greens)
    print(f"{args.date} {args.time} 뒤 실제 신호 — 왼쪽: 원자료 그대로, 오른쪽: 전적색 보정({variant_signature()})")
    width = max(len(r) for r in raw) + 4
    for i in range(max(len(raw), len(fixed))):
        left = raw[i] if i < len(raw) else ""
        right = fixed[i] if i < len(fixed) else ""
        print(left.ljust(width) + right)
    print("보정은 끝나는 녹색의 마지막 몇 초를 빼서 황색 뒤 전적색으로 둔다. 그래서 다음 녹색이 켜지는 시각은 원자료와 같다.")


def _default_model() -> Path | None:
    for drive in sorted(Path.home().glob("Library/CloudStorage/GoogleDrive-*")):
        for base in (drive / "내 드라이브" / "FutureTraffic", drive / "My Drive" / "FutureTraffic"):
            for name in ("models_v6/pet1/seed_42", "models_v5/seed_42"):
                if (base / name / "dqn_204820.zip").exists():
                    return base / name
    return None


def mlp(args) -> None:
    from focused_sumo import configure_sumo
    configure_sumo()
    from stable_baselines3 import DQN
    model_dir = Path(args.model) if args.model else _default_model()
    if model_dir is None or not (model_dir / "dqn_204820.zip").exists():
        raise SystemExit("모델 파일을 찾지 못했습니다. --model <학습 폴더>를 주세요(예: Drive의 FutureTraffic/models_v5/seed_42).")
    model = DQN.load(model_dir / "dqn_204820.zip", device="cpu")
    q, target = model.policy.q_net, model.policy.q_net_target
    n = sum(p.numel() for p in q.parameters())
    print(f"모델: {model_dir}")
    print(f"입력 {model.observation_space.shape[0]}개 → 출력 {model.action_space.n}개(녹색 4개 각각의 Q값)")
    print(f"은닉층 {model.policy.net_arch}, 활성화 함수 {model.policy.activation_fn.__name__}, 가중치 {n:,}개")
    print("본 망(q_net):")
    print(q.q_net)
    same = all(a.shape == b.shape for a, b in zip(q.parameters(), target.parameters()))
    print(f"목표망(q_net_target): 본 망과 같은 모양 {same}, 5,000단계마다 본 망을 복사한다")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("north")
    sub.add_parser("rules")
    s = sub.add_parser("signal")
    s.add_argument("--date", default="20220928")
    s.add_argument("--time", default="0800", help="HHMM")
    s.add_argument("--count", type=int, default=6, help="보여 줄 녹색 수")
    m = sub.add_parser("mlp")
    m.add_argument("--model", default=None, help="학습 폴더(없으면 Drive에서 찾음)")
    args = p.parse_args()
    {"north": north, "rules": rules, "signal": signal, "mlp": mlp}[args.cmd](args)


if __name__ == "__main__":
    main()
