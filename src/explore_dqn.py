"""[추가 탐색] DQN 학습 방법·보상 변형 — 지문 밖, 잠근 focused_rl.py를 고치지 않고 이어받는다(진행 기록 81절).

- Double DQN(B): HoldAwareDQN과 같고, 목표값의 '다음 행동'만 학습 중인 망이 그대로 실행될 행동(관측 끝 표시) 안에서 고른다.
  그 행동의 값은 목표망으로 잰다(van Hasselt 2016). 행동 선택·재생 버퍼·운영 규칙은 그대로다.
  n-step(B)은 stable-baselines3 2.9의 DQN(n_steps=…) 기능을 그대로 쓴다. HoldAwareDQN.train이 replay_data.discounts를 이미 쓴다.
- 에이징 보상(C): −0.1 × Σ(1 + 그 차의 누적 대기 ÷ AGING_S). 같은 감지 구간의 멈춘 차(SUMO 정지 기준)와 진입 대기 차(무게 1)다.
  기존 보상(zone_queue_reward)은 모든 차를 1로 센다. 오래 기다린 차를 무겁게 세어 좌회전처럼 적은 이동이 오래 굶지 않게 한다
  (운영체제 스케줄링의 에이징과 같은 생각).
- 학습에서 고르는 법(focused_train.py): FOCUSED_DQN_DOUBLE=1, FOCUSED_DQN_NSTEPS=5, FOCUSED_REWARD=aging. 학습 기록 explore에 남는다.
- 평가는 기존 평가 코드 그대로다(평가는 predict만 쓰므로 HoldAwareDQN으로 불러도 같다).
"""
from __future__ import annotations

AGING_S = 60.0


from focused_rl import hold_aware_dqn_class as _hold_aware_dqn_class   # 원래 클래스(12개 지도에서 바꿔 끼워도 여기는 원래 것)


def double_hold_aware_dqn_class():
    import torch as th

    class DoubleHoldAwareDQN(_hold_aware_dqn_class()):
        def allowed_next_values(self, next_q: "th.Tensor", next_obs: "th.Tensor") -> "th.Tensor":
            n = self.action_space.n
            mask = next_obs[:, -n:] > 0.5
            online = self.q_net(next_obs)
            pick = online.masked_fill(~mask, -th.inf).argmax(dim=1, keepdim=True)
            current = next_obs[:, :n].argmax(dim=1, keepdim=True)
            action = th.where(mask.any(dim=1, keepdim=True), pick, current)
            return next_q.gather(1, action).squeeze(1)

    return DoubleHoldAwareDQN


def aging_queue_reward(ts) -> float:
    from focused_rl import ALL_ZONE_EDGES, REWARD_SCALE
    sumo = ts.sumo
    total = 0.0
    for edge in ALL_ZONE_EDGES:
        for v in sumo.edge.getLastStepVehicleIDs(edge):
            if sumo.vehicle.getSpeed(v) < 0.1:
                total += 1.0 + sumo.vehicle.getAccumulatedWaitingTime(v) / AGING_S
    total += len(sumo.simulation.getPendingVehicles())
    return -REWARD_SCALE * total
