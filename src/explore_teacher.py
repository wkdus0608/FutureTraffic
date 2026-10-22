"""[추가 탐색] 선생님 DQN(TEACH) — 예측 제어 P0의 판단을 먼저 흉내 낸 뒤 강화학습으로 다듬는다(진행 기록 103절, DQfD 계열). 지문 밖.

단계(103.1절, 값은 아래 상수 그대로):
1. collect: DQN 학습과 같은 학습 환경(학습 13일, 준비 30분 + 60분, pet1)에서 선생님 P0(120초)가 신호를 고르게 하고, 5초 판단마다
   DQN 관측(64칸)·선생님 녹색·보상·다음 관측·끝났는지·고를 수 있었는지를 남긴다. 시나리오 뽑기 seed 1001~1005로 28개씩.
   선생님은 운영 규칙이 허용한 녹색 안에서만 고르므로 규칙이 바꾼 판단과 관측 끝 표시가 허용 목록과 다른 판단은 0이어야 한다(아니면 멈춤).
2. 흉내 내기: 시범만으로 PRETRAIN_UPDATES번 고친다. 손실 = 지금 DQN과 같은 TD 손실(n-step, Double, 실행할 수 있는 녹색만)
   + MARGIN_WEIGHT × 차이 손실. 차이 손실 = max_허용 a [Q(s, a) + MARGIN·(a ≠ 선생님)] − Q(s, 선생님), 고를 수 있었던 판단에만.
   목표망은 TARGET_EVERY번마다 맞춘다(학습 때 목표망 5,000단계와 같은 간격). 끝나면 '흉내만 낸 DQN'으로 따로 저장하고,
   확인용 시범(seed 1005의 마지막 14개 에피소드)에서 고를 수 있었던 판단의 선생님 일치율을 잰다.
3. 다듬기: 그 모델에서 10만 단계 강화학습. 64개 가운데 16개(DEMO_SHARE)는 시범(확인용 제외)에서 꺼내 차이 손실도 건다.
   무작위 행동은 처음부터 끝까지 EPSILON(학습 전 무작위 탐색도 하지 않음, 판단은 처음부터 정책 + 5% 무작위). 나머지 설정은 같은 지도의
   가장 좋은 DQN과 같다(단독 H1: n = 10, 12개 지도 B12: n = 5, 둘 다 Double). 학습 환경·보상·운영 규칙·체크포인트·기록은 그대로다.

잠근 파일·focused_train.py·network_rl.py는 고치지 않는다. 실행할 때만 focused_rl.hold_aware_dqn_class를 이 학습기로 바꿔 끼우고
focused_train·network_rl의 학습 명령을 그대로 부른다. 평가는 기존 평가 코드 그대로(평가는 predict만 쓰므로 원래 학습기로 불러도 같다).

사용(Colab 32·33번 칸):
  FOCUSED_RULESET=pet1 FOCUSED_NORTH_EXT=240 python src/explore_teacher.py collect --map single --seed 1001 --out <Drive>/explore_teacher/demos
  FOCUSED_RULESET=pet1 FOCUSED_NORTH_EXT=240 FOCUSED_DQN_NSTEPS=10 python src/explore_teacher.py train --map single --demos <demos> \
      --seed 42 --timesteps 100000 --out <Drive>/explore_teacher/models_single --resume
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

import explore_dqn  # noqa: E402  원래 학습기를 바꿔 끼우기 전에 가져온다(Double 학습기가 원래 HoldAwareDQN을 이어받게)

TEACHER_VARIANT, TEACHER_HORIZON_S = "p0", 120
TEACHER_CODE = "9440fe1d3869"          # explore_pred.py(100절에서 결과를 낸 코드)
COLLECT_SEEDS = [1001, 1002, 1003, 1004, 1005]
EPISODES_PER_SEED = 28
HOLDOUT_SEED, HOLDOUT_FROM = 1005, 14  # seed 1005의 14번째 에피소드부터(마지막 14개)는 확인용
PRETRAIN_UPDATES, PRETRAIN_BATCH = 20_000, 64
TARGET_EVERY = 5_000
MARGIN, MARGIN_WEIGHT = 0.8, 1.0
DEMO_SHARE = 0.25
EPSILON = 0.05
N_STEPS = {"single": 10, "network12": 5}
MAPS = ("single", "network12")


def code_fingerprint() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]


def demo_file(folder: Path, kind: str, seed: int) -> Path:
    return Path(folder) / f"demo_{kind}_{seed}.npz"


# ------------------------------------------------------------------ 1. 시범 모으기(SUMO)
def _training_env(kind: str, seed: int, workdir: Path):
    from focused_rl import SignalConstraints
    constraints = SignalConstraints()
    if kind == "single":
        from focused_data import COUNTS, load_counts
        from focused_train import WarmStartTrainingEnv, training_scenarios
        counts = load_counts(COUNTS)
        env = WarmStartTrainingEnv(training_scenarios(counts), counts, seed, workdir, constraints=constraints)
    else:
        import network_rl as nr
        final = nr.final_environment()
        env = nr.WarmStart12Env(nr.training_scenarios(), seed, workdir, constraints=constraints, uturn=final["uturn"],
                                background=final["background"], depart_fix=final["depart_fix"])
    return env, constraints


def collect(kind: str, seed: int, episodes: int, out: Path) -> dict:
    """선생님 P0가 학습 환경을 episodes번 몰며 남긴 시범을 npz 하나로."""
    import explore_pred as ep
    from focused_data import TARGET_TLS, environment_fingerprint
    if ep.code_fingerprint() != TEACHER_CODE:
        raise SystemExit(f"선생님 코드 {ep.code_fingerprint()}가 100절의 예측 제어 {TEACHER_CODE}와 다릅니다.")
    Teacher = ep.controller_class(TEACHER_VARIANT, TEACHER_HORIZON_S)
    env, constraints = _training_env(kind, seed, Path(tempfile.mkdtemp(prefix=f"teacher_{kind}_{seed}_")))
    n = env.action_space.n
    rec = {k: [] for k in ("obs", "next_obs", "action", "reward", "truncated", "free", "episode")}
    problems = {"changed_by_rules": 0, "mask_differs": 0}
    started = time.time()
    try:
        for e in range(episodes):
            obs, _ = env.reset()
            sim = env.current
            teacher = Teacher(Teacher.movements(env.route), [p.state for p in sim.traffic_signals[TARGET_TLS].green_phases])
            done = False
            while not done:
                allowed = constraints.allowed(sim)
                if [i for i in range(n) if obs[-n + i] > 0.5] != allowed:
                    problems["mask_differs"] += 1
                a = int(teacher(sim, allowed))
                nxt, r, terminated, truncated, info = env.step(a)
                if info["executed_action"] != a:
                    problems["changed_by_rules"] += 1
                for k, v in (("obs", obs), ("next_obs", nxt), ("action", info["executed_action"]), ("reward", r),
                             ("truncated", truncated and not terminated), ("free", len(allowed) > 1), ("episode", e)):
                    rec[k].append(v)
                obs, done = nxt, terminated or truncated
            print(f"  {kind} seed {seed}: 에피소드 {e + 1}/{episodes} {env.history[-1]}, 판단 {len(rec['action'])}개", flush=True)
    finally:
        env.close()
    meta = {"map": kind, "seed": seed, "episodes": episodes, "transitions": len(rec["action"]),
            "free_decisions": int(sum(rec["free"])), "scenarios": [list(s) for s in env.history[:episodes]],
            "teacher": {"variant": TEACHER_VARIANT, "horizon_s": TEACHER_HORIZON_S, "code": TEACHER_CODE},
            "environment_fingerprint": environment_fingerprint(), "teacher_code_file": code_fingerprint(),
            "problems": problems, "seconds": round(time.time() - started, 1)}
    if kind == "network12":
        import network_rl as nr
        meta["network12_fingerprint"] = nr._code_fingerprint()
    if any(problems.values()):
        raise SystemExit(f"시범 이상: {problems}(선생님은 허용된 녹색 안에서만 고르므로 0이어야 함)")
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, obs=np.asarray(rec["obs"], dtype=np.float32), next_obs=np.asarray(rec["next_obs"], dtype=np.float32),
                        action=np.asarray(rec["action"], dtype=np.int64), reward=np.asarray(rec["reward"], dtype=np.float32),
                        truncated=np.asarray(rec["truncated"], dtype=bool), free=np.asarray(rec["free"], dtype=bool),
                        episode=np.asarray(rec["episode"], dtype=np.int32), meta=json.dumps(meta, ensure_ascii=False))
    tmp.replace(out)
    return meta


def load_demos(files: list[Path], kind: str) -> tuple[list[dict], list[dict]]:
    """(흉내 내기·다듬기용 에피소드들, 확인용 에피소드들). 에피소드마다 판단 순서 그대로(n-step 계산에 필요)."""
    from focused_data import environment_fingerprint
    train, hold = [], []
    for f in files:
        d = np.load(f)
        meta = json.loads(str(d["meta"]))
        if meta["map"] != kind or meta["teacher"]["code"] != TEACHER_CODE or meta["environment_fingerprint"] != environment_fingerprint():
            raise SystemExit(f"{f}: 이 학습과 맞지 않는 시범입니다({meta['map']}, {meta['teacher']}, {meta['environment_fingerprint']}).")
        for e in np.unique(d["episode"]):
            idx = np.flatnonzero(d["episode"] == e)
            ep_ = {k: d[k][idx] for k in ("obs", "next_obs", "action", "reward", "truncated", "free")}
            (hold if meta["seed"] == HOLDOUT_SEED and e >= HOLDOUT_FROM else train).append(ep_)
    return train, hold


# ------------------------------------------------------------------ 2·3. 학습기
def margin_loss(q, actions, allowed, free, margin: float = MARGIN):
    """max_허용 a [Q(s, a) + margin·(a ≠ 선생님)] − Q(s, 선생님)의 평균(고를 수 있었던 판단만). 없으면 0."""
    import torch as th
    if not bool(free.any()):
        return q.sum() * 0.0
    q, actions, allowed = q[free], actions[free].reshape(-1), allowed[free]
    bonus = th.full_like(q, margin)
    bonus.scatter_(1, actions.unsqueeze(1), 0.0)
    best = (q + bonus).masked_fill(~allowed, -th.inf).max(dim=1).values
    return (best - q.gather(1, actions.unsqueeze(1)).squeeze(1)).mean()


def teacher_dqn_class():
    import torch as th
    import torch.nn.functional as F
    from stable_baselines3.common.type_aliases import ReplayBufferSamples

    Base = explore_dqn.double_hold_aware_dqn_class()

    class TeacherDQN(Base):
        demo_files: list[str] = []          # 학습 명령이 정한다(저장하지 않음)
        demo_map: str = "single"
        imitation_dir: str | None = None
        teacher_pretrained = False          # 저장됨: 이어 학습하면 흉내 내기를 다시 하지 않는다

        def _excluded_save_params(self) -> list[str]:
            return [*super()._excluded_save_params(), "_demo_buffer", "_holdout"]

        # ---------------------------------------------- 시범 버퍼(재생 버퍼와 같은 종류: n-step을 같은 방법으로 계산)
        def _setup_demos(self) -> None:
            train, hold = load_demos([Path(f) for f in self.demo_files], self.demo_map)
            size = sum(len(e["action"]) for e in train)
            self._demo_buffer = self.replay_buffer_class(size, self.observation_space, self.action_space, device=self.device,
                                                         n_envs=1, optimize_memory_usage=False, **self.replay_buffer_kwargs)
            for e in train:
                for i in range(len(e["action"])):
                    self._demo_buffer.add(e["obs"][i][None], e["next_obs"][i][None], np.array([[e["action"][i]]]),
                                          np.array([e["reward"][i]]), np.array([bool(e["truncated"][i])]),
                                          [{"TimeLimit.truncated": bool(e["truncated"][i])}])
            self._holdout = hold

        def _demo_batch(self, k: int) -> ReplayBufferSamples:
            return self._demo_buffer.sample(k, env=self._vec_normalize_env)

        def _losses(self, data: ReplayBufferSamples, demo_rows: "th.Tensor"):
            """TD 손실(모든 줄) + 차이 손실(demo_rows가 참인 줄 가운데 고를 수 있었던 판단)."""
            n = self.action_space.n
            discounts = data.discounts if data.discounts is not None else self.gamma
            with th.no_grad():
                next_q = self.q_net_target(data.next_observations)
                next_values = self.allowed_next_values(next_q, data.next_observations).reshape(-1, 1)
                target = data.rewards + (1 - data.dones) * discounts * next_values
            q = self.q_net(data.observations)
            td = F.smooth_l1_loss(th.gather(q, dim=1, index=data.actions.long()), target)
            allowed = data.observations[:, -n:] > 0.5
            free = demo_rows & (allowed.sum(dim=1) > 1)
            m = margin_loss(q, data.actions.long(), allowed, free)
            return td, m

        def _step(self, loss) -> None:
            self.policy.optimizer.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        def agreement(self, episodes: list[dict]) -> dict:
            """고를 수 있었던 판단에서 학생(허용 녹색 중 Q 최댓값)이 선생님과 같은 녹색을 고른 비율."""
            n = self.action_space.n
            obs = np.concatenate([e["obs"] for e in episodes]) if episodes else np.zeros((0, *self.observation_space.shape))
            act = np.concatenate([e["action"] for e in episodes]) if episodes else np.zeros(0)
            free = np.concatenate([e["free"] for e in episodes]) if episodes else np.zeros(0, dtype=bool)
            if not free.any():
                return {"free_decisions": 0, "agreement": None}
            with th.no_grad():
                x = th.as_tensor(obs[free], device=self.device)
                q = self.q_net(x)
                allowed = x[:, -n:] > 0.5
                pick = q.masked_fill(~allowed, -th.inf).argmax(dim=1).cpu().numpy()
            return {"free_decisions": int(free.sum()), "agreement": round(float((pick == act[free]).mean()), 4)}

        def pretrain(self) -> dict:
            self.policy.set_training_mode(True)
            started, td_log, m_log = time.time(), [], []
            for u in range(1, PRETRAIN_UPDATES + 1):
                data = self._demo_batch(PRETRAIN_BATCH)
                td, m = self._losses(data, th.ones(PRETRAIN_BATCH, dtype=th.bool, device=self.device))
                self._step(td + MARGIN_WEIGHT * m)
                td_log.append(td.item())
                m_log.append(m.item())
                if u % TARGET_EVERY == 0:
                    self.q_net_target.load_state_dict(self.q_net.state_dict())
            self.q_net_target.load_state_dict(self.q_net.state_dict())
            self.teacher_pretrained = True
            report = {"updates": PRETRAIN_UPDATES, "batch": PRETRAIN_BATCH, "margin": MARGIN, "margin_weight": MARGIN_WEIGHT,
                      "seconds": round(time.time() - started, 1), "demo_transitions": int(self._demo_buffer.size()),
                      "td_loss_last1000": round(float(np.mean(td_log[-1000:])), 5),
                      "margin_loss_last1000": round(float(np.mean(m_log[-1000:])), 5),
                      "holdout": self.agreement(self._holdout)}
            if self.imitation_dir:
                folder = Path(self.imitation_dir)
                folder.mkdir(parents=True, exist_ok=True)
                self.save(folder / "dqn_204820")
                (folder / "imitation.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
            print(f"흉내 내기 끝: 확인용 일치율 {report['holdout']}", flush=True)
            return report

        # ---------------------------------------------- 학습
        def learn(self, *args, **kwargs):
            if getattr(self, "_demo_buffer", None) is None:
                self._setup_demos()
            if not self.teacher_pretrained:
                self.pretrain()
            return super().learn(*args, **kwargs)

        def _sample_action(self, learning_starts, action_noise=None, n_envs=1):
            # 학습 전 무작위 탐색 없이 처음부터 정책 + EPSILON 무작위(103.1절). 경사 하강은 learning_starts 뒤에 시작한다.
            return super()._sample_action(0, action_noise, n_envs)

        def train(self, gradient_steps: int, batch_size: int = 100) -> None:
            self.policy.set_training_mode(True)
            self._update_learning_rate(self.policy.optimizer)
            k = int(round(batch_size * DEMO_SHARE))
            losses = []
            for _ in range(gradient_steps):
                own = self.replay_buffer.sample(batch_size - k, env=self._vec_normalize_env)
                demo = self._demo_batch(k)
                data = ReplayBufferSamples(*(th.cat([a, b]) if a is not None else None for a, b in zip(own, demo)))
                rows = th.cat([th.zeros(batch_size - k, dtype=th.bool), th.ones(k, dtype=th.bool)]).to(self.device)
                td, m = self._losses(data, rows)
                loss = td + MARGIN_WEIGHT * m
                self._step(loss)
                losses.append(loss.item())
            self._n_updates += gradient_steps
            self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
            self.logger.record("train/loss", np.mean(losses))

    return TeacherDQN


# ------------------------------------------------------------------ 학습 명령
def _check_settings(kind: str) -> None:
    want = str(N_STEPS[kind])
    if os.environ.get("FOCUSED_DQN_NSTEPS") != want:
        raise SystemExit(f"{kind}의 기준 DQN과 같게 FOCUSED_DQN_NSTEPS={want}로 돌린다.")
    for name in ("FOCUSED_DQN_GAMMA", "FOCUSED_DQN_TARGET_UPDATE", "FOCUSED_DQN_DOUBLE"):
        if os.environ.get(name):
            raise SystemExit(f"{name}은 비워 둔다(Double은 이 학습기에 들어 있다).")
    if os.environ.get("FOCUSED_REWARD", "queue") != "queue":
        raise SystemExit("보상은 기본(queue)이어야 한다.")


def train(kind: str, demos: Path, rest: list[str]) -> None:
    _check_settings(kind)
    import focused_rl
    import focused_train
    files = [demo_file(demos, kind, s) for s in COLLECT_SEEDS]
    missing = [str(f) for f in files if not f.exists()]
    if missing:
        raise SystemExit(f"시범이 없습니다: {missing}")
    value = lambda name, default=None: rest[rest.index(name) + 1] if name in rest else default
    if value("--out") is None:
        raise SystemExit("--out(모델 폴더)이 필요하다.")
    seed_dir = Path(value("--out")) / f"seed_{value('--seed', '42')}"
    record = {"stage": "teacher", "teacher": {"variant": TEACHER_VARIANT, "horizon_s": TEACHER_HORIZON_S, "code": TEACHER_CODE},
              "code": code_fingerprint(), "double_dqn": True, "n_steps": N_STEPS[kind], "epsilon": EPSILON,
              "pretrain_updates": PRETRAIN_UPDATES, "margin": MARGIN, "margin_weight": MARGIN_WEIGHT, "demo_share": DEMO_SHARE,
              "demo_files": {f.name: hashlib.sha256(f.read_bytes()).hexdigest()[:12] for f in files},
              "note": "진행 기록 103절, src/explore_teacher.py"}
    rec_file = seed_dir / "teacher.json"
    if rec_file.exists():
        saved = json.loads(rec_file.read_text())
        if (saved.get("code"), saved.get("demo_files")) != (record["code"], record["demo_files"]):
            raise SystemExit(f"{rec_file}: 이어 학습할 모델의 코드·시범이 지금과 다릅니다. 처음부터 다시 학습한다.")
    seed_dir.mkdir(parents=True, exist_ok=True)
    rec_file.write_text(json.dumps(record, ensure_ascii=False, indent=1))
    focused_train.DQN_SETTINGS.update(exploration_initial_eps=EPSILON, exploration_final_eps=EPSILON)
    cls = teacher_dqn_class()
    cls.demo_files, cls.demo_map, cls.imitation_dir = [str(f) for f in files], kind, str(seed_dir / "imitation")
    focused_rl.hold_aware_dqn_class = lambda: cls
    if kind == "single":
        sys.argv = [sys.argv[0], *rest]
        focused_train.main()
    else:
        import network_rl
        sys.argv = [sys.argv[0], "train", *rest]
        network_rl.main()
    write_imitation_config(seed_dir, record)


def write_imitation_config(seed_dir: Path, record: dict) -> None:
    """'흉내만 낸 DQN'도 기존 평가 도구로 평가하도록 학습 기록을 옮겨 적는다(학습량 0, 단계 표시)."""
    cfg = json.loads((seed_dir / "training_config.json").read_text())
    imitation = seed_dir / "imitation"
    if not (imitation / "dqn_204820.zip").exists():
        raise SystemExit(f"{imitation}: 흉내만 낸 모델이 없습니다.")
    (imitation / "training_config.json").write_text(json.dumps(
        {**cfg, "timesteps": 0, "stage": "imitation_only", "teacher": record}, ensure_ascii=False, indent=2))
    (seed_dir / "teacher.json").write_text(json.dumps({**record, "imitation": json.loads((imitation / "imitation.json").read_text())},
                                                      ensure_ascii=False, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--map", choices=MAPS, required=True)
    c.add_argument("--seed", type=int, choices=COLLECT_SEEDS, required=True)
    c.add_argument("--episodes", type=int, default=EPISODES_PER_SEED)
    c.add_argument("--out", required=True)
    t = sub.add_parser("train")
    t.add_argument("--map", choices=MAPS, required=True)
    t.add_argument("--demos", required=True)
    args, rest = ap.parse_known_args()
    if args.cmd == "collect":
        if args.episodes != EPISODES_PER_SEED:
            raise SystemExit(f"103.1절: 에피소드는 seed마다 {EPISODES_PER_SEED}개다.")
        out = demo_file(Path(args.out), args.map, args.seed)
        if out.exists():
            print(f"이미 있음 → 건너뜀: {out}")
            return
        meta = collect(args.map, args.seed, args.episodes, out)
        print(json.dumps({k: v for k, v in meta.items() if k != "scenarios"}, ensure_ascii=False))
    else:
        train(args.map, Path(args.demos), rest)


if __name__ == "__main__":
    main()
