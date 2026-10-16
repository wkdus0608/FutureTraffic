"""DQN 신경망의 크기·모양이 모자라는지 보는 진단. 학습 기록만 쓰고 SUMO는 돌리지 않는다(환경·평가 지문 밖).

학습 때 저장한 경험 기록(replay_buffer.pkl, 10만 단계, 시간 순서)에서 판단마다 그 뒤 실제로 받은 할인 누적 점수
(몬테카를로 반환, 할인율 0.99 = DQN과 같음)를 계산한다. 그다음 (관측 64개, 고른 녹색)에서 그 값을 맞히는 문제를
여러 신경망에 같은 자료·같은 방법으로 풀게 하고, 학습에 쓰지 않은 에피소드(검증)에서 얼마나 맞히는지(R²) 비교한다.

- 큰 망이 검증 R²를 뚜렷이 올리면 → 지금 망(64-64)이 모자랄 수 있다(실제 학습으로 확인할 가치가 있다)
- 비슷하면 → 크기는 병목이 아니다
- 선형(은닉층 없음)이 훨씬 낮으면 → 은닉층(비선형)이 필요하다 = MLP를 쓰는 이유
- 같은 문제를 학습된 DQN의 Q값으로 맞히면(선형 보정) 그 모델이 실제 결과의 순서를 얼마나 아는지 본다
- 학습된 DQN의 Q값이 '몇 단계 앞까지 더한 점수'와 가장 가까운지 본다(DQN이 실제로 배운 시야)

자료를 고르는 방법
- 탐험이 거의 끝난 3만 단계 뒤의 에피소드만 쓴다(행동 정책이 거의 고정 — ε 0.05).
- 각 에피소드(720단계)의 앞 360단계만 쓴다. 에피소드가 60분에서 잘려 끝 쪽 반환은 덜 더해졌기 때문이다
  (360단계 뒤 할인 0.99^360 ≈ 0.027).
- 학습·검증은 에피소드 단위로 나눈다(80% / 20%). 이웃한 판단끼리 매우 비슷해서, 판단 단위로 나누면 검증이 부풀려진다.

한계: 이것은 '가치를 얼마나 잘 나타내는가'를 보는 진단이다. 실제 제어 성능(차량당 지연)은 같은 조건의 학습·평가로만 확정한다.

사용: .venv/bin/python src/focused_capacity.py            (Drive models_v5/seed_42~46의 기록을 찾아 쓴다)
결과: results/focused_204820/capacity/capacity_<날짜>.json, .md
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

GAMMA = 0.99
ARCHS = {"선형(은닉층 없음)": [], "32-32": [32, 32], "64-64 (지금)": [64, 64], "128-128": [128, 128],
         "256-256": [256, 256], "64-64-64 (층 3개)": [64, 64, 64], "256-256-256": [256, 256, 256]}
STATE_ONLY = "64-64, 고른 녹색을 모름"     # 행동이 반환을 얼마나 바꾸는지 보려고(입력에서 행동을 뺀 비교용)
OUT = Path(__file__).resolve().parents[1] / "results/focused_204820/capacity"


def default_buffers() -> list[Path]:
    found = []
    for drive in sorted(Path.home().glob("Library/CloudStorage/GoogleDrive-*")):
        base = drive / "내 드라이브" / "FutureTraffic" / "models_v5"
        found += sorted(base.glob("seed_*/checkpoint/replay_buffer.pkl"))
    return found


def episodes_from_buffer(path: Path, start_step: int, keep: int):
    """(관측, 행동, 반환, 에피소드 번호) — 끝까지 끝난 에피소드만, 앞 keep단계만."""
    from stable_baselines3.common.save_util import load_from_pkl
    rb = load_from_pkl(str(path))
    n = rb.buffer_size if rb.full else rb.pos
    obs, act = rb.observations[:n, 0], rb.actions[:n, 0, 0].astype(np.int64)
    rew, done = rb.rewards[:n, 0].astype(np.float64), rb.dones[:n, 0]
    ends = np.flatnonzero(done)
    X, A, G, E = [], [], [], []
    begin = 0
    for e, end in enumerate(ends):
        if begin >= start_step and end - begin + 1 > keep:
            g = np.zeros(end - begin + 1)
            acc = 0.0
            for t in range(end, begin - 1, -1):
                acc = rew[t] + GAMMA * acc
                g[t - begin] = acc
            X.append(obs[begin:begin + keep]); A.append(act[begin:begin + keep]); G.append(g[:keep])
            E.append(np.full(keep, e))
        begin = end + 1
    return np.concatenate(X), np.concatenate(A), np.concatenate(G), np.concatenate(E)


def build(hidden: list[int], n_in: int, n_out: int):
    import torch.nn as nn
    layers, d = [], n_in
    for h in hidden:
        layers += [nn.Linear(d, h), nn.ReLU()]
        d = h
    return nn.Sequential(*layers, nn.Linear(d, n_out))


def fit(hidden, Xtr, Atr, Ytr, Xva, Ava, Yva, seed: int, max_epochs: int, patience: int,
        state_only: bool = False) -> dict:
    """검증 R²: 전체, 그리고 실제로 고를 수 있었던 판단(실행 가능한 녹색이 2개 이상)만."""
    import torch
    torch.manual_seed(seed)
    net = build(hidden, Xtr.shape[1], 1 if state_only else 4)
    pick = (lambda out, a: out[:, 0]) if state_only else (lambda out, a: out.gather(1, a[:, None])[:, 0])
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    xt, at, yt = torch.tensor(Xtr), torch.tensor(Atr), torch.tensor(Ytr, dtype=torch.float32)
    xv, av, yv = torch.tensor(Xva), torch.tensor(Ava), torch.tensor(Yva, dtype=torch.float32)
    gen = torch.Generator().manual_seed(seed)

    def mse(x, a, y):
        with torch.no_grad():
            return float(((pick(net(x), a) - y) ** 2).mean())

    best, best_state, bad, epochs = float("inf"), None, 0, 0
    for epoch in range(max_epochs):
        perm = torch.randperm(len(xt), generator=gen)
        for i in range(0, len(xt), 256):
            b = perm[i:i + 256]
            loss = ((pick(net(xt[b]), at[b]) - yt[b]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        v = mse(xv, av, yv)
        epochs = epoch + 1
        if v < best - 1e-5:
            best, bad = v, 0
            best_state = {k: t.clone() for k, t in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(best_state)
    choice = xv[:, -4:].sum(1) > 1.5         # 관측 끝 4칸 = 그대로 실행될 녹색 표시
    return {"val_r2": 1 - mse(xv, av, yv) / float(yv.var(unbiased=False)),
            "val_r2_choice": 1 - mse(xv[choice], av[choice], yv[choice]) / float(yv[choice].var(unbiased=False)),
            "train_r2": 1 - mse(xt, at, yt) / float(yt.var(unbiased=False)),
            "epochs": epochs, "weights": sum(p.numel() for p in net.parameters()),
            "choice_share": float(choice.float().mean())}


def trained_dqn_r2(model_zip: Path, Xva, Ava, Yva) -> float | None:
    """학습된 DQN의 Q(s, a)를 선형 보정(a·Q + b)해 검증 반환을 맞힐 때의 R². 모델이 결과의 순서를 아는 정도."""
    if not model_zip.exists():
        return None
    import torch
    from stable_baselines3 import DQN
    model = DQN.load(model_zip, device="cpu")
    with torch.no_grad():
        q = model.policy.q_net(torch.tensor(Xva)).gather(1, torch.tensor(Ava)[:, None])[:, 0].numpy().astype(np.float64)
    coef = np.polyfit(q, Yva, 1)
    pred = np.polyval(coef, q)
    return float(1 - ((pred - Yva) ** 2).mean() / Yva.var())


HORIZONS = [5, 10, 15, 20, 25, 30, 40, 60, 100, 360]


def horizon_match(path: Path, start_step: int, keep: int) -> dict:
    """학습된 DQN의 Q(s, a)가 '몇 단계 앞까지 더한 할인 점수'와 가장 가까운지.

    DQN은 목표망을 바꿀 때마다 벨만 갱신을 대략 한 번 하는 셈이라, 목표망 교체가 N번이면 Q는 대략 N단계 앞까지의
    점수만 담는다(처음 Q가 0 근처에서 시작할 때). 할인율 0.99가 뜻하는 시야(약 100단계)를 다 배웠는지 본다.
    """
    import torch
    from stable_baselines3 import DQN
    from stable_baselines3.common.save_util import load_from_pkl
    rb = load_from_pkl(str(path))
    n = rb.buffer_size if rb.full else rb.pos
    obs, act = rb.observations[:n, 0], rb.actions[:n, 0, 0].astype(np.int64)
    rew, done = rb.rewards[:n, 0].astype(np.float64), rb.dones[:n, 0]
    model = DQN.load(path.parent.parent / "dqn_204820.zip", device="cpu")
    qs, gs = [], {k: [] for k in HORIZONS}
    begin = 0
    for end in np.flatnonzero(done):
        if begin >= start_step and end - begin + 1 > keep:
            r = rew[begin:end + 1]
            L = len(r)
            disc = GAMMA ** np.arange(L)
            prefix = np.concatenate([[0.0], np.cumsum(r * disc)])
            t = np.arange(keep)
            for k in HORIZONS:
                gs[k].append((prefix[np.minimum(t + k, L)] - prefix[t]) / disc[t])
            with torch.no_grad():
                q = model.policy.q_net(torch.tensor(obs[begin:begin + keep])).numpy()
            qs.append(q[t, act[begin:begin + keep]])
        begin = end + 1
    q = np.concatenate(qs)
    g = {k: np.concatenate(v) for k, v in gs.items()}
    mse = {k: float(((g[k] - q) ** 2).mean()) for k in HORIZONS}
    best = min(mse, key=mse.get)
    return {"q_mean": float(q.mean()), "g_mean": {k: float(v.mean()) for k, v in g.items()}, "best_k": best,
            "corr_best": float(np.corrcoef(q, g[best])[0, 1]), "corr_full": float(np.corrcoef(q, g[HORIZONS[-1]])[0, 1])}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--buffers", nargs="*", default=None)
    p.add_argument("--start-step", type=int, default=30_000)
    p.add_argument("--keep", type=int, default=360)
    p.add_argument("--max-epochs", type=int, default=300)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--threads", type=int, default=2)
    args = p.parse_args()
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from focused_sumo import configure_sumo
    configure_sumo()
    import torch
    torch.set_num_threads(args.threads)
    buffers = [Path(b) for b in args.buffers] if args.buffers else default_buffers()
    if not buffers:
        raise SystemExit("경험 기록(replay_buffer.pkl)을 찾지 못했습니다. --buffers로 주세요.")
    results: dict[str, list] = {}
    dqn_r2, data_info = [], []
    for k, path in enumerate(buffers):
        X, A, G, E = episodes_from_buffer(path, args.start_step, args.keep)
        eps = np.unique(E)
        rng = np.random.default_rng(k)
        val_eps = set(rng.choice(eps, size=max(1, round(0.2 * len(eps))), replace=False).tolist())
        va = np.array([e in val_eps for e in E])
        mu, sd = G[~va].mean(), G[~va].std()
        Y = ((G - mu) / sd).astype(np.float32)
        data_info.append({"buffer": str(path), "episodes": int(len(eps)), "train": int((~va).sum()), "val": int(va.sum()),
                          "return_mean": float(mu), "return_sd": float(sd)})
        print(f"[{k + 1}/{len(buffers)}] {path.parent.parent.name}: 에피소드 {len(eps)}개(검증 {len(val_eps)}개), "
              f"판단 {len(G):,}개, 반환 평균 {mu:.1f} ± {sd:.1f}", flush=True)
        for name, hidden in list(ARCHS.items()) + [(STATE_ONLY, [64, 64])]:
            t0 = time.time()
            r = fit(hidden, X[~va], A[~va], Y[~va], X[va], A[va], Y[va], seed=k, max_epochs=args.max_epochs,
                    patience=args.patience, state_only=name == STATE_ONLY)
            results.setdefault(name, []).append(r)
            print(f"    {name:18s} 검증 R² {r['val_r2']:.3f}  고를 수 있던 판단만 {r['val_r2_choice']:.3f} "
                  f"(검증의 {r['choice_share']:.0%})  학습 R² {r['train_r2']:.3f}  ({r['epochs']} 에폭, "
                  f"{time.time() - t0:.0f}초)", flush=True)
        q = trained_dqn_r2(path.parent.parent / "dqn_204820.zip", X[va], A[va], Y[va].astype(np.float64))
        if q is not None:
            dqn_r2.append(q)
            print(f"    {'학습된 DQN(64-64)':18s} 검증 R² {q:.3f}  (Q값을 선형 보정)", flush=True)
    horizon = []
    for path in buffers:
        if (path.parent.parent / "dqn_204820.zip").exists():
            h = horizon_match(path, args.start_step, args.keep)
            horizon.append({"buffer": str(path), **h})
            print(f"{path.parent.parent.name}: Q 평균 {h['q_mean']:.0f}, 가장 가까운 시야 {h['best_k']}단계"
                  f"(상관 {h['corr_best']:.3f}, 전체 반환과는 {h['corr_full']:.3f})", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d")
    summary = {name: {"val_r2_mean": float(np.mean([r["val_r2"] for r in rs])), "val_r2_sd": float(np.std([r["val_r2"] for r in rs])),
                      "val_r2_choice_mean": float(np.mean([r["val_r2_choice"] for r in rs])),
                      "val_r2_choice_sd": float(np.std([r["val_r2_choice"] for r in rs])),
                      "train_r2_mean": float(np.mean([r["train_r2"] for r in rs])), "weights": rs[0]["weights"]}
               for name, rs in results.items()}
    (OUT / f"capacity_{stamp}.json").write_text(json.dumps(
        {"settings": vars(args), "data": data_info, "summary": summary, "per_buffer": results,
         "trained_dqn_val_r2": dqn_r2, "horizon": horizon}, ensure_ascii=False, indent=1))
    lines = [f"| 신경망 | 가중치 | 검증 R²(평균 ± 표준편차, 기록 {len(buffers)}개) | 고를 수 있던 판단만 | 학습 R² |",
             "|---|---:|---:|---:|---:|"]
    for name, s in summary.items():
        lines.append(f"| {name} | {s['weights']:,} | {s['val_r2_mean']:.3f} ± {s['val_r2_sd']:.3f} | "
                     f"{s['val_r2_choice_mean']:.3f} ± {s['val_r2_choice_sd']:.3f} | {s['train_r2_mean']:.3f} |")
    if dqn_r2:
        lines.append(f"| 학습된 DQN(64-64)의 Q값(선형 보정) | 8,580 | {np.mean(dqn_r2):.3f} ± {np.std(dqn_r2):.3f} | — | — |")
    if horizon:
        lines += ["", "| 학습 기록 | DQN Q값 평균 | " + " | ".join(f"{k}단계 점수" for k in HORIZONS) + " | Q와 가장 가까운 시야 |",
                  "|---|---:|" + "---:|" * len(HORIZONS) + "---|"]
        for h in horizon:
            lines.append(f"| {Path(h['buffer']).parent.parent.name} | {h['q_mean']:.0f} | "
                         + " | ".join(f"{h['g_mean'][k]:.0f}" for k in HORIZONS)
                         + f" | {h['best_k']}단계(상관 {h['corr_best']:.3f}) |")
    (OUT / f"capacity_{stamp}.md").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
