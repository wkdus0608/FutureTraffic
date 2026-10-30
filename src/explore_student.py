"""[추가 탐색] 학생 재평가 — 학습 길이·바꿀 문턱·실행 정책을 먼저 고정(진행 기록 131절). 지문 밖, 저장된 자료만 읽음(SUMO 없음).

127절(explore_distill.py)과 같은 자료·목표·MLP·3분할·지표를 쓰되, 다음을 미리 정했다.
- 실행 정책: 시작 seed 0~4 다섯 모델의 예측 평균. 예측 최소 후보 b의 예측 이득(P0 선택 − b)이 문턱 τ보다 클 때만 b, 아니면 P0
  (P0+신경망 혼합 제어; 좌회전 정체 시점에서만 고름).
- 후보: 학습 길이 500·2,000·10,000번 갱신 × τ 0·500·1,000·2,000대·초, 그리고 바꾸지 않음(P0).
- 고르는 법: 평가할 묶음을 빼고 남은 두 묶음으로 안쪽 2분할(서로 하나씩 빼 봄), 안쪽 이득 합이 가장 큰 것(같으면 τ 큰 쪽, 그다음
  짧은 쪽). 가장 큰 값이 0 이하이면 P0. 고른 학습 길이로 두 묶음 전체에 다시 학습하고, 그 가중치를 저장한다(연속 제어용).

사용: .venv/bin/python src/explore_student.py --runs results/explore/online_20261023/runs --out results/explore/student_20261024
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

import explore_distill as ed

PROJECT = ed.PROJECT
STEPS = (500, 2_000, 10_000)
THRESHOLDS = (0.0, 500.0, 1_000.0, 2_000.0)          # 대·초
SEEDS = (0, 1, 2, 3, 4)


def code_fingerprint() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]


# ------------------------------------------------------------------ 정책
def hybrid(pred: np.ndarray, sample: dict, tau: float) -> int:
    """예측 최소 허용 후보(같으면 P0, 그다음 작은 번호)를, 예측 이득이 τ(대·초)보다 클 때만 고른다. τ = 0이면 127절 규칙 F와 같다."""
    best = ed.rule_f(pred, sample)
    return best if (pred[sample["p0"]] - pred[best]) * ed.SCALE > tau else sample["p0"]


def gain(samples: list[dict], picks: list[int]) -> float:
    """P0 선택 대비 줄인 비용의 합(8개 평균 비용, 반올림 없음)."""
    return sum(ed.mean_cost(s, s["p0"]) - ed.mean_cost(s, p) for s, p in zip(samples, picks))


def select(scores: dict) -> tuple[int, float] | None:
    """안쪽 점수가 가장 큰 (학습 길이, τ). 같으면 τ 큰 쪽, 그다음 학습 길이 짧은 쪽. 가장 큰 값이 0 이하이면 None(P0)."""
    (steps, tau), best = max(scores.items(), key=lambda kv: (kv[1], kv[0][1], -kv[0][0]))
    return (steps, tau) if best > 0 else None


# ------------------------------------------------------------------ 모델
def make_net():
    import torch as th
    layers, width = [], 72
    for h in ed.MLP_HIDDEN:
        layers += [th.nn.Linear(width, h), th.nn.ReLU()]
        width = h
    return th.nn.Sequential(*layers, th.nn.Linear(width, ed.N_GREENS))


def train(X: np.ndarray, Y: np.ndarray, seed: int, steps=STEPS) -> dict:
    """127절과 같은 학습(전체 자료, 허용된 녹색만 제곱 오차)을 가장 긴 길이까지 한 번 돌리고, 길이마다 가중치를 떼어 둔다."""
    import torch as th
    th.manual_seed(seed)
    net = make_net()
    opt = th.optim.Adam(net.parameters(), lr=ed.MLP_LR, weight_decay=ed.MLP_DECAY)
    x, mask, y = th.as_tensor(X), th.as_tensor(~np.isnan(Y)), th.as_tensor(np.nan_to_num(Y))
    out = {}
    for i in range(1, max(steps) + 1):
        opt.zero_grad()
        loss = (((net(x) - y) ** 2) * mask).sum() / mask.sum()
        loss.backward()
        opt.step()
        if i in steps:
            out[i] = {k: v.detach().clone() for k, v in net.state_dict().items()}
    return out


def predict(states: list[dict], X: np.ndarray) -> np.ndarray:
    """모델 여러 개의 예측 평균(앙상블)."""
    import torch as th
    preds = []
    for state in states:
        net = make_net()
        net.load_state_dict(state)
        net.eval()
        with th.no_grad():
            preds.append(net(th.as_tensor(X)).numpy())
    return np.mean(preds, axis=0)


def fit_ensemble(train_set: list[dict], steps=STEPS, seeds=SEEDS) -> dict:
    """길이마다 seed별 가중치 목록 {길이: [state, ...]}."""
    X, Y = np.stack([s["obs"] for s in train_set]), np.stack([ed.targets(s) for s in train_set])
    per_seed = [train(X, Y, seed, steps) for seed in seeds]
    return {n: [w[n] for w in per_seed] for n in steps}


def picks_for(states: list[dict], samples: list[dict], tau: float) -> list[int]:
    pred = predict(states, np.stack([s["obs"] for s in samples]))
    return [hybrid(p, s, tau) for p, s in zip(pred, samples)]


# ------------------------------------------------------------------ 평가
def inner_scores(train_set: list[dict], steps=STEPS, thresholds=THRESHOLDS, seeds=SEEDS) -> dict:
    """안쪽 2분할: 남은 묶음을 하나씩 빼고 학습해, 뺀 묶음에서 잰 이득의 합."""
    scores = {(n, tau): 0.0 for n in steps for tau in thresholds}
    for v in sorted({s["group"] for s in train_set}):
        fit_on, val = [s for s in train_set if s["group"] != v], [s for s in train_set if s["group"] == v]
        ens = fit_ensemble(fit_on, steps, seeds)
        for n in steps:
            for tau in thresholds:
                scores[(n, tau)] += gain(val, picks_for(ens[n], val, tau))
    return scores


def cross_validate(samples: list[dict], out: Path | None = None, steps=STEPS, thresholds=THRESHOLDS, seeds=SEEDS) -> dict:
    import torch as th
    groups = sorted({s["group"] for s in samples})
    served_by = ed.left_greens()
    folds, order = {}, []
    for g in groups:
        train_set, test = [s for s in samples if s["group"] != g], [s for s in samples if s["group"] == g]
        scores = inner_scores(train_set, steps, thresholds, seeds)
        choice = select(scores)
        ens = fit_ensemble(train_set, steps, seeds)
        picks = {"student": picks_for(ens[choice[0]], test, choice[1]) if choice else [s["p0"] for s in test],
                 "left_first": [ed.left_first(s, served_by) for s in test], "p0": [s["p0"] for s in test]}
        reference = {f"{n}_{int(tau)}": ed.evaluate(test, picks_for(ens[n], test, tau))["gain"] for n in steps for tau in thresholds}
        folds[g] = {"train_groups": sorted({s["group"] for s in train_set}), "train": len(train_set), "test": len(test),
                    "inner_scores": {f"{n}_{int(tau)}": round(v, 1) for (n, tau), v in scores.items()},
                    "choice": {"steps": choice[0], "tau": choice[1]} if choice else None,
                    **{name: ed.evaluate(test, p) for name, p in picks.items()},
                    "reference_gain_not_for_judgment": reference}
        order.append((test, picks))
        if out is not None:
            d = out / "policies" / g
            d.mkdir(parents=True, exist_ok=True)
            if choice:
                for seed, state in zip(seeds, ens[choice[0]]):
                    th.save(state, d / f"seed{seed}.pt")
            (d / "policy.json").write_text(json.dumps({"group": g, "train_groups": folds[g]["train_groups"], "choice": folds[g]["choice"],
                                                       "seeds": list(seeds), "code": code_fingerprint()}, ensure_ascii=False, indent=1))
    total = {name: ed.evaluate([s for t, _ in order for s in t], [p for _, pk in order for p in pk[name]])
             for name in ("student", "left_first", "p0")}
    return {"groups": groups, "folds": folds, "total": total}


def reading(total: dict, folds: dict) -> dict:
    """131절 읽는 법(126절과 같은 기준을 고정된 정책에)."""
    m = total["student"]
    positive_folds = sum(f["student"]["gain"] > 0 for f in folds.values())
    beats_rule = m["gain"] > total["left_first"]["gain"]
    share = m["share"] or 0.0
    if m["gain"] > 0 and positive_folds >= 2 and share >= ed.SHARE_MIN and beats_rule:
        word = "배울 수 있어 보인다(혼합 제어 후보)"
    elif m["gain"] <= 0 or not beats_rule:
        word = "이 자료로는 배우지 못했다"
    else:
        word = "불분명"
    return {"word": word, "gain": m["gain"], "share": m["share"], "positive_folds": positive_folds,
            "left_first_gain": total["left_first"]["gain"], "choices": {g: f["choice"] for g, f in folds.items()}}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs", default=str(PROJECT / "results/explore/online_20261023/runs"))
    ap.add_argument("--out", default=str(PROJECT / "results/explore/student_20261024"))
    args = ap.parse_args()
    out = Path(args.out)
    if (out / "result.json").exists():
        raise SystemExit(f"이미 있음: {out / 'result.json'} (131절: 한 차례만 잰다)")
    samples = ed.load(Path(args.runs))
    cv = cross_validate(samples, out)
    result = {"code": code_fingerprint(), "distill_code": ed.code_fingerprint(), "samples": len(samples),
              "steps": list(STEPS), "thresholds": list(THRESHOLDS), "seeds": list(SEEDS), **cv}
    result["reading"] = reading(cv["total"], cv["folds"])
    (out / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=1))
    print(json.dumps(result["reading"], ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
