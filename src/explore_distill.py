"""[추가 탐색] 2단계 첫걸음 — 작은 신경망이 탐색의 선택을 배울 수 있나(진행 기록 126절). 지문 밖, 저장된 자료만 읽음(SUMO 없음).

자료: 125절 탐색 실행의 탐색 시점(decisions.jsonl의 searched). 시점마다 관측 72칸, 허용 녹색, P0·탐색의 선택, 후보마다 시나리오 8개의
600초 비용. 신경망은 관측 72칸 → 녹색 4개의 상대 비용(후보 평균 비용 − 허용 후보 평균, ÷ 1,000)을 배운다(허용된 녹색만 손실).
모델: MLP 72-32-32-4(주, 시작 seed 5개) · 릿지(비교). 규칙: R(P0 위에서 고침, 주) · F(자유). 기준: P0, 탐색(위쪽), 좌회전 우선 규칙.
평가: 날짜·시간대 묶음 하나(두 seed)를 빼는 3분할. 읽는 법은 126절.

사용: .venv/bin/python src/explore_distill.py --runs results/explore/online_20261023/runs --out results/explore/distill_20261024
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

N_GREENS = 4
MARK = "600"
SCALE = 1000.0
MLP_HIDDEN, MLP_LR, MLP_DECAY, MLP_EPOCHS, MLP_SEEDS = (32, 32), 1e-3, 1e-3, 500, (0, 1, 2, 3, 4)
RIDGE_ALPHA = 1.0
SHARE_MIN = 0.25
LEFT_CUT = {"left_east": 8, "left_south": 7}         # 123절 좌회전 정체 조건과 같음
AMBIGUOUS_SE = 2.0


def code_fingerprint() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]


# ------------------------------------------------------------------ 자료
def load(runs: Path) -> list[dict]:
    """탐색 실행(*_search)의 탐색 시점. 묶음 = 날짜_시간대(두 seed를 같은 묶음으로)."""
    out = []
    for f in sorted(Path(runs).glob("*_search/decisions.jsonl")):
        date, period, seed = f.parent.name.split("_")[:3]
        for line in f.read_text().splitlines():
            r = json.loads(line)
            if not r["searched"]:
                continue
            costs = {int(c): [x[MARK]["cost"] for x in v] for c, v in r["scores"].items()}
            out.append({"group": f"{date}_{period}", "seed": int(seed[1:]), "t0": r["t0"], "obs": np.asarray(r["obs72"], dtype=np.float32),
                        "allowed": list(r["allowed"]), "p0": int(r["p0_action"]), "search": int(r["chosen"]), "costs": costs,
                        **{k: r[k] for k in ("left_east", "left_south", "left_east_served", "left_south_served")}})
    return out


def targets(sample: dict) -> np.ndarray:
    """녹색 4개의 상대 비용(허용 안 된 녹색은 nan)."""
    means = {c: statistics.fmean(v) for c, v in sample["costs"].items()}
    center = statistics.fmean(means.values())
    y = np.full(N_GREENS, np.nan, dtype=np.float32)
    for c, m in means.items():
        y[c] = (m - center) / SCALE
    return y


def mean_cost(sample: dict, c: int) -> float:
    return statistics.fmean(sample["costs"][c])


def ambiguous(sample: dict) -> bool:
    """탐색 선택과 P0 선택의 비용 차이가 그 차이의 표준오차(시나리오 짝) 2배 미만이면 '구별 어려움'. 둘이 같으면 아님."""
    if sample["search"] == sample["p0"]:
        return False
    d = [a - b for a, b in zip(sample["costs"][sample["p0"]], sample["costs"][sample["search"]])]
    se = statistics.stdev(d) / math.sqrt(len(d))
    return abs(statistics.fmean(d)) < AMBIGUOUS_SE * se


# ------------------------------------------------------------------ 정책
def left_greens() -> dict:
    """좌회전마다 그 좌회전을 보호 녹색(G)으로 주는 녹색 번호(실제 운영 신호의 녹색 순서 = DQN 행동 순서)."""
    from explore_rollout import LEFTS
    from focused_rl import MOVEMENT_LINKS
    from focused_scenario import signal_greens
    greens = signal_greens()
    return {name: [g for g, s in enumerate(greens) if any(s[l] == "G" for l in MOVEMENT_LINKS[m])] for name, m in LEFTS.items()}


def left_first(sample: dict, served_by: dict) -> int:
    """126절 좌회전 우선 규칙: 조건을 만족하는 좌회전(둘이면 기다리는 차가 많은 쪽)을 주는 허용 녹색, 없으면 P0."""
    want = sorted((n for n, cut in LEFT_CUT.items() if not sample[f"{n}_served"] and sample[n] >= cut), key=lambda n: -sample[n])
    for n in want:
        ok = [g for g in served_by[n] if g in sample["allowed"]]
        if ok:
            return ok[0]
    return sample["p0"]


def rule_r(pred: np.ndarray, sample: dict) -> int:
    """규칙 R: P0의 선택보다 예측 상대 비용이 작은 허용 후보가 있으면 그중 가장 작은 것, 없으면 P0."""
    better = [c for c in sample["allowed"] if pred[c] < pred[sample["p0"]]]
    return min(better, key=lambda c: pred[c]) if better else sample["p0"]


def rule_f(pred: np.ndarray, sample: dict) -> int:
    """규칙 F: 예측 상대 비용이 가장 작은 허용 후보(같으면 P0)."""
    low = min(pred[c] for c in sample["allowed"])
    return sample["p0"] if pred[sample["p0"]] <= low + 1e-9 else min(c for c in sample["allowed"] if pred[c] <= low + 1e-9)


# ------------------------------------------------------------------ 모델
def fit_ridge(X: np.ndarray, Y: np.ndarray, alpha: float = RIDGE_ALPHA):
    """녹색마다 따로 릿지(허용된 시점만). 돌려주는 것: 예측 함수."""
    Xb = np.hstack([X, np.ones((len(X), 1), dtype=np.float32)])
    W = np.zeros((Xb.shape[1], N_GREENS), dtype=np.float64)
    for g in range(N_GREENS):
        rows = ~np.isnan(Y[:, g])
        if rows.sum() == 0:
            continue
        A = Xb[rows].astype(np.float64)
        reg = alpha * np.eye(A.shape[1])
        reg[-1, -1] = 0.0                                 # 절편은 벌하지 않음
        W[:, g] = np.linalg.solve(A.T @ A + reg, A.T @ Y[rows, g].astype(np.float64))
    return lambda Xn: np.hstack([Xn, np.ones((len(Xn), 1), dtype=np.float32)]) @ W


def fit_mlp(X: np.ndarray, Y: np.ndarray, seed: int):
    import torch as th
    th.manual_seed(seed)
    layers, width = [], X.shape[1]
    for h in MLP_HIDDEN:
        layers += [th.nn.Linear(width, h), th.nn.ReLU()]
        width = h
    net = th.nn.Sequential(*layers, th.nn.Linear(width, N_GREENS))
    opt = th.optim.Adam(net.parameters(), lr=MLP_LR, weight_decay=MLP_DECAY)
    x = th.as_tensor(X)
    mask = th.as_tensor(~np.isnan(Y))
    y = th.as_tensor(np.nan_to_num(Y))
    for _ in range(MLP_EPOCHS):
        opt.zero_grad()
        loss = (((net(x) - y) ** 2) * mask).sum() / mask.sum()
        loss.backward()
        opt.step()
    net.eval()

    def predict(Xn):
        with th.no_grad():
            return net(th.as_tensor(Xn)).numpy()
    return predict


# ------------------------------------------------------------------ 평가
def evaluate(samples: list[dict], picks: list[int]) -> dict:
    """정책의 선택(picks)을 126절 지표로 잰다(대·초, 8개 평균 비용 기준)."""
    gain = sum(mean_cost(s, s["p0"]) - mean_cost(s, p) for s, p in zip(samples, picks))
    search_gain = sum(mean_cost(s, s["p0"]) - mean_cost(s, s["search"]) for s in samples)
    differ = [(s, p) for s, p in zip(samples, picks) if s["search"] != s["p0"]]
    agree = [(s, p) for s, p in zip(samples, picks) if s["search"] == s["p0"]]
    wrong = [(s, p, mean_cost(s, p) - mean_cost(s, s["search"])) for s, p in zip(samples, picks) if p != s["search"]]
    part = lambda rows: {"n": len(rows), "mean_regret": round(statistics.fmean(r for *_, r in rows), 1) if rows else 0.0,
                         "max_regret": round(max((r for *_, r in rows), default=0.0), 1)}
    return {"decisions": len(samples), "gain": round(gain, 1), "search_gain": round(search_gain, 1),
            "share": round(gain / search_gain, 3) if search_gain > 0 else None,
            "search_differs": len(differ), "same_as_search_when_differs": sum(p == s["search"] for s, p in differ),
            "differs_from_p0_when_search_differs": sum(p != s["p0"] for s, p in differ),
            "search_agrees": len(agree), "needless_changes": sum(p != s["p0"] for s, p in agree),
            "needless_cost": round(sum(mean_cost(s, p) - mean_cost(s, s["p0"]) for s, p in agree if p != s["p0"]), 1),
            "picks_differ_from_p0": sum(p != s["p0"] for s, p in zip(samples, picks)),
            "wrong_all": part(wrong), "wrong_ambiguous": part([w for w in wrong if ambiguous(w[0])]),
            "wrong_clear": part([w for w in wrong if not ambiguous(w[0])])}


def cross_validate(samples: list[dict]) -> dict:
    groups = sorted({s["group"] for s in samples})
    served_by = left_greens()
    policies = {"mlp_R": [], "mlp_F": [], "ridge_R": [], "ridge_F": [], "left_first": [], "p0": []}
    by_fold = {}
    order = []
    for g in groups:
        train = [s for s in samples if s["group"] != g]
        test = [s for s in samples if s["group"] == g]
        X, Y = np.stack([s["obs"] for s in train]), np.stack([targets(s) for s in train])
        Xt = np.stack([s["obs"] for s in test])
        preds_mlp = [fit_mlp(X, Y, seed)(Xt) for seed in MLP_SEEDS]
        ridge = fit_ridge(X, Y)(Xt)
        fold = {"train": len(train), "test": len(test)}
        picks = {"ridge_R": [rule_r(p, s) for p, s in zip(ridge, test)], "ridge_F": [rule_f(p, s) for p, s in zip(ridge, test)],
                 "left_first": [left_first(s, served_by) for s in test], "p0": [s["p0"] for s in test]}
        for name in picks:
            fold[name] = evaluate(test, picks[name])
        for rule, fn in (("R", rule_r), ("F", rule_f)):
            per_seed = [evaluate(test, [fn(p, s) for p, s in zip(pm, test)]) for pm in preds_mlp]
            fold[f"mlp_{rule}"] = {"per_seed_gain": [e["gain"] for e in per_seed], **average(per_seed)}
        by_fold[g] = fold
        order.append((test, picks))
    total = {}
    for name in ("ridge_R", "ridge_F", "left_first", "p0"):
        total[name] = evaluate([s for t, _ in order for s in t], [p for _, pk in order for p in pk[name]])
    for name in ("mlp_R", "mlp_F"):                       # 분할마다 seed 5개 평균을 낸 뒤 분할을 합친다
        total[name] = {"per_fold_gain": [by_fold[g][name]["gain"] for g in groups], **sum_folds([by_fold[g][name] for g in groups])}
    return {"groups": groups, "folds": by_fold, "total": total}


def average(evals: list[dict]) -> dict:
    """seed별 평가의 평균(숫자 항목만, 중첩 dict는 항목별 평균)."""
    out = {}
    for k, v in evals[0].items():
        if isinstance(v, dict):
            out[k] = average([e[k] for e in evals])
        elif isinstance(v, (int, float)) and v is not None:
            vals = [e[k] for e in evals if e[k] is not None]
            out[k] = round(statistics.fmean(vals), 3) if vals else None
        else:
            out[k] = v
    return out


def sum_folds(folds: list[dict]) -> dict:
    """분할별 (seed 평균) 지표를 합친다: 수·합은 더하고, 몫은 합으로 다시 계산한다."""
    keys_sum = ("decisions", "gain", "search_gain", "search_differs", "same_as_search_when_differs", "differs_from_p0_when_search_differs",
                "search_agrees", "needless_changes", "needless_cost", "picks_differ_from_p0")
    out = {k: round(sum(f[k] for f in folds), 1) for k in keys_sum}
    out["share"] = round(out["gain"] / out["search_gain"], 3) if out["search_gain"] > 0 else None
    for part in ("wrong_all", "wrong_ambiguous", "wrong_clear"):
        n = sum(f[part]["n"] for f in folds)
        out[part] = {"n": round(n, 1), "mean_regret": round(sum(f[part]["n"] * f[part]["mean_regret"] for f in folds) / n, 1) if n else 0.0,
                     "max_regret": round(max(f[part]["max_regret"] for f in folds), 1)}
    return out


def reading(total: dict, folds: dict) -> dict:
    """126절 읽는 법(주 = MLP·규칙 R)."""
    m = total["mlp_R"]
    positive_folds = sum(f["mlp_R"]["gain"] > 0 for f in folds.values())
    beats_rule = m["gain"] > total["left_first"]["gain"]
    share = m["share"] or 0.0
    if m["gain"] > 0 and positive_folds >= 2 and share >= SHARE_MIN and beats_rule:
        word = "작은 신경망이 배울 수 있어 보인다"
    elif m["gain"] <= 0 or not beats_rule:
        word = "이 자료로는 배우지 못했다"
    else:
        word = "불분명"
    return {"word": word, "gain": m["gain"], "share": m["share"], "positive_folds": positive_folds, "left_first_gain": total["left_first"]["gain"]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs", default=str(PROJECT / "results/explore/online_20261023/runs"))
    ap.add_argument("--out", default=str(PROJECT / "results/explore/distill_20261024"))
    args = ap.parse_args()
    samples = load(Path(args.runs))
    cv = cross_validate(samples)
    result = {"code": code_fingerprint(), "samples": len(samples),
              "per_group": {g: sum(s["group"] == g for s in samples) for g in cv["groups"]},
              "search_differs": sum(s["search"] != s["p0"] for s in samples),
              "ambiguous_differs": sum(ambiguous(s) for s in samples), **cv}
    result["reading"] = reading(cv["total"], cv["folds"])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=1))
    print(json.dumps({k: result[k] for k in ("samples", "per_group", "search_differs", "ambiguous_differs", "reading")}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
