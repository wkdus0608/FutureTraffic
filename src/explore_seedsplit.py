"""[추가 탐색] 같은 수요의 seed 전이 검사(진행 기록 134절). 지문 밖, 저장된 자료만 읽음(SUMO 없음). 한 차례 탐색적 진단.

세 에피소드의 seed 42 자료를 합쳐 학습하고 seed 43에서 재며, 반대 방향도 잰다. 설정은 미리 고정했다(다시 고르지 않음):
10,000번 갱신, 문턱 0(P0로 돌아가는 정책만 보면 전이 능력을 못 봄), 시작 seed 0~4 다섯 모델의 예측 평균. 그 밖은 131절과 같다.

사용: .venv/bin/python src/explore_seedsplit.py --runs results/explore/online_20261023/runs --out results/explore/seedsplit_20261024
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import explore_distill as ed
import explore_student as es

PROJECT = ed.PROJECT
STEPS = 10_000
TAU = 0.0
DIRECTIONS = ((42, 43), (43, 42))


def code_fingerprint() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]


def changes(samples: list[dict], picks: list[int]) -> dict:
    """실제로 P0와 다르게 고른 시점만: 몇 번, 탐색과 같은 후보, 이득이 난 변경·손해 난 변경(8개 평균 비용)."""
    rows = [(s, p, ed.mean_cost(s, s["p0"]) - ed.mean_cost(s, p)) for s, p in zip(samples, picks) if p != s["p0"]]
    good, bad = [g for *_, g in rows if g > 0], [g for *_, g in rows if g <= 0]
    return {"changes": len(rows), "changes_same_as_search": sum(p == s["search"] for s, p, _ in rows),
            "good_changes": len(good), "good_gain": round(sum(good), 1), "bad_changes": len(bad), "bad_loss": round(-sum(bad), 1)}


def measure(samples: list[dict], picks: list[int]) -> dict:
    return {**ed.evaluate(samples, picks), **changes(samples, picks)}


def per_episode(samples: list[dict], picks: list[int]) -> dict:
    out = {}
    for g in sorted({s["group"] for s in samples}):
        rows = [(s, p) for s, p in zip(samples, picks) if s["group"] == g]
        out[g] = measure([s for s, _ in rows], [p for _, p in rows])
    return out


def run_direction(samples: list[dict], train_seed: int, test_seed: int, steps: int = STEPS, tau: float = TAU, seeds=es.SEEDS) -> tuple:
    train = [s for s in samples if s["seed"] == train_seed]
    test = [s for s in samples if s["seed"] == test_seed]
    ens = es.fit_ensemble(train, (steps,), seeds)
    picks = es.picks_for(ens[steps], test, tau)
    result = {"train_seed": train_seed, "test_seed": test_seed, "train": len(train), "test": len(test),
              "all": measure(test, picks), "per_episode": per_episode(test, picks)}
    return result, test, picks


def check(samples: list[dict], directions=DIRECTIONS, steps: int = STEPS, tau: float = TAU, seeds=es.SEEDS) -> dict:
    served_by = ed.left_greens()
    dirs, tests, picks = [], [], []
    for a, b in directions:
        r, t, p = run_direction(samples, a, b, steps, tau, seeds)
        dirs.append(r)
        tests += t
        picks += p
    return {"directions": dirs, "total": measure(tests, picks), "total_per_episode": per_episode(tests, picks),
            "left_first": ed.evaluate(tests, [ed.left_first(s, served_by) for s in tests])}


def reading(result: dict) -> dict:
    """134절 읽는 법."""
    t = result["total"]
    both = all(d["all"]["gain"] > 0 for d in result["directions"])
    episodes = sum(e["gain"] > 0 for e in result["total_per_episode"].values())
    share = t["share"] or 0.0
    if t["gain"] > 0 and both and share >= ed.SHARE_MIN and episodes >= 2 and t["gain"] > result["left_first"]["gain"]:
        word = "같은 수요 안에서는 옮길 단서가 있다"
    elif t["gain"] <= 0:
        word = "같은 수요 안에서도 옮기지 못했다"
    else:
        word = "불분명"
    return {"word": word, "gain": t["gain"], "share": t["share"], "both_directions_positive": both, "positive_episodes": episodes,
            "changes": t["changes"], "good_changes": t["good_changes"], "bad_changes": t["bad_changes"],
            "left_first_gain": result["left_first"]["gain"]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs", default=str(PROJECT / "results/explore/online_20261023/runs"))
    ap.add_argument("--out", default=str(PROJECT / "results/explore/seedsplit_20261024"))
    args = ap.parse_args()
    out = Path(args.out)
    if (out / "result.json").exists():
        raise SystemExit(f"이미 있음: {out / 'result.json'} (134절: 한 차례만 잰다)")
    samples = ed.load(Path(args.runs))
    result = {"code": code_fingerprint(), "student_code": es.code_fingerprint(), "distill_code": ed.code_fingerprint(),
              "samples": len(samples), "steps": STEPS, "tau": TAU, "seeds": list(es.SEEDS), **check(samples)}
    result["reading"] = reading(result)
    out.mkdir(parents=True, exist_ok=True)
    (out / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=1))
    print(json.dumps(result["reading"], ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
