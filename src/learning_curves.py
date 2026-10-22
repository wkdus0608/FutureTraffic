"""DQN 학습곡선과 '10만 단계에서 아직 나아지는 중이었나' 판단 — 지문 밖, 저장된 학습 기록(monitor.csv)만 읽는다(새 학습 없음).

학습 기록: 에피소드(720단계 = 60분)마다 받은 점수 합. 단계당 보상 = 점수 합 ÷ 720 = −0.1 × (감지 구간에서 기다린 차 + 진입 대기 차)의
에피소드 평균이라, 0에 가까울수록 덜 기다린 것이다.

난이도 보정: 에피소드마다 학습 날짜의 한 시각을 무작위로 골라서(저녁 첨두와 새벽은 점수 차이가 크다), 그대로 그리면 시나리오 운이
곡선을 흔든다. 그래서 탐험이 끝난 뒤(2만 단계부터) 에피소드로 '시각(0~23시) 효과 + 날짜 효과'를 더하는 모형을 같은 설정의 seed 5개를
합쳐 맞추고, 그 효과를 뺀 값(보정 보상)을 쓴다. 시나리오는 시간과 상관없이 무작위로 뽑으므로 보정은 흐름(기울기)을 바꾸지 않는다.

판단 규칙(숫자를 보기 전에 정함, 진행 기록 102절):
- 판단 구간: 뒤쪽 절반(5만~10만 단계)의 에피소드
- seed마다 보정 보상을 단계에 직선으로 맞춘 기울기(1만 단계당), seed 5개 평균과 95% 신뢰구간(t, 자유도 4)
- 신뢰구간 아래 끝 > 0 → '아직 나아지는 중', 위 끝 < 0 → '나빠지는 중', 0을 포함하면 '평평(나아진다고 말할 수 없음)'
- 함께 적음: 6만~8만 대 8만~10만 단계의 보정 보상 평균 차이(seed 평균)

이어 학습한 실행: 끊긴 실행의 monitor.csv에는 체크포인트 뒤 버려진 에피소드가 남는다(시각 t가 다시 작아지는 곳에서 갈림).
끊긴 부분의 시나리오는 학습 환경과 같은 방법(random.Random(seed).choice)으로 다시 만들고(기록이 남은 실행에서 모두 같음을 확인),
이어 학습 시작 단계는 체크포인트 간격의 배수로 정해, 그 뒤에 끝난 끊긴 부분 에피소드는 뺀다.

사용: .venv/bin/python src/learning_curves.py --data results/learning_curves_20261022/data --out results/learning_curves_20261022
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

EPISODE_STEPS = 720
EXPLORATION_END = 20_000
JUDGE_FROM = 50_000
BLOCKS = ((60_000, 80_000), (80_000, 100_000))
SHAPE_BLOCKS = ((0, 10_000), (10_000, 20_000), (20_000, 35_000), (35_000, 50_000), (50_000, 75_000), (75_000, 100_000))  # 곡선 모양(설명용)
T_975_DF4 = 2.776
SEEDS = [42, 43, 44, 45, 46]
CHECKPOINT_FREQ = {"single": 10_000, "network12": 5_000}   # focused_train·network_rl의 --checkpoint-freq 기본값
# (폴더 이름, 지도, 그림 제목, 색 순서 칸) — 같은 방법은 두 지도에서 같은 색
SETS = [("single_v6", "single", "기존 DQN(공식 결과 모델)", 0), ("single_n5", "single", "n = 5 + Double", 1),
        ("single_n10", "single", "n = 10 + Double(H1)", 2),
        ("net12_base", "network12", "기존 DQN(공식 결과 모델)", 0), ("net12_n5", "network12", "n = 5 + Double(B12)", 1),
        ("net12_coord", "network12", "n = 5 + Double + 연동 정보(COORD)", 3)]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]       # 참고 팔레트 1~4번(검사기 통과, 3:1 미만 두 색은 제목·표로 보완)
INK, INK2, GRID, SURFACE, BAND = "#0b0b0b", "#52514e", "#e4e3de", "#fcfcfb", "#efeeea"


# ------------------------------------------------------------------ 기록 읽기
def _scenarios(kind: str) -> list[tuple[str, str]]:
    if kind == "single":
        from focused_data import COUNTS, load_counts
        from focused_train import training_scenarios
        return training_scenarios(load_counts(COUNTS))
    import network_rl
    return network_rl.training_scenarios()


def episodes(seed_dir: Path, kind: str, seed: int) -> list[dict]:
    """에피소드 표: 시작·끝 단계, 단계당 보상, 시나리오(날짜·시각). 이어 학습이면 버려진 에피소드를 빼고 시나리오를 다시 만든다."""
    tc = json.loads((seed_dir / "training_config.json").read_text())
    rows = [r for r in csv.reader(open(seed_dir / "monitor.csv")) if r and r[0][0] in "-0123456789"]
    rewards = [(float(r[0]), int(r[1]), float(r[2])) for r in rows]
    used = [tuple(x) for x in tc["scenarios_used_this_run"]]
    if not tc["resumed"]:
        parts = [(0, rewards, used)]
    else:
        cut = next(i for i in range(1, len(rewards)) if rewards[i][2] < rewards[i - 1][2])   # 시각 t가 다시 작아진 곳
        first, second = rewards[:cut], rewards[cut:]
        start2 = (tc["timesteps"] - EPISODE_STEPS * len(second)) // CHECKPOINT_FREQ[kind] * CHECKPOINT_FREQ[kind]
        rng = random.Random(seed)
        scen = _scenarios(kind)
        first_used = [rng.choice(scen) for _ in first]
        keep = [i for i in range(len(first)) if (i + 1) * EPISODE_STEPS <= start2]
        parts = [(0, [first[i] for i in keep], [first_used[i] for i in keep]), (start2, second, used)]
    out = []
    for start, eps, scen in parts:
        for k, ((r, length, _), (date, period)) in enumerate(zip(eps, scen)):
            s0 = start + k * EPISODE_STEPS
            out.append({"start": s0, "end": s0 + length, "reward": r / length, "date": date, "hour": int(period[:2])})
    return out


# ------------------------------------------------------------------ 보정과 판단
def fit_difficulty(rows: list[dict], sweeps: int = 50) -> tuple[float, dict, dict]:
    """보상 = 평균 + 시각 효과 + 날짜 효과(더하기 모형, 번갈아 평균으로 맞춤). 탐험 뒤 에피소드만."""
    rows = [r for r in rows if r["start"] >= EXPLORATION_END]
    mu = statistics.fmean(r["reward"] for r in rows)
    hour = {r["hour"]: 0.0 for r in rows}
    date = {r["date"]: 0.0 for r in rows}
    for _ in range(sweeps):
        for key, eff, other, okey in (("hour", hour, date, "date"), ("date", date, hour, "hour")):
            acc: dict = {}
            for r in rows:
                acc.setdefault(r[key], []).append(r["reward"] - mu - other[r[okey]])
            for k, v in acc.items():
                eff[k] = statistics.fmean(v)
        shift = statistics.fmean(date.values())
        for k in date:
            date[k] -= shift
        mu += shift
    return mu, hour, date


def adjusted(rows: list[dict], fit) -> list[dict]:
    mu, hour, date = fit
    return [{**r, "adjusted": r["reward"] - hour.get(r["hour"], 0.0) - date.get(r["date"], 0.0)} for r in rows]


def slope_per_10k(rows: list[dict], lo: int = JUDGE_FROM) -> float:
    pts = [((r["start"] + r["end"]) / 2, r["adjusted"]) for r in rows if r["start"] >= lo]
    mx = statistics.fmean(x for x, _ in pts)
    my = statistics.fmean(y for _, y in pts)
    return 10_000 * sum((x - mx) * (y - my) for x, y in pts) / sum((x - mx) ** 2 for x, _ in pts)


def judge(per_seed: dict[int, list[dict]]) -> dict:
    slopes = [slope_per_10k(rows) for rows in per_seed.values()]
    m = statistics.fmean(slopes)
    half = T_975_DF4 * statistics.stdev(slopes) / math.sqrt(len(slopes))
    lo, hi = m - half, m + half
    word = "아직 나아지는 중" if lo > 0 else "나빠지는 중" if hi < 0 else "평평(나아진다고 말할 수 없음)"
    blocks = []
    for a, b in BLOCKS:
        blocks.append(statistics.fmean(statistics.fmean(r["adjusted"] for r in rows if a <= r["start"] < b)
                                       for rows in per_seed.values()))
    shape = {f"{a // 1000}k-{b // 1000}k": round(statistics.fmean(statistics.fmean(r["adjusted"] for r in rows if a <= r["start"] < b)
                                                                    for rows in per_seed.values()), 3) for a, b in SHAPE_BLOCKS}
    return {"slope_per_10k": round(m, 4), "ci95": [round(lo, 4), round(hi, 4)], "seed_slopes": [round(s, 4) for s in slopes],
            "shape": shape,
            "verdict": word, "block_means": {f"{a // 1000}k-{b // 1000}k": round(v, 4) for (a, b), v in zip(BLOCKS, blocks)},
            "block_change": round(blocks[1] - blocks[0], 4)}


# ------------------------------------------------------------------ 그림
def rolling(xs: list[float], ys: list[float], window: int = 15) -> tuple[list[float], list[float]]:
    half = window // 2
    out_x, out_y = [], []
    for i in range(half, len(ys) - half):
        out_x.append(xs[i])
        out_y.append(statistics.fmean(ys[i - half:i + half + 1]))
    return out_x, out_y


def draw(data: dict, out: Path, key: str, ylabel: str, title: str) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": ["AppleGothic", "NanumGothic", "Noto Sans CJK KR", "DejaVu Sans"],
                         "axes.unicode_minus": False, "font.size": 10})
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.4), sharex=True, facecolor=SURFACE)
    for row, kind in enumerate(("single", "network12")):
        sets = [s for s in SETS if s[1] == kind]
        smooth = [v for name, *_ in sets for rows in data[name]["seeds"].values()
                  for v in rolling([0.0] * len(rows), [r[key] for r in rows])[1]]      # 그린 선의 범위(튀는 에피소드 하나로 눌리지 않게)
        lo, hi = min(smooth), max(smooth)
        for col, (name, _, label, color_slot) in enumerate(sets):
            ax = axes[row][col]
            ax.set_facecolor(SURFACE)
            color = COLORS[color_slot]
            ax.axvspan(0, EXPLORATION_END / 1000, color=BAND, zorder=0)
            ax.axvline(JUDGE_FROM / 1000, color=INK2, lw=0.8, ls=(0, (3, 3)), zorder=1)
            curves = []
            for seed, rows in sorted(data[name]["seeds"].items()):
                xs = [(r["start"] + r["end"]) / 2000 for r in rows]
                rx, ry = rolling(xs, [r[key] for r in rows])
                ax.plot(rx, ry, color=color, lw=1, alpha=0.35, zorder=2)
                curves.append((rx, ry))
            n = min(len(c[1]) for c in curves)
            ax.plot(curves[0][0][:n], [statistics.fmean(c[1][i] for c in curves) for i in range(n)], color=color, lw=2.2, zorder=3)
            j = data[name]["judge"]
            ax.set_title(label, fontsize=10.5, color=INK, loc="left", pad=20)
            ax.text(0.0, 1.02, f"5만~10만 기울기 {j['slope_per_10k']:+.3f} [{j['ci95'][0]:+.2f}, {j['ci95'][1]:+.2f}] → {j['verdict'].split('(')[0]}",
                    transform=ax.transAxes, ha="left", va="bottom", fontsize=8.3, color=INK2)
            pad = 0.15 * (hi - lo)
            ax.set_ylim(lo - pad, hi + pad)
            ax.grid(axis="y", color=GRID, lw=0.8)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color(GRID)
            ax.tick_params(colors=INK2, labelsize=8.5)
            if col == 0:
                ax.set_ylabel(("단독 모형\n" if kind == "single" else "12개 지도\n") + ylabel, color=INK, fontsize=9.5)
            if row == 1:
                ax.set_xlabel("학습 단계(천)", color=INK2, fontsize=9)
        axes[row][0].text(EXPLORATION_END / 2000, hi + 0.08 * (hi - lo), "탐험 구간", ha="center", fontsize=8, color=INK2)
    fig.suptitle(title, x=0.01, ha="left", fontsize=12.5, color=INK)
    fig.text(0.01, 0.005, "가는 선 = 학습 seed 5개(에피소드 15개 이동 평균), 굵은 선 = seed 평균. 회색 띠 = 탐험(무작위 행동이 많음), "
                          "점선 = 판단 구간 시작(5만 단계). 0에 가까울수록 덜 기다림.", fontsize=8.5, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    path = out / f"{key}_curves.png"
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", required=True, help="설정별 seed 폴더(monitor.csv·training_config.json)가 있는 곳")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    data_dir, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    data, summary = {}, {}
    for name, kind, label, _ in SETS:
        raw = {s: episodes(data_dir / name / f"seed_{s}", kind, s) for s in SEEDS}
        fit = fit_difficulty([r for rows in raw.values() for r in rows])
        seeds = {s: adjusted(rows, fit) for s, rows in raw.items()}
        data[name] = {"seeds": seeds}
        data[name]["judge"] = judge(seeds)
        summary[name] = {"map": kind, "label": label, "episodes": {s: len(v) for s, v in seeds.items()},
                         "resumed_seeds": [s for s in SEEDS if json.loads((data_dir / name / f"seed_{s}" / "training_config.json").read_text())["resumed"]],
                         **data[name]["judge"]}
    paths = [draw(data, out, "adjusted", "단계당 보상(난이도 보정)", "DQN 학습곡선 — 시나리오 난이도를 뺀 단계당 보상"),
             draw(data, out, "reward", "단계당 보상(보정 없음)", "DQN 학습곡선 — 보정하지 않은 단계당 보상")]
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    heads = [f"{a // 1000}~{b // 1000}k" for a, b in SHAPE_BLOCKS]
    lines = ["| 지도 | 설정 | 뒤쪽 절반 기울기(1만 단계당) | 95% 구간 | 판단 | 6만~8만 → 8만~10만 | " + " | ".join(heads) + " |",
             "|---|---|---:|---|---|---:|" + "---:|" * len(heads)]
    for name, s in summary.items():
        lines.append(f"| {'단독' if s['map'] == 'single' else '12개 지도'} | {s['label']} | {s['slope_per_10k']:+.4f} | "
                     f"[{s['ci95'][0]:+.4f}, {s['ci95'][1]:+.4f}] | {s['verdict']} | {s['block_change']:+.4f} | "
                     + " | ".join(f"{v:.2f}" for v in s["shape"].values()) + " |")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("그림:", *paths)


if __name__ == "__main__":
    main()
