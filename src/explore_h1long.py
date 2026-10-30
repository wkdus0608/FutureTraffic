"""[추가 학습] H1 이어 학습(H1L, 진행 기록 162절) — 학습량만 10만 → 20만 → 30만 단계로 늘린 비교. 지문 밖.

질문: 다른 것은 모두 그대로 두고 학습량만 늘리면 H1의 검증 성능이 나아지는가?
- 출발점(학습 seed 42~46): 87절에서 평가한 최종 H1 모델(dqn_204820.zip, 지문 확인)과 체크포인트의 경험 저장소. 체크포인트 모델은
  최종 모델보다 갱신이 1번 적고 목표망이 교체 전 것이라 쓰지 않는다. H1 원본 폴더는 읽기만 한다.
- 경계 처리: 저장소는 끊긴 에피소드 중간에서 끝 표시 없이 끝난다. 이어 학습 전에 마지막 유효 칸에 '시간 제한 끊김' 표시를 달아
  10단계 묶음이 새 에피소드의 보상을 섞지 않게 한다(그 칸의 실제 다음 상태에서 가치를 잇는다).
- 학습을 두 번 나눠 부른다(→ 20만, → 30만). 두 번째 호출은 에피소드를 끊지 않고 잇고, 호출이 끝난 직후 모델을 저장한다.
  탐험률은 0.05 그대로여야 한다(아니면 멈춤). 환경·모델 seed = 학습 seed + 1000(+ 100 × 다시 시작한 횟수).
- 평가: 검증 2일 × 학습 seed 5개(평가 seed 42, 평가 지문 2030250938d6) × 10만·20만·30만. 10만은 87절 평가의 재현 확인이다.
- 판정: 114.4절 규칙(explore_decide의 summary·collapses·verdict_choice, 문턱 1.3초). 주 = 30만 대 10만.

명령(Colab 43번 칸; 학습 환경 변수 FOCUSED_RULESET=pet1 FOCUSED_NORTH_EXT=240 FOCUSED_DQN_NSTEPS=10 FOCUSED_DQN_DOUBLE=1)
  prepare  --h1 <explore_single/models_nstep10_double/pet1> --out <Drive>/explore_h1long
  train    --h1 ... --out ... --seed 42 [--stages 200000 300000]          (seed마다 따로, 동시에)
  evaluate --out ... --workers 5
  summary  --out ... --h1-eval <explore_single/eval/FP> --v6 <v6_check/FP> --pred <explore_pred/single/FP>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

SEEDS = (42, 43, 44, 45, 46)
H1_SHA10 = {42: "0b12e12bd9", 43: "0763a1351b", 44: "97ee61162f", 45: "fa6dd9cedc", 46: "e4e73e9a2c"}   # 87절 평가 모델
START = 100_000
STAGES = (200_000, 300_000)
STEPS = (START, *STAGES)
CHECKPOINT_EVERY = 10_000
RESEED, RESTART_STEP = 1000, 100
FINAL_EPS = 0.05
ENV_FP, EVAL_FP = "b20888514ea3", "2030250938d6"
MODEL = "dqn_204820"


def code_fingerprint() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# ------------------------------------------------------------------ 순수 계산
def zip_data(path: Path) -> dict:
    """SB3 저장 파일의 학습 진행 값(단계·호출·갱신 횟수, 탐험률)."""
    data = json.loads(zipfile.ZipFile(path).read("data"))
    return {k: data.get(k) for k in ("num_timesteps", "_n_calls", "_n_updates", "_episode_num", "exploration_rate", "n_steps", "gamma",
                                     "exploration_fraction", "target_update_interval")}


def mark_boundary(rb) -> dict:
    """저장소의 마지막 유효 칸이 에피소드 끝(done)이 아니면 '시간 제한 끊김'으로 표시한다(메모리에서만, 원본 파일은 그대로).
    NStepReplayBuffer는 done·timeout이 있는 칸에서 묶음을 끊고, 그 칸의 다음 관측으로 가치를 잇는다."""
    if rb.pos == 0 and not rb.full:
        return {"index": None, "marked": False, "reason": "빈 저장소"}
    i = (rb.pos - 1) % rb.buffer_size
    before = {"done": bool(rb.dones[i].any()), "timeout": bool(rb.timeouts[i].any())}
    marked = not before["done"] and not before["timeout"]
    if marked:
        rb.timeouts[i] = True
    return {"index": int(i), "pos": int(rb.pos), "full": bool(rb.full), "before": before, "marked": marked}


def exploration_at(num: int, total: int, fraction: float = 0.2, start: float = 1.0, end: float = FINAL_EPS) -> float:
    """SB3 DQN의 탐험률 일정(진행률 = 1 − 지금 단계 ÷ 이번 학습의 끝 단계)."""
    from stable_baselines3.common.utils import LinearSchedule
    return float(LinearSchedule(start, end, fraction)(1.0 - num / total))


def episode_windows(monitor_csv: Path, first_step: int, width: int = 50_000) -> list[dict]:
    """monitor.csv의 에피소드 보상(보정 없음)을 누적 단계 기준 width 구간으로 평균한다. first_step = 첫 에피소드 시작 단계."""
    import csv
    rows = list(csv.DictReader(line for line in Path(monitor_csv).read_text().splitlines() if not line.startswith("#")))
    out, step = {}, first_step
    for r in rows:
        step += int(r["l"])
        k = (step - 1) // width
        out.setdefault(k, []).append(float(r["r"]) / int(r["l"]))
    return [{"from": k * width, "to": (k + 1) * width, "episodes": len(v), "reward_per_step": round(statistics.fmean(v), 4)}
            for k, v in sorted(out.items())]


def eval_model_dir(out: Path, steps: int, seed: int) -> Path:
    return Path(out) / "models" / f"step_{steps}" / f"seed_{seed}"


def stage_config(h1_config: dict, steps: int, extra: dict) -> dict:
    """평가 도구(focused_eval.model_identity·run_day)가 읽는 학습 기록: H1 기록을 그대로 두고 단계 수와 이어 학습 정보만 바꾼다."""
    return {**h1_config, "timesteps": steps, "resumed": steps != START, "h1long": extra}


# ------------------------------------------------------------------ 출발점 확인(prepare)
def check_start(h1: Path, seed: int) -> dict:
    d = Path(h1) / f"seed_{seed}"
    cfg = json.loads((d / "training_config.json").read_text())
    final, ckpt, buf = d / f"{MODEL}.zip", d / "checkpoint" / "model.zip", d / "checkpoint" / "replay_buffer.pkl"
    problems = []
    if sha256(final)[:10] != H1_SHA10[seed]:
        problems.append(f"최종 모델 지문 {sha256(final)[:10]} ≠ 87절 {H1_SHA10[seed]}")
    if cfg["environment_fingerprint"] != ENV_FP or cfg["timesteps"] != START or cfg["seed"] != seed:
        problems.append("학습 기록의 환경·단계·seed가 H1과 다르다")
    if not (cfg["explore"]["double_dqn"] and cfg["dqn"].get("n_steps") == 10):
        problems.append("H1 설정(n = 10, Double)이 아니다")
    if json.loads((d / "checkpoint" / "checkpoint.json").read_text())["timesteps"] != START:
        problems.append("체크포인트가 10만 단계가 아니다")
    a, b = zip_data(final), zip_data(ckpt)
    if a["num_timesteps"] != START or a["exploration_rate"] != FINAL_EPS:
        problems.append(f"최종 모델의 단계·탐험률이 다르다: {a}")
    if problems:
        raise SystemExit(f"seed {seed}: " + "; ".join(problems))
    return {"seed": seed, "final_model": str(final), "final_sha256": sha256(final), "final": a, "checkpoint_model_sha256": sha256(ckpt),
            "checkpoint": b, "replay_buffer": str(buf), "replay_buffer_sha256": sha256(buf), "h1_config": cfg}


def prepare(args) -> None:
    out = Path(args.out)
    for seed in SEEDS:
        f = out / "start" / f"seed_{seed}.json"
        if f.exists() and json.loads(f.read_text()).get("code") == code_fingerprint():
            print(f"이미 있음 → 건너뜀: {f}")
            continue
        rec = check_start(Path(args.h1), seed)
        d = eval_model_dir(out, START, seed)                       # 10만 = H1 최종 모델 그대로(재현 확인용 평가)
        d.mkdir(parents=True, exist_ok=True)
        shutil.copy2(rec["final_model"], d / f"{MODEL}.zip")
        (d / "training_config.json").write_text(json.dumps(rec["h1_config"], ensure_ascii=False, indent=2))
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps({**{k: v for k, v in rec.items() if k != "h1_config"}, "code": code_fingerprint()}, ensure_ascii=False, indent=1))
        print(f"출발점 seed {seed}: 최종 모델 {rec['final_sha256'][:10]}(갱신 {rec['final']['_n_updates']}번) / 체크포인트 갱신 "
              f"{rec['checkpoint']['_n_updates']}번 → 최종 모델로 출발", flush=True)


# ------------------------------------------------------------------ 이어 학습(train)
def _latest_callback(folder: Path, restarts: int, every: int = CHECKPOINT_EVERY):
    """every 단계마다 모델·저장소를 덮어써 둔다(Colab이 끊겼을 때만 쓴다)."""
    from stable_baselines3.common.callbacks import BaseCallback

    class Latest(BaseCallback):
        def _on_step(self) -> bool:
            if self.num_timesteps % every == 0:
                folder.mkdir(parents=True, exist_ok=True)
                self.model.save(folder / "model")
                self.model.save_replay_buffer(folder / "replay_buffer")
                (folder / "checkpoint.json").write_text(json.dumps({"timesteps": self.num_timesteps, "restarts": restarts,
                                                                    "saved": time.strftime("%Y-%m-%d %H:%M:%S")}))
            return True

    return Latest()


def train(args) -> None:
    import torch
    torch.set_num_threads(1)
    import focused_train as ft
    from explore_dqn import double_hold_aware_dqn_class
    from focused_data import COUNTS, environment_fingerprint, load_counts
    from focused_rl import SignalConstraints
    from stable_baselines3.common.monitor import Monitor
    if environment_fingerprint() != ENV_FP:
        raise SystemExit(f"환경 지문 {environment_fingerprint()}이 H1의 {ENV_FP}와 다르다")
    if not (ft.EXPLORE["double_dqn"] and ft.DQN_SETTINGS.get("n_steps") == 10):
        raise SystemExit("FOCUSED_DQN_NSTEPS=10 FOCUSED_DQN_DOUBLE=1로 실행한다(H1 설정)")
    out, seed = Path(args.out), args.seed
    sd = out / "train" / f"seed_{seed}"
    stages = tuple(args.stages)
    if (sd / f"model_{stages[-1]}.zip").exists():
        print(f"seed {seed}: 이미 끝남 → 건너뜀")
        return
    start = json.loads((out / "start" / f"seed_{seed}.json").read_text())
    if start["code"] != code_fingerprint():
        raise SystemExit("출발점 확인(prepare)을 지금 코드로 다시 한다")
    state_f = sd / "state.json"
    state = json.loads(state_f.read_text()) if state_f.exists() else {"restarts": -1, "log": []}
    state["restarts"] += 1
    latest = sd / "latest"
    DQN = double_hold_aware_dqn_class()
    counts = load_counts(COUNTS)
    scenarios = ft.training_scenarios(counts)
    cfg = start["h1_config"] if "h1_config" in start else json.loads((Path(start["final_model"]).parent / "training_config.json").read_text())
    env_seed = seed + RESEED + RESTART_STEP * state["restarts"]
    workdir = Path(tempfile.mkdtemp(prefix=f"h1long_{seed}_", dir=args.work))
    constraints = SignalConstraints()
    env = ft.WarmStartTrainingEnv(scenarios, counts, env_seed, workdir, min_green=cfg["environment"]["min_green"],
                                  max_green=cfg["environment"]["max_green"], yellow=cfg["environment"]["yellow_time"], constraints=constraints)
    sd.mkdir(parents=True, exist_ok=True)
    from_start = state["restarts"] == 0 or not (latest / "checkpoint.json").exists()      # 첫 체크포인트 전에 끊겼으면 출발점부터 다시
    monitored = Monitor(env, filename=str(sd / "monitor.csv"), override_existing=from_start)
    if from_start:
        if sha256(Path(start["final_model"])) != start["final_sha256"] or sha256(Path(start["replay_buffer"])) != start["replay_buffer_sha256"]:
            raise SystemExit(f"seed {seed}: H1 원본 파일이 출발점 확인 때와 다르다")
        model = DQN.load(start["final_model"], env=monitored)
        model.load_replay_buffer(start["replay_buffer"])
        source = "H1 최종 모델 + H1 체크포인트 저장소" + ("" if state["restarts"] == 0 else "(첫 체크포인트 전에 끊겨 처음부터 다시)")
    else:
        model = DQN.load(latest / "model", env=monitored)
        model.load_replay_buffer(latest / "replay_buffer")
        source = f"최신 체크포인트({json.loads((latest / 'checkpoint.json').read_text())['timesteps']}단계)"
    if model.replay_buffer.n_steps != 10 or abs(model.replay_buffer.gamma - 0.99) > 1e-12:
        raise SystemExit("저장소가 H1의 10단계·할인율 0.99가 아니다")
    boundary = mark_boundary(model.replay_buffer)
    model.set_random_seed(env_seed)
    entry = {"restart": state["restarts"], "source": source, "from_timesteps": int(model.num_timesteps), "env_seed": env_seed,
             "boundary": boundary, "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    state["log"].append(entry)
    state_f.write_text(json.dumps(state, ensure_ascii=False, indent=1))
    print(f"seed {seed}: {source}에서 {model.num_timesteps}단계부터, 경계 표시 {boundary['marked']}(칸 {boundary['index']})", flush=True)
    started = time.time()
    for target in stages:
        if (sd / f"model_{target}.zip").exists() and model.num_timesteps >= target:
            continue
        if model.num_timesteps < target:
            model.learn(total_timesteps=target - model.num_timesteps, reset_num_timesteps=False, progress_bar=False,
                        callback=_latest_callback(latest, state["restarts"], args.checkpoint_every))
        if model.num_timesteps != target:
            raise SystemExit(f"seed {seed}: {target}단계에서 멈추지 않았다({model.num_timesteps})")
        if abs(model.exploration_rate - FINAL_EPS) > 1e-12:
            raise SystemExit(f"seed {seed}: {target}단계의 탐험률 {model.exploration_rate} ≠ {FINAL_EPS}")
        model.save(sd / f"model_{target}")
        info = {"timesteps": int(model.num_timesteps), "n_calls": int(model._n_calls), "n_updates": int(model._n_updates),
                "exploration_rate": float(model.exploration_rate), "episodes_this_process": len(env.history),
                "distinct_scenarios_this_process": len(set(env.history)), "seconds_this_process": round(time.time() - started, 1),
                "restarts": state["restarts"], "model_sha256": sha256(sd / f"model_{target}.zip"), "code": code_fingerprint(),
                "start": {k: start[k] for k in ("final_sha256", "replay_buffer_sha256")}, "log": state["log"]}
        (sd / f"model_{target}.json").write_text(json.dumps(info, ensure_ascii=False, indent=1))
        d = eval_model_dir(out, target, seed)
        d.mkdir(parents=True, exist_ok=True)
        shutil.copy2(sd / f"model_{target}.zip", d / f"{MODEL}.zip")
        extra = {k: info[k] for k in ("n_calls", "n_updates", "exploration_rate", "restarts", "code")}
        conf = stage_config(cfg, target, {**extra, "start_model_sha256": start["final_sha256"], "boundary": state["log"][0]["boundary"]})
        conf.update(scenarios_used_this_run=env.history, episodes_this_run=len(env.history), seconds_this_run=info["seconds_this_process"])
        (d / "training_config.json").write_text(json.dumps(conf, ensure_ascii=False, indent=2))
        print(f"seed {seed}: {target}단계 저장(갱신 {info['n_updates']}번, 탐험률 {info['exploration_rate']}, "
              f"에피소드 {info['episodes_this_process']}개, {info['seconds_this_process']}초)", flush=True)
    monitored.close()
    shutil.rmtree(workdir, ignore_errors=True)


# ------------------------------------------------------------------ 평가(evaluate)
def evaluate(args) -> None:
    from focused_eval import evaluation_fingerprint
    if evaluation_fingerprint() != EVAL_FP:
        raise SystemExit(f"평가 지문 {evaluation_fingerprint()}이 87절의 {EVAL_FP}와 다르다")
    out = Path(args.out)
    dates = json.loads((PROJECT / "config/focused_split.json").read_text())["validation"]
    for steps in STEPS:
        models = [str(eval_model_dir(out, steps, s)) for s in SEEDS]
        missing = [m for m in models if not (Path(m) / f"{MODEL}.zip").exists()]
        if missing:
            raise SystemExit(f"{steps}단계 모델이 다 없다: {missing}")
        print(f"===== 평가 {steps}단계", flush=True)
        subprocess.run([sys.executable, str(PROJECT / "src/focused_eval.py"), "run", "--dates", *dates, "--policies", "dqn",
                        "--models", *models, "--arrivals", "random", "--workers", str(args.workers), "--out", str(out / "eval")],
                       check=True, cwd=PROJECT)


# ------------------------------------------------------------------ 정리(summary)
WORDS = {"낫다": "추가 20만 단계로 나아졌다(개발 평가)", "차이 없음": "이 설정에서 추가 20만 단계의 효과를 확인하지 못했다",
         "못하다": "이 설정에서 추가 학습 뒤 나빠졌다"}


def paired_seeds(x: dict, y: dict) -> dict:
    """seed마다 (y − x) 지연(양수면 x가 짧음)의 평균과 95% t 구간(자유도 4)."""
    import explore_rollout as er
    d = [y[s] - x[s] for s in sorted(x)]
    mean, lo, hi = er.t_interval(d)
    return {"per_seed": [round(v, 2) for v in d], "mean": round(mean, 3), "interval": [round(lo, 3), round(hi, 3)],
            "shorter_seeds": sum(v > 0 for v in d)}


def reading(s: dict, per_seed: dict) -> dict:
    import explore_decide as dc
    v = lambda a, b: dc.verdict_choice(s[a], s[b])
    main = v("step_300000", "step_100000")
    out = {"main": {**main, "sentence": WORDS[main["word"]], "paired_seeds": paired_seeds(per_seed["step_300000"], per_seed["step_100000"])},
           "200k_vs_100k": {**v("step_200000", "step_100000"), "paired_seeds": paired_seeds(per_seed["step_200000"], per_seed["step_100000"])},
           "300k_vs_200k": v("step_300000", "step_200000")}
    for k in STEPS:
        out[f"step_{k}_vs_greedy"] = v(f"step_{k}", "greedy")
        out[f"step_{k}_vs_P0"] = v(f"step_{k}", "P0")
        out[f"step_{k}_vs_fixedc"] = dc.versus_fixedc(s[f"step_{k}"], s["fixedc"], "mean_delay_s")
    return out


def summary(args) -> None:
    import explore_decide as dc
    out = Path(args.out)
    ev = out / "eval" / EVAL_FP
    groups = {f"step_{k}": dc._dqn_rows(ev, dc.VALIDATION, [str(eval_model_dir(out, k, s)) for s in SEEDS], "random") for k in STEPS}
    old = dc._dqn_rows(Path(args.h1_eval), dc.VALIDATION, [str(eval_model_dir(out, START, s)) for s in SEEDS], "random")
    groups.update({"greedy": dc._single_rows(Path(args.v6), dc.VALIDATION, "greedy", "random"),
                   "fixedc": dc._single_rows(Path(args.v6), dc.VALIDATION, "fixedc", "random"),
                   "P0": dc._single_rows(Path(args.pred), dc.VALIDATION, dc.PRED_TAGS["p0"], "random")})
    s = {k: {**dc.summary(v), "collapses": dc.collapses(v, groups["fixedc"])} for k, v in groups.items()}
    per_seed = {f"step_{k}": dc.per_seed_delay(groups[f"step_{k}"]) for k in STEPS}
    key = lambda r: (r["date"], r["train_seed"])
    old_by = {key(r): r["all"]["mean_delay_s"] for r in old}
    repro = [{"date": r["date"], "seed": r["train_seed"], "now": r["all"]["mean_delay_s"], "h1_87": old_by.get(key(r))}
             for r in groups[f"step_{START}"]]
    training = {}
    for seed in SEEDS:
        sd = out / "train" / f"seed_{seed}"
        rec = {f"{k}": json.loads((sd / f"model_{k}.json").read_text()) for k in STAGES}
        training[seed] = {"restarts": rec[str(STAGES[-1])]["restarts"], "log": rec[str(STAGES[-1])]["log"],
                          **{k: {x: r[x] for x in ("n_updates", "exploration_rate", "episodes_this_process", "distinct_scenarios_this_process",
                                                   "seconds_this_process")} for k, r in rec.items()},
                          "reward_windows": episode_windows(sd / "monitor.csv", START),
                          "h1_reward_windows": episode_windows(Path(json.loads((out / "start" / f"seed_{seed}.json").read_text())["final_model"]).parent
                                                               / "monitor.csv", 0)[-2:]}
    result = {"code": code_fingerprint(), "evaluation_fingerprint": EVAL_FP, "dates": dc.VALIDATION, "threshold_s": dc.DELAY_S,
              "summary": s, "per_seed_delay": per_seed, "reading": reading(s, per_seed),
              "reproduction": {"same": all(x["h1_87"] is not None and abs(x["now"] - x["h1_87"]) < 1e-9 for x in repro), "runs": repro},
              "training": training}
    (out / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=1, default=str))
    r = result["reading"]
    names = {"step_100000": "H1 10만(출발)", "step_200000": "H1 20만", "step_300000": "H1 30만", "greedy": "단순 제어", "P0": "P0",
             "fixedc": "보정 고정신호"}
    lines = ["# H1 이어 학습 — 학습량만 10만 → 20만 → 30만(진행 기록 162절)", "",
             f"- 주 판정(30만 대 10만): **{r['main']['word']}** — {r['main']['sentence']}. 지연 차이 {r['main']['delay_diff_s']}초"
             f"(seed 짝 평균 {r['main']['paired_seeds']['mean']}초 [{r['main']['paired_seeds']['interval'][0]}, "
             f"{r['main']['paired_seeds']['interval'][1]}], 짧아진 seed {r['main']['paired_seeds']['shorter_seeds']}/5)",
             f"- 20만 대 10만: **{r['200k_vs_100k']['word']}** — 지연 차이 {r['200k_vs_100k']['delay_diff_s']}초; 30만 대 20만: "
             f"{r['300k_vs_200k']['word']}({r['300k_vs_200k']['delay_diff_s']}초)",
             f"- 재현 확인(10만 = 87절 H1 평가와 같은가): {'같음' if result['reproduction']['same'] else '다름'}",
             "- 다시 시작: " + ", ".join(f"seed {k} {v['restarts']}번" for k, v in training.items()), "",
             "| 제어 | 차량당 지연 | seed 표준편차 | 300초 초과 | 동쪽 좌회전 | 남쪽 좌회전 | 무너진 실행 | 확인 안 된 충돌 | 대 단순 제어 | 대 P0 | 대 보정 고정 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---|---|---:|"]
    for k in (*[f"step_{x}" for x in STEPS], "greedy", "P0", "fixedc"):
        x = s[k]
        step = k.startswith("step_")
        vs = lambda t: f"{r[f'{k}_vs_{t}']['word']}({r[f'{k}_vs_{t}']['delay_diff_s']}초)" if step else "—"
        fc = f"{r[f'{k}_vs_fixedc']['percent']}%" if step else "—"
        lines.append(f"| {names[k]} | {x['mean_delay_s']} | {x.get('mean_delay_s_std') or '—'} | {x['long_waits']} | {x['left_east']} | "
                     f"{x['left_south']} | {x['collapses']} | {x['unexplained']} | {vs('greedy')} | {vs('P0')} | {fc} |")
    lines += ["", "| seed | 10만 | 20만 | 30만 |", "|---:|---:|---:|---:|"]
    for seed in SEEDS:
        lines.append(f"| {seed} | " + " | ".join(str(per_seed[f'step_{k}'][seed]) for k in STEPS) + " |")
    lines += ["", "- 학습 중 에피소드 보상(보정 없음, 단계당; 5만~10만은 H1 원래 학습): " + "; ".join(
        f"seed {k} " + ", ".join(f"{w['from'] // 1000}~{w['to'] // 1000}천 {w['reward_per_step']}" for w in v["h1_reward_windows"] + v["reward_windows"])
        for k, v in training.items())]
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--h1", required=True)
    p.add_argument("--out", required=True)
    t = sub.add_parser("train")
    t.add_argument("--h1", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--seed", type=int, required=True, choices=SEEDS)
    t.add_argument("--work", default=None)
    t.add_argument("--stages", type=int, nargs="+", default=list(STAGES), help="맥 장치 확인에서만 작은 값을 준다")
    t.add_argument("--checkpoint-every", type=int, default=CHECKPOINT_EVERY, help="맥 장치 확인에서만 작은 값을 준다")
    e = sub.add_parser("evaluate")
    e.add_argument("--out", required=True)
    e.add_argument("--workers", type=int, default=5)
    m = sub.add_parser("summary")
    m.add_argument("--out", required=True)
    m.add_argument("--h1-eval", required=True)
    m.add_argument("--v6", required=True)
    m.add_argument("--pred", required=True)
    args = ap.parse_args()
    {"prepare": prepare, "train": train, "evaluate": evaluate, "summary": summary}[args.cmd](args)


if __name__ == "__main__":
    main()
