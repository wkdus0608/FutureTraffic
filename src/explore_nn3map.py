"""[추가 탐색] NN3MAP — 라운드 정책을 12개 지도에서 잰다(사전 등록 results/explore/nn3r3_20261105/사전등록.md).

R2W(N5 + 학생 2개)와, 라운드 3이 채택되면 R3W(N5 + 학생 3개)를 NN12와 같은 46조각 × seed 5개에서 돌린다. 학습은 없다.
조각·seed·모델 돌려 쓰기가 NN12의 N5와 같아 N5와 조각별로 짝 비교한다. N5·TM4·보정 고정신호·P2는 다시 돌리지 않고 읽는다.
평가 코드(network_eval)는 고치지 않고, 실행할 때만 '단순 제어' 자리에 끼운다(NN12 그대로). 신경망 실행은 계획기를 부르면 멈춘다.
라운드 3의 어떤 단계도 이 결과를 읽지 않는다.

사용(FOCUSED_RULESET=pet1 FOCUSED_NORTH_EXT=240):
  python src/explore_nn3map.py check --n3 <Drive>/explore_nn3 --stored-n5 <Drive>/explore_nn12/N5 --out <dir> --work /content/n3mc
  python src/explore_nn3map.py run --policy R2W --n3 <Drive>/explore_nn3 --out <Drive>/explore_nn3/map/R2W --work /content/n3m --workers 8
  python src/explore_nn3map.py judge --root <Drive>/explore_nn3/map --r3-judge <Drive>/explore_nn3/r3/judge/judge.json --nn12 <Drive>/explore_nn12 --joint2 <Drive>/explore_joint2 --out <Drive>/explore_nn3/map/judge
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import sys
import time
from collections import Counter
from concurrent.futures import as_completed
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import arena3_referee as ref  # noqa: E402
import arena3_stage2_eval as s2e  # noqa: E402
import explore_nn1 as nn  # noqa: E402
import explore_nn12 as n12  # noqa: E402
import explore_nn3 as n3  # noqa: E402
import explore_nn3r2 as r2  # noqa: E402
import explore_nn3r3 as r3  # noqa: E402
import explore_weight as w  # noqa: E402
import focused_eval  # noqa: E402
import network_eval as ne  # noqa: E402
import network_rl as nr  # noqa: E402
from focused_data import environment_fingerprint  # noqa: E402

EXPERIMENT = "NN3MAP_V1"
SOURCES = ("src/explore_nn3map.py", "src/explore_nn3r3.py", "config/explore_nn3r3_v1.json", "results/explore/nn3r3_20261105/사전등록.md", "src/explore_nn3r2.py", "src/explore_nn3.py", *n12.SOURCES)
POLICIES = ("R2W", "R3W")
RECHECK = {"window": ["20220813", 22], "seed": 42}


# ----------------------------------------------------------------------------- 정책
def code_record() -> dict:
    return {"experiment": EXPERIMENT, "sources": {name: w.file_hash(ROOT / name) for name in dict.fromkeys(SOURCES)}, "evaluation_fingerprint": ne.evaluation_fingerprint(), "network12_fingerprint": nr._code_fingerprint(), "environment_fingerprint": environment_fingerprint()}


def code_fingerprint() -> str:
    return w.digest(code_record())


def winner_of(judge: dict) -> str | None:
    """라운드 3 판정에서 채택 후보를 읽는다. 끝나지 않았거나 미채택이면 None."""
    if judge.get("experiment") != r3.EXPERIMENT or judge.get("status") != "done":
        return None
    reading = judge["reading"]
    return reading["winner"] if reading["adopted"] else None


def policy_parts(name: str, n3_root: Path, cfg: dict, judge: dict | None = None) -> tuple[dict[int, Path], dict[int, list]]:
    """정책 이름 → 난수별 (N5 폴더, 학생 [(폴더, 문턱)]). R2W는 설정의 해시로, R3W는 라운드 3 기록의 해시로 대조한다."""
    n3_root = Path(n3_root)
    models, stacks = r3.base_policy(n3_root / "prep", n3_root / "r1" / "students", n3_root / "r2" / "students", cfg)
    if name == "R2W":
        return models, stacks
    if name != "R3W":
        raise ValueError(f"정책은 {POLICIES} 중 하나: {name}")
    winner = winner_of(judge or {})
    if winner is None:
        raise ValueError("라운드 3이 채택되지 않아 R3W가 없습니다.")
    students = n3_root / "r3" / "students"
    stacked = r3.candidate_stacks(students, winner, stacks)
    data, _ = r3.parse_candidate(winner)
    saved = {(s["data"], int(s["seed"])): s["model_sha256"] for s in w.read(students / "done.json")["students"]}
    for seed in stacked:
        folder = students / data / f"seed_{seed}"
        if nn.model_identity(folder) != saved[(data, seed)]:
            raise ValueError(f"라운드 3 학생이 기록과 다릅니다: {folder}")
    return models, stacked


# ----------------------------------------------------------------------------- 실행 계획(순수 계산)
def plan_jobs(name: str, models: dict[int, Path], stacks: dict[int, list], nn12cfg: dict, out: Path, work: Path) -> list[dict]:
    """조각·seed·모델 돌려 쓰기는 NN12의 N5와 같다. 입력 벡터는 저장하지 않는다."""
    jobs = []
    for job in n12.plan_jobs("N5", nn12cfg, out, work, models=Path("unused")):
        seed = int(job["model_seed"])
        if nn.model_identity(models[seed]) != job["model_identity"]:
            raise ValueError("N5 모델이 NN12에 고정한 것과 다릅니다.")
        jobs.append({**job, "name": name, "save_inputs": False, "model_dir": str(models[seed]), "stack": [[str(d), float(t)] for d, t in stacks[seed]], "n5_identity": job["model_identity"], "model_identity": r2.data_identity(models[seed], stacks[seed])})
    return jobs


# ----------------------------------------------------------------------------- 한 판
def execute(job: dict, run_dir: Path, labels: dict) -> dict:
    """N5 자리에 '신경망 + 쌓은 학생'을 앉힌다. 입력·선택은 단독 지도의 StackPolicy 그대로다."""
    window = {"date": job["date"], "hour": job["hour"], "policy": "greedy", "eval_seed": job["eval_seed"], "run_dir": str(run_dir), "fingerprint": n12.EVALUATION_FP}
    n12.ej.install_metrics()
    record: Counter = Counter()
    net, manifest = nn.load_model(Path(job["model_dir"]))
    if nn.model_identity(Path(job["model_dir"])) != job["n5_identity"] or manifest["world"] != n3.WORLD:
        raise ValueError("모델이 설정에 고정한 것과 다릅니다.")
    built = [(nn.load_model(Path(d))[0], float(t)) for d, t in job["stack"]]
    policy = n3.StackPolicy(net, job["name"], built)
    with n12.seated(n12.seat_class(n12.NetSeat, {**job, "name": "N5"}, labels, record, policy=policy)), nn.planner_free(None):
        ne.run_window(window)
    return {"stats": {**policy.info(), **record}, "model_seed": job["model_seed"], "model_identity": job["model_identity"], "planner_free": True}


def run_job(job: dict) -> dict:
    """한 판을 작업 폴더에서 돌리고, 끝나면 결과만 대상 폴더로 옮긴다. metrics.json을 마지막에 써서 있으면 끝난 판이다."""
    started = time.monotonic()
    work, target = Path(job["work"]), Path(job["target"])
    try:
        w.configure()
        torch.set_num_threads(1)
        shutil.rmtree(work, ignore_errors=True)
        work.mkdir(parents=True)
        extra = execute(job, work, job["labels"])
        metrics = w.read(work / "metrics.json")
        metrics["nn12"] = {"name": job["name"], "code": job["code"], "job_seconds": time.monotonic() - started, **extra}
        target.mkdir(parents=True, exist_ok=True)
        if (work / "vehicles.csv.gz").exists():
            shutil.copyfile(work / "vehicles.csv.gz", target / "vehicles.csv.gz")
        w.write(target / "metrics.json", metrics)
        return {"ok": True, "tag": job["tag"], "seconds": time.monotonic() - started}
    except BaseException as error:
        return {"ok": False, "tag": job["tag"], "seconds": time.monotonic() - started, "error": f"{type(error).__name__}: {error}"}
    finally:
        shutil.rmtree(work, ignore_errors=True)
        focused_eval.QueueFirst = n12.ORIGINAL_SEAT


def run_policy(name: str, n3_root: Path, out: Path, work: Path, workers: int, cfg: dict, nn12cfg: dict, *, judge: dict | None = None, worker=run_job, executor=None) -> dict:
    out = Path(out)
    code = code_fingerprint()
    (out / "runs").mkdir(parents=True, exist_ok=True)
    s2e._freeze(out / "experiment.json", code_record())
    models, stacks = policy_parts(name, n3_root, cfg, judge)
    jobs = plan_jobs(name, models, stacks, nn12cfg, out, work)
    todo = [{**j, "labels": nn12cfg["neighbor_labels"], "code": code} for j in jobs if not n12.finished(j, code)]
    ref.stamp(f"NN3MAP {name}: {len(jobs)}판 중 새로 {len(todo)}판 → {out}")
    if worker is run_job and executor is None and todo:
        n12.prepare_caches()
    failed, done = [], 0
    if todo:
        pool = executor or n12.pool_of(workers)
        try:
            futures = {pool.submit(worker, j): j for j in todo}
            for n, future in enumerate(as_completed(futures), 1):
                r = future.result()
                if r["ok"]:
                    done += 1
                else:
                    failed.append(r)
                    w.write(Path(futures[future]["target"]).with_name(r["tag"] + ".failed.json"), r)
                if n % 10 == 0 or n == len(todo):
                    ref.stamp(f"  {n}/{len(todo)} 끝, 실패 {len(failed)}, 마지막 {r['tag']} {r['seconds']:.0f}초")
        finally:
            if executor is None:
                pool.shutdown()
    status = {"total": len(jobs), "done_now": done, "failed": failed, "updated_unix": time.time()}
    s2e._write(out / "status.json", status)
    return status


# ----------------------------------------------------------------------------- 실행 전 점검
def single_job(name: str, models: dict[int, Path], stacks: dict[int, list], date: str, hour: int, seed: int, out: Path, work: Path, labels: dict, code: str) -> dict:
    """점검용 한 판(모델은 난수 42)."""
    chosen = 42
    tag = n12.run_tag(date, hour, seed)
    identity = nn.model_identity(models[chosen])
    return {"name": name, "tag": tag, "date": date, "hour": int(hour), "eval_seed": seed, "work": str(Path(work) / name / tag), "target": str(Path(out) / name / tag), "save_inputs": False, "model_seed": chosen, "model_dir": str(models[chosen]), "stack": [[str(d), float(t)] for d, t in stacks[chosen]],
            "n5_identity": identity, "model_identity": r2.data_identity(models[chosen], stacks[chosen]), "labels": labels, "code": code}


def same_values(a: dict, b: dict) -> bool:
    return all(a[x][y] == b[x][y] for x, y in n12.SAME) and a["action_counts"] == b["action_counts"]


def check(n3_root: Path, stored_n5: Path | None, out: Path, work: Path, cfg: dict, nn12cfg: dict) -> dict:
    """(1) 학생을 뺀 이 경로가 N5와 같은 값 (2) R2W 한 판: 계획기 없음·규칙 변경 0·시간표 일치·학생이 바꿈. 12개 지도는 맥과 Colab의 값이 달라 저장된 N5와의 대조는 Colab에서만 한다(stored_n5)."""
    out, work = Path(out), Path(work)
    if ne.evaluation_fingerprint() != n12.EVALUATION_FP or nr._code_fingerprint() != n12.NETWORK_FP or environment_fingerprint() != n12.ENVIRONMENT_FP:
        raise ValueError("12개 지도·평가·환경 지문이 기존 기준 실행과 다릅니다.")
    out.mkdir(parents=True, exist_ok=True)
    code = code_fingerprint()
    labels = nn12cfg["neighbor_labels"]
    models, stacks = policy_parts("R2W", n3_root, cfg)
    n12.prepare_caches()
    date, hour = RECHECK["window"]
    window, seed = nn12cfg["check"]["window"], nn12cfg["check"]["seed"]
    # 저장된 N5의 그 조각은 NN12 돌려 쓰기 규칙이 정한 모델로 돌았다. 그 모델로 맞춘다.
    i = [tuple(x) for x in nn12cfg["windows"]].index((date, hour))
    chosen = nn12cfg["models"]["N5"]["seeds"][n12.model_slot(i, nn12cfg["seeds"].index(RECHECK["seed"]))]
    empty = {**single_job("EMPTY", models, {42: []}, date, hour, RECHECK["seed"], out, work, labels, code), "model_seed": chosen, "model_dir": str(models[chosen]), "n5_identity": nn.model_identity(models[chosen]), "model_identity": r2.data_identity(models[chosen], [])}
    own = single_job("R2W", models, stacks, window[0], window[1], seed, out, work, labels, code)
    with n12.pool_of(2) as pool:
        got = list(pool.map(run_job, [empty, own]))
    bad = [g for g in got if not g["ok"]]
    if bad:
        raise RuntimeError(f"점검 실행이 실패했습니다: {bad}")
    a, b = w.read(Path(empty["target"]) / "metrics.json"), w.read(Path(own["target"]) / "metrics.json")
    stats = b["nn12"]["stats"]
    checks = {"finished": b["evaluation_fingerprint"] == n12.EVALUATION_FP and b["network12_fingerprint"] == n12.NETWORK_FP, "planner_free": b["nn12"]["planner_free"] is True, "rules_not_changed": sum(b["guard"]["changed_by_rules"].values()) == 0, "timetable_same": stats.get("timetable_checked", 0) > 0 and stats.get("timetable_mismatch", 0) == 0, "student_changes": stats.get("student_changed", 0) > 0, "empty_changes_nothing": a["nn12"]["stats"].get("student_changed", 0) == 0}
    if stored_n5 is not None:
        stored = w.read(Path(stored_n5) / "runs" / n12.run_tag(date, hour, RECHECK["seed"]) / "metrics.json")
        checks["same_as_stored_n5"] = same_values(a, stored)
    report = {"seconds": {"empty": a["nn12"]["job_seconds"], "own": b["nn12"]["job_seconds"]}, "change_share": stats.get("student_changed", 0) / max(stats.get("planned", 1), 1), "empty_values": n12.values_of(a), "own_values": n12.values_of(b), "code": code, "checks": checks, "ok": all(checks.values())}
    s2e._write(out / "check.json", report)
    ref.stamp(f"NN3MAP 점검: {'통과' if report['ok'] else '실패'} {checks} | 판당 {report['seconds']}")
    return report


# ----------------------------------------------------------------------------- 판정
def reference_data(nn12_root: Path, joint2: Path, nn12cfg: dict) -> dict:
    out = {}
    for name in ("FIXC", "P2"):
        out[name] = n12.read_many(n12.reference_paths(name, nn12cfg, joint2, Path("unused"), Path("unused")))
    for name in ("TM4", "N5"):
        out[name] = n12.read_many(n12.policy_paths(name, nn12cfg, nn12_root))
    return out


def word_of(pair: dict) -> str:
    """사전 등록의 세 낱말. 하한 > 0 남음, 상한 < 0 나빠짐, 그 밖은 구분 안 됨."""
    if pair["t95"][0] > 0:
        return "남음"
    if pair["t95"][1] < 0:
        return "나빠짐"
    return "구분 안 됨"


READING = {("남음", "남음"): "단독 지도 라운드를 그대로 이어도 된다.", "partial": "구간 이득은 옮겨지지만 도로망에 남지 않는다. 라운드를 더 하기 전에 주 지표를 도로망까지 넓힐지, 라벨에 204822 앞 대기를 넣을지 사용자와 정한다.",
           "none": "단독 지도의 몫이 12개 지도로 옮겨진다고 말할 수 없다. 라운드 4를 보류하고 12개 지도 상태 라벨을 검토한다."}


def reading_text(segment_word: str, network_word: str) -> str:
    if segment_word != "남음":
        return READING["none"]
    return READING[("남음", "남음")] if network_word == "남음" else READING["partial"]


def collect_report(root: Path, r3_judge: dict | None, nn12_root: Path, joint2: Path, nn12cfg: dict) -> dict:
    root = Path(root)
    winner = winner_of(r3_judge or {})
    names = ["R2W", *(["R3W"] if winner else [])]
    expected = len(nn12cfg["windows"]) * len(nn12cfg["seeds"])
    data = {name: n12.read_many(n12.policy_paths(name, nn12cfg, root)) for name in names}
    missing = {k: expected - len(v) for k, v in data.items() if len(v) != expected}
    if missing:
        return {"experiment": EXPERIMENT, "status": "incomplete", "missing": missing}
    refs = reference_data(Path(nn12_root), Path(joint2), nn12cfg)
    for name, rows in refs.items():
        if len(rows) != expected:
            raise ValueError(f"기준 실행 {name}이 {expected}판이 아닙니다: {len(rows)}")
    data = {**refs, **data}
    code = code_fingerprint()
    for name in names:
        for key, m in data[name].items():
            if m["nn12"]["code"] != code:
                raise ValueError(f"다른 코드로 돈 판이 있습니다: {name} {key}")
    for name, runs in data.items():
        for key, m in runs.items():
            if m.get("evaluation_fingerprint") != n12.EVALUATION_FP or m.get("network12_fingerprint") != n12.NETWORK_FP:
                raise ValueError(f"평가 지문이 다릅니다: {name} {key}")
    for name in names:
        for key, m in data[name].items():
            if m["network"]["vehicles"] != data["FIXC"][key]["network"]["vehicles"]:
                raise ValueError(f"같은 조각·seed의 도로망 차 수가 다릅니다: {name} {key}")
            if m["nn12"]["model_seed"] != data["N5"][key]["nn12"]["model_seed"]:
                raise ValueError(f"N5와 모델 짝이 다릅니다: {name} {key}")
    fixed = n12.weighted_delay(list(data["FIXC"].values()))
    order = ("FIXC", "P2", "TM4", "N5", *names)
    table = {name: n12.summary_row("N5" if name == "N5" else "TM4" if name == "TM4" else "P2" if name == "P2" else "FIXC", data[name], fixed, nn12cfg) for name in order}
    for name in names:
        stats = [m["nn12"]["stats"] for m in data[name].values()]
        planned = sum(s.get("planned", 0) for s in stats)
        models = sorted({m["nn12"]["model_seed"] for m in data[name].values()})
        table[name].update(decisions_per_run=planned / expected, change_share=sum(s.get("student_changed", 0) for s in stats) / max(planned, 1), rules_changed=sum(sum(m["guard"]["changed_by_rules"].values()) for m in data[name].values()), seconds_per_run=statistics.fmean(m["nn12"]["job_seconds"] for m in data[name].values()),
                           timetable_mismatch=sum(s.get("timetable_mismatch", 0) for s in stats), per_model={str(s): {"runs": sum(m["nn12"]["model_seed"] == s for m in data[name].values()), "weighted_delay_s": n12.weighted_delay([m for m in data[name].values() if m["nn12"]["model_seed"] == s])} for s in models})
    seg = {name: n12.per_window(data[name], n12.segment) for name in order}
    net = {name: n12.per_window(data[name], n12.network) for name in order}
    via = {name: n12.per_window(data[name], lambda m: m["via_204822"]["mean_delay_s"]) for name in order}
    unstable = nn12cfg["rules"]["unstable_window"]
    comparisons = [*((n, "N5") for n in names), *((n, "FIXC") for n in names), *((n, "TM4") for n in names), *((n, "P2") for n in names), ("N5", "FIXC"), *([("R3W", "R2W")] if winner else [])]
    pairs = {f"{a}-{b}": {"segment": n12.paired(seg[a], seg[b]), "segment_without_unstable": n12.paired(seg[a], seg[b], unstable), "network": n12.paired(net[a], net[b]), "network_without_unstable": n12.paired(net[a], net[b], unstable), "via_204822": n12.paired(via[a], via[b])} for a, b in comparisons}
    bands = {}
    for label, hours in n12.BANDS.items():
        keys = [k for k in seg["FIXC"] if int(k.split("_")[1]) in hours]
        bands[label] = {name: {"segment_vs_n5_s": statistics.fmean(seg["N5"][k] - seg[name][k] for k in keys), "network_vs_n5_s": statistics.fmean(net["N5"][k] - net[name][k] for k in keys), "segment_vs_fixc_s": statistics.fmean(seg["FIXC"][k] - seg[name][k] for k in keys), "network_vs_fixc_s": statistics.fmean(net["FIXC"][k] - net[name][k] for k in keys)} for name in names}
    top = names[-1]
    p = pairs[f"{top}-N5"]
    words = {"segment": word_of(p["segment"]), "network": word_of(p["network"])}
    return {"experiment": EXPERIMENT, "status": "done", "policies": names, "r3_winner": winner, "top": top, "table": table, "pairs": pairs, "bands": bands, "reading": {**words, "text": reading_text(words["segment"], words["network"])}, "code": code, "evaluation_fingerprint": n12.EVALUATION_FP, "updated_unix": time.time()}


NAMES = {"FIXC": "보정 고정신호", "P2": "P2", "TM4": "TM4(계획기)", "N5": "N5(흉내)", "R2W": "R2W(N5 + 학생 2개)", "R3W": "R3W(N5 + 학생 3개)"}


def render(report: dict) -> str:
    if report["status"] != "done":
        return f"# NN3MAP — 아직 끝나지 않았습니다\n\n남은 판: {json.dumps(report['missing'], ensure_ascii=False)}\n"
    f = lambda x: f"{x:+.2f}"  # noqa: E731
    v = lambda x, nd=2: "-" if x is None else f"{x:.{nd}f}"  # noqa: E731
    lines = ["# NN3MAP — 라운드 정책을 12개 지도에서 잰 결과(46조각 × seed 5개)", "", "검증 2일 개발 비교다. 단독 지도 감소율과 직접 비교하지 않는다. 라운드 3 후보 고르기에 쓰지 않았다.", "",
             "| 제어 | 구간 지연 | 고정신호 대비 | 도로망 지연 | 이웃 지나는 차 | 300초 초과(구간, 판당) | 동 좌회전 | 남 좌회전 | 저녁 동→서 직진 | 충돌(석천사거리/이웃/전체) |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, r in report["table"].items():
        lines.append(f"| {NAMES[name]} | {r['weighted_delay_s']:.2f} | {-r['reduction_vs_fixc_pct']:.1f}% | {v(r.get('network_mean_s'))} | {v(r.get('via_204822_mean_s'))} | {r['long_delays_per_run']:.1f} | {v(r['left_east_s'], 1)} | {v(r['left_south_s'], 1)} | {v(r['evening_east_west_s'], 1)} | {r['collisions_204820']}/{r.get('collisions_204822', '-')}/{r['collisions_total']} |")
    lines += ["", "조각 짝 비교(46조각, +면 앞쪽이 짧음)", "", "| 비교 | 구간 | t95 | 앞쪽이 나은 조각 | 흔들리는 조각 뺌 | 도로망 | t95 | 앞쪽이 나은 조각 | 이웃 지나는 차 |", "|---|---:|---|---:|---:|---:|---|---:|---:|"]
    for key, row in report["pairs"].items():
        s, n, x, y = row["segment"], row["network"], row["segment_without_unstable"], row["via_204822"]
        lines.append(f"| {key.replace('-', ' − ')} | {f(s['mean'])} | [{f(s['t95'][0])}, {f(s['t95'][1])}] | {s['positive']}/{s['n']} | {f(x['mean'])} | {f(n['mean'])} | [{f(n['t95'][0])}, {f(n['t95'][1])}] | {n['positive']}/{n['n']} | {f(y['mean'])} |")
    labels = list(report["bands"])
    for title, key in (("시간대별 N5 대비 구간 지연(초, +면 짧음)", "segment_vs_n5_s"), ("시간대별 N5 대비 도로망 지연(초, +면 짧음)", "network_vs_n5_s"), ("시간대별 보정 고정신호 대비 도로망 지연(초, +면 짧음)", "network_vs_fixc_s")):
        lines += ["", title, "", "| 제어 | " + " | ".join(labels) + " |", "|---|" + "---:|" * len(labels)]
        for name in report["policies"]:
            lines.append(f"| {name} | " + " | ".join(f(report["bands"][b][name][key]) for b in labels) + " |")
    lines += ["", "| 정책 | 판당 결정 | 학생이 바꾼 비율 | 규칙이 바꾼 판단 | 시간표 불일치 | 판당 초 | 모델별 구간 지연 |", "|---|---:|---:|---:|---:|---:|---|"]
    for name in report["policies"]:
        r = report["table"][name]
        each = " / ".join(f"{x['weighted_delay_s']:.2f}({x['runs']}판)" for x in r["per_model"].values())
        lines.append(f"| {name} | {r['decisions_per_run']:.1f} | {r['change_share']:.1%} | {r['rules_changed']} | {r['timetable_mismatch']} | {r['seconds_per_run']:.0f} | {each} |")
    g = report["reading"]
    lines += ["", f"**읽기({report['top']} − N5):** 구간 {g['segment']}, 도로망 {g['network']}. {g['text']}", ""]
    if report["r3_winner"] is None:
        lines.append("- 라운드 3이 끝나지 않았거나 미채택이라 R2W만 쟀다.")
    else:
        lines.append(f"- R3W는 라운드 3 채택 후보 {report['r3_winner']}이다.")
    lines.append("- 이 읽기는 다음 단계의 근거이며 자동으로 무언가를 멈추거나 시작하지 않는다. 구간이 0을 지나도 '효과 없음'이 아니라 '구분 안 됨'이다.")
    return "\n".join(lines) + "\n"


def save_report(out: Path, report: dict) -> None:
    out = Path(out)
    s2e._write(out / "judge.json", w.plain(report))
    (out / "summary.md").write_text(render(report), encoding="utf-8")


# ----------------------------------------------------------------------------- 명령
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("check")
    p.add_argument("--n3", required=True)
    p.add_argument("--stored-n5", help="explore_nn12/N5 폴더. Colab에서만 준다(맥은 소수점 차이로 값이 다르다)")
    p.add_argument("--out", required=True)
    p.add_argument("--work", required=True)
    p = sub.add_parser("run")
    p.add_argument("--policy", required=True, choices=POLICIES)
    p.add_argument("--n3", required=True)
    p.add_argument("--r3-judge", help="R3W일 때 라운드 3 judge.json")
    p.add_argument("--out", required=True)
    p.add_argument("--work", required=True)
    p.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1))
    p = sub.add_parser("judge")
    p.add_argument("--r3-judge", help="없으면 R2W만 읽는다")
    for name in ("root", "nn12", "joint2", "out"):
        p.add_argument(f"--{name}", required=True)
    args = parser.parse_args()
    w.configure()
    torch.set_num_threads(1)
    cfg, nn12cfg = r3.config(), n12.config()
    if args.cmd == "check":
        report = check(Path(args.n3), Path(args.stored_n5) if args.stored_n5 else None, Path(args.out), Path(args.work), cfg, nn12cfg)
        return 0 if report["ok"] else 2
    judge = w.read(Path(args.r3_judge)) if args.r3_judge and Path(args.r3_judge).exists() else None
    if args.cmd == "run":
        status = run_policy(args.policy, Path(args.n3), Path(args.out), Path(args.work), args.workers, cfg, nn12cfg, judge=judge)
        return 1 if status["failed"] else 0
    report = collect_report(Path(args.root), judge, Path(args.nn12), Path(args.joint2), nn12cfg)
    save_report(Path(args.out), report)
    print(render(report))
    return 0 if report["status"] == "done" else 3


if __name__ == "__main__":
    raise SystemExit(main())
