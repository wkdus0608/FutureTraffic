"""Colab-only overnight WEIGHT1 learning-duration study.

The original failed pilot remains unchanged. Only training duration changes:
three independent seeds, predeclared checkpoints, final endpoint comparison.
Whole seed runs are split into <=42-episode jobs to keep two CPUs occupied.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import fcntl
import gzip
import hashlib
import importlib.metadata
import json
import math
import multiprocessing
import os
from pathlib import Path
import signal
import sys
import time
import uuid

import numpy as np

import explore_weight as w

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/explore_weight_long_v1.json"


class StudyStopped(RuntimeError):
    pass


def available_cpus():
    cores = os.cpu_count() or 1
    if hasattr(os, "sched_getaffinity"):
        cores = min(cores, len(os.sched_getaffinity(0)))
    quota = Path("/sys/fs/cgroup/cpu.max")
    if quota.exists():
        fields = quota.read_text().split()
        if len(fields) == 2 and fields[0] != "max":
            cores = min(cores, max(1, int(float(fields[0]) / float(fields[1]))))
    return max(1, cores)


def choose_profile(cpus):
    cfg = w.read(CONFIG)
    if cpus < 2:
        raise StudyStopped("CPU가1개라 3모델의 하루 실험을 구성할 수 없습니다. CPU2개 이상인 Colab 런타임이 필요합니다")
    name = "three_cpu" if cpus >= 3 else "two_cpu"
    return {"name": name, **cfg["profiles"][name]}


def groups(section):
    part = w.read(CONFIG)[section]
    return [(d, p) for d in part["dates"] for p in part["periods"]]


def register(source, out, workers, wall_hours, compute_hours, *, check_runtime=True):
    w.configure()
    from explore_weight_long_train import validate_parent
    parent = validate_parent(Path(source), check_runtime=check_runtime)
    cfg = w.read(CONFIG)
    existing = Path(out) / "experiment.json"
    cpus = available_cpus()
    if cpus < 2:
        raise StudyStopped("학습 재개에도 CPU2개 이상인 Colab 런타임이 필요합니다")
    profile = w.read(existing)["profile"] if existing.exists() else choose_profile(cpus)
    count = min(3, cpus) if workers == "auto" else min(int(workers), cpus, 3)
    if count < 1 or count > 3:
        raise ValueError("workers는auto또는1~3이어야 합니다")
    if not (0 < wall_hours <= 24 and 0 < compute_hours <= 96):
        raise ValueError("실행상한은24시간·누적96시간 이내여야 합니다")
    prior = float(w.read(Path(source) / "budget.json")["used_s"]) + cfg["budget"]["prior_local_seconds"]
    frozen = {"kind": "WEIGHT1_LONG_V1", "config": cfg, "profile": profile,
              "parent_fingerprint": parent["fingerprint"],
              "parent_checkpoint_sha256": parent["checkpoint_sha256"],
              "parent_budget_sha256": w.file_hash(Path(source) / "budget.json"),
              "code": {n: w.file_hash(ROOT / n) for n in ("src/explore_weight_long.py", "src/explore_weight_long_train.py", "config/explore_weight_long_v1.json")},
              "wall_hours": wall_hours, "compute_hours": compute_hours, "prior_compute_s": prior}
    fp = w.digest(frozen)
    record = {"fingerprint": fp, **frozen}
    forecast = forecast_budget(record, count)
    if forecast["conservative_total_s"] > compute_hours * 3600:
        raise StudyStopped("사전 예상 비용이 누적 예산을 넘어 실험을 시작하지 않습니다")
    if existing.exists() and w.read(existing) != record:
        raise ValueError("이어 실행의 설정·부모 자료·코드가 다릅니다. 기존 결과를 보존하세요")
    Path(out).mkdir(parents=True, exist_ok=True)
    w.write(existing, record)
    w.write(Path(out) / "forecast.json", forecast)
    return record, count


def forecast_budget(record, workers):
    cfg = record["config"]
    milestones = record["profile"]["milestones"]
    episodes = len(cfg["train_seeds"]) * max(milestones) - 42
    dev_runs = len(cfg["train_seeds"]) * len(groups("development")) * (
        (len(milestones)-1) * len(cfg["development"]["interim_seeds"])
        + len(cfg["development"]["primary_seeds"]))
    confirmation_runs = (len(cfg["train_seeds"])+2) * len(groups("confirmation")) * len(cfg["confirmation"]["eval_seeds"])
    new_seconds = episodes * cfg["budget"]["episode_seconds_estimate"] + (dev_runs+confirmation_runs) * cfg["budget"]["evaluation_seconds_estimate"]
    return {"new_training_episodes": episodes, "development_runs": dev_runs,
            "confirmation_runs_if_gate_passes": confirmation_runs,
            "nominal_new_s": new_seconds, "nominal_wall_hours": new_seconds / workers / 3600,
            "conservative_total_s": record["prior_compute_s"] + new_seconds * cfg["budget"]["forecast_margin"],
            "includes_prior_s": record["prior_compute_s"],
            "note": "이전 Colab 실측 기반 예상이며 속도를 보장하지 않는다. 하루를 넘으면 저장 후 재개한다."}


class Ledger:
    def __init__(self, out, record):
        self.path = Path(out) / "budget.json"
        self.limit = float(record["compute_hours"]) * 3600
        self.prior = float(record["prior_compute_s"])
        saved = w.read(self.path) if self.path.exists() else {}
        self.entries = saved.get("entries", [])
        self.pending = saved.get("pending", {})
        if saved and (saved["limit_s"] != self.limit or saved["prior_compute_s"] != self.prior):
            raise ValueError("기존 계산 장부의 한도가 다릅니다")
        for token, entry in self.pending.items():
            self.entries.append({**entry, "token": token, "seconds": entry["cap_s"],
                                 "status": "interrupted_conservative_cap"})
        self.pending = {}
        self.save()

    @property
    def used(self):
        return self.prior + sum(e["seconds"] for e in self.entries)

    @property
    def remaining(self):
        return self.limit - self.used - sum(e["cap_s"] for e in self.pending.values())

    def save(self):
        w.write(self.path, {"limit_s": self.limit, "prior_compute_s": self.prior,
                            "used_s": self.used, "entries": self.entries, "pending": self.pending,
                            "accounting": "sum of elapsed worker job seconds, including parallel jobs; prior Colab and Mac retained"})

    def begin(self, name, cap):
        if not math.isfinite(cap) or cap <= 0 or cap > self.remaining:
            raise StudyStopped("누적 계산 예산이 부족해 새 작업을 시작하지 않습니다")
        token = uuid.uuid4().hex
        self.pending[token] = {"job": name, "cap_s": float(cap), "started_unix": time.time()}
        self.save()
        return token

    def finish(self, token, seconds, status):
        entry = self.pending.pop(token)
        self.entries.append({**entry, "token": token, "seconds": max(0, float(seconds)), "status": status})
        self.save()


def _worker_init():
    try:
        os.setsid()
    except OSError:
        pass
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"


def _kill_pool(pool):
    processes = list((getattr(pool, "_processes", None) or {}).values())
    for process in processes:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
    until = time.monotonic() + 5
    for process in processes:
        process.join(timeout=max(0, until-time.monotonic()))
    # Workers and their SUMO descendants each belong to our own process group.
    # Kill that group even if its worker already exited but a descendant stayed.
    for process in processes:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    pool.shutdown(wait=False, cancel_futures=True)


def _training_worker(job):
    from explore_weight_long_train import run_training_job
    return run_training_job(job)


def checkpoint_episodes(folder):
    path = Path(folder) / "latest.json"
    if not path.exists():
        return None
    latest = w.read(path)
    cp = (Path(folder) / latest["checkpoint"]).resolve()
    if cp.parent != (Path(folder) / "checkpoints").resolve():
        raise ValueError("체크포인트 경로가 모델 폴더를 벗어났습니다")
    return int(w.read(cp / "manifest.json")["episodes"])


def next_chunk(current, milestones, chunk_size=42):
    future = [m for m in milestones if m > current]
    return min(min(future), current + chunk_size) if future else None


def train_all(source, out, work, record, workers, ledger, deadline):
    seeds = record["config"]["train_seeds"]
    milestones = record["profile"]["milestones"]
    current = {s: checkpoint_episodes(Path(out) / "models" / f"seed_{s}") for s in seeds}
    current = {s: (n if n is not None else (42 if s == 42 else 0)) for s, n in current.items()}
    active, failed = {}, False
    pool = ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"), initializer=_worker_init)
    try:
        while any(n < max(milestones) for n in current.values()) or active:
            if time.time() >= deadline - 10:
                raise StudyStopped("24시간 실행 상한에 도달했습니다. 마지막 완성 에피소드에서 재개할 수 있습니다")
            busy = {j["seed"] for j, _, _ in active.values()}
            ready = sorted((s for s in seeds if s not in busy and next_chunk(current[s], milestones) is not None),
                           key=lambda s: (current[s], s))
            for seed in ready[:max(0, workers-len(active))]:
                target = next_chunk(current[seed], milestones, record["config"]["chunk_episodes"])
                estimate = ((target-current[seed]) * record["config"]["budget"]["episode_seconds_estimate"] * 1.7 + 180)
                wall_left = deadline - time.time() - 10
                cap = min(estimate, wall_left, ledger.remaining)
                if cap < 180:
                    if active:
                        break
                    raise StudyStopped("다음 학습 구간의 예산이 부족합니다")
                job = {"seed": seed, "source_root": str(source), "out": str(out), "work": str(work),
                       "episodes": [target], "preserve_milestone": target in milestones,
                       "study_fingerprint": record["fingerprint"], "parent_fingerprint": record["parent_fingerprint"],
                       "global_deadline_unix": deadline-5, "job_cap_s": cap}
                token = ledger.begin(f"train/seed{seed}/{current[seed]}-{target}", cap)
                future = pool.submit(_training_worker, job)
                active[future] = (job, token, time.monotonic())
            if not active:
                continue
            done, _ = wait(active, timeout=min(20, max(1, deadline-time.time())), return_when=FIRST_COMPLETED)
            for future in done:
                job, token, started = active.pop(future)
                try:
                    result = future.result()
                except Exception as e:
                    ledger.finish(token, time.monotonic()-started, "failed")
                    raise RuntimeError(f"seed{job['seed']} 학습 오류: {e}") from e
                status = result.get("status", "complete")
                ledger.finish(token, max(result.get("seconds", 0), time.monotonic()-started), status)
                current[job["seed"]] = int(result["completed_episodes"])
                w.write(Path(out) / "training_progress.json", {"episodes": current, "primary": max(milestones)})
                print(f"LONG seed{job['seed']}: {current[job['seed']]}/{max(milestones)}에피소드, "
                      f"누적계산 {ledger.used/3600:.2f}/96시간", flush=True)
                if current[job["seed"]] < job["episodes"][-1]:
                    raise StudyStopped("작업 시간 한도에서 저장 후 중단했습니다. 같은 명령으로 재개하세요")
    except BaseException:
        failed = True
        _kill_pool(pool)
        for job, token, started in active.values():
            ledger.finish(token, time.monotonic()-started, "stopped_by_coordinator")
        raise
    finally:
        if not failed:
            pool.shutdown(wait=True)


def _eval_worker(job):
    started = time.monotonic()
    w.configure()
    from explore_weight_episode import run_episode
    policy = job["mode"]
    if job.get("model"):
        from explore_weight_train import load_model, predict_mode
        model = load_model(Path(job["model"]), fingerprint=job["parent_fingerprint"])
        policy = lambda obs, snapshot: predict_mode(model, obs)
    target = Path(job["target"])
    target.mkdir(parents=True, exist_ok=True)
    r = run_episode(job["date"], job["period"], job["seed"], Path(job["work"]), policy,
                    trace=True, max_seconds=max(1, job["cap_s"]-10-(time.monotonic()-started)))
    metrics = {**r.metrics, "policy": job["policy"], "study_fingerprint": job["study_fingerprint"],
               "parent_fingerprint": job["parent_fingerprint"], "model_sha256": job.get("model_sha256")}
    w.write(target / "metrics.json", metrics)
    with gzip.open(target / "decisions.jsonl.gz", "wt") as f:
        for row in r.decisions:
            f.write(json.dumps(w.plain(row), ensure_ascii=False) + "\n")
    seconds = time.monotonic()-started
    w.write(target / "cache.json", {"study_fingerprint": job["study_fingerprint"],
                                    "model_sha256": job.get("model_sha256"), "seconds": seconds,
                                    "metrics_sha256": w.file_hash(target / "metrics.json"),
                                    "trace_sha256": w.file_hash(target / "decisions.jsonl.gz")})
    return {"seconds": seconds, "metrics": metrics}


def reference_rows(source, policy, seeds):
    result = []
    parent_fingerprint = w.read(Path(source) / "experiment.json")["fingerprint"]
    for d,p in groups("development"):
        for seed in seeds:
            path = Path(source) / "runs/development" / f"{policy}_{d}_{p}_s{seed}"
            cache = w.read(path / "cache.json")
            if (cache["fingerprint"] != parent_fingerprint or
                    cache["metrics_sha256"] != w.file_hash(path / "metrics.json") or
                    cache["decisions_sha256"] != w.file_hash(path / "decisions.jsonl.gz")):
                raise ValueError("부모 개발 평가 캐시의 원시 파일 해시가 다릅니다")
            result.append(w.read(path / "metrics.json"))
    return result


def evaluations(out, work, record, workers, ledger, deadline, stage, policies, gs, eval_seeds):
    from explore_weight_train import checkpoint_identity
    results, jobs = {p: [] for p,_,_ in policies}, []
    for policy, mode, model in policies:
        identity = checkpoint_identity(Path(model), record["parent_fingerprint"]) if model else None
        for d,p in gs:
            for seed in eval_seeds:
                path = Path(out) / "runs" / stage / f"{policy}_{d}_{p}_s{seed}"
                if (path / "cache.json").exists():
                    cache = w.read(path / "cache.json")
                    if (cache["study_fingerprint"] != record["fingerprint"] or cache["model_sha256"] != identity or
                            cache["metrics_sha256"] != w.file_hash(path / "metrics.json") or
                            cache["trace_sha256"] != w.file_hash(path / "decisions.jsonl.gz")):
                        raise ValueError("학습량 평가의 캐시/모델/설정이 바뀌었습니다")
                    results[policy].append(w.read(path / "metrics.json"))
                else:
                    jobs.append({"date":d,"period":p,"seed":seed,"policy":policy,"mode":mode,
                                 "model":str(model) if model else None,"model_sha256":identity,
                                 "target":str(path),"work":str(Path(work)/stage/path.name),
                                 "study_fingerprint":record["fingerprint"],"parent_fingerprint":record["parent_fingerprint"]})
    pool = ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"), initializer=_worker_init)
    active, failed = {}, False
    try:
        while jobs or active:
            if time.time() >= deadline-10:
                raise StudyStopped("평가 중 24시간 상한에 도달했습니다. 완료된 실행을 보존하고 재개할 수 있습니다")
            while jobs and len(active)<workers:
                cap = min(900.0, deadline-time.time()-10, ledger.remaining)
                if cap < 60:
                    if active:break
                    raise StudyStopped("다음 평가를 위한 계산 예산이 부족합니다")
                job=jobs.pop(0);job["cap_s"]=cap
                token=ledger.begin(f"{stage}/{Path(job['target']).name}",cap)
                active[pool.submit(_eval_worker,job)]=(job,token,time.monotonic())
            if not active:continue
            done,_=wait(active,timeout=min(20,max(1,deadline-time.time())),return_when=FIRST_COMPLETED)
            for future in done:
                job,token,started=active.pop(future)
                try:r=future.result()
                except Exception:
                    ledger.finish(token,time.monotonic()-started,"failed")
                    raise
                ledger.finish(token,max(r["seconds"],time.monotonic()-started),"complete")
                results[job["policy"]].append(r["metrics"])
                print(f"{stage} {job['policy']} {job['date']} {job['period']} seed{job['seed']}: "
                      f"{r['metrics']['mean_delay_s']:.4f}초",flush=True)
    except BaseException:
        failed=True;_kill_pool(pool)
        for job,token,started in active.values():ledger.finish(token,time.monotonic()-started,"stopped_by_coordinator")
        raise
    finally:
        if not failed:pool.shutdown(wait=True)
    return results


def development_reading(rows, refs):
    if set(rows) != set(w.read(CONFIG)["train_seeds"]) or set(refs) != {"P0", "M4"}:
        raise ValueError("개발 비교는 모델3개와 P0·M4를 모두 포함해야 합니다")
    all_rows=[m for rs in rows.values() for m in rs]
    # Evaluate each seed separately, then aggregate three equal-cohort models.
    separate={seed:w.pilot_reading(rs,refs) for seed,rs in rows.items()}
    candidate=w.aggregate(all_rows);comparisons={}
    for name,ref in refs.items():
        reference=w.aggregate(ref)
        gain=reference["mean_delay_s"]-candidate["mean_delay_s"]
        by_model={s:v["comparisons"][name]["gain_s"] for s,v in separate.items()}
        by_group={g:sum(v["comparisons"][name]["groups"][g] for v in separate.values())/len(rows)
                  for g in next(iter(separate.values()))["comparisons"][name]["groups"]}
        checks={**w.guards(candidate,reference),"mean_improves":gain>0,
                "two_models_positive":sum(x>0 for x in by_model.values())>=2,
                "three_groups_positive":sum(x>0 for x in by_group.values())>=3}
        comparisons[name]={"gain_s":gain,"by_model":by_model,"by_group":by_group,"checks":checks,"passed":all(checks.values())}
    return {"passed":all(v["passed"] for v in comparisons.values()),"candidate":candidate,"comparisons":comparisons}


def confirmation_reading(candidates,refs,cfg,repeats=None):
    seeds=cfg["train_seeds"];drivers=cfg["confirmation"]["eval_seeds"];gs=groups("confirmation")
    expected={(d,p,s) for d,p in gs for s in drivers}
    if set(candidates)!=set(seeds):raise ValueError("최종 모델3개가 모두 있어야 합니다")
    if set(refs)!={"P0","M4"}:raise ValueError("P0와 M4를 모두 비교해야 합니다")
    for rows in [*candidates.values(),*refs.values()]:
        if len(rows)!=len(expected) or {(r['date'],r['period'],r['seed']) for r in rows}!=expected:
            raise ValueError("확인 평가 조건이 누락되거나 중복됐습니다")
    for rows in candidates.values():
        for ref in refs.values():w.verify_pairs(rows,ref)
    def arrays(rows):
        m={(r['date'],r['period'],r['seed']):r for r in rows}
        return (np.array([[m[d,p,s]['actual_total_delay'] for d,p in gs] for s in drivers]),
                np.array([[m[d,p,s]['vehicles'] for d,p in gs] for s in drivers]))
    costs=np.stack([arrays(candidates[s])[0] for s in seeds]);counts=arrays(candidates[seeds[0]])[1]
    n=cfg["gate"]["bootstrap_repeats"] if repeats is None else repeats
    rng=np.random.default_rng(cfg["gate"]["bootstrap_seed"])
    mi=rng.integers(0,len(seeds),(n,len(seeds)));di=rng.integers(0,len(drivers),(n,len(drivers)))
    candidate=w.aggregate([r for rows in candidates.values() for r in rows]);comparisons={}
    for name,ref in refs.items():
        rc,_=arrays(ref);base=w.aggregate(ref)
        distribution=[float((rc[d].sum()-costs[m][:,d].sum(axis=(1,2)).mean())/counts[d].sum()) for m,d in zip(mi,di)]
        ci=[float(x) for x in np.percentile(distribution,[2.5,97.5])]
        by_seed={s:base['mean_delay_s']-w.aggregate(candidates[s])['mean_delay_s'] for s in seeds}
        by_group={f'{d}_{p}':w.aggregate([r for r in ref if (r['date'],r['period'])==(d,p)])['mean_delay_s']-
                  w.aggregate([r for rs in candidates.values() for r in rs if (r['date'],r['period'])==(d,p)])['mean_delay_s'] for d,p in gs}
        checks={**w.guards(candidate,base),'mean_improves':base['mean_delay_s']>candidate['mean_delay_s'],
                'ci_lower_positive':ci[0]>0,'all_three_models_positive':all(g>0 for g in by_seed.values()),
                'six_groups_positive':sum(g>0 for g in by_group.values())>=6}
        comparisons[name]={'gain_s':base['mean_delay_s']-candidate['mean_delay_s'],'ci95_s':ci,
                           'by_training_seed':by_seed,'by_group':by_group,'checks':checks,'passed':all(checks.values())}
    return {'passed':all(v['passed'] for v in comparisons.values()),'candidate':candidate,'comparisons':comparisons,
            'interpretation':'학습seed3개·고정8수요조건의 추가 개발 확인 평가; 새 날짜/실제 도로 일반화 아님'}


def evaluate_stages(source,out,work,record,workers,ledger,deadline):
    cfg=record['config'];primary=record['profile']['primary_episodes']
    for episode in record['profile']['milestones']:
        seeds=cfg['development']['primary_seeds'] if episode==primary else cfg['development']['interim_seeds']
        policies=[(f'RL{s}',None,Path(out)/'models'/f'seed_{s}'/'stages'/f'episode_{episode:06d}') for s in cfg['train_seeds']]
        rows=evaluations(out,work,record,workers,ledger,deadline,f'dev_ep{episode}',policies,groups('development'),seeds)
        refs={'P0':reference_rows(source,'M0',seeds),'M4':reference_rows(source,'M4',seeds)}
        result=development_reading({s:rows[f'RL{s}'] for s in cfg['train_seeds']},refs)
        result.update(episodes=episode,primary=episode==primary)
        w.write(Path(out)/f'development_{episode}.json',result)
    main=w.read(Path(out)/f'development_{primary}.json')
    if not main['passed']:
        w.write(Path(out)/'outcome.json',{'status':'primary_development_not_passed','primary_episodes':primary,
                                        'message':'계획한 최종 학습까지 완료했으나 M4/P0 동시 개선 기준 미통과; 확인 평가 확대 없음'})
        return
    from explore_weight_train import checkpoint_identity
    policies=[('P0',None,None),('M4',4,None)]+[(f'RL{s}',None,Path(out)/'models'/f'seed_{s}'/'stages'/f'episode_{primary:06d}') for s in cfg['train_seeds']]
    identities={name:checkpoint_identity(Path(model),record['parent_fingerprint']) for name,_,model in policies if model}
    freeze=Path(out)/'confirmation_models.json'
    if freeze.exists() and w.read(freeze)!=identities:raise ValueError('확인 평가 모델이 바뀌었습니다')
    w.write(freeze,identities)
    rows=evaluations(out,work,record,workers,ledger,deadline,'confirmation',policies,groups('confirmation'),cfg['confirmation']['eval_seeds'])
    result=confirmation_reading({s:rows[f'RL{s}'] for s in cfg['train_seeds']},{'P0':rows['P0'],'M4':rows['M4']},cfg)
    w.write(Path(out)/'confirmation_summary.json',result)
    w.write(Path(out)/'outcome.json',{'status':'confirmation_passed' if result['passed'] else 'confirmation_not_passed','primary_episodes':primary})


def summary(out):
    out=Path(out)
    if not (out/'experiment.json').exists():return {}
    record=w.read(out/'experiment.json');rows=[]
    for path in sorted(out.glob('development_*.json'),key=lambda p:int(p.stem.split('_')[-1])):
        r=w.read(path);rows.append(r)
    result={'experiment':record['config']['experiment'],'profile':record['profile'],
            'development':rows,'budget':w.read(out/'budget.json') if (out/'budget.json').exists() else None}
    for name in ('outcome','stopped','training_progress','confirmation_summary'):
        if (out/f'{name}.json').exists():result[name]=w.read(out/f'{name}.json')
    result['completed_episodes']={seed:checkpoint_episodes(out/'models'/f'seed_{seed}') or 0
                                  for seed in record['config']['train_seeds']}
    w.write(out/'summary.json',result)
    lines=['# WEIGHT1 긴 학습 결과','','원래 pilot미통과는 보존한다. 반복 사용된 날짜의 추가 개발 평가다.','',
           f"- 고정 최종 학습: {record['profile']['primary_episodes']}에피소드 ×3모델",
           '- 완료 에피소드: '+', '.join(f"seed{s}={n}" for s,n in result['completed_episodes'].items()),
           '','| 에피소드 | 평균 지연(초) | P0보다 줄인 지연 | M4보다 줄인 지연 |','|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['episodes']} | {r['candidate']['mean_delay_s']:.6f} | {r['comparisons']['P0']['gain_s']:.6f} | {r['comparisons']['M4']['gain_s']:.6f} |")
    if result['budget']:lines+=['',f"- 누적 계산: {result['budget']['used_s']/3600:.3f}/96시간(이전 실행 포함)"]
    for name in ('outcome','stopped'):
        if name in result:lines+=['',f"- {name}: {json.dumps(result[name],ensure_ascii=False)}"]
    if 'confirmation_summary' in result:
        lines+=['',f"- 확인 평가: {'통과' if result['confirmation_summary']['passed'] else '미통과'}"]
    (out/'summary.md').write_text('\n'.join(lines)+'\n')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','run','summary'))
    parser.add_argument('--source',required=True)
    parser.add_argument('--out',required=True)
    parser.add_argument('--work',default='/content/weight_long_work')
    parser.add_argument('--workers',default='auto')
    parser.add_argument('--wall-hours',type=float,default=24)
    parser.add_argument('--compute-hours',type=float,default=96)
    args=parser.parse_args();source=Path(args.source).resolve();out=Path(args.out).resolve();work=Path(args.work).resolve()
    if source==out or source in out.parents or out in source.parents:raise ValueError('새 결과는원본WEIGHT1폴더밖의별도폴더에저장하세요')
    if work==source or source in work.parents or work in source.parents:
        raise ValueError('작업 폴더는 원본 WEIGHT1 자료와 겹칠 수 없습니다')
    out.mkdir(parents=True,exist_ok=True)
    with (out/'.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as e:raise SystemExit('같은 긴 학습이 이미 실행 중입니다') from e
        if args.command=='summary':summary(out);return
        if sys.platform=='darwin':raise SystemExit('이 명령은Colab실행전용입니다. 맥학습은시작하지않습니다')
        setup_started=time.monotonic()
        record,workers=register(source,out,args.workers,args.wall_hours,args.compute_hours)
        print(json.dumps({'profile':record['profile'],'workers':workers,'wall_hours':args.wall_hours,
                          'compute_hours':args.compute_hours,'prior_hours':record['prior_compute_s']/3600},ensure_ascii=False),flush=True)
        ledger=Ledger(out,record)
        setup_seconds=time.monotonic()-setup_started
        token=ledger.begin('setup/validation',max(.001,setup_seconds))
        ledger.finish(token,setup_seconds,'complete')
        if args.command=='prepare':return
        deadline=time.time()+args.wall_hours*3600-setup_seconds
        def terminate(signum, frame):
            raise StudyStopped('실행 중단 요청을 받아 완료된 에피소드를 보존합니다')
        signal.signal(signal.SIGTERM, terminate)
        w.write(out/'session.json',{'started_unix':time.time(),'deadline_unix':deadline,'workers':workers})
        (out/'stopped.json').unlink(missing_ok=True)
        try:
            train_all(source,out,work,record,workers,ledger,deadline)
            evaluate_stages(source,out,work,record,workers,ledger,deadline)
        except (StudyStopped,KeyboardInterrupt) as e:
            w.write(out/'stopped.json',{'reason':'time_budget_or_interrupt','message':str(e),'resume':'same command','models_preserved':True})
            print(str(e),flush=True)
        except BaseException as e:
            w.write(out/'stopped.json',{'reason':'execution_error','message':f'{type(e).__name__}: {e}','models_preserved':True})
            raise
        finally:
            summary(out)


if __name__=='__main__':
    main()
