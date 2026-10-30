"""Colab-only FIXED24: predeclared fixed-policy development sweep, then reused-date comparison.

No training; existing results are read-only. Restart with the same output directory after interruption.
"""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, FIRST_COMPLETED, wait
import fcntl
import hashlib
import importlib.metadata
import json
import multiprocessing
import os
from pathlib import Path
import random
import signal
import sys
import time

import explore_weight as w
import explore_weight_long as long
import explore_night as night

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'config/explore_fixed24_v1.json'
CODE = ('src/explore_fixed24.py', 'config/explore_fixed24_v1.json')


def config():
    return w.read(CONFIG)


def conditions():
    return [(c['date'], c['period'], int(c['seed'])) for c in config()['conditions']]


def confirmation_conditions():
    c = config()['confirmation']
    return [(d, p, s) for d in c['dates'] for p in c['periods'] for s in c['seeds']]


def verify_config(available=None):
    cfg = config()
    cs = conditions()
    if len(cs) != 24 or len(set((d, p) for d, p, s in cs)) != 24 or len({d for d, p, s in cs}) != 11:
        raise ValueError('사전 등록된 24개 상황 또는 11개 날짜가 다릅니다.')
    if Counter(p for _, p, _ in cs) != {p: 6 for p in ('0730', '1230', '1730', '2130')}:
        raise ValueError('시간대별 6개씩이 아닙니다.')
    if available is not None and any((d,p) not in available for d,p,s in cs):
        raise ValueError('학습 날짜의 실제 차량 수요에 없는 상황이 있습니다.')
    expected = [('P0',None,None),('M4',4,None),('W4',None,[4,4]),('W6',None,[6,6]),('W8',None,[8,8])]
    if [(p['tag'],p['mode'],p['factors']) for p in cfg['policies']] != expected or len(confirmation_conditions()) != 64:
        raise ValueError('정책 또는 비교 조건이 사전 설정과 다릅니다.')


def prior_seconds(night_dir, resume_dir):
    ledger = Path(night_dir)/'ledger'
    p = w.read(ledger/'prior.json')
    base = w.read(Path(resume_dir)/'budget.json')
    if base.get('pending') or abs(float(p['prior_used_s'])-float(base['used_s'])) > 0.001:
        raise ValueError('이전 장부에 미정산 작업이 있거나 밤샘 장부와 다릅니다.')
    total = float(p['prior_used_s'])
    for role in ('checks','evaluate','train_42','train_43','train_44'):
        total += float(w.read(ledger/f'{role}.json')['seconds'])
    for role in ('checks','evaluate'):
        status=w.read(Path(night_dir)/'status'/f'{role}.json')
        if status.get('failed') or status.get('stopped') or status.get('unsubmitted') or status.get('done') != 96:
            raise ValueError(f'이전 {role} 작업이 완료 상태가 아닙니다.')
    for seed in (42,43,44):
        status=w.read(Path(night_dir)/'status'/f'train_{seed}.json')
        if status.get('state') != 'complete' or status.get('completed') != 139:
            raise ValueError('이전 학습의 완료 상태가 다릅니다.')
    return total


def register(out, night_dir, resume_dir, source):
    w.configure()
    from focused_data import ENV_FILES, environment_fingerprint
    verify_config(set(w.training_scenarios()))
    w.validate_implementation(w.config())
    cfg=config()
    if environment_fingerprint()!=cfg['environment']['fingerprint'] or w.file_hash(ROOT/'src/explore_pred.py')[:12]!=cfg['environment']['p0_code']:
        raise ValueError('환경 또는 P0 지문이 다릅니다.')
    n=w.read(Path(night_dir)/'experiment.json')
    if n['fingerprint'] != w.digest({k:v for k,v in n.items() if k!='fingerprint'}):
        raise ValueError('밤샘 실험 지문이 다릅니다.')
    for name, sha in n['files'].items():
        if w.file_hash(ROOT/name) != sha:
            raise ValueError(f'밤샘 실험과 코드·자료가 다릅니다: {name}')
    versions={name:importlib.metadata.version(name) for name in night.PACKAGES}
    if versions != n['packages']:
        raise ValueError('밤샘 실험과 패키지 버전이 다릅니다.')
    resume=w.read(Path(resume_dir)/'experiment.json')
    if resume['fingerprint'] != w.digest({k:v for k,v in resume.items() if k!='fingerprint'}):
        raise ValueError('이전 64조건 실험 지문이 다릅니다.')
    if resume['parent_fingerprint'] != w.read(Path(source)/'experiment.json')['fingerprint']:
        raise ValueError('이전 학습과 원본 실험의 지문이 다릅니다.')
    prior=prior_seconds(night_dir,resume_dir)
    frozen={'kind':cfg['experiment'],'config':cfg,'night_fingerprint':n['fingerprint'],
            'resume_fingerprint':resume['fingerprint'], 'prior_compute_s':prior,'compute_hours':96,
            'files':{p:w.file_hash(ROOT/p) for p in CODE},'packages':versions}
    record={'fingerprint':w.digest(frozen),**frozen}
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    path=out/'experiment.json'
    if path.exists() and w.read(path)!=record:
        raise ValueError('기존 실행의 코드·설정·자료·이전 장부가 달라 재개할 수 없습니다.')
    if not path.exists():w.write(path,record)
    return record


def jobs_for(out, work, fp, part, policies, cs):
    jobs=[]
    for c in cs:
        for p in policies:
            jobs += night.eval_jobs(out, work, fp, part, p['tag'], [c], mode=p['mode'], factors=p['factors'])
    for j in jobs:
        j['max_seconds']=config()['budget']['job_timeout_s']
    return jobs


def read_job(job):
    m=night.cached(job)
    if m is not None:
        cache=w.read(Path(job['target'])/'cache.json')
        trace=Path(job['target'])/'decisions.jsonl.gz'
        if not trace.is_file() or w.file_hash(trace)!=cache.get('decisions_sha256'):
            raise ValueError(f"결정 기록 해시 오류: {job['target']}")
        if (m.get('policy'),m.get('date'),m.get('period'),int(m.get('seed',-1))) != (job['tag'],job['date'],job['period'],job['seed']):
            raise ValueError(f"저장된 정책 또는 조건이 다릅니다: {job['target']}")
        if m.get('factors') != job.get('factors') or m.get('experiment_fingerprint') != job['fingerprint']:
            raise ValueError(f"저장된 가중치 또는 지문이 다릅니다: {job['target']}")
    return m


def eval_worker(job):
    # Workers are reused across policies. A weighted run mutates MODE_FACTORS;
    # restore the original modes before and after every task, including failures.
    import explore_weight_core as core
    core.MODE_FACTORS=tuple(tuple(x) for x in w.config()['modes'])
    try:
        return night._eval_worker(job)
    finally:
        core.MODE_FACTORS=tuple(tuple(x) for x in w.config()['modes'])


def execute(jobs, ledger, workers, deadline):
    pending=[j for j in jobs if read_job(j) is None]
    print(f'총 {len(jobs)}회 중 {len(jobs)-len(pending)}회 재사용, {len(pending)}회 실행.',flush=True)
    pool=ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),initializer=long._worker_init)
    active={}; failed=False; cfg=config()['budget']
    try:
        while pending or active:
            while pending and len(active)<workers and time.time()<deadline:
                cap=cfg['job_reserve_s']
                if ledger.remaining < cap+cfg['stop_margin_s']:
                    break
                job=pending.pop(0)
                token=ledger.begin(job['part']+'/'+Path(job['target']).name,cap)
                try: future=pool.submit(eval_worker,job)
                except BaseException:
                    ledger.finish(token,0,'submit_failed'); raise
                active[future]=(job,token,time.monotonic())
            if not active:
                if pending: raise long.StudyStopped('시간 또는 계산 한도에 도달해 남은 작업을 시작하지 않았습니다.')
                break
            done,_=wait(active,timeout=10,return_when=FIRST_COMPLETED)
            for future in done:
                job,token,started=active.pop(future)
                elapsed=time.monotonic()-started
                try: result=future.result()
                except BaseException as e:
                    ledger.finish(token,elapsed,'failed');raise RuntimeError(f'작업 실패 {job["target"]}: {e}') from e
                ledger.finish(token,max(elapsed,float(result['seconds'])),'complete' if result['ok'] else 'failed')
                if not result['ok']: raise RuntimeError(f"실행 실패 {result['name']}: {result.get('error')}")
                read_job(job)
                print(f"완료 {result['name']} | 누적 {ledger.used/3600:.2f}/96 작업시간",flush=True)
        return True
    except BaseException:
        failed=True
        long._kill_pool(pool)
        for job,token,started in active.values():
            ledger.finish(token,time.monotonic()-started,'interrupted')
        raise
    finally:
        if not failed: pool.shutdown(wait=True)


def grouped(jobs):
    groups={}
    for j in jobs:
        m=read_job(j)
        if m is None: raise ValueError('완료되지 않은 작업이 있습니다.')
        groups.setdefault(j['tag'],[]).append(m)
    return groups


def choose_and_freeze(out, jobs):
    rows=grouped(jobs)
    expected=[p['tag'] for p in config()['policies']]
    if set(rows)!=set(expected) or any(len(rows[t])!=24 for t in expected): raise ValueError('24상황의 120개 결과가 없습니다.')
    for tag in expected:
        w.verify_pairs(rows[tag],rows['P0'])
    invalid={tag:[{'date':m['date'],'period':m['period'],'seed':m['seed'],'errors':w.valid_metrics(m)} for m in rows[tag] if w.valid_metrics(m)] for tag in expected}
    if any(invalid.values()):
        w.write(Path(out)/'quality_issues.json',invalid)
        raise ValueError('품질 오류가 있어 후보를 자동 선택하지 않습니다. quality_issues.json을 확인하세요.')
    results={tag:w.aggregate(rows[tag]) for tag in expected}
    chosen=min(expected,key=lambda tag:(results[tag]['mean_delay_s'],expected.index(tag)))
    digest=w.digest({j['tag']+'_'+j['date']+'_'+j['period']+'_'+str(j['seed']):w.file_hash(Path(j['target'])/'metrics.json') for j in jobs})
    record={'selected':chosen,'rule':config()['selection'],'sweep_metrics_digest':digest,
            'mean_delay_s':{k:results[k]['mean_delay_s'] for k in expected}}
    path=Path(out)/'selected.json'
    if path.exists() and w.read(path)!=record: raise ValueError('고정된 선택 결과 또는 원시 지표가 바뀌었습니다.')
    if not path.exists():w.write(path,record)
    return chosen,rows,results


def reference_rows(resume_dir, fp):
    root=Path(resume_dir)/'runs'/'confirmation'
    identities=w.read(Path(resume_dir)/'confirmation_models.json')
    def load_one(item):
        tag,d,p,s=item
        folder=root/f'{tag}_{d}_{p}_s{s}'
        c=w.read(folder/'cache.json')
        raw=(folder/'metrics.json').read_bytes();m=json.loads(raw)
        if c.get('study_fingerprint')!=fp or m.get('study_fingerprint')!=fp or c.get('metrics_sha256')!=hashlib.sha256(raw).hexdigest():
            raise ValueError(f'기존 비교 결과의 지문·해시 불일치: {folder}')
        if (m['policy'],m['date'],m['period'],int(m['seed']))!=(tag,d,p,s):
            raise ValueError(f'기존 비교 결과의 조건 불일치: {folder}')
        if c.get('model_sha256') != identities.get(tag) or m.get('model_sha256') != identities.get(tag):
            raise ValueError(f'기존 모델 식별이 다릅니다: {folder}')
        classification=m.get('collision_classification',{})
        if classification.get('confirmed_events'):
            raise ValueError(f'기존과 새 결과의 충돌 판정이 달라 검토가 필요합니다: {folder}')
        return tag,m
    items=[(tag,d,p,s) for tag in config()['confirmation']['references'] for d,p,s in confirmation_conditions()]
    refs={tag:[] for tag in config()['confirmation']['references']}
    with ThreadPoolExecutor(max_workers=16) as pool:
        for tag,m in pool.map(load_one,items):refs[tag].append(m)
    for tag in refs:w.verify_pairs(refs[tag],refs['P0'])
    return refs


def comparison(rows, refs):
    agg=w.aggregate(rows); out={}
    def movements(data):
        names=set().union(*(m.get('by_movement',{}) for m in data))
        result={}
        for name in sorted(names):
            entries=[m.get('by_movement',{}).get(name,{}) for m in data]
            count=sum(x.get('vehicles',0) for x in entries)
            if count:
                result[name]={'vehicles':count,'mean_delay_s':sum(x.get('vehicles',0)*x.get('mean_delay_s',0) for x in entries)/count}
        return result
    candidate_movements=movements(rows)
    for name,baseline in refs.items():
        w.verify_pairs(rows,baseline)
        a=w.aggregate(baseline)
        gains=[]
        rm={(m['date'],m['period'],int(m['seed'])):m for m in rows}
        for m in baseline:
            key=(m['date'],m['period'],int(m['seed']))
            gains.append({'date':key[0],'period':key[1],'seed':key[2], 'gain_s':m['mean_delay_s']-rm[key]['mean_delay_s']})
        groups={}
        for d,p in sorted({(m['date'],m['period']) for m in baseline}):
            b=[m for m in baseline if (m['date'],m['period'])==(d,p)]
            a_rows=[m for m in rows if (m['date'],m['period'])==(d,p)]
            groups[f'{d}_{p}']=w.aggregate(b)['mean_delay_s']-w.aggregate(a_rows)['mean_delay_s']
        out[name]={'gain_s':a['mean_delay_s']-agg['mean_delay_s'],'reference':a,'guards':w.guards(agg,a),
                   'paired_gains':gains,'groups':groups,'reference_movements':movements(baseline)}
    return {'candidate':agg,'candidate_movements':candidate_movements,'comparisons':out}


def exploratory_date_bootstrap(rows, baseline):
    """Paired date resampling; exploratory only, never a post-selection success gate."""
    w.verify_pairs(rows,baseline)
    indexed={(m['date'],m['period'],int(m['seed'])):m for m in rows}
    days={}
    for m in baseline:
        k=(m['date'],m['period'],int(m['seed']))
        day=days.setdefault(m['date'],[0.0,0])
        day[0]+=m['actual_total_delay']-indexed[k]['actual_total_delay']
        day[1]+=m['vehicles']
    keys=sorted(days)
    if len(keys)<2:raise ValueError('날짜별 구간을 계산할 상황이 부족합니다.')
    cfg=config()['exploratory_bootstrap'];rng=random.Random(cfg['seed']);samples=[]
    for _ in range(cfg['repetitions']):
        chosen=rng.choices(keys,k=len(keys))
        samples.append(sum(days[d][0] for d in chosen)/sum(days[d][1] for d in chosen))
    samples.sort()
    def q(p):
        i=(len(samples)-1)*p;lo=int(i);hi=min(lo+1,len(samples)-1)
        return samples[lo]+(samples[hi]-samples[lo])*(i-lo)
    return {'lower_95':q(.025),'upper_95':q(.975),'days':len(keys),'repetitions':len(samples),'note':cfg['note']}


def make_summary(out, sweep, selected, refs=None, selected_rows=None):
    result={'note':config()['interpretation'],'selected':selected,'sweep':{t:w.aggregate(v) for t,v in sweep.items()},
            'sweep_comparison':comparison(sweep[selected],{'P0':sweep['P0'],'M4':sweep['M4']})}
    result['exploratory_vs_m4']={tag:exploratory_date_bootstrap(rows,sweep['M4']) for tag,rows in sweep.items()}
    indexed={tag:{(m['date'],m['period'],int(m['seed'])):m for m in rows} for tag,rows in sweep.items()}
    order=[p['tag'] for p in config()['policies']]
    result['situations']=[]
    for d,p,s in conditions():
        vals={tag:indexed[tag][(d,p,s)]['mean_delay_s'] for tag in order}
        result['situations'].append({'date':d,'period':p,'seed':s,'mean_delay_s':vals,
            'hindsight_best_fixed':min(order,key=lambda tag:(vals[tag],order.index(tag)))})
    if refs is not None and selected_rows is not None:
        result['reused64']=comparison(selected_rows,refs)
        combined=[m for tag in ('RL42','RL43','RL44') for m in refs[tag]]
        rl=w.aggregate(combined)
        result['reused64']['rl_three_mean_delay_s']=rl['mean_delay_s']
        result['reused64']['gain_vs_rl_three_s']=rl['mean_delay_s']-result['reused64']['candidate']['mean_delay_s']
        result['reused64']['guards_vs_rl_three']=w.guards(result['reused64']['candidate'],rl)
    w.write(Path(out)/'summary.json',result)
    lines=['# 고정 가중치 24상황 비교', '', '반복 사용한 날짜의 추가 개발 비교. RL 학습과 독립 최종 검증이 아닙니다.',
           '',f'선정 후보: **{selected}** (24상황 평균 지연이 가장 낮은 정책, 품질 검사는 별도 표시)', '',
           '| 정책 | 차량당 평균 지연(초) | 동쪽 좌회전(초) | 남쪽 좌회전(초) | 300초 초과 대기 비율 | M4 대비 탐색 구간(초) |',
           '|---|---:|---:|---:|---:|---:|']
    for tag,v in result['sweep'].items():
        def f(x):return '-' if x is None else f'{x:.3f}'
        ci=result['exploratory_vs_m4'][tag]
        lines.append(f"| {tag} | {f(v['mean_delay_s'])} | {f(v['left_east'])} | {f(v['left_south'])} | {v['long_wait_rate']:.6f} | [{ci['lower_95']:+.2f}, {ci['upper_95']:+.2f}] |")
    lines += ['', 'M4 대비 구간은 11일을 날짜 단위로 10,000번 재표집한 탐색적 구간입니다. 후보를 고른 편향과 날짜 선정의 한계가 있어 최종 성공 판정에 쓰지 않습니다.']
    lines += ['', '## 상황별 결과', '', '| 날짜 | 시각 | 난수 | P0 | M4 | 4배 | 6배 | 8배 | 사후 최선 고정 |',
              '|---|---|---:|---:|---:|---:|---:|---:|---|']
    for row in result['situations']:
        v=row['mean_delay_s']
        lines.append(f"| {row['date']} | {row['period']} | {row['seed']} | {v['P0']:.2f} | {v['M4']:.2f} | {v['W4']:.2f} | {v['W6']:.2f} | {v['W8']:.2f} | {row['hindsight_best_fixed']} |")
    lines += ['', '각 줄의 사후 최선은 결과를 보고 고른 참고값이며 동적 RL의 성능 상한이 아닙니다.']
    if 'reused64' in result:
        rr=result['reused64'];lines += ['', '## 기존 64조건 추가 개발 비교', '',f"후보 평균 {rr['candidate']['mean_delay_s']:.3f}초. 기존 RL 3개 평균 {rr['rl_three_mean_delay_s']:.3f}초, 후보의 RL 평균 대비 감소 {rr['gain_vs_rl_three_s']:+.3f}초."]
        for tag,c in rr['comparisons'].items():
            lines.append(f"- {tag} 대비 감소 {c['gain_s']:+.3f}초; 품질·대기 검사: {c['guards']}")
        lines += ['', '기존 미확인 충돌 등 품질 문제가 있으면 성능 채택 판정이 아닙니다. 같은 날짜에서 결과를 보고 개발했으므로 독립 검증이 아닙니다.']
    else:lines += ['', '기존 64조건 비교는 아직 완료되지 않았습니다.']
    lines += ['', '상황별 원시 짝 비교와 품질 정보는 summary.json에 있습니다. 24상황 결과의 사후 선택 편향과 하루 안의 상관을 고려하세요.']
    (Path(out)/'summary.md').write_text('\n'.join(lines)+'\n')


def run(args):
    out,work,night_dir,resume,source=map(Path,(args.out,args.work,args.night,args.resume,args.source))
    paths=[p.resolve() for p in (out,work,night_dir,resume,source)]
    for i,a in enumerate(paths):
        for b in paths[i+1:]:
            if a==b or a in b.parents or b in a.parents:raise ValueError('출력·원본·작업 경로가 겹칩니다.')
    out.mkdir(parents=True,exist_ok=True)
    with (out/'.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('같은 결과 폴더에서 이미 실행 중입니다.')
        rec=register(out,night_dir,resume,source); ledger=long.Ledger(out,rec)
        cs=conditions();jobs=jobs_for(out,work,rec['fingerprint'],'sweep',config()['policies'],cs)
        workers=max(1,min(args.workers,long.available_cpus(),4));deadline=time.time()+config()['budget']['wall_hours']*3600
        print(f'지문 {rec["fingerprint"][:12]}, 이전 사용 {ledger.used/3600:.2f}/96 작업시간, CPU 작업자 {workers}개',flush=True)
        # First paired condition is a real part of the sweep, not extra unbudgeted smoke runs.
        execute(jobs[:5],ledger,workers,deadline)
        first=grouped(jobs[:5])
        for tag in first:
            w.verify_pairs(first[tag],first['P0'])
            if w.valid_metrics(first[tag][0]):raise ValueError(f'첫 상황 점검 실패 {tag}: {w.valid_metrics(first[tag][0])}')
        print('첫 상황 5정책 실행·차량 짝·정산 검사 통과',flush=True)
        execute(jobs,ledger,workers,deadline)
        chosen,sweep,_=choose_and_freeze(out,jobs)
        make_summary(out,sweep,chosen)
        print(f'24상황 완료, 사전 규칙의 후보 {chosen}. 기존 64조건 자료를 검증합니다.',flush=True)
        refs=reference_rows(resume,rec['resume_fingerprint'])
        if chosen in ('P0','M4'):
            selected_rows=refs[chosen]
        else:
            policy=next(p for p in config()['policies'] if p['tag']==chosen)
            check_jobs=jobs_for(out,work,rec['fingerprint'],'reused64',[policy],confirmation_conditions())
            execute(check_jobs,ledger,workers,deadline)
            selected_rows=grouped(check_jobs)[chosen]
        make_summary(out,sweep,chosen,refs,selected_rows)
        print(f'완료. 누적 {ledger.used/3600:.2f}/96 작업시간. 요약: {out/"summary.md"}',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('out','work','night','resume','source'):p.add_argument('--'+name,required=True)
    p.add_argument('--workers',type=int,default=4)
    args=p.parse_args()
    if sys.platform=='darwin':raise SystemExit('실제 실행은 Colab에서만 합니다. 맥에서는 실행하지 않습니다.')
    if args.workers<1:raise SystemExit('작업자 수는 1 이상이어야 합니다.')
    def stop(signum, frame):
        raise KeyboardInterrupt('중단 요청을 받아 진행 상황을 저장합니다.')
    signal.signal(signal.SIGTERM,stop)
    run(args)

if __name__=='__main__':main()
