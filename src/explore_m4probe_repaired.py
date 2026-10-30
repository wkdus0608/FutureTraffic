"""Run the M4 diagnostic with validated initial pending-vehicle restoration and no compute limits.

Uses the original frozen scientific protocol and verifies/reuses existing cached states and rollouts.
Existing results and their original experiment fingerprint remain intact; this continuation records
its own code fingerprint and a separate execution journal. Never run alongside the original notebook.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, FIRST_COMPLETED, wait
import fcntl
import multiprocessing
from pathlib import Path
import signal
import sys
import time
import uuid

import explore_m4probe as base
import explore_m4probe_pending as repair
import explore_weight as w
import explore_weight_long as long

ROOT=Path(__file__).resolve().parents[1]


class Journal:
    """Reproducibility journal only: no compute budget, deadline, or cost gate."""
    def __init__(self,out,experiment_fingerprint):
        self.path=Path(out)/'continuation_journal.json'
        if self.path.exists():
            r=w.read(self.path)
            if r['experiment_fingerprint']!=experiment_fingerprint:raise ValueError('재개 장부의 실험 지문 오류')
            self.entries=r['entries'];self.pending=r['pending'];self.legacy=r['legacy_budget']
            # A hard Colab interruption can leave unresolved jobs. Do not claim an exact elapsed time.
            for token,item in self.pending.items():
                self.entries.append({**item,'token':token,'seconds':None,'status':'interrupted_duration_unknown'})
            self.pending={}
        else:
            old=Path(out)/'budget.json'
            self.legacy={'path':str(old),'sha256':w.file_hash(old) if old.exists() else None,
                         'snapshot':w.read(old) if old.exists() else None}
            self.entries=[];self.pending={}
        self.fingerprint=experiment_fingerprint
        self.save()

    def save(self):
        w.write(self.path,{'experiment_fingerprint':self.fingerprint,'legacy_budget':self.legacy,
                           'entries':self.entries,'pending':self.pending,
                           'note':'기록만 보존한다. 계산 예산·실행 시간으로 중단하지 않는다.'})

    def begin(self,name):
        token=uuid.uuid4().hex
        self.pending[token]={'job':name,'started_unix':time.time()};self.save();return token

    def finish(self,token,seconds,status):
        item=self.pending.pop(token)
        self.entries.append({**item,'token':token,'seconds':max(0.,float(seconds)),'status':status});self.save()


def collect_worker(job):
    return repair.collect_worker(job)


def rollout_job(out,fp,record,state,phase,candidate,index):
    job=base.rollout_job(out,fp,record,state,phase,candidate,index)
    job['expected_snapshot']={key:state[key] for key in ('allowed','mode_actions','mode_mask','p0_action','observation')}
    return job


def execute(jobs,kind,journal,workers):
    read=base.collect_cached if kind=='collect' else base.rollout_cached
    worker=collect_worker if kind=='collect' else repair.rollout_worker
    pending=[j for j in jobs if read(j) is None]
    print(f'{kind}: 전체 {len(jobs)}개, 재사용 {len(jobs)-len(pending)}개, 남은 {len(pending)}개',flush=True)
    if not pending:return
    pool=ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),initializer=long._worker_init)
    active={};failed=False
    try:
        while pending or active:
            while pending and len(active)<workers:
                job=pending.pop(0);token=journal.begin(kind+'/'+Path(job['target']).name)
                try:future=pool.submit(worker,job)
                except BaseException:journal.finish(token,0,'submit_failed');raise
                active[future]=(job,token,time.monotonic())
            if not active:break
            done,_=wait(active,timeout=10,return_when=FIRST_COMPLETED)
            for future in done:
                job,token,start=active.pop(future);elapsed=time.monotonic()-start
                try:res=future.result()
                except BaseException as e:
                    journal.finish(token,elapsed,'failed');raise RuntimeError('작업 프로세스 오류') from e
                journal.finish(token,max(elapsed,float(res['seconds'])),'complete' if res['ok'] else 'failed')
                if not res['ok']:raise RuntimeError(f"{kind} 실패 {res['name']}: {res.get('error')}")
                read(job)
                print(f"{kind} 완료 {res['name']}",flush=True)
    except BaseException:
        failed=True;long._kill_pool(pool)
        for job,token,start in active.values():journal.finish(token,time.monotonic()-start,'interrupted')
        raise
    finally:
        if not failed:pool.shutdown(wait=True)


def register_continuation(out,record):
    frozen={'kind':'M4_ACTION_PROBE_PENDING_REPAIR_V1','original_experiment_fingerprint':record['fingerprint'],
            'driver_sha256':w.file_hash(ROOT/'src/explore_m4probe_repaired.py'),
            'worker_sha256':w.file_hash(ROOT/'src/explore_m4probe.py'),
            'repair_sha256':w.file_hash(ROOT/'src/explore_m4probe_pending.py'),
            'scientific_protocol':'기존 상황·후보·미래·판정을 유지하며 초기 대기 차량 목록을 복원하고 원래 M4 궤적 300초를 매 단계 검증한다.',
            'runtime_limits':'no user-imposed compute-budget, per-job compute deadline, or wall-time cutoff'}
    protocol={'fingerprint':w.digest(frozen),**frozen}
    path=Path(out)/'continuation_protocol.json'
    if path.exists() and w.read(path)!=protocol:
        raise ValueError('재개 실행의 코드나 기존 실험 지문이 다릅니다.')
    if not path.exists():w.write(path,protocol)
    return protocol


def run(args):
    out,work,fixed,night,source=map(Path,(args.out,args.work,args.fixed,args.night,args.source))
    paths=[x.resolve() for x in (out,work,fixed,night,source)]
    for i,a in enumerate(paths):
        for b in paths[i+1:]:
            if a==b or a in b.parents or b in a.parents:raise ValueError('출력·원본·작업 폴더가 겹칩니다.')
    out.mkdir(parents=True,exist_ok=True)
    with (out/'.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('같은 결과 폴더가 실행 중입니다. 기존 실행을 먼저 중지하세요.')
        rec=base.register(out,fixed,night,source)
        protocol=register_continuation(out,rec)
        journal=Journal(out,rec['fingerprint'])
        cpus=long.available_cpus()
        workers=cpus if args.workers==0 else min(cpus,args.workers)
        print(f"M4 복원 검증 {rec['fingerprint'][:12]} 시작; 시간·계산량 제한 없음; 작업자 {workers}개",flush=True)
        print(f"실행 코드 지문 {protocol['fingerprint'][:12]} (기존 결과·설정은 유지)",flush=True)
        import focused_train as ft
        from focused_scenario import dqn_template_network
        dqn_template_network(ft.CACHE)
        cj=base.collect_jobs(out,work,rec['fingerprint'])
        failed_case=next(j for j in cj if base.scenario_id(j['condition'])=='20220812_0730_s43')
        print('이전에 복원 검사가 실패한 상황부터 300초 연속 재현을 검증합니다.',flush=True)
        execute([failed_case],'collect',journal,1)
        print('실패했던 상황의 300초 복원 검증 통과',flush=True)
        execute(cj,'collect',journal,workers)
        records=[base.collect_cached(j) for j in cj]
        selectable=[]
        for r in records:
            for s in r['states']:
                for a in sorted(set(s['mode_actions'])):
                    for i in range(4):selectable.append(rollout_job(out,rec['fingerprint'],r,s,'select',a,i))
        if not selectable:
            base.make_summary(out,records,{},{});print('선택 지점이 없어 근거 부족으로 종료',flush=True);return
        pilot=selectable[:4]
        execute(pilot,'rollout',journal,1)
        print('첫 표본 미래 검사 통과. 시간이나 예산으로 중단하지 않고 이어갑니다.',flush=True)
        execute(selectable,'rollout',journal,workers)
        choices={};validation_jobs=[]
        for r in records:
            for s in r['states']:
                candidates=sorted(set(s['mode_actions']))
                costs={a:[base.rollout_cached(rollout_job(out,rec['fingerprint'],r,s,'select',a,i))['cost'] for i in range(4)] for a in candidates}
                selected=base.choose(s['m4_action'],candidates,costs)
                import statistics
                choices[s['id']]={'m4':s['m4_action'],'chosen':selected,'selection_mean_costs':{str(k):statistics.fmean(v) for k,v in costs.items()}}
                if selected!=s['m4_action']:
                    for action in (s['m4_action'],selected):
                        for i in range(4):validation_jobs.append(rollout_job(out,rec['fingerprint'],r,s,'validate',action,i))
        choice_path=out/'choices.json';choice_record={'fingerprint':rec['fingerprint'],'choices':choices}
        if choice_path.exists() and w.read(choice_path)!=choice_record:raise ValueError('기존 후보 선택이 바뀌었습니다.')
        if not choice_path.exists():w.write(choice_path,choice_record)
        execute(validation_jobs,'rollout',journal,workers)
        validation={}
        for r in records:
            for s in r['states']:
                selected=choices[s['id']]['chosen'];baseline=s['m4_action']
                if selected!=baseline:
                    validation[s['id']]={(a,i):base.rollout_cached(rollout_job(out,rec['fingerprint'],r,s,'validate',a,i))
                                         for a in (baseline,selected) for i in range(4)}
        result=base.make_summary(out,records,choices,validation)
        print(f"완료: {result['verdict']}. 요약: {out/'summary.md'}",flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('out','work','fixed','night','source'):p.add_argument('--'+name,required=True)
    p.add_argument('--workers',type=int,default=0,help='0: 사용 가능한 CPU 자동 설정')
    args=p.parse_args()
    if sys.platform=='darwin':raise SystemExit('실제 진단은 Colab에서만 합니다.')
    if args.workers<0:raise ValueError('작업자 수는 0 이상이어야 합니다.')
    def stop(signum,frame):raise KeyboardInterrupt('중단 요청')
    signal.signal(signal.SIGTERM,stop)
    run(args)

if __name__=='__main__':main()
