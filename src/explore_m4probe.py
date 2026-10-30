"""Colab-only diagnostic: candidate green versus M4 on the same saved M4 state.

Past-based synthetic futures are split into independent selection and validation sets.
The 300-second proxy is not the full-episode objective; this program does not train RL.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, FIRST_COMPLETED, wait
import copy
import fcntl
import hashlib
import importlib.metadata
import json
import math
import multiprocessing
import os
from pathlib import Path
import pickle
import shutil
import signal
import statistics
import sys
import tempfile
import time

import explore_weight as w
import explore_weight_long as long

ROOT=Path(__file__).resolve().parents[1]
CONFIG=ROOT/'config/explore_m4probe_v1.json'
CODE=('src/explore_m4probe.py','config/explore_m4probe_v1.json')


def config():return w.read(CONFIG)


def scenario_id(c):return f"{c['date']}_{c['period']}_s{int(c['seed'])}"


def state_id(c,index):return f"{scenario_id(c)}_w{index}"


def future_seed(state, phase, index):
    if phase not in ('select','validate') or index not in range(4):raise ValueError('미래 표본 키 오류')
    raw=f"{config()['rollout']['seed_salt']}|{state}|{phase}|{index}".encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:4],'big')


def validate_config(available=None):
    c=config();cases=c['conditions']
    split=w.read(ROOT/'config/focused_split.json')
    weekdays=set(split['day_types']['weekday']);weekends=set(split['day_types']['weekend'])
    if len(cases)!=16 or len({scenario_id(x) for x in cases})!=16:raise ValueError('상황 16개가 유일하지 않습니다.')
    if len({x['date'] for x in cases})!=4:raise ValueError('학습 날짜 4개가 아닙니다.')
    for period in ('0730','1230','1730','2130'):
        group=[x for x in cases if x['period']==period]
        if len(group)!=4 or sorted(x['seed'] for x in group)!=[42,42,43,43]:raise ValueError('시간대별 난수가 균형이 아닙니다.')
        if sum(x['date'] in weekdays for x in group)!=2 or sum(x['date'] in weekends for x in group)!=2:
            raise ValueError('각 시간대에는 평일과 주말이 각각 두 실행씩 있어야 합니다.')
    if available is not None and any((x['date'],x['period']) not in available for x in cases):raise ValueError('학습 수요에 없는 상황입니다.')
    if c['windows_s']!=[[600,1500],[1800,2700]] or c['rollout']['selection_futures']!=4 or c['rollout']['validation_futures']!=4:
        raise ValueError('시간창 또는 미래 선택 규칙이 다릅니다.')
    if c['mode_weights']!=w.config()['modes']:raise ValueError('기존 5모드와 가중치가 다릅니다.')
    seeds=[future_seed(state_id(x,i),p,j) for x in cases for i in range(2) for p in ('select','validate') for j in range(4)]
    if len(seeds)!=len(set(seeds)):raise ValueError('미래 표본 난수 충돌이 있습니다.')


def register(out, fixed, night, source):
    from focused_data import environment_fingerprint
    import explore_rollout as er
    w.configure();w.validate_implementation(w.config());validate_config(set(w.training_scenarios()))
    if er.HISTORY_S!=900 or er.TERMINAL_S!=config()['rollout']['terminal_s']:
        raise ValueError('표본 미래의 과거 구간 또는 종단 비용이 설정과 다릅니다.')
    n=w.read(Path(night)/'experiment.json');f=w.read(Path(fixed)/'experiment.json');b=w.read(Path(fixed)/'budget.json')
    for name,exp in (('night',n),('fixed24',f)):
        if exp['fingerprint']!=w.digest({k:v for k,v in exp.items() if k!='fingerprint'}):raise ValueError(f'{name} 지문 불일치')
    if b.get('pending'):raise ValueError('이전 FIXED24 장부에 미정산 작업이 있습니다.')
    if abs(float(b['used_s'])-(float(b['prior_compute_s'])+sum(float(x['seconds']) for x in b['entries'])))>1e-5:
        raise ValueError('이전 FIXED24 장부 합산이 다릅니다.')
    if f['night_fingerprint']!=n['fingerprint'] or f.get('kind')!='WEIGHT_FIXED24_V1':raise ValueError('이전 실험 계보가 다릅니다.')
    if w.read(Path(fixed)/'selected.json')['selected']!='M4':raise ValueError('고정 후보가 M4가 아닙니다.')
    previous_summary=w.read(Path(fixed)/'summary.json')
    if previous_summary.get('selected')!='M4' or len(previous_summary.get('situations',[]))!=24 or 'reused64' not in previous_summary:
        raise ValueError('이전 FIXED24의 24상황 및 기존 64조건 비교가 완성되지 않았습니다.')
    if environment_fingerprint()!=n['environment_fingerprint']:raise ValueError('환경 지문이 이전 실험과 다릅니다.')
    for name,sha in n['files'].items():
        if w.file_hash(ROOT/name)!=sha:raise ValueError(f'기존 환경·코드 지문이 다릅니다: {name}')
    for name,sha in f['files'].items():
        if w.file_hash(ROOT/name)!=sha:raise ValueError(f'FIXED24 코드 지문이 다릅니다: {name}')
    versions={k:importlib.metadata.version(k) for k in n['packages']}
    if versions!=n['packages']:raise ValueError('기존 Colab 패키지 버전이 다릅니다.')
    source_exp=w.read(Path(source)/'experiment.json')
    if source_exp.get('environment_fingerprint')!=n['environment_fingerprint']:raise ValueError('원본 환경이 다릅니다.')
    frozen={'kind':config()['experiment'],'config':config(),'prior_fingerprint':f['fingerprint'],
            'night_fingerprint':n['fingerprint'],'prior_compute_s':float(b['used_s']),'compute_hours':96,
            'files':{x:w.file_hash(ROOT/x) for x in CODE},'packages':versions}
    rec={'fingerprint':w.digest(frozen),**frozen};path=Path(out)/'experiment.json'
    if path.exists() and w.read(path)!=rec:raise ValueError('이어 실행의 설정·자료·장부가 바뀌었습니다.')
    if not path.exists():w.write(path,rec)
    return rec


def digest_files(folder,names):return {name:w.file_hash(Path(folder)/name) for name in names}


def collect_cached(job):
    folder=Path(job['target']);record=folder/'episode.json'
    if not record.exists():
        if folder.exists():raise ValueError(f'완료 기록 없는 상태 폴더가 있어 검토가 필요합니다: {folder}')
        return None
    rec=w.read(record)
    if rec.get('fingerprint')!=job['fingerprint'] or rec.get('condition')!=job['condition']:
        raise ValueError('저장 상태의 지문 또는 조건 오류')
    if digest_files(folder,rec['files'])!=rec['files']:raise ValueError('저장 상태 또는 차량 파일 해시 오류')
    for s in rec['states']:
        if not s['restore_ok']:raise ValueError('복원 검사 미통과')
    return rec


def collect_worker(job):
    import explore_rollout as er
    from explore_weight_core import ModePlanner
    from explore_weight_episode import _Episode
    c=job['condition'];er.SUMO_SEED=int(c['seed']);started=time.monotonic()
    local=Path(job['work']);target=Path(job['target']);shutil.rmtree(local,ignore_errors=True);local.mkdir(parents=True)
    states=[];ep=None
    try:
        ep=_Episode(c['date'],c['period'],int(c['seed']),local/'episode',deadline=time.monotonic()+540)
        ep.date=c['date'];ep.period=c['period']
        planner=ModePlanner(ep.movement_of,[p.state for p in ep.ts.green_phases],ep.handover+3600)
        stem_dir=local/'states';stem_dir.mkdir()
        while float(ep.env.sim_step)-ep.handover < config()['windows_s'][-1][1]:
            snap=planner.snapshot(ep.env);elapsed=float(ep.env.sim_step)-ep.handover
            wanted=next((i for i,(a,b) in enumerate(config()['windows_s']) if a<=elapsed<b and not any(z['window']==i for z in states)),None)
            rec=None
            if wanted is not None and snap.mode_disagrees:
                name=state_id(c,wanted);stem=stem_dir/name
                saved=er.Episode.snapshot(ep,stem,snap.allowed,snap.p0_action)
                rec={'id':name,'window':wanted,'elapsed_s':elapsed,'t0':snap.now,'allowed':list(snap.allowed),
                     'mode_actions':list(snap.mode_actions),'mode_mask':list(snap.mode_mask),'m4_action':int(snap.mode_actions[4]),
                     'p0_action':int(snap.p0_action),'trace_before':er.trace_fingerprint(ep.env.sumo),
                     'history_cutoff':saved['t0'],'route':ep.route.name,'observation':snap.observation.tolist()}
            action=int(snap.mode_actions[4]);corrected,executed=ep.apply(action)
            if corrected!=action or executed!=action:raise RuntimeError('M4 실행 규칙이 선택을 바꿨습니다.')
            if rec is not None:
                rec['trace_after']=er.trace_fingerprint(ep.env.sumo);states.append(rec)
            if ep.done:raise RuntimeError('시간창을 끝내기 전에 에피소드가 끝났습니다.')
        route=ep.route;ep.close();ep=None
        from focused_eval import judge_collisions
        collision=judge_collisions(local/'episode'/'collisions.xml')
        if collision['collisions_at_204820'] or collision['collisions_unexplained']:
            raise RuntimeError('M4 상태 수집 중 미확인 충돌이 발생했습니다.')
        staging=local/'output';staging.mkdir()
        shutil.copy2(route,staging/route.name)
        for s in states:
            for ext in ('.xml','.pkl'):shutil.copy2(stem_dir/(s['id']+ext),staging/(s['id']+ext))
        # Actual future is used ONLY to verify exact state restoration; no candidate evaluation uses it.
        for s in states:
            env=cons=ts=ctl=None
            try:
                env,cons,ts,ctl,saved,_,_=er.restore(staging/s['id'],staging,'real',horizon=10)
                restored=ModePlanner(ctl.movement_of,[p.state for p in ts.green_phases],saved['handover']+3600).snapshot(env)
                before=er.trace_fingerprint(env.sumo)
                a=int(restored.mode_actions[4]);corrected=int(cons(env,a));env.step(corrected)
                after=er.trace_fingerprint(env.sumo)
                s['restore_ok']=(before==s['trace_before'] and after==s['trace_after'] and a==s['m4_action']
                                 and list(restored.mode_actions)==s['mode_actions'] and list(restored.mode_mask)==s['mode_mask']
                                 and list(restored.allowed)==s['allowed'] and corrected==a
                                 and len(restored.observation)==len(s['observation'])
                                 and all(abs(float(a)-float(b))<1e-6 for a,b in zip(restored.observation,s['observation'])))
                if not s['restore_ok']:raise RuntimeError(f"저장 상태 복원이 원래 M4와 다릅니다: {s['id']}")
            finally:
                if env is not None:env.close()
        files=[route.name]+[s['id']+ext for s in states for ext in ('.xml','.pkl')]
        rec={'fingerprint':job['fingerprint'],'condition':c,'states':states,'files':digest_files(staging,files),
             'elapsed_s':time.monotonic()-started,'note':'실제 미래는 상태 복원 검증에만 사용'}
        w.write(staging/'episode.json',rec)
        target.parent.mkdir(parents=True,exist_ok=True)
        temporary=target.parent/('.'+target.name+'.incomplete')
        if temporary.exists():shutil.rmtree(temporary)
        if target.exists():raise ValueError('기존 완료 상태 폴더를 덮어쓸 수 없습니다.')
        shutil.copytree(staging,temporary)
        if digest_files(temporary,files)!=rec['files']:raise ValueError('저장 상태 복사 해시가 다릅니다.')
        temporary.rename(target)
        return {'ok':True,'seconds':time.monotonic()-started,'name':scenario_id(c)}
    except BaseException as e:
        return {'ok':False,'seconds':time.monotonic()-started,'name':scenario_id(c),'error':f'{type(e).__name__}: {e}'}
    finally:
        if ep is not None:ep.close()
        shutil.rmtree(local,ignore_errors=True)


def rollout_cached(job):
    p=Path(job['target'])
    if not p.exists():return None
    digest=p.with_suffix('.sha256')
    if not digest.exists():return None  # interrupted between atomic result and hash writes
    if digest.read_text().strip()!=w.file_hash(p):raise ValueError(f'롤아웃 결과 해시 오류 {p}')
    r=w.read(p)
    for k in ('fingerprint','state','phase','future_seed','candidate','snapshot_hash'):
        if r.get(k)!=job[k]:raise ValueError(f'저장 롤아웃 정보 불일치 {p}: {k}')
    if r.get('leaks') or r.get('changed_by_rules') or r.get('collisions_unexplained') or r.get('collisions_at_204820') or not math.isfinite(float(r['cost'])):
        raise ValueError(f'저장 롤아웃 품질 오류 {p}')
    return r


def rollout_worker(job):
    import explore_rollout as er
    from explore_weight_core import ModePlanner
    started=time.monotonic();er.SUMO_SEED=int(job['driver_seed']);env=None
    trip_fd,trip_name=tempfile.mkstemp(prefix='m4probe_',suffix='.xml');os.close(trip_fd);trip=Path(trip_name)
    collision_fd,collision_name=tempfile.mkstemp(prefix='m4probe_collision_',suffix='.xml');os.close(collision_fd);collision_file=Path(collision_name)
    original_env=er._env
    try:
        stem=Path(job['state_dir'])/job['state'];folder=Path(job['state_dir'])
        if w.file_hash(stem.with_suffix('.xml'))!=job['snapshot_hash']:raise ValueError('상태 해시가 다릅니다.')
        def env_with_collision(route,extra,duration):
            return original_env(route,f'{extra} --collision-output {collision_file}',duration)
        er._env=env_with_collision
        env,cons,ts,ctl,snap,injected,vehicles=er.restore(stem,folder,'sampled',int(job['future_seed']),300,tripinfo=trip)
        planner=ModePlanner(ctl.movement_of,[p.state for p in ts.green_phases],snap['handover']+3600)
        check=planner.snapshot(env)
        if list(check.mode_actions)!=job['mode_actions'] or int(check.mode_actions[4])!=job['m4_action']:
            raise ValueError('표본 미래 주입 후 현재 행동이 저장 상태와 다릅니다.')
        if job['candidate'] not in set(check.mode_actions):raise ValueError('모드가 내지 않는 후보 행동입니다.')
        meter=er.DelayMeter(env,snap['t0'],vehicles,injected,marks=(300,))
        def after_step():meter.step()
        def m4(e,allowed):
            s=planner.snapshot(e);a=int(s.mode_actions[4])
            if a not in allowed:raise ValueError('M4 행동이 실행 가능 목록 밖입니다.')
            return a
        _,changed=er.drive(env,cons,m4,int(job['candidate']),snap['t0'],300,after_step)
        env.close();env=None;meter.finish(trip)
        if 300 not in meter.at:raise ValueError('300초 비용이 없습니다.')
        from focused_eval import judge_collisions
        collision=judge_collisions(collision_file)
        data=meter.at[300];r={k:job[k] for k in ('fingerprint','state','phase','future_seed','candidate','snapshot_hash')}
        r.update(data,leaks=len(meter.leaks),changed_by_rules=changed,injected=len(injected),seconds=time.monotonic()-started,
                 collisions_at_204820=collision['collisions_at_204820'],
                 collisions_unexplained=len(collision['collisions_unexplained']),collision_details=collision)
        if r['leaks'] or changed or r['collisions_at_204820'] or r['collisions_unexplained']:
            raise ValueError('미래 누출, 규칙 변경 또는 미확인 충돌이 발생했습니다.')
        w.write(Path(job['target']),r)
        digest=Path(job['target']).with_suffix('.sha256');tmp=digest.with_suffix('.sha256.tmp')
        tmp.write_text(w.file_hash(Path(job['target']))+'\n');tmp.replace(digest)
        return {'ok':True,'seconds':time.monotonic()-started,'name':Path(job['target']).name}
    except BaseException as e:
        return {'ok':False,'seconds':time.monotonic()-started,'name':Path(job['target']).name,'error':f'{type(e).__name__}: {e}'}
    finally:
        er._env=original_env
        if env is not None:env.close()
        trip.unlink(missing_ok=True)
        collision_file.unlink(missing_ok=True)


def execute(jobs,kind,ledger,workers,deadline):
    read=collect_cached if kind=='collect' else rollout_cached
    fn=collect_worker if kind=='collect' else rollout_worker
    pending=[x for x in jobs if read(x) is None]
    print(f'{kind}: 총 {len(jobs)}개, 재사용 {len(jobs)-len(pending)}, 남은 {len(pending)}개',flush=True)
    if not pending:return
    cfg=config()['budget'];cap=cfg['collect_reserve_s'] if kind=='collect' else cfg['rollout_reserve_s']
    pool=ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),initializer=long._worker_init)
    active={};failed=False
    try:
        while pending or active:
            while pending and len(active)<workers and time.time()<deadline:
                reserved=sum(x['cap_s'] for x in ledger.pending.values())
                new_used=ledger.used-ledger.prior
                if ledger.remaining<cap+cfg['stop_margin_s'] or new_used+reserved+cap>cfg['new_cap_s']:break
                job=pending.pop(0);token=ledger.begin(kind+'/'+Path(job['target']).name,cap)
                try:future=pool.submit(fn,job)
                except BaseException:ledger.finish(token,0,'submit_failed');raise
                active[future]=(job,token,time.monotonic())
            if not active:
                if pending:raise long.StudyStopped('계산 또는 실행 시간 한도에 도달했습니다. 완료한 결과는 보존합니다.')
                break
            done,_=wait(active,timeout=5,return_when=FIRST_COMPLETED)
            if any(time.monotonic()-start>cap-10 for _,_,start in active.values()):
                raise TimeoutError(f'{kind} 작업 하나가 예약 시간 직전에 도달했습니다.')
            for future in done:
                job,token,start=active.pop(future);seconds=time.monotonic()-start
                try:r=future.result()
                except BaseException as e:
                    ledger.finish(token,seconds,'failed');raise RuntimeError('작업 프로세스 오류') from e
                ledger.finish(token,max(seconds,float(r['seconds'])),'complete' if r['ok'] else 'failed')
                if not r['ok']:raise RuntimeError(f"{kind} 실패 {r['name']}: {r.get('error')}")
                read(job)
                print(f"{kind} 완료 {r['name']} | 누적 {ledger.used/3600:.2f}/96시간",flush=True)
    except BaseException:
        failed=True;long._kill_pool(pool)
        for job,token,start in active.values():ledger.finish(token,time.monotonic()-start,'interrupted')
        raise
    finally:
        if not failed:pool.shutdown(wait=True)


def collect_jobs(out,work,fp):
    return [{'condition':c,'fingerprint':fp,'target':str(Path(out)/'states'/scenario_id(c)),
             'work':str(Path(work)/'collect'/scenario_id(c))} for c in config()['conditions']]


def rollout_job(out,fp,record,state,phase,candidate,ix):
    sid=state['id'];seed=future_seed(sid,phase,ix);folder=Path(out)/'states'/scenario_id(record['condition'])
    return {'fingerprint':fp,'state':sid,'phase':phase,'future_seed':seed,'candidate':int(candidate),
            'snapshot_hash':record['files'][sid+'.xml'],'state_dir':str(folder),'driver_seed':record['condition']['seed'],
            'mode_actions':state['mode_actions'],'m4_action':state['m4_action'],
            'target':str(Path(out)/'rollouts'/phase/sid/f'a{candidate}_s{seed}.json')}


def choose(m4,candidates,costs):
    if m4 not in candidates:raise ValueError('M4 행동이 후보에 없습니다.')
    if any(c not in costs or len(costs[c])!=4 for c in candidates):raise ValueError('선택용 미래 비용이 불완전합니다.')
    return min(candidates,key=lambda a:(statistics.fmean(costs[a]),a!=m4,a))


def make_summary(out,records,choices,validation):
    cfg=config();bydate={c['date']:[] for c in cfg['conditions']};byep={};details=[]
    for rec in records:
        ep=scenario_id(rec['condition']);byep[ep]=len(rec['states'])
        for s in rec['states']:
            ch=choices[s['id']];base=s['m4_action']
            if ch['chosen']==base:gains=[0.0]*4
            else:
                pair=validation[s['id']]
                gains=[pair[(base,i)]['cost']-pair[(ch['chosen'],i)]['cost'] for i in range(4)]
            mean=statistics.fmean(gains);bydate[rec['condition']['date']].append(mean)
            details.append({'state':s['id'],'date':rec['condition']['date'],'period':rec['condition']['period'],
                            'm4_action':base,'chosen_action':ch['chosen'],'validation_gains_vehicle_seconds':gains,'mean_gain':mean})
    means={d:statistics.fmean(v) for d,v in bydate.items() if v}
    coverage=sum(n>0 for n in byep.values());dates=len(means)
    avg=statistics.fmean(means.values()) if means else None
    ci=None
    if dates==4:
        se=statistics.stdev(means.values())/math.sqrt(dates);ci=[avg-3.182446*se,avg+3.182446*se]
    positives=sum(x>0 for x in means.values())
    gate=bool(dates==4 and coverage>=cfg['diagnostic_gate']['min_episodes_with_state'] and positives>=3 and ci[0]>0)
    result={'interpretation':cfg['interpretation'],'gate_rule':cfg['diagnostic_gate']['rule'],
            'verdict':'재설계 단서' if gate else '근거 부족', 'episodes':len(records),'episodes_with_state':coverage,
            'states':len(details),'date_means':means,'mean_date_gain':avg,'exploratory_t95':ci,
            'positive_dates':positives,'per_episode_state_count':byep,'details':details,
            'sampled_future_limit':'두 표본 세트는 분리되지만 동일한 미래 생성 모형을 사용한다.',
            'units':'각 선택 지점의 300초 대리 비용 차이(차량·초); 차량당 지연이나 시간당 이득이 아님'}
    w.write(Path(out)/'summary.json',result)
    lines=['# M4 선택 지점 진단 — 개발 파일럿','',f"- 판정: **{result['verdict']}**",
           f"- 상황 {len(records)}개 중 선택 지점이 있는 상황 {coverage}개; 상태 {len(details)}개; 날짜별 양수 {positives}/{dates}",
           f"- 날짜별 평균 대리 비용 차이(차량·초): {means}",f"- 날짜 평균 {avg}, 탐색적 95% t구간 {ci}",
           '', '양수는 M4보다 후보의 300초 대리 비용이 낮다는 뜻입니다. 실제 60분 차량당 지연이나 RL 성공 판정이 아닙니다.',
           '표본 미래와 75초 종단 비용에 의존하며 상태별 이득을 시간당 개선으로 합산할 수 없습니다.',
           '네 날짜는 독립 무작위 표본이 아닙니다. 근거 부족은 기회가 없다는 증명이 아닙니다.',
           '', '| 상태 | M4 첫 행동 | 선택된 행동 | 독립 미래의 평균 차이(차량·초) |','|---|---:|---:|---:|']
    for d in details:lines.append(f"| {d['state']} | {d['m4_action']} | {d['chosen_action']} | {d['mean_gain']:+.2f} |")
    (Path(out)/'summary.md').write_text('\n'.join(lines)+'\n')
    return result


def run(args):
    out,work,fixed,night,source=map(Path,(args.out,args.work,args.fixed,args.night,args.source))
    paths=[p.resolve() for p in (out,work,fixed,night,source)]
    for i,a in enumerate(paths):
        for b in paths[i+1:]:
            if a==b or a in b.parents or b in a.parents:raise ValueError('출력·원본·작업 폴더는 겹칠 수 없습니다.')
    out.mkdir(parents=True,exist_ok=True)
    with (out/'.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('이미 같은 폴더의 진단이 실행 중입니다.')
        rec=register(out,fixed,night,source);ledger=long.Ledger(out,rec)
        workers=max(1,min(args.workers,long.available_cpus(),config()['budget']['workers_max']))
        deadline=time.time()+config()['budget']['wall_hours']*3600
        print(f"이전 누적 {ledger.used/3600:.2f}/96 작업시간. 이번 파일럿 최대 5시간, 작업자 {workers}개.",flush=True)
        # Network template is generated once before workers to avoid concurrent writes.
        import focused_train as ft
        from focused_scenario import dqn_template_network
        dqn_template_network(ft.CACHE)
        cj=collect_jobs(out,work,rec['fingerprint'])
        execute(cj[:1],'collect',ledger,1,deadline)
        first=collect_cached(cj[0])
        if first['states']:print('첫 M4 상황의 저장·복원 검사 통과',flush=True)
        else:print('첫 M4 상황에는 선택 지점이 없어 다음 상황을 이어갑니다.',flush=True)
        execute(cj,'collect',ledger,workers,deadline)
        records=[collect_cached(j) for j in cj]
        selectable=[]
        for r in records:
            for s in r['states']:
                for a in sorted(set(s['mode_actions'])):
                    for i in range(4):selectable.append(rollout_job(out,rec['fingerprint'],r,s,'select',a,i))
        if not selectable:
            make_summary(out,records,{},{});print('선택 지점을 찾지 못해 근거 부족으로 종료',flush=True);return
        pilot=selectable[:config()['budget']['pilot_rollouts']]
        execute(pilot,'rollout',ledger,1,deadline)
        seconds=[rollout_cached(j)['seconds'] for j in pilot]
        states=sum(len(r['states']) for r in records)
        remaining_select=len(selectable)-len(pilot)
        # Assume every state selects a non-M4 alternative: 2 greens × 4 validation futures.
        max_validation=states*2*config()['rollout']['validation_futures']
        budget=config()['budget'];reserved=sum(v['cap_s'] for v in ledger.pending.values())
        available=min(ledger.remaining-budget['stop_margin_s'],budget['new_cap_s']-(ledger.used-ledger.prior)-reserved)
        estimate=(remaining_select+max_validation)*statistics.fmean(seconds)*budget['forecast_margin']
        forecast={'pilot_rollouts':len(pilot),'mean_seconds':statistics.fmean(seconds),'remaining_selection':remaining_select,
                  'maximum_validation':max_validation,'margin':budget['forecast_margin'],'estimated_s':estimate,
                  'available_s':available,'fits':estimate<=available,
                  'note':'초기 4회의 속도를 사용한 보수적 상한 예상. 속도 변화와 실패 때문에 완주를 보장하지 않는다.'}
        w.write(out/'forecast.json',forecast)
        print(f"표본 미래 검사 통과. 남은 작업 예상 {estimate/3600:.2f}h, 사용 가능 {available/3600:.2f}h",flush=True)
        if not forecast['fits']:
            raise long.StudyStopped('사전 비용 예상이 남은 파일럿 예산보다 커서 본 비교를 시작하지 않습니다.')
        execute(selectable,'rollout',ledger,workers,deadline)
        choices={};vj=[]
        for r in records:
            for s in r['states']:
                candidates=sorted(set(s['mode_actions']))
                costs={a:[rollout_cached(rollout_job(out,rec['fingerprint'],r,s,'select',a,i))['cost'] for i in range(4)] for a in candidates}
                a=choose(s['m4_action'],candidates,costs)
                choices[s['id']]={'m4':s['m4_action'],'chosen':a,'selection_mean_costs':{str(k):statistics.fmean(v) for k,v in costs.items()}}
                if a!=s['m4_action']:
                    for x in (s['m4_action'],a):
                        for i in range(4):vj.append(rollout_job(out,rec['fingerprint'],r,s,'validate',x,i))
        choice_path=out/'choices.json';choice_record={'fingerprint':rec['fingerprint'],'choices':choices}
        if choice_path.exists() and w.read(choice_path)!=choice_record:raise ValueError('고정된 선택 결과가 바뀌었습니다.')
        if not choice_path.exists():w.write(choice_path,choice_record)
        execute(vj,'rollout',ledger,workers,deadline)
        validation={}
        for r in records:
            for s in r['states']:
                chosen=choices[s['id']]['chosen'];base=s['m4_action']
                if chosen!=base:
                    validation[s['id']]={(a,i):rollout_cached(rollout_job(out,rec['fingerprint'],r,s,'validate',a,i)) for a in (base,chosen) for i in range(4)}
        result=make_summary(out,records,choices,validation)
        print(f"완료: {result['verdict']}, 누적 {ledger.used/3600:.2f}/96 작업시간, {out/'summary.md'}",flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('out','work','fixed','night','source'):p.add_argument('--'+n,required=True)
    p.add_argument('--workers',type=int,default=4);args=p.parse_args()
    if sys.platform=='darwin':raise SystemExit('실제 진단은 Colab에서만 합니다.')
    if args.workers<1:raise ValueError('작업자 수 오류')
    def stop(signum,frame):raise KeyboardInterrupt('중단 요청')
    signal.signal(signal.SIGTERM,stop)
    run(args)

if __name__=='__main__':main()
