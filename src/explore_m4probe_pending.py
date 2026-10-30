"""Restore the observed insertion queue at the saved instant, with strict replay checks.

SUMO may not expose a previously pending vehicle until the next simulation step. We save its
original pending ID/route/delay, require the loaded simulator to know the same vehicle, and
repair only the immediate read API at the saved instant. We never insert a synthetic past
vehicle. Before accepting a state, an independent 300-second M4 replay must match the original
at every 5-second decision, including the full observation and the vehicle/signal fingerprint.
"""
from __future__ import annotations
import copy
from pathlib import Path
import pickle
import shutil
import time

import explore_weight as w
import explore_m4probe as base
import explore_rollout as er


def capture_pending(sumo):
    ids=sorted(sumo.simulation.getPendingVehicles())
    return {v:{'route':list(sumo.vehicle.getRoute(v)), 'depart_delay':float(sumo.vehicle.getDepartDelay(v))} for v in ids}


def install_initial_pending(env,snap):
    """Expose only the saved t0 insertion queue; SUMO itself retains vehicle movement.

    Verify underlying vehicle route and departure delay before any correction; never hide an
    unexpected pending vehicle. On the first simulation step all lookups return to native SUMO.
    """
    if 'pending_snapshot' not in snap:raise ValueError('저장 상태에 원래 진입 대기 목록이 없습니다.')
    saved=snap['pending_snapshot'];sim=env.sumo.simulation;veh=env.sumo.vehicle;t0=float(snap['t0'])
    native_pending=sim.getPendingVehicles;native_route=veh.getRoute;native_delay=veh.getDepartDelay
    current=set(native_pending());unexpected=current-set(saved)
    if unexpected:raise ValueError(f'복원 상태에 원래 없던 진입 대기 차량이 있습니다: {sorted(unexpected)[:10]}')
    missing=set(saved)-current
    for v in sorted(missing):
        try:
            route=list(native_route(v));delay=float(native_delay(v))
        except Exception as e:
            raise ValueError(f'복원 직후 진입 대기 차량의 실제 존재를 확인할 수 없습니다: {v}: {type(e).__name__}: {e}') from e
        if route!=saved[v]['route'] or abs(delay-float(saved[v]['depart_delay']))>1e-6:
            raise ValueError(f"복원 직후 대기 차량 불일치 {v}: 원래 경로={saved[v]['route']} 복원 경로={route}, "
                             f"원래 대기={saved[v]['depart_delay']} 복원 대기={delay}")
    def at_initial():return abs(float(sim.getTime())-t0)<1e-9
    def pending():
        if at_initial():return tuple(sorted(saved))
        return native_pending()
    def route(v):
        if at_initial() and v in missing:return tuple(saved[v]['route'])
        return native_route(v)
    def delay(v):
        if at_initial() and v in missing:return float(saved[v]['depart_delay'])
        return native_delay(v)
    sim.getPendingVehicles=pending;veh.getRoute=route;veh.getDepartDelay=delay
    return {'saved':len(saved),'missing_initial':sorted(missing),'matched_underlying':True}


_ORIGINAL_RESTORE=er.restore

def restore(stem,route_dir,future,seed=None,horizon=er.H_SENS,tripinfo=None):
    env,cons,ts,ctl,snap,injected,vehicles=_ORIGINAL_RESTORE(stem,route_dir,future,seed,horizon,tripinfo=tripinfo)
    try:
        snap['pending_fix_check']=install_initial_pending(env,snap)
        return env,cons,ts,ctl,snap,injected,vehicles
    except BaseException:
        env.close();raise


def same_snapshot(actual,expected):
    return (list(actual.allowed)==expected['allowed'] and list(actual.mode_actions)==expected['mode_actions']
            and list(actual.mode_mask)==expected['mode_mask'] and int(actual.p0_action)==expected['p0_action']
            and len(actual.observation)==len(expected['observation'])
            and all(abs(float(a)-float(b))<1e-6 for a,b in zip(actual.observation,expected['observation'])))


def saved_snapshot(snapshot):
    return {'allowed':list(snapshot.allowed),'mode_actions':list(snapshot.mode_actions),
            'mode_mask':list(snapshot.mode_mask),'p0_action':int(snapshot.p0_action),
            'observation':snapshot.observation.tolist()}


def collect_worker(job):
    from explore_weight_core import ModePlanner
    from explore_weight_episode import _Episode
    from focused_eval import judge_collisions
    c=job['condition'];er.SUMO_SEED=int(c['seed']);started=time.monotonic()
    local=Path(job['work']);target=Path(job['target']);shutil.rmtree(local,ignore_errors=True);local.mkdir(parents=True)
    states=[];ep=None
    try:
        ep=_Episode(c['date'],c['period'],int(c['seed']),local/'episode',deadline=None)
        ep.date=c['date'];ep.period=c['period']
        planner=ModePlanner(ep.movement_of,[p.state for p in ep.ts.green_phases],ep.handover+3600)
        stem_dir=local/'states';stem_dir.mkdir()
        while True:
            now=float(ep.env.sim_step);elapsed=now-ep.handover
            required_end=max([base.config()['windows_s'][-1][1]]+[s['elapsed_s']+300 for s in states])
            if elapsed>=required_end:break
            snap=planner.snapshot(ep.env)
            wanted=next((i for i,(a,b) in enumerate(base.config()['windows_s']) if a<=elapsed<b and not any(z['window']==i for z in states)),None)
            if wanted is not None and snap.mode_disagrees:
                name=base.state_id(c,wanted);stem=stem_dir/name
                saved=er.Episode.snapshot(ep,stem,snap.allowed,snap.p0_action)
                saved['pending_snapshot']=capture_pending(ep.env.sumo)
                stem.with_suffix('.pkl').write_bytes(pickle.dumps(saved))
                states.append({'id':name,'window':wanted,'elapsed_s':elapsed,'t0':snap.now,
                    **saved_snapshot(snap),'m4_action':int(snap.mode_actions[4]),'route':ep.route.name,
                    'pending_snapshot_count':len(saved['pending_snapshot']),
                    'trace_before':er.trace_fingerprint(ep.env.sumo),'timeline':[]})
            current=[s for s in states if s['t0']<=now<s['t0']+300]
            action=int(snap.mode_actions[4]);corrected,executed=ep.apply(action)
            if corrected!=action or executed!=action:raise RuntimeError('원본 M4 실행 규칙이 선택을 바꿨습니다.')
            if current:
                fingerprint=er.trace_fingerprint(ep.env.sumo)
                for s in current:s['timeline'].append({'snapshot':saved_snapshot(snap),'action':action,'after':fingerprint})
            if ep.done:raise RuntimeError('복원 검증 기간 전에 에피소드가 끝났습니다.')
        route=ep.route;ep.close();ep=None
        collision=judge_collisions(local/'episode'/'collisions.xml')
        if collision['collisions_at_204820'] or collision['collisions_unexplained']:raise RuntimeError('M4 수집 중 미확인 충돌이 발생했습니다.')
        staging=local/'output';staging.mkdir();shutil.copy2(route,staging/route.name)
        for s in states:
            for ext in ('.xml','.pkl'):shutil.copy2(stem_dir/(s['id']+ext),staging/(s['id']+ext))
        for s in states:
            env=None
            try:
                env,cons,ts,ctl,saved,_,_=restore(staging/s['id'],staging,'real',horizon=300)
                if er.trace_fingerprint(env.sumo)!=s['trace_before']:
                    raise RuntimeError(f"M4 복원 직후 차량·신호·대기 상태가 다릅니다: {s['id']}")
                check=ModePlanner(ctl.movement_of,[p.state for p in ts.green_phases],saved['handover']+3600)
                if len(s['timeline'])!=60:raise RuntimeError(f"원본 M4 300초 기준 기록이 부족합니다: {s['id']}")
                for step,ref in enumerate(s['timeline']):
                    actual=check.snapshot(env)
                    if not same_snapshot(actual,ref['snapshot']):
                        raise RuntimeError(f"M4 복원 후 관측/모드가 다릅니다: {s['id']} step={step}")
                    action=int(actual.mode_actions[4]);corrected=int(cons(env,action))
                    if action!=ref['action'] or corrected!=action:
                        raise RuntimeError(f"M4 복원 후 행동이 다릅니다: {s['id']} step={step}")
                    env.step(corrected)
                    if er.trace_fingerprint(env.sumo)!=ref['after']:
                        raise RuntimeError(f"M4 복원 후 차량·신호·대기 상태가 다릅니다: {s['id']} step={step+1}")
                s['restore_ok']=True;s['restore_horizon_s']=300;s['pending_fix_check']=saved['pending_fix_check']
            finally:
                if env is not None:env.close()
            # Timeline remains in a separately hashed file for audit, not in the main index.
            w.write(staging/(s['id']+'.timeline.json'),s.pop('timeline'))
        files=[route.name]+[s['id']+ext for s in states for ext in ('.xml','.pkl','.timeline.json')]
        rec={'fingerprint':job['fingerprint'],'condition':c,'states':states,'files':base.digest_files(staging,files),
             'elapsed_s':time.monotonic()-started,'note':'실제 미래는 300초 M4 복원 검증에만 사용'}
        w.write(staging/'episode.json',rec)
        target.parent.mkdir(parents=True,exist_ok=True);temporary=target.parent/('.'+target.name+'.incomplete')
        if temporary.exists():shutil.rmtree(temporary)
        if target.exists():raise ValueError('기존 상태를 덮어쓸 수 없습니다.')
        shutil.copytree(staging,temporary)
        if base.digest_files(temporary,files)!=rec['files']:raise ValueError('저장 상태 복사 해시 오류')
        temporary.rename(target)
        return {'ok':True,'seconds':time.monotonic()-started,'name':base.scenario_id(c)}
    except BaseException as e:
        # Save failure without discarding or relaxing any state.
        report=Path(job['target']).parent.parent/'restore_diagnostics'/f"{base.scenario_id(c)}.json"
        w.write(report,{'scenario':base.scenario_id(c),'ok':False,'error':f'{type(e).__name__}: {e}',
                        'collected_states':[{'id':s['id'],'pending_snapshot_count':s['pending_snapshot_count']} for s in states],
                        'note':'복원·차량·신호·관측이 다르면 해당 상태를 채택하지 않습니다.'})
        return {'ok':False,'seconds':time.monotonic()-started,'name':base.scenario_id(c),
                'error':f'{type(e).__name__}: {e}; 진단: {report}'}
    finally:
        if ep is not None:ep.close()
        shutil.rmtree(local,ignore_errors=True)


def rollout_worker(job):
    from explore_weight_core import ModePlanner
    previous=er.restore;previous_snapshot=ModePlanner.snapshot;first=[True]
    def checked(planner,env):
        result=previous_snapshot(planner,env)
        if first[0]:
            first[0]=False
            if not same_snapshot(result,job['expected_snapshot']):
                raise ValueError('표본 미래 주입 뒤 현재 관측·마스크·모드가 원래 상태와 다릅니다.')
        return result
    er.restore=restore;ModePlanner.snapshot=checked
    try:return base.rollout_worker(job)
    finally:er.restore=previous;ModePlanner.snapshot=previous_snapshot
