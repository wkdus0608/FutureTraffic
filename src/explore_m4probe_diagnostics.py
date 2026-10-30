"""Explain an M4 saved-state restoration mismatch without weakening the check.

Only observes the original collector. It preserves the original strict pass/fail rules and
writes a concise per-episode report before the collector's temporary files are removed.
"""
from __future__ import annotations
import math
from pathlib import Path
import explore_weight as w


def snapshot_summary(s):
    return {'time':float(s.now),'mode_actions':list(map(int,s.mode_actions)),
            'mode_mask':list(map(bool,s.mode_mask)),'allowed':list(map(int,s.allowed)),
            'p0_action':int(s.p0_action),'observation':[float(x) for x in s.observation]}


def physical(sumo):
    from focused_data import TARGET_TLS
    ids=sorted(sumo.vehicle.getIDList())
    return {'time':float(sumo.simulation.getTime()),
            'signal':sumo.trafficlight.getRedYellowGreenState(TARGET_TLS),
            'pending':sorted(sumo.simulation.getPendingVehicles()),
            'vehicles':{v:[sumo.vehicle.getLaneID(v),float(sumo.vehicle.getLanePosition(v)),float(sumo.vehicle.getSpeed(v))] for v in ids}}


def compare_physical(expected,actual):
    if expected is None or actual is None:return {'available':False}
    a,b=expected['vehicles'],actual['vehicles'];common=set(a)&set(b)
    changed=[]
    for v in sorted(common):
        left,right=a[v],b[v]
        if left!=right:
            changed.append({'vehicle':v,'original':left,'restored':right,
                            'position_difference':right[1]-left[1],'speed_difference':right[2]-left[2]})
    return {'available':True,'time_original':expected['time'],'time_restored':actual['time'],
            'signal_equal':expected['signal']==actual['signal'],
            'pending_equal':expected['pending']==actual['pending'],
            'pending_only_original':sorted(set(expected['pending'])-set(actual['pending']))[:20],
            'pending_only_restored':sorted(set(actual['pending'])-set(expected['pending']))[:20],
            'vehicles_only_original':sorted(set(a)-set(b))[:20],
            'vehicles_only_restored':sorted(set(b)-set(a))[:20],
            'changed_count':len(changed),'changed_examples':changed[:10],
            'max_position_difference':max((abs(x['position_difference']) for x in changed),default=0.),
            'max_speed_difference':max((abs(x['speed_difference']) for x in changed),default=0.)}


def compare_snapshot(expected,actual):
    if expected is None or actual is None:return {'available':False}
    a,b=expected['observation'],actual['observation']
    diffs=[{'index':i,'original':x,'restored':y,'difference':y-x}
           for i,(x,y) in enumerate(zip(a,b)) if not math.isclose(x,y,rel_tol=0.,abs_tol=1e-6)]
    return {'available':True,'time_equal':expected['time']==actual['time'],
            'actions_equal':expected['mode_actions']==actual['mode_actions'],
            'actions_original':expected['mode_actions'],'actions_restored':actual['mode_actions'],
            'mask_equal':expected['mode_mask']==actual['mode_mask'],
            'allowed_equal':expected['allowed']==actual['allowed'],
            'p0_equal':expected['p0_action']==actual['p0_action'],
            'observation_length_equal':len(a)==len(b),'observation_difference_count':len(diffs),
            'observation_differences':diffs[:20],
            'observation_max_abs_difference':max((abs(x['difference']) for x in diffs),default=0.)}


def collect(job,collector):
    import explore_rollout as er
    from explore_weight_core import ModePlanner
    original_snapshot=ModePlanner.snapshot
    original_save=er.Episode.snapshot
    original_restore=er.restore
    original_trace=er.trace_fingerprint
    ctx={'last':None,'recording':None,'restoring':None,'states':{},'capture_errors':[]}

    def instrumented_snapshot(planner,env):
        result=original_snapshot(planner,env);ctx['last']=result
        if ctx['restoring'] is not None:
            try:ctx['states'][ctx['restoring']]['restored_snapshot']=snapshot_summary(result)
            except Exception as e:ctx['capture_errors'].append(f'restored observation: {type(e).__name__}: {e}')
        return result

    def instrumented_save(ep,stem,allowed,action):
        saved=original_save(ep,stem,allowed,action);name=Path(stem).name
        ctx['recording']=name;ctx['restoring']=None
        ctx['states'][name]={'original_frames':[],'restored_frames':[]}
        try:ctx['states'][name]['original_snapshot']=snapshot_summary(ctx['last'])
        except Exception as e:ctx['capture_errors'].append(f'original observation: {type(e).__name__}: {e}')
        return saved

    def instrumented_restore(stem,*args,**kwargs):
        result=original_restore(stem,*args,**kwargs);name=Path(stem).name
        ctx['restoring']=name;ctx['recording']=None
        env=result[0];original_step=env.step
        def step(action):
            ctx['states'][name]['restored_executed_action']=int(action)
            return original_step(action)
        env.step=step
        return result

    def instrumented_trace(sumo):
        digest=original_trace(sumo)
        name=ctx['restoring'] or ctx['recording']
        if name is not None:
            key='restored_frames' if ctx['restoring'] else 'original_frames'
            try:ctx['states'][name][key].append({'fingerprint':digest,'physical':physical(sumo)})
            except Exception as e:ctx['capture_errors'].append(f'traffic trace: {type(e).__name__}: {e}')
        return digest

    try:
        ModePlanner.snapshot=instrumented_snapshot;er.Episode.snapshot=instrumented_save
        er.restore=instrumented_restore;er.trace_fingerprint=instrumented_trace
        result=collector(job)
    finally:
        ModePlanner.snapshot=original_snapshot;er.Episode.snapshot=original_save
        er.restore=original_restore;er.trace_fingerprint=original_trace
    details=[]
    for name,state in ctx['states'].items():
        o,r=state.get('original_frames',[]),state.get('restored_frames',[])
        row={'state':name,'observation':compare_snapshot(state.get('original_snapshot'),state.get('restored_snapshot')),
             'restored_executed_action':state.get('restored_executed_action'),'frames':[]}
        for i,label in enumerate(('before','after_5s')):
            x=o[i] if len(o)>i else None;y=r[i] if len(r)>i else None
            row['frames'].append({'label':label,'fingerprints_equal':x['fingerprint']==y['fingerprint'] if x and y else None,
                                  'physical':compare_physical(x['physical'] if x else None,y['physical'] if y else None)})
        details.append(row)
    report={'scenario':Path(job['target']).name,'collector_ok':bool(result.get('ok')),
            'error':result.get('error'),'states':details,'diagnostic_capture_errors':ctx['capture_errors'],
            'note':'진단만 기록하며 기존 저장/복원 통과 기준은 완화하지 않는다.'}
    dest=Path(job['target']).parent.parent/'restore_diagnostics'/f"{Path(job['target']).name}.json"
    w.write(dest,report)
    if not result.get('ok'):
        print(f'복원 차이 진단 파일: {dest}',flush=True)
        for row in details:
            obs=row['observation'];frames=row['frames']
            print(f"  {row['state']} 신호후보 일치={obs.get('actions_equal')}, 관측 차이={obs.get('observation_difference_count')}, "
                  f"복원 전후 차량상태={[x['fingerprints_equal'] for x in frames]}",flush=True)
    return result
