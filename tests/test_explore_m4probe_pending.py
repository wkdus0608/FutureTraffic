"""Strict pending-vehicle restoration and 300-second replay checks without SUMO."""
import pickle,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import explore_m4probe_pending as f
import explore_m4probe as base
import explore_rollout as er
import explore_weight_core as core
import explore_weight_episode as episode
import focused_eval

class Sim:
 def __init__(self,env):self.env=env
 def getTime(self):return self.env.sim_step
 def getPendingVehicles(self):
  return () if self.env.restored and self.env.sim_step==self.env.start else ('v',)
class Vehicles:
 def getRoute(self,v):
  if v!='v':raise ValueError('unknown')
  return ('edge1','edge2')
 def getDepartDelay(self,v):return 7.
class Env:
 def __init__(self,now,restored=False,drift=False):
  self.sim_step=now;self.start=now;self.restored=restored;self.drift=drift;self.steps=0
  self.sumo=SimpleNamespace(simulation=Sim(self),vehicle=Vehicles(),env=self)
 def step(self,a):self.sim_step+=5;self.steps+=1
 def close(self):pass
class Cons:
 def __call__(self,env,a):return a
class Episode:
 def __init__(self,date,period,seed,work,deadline=None):
  work.mkdir(parents=True);self.route=work/f'{date}_{period}.rou.xml';self.route.write_text('routes')
  (work/'collisions.xml').write_text('<collisions/>');self.env=Env(1800)
  self.ts=SimpleNamespace(green_phases=[SimpleNamespace(state='x')]*4);self.cons=Cons()
  self.movement_of={};self.handover=1800;self.vehicles={};self.done=False
 def apply(self,a):self.env.step(a);return a,a
 def close(self):pass
class Planner:
 def __init__(self,*args):pass
 def snapshot(self,env):return SimpleNamespace(now=env.sim_step,mode_disagrees=True,
  mode_actions=[0,1,2,2,3],mode_mask=[True,True,True,False,True],allowed=[0,1,2,3],
  p0_action=0,observation=np.zeros(110))

def save(ep,stem,allowed,action):
 stem.with_suffix('.xml').write_text('state')
 saved={'t0':ep.env.sim_step,'route':ep.route.name,'handover':ep.handover}
 stem.with_suffix('.pkl').write_bytes(pickle.dumps(saved))
 return saved

def fingerprint(sumo):
 env=sumo.env
 pending=','.join(sumo.simulation.getPendingVehicles())
 extra='drift' if env.restored and env.drift and env.steps>=10 else ''
 return f'{env.sim_step}:{pending}:{extra}'

def check_collector(tmp,drift=False):
 condition=base.config()['conditions'][1]
 job={'condition':condition,'fingerprint':'fp','target':str(Path(tmp)/'states'/'scenario'),'work':str(Path(tmp)/'work')}
 def restore(stem,route_dir,future,seed=None,horizon=300,tripinfo=None):
  saved=pickle.loads(stem.with_suffix('.pkl').read_bytes());env=Env(saved['t0'],True,drift)
  ts=SimpleNamespace(green_phases=[SimpleNamespace(state='x')]*4)
  return env,Cons(),ts,SimpleNamespace(movement_of={}),saved,[],{}
 good={'collisions_at_204820':0,'collisions_unexplained':[]}
 with patch.object(episode,'_Episode',Episode),patch.object(core,'ModePlanner',Planner), \
      patch.object(er.Episode,'snapshot',side_effect=save),patch.object(f,'_ORIGINAL_RESTORE',side_effect=restore), \
      patch.object(er,'trace_fingerprint',side_effect=fingerprint),patch.object(focused_eval,'judge_collisions',return_value=good):
  result=f.collect_worker(job)
 return result,job

class PendingRepairTest(unittest.TestCase):
 def test_virtual_pending_is_exact_only_at_saved_instant(self):
  env=Env(10,True);snap={'t0':10,'pending_snapshot':{'v':{'route':['edge1','edge2'],'depart_delay':7.}}}
  detail=f.install_initial_pending(env,snap)
  self.assertEqual(detail['missing_initial'],['v'])
  self.assertEqual(env.sumo.simulation.getPendingVehicles(),('v',))
  self.assertEqual(env.sumo.vehicle.getRoute('v'),('edge1','edge2'))
  env.step(0)
  self.assertEqual(env.sumo.simulation.getPendingVehicles(),('v',))
 def test_unknown_or_mismatched_underlying_vehicle_must_fail(self):
  env=Env(10,True);bad={'t0':10,'pending_snapshot':{'v':{'route':['wrong'],'depart_delay':7.}}}
  with self.assertRaisesRegex(ValueError,'대기 차량 불일치'):f.install_initial_pending(env,bad)
  env=Env(10,True);unknown={'t0':10,'pending_snapshot':{'missing':{'route':[],'depart_delay':0.}}}
  with self.assertRaisesRegex(ValueError,'존재를 확인'):f.install_initial_pending(env,unknown)
 def test_collector_requires_and_records_300_second_identical_replay(self):
  with tempfile.TemporaryDirectory() as tmp:
   result,job=check_collector(tmp)
   self.assertTrue(result['ok'],result)
   rec=base.collect_cached(job)
   self.assertEqual(len(rec['states']),2)
   for s in rec['states']:
    self.assertEqual(s['restore_horizon_s'],300)
    self.assertEqual(s['pending_fix_check']['missing_initial'],['v'])
    self.assertEqual(len(base.w.read(Path(job['target'])/(s['id']+'.timeline.json'))),60)
 def test_later_divergence_is_rejected(self):
  with tempfile.TemporaryDirectory() as tmp:
   result,job=check_collector(tmp,True)
   self.assertFalse(result['ok'])
   self.assertIn('차량·신호·대기 상태가 다릅니다',result['error'])
   self.assertFalse((Path(job['target'])/'episode.json').exists())
   self.assertTrue((Path(tmp)/'restore_diagnostics'/'20220812_0730_s43.json').exists())
 def test_sampled_future_must_preserve_original_observation(self):
  expected={'allowed':[0,1,2,3],'mode_actions':[0,1,2,2,3],
            'mode_mask':[True,True,True,False,True],'p0_action':0,'observation':[0.]*110}
  dummy=SimpleNamespace(allowed=expected['allowed'],mode_actions=expected['mode_actions'],
   mode_mask=expected['mode_mask'],p0_action=0,observation=np.zeros(110))
  def fake(job):
   try:core.ModePlanner.snapshot(None,None);return {'ok':True}
   except Exception as e:return {'ok':False,'error':str(e)}
  with patch.object(core.ModePlanner,'snapshot',return_value=dummy),patch.object(base,'rollout_worker',side_effect=fake):
   self.assertTrue(f.rollout_worker({'expected_snapshot':expected})['ok'])
   changed={**expected,'observation':[1.]+[0.]*109}
   self.assertFalse(f.rollout_worker({'expected_snapshot':changed})['ok'])

if __name__=='__main__':unittest.main()
