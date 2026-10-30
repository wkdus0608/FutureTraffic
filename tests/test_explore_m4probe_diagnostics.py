"""Diagnostics never declare a mismatch acceptable or modify the collector's result."""
from pathlib import Path
import sys,tempfile,unittest
from contextlib import redirect_stdout
from io import StringIO
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import explore_m4probe_diagnostics as d
import explore_rollout as er
from explore_weight_core import ModePlanner
import explore_weight as w
import explore_weight_episode as episode
import explore_m4probe as probe
import focused_eval

class DiagnosticTest(unittest.TestCase):
 def test_distinguishes_subtle_trajectory_and_observation_differences(self):
  before={'time':1.,'signal':'rr','pending':[],'vehicles':{'a':['lane',10.,2.]}}
  after={'time':1.,'signal':'rr','pending':[],'vehicles':{'a':['lane',10.00000001,2.]}}
  result=d.compare_physical(before,after)
  self.assertEqual(result['changed_count'],1)
  self.assertGreater(result['max_position_difference'],0.)
  s={'time':1.,'mode_actions':[0,1,2,2,3],'mode_mask':[True,True,True,False,True],
     'allowed':[0,1,2,3],'p0_action':0,'observation':[0.,1.]}
  other={**s,'mode_actions':[0,1,2,2,2],'observation':[0.,1.1]}
  comparison=d.compare_snapshot(s,other)
  self.assertFalse(comparison['actions_equal'])
  self.assertEqual(comparison['observation_difference_count'],1)

 def test_failed_collection_writes_report_and_restores_hooks(self):
  original_snapshot=ModePlanner.snapshot;original_save=er.Episode.snapshot;original_restore=er.restore;original_trace=er.trace_fingerprint
  with tempfile.TemporaryDirectory() as tmp:
   job={'target':str(Path(tmp)/'states'/'scenario')}
   with redirect_stdout(StringIO()):
    result=d.collect(job,lambda _: {'ok':False,'error':'strict mismatch'})
   self.assertEqual(result['error'],'strict mismatch')
   report=w.read(Path(tmp)/'restore_diagnostics'/'scenario.json')
   self.assertFalse(report['collector_ok'])
   self.assertEqual(report['error'],'strict mismatch')
  self.assertIs(ModePlanner.snapshot,original_snapshot)
  self.assertIs(er.Episode.snapshot,original_save)
  self.assertIs(er.restore,original_restore)
  self.assertIs(er.trace_fingerprint,original_trace)

 def test_diagnostic_retains_exact_restore_failure_and_explains_actions(self):
  saves={}
  class Env:
   def __init__(self,now,restored=False):self.sim_step=now;self.restored=restored;self.sumo=SimpleNamespace(env=self)
   def step(self,a):self.sim_step+=5
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
  def snapshot(planner,env):
   actions=[0,1,2,3,3] if env.restored else [0,1,2,2,3]
   return SimpleNamespace(now=env.sim_step,mode_disagrees=True,mode_actions=actions,
    mode_mask=[True,True,True,False,True],allowed=[0,1,2,3],p0_action=0,observation=np.zeros(110))
  def save(ep,stem,allowed,action):
   stem.with_suffix('.xml').write_text('state');stem.with_suffix('.pkl').write_text('saved')
   saves[stem.name]=ep.env.sim_step
   return {'t0':ep.env.sim_step,'route':ep.route.name}
  def restore(stem,route_dir,future,horizon):
   env=Env(saves[stem.name],True);ts=SimpleNamespace(green_phases=[SimpleNamespace(state='x')]*4)
   return env,Cons(),ts,SimpleNamespace(movement_of={}),{'handover':1800},[],{}
  with tempfile.TemporaryDirectory() as tmp:
   c=probe.config()['conditions'][0]
   job={'condition':c,'fingerprint':'fp','target':str(Path(tmp)/'states'/'scenario'),'work':str(Path(tmp)/'work')}
   good={'collisions_at_204820':0,'collisions_unexplained':[]}
   with patch.object(episode,'_Episode',Episode),patch.object(ModePlanner,'__init__',return_value=None),patch.object(ModePlanner,'snapshot',snapshot), \
        patch.object(er.Episode,'snapshot',side_effect=save),patch.object(er,'restore',side_effect=restore), \
        patch.object(er,'trace_fingerprint',side_effect=lambda sumo:str(sumo.env.sim_step)), \
        patch.object(d,'physical',side_effect=lambda sumo:{'time':sumo.env.sim_step,'signal':'r','pending':[],'vehicles':{}}), \
        patch.object(focused_eval,'judge_collisions',return_value=good):
    with redirect_stdout(StringIO()):
     result=d.collect(job,probe.collect_worker)
   self.assertFalse(result['ok'])
   self.assertIn('복원이 원래 M4와 다릅니다',result['error'])
   report=w.read(Path(tmp)/'restore_diagnostics'/'scenario.json')
   self.assertFalse(report['states'][0]['observation']['actions_equal'])
   self.assertEqual([f['fingerprints_equal'] for f in report['states'][0]['frames']],[True,True])
   self.assertFalse((Path(job['target'])/'episode.json').exists())

if __name__=='__main__':unittest.main()
