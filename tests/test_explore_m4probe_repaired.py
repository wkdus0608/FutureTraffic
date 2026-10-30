from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch
import sys,tempfile,unittest
from types import SimpleNamespace
from contextlib import redirect_stdout
from io import StringIO
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import explore_m4probe_repaired as r
import explore_m4probe as base
import explore_weight as w

class RepairedRunTest(unittest.TestCase):
 def test_no_budget_limit_reuses_completed_states(self):
  with tempfile.TemporaryDirectory() as tmp:
   out=Path(tmp);journal=r.Journal(out,'fp');jobs=base.collect_jobs(out,out/'work','fp')[:3]
   def fake(job):
    w.write(Path(job['target'])/'episode.json',{'fingerprint':'fp','condition':job['condition'],'states':[],'files':{}})
    return {'ok':True,'seconds':20000.,'name':base.scenario_id(job['condition'])}
   def pool(max_workers,**kwargs):return ThreadPoolExecutor(max_workers=max_workers)
   with patch.object(r,'ProcessPoolExecutor',side_effect=pool),patch.object(r,'collect_worker',side_effect=fake):
    r.execute(jobs,'collect',journal,2);self.assertEqual(len(journal.entries),3)
    r.execute(jobs,'collect',journal,2);self.assertEqual(len(journal.entries),3)
   self.assertEqual(sum(x['seconds'] for x in journal.entries),60000.)
 def test_protocol_pins_repair_and_scientific_parent(self):
  with tempfile.TemporaryDirectory() as tmp:
   rec=r.register_continuation(Path(tmp),{'fingerprint':'old'})
   self.assertEqual(rec['original_experiment_fingerprint'],'old')
   self.assertIn('repair_sha256',rec)
   with self.assertRaises(ValueError):r.register_continuation(Path(tmp),{'fingerprint':'other'})
 def test_rollout_job_carries_original_current_observation(self):
  record={'condition':base.config()['conditions'][0],'files':{'id.xml':'sha'}}
  state={'id':'id','m4_action':3,'mode_actions':[0,1,2,2,3],
         'allowed':[0,1,2,3],'mode_mask':[True,True,True,False,True],
         'p0_action':0,'observation':[0.]*110}
  job=r.rollout_job('/tmp/out','fp',record,state,'select',1,0)
  self.assertEqual(job['expected_snapshot']['observation'],[0.]*110)
  self.assertEqual(job['snapshot_hash'],'sha')

 def test_complete_fake_diagnostic_uses_selection_then_independent_validation(self):
  import focused_scenario
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);out=root/'out'
   args=SimpleNamespace(out=str(out),work=str(root/'work'),fixed=str(root/'fixed'),
                        night=str(root/'night'),source=str(root/'source'),workers=2)
   def fake_collect(job):
    c=job['condition'];sid=base.state_id(c,0);folder=Path(job['target']);folder.mkdir(parents=True)
    (folder/(sid+'.xml')).write_text('state')
    st={'id':sid,'m4_action':0,'mode_actions':[0,1,0,0,0],'allowed':[0,1],
        'mode_mask':[True,True,False,False,False],'p0_action':0,'observation':[0.]*110,'restore_ok':True}
    rec={'fingerprint':'fp','condition':c,'states':[st],'files':{sid+'.xml':w.file_hash(folder/(sid+'.xml'))}}
    w.write(folder/'episode.json',rec)
    return {'ok':True,'seconds':.01,'name':base.scenario_id(c)}
   def fake_rollout(job):
    result={k:job[k] for k in ('fingerprint','state','phase','future_seed','candidate','snapshot_hash')}
    result.update({'cost':10. if job['candidate']==0 else 5.,'leaks':0,'changed_by_rules':0,
                   'collisions_unexplained':0,'collisions_at_204820':0,'seconds':.01})
    path=Path(job['target']);w.write(path,result);path.with_suffix('.sha256').write_text(w.file_hash(path)+'\n')
    return {'ok':True,'seconds':.01,'name':path.name}
   def pool(max_workers,**kwargs):return ThreadPoolExecutor(max_workers=max_workers)
   with patch.object(base,'register',return_value={'fingerprint':'fp'}),patch.object(r.long,'available_cpus',return_value=2), \
        patch.object(r,'ProcessPoolExecutor',side_effect=pool),patch.object(r,'collect_worker',side_effect=fake_collect), \
        patch.object(r.repair,'rollout_worker',side_effect=fake_rollout), \
        patch.object(focused_scenario,'dqn_template_network',return_value=None):
    with redirect_stdout(StringIO()):r.run(args)
   result=w.read(out/'summary.json')
   self.assertEqual(result['states'],16)
   self.assertEqual(result['verdict'],'재설계 단서')
   self.assertTrue(all(item['chosen_action']==1 for item in result['details']))
   phases={item['phase'] for item in (w.read(p) for p in (out/'rollouts').rglob('*.json'))}
   self.assertEqual(phases,{'select','validate'})

if __name__=='__main__':unittest.main()
