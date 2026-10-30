"""M4 diagnostic tests use fakes only; no SUMO or real training."""
import gzip
import json
import sys
import tempfile
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import explore_m4probe as p
import explore_weight as w
import explore_weight_long as long
import explore_rollout as er
import explore_weight_core as core
import explore_weight_episode as episode
import focused_eval


class M4ProbeTest(unittest.TestCase):
    def test_conditions_are_balanced_and_future_streams_are_distinct(self):
        p.validate_config(set(w.training_scenarios()))
        cfg=p.config();self.assertEqual(len(cfg['conditions']),16)
        for c in cfg['conditions']:
            for win in range(2):
                sid=p.state_id(c,win)
                sel={p.future_seed(sid,'select',i) for i in range(4)}
                val={p.future_seed(sid,'validate',i) for i in range(4)}
                self.assertFalse(sel&val)
                self.assertEqual(len(sel|val),8)

    def test_select_uses_only_four_selection_costs_and_ties_keep_m4(self):
        self.assertEqual(p.choose(2,[0,2],{0:[1,1,1,1],2:[1,1,1,1]}),2)
        self.assertEqual(p.choose(2,[0,2],{0:[1,1,1,1],2:[2,2,2,2]}),0)
        with self.assertRaises(ValueError):p.choose(2,[0,2],{0:[1,1,1,1],2:[2,2]})

    def test_positive_and_inconclusive_summary_are_not_full_policy_success(self):
        records=[];choices={};validation={}
        for i,c in enumerate(p.config()['conditions'][:]):
            sid=p.state_id(c,0);states=[] if i%2 else [{'id':sid,'m4_action':2}]
            records.append({'condition':c,'states':states})
            if states:
                choices[sid]={'chosen':1}
                validation[sid]={(a,k):{'cost':10 if a==2 else 5} for a in (2,1) for k in range(4)}
        with tempfile.TemporaryDirectory() as d:
            result=p.make_summary(Path(d),records,choices,validation)
            self.assertEqual(result['verdict'],'재설계 단서')
            self.assertEqual(result['states'],8)
            self.assertEqual(result['exploratory_t95'],[5.0,5.0])
            self.assertIn('차량당 지연이나 시간당 이득이 아님',result['units'])
            for sid in choices: choices[sid]['chosen']=2
            r=p.make_summary(Path(d),records,choices,{})
            self.assertEqual(r['verdict'],'근거 부족')

    def test_rollout_changes_only_first_action_then_returns_to_m4(self):
        class FakeEnv:
            def __init__(self):self.sim_step=0;self._sumo_step=lambda:None;self.actions=[]
            def step(self,a):self.actions.append(a);self.sim_step+=5;self._sumo_step()
            def close(self):pass
        class FakeCons:
            def __call__(self,env,a):return a
            def allowed(self,env):return [0,1,2,3]
        class FakePlanner:
            def __init__(self,*args):pass
            def snapshot(self,e):return SimpleNamespace(mode_actions=[0,1,2,2,3])
        class FakeMeter:
            def __init__(self,*args,**kw):self.leaks=set();self.at={}
            def step(self):pass
            def finish(self,trip):self.at[300]={'cost':100.,'delay':40.,'terminal':60.}
        with tempfile.TemporaryDirectory() as d:
            state=Path(d)/'state';state.write_text('') if False else None
            xml=state.with_suffix('.xml');xml.write_text('saved state')
            target=Path(d)/'result.json';env=FakeEnv();cons=FakeCons()
            job={'fingerprint':'fp','state':'state','phase':'select','future_seed':123,'candidate':1,
                 'snapshot_hash':w.file_hash(xml),'state_dir':d,'driver_seed':42,'mode_actions':[0,1,2,2,3],
                 'm4_action':3,'target':str(target)}
            ts=SimpleNamespace(green_phases=[SimpleNamespace(state='x')]*4)
            ctl=SimpleNamespace(movement_of={})
            ret=(env,cons,ts,ctl,{'handover':0,'t0':0},[],{})
            good={'collisions_at_204820':0,'collisions_unexplained':[]}
            with patch.object(er,'restore',return_value=ret),patch.object(er,'DelayMeter',FakeMeter), \
                 patch.object(core,'ModePlanner',FakePlanner),patch.object(focused_eval,'judge_collisions',return_value=good):
                result=p.rollout_worker(job)
            self.assertTrue(result['ok'],result)
            self.assertEqual(env.actions[0],1)
            self.assertEqual(set(env.actions[1:]),{3})
            self.assertEqual(len(env.actions),60)
            self.assertEqual(p.rollout_cached(job)['cost'],100.)
            data=w.read(target);data['cost']=101;w.write(target,data)
            with self.assertRaisesRegex(ValueError,'해시 오류'):p.rollout_cached(job)

    def test_missing_or_bad_fingerprint_state_is_not_reused(self):
        with tempfile.TemporaryDirectory() as d:
            target=Path(d)/'episode';target.mkdir()
            with self.assertRaisesRegex(ValueError,'완료 기록 없는'):p.collect_cached({'target':str(target)})

    def test_collector_saves_first_choice_in_each_window_and_verifies_restore(self):
        saves={}
        class FakeEnv:
            def __init__(self,now):
                self.sim_step=now;self.sumo=SimpleNamespace(env=self)
            def step(self,action):self.sim_step+=5
            def close(self):pass
        class FakeCons:
            def __call__(self,env,action):return action
        class FakeEpisode:
            def __init__(self,date,period,seed,work,deadline=None):
                work.mkdir(parents=True);self.route=work/f'{date}_{period}.rou.xml';self.route.write_text('routes')
                self.collisions=work/'collisions.xml';self.collisions.write_text('<collisions/>')
                self.env=FakeEnv(1800);self.ts=SimpleNamespace(green_phases=[SimpleNamespace(state='x')]*4)
                self.cons=FakeCons();self.movement_of={};self.handover=1800;self.vehicles={};self.done=False
            def apply(self,a):self.env.step(a);return a,a
            def close(self):pass
        class FakePlanner:
            def __init__(self,*args):pass
            def snapshot(self,env):return SimpleNamespace(now=env.sim_step,mode_disagrees=True,
                mode_actions=[0,1,2,2,3],mode_mask=[True,True,True,False,True],allowed=[0,1,2,3],
                p0_action=0,observation=np.zeros(110))
        def save(ep,stem,allowed,action):
            stem.with_suffix('.xml').write_text('state');stem.with_suffix('.pkl').write_text('saved')
            saves[stem.name]=ep.env.sim_step
            return {'t0':ep.env.sim_step,'route':ep.route.name}
        def restore(stem,route_dir,future,horizon):
            env=FakeEnv(saves[stem.name]);ts=SimpleNamespace(green_phases=[SimpleNamespace(state='x')]*4)
            return env,FakeCons(),ts,SimpleNamespace(movement_of={}),{'handover':1800},[],{}
        with tempfile.TemporaryDirectory() as d:
            c=p.config()['conditions'][0]
            job={'condition':c,'fingerprint':'fp','target':str(Path(d)/'out'),'work':str(Path(d)/'work')}
            good={'collisions_at_204820':0,'collisions_unexplained':[]}
            with patch.object(episode,'_Episode',FakeEpisode),patch.object(core,'ModePlanner',FakePlanner), \
                 patch.object(er.Episode,'snapshot',side_effect=save),patch.object(er,'restore',side_effect=restore), \
                 patch.object(er,'trace_fingerprint',side_effect=lambda sumo:str(sumo.env.sim_step)), \
                 patch.object(focused_eval,'judge_collisions',return_value=good):
                result=p.collect_worker(job)
            self.assertTrue(result['ok'],result)
            saved=p.collect_cached(job)
            self.assertEqual([s['elapsed_s'] for s in saved['states']],[600,1800])
            self.assertTrue(all(s['restore_ok'] for s in saved['states']))

    def test_mac_execution_is_blocked(self):
        argv=['x','--out','a','--work','b','--fixed','c','--night','d','--source','e']
        with patch.object(sys,'argv',argv),patch.object(sys,'platform','darwin'):
            with self.assertRaisesRegex(SystemExit,'Colab'):p.main()

    def test_parallel_budget_charges_each_run_and_reuses_complete_states(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/'out';out.mkdir()
            ledger=long.Ledger(out,{'compute_hours':96,'prior_compute_s':100.0})
            jobs=p.collect_jobs(out,Path(d)/'work','fp')[:3]
            def fake(job):
                w.write(Path(job['target'])/'episode.json',{'fingerprint':'fp','condition':job['condition'],'states':[],'files':{}})
                return {'ok':True,'seconds':1.0,'name':p.scenario_id(job['condition'])}
            def thread_pool(max_workers,**kwargs):return ThreadPoolExecutor(max_workers=max_workers)
            with patch.object(p,'ProcessPoolExecutor',side_effect=thread_pool),patch.object(p,'collect_worker',side_effect=fake):
                p.execute(jobs,'collect',ledger,2,time.time()+60)
                charged=ledger.used
                self.assertEqual(charged,103.0)
                p.execute(jobs,'collect',ledger,2,time.time()+60)
                self.assertEqual(ledger.used,charged)

    def test_new_compute_cap_stops_incomplete_without_success(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/'out';out.mkdir();ledger=long.Ledger(out,{'compute_hours':96,'prior_compute_s':100.0})
            jobs=p.collect_jobs(out,Path(d)/'work','fp')[:2]
            cfg=p.config();cfg['budget']['new_cap_s']=600
            def fake(job):
                w.write(Path(job['target'])/'episode.json',{'fingerprint':'fp','condition':job['condition'],'states':[],'files':{}})
                return {'ok':True,'seconds':1.0,'name':p.scenario_id(job['condition'])}
            def thread_pool(max_workers,**kwargs):return ThreadPoolExecutor(max_workers=max_workers)
            with patch.object(p,'config',return_value=cfg),patch.object(p,'ProcessPoolExecutor',side_effect=thread_pool),patch.object(p,'collect_worker',side_effect=fake):
                with self.assertRaisesRegex(long.StudyStopped,'한도'):p.execute(jobs,'collect',ledger,1,time.time()+60)
            self.assertEqual(ledger.used,101.0)
            self.assertFalse((out/'summary.json').exists())

if __name__=='__main__':unittest.main()
