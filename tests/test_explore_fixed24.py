"""FIXED24 registration, pairing, selection and recovery checks without SUMO."""
import gzip
import json
import sys
import tempfile
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import explore_fixed24 as f
import explore_weight as w
import explore_weight_long as long
import explore_weight_core as core


def save(job, mean=10, quality=True):
    path=Path(job['target']);path.mkdir(parents=True,exist_ok=True)
    metrics={'policy':job['tag'],'date':job['date'],'period':job['period'],'seed':job['seed'],
      'factors':job['factors'],'experiment_fingerprint':job['fingerprint'],'vehicles':100,'total_delay':mean*100,
      'actual_total_delay':mean*100,'mean_delay_s':mean,'long_waits':0,'missing':0,'unfinished':0,
      'changed_by_rules':0,'mode_action_errors':0,'collisions_at_204820':0,'collisions_unexplained':0,
      'cost_check':{'passed':quality},'route_sha256':'route','cohort_sha256':'cohort','by_movement':{
      '477706392_02->477706883_02':{'vehicles':20,'mean_delay_s':mean},
      '477706883_01->477707013_01':{'vehicles':20,'mean_delay_s':mean}}}
    w.write(path/'metrics.json',metrics)
    with gzip.open(path/'decisions.jsonl.gz','wt') as h:h.write('{}\n')
    w.write(path/'cache.json',{'fingerprint':job['fingerprint'],'model_identity':None,
      'metrics_sha256':w.file_hash(path/'metrics.json'),'decisions_sha256':w.file_hash(path/'decisions.jsonl.gz')})


class Fixed24Test(unittest.TestCase):
    def test_scenarios_and_jobs_are_balanced_and_paired(self):
        f.verify_config(set(w.training_scenarios()))
        jobs=f.jobs_for('/out','/work','finger','sweep',f.config()['policies'],f.conditions())
        self.assertEqual(len(jobs),120)
        self.assertEqual(len(f.confirmation_conditions()),64)
        for i in range(0,120,5):
            block=jobs[i:i+5]
            self.assertEqual(len({(j['date'],j['period'],j['seed']) for j in block}),1)
            self.assertEqual([j['tag'] for j in block],['P0','M4','W4','W6','W8'])

    def test_modes_restore_between_different_policy_jobs_even_after_error(self):
        original=tuple(tuple(x) for x in w.config()['modes'])
        def fake(_):
            self.assertEqual(core.MODE_FACTORS,original)
            core.MODE_FACTORS=original[:4]+((8,8),)
            return {'ok':True}
        with patch.object(f.night,'_eval_worker',side_effect=fake):
            f.eval_worker({});f.eval_worker({})
        self.assertEqual(core.MODE_FACTORS,original)
        def fail(_):
            core.MODE_FACTORS=original[:4]+((8,8),)
            raise ValueError('boom')
        with patch.object(f.night,'_eval_worker',side_effect=fail):
            with self.assertRaises(ValueError):f.eval_worker({})
        self.assertEqual(core.MODE_FACTORS,original)

    def test_frozen_selection_and_pair_integrity(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/'out';jobs=f.jobs_for(out,Path(d)/'work','fp','sweep',f.config()['policies'],f.conditions())
            means={'P0':10,'M4':9,'W4':8,'W6':8,'W8':11}
            for j in jobs:save(j,means[j['tag']])
            chosen,rows,_=f.choose_and_freeze(out,jobs)
            self.assertEqual(chosen,'W4') # ties choose lower mode in registered order
            self.assertEqual(f.choose_and_freeze(out,jobs)[0],chosen)
            save(jobs[0],10,quality=False)
            with self.assertRaisesRegex(ValueError,'품질 오류'):f.choose_and_freeze(out,jobs)
            save(jobs[0],10)
            target=Path(jobs[1]['target'])/'metrics.json';m=w.read(target);m['cohort_sha256']='wrong';w.write(target,m)
            cpath=target.parent/'cache.json';c=w.read(cpath);c['metrics_sha256']=w.file_hash(target);w.write(cpath,c)
            with self.assertRaisesRegex(ValueError,'차량/경로'):f.choose_and_freeze(out,jobs)

    def test_cache_rejects_changed_policy_and_trace(self):
        with tempfile.TemporaryDirectory() as d:
            j=f.jobs_for(d,Path(d)/'w','fp','sweep',f.config()['policies'][:1],f.conditions()[:1])[0]
            save(j)
            self.assertIsNotNone(f.read_job(j))
            (Path(j['target'])/'decisions.jsonl.gz').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'해시'):f.read_job(j)

    def test_interrupted_reservations_are_charged_on_restart(self):
        with tempfile.TemporaryDirectory() as d:
            record={'compute_hours':96,'prior_compute_s':100.0}
            ledger=long.Ledger(Path(d),record)
            ledger.begin('sweep/test',600)
            resumed=long.Ledger(Path(d),record)
            self.assertEqual(resumed.used,700)
            self.assertEqual(resumed.pending,{})
            self.assertEqual(resumed.entries[-1]['status'],'interrupted_conservative_cap')

    def test_fake_scheduler_restarts_without_charging_cached_results(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/'out';jobs=f.jobs_for(out,Path(d)/'work','fp','sweep',f.config()['policies'],f.conditions()[:1])
            out.mkdir();ledger=long.Ledger(out,{'compute_hours':96,'prior_compute_s':100.0})
            def fake(j):
                save(j)
                return {'ok':True,'seconds':1.0,'name':Path(j['target']).name}
            def thread_pool(max_workers,**kwargs):return ThreadPoolExecutor(max_workers=max_workers)
            with patch.object(f,'ProcessPoolExecutor',side_effect=thread_pool),patch.object(f,'eval_worker',side_effect=fake):
                f.execute(jobs,ledger,2,time.time()+60)
                after=ledger.used
                self.assertEqual(after,105.0)
                f.execute(jobs,ledger,2,time.time()+60)
                self.assertEqual(ledger.used,after)

    def test_summary_has_all_situations_and_movement_results(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/'out';jobs=f.jobs_for(out,Path(d)/'work','fp','sweep',f.config()['policies'],f.conditions())
            for j in jobs:save(j,10 if j['tag']=='P0' else 9)
            chosen,rows,_=f.choose_and_freeze(out,jobs)
            f.make_summary(out,rows,chosen)
            summary=w.read(out/'summary.json')
            self.assertEqual(len(summary['situations']),24)
            self.assertTrue(summary['sweep_comparison']['candidate_movements'])
            self.assertEqual(summary['exploratory_vs_m4']['M4']['lower_95'],0.0)
            self.assertNotIn('reused64',summary)
            refs={'P0':rows['P0'],'M4':rows['M4'],'RL42':rows['W4'],'RL43':rows['W4'],'RL44':rows['W4']}
            f.make_summary(out,rows,chosen,refs,rows[chosen])
            compared=w.read(out/'summary.json')['reused64']
            self.assertEqual(compared['gain_vs_rl_three_s'],0.0)
            self.assertEqual(len(compared['comparisons']['M4']['groups']),24)

    def test_main_refuses_mac_simulation(self):
        with patch.object(sys,'argv',['x','--out','a','--work','b','--night','c','--resume','d','--source','e']), patch.object(sys,'platform','darwin'):
            with self.assertRaisesRegex(SystemExit,'Colab'):f.main()

if __name__=='__main__':unittest.main()
