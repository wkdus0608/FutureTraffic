"""Resume an existing M4 probe without compute limits, using fakes only."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch
import json
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import explore_m4probe as base
import explore_m4probe_continue as c
import explore_m4probe_diagnostics as diag
import explore_weight as w
import explore_weight_episode as episode


class ContinueTest(unittest.TestCase):
    def test_budget_free_scheduler_reuses_original_states(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/'out';out.mkdir()
            jobs=base.collect_jobs(out,Path(d)/'work','fp')[:3]
            w.write(out/'budget.json',{'used_s':300.,'pending':{},'entries':[]})
            journal=c.Journal(out,'fp')
            def fake(job):
                w.write(Path(job['target'])/'episode.json',{'fingerprint':'fp','condition':job['condition'],'states':[],'files':{}})
                return {'ok':True,'seconds':20000.,'name':base.scenario_id(job['condition'])}
            def thread_pool(max_workers,**kwargs):return ThreadPoolExecutor(max_workers=max_workers)
            with patch.object(c,'ProcessPoolExecutor',side_effect=thread_pool),patch.object(c,'collect_worker',side_effect=fake):
                c.execute(jobs,'collect',journal,2)
                self.assertEqual(len(journal.entries),3)
                self.assertEqual(sum(x['seconds'] for x in journal.entries),60000.)
                c.execute(jobs,'collect',journal,2)
                self.assertEqual(len(journal.entries),3)
            self.assertEqual(w.read(out/'budget.json')['used_s'],300.)

    def test_interrupted_work_is_marked_unknown_not_made_up(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d);w.write(out/'budget.json',{'used_s':12.,'pending':{'old':{'job':'prior'}}})
            first=c.Journal(out,'fp');first.begin('partial')
            restored=c.Journal(out,'fp')
            self.assertEqual(restored.entries[-1]['status'],'interrupted_duration_unknown')
            self.assertIsNone(restored.entries[-1]['seconds'])
            self.assertIn('old',restored.legacy['snapshot']['pending'])
            self.assertEqual(restored.pending,{})

    def test_collector_drops_compute_deadline_only(self):
        seen=[]
        def original(*args,deadline=None,**kwargs):seen.append(deadline);return 'fake'
        def fake_collect(job):return episode._Episode('a','b',42,Path('/tmp/fake'),deadline=123)
        with patch.object(episode,'_Episode',side_effect=original),patch.object(base,'collect_worker',side_effect=fake_collect), \
             patch.object(diag,'collect',side_effect=lambda job,fn:fn(job)):
            self.assertEqual(c.collect_worker({}),'fake')
        self.assertEqual(seen,[None])

    def test_protocol_records_original_identity_and_cannot_change(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d);r=c.register_continuation(out,{'fingerprint':'old-fp'})
            self.assertEqual(r['original_experiment_fingerprint'],'old-fp')
            self.assertEqual(c.register_continuation(out,{'fingerprint':'old-fp'}),r)
            with self.assertRaisesRegex(ValueError,'다릅니다'):c.register_continuation(out,{'fingerprint':'different'})

    def test_mac_execution_is_blocked(self):
        argv=['x','--out','a','--work','b','--fixed','c','--night','d','--source','e']
        with patch.object(sys,'argv',argv),patch.object(sys,'platform','darwin'):
            with self.assertRaisesRegex(SystemExit,'Colab'):c.main()

if __name__=='__main__':unittest.main()
