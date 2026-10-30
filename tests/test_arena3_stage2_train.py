"""Teacher/RL continuation and accounting checks, with no real simulator."""
import copy
import json
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import arena3_common as ac
import arena3_g1 as g1
import arena3_stage2_train as s2


def observation():
    greens = [0, 0, 1, 0, 1, 2]
    mask = ac.canonical_mask6(greens)
    obs = np.zeros(119, np.float32)
    for m,g in enumerate(greens[:5]):
        obs[84+m*4+g]=1
    obs[104:109]=mask[:5]
    obs[110+greens[5]]=1
    obs[114]=mask[5]
    return obs


def dataset():
    obs=np.stack([observation() for _ in range(12)])
    mask=g1.mode_mask(obs)
    gain=np.zeros((12,6),np.float32)
    gain[:,5]=0.5
    return {'observations':obs,'gains':gain,'gain_se':np.ones_like(gain)*0.01,
            'mask':mask,'dates':np.asarray([ac.config()['dates']['train'][0]]*12),
            'fingerprint':'fake-teacher-fingerprint','manifest':{'test':True}}


def fake_pretrain(net,data,variant,seed,**kwargs):
    with torch.no_grad():
        net.advantage.bias[5]=0.5
    return {'seconds':0.0,'epochs':1,'loss':0.1}


def fake_worker(job):
    net=g1._net_from_bytes(job['net'])
    policy=g1.EpsilonPolicy(net,job['base_choices'],job['rng_seed'],g1.gcfg())
    obs=observation()
    greens=[0,0,1,0,1,2]
    for step in [20,40]:
        snap=ac.ArenaSnapshot(base=None,greens=greens,mask=ac.canonical_mask6(greens),
                              choice=True,observation=obs)
        policy.decide(snap,type('Ctx',(),{'step':step})())
    return {'ok':True,'status':'ok','errors':[],'metrics':{'mean_delay_s':30.0,'choice_points':2},
            'choices':policy.taken,'rewards':[-0.01]*720,'seconds':10.0}


class Stage2TrainTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.cfg=copy.deepcopy(g1.gcfg())
        self.cfg.update(replay_size=24,batch_size=4,learning_starts=2,updates_per_transition=1,
                        cpu_hours_per_seed=0.01)
        self.patch=patch.object(g1,'gcfg',return_value=self.cfg)
        self.patch.start()
        self.data=dataset()

    def tearDown(self):
        self.patch.stop()

    def test_variants_have_identical_scenarios_and_no_evaluation_dates(self):
        for seed in [42,43,44]:
            for ep in range(8):
                s=s2.scenario_for(seed,ep)
                self.assertIn(s['date'],ac.config()['dates']['train'])
                self.assertNotIn(s['date'],ac.config()['dates']['eval'])
                self.assertEqual(s,s2.scenario_for(seed,ep))

    def test_pretrain_then_resume_once_and_final_budget_includes_pretrain(self):
        with tempfile.TemporaryDirectory() as tmp, patch('arena3_stage2_data.pretrain',side_effect=fake_pretrain) as pre:
            out=Path(tmp)/'train';work=Path(tmp)/'work'
            a=s2.train(42,out,work,self.data,'advantage',executor=ThreadPoolExecutor(2),worker=fake_worker,max_batches=1)
            self.assertFalse(a['finished'])
            self.assertEqual(a['episodes'],2)
            self.assertEqual(pre.call_count,1)
            warm=s2.load_model(out/'seed_42/pretrained')
            self.assertEqual(g1.greedy_mode(warm,observation()),5)
            b=s2.train(42,out,work,self.data,'advantage',executor=ThreadPoolExecutor(2),worker=fake_worker,max_batches=1)
            self.assertTrue(b['finished'])
            self.assertEqual(b['episodes'],4)
            self.assertEqual(pre.call_count,1)
            self.assertTrue((out/'seed_42/final/manifest.json').exists())
            log=[json.loads(s) for s in (out/'seed_42/spend.jsonl').read_text().splitlines()]
            self.assertEqual(log[0]['note'],'pretraining')
            self.assertGreaterEqual(b['cpu_hours']*3600,40.0)
            self.assertEqual(s2.model_identity(out/'seed_42/final'),json.loads((out/'seed_42/final/manifest.json').read_text())['model_sha256'])

    def test_resume_rejects_variant_and_teacher_change(self):
        with tempfile.TemporaryDirectory() as tmp, patch('arena3_stage2_data.pretrain',side_effect=fake_pretrain):
            out=Path(tmp)/'out'
            s2.train(42,out,Path(tmp)/'work',self.data,'policy',worker=fake_worker,max_batches=0)
            with self.assertRaisesRegex(ValueError,'mismatch'):
                s2.train(42,out,Path(tmp)/'work',self.data,'advantage',worker=fake_worker,max_batches=0)
            changed={**self.data,'fingerprint':'different'}
            with self.assertRaisesRegex(ValueError,'mismatch'):
                s2.train(42,out,Path(tmp)/'work',changed,'policy',worker=fake_worker,max_batches=0)

    def test_model_rejects_byte_and_metadata_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            learner=s2.Learner(42,self.data,'policy')
            state={'episodes':0,'control_steps':0,'choices':0}
            p=Path(tmp)/'model'
            s2.save_model(p,learner,state,'pretrained')
            manifest=json.loads((p/'manifest.json').read_text())
            manifest['kind']='bad'
            (p/'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):s2.model_identity(p)
            s2.save_model(p,learner,state,'pretrained')
            with (p/'model.pt').open('ab') as f:f.write(b'bad')
            with self.assertRaisesRegex(ValueError,'SHA'):s2.load_model(p)

    def test_failed_episode_time_is_not_free(self):
        def bad(job):return {'ok':False,'status':'error','errors':['synthetic'],'seconds':13.0}
        with tempfile.TemporaryDirectory() as tmp,patch('arena3_stage2_data.pretrain',side_effect=fake_pretrain):
            out=Path(tmp)/'out'
            with self.assertRaises(RuntimeError):
                s2.train(42,out,Path(tmp)/'work',self.data,'policy',executor=ThreadPoolExecutor(2),worker=bad,max_batches=1)
            self.assertGreaterEqual(s2.SpendLog(out/'seed_42').total(),26)
            self.assertFalse((out/'seed_42/final').exists())

    def test_interrupted_reservation_charged_once_without_downtime(self):
        with tempfile.TemporaryDirectory() as tmp:
            spend=s2.SpendLog(Path(tmp))
            pending={'operation_id':'one','reserved_seconds':12.5,'note':'pretraining','heartbeat_unix':0}
            spend.pending.write_text(json.dumps(pending))
            spend.recover()
            self.assertEqual(spend.total(),12.5)
            spend.pending.write_text(json.dumps(pending))
            spend.recover()
            self.assertEqual(spend.total(),12.5)

    def test_teacher_weight_constant_unless_decay_given(self):
        learner=s2.Learner(42,self.data,'advantage')
        self.assertIsNone(learner.finetune['teacher_decay_choices'])
        learner.choices=50_000
        self.assertEqual(learner.teacher_weight(),learner.finetune['teacher_weight'])
        learner.finetune={**learner.finetune,'teacher_decay_choices':100}
        learner.choices=50
        self.assertAlmostEqual(learner.teacher_weight(),learner.finetune['teacher_weight']*0.5)
        learner.choices=150
        self.assertEqual(learner.teacher_weight(),0.0)

    def test_teacher_evaluation_dates_rejected(self):
        bad={**self.data,'dates':np.asarray([ac.config()['dates']['eval'][0]]*12)}
        with tempfile.TemporaryDirectory() as tmp,self.assertRaisesRegex(ValueError,'evaluation'):
            s2.train(42,Path(tmp),Path(tmp)/'work',bad,'policy',worker=fake_worker,max_batches=0)

    def test_real_mac_training_is_blocked(self):
        with patch.object(sys,'platform','darwin'),tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError,'Colab'):
                s2.train(42,Path(tmp),Path(tmp)/'work',self.data,'advantage',max_batches=0)


if __name__=='__main__':unittest.main()
