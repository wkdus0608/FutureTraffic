"""FIXED2: restore diagnostics and collect-all reporting, using fakes only (no SUMO)."""
import pickle
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import explore_m4probe as base
import explore_m4probe_fixed2 as f
import explore_m4probe_pending as repair
import explore_m4probe_repaired as rep
import explore_rollout as er
import explore_weight as w
import explore_weight_core as core
import explore_weight_episode as episode
import focused_eval


class Sim:
    def __init__(self, env):
        self.env = env

    def getTime(self):
        return self.env.sim_step

    def getPendingVehicles(self):
        return () if self.env.restored and self.env.sim_step == self.env.start else ('v',)


class Vehicles:
    def __init__(self, env):
        self.env = env

    def getIDList(self):
        return ('a', 'b')

    def getLaneID(self, v):
        return 'lane_' + v

    def getLanePosition(self, v):
        e = self.env
        extra = 1.0 if (e.restored and e.drift is not None and e.steps >= e.drift and v == 'b') else 0.0
        return 10.0 + 0.1 * e.sim_step + extra

    def getSpeed(self, v):
        return 5.0

    def getRoute(self, v):
        if v != 'v':
            raise ValueError('unknown')
        return ('edge1', 'edge2') if self.env.route_ok else ('wrong',)

    def getDepartDelay(self, v):
        return 7.0


class TL:
    def getRedYellowGreenState(self, tls):
        return 'GGrr'


class Env:
    def __init__(self, now, restored=False, drift=None, obs_drift_step=None, route_ok=True):
        self.sim_step = now
        self.start = now
        self.restored = restored
        self.drift = drift
        self.obs_drift_step = obs_drift_step
        self.route_ok = route_ok
        self.steps = 0
        self.sumo = SimpleNamespace(simulation=Sim(self), vehicle=Vehicles(self), trafficlight=TL(), env=self)

    def step(self, a):
        self.sim_step += 5
        self.steps += 1

    def close(self):
        pass


class Cons:
    def __call__(self, env, a):
        return a


class Episode:
    def __init__(self, date, period, seed, work, deadline=None):
        work.mkdir(parents=True)
        self.route = work / ('%s_%s.rou.xml' % (date, period))
        self.route.write_text('routes')
        (work / 'collisions.xml').write_text('<collisions/>')
        self.env = Env(1800)
        self.ts = SimpleNamespace(green_phases=[SimpleNamespace(state='x')] * 4)
        self.cons = Cons()
        self.movement_of = {}
        self.handover = 1800
        self.vehicles = {}
        self.done = False

    def apply(self, a):
        self.env.step(a)
        return a, a

    def close(self):
        pass


class Planner:
    def __init__(self, *args):
        pass

    def snapshot(self, env):
        obs = np.zeros(110)
        if env.restored and env.obs_drift_step is not None and env.steps == env.obs_drift_step:
            obs[54] = 0.5
        return SimpleNamespace(now=env.sim_step, mode_disagrees=True, mode_actions=[0, 1, 2, 2, 3],
                               mode_mask=[True, True, True, False, True], allowed=[0, 1, 2, 3], p0_action=0, observation=obs)


def save(ep, stem, allowed, action):
    stem.with_suffix('.xml').write_text('state')
    saved = {'t0': ep.env.sim_step, 'route': ep.route.name, 'handover': ep.handover}
    stem.with_suffix('.pkl').write_bytes(pickle.dumps(saved))
    return saved


def fingerprint(sumo):
    e = sumo.env
    pending = ','.join(sumo.simulation.getPendingVehicles())
    drift = e.restored and e.drift is not None and e.steps >= e.drift
    return '%s:%s:%s' % (e.sim_step, pending, 'drift' if drift else '')


def run_collector(tmp, **env_kwargs):
    condition = base.config()['conditions'][1]
    job = {'condition': condition, 'fingerprint': 'fp', 'target': str(Path(tmp) / 'states' / base.scenario_id(condition)),
           'work': str(Path(tmp) / 'work')}

    def restore(stem, route_dir, future, seed=None, horizon=300, tripinfo=None):
        saved = pickle.loads(stem.with_suffix('.pkl').read_bytes())
        env = Env(saved['t0'], True, **env_kwargs)
        return env, Cons(), SimpleNamespace(green_phases=[SimpleNamespace(state='x')] * 4), SimpleNamespace(movement_of={}), saved, [], {}
    good = {'collisions_at_204820': 0, 'collisions_unexplained': []}
    with patch.object(episode, '_Episode', Episode), patch.object(core, 'ModePlanner', Planner), \
            patch.object(er.Episode, 'snapshot', side_effect=save), patch.object(repair, '_ORIGINAL_RESTORE', side_effect=restore), \
            patch.object(er, 'trace_fingerprint', side_effect=fingerprint), patch.object(focused_eval, 'judge_collisions', return_value=good):
        result = f.collect_worker(job)
    return result, job


class Fixed2CollectorTest(unittest.TestCase):
    def test_pass_keeps_raw_tables_and_same_cache_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            result, job = run_collector(tmp)
            self.assertTrue(result['ok'], result)
            rec = base.collect_cached(job)
            self.assertEqual(len(rec['states']), 2)
            for s in rec['states']:
                self.assertTrue(s['restore_ok'])
                self.assertEqual(s['restore_horizon_s'], 300)
                self.assertEqual(s['pending_fix_check']['missing_initial'], ['v'])
                self.assertNotIn('timeline', s)
                self.assertNotIn('physical_before', s)
                line = w.read(Path(job['target']) / (s['id'] + '.timeline.json'))
                self.assertEqual(len(line['steps']), 60)
                self.assertEqual(line['physical_before']['pending'], ['v'])
                self.assertIn('a', line['steps'][0]['physical_after']['vehicles'])
            report = w.read(Path(tmp) / 'restore_diagnostics' / '20220812_0730_s43.json')
            self.assertTrue(report['collector_ok'])
            self.assertFalse((Path(tmp) / 'restore_failed').exists())

    def test_later_vehicle_drift_is_explained_and_both_states_are_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            result, job = run_collector(tmp, drift=10)
            self.assertFalse(result['ok'])
            self.assertEqual([x['kind'] for x in result['failures']], ['state_after_step', 'state_after_step'])
            self.assertEqual([x['step'] for x in result['failures']], [10, 10])   # 10번째 5초 단계를 마친 직후
            self.assertFalse((Path(job['target']) / 'episode.json').exists())
            report = w.read(Path(tmp) / 'restore_diagnostics' / '20220812_0730_s43.json')
            self.assertEqual(len(report['states']), 2)
            phys = report['states'][0]['failure']['physical']
            self.assertTrue(phys['available'])
            self.assertEqual(phys['changed_count'], 1)
            self.assertEqual(phys['changed_examples'][0]['vehicle'], 'b')
            self.assertAlmostEqual(phys['max_position_difference'], 1.0)
            kept = Path(result['kept_states'])
            self.assertTrue(any(p.name.endswith('.timeline.json') for p in kept.iterdir()))
            self.assertTrue(any(p.name.endswith('.xml') for p in kept.iterdir()))

    def test_observation_difference_names_the_cell_and_group(self):
        with tempfile.TemporaryDirectory() as tmp:
            result, job = run_collector(tmp, obs_drift_step=3)
            self.assertFalse(result['ok'])
            failure = w.read(Path(tmp) / 'restore_diagnostics' / '20220812_0730_s43.json')['states'][0]['failure']
            self.assertEqual(failure['kind'], 'observation_or_mode')
            self.assertEqual(failure['step'], 3)
            diff = failure['snapshot']['observation_differences'][0]
            self.assertEqual((diff['index'], diff['group']), (54, '기본 관측'))
            self.assertAlmostEqual(diff['restored'], 0.5)
            lines = f.describe_failure('state', failure)
            self.assertTrue(any('관측 54번' in x for x in lines))

    def test_pending_route_mismatch_is_a_recorded_failure_not_a_crash(self):
        with tempfile.TemporaryDirectory() as tmp:
            result, job = run_collector(tmp, route_ok=False)
            self.assertFalse(result['ok'])
            self.assertEqual({x['kind'] for x in result['failures']}, {'exception'})
            failure = w.read(Path(tmp) / 'restore_diagnostics' / '20220812_0730_s43.json')['states'][0]['failure']
            self.assertIn('대기 차량 불일치', failure['error'])

    def test_strict_rule_is_not_relaxed_by_the_diagnostics(self):
        # same input that the unchanged collector rejects must be rejected here as well
        with tempfile.TemporaryDirectory() as tmp:
            old, job = None, None
            condition = base.config()['conditions'][1]
            job = {'condition': condition, 'fingerprint': 'fp', 'target': str(Path(tmp) / 'states' / 'scenario'),
                   'work': str(Path(tmp) / 'work')}

            def restore(stem, route_dir, future, seed=None, horizon=300, tripinfo=None):
                saved = pickle.loads(stem.with_suffix('.pkl').read_bytes())
                return Env(saved['t0'], True, drift=10), Cons(), SimpleNamespace(green_phases=[SimpleNamespace(state='x')] * 4), \
                    SimpleNamespace(movement_of={}), saved, [], {}
            good = {'collisions_at_204820': 0, 'collisions_unexplained': []}
            with patch.object(episode, '_Episode', Episode), patch.object(core, 'ModePlanner', Planner), \
                    patch.object(er.Episode, 'snapshot', side_effect=save), patch.object(repair, '_ORIGINAL_RESTORE', side_effect=restore), \
                    patch.object(er, 'trace_fingerprint', side_effect=fingerprint), patch.object(focused_eval, 'judge_collisions', return_value=good):
                old = repair.collect_worker(job)
            self.assertFalse(old['ok'])
            result, _ = run_collector(tmp + '/second', drift=10)
            self.assertFalse(result['ok'])


class Fixed2RunTest(unittest.TestCase):
    def _args(self, root):
        return SimpleNamespace(out=str(root / 'out'), work=str(root / 'work'), fixed=str(root / 'fixed'),
                               night=str(root / 'night'), source=str(root / 'source'), workers=2)

    def test_execute_all_keeps_going_after_a_failure_and_returns_all_of_them(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            journal = rep.Journal(out, 'fp')
            jobs = base.collect_jobs(out, out / 'work', 'fp')[:4]
            seen = []

            def fake(job):
                seen.append(base.scenario_id(job['condition']))
                if job is jobs[1]:
                    return {'ok': False, 'seconds': 1., 'name': base.scenario_id(job['condition']), 'error': 'x', 'failures': [], 'diagnostic_file': 'd'}
                w.write(Path(job['target']) / 'episode.json', {'fingerprint': 'fp', 'condition': job['condition'], 'states': [], 'files': {}})
                return {'ok': True, 'seconds': 1., 'name': base.scenario_id(job['condition'])}

            def pool(max_workers, **kwargs):
                return ThreadPoolExecutor(max_workers=max_workers)
            with patch.object(f, 'ProcessPoolExecutor', side_effect=pool), patch.object(f, 'collect_worker', side_effect=fake):
                failures = f.execute_all(jobs, journal, 2)
            self.assertEqual(sorted(seen), sorted(base.scenario_id(j['condition']) for j in jobs))
            self.assertEqual([x['name'] for x in failures], [base.scenario_id(jobs[1]['condition'])])
            self.assertEqual(sorted(x['status'] for x in journal.entries), ['complete'] * 3 + ['failed'])

    def test_run_stops_before_rollouts_when_any_restore_failed_and_writes_the_report(self):
        import focused_scenario
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            failure = {'ok': False, 'seconds': 1., 'name': '20220812_0730_s43', 'error': 'bad', 'failures': [{'state': 's', 'kind': 'x', 'step': 1}],
                       'diagnostic_file': 'd'}
            continuation = MagicMock()
            with patch.object(base, 'register', return_value={'fingerprint': 'fp'}), patch.object(f.long, 'available_cpus', return_value=2), \
                    patch.object(f, 'execute_all', side_effect=[[failure], []]), patch.object(rep, 'run', continuation), \
                    patch.object(focused_scenario, 'dqn_template_network', return_value=None):
                with self.assertRaises(f.RestoreFailures):
                    f.run(self._args(root))
            continuation.assert_not_called()
            report = w.read(root / 'out' / 'restore_report.json')
            self.assertEqual(report['summary']['failed'], 1)
            self.assertEqual(next(r for r in report['scenarios'] if r['status'] == 'failed')['scenario'], '20220812_0730_s43')
            self.assertTrue((root / 'out' / 'fixed2_stages.json').exists())

    def test_run_hands_over_to_the_unchanged_continuation_only_after_everything_passed(self):
        import focused_scenario
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            continuation = MagicMock()
            args = self._args(root)
            with patch.object(base, 'register', return_value={'fingerprint': 'fp'}), patch.object(f.long, 'available_cpus', return_value=2), \
                    patch.object(f, 'execute_all', side_effect=[[], []]), patch.object(rep, 'run', continuation), \
                    patch.object(focused_scenario, 'dqn_template_network', return_value=None):
                f.run(args)
            continuation.assert_called_once_with(args)


if __name__ == '__main__':
    unittest.main()
