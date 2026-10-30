"""FIXED3: tolerance comparison, bounded exclusion and reproducibility probe, using fakes only (no SUMO)."""
import pickle
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import explore_m4probe as base
import explore_m4probe_fixed3 as f
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
        extra = 0.0
        if e.restored and v == 'b':
            if e.drift is not None and e.steps >= e.drift:
                extra += e.drift_size
            if e.noise is not None and e.steps >= e.noise:
                extra += 4e-9
        return 10.0 + 0.1 * e.sim_step + extra

    def getSpeed(self, v):
        return 5.0

    def getRoute(self, v):
        if v != 'v':
            raise ValueError('unknown')
        return ('edge1', 'edge2')

    def getDepartDelay(self, v):
        return 7.0


class TL:
    def getRedYellowGreenState(self, tls):
        return 'GGrr'


class Env:
    def __init__(self, now, restored=False, drift=None, noise=None, drift_size=1.0):
        self.sim_step = now
        self.start = now
        self.restored = restored
        self.drift = drift
        self.noise = noise
        self.drift_size = drift_size
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
        return SimpleNamespace(now=env.sim_step, mode_disagrees=True, mode_actions=[0, 1, 2, 2, 3],
                               mode_mask=[True, True, True, False, True], allowed=[0, 1, 2, 3], p0_action=0, observation=np.zeros(110))


def save(ep, stem, allowed, action):
    stem.with_suffix('.xml').write_text('state')
    saved = {'t0': ep.env.sim_step, 'route': ep.route.name, 'handover': ep.handover}
    stem.with_suffix('.pkl').write_bytes(pickle.dumps(saved))
    return saved


def fingerprint(sumo):
    e = sumo.env
    pending = ','.join(sumo.simulation.getPendingVehicles())
    flag = ''
    if e.restored and e.drift is not None and e.steps >= e.drift:
        flag += 'drift'
    if e.restored and e.noise is not None and e.steps >= e.noise:
        flag += 'noise'
    return '%s:%s:%s' % (e.sim_step, pending, flag)


def run_collector(tmp, restore_kwargs=None):
    """restore_kwargs: a dict, or a list of dicts used for the 1st, 2nd, ... restoration of each state."""
    condition = base.config()['conditions'][1]
    job = {'condition': condition, 'fingerprint': 'fp', 'target': str(Path(tmp) / 'states' / base.scenario_id(condition)),
           'work': str(Path(tmp) / 'work')}
    calls = {'n': 0}

    def restore(stem, route_dir, future, seed=None, horizon=300, tripinfo=None):
        saved = pickle.loads(stem.with_suffix('.pkl').read_bytes())
        kw = restore_kwargs
        if isinstance(kw, list):
            kw = kw[min(calls['n'], len(kw) - 1)]
        calls['n'] += 1
        env = Env(saved['t0'], True, **(kw or {}))
        return env, Cons(), SimpleNamespace(green_phases=[SimpleNamespace(state='x')] * 4), SimpleNamespace(movement_of={}), saved, [], {}
    good = {'collisions_at_204820': 0, 'collisions_unexplained': []}
    with patch.object(episode, '_Episode', Episode), patch.object(core, 'ModePlanner', Planner), \
            patch.object(er.Episode, 'snapshot', side_effect=save), patch.object(repair, '_ORIGINAL_RESTORE', side_effect=restore), \
            patch.object(er, 'trace_fingerprint', side_effect=fingerprint), patch.object(focused_eval, 'judge_collisions', return_value=good):
        result = f.collect_worker(job)
    return result, job


def table(vehicles, signal='GGrr', pending=(), time=10.0):
    return {'time': time, 'signal': signal, 'pending': list(pending), 'vehicles': vehicles}


class TablesMatchTest(unittest.TestCase):
    def test_tolerance_accepts_rounding_and_rejects_real_differences(self):
        a = table({'a': ['l1', 10.0, 5.0], 'b': ['l1', 20.0, 5.0]})
        self.assertTrue(f.tables_match(a, table({'a': ['l1', 10.0 + 4e-9, 5.0 + 4e-9], 'b': ['l1', 20.0, 5.0]}))['ok'])
        self.assertTrue(f.tables_match(a, table({'a': ['l1', 10.0 - 3.7e-21, 5.0], 'b': ['l1', 20.0, 5.0]}))['ok'])
        bad = f.tables_match(a, table({'a': ['l1', 10.0 + 2e-6, 5.0], 'b': ['l1', 20.0, 5.0]}))
        self.assertFalse(bad['ok'])
        self.assertAlmostEqual(bad['max_position_difference'], 2e-6, places=12)
        self.assertFalse(f.tables_match(a, table({'a': ['l1', 10.0, 5.0 + 1.8], 'b': ['l1', 20.0, 5.0]}))['ok'])

    def test_lane_signal_pending_time_and_vehicle_set_must_be_identical(self):
        a = table({'a': ['l1', 10.0, 5.0]}, pending=['p'])
        self.assertFalse(f.tables_match(a, table({'a': ['l2', 10.0, 5.0]}, pending=['p']))['ok'])
        self.assertFalse(f.tables_match(a, table({'a': ['l1', 10.0, 5.0]}, signal='rrGG', pending=['p']))['ok'])
        self.assertFalse(f.tables_match(a, table({'a': ['l1', 10.0, 5.0]}, pending=[]))['ok'])
        self.assertFalse(f.tables_match(a, table({'a': ['l1', 10.0, 5.0]}, pending=['p'], time=11.0))['ok'])
        self.assertFalse(f.tables_match(a, table({'a': ['l1', 10.0, 5.0], 'c': ['l1', 1.0, 1.0]}, pending=['p']))['ok'])
        self.assertIsNone(f.tables_match(None, a)['ok'])


class Fixed3CollectorTest(unittest.TestCase):
    def test_rounding_noise_passes_and_is_counted(self):
        with tempfile.TemporaryDirectory() as tmp:
            result, job = run_collector(tmp, {'noise': 2})
            self.assertTrue(result['ok'], result)
            self.assertEqual(result['excluded'], [])
            rec = base.collect_cached(job)
            self.assertEqual(len(rec['states']), 2)
            for s in rec['states']:
                self.assertTrue(s['restore_ok'])
                self.assertGreater(s['tolerance_steps'], 0)
                self.assertLess(s['max_position_difference'], 1e-8)
            self.assertFalse((Path(tmp) / 'restore_failed').exists())

    def test_real_divergence_is_excluded_not_relaxed(self):
        with tempfile.TemporaryDirectory() as tmp:
            result, job = run_collector(tmp, {'drift': 10, 'drift_size': 3.6})
            self.assertTrue(result['ok'], result)
            self.assertEqual([e['kind'] for e in result['excluded']], ['state_after_step', 'state_after_step'])
            self.assertEqual([e['step'] for e in result['excluded']], [10, 10])
            self.assertTrue(all(e['restore_reproducible'] is True for e in result['excluded']))
            self.assertAlmostEqual(result['excluded'][0]['max_position_difference'], 3.6)
            rec = base.collect_cached(job)
            self.assertEqual(rec['states'], [])
            self.assertEqual(len(rec['excluded_states']), 2)
            kept = Path(tmp) / 'restore_failed' / '20220812_0730_s43'
            self.assertTrue(any(p.name.endswith('.timeline.json') for p in kept.iterdir()))
            self.assertFalse(any('_w' in p.name for p in Path(job['target']).iterdir()))   # 제외된 상태 파일은 결과 폴더에 남지 않는다
            report = w.read(Path(tmp) / 'restore_diagnostics' / '20220812_0730_s43.json')
            self.assertFalse(report['collector_ok'])
            self.assertEqual(report['tolerance_m_and_mps'], 1e-6)

    def test_irreproducible_restoration_is_reported_as_such(self):
        with tempfile.TemporaryDirectory() as tmp:
            # restoration order: replay(w0), replay(w1), probe(w0), probe(w1) -> both replays diverge, both probes are perfect
            bad = {'drift': 10, 'drift_size': 3.6}
            result, job = run_collector(tmp, [bad, bad, {}, {}])
            self.assertTrue(all(e['restore_reproducible'] is False for e in result['excluded']), result['excluded'])

    def test_pending_mismatch_is_an_excluded_exception_with_its_message(self):
        with tempfile.TemporaryDirectory() as tmp:
            condition = base.config()['conditions'][1]
            with patch.object(repair, 'install_initial_pending', side_effect=ValueError('복원 직후 대기 차량 불일치 v')):
                result, job = run_collector(tmp, {})
            self.assertEqual({e['kind'] for e in result['excluded']}, {'exception'})
            failure = w.read(Path(tmp) / 'restore_diagnostics' / '20220812_0730_s43.json')['states'][0]['failure']
            self.assertIn('대기 차량 불일치', failure['error'])


class Fixed3ReportTest(unittest.TestCase):
    def test_report_counts_exclusions_tolerance_use_and_pending_repairs(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            jobs = base.collect_jobs(out, out / 'work', 'fp')[:3]
            for j, (excluded, tolerance, shim) in zip(jobs[:2], (([{'id': 'x_w1', 'kind': 'state_after_step', 'step': 4}], 0, False), ([], 3, True))):
                state = {'id': 's_w0', 'restore_ok': True, 'tolerance_steps': tolerance,
                         'pending_fix_check': {'missing_initial': ['v'] if shim else []}}
                w.write(Path(j['target']) / 'episode.json', {'fingerprint': 'fp', 'condition': j['condition'], 'states': [state],
                                                              'excluded_states': excluded, 'files': {}})
            summary = f.write_report(out, jobs, [])
            self.assertEqual((summary['ok'], summary['failed'], summary['not_run']), (2, 0, 1))
            self.assertEqual(summary['excluded_states'], 1)
            self.assertEqual(summary['states_using_tolerance'], 1)
            report = w.read(out / 'restore_report.json')
            self.assertEqual(report['excluded'][0]['scenario'], base.scenario_id(jobs[0]['condition']))
            self.assertEqual(report['scenarios'][1]['states_needing_pending_fix'], ['s_w0'])
            self.assertEqual(report['scenarios'][2]['status'], 'not_run')


class Fixed3RunTest(unittest.TestCase):
    def _args(self, root):
        return SimpleNamespace(out=str(root / 'out'), work=str(root / 'work'), fixed=str(root / 'fixed'),
                               night=str(root / 'night'), source=str(root / 'source'), workers=2)

    def _run(self, summary, failures=()):
        import focused_scenario
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            continuation = MagicMock()
            args = self._args(root)
            with patch.object(base, 'register', return_value={'fingerprint': 'fp'}), patch.object(f.long, 'available_cpus', return_value=2), \
                    patch.object(f, 'execute_all', return_value=list(failures)), patch.object(f, 'write_report', return_value=summary), \
                    patch.object(rep, 'run', continuation), patch.object(focused_scenario, 'dqn_template_network', return_value=None):
                try:
                    f.run(args)
                    error = None
                except f.RestoreFailures as e:
                    error = e
            return continuation, error, args

    def summary(self, excluded, failed=0):
        return {'ok': 16 - failed, 'failed': failed, 'not_run': 0, 'excluded_states': excluded,
                'excluded_limit': f.MAX_EXCLUDED_STATES, 'states_using_tolerance': 2}

    def test_within_the_exclusion_limit_the_unchanged_continuation_runs(self):
        continuation, error, args = self._run(self.summary(1))
        self.assertIsNone(error)
        continuation.assert_called_once_with(args)

    def test_too_many_exclusions_stop_before_any_rollout(self):
        continuation, error, _ = self._run(self.summary(f.MAX_EXCLUDED_STATES + 1))
        self.assertIsNotNone(error)
        self.assertIn('상한', str(error))
        continuation.assert_not_called()

    def test_scenario_that_could_not_be_collected_stops_the_run(self):
        failure = {'ok': False, 'seconds': 1., 'name': '20220812_0730_s43', 'error': 'x', 'diagnostic_file': 'd'}
        continuation, error, _ = self._run(self.summary(0, failed=1), [failure])
        self.assertIsNotNone(error)
        continuation.assert_not_called()

    def test_earlier_diagnostics_are_moved_once_never_deleted(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / 'restore_failed' / 's').mkdir(parents=True)
            (out / 'restore_failed' / 's' / 'x.xml').write_text('state')
            (out / 'restore_diagnostics').mkdir()
            (out / 'restore_diagnostics' / 's.json').write_text('{}')
            (out / 'restore_report.json').write_text('{}')
            self.assertTrue(f.archive_earlier_diagnostics(out))
            self.assertEqual((out / 'archive_fixed2' / 'restore_failed' / 's' / 'x.xml').read_text(), 'state')
            self.assertFalse((out / 'restore_failed').exists())
            (out / 'restore_diagnostics').mkdir()      # this run's own output must not be moved on a second call
            self.assertFalse(f.archive_earlier_diagnostics(out))
            self.assertTrue((out / 'restore_diagnostics').exists())


if __name__ == '__main__':
    unittest.main()
