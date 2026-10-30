"""M4 selection-point pilot, stage 1, FIXED3: numerical tolerance, bounded exclusion, determinism probe.

What FIXED2 showed on Colab (2026-10-27, 16 scenarios, 32 saved states): 29 states passed the strict 300 s replay,
including the two states that had a pending vehicle (the pending repair works), and three failed:

* ``20221030_2130_s42_w0`` right after restore: one vehicle's lane position differed by 3.7e-21 m (the state file writes
  positions with 20 decimals, so 2**-48 became 3.55271e-15);
* ``20220812_1230_s42_w0`` at step 2: one vehicle's position/speed differed by 4.0e-9 m / 4.0e-9 m/s;
  both were rejected only because the vehicle fingerprint is a hash of exact floats;
* ``20220906_1730_s43_w1`` at step 4: three queued vehicles differed by up to 3.6 m and 1.8 m/s (a real divergence).

Changes (decided before any rollout outcome exists; everything else is exactly FIXED2 / the unchanged pilot):

1. The exact-float fingerprint is replaced, only when the hashes differ, by a tolerance comparison of the same raw
   tables (lane id, signal, pending list and vehicle set must be identical; position and speed within 1e-6, the same
   tolerance the 110-cell observation check already uses). A real divergence is still caught at the first step where it
   exceeds the tolerance.
2. A state that still fails is excluded from the pilot, never relaxed: it is recorded with its diagnosis in the episode
   file and ``restore_failed/``; a second independent restore is replayed to say whether the restored dynamics are
   reproducible. At most ``MAX_EXCLUDED_STATES`` (2 of 32) may be excluded; a larger number means the restore is not
   reliable and the run stops before any rollout.
3. Exclusion depends only on restore fidelity, never on any rollout outcome, and the count is reported.

When the restore check is satisfied, the unchanged continuation (``explore_m4probe_repaired.run``) runs the rollouts.
"""
from __future__ import annotations
import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import fcntl
import multiprocessing
from pathlib import Path
import pickle
import shutil
import signal
import sys
import time

import explore_m4probe as base
import explore_m4probe_diagnostics as diag
import explore_m4probe_pending as repair
import explore_m4probe_repaired as rep
import explore_rollout as er
import explore_weight as w
import explore_weight_long as long

ROOT = Path(__file__).resolve().parents[1]
REPLAY_STEPS = 60
TOLERANCE = 1e-6            # metres and metres/second, same as the observation check
DETERMINISM_TOLERANCE = 1e-9
MAX_EXCLUDED_STATES = 2     # of 32 saved states; fixed before any rollout outcome exists
OBS_GROUPS = ((0, 64, '기본 관측'), (64, 72, '이동별 대기행렬'), (72, 80, '이동별 최장 대기'), (80, 84, '후보 비용 차이'),
              (84, 104, '모드별 행동 원-핫'), (104, 109, '모드 마스크'), (109, 110, '남은 시간'))


class RestoreFailures(RuntimeError):
    def __init__(self, failures, reason=None):
        self.failures = failures
        super().__init__(reason or ('복원 검사 실패 상황 %d개: %s' % (len(failures), ', '.join(f['name'] for f in failures))))


def group_of(index):
    return next((name for lo, hi, name in OBS_GROUPS if lo <= index < hi), '알 수 없음')


def raw_tables(sumo, errors):
    """Raw vehicle/pending/signal table; used to explain a mismatch and for the tolerance comparison."""
    try:
        return diag.physical(sumo)
    except Exception as e:
        errors.append('%s: %s' % (type(e).__name__, e))
        return None


def tables_match(expected, actual, tolerance=TOLERANCE):
    """Same fields as ``trace_fingerprint`` (time, signal, pending list, vehicle ids, lane, position, speed) but
    positions and speeds are compared within ``tolerance``. ``ok`` is None if a table is missing."""
    if expected is None or actual is None:
        return {'ok': None}
    out = {'ok': False, 'max_position_difference': 0.0, 'max_speed_difference': 0.0}
    a, b = expected['vehicles'], actual['vehicles']
    if (expected['time'] != actual['time'] or expected['signal'] != actual['signal']
            or list(expected['pending']) != list(actual['pending']) or set(a) != set(b)):
        return out
    worst_position = worst_speed = 0.0
    lanes_equal = True
    for v, (lane, position, speed) in a.items():
        lane2, position2, speed2 = b[v]
        lanes_equal &= lane == lane2
        worst_position = max(worst_position, abs(position2 - position))
        worst_speed = max(worst_speed, abs(speed2 - speed))
    out['max_position_difference'], out['max_speed_difference'] = worst_position, worst_speed
    out['ok'] = bool(lanes_equal and worst_position <= tolerance and worst_speed <= tolerance)
    return out


def explain_snapshot(expected, actual):
    obs = [float(x) for x in actual.observation]
    exp = [float(x) for x in expected['observation']]
    diffs = [{'index': i, 'group': group_of(i), 'original': a, 'restored': b, 'difference': b - a}
             for i, (a, b) in enumerate(zip(exp, obs)) if abs(a - b) >= 1e-6]
    return {'actions_equal': list(actual.mode_actions) == list(expected['mode_actions']),
            'actions_original': [int(x) for x in expected['mode_actions']],
            'actions_restored': [int(x) for x in actual.mode_actions],
            'mask_equal': list(actual.mode_mask) == list(expected['mode_mask']),
            'allowed_equal': list(actual.allowed) == list(expected['allowed']),
            'p0_equal': int(actual.p0_action) == int(expected['p0_action']),
            'observation_length_equal': len(exp) == len(obs),
            'observation_difference_count': len(diffs), 'observation_differences': diffs[:20]}


def replay_state(s, staging, errors):
    """Strict 300 s replay of one saved state: identical decisions and, within the tolerance, identical traffic."""
    from explore_weight_core import ModePlanner
    out = {'state': s['id'], 'ok': False, 'saved_pending': s['pending_snapshot_count'],
           'tolerance_steps': 0, 'max_position_difference': 0.0, 'max_speed_difference': 0.0}
    env = None

    def same_traffic(fingerprint_now, fingerprint_expected, expected_table):
        """Exact hash first; if it differs, accept only a within-tolerance match of the raw tables."""
        if fingerprint_now == fingerprint_expected:
            return True, None
        table = raw_tables(env.sumo, errors)
        match = tables_match(expected_table, table)
        if match['ok']:
            out['tolerance_steps'] += 1
            out['max_position_difference'] = max(out['max_position_difference'], match['max_position_difference'])
            out['max_speed_difference'] = max(out['max_speed_difference'], match['max_speed_difference'])
            return True, table
        return False, table
    try:
        env, cons, ts, ctl, saved, _, _ = repair.restore(staging / s['id'], staging, 'real', horizon=300)
        out['pending_fix_check'] = saved.get('pending_fix_check')
        ok, table = same_traffic(er.trace_fingerprint(env.sumo), s['trace_before'], s.get('physical_before'))
        if not ok:
            out['failure'] = {'kind': 'right_after_restore', 'step': 0, 'restored_table': table,
                              'physical': diag.compare_physical(s.get('physical_before'), table)}
            return out
        check = ModePlanner(ctl.movement_of, [p.state for p in ts.green_phases], saved['handover'] + 3600)
        if len(s['timeline']) != REPLAY_STEPS:
            out['failure'] = {'kind': 'timeline_incomplete', 'length': len(s['timeline'])}
            return out
        for step, ref in enumerate(s['timeline']):
            actual = check.snapshot(env)
            if not repair.same_snapshot(actual, ref['snapshot']):
                before = s.get('physical_before') if step == 0 else s['timeline'][step - 1].get('physical_after')
                table = raw_tables(env.sumo, errors)
                out['failure'] = {'kind': 'observation_or_mode', 'step': step, 'restored_table': table,
                                  'snapshot': explain_snapshot(ref['snapshot'], actual),
                                  'physical': diag.compare_physical(before, table)}
                return out
            action = int(actual.mode_actions[4])
            corrected = int(cons(env, action))
            if action != ref['action'] or corrected != action:
                out['failure'] = {'kind': 'action', 'step': step, 'original_action': ref['action'],
                                  'restored_action': action, 'restored_corrected': corrected}
                return out
            env.step(corrected)
            ok, table = same_traffic(er.trace_fingerprint(env.sumo), ref['after'], ref.get('physical_after'))
            if not ok:
                out['failure'] = {'kind': 'state_after_step', 'step': step + 1, 'restored_table': table,
                                  'physical': diag.compare_physical(ref.get('physical_after'), table)}
                return out
        out['ok'] = True
    except Exception as e:
        out['failure'] = {'kind': 'exception', 'error': '%s: %s' % (type(e).__name__, e)}
    finally:
        if env is not None:
            try:
                env.close()
            except Exception as e:
                errors.append('close: %s: %s' % (type(e).__name__, e))
    return out


def reproducibility(first, second):
    """Do two independent restorations of the same saved state behave identically? (True/False/None)"""
    if second['ok'] != first['ok']:
        return False
    if first['ok']:
        return True
    a, b = first.get('failure', {}), second.get('failure', {})
    if (a.get('kind'), a.get('step')) != (b.get('kind'), b.get('step')):
        return False
    ta, tb = a.get('restored_table'), b.get('restored_table')
    if ta is None or tb is None:
        return None
    return bool(tables_match(ta, tb, DETERMINISM_TOLERANCE)['ok'])


def describe_failure(state, failure):
    """Short human-readable lines for the Colab log."""
    lines = ['  [실패] %s: 종류=%s%s' % (state, failure['kind'], (' step=%s' % failure['step']) if 'step' in failure else '')]
    if failure['kind'] == 'exception':
        lines.append('     %s' % failure['error'])
    snap = failure.get('snapshot')
    if snap:
        lines.append('     모드 행동 일치=%s 마스크 일치=%s 허용 현시 일치=%s 관측 차이 %d칸' % (
            snap['actions_equal'], snap['mask_equal'], snap['allowed_equal'], snap['observation_difference_count']))
        for d in snap['observation_differences'][:6]:
            lines.append('     관측 %d번(%s): 원본 %.6g → 복원 %.6g' % (d['index'], d['group'], d['original'], d['restored']))
    phys = failure.get('physical')
    if phys and phys.get('available'):
        lines.append('     신호 일치=%s 대기 목록 일치=%s 원본에만 대기=%s 복원에만 대기=%s' % (
            phys['signal_equal'], phys['pending_equal'], phys['pending_only_original'][:4], phys['pending_only_restored'][:4]))
        lines.append('     차량: 원본에만 %d, 복원에만 %d, 값이 다른 차량 %d (위치 최대 %.3g m, 속도 최대 %.3g m/s)' % (
            len(phys['vehicles_only_original']), len(phys['vehicles_only_restored']), phys['changed_count'],
            phys['max_position_difference'], phys['max_speed_difference']))
    if failure['kind'] == 'action':
        lines.append('     원본 행동 %s → 복원 %s (규칙 적용 후 %s)' % (failure['original_action'], failure['restored_action'], failure['restored_corrected']))
    if 'restore_reproducible' in failure:
        lines.append('     같은 상태를 한 번 더 복원했을 때 같은 어긋남이 같은 값으로 나왔는가: %s' % failure['restore_reproducible'])
    return lines


def collect_worker(job):
    from explore_weight_core import ModePlanner
    from explore_weight_episode import _Episode
    from focused_eval import judge_collisions
    c = job['condition']
    er.SUMO_SEED = int(c['seed'])
    started = time.monotonic()
    local, target = Path(job['work']), Path(job['target'])
    shutil.rmtree(local, ignore_errors=True)
    local.mkdir(parents=True)
    scenario = base.scenario_id(c)
    out_dir = target.parent.parent
    states, ep, errors = [], None, []
    report_path = out_dir / 'restore_diagnostics' / ('%s.json' % scenario)
    try:
        ep = _Episode(c['date'], c['period'], int(c['seed']), local / 'episode', deadline=None)
        ep.date, ep.period = c['date'], c['period']
        planner = ModePlanner(ep.movement_of, [p.state for p in ep.ts.green_phases], ep.handover + 3600)
        stem_dir = local / 'states'
        stem_dir.mkdir()
        while True:
            now = float(ep.env.sim_step)
            elapsed = now - ep.handover
            required_end = max([base.config()['windows_s'][-1][1]] + [s['elapsed_s'] + 300 for s in states])
            if elapsed >= required_end:
                break
            snap = planner.snapshot(ep.env)
            wanted = next((i for i, (a, b) in enumerate(base.config()['windows_s'])
                           if a <= elapsed < b and not any(z['window'] == i for z in states)), None)
            if wanted is not None and snap.mode_disagrees:
                name = base.state_id(c, wanted)
                stem = stem_dir / name
                saved = er.Episode.snapshot(ep, stem, snap.allowed, snap.p0_action)
                saved['pending_snapshot'] = repair.capture_pending(ep.env.sumo)
                stem.with_suffix('.pkl').write_bytes(pickle.dumps(saved))
                states.append({'id': name, 'window': wanted, 'elapsed_s': elapsed, 't0': snap.now,
                               **repair.saved_snapshot(snap), 'm4_action': int(snap.mode_actions[4]),
                               'route': ep.route.name, 'pending_snapshot_count': len(saved['pending_snapshot']),
                               'trace_before': er.trace_fingerprint(ep.env.sumo),
                               'physical_before': raw_tables(ep.env.sumo, errors), 'timeline': []})
            current = [s for s in states if s['t0'] <= now < s['t0'] + 300]
            action = int(snap.mode_actions[4])
            corrected, executed = ep.apply(action)
            if corrected != action or executed != action:
                raise RuntimeError('원본 M4 실행 규칙이 선택을 바꿨습니다.')
            if current:
                fingerprint = er.trace_fingerprint(ep.env.sumo)
                tables = raw_tables(ep.env.sumo, errors)
                shot = repair.saved_snapshot(snap)
                for s in current:
                    s['timeline'].append({'snapshot': shot, 'action': action, 'after': fingerprint, 'physical_after': tables})
            if ep.done:
                raise RuntimeError('복원 검증 기간 전에 에피소드가 끝났습니다.')
        route = ep.route
        ep.close()
        ep = None
        collision = judge_collisions(local / 'episode' / 'collisions.xml')
        if collision['collisions_at_204820'] or collision['collisions_unexplained']:
            raise RuntimeError('M4 수집 중 미확인 충돌이 발생했습니다.')
        staging = local / 'output'
        staging.mkdir()
        shutil.copy2(route, staging / route.name)
        for s in states:
            for ext in ('.xml', '.pkl'):
                shutil.copy2(stem_dir / (s['id'] + ext), staging / (s['id'] + ext))
        results = [replay_state(s, staging, errors) for s in states]
        for s, r in zip(states, results):
            if not r['ok']:   # is the restored behaviour reproducible? (second independent restoration)
                second = replay_state(s, staging, errors)
                r['failure']['restore_reproducible'] = reproducibility(r, second)
                r['failure']['second_replay'] = {'ok': second['ok'], 'kind': second.get('failure', {}).get('kind'),
                                                  'step': second.get('failure', {}).get('step')}
        for s in states:   # timelines (with raw tables) stay next to the saved state for audit and post-mortem
            w.write(staging / (s['id'] + '.timeline.json'), {'physical_before': s.pop('physical_before'), 'steps': s.pop('timeline')})
        kept = [(s, r) for s, r in zip(states, results) if r['ok']]
        dropped = [(s, r) for s, r in zip(states, results) if not r['ok']]
        report = {'scenario': scenario, 'collector_ok': not dropped, 'states': results, 'diagnostic_capture_errors': errors,
                  'tolerance_m_and_mps': TOLERANCE,
                  'note': '허용오차 이내의 수치 차이만 통과로 보고, 실제 어긋남은 제외하며 제외 개수에는 상한이 있다.'}
        w.write(report_path, report)
        excluded = [{'id': s['id'], 'window': s['window'], 'elapsed_s': s['elapsed_s'], 'kind': r['failure']['kind'],
                     'step': r['failure'].get('step'), 'restore_reproducible': r['failure'].get('restore_reproducible'),
                     'max_position_difference': ((r['failure'].get('physical') or {}).get('max_position_difference')),
                     'max_speed_difference': ((r['failure'].get('physical') or {}).get('max_speed_difference'))}
                    for s, r in dropped]
        if dropped:
            archived = out_dir / 'restore_failed' / scenario
            shutil.rmtree(archived, ignore_errors=True)
            archived.mkdir(parents=True)
            shutil.copy2(staging / route.name, archived / route.name)
            for s, r in dropped:
                for ext in ('.xml', '.pkl', '.timeline.json'):
                    shutil.copy2(staging / (s['id'] + ext), archived / (s['id'] + ext))
                    (staging / (s['id'] + ext)).unlink()
                for line in describe_failure(s['id'], r['failure']):
                    print(line, flush=True)
        for s, r in kept:
            s['restore_ok'] = True
            s['restore_horizon_s'] = 300
            s['pending_fix_check'] = r.get('pending_fix_check')
            s['tolerance_steps'] = r['tolerance_steps']
            s['max_position_difference'] = r['max_position_difference']
            s['max_speed_difference'] = r['max_speed_difference']
        kept_states = [s for s, _ in kept]
        files = [route.name] + [s['id'] + ext for s in kept_states for ext in ('.xml', '.pkl', '.timeline.json')]
        rec = {'fingerprint': job['fingerprint'], 'condition': c, 'states': kept_states, 'excluded_states': excluded,
               'files': base.digest_files(staging, files), 'elapsed_s': time.monotonic() - started,
               'note': '실제 미래는 300초 M4 복원 검증에만 사용. 제외된 상태는 복원 어긋남 때문이며 결과와 무관하다.'}
        w.write(staging / 'episode.json', rec)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.parent / ('.' + target.name + '.incomplete')
        if temporary.exists():
            shutil.rmtree(temporary)
        if target.exists():
            raise ValueError('기존 상태를 덮어쓸 수 없습니다.')
        shutil.copytree(staging, temporary)
        if base.digest_files(temporary, files) != rec['files']:
            raise ValueError('저장 상태 복사 해시 오류')
        temporary.rename(target)
        return {'ok': True, 'seconds': time.monotonic() - started, 'name': scenario, 'excluded': excluded}
    except BaseException as e:
        report = {'scenario': scenario, 'collector_ok': False, 'error': '%s: %s' % (type(e).__name__, e),
                  'collected_states': [{'id': s['id'], 'pending_snapshot_count': s['pending_snapshot_count']} for s in states],
                  'diagnostic_capture_errors': errors, 'note': '수집 단계에서 중단됨(복원 비교 이전 또는 도중의 예외).'}
        try:
            w.write(report_path, report)
        except Exception:
            pass
        return {'ok': False, 'seconds': time.monotonic() - started, 'name': scenario,
                'error': '%s: %s' % (type(e).__name__, e), 'failures': [], 'diagnostic_file': str(report_path)}
    finally:
        if ep is not None:
            ep.close()
        shutil.rmtree(local, ignore_errors=True)


def print_failure(res):
    print('collect 실패 %s: %s' % (res['name'], res.get('error')), flush=True)
    print('  진단 파일: %s' % res.get('diagnostic_file'), flush=True)


def execute_all(jobs, journal, workers):
    """Run collection jobs; a scenario that cannot be collected is recorded and the others still run."""
    pending = [j for j in jobs if base.collect_cached(j) is None]
    print('collect: 전체 %d개, 재사용 %d개, 남은 %d개' % (len(jobs), len(jobs) - len(pending), len(pending)), flush=True)
    failures = []
    if not pending:
        return failures
    pool = ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'), initializer=long._worker_init)
    active = {}
    try:
        while pending or active:
            while pending and len(active) < workers:
                job = pending.pop(0)
                token = journal.begin('collect/' + Path(job['target']).name)
                try:
                    future = pool.submit(collect_worker, job)
                except BaseException:
                    journal.finish(token, 0, 'submit_failed')
                    raise
                active[future] = (job, token, time.monotonic())
            if not active:
                break
            done, _ = wait(active, timeout=10, return_when=FIRST_COMPLETED)
            for future in done:
                job, token, start = active.pop(future)
                elapsed = time.monotonic() - start
                try:
                    res = future.result()
                except BaseException as e:
                    journal.finish(token, elapsed, 'failed')
                    raise RuntimeError('작업 프로세스 오류') from e
                journal.finish(token, max(elapsed, float(res['seconds'])), 'complete' if res['ok'] else 'failed')
                if res['ok']:
                    base.collect_cached(job)
                    note = (' (제외된 상태 %d개)' % len(res['excluded'])) if res.get('excluded') else ''
                    print('collect 완료 %s%s' % (res['name'], note), flush=True)
                else:
                    failures.append(res)
                    print_failure(res)
    except BaseException:
        long._kill_pool(pool)
        for job, token, start in active.values():
            journal.finish(token, time.monotonic() - start, 'interrupted')
        raise
    else:
        pool.shutdown(wait=True)
    return failures


def write_report(out, jobs, failures):
    failed = {f['name']: f for f in failures}
    rows, excluded_total, tolerance_states = [], [], 0
    for j in jobs:
        sid = base.scenario_id(j['condition'])
        rec = None
        if sid not in failed:
            try:
                rec = base.collect_cached(j)
            except ValueError:
                rec = None
        if sid in failed:
            f = failed[sid]
            rows.append({'scenario': sid, 'status': 'failed', 'error': f.get('error'), 'diagnostic_file': f.get('diagnostic_file')})
        elif rec is not None:
            excluded = rec.get('excluded_states') or []
            excluded_total += [dict(e, scenario=sid) for e in excluded]
            used = [s['id'] for s in rec['states'] if (s.get('pending_fix_check') or {}).get('missing_initial')]
            tol = [s['id'] for s in rec['states'] if s.get('tolerance_steps')]
            tolerance_states += len(tol)
            rows.append({'scenario': sid, 'status': 'ok', 'states': len(rec['states']), 'excluded_states': excluded,
                         'states_needing_pending_fix': used, 'states_using_tolerance': tol})
        else:
            rows.append({'scenario': sid, 'status': 'not_run'})
    summary = {'ok': sum(r['status'] == 'ok' for r in rows), 'failed': sum(r['status'] == 'failed' for r in rows),
               'not_run': sum(r['status'] == 'not_run' for r in rows), 'excluded_states': len(excluded_total),
               'excluded_limit': MAX_EXCLUDED_STATES, 'states_using_tolerance': tolerance_states}
    w.write(Path(out) / 'restore_report.json', {'summary': summary, 'excluded': excluded_total, 'scenarios': rows,
                                                 'tolerance_m_and_mps': TOLERANCE,
                                                 'note': '제외 상태는 복원 어긋남 때문이며 후보 비교 결과와 무관하다. 상한을 넘으면 후보 비교를 시작하지 않는다.'})
    return summary


def archive_earlier_diagnostics(out):
    """FIXED2's evidence (diagnostics and kept states of the three failed scenarios) is moved, never deleted."""
    marker = Path(out) / 'archive_fixed2' / 'archived.json'
    if marker.exists():
        return False
    moved = []
    for name in ('restore_diagnostics', 'restore_failed', 'restore_report.json'):
        src = Path(out) / name
        if src.exists():
            dst = Path(out) / 'archive_fixed2' / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
            moved.append(name)
    w.write(marker, {'moved': moved, 'time_unix': time.time(),
                     'note': 'FIXED2 실행 때 만든 복원 진단 기록을 FIXED3 실행 전에 옮겨 보존했다(삭제하지 않음).'})
    return bool(moved)


def register_stage(out, protocol):
    path = Path(out) / 'fixed3_stages.json'
    record = w.read(path) if path.exists() else {'stages': []}
    sha = w.file_hash(ROOT / 'src/explore_m4probe_fixed3.py')
    record['stages'].append({'module_sha256': sha, 'continuation_protocol': protocol['fingerprint'], 'started_unix': time.time(),
                             'tolerance_m_and_mps': TOLERANCE, 'max_excluded_states': MAX_EXCLUDED_STATES})
    w.write(path, record)
    return sha


def run(args):
    out, work, fixed, night, source = map(Path, (args.out, args.work, args.fixed, args.night, args.source))
    paths = [x.resolve() for x in (out, work, fixed, night, source)]
    for i, a in enumerate(paths):
        for b in paths[i + 1:]:
            if a == b or a in b.parents or b in a.parents:
                raise ValueError('출력·원본·작업 폴더가 겹칩니다.')
    out.mkdir(parents=True, exist_ok=True)
    with (out / '.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('같은 결과 폴더가 실행 중입니다. 기존 실행을 먼저 중지하세요.')
        record = base.register(out, fixed, night, source)
        protocol = rep.register_continuation(out, record)
        sha = register_stage(out, protocol)
        if archive_earlier_diagnostics(out):
            print('FIXED2의 복원 진단 기록을 archive_fixed2/로 옮겨 보존했습니다(삭제하지 않음).', flush=True)
        journal = rep.Journal(out, record['fingerprint'])
        cpus = long.available_cpus()
        workers = cpus if args.workers == 0 else min(cpus, args.workers)
        print('M4 복원 검증(FIXED3) %s 시작; 시간·계산량 제한 없음; 작업자 %d개; 이 단계 코드 %s; 허용오차 %g, 제외 상한 %d개'
              % (record['fingerprint'][:12], workers, sha[:12], TOLERANCE, MAX_EXCLUDED_STATES), flush=True)
        import focused_train as ft
        from focused_scenario import dqn_template_network
        dqn_template_network(ft.CACHE)
        jobs = base.collect_jobs(out, work, record['fingerprint'])
        failures = execute_all(jobs, journal, workers)
        summary = write_report(out, jobs, failures)
        print('복원 검증 요약: 통과 %d, 수집 실패 %d, 미실행 %d, 제외된 상태 %d개(상한 %d), 허용오차를 쓴 상태 %d개 → %s'
              % (summary['ok'], summary['failed'], summary['not_run'], summary['excluded_states'], MAX_EXCLUDED_STATES,
                 summary['states_using_tolerance'], out / 'restore_report.json'), flush=True)
        if failures:
            raise RestoreFailures(failures)
        if summary['excluded_states'] > MAX_EXCLUDED_STATES:
            raise RestoreFailures([], '제외된 상태가 %d개로 상한 %d개를 넘었습니다. 복원이 믿을 만하지 않아 후보 비교를 시작하지 않습니다.'
                                  % (summary['excluded_states'], MAX_EXCLUDED_STATES))
    print('복원 검증을 마쳤습니다. 이어서 후보 비교(롤아웃)를 시작합니다.', flush=True)
    rep.run(args)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('out', 'work', 'fixed', 'night', 'source'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--workers', type=int, default=0, help='0: 사용 가능한 CPU 자동 설정')
    args = p.parse_args()
    if sys.platform == 'darwin':
        raise SystemExit('실제 진단은 Colab에서만 합니다.')
    if args.workers < 0:
        raise ValueError('작업자 수는 0 이상이어야 합니다.')

    def stop(signum, frame):
        raise KeyboardInterrupt('중단 요청')
    signal.signal(signal.SIGTERM, stop)
    run(args)


if __name__ == '__main__':
    main()
