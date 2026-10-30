"""M4 selection-point pilot, stage 1 with full restore diagnostics (FIXED2).

Nothing scientific changes: same 16 scenarios, same first-disagreement states, same candidates, futures and
gate, and the same strict pass/fail as ``explore_m4probe_pending`` (identical 110-cell observation, mode
actions, mask, executed action and vehicle/signal/pending fingerprint at every 5-second decision for 300 s).
Only the reporting and the stopping rule differ:

* the original raw vehicle / pending / signal tables are kept for the 300 s after every saved state, so a mismatch
  is explained (which observation cells, which vehicles, which pending lists) instead of one error line;
* every state of a scenario is replayed even if an earlier one failed, and every scenario is processed even if
  another one failed, so a single Colab run shows all restoration problems;
* a failed scenario's saved states and timelines are kept under ``restore_failed/`` for post-mortem;
* the run stops before any rollout if any state failed; nothing is dropped or relaxed.

When every state passes, the unchanged continuation (``explore_m4probe_repaired.run``) takes over and runs the
rollouts, so the rollout and gate code are exactly the ones already reviewed.
"""
from __future__ import annotations
import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import fcntl
import multiprocessing
import os
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
FAILED_CASE = '20220812_0730_s43'
REPLAY_STEPS = 60
OBS_GROUPS = ((0, 64, '기본 관측'), (64, 72, '이동별 대기행렬'), (72, 80, '이동별 최장 대기'), (80, 84, '후보 비용 차이'),
              (84, 104, '모드별 행동 원-핫'), (104, 109, '모드 마스크'), (109, 110, '남은 시간'))


class RestoreFailures(RuntimeError):
    def __init__(self, failures):
        self.failures = failures
        super().__init__('복원 검사 실패 상황 %d개: %s' % (len(failures), ', '.join(f['name'] for f in failures)))


def group_of(index):
    return next((name for lo, hi, name in OBS_GROUPS if lo <= index < hi), '알 수 없음')


def raw_tables(sumo, errors):
    """Raw vehicle/pending/signal table used only to explain a mismatch; never changes pass/fail."""
    try:
        return diag.physical(sumo)
    except Exception as e:
        errors.append('%s: %s' % (type(e).__name__, e))
        return None


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
    """Strict 300 s replay of one saved state; returns the verdict plus, on failure, what differed."""
    from explore_weight_core import ModePlanner
    out = {'state': s['id'], 'ok': False, 'saved_pending': s['pending_snapshot_count']}
    env = None
    try:
        env, cons, ts, ctl, saved, _, _ = repair.restore(staging / s['id'], staging, 'real', horizon=300)
        out['pending_fix_check'] = saved.get('pending_fix_check')
        if er.trace_fingerprint(env.sumo) != s['trace_before']:
            out['failure'] = {'kind': 'right_after_restore', 'step': 0,
                              'physical': diag.compare_physical(s.get('physical_before'), raw_tables(env.sumo, errors))}
            return out
        check = ModePlanner(ctl.movement_of, [p.state for p in ts.green_phases], saved['handover'] + 3600)
        if len(s['timeline']) != REPLAY_STEPS:
            out['failure'] = {'kind': 'timeline_incomplete', 'length': len(s['timeline'])}
            return out
        for step, ref in enumerate(s['timeline']):
            actual = check.snapshot(env)
            if not repair.same_snapshot(actual, ref['snapshot']):
                before = s.get('physical_before') if step == 0 else s['timeline'][step - 1].get('physical_after')
                out['failure'] = {'kind': 'observation_or_mode', 'step': step,
                                  'snapshot': explain_snapshot(ref['snapshot'], actual),
                                  'physical': diag.compare_physical(before, raw_tables(env.sumo, errors))}
                return out
            action = int(actual.mode_actions[4])
            corrected = int(cons(env, action))
            if action != ref['action'] or corrected != action:
                out['failure'] = {'kind': 'action', 'step': step, 'original_action': ref['action'],
                                  'restored_action': action, 'restored_corrected': corrected}
                return out
            env.step(corrected)
            if er.trace_fingerprint(env.sumo) != ref['after']:
                out['failure'] = {'kind': 'state_after_step', 'step': step + 1,
                                  'physical': diag.compare_physical(ref.get('physical_after'), raw_tables(env.sumo, errors))}
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
        # Timelines (with raw tables) are kept next to the saved state for audit and post-mortem.
        for s in states:
            w.write(staging / (s['id'] + '.timeline.json'), {'physical_before': s.pop('physical_before'), 'steps': s.pop('timeline')})
        failed = [r for r in results if not r['ok']]
        report = {'scenario': scenario, 'collector_ok': not failed, 'states': results, 'diagnostic_capture_errors': errors,
                  'note': '진단만 기록하며 저장/복원 통과 기준은 완화하지 않는다.'}
        w.write(report_path, report)
        if failed:
            kept = out_dir / 'restore_failed' / scenario
            shutil.rmtree(kept, ignore_errors=True)
            shutil.copytree(staging, kept)
            for r in failed:
                for line in describe_failure(r['state'], r['failure']):
                    print(line, flush=True)
            return {'ok': False, 'seconds': time.monotonic() - started, 'name': scenario,
                    'error': '저장 상태 복원이 원래 M4와 다릅니다: ' + ', '.join(r['state'] for r in failed),
                    'failures': [{'state': r['state'], 'kind': r['failure']['kind'], 'step': r['failure'].get('step')} for r in failed],
                    'diagnostic_file': str(report_path), 'kept_states': str(kept)}
        for s, r in zip(states, results):
            s['restore_ok'] = True
            s['restore_horizon_s'] = 300
            s['pending_fix_check'] = r.get('pending_fix_check')
        files = [route.name] + [s['id'] + ext for s in states for ext in ('.xml', '.pkl', '.timeline.json')]
        rec = {'fingerprint': job['fingerprint'], 'condition': c, 'states': states, 'files': base.digest_files(staging, files),
               'elapsed_s': time.monotonic() - started, 'note': '실제 미래는 300초 M4 복원 검증에만 사용'}
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
        return {'ok': True, 'seconds': time.monotonic() - started, 'name': scenario}
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
    if res.get('kept_states'):
        print('  저장 상태 보존: %s' % res['kept_states'], flush=True)


def execute_all(jobs, journal, workers):
    """Run collection jobs; a restore failure is recorded and the other jobs still run."""
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
                    print('collect 완료 %s' % res['name'], flush=True)
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
    rows = []
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
            rows.append({'scenario': sid, 'status': 'failed', 'error': f.get('error'), 'failures': f.get('failures'),
                         'diagnostic_file': f.get('diagnostic_file')})
        elif rec is not None:
            shim = [s['id'] for s in rec['states'] if (s.get('pending_fix_check') or {}).get('missing_initial')]
            rows.append({'scenario': sid, 'status': 'ok', 'states': len(rec['states']), 'states_needing_pending_fix': shim})
        else:
            rows.append({'scenario': sid, 'status': 'not_run'})
    summary = {'ok': sum(r['status'] == 'ok' for r in rows), 'failed': sum(r['status'] == 'failed' for r in rows),
               'not_run': sum(r['status'] == 'not_run' for r in rows)}
    w.write(Path(out) / 'restore_report.json', {'summary': summary, 'scenarios': rows,
                                                 'note': '엄격한 통과 기준은 그대로이며, 하나라도 실패하면 후보 비교를 시작하지 않는다.'})
    return summary


def register_stage(out, protocol):
    path = Path(out) / 'fixed2_stages.json'
    record = w.read(path) if path.exists() else {'stages': []}
    sha = w.file_hash(ROOT / 'src/explore_m4probe_fixed2.py')
    record['stages'].append({'module_sha256': sha, 'continuation_protocol': protocol['fingerprint'], 'started_unix': time.time()})
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
        journal = rep.Journal(out, record['fingerprint'])
        cpus = long.available_cpus()
        workers = cpus if args.workers == 0 else min(cpus, args.workers)
        print('M4 복원 검증(FIXED2) %s 시작; 시간·계산량 제한 없음; 작업자 %d개; 이 단계 코드 %s' % (record['fingerprint'][:12], workers, sha[:12]), flush=True)
        import focused_train as ft
        from focused_scenario import dqn_template_network
        dqn_template_network(ft.CACHE)
        jobs = base.collect_jobs(out, work, record['fingerprint'])
        failed_case = next(j for j in jobs if base.scenario_id(j['condition']) == FAILED_CASE)
        print('이전에 복원 검사가 실패한 상황(20220812 07:30 난수 43)을 먼저 검증합니다.', flush=True)
        failures = execute_all([failed_case], journal, 1)
        if failures:
            print('실패했던 상황이 다시 실패했습니다. 원인을 기록하고 나머지 상황도 모두 검사합니다.', flush=True)
            rest = [j for j in jobs if j is not failed_case]
        else:
            print('실패했던 상황의 300초 복원 검증 통과', flush=True)
            rest = jobs
        failures = failures + execute_all(rest, journal, workers)
        summary = write_report(out, jobs, failures)
        print('복원 검증 요약: 통과 %d, 실패 %d, 미실행 %d → %s' % (summary['ok'], summary['failed'], summary['not_run'], out / 'restore_report.json'), flush=True)
        if failures:
            raise RestoreFailures(failures)
    print('모든 상태의 300초 복원 검증을 통과했습니다. 이어서 후보 비교(롤아웃)를 시작합니다.', flush=True)
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
