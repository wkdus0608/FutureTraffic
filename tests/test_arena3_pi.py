"""ARENA3 정책 반복(PI) 검사 — 설정·녹색 마스크·수집기·롤아웃 이어 달리기·자료·학생·평가 연결(가짜 SUMO, 실제 시뮬레이터 없음)."""
import copy
import gzip
import json
import sys
import tempfile
import unittest
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import arena3_bprime as bp  # noqa: E402
import arena3_common as ac  # noqa: E402
import arena3_g1 as g1  # noqa: E402
import arena3_pi as pi  # noqa: E402
import arena3_pi_learn as pl  # noqa: E402
import explore_weight as w  # noqa: E402

CFG = pi.config("R1")
TABLE = {"time": 1.0, "tls": "GGrr", "vehicles": [], "pending": []}


def make_obs(greens6, allowed, rng=None):
    """관측 119칸 모양(검사용): 허용 녹색(60:64), 모드 0~4 첫 녹색(84:104)·유효(104:109), W4(110:115). 나머지는 0 또는 난수."""
    obs = np.zeros(119, np.float32)
    if rng is not None:
        obs[:60] = rng.random(60)
        obs[64:84] = rng.random(20)
        obs[109] = rng.random()
    obs[60:64] = [float(g in allowed) for g in range(4)]
    mask = ac.canonical_mask6(list(greens6))
    for m, g in enumerate(greens6[:5]):
        obs[84 + 4 * m + g] = 1.0
    obs[104:109] = mask[:5]
    obs[110 + greens6[5]] = 1.0
    obs[114] = mask[5]
    return obs


def snapshot(greens6, allowed, now=0.0, rng=None):
    greens6 = list(greens6)
    return ac.ArenaSnapshot(base=SimpleNamespace(allowed=list(allowed), p0_action=int(allowed[0]), now=float(now)),
                            greens=greens6, mask=ac.canonical_mask6(greens6), choice=len(set(greens6)) > 1,
                            observation=make_obs(greens6, allowed, rng))


CHOICE = ([0, 0, 0, 0, 1, 1], [0, 1, 2, 3])        # 갈림: 모드 녹색 {0, 1}, M4 = 1, 모드 밖 녹색 {2, 3}
AGREE = ([1] * 6, [1, 3])                          # 합의: 모두 1, 허용 {1, 3}
FORCED = ([2] * 6, [2])                            # 강제


def biased_net(prefer: int, amount: float = 1.0) -> pi.GreenNet:
    net = pi.GreenNet()
    with torch.no_grad():
        net.advantage.bias[prefer] = amount
    return net


class ConfigTest(unittest.TestCase):
    def test_situations_are_new_training_placements(self):
        lock = ac.config()
        sits = CFG["situations_data"]
        self.assertEqual(len(sits), 64)
        self.assertEqual({s["date"] for s in sits}, set(lock["dates"]["train"]))
        for d in lock["dates"]["train"]:
            self.assertEqual(sum(s["date"] == d for s in sits), 8)
        self.assertEqual([s["tag"] for s in sits], [f"R1T{i}" for i in range(1, 65)])
        self.assertTrue(all(s["sumo_seed"] == ac.sumo_seed(s["tag"], s["date"], s["period"]) for s in sits))
        self.assertTrue(all(s["period"] in ac.TRAIN_SLOTS for s in sits))
        old = {x["route_sha256"] for p in ("eval", "confirmation", "bprime_data") for x in lock[f"situations_{p}"]}
        self.assertFalse({s["route_sha256"] for s in sits} & old)
        self.assertEqual(pi.data_situations(lock, CFG["situations"])[5]["period"], sits[5]["period"])

    def test_route_files_reproduce_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            for s in CFG["situations_data"][:2]:
                self.assertEqual(ac.route_file_sha(s, Path(tmp)), s["route_sha256"])

    def test_base_policy_and_constants(self):
        b = CFG["base_policy"]
        self.assertEqual((b["name"], b["kind"], b["seed"], b["stage"], b["stage2_variant"]),
                         ("S2A_pre_42", "mode", 42, "pretrained", "advantage"))
        broken = copy.deepcopy(CFG)
        broken["rollout"]["terminal_s"] = 60.0
        with self.assertRaises(ValueError):
            pi.check_constants(broken)

    def test_agree_sample_rule_is_deterministic_and_near_fraction(self):
        picks = [pi.agree_sampled(CFG, "20220812_0730_R1T1", k) for k in range(4000)]
        self.assertEqual(picks, [pi.agree_sampled(CFG, "20220812_0730_R1T1", k) for k in range(4000)])
        self.assertAlmostEqual(sum(picks) / len(picks), CFG["agree_sample"]["fraction"], delta=0.03)
        self.assertNotEqual(pi.future_seed(CFG, "x_p000", 0), bp.future_seed("x_p000", 0))


class MaskAndPolicyTest(unittest.TestCase):
    def test_masks_by_variant(self):
        rows = {v: [pi.action_mask(make_obs(*case), v).tolist() for case in (CHOICE, AGREE, FORCED)] for v in pi.VARIANTS}
        self.assertEqual(rows["M"], [[True, True, False, False], [False, True, False, False], [False, False, True, False]])
        self.assertEqual(rows["G"], [[True, True, True, True], [False, True, False, False], [False, False, True, False]])
        self.assertEqual(rows["A"], [[True, True, True, True], [False, True, False, True], [False, False, True, False]])

    def test_untrained_student_is_m4_everywhere(self):
        net = pi.GreenNet()
        rng = np.random.default_rng(0)
        for case in (CHOICE, AGREE, FORCED):
            for v in pi.VARIANTS:
                self.assertEqual(pi.greedy_green(net, make_obs(*case, rng), v), case[0][ac.M4])

    def test_greedy_respects_variant(self):
        net = biased_net(2)                                 # 모드 밖 녹색 2를 가장 좋게 보는 신경망
        self.assertEqual(pi.greedy_green(net, make_obs(*CHOICE), "M"), 1)   # 모드 녹색 {0, 1}: 둘 다 0 → M4
        self.assertEqual(pi.greedy_green(net, make_obs(*CHOICE), "G"), 2)
        net3 = biased_net(3)
        self.assertEqual(pi.greedy_green(net3, make_obs(*AGREE), "G"), 1)
        self.assertEqual(pi.greedy_green(net3, make_obs(*AGREE), "A"), 3)
        obs = np.stack([make_obs(*c) for c in (CHOICE, AGREE, FORCED)])
        for v in pi.VARIANTS:
            batch = pl.predict(net3, obs, pi.action_mask(obs, v))
            self.assertEqual(batch.tolist(), [pi.greedy_green(net3, o, v) for o in obs])

    def test_mode_base_matches_stage2_referee_policy(self):
        torch.manual_seed(1)
        net = g1.DuelingNet()
        with torch.no_grad():
            net.advantage.weight.normal_(0, 0.5)
        base, referee = pi.ModeBase(net), g1.GreedyPolicy(net)
        rng = np.random.default_rng(2)
        for _ in range(200):
            greens = [int(x) for x in rng.integers(0, 4, size=6)]
            allowed = sorted(set(greens) | {int(rng.integers(0, 4))})
            snap = snapshot(greens, allowed, rng=rng)
            expected = referee.decide(snap, None) if snap.choice else greens[0]
            self.assertEqual(base.act(snap), expected)

    def test_green_policy_is_called_every_step_and_leaves_p0_check(self):
        policy = pi.GreenPolicy(biased_net(3), "A", "R1A_m0")
        self.assertIsInstance(policy, ac.P0Policy)          # 실행기가 모든 제어 단계에서 부르는 종류
        ctx = SimpleNamespace(mismatch=0)
        self.assertEqual(policy.decide(snapshot(*AGREE), ctx), 3)
        self.assertEqual(policy.decide(snapshot(*FORCED), ctx), 2)
        self.assertEqual(ctx.mismatch, 0)
        self.assertEqual(policy.name, "R1A_m0")

    def test_payload_round_trip(self):
        net = biased_net(2)
        payload = {"kind": "green", "variant": "G", "name": "x", "sha256": "s", "state": pi._state_bytes(net)}
        base = pi.base_from_payload(payload)
        self.assertEqual(base.act(snapshot(*CHOICE)), 2)


class FakePool:
    """submit를 바로 계산해 끝난 Future를 돌려준다. 롤아웃 비용 = 후보·미래로 정한 가짜 값."""

    def __init__(self, cost=None, verify_ok=True, fail=None):
        self.cost = cost or (lambda job: 100.0 + 10 * job["candidate"] - job["index"])
        self.verify_ok, self.fail, self.calls = verify_ok, fail, []

    def submit(self, fn, job):
        self.calls.append((fn, job))
        f = Future()
        if fn is pi.rollout_worker:
            if self.fail and self.fail(job):
                f.set_result({"ok": False, "error": "가짜 실패", "seconds": 1.0})
            else:
                f.set_result({"ok": True, "cost": self.cost(job), "delay": 1.0, "residual": 0, "future_seed": job["future_seed"],
                              "leaks": 0, "changed_by_rules": 0, "collisions_unexplained": 0, "collisions_at_204820": 0,
                              "candidate": job["candidate"], "future": job["index"], "seconds": 2.0})
        elif fn is bp.verify_worker:
            f.set_result({"ok": self.verify_ok, "table": TABLE, "seconds": 0.5})
        else:
            raise AssertionError(f"알 수 없는 작업 함수: {fn}")
        return f

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def fake_saving(test):
    import explore_m4probe_pending as pend
    import explore_rollout as er

    def save(ep, stem, allowed, p0):
        stem.parent.mkdir(parents=True, exist_ok=True)
        stem.with_suffix(".xml").write_text("<state/>")
        return {"t0": 0.0}
    for p in (patch.object(er.Episode, "snapshot", side_effect=save), patch.object(pend, "capture_pending", return_value={}),
              patch.object(er, "types_only_file", return_value=None), patch.object(bp, "trace_table", return_value=TABLE)):
        p.start()
        test.addCleanup(p.stop)


def drive_collector(policy, snaps, work):
    """실행기 흉내: P0 종류라 모든 단계에서 decide → 실행 직후 after_step."""
    ep = SimpleNamespace(env=SimpleNamespace(sumo=None), route=Path(work) / "r.rou.xml")
    actions = []
    for step, snap in enumerate(snaps):
        ctx = SimpleNamespace(ep=ep, step=step, mismatch=0)
        a = policy.decide(snap, ctx)
        actions.append(a)
        policy.after_step(ctx, snap, a, a)
    return actions


def episode_snaps(n=40):
    return [snapshot(*(CHOICE if k % 5 == 0 else AGREE if k % 5 in (1, 2) else FORCED)) for k in range(n)]


class CollectorTest(unittest.TestCase):
    def setUp(self):
        fake_saving(self)
        self.cfg = copy.deepcopy(CFG)
        self.cfg["rollout"]["futures"] = 4

    def test_actions_are_base_choices_and_jobs_cover_allowed_greens(self):
        with tempfile.TemporaryDirectory() as tmp:
            pool = FakePool()
            base = pi.GreenBase(biased_net(2), "G")
            s = self.cfg["situations_data"][0]
            policy = pi.Collector(base, pool, self.cfg, s, Path(tmp) / "states")
            snaps = episode_snaps()
            actions = drive_collector(policy, snaps, tmp)
            self.assertEqual(actions, [base.act(x) for x in snaps])
            sid = ac.situation_id(s)
            agree_steps = [k for k in range(40) if k % 5 in (1, 2) and pi.agree_sampled(self.cfg, sid, k)]
            self.assertEqual([p["step"] for p in policy.points], sorted([k for k in range(0, 40, 5)] + agree_steps))
            self.assertEqual(policy.counts, {"forced": 16, "choice": 8, "agree": 16, "agree_sampled": len(agree_steps)})
            rollouts = [j for fn, j in pool.calls if fn is pi.rollout_worker]
            self.assertEqual(len(rollouts), sum(len(p["allowed"]) for p in policy.points) * 4)
            self.assertEqual(len([1 for fn, _ in pool.calls if fn is bp.verify_worker]), len(policy.points))
            first = [j for j in rollouts if j["stem"].endswith("_p000")]
            self.assertEqual(sorted({j["candidate"] for j in first}), [0, 1, 2, 3])
            self.assertEqual(len({j["future_seed"] for j in first}), 4)        # 후보끼리 같은 미래 4개
            self.assertTrue(all(p["executed"] == base.act(snaps[p["step"]]) for p in policy.points))
            rc = policy.finish()
            self.assertEqual((rc["checked"], rc["mismatched"]), (len(policy.points), 0))
            self.assertTrue(all(p["restore_check"]["ok"] and p["valid_futures"] == 4 for p in policy.points))
            self.assertEqual(policy.points[0]["rollouts"]["2"][1]["cost"], 100.0 + 20 - 1)
            self.assertFalse(list((Path(tmp) / "states").glob("*.xml")))       # 다 쓴 상태 파일은 지운다

    def test_limit_and_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            policy = pi.Collector(pi.ModeBase(g1.DuelingNet()), FakePool(fail=lambda j: j["candidate"] == 3), self.cfg,
                                  self.cfg["situations_data"][0], Path(tmp) / "s", limit={"choice": 2, "agree": 1})
            drive_collector(policy, episode_snaps(), tmp)
            self.assertEqual(sum(p["kind"] == "choice" for p in policy.points), 2)
            self.assertLessEqual(sum(p["kind"] == "agree" for p in policy.points), 1)
            with self.assertRaises(RuntimeError):
                policy.finish()                                   # 실패 12개 > 허용 3개

    def test_few_failed_rollouts_leave_holes_and_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            once = lambda j: j["candidate"] == 2 and j["index"] == 1 and j["stem"].endswith("_p000")  # noqa: E731
            policy = pi.Collector(pi.ModeBase(g1.DuelingNet()), FakePool(fail=once), self.cfg,
                                  self.cfg["situations_data"][0], Path(tmp) / "s", limit={"choice": 2, "agree": 0})
            drive_collector(policy, episode_snaps(), tmp)
            rc = policy.finish()
            self.assertEqual((rc["rollout_failures"], rc["rollouts"]), (1, 2 * 4 * 4))
            first = policy.points[0]
            self.assertIsNone(first["rollouts"]["2"][1])
            self.assertEqual(first["rollout_errors"][0]["green"], 2)
            self.assertEqual((first["valid_futures"], policy.points[1]["valid_futures"]), (3, 4))

    def test_restore_mismatch_and_collision_futures(self):
        with tempfile.TemporaryDirectory() as tmp:
            policy = pi.Collector(pi.ModeBase(g1.DuelingNet()), FakePool(verify_ok=False), self.cfg,
                                  self.cfg["situations_data"][0], Path(tmp) / "s", limit={"choice": 1, "agree": 0})
            drive_collector(policy, episode_snaps(6), tmp)
            rc = policy.finish()
            self.assertEqual(rc["mismatched"], 1)
            point = policy.points[0]
            point["rollouts"]["2"][0]["collisions_unexplained"] = 1
            self.assertEqual(pi.valid_futures(point), [1, 2, 3])


class RolloutContinuationTest(unittest.TestCase):
    """rollout_worker: 첫 5초 후보 → 지금 정책 → 제어 60분 뒤 P0(가짜 환경으로 이어 달리기만 본다)."""

    def test_follows_base_then_p0_after_control_end(self):
        import focused_eval
        executed = []

        class Env:
            sim_step = 3500.0

            def _sumo_step(self):
                pass

            def step(self, a):
                for _ in range(5):
                    self._sumo_step()
                executed.append((self.sim_step, int(a)))
                self.sim_step += 5.0

            def close(self):
                pass

        env = Env()
        cons = type("Cons", (), {"__call__": lambda self, e, a: a, "allowed": lambda self, e: [0, 1, 2, 3]})()
        ctl = type("P0", (), {"__call__": lambda self, e, allowed: 3, "movement_of": {}})()
        ts = SimpleNamespace(green_phases=[SimpleNamespace(state="G")] * 4)
        snap0 = {"t0": 3500.0, "handover": 0.0}

        class Planner:
            def __init__(self, *args):
                pass

            def arena_snapshot(self, e):
                return snapshot(*CHOICE, now=e.sim_step)

        class Meter:
            def __init__(self, *args, marks=()):
                self.at, self.leaks = {}, set()

            def step(self):
                pass

            def finish(self, trip):
                self.at[300] = {"delay": 5.0, "cost": 7.0, "residual": 0}

        class Base:
            def act(self, snap):
                return 2
        job = {"stem": "x", "route_dir": "y", "sumo_seed": 1, "horizon_s": 300, "future_seed": 9, "index": 0, "candidate": 1,
               "greens": CHOICE[0], "allowed": CHOICE[1], "observation": make_obs(*CHOICE).tolist()}
        with patch.object(bp, "_restore_env", return_value=(env, cons, ts, ctl, snap0, [], {})), \
                patch.object(ac, "_planner_class", return_value=Planner), \
                patch.object(bp, "_meter_class", return_value=Meter), \
                patch.object(focused_eval, "judge_collisions", return_value={"collisions_unexplained": [], "collisions_at_204820": 0}), \
                patch.object(pi, "_BASE", Base()):
            r = pi.rollout_worker(job)
        self.assertTrue(r["ok"], r)
        self.assertEqual(r["cost"], 7.0)
        self.assertEqual(executed[0], (3500.0, 1))                                   # 첫 5초 = 후보
        self.assertTrue(all(a == 2 for t, a in executed[1:] if t < 3600))            # 제어 안 = 지금 정책
        self.assertTrue(all(a == 3 for t, a in executed if t >= 3600))               # 제어 뒤 = P0
        self.assertEqual(executed[-1][0], 3795.0)

    def test_requires_pool_initializer(self):
        with patch.object(pi, "_BASE", None):
            r = pi.rollout_worker({"candidate": 0, "index": 0})
        self.assertFalse(r["ok"])
        self.assertIn("지금 정책이 없습니다", r["error"])


def fake_point(state, kind, greens, allowed, costs, *, executed=None, ok=True, futures=4, seeds=None):
    """costs[g][j] = 비용(또는 None)."""
    obs = make_obs(greens, allowed)
    rollouts = {str(g): [None if costs[g][j] is None else
                         {"cost": costs[g][j], "future_seed": (seeds or {}).get(j, 1000 + j), "collisions_unexplained": 0,
                          "collisions_at_204820": 0} for j in range(futures)] for g in allowed}
    return {"state": state, "kind": kind, "step": 3, "allowed": list(allowed), "greens": list(greens), "m4": greens[ac.M4],
            "executed": greens[ac.M4] if executed is None else executed, "observation": obs.tolist(), "rollouts": rollouts,
            "restore_check": {"ok": ok}}


class DatasetRowTest(unittest.TestCase):
    def setUp(self):
        self.cfg = copy.deepcopy(CFG)
        self.cfg["rollout"]["futures"] = 4

    def test_gains_are_relative_to_m4_green_with_fit_score_halves(self):
        costs = {0: [10, 12, 14, 16], 1: [11, 11, 11, 11], 2: [5, 9, 13, 21], 3: [20, 20, 20, 20]}
        row, why = pl.point_to_row(fake_point("a", "choice", *CHOICE, costs), "20220812", "s", self.cfg)
        self.assertIsNone(why)
        m4 = [11, 11, 11, 11]
        for g, c in costs.items():
            diffs = [(a - b) / 1000 for a, b in zip(m4, c)]
            self.assertAlmostEqual(float(row["gains"][g]), sum(diffs) / 4, places=6)
            self.assertAlmostEqual(float(row["fit_gains"][g]), sum(diffs[:2]) / 2, places=6)
            self.assertAlmostEqual(float(row["score_gains"][g]), sum(diffs[2:]) / 2, places=6)
        self.assertEqual(float(row["gains"][1]), 0.0)
        self.assertEqual(row["proposed"].tolist(), [True, True, False, False])
        self.assertEqual(row["kinds"], "choice")

    def test_exclusions_and_errors(self):
        costs = {1: [1, 2, 3, 4], 3: [2, 2, 2, 2]}
        _, why = pl.point_to_row(fake_point("b", "agree", *AGREE, costs, ok=False), "d", "s", self.cfg)
        self.assertEqual(why, "restore_mismatch")
        dropped = copy.deepcopy(costs)
        dropped[3][0] = None
        _, why = pl.point_to_row(fake_point("c", "agree", *AGREE, dropped), "d", "s", self.cfg)
        self.assertEqual(why, "insufficient_valid_futures")          # 4개 중 하나가 빠지면 최소 4개 미달
        with self.assertRaises(ValueError):
            pl.point_to_row(fake_point("d", "choice", *AGREE, costs), "d", "s", self.cfg)    # 구분이 관측과 다름
        other = fake_point("e", "agree", *AGREE, costs)
        other["rollouts"]["3"][2]["future_seed"] = 77                  # 한 후보만 다른 미래
        with self.assertRaises(ValueError):
            pl.point_to_row(other, "d", "s", self.cfg)
        bad = fake_point("f", "agree", *AGREE, costs)
        bad["allowed"] = [1, 2]
        with self.assertRaises(ValueError):
            pl.point_to_row(bad, "d", "s", self.cfg)


def synthetic_dataset(cfg, n_per_date=60, dates=("20220812", "20220822", "20220907"), seed=0):
    """갈림: 녹색 2(모드 밖)가 obs[0] > 0.5일 때 +0.5, 아니면 −0.5. 합의: 녹색 3이 obs[1] > 0.5일 때 +0.4, 아니면 −0.4."""
    rng = np.random.default_rng(seed)
    rows = []
    for d in dates:
        for k in range(n_per_date):
            kind = "choice" if k % 2 == 0 else "agree"
            greens, allowed = (CHOICE if kind == "choice" else AGREE)
            obs = make_obs(greens, allowed, rng)
            gains = np.zeros(4, np.float32)
            if kind == "choice":
                gains[[0, 2, 3]] = (-0.2, 0.5 if obs[0] > 0.5 else -0.5, -1.0)
            else:
                gains[3] = 0.4 if obs[1] > 0.5 else -0.4
            noise = rng.normal(0, 0.02, size=4).astype(np.float32) * pi.allowed_mask(obs)
            noise[greens[ac.M4]] = 0.0
            rows.append({"observations": obs, "allowed": pi.allowed_mask(obs),
                         "proposed": np.eye(4, dtype=bool)[np.asarray(greens)].any(axis=0), "m4": greens[ac.M4],
                         "executed": greens[ac.M4], "kinds": kind, "gains": gains, "gain_se": np.full(4, 0.05, np.float32),
                         "fit_gains": gains + noise, "fit_gain_se": np.full(4, 0.07, np.float32), "score_gains": gains - noise,
                         "dates": d, "state_ids": f"{d}_{k}", "situations": f"{d}_s{k % 3}", "valid_futures": 8})
    ds = {k: np.asarray([r[k] for r in rows]) for k in pl.ARRAY_KEYS}
    ds["masks"] = {v: pi.action_mask(ds["observations"], v) for v in pi.VARIANTS}
    ds["fingerprint"], ds["manifest"] = "synthetic", {"kind": "synthetic"}
    return ds


class StudentTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.cfg = copy.deepcopy(CFG)
        self.cfg["student"]["epochs"] = 60

    def test_student_learns_when_to_leave_m4_and_masks_hold(self):
        ds = synthetic_dataset(self.cfg)
        net, stats = pl.train_student(ds, self.cfg, 42)
        self.assertLess(stats["final_full_loss"], stats["initial_full_loss"])
        obs = ds["observations"]
        choice, agree = ds["kinds"] == "choice", ds["kinds"] == "agree"
        g = pl.predict(net, obs, ds["masks"]["G"])
        a = pl.predict(net, obs, ds["masks"]["A"])
        m = pl.predict(net, obs, ds["masks"]["M"])
        want_g = np.where(obs[:, 0] > 0.5, 2, 1)
        self.assertGreater(np.mean(g[choice] == want_g[choice]), 0.85)
        self.assertTrue(np.all(g[agree] == 1) and np.all(m[agree] == 1))
        self.assertTrue(set(m[choice].tolist()) <= {0, 1})
        want_a = np.where(obs[:, 1] > 0.5, 3, 1)
        self.assertGreater(np.mean(a[agree] == want_a[agree]), 0.85)
        again, _ = pl.train_student(ds, self.cfg, 42)
        for x, y in zip(net.state_dict().values(), again.state_dict().values()):
            self.assertTrue(torch.equal(x, y))

    def test_score_and_offline_report(self):
        ds = synthetic_dataset(self.cfg, n_per_date=40)
        s = pl.score(ds, ds["m4"], self.cfg)
        self.assertEqual(s["choice"]["mean"], 0.0)
        best = pl.teacher_greens(ds, ds["masks"]["A"], "score_gains")
        self.assertGreater(pl.score(ds, best, self.cfg)["per_hour_rough"], 0)
        with tempfile.TemporaryDirectory() as tmp:
            self.cfg["student"]["epochs"] = 5
            report = pl.offline_report(ds, self.cfg, Path(tmp))
            self.assertIn("학생 G(mean)", report["methods"])
            self.assertTrue((Path(tmp) / "offline_report.md").exists())
            self.assertEqual(len(report["folds"]), 3 * len(self.cfg["offline"]["targets"]))

    def test_save_load_identity_and_tamper(self):
        net = biased_net(2, 0.3)
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "seed_42"
            meta = {"round": "R1", "seed": 42, "stage": "final", "dataset_fingerprint": "fp", "network": [64, 64],
                    "target": "mean"}
            sha = pl.save_student(folder, net, meta)
            self.assertEqual(pl.student_identity(folder), sha)
            loaded = pl.load_student(folder)
            self.assertEqual(pi.greedy_green(loaded, make_obs(*CHOICE), "G"), 2)
            with open(folder / "model.pt", "ab") as f:
                f.write(b"x")
            with self.assertRaises(ValueError):
                pl.student_identity(folder)


def make_students(root, cfg, dataset_fp="fp", seeds=None):
    for seed in seeds or cfg["student"]["seeds"]:
        pl.save_student(Path(root) / f"seed_{seed}", biased_net(2, 0.1 * seed),
                        {"round": cfg["round"], "seed": seed, "stage": "final", "dataset_fingerprint": dataset_fp,
                         "network": list(cfg["student"]["network"]), "target": cfg["student"]["target"]})
    return pl.student_dirs(root, cfg)


class EvaluationTest(unittest.TestCase):
    def test_declaration_and_jobs(self):
        with tempfile.TemporaryDirectory() as tmp:
            dirs = make_students(Path(tmp) / "students", CFG)
            decl = pl.declaration("R1G", dirs, "eval", CFG)
            self.assertEqual((decl["variant"], decl["seeds"]), ("G", [42, 43, 44]))
            with self.assertRaises(ValueError):
                pl.declaration("R1G", list(reversed(dirs)), "eval", CFG)
            with self.assertRaises(ValueError):
                pl.declaration("P2", dirs, "eval", CFG)
            captured = {}

            def fake_run_jobs(jobs, workers, ledger, worker=None, executor=None):
                captured["jobs"] = jobs
                return {"done": len(jobs), "failed": []}
            with patch.object(pl.ref, "register", return_value="fake-fp"), \
                    patch.object(pl.ref, "run_jobs", side_effect=fake_run_jobs):
                pl.evaluate("R1A", dirs, Path(tmp) / "eval", Path(tmp) / "w", 2, CFG, worker=lambda job: None)
            jobs = captured["jobs"]
            self.assertEqual(len(jobs), 3 * 84)
            self.assertEqual({(j["kind"], j["variant"]) for j in jobs}, {("pi", "A")})
            self.assertEqual({j["policy"] for j in jobs}, {"R1A_m0", "R1A_m1", "R1A_m2"})

    def test_mixed_datasets_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            dirs = make_students(Path(tmp), CFG, seeds=[42, 43])
            make_students(Path(tmp), CFG, dataset_fp="other", seeds=[44])
            with self.assertRaises(ValueError):
                pl.declaration("R1M", dirs, "eval", CFG)

    def test_winner_and_pairs(self):
        results = {"S2A_pre": {"status": "pass", "gain_date_mean_s": 9.0},
                   "R1M": {"status": "fail", "gain_date_mean_s": 5.0},
                   "R1G": {"status": "hold_shared_collision", "gain_date_mean_s": 1.0}}
        self.assertIsNone(pl.choose_winner(results, CFG))
        results["R1G"]["status"] = "pass"
        self.assertEqual(pl.choose_winner(results, CFG), "R1G")
        dates = ["d1", "d2", "d3"]
        fake = lambda g: {"status": "pass", "gain_date_mean_s": g, "per_date_gain_s": {d: g + i * 0.1 for i, d in enumerate(dates)}}  # noqa: E731
        with patch.object(pl.s2e, "verify_baseline", return_value={"evaluation_fingerprint": "e"}), \
                patch.object(pl.s2e, "verified_runs", return_value=[]), \
                patch.object(pl.ref, "quality_errors", return_value=[]), \
                patch.object(pl, "_pi_runs", side_effect=lambda spec, *a: (spec.split("=")[0], {"d": 1}, [[1]] * 3)
                             if not spec.startswith("R1A") else ("R1A", None, None)), \
                patch.object(pl.s2e, "situations_for", return_value=[{}]), \
                patch.object(pl.s2e, "candidate_models", return_value=("S2A_pre", Path("x"), [[1]] * 3)), \
                patch.object(pl.s2e, "folder_record", return_value={"evaluation_fingerprint": "e"}), \
                patch.object(pl.ref, "judge", side_effect=[fake(0.9), fake(0.7), fake(0.6), fake(0.0)]):
            report = pl.collect_report(Path("b"), ["R1G=x", "R1M=y", "R1A=z"], ["S2A_pre=q"], CFG)
        self.assertEqual(report["pending"], {"R1A": "아직 평가 결과 없음"})
        self.assertAlmostEqual(report["paired"]["R1G-R1M"]["mean_s"], 0.2)
        self.assertAlmostEqual(report["paired"]["R1M-S2A_pre"]["mean_s"], 0.1)
        self.assertNotIn("R1A-R1G", report["paired"])


class CollectLoopTest(unittest.TestCase):
    """collect → finalize → 자료 읽기 → 학생 학습까지 가짜 실행기로 한 번에(재실행은 끝난 판을 건너뜀)."""

    def setUp(self):
        fake_saving(self)
        torch.set_num_threads(1)

    def test_collect_resume_dataset_and_train(self):
        cfg = copy.deepcopy(CFG)
        cfg["rollout"]["futures"] = 4
        cfg["student"]["epochs"] = 3
        cfg["situations_data"] = cfg["situations_data"][:2]
        net = g1.DuelingNet()
        payload = {"kind": "mode", "variant": None, "name": "S2A_pre_42", "sha256": "base-sha", "state": pi._state_bytes(net)}
        ran = []

        def fake_run(s, work, policy, max_seconds=None, on_step=None):
            ran.append(ac.situation_id(s))
            snaps = episode_snaps(25)
            ep = SimpleNamespace(env=SimpleNamespace(sumo=None), route=Path(work) / "r.rou.xml")
            for step, snap in enumerate(snaps):
                ctx = SimpleNamespace(ep=ep, step=step, mismatch=0)
                a = policy.decide(snap, ctx)
                on_step(ctx, snap, a, a)
            metrics = {"situation": ac.situation_id(s), "date": s["date"], "route_sha256": s["route_sha256"],
                       "mean_delay_s": 30.0, "seconds": 5.0, "p0_mismatch": 0, "mode_action_errors": 0,
                       "cost_check": {"passed": True}, "collisions_at_204820": 0, "collisions_unexplained": 0,
                       "choice_points": 5, "vehicles": 10}
            return ac.ArenaResult(metrics, [{"executed_green": 1}], [], [])

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "data"
            with patch.object(pi.ref, "register", return_value="fake-fp"), \
                    patch.object(pi, "base_payload", return_value=payload):
                code = pi.collect(cfg, out, Path(tmp) / "w", 0, 1, Path(tmp), 1, executor=FakePool(), run_episode=fake_run)
                self.assertEqual(code, 0)
                self.assertEqual(len(ran), 2)
                status = w.read(out / "status/collect_shard0.json")
                self.assertEqual((status["done"], status["failed"]), (2, []))
                self.assertEqual(pi.restore_totals(out)[1], 0)
                code = pi.collect(cfg, out, Path(tmp) / "w", 0, 1, Path(tmp), 1, executor=FakePool(),
                                  run_episode=lambda *a, **k: self.fail("끝난 판을 다시 돌렸습니다"))
                self.assertEqual(code, 0)
            w.write(out / "experiment.json", {"fingerprint": "fake-fp", "evaluation_fingerprint": "e",
                                              "evaluation": {"lock_sha256": w.file_hash(ac.CONFIG)}})
            cfg["base_policy"]["model_sha256"] = "base-sha"
            ds = pl.load_dataset(out, cfg, check_provenance=False)
            points = [p for s in cfg["situations_data"]
                      for p in pi.read_points(out / "runs" / pi.POLICY / ac.situation_id(s))]
            self.assertEqual(len(ds["observations"]), len(points))
            self.assertEqual(ds["manifest"]["used_points"], len(points))
            self.assertEqual(set(ds["kinds"].tolist()), {"choice", "agree"})
            # 가짜 비용: 녹색이 클수록 10씩 비싸다 → M4(1) 대비 녹색 0은 +0.010, 2는 −0.010
            choice = np.flatnonzero(ds["kinds"] == "choice")[0]
            np.testing.assert_allclose(ds["gains"][choice], [0.010, 0.0, -0.010, -0.020], atol=1e-6)
            with gzip.open(out / "runs" / pi.POLICY / ac.situation_id(cfg["situations_data"][0]) / "points.jsonl.gz", "rt") as f:
                self.assertTrue(all("main_after" not in json.loads(x) for x in f))
            with patch.object(pl, "load_dataset", return_value=ds):
                summary = pl.train_students(out, Path(tmp) / "students", cfg)
            self.assertEqual(sorted(summary), [42, 43, 44])
            dirs = pl.student_dirs(Path(tmp) / "students", cfg)
            self.assertEqual(pl.declaration("R1G", dirs, "eval", cfg)["dataset_fingerprint"], ds["fingerprint"])


if __name__ == "__main__":
    unittest.main()
