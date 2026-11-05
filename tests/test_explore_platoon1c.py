"""[추가 탐색] PLATOON1C 검사 — SUMO 실행 없음(가짜 신경망·가짜 실행 기록)."""
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

import arena3_common as ac  # noqa: E402
import arena3_referee as ref  # noqa: E402
import explore_platoon1 as pt  # noqa: E402
import explore_platoon1_learn as pl  # noqa: E402
import explore_platoon1c as pc  # noqa: E402
import explore_weight as w  # noqa: E402
from test_explore_platoon1 import fake_run, obs167  # noqa: E402

w.configure()
CFG = pc.config()
PCFG = pt.config()
LOCK = ac.config()
SITS = PCFG["situations_eval"]


class FakeNet:
    """TM4 녹색 대비 값을 그대로 돌려주는 신경망(값은 TM4 녹색이 0이 되게 만든다)."""

    def __init__(self, values):
        self.values = values

    def __call__(self, obs):
        anchor = int(obs[0, pt.TM4_ONEHOT].argmax())
        return torch.tensor([[v - self.values[anchor] for v in self.values]], dtype=torch.float32)


class ConfigTest(unittest.TestCase):
    def test_candidates_are_the_preregistered_three(self):
        self.assertEqual({k: v["tau"] for k, v in CFG["candidates"].items()}, {"t09": 0.9, "t20": 2.0, "t30": 3.0})
        self.assertEqual(CFG["students"]["seeds"], PCFG["student"]["seeds"])
        self.assertEqual(len(set(CFG["students"]["model_sha256"])), 3)
        self.assertEqual(CFG["rules"]["learning_min_gain_s"], PCFG["rules"]["learning_min_gain_s"])
        self.assertEqual(CFG["rules"]["learning_min_positive_dates"], PCFG["rules"]["learning_min_positive_dates"])

    def test_jobs_keep_candidates_and_models_apart(self):
        folders = [Path(f"/m/seed_{s}") for s in CFG["students"]["seeds"]]
        jobs = pc.jobs_for(Path("/o/t20"), Path("/w"), "fp", "t20", 2.0, SITS, folders, CFG)
        self.assertEqual(len(jobs), 252)
        self.assertEqual(len({j["target"] for j in jobs}), 252)
        self.assertEqual({j["model_identity"].split("|")[1] for j in jobs}, {"tau=2"})
        self.assertEqual({j["tau"] for j in jobs}, {2.0})
        self.assertEqual({j["model_sha256"] for j in jobs}, set(CFG["students"]["model_sha256"]))
        self.assertNotEqual(pc.identity("a", 0.9), pc.identity("a", 2.0))


class SureGreenTest(unittest.TestCase):
    def test_threshold(self):
        obs = obs167((0, 1, 2), tm4=1)
        net = FakeNet([0.5, 0.0, 2.5, 9.0])                         # 녹색 3은 허용 밖
        self.assertEqual(pc.sure_green(net, obs, 2.0), (2, 2.5))
        self.assertEqual(pc.sure_green(net, obs, 3.0), (1, 2.5))    # 이득이 문턱 아래면 TM4
        self.assertEqual(pc.sure_green(net, obs, 2.5), (2, 2.5))    # 문턱과 같으면 바꾼다
        low = FakeNet([-0.5, 0.0, -1.0, 9.0])
        self.assertEqual(pc.sure_green(low, obs, 0.0), (1, -0.5))   # TM4보다 나은 녹색이 없으면 TM4

    def test_never_and_forced(self):
        obs = obs167((0, 1), tm4=0)
        self.assertEqual(pc.sure_green(FakeNet([0.0, 50.0, 0, 0]), obs, 1e9)[0], 0)
        with self.assertRaises(ValueError):
            pc.sure_green(FakeNet([0, 0, 0, 0]), obs167((1,), tm4=1), 0.0)      # 고를 것 없음
        with self.assertRaises(ValueError):
            pc.sure_green(FakeNet([0, 0, 0, 0]), obs167((0, 1), tm4=2), 0.0)    # TM4 녹색이 허용 밖
        with self.assertRaises(ValueError):
            pc.sure_green(FakeNet([0, float("nan"), 0, 0]), obs, 0.0)

    def test_small_tau_is_pt1_greedy(self):
        rng = np.random.default_rng(7)
        torch.manual_seed(7)
        net = pt.PlatoonNet().eval()
        for p in net.advantage.parameters():
            torch.nn.init.normal_(p, std=0.5)
        hits = 0
        for k in range(300):
            allowed = tuple(sorted(rng.choice(4, size=int(rng.integers(2, 5)), replace=False).tolist()))
            tm4 = int(rng.choice(allowed))
            obs = obs167(allowed, tm4, extra_seed=k)
            action = pc.sure_green(net, obs, float(CFG["smoke"]["always_tau"]))[0]
            self.assertEqual(action, pt.greedy_green(net, obs))
            hits += action != tm4
        self.assertGreater(hits, 30)


class PolicyTest(unittest.TestCase):
    def decide(self, policy, allowed, tm4):
        core = SimpleNamespace(observe=lambda env: None,
                               view=lambda env, a: {"green": tm4, "extra": obs167(allowed, tm4)[119:].tolist()})
        snap = SimpleNamespace(observation=obs167(allowed, tm4)[:119])
        ctx = SimpleNamespace(ep=SimpleNamespace(env=None), allowed=list(allowed))
        with mock.patch.object(pt, "episode_core", lambda holder, c: core):
            return policy.decide(snap, ctx)

    def test_counts_and_info(self):
        policy = pc.SureStudentPolicy(FakeNet([0.0, 2.5, 0.0, 0.0]), 2.0, "t20_m0")
        self.assertEqual(self.decide(policy, (0, 1), 0), 1)
        self.assertEqual(self.decide(policy, (1,), 1), 1)           # 허용 녹색 하나: 세지 않음
        strict = pc.SureStudentPolicy(FakeNet([0.0, 2.5, 0.0, 0.0]), 3.0, "t30_m0")
        self.assertEqual(self.decide(strict, (0, 1), 0), 0)
        info = policy.info()
        self.assertEqual((info["planned"], info["differs_from_tm4"], info["tau"]), (1, 1, 2.0))
        self.assertEqual(info["pairs"], {"0->1": 1})
        self.assertEqual(info["above"], {"0": 1, "0.9": 1, "2": 1, "3": 0})
        self.assertEqual((strict.stats["differs_from_tm4"], strict.stats["planned"]), (0, 1))
        self.assertIsInstance(policy, ac.P0Policy)                  # 실행기가 매 제어 단계 부른다


def rows(policy, gain_by_date, information=None, sits=SITS):
    out = []
    for s in sits:
        m = fake_run(s, 40.0 - gain_by_date.get(s["date"], 0.0), policy)
        m["information"] = information or {"planned": 100, "differs_from_tm4": 10, "pairs": {"1->2": 10}}
        out.append(m)
    return out


class JudgeTest(unittest.TestCase):
    def judged(self, gains):
        m4 = rows("M4", {})
        return ref.judge([rows("c", dict(zip(LOCK["dates"]["eval"], gains)))], m4, None, LOCK, SITS)

    def test_select_largest_learning_among_passed(self):
        t975 = LOCK["judge"]["t975"]
        tm4 = self.judged([3.8] * 7)
        good = self.judged((4.5, 4.4, 4.6, 4.2, 4.5, 4.7, 4.3))
        better = self.judged((4.9, 4.8, 5.0, 4.6, 4.9, 5.1, 4.7))
        bad = self.judged((3.0, 3.2, 2.9, 3.1, 3.0, 3.3, 2.8))
        paired = {f"{n}-TM4": pl.paired_difference(r, tm4, t975) for n, r in (("t09", bad), ("t20", good), ("t30", better))}
        verdicts = {n: pl.main_verdict(r, paired[f"{n}-TM4"], CFG["rules"]) for n, r in (("t09", bad), ("t20", good), ("t30", better))}
        self.assertEqual([verdicts[n]["passed"] for n in ("t09", "t20", "t30")], [False, True, True])
        self.assertEqual(pc.select(verdicts, paired), "t30")
        self.assertIsNone(pc.select({"t09": verdicts["t09"]}, paired))

    def test_deviation_summary(self):
        a = rows("m0", {}, {"planned": 100, "differs_from_tm4": 20, "pairs": {"1->2": 15, "1->0": 5}}, SITS[:2])
        b = rows("m1", {}, {"planned": 100, "differs_from_tm4": 10, "pairs": {"1->2": 10}}, SITS[:2])
        d = pc.deviation_summary([a, b])
        self.assertAlmostEqual(d["share"], 0.15)
        self.assertEqual(list(d["pairs"]), ["1->2", "1->0"])
        self.assertAlmostEqual(d["pairs"]["1->2"]["share"], 50 / 60)

    def test_render_incomplete_and_no_confirmation_without_winner(self):
        self.assertIn("끝나지 않은", pc.render({"status": "incomplete", "missing": {"t09": [1, 0, 0]}}, "개발"))
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            pc.s2e._write(tmp / "judge" / "judge.json", {"status": "done", "part": "eval", "selected": None, "verdicts": {}})
            result = pc.confirm(tmp / "judge", tmp / "students", tmp / "confirm", tmp / "work", 1, CFG)
            self.assertEqual(result["status"], "not_run_development_failed")
            pc.s2e._write(tmp / "judge2" / "judge.json", {"status": "incomplete", "part": "eval"})
            with self.assertRaises(ValueError):
                pc.confirm(tmp / "judge2", tmp / "students", tmp / "confirm2", tmp / "work", 1, CFG)

    def test_confirmation_report_reads_folders_and_checks_identity(self):
        sits = pl.situations(PCFG, "confirmation")
        dates = LOCK["dates"]["eval"]
        sha = CFG["students"]["model_sha256"]

        def write(folder, name, model_identity, gains, fingerprint):
            for m in rows(name, dict(zip(dates, gains)), sits=sits):
                result = ac.ArenaResult(metrics=m, decisions=[], rewards=[])
                ac.write_run(ref.target_dir(folder, name, m["situation"]), result,
                             {"policy": name, "fingerprint": fingerprint, "model_identity": model_identity})

        with tempfile.TemporaryDirectory() as tmp:
            base, cand = Path(tmp) / "baseline", Path(tmp) / "t20"
            fp_base, fp_cand = ref.register(base, extra_files=("src/explore_platoon1_learn.py",)), ref.register(cand, extra_files=pc.SOURCES)
            write(base, "M4", "P/M4", [0.0] * 7, fp_base)
            write(base, "TM4", "P/TM4", [3.8] * 7, fp_base)
            for i, name in enumerate(pc.policy_names("t20")):
                if i < 2:
                    write(cand, name, pc.identity(sha[i], 2.0), (4.5, 4.4, 4.6, 4.2, 4.5, 4.7, 4.3), fp_cand)
            self.assertEqual(pc.confirmation_report(cand, base, "t20", CFG)["status"], "incomplete")
            write(cand, "t20_m2", pc.identity(sha[2], 2.0), (4.5, 4.4, 4.6, 4.2, 4.5, 4.7, 4.3), fp_cand)
            rep = pc.confirmation_report(cand, base, "t20", CFG)
            self.assertEqual(rep["status"], "done")
            self.assertAlmostEqual(rep["paired"]["t20-TM4"]["mean_s"], 0.657142857, places=6)
            self.assertIn("t20 − TM4", pc.render(rep, "확인"))
            with self.assertRaises(ValueError):                      # 다른 문턱으로 돌린 실행은 읽지 않는다
                pc._candidate_rows(cand, "t20", 3.0, sits, CFG)


if __name__ == "__main__":
    unittest.main()
