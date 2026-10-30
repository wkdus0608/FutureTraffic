"""Teacher labels, future separation and date-heldout learning; no SUMO."""
from pathlib import Path
import copy
import gzip
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import arena3_stage2_data as sd
import arena3_g1 as g1
import arena3_referee as ref


def point(name="state", gain=1200.0, valid=8):
    greens = [0, 0, 1, 0, 0, 2]
    mask = sd.ac.canonical_mask6(greens)
    obs = np.zeros(119, np.float32)
    obs[0] = float(gain > 0)
    for m, g in enumerate(greens[:5]):
        obs[84 + m * 4 + g] = 1
    obs[104:109] = mask[:5]
    obs[110 + greens[5]] = 1
    obs[114] = mask[5]
    obs[115:119] = [0.0, 0.5, 1.0, 0.7]
    rollouts = {str(g): [] for g in sorted(set(greens))}
    for j in range(8):
        for g in sorted(set(greens)):
            cost = 10000.0 + 10 * j - (gain if g == 1 else -500.0 if g == 2 else 0.0)
            rollouts[str(g)].append({"cost": cost, "future_seed": j, "collisions_unexplained": 0,
                                     "collisions_at_204820": 0} if j < valid else None)
    return {"state": name, "greens": greens, "mask": mask, "observation": obs.tolist(),
            "allowed": [0, 1, 2, 3], "m4": 0, "chosen": 1 if gain > 0 else 0,
            "candidates": [0, 1, 2], "rollouts": rollouts, "restore_check": {"ok": True}}


def dataset(n_dates=3):
    rows = []
    for d in range(n_dates):
        for i in range(12):
            p = point(f"d{d}s{i}", 1200.0 if i % 2 else -1000.0)
            r, _ = sd.point_to_row(p, f"date{d}", f"sit{d}")
            rows.append(r)
    keys = ["observations", "mask", "gains", "gain_se", "fit_gains", "fit_gain_se", "score_gains",
            "teacher_mode", "fit_teacher_mode", "anchor_mode", "w4_mode", "rule_mode", "dates", "state_ids", "situations"]
    return {**{k: np.asarray([r[k] for r in rows]) for k in keys}, "fingerprint": "synthetic",
            "manifest": {"used_points": len(rows), "excluded": {}, "dates": [f"date{i}" for i in range(n_dates)]}}


class PointTest(unittest.TestCase):
    def test_gains_units_canonical_modes_and_clean_restore(self):
        r, reason = sd.point_to_row(point(), "train", "sid")
        self.assertIsNone(reason)
        self.assertEqual(r["observations"].shape, (119,))
        self.assertEqual(r["anchor_mode"], 0)
        self.assertEqual(r["teacher_mode"], 2)
        self.assertAlmostEqual(r["gains"][2], 1.2, places=6)
        self.assertEqual(r["gain_se"][2], 0)
        self.assertAlmostEqual(r["gains"][5], -0.5)
        self.assertEqual(set(r["fit_future_indices"]), {0, 1, 2, 3})
        self.assertEqual(set(r["score_future_indices"]), {4, 5, 6, 7})

    def test_future_halves_do_not_overlap_with_seven_or_four(self):
        for n in [7, 4]:
            r, reason = sd.point_to_row(point(valid=n), "train", "sid")
            self.assertIsNone(reason)
            self.assertFalse(set(r["fit_future_indices"]) & set(r["score_future_indices"]))
            self.assertEqual(len(r["fit_future_indices"]) + len(r["score_future_indices"]), n)

    def test_restore_bad_is_explicitly_excluded(self):
        p = point()
        p["restore_check"]["ok"] = False
        self.assertEqual(sd.point_to_row(p, "train", "sid"), (None, "restore_mismatch"))

    def test_future_collision_drops_same_index_for_every_candidate(self):
        p = point()
        p["rollouts"]["1"][3]["collisions_unexplained"] = 1
        r, _ = sd.point_to_row(p, "train", "sid")
        self.assertEqual(r["valid_futures"], 7)
        self.assertNotIn(3, r["fit_future_indices"] + r["score_future_indices"])
        self.assertAlmostEqual(r["gains"][2], 1.2, places=6)

    def test_too_few_futures_excludes_and_malformed_pair_fails(self):
        self.assertEqual(sd.point_to_row(point(valid=3), "train", "sid"), (None, "insufficient_valid_futures"))
        p = point()
        p["rollouts"]["1"][0]["future_seed"] = 999
        with self.assertRaises(ValueError):
            sd.point_to_row(p, "train", "sid")

    def test_chosen_and_observation_must_match(self):
        p = point()
        p["chosen"] = 2
        with self.assertRaises(ValueError):
            sd.point_to_row(p, "train", "sid")
        p = point()
        p["observation"][84] = 0
        with self.assertRaises(ValueError):
            sd.point_to_row(p, "train", "sid")


class LearningTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_both_losses_finite_and_canonical_greedy(self):
        d = dataset(2)
        b = sd.tensor_batch(d, [0, 1, 2, 3])
        for variant in sd.VARIANTS:
            net = g1.DuelingNet()
            loss = sd.supervised_loss(net, b["obs"], b["gains"], b["gain_se"], b["mask"], variant)
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertTrue(any(p.grad is not None for p in net.parameters()))
            picks = sd.predict_modes(net, d["observations"], d["mask"])
            self.assertTrue(np.all(d["mask"][np.arange(len(picks)), picks]))

    def test_both_pretrain_losses_improve_on_predictable_inputs(self):
        d = dataset(2)
        for variant in sd.VARIANTS:
            torch.manual_seed(42)
            net = g1.DuelingNet()
            meta = sd.pretrain(net, d, variant, 42, epochs=60)
            self.assertLess(meta["final_full_loss"], meta["initial_full_loss"])
            self.assertEqual(meta["points"], len(d["observations"]))
            self.assertEqual(meta["dataset_fingerprint"], "synthetic")

    def test_cv_holds_entire_date_and_failure_is_advisory(self):
        d = dataset(3)
        cfg = {**sd.DEFAULT_PRETRAIN, "epochs": 1}
        with tempfile.TemporaryDirectory() as tmp, patch.object(sd, "stage2_settings", return_value=cfg):
            r = sd.offline_report(d, Path(tmp))
            self.assertFalse(r["blocks_rl"])
            self.assertEqual(r["status"], "advisory_only")
            for fold in r["folds"]:
                self.assertNotIn(fold["heldout_date"], fold["train_dates"])
                self.assertEqual(fold["heldout_points"], 12)
            self.assertEqual(r["methods"]["M4"]["date_mean_gain_vehicle_seconds_per_decision"], 0)
            self.assertTrue((Path(tmp) / "offline_report.md").is_file())

    def test_scoring_uses_second_half_not_fit_targets(self):
        d = dataset(2)
        d["score_gains"][:, 2] = -4
        d["fit_gains"][:, 2] = 5
        score = sd._score(d, np.full(len(d["observations"]), 2))
        self.assertEqual(score["date_mean_gain_vehicle_seconds_per_decision"], -4000)
        self.assertEqual(score["harmful_switches_on_score_futures"], len(d["observations"]))


class DataIntegrityTest(unittest.TestCase):
    def make_folder(self, root):
        cfg = copy.deepcopy(sd.ac.config())
        cfg["dates"]["train"] = ["train1", "train2"]
        cfg["dates"]["eval"] = ["eval1"]
        cfg["situations_bprime_data"] = [
            {"date": date, "period": "0800", "tag": "T" + str(i), "route_sha256": "a" * 64}
            for i, date in enumerate(cfg["dates"]["train"])]
        evaluation = {"kind": sd.ac.EXPERIMENT, "environment_fingerprint": cfg["environment"]["fingerprint"],
                      "lock_sha256": sd.sha(sd.ac.CONFIG.read_bytes()),
                      "files": {p: sd.sha((ROOT / p).read_bytes()) for p in ref.SOURCES}, "packages": {}}
        frozen = {"evaluation": evaluation, "evaluation_fingerprint": sd.digest(evaluation),
                  "extra_files": {"src/arena3_bprime.py": sd.sha((ROOT / 'src/arena3_bprime.py').read_bytes())}}
        fingerprint = sd.digest(frozen)
        sd.write_json(root / "experiment.json", {"fingerprint": fingerprint, **frozen})
        for s in cfg["situations_bprime_data"]:
            sid = sd.ac.situation_id(s)
            folder = root / "runs/BPRIME" / sid
            folder.mkdir(parents=True)
            points = [point(sid + "_p0")]
            raw = gzip.compress((json.dumps(points[0]) + "\n").encode())
            (folder / "points.jsonl.gz").write_bytes(raw)
            m = {"situation": sid, "date": s["date"], "route_sha256": s["route_sha256"], "policy": "BPRIME", "role": "data",
                 "fingerprint": fingerprint, "model_identity": None, "points_sha256": sd.sha(raw), "bprime_points": 1,
                 "cost_check": {"passed": True}}
            mraw = json.dumps(m).encode()
            (folder / "metrics.json").write_bytes(mraw)
            sd.write_json(folder / "cache.json", {"fingerprint": fingerprint, "model_identity": None,
                                                  "metrics_sha256": sd.sha(mraw), "decisions_sha256": "not-read"})
        return cfg

    def test_full_loader_and_portable_fingerprint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "teacher"
            cfg = self.make_folder(root)
            with patch.object(sd.ac, "config", return_value=cfg):
                d = sd.load_dataset(root)
                self.assertEqual(d["observations"].shape, (2, 119))
                self.assertNotIn(str(root), json.dumps(d["manifest"]))
                import shutil
                moved = Path(tmp) / "other"
                shutil.copytree(root, moved)
                self.assertEqual(sd.load_dataset(moved)["fingerprint"], d["fingerprint"])

    def test_tampered_metrics_and_extra_situation_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = self.make_folder(root)
            with patch.object(sd.ac, "config", return_value=cfg):
                m = next((root / "runs/BPRIME").glob('*/metrics.json'))
                m.write_text(m.read_text() + ' ')
                with self.assertRaises(ValueError):
                    sd.load_dataset(root)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = self.make_folder(root)
            (root / 'runs/BPRIME/eval1_0800_E1').mkdir()
            with patch.object(sd.ac, "config", return_value=cfg), self.assertRaises(ValueError):
                sd.load_dataset(root)


if __name__ == '__main__':
    unittest.main()
