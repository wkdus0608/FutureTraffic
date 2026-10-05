"""Mapping gates and continuous whole-network signal replay; no DQN training."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from network_signals import (ReplayOptions, SignalGateError, audit_signal_links,
    audit_priority_green_conflicts, build_network_signal_programs, write_network_signal_additional)


def seconds(phases):
    return [state for duration, state in phases for _ in range(int(duration))]


def source_windows(cycle, slots):
    """Independently model a recorder omitting the phase crossing a 15m end."""
    timeline = []
    t, i = 0, 0
    while t < slots * 900:
        d, s = cycle[i % len(cycle)]
        timeline.append((t, t + d, s))
        t += d
        i += 1
    windows = {}
    for q in range(slots):
        begin, end = q * 900, (q + 1) * 900
        phases = []
        for a, b, s in timeline:
            if b <= begin or a >= end:
                continue
            # The first phase can be a remaining portion; the final unfinished
            # phase is not written by the source recorder.
            if b <= end:
                phases.append((b - max(begin, a), s))
        key = f"{begin // 3600:02d}{begin % 3600 // 60:02d}"
        windows[key] = phases
    actual = [s for a, b, s in timeline for _ in range(int(min(b, slots * 900) - a))]
    return windows, actual


class SignalFixtures(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.folder = Path(self.tmp.name)
        self.net = self.folder / "net.xml"

    def write_net(self, extra=False):
        root = ET.Element("net")
        for tls, width in [("main", 2)] + ([("mismatch", 3)] if extra else []):
            logic = ET.SubElement(root, "tlLogic", id=tls, type="static", offset="0", programID="0")
            for d, s in [(20, "Gr"), (3, "yr"), (30, "rG"), (3, "ry")]:
                ET.SubElement(logic, "phase", duration=str(d), state=s + ("r" * (width - 2)))
            incoming = ET.SubElement(root, "edge", id=f"{tls}_in", to=f"node_{tls}", **{"from": "source"})
            for i in range(width):
                ET.SubElement(incoming, "lane", id=f"{tls}_in_{i}", index=str(i))
                target = ET.SubElement(root, "edge", id=f"{tls}_out{i}", **{"from": f"node_{tls}", "to": "sink"})
                ET.SubElement(target, "lane", id=f"{tls}_out{i}_0", index="0")
                internal = ET.SubElement(root, "edge", id=f":{tls}_{i}", function="internal")
                ET.SubElement(internal, "lane", id=f":{tls}_{i}_0", index="0")
            junction = ET.SubElement(root, "junction", id=f"node_{tls}", type="traffic_light",
                incLanes=" ".join(f"{tls}_in_{i}" for i in range(width)),
                intLanes=" ".join(f":{tls}_{i}_0" for i in range(width)))
            for i in range(width):
                ET.SubElement(junction, "request", index=str(i), foes="0" * width, response="0" * width, cont="0")
            for i in range(width):
                ET.SubElement(root, "connection", tl=tls, linkIndex=str(i), **{
                    "from": f"{tls}_in", "to": f"{tls}_out{i}", "fromLane": str(i),
                    "toLane": "0", "dir": "s", "via": f":{tls}_{i}_0"})
                ET.SubElement(root, "connection", **{"from": f":{tls}_{i}", "to": f"{tls}_out{i}",
                    "fromLane": "0", "toLane": "0"})
        ET.ElementTree(root).write(self.net)
        return self.net

    def write_sources(self, windows, extra=False):
        paths = {}
        for key, phases in windows.items():
            root = ET.Element("additional")
            for tls in (["main", "mismatch"] if extra else ["main"]):
                logic = ET.SubElement(root, "tlLogic", id=tls, type="static", offset="0")
                for d, s in phases:
                    ET.SubElement(logic, "phase", duration=str(d), state=s)
            paths[key] = self.folder / f"{key}.xml"
            ET.ElementTree(root).write(paths[key])
        return paths

    def set_foes(self, root, tls="main", a=0, b=1, asymmetric=False):
        junction = root.find(f"junction[@id='node_{tls}']")
        width = len(junction.findall("request"))
        for left, right in [(a, b)] + ([] if asymmetric else [(b, a)]):
            req = junction.find(f"request[@index='{left}']")
            bits = list("0" * width)
            bits[-1 - right] = "1"
            req.set("foes", "".join(bits))

    def test_request_mapping_uses_incoming_lanes_not_tls_index(self):
        self.write_net(extra=True)
        root = ET.parse(self.net).getroot()
        # Three requests: reorder their signal columns without changing ROW.
        links = root.findall("connection[@tl='mismatch']")
        for conn, idx in zip(links, [2, 0, 1]):
            conn.set("linkIndex", str(idx))
        self.set_foes(root, tls="mismatch", a=0, b=2)
        ET.ElementTree(root).write(self.net)
        audit = audit_priority_green_conflicts(self.net,
            {"main": [(1, "Gg")], "mismatch": [(1, "rGG")]})
        checked = audit["tls"]["mismatch"]
        self.assertTrue(checked["mapping_complete"])
        self.assertFalse(audit["static_priority_green_safe"])
        pair = checked["conflicting_states"][0]["pairs"][0]
        self.assertEqual({pair["left"]["request_index"], pair["right"]["request_index"]}, {0, 2})
        self.assertEqual({pair["left"]["link_index"], pair["right"]["link_index"]}, {1, 2})
        # The same TLS columns are safe when they activate requests 1 and 2.
        safe = audit_priority_green_conflicts(self.net,
            {"main": [(1, "Gg")], "mismatch": [(1, "GGr")]})
        self.assertTrue(safe["static_priority_green_safe"])

    def test_split_internal_lanes_are_traced_and_broken_paths_blocked(self):
        self.write_net()
        root = ET.parse(self.net).getroot()
        edge = ET.SubElement(root, "edge", id=":main_continuation", function="internal")
        ET.SubElement(edge, "lane", id=":main_after_wait", index="0")
        root.find("connection[@from=':main_0']").set("via", ":main_after_wait")
        ET.SubElement(root, "connection", **{"from": ":main_continuation", "to": "main_out0",
            "fromLane": "0", "toLane": "0"})
        root.find("junction").set("intLanes", ":main_after_wait :main_1_0")
        ET.ElementTree(root).write(self.net)
        checked = audit_priority_green_conflicts(self.net)["tls"]["main"]
        self.assertTrue(checked["mapping_complete"])
        self.assertEqual(checked["request_mapping"][0]["internal_lane_trace"],
                         [":main_0_0", ":main_after_wait"])
        root.remove(root.find("connection[@from=':main_continuation']"))
        ET.ElementTree(root).write(self.net)
        with self.assertRaises(SignalGateError) as error:
            build_network_signal_programs(self.net, {}, 0, 60, "embedded",
                options=ReplayOptions(allow_priority_conflicts=True))
        self.assertIn("incomplete", str(error.exception))

    def test_conflicting_green_blocks_embedded_and_reconstructed_or_is_explicit_diagnostic(self):
        self.write_net()
        root = ET.parse(self.net).getroot()
        self.set_foes(root)
        root.find("tlLogic/phase").set("state", "GG")
        ET.ElementTree(root).write(self.net)
        with self.assertRaises(SignalGateError) as error:
            build_network_signal_programs(self.net, {}, 0, 60, "embedded")
        self.assertFalse(error.exception.report["static_priority_green_safe"])
        _, meta = build_network_signal_programs(self.net, {}, 0, 60, "embedded",
            options=ReplayOptions(allow_priority_conflicts=True))
        self.assertTrue(meta["diagnostic_only"])
        self.assertIs(meta["learning_allowed"], False)
        cycle = [(20, "GG"), (3, "yy"), (33, "rr")]
        windows, _ = source_windows(cycle, 1)
        paths = self.write_sources(windows)
        with self.assertRaises(SignalGateError) as error:
            build_network_signal_programs(self.net, paths, 0, 900, "reconstructed",
                options=ReplayOptions(allow_identity_mapping=True))
        self.assertIn("priority G/G", str(error.exception))

    def test_permissive_green_is_distinct_and_same_column_foes_are_detected(self):
        self.write_net()
        root = ET.parse(self.net).getroot()
        self.set_foes(root, asymmetric=True)
        ET.ElementTree(root).write(self.net)
        check = audit_priority_green_conflicts(self.net, {"main": [(10, "Gg")]})
        self.assertTrue(check["static_priority_green_safe"])
        check = audit_priority_green_conflicts(self.net, {"main": [(10, "GG")]})
        pair = check["tls"]["main"]["conflicting_states"][0]["pairs"][0]
        self.assertTrue(pair["left_foes_right"])
        self.assertFalse(pair["right_foes_left"])
        root.findall("connection[@tl='main']")[1].set("linkIndex", "0")
        ET.ElementTree(root).write(self.net)
        check = audit_priority_green_conflicts(self.net, {"main": [(10, "Gr")]})
        self.assertFalse(check["static_priority_green_safe"])

    def test_missing_matrix_or_tls_is_never_safe(self):
        self.write_net(extra=True)
        check = audit_priority_green_conflicts(self.net, {"main": [(1, "Gr")]})
        self.assertEqual(check["missing_tls"], ["mismatch"])
        self.assertFalse(check["static_priority_green_safe"])
        root = ET.parse(self.net).getroot()
        node = root.find("junction[@id='node_main']")
        node.find("request[@index='0']").set("foes", "000")
        ET.ElementTree(root).write(self.net)
        check = audit_priority_green_conflicts(self.net)
        self.assertFalse(check["tls"]["main"]["mapping_complete"])

    def test_width_equality_is_not_certification_or_automatic_replay(self):
        windows, _ = source_windows([(20, "Gr"), (3, "yr"), (30, "rG"), (3, "ry")], 2)
        paths = self.write_sources(windows)
        audit = audit_signal_links(self.write_net(), paths)
        self.assertTrue(audit["tls"]["main"]["width_compatible"])
        self.assertFalse(audit["tls"]["main"]["mapping_certified"])
        self.assertEqual(audit["tls"]["main"]["links"][1]["to_edge"], "main_out1")
        with self.assertRaises(SignalGateError) as error:
            build_network_signal_programs(self.net, paths, 0, 1800, "reconstructed")
        self.assertIn("identity", error.exception.report["gates"]["main"])

    def test_boundary_recorder_is_reconstructed_without_clock_drift(self):
        cycle = [(20, "Gr"), (3, "yr"), (30, "rG"), (3, "ry")]
        windows, actual = source_windows(cycle, 3)
        paths = self.write_sources(windows)
        programs, metadata = build_network_signal_programs(self.write_net(), paths, 0, 2700,
            "reconstructed", options=ReplayOptions(allow_identity_mapping=True))
        self.assertEqual(seconds(programs["main"]), actual)
        self.assertAlmostEqual(sum(d for d, _ in programs["main"]), 2700)
        self.assertTrue(any(n["kind"] == "period_tail_successor_assumption"
                            for n in metadata["tls"]["main"]["modifications"]))
        self.assertFalse(metadata["tls"]["main"]["mapping_certified"])
        # A partial run must slice the same continuous timeline.
        sliced, _ = build_network_signal_programs(self.net, paths, 871, 987, "reconstructed",
            options=ReplayOptions(allow_identity_mapping=True))
        self.assertEqual(seconds(sliced["main"]), actual[871:1858])

    def test_mismatch_never_pads_and_mixed_fallback_is_visible(self):
        windows, _ = source_windows([(20, "Gr"), (3, "yr"), (30, "rG"), (3, "ry")], 2)
        paths = self.write_sources(windows, extra=True)
        options = ReplayOptions(allow_identity_mapping=True)
        with self.assertRaises(SignalGateError) as error:
            build_network_signal_programs(self.write_net(extra=True), paths, 0, 1800,
                                          "reconstructed", options=options)
        self.assertEqual(set(error.exception.report["gates"]), {"mismatch"})
        programs, meta = build_network_signal_programs(self.net, paths, 0, 1800,
            "reconstructed_mixed", options=options)
        self.assertEqual(meta["tls"]["main"]["baseline"], "reconstructed")
        self.assertEqual(meta["tls"]["mismatch"]["baseline"], "embedded_fallback")
        self.assertTrue(all(len(s) == 3 for _, s in programs["mismatch"]))
        self.assertIn("explicit columns", meta["gates"]["mismatch"])

    def test_explicit_maps_validate_bounds_duplicates_and_evidence(self):
        windows, _ = source_windows([(20, "Gr"), (3, "yr"), (30, "rG"), (3, "ry")], 1)
        paths = self.write_sources(windows)
        self.write_net()
        for spec in [{"indices": [0, 2], "reason": "bad"}, {"indices": [0, 0], "reason": "bad"},
                     {"indices": [0, 1]}, {"indices": [0, 1], "reason": "bad", "certified": True}]:
            with self.subTest(spec=spec), self.assertRaises(SignalGateError):
                build_network_signal_programs(self.net, paths, 0, 900, "reconstructed",
                                              column_maps={"main": spec})
        p, meta = build_network_signal_programs(self.net, paths, 0, 900, "reconstructed",
            column_maps={"main": {"indices": [1, 0], "reason": "explicit fixture assumption"}})
        self.assertEqual(p["main"][0][1], "rG")
        self.assertFalse(meta["tls"]["main"]["mapping_certified"])

    def test_missing_file_is_gated_instead_of_silent_compression(self):
        windows, _ = source_windows([(20, "Gr"), (3, "yr"), (30, "rG"), (3, "ry")], 3)
        del windows["0015"]
        paths = self.write_sources(windows)
        with self.assertRaises(SignalGateError) as error:
            build_network_signal_programs(self.write_net(), paths, 0, 2700, "reconstructed",
                options=ReplayOptions(allow_identity_mapping=True))
        self.assertIn("missing quarter", error.exception.report["gates"]["main"])

    def test_phase_shift_is_an_explicit_recorded_option(self):
        windows, _ = source_windows([(20, "Gr"), (3, "yr"), (30, "rG"), (3, "ry")], 1)
        paths = self.write_sources(windows)
        p, meta = build_network_signal_programs(self.write_net(), paths, 0, 900, "reconstructed",
            options=ReplayOptions(allow_identity_mapping=True, phase_shifts={"main": -1}))
        self.assertEqual(p["main"][0], (20, "rG"))
        self.assertTrue(any(n["kind"] == "phase_row_shift_assumption"
                            for n in meta["tls"]["main"]["modifications"]))

    def test_missing_yellow_is_gated_or_repaired_with_exact_duration(self):
        windows, _ = source_windows([(20, "Gr"), (30, "rG")], 2)
        paths = self.write_sources(windows)
        self.write_net()
        with self.assertRaises(SignalGateError):
            build_network_signal_programs(self.net, paths, 0, 1800, "reconstructed",
                options=ReplayOptions(allow_identity_mapping=True))
        p, meta = build_network_signal_programs(self.net, paths, 0, 1800, "reconstructed",
            options=ReplayOptions(allow_identity_mapping=True, repair_yellow=True))
        self.assertEqual(sum(d for d, _ in p["main"]), 1800)
        self.assertEqual(meta["tls"]["main"]["transition_check"]["priority_green_to_red_without_yellow"], 0)
        self.assertTrue(any(n["kind"] == "missing_yellow_assumption"
                            for n in meta["tls"]["main"]["modifications"]))

    def test_drain_extension_and_source_protection(self):
        windows, _ = source_windows([(20, "Gr"), (3, "yr"), (30, "rG"), (3, "ry")], 1)
        paths = self.write_sources(windows)
        self.write_net()
        with self.assertRaises(SignalGateError):
            build_network_signal_programs(self.net, paths, 0, 1200, "reconstructed",
                options=ReplayOptions(allow_identity_mapping=True))
        p, meta = build_network_signal_programs(self.net, paths, 0, 1200, "reconstructed",
            options=ReplayOptions(allow_identity_mapping=True, extend_last_cycle=True))
        self.assertEqual(sum(d for d, _ in p["main"]), 1200)
        self.assertTrue(any(n["kind"] == "drain_cycle_extension_assumption"
                            for n in meta["tls"]["main"]["modifications"]))
        before = self.net.read_bytes()
        with self.assertRaises(ValueError):
            write_network_signal_additional(p, self.net, meta)
        self.assertEqual(before, self.net.read_bytes())
        output = write_network_signal_additional(p, self.folder / "tls.add.xml", meta)
        self.assertEqual(ET.parse(output).getroot().tag, "additional")
        saved = json.loads(output.with_suffix(".metadata.json").read_text())
        self.assertEqual(saved["fingerprint"], meta["fingerprint"])
        self.assertEqual(meta["implementation_sha256"], __import__("hashlib").sha256(
            (ROOT / "src/network_signals.py").read_bytes()).hexdigest())

    def test_long_green_interpolation_is_optional_and_never_certified(self):
        cycle = [(20, "Gr"), (3, "yr"), (30, "rG"), (3, "ry")]
        # A long source green surrounded by several ordinary cycles. It could
        # be a source dropout or intentional control; default must preserve it.
        phases = cycle * 3 + [(200, "Gr"), (3, "yr"), (30, "rG"), (3, "ry")] + cycle * 5
        paths = self.write_sources({"0000": phases})
        self.write_net()
        p, normal = build_network_signal_programs(self.net, paths, 0, 900, "reconstructed",
            options=ReplayOptions(allow_identity_mapping=True))
        changed, meta = build_network_signal_programs(self.net, paths, 0, 900, "reconstructed",
            options=ReplayOptions(allow_identity_mapping=True, interpolate_long_greens=True))
        self.assertIn((200, "Gr"), p["main"])
        self.assertLess(max(d for d, s in changed["main"] if s == "Gr"), 200)
        self.assertAlmostEqual(sum(d for d, _ in changed["main"]), 900)
        self.assertFalse(any(n["kind"] == "long_green_interpolation_assumption"
                             for n in normal["tls"]["main"]["modifications"]))
        self.assertTrue(any(n["kind"] == "long_green_interpolation_assumption"
                            for n in meta["tls"]["main"]["modifications"]))
        self.assertFalse(meta["tls"]["main"]["mapping_certified"])

    def test_fingerprint_changes_with_source_and_reconstruction_options(self):
        windows, _ = source_windows([(20, "Gr"), (3, "yr"), (30, "rG"), (3, "ry")], 1)
        paths = self.write_sources(windows)
        self.write_net()
        opts = ReplayOptions(allow_identity_mapping=True)
        _, before = build_network_signal_programs(self.net, paths, 0, 900, "reconstructed", options=opts)
        tree = ET.parse(paths["0000"])
        tree.getroot().find("tlLogic/phase").set("duration", "21")
        tree.write(paths["0000"])
        _, after = build_network_signal_programs(self.net, paths, 0, 900, "reconstructed", options=opts)
        self.assertNotEqual(before["fingerprint"], after["fingerprint"])
        self.assertNotEqual(before["audit"]["source_files"][0]["sha256"],
                            after["audit"]["source_files"][0]["sha256"])
        _, shifted = build_network_signal_programs(self.net, paths, 0, 900, "reconstructed",
            options=ReplayOptions(allow_identity_mapping=True, phase_shifts={"main": -1}))
        self.assertNotEqual(after["fingerprint"], shifted["fingerprint"])

    def test_embedded_baseline_does_not_need_source_or_identity_claim(self):
        p, meta = build_network_signal_programs(self.write_net(extra=True), {}, 18000, 1200, "embedded")
        self.assertEqual(set(p), {"main", "mismatch"})
        self.assertTrue(all(sum(d for d, _ in phases) == 1200 for phases in p.values()))
        self.assertTrue(all(d["baseline"] == "embedded" for d in meta["tls"].values()))


class ProvidedSignalData(unittest.TestCase):
    @unittest.skipUnless((ROOT / "results/focused_204820/signals_all/20220810").is_dir(),
                         "provided daily signal files are not installed")
    def test_all_twelve_tls_and_three_width_mismatches_are_preserved(self):
        net = next((ROOT / "Sample").rglob("1200012n.net.xml"))
        files = {p.name[14:18]: p for p in sorted(
            (ROOT / "results/focused_204820/signals_all/20220810").glob("*s.xml"))}
        audit = audit_signal_links(net, files)
        self.assertEqual(audit["tls_count"], 12)
        bad = {k for k, v in audit["tls"].items() if not v["width_compatible"]}
        self.assertEqual(bad, {"201306", "201330", "cluster_204796_205202"})
        p, meta = build_network_signal_programs(net, files, 0, 86400, "reconstructed_mixed",
            options=ReplayOptions(allow_identity_mapping=True, repair_yellow=True, allow_priority_conflicts=True))
        self.assertEqual(set(p), set(audit["tls"]))
        self.assertTrue(all(sum(d for d, _ in x) == 86400 for x in p.values()))
        self.assertTrue(all(meta["tls"][k]["baseline"] == "embedded_fallback" for k in bad))
        self.assertEqual(meta["tls"]["204822"]["baseline"], "reconstructed")
        self.assertFalse(meta["tls"]["204822"]["mapping_certified"])
        safety = meta["priority_green_audit"]
        self.assertEqual(safety["tls_count"], 12)
        self.assertTrue(all(v["mapping_complete"] for v in safety["tls"].values()))
        self.assertEqual(sum(v["mapped_connections"] for v in safety["tls"].values()), 249)
        self.assertFalse(meta["static_priority_green_safe"])
        self.assertFalse(meta["learning_allowed"])
        cluster = safety["tls"]["cluster_204796_205202"]
        pair = next(c for c in cluster["conflicting_states"]
                    if c["state"] == "rrrrrGGGrrrrrrrrrGGGrrrr")["pairs"][0]
        self.assertEqual({pair["left"]["request_index"], pair["right"]["request_index"]}, {5, 17})


if __name__ == "__main__":
    unittest.main()
