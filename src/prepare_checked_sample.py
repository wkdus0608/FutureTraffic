"""검증한 샘플 보완을 별도 사본에 적용한다. 기존 시나리오는 바꾸지 않는다.

도로 ID 대응은 공급자의 공식 대응표가 없는 조건부 가설이다. 이 스크립트는
확인한 샘플 도로망에서만 실행되며, 새 도로 연결이나 신호를 만들지 않는다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from prepare_sumo_scenario import VEHICLE_TYPES, find_one

NETWORK_SHA256 = "c041e7ccf0c389ea60425a5916ba36e3c89838177ce61b2ddf1d6229cf67223d"
OLD_EDGE, NEW_EDGE = "477706911_02", "477706911_02_02"
PROJECT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_xml(tree: ET.ElementTree, path: Path) -> None:
    ET.indent(tree, space="  ")
    tree.write(path, encoding="utf-8", xml_declaration=True)


def prepare_copies(sample: Path, prefix: str, bundle: Path) -> dict:
    names = ["1200012n.net.xml", "possible_routes.rou.xml", "exp.rou.xml"]
    names += [f"{prefix}{code}.xml" for code in VEHICLE_TYPES]
    sources = {name: find_one(sample, name) for name in names}
    source_hashes = {name: digest(path) for name, path in sources.items()}
    if digest(sources[names[0]]) != NETWORK_SHA256:
        raise ValueError("확인한 샘플과 다른 도로망입니다. 자동 보완을 적용하지 않습니다.")
    network = ET.parse(sources[names[0]]).getroot()
    edges = {e.get("id"): e for e in network.findall("edge") if not e.get("id", "").startswith(":")}
    connections = {(c.get("from"), c.get("to")) for c in network.findall("connection")
                   if not c.get("from", "").startswith(":")}
    incoming = {b for a, b in connections}
    outgoing = {a for a, b in connections}
    lengths = {name: float(e.find("lane").get("length")) for name, e in edges.items()}
    if OLD_EDGE in edges or edges[NEW_EDGE].get("from") != "204820":
        raise ValueError("샘플 도로 ID 대응의 전제와 다릅니다.")

    # 원본 보호: 출력 경로가 새 폴더일 때만 생성한다.
    if bundle.exists():
        raise FileExistsError(f"기존 결과를 덮어쓰지 않습니다: {bundle}")
    if bundle == sample or sample in bundle.parents:
        raise ValueError("원본 Sample 내부에는 결과를 만들 수 없습니다.")
    inputs = bundle / "inputs"
    inputs.mkdir(parents=True)
    for name, source in sources.items():
        shutil.copy2(source, inputs / name)

    mapping_count = 0
    measured_pairs = set()
    for code in VEHICLE_TYPES:
        path = inputs / f"{prefix}{code}.xml"
        tree = ET.parse(path)
        for relation in tree.findall(".//edgeRelation"):
            if relation.get("to") == OLD_EDGE:
                if (relation.get("from"), NEW_EDGE) not in connections:
                    raise ValueError("도착 ID를 바꿔도 허용되는 회전 연결이 없습니다.")
                mapping_count += int(float(relation.get("count", "0")))
                relation.set("to", NEW_EDGE)
            if float(relation.get("count", "0")) > 0:
                measured_pairs.add((relation.get("from"), relation.get("to")))
        write_xml(tree, path)

    path = inputs / "possible_routes.rou.xml"
    tree = ET.parse(path)
    original = [tuple(r.get("edges").split()) for r in tree.findall(".//route")]
    coverage = {pair for route in original for pair in zip(route, route[1:])}
    missing_allowed = sorted((measured_pairs - coverage) & connections)
    added = []
    for a, b in missing_allowed:
        # 제공 경로의 앞·뒤 구간을 연결한다. 교차로 바로 앞에서 생성하고
        # 바로 뒤에서 삭제하는 2-edge 진단 경로는 사용하지 않는다.
        prefixes = {r[:i+1] for r in original for i, e in enumerate(r) if e == a
                    and (a not in incoming or sum(lengths[x] for x in r[:i]) >= 100)}
        suffixes = {r[i:] for r in original for i, e in enumerate(r) if e == b
                    and (r[-1] not in outgoing or sum(lengths[x] for x in r[i+1:]) >= 100)}
        candidates = {p+s for p in prefixes for s in suffixes
                      if len(p+s) >= 3 and len(set(p+s)) == len(p+s)
                      and all(pair in connections for pair in zip(p+s, (p+s)[1:]))}
        if not candidates:
            raise ValueError(f"전체 경로를 구성할 수 없습니다: {a} → {b}")

        def score(route: tuple) -> tuple:
            others = len(set(zip(route, route[1:])) & (measured_pairs - {(a, b)}))
            boundary_penalty = int(route[0] in incoming) + int(route[-1] in outgoing)
            return boundary_penalty, others, sum(lengths[x] for x in route), route

        for route in sorted(candidates, key=score)[:8]:
            identifier = f"checked_supplement_{len(added)}"
            vehicle = ET.SubElement(tree.getroot(), "vehicle", id=identifier, depart="0")
            ET.SubElement(vehicle, "route", edges=" ".join(route))
            added.append({"id": identifier, "movement": [a, b], "edges": list(route),
                          "length_m": round(sum(lengths[x] for x in route), 2),
                          "origin_is_boundary": route[0] not in incoming,
                          "destination_is_boundary": route[-1] not in outgoing})
    write_xml(tree, path)
    manifest = {
        "status": "experimental_sample_correction",
        "network_changed": False,
        "source_files": {name: {"path": str(p), "sha256": source_hashes[name]} for name, p in sources.items()},
        "conditional_id_mapping": {"from": OLD_EDGE, "to": NEW_EDGE, "mapped_observations": mapping_count,
                                   "supplier_mapping_confirmed": False},
        "added_candidate_routes": added,
        "covered_new_movements": [list(pair) for pair in missing_allowed],
        "note": "제공 경로의 앞뒤 구간을 결합한 후보이며, 실제 차량의 출발지·목적지를 확인한 결과는 아닙니다.",
    }
    # 생성 후에도 원본 지문을 다시 확인한다.
    for name, record in manifest["source_files"].items():
        if digest(sources[name]) != record["sha256"]:
            raise RuntimeError("실행 중 원본이 변경됐습니다.")
    write_json(bundle / "correction_manifest.json", manifest)
    return manifest


def print_summary(bundle: Path) -> None:
    """생성 때와 조회 때 같은 한국어 요약을 출력한다."""
    scenario = bundle / "scenario"
    summary = json.loads((scenario / "data_quality_summary.json").read_text())
    metadata = json.loads((scenario / "scenario_metadata.json").read_text())
    begin, end = int(metadata["source_begin_sec"]), int(metadata["source_end_sec"])
    print(f"생성 차량: {summary['generated_vehicle_count']:,}대")
    print(f"원본 시간: {begin}~{end}초 → SUMO 시간: 0~{end - begin}초")
    print(
        f"회전교통량 누락: {summary['missing_turn_count']}/{summary['original_turn_count']} "
        f"({float(summary['mismatch_rate']) * 100:.2f}%)"
    )
    print(f"데이터 품질 판정: {summary['status']}")
    print(f"결과 폴더: {scenario}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-root", type=Path, default=PROJECT / "Sample")
    parser.add_argument("--turn-prefix", default="90001220220810050000")
    parser.add_argument("--output-bundle", type=Path,
                        default=PROJECT / "results/data_quality_audit_20261001/checked_sample")
    parser.add_argument("--show-summary", action="store_true",
                        help="이미 생성한 결과의 요약만 출력합니다. 파일을 변경하지 않습니다.")
    args = parser.parse_args()
    bundle = args.output_bundle.expanduser().resolve()
    if args.show_summary:
        if not (bundle / "scenario/data_quality_summary.json").is_file():
            parser.error(f"이 폴더에 생성 결과가 없습니다: {bundle}")
        print_summary(bundle)
        return
    manifest = prepare_copies(args.sample_root.expanduser().resolve(), args.turn_prefix, bundle)
    command = [sys.executable, str(PROJECT / "src/prepare_sumo_scenario.py"),
               "--sample-root", str(bundle / "inputs"), "--turn-prefix", args.turn_prefix,
               "--output-dir", str(bundle / "scenario")]
    with (bundle / "prepare.log").open("w", encoding="utf-8") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    metadata_path = bundle / "scenario/scenario_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["correction_status"] = manifest["status"]
    metadata["supplier_id_mapping_confirmed"] = False
    write_json(metadata_path, metadata)
    print_summary(bundle)


if __name__ == "__main__":
    main()
