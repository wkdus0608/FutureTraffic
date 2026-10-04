"""원본을 보존하고 lcStrategic=1.5의 별도 검증 시나리오를 만든다.

차로 변경을 일찍 계획하는 시뮬레이션 가정이며, 도로망 수리/현장 보정이 아니다.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET


def prepare(source: Path, output: Path):
    if output.exists():
        raise FileExistsError(f"기존 결과를 덮어쓰지 않습니다: {output}")
    if source in output.parents or output in source.parents:
        raise ValueError("원본과 출력 폴더는 서로 포함할 수 없습니다.")
    before={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in source.rglob('*') if p.is_file()}
    routes=list((source/'scenario').glob('traffic_*.rou.xml'))
    if len(routes)!=1:
        raise ValueError("차량 경로 파일이 정확히 하나여야 합니다.")
    network=source/'scenario/network.net.xml'
    if hashlib.sha256(network.read_bytes()).hexdigest()!='c041e7ccf0c389ea60425a5916ba36e3c89838177ce61b2ddf1d6229cf67223d':
        raise ValueError("이번 검증과 다른 도로망에는 자동 적용하지 않습니다.")
    tree=ET.parse(routes[0]);baseline=ET.parse(routes[0]).getroot()
    for v in tree.findall('vType'):
        if v.get('lcStrategic') not in (None,'1','1.0'):
            raise ValueError("이미 수정된 차로 변경 설정")
        v.set('lcStrategic','1.5')
    shutil.copytree(source,output)
    dest=output/'scenario'/routes[0].name
    tree.write(dest,encoding='utf-8',xml_declaration=True)
    new=ET.parse(dest).getroot()
    if [ET.tostring(v) for v in baseline.findall('vehicle')]!=[ET.tostring(v) for v in new.findall('vehicle')]:
        raise AssertionError("차량/경로/출발 조건이 변경됐습니다.")
    for a,b in zip(baseline.findall('vType'),new.findall('vType'),strict=True):
        attrs=dict(b.attrib);assert attrs.pop('lcStrategic')=='1.5';assert attrs==a.attrib
    result={'status':'separate simulation variant; not real-road calibration or topology repair',
            'source_bundle':str(source),'lcStrategic':1.5,'other_vtype_attributes_preserved':True,
            'vehicles_routes_departures_preserved':True,
            'network_sha256':hashlib.sha256(network.read_bytes()).hexdigest(),
            'route_file_sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),
            'selection_basis':'earlier isolated validation on seeds42/43; no held-out-date evidence',
            'source_doc':'https://sumo.dlr.de/docs/Definition_of_Vehicles,_Vehicle_Types,_and_Routes.html#lane-changing-models'}
    (output/'behavior_manifest.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    for name,sha in before.items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=sha:
            raise AssertionError("원본 보존 실패")
    print(output)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-bundle',type=Path,required=True)
    p.add_argument('--output-bundle',type=Path,required=True)
    args=p.parse_args();prepare(args.source_bundle.resolve(),args.output_bundle.resolve())
