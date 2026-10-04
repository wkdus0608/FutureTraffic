"""직접 연결 없는 자동차 차로에서 차로 변경 후 출구까지 주행 가능한지 검사한다.

별도 진단 차량 96대를 사용한다. 실제 교통량 또는 현장 통행 허용 검증은 아니다.
"""
from collections import defaultdict, deque
import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET
from rl_environment import configure_sumo
from audit_network_structure import CLASSES, permits


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--network',type=Path,required=True)
    p.add_argument('--audit',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    audit=json.loads(args.audit.read_text());root=ET.parse(args.network).getroot()
    edges={e.get('id'):e for e in root.findall('edge')};lanes={l.get('id'):l for e in edges.values() for l in e.findall('lane')}
    connections=root.findall('connection'); graphs={k:defaultdict(set) for k in CLASSES}
    for c in connections:
        a,b=c.get('from'),c.get('to')
        if a.startswith(':') or b.startswith(':'):continue
        la=edges[a].find(f"lane[@index='{c.get('fromLane')}']");lb=edges[b].find(f"lane[@index='{c.get('toLane')}']")
        used=[la,lb]+([lanes[c.get('via')]] if c.get('via') else [])
        for kind in CLASSES:
            if all(permits(l,kind) for l in used):graphs[kind][a].add(b)
    targets=set(audit['boundary_exit_edges']);routes=ET.Element('routes');specs={}
    for kind in CLASSES:ET.SubElement(routes,'vType',id=kind,vClass=kind)
    for item in audit['lane_termination_review']:
        for kind in item['classes']:
            start=item['edge'];q=deque([[start]]);seen={start};path=None
            while q:
                route=q.popleft()
                if route[-1] in targets:path=route;break
                for nxt in sorted(graphs[kind][route[-1]]):
                    if nxt not in seen:seen.add(nxt);q.append(route+[nxt])
            if path is None:raise ValueError(f'no probe route {start}/{kind}')
            number=len(specs);name=f'probe_{number:03d}'
            vehicle=ET.SubElement(routes,'vehicle',id=name,type=kind,depart=str(number*3),departLane=str(lanes[item['lane']].get('index')),departSpeed='0')
            ET.SubElement(vehicle,'route',edges=' '.join(path))
            specs[name]={'start_lane':item['lane'],'vehicle_class':kind,'route':path,'observed_lateral_change':False,'departed':False,'arrived':False}
    route_file=args.output/'probe.rou.xml';ET.ElementTree(routes).write(route_file,encoding='utf-8',xml_declaration=True)
    configure_sumo();import traci
    connection=traci.start(['sumo','-n',str(args.network),'-r',str(route_file),'-e','1800','--seed','42','--time-to-teleport','-1','--no-step-log','true','--duration-log.disable','true','--tripinfo-output',str(args.output/'tripinfo.xml')],label='lane_probe')
    conn=traci.getConnection('lane_probe');collisions=teleports=0
    try:
        while conn.simulation.getTime()<1800 and conn.simulation.getMinExpectedNumber()>0:
            conn.simulationStep();collisions+=conn.simulation.getCollidingVehiclesNumber();teleports+=conn.simulation.getStartingTeleportNumber()
            for name in conn.simulation.getDepartedIDList():
                specs[name]['departed']=True;specs[name]['actual_depart_lane']=conn.vehicle.getLaneID(name)
            for name in conn.vehicle.getIDList():
                lane=conn.vehicle.getLaneID(name);s=specs[name]
                if conn.vehicle.getRoadID(name)==s['route'][0] and lane!=s['start_lane']:s['observed_lateral_change']=True
            for name in conn.simulation.getArrivedIDList():specs[name]['arrived']=True
        result={'vehicles':len(specs),'departed':sum(s['departed'] for s in specs.values()),'arrived':sum(s['arrived'] for s in specs.values()),'observed_lateral_changes':sum(s['observed_lateral_change'] for s in specs.values()),'departure_lane_matches':sum(s.get('actual_depart_lane')==s['start_lane'] for s in specs.values()),'collisions':collisions,'teleports':teleports,'finished_at_sec':conn.simulation.getTime(),'probes':specs,'scope':'synthetic lane access probe; not real-road legality or demand evaluation'}
        (args.output/'lane_probe_result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
        print(json.dumps({k:v for k,v in result.items() if k!='probes'},ensure_ascii=False))
    finally:conn.close()


if __name__=='__main__':main()
