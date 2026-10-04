"""DQN 평가에서 미도착한 차량의 차로·다음 도로·신호·차로 변경 상태를 기록한다."""
import argparse
import json
from pathlib import Path
from rl_environment import make_environment
from stable_baselines3 import DQN


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scenario-dir',type=Path,required=True)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--seed',type=int,default=43)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();model=DQN.load(args.model)
    env=make_environment(seed=args.seed,scenario_dir=args.scenario_dir,record_per_second_queue=True,additional_sumo_cmd='--no-step-log true')
    trail=[];last={}
    try:
        obs,_=env.reset(seed=args.seed)
        for step in range(360):
            action,_=model.predict(obs,deterministic=True)
            obs,_,term,trunc,_=env.step(int(action))
            if env.sumo.simulation.getTime()>=1200:
                snapshot=[]
                for name in env.sumo.vehicle.getIDList():
                    v=env.sumo.vehicle;route=list(v.getRoute(name));idx=v.getRouteIndex(name)
                    row={'id':name,'lane':v.getLaneID(name),'position':v.getLanePosition(name),
                         'speed':v.getSpeed(name),'waiting_sec':v.getWaitingTime(name),
                         'route_index':idx,'next_edge':route[idx+1] if idx+1<len(route) else None,
                         'next_tls':v.getNextTLS(name),'leader':v.getLeader(name,100),
                         'left':v.getLaneChangeStatePretty(name,1),'right':v.getLaneChangeStatePretty(name,-1)}
                    snapshot.append(row);last[name]=row
                if step%12==11:trail.append({'time':env.sumo.simulation.getTime(),'vehicles':snapshot})
            if term or trunc:break
        remaining=[last[n] for n in env.sumo.vehicle.getIDList()]
        result={'final_time':env.sumo.simulation.getTime(),'seed':args.seed,'remaining':remaining,
                'trail':trail,'interpretation':'diagnostic replay; separate from evaluation'}
        args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2))
        print(json.dumps(remaining,ensure_ascii=False,indent=2))
    finally:env.close()


if __name__=='__main__':main()
