"""12개 지도 학습 환경의 LocalView 검사(가짜 SUMO 연결, SUMO 실행 없음).

단독 모형은 모든 차가 석천사거리 구간 입구에서 출발한다(넘친 줄은 '못 들어온 차'). 12개 지도에서 '못 들어온 차'·'출발 대기'·
'누적 대기'가 단독 모형과 같은 뜻(석천사거리 앞에서 기다리는 것)이 되는지 본다."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import network_rl as nr  # noqa: E402
from focused_rl import MOVEMENT_LINKS  # noqa: E402

THROUGH = list(next(iter(MOVEMENT_LINKS)))            # 석천사거리 이동 하나를 지나는 경로


class FakeSimulation:
    def __init__(self):
        self.pending = ["local_wait", "far_wait"]
        self.time = 10.0

    def getPendingVehicles(self):
        return tuple(self.pending)

    def getTime(self):
        return self.time


class FakeVehicle:
    def __init__(self):
        self.depart_delay = {"local": 30.0, "far": 200.0, "up": 50.0, "other": 0.0}
        self.acc = {"local": 50.0, "far": 70.0, "up": 40.0, "other": 5.0}
        self.speed = {"up": 0.0, "other": 0.0}
        self.route = {"up": ["feeder"] + THROUGH, "other": ["feeder", "side"]}

    def getDepartDelay(self, v):
        return self.depart_delay[v]

    def getAccumulatedWaitingTime(self, v):
        return self.acc[v]

    def getSpeed(self, v):
        return self.speed[v]

    def getRoute(self, v):
        return self.route[v]


class FakeEdge:
    def __init__(self):
        self.on = {"entry": [], "feeder": []}

    def getLastStepVehicleIDs(self, e):
        return tuple(self.on.get(e, ()))

    def getLastStepHaltingNumber(self, e):
        return len(self.on.get(e, ()))


class FakeConnection:
    def __init__(self):
        self.simulation, self.vehicle, self.edge = FakeSimulation(), FakeVehicle(), FakeEdge()
        self.trafficlight = "tl"


class LocalViewTest(unittest.TestCase):
    def setUp(self):
        self.conn = FakeConnection()
        self.keep = {"local", "local_wait"}
        self.view = nr.LocalView(self.conn, self.keep, ["entry"], ["feeder"])

    def test_pending_only_from_204820_roads(self):
        self.assertEqual(self.view.simulation.getPendingVehicles(), ("local_wait",))
        self.assertEqual(self.view.simulation.getTime(), 10.0)               # 나머지는 그대로 넘긴다

    def test_depart_delay_only_for_local_origins(self):
        self.assertEqual(self.view.vehicle.getDepartDelay("local"), 30.0)
        self.assertEqual(self.view.vehicle.getDepartDelay("far"), 0.0)        # 멀리서 출발한 차의 출발 대기는 빼고

    def test_waiting_before_entry_is_removed(self):
        self.conn.edge.on["entry"] = ["far", "local"]
        self.view.track()                                                     # 진입 구간에 처음 보인 때: far 70, local 50
        self.conn.vehicle.acc = {"local": 80.0, "far": 95.0}
        self.assertEqual(self.view.vehicle.getAccumulatedWaitingTime("far"), 25.0)     # 95 − 70(이웃 신호 대기)
        self.assertEqual(self.view.vehicle.getAccumulatedWaitingTime("local"), 80.0)   # 구간에서 출발한 차는 그대로
        self.conn.vehicle.acc["far"] = 100.0
        self.view.track()                                                     # 다시 보여도 처음 값을 유지
        self.assertEqual(self.view.vehicle.getAccumulatedWaitingTime("far"), 30.0)
        self.assertEqual(self.view.trafficlight, "tl")

    def test_queue_above_the_entry_counts_as_waiting(self):
        self.conn.edge.on["feeder"] = ["up", "other"]                         # 둘 다 멈춤, other는 석천사거리로 안 감
        self.view.track()
        self.assertEqual(self.view.simulation.getPendingVehicles(), ("local_wait", "up"))
        self.conn.simulation.time = 25.0
        self.view.track()
        self.assertEqual(self.view.vehicle.getDepartDelay("up"), 15.0)        # 구간 위에서 기다린 초 = 출발 대기
        self.conn.edge.on = {"feeder": [], "entry": ["up"]}                   # 40초에 구간에 들어옴
        self.conn.simulation.time = 40.0
        self.view.track()
        self.assertEqual(self.view.simulation.getPendingVehicles(), ("local_wait",))
        self.assertEqual(self.view.vehicle.getDepartDelay("up"), 30.0)        # 기다린 30초는 그대로 남는다
        self.assertEqual(self.view.vehicle.getAccumulatedWaitingTime("up"), 0.0)       # 그 뒤 대기만 센다(40 − 40)


if __name__ == "__main__":
    unittest.main()
