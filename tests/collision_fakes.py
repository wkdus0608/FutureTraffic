"""충돌 재현 도구 시험용 가짜 SUMO 연결·환경(SUMO 없음). test_collision_tool·test_network_collision이 함께 쓴다."""
from types import SimpleNamespace

COLLISION_T = 100.0
ROUTE = ["e0", "e1", "e2", "e3", "e4"]


def fake_connection(clock: dict) -> SimpleNamespace:
    """두 차: 피해 차 b는 교차로 :J_ 안에 멈춰 있고, 가해 차 a는 95초에 교차로에 들어온다. 다른 차 c는 교차로 밖."""
    def lane(v):
        if v == "b":
            return ":J_1_0"
        if v == "a":
            return ":J_2_0" if clock["t"] >= 95 else "e1_0"
        return "x_0"
    return SimpleNamespace(
        vehicle=SimpleNamespace(
            getIDList=lambda: ["a", "b", "c"], getLaneID=lane, getTypeID=lambda v: "passenger",
            getPosition=lambda v: (1.0, 2.0), getLanePosition=lambda v: 3.0,
            getSpeed=lambda v: 0.0 if v == "b" else 5.0, getAcceleration=lambda v: 0.0, getAngle=lambda v: 90.0,
            getLength=lambda v: 5.0, getNextTLS=lambda v: [("204820", 0, 10.0, "r")], getJunctionFoes=lambda v, d: [],
            getRoute=lambda v: ROUTE, getRouteIndex=lambda v: 1),
        trafficlight=SimpleNamespace(getRedYellowGreenState=lambda tls: "rG"),
        edge=SimpleNamespace(getLastStepVehicleNumber=lambda e: 4, getLastStepHaltingNumber=lambda e: 3),
        lane=SimpleNamespace(getLength=lambda lane_id: 25.0))


class LocalView:
    """12개 지도의 감싼 연결 흉내: 기록 장치가 이것으로 읽으면 실패한다(감싸지 않은 _conn으로 읽어야 함)."""

    def __init__(self, conn):
        self._conn = conn

    def __getattr__(self, name):
        raise AssertionError(f"LocalView.{name}을 읽었다 — 감싸지 않은 연결로 읽어야 한다")


class FakeEnv:
    def __init__(self, wrap: bool = False):
        self.sim_step, self.clock = 0.0, {"t": 0.0}
        conn = fake_connection(self.clock)
        self.sumo = LocalView(conn) if wrap else conn
        self.steps = 0

    def _sumo_step(self):
        self.sim_step += 1
        self.clock["t"] = self.sim_step
        self.steps += 1


def write_collisions(run_dir, events) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    body = "".join("<collision " + " ".join(f'{k}="{v}"' for k, v in e.items()) + "/>" for e in events)
    (run_dir / "collisions.xml").write_text(f"<collisions>{body}</collisions>")


def run_until_stopped(env, limit: int = 1000) -> None:
    """평가 반복 흉내: 기록 장치가 멈출 때까지 1초씩. 끝까지 가면 그대로 돌아온다(충돌 시각 전에 끝난 경우)."""
    for _ in range(limit):
        env._sumo_step()
