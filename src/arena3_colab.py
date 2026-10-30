"""ARENA3 Colab 도우미 — 노트북 세 개가 같은 방식으로 명령을 돌리고, 병렬 작업을 띄우고, 진행을 보여 준다.

- run_live: 한 명령을 돌리며 출력을 그대로 보여 주고, 실패하면 멈춘다(Ctrl+C면 그 프로세스 묶음을 끝낸다).
- launch: 여러 명령을 각자 새 세션으로 띄우고 출력은 Drive의 로그 파일로 보낸다(셀을 멈춰도 계속 돈다).
  같은 이름의 작업이 이미 돌고 있으면 다시 띄우지 않는다(pid 파일).
- watch: 상태 파일(JSON)을 주기적으로 읽어 한 줄씩 보여 주고, 모든 작업이 끝나면 종료 코드를 돌려준다.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def env() -> dict:
    return {**os.environ, "FOCUSED_RULESET": "pet1", "FOCUSED_NORTH_EXT": "240", "PYTHONUNBUFFERED": "1",
            "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}


def run_live(args: list[str], cwd: Path, check: bool = True) -> int:
    process = subprocess.Popen([sys.executable, "-u", *args], cwd=cwd, env=env(), stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, start_new_session=True)
    try:
        for line in process.stdout:
            print(line.rstrip(), flush=True)
        code = process.wait()
    except KeyboardInterrupt:
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=20)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        raise
    if check and code != 0:
        raise RuntimeError(f"작업이 멈췄습니다(종료 코드 {code}). 위의 원인을 확인해 주세요.")
    return code


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    try:                                             # 끝났지만 거두지 않은 자식(좀비)은 끝난 것으로 본다
        done, _ = os.waitpid(pid, os.WNOHANG)
        return done == 0
    except ChildProcessError:
        return Path(f"/proc/{pid}").exists()


def launch(jobs: dict[str, list[str]], cwd: Path, logs: Path) -> dict[str, int]:
    """jobs = {이름: 명령 인수}. 이미 돌고 있는 이름은 건너뛴다. {이름: pid}"""
    logs = Path(logs)
    logs.mkdir(parents=True, exist_ok=True)
    pids = {}
    for name, args in jobs.items():
        pid_file = logs / f"{name}.pid"
        if pid_file.exists():
            old = int(pid_file.read_text().strip() or 0)
            if old and _alive(old):
                print(f"{name}: 이미 실행 중(pid {old})")
                pids[name] = old
                continue
        log = (logs / f"{name}.log").open("a", encoding="utf-8")
        log.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} 시작: {' '.join(args)}\n")
        log.flush()
        (logs / f"{name}.exit").unlink(missing_ok=True)
        wrapper = [sys.executable, "-u", str(Path(__file__).resolve()), "wrap", str(logs / f"{name}.exit"), "--", *args]
        p = subprocess.Popen(wrapper, cwd=cwd, env=env(), stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        pid_file.write_text(str(p.pid))
        pids[name] = p.pid
        print(f"{name}: 시작(pid {p.pid}) → 로그 {logs / (name + '.log')}")
    return pids


def tail(path: Path, n: int = 3) -> list[str]:
    try:
        lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    return lines[-n:]


def watch(pids: dict[str, int], logs: Path, every_s: int = 120, describe=None) -> int:
    """모든 작업이 끝날 때까지 every_s초마다 각 작업의 마지막 로그 줄(또는 describe(이름))을 보여 준다."""
    logs = Path(logs)
    while True:
        alive = {name: _alive(pid) for name, pid in pids.items()}
        print(time.strftime("%H:%M:%S"), "| 실행 중:", [n for n, a in alive.items() if a] or "없음", flush=True)
        for name in pids:
            info = describe(name) if describe else None
            print(f"  {name}: {info if info else ' / '.join(tail(logs / f'{name}.log', 1))}", flush=True)
        if not any(alive.values()):
            break
        time.sleep(every_s)
    codes = {name: (logs / f"{name}.exit").read_text().strip() if (logs / f"{name}.exit").exists() else "없음"
             for name in pids}
    failed = [name for name, code in codes.items() if code != "0"]
    print("종료 코드:", codes)
    for name in failed:
        print(f"----- {name} 로그 마지막 40줄 (종료 코드 {codes[name]})")
        print("\n".join(tail(logs / f"{name}.log", 40)))
    return 1 if failed else 0


def read_json(path: Path) -> dict | None:
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def _wrap(exit_file: str, args: list[str]) -> int:
    """launch가 띄우는 감싸기: 명령의 종료 코드를 파일로 남긴다(셀이 다시 붙어도 성공·실패를 알 수 있게)."""
    code = subprocess.call([sys.executable, "-u", *args])
    Path(exit_file).write_text(f"{code}\n")
    return code


if __name__ == "__main__":
    if len(sys.argv) > 3 and sys.argv[1] == "wrap" and sys.argv[3] == "--":
        raise SystemExit(_wrap(sys.argv[2], sys.argv[4:]))
    raise SystemExit("사용: python src/arena3_colab.py wrap <종료코드 파일> -- <명령 인수…>")
