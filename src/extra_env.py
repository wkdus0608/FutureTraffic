"""새 12일을 단독 모형에서 읽게 한다 — 지문 밖, 잠근 파일을 고치지 않는다(진행 기록 89~91절).

`activate()`는 실행할 때만, 단독 모형이 자료를 읽는 네 곳을 31일 자료로 바꿔 끼운다.
- 204820 회전 표: data/derived/counts_204820.csv → data/extra/derived/counts_204820_31days.csv
- 서쪽 갈래 유입 표: branch_inflow_477748652_01.csv → *_31days.csv
- 동쪽 1분 통과 표(무리 도착): east_upstream_204822.csv → *_31days.csv
- 신호 폴더: data/raw/signals → data/extra/raw/signals(교차로_26 파일, 31일)
31일 자료는 겹치는 19일에서 기존 자료와 행·신호까지 같다(90절, data/extra/verify_overlap.json). 그래서 기존 19일 결과는
바뀌지 않고, 새 12일만 더 읽을 수 있게 된다. 환경 지문은 잠근 파일 내용으로 계산하므로 그대로다.
"""
from __future__ import annotations

import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
EXTRA = PROJECT / "data/extra"
NEW_DATES = ["20220815", "20220912", "20220913", "20220914", "20220915", "20220916",
             "20220930", "20221001", "20221002", "20221026", "20221027", "20221031"]
FILES = {"counts": EXTRA / "derived/counts_204820_31days.csv",
         "branch": EXTRA / "derived/branch_inflow_477748652_01_31days.csv",
         "upstream": EXTRA / "derived/east_upstream_204822_31days.csv",
         "signals": EXTRA / "raw/signals"}
_ACTIVE = False


def activate() -> dict:
    """31일 자료로 바꿔 끼운다(여러 번 불러도 한 번만). 같은 자료 확인 기록이 없거나 다르면 멈춘다."""
    global _ACTIVE
    verify = json.loads((EXTRA / "verify_overlap.json").read_text())
    ok = all(verify[k]["same_on_19_days"] for k in ("counts_204820", "branch_inflow_477748652_01", "east_upstream_204822"))
    ok = ok and all(verify[k]["mismatch"] == 0 for k in ("signal_204820", "signal_204822"))
    missing = [str(p) for p in FILES.values() if not p.exists()]
    if not ok or missing:
        raise SystemExit(f"새 날짜 자료가 준비되지 않았습니다(같은 자료 확인 {ok}, 없는 파일 {missing}). extra_days.py extract·verify를 먼저.")
    if _ACTIVE:
        return dict(FILES)
    import focused_data as fd
    import focused_day
    import focused_platoon as fp
    import focused_scenario as fs
    fd.COUNTS = FILES["counts"]
    fd.BRANCH_INFLOW = FILES["branch"]
    fd.SIGNALS = FILES["signals"]
    fs.SIGNALS = FILES["signals"]
    focused_day.COUNTS = FILES["counts"]
    fp.UPSTREAM_COUNTS = FILES["upstream"]
    fp.load_upstream_counts.__defaults__ = (FILES["upstream"],)
    fd.load_branch_inflow.__defaults__ = (FILES["branch"],)
    if hasattr(fs, "_branch_inflow_cache") and hasattr(fs._branch_inflow_cache, "cache_clear"):
        fs._branch_inflow_cache.cache_clear()
    _ACTIVE = True
    return dict(FILES)
