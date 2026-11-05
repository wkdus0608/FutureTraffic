"""ARENA3 정책 반복 1라운드 노트북 두 개를 만든다(묶음 검증·패키지·시험 칸은 검증된 PPO 노트북에서 가져온다).

사용: python src/arena3_pi_notebooks.py   (저장소의 notebooks/ARENA3_PI_R1_*.ipynb를 다시 쓴다)
"""
import json
import sys
from pathlib import Path

repo = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
base = json.loads((repo / "notebooks/ARENA3_PPO_P1.ipynb").read_text())
src = lambda i: "".join(base["cells"][i]["source"])  # noqa: E731
zip_cell = src(2).replace('"ARENA3_PPO_V1", "ARENA3 PPO 전용 ZIP이 아닙니다"', '"ARENA3_PI_R1", "ARENA3 정책 반복 1라운드 전용 ZIP이 아닙니다"') \
                 .replace("f\"| PPO 설정 {package_manifest['ppo_config_sha256'][:12]}\"",
                          "f\"| 정책 반복 설정 {package_manifest['pi_config_sha256'][:12]}\"")
assert '"ARENA3_PI_R1"' in zip_cell and "pi_config_sha256" in zip_cell, "ZIP 칸 바꾸기 실패"
packages_cell, tests_cell = src(3), src(4)
assert "EXPECTED" in packages_cell and "test_arena3*.py" in tests_cell


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


def markdown(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def notebook(cells):
    # Colab은 런타임 종류를 노트북마다 저장한다. "hm" = 높은 RAM(CPU 8개).
    metadata = {**base["metadata"], "colab": {**base["metadata"].get("colab", {}), "machine_shape": "hm"}}
    return {"cells": cells, "metadata": metadata, "nbformat": base["nbformat"], "nbformat_minor": base["nbformat_minor"]}


SETUP = '''
from google.colab import drive
from pathlib import Path
import os, sys, json

drive.mount("/content/drive")
DRIVE = Path("/content/drive/MyDrive/FutureTraffic")
ARCHIVE = DRIVE / "FutureTraffic_ARENA3_PI.zip"
A3 = DRIVE / "arena3"
OUT = A3 / "pi" / "R1"
WORKSPACE = Path("/content/FutureTraffic_ARENA3_PI")
MIN_CPUS = 8
assert ARCHIVE.is_file(), f"묶음이 없습니다: {ARCHIVE}"
CPUS = os.cpu_count() or 1
'''


def data_notebook():
    cells = [markdown("""
# ARENA3 정책 반복 1라운드 — 자료 모으기 (런타임 3개 동시)

지금까지 성능을 실제로 올린 것은 'SUMO 안에서 같은 순간을 되돌려 후보를 비교한 자료'뿐이었습니다(B′ +0.95초, 그 흉내 S2A_pre +0.60초).
이번에는 그 방식을 **강화학습의 정책 반복**으로 돌립니다.
- 지금 정책 = **S2A_pre(난수 42)**. 이 정책이 학습 날짜의 새 상황 64판을 운전합니다.
- 고를 수 있는 순간마다 상태를 저장하고, **허용된 녹색을 모두** 같은 표본 미래 8개로 300초씩 굴려 봅니다(뒤는 지금 정책이 운전).
  - 갈림 지점(모드들이 다른 녹색을 냄, 시간당 약 29곳): 모두
  - 합의 지점(고를 수는 있는데 모드가 모두 같음, 시간당 약 59곳): 25%만
- 다음 노트북 `ARENA3_PI_R1_learn`에서 신경망이 '어느 녹색이 더 나은가'를 배우고, 1단계와 같은 심판으로 잽니다.

**실행 방법**
1. 런타임 유형을 **CPU + 높은 RAM**(CPU 8개)으로 고릅니다.
2. 이 노트북을 **런타임 3개**에서 엽니다. 첫 칸의 `SHARD`를 각각 **0, 1, 2**로 바꾸고(`SHARDS = 3`은 그대로) **런타임 → 모두 실행**을 누릅니다.
   런타임을 4개 쓸 수 있으면 네 곳 모두 `SHARDS = 4`로, `SHARD`는 0~3으로 하면 더 빨리 끝납니다.
3. 순서: 실행 전 점검(진짜 SUMO, 약 15분) → 자료 모으기(런타임 3개 기준 약 15~17시간, 4개면 약 11~13시간) → 진행 확인.
4. 끊기면 처음 네 칸을 실행한 뒤 멈춘 칸부터 다시 실행하세요(또는 **모두 실행**). 끝난 판은 건너뛰고, 끊긴 판만 처음부터 다시 합니다.
5. 마지막 칸에 **64/64**가 나오면 `ARENA3_PI_R1_learn`을 실행하세요.
"""),
             code(SETUP + '''
SHARD = 0          # 런타임마다 0, 1, 2 (런타임 4개면 0~3)
SHARDS = 3         # 모든 런타임에서 같은 값
BASE_ROOT = A3 / "stage2" / "S2A_train"      # 지금 정책 S2A_pre(난수 42)가 있는 곳
assert 0 <= SHARD < SHARDS, "SHARD는 0 이상 SHARDS 미만이어야 합니다."
print("CPU 개수:", CPUS, "| 조각:", SHARD, "/", SHARDS, "| 결과 폴더:", OUT / "data")
assert CPUS >= MIN_CPUS, (f"CPU가 {CPUS}개입니다. 런타임 → 런타임 유형 변경에서 CPU와 '높은 RAM'을 고른 뒤 다시 실행하세요. "
                          "(롤아웃을 CPU 수만큼 동시에 돌립니다. CPU 2개면 4배 오래 걸려 24시간 한도에 걸립니다.)")
assert (BASE_ROOT / "seed_42" / "pretrained" / "model.pt").exists(), f"지금 정책 모델이 없습니다: {BASE_ROOT}"
'''),
             code(zip_cell), code(packages_cell), code(tests_cell),
             code('''
# 1) 실행 전 점검(약 15분): S2A_pre로 학습 상황 하나를 그냥 돌린 것과, 상태를 저장·롤아웃하며 돌린 것이 소수점까지 같은지,
#    롤아웃 복원이 맞는지, 판당 시간이 얼마인지 봅니다. 결과는 이 런타임 전용 점검 폴더에만 씁니다.
SMOKE = OUT / "smoke" / f"shard{SHARD}"
previous = colab.read_json(SMOKE / "smoke.json")
if previous and previous.get("passed"):
    print("이전 점검이 통과해 건너뜁니다. 판당 약", round(previous["estimate_cpu_hours_per_episode"], 2), "CPU-시간")
else:
    colab.run_live(["src/arena3_pi.py", "smoke", "--out", str(SMOKE), "--work", "/content/pi_smoke",
                    "--base-root", str(BASE_ROOT), "--workers", str(CPUS)], WORKSPACE)
    s = colab.read_json(SMOKE / "smoke.json")
    print("점검 통과. 판당 약", round(s["estimate_cpu_hours_per_episode"], 2), "CPU-시간, 64판 합 약",
          round(s["estimate_cpu_hours_all"]), "CPU-시간")
'''),
             code('''
# 2) 자료 모으기: 이 런타임 몫(64판 중 SHARD번째부터 SHARDS칸씩). 셀을 멈춰도 계속 돕니다.
#    런타임이 끊기면 처음 네 칸 → 이 칸을 다시 실행하세요. 끝난 판은 건너뜁니다.
assert "colab" in globals(), "처음 네 칸(1~4)을 먼저 실행하세요. 런타임이 끊기면 변수와 작업 폴더가 사라집니다."
DATA = OUT / "data"
LOGS = OUT / "logs"
NAME = f"collect_shard{SHARD}"


def describe(name):
    s = colab.read_json(DATA / "status" / f"{name}.json")
    if not s:
        return None
    return (f"끝난 판 {s['done']}/{s['total']}, 롤아웃 중 {len(s['inflight'])}판, 실패 {len(s['failed'])}, "
            f"복원 불일치 {s['restore_mismatched']}/{s['restore_checked']}")


pids = colab.launch({NAME: ["src/arena3_pi.py", "collect", "--out", str(DATA), "--work", "/content/pi",
                            "--shard", str(SHARD), "--shards", str(SHARDS), "--base-root", str(BASE_ROOT),
                            "--workers", str(CPUS)]}, WORKSPACE, LOGS)
code = colab.watch(pids, LOGS, 600, describe)
assert code == 0, "자료 모으기에서 오류가 났습니다. 위 로그를 저에게 보내 주세요(다시 실행하면 끝난 판은 건너뜁니다)."
'''),
             code('''
# 3) 모든 런타임의 진행(언제든 다시 실행해 볼 수 있습니다).
for f in sorted((OUT / "data" / "status").glob("collect_shard*.json")):
    s = colab.read_json(f) or {}
    print(f.stem, f"{s.get('done')}/{s.get('total')}", "| 실패", len(s.get("failed", [])),
          "| 복원 불일치", f"{s.get('restore_mismatched')}/{s.get('restore_checked')}")
done = len(list((OUT / "data" / "runs" / "PI_DATA").glob("*/cache.json")))
print(f"끝난 판 {done}/64")
if done == 64:
    print("64판이 모두 끝났습니다. ARENA3_PI_R1_learn 노트북을 실행하세요.")
''')]
    return notebook(cells)


def learn_notebook():
    cells = [markdown("""
# ARENA3 정책 반복 1라운드 — 학생 학습·심판 평가·판정

`ARENA3_PI_R1_data`의 64판이 모두 끝난 뒤 **런타임 하나**(CPU + 높은 RAM)에서 **모두 실행**합니다.
1. 날짜 보류 진단(참고, 약 10분): 학습에 안 쓴 날짜에서 '더 나은 녹색'을 얼마나 맞히는지 봅니다. 판정·선택에는 쓰지 않습니다.
2. 학생 학습(약 5분): 같은 자료로 난수 42·43·44 세 신경망을 학습합니다. 입력은 관측 119칸뿐입니다.
3. 심판 평가: 세 변형 × 모델 3개 × 84상황 = 756판(CPU 8개 기준 약 5~6시간). 1단계와 같은 심판·같은 84상황입니다.
   - **R1M**: 갈림 지점에서 모드가 내는 녹색만(지금 행동 범위 안의 정책 반복)
   - **R1G**: 갈림 지점에서 허용 녹색 전부(새 행동)
   - **R1A**: 갈림·합의 지점 모두에서 허용 녹색 전부
4. 판정: M4 대비 판정 + 짝 비교(R1M − S2A_pre, R1G − R1M, R1A − R1G, 각 변형 − S2B, 각 변형 − B′).
5. 확인: 통과한 변형 중 날짜 평균이 가장 큰 하나만 C1·C2 84상황에서 다시 잽니다(약 2시간). 통과가 없으면 SUMO를 돌리지 않고 끝납니다.

빨리 끝내려면: 2번(학생 학습)이 끝난 뒤 다른 런타임 두 곳에서 이 노트북을 열고 첫 칸의 `VARIANTS`를 각각 `("G",)`, `("A",)`로,
이 런타임은 `("M",)`으로 바꿔 3번 칸까지 실행하세요. 4번(판정)·5번(확인)은 세 평가가 끝난 뒤 한 곳에서만 실행합니다.
"""),
             code(SETUP + '''
VARIANTS = ("M", "G", "A")      # 이 런타임에서 평가할 변형(나눠 돌릴 때만 바꿉니다)
print("CPU 개수:", CPUS, "| 결과 폴더:", OUT, "| 평가할 변형:", VARIANTS)
if CPUS < MIN_CPUS:
    print(f"주의: CPU가 {CPUS}개입니다. 평가가 오래 걸립니다. 가능하면 CPU + 높은 RAM 런타임을 쓰세요.")
done = len(list((OUT / "data" / "runs" / "PI_DATA").glob("*/cache.json")))
assert done == 64, f"자료가 64판 중 {done}판만 끝났습니다. ARENA3_PI_R1_data를 먼저 끝내세요."
'''),
             code(zip_cell), code(packages_cell), code(tests_cell),
             code('''
# 1) 날짜 보류 진단(참고, SUMO 없음).
OFFLINE = OUT / "offline"
if not (OFFLINE / "offline_report.md").exists():
    colab.run_live(["src/arena3_pi.py", "offline", "--data", str(OUT / "data"), "--out", str(OFFLINE)], WORKSPACE)
display(Markdown((OFFLINE / "offline_report.md").read_text()))
'''),
             code('''
# 2) 학생 학습(난수 42·43·44, SUMO 없음). 이미 있으면 같은 자료로 학습했는지만 확인하고 건너뜁니다.
assert "colab" in globals(), "처음 네 칸(1~4)을 먼저 실행하세요."
colab.run_live(["src/arena3_pi.py", "train", "--data", str(OUT / "data"), "--out", str(OUT / "students")], WORKSPACE)
'''),
             code('''
# 3) 심판 평가: 변형마다 모델 3개 × 84상황 = 252판. 끊겨도 다시 실행하면 끝난 판은 건너뜁니다.
assert "colab" in globals(), "처음 네 칸(1~4)을 먼저 실행하세요. 런타임이 끊기면 변수와 작업 폴더가 사라집니다."
missing = [s for s in (42, 43, 44) if not (OUT / "students" / f"seed_{s}" / "model.pt").exists()]
assert not missing, f"학생 모델이 없습니다(난수 {missing}). 2)를 먼저 실행하세요."
for v in VARIANTS:
    colab.run_live(["src/arena3_pi.py", "evaluate", "--variant", v, "--students", str(OUT / "students"),
                    "--out", str(OUT / "eval" / f"R1{v}"), "--work", f"/content/pi_eval_{v}", "--workers", str(CPUS)],
                   WORKSPACE)
    print(f"R1{v} 평가 끝")
'''),
             code('''
# 4) 판정(SUMO 없음, 몇 분). 세 변형의 평가가 모두 끝나야 합니다.
assert "colab" in globals(), "처음 네 칸(1~4)을 먼저 실행하세요."
S2 = A3 / "stage2"
args = ["src/arena3_pi.py", "judge", "--baseline", str(A3 / "referee" / "baseline"), "--out", str(OUT / "judge"),
        "--reference", f"BPRIME={A3 / 'referee' / 'bprime'}"]
for v in ("M", "G", "A"):
    args += ["--pi", f"R1{v}={OUT / 'eval' / f'R1{v}'}"]
for n in ("S2A_pre", "S2A", "S2B_pre", "S2B"):
    args += ["--reference", f"{n}={S2 / 'eval' / n}"]
code = colab.run_live(args, WORKSPACE, check=False)
assert code == 0, "아직 평가가 끝나지 않은 변형이 있습니다(위 '아직 평가 중'). 평가가 끝난 뒤 다시 실행하세요."
display(Markdown((OUT / "judge" / "summary.md").read_text()))
'''),
             code('''
# 5) 확인(C1·C2 84상황): 통과한 변형 중 날짜 평균이 가장 큰 하나만. 승자는 처음 한 번 고정되어, 다시 실행해도 바뀌지 않습니다.
assert "colab" in globals(), "처음 네 칸(1~4)을 먼저 실행하세요."
CONFIRM_BASE = A3 / "stage2" / "confirm" / "baseline"     # 2단계 확인 때 돌린 M4·P0·W4(C1·C2) 252판
colab.run_live(["src/arena3_pi.py", "confirm", "--judge", str(OUT / "judge"), "--baseline", str(CONFIRM_BASE),
                "--reference", f"S2B={A3 / 'stage2' / 'confirm' / 'S2B'}", "--out", str(OUT / "judge" / "confirm"),
                "--work", "/content/pi_confirm", "--workers", str(CPUS)], WORKSPACE)
result = colab.read_json(OUT / "judge" / "confirm" / "confirmation.json")
print("확인 결과:", result["status"], "| 승자:", (result.get("selection") or {}).get("winner"))
if (OUT / "judge" / "confirm" / "summary.md").exists():
    display(Markdown((OUT / "judge" / "confirm" / "summary.md").read_text()))
''')]
    return notebook(cells)


SHARD_INTRO = """
# ARENA3 정책 반복 1라운드 — 자료 모으기, 조각 {k} / 3

**이 파일 하나 = 런타임 하나입니다.** Colab은 같은 노트북 파일을 탭 여러 개로 열어도 런타임 하나에 붙어서, 조각마다 파일을 따로 둡니다.
- 조각 0 = `ARENA3_PI_R1_data.ipynb`, 조각 1 = `ARENA3_PI_R1_data_shard1.ipynb`, 조각 2 = `ARENA3_PI_R1_data_shard2.ipynb`
- 세 파일을 하나씩 열어(런타임 유형 **CPU + 높은 RAM**) 각각 **런타임 → 모두 실행**을 누르세요. 첫 칸의 `SHARD`는 이미 {k}로 정해져 있으니 바꾸지 마세요.
- 같은 조각을 두 런타임에서 동시에 돌리지 마세요(같은 결과 파일을 서로 덮어씁니다).

하는 일과 순서는 조각 0 파일과 같습니다: 실행 전 점검(약 15분) → 자료 모으기(약 15~17시간) → 진행 확인.
끊기면 다시 **모두 실행**하면 끝난 판은 건너뜁니다. 마지막 칸에 **64/64**가 나오면(세 조각이 다 끝난 것) `ARENA3_PI_R1_learn`을 실행하세요.
"""


def shard_notebook(generic: dict, k: int) -> dict:
    """조각 0 노트북과 같고, 소개와 SHARD 값만 다르다."""
    nb = json.loads(json.dumps(generic))
    nb["cells"][0] = markdown(SHARD_INTRO.format(k=k))
    first = "".join(nb["cells"][1]["source"])
    line = "SHARD = 0          # 런타임마다 0, 1, 2 (런타임 4개면 0~3)"
    assert first.count(line) == 1, "SHARD 줄 바꾸기 실패"
    nb["cells"][1]["source"] = first.replace(line, f"SHARD = {k}          # 이 파일 = 조각 {k}(바꾸지 마세요)") \
                                    .splitlines(keepends=True)
    return nb


out = repo / "notebooks"
generic = data_notebook()
for name, nb in (("ARENA3_PI_R1_data.ipynb", generic), ("ARENA3_PI_R1_data_shard1.ipynb", shard_notebook(generic, 1)),
                 ("ARENA3_PI_R1_data_shard2.ipynb", shard_notebook(generic, 2)), ("ARENA3_PI_R1_learn.ipynb", learn_notebook())):
    (out / name).write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n")
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"{name}[{i}]", "exec")
    print(name, len(nb["cells"]), "cells, syntax ok")
