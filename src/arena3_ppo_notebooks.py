"""ARENA3 PPO 노트북 세 개를 만든다(묶음 검증·패키지·시험 칸은 검증된 S2B 노트북에서 가져온다).

사용: python src/arena3_ppo_notebooks.py   (저장소의 notebooks/ARENA3_PPO_*.ipynb를 다시 쓴다)
"""
import json
import sys
from pathlib import Path

repo = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
base = json.loads((repo / "notebooks/ARENA3_S2B_policy.ipynb").read_text())
src = lambda i: "".join(base["cells"][i]["source"])
zip_cell = src(2).replace('"ARENA3_S2_V1", "ARENA3 2단계 전용 ZIP이 아닙니다"', '"ARENA3_PPO_V1", "ARENA3 PPO 전용 ZIP이 아닙니다"') \
                 .replace("f\"| 2단계 설정 {package_manifest['stage2_config_sha256'][:12]}\"", "f\"| PPO 설정 {package_manifest['ppo_config_sha256'][:12]}\"")
assert '"ARENA3_PPO_V1"' in zip_cell and "ppo_config_sha256" in zip_cell, "ZIP 칸 바꾸기 실패"
packages_cell, tests_cell = src(3), src(4)
assert "EXPECTED" in packages_cell and "test_arena3*.py" in tests_cell


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


def markdown(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def notebook(cells):
    # Colab은 런타임 종류를 노트북마다 저장한다. "hm" = 높은 RAM(CPU 8개). G1 노트북에는 있었고 S2 노트북에는 없었다.
    metadata = {**base["metadata"], "colab": {**base["metadata"].get("colab", {}), "machine_shape": "hm"}}
    return {"cells": cells, "metadata": metadata, "nbformat": base["nbformat"], "nbformat_minor": base["nbformat_minor"]}


INTRO = {
    "P1": ("P1 (M4 출발, 순수 PPO)",
           "G1과 같이 **M4에서 출발**합니다. 처음 탐험도 G1과 같게(후보 2개일 때 M4 확률 0.75) 맞췄습니다.\n"
           "묻는 것: **같은 출발·같은 학습 상황에서 PPO가 DQN(G1)보다 나은가.**"),
    "P2": ("P2 (S2A_pre 출발, PPO 미세조정)",
           "같은 난수의 **S2A_pre(사전학습만 한 모델)에서 출발**합니다. 처음 탐욕 선택은 S2A_pre와 완전히 같습니다(맥에서 1,802지점 확인).\n"
           "묻는 것: **PPO 미세조정이 사전학습 위에 이득을 더하는가**(2단계 DQN 미세조정은 더하지 못했습니다)."),
}


def variant_notebook(variant):
    title, what = INTRO[variant]
    cells = [markdown(f"""
# ARENA3 PPO — {title}

교수님이 제안한 Stable-Baselines3 계열의 PPO를 지금까지와 같은 조건에서 시험합니다.
- 같은 것: 관측 119칸, 6모드 행동(갈림길에서만 고름), 정확히 정산한 지연 보상, G1과 같은 학습 상황, 같은 심판 84상황.
- 다른 것: 학습 규칙만 PPO(SB3 기본값: 클립 0.2, GAE 0.95, 에포크 10, 미니배치 64, 학습률 3e-4).
- {what}
- 예산: 난수마다 에피소드 448판(G1 수준). 시간이 아니라 **판 수**로 셉니다.

1. 런타임 유형을 **CPU + 높은 RAM**(CPU 8개)으로 고르고 **런타임 → 모두 실행**을 누르세요. 첫 칸이 CPU 수를 확인합니다. 다른 PPO 노트북과 **동시에** 돌리면 됩니다.
2. 순서: 실행 전 점검(진짜 SUMO 4판, 약 10분) → 학습 난수 3개 동시(약 10시간) → 심판 평가 252판(약 1.5시간) → 참고 판정.
3. 끊기면 처음 네 칸을 실행한 뒤 멈춘 칸부터 다시 실행하세요(또는 **모두 실행**). 끝난 단계는 건너뛰고, 학습은 체크포인트에서 이어집니다.
"""),
             code(f'''
from google.colab import drive
from pathlib import Path
import os, sys, json

drive.mount("/content/drive")
DRIVE = Path("/content/drive/MyDrive/FutureTraffic")
ARCHIVE = DRIVE / "FutureTraffic_ARENA3_PPO.zip"
A3 = DRIVE / "arena3"
VARIANT = "{variant}"
OUT = A3 / f"ppo_{{VARIANT}}"           # 런타임마다 다른 폴더(두 런타임이 같은 이름 폴더를 동시에 만들지 않게)
S2A_TRAIN = A3 / "stage2" / "S2A_train"  # P2의 출발 모델(S2A_pre)
WORKSPACE = Path("/content/FutureTraffic_ARENA3_PPO")
MIN_CPUS = 8
assert ARCHIVE.is_file(), f"묶음이 없습니다: {{ARCHIVE}}"
CPUS = os.cpu_count() or 1
print("CPU 개수:", CPUS, "| 방식:", VARIANT, "| 결과 폴더:", OUT)
assert CPUS >= MIN_CPUS, (f"CPU가 {{CPUS}}개입니다. 런타임 → 런타임 유형 변경에서 CPU와 '높은 RAM'을 고른 뒤 다시 실행하세요. "
                          "(경험 양은 판 수로 정해져 같지만, CPU 2개면 학습이 약 4배 오래 걸려 24시간 한도에 걸립니다.)")
if VARIANT == "P2":
    missing = [s for s in (42, 43, 44) if not (S2A_TRAIN / f"seed_{{s}}" / "pretrained" / "model.pt").exists()]
    assert not missing, f"S2A_pre 모델이 없습니다(난수 {{missing}}): {{S2A_TRAIN}}"
'''),
             code(zip_cell), code(packages_cell), code(tests_cell),
             code('''
# 1) 실행 전 점검: 진짜 SUMO로 에피소드 4판·갱신 2번(평가자 예열 1번 + PPO 1번)을 점검용 폴더에서 돌립니다(약 10분).
#    본 학습 폴더와 섞이지 않습니다.
SMOKE = OUT / "smoke"
if (SMOKE / "smoke.json").exists():
    print("이전 점검이 통과해 건너뜁니다:", colab.read_json(SMOKE / "smoke.json")["episodes"], "에피소드")
else:
    args = ["src/arena3_ppo.py", "smoke", "--variant", VARIANT, "--seed", "42", "--out", str(SMOKE),
            "--work", "/content/ppo_smoke"]
    if VARIANT == "P2":
        args += ["--init-root", str(S2A_TRAIN)]
    colab.run_live(args, WORKSPACE)
'''),
             code('''
# 2) 학습: 난수 42·43·44를 동시에(난수마다 에피소드 2판씩 병렬). 난수마다 448판에서 멈춥니다(약 10시간).
#    셀을 멈춰도 학습은 계속 돕니다. 런타임이 끊기면 처음 네 칸 → 이 칸을 다시 실행하면 체크포인트에서 이어집니다.
assert "colab" in globals(), "처음 네 칸(1~4)을 먼저 실행하세요. 런타임이 끊기면 변수와 작업 폴더가 사라집니다."
TRAIN = OUT / "train"
LOGS = OUT / "logs"


def command(seed):
    args = ["src/arena3_ppo.py", "train", "--variant", VARIANT, "--seed", str(seed), "--out", str(TRAIN),
            "--work", f"/content/ppo_{seed}"]
    return args + (["--init-root", str(S2A_TRAIN)] if VARIANT == "P2" else [])


def describe(job):
    s = colab.read_json(TRAIN / f"seed_{job[-2:]}" / "status.json")
    if not s:
        return None
    last = s.get("last_update") or {}
    return (f"에피소드 {s['episodes']}/{s['budget_episodes']}, 갱신 {s['updates']}, 선택 {s['choices']}, "
            f"버린 판 {s['discarded']}, KL {last.get('approx_kl')}, 끝남 {s['finished']}")


pids = colab.launch({f"{VARIANT}_seed{s}": command(s) for s in (42, 43, 44)}, WORKSPACE, LOGS)
code = colab.watch(pids, LOGS, 300, describe)
assert code == 0, f"{VARIANT} 학습에서 오류가 났습니다. 위 로그를 저에게 보내 주세요."
'''),
             code('''
# 3) 심판 평가: 최종 모델 3개 × 84상황 = 252판. 1단계와 같은 심판·같은 84상황입니다. 끊겨도 다시 실행하면 끝난 판은 건너뜁니다.
assert "colab" in globals(), "처음 네 칸(1~4)을 먼저 실행하세요. 런타임이 끊기면 변수와 작업 폴더가 사라집니다."
TRAIN = OUT / "train"  # 학습 칸(2)을 건너뛰고 이 칸만 실행해도 되도록 여기서도 정한다
missing = [s for s in (42, 43, 44) if not (TRAIN / f"seed_{s}" / "final" / "model.pt").exists()]
assert not missing, f"최종 모델이 아직 없습니다(난수 {missing}). 2)를 다시 실행하세요."
colab.run_live(["src/arena3_ppo.py", "evaluate", "--variant", VARIANT, "--train", str(TRAIN), "--out", str(OUT / "eval"),
                "--work", "/content/ppo_eval", "--workers", str(CPUS)], WORKSPACE)
print(f"{VARIANT} 평가 끝. 다른 PPO 노트북도 끝나면 ARENA3_PPO_judge를 실행하세요.")
'''),
             code('''
# 4) 참고 판정: 이 방식과 기존 후보(G1·S2A_pre·S2A·S2B_pre·S2B)를 같은 심판으로 봅니다(SUMO 없음, 몇 분).
#    최종 판정과 C1·C2 확인은 ARENA3_PPO_judge 노트북에서 합니다.
assert "colab" in globals(), "처음 네 칸(1~4)을 먼저 실행하세요."
from IPython.display import display, Markdown
S2 = A3 / "stage2"
args = ["src/arena3_ppo.py", "judge", "--baseline", str(A3 / "referee" / "baseline"),
        "--ppo", f"{VARIANT}={OUT / 'eval'}", "--reference", f"G1={A3 / 'referee' / 'G1'}", "--out", str(OUT / "judge")]
for n in ("S2A_pre", "S2A", "S2B_pre", "S2B"):
    args += ["--reference", f"{n}={S2 / 'eval' / n}"]
colab.run_live(args, WORKSPACE, check=False)
if (OUT / "judge" / "summary.md").exists():
    display(Markdown((OUT / "judge" / "summary.md").read_text()))
''')]
    return notebook(cells)


def judge_notebook():
    cells = [markdown("""
# ARENA3 PPO — 판정·확인

**P1·P2 노트북이 끝난 뒤** 새 런타임 하나에서 실행합니다.
1. 개발 판정: P1·P2와 기존 후보(G1·S2A_pre·S2A·S2B_pre·S2B)를 1단계와 같은 심판·같은 84상황으로 M4와 비교합니다. 짝 비교(P1 − G1, P2 − S2A_pre 등)도 함께 냅니다.
2. 확인: 개발 판정을 **통과한 PPO**(P1·P2 중 날짜 평균이 큰 하나)만 새 배치 C1·C2 84상황에서 다시 잽니다.
   - 기준선(M4·P0·W4)은 2단계 확인 때 돌린 것을 그대로 씁니다. 후보 252판만 돕니다(CPU 8개 기준 약 1.5시간).
   - 통과한 PPO가 없으면 SUMO를 돌리지 않고 끝납니다. 승자는 처음 한 번 고정되어, 다시 실행해도 바뀌지 않습니다.
3. P2만 돌렸다면 첫 칸의 `RUN`을 `("P2",)`로 바꾸세요.
"""),
             code('''
from google.colab import drive
from pathlib import Path
import os, sys, json

drive.mount("/content/drive")
DRIVE = Path("/content/drive/MyDrive/FutureTraffic")
ARCHIVE = DRIVE / "FutureTraffic_ARENA3_PPO.zip"
A3 = DRIVE / "arena3"
RUN = ("P1", "P2")                     # 실제로 돌린 PPO 방식(P2만 돌렸다면 ("P2",))
JUDGE = A3 / "ppo_judge"
WORKSPACE = Path("/content/FutureTraffic_ARENA3_PPO")
assert ARCHIVE.is_file(), f"묶음이 없습니다: {ARCHIVE}"
CPUS = os.cpu_count() or 1
print("CPU 개수:", CPUS, "| 판정 폴더:", JUDGE)
if CPUS < 8:
    print(f"주의: CPU가 {CPUS}개입니다. 확인 252판이 오래 걸립니다. 가능하면 CPU + 높은 RAM 런타임을 쓰세요.")
'''),
             code(zip_cell), code(packages_cell), code(tests_cell),
             code('''
# 1) 개발 판정(84상황, M4 대비). 아직 평가가 안 끝난 PPO가 있으면 멈춥니다.
from IPython.display import display, Markdown
S2 = A3 / "stage2"
args = ["src/arena3_ppo.py", "judge", "--baseline", str(A3 / "referee" / "baseline"), "--out", str(JUDGE),
        "--reference", f"G1={A3 / 'referee' / 'G1'}"]
for v in RUN:
    args += ["--ppo", f"{v}={A3 / f'ppo_{v}' / 'eval'}"]
for n in ("S2A_pre", "S2A", "S2B_pre", "S2B"):
    args += ["--reference", f"{n}={S2 / 'eval' / n}"]
code = colab.run_live(args, WORKSPACE, check=False)
assert code == 0, "아직 평가가 끝나지 않은 PPO가 있습니다(위 '아직 평가 중'). P1·P2 노트북이 끝난 뒤 다시 실행하세요."
display(Markdown((JUDGE / "summary.md").read_text()))
'''),
             code('''
# 2) 확인(C1·C2 84상황): 통과한 PPO 중 날짜 평균이 가장 큰 하나만 다시 잽니다.
assert "colab" in globals(), "처음 네 칸(1~4)을 먼저 실행하세요. 런타임이 끊기면 변수와 작업 폴더가 사라집니다."
from IPython.display import display, Markdown
CONFIRM_BASE = A3 / "stage2" / "confirm" / "baseline"   # 2단계 확인 때 돌린 M4·P0·W4(C1·C2) 252판
colab.run_live(["src/arena3_ppo.py", "confirm", "--judge", str(JUDGE), "--baseline", str(CONFIRM_BASE),
                "--out", str(JUDGE / "confirm"), "--work", "/content/ppo_confirm", "--workers", str(CPUS)], WORKSPACE)
result = colab.read_json(JUDGE / "confirm" / "confirmation.json")
print("확인 결과:", result["status"], "| 승자:", result["selection"]["winner"] if "selection" in result else None)
if (JUDGE / "confirm" / "summary.md").exists():
    display(Markdown((JUDGE / "confirm" / "summary.md").read_text()))
''')]
    return notebook(cells)


out = repo / "notebooks"
for name, nb in (("ARENA3_PPO_P1.ipynb", variant_notebook("P1")), ("ARENA3_PPO_P2.ipynb", variant_notebook("P2")),
                 ("ARENA3_PPO_judge.ipynb", judge_notebook())):
    (out / name).write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n")
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"{name}[{i}]", "exec")
    print(name, len(nb["cells"]), "cells, syntax ok")
