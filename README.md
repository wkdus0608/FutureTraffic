# 석천사거리 신호 비교 실험 (AI Hub 교통량 + SUMO + DQN)

**질문: 교통 수요와 운전자 조건을 고정하고 신호제어만 바꾸면 무엇이 달라지는가?**

AI Hub 71572의 부천 교통량으로 석천사거리(신호 번호 `204820`)를 SUMO 가상 도로에 만들고, 같은 차량·같은 운전자 조건에서 신호만 바꿔 비교한다.

| 비교하는 신호 | 뜻 |
|---|---|
| 고정신호 | AI Hub에 기록된 실제 운영 신호를 그대로 재생 |
| 감응식 | 실제 신호의 녹색 4개를 차량 감지에 따라 늘이고 줄임(참고 기준) |
| DQN | 강화학습이 5초마다 녹색 4개 중 하나를 고름(보행 시간 등 운영 규칙은 지킴) |

결론의 범위는 "AI Hub 교통량으로 만든 SUMO 가상 도로에서의 비교"다. 실제 도로 성능을 주장하지 않는다.

## 지금 단계 (2026-10-05)

- 석천사거리 단독 모형으로 진행한다. 12개 교차로 전체 모형은 보류(포기 아님).
- 가상 도로 점검 완료: 학습·검증 15일을 하루씩 끊지 않고 돌려도 교착·충돌·순간이동 없이 실제 통과량을 따라간다.
- 비교 전에 고칠 것이 남아 있다([전체 점검](docs/석천사거리_전체점검_20261005.md) 5.2절). **고치기 전에는 본 학습·평가를 하지 않는다.**
- 쓰지 않는 코드·자료·예전 결과는 프로젝트 밖 `../자주프_보관_20261005/`로 옮겼다(지우지 않음).

## 폴더

```
config/   driver_model.json(운전자 설정) · focused_split.json(학습13·검증2·시험4일 분할)
data/     raw(AI Hub 원본: 지도·회전교통량·신호) · derived(원본에서 다시 만드는 표) — data/README.md
src/      focused_data.py      자료 읽기
          focused_scenario.py  차량 파일 만들기 · 실제 신호 복원
          focused_sumo.py      SUMO 실행 도구
          focused_day.py       하루 연속 환경 점검(고정신호)
          focused_rl.py        DQN이 보는 것·행동 규칙·보상
          focused_experiment.py  DQN 학습 · 정책 평가 · 보고
          focused_view.py      SUMO 화면으로 장면 보기
          focused_package.py   Colab용 묶음(zip) 만들기
tests/    자동 검사 16개
notebooks/ FutureTraffic_Colab.ipynb(Colab에서 설치·검사·재현성 확인)
docs/     설계 쉬운 설명 · 전체 점검 · 가상 도로 정상화 · 중간보고
results/  focused_204820/day/check(15일 환경 점검 결과) · claude_audit_20261005(점검 기록)
```

## 자주 쓰는 명령

```bash
.venv/bin/python -m unittest discover -s tests
```

```bash
.venv/bin/python src/focused_day.py check --dates 20220811 --workers 1
```

```bash
.venv/bin/python src/focused_data.py
```

- 첫째: 자동 검사(약 50초). 둘째: 실제 운영 신호로 하루를 돌려 가상 도로를 점검(하루 약 1~2분). 셋째: 원본에서 `data/derived` 표를 다시 만든다.
- 시험 날짜(08-10, 10-28, 08-21, 09-10)는 설정을 고르는 데 쓰지 않는다.

## SUMO 화면으로 보기 (이 맥에서)

```bash
.venv/bin/python src/focused_view.py --date 20220811 --start 0800
```

- 실험과 같은 차량·신호로 장면을 만들어 SUMO 창을 연다. 화면 시계는 실제 시각이고, 시작 10여 분 전부터 차를 넣어 대기열이 있는 상태로 시작 시각에서 멈춘다. ▶(Run)을 누르면 이어진다.
- `--start 1800`처럼 시각을 바꾸고, `--policy actuated`로 감응식을 본다. DQN 장면은 학습한 모델이 생기면 추가한다.
- 창이 열리지 않으면 XQuartz를 한 번 실행한 뒤 다시 시도한다.

## Colab에서 돌리기

1. 맥에서 `.venv/bin/python src/focused_package.py` → `results/colab/FutureTraffic_focused.zip`(약 25MB)
2. 이 zip을 Google Drive `내 드라이브/FutureTraffic/`에 올린다.
3. `notebooks/FutureTraffic_Colab.ipynb`를 Colab에서 열고(런타임: CPU) 위에서부터 실행한다.
- 코드를 고칠 때마다 1~2를 다시 한다. 한 번의 비교 실험(고정·감응식·DQN)은 같은 곳(모두 Colab 또는 모두 맥)에서 돌린다.

## 문서

- [종합 보고](docs/종합보고_20261005.md): 처음 문제부터 지금까지, 그리고 코드가 정한 규칙 전체
- [설계 쉬운 설명](docs/설계_쉬운설명_20261005.md): 자료·코드 흐름을 예시와 함께 설명
- [전체 점검](docs/석천사거리_전체점검_20261005.md): 자료·지도·코드 점검 결과와 남은 일
- [가상 도로 정상화](docs/가상도로_정상화_20261004.md): 신호 복원·차로 변경·갈래·운전자 설정을 정한 근거
- [중간보고](docs/중간보고_20261005.md), [진행 기록](PROJECT_PROGRESS.md)(가장 자세한 기록, 작업 전에 먼저 읽는다)
