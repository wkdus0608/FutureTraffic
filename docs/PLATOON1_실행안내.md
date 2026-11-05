# PLATOON1 실행 안내 (1단계: 무리 도착 세계에서 정책 반복 1라운드)

- **무엇:** PLATOON0(진행 기록 241절)에서 동쪽 차가 이웃 신호(204822)에 맞춰 무리로 오는 세계(P)에서는 이웃 시간표를 쓰는
  계획기 TM4가 M4보다 3.82초 짧았다. 이번에는 그 정보를 보는 **강화학습(정책 반복, PI R1과 같은 방법)** 학생이 TM4보다
  더 짧아지는지(학습의 몫) 잰다. 기준은 `results/explore/platoon1_20261101/사전등록.md`(실행 전 고정).
- **자료:** TM4가 학습 날짜의 P 세계 64판(PI R1과 같은 상황)을 운전한다. 고를 수 있는 순간의 25%에서 상태를 저장하고,
  허용 녹색을 모두 무리 규칙으로 만든 표본 미래 8개 × 300초로 굴려 짝 비교한다(뒤는 저장 기록을 이어받은 TM4).
- **학생:** 관측 167칸(ARENA3 119칸 + TM4 녹색·녹색별 계획 비용 차·동쪽 예측 도착·204822 시간표 48칸) → 녹색 4개의 TM4 대비 값.
  학습 전에는 TM4와 똑같이 움직인다.
- **판정:** P 세계 평가 84상황. 기준 M4·TM4·OM4·P1M4는 PLATOON0 결과를 다시 돌리지 않고 쓴다.
  PT1이 M4 대비 ARENA3 판정을 통과하고 L = PT1 − TM4가 0.3초 이상·t95 하한 > 0·양수 날짜 5개 이상이면 통과.

## 순서
1. Drive `FutureTraffic/`에 `FutureTraffic_PLATOON1.zip`과 노트북 네 개가 있는지 확인한다. `explore_platoon0/`(PLATOON0 결과)는 읽기만 한다.
2. 자료 모으기: `PLATOON1_data.ipynb`(조각 0), `PLATOON1_data_shard1.ipynb`, `PLATOON1_data_shard2.ipynb`를 **런타임 하나에 파일 하나씩**
   열고(CPU + 높은 RAM) 각각 **모두 실행**. 칸 순서: 묶음 검증 → 패키지 → 시험 → 실행 전 점검(약 20~30분) → 자료 모으기(백그라운드)
   → 진행 확인. 세 런타임이면 약 11~16시간(어림, 점검 칸이 실제 값으로 다시 어림).
3. 마지막 칸에 64/64가 나오면 `PLATOON1_learn.ipynb`를 런타임 하나에서 **모두 실행**: 날짜 보류 진단(참고) → 학생 학습 → 평가 252판
   (약 3시간) → 판정 → 확인(판정 통과 때만, 약 4~5시간).
4. 끊기면 다시 **모두 실행**한다. 끝난 판(`cache.json`이 있는 폴더)은 건너뛴다. 같은 조각을 두 런타임에서 돌리지 않는다.

## 결과
- 자료: Drive `FutureTraffic/explore_platoon1/data/runs/PT1_DATA/<상황>/`(points.jsonl.gz·metrics.json), 진행 `data/status/`.
- 학생: `explore_platoon1/students/seed_42|43|44/`, 날짜 보류 진단 `explore_platoon1/offline/`.
- 평가·판정: `explore_platoon1/eval/PT1/`, `explore_platoon1/judge/judge.json`·`summary.md`, 확인 `judge/confirm/`.
- 맥에서 다시 계산: `.venv/bin/python src/explore_platoon1.py judge --eval <받은 eval/PT1> --platoon0 <PLATOON0 폴더> --out <폴더>`
