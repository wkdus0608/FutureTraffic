# JOINT2 실행 안내 (0단계: 두 곳 함께 예측 제어로 여지 재기)

- 무엇: 석천사거리와 동쪽 이웃(204822)을 함께 예측 제어하면, 석천사거리만 바꾼 기존 P2보다 도로망 지연이 줄어드는지 잰다.
  학습은 없다. 기준은 `results/explore/joint2_20261031/사전등록.md`(실행 전 고정), 근거는 `docs/12개교차로_제어_검토안_20261031.md` 4절.
- 정책 4개: 보정 고정신호, 기존 P2, E(석천사거리만, 이웃 시간표를 정확히 앎), J2(이웃도 예측 제어).
  이웃 제어는 실제 신호와 같은 녹색 4개·순서·황색 4초를 지키고 녹색 길이만 그 날 실제 길이의 5~95% 범위에서 바꾼다.
- 규모: 12개 지도 검증 2일 46조각 × seed 5개 × 정책 4개 = 920번. Colab CPU 8개로 약 6~8시간.

## 순서
1. Drive `FutureTraffic/`에 `FutureTraffic_JOINT2.zip`과 `JOINT2_Colab.ipynb`가 있는지 확인한다(`FutureTraffic_network12.zip`, `explore_pred/network12/`는 기존 것을 읽기만 한다).
2. 노트북을 열고 런타임 유형을 **CPU, 높은 RAM(CPU 8개)** 으로 고른 뒤 **모두 실행**.
3. 칸 순서: 압축 검증 → 패키지·지문 확인 → 단위 시험 → 작은 실행(학습 날짜 한 조각, 이웃 규칙 확인) → 본 실행 → 판정.
4. 끊기면 다시 모두 실행한다. 끝난 실행(`metrics.json`이 있는 폴더)은 건너뛴다.

## 결과
- Drive `FutureTraffic/explore_joint2/9d3b7504064b/windows/<날짜>_<시>/<정책>_eval<seed>/metrics.json`
- 판정 `FutureTraffic/explore_joint2/report.json`: 재현 점검(이번 P2 = 기존 P2), 정책별 평균, J2 대 P2 판정.
