# 자료 폴더

석천사거리(204820) 실험에 쓰는 자료는 모두 여기에 있다. 코드에서 경로는 `src/focused_data.py` 한 곳에만 적혀 있다.

## raw — AI Hub 원본 (읽기만 한다)

| 파일 | 내용 | 크기 |
|---|---|---|
| `network/1200012n.net.xml` | 부천 도로망 지도: 도로·차로·교차로·신호 연결 | 1.7MB |
| `network/exp.rou.xml` | 차종 정의: 승용·버스·트럭·이륜 | 1KB |
| `turn_counts/all_turn_xml_compressed_block.bin` | 회전교통량 XML 7,296개(19일 × 96개 15분 × 4차종)가 압축된 구간 그대로 | 21.5MB |
| `turn_counts/turn_block_metadata.json`, `turn_entries.json` | 위 구간에서 각 파일을 꺼낼 위치·크기·CRC32 | 1.8MB |
| `signals/<날짜>/*s.xml` | 15분 단위 신호 기록 1,824개(19일 × 96개). 12개 신호교차로가 한 파일에 같이 있다 | 56MB |
| `manifest.json` | 출처와 검사값 | |

- 출처: Google Drive에 있는 AI Hub 71572 라벨링 데이터 분할 압축(TL.z01~z04, TL.zip, 약 499GB) 안의 `신호최적화데이터/부천시/`. 압축 전체를 풀지 않고 필요한 구간만 읽어 왔다.
- 확인(2026-10-05): 신호 1,824개와 지도 2개는 Drive 원본 압축 목록의 크기·CRC32와 같다. 회전교통량은 꺼낼 때마다 CRC32를 검사한다.
- 회전교통량 한 줄의 뜻: "1분 동안, 이 교차로에서, 이 차종이, A 도로에서 B 도로로 몇 대". 교차로마다 따로 센 장부이고, 같은 차가 다음 교차로에서 어디로 갔는지는 이어져 있지 않다.

## derived — 원본에서 다시 만들 수 있는 표

| 파일 | 내용 | 다시 만드는 법 |
|---|---|---|
| `counts_204820.csv` | 204820 진입 도로에서 시작하는 회전 기록만 모은 표(30만 1,878줄, 185만 5,756대) | `.venv/bin/python src/focused_data.py` |
| `counts_204820_summary.json` | 위 표의 요약 | 같음 |
| `branch_inflow_477748652_01.csv` | 서쪽 접근로 4차로 가지로 들어가는 상류 회전 기록(시간대별 합) | 같음 |

`data/raw`와 `data/derived`는 Git에 넣지 않는다(`.gitignore`). 이 설명 파일만 Git에 넣는다.
