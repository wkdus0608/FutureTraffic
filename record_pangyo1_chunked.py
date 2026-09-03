#!/usr/bin/env python3

"""Record the Pangyo-1 HLS stream in short chunks with fresh API URLs.

The ITS CCTV URL is session-based. This recorder fetches a fresh URL for each
chunk and keeps a wall-clock manifest so the video can be aligned with VDS
timestamps even when a chunk ends early.
"""

import argparse
import csv
import re
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path

import requests


from its_common import get_api_key

PROJECT_DIR = Path(__file__).resolve().parent
RECORDINGS_DIR = PROJECT_DIR / "recordings"

CCTV_NAME = "[경부선] 판교1"
VDS_ID = "0010VDS31400"

API_URL = "https://openapi.its.go.kr:9443/cctvInfo"
API_BOX = {
    "type": "all",
    "minX": "127.00",
    "maxX": "127.30",
    "minY": "37.30",
    "maxY": "37.60",
    "getType": "json",
}


def now_local():
    return datetime.now().astimezone()


def fetch_cctv(cctv_type):
    params = dict(API_BOX)
    params["cctvType"] = str(cctv_type)
    params["apiKey"] = get_api_key()

    response = requests.get(API_URL, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()["response"]["data"]

    candidates = [item for item in data if item.get("cctvname") == CCTV_NAME]
    if not candidates:
        raise RuntimeError(f"CCTV를 찾을 수 없습니다: {CCTV_NAME}")

    item = candidates[0]
    stream_url = item.get("cctvurl")
    if not stream_url:
        raise RuntimeError("CCTV 스트림 주소가 비어 있습니다.")

    return item, stream_url


def redact_urls(text):
    """FFmpeg 로그에서 임시 CCTV URL의 query string을 제거합니다."""
    return re.sub(
        r"(https?://[^?\s'\"]+)\?[^\s'\"]+",
        r"\1?<redacted>",
        text,
    )


def classify_ffmpeg_log(log_file):
    """FFmpeg stderr를 playlist/segment/네트워크 오류로 요약합니다."""
    text = log_file.read_text(encoding="utf-8", errors="replace")
    lower = text.lower()

    status_codes = sorted(
        set(
            re.findall(
                r"(?:http error|server returned)\s+(\d{3})",
                text,
                flags=re.IGNORECASE,
            )
        )
    )
    http_status = ",".join(status_codes)

    # HLS 개별 미디어 조각 오류는 playlist reload 문구와 구분합니다.
    if re.search(
        r"failed to (?:open|load) segment|error when loading segment|segment\s+\d+.*(?:404|not found)",
        lower,
    ):
        if "404" in status_codes:
            return "SEGMENT_404", http_status
        return "SEGMENT_ERROR", http_status

    # 현재 관찰된 404는 모두 playlist를 읽거나 다시 불러오는 단계에서 발생합니다.
    if "404" in status_codes and re.search(
        r"playlist|parsing playlist|error opening input",
        lower,
    ):
        return "PLAYLIST_404", http_status

    if "404" in status_codes:
        return "HTTP_404_UNCLASSIFIED", http_status
    if re.search(r"timed out|timeout|timedout", lower):
        return "TIMEOUT", http_status
    if re.search(
        r"connection reset|connection refused|network is unreachable|could not resolve|broken pipe",
        lower,
    ):
        return "NETWORK_ERROR", http_status
    if re.search(r"failed to reload playlist|error when loading", lower):
        return "HLS_ERROR", http_status
    if re.search(r"error|failed|invalid data|end of file", lower):
        return "OTHER_ERROR", http_status
    return "OK", http_status


def write_diagnostic_summary(summary_file, counts):
    with summary_file.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["오류분류", "조각수"])
        for category, count in sorted(counts.items()):
            writer.writerow([category, count])


def probe_duration(video_file):
    if not video_file.exists() or video_file.stat().st_size == 0:
        return 0.0

    result = subprocess.run(
        [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "csv=p=0",
            str(video_file),
        ],
        capture_output=True,
        text=True,
    )

    try:
        return float(result.stdout.strip())
    except ValueError:
        return 0.0


def record_one_chunk(stream_url, video_file, log_file, seconds):
    command = [
        "ffmpeg",
        "-hide_banner",
        "-y",
        "-http_persistent", "0",
        "-user_agent", "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "-headers", "Referer: https://www.its.go.kr/\r\n",
        "-rw_timeout", "15000000",

        # 입력 스트림
        "-i", stream_url,

        # 이번 조각의 최대 길이
        "-t", str(max(1, int(seconds))),

        # 영상만 원본 그대로 저장
        "-map", "0:v:0",
        "-c", "copy",
        "-f", "mpegts",
        str(video_file),
    ]

    # 임시 HLS 주소가 터미널에 노출되지 않도록 FFmpeg 로그를 별도 저장
    with log_file.open("w", encoding="utf-8") as log:
        result = subprocess.run(
            command,
            stdout=subprocess.DEVNULL,
            stderr=log,
        )

    return result.returncode


def append_manifest(manifest_file, row, header):
    file_exists = manifest_file.exists() and manifest_file.stat().st_size > 0
    encoding = "utf-8" if file_exists else "utf-8-sig"
    with manifest_file.open("a", encoding=encoding, newline="") as file:
        writer = csv.DictWriter(file, fieldnames=header)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(
        description="경부선 판교1 CCTV를 새 HLS 주소로 분할 녹화합니다."
    )
    parser.add_argument(
        "--minutes",
        type=float,
        default=5,
        help="전체 녹화 시간(분), 기본값 5",
    )
    parser.add_argument(
        "--chunk-seconds",
        type=int,
        default=120,
        help="각 조각의 최대 길이(초), 기본값 120",
    )
    parser.add_argument(
        "--cctv-type",
        choices=("1", "4"),
        default="4",
        help="ITS CCTV 유형: 1=HLS, 4=HTTPS HLS, 기본값 4",
    )
    args = parser.parse_args()

    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise SystemExit("ffmpeg와 ffprobe가 모두 설치되어 있어야 합니다.")
    if args.minutes <= 0 or args.chunk_seconds <= 0:
        raise SystemExit("녹화 시간과 조각 길이는 0보다 커야 합니다.")

    total_seconds = int(args.minutes * 60)
    session_start = now_local()
    session_stamp = session_start.strftime("%Y%m%d_%H%M%S")
    session_dir = RECORDINGS_DIR / f"pangyo1_chunked_{session_stamp}"
    session_dir.mkdir(parents=True, exist_ok=True)
    manifest_file = session_dir / "manifest.csv"
    diagnostic_file = session_dir / "diagnostic_summary.csv"

    manifest_header = [
        "조각번호", "CCTV명", "VDS_ID", "API_CCTV좌표X", "API_CCTV좌표Y",
        "API_CCTV유형",
        "시작시각", "종료시각", "요청길이_초", "실제영상길이_초",
        "FFmpeg종료코드", "상태", "오류분류", "HTTP상태",
        "영상파일", "FFmpeg로그",
    ]

    print("분할 녹화를 시작합니다.")
    print("CCTV:", CCTV_NAME)
    print("VDS_ID:", VDS_ID)
    print("CCTV 유형:", args.cctv_type, "(1=HLS, 4=HTTPS HLS)")
    print("전체 녹화 시간:", total_seconds, "초")
    print("조각 길이:", args.chunk_seconds, "초")
    print("저장 폴더:", session_dir)
    print()

    session_deadline = time.monotonic() + total_seconds
    part_number = 1
    consecutive_failures = 0
    full_parts = 0
    partial_parts = 0
    failed_parts = 0
    media_seconds = 0.0
    diagnostic_counts = {}

    while time.monotonic() < session_deadline:
        remaining = session_deadline - time.monotonic()
        # 남은 시간이 너무 짧으면 의미 없는 마지막 조각을 만들지 않습니다.
        if remaining < 5:
            break
        requested_seconds = min(args.chunk_seconds, max(1, int(remaining)))

        try:
            cctv, stream_url = fetch_cctv(args.cctv_type)
        except Exception as error:
            consecutive_failures += 1
            print(f"[{part_number:03d}] CCTV 주소 요청 실패: {error}")
            if consecutive_failures >= 20:
                print("연속 실패가 많아 녹화를 중단합니다.")
                break
            time.sleep(3)
            continue

        part_start = now_local()
        part_stamp = part_start.strftime("%Y%m%d_%H%M%S")
        video_file = session_dir / f"part_{part_number:03d}_{part_stamp}.ts"
        ffmpeg_log = session_dir / f"part_{part_number:03d}_{part_stamp}.ffmpeg.log"

        print(
            f"[{part_number:03d}] 녹화 중 "
            f"({requested_seconds}초 요청, 시작 {part_start.strftime('%H:%M:%S')})",
            flush=True,
        )

        return_code = record_one_chunk(
            stream_url,
            video_file,
            ffmpeg_log,
            requested_seconds,
        )
        # FFmpeg 로그에는 임시 URL의 query string이 포함될 수 있으므로
        # 먼저 URL을 가린 뒤, 그 로그를 기준으로 오류를 분류합니다.
        raw_log = ffmpeg_log.read_text(encoding="utf-8", errors="replace")
        ffmpeg_log.write_text(redact_urls(raw_log), encoding="utf-8")
        error_category, http_status = classify_ffmpeg_log(ffmpeg_log)
        part_end = now_local()
        actual_seconds = probe_duration(video_file)
        media_seconds += actual_seconds

        if (
            return_code == 0
            and actual_seconds > 0.5
            and actual_seconds >= requested_seconds - 2
        ):
            status = "SUCCESS"
            full_parts += 1
            consecutive_failures = 0
        elif actual_seconds > 0:
            status = "PARTIAL"
            partial_parts += 1
            consecutive_failures = 0
        else:
            status = "FAILED"
            failed_parts += 1
            consecutive_failures += 1

        if status == "SUCCESS" and error_category == "OK":
            error_category = "OK"
        diagnostic_counts[error_category] = diagnostic_counts.get(error_category, 0) + 1

        append_manifest(
            manifest_file,
            {
                "조각번호": part_number,
                "CCTV명": cctv.get("cctvname", CCTV_NAME),
                "VDS_ID": VDS_ID,
                "API_CCTV좌표X": cctv.get("coordx", ""),
                "API_CCTV좌표Y": cctv.get("coordy", ""),
                "API_CCTV유형": args.cctv_type,
                "시작시각": part_start.isoformat(),
                "종료시각": part_end.isoformat(),
                "요청길이_초": requested_seconds,
                "실제영상길이_초": f"{actual_seconds:.3f}",
                "FFmpeg종료코드": return_code,
                "상태": status,
                "오류분류": error_category,
                "HTTP상태": http_status,
                "영상파일": str(video_file) if video_file.exists() else "",
                "FFmpeg로그": str(ffmpeg_log),
            },
            manifest_header,
        )

        print(
            f"[{part_number:03d}] {status}, "
            f"실제 {actual_seconds:.2f}초, "
            f"종료코드 {return_code}",
            flush=True,
        )

        if consecutive_failures >= 20:
            print("연속 실패가 많아 녹화를 중단합니다.")
            break

        part_number += 1

        # 다음 API 주소를 받기 전 짧게 대기합니다.
        if status != "SUCCESS":
            time.sleep(2)

    session_end = now_local()
    elapsed = (session_end - session_start).total_seconds()

    print()
    print("분할 녹화가 종료되었습니다.")
    print("시작:", session_start.isoformat())
    print("종료:", session_end.isoformat())
    print("실제 경과 시간:", f"{elapsed:.1f}초")
    print("저장된 영상 시간 합계:", f"{media_seconds:.1f}초")
    coverage = (media_seconds / total_seconds * 100) if total_seconds else 0.0
    print("수집률:", f"{coverage:.1f}%")
    print("정상 조각:", full_parts)
    print("부분 조각:", partial_parts)
    print("실패 조각:", failed_parts)
    print("manifest:", manifest_file)
    write_diagnostic_summary(diagnostic_file, diagnostic_counts)
    print("오류 분류:", diagnostic_file)
    for category, count in sorted(diagnostic_counts.items()):
        print(f"  - {category}: {count}개")
    print()
    print("각 조각의 실제 시간과 VDS 연결에는 manifest.csv를 사용하세요.")


if __name__ == "__main__":
    main()
