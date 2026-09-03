#!/usr/bin/env python3

"""Check whether HLS 404 comes from ITS/CDN or only from FFmpeg.

Prints HTTP status only. Stream tokens are not written to disk or stdout.
"""

from __future__ import annotations

import argparse
import subprocess
import time
from urllib.parse import urlsplit

from its_common import fetch_cctv, load_config

CHROME_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
FFMPEG_UA = "Lavf/61.7.100"
REFERER = "https://www.its.go.kr/"


def host_path(url: str) -> str:
    parts = urlsplit(url)
    path = parts.path
    if "/8152/" in path:
        path = "/8152/<redacted>"
    elif path.endswith(".m3u8"):
        path = path.rsplit("/", 1)[0] + "/<file>.m3u8"
    return f"{parts.scheme}://{parts.netloc}{path}"


def curl_status(url: str, user_agent: str, follow: bool = True) -> dict:
    cmd = [
        "curl",
        "-sS",
        "-o", "/dev/null",
        "-D", "-",
        "-A", user_agent,
        "-H", f"Referer: {REFERER}",
        "--max-time", "15",
        url,
    ]
    if follow:
        cmd[1:1] = ["-L", "--max-redirs", "5"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    statuses = []
    locations = []
    for line in result.stdout.splitlines():
        if line.upper().startswith("HTTP/"):
            statuses.append(line.strip())
        elif line.lower().startswith("location:"):
            loc = line.split(":", 1)[1].strip()
            locations.append(host_path(loc))
    final = ""
    if statuses:
        final = statuses[-1].split()[1] if len(statuses[-1].split()) > 1 else statuses[-1]
    return {
        "ok": result.returncode == 0,
        "curl_exit": result.returncode,
        "stderr": (result.stderr or "").strip()[:200],
        "http_lines": statuses,
        "final_status": final,
        "redirect_hosts": locations,
        "url_host": host_path(url),
    }


def print_probe(label: str, probe: dict) -> None:
    print(f"  {label}")
    print(f"    대상: {probe['url_host']}")
    print(f"    HTTP: {' -> '.join(probe['http_lines']) or '(응답 헤더 없음)'}")
    print(f"    최종코드: {probe['final_status'] or '-'}")
    if probe["redirect_hosts"]:
        print(f"    리다이렉트: {' -> '.join(probe['redirect_hosts'])}")
    if probe["stderr"]:
        print(f"    curl stderr: {probe['stderr']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="ITS HLS 404를 curl로 교차검증합니다.")
    parser.add_argument("--repeats", type=int, default=5, help="새 API URL을 몇 번 받아 curl할지")
    parser.add_argument("--poll-seconds", type=int, default=40, help="200인 URL을 얼마나 재요청할지")
    args = parser.parse_args()

    config = load_config()
    print("1) ITS API가 준 주소를 바로 curl (FFmpeg 없음)")
    already_dead = 0
    live_url = None
    for i in range(1, args.repeats + 1):
        _item, url = fetch_cctv(config)
        chrome = curl_status(url, CHROME_UA)
        ffmpeg_ua = curl_status(url, FFMPEG_UA)
        print(f"[{i:02d}] API URL {chrome['url_host']}")
        print_probe("Chrome UA + Referer", chrome)
        print_probe("FFmpeg Lavf UA + Referer", ffmpeg_ua)
        chrome_code = chrome["final_status"]
        if chrome_code == "404":
            already_dead += 1
        elif chrome_code.startswith("2") and live_url is None:
            live_url = url
        time.sleep(1)

    print()
    print(f"API가 주자마자 curl 404인 횟수: {already_dead}/{args.repeats}")
    if already_dead:
        print("→ FFmpeg를 거치지 않고도 404다. 죽은 주소를 ITS/CDN이 준 것이다.")

    if live_url is None:
        print()
        print("살아 있는 URL을 못 받아서 TTL 측정은 건너뜁니다.")
        return

    print()
    print("2) 방금 200이던 같은 URL을 몇 초마다 curl (여전히 FFmpeg 없음)")
    started = time.monotonic()
    while time.monotonic() - started <= args.poll_seconds:
        elapsed = int(time.monotonic() - started)
        probe = curl_status(live_url, CHROME_UA)
        print(f"  t+{elapsed:02d}s  {probe['final_status']}  {probe['url_host']}")
        if probe["final_status"] == "404":
            print("→ 같은 주소가 시간이 지나며 404로 바뀌었다. 세션/토큰 만료로 보는 것이 맞다.")
            break
        time.sleep(5)
    else:
        print("  이 구간에서는 아직 200이다. --poll-seconds 를 늘려 다시 보면 된다.")


if __name__ == "__main__":
    main()
