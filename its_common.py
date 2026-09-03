#!/usr/bin/env python3

"""Shared ITS CCTV helpers and config loading."""

from __future__ import annotations

import os
import re
from pathlib import Path

import requests
import yaml
from dotenv import load_dotenv

PROJECT_DIR = Path(__file__).resolve().parent
RECORDINGS_DIR = PROJECT_DIR / "recordings"
DEFAULT_CONFIG = PROJECT_DIR / "config.yaml"


def load_config(path: str | Path | None = None) -> dict:
    config_path = Path(path) if path else DEFAULT_CONFIG
    with config_path.open(encoding="utf-8") as file:
        return yaml.safe_load(file)


def get_api_key() -> str:
    load_dotenv(PROJECT_DIR / ".env")
    for name in ("ITS_API_KEY", "OPENAPI", "API_KEY"):
        value = os.environ.get(name, "").strip()
        if value:
            return value
    raise RuntimeError(
        ".env에 ITS_API_KEY 또는 OPENAPI 키가 필요합니다."
    )


def fetch_cctv(config: dict) -> tuple[dict, str]:
    its = config["its"]
    site = config["site"]
    params = {
        "apiKey": get_api_key(),
        "type": its["type"],
        "cctvType": str(its["cctv_type"]),
        "minX": its["minX"],
        "maxX": its["maxX"],
        "minY": its["minY"],
        "maxY": its["maxY"],
        "getType": its.get("getType", "json"),
    }
    response = requests.get(its["cctv_url"], params=params, timeout=30)
    header = {}
    payload = None
    try:
        payload = response.json()
        header = payload.get("header") or {}
    except ValueError:
        payload = None
    result_code = str(header.get("resultCode") or "")
    result_msg = str(header.get("resultMsg") or "")
    if response.status_code != 200 or result_code in {"4001", "400"}:
        detail = " ".join(part for part in (result_code, result_msg) if part).strip()
        raise RuntimeError(
            f"cctvInfo HTTP {response.status_code}: {detail or response.reason}"
        )
    response.raise_for_status()
    data = (payload or response.json())["response"]["data"]
    name = site["cctv_name"]
    candidates = [item for item in data if item.get("cctvname") == name]
    if not candidates:
        raise RuntimeError(f"CCTV를 찾을 수 없습니다: {name}")
    item = candidates[0]
    stream_url = item.get("cctvurl")
    if not stream_url:
        raise RuntimeError("CCTV 스트림 주소가 비어 있습니다.")
    return item, stream_url


def redact_urls(text: str) -> str:
    return re.sub(
        r"(https?://[^?\s'\"]+)\?[^\s'\"]+",
        r"\1?<redacted>",
        text,
    )


def classify_ffmpeg_log(text: str) -> tuple[str, str]:
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
    if re.search(
        r"failed to (?:open|load) segment|error when loading segment",
        lower,
    ):
        if "404" in status_codes:
            return "SEGMENT_404", http_status
        return "SEGMENT_ERROR", http_status
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
