#!/usr/bin/env python3

"""Build pangyo1_15min_dataset.csv from the 5-minute master. Does not train."""

from __future__ import annotations

import argparse

from dataset_io import build_15min_dataset
from its_common import dataset_15min_path, load_config, master_csv_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="5분 master에서 15분 입력 / 다음 15분 속도 정답 CSV를 만듭니다."
    )
    parser.add_argument("--master", default=None, help="pangyo1_5min_master.csv")
    parser.add_argument("--out", default=None, help="pangyo1_15min_dataset.csv")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    dest = build_15min_dataset(args.master, args.out, args.config)
    config = load_config(args.config)
    print("저장:", dest)
    print("입력:", args.master or master_csv_path(config))
    print("기본 출력:", args.out or dataset_15min_path(config))


if __name__ == "__main__":
    main()
