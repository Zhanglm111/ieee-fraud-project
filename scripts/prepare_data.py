from __future__ import annotations

import argparse

from fraud_detection.data import prepare_data
from fraud_detection.utils import load_yaml, save_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/xgb.yaml")
    args = parser.parse_args()
    config = load_yaml(args.config)
    paths = prepare_data(config)
    save_json(paths, "outputs/prepared_data_paths.json")
    print(paths)


if __name__ == "__main__":
    main()
