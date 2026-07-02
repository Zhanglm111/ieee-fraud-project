from __future__ import annotations

import argparse

from fraud_detection.pipeline import run_training
from fraud_detection.utils import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/xgb.yaml")
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()
    run_training(load_yaml(args.config), args.output_dir)


if __name__ == "__main__":
    main()
