#!/usr/bin/env python
"""Run main.py for a given method/dataset using its YAML config.

Usage:
    python run_config.py <method> <dataset> [--<extra_arg> <value> ...]
    e.g. python run_config.py pace imagenet_c --batch_size 32 --cma_init_sigma 0.1

Any extra --<arg> <value> pairs are forwarded to main.py after the config's
flags, so they override the corresponding config value (argparse keeps the
last occurrence of a repeated flag).
"""

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

CONFIGS_DIR = Path(__file__).resolve().parent / "configs"


def load_config(method, dataset):
    common_path = CONFIGS_DIR / "common.yaml"
    method_path = CONFIGS_DIR / dataset / f"{method}.yaml"

    if not method_path.exists():
        raise FileNotFoundError(f"No config found at {method_path}")

    common = yaml.safe_load(common_path.read_text()) or {}
    specific = yaml.safe_load(method_path.read_text()) or {}

    config = {**common, **specific}
    config["algorithm"] = method
    config["dataset"] = dataset
    return config


def build_command(config):
    command = ["python", "main.py"]
    for key, value in config.items():
        flag = f"--{key}"
        if isinstance(value, bool):
            if value:
                command.append(flag)
        else:
            command.extend([flag, str(value)])
    return command


def main():
    parser = argparse.ArgumentParser(description="Run a TTA method with its config for a given dataset.")
    parser.add_argument("method", help="algorithm name, e.g. pace, foa, sar")
    parser.add_argument("dataset", help="dataset name, e.g. imagenet_c, imagenet_r, domainnet126")
    args, overrides = parser.parse_known_args()

    config = load_config(args.method, args.dataset)
    command = build_command(config) + overrides

    print(" ".join(command))
    print()

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    for line in process.stdout:
        print(line, end="", flush=True)

    returncode = process.wait()
    if returncode != 0:
        print("--ERROR--" * 20)
        sys.exit(returncode)


if __name__ == "__main__":
    main()
