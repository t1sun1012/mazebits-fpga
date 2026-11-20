"""Verify exported integer traces against both integer inference engines."""

import argparse
import json
from pathlib import Path

from mazebits.fixed_vectors import verify_vectors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--vectors-dir", type=Path, required=True)
    parser.add_argument("--case-limit", type=int)
    args = parser.parse_args()
    print(json.dumps(verify_vectors(**vars(args)), indent=2))


if __name__ == "__main__":
    main()
