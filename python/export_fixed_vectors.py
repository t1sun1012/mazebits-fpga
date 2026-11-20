"""Export full integer-CNN traces for independent checks and future RTL tests."""

import argparse
import json
from pathlib import Path

from mazebits.fixed_vectors import export_vectors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--training-metadata", type=Path)
    args = parser.parse_args()
    manifest = export_vectors(args.model_dir, args.output_dir, args.training_metadata)
    print(json.dumps({"output_dir": str(args.output_dir.resolve()),
                      "bits": manifest["bits"],
                      "verification": manifest["verification"]}, indent=2))


if __name__ == "__main__":
    main()
