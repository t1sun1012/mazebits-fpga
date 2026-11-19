"""Generate deterministic maze layouts and BFS imitation-learning examples."""

import argparse
import json
from pathlib import Path

from mazebits.dataset import generate_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/maze-v1"))
    parser.add_argument("--train-layouts", type=int, default=800)
    parser.add_argument("--val-layouts", type=int, default=150)
    parser.add_argument("--test-layouts", type=int, default=150)
    parser.add_argument("--seed", type=int, default=311)
    parser.add_argument("--goals-per-layout", type=int, default=3)
    parser.add_argument("--episodes-per-goal", type=int, default=8)
    parser.add_argument("--min-component", type=int, default=24)
    args = parser.parse_args()
    manifest = generate_dataset(**vars(args))
    print(json.dumps({
        "output_dir": str(args.output_dir.resolve()),
        "splits": {key: {name: value[name] for name in
                        ("layouts", "goals", "samples", "episodes")}
                   for key, value in manifest["splits"].items()},
        "audit": manifest["audit"],
        "test_status": manifest["test_status"],
    }, indent=2))


if __name__ == "__main__":
    main()
