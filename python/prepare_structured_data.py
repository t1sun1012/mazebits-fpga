"""Generate structured barriers/corridors for the MazeBits base-model improvement."""
import argparse
import json
from pathlib import Path
from mazebits.structured_dataset import generate_structured_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/maze-v2"))
    parser.add_argument("--train-layouts", type=int, default=1200)
    parser.add_argument("--val-layouts", type=int, default=200)
    parser.add_argument("--test-layouts", type=int, default=200)
    parser.add_argument("--seed", type=int, default=312)
    parser.add_argument("--goals-per-layout", type=int, default=3)
    parser.add_argument("--episodes-per-goal", type=int, default=8)
    parser.add_argument("--min-component", type=int, default=24)
    parser.add_argument("--exclude-manifest", action="append", type=Path, default=[],
                        help="Additional dataset audit manifests; the v1 manifest is always excluded.")
    parser.add_argument("--exclude-maze-file", action="append", type=Path, default=[])
    args = parser.parse_args()
    values = vars(args).copy()
    prior = Path(__file__).absolute().parents[1] / "results/dataset/manifest.json"
    values["exclude_manifests"] = [prior, *values.pop("exclude_manifest")]
    values["exclude_maze_files"] = values.pop("exclude_maze_file")
    manifest = generate_structured_dataset(**values)
    print(json.dumps({"output_dir": str(args.output_dir.absolute()),
        "splits": {key: {name: value[name] for name in
                         ("layouts", "goals", "samples", "episodes", "distance_max", "by_family")}
                   for key, value in manifest["splits"].items()},
        "audit": manifest["audit"], "test_status": manifest["test_status"]}, indent=2))


if __name__ == "__main__":
    main()
