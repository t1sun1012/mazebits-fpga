"""Train the MazeBits CNN on CPU and select it using validation rollouts."""

import argparse
from pathlib import Path

from mazebits.training import train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/maze-v1"))
    parser.add_argument("--run-dir", type=Path, default=Path("runs/cnn-baseline"))
    parser.add_argument("--seed", type=int, default=311)
    parser.add_argument("--channels", type=int, default=16)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--no-augment", action="store_true")
    args = vars(parser.parse_args())
    args["augment"] = not args.pop("no_augment")
    train(**args)


if __name__ == "__main__":
    main()
