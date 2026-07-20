# This Python file is used for smoke testing.
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent

PYTHON = Path(r"C:\Users\yrz\anaconda3\envs\selection\python.exe")
TRAIN = WORKSPACE / "data" / "cifar10_png" / "cifar10" / "train"
TEST = WORKSPACE / "data" / "cifar10_png" / "cifar10" / "test"

# Default model.
WEIGHTS = ROOT / "vgg11_bn.pt"

DATASET = "cifar10"
MODEL = "vgg11_bn"
DEVICE = None

# Coverage granularity: choose from "all", "basc", "iasc", "irsc".
COVERAGE_MODE = "basc"

# Short run settings.
LABEL = 0
MAX_IMAGES = 6000
BATCH_SIZE = 32
MAX_SEEDS = 10
BUDGET = 100
ROLLOUT_DEPTH = 10
TOP_K = 10

RUNS = ROOT / "runs"
BOUNDS = RUNS / "short_vgg11_bounds.json"
RESPONSE_SCOPE = RUNS / "short_vgg11_response_scope.json"
OUTPUT_DIR = RUNS / f"short_vgg11_{COVERAGE_MODE}_images"


def run(args: list[str]) -> None:
    print("\n> " + " ".join(str(x) for x in args), flush=True)
    subprocess.run(args, cwd=ROOT, check=True)


def existing(path: Path, name: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{name} does not exist: {path}")


def label_args() -> list[str]:
    return ["--label", str(LABEL)] if LABEL is not None else []


def max_images_args() -> list[str]:
    return ["--max-images", str(MAX_IMAGES)] if MAX_IMAGES is not None else []


def main() -> None:
    python = PYTHON if PYTHON.exists() else Path(sys.executable)
    existing(TRAIN, "train directory")
    existing(TEST, "test directory")
    existing(WEIGHTS, "model weights")
    RUNS.mkdir(parents=True, exist_ok=True)

    common = [
        "--dataset",
        DATASET,
        "--model",
        MODEL,
        "--weights",
        str(WEIGHTS),
    ]
    if DEVICE:
        common += ["--device", DEVICE]

    run(
        [
            str(python),
            str(ROOT / "main.py"),
            "prepare-response-stats",
        ]
        + common
        + [
            "--data-dir",
            str(TRAIN),
            "--output",
            str(BOUNDS),
            "--batch-size",
            str(BATCH_SIZE),
        ]
        + max_images_args()
        + label_args()
    )

    run(
        [
            str(python),
            str(ROOT / "main.py"),
            "select-response-scope",
        ]
        + common
        + [
            "--data-dir",
            str(TRAIN),
            "--output",
            str(RESPONSE_SCOPE),
            "--batch-size",
            str(BATCH_SIZE),
            "--top-per-layer",
            "5",
        ]
        + max_images_args()
    )

    run(
        [
            str(python),
            str(ROOT / "main.py"),
            "fuzz",
        ]
        + common
        + [
            "--seed-dir",
            str(TEST),
            "--response-scope-config",
            str(RESPONSE_SCOPE),
            "--bounds",
            str(BOUNDS),
            "--output-dir",
            str(OUTPUT_DIR),
            "--max-seeds",
            str(MAX_SEEDS),
            "--budget",
            str(BUDGET),
            "--rollout-depth",
            str(ROLLOUT_DEPTH),
            "--top-k",
            str(TOP_K),
            "--coverage-mode",
            COVERAGE_MODE,
        ]
        + label_args()
    )

    print(f"\nDone. Results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
