from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from starfuzz.benchmarks import DATASETS, MODELS
from starfuzz.experiments.common import ModelConfig
from starfuzz.experiments.speed import SpeedBenchmarkConfig, run_speed_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description="Run StaRFuzz speed benchmark.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset", default="cifar10", choices=DATASETS)
    parser.add_argument("--model", default=None, choices=MODELS)
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--random-seeds", type=int, default=5)
    parser.add_argument("--budget-per-seed", type=int, default=100)
    parser.add_argument("--total-candidates", type=int, default=100)
    parser.add_argument("--rollout-depth", type=int, default=8)
    parser.add_argument("--rng-seed", type=int, default=20260701)
    args = parser.parse_args()
    cfg = SpeedBenchmarkConfig(
        output_dir=args.output_dir,
        dataset=args.dataset,
        random_seeds=args.random_seeds,
        budget_per_seed=args.budget_per_seed,
        total_candidates=args.total_candidates,
        rollout_depth=args.rollout_depth,
        rng_seed=args.rng_seed,
    )
    out = run_speed_benchmark(cfg, ModelConfig(dataset=args.dataset, model=args.model, weights=args.weights, device=args.device))
    print(f"saved speed summary: {out}")


if __name__ == "__main__":
    main()

