from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from starfuzz.benchmarks import DATASETS, MODELS
from starfuzz.experiments.common import ModelConfig
from starfuzz.experiments.retrain import RetrainConfig, run_retrain_repair


def main() -> None:
    parser = argparse.ArgumentParser(description="Run StaRFuzz retrain/repair experiment.")
    parser.add_argument("--fdr-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset", default="cifar10", choices=DATASETS)
    parser.add_argument("--clean-seed-dir", type=Path, default=None)
    parser.add_argument("--model", default=None, choices=MODELS)
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-failures", type=int, default=1000)
    args = parser.parse_args()
    cfg = RetrainConfig(
        fdr_summary=args.fdr_summary,
        output_dir=args.output_dir,
        dataset=args.dataset,
        clean_seed_dir=args.clean_seed_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_failures=args.max_failures,
    )
    out = run_retrain_repair(cfg, ModelConfig(dataset=args.dataset, model=args.model, weights=args.weights, device=args.device))
    print(f"saved retrain summary: {out}")


if __name__ == "__main__":
    main()


