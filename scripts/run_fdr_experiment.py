from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from starfuzz.benchmarks import DATASETS, MODELS
from starfuzz.experiments.common import ExperimentConfig, ModelConfig
from starfuzz.experiments.fdr import run_paper_fdr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run paper-style StaRFuzz FDR experiment.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset", default="cifar10", choices=DATASETS)
    parser.add_argument("--seed-dir", type=Path, default=None)
    parser.add_argument("--response-scope-config", type=Path, default=ROOT.parent / "ResponseScope" / "cifar10_vgg11.json")
    parser.add_argument("--bounds", type=Path, default=ROOT.parent / "boundary" / "cifar10_vgg11.json")
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--model", default=None, choices=MODELS)
    parser.add_argument("--random-seeds", type=int, default=20)
    parser.add_argument("--total-candidates", type=int, default=1000)
    parser.add_argument("--budget-per-seed", type=int, default=1000)
    parser.add_argument("--schedule-interval", type=int, default=500)
    parser.add_argument("--rollout-depth", type=int, default=8)
    parser.add_argument("--rng-seed", type=int, default=20260701)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--zeta-path", type=Path, default=None)
    parser.add_argument("--reuse-zeta", action="store_true")
    parser.add_argument("--no-zeta", action="store_true")
    parser.add_argument("--all-seeds", action="store_true", help="do not filter originally wrong seeds")
    parser.add_argument("--save-candidate-images", action="store_true")
    parser.add_argument("--coverage-mode", choices=["all", "basc", "iasc", "irsc"], default="all")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    exp_cfg = ExperimentConfig(
        dataset=args.dataset,
        seed_dir=args.seed_dir,
        output_dir=args.output_dir,
        response_scope_config=args.response_scope_config,
        bounds=args.bounds,
        random_seeds=args.random_seeds,
        total_candidates=args.total_candidates,
        budget_per_seed=args.budget_per_seed,
        schedule_interval=args.schedule_interval,
        rollout_depth=args.rollout_depth,
        rng_seed=args.rng_seed,
        require_correct_seeds=not args.all_seeds,
        zeta_path=args.zeta_path,
        reuse_zeta=args.reuse_zeta,
        use_zeta=not args.no_zeta,
        coverage_mode=args.coverage_mode,
    )
    model_cfg = ModelConfig(dataset=args.dataset, model=args.model, weights=args.weights, device=args.device)
    result = run_paper_fdr(exp_cfg, model_cfg=model_cfg, save_candidate_images=args.save_candidate_images)
    print(f"FDR={result.fdr:.4f}, strict={result.strict_fdr:.4f}, saved={result.summary_path}")


if __name__ == "__main__":
    main()

