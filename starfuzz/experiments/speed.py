from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from starfuzz.experiments.common import ExperimentConfig, ModelConfig, config_to_json, write_json
from starfuzz.experiments.fdr import run_paper_fdr


@dataclass
class SpeedBenchmarkConfig:
    output_dir: Path
    dataset: str = "cifar10"
    random_seeds: int = 5
    budget_per_seed: int = 100
    total_candidates: int = 100
    rollout_depth: int = 8
    require_correct_seeds: bool = False
    rng_seed: int = 20260701


def run_speed_benchmark(speed_cfg: SpeedBenchmarkConfig, model_cfg: ModelConfig | None = None) -> Path:
    exp_cfg = ExperimentConfig(
        dataset=speed_cfg.dataset,
        output_dir=speed_cfg.output_dir,
        random_seeds=speed_cfg.random_seeds,
        budget_per_seed=speed_cfg.budget_per_seed,
        total_candidates=speed_cfg.total_candidates,
        rollout_depth=speed_cfg.rollout_depth,
        require_correct_seeds=speed_cfg.require_correct_seeds,
        rng_seed=speed_cfg.rng_seed,
        schedule_interval=max(1, speed_cfg.budget_per_seed),
    )
    if model_cfg is not None:
        model_cfg.dataset = speed_cfg.dataset
    result = run_paper_fdr(exp_cfg, model_cfg=model_cfg, save_candidate_images=False)
    summary_path = result.summary_path
    speed_summary = {
        "speed_config": config_to_json(speed_cfg),
        "fdr_summary": str(summary_path),
        "fdr": result.fdr,
        "strict_fdr": result.strict_fdr,
        "total_failures": result.total_failures,
        "total_candidates": result.total_candidates,
    }
    out = speed_cfg.output_dir / "speed_summary.json"
    write_json(out, speed_summary)
    return out


