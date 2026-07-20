from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from starfuzz.experiments.common import ROOT, WORKSPACE, write_json


@dataclass
class AblationCase:
    name: str
    output_dir: Path
    response_scope_config: Path = WORKSPACE / "ResponseScope" / "cifar10_vgg11.json"
    budget_per_seed: int = 1000
    rollout_depth: int = 8
    schedule_interval: int = 500
    require_correct_seeds: bool = True
    reuse_zeta: bool = True
    zeta_path: Path | None = None
    extra_args: list[str] = field(default_factory=list)


@dataclass
class AblationSuite:
    output_dir: Path
    random_seeds: int = 20
    total_candidates: int = 1000
    rng_seed: int = 20260701
    device: str = "cuda"
    cases: list[AblationCase] = field(default_factory=list)


def default_ablation_suite(output_dir: Path) -> AblationSuite:
    zeta = output_dir / "zeta_base.json"
    return AblationSuite(
        output_dir=output_dir,
        cases=[
            AblationCase("full_b1000_d8", output_dir / "full_b1000_d8", zeta_path=zeta, reuse_zeta=False),
            AblationCase("no_zeta", output_dir / "no_zeta", zeta_path=zeta, reuse_zeta=False, extra_args=["--no-zeta"]),
            AblationCase("basc_only", output_dir / "basc_only", zeta_path=zeta, extra_args=["--coverage-mode", "basc"]),
            AblationCase("iasc_only", output_dir / "iasc_only", zeta_path=zeta, extra_args=["--coverage-mode", "iasc"]),
            AblationCase("irsc_only", output_dir / "irsc_only", zeta_path=zeta, extra_args=["--coverage-mode", "irsc"]),
            AblationCase("full_b2000_d8", output_dir / "full_b2000_d8", budget_per_seed=2000, zeta_path=zeta),
            AblationCase("depth16", output_dir / "depth16", rollout_depth=16, zeta_path=zeta),
            AblationCase("no_dynamic_schedule", output_dir / "no_dynamic_schedule", schedule_interval=10**12, zeta_path=zeta),
            AblationCase("all_seed_oracle", output_dir / "all_seed_oracle", require_correct_seeds=False, zeta_path=zeta),
            AblationCase("ratio_1pct", output_dir / "ratio_1pct", response_scope_config=ROOT / "runs" / "response_scope_vgg11_ratio001.json", zeta_path=output_dir / "zeta_ratio_1pct.json", reuse_zeta=False),
            AblationCase("ratio_1p5pct", output_dir / "ratio_1p5pct", response_scope_config=ROOT / "runs" / "response_scope_vgg11_ratio0015.json", zeta_path=output_dir / "zeta_ratio_1p5pct.json", reuse_zeta=False),
        ],
    )


def command_for_case(suite: AblationSuite, case: AblationCase) -> list[str]:
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "evaluate_paper_fdr.py"),
        "--random-seeds",
        str(suite.random_seeds),
        "--total-candidates",
        str(suite.total_candidates),
        "--budget-per-seed",
        str(case.budget_per_seed),
        "--schedule-interval",
        str(case.schedule_interval),
        "--rollout-depth",
        str(case.rollout_depth),
        "--device",
        suite.device,
        "--rng-seed",
        str(suite.rng_seed),
        "--response-scope-config",
        str(case.response_scope_config),
        "--output-dir",
        str(case.output_dir),
    ]
    if case.require_correct_seeds:
        cmd.append("--require-correct-seeds")
    if case.zeta_path is not None:
        cmd += ["--zeta-path", str(case.zeta_path)]
    if case.reuse_zeta:
        cmd.append("--reuse-zeta")
    cmd += case.extra_args
    return cmd


def write_ablation_plan(suite: AblationSuite) -> Path:
    suite.output_dir.mkdir(parents=True, exist_ok=True)
    plan = []
    for case in suite.cases:
        plan.append(
            {
                "name": case.name,
                "output_dir": str(case.output_dir),
                "command": command_for_case(suite, case),
            }
        )
    path = suite.output_dir / "ablation_plan.json"
    write_json(path, {"cases": plan})
    return path


def run_ablation_suite(suite: AblationSuite, dry_run: bool = True) -> Path:
    plan_path = write_ablation_plan(suite)
    if dry_run:
        return plan_path
    for case in suite.cases:
        case.output_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(command_for_case(suite, case), check=True)
    return plan_path

