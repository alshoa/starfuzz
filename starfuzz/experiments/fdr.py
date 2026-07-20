from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from starfuzz.coverage import StabilityCoverage
from starfuzz.experiments.common import (
    ExperimentConfig,
    ModelConfig,
    config_to_json,
    coverage_weights,
    load_response_scope_for_seed,
    load_model_bundle,
    load_or_calibrate_zeta,
    nvidia_smi_snapshot,
    pick_random_seeds,
    set_reproducible,
    write_json,
)
from starfuzz.paper_mcts import PaperEvaluation, PaperMCTSTree, PaperNode
from starfuzz.transforms import default_action_pool


@dataclass
class FdrRunResult:
    summary_path: Path
    fdr: float
    strict_fdr: float
    total_failures: int
    total_candidates: int


def _reallocate_units(states) -> None:
    remaining = sum(state["units"] for state in states)
    active = [state for state in states if state["units"] > 0]
    if remaining <= 0 or not active:
        return
    scores = np.array([max((n.avg_reward for n in state["tree"].nodes), default=0.0) for state in active])
    scores = np.maximum(scores, 0.0) + 1e-6
    raw = scores / scores.sum() * remaining
    units = np.floor(raw).astype(int)
    leftover = remaining - int(units.sum())
    for idx in np.argsort(-(raw - units))[:leftover]:
        units[idx] += 1
    for state in states:
        state["units"] = 0
    for state, unit in zip(active, units.tolist(), strict=True):
        state["units"] = int(unit)


def run_paper_fdr(
    exp_cfg: ExperimentConfig,
    model_cfg: ModelConfig | None = None,
    save_candidate_images: bool = False,
) -> FdrRunResult:
    """Run paper-style StaRFuzz and compute FDR.

    This is the reusable module behind the CLI scripts. It keeps the paper
    mechanics: zeta calibration, multi-seed budget scheduling, global top-k
    candidate collection, and stability coverage reward.
    """

    model_cfg = model_cfg or ModelConfig(dataset=exp_cfg.dataset, device="cuda")
    model_cfg.dataset = exp_cfg.dataset
    exp_cfg.output_dir.mkdir(parents=True, exist_ok=True)
    rng = set_reproducible(exp_cfg.rng_seed)
    bundle = load_model_bundle(model_cfg)
    actions = default_action_pool()
    zeta_rng = set_reproducible(exp_cfg.rng_seed + 100003)
    zeta, zeta_path, zeta_seconds = load_or_calibrate_zeta(exp_cfg, bundle, actions, zeta_rng)
    bounds = StabilityCoverage.load_bounds(exp_cfg.bounds)
    seeds = pick_random_seeds(exp_cfg, exp_cfg.random_seeds, rng, bundle.image_size)
    states = []

    for seed in seeds:
        response_scope = load_response_scope_for_seed(exp_cfg, seed.label, bundle.layer_sizes)
        coverage = StabilityCoverage(
            bundle.model,
            bundle.preprocess,
            response_scope,
            bounds=bounds,
            zeta=zeta,
            num_bins=exp_cfg.num_bins,
            weights=coverage_weights(exp_cfg.coverage_mode),
        )
        _original_snapshot, original_logits = coverage.evaluate([seed.image])
        original_pred = int(original_logits.argmax(dim=1)[0])
        if exp_cfg.require_correct_seeds and original_pred != seed.label:
            continue

        def make_evaluate(cov: StabilityCoverage):
            def evaluate(images: list[np.ndarray]) -> PaperEvaluation:
                snapshot, logits = cov.evaluate(images)
                return PaperEvaluation(
                    stability_coverage=snapshot.total,
                    pred=int(logits.argmax(dim=1)[-1]),
                )

            return evaluate

        tree = PaperMCTSTree(
            seed.image,
            seed.label,
            actions,
            make_evaluate(coverage),
            rollout_depth=exp_cfg.rollout_depth,
            rng=rng,
        )
        states.append(
            {
                "seed": seed,
                "coverage": coverage,
                "original_pred": original_pred,
                "tree": tree,
                "units": 0,
            }
        )

    if not states:
        raise RuntimeError("No seeds selected after filtering.")

    total_budget = exp_cfg.budget_per_seed * len(states)
    base_units = total_budget // len(states)
    extra = total_budget % len(states)
    for idx, state in enumerate(states):
        state["units"] = base_units + (1 if idx < extra else 0)

    start = time.perf_counter()
    used = 0
    while sum(state["units"] for state in states) > 0:
        for state in states:
            if state["units"] <= 0:
                continue
            state["tree"].iterate()
            state["units"] -= 1
            used += 1
            if used % exp_cfg.schedule_interval == 0:
                _reallocate_units(states)
    fuzz_seconds = time.perf_counter() - start

    all_nodes: list[tuple[dict, PaperNode]] = []
    for state in states:
        for node in state["tree"].nodes:
            if node is not state["tree"].root and node.visits > 0:
                all_nodes.append((state, node))
    all_nodes.sort(key=lambda item: item[1].avg_reward, reverse=True)
    selected = all_nodes[: exp_cfg.total_candidates]

    total_failures = strict_failures = strict_total = 0
    records = []
    per_seed: dict[str, dict] = {}
    candidates_dir = exp_cfg.output_dir / "candidates"
    if save_candidate_images:
        candidates_dir.mkdir(parents=True, exist_ok=True)

    for rank, (state, node) in enumerate(selected):
        seed = state["seed"]
        path_images = [item.image for item in node.path()]
        snapshot, logits = state["coverage"].evaluate(path_images)
        pred = int(logits.argmax(dim=1)[-1])
        is_failure = pred != seed.label
        total_failures += int(is_failure)
        if state["original_pred"] == seed.label:
            strict_total += 1
            strict_failures += int(is_failure)
        image_path = None
        if save_candidate_images:
            image_path = candidates_dir / f"rank_{rank:05d}_label_{seed.label}_pred_{pred}_failure_{int(is_failure)}.png"
            Image.fromarray(node.image).save(image_path)
        key = str(seed.path)
        per_info = per_seed.setdefault(
            key,
            {
                "seed": key,
                "label": seed.label,
                "original_prediction": state["original_pred"],
                "original_correct": state["original_pred"] == seed.label,
                "candidates": 0,
                "failures": 0,
            },
        )
        per_info["candidates"] += 1
        per_info["failures"] += int(is_failure)
        records.append(
            {
                "rank": rank,
                "seed": key,
                "label": seed.label,
                "prediction": pred,
                "is_model_failure": is_failure,
                "avg_reward": node.avg_reward,
                "image": str(image_path) if image_path else None,
                "stability_coverage": {
                    "basc": snapshot.basc,
                    "iasc": snapshot.iasc,
                    "irsc": snapshot.irsc,
                    "total": snapshot.total,
                },
                "response_variation": snapshot.variation,
            }
        )

    for item in per_seed.values():
        item["fdr"] = item["failures"] / max(1, item["candidates"])

    summary = {
        "mode": "paper_mcts",
        "experiment_config": config_to_json(exp_cfg),
        "model_config": config_to_json(model_cfg),
        "zeta_path": str(zeta_path),
        "zeta_seconds": zeta_seconds,
        "fuzz_seconds": fuzz_seconds,
        "iterations_per_second": total_budget / max(1e-9, fuzz_seconds),
        "num_seeds": len(states),
        "total_budget": total_budget,
        "total_candidates": len(selected),
        "total_failures": total_failures,
        "fdr": total_failures / max(1, len(selected)),
        "strict_failures": strict_failures,
        "strict_total": strict_total,
        "strict_fdr": strict_failures / max(1, strict_total),
        "seed_accuracy": sum(1 for s in states if s["original_pred"] == s["seed"].label) / max(1, len(states)),
        "gpu": nvidia_smi_snapshot(),
        "per_seed": list(per_seed.values()),
        "top_candidates": records,
    }
    path = exp_cfg.output_dir / "fdr_summary.json"
    write_json(path, summary)
    return FdrRunResult(path, summary["fdr"], summary["strict_fdr"], total_failures, len(selected))

