from __future__ import annotations

import json
import random
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from starfuzz.benchmarks import default_data_dir, default_layers, default_model_for_dataset, load_benchmark
from starfuzz.coverage import StabilityCoverage
from starfuzz.datasets import Seed, iter_seeds, load_class_map
from starfuzz.hooks import ActivationExtractor
from starfuzz.response_scope import InternalResponseScope, load_response_scope


ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = ROOT.parent


@dataclass
class ModelConfig:
    dataset: str = "cifar10"
    model: str | None = None
    weights: Path | None = None
    device: str = "cuda"


@dataclass
class ExperimentConfig:
    dataset: str = "cifar10"
    seed_dir: Path | None = None
    response_scope_config: Path = WORKSPACE / "ResponseScope" / "cifar10_vgg11.json"
    bounds: Path = WORKSPACE / "boundary" / "cifar10_vgg11.json"
    output_dir: Path = ROOT / "runs" / "experiment"
    random_seeds: int = 20
    total_candidates: int = 1000
    budget_per_seed: int = 1000
    schedule_interval: int = 500
    rollout_depth: int = 8
    num_bins: int = 15
    rng_seed: int = 20260701
    require_correct_seeds: bool = True
    zeta_images_per_class: int = 20
    zeta_mutants_per_image: int = 8
    zeta_path: Path | None = None
    reuse_zeta: bool = False
    use_zeta: bool = True
    coverage_mode: str = "all"


def coverage_weights(mode: str) -> tuple[float, float, float]:
    modes = {
        "all": (1.0, 1.0, 1.2),
        "basc": (1.0, 0.0, 0.0),
        "iasc": (0.0, 1.0, 0.0),
        "irsc": (0.0, 0.0, 1.0),
    }
    if mode not in modes:
        raise ValueError(f"Unknown coverage mode {mode}. Choose one of {', '.join(modes)}")
    return modes[mode]


@dataclass
class ModelBundle:
    model: torch.nn.Module
    preprocess: Any
    device: torch.device
    layer_sizes: dict[str, int]
    image_size: int
    num_classes: int


def set_reproducible(seed: int) -> random.Random:
    rng = random.Random(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return rng


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_model_bundle(model_cfg: ModelConfig) -> ModelBundle:
    model_name = model_cfg.model or default_model_for_dataset(model_cfg.dataset)
    bundle = load_benchmark(model_cfg.dataset, model_name, weights=model_cfg.weights, device=model_cfg.device, workspace=WORKSPACE)
    layer_sizes = infer_layer_sizes(
        bundle.model,
        bundle.preprocess,
        layers=default_layers(model_name, WORKSPACE),
        image_size=bundle.dataset.image_size,
    )
    return ModelBundle(
        model=bundle.model,
        preprocess=bundle.preprocess,
        device=bundle.device,
        layer_sizes=layer_sizes,
        image_size=bundle.dataset.image_size,
        num_classes=bundle.dataset.num_classes,
    )


def infer_layer_sizes(model: torch.nn.Module, preprocess, layers: list[str], image_size: int) -> dict[str, int]:
    dummy = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    with torch.no_grad(), ActivationExtractor(model, layers) as extractor:
        batch = torch.stack([preprocess(Image.fromarray(dummy))], dim=0).to(next(model.parameters()).device)
        model(batch)
    return {layer: int(out.shape[1]) for layer, out in extractor.outputs.items()}


def resolved_seed_dir(config: ExperimentConfig) -> Path:
    return config.seed_dir or default_data_dir(config.dataset, WORKSPACE, seed=True)


def pick_random_seeds(config: ExperimentConfig, count: int, rng: random.Random, image_size: int) -> list[Seed]:
    seed_dir = resolved_seed_dir(config)
    class_map = load_class_map(seed_dir, config.dataset)
    all_items = list(iter_seeds(seed_dir, max_images=None, image_size=image_size, class_map=class_map))
    if not all_items:
        return []
    return rng.sample(all_items, min(count, len(all_items)))


def class_image_samples(config: ExperimentConfig, label: int, count: int, rng: random.Random, image_size: int) -> list[np.ndarray]:
    seed_dir = resolved_seed_dir(config)
    class_map = load_class_map(seed_dir, config.dataset)
    items = [seed for seed in iter_seeds(seed_dir, image_size=image_size, class_map=class_map) if seed.label == label]
    if len(items) > count:
        items = rng.sample(items, count)
    return [seed.image for seed in items]


def load_response_scope_for_seed(config: ExperimentConfig, label: int, layer_sizes: dict[str, int]) -> InternalResponseScope:
    return load_response_scope(config.response_scope_config, label, layer_sizes=layer_sizes)


def calibrate_zeta_ranges(
    config: ExperimentConfig,
    bundle: ModelBundle,
    actions,
    rng: random.Random,
) -> dict[str, dict[str, dict[str, float]]]:
    bounds = StabilityCoverage.load_bounds(config.bounds)
    zeta: dict[str, dict[str, dict[str, float]]] = {}
    for label in range(bundle.num_classes):
        response_scope = load_response_scope_for_seed(config, label, bundle.layer_sizes)
        coverage = StabilityCoverage(
            bundle.model,
            bundle.preprocess,
            response_scope,
            bounds=bounds,
            zeta=None,
            num_bins=config.num_bins,
        )
        values = {"basc": [], "iasc": [], "irsc": []}
        for image in class_image_samples(config, label, config.zeta_images_per_class, rng, bundle.image_size):
            images = [image]
            for _ in range(config.zeta_mutants_per_image):
                images.append(rng.choice(actions).apply(image))
            snapshot, _ = coverage.evaluate(images)
            values["basc"].append(snapshot.basc)
            values["iasc"].append(snapshot.iasc)
            values["irsc"].append(snapshot.irsc)
        zeta[str(label)] = {
            name: {
                "min": float(np.min(vals)) if vals else 0.0,
                "max": float(np.max(vals)) if vals else 1.0,
            }
            for name, vals in values.items()
        }
    return zeta


def load_or_calibrate_zeta(config: ExperimentConfig, bundle: ModelBundle, actions, rng: random.Random) -> tuple[dict, Path, float]:
    zeta_path = config.zeta_path or config.output_dir / "zeta_ranges.json"
    if not config.use_zeta:
        return {}, zeta_path, 0.0
    start = time.perf_counter()
    if config.reuse_zeta and zeta_path.exists():
        return read_json(zeta_path), zeta_path, 0.0
    zeta = calibrate_zeta_ranges(config, bundle, actions, rng)
    elapsed = time.perf_counter() - start
    write_json(zeta_path, zeta)
    return zeta, zeta_path, elapsed


def nvidia_smi_snapshot() -> dict[str, Any]:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return {}
    if not output:
        return {}
    name, mem_used, mem_total, util = [part.strip() for part in output.splitlines()[0].split(",")]
    return {
        "gpu_name": name,
        "memory_used_mib": float(mem_used),
        "memory_total_mib": float(mem_total),
        "utilization_percent": float(util),
    }


def config_to_json(config: Any) -> dict[str, Any]:
    data = asdict(config)
    for key, value in list(data.items()):
        if isinstance(value, Path):
            data[key] = str(value)
    return data

