from __future__ import annotations

from pathlib import Path

import torch

from .benchmarks import get_dataset_spec, load_model, make_preprocess, resolve_workspace


def resolve_project_root(start: Path | None = None) -> Path:
    return resolve_workspace(start)


def load_cifar_vgg(
    model_name: str,
    weights: Path | None = None,
    device: str | torch.device | None = None,
    project_root: Path | None = None,
) -> torch.nn.Module:
    return load_model("cifar10", model_name, weights=weights, device=device, workspace=project_root)


def cifar10_preprocess(device: torch.device):
    return make_preprocess(get_dataset_spec("cifar10"), device)
