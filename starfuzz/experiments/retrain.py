from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from starfuzz.benchmarks import default_data_dir
from starfuzz.datasets import iter_seeds, load_class_map
from starfuzz.experiments.common import ModelConfig, WORKSPACE, load_model_bundle, set_reproducible, write_json


@dataclass
class RetrainConfig:
    fdr_summary: Path
    output_dir: Path
    dataset: str = "cifar10"
    clean_seed_dir: Path | None = None
    max_clean_per_class: int = 200
    max_failures: int = 1000
    batch_size: int = 64
    epochs: int = 3
    lr: float = 1e-4
    rng_seed: int = 20260701
    save_name: str = "vgg11_starfuzz_retrained.pt"


class ImagePathDataset(Dataset):
    def __init__(self, samples: list[tuple[Path, int]], preprocess):
        self.samples = samples
        self.preprocess = preprocess

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        return self.preprocess(image), int(label)


def clean_samples(dataset: str, seed_dir: Path, max_per_class: int, rng: random.Random) -> list[tuple[Path, int]]:
    class_map = load_class_map(seed_dir, dataset)
    by_label: dict[int, list[Path]] = {}
    for seed in iter_seeds(seed_dir, class_map=class_map):
        by_label.setdefault(seed.label, []).append(seed.path)
    samples: list[tuple[Path, int]] = []
    for label, paths in by_label.items():
        paths = sorted(paths)
        if len(paths) > max_per_class:
            paths = rng.sample(paths, max_per_class)
        samples.extend((path, label) for path in paths)
    return samples


def failure_samples_from_summary(summary_path: Path, max_failures: int, rng: random.Random) -> list[tuple[Path, int]]:
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    items = []
    for cand in data.get("top_candidates", []):
        image = cand.get("image")
        if cand.get("is_model_failure") and image:
            path = Path(image)
            if path.exists():
                items.append((path, int(cand["label"])))
    if len(items) > max_failures:
        items = rng.sample(items, max_failures)
    return items


@torch.no_grad()
def evaluate_dataset(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    total = correct = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        pred = model(images).argmax(dim=1)
        total += int(labels.numel())
        correct += int((pred == labels).sum().item())
    acc = correct / max(1, total)
    return {"total": total, "correct": correct, "accuracy": acc, "fdr": 1.0 - acc}


def run_retrain_repair(config: RetrainConfig, model_cfg: ModelConfig | None = None) -> Path:
    """Retrain/repair experiment.

    The FDR summary must be produced with `save_candidate_images=True`, because
    retraining needs the generated failure images on disk.
    """

    rng = set_reproducible(config.rng_seed)
    model_cfg = model_cfg or ModelConfig(dataset=config.dataset, device="cuda")
    model_cfg.dataset = config.dataset
    bundle = load_model_bundle(model_cfg)
    device = bundle.device
    clean_seed_dir = config.clean_seed_dir or default_data_dir(config.dataset, WORKSPACE, seed=True)
    clean = clean_samples(config.dataset, clean_seed_dir, config.max_clean_per_class, rng)
    failures = failure_samples_from_summary(config.fdr_summary, config.max_failures, rng)
    if not failures:
        raise RuntimeError("No saved failure images found. Run FDR with save_candidate_images enabled first.")

    config.output_dir.mkdir(parents=True, exist_ok=True)
    train_samples = clean + failures
    rng.shuffle(train_samples)
    train_loader = DataLoader(
        ImagePathDataset(train_samples, bundle.preprocess),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
    )
    clean_loader = DataLoader(
        ImagePathDataset(clean, bundle.preprocess),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
    )
    failure_loader = DataLoader(
        ImagePathDataset(failures, bundle.preprocess),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
    )

    before_clean = evaluate_dataset(bundle.model, clean_loader, device)
    before_failures = evaluate_dataset(bundle.model, failure_loader, device)

    optimizer = torch.optim.Adam(bundle.model.parameters(), lr=config.lr)
    bundle.model.train()
    history = []
    for epoch in range(config.epochs):
        total_loss = 0.0
        seen = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(bundle.model(images), labels)
            loss.backward()
            optimizer.step()
            batch = int(labels.numel())
            seen += batch
            total_loss += float(loss.item()) * batch
        history.append({"epoch": epoch + 1, "loss": total_loss / max(1, seen)})

    after_clean = evaluate_dataset(bundle.model, clean_loader, device)
    after_failures = evaluate_dataset(bundle.model, failure_loader, device)
    checkpoint = config.output_dir / config.save_name
    torch.save(bundle.model.state_dict(), checkpoint)
    summary = {
        "config": {
            "fdr_summary": str(config.fdr_summary),
            "output_dir": str(config.output_dir),
            "dataset": config.dataset,
            "clean_seed_dir": str(clean_seed_dir),
            "max_clean_per_class": config.max_clean_per_class,
            "max_failures": config.max_failures,
            "batch_size": config.batch_size,
            "epochs": config.epochs,
            "lr": config.lr,
            "rng_seed": config.rng_seed,
        },
        "num_clean": len(clean),
        "num_failures": len(failures),
        "history": history,
        "before": {"clean": before_clean, "failures": before_failures},
        "after": {"clean": after_clean, "failures": after_failures},
        "checkpoint": str(checkpoint),
    }
    out = config.output_dir / "retrain_summary.json"
    write_json(out, summary)
    return out


