from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
from PIL import Image


CIFAR10_CLASSES = {
    "airplane": 0,
    "automobile": 1,
    "bird": 2,
    "cat": 3,
    "deer": 4,
    "dog": 5,
    "frog": 6,
    "horse": 7,
    "ship": 8,
    "truck": 9,
}

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


@dataclass(frozen=True)
class Seed:
    path: Path
    label: int
    image: np.ndarray


def load_class_map(root: Path, dataset: str | None = None) -> dict[str, int]:
    dataset_name = (dataset or "").lower()
    if dataset_name == "cifar10":
        return CIFAR10_CLASSES
    if dataset_name == "mnist":
        return {str(i): i for i in range(10)}
    wnids = root / "wnids.txt"
    if not wnids.exists():
        wnids = root.parent / "wnids.txt"
    if dataset_name in {"tiny-imagenet", "tinyimagenet"} and wnids.exists():
        return {line.strip(): idx for idx, line in enumerate(wnids.read_text(encoding="utf-8").splitlines()) if line.strip()}
    dirs = sorted(p.name for p in root.iterdir() if p.is_dir()) if root.exists() else []
    return {name.lower(): idx for idx, name in enumerate(dirs)}


def infer_label(path: Path, default_label: int | None = None, class_map: dict[str, int] | None = None) -> int:
    if default_label is not None:
        return int(default_label)
    class_map = {key.lower(): value for key, value in (class_map or CIFAR10_CLASSES).items()}
    for parent in [path.parent.name.lower(), path.parent.parent.name.lower()]:
        if parent in class_map:
            return int(class_map[parent])
        if parent.startswith("class_") and parent[6:].isdigit():
            return int(parent[6:])
        if parent.isdigit():
            return int(parent)
    stem_parts = path.stem.replace("-", "_").split("_")
    for part in stem_parts:
        if part.isdigit():
            value = int(part)
            if 0 <= value <= 999:
                return value
    raise ValueError(f"Cannot infer label from {path}; pass --label for a single-class seed directory.")


def load_image(path: Path, image_size: int | None = None) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    if image_size is not None:
        image = image.resize((image_size, image_size), Image.BILINEAR)
    return np.asarray(image, dtype=np.uint8)


def iter_seeds(
    root: Path,
    label: int | None = None,
    max_images: int | None = None,
    image_size: int | None = None,
    class_map: dict[str, int] | None = None,
) -> Iterator[Seed]:
    paths = sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
    if max_images is not None:
        paths = paths[: max(0, max_images)]
    for path in paths:
        yield Seed(path=path, label=infer_label(path, label, class_map), image=load_image(path, image_size))
