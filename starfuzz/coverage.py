from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .hooks import ActivationExtractor
from .response_scope import InternalResponseScope


@dataclass
class StabilitySnapshot:
    basc: float
    iasc: float
    irsc: float
    total: float
    variation: float
    vector: np.ndarray
    bins: set[tuple[str, int, int]]


def entropy(values: np.ndarray, low: float, high: float, bins: int) -> tuple[float, set[int]]:
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low = float(np.min(values))
        high = float(np.max(values))
        if high <= low:
            high = low + 1e-6
    clipped = np.clip(values.astype(np.float64), low, high)
    ids = np.floor((clipped - low) / (high - low + 1e-12) * bins).astype(int)
    ids = np.clip(ids, 0, bins - 1)
    counts = np.bincount(ids, minlength=bins).astype(np.float64)
    prob = counts[counts > 0] / max(1.0, counts.sum())
    h = float(-(prob * np.log2(prob)).sum()) if prob.size else 0.0
    return h / max(1e-12, math.log2(bins)), set(int(i) for i in np.unique(ids))


class StabilityCoverage:
    def __init__(
        self,
        model: torch.nn.Module,
        preprocess,
        response_scope: InternalResponseScope,
        bounds: dict | None = None,
        zeta: dict | None = None,
        num_bins: int = 15,
        weights: tuple[float, float, float] = (1.0, 1.0, 1.2),
    ):
        self.model = model
        self.preprocess = preprocess
        self.response_scope = response_scope
        self.bounds = bounds or {}
        self.zeta = zeta or {}
        self.num_bins = int(num_bins)
        self.weights = weights
        self.device = next(model.parameters()).device

    @staticmethod
    def load_bounds(path: Path | None) -> dict | None:
        if path is None:
            return None
        if not path.exists():
            raise FileNotFoundError(path)
        return json.loads(path.read_text(encoding="utf-8"))

    def _bound(self, label: int, layer: str, index: int) -> tuple[float, float]:
        class_bounds = self.bounds.get(str(label), self.bounds.get(label, {}))
        layer_bounds = class_bounds.get(layer, {})
        value = layer_bounds.get(str(index), layer_bounds.get(index))
        if isinstance(value, list) and len(value) >= 2:
            return float(value[0]), float(value[1])
        if isinstance(value, dict):
            return float(value.get("min", 0.0)), float(value.get("max", 1.0))
        return float("nan"), float("nan")

    def _normalize_metric(self, name: str, raw: float) -> float:
        info = self.zeta.get(str(self.response_scope.label), self.zeta.get(self.response_scope.label, self.zeta))
        value = info.get(name) if isinstance(info, dict) else None
        if not isinstance(value, dict):
            return float(raw)
        zmin = float(value.get("min", raw))
        zmax = float(value.get("max", raw))
        if zmax <= zmin:
            return float(raw)
        return float(np.clip((raw - zmin) / (zmax - zmin), 0.0, 1.0))

    def tensorize(self, images: list[np.ndarray]) -> torch.Tensor:
        tensors = [self.preprocess(Image.fromarray(img)) for img in images]
        return torch.stack(tensors, dim=0).to(self.device)

    @staticmethod
    def response_variation(values: np.ndarray, low: float, high: float) -> np.ndarray:
        baseline = float(values[0])
        if values.size <= 1:
            return np.asarray([0.0], dtype=np.float64)
        if np.isfinite(low) and np.isfinite(high) and high > low:
            scale = max(1e-12, high - low)
        else:
            scale = max(1.0, abs(baseline))
        return np.abs(values[1:].astype(np.float64) - baseline) / scale

    @torch.no_grad()
    def evaluate(self, images: list[np.ndarray]) -> tuple[StabilitySnapshot, torch.Tensor]:
        grouped = self.response_scope.by_layer()
        with ActivationExtractor(self.model, grouped.keys()) as extractor:
            logits = self.model(self.tensorize(images))
            outputs = extractor.outputs

        basc_scores: list[float] = []
        layer_scores: list[float] = []
        vector_parts: list[float] = []
        reached_bins: set[tuple[str, int, int]] = set()

        for layer, indices in grouped.items():
            layer_values = []
            acts = outputs[layer].numpy()
            for idx in indices:
                if idx < 0 or idx >= acts.shape[1]:
                    continue
                values = acts[:, idx]
                low, high = self._bound(self.response_scope.label, layer, idx)
                variation = self.response_variation(values, low, high)
                h, bins = entropy(variation, 0.0, max(1.0, float(np.max(variation))), self.num_bins)
                basc_scores.append(h)
                layer_values.append(h)
                vector_parts.append(float(np.mean(variation)))
                reached_bins.update((layer, idx, b) for b in bins)
            if layer_values:
                layer_scores.append(float(np.mean(layer_values)))

        basc_raw = float(np.mean(basc_scores)) if basc_scores else 0.0
        iasc_raw = float(np.mean(layer_scores)) if layer_scores else 0.0
        if len(layer_scores) > 1:
            mu = float(np.mean(layer_scores))
            var = float(np.var(layer_scores))
            irsc_raw = float((mu ** 3) / (var + mu ** 2 + 1e-12))
            irsc_raw = max(0.0, min(1.0, irsc_raw))
        else:
            irsc_raw = iasc_raw
        basc = self._normalize_metric("basc", basc_raw)
        iasc = self._normalize_metric("iasc", iasc_raw)
        irsc = self._normalize_metric("irsc", irsc_raw)
        total = self.weights[0] * basc + self.weights[1] * iasc + self.weights[2] * irsc
        vector = np.asarray(vector_parts, dtype=np.float64)
        variation = float(np.mean(vector)) if vector.size else 0.0
        return StabilitySnapshot(basc, iasc, irsc, total, variation, vector, reached_bins), logits.detach().cpu()


def response_variation_score(snapshot: StabilitySnapshot) -> float:
    return float(snapshot.variation)
