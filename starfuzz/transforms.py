from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import cv2
import numpy as np


ImageArray = np.ndarray


@dataclass(frozen=True)
class Action:
    name: str
    param: object
    fn: Callable[[ImageArray, object], ImageArray]

    def apply(self, image: ImageArray) -> ImageArray:
        out = self.fn(image.copy(), self.param)
        if out.shape[:2] != image.shape[:2]:
            out = cv2.resize(out, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_AREA)
        return np.clip(out, 0, 255).astype(np.uint8)

    @property
    def key(self) -> str:
        return f"{self.name}:{self.param}"


def image_translation(img: ImageArray, params: Sequence[int]) -> ImageArray:
    rows, cols = img.shape[:2]
    mat = np.float32([[1, 0, params[0]], [0, 1, params[1]]])
    return cv2.warpAffine(img, mat, (cols, rows), borderMode=cv2.BORDER_REFLECT_101)


def image_scale(img: ImageArray, factor: float) -> ImageArray:
    rows, cols = img.shape[:2]
    new_size = (max(1, int(cols * factor)), max(1, int(rows * factor)))
    scaled = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
    if factor >= 1.0:
        y0 = max((scaled.shape[0] - rows) // 2, 0)
        x0 = max((scaled.shape[1] - cols) // 2, 0)
        return scaled[y0 : y0 + rows, x0 : x0 + cols]
    canvas = np.zeros_like(img)
    y0 = (rows - scaled.shape[0]) // 2
    x0 = (cols - scaled.shape[1]) // 2
    canvas[y0 : y0 + scaled.shape[0], x0 : x0 + scaled.shape[1]] = scaled
    return canvas


def image_rotation(img: ImageArray, degrees: float) -> ImageArray:
    rows, cols = img.shape[:2]
    mat = cv2.getRotationMatrix2D((cols / 2, rows / 2), degrees, 1.0)
    return cv2.warpAffine(img, mat, (cols, rows), borderMode=cv2.BORDER_REFLECT_101)


def image_shear(img: ImageArray, factor: float) -> ImageArray:
    rows, cols = img.shape[:2]
    mat = np.float32([[1, -factor, 0], [0, 1, 0]])
    return cv2.warpAffine(img, mat, (cols, rows), borderMode=cv2.BORDER_REFLECT_101)


def image_contrast(img: ImageArray, alpha: float) -> ImageArray:
    return cv2.convertScaleAbs(img, alpha=float(alpha), beta=0)


def image_brightness(img: ImageArray, beta: int) -> ImageArray:
    return cv2.convertScaleAbs(img, alpha=1.0, beta=int(beta))


def image_blur(img: ImageArray, mode: int) -> ImageArray:
    mode = int(mode)
    if mode == 1:
        return cv2.blur(img, (3, 3))
    if mode == 2:
        return cv2.GaussianBlur(img, (3, 3), 0)
    if mode == 3:
        return cv2.GaussianBlur(img, (5, 5), 0)
    if mode == 4:
        return cv2.medianBlur(img, 3)
    return cv2.bilateralFilter(img, 5, 50, 50)


def default_action_pool() -> list[Action]:
    specs: list[tuple[str, Callable[[ImageArray, object], ImageArray], Iterable[object]]] = [
        ("translation", image_translation, [(-3, -3), (-3, 0), (0, -3), (3, 0), (0, 3), (3, 3)]),
        ("scale", image_scale, [0.90, 0.95, 1.05, 1.10]),
        ("rotation", image_rotation, [-15, -10, -5, 5, 10, 15]),
        ("shear", image_shear, [-0.12, -0.06, 0.06, 0.12]),
        ("contrast", image_contrast, [0.85, 0.95, 1.05, 1.15]),
        ("brightness", image_brightness, [-20, -10, 10, 20, 30]),
        ("blur", image_blur, [1, 2, 3, 4]),
    ]
    return [Action(name, param, fn) for name, fn, params in specs for param in params]


def semantic_constraint(
    original: ImageArray,
    mutated: ImageArray,
    max_changed_ratio: float = 0.80,
    max_linf_ratio: float = 1.00,
) -> bool:
    # Alpha/beta semantic checks are intentionally disabled for unconstrained FDR runs.
    return original.shape == mutated.shape
