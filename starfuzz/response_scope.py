from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class InternalResponseRef:
    layer: str
    index: int


@dataclass(frozen=True)
class InternalResponseScope:
    label: int
    responses: tuple[InternalResponseRef, ...]

    @property
    def layers(self) -> list[str]:
        return list(dict.fromkeys(response.layer for response in self.responses))

    def by_layer(self) -> dict[str, list[int]]:
        grouped: dict[str, list[int]] = {}
        for response in self.responses:
            grouped.setdefault(response.layer, []).append(response.index)
        return grouped


def _layer_offsets(layer_sizes: dict[str, int]) -> dict[str, int]:
    offset = 0
    out = {}
    for layer, size in layer_sizes.items():
        out[layer] = offset
        offset += size
    return out


def local_index(layer: str, raw_index: int, layer_sizes: dict[str, int] | None = None) -> int:
    if layer_sizes is None:
        return int(raw_index)
    size = layer_sizes.get(layer)
    if size is None or 0 <= raw_index < size:
        return int(raw_index)
    offsets = _layer_offsets(layer_sizes)
    idx = raw_index - offsets.get(layer, 0)
    if 0 <= idx < size:
        return int(idx)
    return int(raw_index)


def load_response_scope(
    path: Path,
    label: int,
    layer_sizes: dict[str, int] | None = None,
    top_n: int | None = None,
) -> InternalResponseScope:
    data = json.loads(path.read_text(encoding="utf-8"))
    item = data[str(label)]
    responses = item["responses"]
    layers = item["layers"]
    refs = [
        InternalResponseRef(str(layer), local_index(str(layer), int(index), layer_sizes))
        for index, layer in zip(responses, layers, strict=True)
    ]
    if top_n is not None:
        refs = refs[:top_n]
    return InternalResponseScope(label=label, responses=tuple(refs))


def response_scope_from_refs(label: int, refs: Iterable[tuple[str, int]]) -> InternalResponseScope:
    return InternalResponseScope(
        label=label,
        responses=tuple(InternalResponseRef(layer, int(index)) for layer, index in refs),
    )
