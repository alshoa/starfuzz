from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Iterable

import torch


def get_module(model: torch.nn.Module, name: str) -> torch.nn.Module:
    current: torch.nn.Module = model
    for part in name.split("."):
        if part.isdigit():
            current = current[int(part)]  # type: ignore[index]
        else:
            current = getattr(current, part)
    return current


class ActivationExtractor(AbstractContextManager["ActivationExtractor"]):
    def __init__(self, model: torch.nn.Module, layers: Iterable[str]):
        self.model = model
        self.layers = list(dict.fromkeys(layers))
        self.outputs: dict[str, torch.Tensor] = {}
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    def __enter__(self) -> "ActivationExtractor":
        for layer_name in self.layers:
            module = get_module(self.model, layer_name)
            self._handles.append(module.register_forward_hook(self._make_hook(layer_name)))
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def _make_hook(self, layer_name: str):
        def hook(_module, _inputs, output):
            if isinstance(output, tuple):
                output = output[0]
            self.outputs[layer_name] = output.detach().flatten(1).cpu()

        return hook

