from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch


DATASETS = ("mnist", "cifar10", "imagenet", "tiny-imagenet")
MODELS = (
    "lenet4",
    "lenet5",
    "alexnet",
    "vgg11_bn",
    "vgg13_bn",
    "vgg16_bn",
    "resnet20",
    "resnet34",
    "densenet121",
    "densenet161",
)


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    num_classes: int
    image_size: int
    channels: int
    mean: tuple[float, ...]
    std: tuple[float, ...]
    default_data_dir: Path
    default_seed_dir: Path


@dataclass(frozen=True)
class ModelSpec:
    name: str
    default_dataset: str
    default_weights: Path | None
    activation_layers: tuple[str, ...]


@dataclass(frozen=True)
class BenchmarkBundle:
    dataset: DatasetSpec
    model_spec: ModelSpec
    model: torch.nn.Module
    preprocess: Callable
    device: torch.device


def resolve_workspace(start: Path | None = None) -> Path:
    here = (start or Path.cwd()).resolve()
    for candidate in [here, *here.parents]:
        if (candidate / "cifar10_models").exists() or ((candidate / "data").exists() and (candidate / "OriModel").exists()):
            return candidate
    return here


def normalize_name(value: str) -> str:
    return value.strip().lower().replace("_", "-").replace(".", "-")


def canonical_dataset(dataset: str) -> str:
    name = normalize_name(dataset)
    aliases = {
        "cifar-10": "cifar10",
        "tinyimagenet": "tiny-imagenet",
        "tiny-imagenet-200": "tiny-imagenet",
    }
    name = aliases.get(name, name)
    if name not in DATASETS:
        raise ValueError(f"Unsupported dataset {dataset}. Choose one of: {', '.join(DATASETS)}")
    return name


def canonical_model(model: str) -> str:
    name = normalize_name(model)
    aliases = {
        "lenet-4": "lenet4",
        "lenet-5": "lenet5",
        "vgg11-bn": "vgg11_bn",
        "vgg13-bn": "vgg13_bn",
        "vgg16-bn": "vgg16_bn",
        "resnet-20": "resnet20",
        "resnet-34": "resnet34",
        "densenet-121": "densenet121",
        "densenet-161": "densenet161",
    }
    name = aliases.get(name, name)
    if name not in MODELS:
        raise ValueError(f"Unsupported model {model}. Choose one of: {', '.join(MODELS)}")
    return name


def dataset_specs(workspace: Path | None = None) -> dict[str, DatasetSpec]:
    root = resolve_workspace(workspace)
    return {
        "mnist": DatasetSpec(
            name="mnist",
            num_classes=10,
            image_size=28,
            channels=1,
            mean=(0.1307,),
            std=(0.3081,),
            default_data_dir=root / "data" / "Mnist_png" / "mnist_png",
            default_seed_dir=root / "data" / "Mnist_png" / "mnist_png",
        ),
        "cifar10": DatasetSpec(
            name="cifar10",
            num_classes=10,
            image_size=32,
            channels=3,
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2470, 0.2435, 0.2616),
            default_data_dir=root / "data" / "cifar10_png" / "cifar10",
            default_seed_dir=root / "seed",
        ),
        "imagenet": DatasetSpec(
            name="imagenet",
            num_classes=1000,
            image_size=224,
            channels=3,
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            default_data_dir=root / "data" / "imagenet" / "val",
            default_seed_dir=root / "data" / "imagenet" / "val",
        ),
        "tiny-imagenet": DatasetSpec(
            name="tiny-imagenet",
            num_classes=200,
            image_size=64,
            channels=3,
            mean=(0.4802, 0.4481, 0.3975),
            std=(0.2770, 0.2691, 0.2821),
            default_data_dir=root / "data" / "tiny-imagenet-200" / "train",
            default_seed_dir=root / "data" / "tiny-imagenet-200" / "train",
        ),
    }


def _existing(path: Path) -> Path | None:
    return path if path.exists() else None


def model_specs(workspace: Path | None = None) -> dict[str, ModelSpec]:
    root = resolve_workspace(workspace)
    state = root / "cifar10_models" / "state_dicts"
    return {
        "lenet4": ModelSpec("lenet4", "mnist", _existing(root / "OriModel" / "LeNet1.pth"), ("fc1", "fc2", "fc3")),
        "lenet5": ModelSpec("lenet5", "mnist", _existing(root / "OriModel" / "LeNet5.pth"), ("fc1", "fc2", "fc3")),
        "alexnet": ModelSpec("alexnet", "imagenet", None, ("classifier.1", "classifier.4", "classifier.6")),
        "vgg11_bn": ModelSpec("vgg11_bn", "cifar10", _existing(state / "vgg11_bn.pt"), ("classifier.0", "classifier.3", "classifier.6")),
        "vgg13_bn": ModelSpec("vgg13_bn", "cifar10", _existing(state / "vgg13_bn.pt"), ("classifier.0", "classifier.3", "classifier.6")),
        "vgg16_bn": ModelSpec("vgg16_bn", "cifar10", _existing(state / "vgg16_bn.pt"), ("classifier.0", "classifier.3", "classifier.6")),
        "resnet20": ModelSpec("resnet20", "cifar10", _existing(state / "resnet20.pt"), ("layer1", "layer2", "layer3")),
        "resnet34": ModelSpec("resnet34", "cifar10", _existing(state / "resnet34.pt"), ("layer2", "layer3", "layer4")),
        "densenet121": ModelSpec("densenet121", "cifar10", _existing(state / "densenet121.pt"), ("features.denseblock2", "features.denseblock3", "features.denseblock4")),
        "densenet161": ModelSpec("densenet161", "cifar10", _existing(state / "densenet161.pt"), ("features.denseblock2", "features.denseblock3", "features.denseblock4")),
    }


def get_dataset_spec(dataset: str, workspace: Path | None = None) -> DatasetSpec:
    return dataset_specs(workspace)[canonical_dataset(dataset)]


def get_model_spec(model: str, workspace: Path | None = None) -> ModelSpec:
    return model_specs(workspace)[canonical_model(model)]


def default_model_for_dataset(dataset: str) -> str:
    return {
        "mnist": "lenet5",
        "cifar10": "vgg11_bn",
        "imagenet": "resnet34",
        "tiny-imagenet": "resnet34",
    }[canonical_dataset(dataset)]


def default_data_dir(dataset: str, workspace: Path | None = None, seed: bool = False) -> Path:
    spec = get_dataset_spec(dataset, workspace)
    return spec.default_seed_dir if seed else spec.default_data_dir


def default_weights(model: str, workspace: Path | None = None) -> Path | None:
    return get_model_spec(model, workspace).default_weights


def default_layers(model: str, workspace: Path | None = None) -> list[str]:
    return list(get_model_spec(model, workspace).activation_layers)


def make_preprocess(dataset: DatasetSpec, device: torch.device):
    from torchvision import transforms

    steps = []
    if dataset.channels == 1:
        steps.append(transforms.Grayscale(num_output_channels=1))
    steps.extend(
        [
            transforms.Resize((dataset.image_size, dataset.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=dataset.mean, std=dataset.std),
            lambda x: x.to(device),
        ]
    )
    return transforms.Compose(steps)


class LeNet5(torch.nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 6, kernel_size=5, stride=1)
        self.pool1 = torch.nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = torch.nn.Conv2d(6, 16, kernel_size=5, stride=1)
        self.pool2 = torch.nn.AvgPool2d(kernel_size=2, stride=2)
        self.fc1 = torch.nn.Linear(256, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class LeNet4(torch.nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 4, kernel_size=5, stride=1)
        self.pool1 = torch.nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = torch.nn.Conv2d(4, 12, kernel_size=5, stride=1)
        self.pool2 = torch.nn.AvgPool2d(kernel_size=2, stride=2)
        self.fc1 = torch.nn.Linear(192, 84)
        self.fc2 = torch.nn.Linear(84, 32)
        self.fc3 = torch.nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


def _strip_module_prefix(state_dict: dict) -> dict:
    if not any(k.startswith("module.") for k in state_dict.keys()):
        return state_dict
    return {k.removeprefix("module."): v for k, v in state_dict.items()}


def _load_state(model: torch.nn.Module, weights: Path | None) -> None:
    if weights is None:
        return
    if not weights.exists():
        raise FileNotFoundError(f"Model weights not found: {weights}")
    state = torch.load(str(weights), map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(_strip_module_prefix(state))


def _load_cifar_model(model_name: str, weights: Path | None, root: Path, num_classes: int) -> torch.nn.Module:
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from cifar10_models import densenet, resnet, vgg

    builders = {
        "vgg11_bn": vgg.vgg11_bn,
        "vgg13_bn": vgg.vgg13_bn,
        "vgg16_bn": vgg.vgg16_bn,
        "resnet34": resnet.resnet34,
        "densenet121": densenet.densenet121,
        "densenet161": densenet.densenet161,
    }
    if model_name == "resnet20":
        raise ValueError("resnet20 is listed for paper completeness, but no ResNet20 builder is present in this artifact.")
    if model_name not in builders:
        raise ValueError(f"Model {model_name} is not available for CIFAR-10 in this artifact.")
    model = builders[model_name]()
    _load_state(model, weights)
    return model


def _load_torchvision_model(model_name: str, num_classes: int) -> torch.nn.Module:
    from torchvision import models

    builders = {
        "alexnet": models.alexnet,
        "resnet34": models.resnet34,
        "densenet121": models.densenet121,
        "densenet161": models.densenet161,
        "vgg11_bn": models.vgg11_bn,
        "vgg13_bn": models.vgg13_bn,
        "vgg16_bn": models.vgg16_bn,
    }
    if model_name not in builders:
        raise ValueError(f"Model {model_name} is not available through torchvision.")
    return builders[model_name](weights=None, num_classes=num_classes)


def load_model(
    dataset: str,
    model: str,
    weights: Path | None = None,
    device: str | torch.device | None = None,
    workspace: Path | None = None,
) -> torch.nn.Module:
    root = resolve_workspace(workspace)
    dataset_spec = get_dataset_spec(dataset, root)
    model_spec = get_model_spec(model, root)
    model_name = model_spec.name
    if weights is None and dataset_spec.name == model_spec.default_dataset:
        weights = model_spec.default_weights

    if dataset_spec.name == "mnist":
        if model_name == "lenet4":
            loaded = LeNet4(dataset_spec.num_classes)
        elif model_name == "lenet5":
            loaded = LeNet5(dataset_spec.num_classes)
        else:
            raise ValueError(f"Dataset MNIST supports lenet4/lenet5 in this artifact, not {model_name}.")
        _load_state(loaded, weights)
    elif dataset_spec.name == "cifar10":
        loaded = _load_cifar_model(model_name, weights, root, dataset_spec.num_classes)
    else:
        loaded = _load_torchvision_model(model_name, dataset_spec.num_classes)
        _load_state(loaded, weights)

    target_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    loaded.eval()
    loaded.to(target_device)
    return loaded


def load_benchmark(
    dataset: str,
    model: str | None = None,
    weights: Path | None = None,
    device: str | torch.device | None = None,
    workspace: Path | None = None,
) -> BenchmarkBundle:
    root = resolve_workspace(workspace)
    dataset_name = canonical_dataset(dataset)
    model_name = canonical_model(model or default_model_for_dataset(dataset_name))
    dataset_spec = get_dataset_spec(dataset_name, root)
    model_spec = get_model_spec(model_name, root)
    loaded = load_model(dataset_name, model_name, weights=weights, device=device, workspace=root)
    target_device = next(loaded.parameters()).device
    return BenchmarkBundle(
        dataset=dataset_spec,
        model_spec=model_spec,
        model=loaded,
        preprocess=make_preprocess(dataset_spec, target_device),
        device=target_device,
    )
