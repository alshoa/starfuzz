from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from .benchmarks import DATASETS, MODELS, default_data_dir, default_layers, default_model_for_dataset, load_benchmark
from .datasets import Seed, iter_seeds, load_class_map
from .response_scope import load_response_scope


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="StaRFuzz stability-aware DNN fuzzer")
    sub = parser.add_subparsers(dest="cmd", required=True)

    common_model = argparse.ArgumentParser(add_help=False)
    common_model.add_argument("--dataset", default="cifar10", choices=DATASETS)
    common_model.add_argument("--model", default=None, choices=MODELS)
    common_model.add_argument("--weights", type=Path, default=None)
    common_model.add_argument("--device", default=None)
    common_model.add_argument("--project-root", type=Path, default=None)

    p = sub.add_parser(
        "prepare-response-stats",
        parents=[common_model],
        help="compute class/layer/internal-response activation statistics",
    )
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--label", type=int, default=None)
    p.add_argument("--max-images", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--image-size", type=int, default=None)
    p.add_argument("--layers", nargs="+", default=None)

    p = sub.add_parser(
        "select-response-scope",
        parents=[common_model],
        help="select class-discriminative internal responses by F-score",
    )
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-images", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--image-size", type=int, default=None)
    p.add_argument("--top-per-layer", type=int, default=5)
    p.add_argument("--top-ratio", type=float, default=None, help="select this ratio of responses per layer")
    p.add_argument("--layers", nargs="+", default=None)

    p = sub.add_parser("fuzz", parents=[common_model], help="run stability coverage-guided fuzzing")
    p.add_argument("--seed-dir", type=Path, default=None)
    p.add_argument("--response-scope-config", type=Path, required=True)
    p.add_argument("--bounds", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=Path("starfuzz_runs"))
    p.add_argument("--label", type=int, default=None)
    p.add_argument("--max-seeds", type=int, default=10)
    p.add_argument("--budget", type=int, default=100)
    p.add_argument("--rollout-depth", type=int, default=20)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--num-bins", type=int, default=15)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--image-size", type=int, default=None)
    p.add_argument("--layers", nargs="+", default=None)

    sub.add_parser("summarize-paper", help="print a concise summary of StaRFuzz")
    return parser


def load_model_and_preprocess(args):
    model_name = args.model or default_model_for_dataset(args.dataset)
    bundle = load_benchmark(args.dataset, model_name, weights=args.weights, device=args.device, workspace=args.project_root)
    return bundle.model, bundle.preprocess, bundle.device, bundle.dataset, bundle.model_spec


def selected_layers(args) -> list[str]:
    return args.layers or default_layers(args.model or default_model_for_dataset(args.dataset), args.project_root)


def selected_image_size(args, dataset_spec) -> int:
    return int(args.image_size or dataset_spec.image_size)


def selected_data_dir(args, seed: bool = False) -> Path:
    path = args.seed_dir if seed and hasattr(args, "seed_dir") else getattr(args, "data_dir", None)
    return path or default_data_dir(args.dataset, args.project_root, seed=seed)


def infer_layer_sizes(model, preprocess, layers: list[str], image_size: int) -> dict[str, int]:
    import numpy as np
    import torch
    from PIL import Image

    from .hooks import ActivationExtractor

    dummy = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    with torch.no_grad(), ActivationExtractor(model, layers) as extractor:
        model(torch.stack([preprocess(Image.fromarray(dummy))], dim=0).to(next(model.parameters()).device))
    return {layer: int(value.shape[1]) for layer, value in extractor.outputs.items()}


def batch(items: list[Seed], size: int):
    for start in range(0, len(items), size):
        yield items[start : start + size]


def command_prepare_response_stats(args) -> None:
    import torch
    from PIL import Image

    from .hooks import ActivationExtractor

    model, preprocess, _device, dataset_spec, _model_spec = load_model_and_preprocess(args)
    data_dir = selected_data_dir(args)
    image_size = selected_image_size(args, dataset_spec)
    layers = selected_layers(args)
    class_map = load_class_map(data_dir, args.dataset)
    seeds = list(iter_seeds(data_dir, label=args.label, max_images=args.max_images, image_size=image_size, class_map=class_map))
    if not seeds:
        raise SystemExit(f"No images found under {data_dir}")
    stats: dict[str, dict[str, dict[str, list[float]]]] = {}
    for group in batch(seeds, args.batch_size):
        tensors = torch.stack([preprocess(Image.fromarray(seed.image)) for seed in group], dim=0).to(next(model.parameters()).device)
        with torch.no_grad(), ActivationExtractor(model, layers) as extractor:
            model(tensors)
        for seed_idx, seed in enumerate(group):
            class_stats = stats.setdefault(str(seed.label), {layer: {} for layer in layers})
            for layer, out in extractor.outputs.items():
                values = out[seed_idx].numpy()
                layer_stats = class_stats.setdefault(layer, {})
                for i, value in enumerate(values):
                    old = layer_stats.get(str(i))
                    v = float(value)
                    if old is None:
                        layer_stats[str(i)] = [v, v]
                    else:
                        old[0] = min(old[0], v)
                        old[1] = max(old[1], v)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"saved response statistics: {args.output}")


def command_select_response_scope(args) -> None:
    import numpy as np
    import torch
    from PIL import Image

    from .hooks import ActivationExtractor

    model, preprocess, _device, dataset_spec, _model_spec = load_model_and_preprocess(args)
    data_dir = selected_data_dir(args)
    image_size = selected_image_size(args, dataset_spec)
    layers = selected_layers(args)
    class_map = load_class_map(data_dir, args.dataset)
    seeds = list(iter_seeds(data_dir, max_images=args.max_images, image_size=image_size, class_map=class_map))
    if not seeds:
        raise SystemExit(f"No images found under {data_dir}")
    labels = np.asarray([s.label for s in seeds], dtype=np.int64)
    activations: dict[str, list[np.ndarray]] = {layer: [] for layer in layers}
    for group in batch(seeds, args.batch_size):
        tensors = torch.stack([preprocess(Image.fromarray(seed.image)) for seed in group], dim=0).to(next(model.parameters()).device)
        with torch.no_grad(), ActivationExtractor(model, layers) as extractor:
            model(tensors)
        for layer, out in extractor.outputs.items():
            activations[layer].append(out.numpy())

    acts = {layer: np.concatenate(parts, axis=0) for layer, parts in activations.items()}
    result: dict[str, dict[str, list]] = {}
    for label in sorted(set(int(x) for x in labels)):
        selected: list[tuple[str, int, float]] = []
        mask = labels == label
        other = ~mask
        if mask.sum() < 2 or other.sum() < 2:
            continue
        for layer in layers:
            layer_acts = acts[layer]
            mu_pos = layer_acts[mask].mean(axis=0)
            mu_neg = layer_acts[other].mean(axis=0)
            var_pos = layer_acts[mask].var(axis=0)
            var_neg = layer_acts[other].var(axis=0)
            f_score = ((mu_pos - mu_neg) ** 2) / (var_pos + var_neg + 1e-12)
            if args.top_ratio is None:
                top_count = args.top_per_layer
            else:
                top_count = max(1, int(round(layer_acts.shape[1] * args.top_ratio)))
            top = np.argsort(-f_score)[: min(top_count, layer_acts.shape[1])]
            selected.extend((layer, int(idx), float(f_score[idx])) for idx in top)
        result[str(label)] = {
            "responses": [idx for _layer, idx, _score in selected],
            "layers": [layer for layer, _idx, _score in selected],
            "scores": [score for _layer, _idx, score in selected],
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"saved response scope config: {args.output}")


def node_action_sequence(node) -> list[str]:
    return [n.action.key for n in node.path() if n.action is not None]


def save_node_image(node, path: Path) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(node.image).save(path)


def command_fuzz(args) -> None:
    import numpy as np
    import torch

    from .coverage import StabilityCoverage
    from .mcts import Evaluation, MCTSFuzzer
    from .transforms import default_action_pool

    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    model, preprocess, _device, dataset_spec, _model_spec = load_model_and_preprocess(args)
    image_size = selected_image_size(args, dataset_spec)
    layers = selected_layers(args)
    seed_dir = selected_data_dir(args, seed=True)
    layer_sizes = infer_layer_sizes(model, preprocess, layers, image_size)
    bounds = StabilityCoverage.load_bounds(args.bounds)
    class_map = load_class_map(seed_dir, args.dataset)
    seeds = list(iter_seeds(seed_dir, label=args.label, max_images=args.max_seeds, image_size=image_size, class_map=class_map))
    if not seeds:
        raise SystemExit(f"No images found under {seed_dir}")
    actions = default_action_pool()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_results = []

    for seed_no, seed in enumerate(seeds):
        response_scope = load_response_scope(args.response_scope_config, seed.label, layer_sizes=layer_sizes)
        coverage = StabilityCoverage(model, preprocess, response_scope, bounds=bounds, num_bins=args.num_bins)
        _original_snapshot, original_logits = coverage.evaluate([seed.image])
        inherited_label = int(seed.label)
        original_pred = int(original_logits.argmax(dim=1)[0])

        def evaluate(images: list[np.ndarray]) -> Evaluation:
            snapshot, logits = coverage.evaluate(images)
            final_pred = int(logits.argmax(dim=1)[-1])
            return Evaluation(stability_coverage=snapshot.total, pred=final_pred)

        fuzzer = MCTSFuzzer(
            seed.image,
            inherited_label,
            actions,
            evaluate,
            rollout_depth=args.rollout_depth,
            rng=rng,
        )
        result = fuzzer.run(args.budget, top_k=args.top_k)
        seed_dir = args.output_dir / f"seed_{seed_no:04d}_label_{inherited_label}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        records = []
        for rank, node in enumerate(result.best_nodes):
            path_images = [item.image for item in node.path()]
            snapshot, logits = coverage.evaluate(path_images)
            pred = int(logits.argmax(dim=1)[-1])
            image_path = seed_dir / f"rank_{rank:03d}_pred_{pred}_reward_{node.avg_reward:.4f}.png"
            save_node_image(node, image_path)
            records.append(
                {
                    "rank": rank,
                    "image": str(image_path),
                    "prediction": pred,
                    "is_model_failure": pred != inherited_label,
                    "avg_reward": node.avg_reward,
                    "visits": node.visits,
                    "stability_coverage": {
                        "basc": snapshot.basc,
                        "iasc": snapshot.iasc,
                        "irsc": snapshot.irsc,
                        "total": snapshot.total,
                    },
                    "response_variation": snapshot.variation,
                    "actions": node_action_sequence(node),
                }
            )

        summary = {
            "seed_path": str(seed.path),
            "label": inherited_label,
            "original_prediction": original_pred,
            "response_scope_layers": response_scope.layers,
            "iterations": result.total_iterations,
            "rollout_failures": result.total_failures,
            "top_nodes": records,
        }
        (seed_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        all_results.append(summary)
        print(f"[{seed_no + 1}/{len(seeds)}] saved {len(records)} candidates in {seed_dir}")

    (args.output_dir / "run_summary.json").write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"saved run summary: {args.output_dir / 'run_summary.json'}")


def command_summarize_paper(_args) -> None:
    print(
        "StaRFuzz performs stability-aware DNN fuzzing: semantic-preserving image transformations "
        "induce internal response variation, BaSC/IaSC/IrSC summarize the variation distribution "
        "as Stability Coverage, and MCTS uses Stability Coverage Gain to expose model-level failures."
    )


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.cmd == "prepare-response-stats":
        command_prepare_response_stats(args)
    elif args.cmd == "select-response-scope":
        command_select_response_scope(args)
    elif args.cmd == "fuzz":
        command_fuzz(args)
    elif args.cmd == "summarize-paper":
        command_summarize_paper(args)
    else:
        raise SystemExit(f"unknown command: {args.cmd}")


if __name__ == "__main__":
    main()
