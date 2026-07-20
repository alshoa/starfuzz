# StaRFuzz

StaRFuzz is a stability-aware fuzzing artifact for image-classification models. The artifact follows the current paper logic:

semantic-preserving transformations -> internal response variation -> Stability Coverage -> coverage-guided MCTS search -> model-level failure exposure.

## Method

StaRFuzz does not treat the final model output as the only fuzzing target. It tracks selected internal responses and asks how their responses vary under semantic-preserving input transformations.

The main workflow is:

1. Compute per-class internal response statistics from seed/training images.
2. Select an internal response scope with F-score over class-discriminative responses.
3. For a rollout path, compute response variation relative to the seed response.
4. Characterize the response-variation distribution with BaSC, IaSC, and IrSC.
5. Use MCTS with reward `Delta SCov` to search transformation sequences.
6. Report model-level failures separately as FDR/failure exposure results.

## Files

- `starfuzz/benchmarks.py`: dataset/model registry, default input paths, default weight paths, activation layers, and dataset-specific preprocessing.
- `starfuzz/response_scope.py`: response scope loading and global-to-local response index conversion.
- `starfuzz/coverage.py`: Stability Coverage over internal response variation, including BaSC, IaSC, and IrSC.
- `starfuzz/mcts.py`: stability coverage-guided Monte Carlo Tree Search.
- `starfuzz/paper_mcts.py`: paper-style MCTS used by the experiment runners.
- `starfuzz/transforms.py`: semantic-preserving image transformations.
- `starfuzz/hooks.py`: PyTorch activation hooks.
- `starfuzz/cli.py`: CLI commands for response statistics, response scope selection, and fuzzing.
- `starfuzz/experiments/`: FDR, speed, retrain/repair, and ablation experiment modules.

## Benchmarks

The artifact registers the four datasets used in the paper:

- `mnist`: default input `../data/Mnist_png/mnist_png`, image size 28, grayscale normalization `(0.1307)/(0.3081)`.
- `cifar10`: default input `../data/cifar10_png/cifar10`, default seed dir `../seed`, image size 32, CIFAR-10 normalization.
- `imagenet`: default input `../data/imagenet/val`, image size 224, ImageNet normalization.
- `tiny-imagenet`: default input `../data/tiny-imagenet-200/train`, image size 64, TinyImageNet normalization.

The ten MuTs are registered as:

`lenet4`, `lenet5`, `alexnet`, `vgg11_bn`, `vgg13_bn`, `vgg16_bn`, `resnet20`, `resnet34`, `densenet121`, and `densenet161`.

Default paths are also listed in `configs/benchmark_paths.json`. At runtime, StaRFuzz selects preprocessing, input size, class-label parsing, default hook layers, and default weights from `--dataset` and `--model`. If a weight file is not present locally, pass `--weights PATH`.

## Environment

```powershell
cd C:\Users\yrz\Desktop\work\ComFuzz\code\JuneFight\StaRFuzz
conda env create -f .\environment.yml
conda activate starfuzz
python -m pip install -e .
```

## Run

Paper summary:

```powershell
python .\main.py summarize-paper
```

Run a small VGG11/CIFAR-10 fuzzing demo:

```powershell
python .\main.py fuzz `
  --dataset cifar10 `
  --model vgg11_bn `
  --seed-dir ..\seed `
  --response-scope-config ..\ResponseScope\cifar10_vgg11.json `
  --bounds ..\boundary\cifar10_vgg11.json `
  --output-dir .\runs\vgg11_seed_demo `
  --max-seeds 2 `
  --budget 20 `
  --rollout-depth 8 `
  --top-k 5
```

Generate response statistics:

```powershell
python .\main.py prepare-response-stats `
  --dataset cifar10 `
  --model vgg11_bn `
  --data-dir ..\seed `
  --output .\runs\vgg11_response_stats.json `
  --max-images 200
```

Generate a response scope config with F-score:

```powershell
python .\main.py select-response-scope `
  --dataset cifar10 `
  --model vgg11_bn `
  --data-dir ..\seed `
  --output .\runs\vgg11_response_scope.json `
  --top-per-layer 5 `
  --max-images 1000
```

Outputs are saved as one folder per seed, including candidate PNGs and `summary.json`; the whole run also gets `run_summary.json`.

## Experiments

See [EXPERIMENTS.md](EXPERIMENTS.md) for FDR, speed, retrain/repair, ablation, and zeta calibration entry points.
