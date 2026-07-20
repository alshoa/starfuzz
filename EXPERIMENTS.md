# StaRFuzz Experiments

This folder contains the paper-style StaRFuzz experiment code.

## Main Method

- `starfuzz/paper_mcts.py`: paper-style stability coverage-guided MCTS.
- `starfuzz/coverage.py`: BaSC, IaSC, and IrSC over internal response variation.
- `starfuzz/experiments/fdr.py`: reusable FDR experiment module.
- `scripts/evaluate_paper_fdr.py`: standalone paper-style FDR runner.

The implementation includes full-tree UCB selection, root-seed mutation, rollout path sample sets, `SCov(S) - SCov(S_parent)` reward, node SCov cache, path back-propagation, multi-seed resource scheduling, and zeta range calibration for BaSC/IaSC/IrSC normalization.

Alpha/beta image constraints are currently disabled in `starfuzz/transforms.py`; only shape consistency is checked.

## FDR

Run FDR and save candidate images for retraining:

```powershell
cd C:\Users\yrz\Desktop\work\ComFuzz\code\JuneFight
C:\Users\yrz\anaconda3\envs\selection\python.exe StaRFuzz\scripts\run_fdr_experiment.py `
  --output-dir StaRFuzz\runs\paper_fdr_saved `
  --dataset cifar10 `
  --model vgg11_bn `
  --response-scope-config ResponseScope\cifar10_vgg11.json `
  --random-seeds 20 `
  --total-candidates 1000 `
  --budget-per-seed 2000 `
  --rollout-depth 8 `
  --save-candidate-images
```

Useful switches:

- `--all-seeds`: include originally wrong seeds.
- `--reuse-zeta --zeta-path PATH`: reuse calibrated zeta ranges.
- `--no-zeta`: ablate zeta normalization.
- `--coverage-mode basc|iasc|irsc`: isolate one Stability Coverage granularity.

## Speed

Run a small throughput benchmark:

```powershell
C:\Users\yrz\anaconda3\envs\selection\python.exe StaRFuzz\scripts\run_speed_experiment.py `
  --output-dir StaRFuzz\runs\speed_vgg11 `
  --dataset cifar10 `
  --model vgg11_bn `
  --random-seeds 5 `
  --budget-per-seed 100 `
  --total-candidates 100
```

Outputs:

- `fdr_summary.json`: zeta time, fuzz time, iterations/sec, and GPU snapshot.
- `speed_summary.json`: compact speed report.

## Retrain / Repair

Retrain VGG11 on generated model-level failure cases plus clean seeds:

```powershell
C:\Users\yrz\anaconda3\envs\selection\python.exe StaRFuzz\scripts\run_retrain_experiment.py `
  --fdr-summary StaRFuzz\runs\paper_fdr_saved\fdr_summary.json `
  --output-dir StaRFuzz\runs\retrain_repair `
  --dataset cifar10 `
  --model vgg11_bn `
  --epochs 3 `
  --batch-size 64 `
  --lr 1e-4
```

The FDR summary must be produced with `--save-candidate-images`, otherwise retraining has no generated images to load.

Outputs:

- `retrain_summary.json`: before/after clean accuracy and generated-failure FDR.
- `vgg11_starfuzz_retrained.pt`: repaired checkpoint.

## Ablation

Write an ablation plan only:

```powershell
C:\Users\yrz\anaconda3\envs\selection\python.exe StaRFuzz\scripts\run_ablation_experiment.py `
  --output-dir StaRFuzz\runs\ablation_plan
```

Actually run all ablations:

```powershell
C:\Users\yrz\anaconda3\envs\selection\python.exe StaRFuzz\scripts\run_ablation_experiment.py `
  --output-dir StaRFuzz\runs\ablation_full `
  --run
```

Default ablations include full method, no zeta normalization, no dynamic resource scheduling, rollout depth 16, all-seed oracle setting, response-scope ratio sweeps, and BaSC/IaSC/IrSC-only variants.
