$ErrorActionPreference = "Stop"

Set-Location (Join-Path $PSScriptRoot "..")

python .\main.py summarize-paper

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


