$ErrorActionPreference = "Stop"

Set-Location (Join-Path $PSScriptRoot "..")

conda env create -f .\environment.yml
conda run -n starfuzz python -m pip install -e .

Write-Host "Done. Activate with: conda activate starfuzz"

