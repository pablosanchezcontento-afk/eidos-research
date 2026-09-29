param(
  [Parameter(Mandatory=$true)][string]$TrainBin,
  [Parameter(Mandatory=$true)][string]$ValBin
)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
python .\experiments\build_matrix.py `
  --config .\configs\eidos_glyph_10m.json `
  --train-bin $TrainBin `
  --val-bin $ValBin `
  --out-dir .\runs\glyph_calibration `
  --script-out .\runs\run_glyph_calibration.ps1 `
  --phase calibration `
  --tokens-per-param 2 `
  --lrs 0.0003 0.0006 0.0012 0.0025 `
  --seeds 1337
Write-Host "Generated runs/run_glyph_calibration.ps1. Calibration tunes LR only; it does not prove architecture quality." -ForegroundColor Green
