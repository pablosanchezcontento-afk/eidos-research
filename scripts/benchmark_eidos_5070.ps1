$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

Write-Host "Installing EIDOS in editable mode..." -ForegroundColor Cyan
python -m pip install -e .

Write-Host "Running correctness tests..." -ForegroundColor Cyan
python -m pytest -q

Write-Host "Checking CUDA..." -ForegroundColor Cyan
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"

New-Item -ItemType Directory -Force -Path results\gpu | Out-Null

Write-Host "EIDOS GPU benchmark without torch.compile..." -ForegroundColor Cyan
python .\benchmarks\benchmark_gpu.py `
  --config .\configs\eidos_glyph_10m.json `
  --seq-len 512 `
  --micro-batch 1 `
  --steps 30 `
  --warmup 8 `
  --precision bf16 `
  --optimizer adafactor `
  --out .\results\gpu\eidos_glyph_5070_eager.json

Write-Host "EIDOS GPU benchmark with torch.compile..." -ForegroundColor Cyan
python .\benchmarks\benchmark_gpu.py `
  --config .\configs\eidos_glyph_10m.json `
  --seq-len 512 `
  --micro-batch 1 `
  --steps 30 `
  --warmup 8 `
  --precision bf16 `
  --optimizer adafactor `
  --compile `
  --compile-mode default `
  --out .\results\gpu\eidos_glyph_5070_compile.json

Write-Host "Done. Send results/gpu/eidos_glyph_5070_compile.json for analysis." -ForegroundColor Green
