$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
python .\experiments\select_lrs.py `
  --runs-dir .\runs\glyph_calibration `
  --lr-grid 0.0003 0.0006 0.0012 0.0025 `
  --out .\runs\glyph_selected_lrs.json
