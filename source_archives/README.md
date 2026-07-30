# Source-complete archive

The source-complete EIDOS v0.4 snapshot is stored as seven Base64 chunks because the connected GitHub writer only supports UTF-8 text uploads.

## Reconstruct on Linux/macOS

```bash
cat eidos-v0.4-source-complete.tar.gz.b64.p* > eidos-v0.4-source-complete.tar.gz.b64
base64 --decode eidos-v0.4-source-complete.tar.gz.b64 > eidos-v0.4-source-complete.tar.gz
sha256sum eidos-v0.4-source-complete.tar.gz
mkdir eidos-v0.4-source-complete
tar -xzf eidos-v0.4-source-complete.tar.gz -C eidos-v0.4-source-complete
```

Expected SHA-256:

```text
784872f11473cd120aa2eda69db9eae687565d2220d788fcefece14441c97fc9
```

## Reconstruct on PowerShell

```powershell
$parts = Get-ChildItem .\eidos-v0.4-source-complete.tar.gz.b64.p* | Sort-Object Name
$b64 = ($parts | ForEach-Object { Get-Content $_.FullName -Raw }) -join ''
[IO.File]::WriteAllBytes('.\eidos-v0.4-source-complete.tar.gz', [Convert]::FromBase64String($b64))
Get-FileHash .\eidos-v0.4-source-complete.tar.gz -Algorithm SHA256
mkdir .\eidos-v0.4-source-complete
 tar -xzf .\eidos-v0.4-source-complete.tar.gz -C .\eidos-v0.4-source-complete
```

This archive contains the latest **source-complete** EIDOS snapshot: model, baselines, training pipeline, data pipeline, benchmarks, tests, experiment gates, configs, reports source and preserved outputs. It does not contain the later unrecovered Sparse16 working tree.
