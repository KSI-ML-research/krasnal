# Cloud Training & Inference with Docker

This guide provides two container variants:
- CPU (works on every Docker host).
- GPU/CUDA (requires NVIDIA GPU, drivers and NVIDIA Container Toolkit).

Suitable for cloud VMs/services that support Docker.

## 0) Local PyTorch choice (non-Docker usage)

Base setup (fast, without torch):

```bash
uv sync
```

Choose torch variant:

```bash
# CPU
uv run python scripts/install_torch.py --target cpu

# CUDA 12.4
uv run python scripts/install_torch.py --target cu124
```

## 1) Build + run training

From repository root:

CPU:

```bash
uv run python scripts/docker_train.py --target cpu --run-eval
```

GPU/CUDA:

```bash
uv run python scripts/docker_train.py --target gpu --run-eval
```

Optional full pipeline (preprocess + train + eval):

```bash
uv run python scripts/docker_train.py --target cpu --run-preprocess --run-eval
```

What it does:
- Builds `docker/Dockerfile.train` into `krasnal-train:latest`.
- Mounts local `data/` and `models/` into the container.
- Runs training (and optional preprocess/eval) inside Docker.

## 2) Build + run inference API

Start HTTP inference service on port 8000:

CPU:

```bash
uv run python scripts/docker_inference.py --target cpu --provider pytorch --port 8000
```

GPU/CUDA:

```bash
uv run python scripts/docker_inference.py --target gpu --provider pytorch --port 8000
```

Health check:

Linux/macOS:

```bash
curl http://localhost:8000/health
```

Windows PowerShell:

```powershell
curl.exe http://localhost:8000/health
# or (native PowerShell)
Invoke-RestMethod -Method Get -Uri "http://localhost:8000/health"
```

Predict move:

Linux/macOS:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"moves":"e2e4 e7e5 g1f3"}'
```

Windows PowerShell:

```powershell
# recommended (native PowerShell)
$body = @{ moves = "e2e4 e7e5 g1f3" } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri "http://localhost:8000/predict" -ContentType "application/json" -Body $body

# alternative (curl.exe)
$json = '{"moves":"e2e4 e7e5 g1f3"}'
curl.exe -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" --data-binary $json
```

Check if curl exists on Windows:

```powershell
Get-Command curl.exe
```

If you see `{"error": "invalid_json"}` in PowerShell, use the native `Invoke-RestMethod` variant above (recommended).

## 3) Runtime configuration

- `ENGINE_PROVIDER`: `pytorch` (default in inference image) or `mock`.
- `ENGINE_TEMPERATURE`: sampling temperature (default `0.0` for greedy).
- `ENGINE_TOP_P`: nucleus sampling cutoff (default `1.0`).

The PyTorch provider expects a model checkpoint at `models/chess_model.pt`.

## 4) Notes about CUDA

- `--target gpu` uses dedicated GPU Dockerfiles and runs container with `--gpus all`.
- If CUDA is unavailable, use `--target cpu`.
- `--provider mock` works without model checkpoint and is useful for smoke tests.

## 5) PR: How to test

Copy-paste this section into your PR description.

### CPU path

Linux/macOS:

```bash
# training + eval in Docker
uv run python scripts/docker_train.py --target cpu --run-eval

# run inference API in Docker (new terminal)
uv run python scripts/docker_inference.py --target cpu --provider pytorch --port 8000

# health
curl http://localhost:8000/health

# sample prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"moves":"e2e4 e7e5 g1f3"}'
```

Windows PowerShell:

```powershell
uv run python scripts/docker_train.py --target cpu --run-eval
uv run python scripts/docker_inference.py --target cpu --provider pytorch --port 8000

curl.exe http://localhost:8000/health
$body = @{ moves = "e2e4 e7e5 g1f3" } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri "http://localhost:8000/predict" -ContentType "application/json" -Body $body
```

### GPU/CUDA path

Linux/macOS:

```bash
# training + eval in Docker
uv run python scripts/docker_train.py --target gpu --run-eval

# run inference API in Docker (new terminal)
uv run python scripts/docker_inference.py --target gpu --provider pytorch --port 8000

# health
curl http://localhost:8000/health

# sample prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"moves":"e2e4 e7e5 g1f3"}'
```

Windows PowerShell:

```powershell
uv run python scripts/docker_train.py --target gpu --run-eval
uv run python scripts/docker_inference.py --target gpu --provider pytorch --port 8000

curl.exe http://localhost:8000/health
$body = @{ moves = "e2e4 e7e5 g1f3" } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri "http://localhost:8000/predict" -ContentType "application/json" -Body $body
```
