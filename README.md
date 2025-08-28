# WorldGen Docker + Localhost API

This setup builds a GPU-enabled container for the official WorldGen repository and exposes a simple REST API on localhost:8000 to generate 3D scenes from text or images.

- Dockerfile: `docker/Dockerfile`
- API server: `app/server.py`
- Compose: `docker-compose.yml`

## Prerequisites
- NVIDIA GPU with recent drivers (compatible with CUDA 12.8)
- Docker Desktop with WSL2 backend (Windows) and GPU support enabled
- NVIDIA Container Toolkit installed and working with Docker (so `--gpus all` works)

Verify with:
```
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

## Build
```
docker compose build
```

## Run
This maps port 8000 and a local `./data` folder for outputs (PLY files):
```
docker compose up
```
Then open http://localhost:8000 for the API root, and http://localhost:8000/docs for Swagger UI.

Alternatively with `docker run`:
```
docker run --rm --gpus all -p 8000:8000 -v %cd%/data:/data worldgen-api:latest
```

## API
- Health: `GET /healthz`
- Root: `GET /`
- Generate from text: `POST /generate/text`
  - Form fields: `prompt` (str, required), `low_vram` (bool), `return_mesh` (bool), `inpaint_bg` (bool)
- Generate from image: `POST /generate/image`
  - Multipart: `image` (file, required)
  - Form fields: `prompt` (optional), `low_vram` (bool), `return_mesh` (bool), `inpaint_bg` (bool)

### Curl examples
Note for Windows PowerShell: use `curl.exe` (not the PowerShell alias) and quote with double quotes.

Text-to-scene (PowerShell):
```
curl.exe -X POST http://localhost:8000/generate/text -F "prompt=A cozy wooden cabin in a snowy forest at night, soft warm lights" -F "low_vram=false"
```

Image-to-scene (PowerShell):
```
curl.exe -X POST http://localhost:8000/generate/image -F "image=@C:\\path\\to\\image.jpg" -F "prompt=Generate a 3D scene matching this image" -F "low_vram=false"
```

Bash examples (Linux/macOS):
```
curl -X POST http://localhost:8000/generate/text \
  -F "prompt=A cozy wooden cabin in a snowy forest at night, soft warm lights" \
  -F "low_vram=false"

curl -X POST http://localhost:8000/generate/image \
  -F image=@example.jpg \
  -F "prompt=Generate a 3D scene matching this image" \
  -F "low_vram=false"
```

Responses include the saved path under `/data` (mounted to `./data`).

## Environment (.env) and Hugging Face
First-time runs may need to download gated weights from Hugging Face. Create a `.env` at the project root (compose loads it automatically) with your token:

```
HUGGING_FACE_HUB_TOKEN=hf_xxx
HF_TOKEN=hf_xxx
# Optional accelerator for downloads
HF_HUB_ENABLE_HF_TRANSFER=1
```

Accept any gated model licenses on their Hugging Face pages, then recreate the container:
```
docker compose up -d --force-recreate
```

If behind a proxy, also add (example):
```
HTTP_PROXY=http://user:pass@proxy:port
HTTPS_PROXY=http://user:pass@proxy:port
```
If `hf-transfer` conflicts with the proxy, set `HF_HUB_ENABLE_HF_TRANSFER=0`.

## Notes
- This image installs PyTorch/TorchVision CUDA 12.8 wheels and then `pip install .` for WorldGen. Heavy downloads are expected on first run.
- VRAM usage: the API parameters default to full usage (`low_vram=false` if unchecked). VRAM will increase as models load; you already have access to the full GPU.
- If you intend to use the background inpainting feature, exec into the container and run:
```
pip install iopaint --no-dependencies
```
Then restart the container.

## Troubleshooting
- If `torch.cuda.is_available()` is false inside the container:
  - Ensure Docker Desktop has GPU support enabled and WSL2 backend is used.
  - Ensure NVIDIA Container Toolkit is installed.
  - Try `docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi`.
- Build failures on `open3d`/`trimesh`/GL libs: the Dockerfile installs `libgl1`, `libxext6`, `libsm6`, `libxrender1`, and `libglib2.0-0` to satisfy typical requirements.

- Hugging Face "cannot find requested files" at runtime:
  - Add `HUGGING_FACE_HUB_TOKEN`/`HF_TOKEN` to `.env` and accept the model licenses.
  - Recreate the container: `docker compose up -d --force-recreate`.
  - If behind a proxy, set `HTTP_PROXY`/`HTTPS_PROXY`. If issues persist, set `HF_HUB_ENABLE_HF_TRANSFER=0`.

## Monitoring
- Logs: `docker compose logs -f worldgen-api`
- GPU usage: `docker compose exec worldgen-api nvidia-smi -l 1`
