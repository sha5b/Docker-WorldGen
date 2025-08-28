# WorldGen Docker + Localhost API

This setup builds a GPU-enabled container for the official WorldGen repository and exposes a simple REST API on localhost:8000 to generate 3D scenes from text or images.

- Dockerfile: `docker/Dockerfile`
- API server: `app/server.py`
- Compose: `docker-compose.yml`

## Prerequisites
- NVIDIA GPU with recent drivers
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
Text-to-scene (splat/ply output):
```
curl -X POST http://localhost:8000/generate/text \
  -F "prompt=A cozy wooden cabin in a snowy forest at night, soft warm lights" \
  -F "low_vram=true"
```

Image-to-scene:
```
curl -X POST http://localhost:8000/generate/image \
  -F image=@example.jpg \
  -F "prompt=Generate a 3D scene matching this image" \
  -F "low_vram=true"
```

Responses include the saved path under `/data` (mounted to `./data`).

## Notes
- This image installs PyTorch and TorchVision CUDA 12.8 wheels and then `pip install .` for WorldGen. Heavy downloads expected on first build.
- Use `low_vram=true` if your GPU has < 24GB VRAM (per README).
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
