import io
import os
import uuid
import time
import threading
import traceback
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torch

from worldgen import WorldGen

app = FastAPI(title="WorldGen API", version="0.1")

# Enable CORS so local files/apps can call the API without rebuilding the container
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OUTPUT_DIR = os.getenv("OUTPUT_DIR", "/data")
os.makedirs(OUTPUT_DIR, exist_ok=True)
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _log_path(job_id: str) -> str:
    return os.path.join(LOG_DIR, f"{job_id}.log")


def _log(job_id: str, msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(_log_path(job_id), "a", encoding="utf-8") as f:
        f.write(f"[{ts}] {msg}\n")
    # Mirror to container stdout for `docker compose logs -f`
    print(f"[{ts}] ({job_id}) {msg}")


def _startup_diag() -> None:
    """Print basic environment diagnostics at startup to aid debugging."""
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    hf_home = os.getenv("HF_HOME")
    tok1 = bool(os.getenv("HF_TOKEN"))
    tok2 = bool(os.getenv("HUGGING_FACE_HUB_TOKEN"))
    hv = os.getenv("HF_HUB_VERBOSE")
    ht = os.getenv("HF_HUB_ENABLE_HF_TRANSFER")
    print(f"[{ts}] (init) HF_HOME={hf_home!r} HF_TOKEN={'set' if tok1 else 'missing'} HUGGING_FACE_HUB_TOKEN={'set' if tok2 else 'missing'} HF_HUB_VERBOSE={hv!r} HF_HUB_ENABLE_HF_TRANSFER={ht!r}")


# Emit startup diagnostics once when the module is imported
_startup_diag()


def _heartbeat(job_id: str, t0: float, stop_evt: threading.Event, interval: float = 2.0) -> None:
    while not stop_evt.is_set():
        elapsed = time.time() - t0
        cuda = torch.cuda.is_available()
        mem = None
        if cuda:
            try:
                mem = {
                    "alloc": torch.cuda.memory_allocated() // (1024*1024),
                    "reserved": torch.cuda.memory_reserved() // (1024*1024),
                }
            except Exception:
                mem = None
        _log(job_id, f"heartbeat: running, elapsed={elapsed:.1f}s" + (f", cuda_mem(MB)={mem}" if mem else ""))
        stop_evt.wait(interval)


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.post("/generate/text")
async def generate_text(
    prompt: str = Form(..., description="Text prompt to describe the scene"),
    low_vram: bool = Form(False),
    return_mesh: bool = Form(False),
    inpaint_bg: bool = Form(False),
):
    try:
        uid = str(uuid.uuid4())
        _log(uid, f"START text generation: low_vram={low_vram} return_mesh={return_mesh} inpaint_bg={inpaint_bg}")
        hb_stop = threading.Event()
        t0 = time.time()
        hb_thr = threading.Thread(target=_heartbeat, args=(uid, t0, hb_stop), daemon=True)
        hb_thr.start()

        worldgen = WorldGen(mode="t2s", device=_device(), low_vram=low_vram, inpaint_bg=inpaint_bg)
        _log(uid, "WorldGen initialized")
        result = worldgen.generate_world(prompt)
        _log(uid, "World generated; saving output")
        if return_mesh:
            # Save mesh
            try:
                import open3d as o3d  # type: ignore
            except Exception as e:
                hb_stop.set()
                _log(uid, f"ERROR: Open3D import failed: {e}")
                return JSONResponse(status_code=500, content={"error": f"Open3D import failed: {e}", "job_id": uid})
            mesh_path = os.path.join(OUTPUT_DIR, f"{uid}.ply")
            o3d.io.write_triangle_mesh(mesh_path, result)
            hb_stop.set()
            _log(uid, f"DONE in {time.time()-t0:.1f}s -> {mesh_path}")
            return {"type": "mesh", "path": mesh_path, "job_id": uid}
        else:
            # Save splat/pointcloud
            ply_path = os.path.join(OUTPUT_DIR, f"{uid}.ply")
            try:
                result.save(ply_path)
            except AttributeError:
                # Fallback if object provides a different saving API
                if hasattr(result, "to_ply"):
                    with open(ply_path, "wb") as f:
                        f.write(result.to_ply())
                else:
                    hb_stop.set()
                    _log(uid, "ERROR: Unknown result type; cannot save .ply")
                    return JSONResponse(status_code=500, content={"error": "Unknown result type; cannot save .ply", "job_id": uid})
            hb_stop.set()
            _log(uid, f"DONE in {time.time()-t0:.1f}s -> {ply_path}")
            return {"type": "splat", "path": ply_path, "job_id": uid}
    except Exception as e:
        jid = uid if 'uid' in locals() else 'unknown'
        _log(jid, f"ERROR: {e}\n{traceback.format_exc()}")
        return JSONResponse(status_code=500, content={"error": str(e), "job_id": (uid if 'uid' in locals() else None)})


@app.post("/generate/image")
async def generate_image(
    image: UploadFile = File(...),
    prompt: Optional[str] = Form(None),
    low_vram: bool = Form(False),
    return_mesh: bool = Form(False),
    inpaint_bg: bool = Form(False),
):
    try:
        uid = str(uuid.uuid4())
        _log(uid, f"START image generation: low_vram={low_vram} return_mesh={return_mesh} inpaint_bg={inpaint_bg}")
        hb_stop = threading.Event()
        t0 = time.time()
        hb_thr = threading.Thread(target=_heartbeat, args=(uid, t0, hb_stop), daemon=True)
        hb_thr.start()

        contents = await image.read()
        pil = Image.open(io.BytesIO(contents)).convert("RGB")
        worldgen = WorldGen(mode="i2s", device=_device(), low_vram=low_vram, inpaint_bg=inpaint_bg)
        _log(uid, "WorldGen initialized")
        result = worldgen.generate_world(image=pil, prompt=prompt or "")
        _log(uid, "World generated; saving output")
        if return_mesh:
            try:
                import open3d as o3d  # type: ignore
            except Exception as e:
                hb_stop.set()
                _log(uid, f"ERROR: Open3D import failed: {e}")
                return JSONResponse(status_code=500, content={"error": f"Open3D import failed: {e}", "job_id": uid})
            mesh_path = os.path.join(OUTPUT_DIR, f"{uid}.ply")
            o3d.io.write_triangle_mesh(mesh_path, result)
            hb_stop.set()
            _log(uid, f"DONE in {time.time()-t0:.1f}s -> {mesh_path}")
            return {"type": "mesh", "path": mesh_path, "job_id": uid}
        else:
            ply_path = os.path.join(OUTPUT_DIR, f"{uid}.ply")
            try:
                result.save(ply_path)
            except AttributeError:
                if hasattr(result, "to_ply"):
                    with open(ply_path, "wb") as f:
                        f.write(result.to_ply())
                else:
                    hb_stop.set()
                    _log(uid, "ERROR: Unknown result type; cannot save .ply")
                    return JSONResponse(status_code=500, content={"error": "Unknown result type; cannot save .ply", "job_id": uid})
            hb_stop.set()
            _log(uid, f"DONE in {time.time()-t0:.1f}s -> {ply_path}")
            return {"type": "splat", "path": ply_path, "job_id": uid}
    except Exception as e:
        jid = uid if 'uid' in locals() else 'unknown'
        _log(jid, f"ERROR: {e}\n{traceback.format_exc()}")
        return JSONResponse(status_code=500, content={"error": str(e), "job_id": (uid if 'uid' in locals() else None)})


@app.get("/")
def root():
    return {"service": "worldgen", "health": "/healthz", "endpoints": ["POST /generate/text", "POST /generate/image"]}




@app.get("/files")
def list_files(limit: int = 50):
    """List generated files in OUTPUT_DIR sorted by modified time desc."""
    try:
        entries = []
        for name in os.listdir(OUTPUT_DIR):
            path = os.path.join(OUTPUT_DIR, name)
            if os.path.isfile(path):
                entries.append((name, os.path.getmtime(path), os.path.getsize(path)))
        entries.sort(key=lambda x: x[1], reverse=True)
        files = [
            {"name": n, "size": s, "modified": m, "download": f"/files/{n}"}
            for n, m, s in entries[: max(1, min(limit, 200))]
        ]
        return {"files": files}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/files/{name}")
def download_file(name: str):
    """Download a generated file by name. Prevent path traversal."""
    safe_name = os.path.basename(name)
    if safe_name != name:
        return JSONResponse(status_code=400, content={"error": "invalid file name"})
    path = os.path.join(OUTPUT_DIR, safe_name)
    if not os.path.isfile(path):
        return JSONResponse(status_code=404, content={"error": "file not found"})
    return FileResponse(path, media_type="application/octet-stream", filename=safe_name)


@app.get("/jobs/{job_id}/logs")
def stream_logs(job_id: str):
    """Stream logs for a given job id using Server-Sent Events (SSE)."""
    path = _log_path(job_id)
    if not os.path.exists(path):
        return JSONResponse(status_code=404, content={"error": "job not found"})

    def event_stream():
        # stream existing content, then append new lines
        with open(path, "r", encoding="utf-8") as f:
            # Send existing lines first
            for line in f:
                yield f"data: {line.rstrip()}\n\n"
            # Then stream new lines as they are written
            while True:
                where = f.tell()
                line = f.readline()
                if not line:
                    time.sleep(1.0)
                    f.seek(where)
                else:
                    yield f"data: {line.rstrip()}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
