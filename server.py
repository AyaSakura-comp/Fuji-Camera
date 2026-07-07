#!/usr/bin/env python3
"""
Fuji Camera — picture pool + serial FLUX.2 film-style processing server.

Frontend uploads photos -> stored in a pool (data/) -> a single worker thread
runs `create_image.py "analog, AnalogRedmAF, F1.2 shallow depth of field, 35mm analog film photo, soft contrast, fine film grain, subtle halation, cinematic bokeh" --refcontrol --image <photo>` one at a time (FLUX.2
RefControl analog flow) -> results saved. Status (pending/processing/done/error)
is persisted in data/db.json and survives restarts.
"""
import io
import json
import os
import re
import subprocess
import tempfile
import threading
import time
import urllib.request
import uuid
from pathlib import Path
from queue import Queue

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps

# ---------------------------------------------------------------- paths
BASE   = Path(__file__).resolve().parent
DATA   = BASE / "data"
ORIG   = DATA / "originals"
THUMB  = DATA / "thumbs"
RESULT = DATA / "results"
DB     = DATA / "db.json"
for d in (ORIG, THUMB, RESULT):
    d.mkdir(parents=True, exist_ok=True)

HOME        = Path.home()
CREATE_IMG  = HOME / ".hermes/skills/create-image/scripts/create_image.py"
FLUX_PY     = HOME / "models-work/flux2/.venv-rocm72/bin/python"
FLUX_CWD    = HOME / "models-work/flux2"

# Host-side generation service (gen_service.py). The worker sends original image
# BYTES here over HTTP and gets processed BYTES back — no shared filesystem, so
# this server can run in a container / elsewhere and just point at the endpoint.
# In a container set FUJI_GEN_URL=http://host.docker.internal:7863.
# On the host (no gen service running) it falls back to a local create_image.py
# subprocess, so running server.py directly on the box still works standalone.
GEN_URL   = os.environ.get("FUJI_GEN_URL", "http://127.0.0.1:7863").rstrip("/")
GEN_STEPS = os.environ.get("FUJI_GEN_STEPS", "2")

THUMB_MAX = 500     # px, longest edge for gallery thumbnails
FULL_MAX  = 2200    # px, longest edge for the full-screen viewer image

# ---------------------------------------------------------------- db (thread-safe)
_db_lock = threading.Lock()

def load_db():
    if DB.exists():
        try:
            return json.loads(DB.read_text())
        except Exception:
            pass
    return {"photos": []}

def save_db(db):
    tmp = DB.with_suffix(".tmp")
    tmp.write_text(json.dumps(db, ensure_ascii=False, indent=2))
    tmp.replace(DB)

def find(db, pid):
    for p in db["photos"]:
        if p["id"] == pid:
            return p
    return None

# ---------------------------------------------------------------- image helpers
def make_thumb(src: Path, dst: Path, longest: int):
    im = Image.open(src)
    im = ImageOps.exif_transpose(im)
    im = im.convert("RGB")
    im.thumbnail((longest, longest), Image.LANCZOS)
    im.save(dst, "JPEG", quality=85)

# ---------------------------------------------------------------- worker queue
work_q: "Queue[str]" = Queue()

def enqueue(pid):
    work_q.put(pid)

# the style prompt (shared default with gen_service.py); overridable via env
FILM_PROMPT = os.environ.get(
    "FUJI_GEN_PROMPT",
    "analog, AnalogRedmAF, F1.2 shallow depth of field, 35mm analog film photo, "
    "soft contrast, fine film grain, subtle halation, cinematic bokeh")

def _gen_ready():
    try:
        with urllib.request.urlopen(GEN_URL + "/health", timeout=3) as r:
            return json.loads(r.read()).get("ready") is True
    except Exception:
        return False

def run_film(orig_bytes: bytes) -> bytes:
    """original image bytes -> processed film image bytes.
    Prefer the host gen service over HTTP (no shared FS); on the host, fall back
    to a local create_image.py subprocess so standalone runs still work."""
    if _gen_ready():
        req = urllib.request.Request(GEN_URL + "/film", data=orig_bytes,
                                     headers={"Content-Type": "image/jpeg"})
        with urllib.request.urlopen(req, timeout=1800) as r:
            return r.read()

    # fallback: local cold create_image.py subprocess (host with the FLUX venv)
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tf:
        tf.write(orig_bytes); tmp = tf.name
    try:
        env = dict(os.environ)
        env["FLUX2_BIG_WMMA_LINEAR"] = "1"
        cmd = [str(FLUX_PY), str(CREATE_IMG), FILM_PROMPT, "--refcontrol",
               "--steps", GEN_STEPS, "--image", tmp]
        proc = subprocess.run(cmd, cwd=str(FLUX_CWD), env=env,
                              capture_output=True, text=True, timeout=1800)
        if proc.returncode != 0:
            raise RuntimeError(f"create_image exit {proc.returncode}: "
                               + (proc.stderr or proc.stdout)[-800:])
        m = re.search(r'"final_path":\s*"([^"]+)"', proc.stdout)
        if not m:
            raise RuntimeError("no final_path in output: " + proc.stdout[-800:])
        final = Path(m.group(1))
        if not final.exists():
            raise RuntimeError(f"final_path missing on disk: {final}")
        return final.read_bytes()
    finally:
        try: os.unlink(tmp)
        except OSError: pass

def process_one(pid):
    with _db_lock:
        db = load_db()
        p = find(db, pid)
        if not p or p["status"] == "done":
            return
        p["status"] = "processing"
        p["started"] = time.time()
        save_db(db)

    err = None
    try:
        orig_bytes = (ORIG / p["orig"]).read_bytes()
        result_bytes = run_film(orig_bytes)

        # store a viewer-sized JPEG + a thumbnail of the result
        res_name   = f"{pid}.jpg"
        thumb_name = f"{pid}_r.jpg"
        im = ImageOps.exif_transpose(Image.open(io.BytesIO(result_bytes))).convert("RGB")
        im.thumbnail((FULL_MAX, FULL_MAX), Image.LANCZOS)
        im.save(RESULT / res_name, "JPEG", quality=92)
        im.thumbnail((THUMB_MAX, THUMB_MAX), Image.LANCZOS)
        im.save(THUMB / thumb_name, "JPEG", quality=85)

        with _db_lock:
            db = load_db()
            p = find(db, pid)
            if p:
                p["status"] = "done"
                p["result"] = res_name
                p["result_thumb"] = thumb_name
                p["finished"] = time.time()
                p.pop("error", None)
                save_db(db)
        return
    except subprocess.TimeoutExpired:
        err = "timeout (>1800s)"
    except Exception as e:
        err = str(e)

    with _db_lock:
        db = load_db()
        p = find(db, pid)
        if p:
            p["status"] = "error"
            p["error"] = err
            p["finished"] = time.time()
            save_db(db)

def worker_loop():
    while True:
        pid = work_q.get()
        try:
            process_one(pid)
        except Exception as e:
            print("worker error:", e, flush=True)
        finally:
            work_q.task_done()

# ---------------------------------------------------------------- app
app = FastAPI(title="Fuji Camera")

@app.on_event("startup")
def _startup():
    # requeue anything left pending / processing from a previous run
    with _db_lock:
        db = load_db()
        requeue = []
        for p in db["photos"]:
            if p["status"] in ("pending", "processing"):
                p["status"] = "pending"
                requeue.append(p["id"])
        save_db(db)
    for pid in requeue:
        enqueue(pid)
    t = threading.Thread(target=worker_loop, daemon=True)
    t.start()

@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    data = await file.read()
    if not data:
        raise HTTPException(400, "empty upload")
    pid = uuid.uuid4().hex[:12]
    orig_name  = f"{pid}.jpg"
    thumb_name = f"{pid}.jpg"
    orig_path  = ORIG / orig_name

    # normalise to JPEG with correct orientation
    try:
        im = ImageOps.exif_transpose(Image.open(__import__("io").BytesIO(data))).convert("RGB")
        im.save(orig_path, "JPEG", quality=92)
    except Exception as e:
        raise HTTPException(400, f"bad image: {e}")
    make_thumb(orig_path, THUMB / thumb_name, THUMB_MAX)

    entry = {
        "id": pid,
        "status": "pending",
        "created": time.time(),
        "orig": orig_name,
        "orig_thumb": thumb_name,
    }
    with _db_lock:
        db = load_db()
        db["photos"].append(entry)
        save_db(db)
    enqueue(pid)
    return {"id": pid, "status": "pending"}

@app.get("/api/photos")
def list_photos():
    with _db_lock:
        db = load_db()
        photos = list(db["photos"])
    photos.sort(key=lambda p: p.get("created", 0), reverse=True)
    counts = {"pending": 0, "processing": 0, "done": 0, "error": 0}
    out = []
    for p in photos:
        counts[p["status"]] = counts.get(p["status"], 0) + 1
        out.append({
            "id": p["id"],
            "status": p["status"],
            "created": p.get("created"),
            "error": p.get("error"),
        })
    return {"photos": out, "counts": counts}

def _file_or_404(path: Path, media="image/jpeg"):
    if not path.exists():
        raise HTTPException(404, "not found")
    return FileResponse(str(path), media_type=media)

@app.get("/api/file/{pid}/thumb")
def get_thumb(pid: str):
    with _db_lock:
        p = find(load_db(), pid)
    if not p:
        raise HTTPException(404, "no such photo")
    if p["status"] == "done" and p.get("result_thumb"):
        return _file_or_404(THUMB / p["result_thumb"])
    return _file_or_404(THUMB / p.get("orig_thumb", f"{pid}.jpg"))

@app.get("/api/file/{pid}/full")
def get_full(pid: str):
    with _db_lock:
        p = find(load_db(), pid)
    if not p:
        raise HTTPException(404, "no such photo")
    if p["status"] == "done" and p.get("result"):
        return _file_or_404(RESULT / p["result"])
    return _file_or_404(ORIG / p.get("orig", f"{pid}.jpg"))

@app.get("/api/file/{pid}/orig")
def get_orig(pid: str):
    with _db_lock:
        p = find(load_db(), pid)
    if not p:
        raise HTTPException(404, "no such photo")
    return _file_or_404(ORIG / p.get("orig", f"{pid}.jpg"))

@app.delete("/api/photo/{pid}")
def delete_photo(pid: str):
    with _db_lock:
        db = load_db()
        p = find(db, pid)
        if not p:
            raise HTTPException(404, "no such photo")
        db["photos"] = [x for x in db["photos"] if x["id"] != pid]
        save_db(db)
    for f in (ORIG / p.get("orig", ""), THUMB / p.get("orig_thumb", ""),
              RESULT / p.get("result", ""), THUMB / p.get("result_thumb", "")):
        try:
            if f.name and f.exists():
                f.unlink()
        except Exception:
            pass
    return {"ok": True}

# static frontend (index.html at /) — mounted last so /api/* wins
app.mount("/", StaticFiles(directory=str(BASE / "static"), html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8090)
