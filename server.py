#!/usr/bin/env python3
"""
Fuji Camera — picture pool + serial FLUX.2 film-style processing server.

Frontend uploads photos -> stored in a pool (data/) -> a single worker thread
runs `create_image.py "analog, AnalogRedmAF, F1.2 shallow depth of field, 35mm analog film photo, soft contrast, fine film grain, subtle halation, cinematic bokeh" --refcontrol --image <photo>` one at a time (FLUX.2
RefControl analog flow) -> results saved. Status (pending/processing/done/error)
is persisted in data/db.json and survives restarts.
"""
import json
import os
import re
import subprocess
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

# Warm resident daemon (film_daemon.py). If it's up we use it (fast: no model
# reload per image); otherwise fall back to the cold create_image.py subprocess.
DAEMON_URL  = "http://127.0.0.1:7863"

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

def _daemon_ready():
    try:
        with urllib.request.urlopen(DAEMON_URL + "/health", timeout=3) as r:
            return json.loads(r.read()).get("ready") is True
    except Exception:
        return False

def run_film(orig_path: Path) -> Path:
    """Produce the film-styled PNG for orig_path; return its path.
    Prefer the warm resident daemon; fall back to the cold subprocess."""
    if _daemon_ready():
        req = urllib.request.Request(
            DAEMON_URL + "/film",
            data=json.dumps({"image": str(orig_path)}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=1800) as r:
            res = json.loads(r.read())
        if "final_path" not in res:
            raise RuntimeError("daemon error: " + json.dumps(res)[:800])
        final = Path(res["final_path"])
        if not final.exists():
            raise RuntimeError(f"daemon final_path missing: {final}")
        return final

    # fallback: cold create_image.py subprocess (reloads model each time)
    env = dict(os.environ)
    env["FLUX2_BIG_WMMA_LINEAR"] = "1"
    cmd = [str(FLUX_PY), str(CREATE_IMG), "analog, AnalogRedmAF, F1.2 shallow depth of field, 35mm analog film photo, soft contrast, fine film grain, subtle halation, cinematic bokeh", "--refcontrol",
           "--steps", "2", "--image", str(orig_path)]
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
    return final

def process_one(pid):
    with _db_lock:
        db = load_db()
        p = find(db, pid)
        if not p or p["status"] == "done":
            return
        p["status"] = "processing"
        p["started"] = time.time()
        save_db(db)

    orig_path = ORIG / p["orig"]
    err = None
    try:
        final = run_film(orig_path)

        # store a viewer-sized JPEG + a thumbnail of the result
        res_name   = f"{pid}.jpg"
        thumb_name = f"{pid}_r.jpg"
        im = ImageOps.exif_transpose(Image.open(final)).convert("RGB")
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
                p["source_png"] = str(final)
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
