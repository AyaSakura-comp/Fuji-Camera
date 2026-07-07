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

import hashlib
import hmac

from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Request
from fastapi.responses import (FileResponse, JSONResponse, Response,
                               HTMLResponse, RedirectResponse)
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

# Passcode gate (for public/Funnel exposure). One or more passcodes; EACH passcode
# is its own gallery (photos are scoped to the passcode used to log in). Set via
# FUJI_PASSCODES="a,b,c" (or the legacy single FUJI_PASSCODE). Unset = no gate.
# The GPU queue is shared, so the "processing" counts are GLOBAL across groups.
_pc_raw = (os.environ.get("FUJI_PASSCODES", "") + "," + os.environ.get("FUJI_PASSCODE", ""))
PASSCODES   = [p.strip() for p in _pc_raw.split(",") if p.strip()]
GATE_ON     = bool(PASSCODES)
COOKIE_NAME = "fuji_auth"
# open paths that never require auth (login POST + PWA icons/manifest)
_OPEN_PATHS = {"/auth", "/logout", "/manifest.webmanifest", "/favicon-32.png",
               "/apple-touch-icon.png", "/icon-192.png", "/icon-512.png"}

# group id per passcode; cookies are signed with a key derived from the actual
# passcodes (secret — not in the public repo) so groups can't be forged.
_SIGN_KEY   = hashlib.sha256(("fuji-sign:" + ",".join(sorted(PASSCODES))).encode()).digest()
def _gid(passcode: str) -> str:
    return hashlib.sha256(("fuji-grp:" + passcode).encode()).hexdigest()[:12]
GROUPS      = {_gid(p): p for p in PASSCODES}
DEFAULT_GID = _gid(PASSCODES[0]) if PASSCODES else ""   # legacy photos (no group) belong here

def _make_cookie(passcode: str) -> str:
    g = _gid(passcode)
    sig = hmac.new(_SIGN_KEY, g.encode(), hashlib.sha256).hexdigest()[:16]
    return g + "." + sig

def _cookie_group(cookie: str):
    if not cookie or "." not in cookie:
        return None
    g, sig = cookie.rsplit(".", 1)
    good = hmac.new(_SIGN_KEY, g.encode(), hashlib.sha256).hexdigest()[:16]
    return g if (g in GROUPS and hmac.compare_digest(sig, good)) else None

_LOGIN_HTML = """<!doctype html><html lang="zh-Hant"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>Fuji 相機</title><style>
*{box-sizing:border-box}html,body{height:100%;margin:0;background:#000;color:#fff;
font-family:-apple-system,BlinkMacSystemFont,"Helvetica Neue",sans-serif}
.wrap{position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;
justify-content:center;gap:18px;padding:24px;text-align:center}
h1{font-size:22px;font-weight:600;margin:0}p{color:#8a8a8e;font-size:14px;margin:0}
form{display:flex;flex-direction:column;gap:12px;width:100%;max-width:280px;margin-top:6px}
input{font-size:20px;text-align:center;letter-spacing:6px;padding:14px;border-radius:14px;
border:none;background:#1c1c1e;color:#fff;outline:none}
button{padding:14px;border:none;border-radius:24px;background:#fff;color:#000;
font-size:17px;font-weight:600;cursor:pointer}
.err{color:#ff453a;font-size:13px;min-height:16px}</style></head>
<body><div class="wrap"><div style="font-size:44px">📷</div>
<h1>Fuji 相機</h1><p>請輸入密碼</p>
<form method="post" action="/auth" autocomplete="off">
<input name="passcode" type="password" inputmode="numeric" autofocus placeholder="••••">
<div class="err">%ERR%</div>
<button type="submit">進入</button></form></div></body></html>"""

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

_cancelled = set()   # pids deleted while still queued / processing (guarded by _db_lock)

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
        if not p or p["status"] == "done" or pid in _cancelled:  # deleted while queued -> skip
            _cancelled.discard(pid)
            return
        p["status"] = "processing"
        p["started"] = time.time()
        save_db(db)

    err = None
    try:
        orig_bytes = (ORIG / p["orig"]).read_bytes()
        result_bytes = run_film(orig_bytes)

        # deleted while it was processing? discard the result, write nothing.
        with _db_lock:
            if find(load_db(), pid) is None or pid in _cancelled:
                _cancelled.discard(pid)
                return

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

@app.middleware("http")
async def _passcode_gate(request: Request, call_next):
    if not GATE_ON or request.url.path in _OPEN_PATHS:
        request.state.group = DEFAULT_GID
        return await call_next(request)
    gid = _cookie_group(request.cookies.get(COOKIE_NAME, ""))
    if gid:
        request.state.group = gid
        return await call_next(request)
    if request.url.path.startswith("/api"):
        return JSONResponse({"error": "auth required"}, status_code=401)
    return HTMLResponse(_LOGIN_HTML.replace("%ERR%", ""), status_code=401)

@app.post("/auth")
async def auth(passcode: str = Form("")):
    if any(hmac.compare_digest(passcode, p) for p in PASSCODES):
        resp = RedirectResponse("/", status_code=303)
        resp.set_cookie(COOKIE_NAME, _make_cookie(passcode), max_age=30 * 24 * 3600,
                        httponly=True, samesite="lax", secure=True)
        return resp
    return HTMLResponse(_LOGIN_HTML.replace("%ERR%", "密碼錯誤"), status_code=401)

@app.post("/logout")
async def logout():
    resp = RedirectResponse("/", status_code=303)
    resp.delete_cookie(COOKIE_NAME)
    return resp

@app.on_event("startup")
def _startup():
    # requeue anything left pending / processing from a previous run;
    # backfill a group on legacy photos (permanently assign them to DEFAULT_GID)
    with _db_lock:
        db = load_db()
        requeue = []
        for p in db["photos"]:
            if "group" not in p:
                p["group"] = DEFAULT_GID
            if p["status"] in ("pending", "processing"):
                p["status"] = "pending"
                requeue.append(p["id"])
        save_db(db)
    for pid in requeue:
        enqueue(pid)
    t = threading.Thread(target=worker_loop, daemon=True)
    t.start()

@app.post("/api/upload")
async def upload(request: Request, file: UploadFile = File(...)):
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
        "group": getattr(request.state, "group", DEFAULT_GID),   # per-passcode gallery
    }
    with _db_lock:
        db = load_db()
        db["photos"].append(entry)
        save_db(db)
    enqueue(pid)
    return {"id": pid, "status": "pending"}

@app.get("/api/photos")
def list_photos(request: Request):
    grp = getattr(request.state, "group", DEFAULT_GID)
    with _db_lock:
        db = load_db()
        photos = list(db["photos"])
    photos.sort(key=lambda p: p.get("created", 0), reverse=True)
    # counts are GLOBAL (shared GPU queue) so "processing N" reflects everyone;
    # the photo list is scoped to the caller's passcode group (their own gallery).
    counts = {"pending": 0, "processing": 0, "done": 0, "error": 0}
    out = []
    for p in photos:
        counts[p["status"]] = counts.get(p["status"], 0) + 1
        if p.get("group", DEFAULT_GID) != grp:
            continue
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

def _find_owned(pid: str, request: Request):
    """find the photo, but only if it belongs to the caller's passcode group."""
    grp = getattr(request.state, "group", DEFAULT_GID)
    with _db_lock:
        p = find(load_db(), pid)
    if not p or p.get("group", DEFAULT_GID) != grp:
        raise HTTPException(404, "no such photo")
    return p

@app.get("/api/file/{pid}/thumb")
def get_thumb(pid: str, request: Request):
    p = _find_owned(pid, request)
    if p["status"] == "done" and p.get("result_thumb"):
        return _file_or_404(THUMB / p["result_thumb"])
    return _file_or_404(THUMB / p.get("orig_thumb", f"{pid}.jpg"))

@app.get("/api/file/{pid}/full")
def get_full(pid: str, request: Request):
    p = _find_owned(pid, request)
    if p["status"] == "done" and p.get("result"):
        return _file_or_404(RESULT / p["result"])
    return _file_or_404(ORIG / p.get("orig", f"{pid}.jpg"))

@app.get("/api/file/{pid}/orig")
def get_orig(pid: str, request: Request):
    p = _find_owned(pid, request)
    return _file_or_404(ORIG / p.get("orig", f"{pid}.jpg"))

@app.delete("/api/photo/{pid}")
def delete_photo(pid: str, request: Request):
    grp = getattr(request.state, "group", DEFAULT_GID)
    with _db_lock:
        db = load_db()
        p = find(db, pid)
        if not p or p.get("group", DEFAULT_GID) != grp:
            raise HTTPException(404, "no such photo")
        db["photos"] = [x for x in db["photos"] if x["id"] != pid]
        save_db(db)
        # if it's still queued/processing, mark it so the worker skips it and
        # discards any in-flight result instead of resurrecting the files
        if p.get("status") in ("pending", "processing"):
            _cancelled.add(pid)
    for f in (ORIG / p.get("orig", ""), THUMB / p.get("orig_thumb", ""),
              RESULT / p.get("result", ""), THUMB / p.get("result_thumb", "")):
        try:
            if f.name and f.exists():
                f.unlink()
        except Exception:
            pass
    return {"ok": True}

# serve the app HTML with no-cache so frontend edits show up immediately
# (single-file app: fresh HTML = fresh inline CSS/JS)
@app.get("/")
def index():
    return FileResponse(str(BASE / "static" / "index.html"), media_type="text/html",
                        headers={"Cache-Control": "no-cache, must-revalidate"})

# static frontend (icons/manifest/etc.) — mounted last so /api/* and / win
app.mount("/", StaticFiles(directory=str(BASE / "static"), html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8090)
