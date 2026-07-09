#!/usr/bin/env python3
"""
Fuji Camera — HOST-side generation service (GPU stays here).

The web app (server.py, possibly in a Docker container elsewhere) POSTs original
image BYTES here; this runs the local FLUX.2 底片風 pipeline on the host GPU and
returns the processed image BYTES. No shared filesystem needed → the app is
portable and can be deployed anywhere as long as it can reach this endpoint.

Runs under SYSTEM python3 (only stdlib) — it just shells out to the FLUX venv's
create_image.py, so it needs no torch itself.

HTTP:
  GET  /health           -> {"ready": true}
  POST /film  (raw image bytes in body)  -> processed PNG bytes (image/png)

Env:
  FUJI_GEN_PORT  (default 7863)
  FUJI_GEN_STEPS (default 2)
"""
import json
import os
import re
import subprocess
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HOME       = Path.home()
CREATE_IMG = HOME / ".hermes/skills/create-image/scripts/create_image.py"
FLUX_PY    = HOME / "models-work/flux2/.venv-rocm72/bin/python"
FLUX_CWD   = HOME / "models-work/flux2"

PORT   = int(os.environ.get("FUJI_GEN_PORT", "7863"))
STEPS  = os.environ.get("FUJI_GEN_STEPS", "2")
PROMPT = os.environ.get(
    "FUJI_GEN_PROMPT",
    "analog, AnalogRedmAF, F1.2 shallow depth of field, 35mm analog film photo, "
    "soft contrast, fine film grain, subtle halation, cinematic bokeh")

_gpu_lock = threading.Lock()   # one FLUX job at a time (single GPU)
_AP_RE = re.compile(r"^\d+(\.\d+)?$")   # aperture must be a plain number


def log(*a):
    print("[gen-service]", *a, flush=True)


def prompt_for(aperture: str) -> str:
    """swap the F-number in the base prompt for the requested aperture (F2.8 etc.)."""
    if not aperture or not _AP_RE.match(aperture):
        return PROMPT
    return re.sub(r"F[0-9.]+", "F" + aperture, PROMPT, count=1)


def run_film_bytes(orig: bytes, aperture: str = "1.2") -> bytes:
    """original image bytes -> processed film PNG bytes (blocks; serialized)."""
    with _gpu_lock:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tf:
            tf.write(orig)
            tmp = tf.name
        try:
            env = dict(os.environ)
            env["FLUX2_BIG_WMMA_LINEAR"] = "1"
            cmd = [str(FLUX_PY), str(CREATE_IMG), prompt_for(aperture), "--refcontrol",
                   "--steps", STEPS, "--image", tmp]
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
            try:
                os.unlink(tmp)
            except OSError:
                pass


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json"):
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode()
            ctype = "application/json"
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            return self._send(200, {"ready": True})
        self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/film":
            return self._send(404, {"error": "not found"})
        try:
            n = int(self.headers.get("Content-Length", 0))
            if n <= 0:
                return self._send(400, {"error": "empty body"})
            data = self.rfile.read(n)
            aperture = self.headers.get("X-Aperture", "1.2")
            out = run_film_bytes(data, aperture)
            self._send(200, out, ctype="image/png")
        except Exception as e:
            import traceback
            traceback.print_exc()
            self._send(500, {"error": str(e)})

    def log_message(self, *a):
        pass


def main():
    srv = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    log(f"listening on 0.0.0.0:{PORT} (steps={STEPS}); GPU via {FLUX_PY}")
    srv.serve_forever()


if __name__ == "__main__":
    main()
