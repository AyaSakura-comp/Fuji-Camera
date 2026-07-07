#!/usr/bin/env python3
"""
Warm resident FLUX.2 film daemon for the Fuji Camera app.

Keeps the FLUX.2 9B-KV pipeline + RefControl(depth) LoRA + Analog-Redmond LoRA +
Depth-Anything-V2 depth model resident in GPU memory, so each 底片風 request skips
the ~200s cold model load and only pays the ~30-90s inference.

Faithfulness: this reuses create_image.py's own helper functions and reproduces the
exact RefControl+film path from its main() (same prompt, same LoRAs/weights, same
depth model, same sizes, same steps/guidance) so results match `/create-image 底片風`
on a photo. Only the model stays warm between requests.

Runs under the FLUX venv: ~/models-work/flux2/.venv-rocm72
HTTP: POST /film  {"image": "/abs/path.jpg"}  ->  {"final_path": "..."}
      GET  /health -> {"ready": bool}
"""
import os
import sys
import json
import time
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# --- env MUST be set before importing create_image / loading the pipe ---
os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")  # REQUIRED (dual-ref attn / OOM)
os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
os.environ.setdefault("FLUX2_BIG_WMMA_LINEAR", "1")                     # tuned wmma kernel
os.environ.setdefault("MIOPEN_FIND_MODE", "FAST")
os.environ.setdefault("MIOPEN_USER_DB_PATH", str(Path.home() / ".cache/miopen-flux2"))
# NOTE: do NOT set HF_HUB_OFFLINE — it makes diffusers refuse to load a LoRA by
# file path ("must specify a weight_name"). create_image.py relies on cache hits
# without offline mode; everything is already cached so no network is hit.

SCRIPTS = Path.home() / ".hermes/skills/create-image/scripts"
sys.path.insert(0, str(SCRIPTS))

import torch                                    # noqa: E402
from PIL import Image, ImageOps                 # noqa: E402
import create_image as ci                       # noqa: E402  (applies contiguous-SDPA + wmma patches)

PORT = 7863
OUT_DIR = Path("/home/chihmin/models-work/flux2/output/create-image")
OUT_DIR.mkdir(parents=True, exist_ok=True)

REFCONTROL_LORA = "/home/chihmin/models-work/flux2/loras/refcontrol_klein9b_depth.safetensors"
ANALOG_LORA     = "/home/chihmin/models-work/flux2/loras/analog_redmond_fluxklein9b.safetensors"

# defaults mirrored from create_image.py argparse for the 底片風 --refcontrol path
STEPS           = 4      # argparse default
GUIDANCE        = 1.0    # argparse default
REFCONTROL_SCALE = 1.0   # argparse default
ANALOG_SCALE    = 0.55   # create_image adjusts lora_scale 0.8 -> 0.55 on refcontrol path
PROMPT          = "refcontrol, analog, AnalogRedmAF, analog, AnalogRedmAF, F1.2 shallow depth of field, 35mm analog film photo, soft contrast, fine film grain, subtle halation, cinematic bokeh"

_pipe = None
_depth = None
_ready = False
_gpu_lock = threading.Lock()   # serialize inference (single GPU)


def log(*a):
    print("[film-daemon]", *a, flush=True)


def load_models():
    global _pipe, _depth, _ready
    t0 = time.perf_counter()
    log("loading FLUX.2 9B-KV pipeline...")
    pipe = ci.Flux2KleinPipeline.from_pretrained(ci.MODEL_ID_9B_KV, torch_dtype=torch.bfloat16)
    pipe = pipe.to("cuda")

    log("loading LoRAs (refcontrol + analog)...")
    pipe.load_lora_weights(REFCONTROL_LORA, adapter_name="refcontrol")
    pipe.load_lora_weights(ANALOG_LORA, adapter_name="analog")
    pipe.set_adapters(["refcontrol", "analog"], adapter_weights=[REFCONTROL_SCALE, ANALOG_SCALE])

    log("loading Depth-Anything-V2-Large...")
    from transformers import pipeline as tpipe
    depth = tpipe("depth-estimation",
                  model="depth-anything/Depth-Anything-V2-Large-hf", device="cuda")

    globals()["_pipe"] = pipe
    globals()["_depth"] = depth
    globals()["_ready"] = True
    log(f"ready in {time.perf_counter()-t0:.0f}s")


def run_film(image_path: str) -> dict:
    if not _ready:
        raise RuntimeError("models not loaded yet")
    with _gpu_lock:
        t_all = time.perf_counter()
        # size buckets from the photo's orientation (same as create_image main())
        aspect = ci.resolve_flux_aspect_ratio(None, image_path)
        gen_w, gen_h, final_w, final_h, aspect_ratio = ci.resolve_flux_sizes(
            "9b-kv", aspect, native=False)

        src = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
        t = time.perf_counter()
        depth_img = _depth(src)["depth"]
        t_depth = time.perf_counter() - t

        ref_cond = ci._cover_resize(src, gen_w, gen_h)
        depth_cond = ci._cover_resize(depth_img, gen_w, gen_h)

        seed = int(time.time() * 1000) % (2**31)
        gen = torch.Generator(device="cuda").manual_seed(seed)

        t = time.perf_counter()
        with torch.inference_mode():
            image = _pipe(
                prompt=PROMPT, width=gen_w, height=gen_h,
                num_inference_steps=STEPS, guidance_scale=GUIDANCE,
                generator=gen, image=[ref_cond, depth_cond],
            ).images[0]
        t_gen = time.perf_counter() - t

        final = image.resize((final_w, final_h), Image.Resampling.LANCZOS)
        ts = time.strftime("%Y%m%d_%H%M%S")
        final_path = OUT_DIR / f"flux2_refcontrol-film-9b-warm_{ts}_{final_w}x{final_h}.png"
        final.save(final_path)

        return {
            "final_path": str(final_path.resolve()),
            "served_by": "fuji-film-daemon",
            "prompt": PROMPT,
            "seed": seed,
            "timings": {"depth": round(t_depth, 1), "generate": round(t_gen, 1),
                        "total": round(time.perf_counter() - t_all, 1)},
        }


class Handler(BaseHTTPRequestHandler):
    def _json(self, code, obj):
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            return self._json(200, {"ready": _ready})
        self._json(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/film":
            return self._json(404, {"error": "not found"})
        try:
            n = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(n) or b"{}")
            image = payload.get("image")
            if not image or not Path(image).exists():
                return self._json(400, {"error": "image path missing/not found"})
            result = run_film(image)
            self._json(200, result)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self._json(500, {"error": str(e)})

    def log_message(self, *a):
        pass


def main():
    # load models in the background so /health answers immediately
    threading.Thread(target=load_models, daemon=True).start()
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    log(f"listening on http://127.0.0.1:{PORT} (loading models...)")
    srv.serve_forever()


if __name__ == "__main__":
    main()
