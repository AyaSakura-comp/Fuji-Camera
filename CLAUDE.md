# Fuji Camera

iPhone-oriented web camera + "picture pool" that auto-applies a **底片風 (analog film)** look to
every photo via the local **FLUX.2** deployment. Shoot on the phone → photos upload to this server →
a single worker processes them one-at-a-time through FLUX.2 RefControl → browse/compare/download in
an iPhone-Photos-style gallery.

Served over Tailscale HTTPS (iOS Safari requires HTTPS for camera access):
**https://aya.crayfish-monitor.ts.net/**

## Architecture

```
iPhone Safari (static SPA)
   │  POST /api/upload (JPEG)
   ▼
server.py  ── FastAPI on 127.0.0.1:8090 ─────────────┐
   │  photo saved to data/, status=pending           │ single worker thread,
   │  worker picks one at a time (serial queue)       │ SERIAL (one GPU job at a time)
   ▼                                                  │
run_film(orig)  →  create_image.py 底片風 subprocess  │  (cold: reloads model each call)
   │                                                  │
   ▼                                                  ▼
data/results/<id>.jpg (processed)          data/db.json (status: pending→processing→done/error)
```

Frontend (`static/index.html`, one file) is served by the same FastAPI app at `/`.
Tailscale `serve` proxies `:443 → 127.0.0.1:8090`.

## Files

- `server.py` — FastAPI app + picture pool + serial worker. Endpoints:
  - `POST /api/upload` — multipart JPEG; normalises orientation, makes a thumb, enqueues, returns `{id}`.
  - `GET /api/photos` — list (newest first) + status counts.
  - `GET /api/file/{id}/{thumb|full|orig}` — `full`/`thumb` return the **processed result if done, else the original**.
  - `DELETE /api/photo/{id}`.
  - Status persisted in `data/db.json`; on startup, unfinished (`pending`/`processing`) items are requeued.
- `static/index.html` — the whole SPA (camera view + gallery + fullscreen viewer). No build step, no deps.
- `film_daemon.py` — a warm/resident FLUX daemon. **Currently DISABLED** (see "Warm daemon" below).
- `data/` — runtime state, gitignore-worthy: `originals/`, `results/`, `thumbs/`, `db.json`.

## The film pipeline (must stay faithful to `/create-image`)

The worker shells out to the shared create-image skill — do **not** reimplement FLUX here:

```bash
FLUX2_BIG_WMMA_LINEAR=1 \
  ~/models-work/flux2/.venv-rocm72/bin/python \
  ~/.hermes/skills/create-image/scripts/create_image.py \
  "底片風" --refcontrol --steps 2 --image <orig>    # cwd: ~/models-work/flux2
```

- `底片風` is a film keyword → the script auto-runs the RefControl analog flow. The prompt the model
  actually sees is `refcontrol, analog, AnalogRedmAF, 底片風` (assembled in `create_image.py`, not here).
  LoRAs: `refcontrol_klein9b_depth` (structure) + `analog_redmond` (film).
- The worker parses `"final_path"` from the script's stdout JSON, then makes a viewer JPEG + thumb.
- **`--steps 2`** is a deliberate speed choice (see Perf).

## Perf reality (measured on this box, gfx1151, 2:3 portrait)

- NOT hires: generates at **832×1248**, then Lanczos-upscales to 1184×1776.
- Bottleneck is the **denoise**, not model load. Skill's own `timing.json`: `load_pipeline ~1s` (page
  cache), `generate ~172s` at 4 steps → **~106s at 2 steps**. Depth ~3s.
- The SKILL's "~90s" figure is optimistic/for landscape 3:2 (where the tuned WMMA kernel applies).

## Warm daemon (film_daemon.py) — DISABLED on purpose

`film_daemon.py` keeps FLUX 9B + both LoRAs + the depth model resident (HTTP on `:7863`,
`run_film` in `server.py` prefers it when up, else falls back to the subprocess). It faithfully
replicates the RefControl flow by importing `create_image`'s helpers.

**It was measured to save almost nothing** (~25s): the model load is already ~1s from page cache and
the denoise dominates. Keeping it resident holds ~33GB and forces `qwen-lcpp` off (OOM). So it is
`disable`d. Only re-enable if you have a reason to hold FLUX warm AND have freed the memory.

## Running / ops

Systemd **user** units (linger is on, so they survive reboot):

```bash
systemctl --user restart fuji-camera        # the server (always on)
systemctl --user status  fuji-camera
journalctl --user -u fuji-camera -f

# Tailscale HTTPS (persists across reboot; only needed once):
sudo tailscale serve --bg --https=443 http://127.0.0.1:8090
```

- `~/.config/systemd/user/fuji-camera.service` — the server (enabled).
- `~/.config/systemd/user/fuji-film-daemon.service` — the warm daemon (**disabled**).
- Static frontend edits are live on refresh (StaticFiles reads disk per request) — no restart needed.
  Restart only for `server.py` changes.

## Frontend notes (static/index.html)

- **Camera**: lens picker `.5×/1×/2×` classified from `enumerateDevices` labels (matches both Chinese
  `後置超廣角/望遠` and English `ultra/tele`; hides iOS virtual composite cams), zoom slider via track
  `zoom` capability, burst-hold shutter, background upload (non-blocking so you can shoot rapidly).
- **Gallery**: `#gallery` is a full-screen scroll container (`position:absolute; inset:0; overflow-y:auto`).
  3-col thumbnail grid, spinner overlay on processing/pending cells, polls `/api/photos` every 3s.
- **Viewer**: windowed filmstrip carousel — one `.vslide` per photo in `[vIndex-1 .. vIndex+1]`, each at
  `left=i*W`, track `translateX(-vIndex*W + dragX)`, so prev/next are preloaded and glued during the
  drag; velocity/distance snap. Pinch-zoom, double-tap, swipe-down-to-dismiss.
  - **Press-and-hold (320ms) = Lightroom before/after**: shows `/orig` while held, back to `/full` on
    release (orig preloaded per done slide).
  - **Download always fetches the processed result** (`/full?dl=<ts>`, `cache:'no-store'`, blob) — never
    the original, even while 看原圖 / holding.

## Gotchas (real bugs hit here)

- **CSS `calc()` needs spaces around `+`/`-`**: `calc(env(...)+18px)` is invalid and the whole value is
  discarded → e.g. the bottom action bar collapsed to the top. Always `calc(env(...) + 18px)`.
- **`/full` returns the ORIGINAL while a photo is still processing** (result doesn't exist yet). Anything
  that must be the processed image needs cache-busting, or it can serve a stale cached original.
- **iOS long-press "Save Image" callout** conflicts with hold-to-compare → disabled via
  `-webkit-touch-callout:none` + `img{pointer-events:none}` + a `contextmenu` blocker.
- **Don't reimplement FLUX** — always go through `create_image.py`. Do not enable the warm daemon
  without freeing memory (it OOMs alongside qwen-lcpp / ComfyUI).
