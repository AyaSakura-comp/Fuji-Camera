# Fuji Camera

iPhone-oriented web camera + "picture pool" that auto-applies a **底片風 (analog film)** look to
every photo via the local **FLUX.2** deployment. Shoot on the phone → photos upload to this server →
a single worker processes them one-at-a-time through FLUX.2 RefControl → browse/compare/download in
an iPhone-Photos-style gallery.

Served over Tailscale HTTPS (iOS Safari requires HTTPS for camera access):
**https://aya.crayfish-monitor.ts.net/**

## Architecture

The web/pool server is **decoupled from the GPU** so it can run in a container / elsewhere while
generation stays on the host. They talk over HTTP with **image bytes** (no shared filesystem).

```
iPhone Safari (static SPA)
   │  POST /api/upload (JPEG)
   ▼
server.py ── FastAPI :8090 (web + pool + serial queue) ── NO GPU, NO torch
   │  worker: run_film(orig_bytes)
   │      POST bytes ─────────────►  gen_service.py  (HOST, :7863)  ── the GPU lives here
   │      ◄──────── processed bytes     └─ shells out to create_image.py 底片風 (FLUX.2), serial
   ▼
data/  (originals/ results/ thumbs/ db.json ; status pending→processing→done/error)
```

- `run_film(orig_bytes) -> bytes`: **prefers the host gen service** at `FUJI_GEN_URL`
  (default `http://127.0.0.1:7863`; in a container set `http://host.docker.internal:7863`).
  If the gen service is unreachable it **falls back to a local `create_image.py` subprocess**, so
  running `server.py` directly on the host still works standalone (no gen service needed).
- Frontend (`static/index.html`, one file) is served by the same FastAPI app at `/`.
- **Two ways to deploy** (both keep the GPU on the host):
  1. **Host-only** (current live): `fuji-camera.service` (server) + `fuji-gen.service` (gen) + host
     `tailscale serve` → `https://aya.<tailnet>.ts.net/`.
  2. **Docker** (`docker-compose.yml`): app container + Tailscale sidecar (new node `fuji-camera` →
     `https://fuji-camera.<tailnet>.ts.net/`), talking to the host `fuji-gen.service`. See below.

## Files

- `gen_service.py` — **host-side** GPU generation service (stdlib http, system python; it only shells
  out to the FLUX venv). `POST /film` (raw image bytes) → processed PNG bytes; `GET /health`. Serial
  (one GPU job at a time). Runs as `fuji-gen.service` on `:7863`.
- `Dockerfile` / `docker-compose.yml` / `.dockerignore` / `ts-serve.json` / `.env.example` — the
  containerised deploy (app + Tailscale sidecar). The image has **no torch/ROCm** — it's tiny.
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

## Deploy from scratch

**Prerequisites**
- The **create-image FLUX.2 skill** must already be deployed on the box — this app shells out to it and
  does not bundle any model. It expects:
  - script: `~/.hermes/skills/create-image/scripts/create_image.py`
  - FLUX venv (torch+diffusers, ROCm): `~/models-work/flux2/.venv-rocm72`
  - LoRAs present under `~/models-work/flux2/loras/` (`refcontrol_klein9b_depth`, `analog_redmond`).
  Sanity check: `~/models-work/flux2/.venv-rocm72/bin/python ~/.hermes/skills/create-image/scripts/create_image.py "底片風" --refcontrol --steps 2 --image some.jpg`
- **System python3** (the server itself is light) with: `fastapi uvicorn pillow python-multipart`
  (`pip3 install --user fastapi uvicorn pillow python-multipart` if missing).
- **Tailscale** up and logged in (for HTTPS; iOS camera needs it).

**Steps**
```bash
# 1. clone
git clone git@github.com:AyaSakura-comp/Fuji-Camera.git ~/src/fuji_camera

# 2. smoke-test the server (Ctrl-C after it says "Application startup complete")
cd ~/src/fuji_camera && python3 -m uvicorn server:app --host 127.0.0.1 --port 8090

# 3. install as a systemd --user service (survives reboot)
loginctl enable-linger "$USER"                       # once, so user services run without a login
mkdir -p ~/.config/systemd/user
cat > ~/.config/systemd/user/fuji-camera.service <<'UNIT'
[Unit]
Description=Fuji Camera picture-pool server
After=network-online.target

[Service]
Type=simple
WorkingDirectory=%h/src/fuji_camera
ExecStart=/usr/bin/python3 -m uvicorn server:app --host 127.0.0.1 --port 8090
Restart=on-failure
RestartSec=3

[Install]
WantedBy=default.target
UNIT
systemctl --user daemon-reload
systemctl --user enable --now fuji-camera.service

# 4. expose over Tailscale HTTPS (persists across reboots; run once)
sudo tailscale serve --bg --https=443 http://127.0.0.1:8090

# 5. open the printed https URL (e.g. https://<host>.<tailnet>.ts.net/) on the iPhone
```

`data/` (originals/results/thumbs/db.json) is created on first run. The warm daemon
(`fuji-film-daemon.service`) is optional and off by default — see "Warm daemon".

## Deploy with Docker (app in a container, GPU on the host)

The container runs only the web/pool server + a Tailscale sidecar; it reaches the host's
`gen_service.py` over HTTP for the actual FLUX work. Because the handoff is **bytes over HTTP**
(no shared filesystem), the container is portable — deploy it anywhere that can reach a gen endpoint.

```bash
# 1. HOST: run the GPU generation service (needs the create-image FLUX.2 skill + venv; see above)
systemctl --user enable --now fuji-gen.service      # gen_service.py on :7863
curl -s localhost:7863/health                        # {"ready": true}

# 2. Tailscale auth key for the sidecar
cp .env.example .env && $EDITOR .env                 # TS_AUTHKEY=tskey-auth-...

# 3. bring up the container + sidecar
docker compose up -d --build
```

- The sidecar registers a **new Tailscale node `fuji-camera`** → the app is served at
  `https://fuji-camera.<your-tailnet>.ts.net/` (a fresh subdomain, distinct from the host's `aya.*`).
- To make it **public on the internet**, set `AllowFunnel` to `true` in `ts-serve.json` (and enable
  Funnel in the tailnet ACL). Otherwise it's tailnet-only. **When public, set a passcode** (below).
- **Passcode gate + per-passcode galleries**: set `FUJI_PASSCODES="8345,2233,..."` (in `.env`;
  legacy `FUJI_PASSCODE` still merged). Every request needs a valid signed cookie; a login page
  collects the passcode and sets a 30-day cookie (icons/manifest stay open for the PWA). **Each
  passcode is its own gallery** — uploads are tagged with the caller's group (`_gid(passcode)`);
  `/api/photos` and `/api/file` + delete are scoped to that group (cross-group access → 404). BUT
  the `counts` in `/api/photos` are **global** across all groups (shared GPU queue), so the
  "processing N" pill/badge reflects everyone. Legacy photos (no group) are backfilled to
  `DEFAULT_GID` (= first passcode's group) on startup. Cookies are signed with a key derived from the
  actual passcodes (secret — not in the repo), so groups can't be forged. `POST /logout` clears the
  cookie (there's a 登出 tab). Unset `FUJI_PASSCODES` = no gate (fine for tailnet-only).
- The container reaches the host via `host.docker.internal` (`extra_hosts: host-gateway`);
  `FUJI_GEN_URL=http://host.docker.internal:7863`. `gen_service.py` binds `0.0.0.0` for this.
- `./data` is bind-mounted so the picture pool persists across container restarts.
- **Don't run the host `fuji-camera.service` and the container at the same time** — they'd both drive
  the single-GPU `fuji-gen.service` and compete. Pick one front-end.

## Running / ops

**LIVE deployment = Docker** (`https://fuji-camera.crayfish-monitor.ts.net/`). The old host-only path
(`fuji-camera.service` on `:8090` + `aya.*` serve) is **stopped/disabled** — don't run both (shared
`./data` + single GPU). Restart everything with the skill: `bash ~/.hermes/skills/restart-fuji-camera/scripts/restart.sh`.

Systemd **user** units (linger on, survive reboot):

```bash
systemctl --user restart fuji-gen.service            # HOST GPU gen service (:7863)
systemctl --user restart fuji-camera-docker.service  # docker compose up -d (app + tailscale)
#   or directly:  cd ~/src/fuji_camera && docker compose up -d
docker compose logs --tail=40                         # app/tailscale logs
journalctl --user -u fuji-gen -n 40 --no-pager        # gen service logs
```

- `~/.config/systemd/user/fuji-gen.service` — host GPU generation service (**enabled**).
- `~/.config/systemd/user/fuji-camera-docker.service` — wraps `docker compose up/down` (**enabled**).
- `~/.config/systemd/user/fuji-camera.service` — old host-only server (**disabled**, superseded).
- `~/.config/systemd/user/fuji-film-daemon.service` — the warm daemon (**disabled**, see below).
- Restart order matters: **app before the ts sidecar** (shares its netns); the skill/script handles it.
- Static frontend edits (`static/index.html`) are live on refresh — no restart. For `server.py` changes
  rebuild the image: `docker compose up -d --build`. For `gen_service.py`: `systemctl --user restart fuji-gen`.

## Frontend notes (static/index.html)

- **Camera**: lens picker `.5×/1×/2×` classified from `enumerateDevices` labels (matches both Chinese
  `後置超廣角/望遠` and English `ultra/tele`; hides iOS virtual composite cams), zoom slider via track
  `zoom` capability + **two-finger pinch** on the preview (shared `applyZoom()`), burst-hold shutter,
  background upload (non-blocking). Camera stream **stops on the gallery tab / reopens on return**.
  - **Capture follows orientation**: grabs the visible `object-fit:cover` region, so portrait phone →
    portrait photo, landscape → landscape (also WYSIWYG with the preview). Front camera mirrored.
  - **Landscape layout** (`@media (orientation:landscape)`): tab nav → vertical bar far-right, shutter
    cluster just left of it, zoom → vertical slider on the left edge, lens picker bottom-centre.
- **Gallery**: `#gallery` is a full-screen scroll container (`position:absolute; inset:0; overflow-y:auto`).
  3-col thumbnail grid, spinner overlay on processing/pending cells, polls `/api/photos` every 3s. A
  **+ button** uploads any device photo(s) into the pool (`/api/upload`, processed the same way).
- **Viewer**: windowed filmstrip carousel — one `.vslide` per photo in `[vIndex-1 .. vIndex+1]`, each at
  `left=i*W`, track `translateX(-vIndex*W + dragX)`, so prev/next are preloaded and glued during the
  drag; velocity/distance snap. Pinch-zoom, double-tap, swipe-down-to-dismiss.
  - **Press-and-hold (320ms) = Lightroom before/after**: shows `/orig` while held, back to `/full` on
    release (orig preloaded per done slide).
  - **Download/save = what you're viewing**: the original when 看原圖 is active (`/orig`, file
    `fuji_<id>_orig.jpg`), else the processed result (`/full`). Cache-busted (`?dl=<ts>`,
    `cache:'no-store'`, blob). On iPhone uses the Web Share API (`navigator.share`) → Save to Photos.

## Gotchas (real bugs hit here)

- **CSS `calc()` needs spaces around `+`/`-`**: `calc(env(...)+18px)` is invalid and the whole value is
  discarded → e.g. the bottom action bar collapsed to the top. Always `calc(env(...) + 18px)`.
- **`/full` returns the ORIGINAL while a photo is still processing** (result doesn't exist yet). Anything
  that must be the processed image needs cache-busting, or it can serve a stale cached original.
- **iOS long-press "Save Image" callout** conflicts with hold-to-compare → disabled via
  `-webkit-touch-callout:none` + `img{pointer-events:none}` + a `contextmenu` blocker.
- **Don't reimplement FLUX** — always go through `create_image.py`. Do not enable the warm daemon
  without freeing memory (it OOMs alongside qwen-lcpp / ComfyUI).
