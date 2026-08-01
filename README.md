# Job Speed Tracker

Real-time vehicle detection, tracking, and speed estimation from public CCTV streams. It crawls a city's CCTV portal for live camera feeds, runs YOLO + BoT-SORT/ByteTrack detection and tracking, estimates vehicle speed, and serves a live dashboard with statistics, history, and per-camera controls.

## Requirements

- **Python** 3.11+ (developed against 3.13)
- **MySQL** 5.7+ / 8.x — optional. If it's not reachable at startup, the app automatically falls back to a local SQLite database so it still runs (see [Database](#database)).
- **FFmpeg** — bundled with `opencv-python`'s wheel on most platforms; no separate install needed.
- Internet access — the app crawls a live CCTV portal for camera streams and (if a listed model checkpoint isn't already present locally) may need to fetch it.
- A YOLO checkpoint in `weights/` — see [Model weights](#model-weights).

## Installation

1. **Get the code**

   ```bash
   cd job_speed_tracker
   ```

2. **Create a virtual environment** (recommended)

   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

   This installs Flask, OpenCV, Ultralytics/PyTorch, and the tracking/ReID stack. First install can take a while (PyTorch is large).

4. **Configure environment variables**

   ```bash
   # Windows
   copy .env.example .env
   # macOS/Linux
   cp .env.example .env
   ```

   Then edit `.env` — at minimum, review `DATABASE_URL` and `YOLO_WEIGHTS_PATH`. See [Configuration](#configuration) for what each variable does.

5. **Add a YOLO model checkpoint** — see [Model weights](#model-weights) below.

6. **Run it**

   ```bash
   python app.py
   ```

   Then open **http://127.0.0.1:8000** (or whatever `HOST`/`PORT` you set) in a browser.

That's it — camera discovery, database setup, and log directories are all handled automatically at startup.

## Model weights

Place a `.pt` YOLO checkpoint under `weights/`. The path in `YOLO_WEIGHTS_PATH` (`.env`) is tried first; if it's missing, the app automatically falls through this priority list (`config.py: Config.MODEL_PRIORITY`) until one loads:

```
weights/yolo26m.pt
weights/yolo11m.pt
weights/yolov8m.pt
weights/yolo26n.pt
weights/yolo11n.pt
weights/yolov8n.pt
```

If a filename matches a standard Ultralytics-hosted checkpoint (e.g. `yolov8n.pt`, `yolo11n.pt`), Ultralytics may download it automatically on first use (internet required). Custom/fine-tuned checkpoints (e.g. `yolo26m.pt`) must be placed in `weights/` yourself. If no checkpoint loads, the app still starts and serves raw video, but detection stays disabled until a valid weights file is available.

## Configuration

All settings the app reads from the environment live in `.env` (copy `.env.example` to start). Everything else is a plain constant in `config.py` — tunables like detection confidence/IoU/image size, tracker parameters, and buffer sizes are not environment variables by design, so the `.env` contract never has to grow; edit `config.py` directly if you need to change those.

| Variable | Purpose | Default |
|---|---|---|
| `HOST` | Bind address for the Flask server | `127.0.0.1` |
| `PORT` | Bind port | `8000` |
| `FLASK_DEBUG` | Flask debug mode (`1`/`0`) | `0` |
| `DATABASE_URL` | SQLAlchemy URI for detection storage | MySQL on localhost |
| `YOLO_WEIGHTS_PATH` | Path to the YOLO checkpoint to load | `weights/yolo26m.pt` |
| `CCTV_STREAM_URL` | CCTV portal to crawl for camera streams, and the fallback stream if crawling finds nothing | — |
| `PIXELS_PER_METER` | Perspective calibration for speed estimation | `8.0` |
| `SPEED_LIMIT` | km/h threshold above which a vehicle is flagged "Overspeed" | `60.0` |
| `MIN_SPEED` / `MAX_SPEED` | Sanity bounds for reported speeds (km/h) | `5.0` / `240.0` |
| `SNAPSHOT_DIR` | Where overspeed/manual snapshot JPEGs are saved | `static/snapshot` |
| `LOG_DIR` | Where rotating log files are written | `logs` |

`PIXELS_PER_METER` and `SPEED_LIMIT` are also adjustable live from the dashboard's settings panel without restarting the app.

## Database

On startup the app:

1. Creates the configured database if it doesn't already exist.
2. Creates/migrates the `detections` table (`db.create_all()` + a small in-app migration step for older installs).
3. If the configured server is unreachable, logs a warning and transparently falls back to a local SQLite file (`database/speed_tracker.db`) so the app still runs without a MySQL server.

No manual schema step is required. `database/schema.sql` is kept only as a reference for manual provisioning if you'd rather set the table up yourself ahead of time.

## Running

```bash
python app.py
```

- Dashboard: `http://<HOST>:<PORT>/`
- Detection history: `http://<HOST>:<PORT>/history`
- Health check: `http://<HOST>:<PORT>/healthz`

The dashboard lets you search/switch cameras, adjust confidence/IoU live, draw the region-of-interest and speed measurement lines, and watch live statistics (counts by vehicle type, average speed, overspeed events, FPS/latency).

There's also a standalone CLI, `speed_tracker.py`, for running detection against a single stream outside the web app — see `python speed_tracker.py --help`.

## Project layout

```
app.py                  Flask app factory / entry point
config.py               All tunables (env-backed + fixed constants)
db.py                   Database bootstrap, migration, SQLite fallback
models/                 SQLAlchemy models
routes/                 Flask blueprints (web pages + JSON API)
services/               Detection pipeline: capture, detector, tracker,
                         speed estimator, preprocessing, vehicle filtering
trackers/                BoT-SORT/ByteTrack tuning profiles for this deployment
weights/                YOLO checkpoints (not all committed — see above)
static/ , templates/    Dashboard frontend
database/               Reference SQL schema + DB helper utilities
```

## Troubleshooting

- **No cameras showing up** — check `CCTV_STREAM_URL` is reachable and that outbound internet access isn't blocked; camera discovery happens in a background thread at startup, so check the logs (`logs/app.log`) for crawl errors.
- **"No YOLO weights found"** — put a valid `.pt` file in `weights/` matching `YOLO_WEIGHTS_PATH` or one of the fallback names above.
- **MySQL connection errors at startup** — expected if no server is running; the app falls back to SQLite automatically. To use MySQL, make sure the server is up and `DATABASE_URL` credentials are correct.
- **High CPU / slow inference** — this is a CPU-bound YOLO pipeline by default; see the tuning notes and rationale directly in `config.py` (detection confidence/IoU/image size) and `trackers/*.yaml` (tracker thresholds) before changing them.
# Job_tracker
