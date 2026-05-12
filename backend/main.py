"""
AIGym Web Backend
=================
Architecture: Browser → JPEG bytes → WebSocket → YOLO → JSON keypoints → Browser

Why this is fast (< 100 ms):
  1. Video stays LOCAL: no annotated frame round-trip, only ~1-2 KB JSON returned.
  2. Ping-pong protocol: next frame is only sent AFTER response is received,
     so the GPU queue never backs up (eliminates MJPEG's compounding delay).
  3. WebSocket overhead is ~0 ms vs HTTP's per-request handshake.
  4. YOLO runs in a thread-pool executor → async loop never blocks.
  5. Frontend draws its OWN webcam stream; backend just supplies skeleton data.
"""

import asyncio
import concurrent.futures
import json
import logging
import os
import time

import cv2
import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO

from session import PoseSession

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH = os.getenv("MODEL_PATH", "yolo11s-pose.pt")
YOLO_CONF  = float(os.getenv("YOLO_CONF", "0.50"))
YOLO_IOU   = float(os.getenv("YOLO_IOU", "0.70"))
NUM_THREADS = int(os.getenv("NUM_THREADS", "4"))

# Incoming frame dimensions (must match frontend SEND_W × SEND_H)
SEND_W = int(os.getenv("SEND_W", "640"))
SEND_H = int(os.getenv("SEND_H", "480"))

# ── Torch CPU optimisation ───────────────────────────────────────────────────
torch.set_num_threads(NUM_THREADS)
log.info(f"Torch threads: {NUM_THREADS}")

app   = FastAPI(title="AIGym Web Backend")
model: YOLO | None = None

# Dedicated single-thread executor keeps YOLO on one thread → avoids CUDA
# context-switching overhead when multiple clients connect simultaneously.
_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "https://*.onrender.com",   # Allow any Render frontend
        "https://*.vercel.app",      # Allow any Vercel frontend
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Startup ───────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup() -> None:
    global model

    log.info(f"Loading model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    # ── Device detection ──────────────────────────────────────────────────
    device = next(model.model.parameters()).device
    cuda_available = torch.cuda.is_available()

    if cuda_available:
        gpu_name = torch.cuda.get_device_name(0)
        log.info(f"✅ GPU detected: {gpu_name}")

        # Optional: enable FP16 for 2× speed on modern GPUs
        try:
            model.model.half()
            log.info("   FP16 (half precision) enabled")
        except Exception:
            log.info("   FP16 not supported, using FP32")

    else:
        log.info(f"ℹ️  Running on CPU (device: {device})")
        log.info("   For better performance, consider GPU or ONNX export")

    # ── Warm-up ───────────────────────────────────────────────────────────
    dummy = np.zeros((SEND_H, SEND_W, 3), dtype=np.uint8)
    kwargs = {"conf": YOLO_CONF, "iou": YOLO_IOU, "verbose": False}
    if cuda_available:
        kwargs["half"] = True

    model(dummy, **kwargs)
    log.info("Model warm-up complete — ready for connections.")


# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    cuda_available = torch.cuda.is_available()
    return {
        "status": "ok",
        "model": MODEL_PATH,
        "device": "cuda" if cuda_available else "cpu",
        "cuda_available": cuda_available,
    }


@app.get("/exercises")
async def exercises():
    return {"exercises": ["pushups", "squats", "lunges", "planks"]}


# ── WebSocket endpoint ────────────────────────────────────────────────────────
@app.websocket("/ws/{exercise}")
async def websocket_pose(websocket: WebSocket, exercise: str) -> None:
    await websocket.accept()
    log.info(f"[CONNECT]  exercise={exercise}  client={websocket.client}")

    try:
        session = PoseSession(exercise)
    except ValueError as e:
        await websocket.send_text(json.dumps({"error": str(e)}))
        await websocket.close()
        return

    loop    = asyncio.get_event_loop()
    frames  = 0
    t_start = time.perf_counter()

    try:
        while True:
            # ── Receive JPEG bytes from browser ──────────────────────────
            data: bytes = await websocket.receive_bytes()

            # ── Decode + infer in thread-pool (never blocks event loop) ──
            payload = await loop.run_in_executor(
                _executor, _process, data, session
            )

            # ── Send lightweight JSON response (~1-2 KB) ─────────────────
            await websocket.send_text(json.dumps(payload))

            frames += 1
            if frames % 100 == 0:
                elapsed = time.perf_counter() - t_start
                log.info(
                    f"  [{exercise}] {frames} frames  "
                    f"avg {elapsed/frames*1000:.1f} ms/frame  "
                    f"({frames/elapsed:.1f} fps)"
                )

    except WebSocketDisconnect:
        log.info(f"[DISCONNECT]  exercise={exercise}  frames={frames}")
    except Exception as exc:
        log.error(f"[ERROR]  {exc}", exc_info=True)


# ── Frame processing (runs in thread-pool) ───────────────────────────────────
def _process(data: bytes, session: "PoseSession") -> dict:
    """Decode JPEG → YOLO → assessment → JSON-safe dict."""
    t0 = time.perf_counter()

    # Decode JPEG
    arr   = np.frombuffer(data, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        return {"error": "bad_frame"}

    h, w = frame.shape[:2]
    t_decode = (time.perf_counter() - t0) * 1000

    # YOLO inference
    t1      = time.perf_counter()
    cuda_available = torch.cuda.is_available()
    kwargs = {"conf": YOLO_CONF, "iou": YOLO_IOU, "verbose": False}
    if cuda_available:
        kwargs["half"] = True

    results = model(frame, **kwargs)
    t_yolo = (time.perf_counter() - t1) * 1000

    # Extract + buffer keypoints
    result = results[0] if results else None
    raw_pts, raw_conf = session.extract_kp(result)

    if raw_pts is not None:
        pts, conf_arr = session.kp_buf.update(raw_pts, raw_conf)
    else:
        pts, conf_arr = None, None

    # Exercise checker + EvidenceGate + rep/hold counting
    t2 = time.perf_counter()
    assessment, state_info = session.process(pts, conf_arr)
    t_logic = (time.perf_counter() - t2) * 1000

    # Serialise keypoints [[x, y, conf], ...]
    kp_list: list[list[float]] = []
    if pts is not None:
        for i in range(len(pts)):
            c = float(conf_arr[i]) if conf_arr is not None else 1.0
            kp_list.append([round(float(pts[i][0]), 1),
                            round(float(pts[i][1]), 1),
                            round(c, 3)])

    return {
        "keypoints":   kp_list,
        "frame_size":  [w, h],
        "assessment":  assessment,
        "state":       state_info,
        "timing_ms": {
            "decode": round(t_decode, 1),
            "yolo":   round(t_yolo,   1),
            "logic":  round(t_logic,  1),
        },
    }