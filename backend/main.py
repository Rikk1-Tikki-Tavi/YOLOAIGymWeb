import asyncio
import concurrent.futures
import json
import logging
import os
import time

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from session import PoseSession

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Set USE_YOLO=true to load YOLO for CLI / test scripts.
# The web path receives keypoints from the browser (TF.js MoveNet) and skips YOLO.
USE_YOLO    = os.getenv("USE_YOLO", "false").lower() == "true"
MODEL_PATH  = os.getenv("MODEL_PATH",  "yolo11s-pose.pt")
YOLO_CONF   = float(os.getenv("YOLO_CONF",  "0.50"))
YOLO_IOU    = float(os.getenv("YOLO_IOU",   "0.70"))
NUM_THREADS = int(os.getenv("NUM_THREADS",   "4"))
SEND_W      = int(os.getenv("SEND_W", "640"))
SEND_H      = int(os.getenv("SEND_H", "480"))

import torch
torch.set_num_threads(NUM_THREADS)
log.info(f"Torch threads: {NUM_THREADS}  |  USE_YOLO={USE_YOLO}")

_CUDA = torch.cuda.is_available()
model = None

app = FastAPI(title="AIGym Web Backend")

_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup() -> None:
    global model

    if USE_YOLO:
        from ultralytics import YOLO
        import cv2
        log.info(f"Loading YOLO model: {MODEL_PATH}")
        model = YOLO(MODEL_PATH)
        if _CUDA:
            log.info(f"GPU: {torch.cuda.get_device_name(0)}")
            try:
                model.model.half()
                log.info("FP16 enabled")
            except Exception:
                pass
        dummy = np.zeros((SEND_H, SEND_W, 3), dtype=np.uint8)
        model(dummy, conf=YOLO_CONF, iou=YOLO_IOU, verbose=False)
        log.info("YOLO warm-up complete.")
    else:
        log.info("YOLO disabled — web path uses browser-side MoveNet keypoints.")

    log.info("Backend ready for connections.")


@app.get("/health")
async def health():
    return {"status": "ok", "use_yolo": USE_YOLO, "cuda": _CUDA}


@app.get("/exercises")
async def exercises():
    return {"exercises": ["pushups", "squats", "lunges", "planks"]}


@app.websocket("/ws/{exercise}/{dominant_foot}")
async def websocket_pose(websocket: WebSocket, exercise: str, dominant_foot: str = "L") -> None:
    await websocket.accept()
    log.info(f"[CONNECT]  exercise={exercise}  dominant_foot={dominant_foot}  client={websocket.client}")

    try:
        session = PoseSession(exercise, dominant_foot=dominant_foot)
    except ValueError as e:
        await websocket.send_text(json.dumps({"error": str(e)}))
        await websocket.close()
        return

    loop   = asyncio.get_event_loop()
    frames = 0
    t_start = time.perf_counter()

    try:
        while True:
            # Receive JSON text: { keypoints: [[x,y,c]×17], frame_size: [w,h] }
            msg: str = await websocket.receive_text()

            payload = await loop.run_in_executor(
                _executor, _process_kp, msg, session
            )

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


def _process_kp(msg: str, session: "PoseSession") -> dict:
    """
    Web path: receive browser keypoints (from TF.js MoveNet), run form logic only.
    Eliminates YOLO inference — form checking takes ~5-20ms vs 100-300ms for YOLO on CPU.
    """
    t0 = time.perf_counter()

    body      = json.loads(msg)
    kp_raw    = body["keypoints"]       # [[x, y, conf] × 17]
    frame_w, frame_h = body["frame_size"]

    pts  = np.array([[k[0], k[1]] for k in kp_raw], dtype=np.float32)
    conf = np.array([k[2]         for k in kp_raw], dtype=np.float32)

    # Smooth keypoints across frames (same KeypointBuffer as before)
    pts, conf = session.kp_buf.update(pts, conf)

    t1 = time.perf_counter()
    assessment, state_info = session.process(pts, conf)
    t_logic = (time.perf_counter() - t1) * 1000

    kp_list = [
        [round(float(pts[i][0]), 1), round(float(pts[i][1]), 1), round(float(conf[i]), 3)]
        for i in range(len(pts))
    ]

    return {
        "keypoints":  kp_list,
        "frame_size": [frame_w, frame_h],
        "assessment": assessment,
        "state":      state_info,
        "timing_ms":  {
            "decode": 0,
            "yolo":   0,
            "logic":  round(t_logic, 1),
        },
    }


def _process_yolo(data: bytes, session: "PoseSession") -> dict:
    """
    CLI / test path: receive JPEG frame, run YOLO, then form logic.
    Only used when USE_YOLO=true (local development / video testing).
    """
    import cv2
    t0 = time.perf_counter()
    arr   = np.frombuffer(data, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        return {"error": "bad_frame"}
    h, w = frame.shape[:2]
    t_decode = (time.perf_counter() - t0) * 1000

    t1      = time.perf_counter()
    results = model(frame, conf=YOLO_CONF, iou=YOLO_IOU, verbose=False)
    t_yolo  = (time.perf_counter() - t1) * 1000

    result = results[0] if results else None
    raw_pts, raw_conf = session.extract_kp(result)

    if raw_pts is not None:
        pts, conf_arr = session.kp_buf.update(raw_pts, raw_conf)
    else:
        pts, conf_arr = None, None

    t2 = time.perf_counter()
    assessment, state_info = session.process(pts, conf_arr)
    t_logic = (time.perf_counter() - t2) * 1000

    kp_list = []
    if pts is not None:
        for i in range(len(pts)):
            c = float(conf_arr[i]) if conf_arr is not None else 1.0
            kp_list.append([round(float(pts[i][0]), 1), round(float(pts[i][1]), 1), round(c, 3)])

    return {
        "keypoints":  kp_list,
        "frame_size": [w, h],
        "assessment": assessment,
        "state":      state_info,
        "timing_ms":  {"decode": round(t_decode, 1), "yolo": round(t_yolo, 1), "logic": round(t_logic, 1)},
    }
