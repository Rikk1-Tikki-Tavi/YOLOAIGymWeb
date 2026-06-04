# AIGym Web — Real-Time Form Correction in the Browser

## Why this is fast (< 100 ms end-to-end)

The core problem with your two previous attempts:

| Approach | Why it lagged |
|---|---|
| MJPEG | Server encodes annotated frame (~100 KB) and sends it back. Frames pile up in a queue. Latency compounds. |
| WebRTC | ICE/STUN negotiation, codec pipelines, and browser policies all add complexity and latency. |

**This solution:** WebSocket **ping-pong** with **keypoints-only response**.

```
Browser webcam (local, 0 ms) ──JPEG 640×480──► FastAPI WS
                                                └─► YOLO on GPU (~20-30 ms)
                                                └─► return JSON keypoints (~1.5 KB)
Browser canvas overlay ◄─────────────────────── JSON keypoints
```

**Why it's fast:**
1. **Video never leaves the browser.** The `<canvas>` draws from the local `<video>` element — zero latency for the video itself.
2. **Ping-pong protocol.** Only ONE frame is in-flight at a time. If GPU takes 30 ms, you get 33 fps with exactly 30 ms of pose latency — never more, because no queue can build up.
3. **Keypoints are tiny.** ~1.5 KB JSON vs ~80 KB annotated JPEG. The response is 50× smaller.
4. **YOLO in thread pool.** `asyncio.get_event_loop().run_in_executor(executor, ...)` keeps the async event loop free while GPU is working.
5. **Identical logic.** The backend imports your `aigym_core.py` directly — no re-implementation, same EvidenceGate and rep counting as desktop.

---

## File Structure

```
aigym-web/
├── backend/
│   ├── main.py              ← FastAPI + WebSocket server
│   ├── session.py           ← PoseSession (wraps your existing logic)
│   ├── aigym_core.py        ← YOUR FILE: copy aigym_pose_detection.py here
│   └── requirements.txt
└── frontend/
    ├── index.html
    ├── package.json
    ├── vite.config.js
    └── src/
        ├── main.jsx
        ├── index.css
        ├── App.jsx              ← Exercise selector
        ├── components/
        │   └── ExerciseView.jsx ← Webcam + canvas + HUD
        ├── hooks/
        │   └── usePoseSocket.js ← WS frame pump (ping-pong)
        └── utils/
            └── skeleton.js      ← Canvas drawing (skeleton + HUD)
```

---

## Setup Instructions

### Step 1 — Copy your existing Python file

```bash
# Rename and copy your existing pose detection file
cp aigym_pose_detection.py aigym-web/backend/aigym_core.py
```

### Step 2 — Backend

```bash
cd aigym-web/backend

# Create a virtual environment (recommended)
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Make sure your YOLO model files are present
# (yolo11n-pose.pt downloads automatically on first run if you have ultralytics)
# If you want max accuracy: change MODEL_PATH in main.py to "yolo11x-pose.pt"

# Start the server
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

You should see:
```
INFO  Loading model: yolo11n-pose.pt
INFO  Model warm-up complete — ready for connections.
INFO  Uvicorn running on http://0.0.0.0:8000
```

### Step 3 — Frontend

Open a **second terminal**:

```bash
cd aigym-web/frontend
npm install
npm run dev
```

You should see:
```
VITE v5.x  ready in 500ms
➜  Local:   http://localhost:5173/
```

### Step 4 — Open the app

Navigate to **http://localhost:5173** in Chrome.

- Select an exercise
- Allow camera access when prompted
- Wait 7 seconds for the countdown
- Start exercising — the skeleton and HUD appear in real-time

---

## Expected Performance

| Metric | Expected value |
|---|---|
| End-to-end latency (RTT shown in debug bar) | 30–70 ms |
| Server FPS (shown in debug bar) | 20–33 fps |
| YOLO inference (yolo11n-pose, RTX GPU) | 8–20 ms |
| YOLO inference (yolo11x-pose, RTX GPU) | 25–50 ms |
| JSON response size | ~1.5 KB |

The debug bar at the bottom of the canvas shows live `RTT`, `SRV fps`, `YOLO ms`, and `TOT ms`.

---

## Model Selection

In `backend/main.py`, line:
```python
MODEL_PATH = "yolo11n-pose.pt"
```

| Model | Speed | Accuracy | Recommended for |
|---|---|---|---|
| `yolo11n-pose.pt` | Fastest (~10 ms) | Good | Thesis demo |
| `yolo11s-pose.pt` | Fast (~15 ms) | Better | Default |
| `yolo11x-pose.pt` | Slower (~35 ms) | Best | Final demo |

---

## Browser Requirements

No special flags needed. This solution uses:
- `getUserMedia` (standard since Chrome 47)
- `WebSocket` (standard since Chrome 16)
- `Canvas 2D API` (standard)
- No WebGPU, no WebRTC, no browser flags

Chrome 90+ is recommended (all standard features).

---

## Deployment Notes (for later)

For deployment beyond localhost:
1. **HTTPS required for `getUserMedia`** — use Let's Encrypt or Cloudflare.
2. Update `vite.config.js` proxy targets to your server's URL.
3. Update `CORSMiddleware` origins in `main.py`.
4. For cloud GPU (e.g., RunPod, Lambda Labs): the latency will increase by network RTT (~20-80 ms for same region), still well under 200 ms.

---

## Troubleshooting

**"Camera error: Permission denied"**
→ Click the camera icon in Chrome's address bar and allow access.

**Backend won't start / import error**
→ Make sure `aigym_core.py` is in the `backend/` folder (copy of your `aigym_pose_detection.py`).

**Skeleton appears but offset**
→ Verify `frame_size` in the debug console (F12). The backend and frontend must agree on frame dimensions.

**High latency (>150 ms)**
→ Switch from `yolo11x-pose.pt` to `yolo11n-pose.pt` in `main.py`.

**"OFFLINE" shown in header**
→ Check that `uvicorn` is running on port 8000 and the Vite proxy is active.
