# AIGym — Deployment Guide

Everything you need to understand, manage, and re-deploy the production server yourself.

---

## 1. What Changed: Architecture Overview

### Old Architecture (localhost only)
```
Browser
  │
  ├─► Sends raw JPEG frame (~10 KB binary) via WebSocket
  │
VPS Backend (FastAPI)
  ├─► Decodes JPEG
  ├─► Runs YOLO11 pose estimation on CPU (~100–300 ms, no GPU)
  ├─► Runs form-analysis logic (~5 ms)
  └─► Returns keypoints + assessment JSON
  │
Browser
  └─► Draws skeleton + HUD
```
**Problem:** 200–400 ms round-trip latency on VPS because YOLO is slow without a GPU.

---

### New Architecture (production)
```
Browser
  ├─► Loads TF.js MoveNet model from CDN (once, cached by browser)
  ├─► Runs pose estimation locally on WebGL/GPU (~10–20 ms)
  ├─► Sends 17 keypoints as JSON (~500 bytes) via WebSocket
  │
VPS Backend (FastAPI)
  ├─► Receives JSON (no image decode, no YOLO)
  ├─► Runs form-analysis logic (~5 ms)
  └─► Returns assessment JSON
  │
Browser
  └─► Draws skeleton immediately from local keypoints (60 fps)
      HUD updates from backend responses (~15–30 fps)
```
**Result:** 15–50 ms total latency regardless of server location.

---

## 2. Key Terms & Tools

| Term | What It Is |
|------|-----------|
| **VPS** | Virtual Private Server — your rented Linux machine at `43.157.201.110` |
| **PM2** | Node.js process manager used to keep the Python backend running 24/7 |
| **Nginx** | Web server that acts as a reverse proxy — receives browser requests on port 80/443 and forwards them to the FastAPI backend on port 8000 |
| **FastAPI** | Python web framework powering the backend WebSocket server |
| **Uvicorn** | ASGI server that runs FastAPI inside PM2 |
| **TF.js / MoveNet** | TensorFlow.js running in the browser — replaces YOLO for pose estimation |
| **WebSocket** | Persistent two-way connection between browser and backend for real-time data |
| **Let's Encrypt** | Free SSL certificate authority — certificate is already on the VPS at `/etc/letsencrypt/live/brevonsolutions.com/` |
| **paramiko / scp** | Python libraries for SSH and file transfer (used by the deploy scripts) |
| **Vite** | Frontend build tool — bundles React JSX into plain JS for production |
| **CDN** | Content Delivery Network — TF.js is loaded from `cdn.jsdelivr.net` instead of bundled locally |

---

## 3. Server Layout

```
/var/www/aigym/
├── backend/              ← Python FastAPI source files
│   ├── main.py           ← WebSocket server entry point
│   ├── aigym_core.py     ← Exercise form detection logic
│   ├── session.py        ← Per-connection state management
│   └── requirements.txt
└── frontend/             ← Built React app (static files served by Nginx)
    ├── index.html
    └── assets/
        ├── index-*.js
        └── index-*.css
```

Nginx serves `/var/www/aigym/frontend/` for the website and proxies `/ws/` requests to `localhost:8000` (the FastAPI backend).

---

## 4. Files Changed in This Refactor

### `frontend/index.html`
Added CDN script tags **before** the React module so TF.js globals are available when React loads:
```html
<!-- Full tf.min.js required — pose-detection needs tf.loadGraphModel (tfjs-converter) -->
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/@tensorflow-models/pose-detection@2.1.3/dist/pose-detection.min.js"></script>
```

### `frontend/src/hooks/usePoseDetector.js` *(new file)*
Loads MoveNet in the browser. Uses `window.tf` and `window.poseDetection` (injected by the CDN scripts above). Returns `{ detectPose, isReady }`.

### `frontend/src/hooks/usePoseSocket.js`
Completely rewritten:
- **Before:** Captured video frame to canvas → encoded to JPEG → sent as binary WebSocket message
- **After:** Calls `detectPose(video)` → mirrors keypoints → sends JSON text `{ keypoints: [[x,y,conf]×17], frame_size: [w,h] }`

### `backend/main.py`
- Added `USE_YOLO = os.getenv("USE_YOLO", "false").lower() == "true"` — YOLO model is not loaded on VPS (saves ~500 MB RAM)
- WebSocket handler changed from `receive_bytes()` to `receive_text()`
- Added `_process_kp(msg, session)` — parses JSON keypoints and runs form logic (no image decode)
- WebSocket route changed to `/ws/{exercise}/{dominant_foot}` (dominant foot passed at connection time)

### `backend/session.py`
- Constructor now accepts `dominant_foot` parameter and stores it in state for lunge detection

### `backend/aigym_core.py`
- Lunge detection rewritten:
  - **Dominant foot selection** replaces dynamic leg detection (user picks L/R upfront)
  - **Var1 (knee overextension):** Angle-only; fires when `front_ka < KNEE_OVER_ANGLE_DEEP` (70 degrees) for 6 consecutive frames. The earlier positional knee-vs-ankle check was removed because it was sensitive to MoveNet x-coordinate noise.
  - **Var2 (wide step):** Latch mechanism using the front-rear knee asymmetry: fires when `(rear_ka - front_ka) > WIDE_STEP_ASYM_DEG` (25 degrees) for 10 consecutive frames; releases when rear knee bends below 100 degrees or user stands.
  - **Var3 (torso lean):** Flexible keypoint fallback for low-confidence frames
- Squat detection updated: correct depth is `ka <= 65`; the 65-70 degree band is the noise margin; `min_angle_this_rep > 70` flags var1 at rep close. The previous "too deep" warning has been removed (deeper squats are treated as equally correct).
- Setup gate removed from all four checkers and the session loop; scoring now begins from the first frame of pose detection.
- Plank detection fixed: `motion_cue` was undefined when `dominant_tag` was `None`

---

## 5. How to Re-Deploy (Full Steps)

### Prerequisites (one-time setup on your Windows machine)
```powershell
pip install paramiko scp
cd "D:\DOCUMENTS\SGU DOCUMENTS\Thesis\aigym-web2\frontend"
npm install
```

### Step-by-step deploy
```powershell
# 1. Build the React frontend
cd "D:\DOCUMENTS\SGU DOCUMENTS\Thesis\aigym-web2\frontend"
npm run build

# 2. Upload frontend + backend and restart PM2
cd "D:\DOCUMENTS\SGU DOCUMENTS\Thesis\aigym-web2"
python deploy_update.py
```

That's it. `deploy_update.py` handles:
- Uploading `frontend/dist/` → `/var/www/aigym/frontend/`
- Uploading `backend/` → `/var/www/aigym/backend/`
- Running `pm2 restart aigym`
- Printing the live URL

---

## 6. Commands to Manage the Server Yourself

### SSH into the server
```bash
ssh ubuntu@43.157.201.110
# password: panda-86$-thunder
```

### PM2 — Process Manager Commands
```bash
pm2 list                        # See all running processes and their status
pm2 logs aigym                  # Live log stream (Ctrl+C to stop)
pm2 logs aigym --lines 50       # Last 50 lines without streaming
pm2 restart aigym               # Restart the backend (use after code changes)
pm2 stop aigym                  # Stop the backend
pm2 start aigym                 # Start the backend
pm2 show aigym                  # Detailed info (PID, uptime, memory, env vars)
```

### Get logs from your Windows machine (without SSH)
```powershell
cd "D:\DOCUMENTS\SGU DOCUMENTS\Thesis\aigym-web2"
python get_logs.py
```

### Nginx — Web Server Commands
```bash
sudo nginx -t                   # Test config for syntax errors (always run before reload)
sudo systemctl reload nginx     # Apply config changes without downtime
sudo systemctl restart nginx    # Full restart (only if reload doesn't work)
sudo systemctl status nginx     # Check if nginx is running
sudo cat /var/log/nginx/error.log | tail -20   # Recent nginx errors
```

### Check what's running on a port
```bash
sudo ss -tlnp | grep :80        # Who's listening on port 80
sudo ss -tlnp | grep :8000      # Confirm FastAPI is on 8000
```

### SSL Certificate
The wildcard cert for `*.brevonsolutions.com` is already installed at:
```
/etc/letsencrypt/live/brevonsolutions.com/fullchain.pem
/etc/letsencrypt/live/brevonsolutions.com/privkey.pem
```
To renew (auto-renews via cron, but manual command if needed):
```bash
sudo certbot renew --dry-run    # Test renewal
sudo certbot renew              # Actually renew
```

---

## 7. Environment Variables (PM2)

The backend reads these at startup:

| Variable | Value on VPS | Effect |
|----------|-------------|--------|
| `USE_YOLO` | `false` | Skips loading YOLO model — saves ~500 MB RAM, required since VPS has no GPU |

To change an env var:
```bash
pm2 set aigym:USE_YOLO false    # Set in PM2 ecosystem
pm2 restart aigym               # Apply it
```
Or edit `/var/www/aigym/ecosystem.config.js` on the server and run `pm2 reload ecosystem.config.js`.

---

## 8. DNS Setup (Required for HTTPS)

HTTPS at `https://aigym.brevonsolutions.com` requires one DNS record added in your domain registrar:

| Type | Name | Value | TTL |
|------|------|-------|-----|
| A | `aigym` | `43.157.201.110` | 300 |

Also make sure **port 443** is open in your Tencent Cloud security group (the firewall for the VPS). Without that, HTTPS connections will hang even if Nginx is configured correctly.

Path: Tencent Cloud Console → Lighthouse/CVM → Security Groups → Add inbound rule → TCP 443

---

## 9. Troubleshooting Quick Reference

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| Site not loading | PM2 crashed | `pm2 restart aigym` on server |
| `frames=0` in PM2 logs | Browser model failed to load | Open DevTools Console on the site and check for JS errors |
| "LOADING AI MODEL" stuck forever | TF.js CDN failed or WebGL blocked | Try a different browser; check Console for errors |
| Skeleton draws but HUD is empty | WebSocket not connecting | Check OFFLINE/LIVE indicator top-right; check Nginx config |
| HTTPS not working | DNS not set or port 443 blocked | See Section 8 above |
| Backend using too much RAM | YOLO accidentally loaded | `pm2 show aigym` → confirm `USE_YOLO=false` |
| Form detection wrong | Logic bug in aigym_core.py | Fix code → rebuild → `python deploy_update.py` |




Great question. The short answer is: YOLO runs on the server, not the user's machine — so it can never use the user's GPU.
  
  Here's the fundamental difference:

  ┌────────────────┬────────────────────────────────────────────────┬───────────────────────────────────────────────┐
  │                │                   YOLO (old)                   │                 MoveNet (new)                 │
  ├────────────────┼────────────────────────────────────────────────┼───────────────────────────────────────────────┤
  │ Where it runs  │ Your VPS (remote server)                       │ User's browser (their own device)             │
  ├────────────────┼────────────────────────────────────────────────┼───────────────────────────────────────────────┤
  │ GPU available? │ Only if VPS has a GPU — yours doesn't          │ Always — browser uses user's GPU via WebGL    │
  ├────────────────┼────────────────────────────────────────────────┼───────────────────────────────────────────────┤
  │ Latency source │ CPU inference (100–300ms) + network round-trip │ Local GPU inference (~10–20ms), no round-trip │
  └────────────────┴────────────────────────────────────────────────┴───────────────────────────────────────────────┘

  The VPS is just a rented Linux machine in a data center. It has no GPU. When YOLO runs there, it's stuck on CPU no matter what. You can't tell a remote server to "use the user's GPU" — the user's GPU is
   physically on their own computer, not on your server.

  MoveNet solves this by flipping the architecture: instead of sending video to the server for inference, the inference runs inside the browser using TensorFlow.js + WebGL. WebGL is a browser API that
  gives JavaScript direct access to the user's local GPU — whether that's an NVIDIA RTX, an Apple M-series chip, or even a phone's mobile GPU. So every user gets GPU-accelerated inference for free, on
  their own hardware.

  The only way YOLO could use a GPU would be to either:
  1. Upgrade the VPS to a GPU instance — possible, but GPU VPS plans cost 10–50× more (typically $300–$1000+/month vs ~$10–30 for a CPU VPS)
  2. Run YOLO in the browser — YOLO models converted to TF.js exist, but YOLO11 is ~100MB to download vs MoveNet Thunder's ~12MB, and it's slower in the browser than MoveNet which was specifically
  designed for real-time browser inference

  For a thesis project on a shared VPS, MoveNet is the right call