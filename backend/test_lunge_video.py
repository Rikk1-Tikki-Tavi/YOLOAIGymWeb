#!/usr/bin/env python3
"""Test lunge detection on MP4 video files."""

import argparse
import cv2
import sys
from pathlib import Path

from ultralytics import YOLO
from aigym_core import extract_kp, draw_hud, draw_skeleton, EXERCISE_KP_MASK
from session import PoseSession

MODEL_PATH = "yolo11s-pose.pt"
YOLO_CONF = 0.50
YOLO_IOU = 0.70

def main():
    parser = argparse.ArgumentParser(description="Test lunge detection on video")
    parser.add_argument("--video", type=str, required=True, help="Path to MP4 video")
    parser.add_argument("--output", type=str, default=None, help="Output video path (optional)")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Error: Video not found: {video_path}")
        sys.exit(1)

    # Load model
    print(f"Loading model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    import numpy as np
    dummy = np.zeros((480, 640, 3), dtype=np.uint8)
    model(dummy, conf=YOLO_CONF, iou=YOLO_IOU, verbose=False)

    # Setup video
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video: {w}x{h} @ {fps} fps")

    # Setup output video if requested
    out = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(args.output, fourcc, fps, (w, h))
        print(f"Output: {args.output}")

    # Setup session (uses PoseSession for full processing)
    session = PoseSession("lunges")

    frame_idx = 0
    print("Processing video...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO detection
        results = model(frame, conf=YOLO_CONF, iou=YOLO_IOU, verbose=False)
        result = results[0] if results else None
        raw_pts, raw_conf = session.extract_kp(result)

        # Update keypoint buffer
        if raw_pts is not None:
            pts, conf_arr = session.kp_buf.update(raw_pts, raw_conf)
        else:
            pts, conf_arr = None, None

        # Run full session processing (includes EvidenceGate, rep counting, etc.)
        assessment, state_info = session.process(pts, conf_arr)

        # Reconstruct motion_cue tuple for draw_hud (session.process returns separate fields)
        assessment["motion_cue"] = (assessment.get("motion_cue_text", ""),
                                     assessment.get("motion_cue_severity", "neutral"))

        # Draw skeleton
        if pts is not None:
            allowed = EXERCISE_KP_MASK.get("lunges", set(range(17)))
            draw_skeleton(frame, pts, conf_arr, allowed, {})

        # Draw HUD
        draw_hud(frame, assessment, session.state, session.ex)

        # Print shin tilt angle and form
        motion_text = assessment.get("motion_cue_text", "")
        form_label = assessment.get("form_label", "")
        if "°" in motion_text or form_label != "Transitioning...":
            print(f"Frame {frame_idx}: {motion_text} | Form: {form_label}")

        # Display frame
        cv2.imshow("Lunge Detection", frame)
        if out:
            out.write(frame)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break

        frame_idx += 1
        if frame_idx % 30 == 0:
            print(f"  Frame {frame_idx} ({frame_idx/fps:.1f}s)")

    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()

    print(f"Done! Processed {frame_idx} frames")

if __name__ == "__main__":
    main()
