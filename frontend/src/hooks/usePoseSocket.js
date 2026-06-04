import { useRef, useEffect, useState, useCallback } from 'react'

/**
 * usePoseSocket
 * -------------
 * Runs in-browser pose estimation (via detectPose from usePoseDetector) then
 * sends the resulting 17 keypoints as JSON text to the backend WebSocket.
 *
 * This replaces the old approach of sending raw JPEG frames (~10 KB each) to
 * the VPS for YOLO inference (100-300ms on CPU). By running pose estimation
 * locally (~10-20ms on WebGL) the round-trip drops from 200-400ms to 15-50ms.
 *
 * Payload sent to backend:
 *   { "keypoints": [[x,y,conf] × 17], "frame_size": [videoWidth, videoHeight] }
 *   — ~500 bytes JSON text vs ~10 KB binary
 */
export function usePoseSocket({ exercise, dominantFoot, videoRef, detectPose, onData, onLocalKP, enabled = true }) {
  const wsRef            = useRef(null)
  const rafRef           = useRef(null)
  const responsePending  = useRef(false)
  const sendTimeRef      = useRef(0)
  const kpSmoothRef      = useRef(null)   // EMA-smoothed keypoints for skeleton rendering

  const [connected,  setConnected]  = useState(false)
  const [latency,    setLatency]    = useState(0)
  const [serverFps,  setServerFps]  = useState(0)
  const [timingMs,   setTimingMs]   = useState({ decode: 0, yolo: 0, logic: 0 })

  const fpsCountRef = useRef(0)
  const fpsTimerRef = useRef(performance.now())

  // WebSocket lifecycle
  useEffect(() => {
    if (!exercise || !enabled) return

    const foot = dominantFoot ?? 'L'
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const ws = new WebSocket(`${protocol}//${window.location.host}/ws/${exercise}/${foot}`)
    wsRef.current = ws

    ws.onopen = () => {
      setConnected(true)
      responsePending.current = false
    }
    ws.onclose = () => {
      setConnected(false)
      responsePending.current = false
    }
    ws.onerror = (e) => console.error('[WS] error', e)

    ws.onmessage = (evt) => {
      const rtt = performance.now() - sendTimeRef.current
      setLatency(Math.round(rtt))
      responsePending.current = false

      fpsCountRef.current++
      const now = performance.now()
      if (now - fpsTimerRef.current >= 1000) {
        setServerFps(fpsCountRef.current)
        fpsCountRef.current = 0
        fpsTimerRef.current = now
      }

      try {
        const data = JSON.parse(evt.data)
        if (data.timing_ms) setTimingMs(data.timing_ms)
        onData?.(data)
      } catch (e) {
        console.error('[WS] parse error', e)
      }
    }

    return () => { ws.close(); wsRef.current = null; responsePending.current = false }
  }, [exercise, dominantFoot, enabled])

  // Main pose-detect-and-send loop
  const runPoseAndSend = useCallback(async () => {
    const video = videoRef.current
    if (!video || video.readyState < 2) return
    if (!detectPose) return

    // Always run pose detection regardless of backend state.
    // This keeps the skeleton rendering at native MoveNet rate (~30-60fps)
    // even when the backend hasn't responded yet (planks, slow connections).
    const rawKP = await detectPose(video)
    if (!rawKP) {
      // No confident pose — clear stale skeleton and reset smoother so the
      // next detection starts fresh (no blend with stale ghost positions).
      kpSmoothRef.current = null
      onLocalKP?.(null)
      return
    }

    const vW = video.videoWidth
    const vH = video.videoHeight

    // Mirror x: the canvas draws the video flipped (selfie mode) so keypoints
    // must be flipped too before sending to the backend and before rendering.
    const mirroredKP = rawKP.map(([x, y, c]) => [vW - x, y, c])

    // EMA-smooth keypoints for the skeleton display only.
    // Alpha=0.65: responsive but stable enough to kill MoveNet noise jitter.
    // Raw mirroredKP is sent to the backend so angle calculations stay accurate.
    const ALPHA = 0.65
    const prev  = kpSmoothRef.current
    const smoothed = (prev && prev.length === mirroredKP.length)
      ? mirroredKP.map(([x, y, c], i) => [
          ALPHA * x + (1 - ALPHA) * prev[i][0],
          ALPHA * y + (1 - ALPHA) * prev[i][1],
          c,
        ])
      : mirroredKP
    kpSmoothRef.current = smoothed

    // Fire smoothed keypoints for 60fps skeleton rendering
    onLocalKP?.(smoothed)

    // Only send raw keypoints to backend when the previous response has been received.
    const ws = wsRef.current
    if (!ws || ws.readyState !== WebSocket.OPEN) return
    if (responsePending.current) return

    // Send keypoints JSON (~500 bytes) instead of a JPEG frame (~10 KB)
    responsePending.current = true
    sendTimeRef.current = performance.now()
    ws.send(JSON.stringify({ keypoints: mirroredKP, frame_size: [vW, vH] }))
  }, [videoRef, detectPose, onLocalKP])

  // RAF loop
  useEffect(() => {
    if (!enabled) return
    let active = true
    const loop = () => {
      if (!active) return
      runPoseAndSend()
      rafRef.current = requestAnimationFrame(loop)
    }
    rafRef.current = requestAnimationFrame(loop)
    return () => {
      active = false
      if (rafRef.current) cancelAnimationFrame(rafRef.current)
    }
  }, [runPoseAndSend, enabled])

  return { connected, latency, serverFps, timingMs }
}
