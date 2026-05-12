import { useRef, useEffect, useState, useCallback } from 'react'

const SEND_W = 640
const SEND_H = 480
const JPEG_Q = 0.65

export function usePoseSocket({ exercise, videoRef, onData, enabled = true }) {
  const wsRef          = useRef(null)
  const captureRef     = useRef(null)
  const responsePending = useRef(false)    // waiting for backend response
  const captureInProgress = useRef(false)  // currently encoding / converting
  const sendTimeRef    = useRef(0)
  const rafRef         = useRef(null)

  const [connected,  setConnected]  = useState(false)
  const [latency,    setLatency]    = useState(0)
  const [serverFps,  setServerFps]  = useState(0)
  const [timingMs,   setTimingMs]   = useState({ decode: 0, yolo: 0, logic: 0 })

  const fpsCountRef = useRef(0)
  const fpsTimerRef = useRef(performance.now())

  // Offscreen canvas
  useEffect(() => {
    const c = document.createElement('canvas')
    c.width  = SEND_W
    c.height = SEND_H
    captureRef.current = c
  }, [])

  // WebSocket lifecycle
  useEffect(() => {
    if (!exercise || !enabled) return

    const ws = new WebSocket(`ws://localhost:8000/ws/${exercise}`)
    ws.binaryType = 'arraybuffer'
    wsRef.current = ws

    ws.onopen = () => {
      setConnected(true)
      responsePending.current = false
      captureInProgress.current = false
    }

    ws.onclose = () => {
      setConnected(false)
      responsePending.current = false
      captureInProgress.current = false
    }

    ws.onerror = (e) => console.error('[WS] error', e)

    ws.onmessage = (evt) => {
      // Message received → response no longer pending
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

    return () => {
      ws.close()
      wsRef.current = null
      responsePending.current = false
      captureInProgress.current = false
    }
  }, [exercise, enabled])

  // Robust JPEG capture with single‑flight guarantee
  const captureAndSend = useCallback(() => {
    const ws      = wsRef.current
    const video   = videoRef.current
    const canvas  = captureRef.current

    if (!ws || ws.readyState !== WebSocket.OPEN) return
    if (responsePending.current || captureInProgress.current) return
    if (!video || video.readyState < 2) return
    if (!canvas) return

    const ctx = canvas.getContext('2d', { willReadFrequently: false })
    if (!ctx) return

    // Mirror
    ctx.save()
    ctx.translate(SEND_W, 0)
    ctx.scale(-1, 1)
    ctx.drawImage(video, 0, 0, SEND_W, SEND_H)
    ctx.restore()

    // Mark encoding in progress
    captureInProgress.current = true

    canvas.toBlob((blob) => {
      captureInProgress.current = false  // encoding done

      // If ws closed / a response is already pending, abort
      if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return
      if (responsePending.current) return

      if (!blob) return

      blob.arrayBuffer().then((buf) => {
        if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return
        if (responsePending.current) return

        // Both flags clear → we can send this frame
        responsePending.current = true
        sendTimeRef.current = performance.now()
        wsRef.current.send(buf)
      }).catch(() => {
        // Conversion failed – reset flags
        captureInProgress.current = false
      })
    }, 'image/jpeg', JPEG_Q)
  }, [videoRef])

  // RAF loop
  useEffect(() => {
    if (!enabled) return
    let active = true
    const loop = () => {
      if (!active) return
      captureAndSend()
      rafRef.current = requestAnimationFrame(loop)
    }
    rafRef.current = requestAnimationFrame(loop)
    return () => {
      active = false
      if (rafRef.current) cancelAnimationFrame(rafRef.current)
    }
  }, [captureAndSend, enabled])

  return { connected, latency, serverFps, timingMs }
}