/**
 * ExerciseView
 * ============
 * Layout:
 *   ┌─────────────────────────────┐
 *   │  <video>  (local webcam)    │  ← browser renders its own stream, zero latency
 *   │  <canvas> (overlay, same   │  ← skeleton + HUD drawn from backend keypoints
 *   │            size, absolute) │
 *   └─────────────────────────────┘
 *
 * The video element is hidden (opacity 0) and the canvas draws on top.
 * This means the user sees: their live feed drawn by the browser (0 ms latency)
 * PLUS the skeleton drawn from ~30-60 ms old keypoints — which is imperceptible.
 */

import { useRef, useEffect, useCallback, useState } from 'react'
import { usePoseSocket } from '../hooks/usePoseSocket'
import {
  drawSkeleton, drawHUD, drawDebugBar,
  scaleKP,
} from '../utils/skeleton'

export default function ExerciseView({ exercise, onBack }) {
  const videoRef        = useRef(null)
  const canvasRef       = useRef(null)
  const streamRef       = useRef(null)

  // Latest data from backend (updated every WS response ~30ms)
  const poseDataRef     = useRef(null)
  const rafRef          = useRef(null)

  const [camReady,  setCamReady]  = useState(false)
  const [camError,  setCamError]  = useState(null)
  const [countdown, setCountdown] = useState(6)
  const [started,   setStarted]   = useState(false)
  const [finished,  setFinished]  = useState(false)
  const [summary,   setSummary]   = useState(null)

  // ── Start webcam ──────────────────────────────────────────────────────────
  useEffect(() => {
    let mounted = true
    navigator.mediaDevices
      .getUserMedia({ video: { width: 1280, height: 720, frameRate: 30 } })
      .then((stream) => {
        if (!mounted) { stream.getTracks().forEach(t => t.stop()); return }
        streamRef.current         = stream
        videoRef.current.srcObject = stream
        videoRef.current.play()
        setCamReady(true)
      })
      .catch((e) => {
        if (mounted) setCamError(e.message)
      })

    return () => {
      mounted = false
      streamRef.current?.getTracks().forEach(t => t.stop())
      if (rafRef.current) cancelAnimationFrame(rafRef.current)
    }
  }, [])

  // ── Countdown then start ──────────────────────────────────────────────────
  useEffect(() => {
    if (!camReady) return
    let n = 6
    setCountdown(n)
    const id = setInterval(() => {
      n--
      setCountdown(n)
      if (n <= 0) { clearInterval(id); setStarted(true) }
    }, 1000)
    return () => clearInterval(id)
  }, [camReady])

  // ── Receive keypoints from backend ────────────────────────────────────────
  const handleData = useCallback((data) => {
    poseDataRef.current = data
  }, [])

  const { connected, latency, serverFps, timingMs } = usePoseSocket({
    exercise,
    videoRef,
    onData:  handleData,
    enabled: started && !finished,
  })

  const handleFinish = useCallback(() => {
    const snapshot = poseDataRef.current?.state || null
    setSummary(snapshot)
    setFinished(true)
    setStarted(false)
  }, [])

  // ── Keyboard shortcut (F to finish) ─────────────────────────────────────
  useEffect(() => {
    const onKeyDown = (e) => {
      if (e.key?.toLowerCase() === 'f') {
        if (!started || finished) return
        handleFinish()
      }
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [handleFinish, started, finished])

  const getVarLabel = useCallback((key, varNames) => {
    if (!key) return ''
    if (key.startsWith('var')) {
      const idx = key.slice(3)
      return (varNames && varNames[idx]) ? varNames[idx] : key
    }
    return key
  }, [])

  // ── RAF render loop — draws every animation frame ─────────────────────────
  useEffect(() => {
    if (!camReady) return

    const canvas = canvasRef.current
    const video  = videoRef.current

    const render = () => {
      rafRef.current = requestAnimationFrame(render)

      const cw = canvas.width
      const ch = canvas.height

      const ctx = canvas.getContext('2d')
      ctx.clearRect(0, 0, cw, ch)

      // ── Draw webcam frame (mirrored) ──────────────────────────────────────
      ctx.save()
      ctx.translate(cw, 0)
      ctx.scale(-1, 1)
      if (video.readyState >= 2) {
        ctx.drawImage(video, 0, 0, cw, ch)
      }
      ctx.restore()

      // ── Overlay darkening + vignette ──────────────────────────────────────
      const grad = ctx.createRadialGradient(cw/2, ch/2, ch*0.3, cw/2, ch/2, ch*0.75)
      grad.addColorStop(0, 'rgba(0,0,0,0)')
      grad.addColorStop(1, 'rgba(0,0,0,0.45)')
      ctx.fillStyle = grad
      ctx.fillRect(0, 0, cw, ch)

      if (!started && !finished) {
        // Countdown overlay
        ctx.fillStyle = 'rgba(8,12,18,0.6)'
        ctx.fillRect(0, 0, cw, ch)
        ctx.fillStyle = '#e8f4ff'
        ctx.font      = `bold ${countdown > 0 ? 120 : 80}px "Share Tech Mono", monospace`
        ctx.textAlign  = 'center'
        ctx.textBaseline = 'middle'
        ctx.fillText(countdown > 0 ? countdown : 'GO!', cw / 2, ch / 2)
        ctx.textAlign    = 'left'
        ctx.textBaseline = 'alphabetic'
        return
      }

      // ── Skeleton + HUD from latest pose data ─────────────────────────────
      const data = poseDataRef.current
      if (!finished && data && !data.error) {
        const { keypoints, frame_size, assessment, state: stateInfo } = data

        if (keypoints && frame_size) {
          // Scale keypoints from backend frame coords → canvas display coords
          const scaledKP = scaleKP(keypoints, frame_size[0], frame_size[1], cw, ch)
          drawSkeleton(ctx, scaledKP, assessment)
        }

        drawHUD(ctx, cw, ch, assessment, stateInfo)
      } else if (started && !finished) {
        // No data yet — show "waiting for signal"
        ctx.fillStyle = 'rgba(8,12,18,0.55)'
        ctx.fillRect(0, 0, cw, 64)
        ctx.fillStyle = '#5a7080'
        ctx.font      = 'bold 14px "Exo 2", sans-serif'
        ctx.textBaseline = 'middle'
        ctx.fillText('WAITING FOR SIGNAL…', 18, 32)
        ctx.textBaseline = 'alphabetic'
      }

      // ── Debug bar (always visible) ────────────────────────────────────────
      drawDebugBar(ctx, cw, ch, { latency, serverFps, timingMs, connected })
    }

    rafRef.current = requestAnimationFrame(render)
    return () => { if (rafRef.current) cancelAnimationFrame(rafRef.current) }
  }, [camReady, started, countdown, connected, latency, serverFps, timingMs])

  // ── Resize canvas to match container ─────────────────────────────────────
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ro = new ResizeObserver(([entry]) => {
      const { width, height } = entry.contentRect
      canvas.width  = Math.floor(width)
      canvas.height = Math.floor(height)
    })
    ro.observe(canvas.parentElement)
    return () => ro.disconnect()
  }, [])

  // ── Render ────────────────────────────────────────────────────────────────
  return (
    <div style={styles.root}>

      {/* ── Header bar ───────────────────────────────────────────────────── */}
      <div style={styles.header}>
        <button style={styles.backBtn} onClick={onBack}>← BACK</button>
        <span style={styles.exLabel}>{exercise.toUpperCase()}</span>
        <button style={styles.finishBtn} onClick={handleFinish} disabled={!started || finished}>
          FINISH
        </button>
        <div style={styles.connDot(connected)} title={connected ? 'Connected' : 'Disconnected'}>
          {connected ? 'LIVE' : 'OFFLINE'}
        </div>
      </div>

      {/* ── Video + Canvas stage ─────────────────────────────────────────── */}
      <div style={styles.stage}>
        {/* Hidden video: only used as source for canvas drawImage */}
        <video
          ref={videoRef}
          style={styles.video}
          playsInline muted autoPlay
        />
        {/* Canvas: renders everything the user actually sees */}
        <canvas ref={canvasRef} style={styles.canvas} />

        {/* Summary overlay */}
        {finished && summary && (
          <div style={styles.summaryOverlay}>
            <div style={styles.summaryCard}>
              <div style={styles.summaryTitle}>SESSION SUMMARY</div>
              <div style={styles.summarySubtitle}>{summary.exercise_label}</div>

              {summary.is_timed ? (
                <div style={styles.summaryBlock}>
                  <div style={styles.summaryRow}>
                    <span style={styles.summaryLabel}>Hold Time</span>
                    <span style={styles.summaryValue}>{summary.hold_time}s</span>
                  </div>
                  <div style={styles.summaryRow}>
                    <span style={styles.summaryLabel}>Grade</span>
                    <span style={styles.summaryBadge(summary.hold_grade)}>{summary.hold_grade}</span>
                  </div>
                  <div style={styles.summarySectionTitle}>Form Issues</div>
                  {summary.form_issues && summary.form_issues.length ? (
                    summary.form_issues.map((key) => (
                      <div key={key} style={styles.summaryIssue}>
                        {getVarLabel(key, summary.var_names)}
                      </div>
                    ))
                  ) : (
                    <div style={styles.summaryIssueGood}>None — great form throughout!</div>
                  )}
                </div>
              ) : (
                <div style={styles.summaryBlock}>
                  <div style={styles.summaryRow}>
                    <span style={styles.summaryLabel}>Score</span>
                    <span style={styles.summaryValue}>
                      {summary.correct_reps}/{summary.total_reps} ({Math.round(summary.score_pct)}%)
                    </span>
                  </div>
                  <div style={styles.summaryRow}>
                    <span style={styles.summaryLabel}>Grade</span>
                    <span style={styles.summaryBadge(summary.grade)}>{summary.grade}</span>
                  </div>
                  <div style={styles.summarySectionTitle}>Breakdown</div>
                  <div style={styles.summaryRow}>
                    <span style={styles.summaryLabel}>Correct</span>
                    <span style={styles.summaryValue}>{summary.rep_breakdown?.correct ?? 0}</span>
                  </div>
                  {["var1", "var2", "var3"].map((key) => (
                    summary.var_names?.[key.slice(3)] ? (
                      <div key={key} style={styles.summaryRow}>
                        <span style={styles.summaryLabel}>{summary.var_names[key.slice(3)]}</span>
                        <span style={styles.summaryValue}>{summary.rep_breakdown?.[key] ?? 0}</span>
                      </div>
                    ) : null
                  ))}
                </div>
              )}

              <div style={styles.summaryActions}>
                <button style={styles.summaryBtn} onClick={onBack}>BACK</button>
              </div>
            </div>
          </div>
        )}

        {/* Camera error overlay */}
        {camError && (
          <div style={styles.errorOverlay}>
            <p style={{ color: '#ff3b3b', fontFamily: 'Exo 2, sans-serif' }}>
              Camera error: {camError}
            </p>
            <p style={{ color: '#5a7080', fontSize: 13, marginTop: 8 }}>
              Allow camera access in your browser and refresh.
            </p>
          </div>
        )}
      </div>

    </div>
  )
}

// ── Inline styles ─────────────────────────────────────────────────────────────
const styles = {
  root: {
    display:       'flex',
    flexDirection: 'column',
    width:         '100%',
    height:        '100vh',
    background:    '#080c12',
    overflow:      'hidden',
  },
  header: {
    display:        'flex',
    alignItems:     'center',
    gap:            16,
    padding:        '10px 20px',
    background:     '#0d1420',
    borderBottom:   '1px solid #1e2d42',
    flexShrink:     0,
  },
  backBtn: {
    background:    'none',
    border:        '1px solid #1e3a5f',
    borderRadius:  4,
    color:         '#3d9eff',
    fontFamily:    '"Exo 2", sans-serif',
    fontSize:      12,
    fontWeight:    600,
    padding:       '5px 12px',
    cursor:        'pointer',
    letterSpacing: 1,
  },
  exLabel: {
    fontFamily:    '"Exo 2", sans-serif',
    fontWeight:    700,
    fontSize:      16,
    letterSpacing: 3,
    color:         '#e8f4ff',
    flex:          1,
  },
  connDot: (ok) => ({
    fontFamily:  '"Share Tech Mono", monospace',
    fontSize:    11,
    color:       ok ? '#00e87a' : '#ff3b3b',
    letterSpacing: 1,
  }),
  finishBtn: {
    background:    'transparent',
    border:        '1px solid #2c3f57',
    borderRadius:  4,
    color:         '#c8d8e8',
    fontFamily:    '"Exo 2", sans-serif',
    fontSize:      12,
    fontWeight:    600,
    padding:       '5px 12px',
    cursor:        'pointer',
    letterSpacing: 1,
    opacity:       0.9,
  },
  stage: {
    position:   'relative',
    flex:       1,
    background: '#000',
    overflow:   'hidden',
  },
  video: {
    position:   'absolute',
    width:      '100%',
    height:     '100%',
    objectFit:  'contain',
    opacity:    0,               // hidden — canvas draws from this element
    pointerEvents: 'none',
  },
  canvas: {
    position: 'absolute',
    inset:    0,
    width:    '100%',
    height:   '100%',
    display:  'block',
  },
  errorOverlay: {
    position:      'absolute',
    inset:         0,
    display:       'flex',
    flexDirection: 'column',
    alignItems:    'center',
    justifyContent:'center',
    background:    'rgba(8,12,18,0.92)',
  },
  summaryOverlay: {
    position:      'absolute',
    inset:         0,
    display:       'flex',
    alignItems:    'center',
    justifyContent:'center',
    background:    'rgba(8,12,18,0.78)',
    backdropFilter:'blur(2px)',
  },
  summaryCard: {
    width:        420,
    maxWidth:     '90vw',
    background:   'rgba(12,18,26,0.95)',
    border:       '1px solid #1f2e44',
    borderRadius: 10,
    padding:      '18px 20px',
    boxShadow:    '0 10px 40px rgba(0,0,0,0.4)',
  },
  summaryTitle: {
    fontFamily:    '"Exo 2", sans-serif',
    fontWeight:    800,
    fontSize:      16,
    letterSpacing: 3,
    color:         '#e8f4ff',
  },
  summarySubtitle: {
    fontFamily:    '"Share Tech Mono", monospace',
    fontSize:      12,
    color:         '#7f9ab3',
    marginTop:     4,
    marginBottom:  12,
  },
  summaryBlock: {
    display:      'flex',
    flexDirection:'column',
    gap:          8,
  },
  summaryRow: {
    display:       'flex',
    justifyContent:'space-between',
    alignItems:    'center',
  },
  summaryLabel: {
    fontFamily: '"Exo 2", sans-serif',
    fontSize:   12,
    color:      '#8ea6bf',
  },
  summaryValue: {
    fontFamily: '"Share Tech Mono", monospace',
    fontSize:   13,
    color:      '#e8f4ff',
  },
  summarySectionTitle: {
    fontFamily: '"Exo 2", sans-serif',
    fontSize:   12,
    fontWeight: 700,
    color:      '#c8d8e8',
    marginTop:  8,
  },
  summaryIssue: {
    fontFamily: '"Share Tech Mono", monospace',
    fontSize:   12,
    color:      '#ff6b6b',
  },
  summaryIssueGood: {
    fontFamily: '"Share Tech Mono", monospace',
    fontSize:   12,
    color:      '#00e87a',
  },
  summaryBadge: (grade) => ({
    fontFamily:    '"Share Tech Mono", monospace',
    fontSize:      16,
    fontWeight:    700,
    letterSpacing: 2,
    color:         grade === 'A' ? '#00e87a' : grade === 'B' ? '#3d9eff' : grade === 'C' ? '#f4c04d' : '#ff6b6b',
  }),
  summaryActions: {
    display:       'flex',
    justifyContent:'flex-end',
    marginTop:     14,
  },
  summaryBtn: {
    background:    '#152131',
    border:        '1px solid #27405a',
    borderRadius:  6,
    color:         '#e8f4ff',
    fontFamily:    '"Exo 2", sans-serif',
    fontSize:      12,
    fontWeight:    700,
    padding:       '6px 14px',
    cursor:        'pointer',
    letterSpacing: 1,
  },
}
