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
import { usePoseDetector } from '../hooks/usePoseDetector'
import {
  drawSkeleton, drawHUD, drawDebugBar,
  scaleKP,
} from '../utils/skeleton'

const FORM_CORRECTIONS = {
  pushups: {
    var1: 'Hip sagging. Engage your core and keep your body straight.',
    var2: 'Hip piking. Lower your hips.',
    var3: 'Insufficient elbow flexion. Go lower next rep.',
  },
  squats: {
    var1: 'Insufficient depth. Go deeper.',
    var2: 'Excessive forward lean. Keep your chest up.',
  },
  lunges: {
    var1: 'Front knee overextension. Take a wider step.',
    var2: 'Wide step. Move your front foot back.',
    var3: 'Torso forward lean. Stand more upright.',
  },
  planks: {
    var1: 'Hip sagging. Tighten your core.',
    var2: 'Hip elevation. Lower your hips.',
  },
}

// ── Debug helper — mirrors backend angle_at_b exactly ────────────────────────
// Enable in browser console: window.__DEBUG_ANGLES = true
function angleBetween(a, b, c) {
  const ba  = [a[0] - b[0], a[1] - b[1]]
  const bc  = [c[0] - b[0], c[1] - b[1]]
  const dot = ba[0] * bc[0] + ba[1] * bc[1]
  const len = Math.hypot(ba[0], ba[1]) * Math.hypot(bc[0], bc[1])
  if (len < 1e-6) return 180
  return Math.round(Math.acos(Math.max(-1, Math.min(1, dot / len))) * 180 / Math.PI)
}

export default function ExerciseView({ exercise, onBack }) {
  const videoRef        = useRef(null)
  const canvasRef       = useRef(null)
  const streamRef       = useRef(null)

  // In-browser pose detector (TF.js MoveNet) — eliminates VPS YOLO round-trip
  const { detectPose, isReady: modelReady, rejectionCount, resetRejectionCount } = usePoseDetector()

  // Local keypoints from MoveNet — updated every frame before the WS round-trip
  const localKPRef      = useRef(null)

  // Latest assessment+state from backend (updated at WS response rate ~15-30fps)
  const poseDataRef     = useRef(null)
  const rafRef          = useRef(null)
  const debugRef        = useRef({ latency: 0, serverFps: 0, timingMs: null, connected: false })
  const prevRepsRef     = useRef(0)
  const speechTimeoutRef = useRef(null)

  const [camReady,      setCamReady]      = useState(false)
  const [camError,      setCamError]      = useState(null)
  const [countdown,     setCountdown]     = useState(7)
  const [started,       setStarted]       = useState(false)
  const [finished,      setFinished]      = useState(false)
  const [summary,       setSummary]       = useState(null)
  const [testCondition, setTestCondition] = useState(null)   // null | 'correct' | 'incorrect'
  const [testSelected,  setTestSelected]  = useState(false)

  // Lunge-specific: dominant foot selection ("L" | "R" | null=not yet chosen)
  const [dominantFoot,  setDominantFoot]  = useState(exercise === 'lunges' ? null : 'L')
  const [footSelected,  setFootSelected]  = useState(exercise !== 'lunges')

  // Ref so the RAF loop can always read the current dominant foot without
  // restarting the animation loop when the user switches sides.
  const dominantFootRef = useRef(dominantFoot)
  useEffect(() => { dominantFootRef.current = dominantFoot }, [dominantFoot])

  // ── Start webcam ──────────────────────────────────────────────────────────
  useEffect(() => {
    let mounted = true
    navigator.mediaDevices
      .getUserMedia({ video: { width: 1280, height: 720, frameRate: 30, facingMode: 'user' } })
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
      if (speechTimeoutRef.current) clearTimeout(speechTimeoutRef.current)
      window.speechSynthesis?.cancel()
    }
  }, [])

  // ── Countdown then start (waits for foot + test condition selection) ─────
  useEffect(() => {
    if (!camReady || !footSelected || !testSelected) return
    prevRepsRef.current = 0
    let n = 7
    setCountdown(n)
    const id = setInterval(() => {
      n--
      setCountdown(n)
      if (n <= 0) { clearInterval(id); setStarted(true) }
    }, 1000)
    return () => clearInterval(id)
  }, [camReady, footSelected, testSelected])

  const speak = useCallback((text) => {
    if (!window.speechSynthesis) return
    const u = new SpeechSynthesisUtterance(text)
    u.rate = 1.1
    window.speechSynthesis.speak(u)
  }, [])

  // ── Receive keypoints from backend ────────────────────────────────────────
  const handleData = useCallback((data) => {
    poseDataRef.current = data

    const st = data?.state
    if (!st) return

    const currentReps = st.reps ?? 0
    if (currentReps > prevRepsRef.current) {
      prevRepsRef.current = currentReps
      const cls = st.last_rep_cls

      if (speechTimeoutRef.current) {
        clearTimeout(speechTimeoutRef.current)
        speechTimeoutRef.current = null
      }
      window.speechSynthesis?.cancel()

      if (cls === 'correct') {
        speak('Correct')
      } else if (cls && cls !== 'neutral') {
        speak('Wrong')
        const correction = FORM_CORRECTIONS[exercise]?.[cls]
        if (correction) {
          speechTimeoutRef.current = setTimeout(() => speak(correction), 3000)
        }
      }
    }
  }, [exercise, speak])

  const { connected, latency, serverFps, timingMs } = usePoseSocket({
    exercise,
    dominantFoot: dominantFoot ?? 'L',
    videoRef,
    detectPose,
    onData:     handleData,
    onLocalKP:  (kp) => { localKPRef.current = kp },
    enabled:    started && !finished,
  })

  // Keep debug values in a ref so the RAF loop doesn't restart on every WS frame
  debugRef.current = { latency, serverFps, timingMs, connected }

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
    const ctx    = canvas.getContext('2d')

    const render = () => {
      rafRef.current = requestAnimationFrame(render)

      const cw = canvas.width
      const ch = canvas.height

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

        // Positioning hint below the countdown number
        ctx.font      = 'bold 18px "Exo 2", sans-serif'
        ctx.fillStyle = '#00e87a'
        if (['pushups', 'squats', 'planks'].includes(exercise)) {
          ctx.fillText('FACE SIDEWAYS TO THE CAMERA', cw / 2, ch / 2 + 90)
        } else if (exercise === 'lunges') {
          const foot = dominantFootRef.current ?? 'L'
          const hint = foot === 'L'
            ? 'FACE LEFT · STEP FORWARD WITH LEFT FOOT'
            : 'FACE RIGHT · STEP FORWARD WITH RIGHT FOOT'
          ctx.fillText(hint, cw / 2, ch / 2 + 90)
        }

        ctx.textAlign    = 'left'
        ctx.textBaseline = 'alphabetic'
        return
      }

      // ── Skeleton + HUD ────────────────────────────────────────────────────
      // Skeleton uses LOCAL keypoints from MoveNet (available every frame, ~60fps).
      // HUD uses assessment+state from the backend WS response (~15-30fps).
      const data    = poseDataRef.current
      const localKP = localKPRef.current
      // `video` is already in scope from the useEffect closure (line 198)
      if (!finished && data && !data.error) {
        const { assessment, state: stateInfo } = data

        if (localKP && video && video.videoWidth > 0) {
          // Scale local keypoints from video pixel coords → canvas display coords
          const scaledKP = scaleKP(localKP, video.videoWidth, video.videoHeight, cw, ch)
          drawSkeleton(ctx, scaledKP, assessment)
        }

        drawHUD(ctx, cw, ch, assessment, stateInfo)
      } else if (started && !finished) {
        // Model or backend not ready yet
        ctx.fillStyle = 'rgba(8,12,18,0.55)'
        ctx.fillRect(0, 0, cw, 64)
        ctx.fillStyle = '#5a7080'
        ctx.font      = 'bold 14px "Exo 2", sans-serif'
        ctx.textBaseline = 'middle'
        ctx.fillText(modelReady ? 'WAITING FOR SIGNAL…' : 'LOADING AI MODEL… PLEASE WAIT', 18, 32)
        ctx.textBaseline = 'alphabetic'
      }

      // ── Debug bar (always visible) ────────────────────────────────────────
      drawDebugBar(ctx, cw, ch, debugRef.current)

      // ── Lunge angle debug overlay ─────────────────────────────────────────
      // Enable: type  window.__DEBUG_ANGLES = true  in the browser console.
      // Disable: window.__DEBUG_ANGLES = false  (or just close the console).
      if (window.__DEBUG_ANGLES && exercise === 'lunges') {
        const kp = localKPRef.current
        if (kp && kp.length === 17 && video && video.videoWidth > 0) {
          const foot = dominantFootRef.current ?? 'L'
          const [fh, fk, fa] = foot === 'L' ? [11, 13, 15] : [12, 14, 16]
          const [rh, rk, ra] = foot === 'L' ? [12, 14, 16] : [11, 13, 15]

          ctx.font         = 'bold 16px "Share Tech Mono", monospace'
          ctx.textBaseline = 'alphabetic'

          if (kp[fh][2] > 0.25 && kp[fk][2] > 0.25 && kp[fa][2] > 0.25) {
            ctx.fillStyle = '#00ff00'
            ctx.fillText(`Front: ${angleBetween(kp[fh], kp[fk], kp[fa])}°`, 20, 40)
          }
          if (kp[rh][2] > 0.25 && kp[rk][2] > 0.25 && kp[ra][2] > 0.25) {
            ctx.fillStyle = '#ffff00'
            ctx.fillText(`Rear:  ${angleBetween(kp[rh], kp[rk], kp[ra])}°`, 20, 70)
          }
        }
      }
    }

    rafRef.current = requestAnimationFrame(render)
    return () => { if (rafRef.current) cancelAnimationFrame(rafRef.current) }
  }, [camReady, started, countdown])

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

        {/* ── Lunge foot selection (lunges only, before test condition) ─── */}
        {camReady && !footSelected && !finished && exercise === 'lunges' && (
          <div style={styles.testOverlay}>
            <div style={styles.testCard}>
              <div style={styles.testTitle}>CHOOSE DOMINANT FOOT</div>
              <div style={styles.testSubtitle}>Which foot will step forward?</div>
              <div style={styles.testOptions}>
                <button
                  style={styles.testOption(dominantFoot === 'L')}
                  onClick={() => setDominantFoot('L')}
                >
                  <span style={styles.testOptionIcon}>←</span>
                  <span>Left Foot</span>
                  <span style={styles.testOptionHint}>Face LEFT — left foot steps forward</span>
                </button>
                <button
                  style={styles.testOption(dominantFoot === 'R')}
                  onClick={() => setDominantFoot('R')}
                >
                  <span style={styles.testOptionIcon}>→</span>
                  <span>Right Foot</span>
                  <span style={styles.testOptionHint}>Face RIGHT — right foot steps forward</span>
                </button>
              </div>
              <button
                style={{
                  ...styles.testStartBtn,
                  opacity: dominantFoot ? 1 : 0.4,
                  cursor: dominantFoot ? 'pointer' : 'not-allowed',
                }}
                disabled={!dominantFoot}
                onClick={() => setFootSelected(true)}
              >
                CONFIRM →
              </button>
            </div>
          </div>
        )}

        {/* Test condition selector — shown after foot selected, before countdown */}
        {camReady && footSelected && !testSelected && !finished && (
          <div style={styles.testOverlay}>
            <div style={styles.testCard}>
              <div style={styles.testTitle}>SELECT TEST CONDITION</div>
              {exercise === 'lunges' && (
                <div style={styles.lungeFacingHint}>
                  {dominantFoot === 'L'
                    ? '← Face LEFT · left foot steps forward'
                    : '→ Face RIGHT · right foot steps forward'}
                </div>
              )}
              <div style={styles.testSubtitle}>What form will you perform this session?</div>
              <div style={styles.testOptions}>
                <button
                  style={styles.testOption(testCondition === 'correct')}
                  onClick={() => setTestCondition('correct')}
                >
                  <span style={styles.testOptionIcon}>✓</span>
                  <span>Correct Form</span>
                  <span style={styles.testOptionHint}>Testing H1 correct detection</span>
                </button>
                <button
                  style={styles.testOption(testCondition === 'incorrect')}
                  onClick={() => setTestCondition('incorrect')}
                >
                  <span style={styles.testOptionIcon}>✗</span>
                  <span>Incorrect Form</span>
                  <span style={styles.testOptionHint}>Testing H1 error detection</span>
                </button>
                <button
                  style={styles.testOption(testCondition === null)}
                  onClick={() => setTestCondition(null)}
                >
                  <span style={styles.testOptionIcon}>○</span>
                  <span>Free Training</span>
                  <span style={styles.testOptionHint}>No accuracy test</span>
                </button>
              </div>
              <button
                style={styles.testStartBtn}
                onClick={() => setTestSelected(true)}
              >
                START SESSION →
              </button>
            </div>
          </div>
        )}

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
                  {testCondition !== null && (() => {
                    const total = summary.total_reps || 0
                    const breakdown = summary.rep_breakdown ?? {}
                    const matching = testCondition === 'correct'
                      ? (breakdown.correct ?? 0)
                      : (breakdown.var1 ?? 0) + (breakdown.var2 ?? 0) + (breakdown.var3 ?? 0)
                    const acc = total > 0 ? (matching / total) * 100 : 0
                    return (
                      <div style={styles.h1Bar}>
                        <div style={styles.h1BarLabel}>
                          <span>H1 CLASSIFICATION ACCURACY</span>
                          <span style={styles.h1Badge(acc >= 80)}>
                            {acc >= 80 ? 'H1 PASSED ✓' : 'H1 NOT MET'}
                          </span>
                        </div>
                        <div style={styles.h1BarCondition}>
                          {testCondition === 'correct' ? 'Condition: Correct Form' : 'Condition: Incorrect Form'}
                          {' — '}{matching}/{total} correctly classified
                        </div>
                        <div style={styles.h1Track}>
                          <div style={styles.h1Fill(acc)} />
                          <div style={styles.h1Threshold} title="80% H1 target" />
                        </div>
                        <div style={styles.h1Pct}>{Math.round(acc)}%</div>
                      </div>
                    )
                  })()}
                  <div style={styles.summarySectionTitle}>Breakdown</div>
                  <div style={styles.summaryRow}>
                    <span style={styles.summaryLabel}>Correct</span>
                    <span style={styles.summaryValue}>{summary.rep_breakdown?.correct ?? 0}</span>
                  </div>
                  {Object.entries(summary.var_names ?? {}).map(([varNum, varLabel]) => (
                    <div key={`var${varNum}`} style={styles.summaryRow}>
                      <span style={styles.summaryLabel}>{varLabel}</span>
                      <span style={styles.summaryValue}>{summary.rep_breakdown?.[`var${varNum}`] ?? 0}</span>
                    </div>
                  ))}
                </div>
              )}

              <div style={styles.summaryActions}>
                {exercise === 'lunges' && (
                  <button
                    style={styles.summaryBtnContinue}
                    onClick={() => {
                      const otherFoot = dominantFoot === 'L' ? 'R' : 'L'
                      setDominantFoot(otherFoot)
                      setFootSelected(true)   // skip foot selection, go straight to test condition
                      setSummary(null)
                      setFinished(false)
                      setStarted(false)
                      setTestSelected(false)
                      setTestCondition(null)
                      prevRepsRef.current = 0
                    }}
                  >
                    CONTINUE → {dominantFoot === 'L' ? 'RIGHT' : 'LEFT'} FOOT
                  </button>
                )}
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
    width:        600,             // was 420 → bigger box
    maxWidth:     '95vw',          // ensure it doesn't overflow on smaller screens
    background:   'rgba(12,18,26,0.95)',
    border:       '1px solid #1f2e44',
    borderRadius: 12,              // slightly rounder
    padding:      '24px 28px',     // more breathing room
    boxShadow:    '0 10px 40px rgba(0,0,0,0.4)',
  },
  summaryTitle: {
    fontFamily:    '"Exo 2", sans-serif',
    fontWeight:    800,
    fontSize:      30,
    letterSpacing: 3,
    color:         '#e8f4ff',
  },
  summarySubtitle: {
    fontFamily:    '"Share Tech Mono", monospace',
    fontSize:      25,
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
    fontSize:   25,
    color:      '#8ea6bf',
  },
  summaryValue: {
    fontFamily: '"Share Tech Mono", monospace',
    fontSize:   30,
    color:      '#e8f4ff',
  },
  summarySectionTitle: {
    fontFamily: '"Exo 2", sans-serif',
    fontSize:   25,
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
  testOverlay: {
    position:       'absolute',
    inset:          0,
    display:        'flex',
    alignItems:     'center',
    justifyContent: 'center',
    background:     'rgba(8,12,18,0.92)',
    zIndex:         20,
  },
  testCard: {
    background:   '#0d1420',
    border:       '1px solid #1e2d42',
    borderRadius: 14,
    padding:      '32px 28px',
    width:        320,
    display:      'flex',
    flexDirection:'column',
    gap:          14,
  },
  testTitle: {
    fontFamily:    '"Share Tech Mono", monospace',
    fontSize:      13,
    letterSpacing: 3,
    color:         '#00e87a',
    textAlign:     'center',
  },
  testSubtitle: {
    fontFamily: '"Exo 2", sans-serif',
    fontSize:   12,
    color:      '#3d6080',
    textAlign:  'center',
  },
  testOptions: {
    display:       'flex',
    flexDirection: 'column',
    gap:           8,
  },
  testOption: (selected) => ({
    display:       'flex',
    alignItems:    'center',
    gap:           10,
    padding:       '10px 14px',
    background:    selected ? '#0a2a1a' : '#0a1422',
    border:        `1px solid ${selected ? '#00e87a' : '#1e2d42'}`,
    borderRadius:  8,
    cursor:        'pointer',
    color:         selected ? '#00e87a' : '#c8d8e8',
    fontFamily:    '"Exo 2", sans-serif',
    fontSize:      13,
    fontWeight:    selected ? 700 : 400,
    textAlign:     'left',
    flexDirection: 'column',
    alignItems:    'flex-start',
    transition:    'border-color 0.15s',
  }),
  testOptionIcon: {
    fontFamily: '"Share Tech Mono", monospace',
    fontSize:   16,
  },
  testOptionHint: {
    fontFamily: '"Share Tech Mono", monospace',
    fontSize:   10,
    color:      '#2a4a60',
    marginTop:  2,
  },
  lungeFacingHint: {
    fontFamily:    '"Share Tech Mono", monospace',
    fontSize:      11,
    letterSpacing: 1,
    color:         '#00e87a',
    textAlign:     'center',
    background:    '#001a0d',
    border:        '1px solid #00e87a33',
    borderRadius:  6,
    padding:       '6px 10px',
  },
  testStartBtn: {
    background:    '#00e87a',
    border:        'none',
    borderRadius:  8,
    color:         '#080c12',
    fontFamily:    '"Exo 2", sans-serif',
    fontSize:      13,
    fontWeight:    700,
    letterSpacing: 1,
    padding:       '11px 0',
    cursor:        'pointer',
    marginTop:     4,
  },
  h1BarCondition: {
    fontFamily: '"Share Tech Mono", monospace',
    fontSize:   10,
    color:      '#2a5a7a',
    marginBottom: 4,
  },
  h1Bar: {
    margin:        '10px 0 4px',
    padding:       '8px 10px',
    background:    '#0d1a26',
    borderRadius:  6,
    border:        '1px solid #1e2d42',
  },
  h1BarLabel: {
    display:        'flex',
    justifyContent: 'space-between',
    alignItems:     'center',
    fontFamily:     '"Share Tech Mono", monospace',
    fontSize:       10,
    letterSpacing:  2,
    color:          '#3d6080',
    marginBottom:   6,
  },
  h1Badge: (passed) => ({
    fontFamily:    '"Share Tech Mono", monospace',
    fontSize:      10,
    fontWeight:    700,
    letterSpacing: 1,
    color:         passed ? '#00e87a' : '#ff6b6b',
  }),
  h1Track: {
    position:     'relative',
    height:       6,
    background:   '#1e2d42',
    borderRadius: 3,
    overflow:     'visible',
  },
  h1Fill: (pct) => ({
    position:     'absolute',
    left:         0,
    top:          0,
    height:       '100%',
    width:        `${Math.min(pct, 100)}%`,
    background:   pct >= 80 ? '#00e87a' : pct >= 60 ? '#f4c04d' : '#ff6b6b',
    borderRadius: 3,
    transition:   'width 0.6s ease',
  }),
  h1Threshold: {
    position:     'absolute',
    left:         '80%',
    top:          -3,
    width:        2,
    height:       12,
    background:   '#ffffff44',
    borderRadius: 1,
  },
  h1Pct: {
    fontFamily:  '"Share Tech Mono", monospace',
    fontSize:    18,
    fontWeight:  700,
    color:       '#e8f4ff',
    textAlign:   'right',
    marginTop:   4,
  },
  summaryBadge: (grade) => ({
    fontFamily:    '"Share Tech Mono", monospace',
    fontSize:      25,
    fontWeight:    700,
    letterSpacing: 2,
    color:         grade === 'A' ? '#00e87a' : grade === 'B' ? '#3d9eff' : grade === 'C' ? '#f4c04d' : '#ff6b6b',
  }),
  summaryActions: {
    display:       'flex',
    justifyContent:'flex-end',
    gap:           8,
    marginTop:     14,
  },
  summaryBtnContinue: {
    background:    '#003d20',
    border:        '1px solid #00e87a',
    borderRadius:  6,
    color:         '#00e87a',
    fontFamily:    '"Exo 2", sans-serif',
    fontSize:      12,
    fontWeight:    700,
    padding:       '6px 14px',
    cursor:        'pointer',
    letterSpacing: 1,
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
