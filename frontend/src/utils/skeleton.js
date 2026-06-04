/**
 * skeleton.js — Canvas drawing utilities
 *
 * Replicates the desktop app's visual language in the browser:
 *   • Bone segments with per-assessment colour coding
 *   • Keypoint dots (white fill, dark outline)
 *   • HUD overlay: motion cue (top) + form class (bottom) + rep counter
 *   • Debug bar: latency / FPS / YOLO ms
 */

// COCO-17 skeleton connectivity (same as ALL_BONES in the Python app)
export const BONES = [
  [0,1],[0,2],[1,3],[2,4],
  [5,6],[5,7],[7,9],[6,8],[8,10],
  [5,11],[6,12],[11,12],
  [11,13],[13,15],[12,14],[14,16],
]

// Severity → CSS colour (matches SEV dict in Python)
export const SEV_COLOR = {
  good:    '#00e87a',
  info:    '#3d9eff',
  warning: '#ffb800',
  error:   '#ff3b3b',
  neutral: '#5a7080',
}

// Form class → colour
export const FORM_COLOR = {
  correct: '#00e87a',
  var1:    '#ff3b3b',
  var2:    '#ff3b3b',
  var3:    '#ff3b3b',
  neutral: '#5a7080',
}

const KP_CONF_THRESH = 0.20

/**
 * Scale keypoints from backend frame coordinates to canvas display coordinates.
 * Backend keypoints are in [frameW × frameH] space.
 * Canvas is displayed at [canvasW × canvasH].
 */
export function scaleKP(kp, frameW, frameH, canvasW, canvasH) {
  if (!kp || kp.length === 0) return []
  const sx = canvasW / frameW
  const sy = canvasH / frameH
  return kp.map(([x, y, c]) => [x * sx, y * sy, c])
}

/**
 * Draw the full skeleton: bones + joints.
 */
export function drawSkeleton(ctx, kp, assessment) {
  if (!kp || kp.length < 17) return

  const fck    = assessment?.form_class_key ?? 'neutral'
  const boneColor = FORM_COLOR[fck] ?? FORM_COLOR.neutral

  // ── Bones ─────────────────────────────────────────────────────────────────
  ctx.lineWidth   = 2.5
  ctx.lineCap     = 'round'

  for (const [a, b] of BONES) {
    if (a >= kp.length || b >= kp.length) continue
    const [ax, ay, ac] = kp[a]
    const [bx, by, bc] = kp[b]
    if (ac < KP_CONF_THRESH || bc < KP_CONF_THRESH) continue
    if (ax === 0 && ay === 0) continue
    if (bx === 0 && by === 0) continue

    ctx.strokeStyle = boneColor
    ctx.globalAlpha = 0.85
    ctx.beginPath()
    ctx.moveTo(ax, ay)
    ctx.lineTo(bx, by)
    ctx.stroke()
  }

  // ── Joints ────────────────────────────────────────────────────────────────
  ctx.globalAlpha = 1.0
  for (let i = 0; i < kp.length; i++) {
    const [x, y, c] = kp[i]
    if (c < KP_CONF_THRESH || (x === 0 && y === 0)) continue

    ctx.fillStyle   = '#ffffff'
    ctx.strokeStyle = '#111'
    ctx.lineWidth   = 1.5
    ctx.beginPath()
    ctx.arc(x, y, 5, 0, Math.PI * 2)
    ctx.fill()
    ctx.stroke()
  }

  ctx.globalAlpha = 1.0
}

/**
 * Draw the two-panel HUD — identical layout to the desktop draw_hud().
 *
 * Panel 1 (top ~70 px):    Motion cue   (coloured by severity)
 * Panel 2 (top 70-120 px): Form class   (CORRECT FORM / INCORRECT FORM)
 */
export function drawHUD(ctx, canvasW, canvasH, assessment, stateInfo) {
  if (!assessment) return

  const {
    motion_cue_text:     cueText  = '',
    motion_cue_severity: cueSev   = 'neutral',
    form_class_key:      fck      = 'neutral',
    form_label:          fLabel   = '',
    primary_angle:       angle    = null,
  } = assessment

  const {
    reps        = 0,
    stage       = 'up',
    is_timed    = false,
    hold_time   = 0,
    grade       = '–',
    correct_reps = 0,
    total_reps   = 0,
    score_pct    = 0,
  } = stateInfo ?? {}

  // ── Panel backgrounds (semi-transparent) ──────────────────────────────────
  const panelH1 = 64
  const panelH2 = 46

  // Panel 1 — motion cue
  const cueCol = SEV_COLOR[cueSev] ?? SEV_COLOR.neutral
  ctx.fillStyle = 'rgba(8,12,18,0.82)'
  ctx.fillRect(0, 0, canvasW, panelH1)
  // left accent bar
  ctx.fillStyle = cueCol
  ctx.fillRect(0, 0, 5, panelH1)

  ctx.fillStyle = cueCol
  ctx.font      = 'bold 15px "Exo 2", sans-serif'
  ctx.textBaseline = 'middle'
  drawTextFit(ctx, cueText.toUpperCase(), 18, panelH1 / 2, canvasW - 150)

  // Angle badge (top-right of panel 1)
  if (angle !== null) {
    const badge = `${Math.round(angle)}°`
    ctx.fillStyle = 'rgba(30,40,55,0.9)'
    roundRect(ctx, canvasW - 90, 12, 80, 40, 6)
    ctx.fill()
    ctx.fillStyle = cueCol
    ctx.font      = '18px "Share Tech Mono", monospace'
    ctx.textAlign = 'center'
    ctx.fillText(badge, canvasW - 50, panelH1 / 2)
    ctx.textAlign = 'left'
  }

  // ── Panel 2 — form class ───────────────────────────────────────────────────
  const y2 = panelH1
  const formCol = FORM_COLOR[fck] ?? FORM_COLOR.neutral
  ctx.fillStyle = 'rgba(8,12,18,0.82)'
  ctx.fillRect(0, y2, canvasW, panelH2)
  ctx.fillStyle = formCol
  ctx.fillRect(0, y2, 5, panelH2)

  ctx.fillStyle = formCol
  ctx.font      = 'bold 17px "Exo 2", sans-serif'
  ctx.textBaseline = 'middle'
  ctx.fillText(fLabel.toUpperCase(), 18, y2 + panelH2 / 2)

  // ── Rep / hold counter (bottom-left corner) ────────────────────────────────
  const counterH = 64
  const counterY = canvasH - 24 - counterH - 12
  ctx.fillStyle = 'rgba(8,12,18,0.82)'
  roundRect(ctx, 12, counterY, 210, counterH, 10)
  ctx.fill()

  if (is_timed) {
    const mm = Math.floor(hold_time / 60).toString().padStart(2, '0')
    const ss = Math.floor(hold_time % 60).toString().padStart(2, '0')
    ctx.fillStyle = '#c8d8e8'
    ctx.font      = '13px "Exo 2", sans-serif'
    ctx.textBaseline = 'top'
    ctx.fillText('GOOD HOLD', 22, counterY + 10)
    ctx.fillStyle = SEV_COLOR.good
    ctx.font      = '34px "Share Tech Mono", monospace'
    ctx.fillText(`${mm}:${ss}`, 22, counterY + 26)
  } else {
    ctx.fillStyle = '#c8d8e8'
    ctx.font      = '13px "Exo 2", sans-serif'
    ctx.textBaseline = 'top'
    ctx.fillText(`REPS  •  STAGE: ${stage.toUpperCase()}`, 22, counterY + 10)
    ctx.fillStyle = '#e8f4ff'
    ctx.font      = '34px "Share Tech Mono", monospace'
    ctx.fillText(String(reps).padStart(2, '0'), 22, counterY + 26)

    if (total_reps > 0) {
      ctx.fillStyle = score_pct >= 85 ? SEV_COLOR.good : score_pct >= 55 ? SEV_COLOR.warning : SEV_COLOR.error
      ctx.font      = '12px "Share Tech Mono", monospace'
      ctx.textBaseline = 'top'
      ctx.fillText(`${correct_reps}/${total_reps} CORRECT  ${Math.round(score_pct)}%  ${grade}`, 78, counterY + 36)
    }
  }

  ctx.textBaseline = 'alphabetic'
}

/**
 * Debug bar at the very bottom: latency, server FPS, YOLO ms.
 */
export function drawDebugBar(ctx, canvasW, canvasH, { latency, serverFps, timingMs, connected }) {
  const h = 24
  const y = canvasH - h
  ctx.fillStyle = 'rgba(0,0,0,0.7)'
  ctx.fillRect(0, y, canvasW, h)

  ctx.font      = '11px "Share Tech Mono", monospace'
  ctx.textBaseline = 'middle'
  const cy = y + h / 2

  const dot   = connected ? '●' : '○'
  const dotC  = connected ? SEV_COLOR.good : SEV_COLOR.error
  ctx.fillStyle = dotC
  ctx.fillText(dot, 8, cy)

  ctx.fillStyle = '#4a6a80'
  const parts = [
    `RTT ${latency}ms`,
    `SRV ${serverFps}fps`,
    timingMs ? `SRV ${timingMs.logic}ms` : '',
  ].filter(Boolean).join('   ')
  ctx.fillText(parts, 22, cy)

  ctx.textBaseline = 'alphabetic'
}

// ── Helpers ──────────────────────────────────────────────────────────────────
function drawTextFit(ctx, text, x, y, maxW) {
  const measured = ctx.measureText(text).width
  if (measured <= maxW) {
    ctx.fillText(text, x, y)
  } else {
    // Shrink font until it fits
    const origFont = ctx.font
    let size = 15
    while (size > 9) {
      size--
      ctx.font = `bold ${size}px "Exo 2", sans-serif`
      if (ctx.measureText(text).width <= maxW) break
    }
    ctx.fillText(text, x, y)
    ctx.font = origFont
  }
}

function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath()
  ctx.moveTo(x + r, y)
  ctx.lineTo(x + w - r, y)
  ctx.quadraticCurveTo(x + w, y, x + w, y + r)
  ctx.lineTo(x + w, y + h - r)
  ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h)
  ctx.lineTo(x + r, y + h)
  ctx.quadraticCurveTo(x, y + h, x, y + h - r)
  ctx.lineTo(x, y + r)
  ctx.quadraticCurveTo(x, y, x + r, y)
  ctx.closePath()
}
