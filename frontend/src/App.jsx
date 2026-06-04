import { useState } from 'react'
import ExerciseView from './components/ExerciseView'

const EXERCISES = [
  {
    key:   'pushups',
    label: 'Push-ups',
    icon:  '💪',
    desc:  'Hip sagging · Hip piking · Elbow depth',
  },
  {
    key:   'squats',
    label: 'Squats',
    icon:  '🦵',
    desc:  'Knee depth · Torso lean',
  },
  {
    key:   'lunges',
    label: 'Lunges',
    icon:  '🏃',
    desc:  'Knee overextension · Wide step · Torso lean',
  },
  {
    key:   'planks',
    label: 'Planks',
    icon:  '⚡',
    desc:  'Hip sagging · Hip elevation',
  },
]

export default function App() {
  const [active, setActive] = useState(null)

  if (active) {
    return <ExerciseView exercise={active} onBack={() => setActive(null)} />
  }

  return (
    <div style={s.root}>
      <header style={s.header}>
        <div style={s.logo}>
          <span style={s.logoMark}>AI</span>
          <span style={s.logoText}>GYM</span>
        </div>
        <p style={s.tagline}>REAL-TIME FORM ANALYSIS · POSE DETECTION · LIVE COACHING</p>
      </header>

      <main style={s.main}>
        <p style={s.select}>SELECT EXERCISE</p>

        <div style={s.grid}>
          {EXERCISES.map((ex) => (
            <button
              key={ex.key}
              style={s.card}
              onClick={() => setActive(ex.key)}
              onMouseEnter={e => Object.assign(e.currentTarget.style, s.cardHover)}
              onMouseLeave={e => Object.assign(e.currentTarget.style, s.card)}
            >
              <span style={s.icon}>{ex.icon}</span>
              <span style={s.cardLabel}>{ex.label.toUpperCase()}</span>
              <span style={s.cardDesc}>{ex.desc}</span>
              <span style={s.cardArrow}>→</span>
            </button>
          ))}
        </div>
      </main>

      <footer style={s.footer}>
        <span style={s.footNote}>POWERED BY MOVENET · FASTAPI · REACT</span>
      </footer>
    </div>
  )
}

/* ── Styles ─────────────────────────────────────────────────────────────── */
const s = {
  root: {
    minHeight:      '100vh',
    display:        'flex',
    flexDirection:  'column',
    alignItems:     'center',
    background:     '#080c12',
    fontFamily:     '"Exo 2", sans-serif',
  },
  header: {
    marginTop:    60,
    textAlign:    'center',
  },
  logo: {
    display:     'flex',
    alignItems:  'baseline',
    gap:         6,
    justifyContent: 'center',
    marginBottom: 12,
  },
  logoMark: {
    fontSize:      64,
    fontWeight:    900,
    color:         '#00e87a',
    letterSpacing: -2,
    lineHeight:    1,
    textShadow:    '0 0 40px rgba(0,232,122,0.4)',
  },
  logoText: {
    fontSize:    64,
    fontWeight:  300,
    color:       '#c8d8e8',
    letterSpacing: 8,
  },
  tagline: {
    fontSize:    11,
    fontFamily:  '"Share Tech Mono", monospace',
    letterSpacing: 3,
    color:       '#2a4a60',
    marginTop:   8,
  },
  main: {
    flex:    1,
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    marginTop: 60,
    width: '100%',
    maxWidth: 720,
    padding: '0 24px',
  },
  select: {
    fontFamily:  '"Share Tech Mono", monospace',
    fontSize:    11,
    letterSpacing: 4,
    color:       '#2a5a7a',
    marginBottom: 28,
  },
  grid: {
    display:             'grid',
    gridTemplateColumns: 'repeat(2, 1fr)',
    gap:                 16,
    width:               '100%',
  },
  card: {
    display:        'flex',
    flexDirection:  'column',
    alignItems:     'flex-start',
    gap:            6,
    padding:        '28px 28px 24px',
    background:     '#0d1420',
    border:         '1px solid #1e2d42',
    borderRadius:   12,
    cursor:         'pointer',
    textAlign:      'left',
    position:       'relative',
    transition:     'border-color 0.2s, background 0.2s',
  },
  cardHover: {
    display:        'flex',
    flexDirection:  'column',
    alignItems:     'flex-start',
    gap:            6,
    padding:        '28px 28px 24px',
    background:     '#111825',
    border:         '1px solid #00e87a',
    borderRadius:   12,
    cursor:         'pointer',
    textAlign:      'left',
    position:       'relative',
    transition:     'border-color 0.2s, background 0.2s',
  },
  icon: {
    fontSize:    32,
    marginBottom: 4,
  },
  cardLabel: {
    fontWeight:  700,
    fontSize:    18,
    letterSpacing: 2,
    color:       '#e8f4ff',
  },
  cardDesc: {
    fontSize:   12,
    color:      '#3d6080',
    fontFamily: '"Share Tech Mono", monospace',
  },
  cardArrow: {
    position:  'absolute',
    right:     20,
    top:       '50%',
    transform: 'translateY(-50%)',
    color:     '#1e3a5f',
    fontSize:  20,
  },
  footer: {
    padding:     '24px 0 32px',
    textAlign:   'center',
  },
  footNote: {
    fontFamily:  '"Share Tech Mono", monospace',
    fontSize:    10,
    letterSpacing: 2,
    color:       '#1a2a38',
  },
}
