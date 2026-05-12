"""
AIGym Pose Detection v10
========================
Major refactor: Causally-Linked HUD Architecture

=== DESIGN SUMMARY (10 bullets) ===
  1. Shared per-frame Assessment dict returned by each checker:
       primary_angle, motion_cue, dominant_tag, form_class_key,
       form_label, evidence_tag, evidence_strength.
  2. EvidenceGate replaces the old (cls_key, var_label) Stabilizer.
     Only a sustained dominant_tag (>= CONFIRM=6 frames) confirms a var class.
     Only >= CLEAR=12 consecutive no-error frames reset to 'correct'.
     In between: 'neutral' — no contradiction is possible.
  3. Anti-contradiction neutral window: when a confirmed-var sees clearing evidence,
     bottom shows 'neutral' for CLEAR/2 frames before switching to 'correct'.
     When confirmed-correct sees a new error tag, bottom shows 'neutral' briefly
     before confirming the var.  Top and bottom are never in direct conflict.
  4. Each checker determines ONE dominant_tag per frame (priority-ordered).
     The motion_cue (top header) always describes the same issue as dominant_tag.
     Causal link: top cue evidence -> dominant_tag -> EvidenceGate -> bottom class.
  5. Per-checker Stabilizers (state["st"]) and the global cls_stab are removed.
     AngleSmoothers (state["sm"]) are kept for angle-value stability.
  6. Rep voting (down_cls_buf) uses EvidenceGate output — so the rep log reflects
     the same causal evidence chain as the live HUD.
  7. Plank hold tracking and form_issues tracking use EvidenceGate output.
  8. Push-up var3 (Insufficient Elbow Flexion) is rep-completion only
     (in update_reps via insuf_depth_threshold=105); EvidenceGate covers var1/var2.
  9. Lunge internal debounce (lunge_down_frames etc.) is kept to pre-filter noise
     before passing dominant_tag to EvidenceGate.
 10. draw_hud() takes the unified assessment dict — both headers read from the
     same object, making coherence structurally enforced.

=== TOP CUE TAG -> BOTTOM CLASS MAPPING ===
  Exercise  | Tag           | Bottom Class | Label
  ----------|---------------|--------------|-------------------------------
  Push-ups  | "sag"         | var1         | Hip Sagging
  Push-ups  | "pike"        | var2         | Hip Piking
  Push-ups  | None (good)   | correct      | Correct Form
  ----------|---------------|--------------|-------------------------------
  Squats    | "lean"        | var2         | Excessive Forward Lean
  Squats    | "depth"       | var1         | Insufficient Depth
  Squats    | None (good)   | correct      | Correct Form
  ----------|---------------|--------------|-------------------------------
  Lunges    | "knee_over"   | var1         | Front Knee Overextension
  Lunges    | "rear_shallow"| var2         | Insufficient Rear Knee Depth
  Lunges    | "torso_lean"  | var3         | Torso Forward Lean
  Lunges    | None (good)   | correct      | Correct Form
  ----------|---------------|--------------|-------------------------------
  Planks    | "hip_sag"     | var1         | Hip Sagging
  Planks    | "hip_high"    | var2         | Hip Elevation
  Planks    | "head_drop"   | var3         | Head Dropping
  Planks    | None (good)   | correct      | Correct Form

=== CONTRADICTIONS FIXED (2 examples) ===
  BEFORE (push-up hip sag fixed):
    Frame 80:  dev=22 -> top "Engage core - hips sagging!" (error)
               Stabilizer lock still holds "CORRECT FORM" for 1.5s -> CONTRADICTION
    Frame 82:  top still red / error; bottom still green "CORRECT FORM"  <- BUG
  AFTER:
    Frame 80:  dominant_tag="sag"; EvidenceGate err_count["sag"]=1 -> cls=neutral
    Frame 86:  err_count["sag"]=6 >= CONFIRM -> cls=var1 "Hip Sagging"
               Top and bottom both show sag evidence. No contradiction.

  BEFORE (squat standing up after depth warning):
    Frame 120: ka=155 (standing) -> top "Standing - begin descent" (info)
               Old Stabilizer still showing "Insufficient Depth" (var1) <- CONTRADICTION
  AFTER:
    Frame 120: dominant_tag=None; EvidenceGate good_count starts accumulating
    Frame 132: good_count=12 >= CLEAR -> cls=correct
               During frames 120-131: bottom shows "neutral/Transitioning"
               Never shows top-info + bottom-var simultaneously.

=== MANUAL WEBCAM TEST CHECKLIST ===
  Push-ups:
    [P1] Drop hips -> top turns red "hips sagging" -> bottom confirms "Hip Sagging" ~0.2s
    [P2] Raise hips high -> top red "raised too high" -> bottom "Hip Piking" ~0.2s
    [P3] Fix form -> both headers transition through neutral -> correct (no contradiction)
    [P4] Do quarter-reps (elbow ~130d) -> score screen shows Insufficient Elbow Flexion
  Squats:
    [S1] Lean >45d -> top "lift chest!" -> bottom "Excessive Forward Lean" ~0.2s
    [S2] Stand upright -> top "begin descent" -> bottom must NOT show var (goes neutral then correct)
    [S3] Stay shallow (>95d in down phase) -> top "go deeper" -> bottom "Insufficient Depth"
    [S4] Reach 65-85d -> top "Good depth" -> bottom "Correct Form" after gate clears
  Lunges:
    [L1] Short step -> top "step wider" -> bottom "Front Knee Overextension"
    [L2] Rear knee stays high -> top "lower rear knee" -> bottom "Insuff. Rear Knee Depth"
    [L3] Lean forward >42d -> top "stand upright" -> bottom "Torso Forward Lean"
    [L4] Good lunge -> top "Good depth" + bottom "Correct Form"
    [L5] Confirm both left and right leg work identically
  Planks:
    [K1] Sag hips -> top "hips sagging" -> bottom "Hip Sagging"
    [K2] Raise hips -> top "raised too high" -> bottom "Hip Elevation"
    [K3] Drop chin -> top "chin dropping" -> bottom "Head Dropping"
    [K4] Hold good form 60s -> score screen grade A
    [K5] Confirm form_issues section lists only observed errors

Usage:
  python aigym_pose_detection_v10.py --exercise squats
  python aigym_pose_detection_v10.py --source video.mp4 --exercise lunges --save
  python aigym_pose_detection_v10.py --exercise pushups
  python aigym_pose_detection_v10.py --exercise planks
"""

import argparse
import collections
import math
import sys
import time
from collections import Counter

import cv2
import numpy as np
from ultralytics import YOLO

# ──────────────────────────────────────────────────────────────────────────────
# Tuneable parameters
# ──────────────────────────────────────────────────────────────────────────────
SMOOTH_WINDOW    = 4
CONFIRM_FRAMES   = 6    # EvidenceGate: frames needed to confirm a var class
CLEAR_FRAMES     = 12   # EvidenceGate: frames needed to reset to correct
START_COUNTDOWN_SEC = 5
KP_CONF_LIVE     = 0.30
KP_CONF_GHOST    = 0.20
MAX_GHOST_FRAMES = 10
KP_BUFFER_SIZE   = 6

# ──────────────────────────────────────────────────────────────────────────────
# Colours
# ──────────────────────────────────────────────────────────────────────────────
SEV = {
    "good":    (34,  197,  94),
    "info":    (59,  130, 246),
    "warning": (234, 179,   8),
    "error":   (239,  68,  68),
    "neutral": (148, 163, 184),
}

CLASS_DISPLAY = {
    "correct": ("CORRECT FORM",     (34,  197,  94)),
    "var1":    ("INCORRECT FORM",   (239,  68,  68)),
    "var2":    ("INCORRECT FORM",   (239,  68,  68)),
    "var3":    ("INCORRECT FORM",   (239,  68,  68)),
    "neutral": ("TRANSITIONING...", (148, 163, 184)),
}

ALL_BONES = [
    (0,1),(0,2),(1,3),(2,4),
    (5,6),(5,7),(7,9),(6,8),(8,10),
    (5,11),(6,12),(11,12),
    (11,13),(13,15),(12,14),(14,16),
]

EXERCISE_KP_MASK = {
    "pushups": {0, 5, 6, 7, 8, 9, 10, 11, 12, 15, 16},
    "squats":  {5, 6, 11, 12, 13, 14, 15, 16},
    "lunges":  {5, 6, 11, 12, 13, 14, 15, 16},
    "planks":  {0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
}

DEFAULT_MODEL = "yolo11x-pose.pt"
DEFAULT_CONF  = 0.5
DEFAULT_IOU   = 0.7


# ──────────────────────────────────────────────────────────────────────────────
# KeypointBuffer  (unchanged from v9)
# ──────────────────────────────────────────────────────────────────────────────
class KeypointBuffer:
    def __init__(self, n_kp=17, buf=KP_BUFFER_SIZE, max_ghost=MAX_GHOST_FRAMES):
        self._n     = n_kp
        self._buf   = [collections.deque(maxlen=buf) for _ in range(n_kp)]
        self._ghost = [0] * n_kp
        self._max_g = max_ghost

    def update(self, pts, conf):
        out_pts  = pts.copy()
        out_conf = conf.copy()
        for i in range(min(self._n, len(pts))):
            live_ok = conf[i] >= KP_CONF_LIVE and not (pts[i] == 0).all()
            if live_ok:
                self._buf[i].append(pts[i].copy())
                self._ghost[i] = 0
            else:
                self._ghost[i] += 1
                if self._buf[i] and self._ghost[i] <= self._max_g:
                    buf_arr = np.array(self._buf[i])
                    w = np.arange(1, len(buf_arr)+1, dtype=float); w /= w.sum()
                    out_pts[i]  = (buf_arr * w[:,None]).sum(axis=0)
                    out_conf[i] = max(KP_CONF_GHOST, conf[i])
        return out_pts, out_conf

    def reset(self):
        for i in range(self._n):
            self._buf[i].clear(); self._ghost[i] = 0


# ──────────────────────────────────────────────────────────────────────────────
# AngleSmoother  (unchanged)
# ──────────────────────────────────────────────────────────────────────────────
class AngleSmoother:
    def __init__(self, n=SMOOTH_WINDOW):
        self._buf = collections.deque(maxlen=n)
    def update(self, v):
        if v is None: self._buf.clear(); return None
        self._buf.append(v); return float(np.mean(self._buf))
    def reset(self): self._buf.clear()
    def last(self):  return float(self._buf[-1]) if self._buf else None


# ──────────────────────────────────────────────────────────────────────────────
# EvidenceGate  (NEW — replaces cls_stab Stabilizer)
#
# Invariant: form_class_key is ONLY derived from sustained dominant_tag evidence.
# Top cue produces dominant_tag → EvidenceGate confirms/clears bottom class.
# Anti-contradiction: 'neutral' shown during any transition window.
# ──────────────────────────────────────────────────────────────────────────────
class EvidenceGate:
    CONFIRM = CONFIRM_FRAMES   # sustained error frames → confirm var class
    CLEAR   = CLEAR_FRAMES     # sustained good frames  → reset to correct

    def __init__(self, tag_to_class: dict, var_labels: dict):
        # tag_to_class: {tag_string: "var1"|"var2"|"var3"}
        self._t2c    = tag_to_class
        self._labels = var_labels   # {cls_key: human_label}
        self._err    = {}           # {tag: consecutive_error_frames}
        self._good   = 0            # consecutive good frames
        self._cls    = None         # confirmed class: None|"correct"|"var1"|"var2"|"var3"

    def update(self, dominant_tag):
        """
        dominant_tag: single string from checker priority logic, or None (good frame).
        Returns (form_class_key, form_label) — safe to render directly.
        """
        if dominant_tag in ("setup_pending", "no_pose"):
            # Transitional states should never render as confirmed correct/var.
            self._good = 0
            for t in list(self._err):
                self._err[t] = max(0, self._err[t] - 2)
            return "neutral", self._labels.get("neutral", "Transitioning...")

        if dominant_tag is None:
            # Good evidence — accumulate toward clear
            self._good += 1
            for t in list(self._err):
                self._err[t] = max(0, self._err[t] - 2)
            if self._good >= self.CLEAR:
                self._cls = "correct"
        else:
            # Error evidence — accumulate toward confirm
            self._good = max(0, self._good - 3)
            self._err[dominant_tag] = self._err.get(dominant_tag, 0) + 1
            for t in list(self._err):
                if t != dominant_tag:
                    self._err[t] = max(0, self._err[t] - 1)
            if self._err.get(dominant_tag, 0) >= self.CONFIRM:
                mapped = self._t2c.get(dominant_tag)
                if mapped:
                    self._cls = mapped

        # ── Anti-contradiction neutral window ──────────────────────────────
        # Case A: confirmed var but good evidence accumulating → show neutral
        #         (prevents top=good + bottom=var)
        if (self._cls in ("var1","var2","var3")
                and dominant_tag is None
                and self._good >= self.CLEAR // 2):
            out = "neutral"
        # Case B: confirmed correct but new error just arrived → show neutral
        #         (prevents top=error + bottom=correct)
        elif (self._cls == "correct"
              and dominant_tag is not None
              and self._good < 2):
            out = "neutral"
        else:
            out = self._cls or "neutral"

        label = self._labels.get(out, "Transitioning...")
        return out, label

    def evidence_strength(self) -> int:
        """Frame count of currently strongest error tag (for debugging/display)."""
        if not self._err: return 0
        return max(self._err.values())

    def reset(self):
        self._err.clear(); self._good = 0; self._cls = None


# ──────────────────────────────────────────────────────────────────────────────
# Assessment builder helper
# ──────────────────────────────────────────────────────────────────────────────
def _assessment(primary_angle, motion_cue, dominant_tag):
    """
    Create the shared per-frame assessment dict.
    form_class_key and form_label are placeholders filled by main loop after
    EvidenceGate.update() — ensuring both headers share the same causal source.
    """
    return {
        "primary_angle":    primary_angle,
        "motion_cue":       motion_cue,       # (text, severity) for top header
        "dominant_tag":     dominant_tag,     # evidence tag driving EvidenceGate
        "form_class_key":   "neutral",        # filled by ev_gate.update()
        "form_label":       "Transitioning...",  # filled by ev_gate.update()
        "evidence_tag":     dominant_tag,
        "evidence_strength": 0,               # filled after ev_gate.update()
    }


# Shared no-person assessment used by both the CLI runner and web session.
_no_pose = _assessment(None, ("Step into frame - full body visible", "neutral"), "no_pose")
_no_pose["form_class_key"] = "neutral"
_no_pose["form_label"]     = "No Pose"


# ──────────────────────────────────────────────────────────────────────────────
# Geometry helpers  (unchanged)
# ──────────────────────────────────────────────────────────────────────────────
def angle_at_b(a, b, c) -> float:
    ba=np.asarray(a,float)-np.asarray(b,float)
    bc=np.asarray(c,float)-np.asarray(b,float)
    d=np.linalg.norm(ba)*np.linalg.norm(bc)
    return float(np.degrees(np.arccos(np.clip(np.dot(ba,bc)/d,-1,1)))) if d>1e-6 else 180.0

def vertical_angle(top, bot) -> float:
    dx=float(top[0])-float(bot[0]); dy=float(bot[1])-float(top[1])
    return abs(math.degrees(math.atan2(abs(dx), max(dy,1e-3))))

def shin_tilt(knee, ankle) -> float:
    vec=np.asarray(ankle,float)-np.asarray(knee,float)
    return abs(math.degrees(math.atan2(abs(vec[0]), max(vec[1],1e-3))))

def signed_deviation(shoulder, hip, ankle) -> float:
    sh=np.asarray(shoulder,float); hp=np.asarray(hip,float); an=np.asarray(ankle,float)
    sa=an-sh; sal=np.linalg.norm(sa)
    if sal<1: return 0.0
    closest=sh+(np.dot(hp-sh,sa)/sal**2)*sa
    dev=180.0-angle_at_b(sh,hp,an)
    return (1.0 if hp[1]>closest[1] else -1.0)*dev

def midpt(a,b): return np.array([(a[0]+b[0])/2.0,(a[1]+b[1])/2.0])
def ipt(a):     return (int(a[0]),int(a[1]))

def vis(pts, conf, *idxs):
    return all(i<len(pts) and conf[i]>=KP_CONF_GHOST and not (pts[i]==0).all() for i in idxs)

def arm_conf(pts, conf, sh, el, wr):
    if vis(pts,conf,sh,el,wr): return float(np.mean([conf[sh],conf[el],conf[wr]]))
    return 0.0

def extract_kp(result):
    if result.keypoints is None or result.keypoints.xy is None: return None, None
    xy,c=result.keypoints.xy,result.keypoints.conf
    if xy.shape[0]==0: return None,None
    return xy[0].cpu().numpy(),(c[0].cpu().numpy() if c is not None else np.ones(17))


# ──────────────────────────────────────────────────────────────────────────────
# Drawing helpers  (unchanged)
# ──────────────────────────────────────────────────────────────────────────────
def draw_skeleton(frame, pts, conf, allowed, highlight):
    for (a,b) in ALL_BONES:
        if a not in allowed or b not in allowed: continue
        if vis(pts,conf,a,b):
            cv2.line(frame,ipt(pts[a]),ipt(pts[b]),highlight.get((a,b),(70,70,70)),2,cv2.LINE_AA)
    for i in allowed:
        if i<len(pts) and conf[i]>=KP_CONF_GHOST and not (pts[i]==0).all():
            cv2.circle(frame,ipt(pts[i]),5,(255,255,255),-1,cv2.LINE_AA)
            cv2.circle(frame,ipt(pts[i]),5,(0,0,0),1,cv2.LINE_AA)

def draw_line(frame,pa,pb,color,thick=2,dashed=False):
    p1,p2=np.asarray(pa,float),np.asarray(pb,float)
    if dashed:
        n=max(int(np.linalg.norm(p2-p1)/12),1)
        for i in range(n):
            if i%2==0:
                cv2.line(frame,ipt(p1+(p2-p1)*(i/n)),ipt(p1+(p2-p1)*((i+1)/n)),color,thick,cv2.LINE_AA)
    else:
        cv2.line(frame,ipt(p1),ipt(p2),color,thick,cv2.LINE_AA)

def draw_arc(frame,a,b,c,color,label):
    ba=np.asarray(a,float)-np.asarray(b,float)
    bc=np.asarray(c,float)-np.asarray(b,float)
    r=max(int(min(np.linalg.norm(ba),np.linalg.norm(bc))*0.38),22)
    a1=math.degrees(math.atan2(-ba[1],ba[0])); a2=math.degrees(math.atan2(-bc[1],bc[0]))
    s,e=min(a1,a2),max(a1,a2)
    if e-s>180: s,e=e,s+360
    cv2.ellipse(frame,ipt(b),(r,r),0,-e,-s,color,2,cv2.LINE_AA)
    mid=math.radians(-(s+e)/2)
    cv2.putText(frame,label,(int(b[0]+(r+18)*math.cos(mid)),int(b[1]+(r+18)*math.sin(mid))),
                cv2.FONT_HERSHEY_SIMPLEX,0.46,color,1,cv2.LINE_AA)


# ──────────────────────────────────────────────────────────────────────────────
# Push-up checker
#
# dominant_tag priority (error → drives EvidenceGate):
#   "sag"  → var1 Hip Sagging        (alignment dev > 20 sag)
#   "pike" → var2 Hip Piking         (alignment dev > 15 pike)
#   None   → good/info (elbow cue shown when no alignment error)
#
# var3 (Insufficient Elbow Flexion) is handled at rep-completion in update_reps.
# ──────────────────────────────────────────────────────────────────────────────
def check_pushup(pts, conf, state):
    hl, lines, arcs = {}, [], []

    lc=arm_conf(pts,conf,5,7,9); rc=arm_conf(pts,conf,6,8,10)
    if lc==0 and rc==0:
        state["primary_angle"] = None
        return _assessment(None, ("Arm keypoints not visible - face sideways","neutral"), None), hl, lines, arcs

    sh,el,wr = (5,7,9) if lc>=rc else (6,8,10)
    raw_ea = angle_at_b(pts[sh],pts[el],pts[wr])
    ea = state["sm"]["elbow"].update(raw_ea)
    state["primary_angle"] = ea
    state["min_angle_this_rep"] = min(state.get("min_angle_this_rep",999), ea)
    phase = state.get("stage","up")

    # Elbow depth cue (used when no alignment error is dominant)
    if ea >= 145:
        elbow_cue = ("Arms extended - lower with control","info")
    elif 80 <= ea <= 90:
        elbow_cue = (f"Good elbow depth ({ea:.0f}d) - push up!","good")
    elif ea < 65:
        elbow_cue = ("Too deep - push up now","error")
    elif ea > 100 and phase == "down":
        elbow_cue = (f"Go lower - {ea:.0f}d above target","warning")
    else:
        elbow_cue = (f"Elbow: {ea:.0f}d - keep lowering","info")

    c = SEV[elbow_cue[1]]; hl[(sh,el)]=c; hl[(el,wr)]=c
    arcs.append((pts[sh],pts[el],pts[wr],c,f"{ea:.0f}d"))

    # Alignment — drives dominant_tag
    dominant_tag = None
    align_cue    = None
    if vis(pts,conf,5,6,11,12,15,16):
        sh_m  = midpt(pts[5],pts[6])
        hip_m = midpt(pts[11],pts[12])
        ank_m = midpt(pts[15],pts[16])
        dev   = state["sm"]["align"].update(signed_deviation(sh_m,hip_m,ank_m))

        if dev > 20:
            dominant_tag = "sag"
            align_cue    = ("Engage core - hips sagging!","error")
        elif dev < -15:
            dominant_tag = "pike"
            align_cue    = ("Lower hips - raised too high!","error")
        elif abs(dev) > 10:
            align_cue    = (f"Alignment {dev:+.0f}d - tighten core","warning")
        else:
            align_cue    = ("Body alignment: straight OK","good")

        ca = SEV[align_cue[1]]
        hl[(5,11)]=ca; hl[(6,12)]=ca; hl[(11,12)]=ca
        lines.append((sh_m,hip_m,ca,False)); lines.append((hip_m,ank_m,ca,False))
    else:
        state["sm"]["align"].reset()

    # Head (informational angle tracking only — no ev_tag)
    if vis(pts,conf,0,5,6):
        sh_m = midpt(pts[5],pts[6])
        ref  = max(np.linalg.norm(pts[sh] - (midpt(pts[11],pts[12]) if vis(pts,conf,11,12) else pts[el])), 10)
        state["sm"]["head"].update((pts[0][0]-sh_m[0])/ref)
    else:
        state["sm"]["head"].reset()

    # Motion cue: alignment error takes priority over elbow coaching
    if dominant_tag in ("sag","pike"):
        motion_cue = align_cue
    elif align_cue and align_cue[1] == "warning":
        # Show alignment warning even without confirming a tag
        motion_cue = align_cue
    else:
        motion_cue = elbow_cue

    return _assessment(ea, motion_cue, dominant_tag), hl, lines, arcs


# ──────────────────────────────────────────────────────────────────────────────
# Squat checker
#
# dominant_tag priority:
#   "lean"  → var2 Excessive Forward Lean  (torso > 45d)
#   "depth" → var1 Insufficient Depth      (ka > 95d during down phase)
#   None    → good/info
# ──────────────────────────────────────────────────────────────────────────────
def check_squat(pts, conf, state):
    hl, lines, arcs = {}, [], []

    if not vis(pts,conf,12,14,16):
        state["primary_angle"] = None
        return _assessment(None, ("Leg keypoints not visible - face sideways","neutral"), None), hl, lines, arcs

    raw_ka = angle_at_b(pts[12],pts[14],pts[16])
    ka = state["sm"]["knee"].update(raw_ka)
    state["primary_angle"] = ka
    state["min_angle_this_rep"] = min(state.get("min_angle_this_rep",999), ka)
    phase = state.get("stage","up")

    # Torso lean — checked first (injury priority)
    lean_tag = None
    torso_cue = None
    torso_deg = 0.0
    if vis(pts,conf,6,12):
        torso_deg = state["sm"]["lean"].update(vertical_angle(pts[6],pts[12]))
        if torso_deg > 45:
            lean_tag  = "lean"
            torso_cue = (f"Lean {torso_deg:.0f}d - lift chest now!","error")
        elif torso_deg > 35:
            torso_cue = (f"Forward lean ({torso_deg:.0f}d) - lift chest","warning")
        else:
            torso_cue = (f"Torso: {torso_deg:.0f}d OK","good")
        cl = SEV[torso_cue[1]]
        hl[(5,6)]=cl; hl[(6,12)]=cl; hl[(5,11)]=cl
        lines.append((pts[6],pts[12],cl,False))
    else:
        state["sm"]["lean"].reset()

    # Knee depth cue
    depth_tag = None
    if ka >= 155:
        knee_cue = ("Standing - begin your descent","info")
    elif 60 <= ka <= 85:
        knee_cue = (f"Good depth ({ka:.0f}d) - drive through heels!","good")
    elif ka < 60:
        knee_cue = ("Too deep - come up slightly","error")
    elif ka > 95 and phase == "down":
        depth_tag = "depth"
        knee_cue  = (f"Go deeper - {ka:.0f}d above target","warning")
    else:
        knee_cue = (f"Knee: {ka:.0f}d - keep lowering","info")

    c = SEV[knee_cue[1]]; hl[(12,14)]=c; hl[(14,16)]=c
    arcs.append((pts[12],pts[14],pts[16],c,f"{ka:.0f}d"))

    # Priority: lean (injury) > depth (technique)
    if lean_tag:
        dominant_tag = "lean"
        motion_cue   = torso_cue
    elif depth_tag:
        dominant_tag = "depth"
        motion_cue   = knee_cue
    else:
        dominant_tag = None
        # Show most informative cue available
        if torso_cue and torso_cue[1] in ("warning","error"):
            motion_cue = torso_cue
        else:
            motion_cue = knee_cue

    return _assessment(ka, motion_cue, dominant_tag), hl, lines, arcs


# ──────────────────────────────────────────────────────────────────────────────
# Lunge checker
#
# dominant_tag priority:
#   "knee_over"    → var1 Front Knee Overextension
#   "rear_shallow" → var2 Insufficient Rear Knee Depth
#   "torso_lean"   → var3 Torso Forward Lean
#   None           → good/info
#
# Internal debounce (lunge_*_frames) pre-filters noise; EvidenceGate adds
# an additional CONFIRM=6 frame gate on top.
# ──────────────────────────────────────────────────────────────────────────────
def _pick_lunge_legs(pts, conf, state):
    l_ok=vis(pts,conf,11,13,15); r_ok=vis(pts,conf,12,14,16)
    if not l_ok and not r_ok: return None, None
    if l_ok and not r_ok: state["front_leg"]="L"; return (11,13,15), None
    if r_ok and not l_ok: state["front_leg"]="R"; return (12,14,16), None
    l_ka=angle_at_b(pts[11],pts[13],pts[15])
    r_ka=angle_at_b(pts[12],pts[14],pts[16])
    if l_ka < r_ka-15: state["front_leg"]="L"
    elif r_ka < l_ka-15: state["front_leg"]="R"
    return ((11,13,15),(12,14,16)) if state.get("front_leg","R")=="L" else ((12,14,16),(11,13,15))


def check_lunge(pts, conf, state):
    hl, lines, arcs = {}, [], []

    front, rear = _pick_lunge_legs(pts, conf, state)
    if front is None:
        state["primary_angle"] = None
        return _assessment(None, ("Leg keypoints not visible - face sideways","neutral"), None), hl, lines, arcs

    fh,fk,fa = front
    raw_ka = angle_at_b(pts[fh],pts[fk],pts[fa])
    ka = state["sm"]["knee"].update(raw_ka)
    state["primary_angle"] = ka
    phase = state.get("stage","up")

    # Internal debounce counter: must be sustained in lunge position
    if phase == "down" and ka <= 120:
        state["lunge_down_frames"] = state.get("lunge_down_frames",0) + 1
    else:
        state["lunge_down_frames"] = 0
        state["lunge_rear_err_frames"] = 0
        state["lunge_torso_err_frames"] = 0

    down_ready = state.get("lunge_down_frames",0) >= 4

    # Front knee angle cue
    if ka >= 160:
        knee_cue = ("Stand tall - step forward to lunge","info")
    elif 80 <= ka <= 100:
        knee_cue = (f"Good depth ({ka:.0f}d) - push off to rise","good")
    elif ka < 70:
        knee_cue = ("Too low - ease up slightly","error")
    elif phase == "down":
        knee_cue = (f"Drop lower - front knee to 90d (now {ka:.0f}d)","warning")
    else:
        knee_cue = (f"Front knee: {ka:.0f}d - lower down","info")

    c = SEV[knee_cue[1]]; hl[(fh,fk)]=c; hl[(fk,fa)]=c
    arcs.append((pts[fh],pts[fk],pts[fa],c,f"{ka:.0f}d"))

    # Shin tilt (knee over toes)
    knee_over = False
    shin_cue  = None
    if ka < 140 and vis(pts,conf,fk,fa):
        tilt = state["sm"]["knee_toe"].update(shin_tilt(pts[fk],pts[fa]))
        if tilt > 45:
            knee_over = True
            shin_cue  = (f"Knee past toes ({tilt:.0f}d) - step wider","error")
            hl[(fk,fa)] = SEV["error"]
        elif tilt > 36:
            shin_cue  = (f"Knee forward ({tilt:.0f}d) - shift weight back","warning")
            hl[(fk,fa)] = SEV["warning"]
        else:
            shin_cue  = (f"Knee position: good ({tilt:.0f}d) OK","good")
            hl[(fk,fa)] = SEV["good"]
        lines.append((pts[fa],(pts[fa][0],pts[fk][1]),SEV[shin_cue[1]],True))
    else:
        state["sm"]["knee_toe"].reset()

    # Rear knee — require LIVE confidence to avoid ghost-from-floor triggers
    rear_shallow = False
    rear_cue     = None
    if rear is not None and ka < 135:
        rh,rk,ra = rear
        rear_live = (vis(pts,conf,rh,rk,ra)
                     and conf[rk] >= KP_CONF_LIVE
                     and conf[ra] >= KP_CONF_LIVE)
        if rear_live:
            rka = state["sm"]["rear_knee"].update(angle_at_b(pts[rh],pts[rk],pts[ra]))
            arcs.append((pts[rh],pts[rk],pts[ra],SEV["info"],f"{rka:.0f}d"))
            if phase == "down" and ka <= 110 and rka > 142:
                state["lunge_rear_err_frames"] = state.get("lunge_rear_err_frames",0)+1
                if down_ready and state["lunge_rear_err_frames"] >= 4:
                    rear_shallow = True
                    rear_cue = (f"Lower rear knee - {rka:.0f}d too high","error")
                    hl[(rh,rk)]=SEV["error"]; hl[(rk,ra)]=SEV["error"]
                else:
                    rear_cue = (f"Rear knee: {rka:.0f}d - descend further","warning")
                    hl[(rh,rk)]=SEV["warning"]; hl[(rk,ra)]=SEV["warning"]
            else:
                state["lunge_rear_err_frames"] = max(state.get("lunge_rear_err_frames",0)-1, 0)
                rear_cue = (f"Rear knee: {rka:.0f}d OK","good")
                hl[(rh,rk)]=SEV["good"]; hl[(rk,ra)]=SEV["good"]
        else:
            state["lunge_rear_err_frames"] = 0
            state["sm"]["rear_knee"].reset()
    else:
        state["lunge_rear_err_frames"] = 0
        state["sm"]["rear_knee"].reset()

    # Torso lean — shoulder midpoint to hip midpoint (v9 fix kept)
    torso_lean = False
    torso_cue  = None
    torso_deg  = 0.0
    if vis(pts,conf,5,6,11,12):
        sh_mid  = midpt(pts[5],pts[6])
        hip_mid = midpt(pts[11],pts[12])
        torso_deg = state["sm"]["lean"].update(vertical_angle(sh_mid,hip_mid))

        # Hard-reset when deep in lunge with good torso — prevents stale votes
        if ka < 95 and torso_deg <= 40:
            state["lunge_torso_err_frames"] = 0

        if phase == "down" and ka <= 112 and torso_deg > 42:
            state["lunge_torso_err_frames"] = state.get("lunge_torso_err_frames",0)+1
            if down_ready and state["lunge_torso_err_frames"] >= 4:
                torso_lean = True
                torso_cue  = (f"Torso lean {torso_deg:.0f}d - stand upright!","error")
            else:
                torso_cue  = (f"Slight lean ({torso_deg:.0f}d) - lift chest","warning")
        elif phase == "down" and ka <= 125 and torso_deg > 34:
            state["lunge_torso_err_frames"] = max(state.get("lunge_torso_err_frames",0)-1, 0)
            torso_cue = (f"Slight lean ({torso_deg:.0f}d) - lift chest","warning")
        else:
            state["lunge_torso_err_frames"] = 0
            torso_cue = (f"Torso: {torso_deg:.0f}d OK","good")

        ct = SEV[torso_cue[1]]
        hl[(5,6)]=ct; hl[(5,11)]=ct; hl[(6,12)]=ct
        lines.append((sh_mid,hip_mid,ct,False))
    else:
        state["lunge_torso_err_frames"] = 0
        state["sm"]["lean"].reset()

    # Priority: knee_over > rear_shallow > torso_lean
    # Motion cue matches dominant_tag — causal coherence enforced here.
    if knee_over:
        dominant_tag = "knee_over"
        motion_cue   = shin_cue
    elif rear_shallow:
        dominant_tag = "rear_shallow"
        motion_cue   = rear_cue or (f"Lower rear knee - too high","error")
    elif torso_lean:
        dominant_tag = "torso_lean"
        motion_cue   = torso_cue
    else:
        dominant_tag = None
        # Best informative cue when no confirmed error
        if shin_cue and shin_cue[1] == "warning":
            motion_cue = shin_cue
        elif torso_cue and torso_cue[1] in ("warning","error"):
            motion_cue = torso_cue
        elif rear_cue and rear_cue[1] == "warning":
            motion_cue = rear_cue
        else:
            motion_cue = knee_cue

    return _assessment(ka, motion_cue, dominant_tag), hl, lines, arcs


# ──────────────────────────────────────────────────────────────────────────────
# Plank checker
#
# dominant_tag priority:
#   "hip_sag"   → var1 Hip Sagging      (dev > 28)
#   "hip_high"  → var2 Hip Elevation    (dev < -28)
#   "head_drop" → var3 Head Dropping    (nb > 15)
#   None        → good/info
# ──────────────────────────────────────────────────────────────────────────────
def check_plank(pts, conf, state):
    hl, lines, arcs = {}, [], []

    if not vis(pts,conf,5,6,11,12,15,16):
        state["primary_angle"] = None
        return _assessment(None, ("Move back - full body must be visible","neutral"), None), hl, lines, arcs

    sh  = midpt(pts[5],pts[6])
    hip = midpt(pts[11],pts[12])
    ank = midpt(pts[15],pts[16])

    # Setup gate: avoid showing Correct Form while user is still upright.
    # vertical_angle() is small when torso is vertical and larger when horizontal.
    torso_from_vertical = vertical_angle(sh, hip)
    if not state.get("setup_ready", False):
        if torso_from_vertical >= 35:
            state["setup_ready"] = True
        else:
            state["primary_angle"] = None
            return _assessment(None, ("Get into plank position to begin", "info"), "setup_pending"), hl, lines, arcs

    # Off-ground gate: allow slight hip dip, but block actual floor contact.
    # Use a floor-line estimate + hysteresis instead of raw ankle-hip ratio.
    torso_len = max(np.linalg.norm(sh - hip), 10.0)
    floor_idxs = [7, 8, 9, 10, 13, 14, 15, 16]
    floor_y_vals = [pts[i][1] for i in floor_idxs
                    if i < len(pts) and conf[i] >= KP_CONF_GHOST and not (pts[i] == 0).all()]
    floor_y = max(floor_y_vals) if floor_y_vals else max(pts[15][1], pts[16][1])
    hip_to_floor = float(floor_y - hip[1])

    # Min clearance keeps plank valid with natural small hip sag.
    min_clear = max(0.035 * torso_len, 6.0)
    if hip_to_floor < min_clear:
        state["plank_floor_fail_frames"] = state.get("plank_floor_fail_frames", 0) + 1
    else:
        state["plank_floor_fail_frames"] = max(state.get("plank_floor_fail_frames", 0) - 1, 0)

    if state.get("plank_floor_fail_frames", 0) >= 5:
        state["primary_angle"] = None
        return _assessment(None, ("Lift hips/chest off the floor into plank", "warning"), "setup_pending"), hl, lines, arcs

    dev = state["sm"]["align"].update(signed_deviation(sh,hip,ank))
    state["primary_angle"] = 180.0 - abs(dev)

    # Alignment — primary plank cue
    dominant_tag = None
    if dev > 14:
        dominant_tag = "hip_sag"
        align_cue    = ("Lift hips - body too close to floor","error")
    elif dev < -28:
        dominant_tag = "hip_high"
        align_cue    = ("Lower hips - raised too high","error")
    elif dev > 8:
        align_cue    = (f"Hips slightly low ({dev:+.0f}d) - lift a bit","warning")
    elif dev < -18:
        align_cue    = (f"Hips slightly high ({dev:+.0f}d) - lower a bit","warning")
    else:
        align_cue    = (f"Body alignment: straight OK ({dev:+.1f}d)","good")

    ca = SEV[align_cue[1]]
    for b in [(5,11),(6,12),(11,12),(11,13),(12,14),(13,15),(14,16)]: hl[b]=ca
    lines.append((sh,hip,ca,False)); lines.append((hip,ank,ca,False))
    arcs.append((sh,hip,ank,ca,f"{dev:+.0f}d"))

    # Head check
    head_drop = False
    head_cue  = None
    if vis(pts,conf,0,6,10):
        nose=pts[0]; sw=pts[10]-pts[6]; sl=np.linalg.norm(sw)
        if sl > 5:
            t      = np.clip(np.dot(nose-pts[6],sw)/sl**2,0,1)
            line_y = pts[6][1]+t*sw[1]
            nb     = state["sm"]["head"].update(float(nose[1]-line_y))
            if nb > 10:
                head_drop = True
                head_cue  = ("Keep head neutral - chin dropping","error")
                hl[(0,1)]=SEV["error"]; hl[(0,2)]=SEV["error"]
                lines.append((nose,(nose[0],line_y),SEV["error"],True))
            elif nb > 2:
                head_cue = ("Head slightly low - keep eyes forward","warning")
            else:
                head_cue = ("Head position: neutral OK","good")
    else:
        state["sm"]["head"].reset()

    # Elbow stack (informational only, no dominant_tag)
    if vis(pts,conf,6,8,11,12):
        ref = max(abs(pts[6][1]-midpt(pts[11],pts[12])[1]),10)
        vd  = state["sm"]["stack"].update(abs(pts[6][1]-pts[8][1])/ref)
        hl[(6,8)] = SEV["warning"] if vd>0.35 else SEV["good"]
    else:
        state["sm"]["stack"].reset()

    # Priority: hip alignment > head drop
    # (hip errors are more structurally significant)
    if dominant_tag:  # hip_sag or hip_high
        motion_cue = align_cue
    elif head_drop:
        dominant_tag = "head_drop"
        motion_cue   = head_cue
    else:
        dominant_tag = None
        # Show head warning if present, else alignment cue
        if head_cue and head_cue[1] in ("warning","error"):
            motion_cue = head_cue
        else:
            motion_cue = align_cue

    return _assessment(state["primary_angle"], motion_cue, dominant_tag), hl, lines, arcs


# ──────────────────────────────────────────────────────────────────────────────
# Exercise registry
# ──────────────────────────────────────────────────────────────────────────────
def _smooth(keys, window=SMOOTH_WINDOW):
    return {k: AngleSmoother(window) for k in keys}

def _make_ev_gate(ex):
    """Build EvidenceGate from exercise config."""
    labels = {"correct": "Correct Form", "neutral": "Transitioning..."}
    for i, lbl in ex.get("var_names", {}).items():
        labels[f"var{i}"] = lbl
    return EvidenceGate(ex.get("ev_tags", {}), labels)

EXERCISES = {
    "pushups": {
        "label":       "Push-ups",
        "description": "Upper body - chest, shoulders, triceps",
        "up_angle": 130.0, "down_angle": 110.0, "tolerance": 10.0,
        "is_timed":    False,
        "checker":     check_pushup,
        "smooth_window": SMOOTH_WINDOW,
        "kp_buffer_size": KP_BUFFER_SIZE,
        "max_ghost_frames": MAX_GHOST_FRAMES,
        "insuf_depth_threshold": 105,  # rep-completion var3 check
        "smooth_keys": ["elbow","align","head"],
        # ev_tags: per-frame tags that EvidenceGate maps to form class
        "ev_tags":     {"sag": "var1", "pike": "var2"},
        # var3 (Insufficient Elbow Flexion) is rep-completion only
        "var_names":   {1:"Hip Sagging", 2:"Hip Piking", 3:"Insufficient Elbow Flexion"},
    },
    "squats": {
        "label":       "Squats",
        "description": "Lower body - quads, glutes, hamstrings",
        "up_angle": 150.0, "down_angle": 100.0, "tolerance": 10.0,
        "is_timed":    False,
        "checker":     check_squat,
        "smooth_window": SMOOTH_WINDOW,
        "kp_buffer_size": KP_BUFFER_SIZE,
        "max_ghost_frames": MAX_GHOST_FRAMES,
        "smooth_keys": ["knee","lean"],
        # Priority in ev_tags: lean > depth (lean is higher injury risk)
        "ev_tags":     {"lean": "var2", "depth": "var1"},
        "var_names":   {1:"Insufficient Depth", 2:"Excessive Forward Lean"},
    },
    "lunges": {
        "label":       "Lunges",
        "description": "Lower body - quads, hip flexors, glutes",
        "up_angle": 155.0, "down_angle": 115.0, "tolerance": 10.0,
        "is_timed":    False,
        "checker":     check_lunge,
        "smooth_window": 2,
        "kp_buffer_size": 4,
        "max_ghost_frames": 3,
        "smooth_keys": ["knee","knee_toe","rear_knee","lean"],
        "ev_tags":     {"knee_over": "var1", "rear_shallow": "var2", "torso_lean": "var3"},
        "var_names":   {1:"Front Knee Overextension", 2:"Insufficient Rear Knee Depth", 3:"Torso Forward Lean"},
    },
    "planks": {
        "label":       "Planks",
        "description": "Core - transverse abdominis, spinal stabilisers",
        "up_angle": 180.0, "down_angle": 160.0, "tolerance": 10.0,
        "is_timed":    True,
        "checker":     check_plank,
        "smooth_window": SMOOTH_WINDOW,
        "kp_buffer_size": KP_BUFFER_SIZE,
        "max_ghost_frames": MAX_GHOST_FRAMES,
        "smooth_keys": ["align","head","stack"],
        "ev_tags":     {"hip_sag": "var1", "hip_high": "var2", "head_drop": "var3"},
        "var_names":   {1:"Hip Sagging", 2:"Hip Elevation", 3:"Head Dropping"},
    },
}


# ──────────────────────────────────────────────────────────────────────────────
# Session state
# ──────────────────────────────────────────────────────────────────────────────
def make_state(ex):
    return {
        "up_angle": ex["up_angle"], "down_angle": ex["down_angle"],
        "tolerance": ex["tolerance"],
        "stage": "up", "reps": 0, "hit_down": False,
        "prev_angle": None, "primary_angle": None,
        "hold_start": None, "total_hold": 0.0,
        "sm": _smooth(ex["smooth_keys"], ex.get("smooth_window", SMOOTH_WINDOW)),
        # EvidenceGate: the ONLY form-classification gating mechanism
        "ev_gate":              _make_ev_gate(ex),
        "rep_log":              [],
        "down_cls_buf":         [],
        "min_angle_this_rep":   999.0,
        "insuf_flex_this_rep":  False,
        "insuf_depth_threshold": ex.get("insuf_depth_threshold", None),
        "front_leg":            None,
        # Lunge internal debounce counters
        "lunge_down_frames":       0,
        "lunge_rear_err_frames":   0,
        "lunge_torso_err_frames":  0,
        "session_start":        None,
        "exercise_name":        None,
        "form_issues":          set(),
        "plank_floor_fail_frames": 0,
        # Push-up specific HUD gating/rep feedback helpers
        "setup_ready":          False,
        "last_rep_cls":         None,
        "last_rep_time":        0.0,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Rep counting  (unchanged logic; cls_key now from EvidenceGate)
# ──────────────────────────────────────────────────────────────────────────────
def update_reps(state, cls_key):
    angle = state.get("primary_angle")
    if angle is None: return
    up, dn, tol = state["up_angle"], state["down_angle"], state["tolerance"]

    if angle <= dn + tol:
        state["stage"]    = "down"
        state["hit_down"] = True
        if cls_key != "neutral":
            state["down_cls_buf"].append(cls_key)

    elif angle >= up - tol:
        if state["hit_down"]:
            buf     = state["down_cls_buf"]
            counts  = Counter(buf)
            dominant = counts.most_common(1)[0][0] if buf else "neutral"

            # Strict rep adjudication: if any down-phase error occurred in the
            # rep, prioritise that variation over transient later 'correct'.
            if buf:
                var_only = [b for b in buf if b in ("var1", "var2", "var3")]
                if var_only:
                    dominant = Counter(var_only).most_common(1)[0][0]

            # Rep-completion insufficient-depth check (push-ups var3)
            insuf_thresh = state.get("insuf_depth_threshold")
            if dominant in ("neutral","correct"):
                if (insuf_thresh is not None and
                        state.get("min_angle_this_rep",999) > insuf_thresh):
                    dominant = "var3"
                    state["insuf_flex_this_rep"] = True
                else:
                    state["insuf_flex_this_rep"] = False
            else:
                state["insuf_flex_this_rep"] = False

            state["rep_log"].append(dominant)
            state["reps"]       += 1
            state["last_rep_cls"]  = dominant
            state["last_rep_time"] = time.time()
            state["hit_down"]    = False
            state["down_cls_buf"] = []
            state["min_angle_this_rep"] = 999.0
        state["stage"] = "up"

    else:
        prev = state["prev_angle"]
        if prev is not None:
            if angle < prev - 0.5:   state["stage"] = "down"
            elif angle > prev + 0.5: state["stage"] = "up"

    state["prev_angle"] = angle


def update_plank(state, is_good):
    now = time.time()
    if is_good:
        if state["hold_start"] is None: state["hold_start"] = now
        else:
            state["total_hold"] += now - state["hold_start"]
            state["hold_start"]  = now
    else:
        state["hold_start"] = None


# ──────────────────────────────────────────────────────────────────────────────
# Scoring  (unchanged)
# ──────────────────────────────────────────────────────────────────────────────
def compute_score(rep_log):
    if not rep_log: return 0, 0, 0.0, "N/A"
    total   = len(rep_log)
    correct = sum(1 for r in rep_log if r=="correct")
    pct     = correct / total * 100
    grade   = "A" if pct>=85 else "B" if pct>=70 else "C" if pct>=55 else "D"
    return correct, total, pct, grade


def show_score_screen(last_frame, rep_log, ex, window_name, state=None):
    h, w = last_frame.shape[:2]
    frame = last_frame.copy()
    ov    = frame.copy()
    cv2.rectangle(ov,(0,0),(w,h),(10,10,10),-1)
    cv2.addWeighted(ov,0.88,frame,0.12,0,frame)

    cv2.putText(frame,"SESSION COMPLETE",(w//2-190,55),
                cv2.FONT_HERSHEY_SIMPLEX,1.2,(200,200,255),3,cv2.LINE_AA)
    cv2.putText(frame,ex["label"].upper(),(w//2-130,95),
                cv2.FONT_HERSHEY_SIMPLEX,0.9,(160,160,200),2,cv2.LINE_AA)

    if ex.get("is_timed"):
        good_t  = state["total_hold"] if state else 0.0
        total_t = ((time.time()-state["session_start"])
                   if (state and state.get("session_start")) else 0.0)
        grade   = "A" if good_t>=60 else "B" if good_t>=45 else "C" if good_t>=30 else "D"
        gc      = SEV["good"] if grade in ("A","B") else SEV["warning"] if grade=="C" else SEV["error"]
        cv2.circle(frame,(w//2,180),60,gc,4,cv2.LINE_AA)
        cv2.putText(frame,grade,(w//2-28,202),cv2.FONT_HERSHEY_SIMPLEX,2.5,gc,5,cv2.LINE_AA)
        cv2.putText(frame,f"Good form hold : {int(good_t)//60}:{int(good_t)%60:02d}",
                    (w//2-200,270),cv2.FONT_HERSHEY_SIMPLEX,0.80,(220,220,220),2,cv2.LINE_AA)
        cv2.putText(frame,f"Total session  : {int(total_t)//60}:{int(total_t)%60:02d}",
                    (w//2-200,300),cv2.FONT_HERSHEY_SIMPLEX,0.68,(160,160,160),1,cv2.LINE_AA)
        cv2.putText(frame,"Grade: A=60s  B=45s  C=30s  D<30s",
                    (w//2-200,326),cv2.FONT_HERSHEY_SIMPLEX,0.50,(120,120,160),1,cv2.LINE_AA)
        vnames = ex.get("var_names",{}); fi = state.get("form_issues",set()) if state else set()
        y = 364
        cv2.putText(frame,"Form issues detected:",(w//2-200,y),
                    cv2.FONT_HERSHEY_SIMPLEX,0.58,(180,180,180),1,cv2.LINE_AA); y+=30
        if not fi:
            cv2.putText(frame,"None - great form throughout!",(w//2-200,y),
                        cv2.FONT_HERSHEY_SIMPLEX,0.58,SEV["good"],1,cv2.LINE_AA)
        else:
            for key in ["var1","var2","var3"]:
                if key in fi:
                    lbl = vnames.get(int(key[-1]),key)
                    cv2.putText(frame,f"  [!] {lbl}",(w//2-200,y),
                                cv2.FONT_HERSHEY_SIMPLEX,0.58,SEV["error"],1,cv2.LINE_AA); y+=28
    else:
        correct,total,pct,grade = compute_score(rep_log)
        gc = SEV["good"] if grade in ("A","B") else SEV["warning"] if grade=="C" else SEV["error"]
        cv2.circle(frame,(w//2,180),60,gc,4,cv2.LINE_AA)
        cv2.putText(frame,grade,(w//2-28,202),cv2.FONT_HERSHEY_SIMPLEX,2.5,gc,5,cv2.LINE_AA)
        if total>0:
            cv2.putText(frame,f"Score: {correct} / {total} reps correct  ({pct:.0f}%)",
                        (w//2-235,278),cv2.FONT_HERSHEY_SIMPLEX,0.80,(220,220,220),2,cv2.LINE_AA)
        else:
            cv2.putText(frame,"No reps recorded - press SPACE before exercising",
                        (w//2-275,278),cv2.FONT_HERSHEY_SIMPLEX,0.60,(180,180,180),1,cv2.LINE_AA)
        vnames = ex.get("var_names",{}); cnts = Counter(rep_log); y = 320
        for key, label in [("correct","Correct Form")]+[(f"var{i}",vnames.get(i,"")) for i in range(1,4)]:
            if not label: continue
            n   = cnts.get(key,0)
            col = SEV["good"] if key=="correct" else (SEV["error"] if n>0 else SEV["neutral"])
            cv2.putText(frame,f"{label:<30}: {n} rep{'s' if n!=1 else ''}",
                        (w//2-230,y),cv2.FONT_HERSHEY_SIMPLEX,0.54,col,1,cv2.LINE_AA); y+=30

    cv2.putText(frame,"Press any key to exit",(w//2-155,h-28),
                cv2.FONT_HERSHEY_SIMPLEX,0.65,(130,130,160),1,cv2.LINE_AA)
    cv2.namedWindow(window_name,cv2.WINDOW_NORMAL); cv2.resizeWindow(window_name,1280,720)
    cv2.imshow(window_name,frame); cv2.waitKey(0)


# ──────────────────────────────────────────────────────────────────────────────
# HUD  — reads from unified assessment dict
#
# Panel 1 (0-100 px):  Motion Coach   — assessment["motion_cue"]
# Panel 2 (100-148px): Form Judge     — assessment["form_class_key/form_label"]
#
# Both panels read from the SAME assessment object: coherence is structural.
# ──────────────────────────────────────────────────────────────────────────────
def draw_hud(frame, assessment, state, ex):
    h, w = frame.shape[:2]

    msg, sev       = assessment["motion_cue"]
    msg_color      = SEV[sev]
    cls_key        = assessment["form_class_key"]
    var_label      = assessment["form_label"]
    _, form_color  = CLASS_DISPLAY.get(cls_key, CLASS_DISPLAY["neutral"])

    # ── Panel 1: Motion Coach ────────────────────────────────────────────
    ov = frame.copy()
    cv2.rectangle(ov,(0,0),(w,100),(15,15,20),-1)
    # Left accent stripe = form status colour (instant glance cue)
    cv2.rectangle(ov,(0,0),(10,100),form_color,-1)
    cv2.addWeighted(ov,0.80,frame,0.20,0,frame)

    cv2.putText(frame,ex["label"].upper(),(20,18),
                cv2.FONT_HERSHEY_SIMPLEX,0.46,(120,120,140),1,cv2.LINE_AA)
    cv2.putText(frame,msg,(20,56),
                cv2.FONT_HERSHEY_SIMPLEX,0.72,msg_color,2,cv2.LINE_AA)

    # Right column
    correct,total,pct,_ = compute_score(state["rep_log"])
    if ex.get("is_timed"):
        t = state["total_hold"]
        cv2.putText(frame,f"Hold: {int(t)//60}:{int(t)%60:02d}",
                    (w-215,30),cv2.FONT_HERSHEY_SIMPLEX,0.76,(230,230,230),2,cv2.LINE_AA)
    else:
        cv2.putText(frame,f"Reps: {state['reps']}",
                    (w-195,28),cv2.FONT_HERSHEY_SIMPLEX,0.76,(230,230,230),2,cv2.LINE_AA)
        s_col = SEV["info"] if state["stage"]=="down" else (110,110,130)
        cv2.putText(frame,state["stage"].upper(),
                    (w-195,50),cv2.FONT_HERSHEY_SIMPLEX,0.50,s_col,1,cv2.LINE_AA)
        if total>0:
            ac = SEV["good"] if pct>=70 else SEV["warning"] if pct>=50 else SEV["error"]
            cv2.putText(frame,f"Form  {correct}/{total}  ({pct:.0f}%)",
                        (w-195,70),cv2.FONT_HERSHEY_SIMPLEX,0.44,ac,1,cv2.LINE_AA)

    # Angle progress bar
    angle = assessment.get("primary_angle")
    if angle is not None:
        up,dn = ex["up_angle"],ex["down_angle"]
        norm  = (max(min(angle,up),dn)-dn)/max(up-dn,1)
        bx,by,bw,bh = 20,80,w-220,10
        cv2.rectangle(frame,(bx,by),(bx+bw,by+bh),(35,35,45),-1)
        cv2.rectangle(frame,(bx,by),(bx+int(norm*bw),by+bh),msg_color,-1)
        cv2.putText(frame,f"{angle:.0f}d",(bx+bw+8,by+10),
                    cv2.FONT_HERSHEY_SIMPLEX,0.44,(150,150,170),1)

    # ── Panel 2: Form Judge ──────────────────────────────────────────────
    ov2 = frame.copy()
    cv2.rectangle(ov2,(0,100),(w,148),(18,18,25),-1)
    cv2.addWeighted(ov2,0.85,frame,0.15,0,frame)
    # Thin divider colour = form status (instant coherence signal)
    cv2.line(frame,(0,100),(w,100),form_color,2,cv2.LINE_AA)

    icon = "[OK]" if cls_key=="correct" else \
           ("[!]" if cls_key.startswith("var") else "[~]")
    cv2.putText(frame,f"{icon}  {var_label}",(20,133),
                cv2.FONT_HERSHEY_SIMPLEX,0.82,form_color,2,cv2.LINE_AA)

    return frame


def draw_ready(frame, ex, countdown_left=None):
    h,w=frame.shape[:2]
    ov=frame.copy()
    cv2.rectangle(ov,(0,0),(w,150),(0,0,0),-1)
    cv2.addWeighted(ov,0.65,frame,0.35,0,frame)
    cv2.putText(frame,ex["label"].upper(),(w//2-100,42),
                cv2.FONT_HERSHEY_SIMPLEX,1.1,(100,150,255),3,cv2.LINE_AA)
    cv2.putText(frame,ex["description"],(w//2-240,80),
                cv2.FONT_HERSHEY_SIMPLEX,0.56,(170,170,255),1,cv2.LINE_AA)
    vn=ex.get("var_names",{})
    vn_str="  ".join([f"V{i}:{vn[i]}" for i in range(1,4) if vn.get(i)])
    cv2.putText(frame,vn_str,(w//2-290,108),
                cv2.FONT_HERSHEY_SIMPLEX,0.38,(130,130,200),1,cv2.LINE_AA)
    if countdown_left is not None:
        cv2.putText(frame,f"Starting in {countdown_left}s... get into position",
                    (w//2-290,136),cv2.FONT_HERSHEY_SIMPLEX,0.70,(200,220,255),2,cv2.LINE_AA)
    else:
        cv2.putText(frame,"Press SPACE to begin  |  q to quit",
                    (w//2-215,136),cv2.FONT_HERSHEY_SIMPLEX,0.70,(200,200,255),2,cv2.LINE_AA)
    return frame


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def parse_args():
    p=argparse.ArgumentParser(description="AIGym v10",
                              formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--source",   default="0")
    p.add_argument("--exercise", default="squats", choices=list(EXERCISES.keys()))
    p.add_argument("--save",     action="store_true")
    p.add_argument("--output",   default="aigym_v10_output.avi")
    p.add_argument("--no-show",  action="store_true")
    p.add_argument("--model",    default=DEFAULT_MODEL)
    p.add_argument("--conf",     type=float, default=DEFAULT_CONF)
    p.add_argument("--iou",      type=float, default=DEFAULT_IOU)
    p.add_argument("--device",   default=None)
    return p.parse_args()


def normalize_device_arg(device):
    if device is None: return None
    d=str(device).strip().lower()
    if not d: return None
    if d in ("cpu","mps"): return d
    if d.startswith("cuda"): return d
    if d.isdigit(): return f"cuda:{d}"
    return device


# ──────────────────────────────────────────────────────────────────────────────
# Main loop
# ──────────────────────────────────────────────────────────────────────────────
def run(args):
    ex      = EXERCISES[args.exercise]
    checker = ex["checker"]
    allowed = EXERCISE_KP_MASK[args.exercise]
    vn      = ex.get("var_names",{})

    print(f"\n{'='*64}")
    print(f"  AIGym Pose Detection v10")
    print(f"  {ex['label']} - {ex['description']}")
    print(f"  V1:{vn.get(1,'--')}  V2:{vn.get(2,'--')}  V3:{vn.get(3,'--')}")
    print(f"  EvidenceGate: confirm={CONFIRM_FRAMES}f  clear={CLEAR_FRAMES}f")
    print(f"  SPACE=start/pause  q=quit+score\n")

    model = YOLO(args.model)
    resolved = normalize_device_arg(args.device)
    active_device = "auto"
    if resolved:
        try:
            model.to(resolved); active_device = resolved
        except (AssertionError, RuntimeError) as e:
            msg = str(e).lower()
            if str(resolved).lower().startswith("cuda") and "cuda" in msg:
                print(f"[WARN] {resolved} unavailable, falling back to CPU.")
                model.to("cpu"); active_device = "cpu"
            else: raise
    print(f"  Device: {active_device}")

    src = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open: {args.source}"); sys.exit(1)

    writer = None
    if args.save:
        fw=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); fh=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps=cap.get(cv2.CAP_PROP_FPS) or 30.0
        writer=cv2.VideoWriter(args.output,cv2.VideoWriter_fourcc(*"mp4v"),fps,(fw,fh))

    is_started = False
    start_countdown_until = None
    state      = make_state(ex)
    state["exercise_name"] = args.exercise
    kp_buf     = KeypointBuffer(buf=ex.get("kp_buffer_size",KP_BUFFER_SIZE),
                                max_ghost=ex.get("max_ghost_frames",MAX_GHOST_FRAMES))
    n_frames   = 0
    last_frame = None
    win_name   = f"AIGym v10 - {ex['label']}"

    if not args.no_show:
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win_name, 1280, 720)

    try:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok: break
            last_frame = frame.copy()

            if (not is_started and start_countdown_until is not None
                    and time.time() >= start_countdown_until):
                is_started = True
                start_countdown_until = None
                print("  [START]")
                state = make_state(ex)
                state["exercise_name"] = args.exercise
                state["session_start"] = time.time()
                kp_buf = KeypointBuffer(buf=ex.get("kp_buffer_size",KP_BUFFER_SIZE),
                                       max_ghost=ex.get("max_ghost_frames",MAX_GHOST_FRAMES))

            results = model.track(frame, persist=True,
                                  conf=args.conf, iou=args.iou, verbose=False)
            raw_pts, raw_conf = extract_kp(results[0]) if results else (None,None)

            if raw_pts is not None:
                pts, conf_arr = kp_buf.update(raw_pts, raw_conf)
            else:
                pts, conf_arr = None, None

            if pts is not None and is_started:
                assessment, skel_hl, form_lines, arcs = checker(pts, conf_arr, state)
            else:
                assessment = _no_pose.copy()
                skel_hl, form_lines, arcs = {}, [], []

            # Missing pose should force transition state, not stale correct.
            if (assessment.get("dominant_tag") is None
                    and assessment.get("primary_angle") is None):
                assessment["dominant_tag"] = "no_pose"
                assessment["evidence_tag"] = "no_pose"

            # Setup gate for rep-based exercises: keep bottom in Transitioning
            # until the user has entered the down zone at least once this set.
            if is_started and not ex.get("is_timed", False):
                angle = assessment.get("primary_angle")
                if (not state.get("setup_ready", False)
                        and angle is not None
                        and angle <= state["down_angle"] + state["tolerance"]):
                    state["setup_ready"] = True
                if not state.get("setup_ready", False) and assessment.get("dominant_tag") is None:
                    assessment["dominant_tag"] = "setup_pending"
                    assessment["evidence_tag"] = "setup_pending"
                    assessment["motion_cue"] = ("Get into position and start your first rep","info")
                elif (state.get("setup_ready", False)
                      and assessment.get("dominant_tag") is None
                      and angle is not None
                      and angle >= state["up_angle"] - state["tolerance"]):
                    # At start/top position, show transition rather than correct.
                    assessment["dominant_tag"] = "setup_pending"
                    assessment["evidence_tag"] = "setup_pending"
                    assessment["motion_cue"] = ("Ready position - begin next rep","info")

            # ── EvidenceGate: converts dominant_tag → stable form class ──
            # This is the ONLY place form_class_key is assigned.
            # Both HUD panels read from assessment → coherence is structural.
            cls_key, var_label = state["ev_gate"].update(assessment["dominant_tag"])
            assessment["form_class_key"]    = cls_key
            assessment["form_label"]        = var_label
            assessment["evidence_strength"] = state["ev_gate"].evidence_strength()

            # Sync primary_angle to state for update_reps / progress bar
            state["primary_angle"] = assessment["primary_angle"]

            if is_started:
                if ex["is_timed"]:
                    update_plank(state, cls_key == "correct")
                    if cls_key in ("var1","var2","var3"):
                        state["form_issues"].add(cls_key)
                else:
                    update_reps(state, cls_key)

                # Push-up var3 is rep-completion-based; surface it briefly on HUD
                # right after the rep closes so bottom header reflects the event.
                if (state.get("exercise_name") == "pushups"
                        and state.get("last_rep_cls") == "var3"
                        and (time.time() - state.get("last_rep_time", 0.0)) <= 1.0):
                    assessment["form_class_key"] = "var3"
                    assessment["form_label"] = ex.get("var_names", {}).get(3, "Insufficient Elbow Flexion")
                    if assessment.get("motion_cue", ("", "neutral"))[1] in ("good", "info"):
                        assessment["motion_cue"] = ("Insufficient depth last rep - go lower next rep", "warning")

            # Render
            annotated = frame.copy()
            if pts is not None:
                draw_skeleton(annotated, pts, conf_arr, allowed, skel_hl)
                if is_started:
                    for (pa,pb,lc,dash) in form_lines: draw_line(annotated,pa,pb,lc,dashed=dash)
                    for arc_args in arcs:              draw_arc(annotated,*arc_args)

            if is_started:
                annotated = draw_hud(annotated, assessment, state, ex)
            else:
                countdown_left = None
                if start_countdown_until is not None:
                    remaining = start_countdown_until - time.time()
                    countdown_left = max(1, int(math.ceil(remaining)))
                annotated = draw_ready(annotated, ex, countdown_left)

            if not args.no_show:
                cv2.imshow(win_name, annotated)
            if writer:
                writer.write(annotated)
            n_frames += 1

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord(" "):
                if is_started:
                    is_started = False
                    start_countdown_until = None
                    print("  [PAUSE]")
                else:
                    if start_countdown_until is None:
                        start_countdown_until = time.time() + START_COUNTDOWN_SEC
                        print(f"  [START IN {START_COUNTDOWN_SEC}]")
                    else:
                        start_countdown_until = None
                        print("  [COUNTDOWN CANCELED]")

    except KeyboardInterrupt:
        print("\n  Interrupted.")
    finally:
        cap.release()
        if writer: writer.release()

        if last_frame is not None:
            show_score_screen(last_frame, state["rep_log"], ex,
                              "AIGym v10 - Final Score", state)
        cv2.destroyAllWindows()

        print(f"\n  -- Final Score --")
        print(f"  Exercise : {ex['label']}")
        if ex.get("is_timed"):
            good_t = state["total_hold"]
            g = "A" if good_t>=60 else "B" if good_t>=45 else "C" if good_t>=30 else "D"
            print(f"  Good hold: {int(good_t)//60}:{int(good_t)%60:02d}  (Grade {g})")
            fi = state.get("form_issues",set())
            if fi:
                for key in ["var1","var2","var3"]:
                    if key in fi: print(f"    [!] {vn.get(int(key[-1]),key)}")
            else:
                print("    No form issues detected")
        else:
            c,t,pct,grade = compute_score(state["rep_log"])
            if t>0:
                print(f"  Reps     : {t}  |  Correct: {c}  |  {pct:.0f}%  (Grade {grade})")
                for key,label in ([("correct","Correct Form")]+
                                  [(f"var{i}",vn.get(i,"")) for i in range(1,4)]):
                    if not label: continue
                    n=Counter(state["rep_log"]).get(key,0)
                    if n: print(f"    {label:<36}: {n}")
            else:
                print("  No reps recorded - press SPACE before exercising")
        print(f"  Frames   : {n_frames}")
        print("  Done.\n")


if __name__ == "__main__":
    run(parse_args())