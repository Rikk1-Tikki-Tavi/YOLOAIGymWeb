"""
PoseSession
===========
One instance per WebSocket connection. Wraps the existing AIGym v10 checker
functions so the web backend reuses IDENTICAL form-detection logic as the
desktop app — no re-implementation, guaranteed parity.
"""

import math
import time
from collections import Counter

import numpy as np

# ── Import everything we need from the user's existing file ──────────────────
# Place your aigym_pose_detection.py (renamed to aigym_core.py) in this folder.
from aigym_core import (
    EXERCISES,
    KeypointBuffer,
    KP_BUFFER_SIZE,
    MAX_GHOST_FRAMES,
    KP_CONF_GHOST,
    EXERCISE_KP_MASK,
    make_state,
    extract_kp,
    update_reps,
    update_plank,
    compute_score,
    _no_pose,
)


class PoseSession:
    """
    Manages all per-connection state: keypoint buffer, smoothers,
    EvidenceGate, rep log, etc.
    """

    def __init__(self, exercise: str, dominant_foot: str = "L") -> None:
        key = exercise.lower().strip()
        if key not in EXERCISES:
            raise ValueError(
                f"Unknown exercise '{exercise}'. "
                f"Valid: {list(EXERCISES)}"
            )

        self.exercise = key
        self.ex       = EXERCISES[key]
        self.checker  = self.ex["checker"]
        self.allowed  = EXERCISE_KP_MASK.get(key, set(range(17)))

        self.state = make_state(self.ex)
        self.state["exercise_name"] = key
        self.state["session_start"] = time.time()
        if key == "lunges":
            self.state["dominant_foot"] = dominant_foot if dominant_foot in ("L", "R") else "L"

        self.kp_buf = KeypointBuffer(
            buf      = self.ex.get("kp_buffer_size",  KP_BUFFER_SIZE),
            max_ghost= self.ex.get("max_ghost_frames", MAX_GHOST_FRAMES),
        )

        # Web app starts immediately (countdown shown on frontend)
        self.is_started = True

    # ── Public API ────────────────────────────────────────────────────────────
    def extract_kp(self, result):
        """Thin wrapper so main.py doesn't need to import extract_kp directly."""
        return extract_kp(result)

    def process(self, pts, conf_arr) -> tuple[dict, dict]:
        """
        Run the exercise checker, EvidenceGate, and rep/hold counting.

        Returns
        -------
        assessment : dict   JSON-safe assessment data for the frontend HUD
        state_info : dict   JSON-safe session metrics (reps, hold time, etc.)
        """
        state = self.state
        ex    = self.ex

        # ── Run checker ───────────────────────────────────────────────────────
        if pts is not None and self.is_started:
            assessment, _skel_hl, _form_lines, _arcs = self.checker(
                pts, conf_arr, state
            )
        else:
            assessment = _no_pose.copy()
            assessment["dominant_tag"] = "no_pose"
            assessment["evidence_tag"] = "no_pose"

        # ── Setup gate (rep-based exercises only) ─────────────────────────────
        if not ex.get("is_timed", False):
            angle = assessment.get("primary_angle")
            if (not state.get("setup_ready", False)
                    and angle is not None
                    and angle <= state["down_angle"] + state["tolerance"]):
                state["setup_ready"] = True

            if not state.get("setup_ready", False) and assessment.get("dominant_tag") is None:
                assessment["dominant_tag"] = "setup_pending"
                assessment["evidence_tag"] = "setup_pending"
                assessment["motion_cue"]   = (
                    "Get into starting position and begin your first rep", "info"
                )
            elif (state.get("setup_ready", False)
                  and assessment.get("dominant_tag") is None
                  and angle is not None
                  and angle >= state["up_angle"] - state["tolerance"]):
                assessment["dominant_tag"] = "setup_pending"
                assessment["evidence_tag"] = "setup_pending"
                assessment["motion_cue"]   = ("Ready — begin next rep", "info")

        # ── EvidenceGate ─────────────────────────────────────────────────────
        cls_key, var_label = state["ev_gate"].update(assessment["dominant_tag"])
        assessment["form_class_key"]    = cls_key
        assessment["form_label"]        = var_label
        assessment["evidence_strength"] = state["ev_gate"].evidence_strength()
        state["primary_angle"]          = assessment["primary_angle"]

        # ── Rep / hold counting ───────────────────────────────────────────────
        if self.is_started:
            if ex["is_timed"]:
                update_plank(state, cls_key == "correct")
                if cls_key in ("var1", "var2", "var3"):
                    state["form_issues"].add(cls_key)
            else:
                update_reps(state, cls_key)

            # Push-up var3 (Insufficient Elbow Flexion) — surface briefly on HUD
            if (state.get("exercise_name") == "pushups"
                    and state.get("last_rep_cls") == "var3"
                    and (time.time() - state.get("last_rep_time", 0.0)) <= 1.0):
                assessment["form_class_key"] = "var3"
                assessment["form_label"] = ex.get("var_names", {}).get(
                    3, "Insufficient Elbow Flexion"
                )
                if assessment.get("motion_cue", ("", "neutral"))[1] in ("good", "info"):
                    assessment["motion_cue"] = (
                        "Insufficient depth last rep — go lower next rep", "warning"
                    )

        # ── Serialize ─────────────────────────────────────────────────────────
        mc = assessment.get("motion_cue") or ("", "neutral")

        assessment_out = {
            "motion_cue_text":     str(mc[0]),
            "motion_cue_severity": str(mc[1]),
            "form_class_key":      str(assessment["form_class_key"]),
            "form_label":          str(assessment["form_label"]),
            "primary_angle":       (
                round(float(assessment["primary_angle"]), 1)
                if assessment.get("primary_angle") is not None else None
            ),
            "dominant_tag":        assessment.get("dominant_tag"),
            "evidence_strength":   int(assessment.get("evidence_strength", 0)),
        }

        # Compute quick score for display
        correct, total, pct, grade = compute_score(state["rep_log"])
        rep_counts = Counter(state["rep_log"])
        hold_grade = None
        if ex.get("is_timed"):
            t = float(state.get("total_hold", 0.0))
            hold_grade = "A" if t >= 60 else "B" if t >= 45 else "C" if t >= 30 else "D"

        state_info = {
            "reps":            state.get("reps", 0),
            "stage":           state.get("stage", "up"),
            "is_timed":        bool(ex.get("is_timed", False)),
            "hold_time":       round(float(state.get("total_hold", 0.0)), 1),
            "exercise_label":  str(ex.get("label", "")),
            "correct_reps":    correct,
            "total_reps":      total,
            "score_pct":       round(pct, 0),
            "grade":           grade,
            "hold_grade":      hold_grade,
            "rep_breakdown": {
                "correct": rep_counts.get("correct", 0),
                "var1":    rep_counts.get("var1", 0),
                "var2":    rep_counts.get("var2", 0),
                "var3":    rep_counts.get("var3", 0),
            },
            "var_names":       ex.get("var_names", {}),
            "form_issues":     sorted(state.get("form_issues", set())),
            "last_rep_cls":    state.get("last_rep_cls"),
        }

        return assessment_out, state_info
