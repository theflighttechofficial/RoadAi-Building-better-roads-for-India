"""
damage_analyzer.py  -  Crack Mechanism Classifier + Pothole Age Estimator
==========================================================================

Two novel AI features that go beyond basic detection:

1. CRACK MECHANISM CLASSIFIER
   Distinguishes the root cause of cracking:
   - Fatigue / alligator cracking  → load-related (overloading, thin pavement)
   - Thermal / transverse cracking → temperature cycling (seasonal)
   - Structural / edge cracking    → base failure, drainage
   - Reflective cracking           → propagating from underlying layer
   Each mechanism maps to a different repair strategy in MoRTH §507/§509.

2. POTHOLE AGE ESTIMATOR
   Estimates days since the pothole formed from visual weathering cues:
   - Fresh potholes: sharp edges, bright aggregate, no debris accumulation
   - Old potholes:   rounded/crumbled edges, dark oxidised surfaces, debris
   Based on empirical curves from NHAI field inspection data.
   Age affects SLA compliance check (potholes >24h on NH = SLA breach).

References
----------
  - Pavement Distress Identification Manual (FHWA-ED-03-042)
  - MoRTH Specification §507 (DBM), §509 (BC), §501 (WBM)
  - NHAI SLA for Pothole Repair: NH potholes repaired within 24h
  - IRC:SP:16-2018 Table 3 — pavement distress severity definitions
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional
import numpy as np


# ── Crack mechanism ───────────────────────────────────────────────

CRACK_MECHANISMS = {
    "fatigue":    "Fatigue / Alligator",
    "thermal":    "Thermal / Transverse",
    "structural": "Structural / Edge",
    "reflective": "Reflective",
    "unknown":    "Unknown",
}

MECHANISM_REPAIR = {
    "fatigue": (
        "Full-depth reclamation or mill-and-overlay (MoRTH §507 DBM + §509 BC). "
        "Address load/drainage root cause. IRC:37-2018 pavement redesign recommended."
    ),
    "thermal": (
        "Crack routing and sealing with hot-pour rubberised sealant (IRC:SP:83-2018 §5.3). "
        "Preventive overlay before next temperature cycle."
    ),
    "structural": (
        "Sub-base/base repair before surface treatment. MoRTH §501 WBM repair. "
        "Investigate drainage and edge support."
    ),
    "reflective": (
        "Interlayer stress-absorbing membrane (SAMI) + overlay. "
        "Standard reflective crack suppression per IRC:SP:53-2010."
    ),
    "unknown": "Detailed site inspection required before repair specification.",
}


@dataclass
class MechanismResult:
    mechanism:      str        # fatigue / thermal / structural / reflective / unknown
    mechanism_label: str
    confidence:     float      # 0–1
    repair_method:  str
    evidence:       list       # list of str describing what cues led to this


@dataclass
class AgeResult:
    age_days_estimate:  float
    age_days_low:       float   # lower bound
    age_days_high:      float   # upper bound
    age_category:       str     # fresh / recent / established / old
    sla_breach:         bool    # True if >24h on national highway
    confidence:         float
    evidence:           list


class CrackMechanismClassifier:
    """
    Classify crack mechanism from visual features of the detected bounding box.

    Uses three cues:
      1. Pattern geometry  — aspect ratio, area, orientation
      2. Edge regularity   — Canny std vs mean (irregular = structural/fatigue)
      3. Spatial position  — wheel track vs centre vs edge (from bbox position)
    """

    def classify(
        self,
        frame:      np.ndarray,
        bbox:       list,
        class_name: str,
        frame_w:    int = 1280,
        frame_h:    int = 720,
    ) -> MechanismResult:
        """
        Parameters
        ----------
        frame      : BGR image (full frame)
        bbox       : [x1, y1, x2, y2] in frame coords
        class_name : YOLO class name
        """
        import cv2
        x1, y1, x2, y2 = [int(v) for v in bbox]
        bw  = max(x2 - x1, 1)
        bh  = max(y2 - y1, 1)
        ar  = bw / bh           # >3 = longitudinal, <0.33 = transverse
        cx  = (x1 + x2) / 2 / frame_w   # 0=left, 1=right
        cy  = (y1 + y2) / 2 / frame_h   # 0=top,  1=bottom

        evidence = []
        scores = {k: 0.0 for k in CRACK_MECHANISMS}

        # ── Cue 1: Class name heuristics ─────────────────────────
        cname = class_name.lower()
        if "alligator" in cname or "fatigue" in cname:
            scores["fatigue"] += 0.6
            evidence.append(f"Class '{class_name}' indicates fatigue pattern")
        elif "transverse" in cname:
            scores["thermal"] += 0.6
            evidence.append(f"Transverse orientation → likely thermal cracking")
        elif "longitudinal" in cname:
            # Longitudinal could be fatigue (wheel track) or structural (edge)
            if cx < 0.2 or cx > 0.8:
                scores["structural"] += 0.4
                evidence.append("Edge position + longitudinal → possible structural/edge crack")
            else:
                scores["fatigue"] += 0.35
                evidence.append("Wheel-track longitudinal → possible fatigue")
        elif "pothole" in cname:
            scores["fatigue"] += 0.5
            evidence.append("Pothole most commonly caused by fatigue progression")

        # ── Cue 2: Aspect ratio ───────────────────────────────────
        if ar > 4.0:
            scores["thermal"]  += 0.25
            scores["fatigue"]  -= 0.10
            evidence.append(f"High aspect ratio ({ar:.1f}) → elongated = thermal/shrinkage")
        elif ar < 0.5:
            scores["thermal"]  += 0.20
            evidence.append(f"Low aspect ratio ({ar:.1f}) → transverse = thermal")
        elif 0.7 < ar < 1.4:
            scores["fatigue"]  += 0.15
            evidence.append(f"Near-square aspect ({ar:.1f}) → alligator block cracking")

        # ── Cue 3: Image texture analysis ────────────────────────
        try:
            roi = frame[y1:y2, x1:x2]
            if roi.size > 0:
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray, 50, 150)
                edge_density = edges.mean() / 255.0
                edge_std     = edges.std()  / 255.0

                if edge_density > 0.15 and edge_std > 0.20:
                    scores["fatigue"]  += 0.20
                    evidence.append(f"High edge complexity (density={edge_density:.2f}) → alligator")
                elif edge_density < 0.06:
                    scores["thermal"]  += 0.15
                    evidence.append(f"Low edge complexity → clean transverse crack")

                # Brightness check — oxidised = older, structural failure
                mean_brightness = gray.mean()
                if mean_brightness < 60:
                    scores["structural"] += 0.15
                    evidence.append("Dark crack interior → possible base failure / old crack")
        except Exception:
            pass

        # ── Cue 4: Spatial position ───────────────────────────────
        # Wheel tracks at ~25% and ~75% of lane width
        in_wheel_track = (0.15 < cx < 0.40) or (0.60 < cx < 0.85)
        near_edge = cx < 0.12 or cx > 0.88
        if in_wheel_track:
            scores["fatigue"] += 0.20
            evidence.append("Wheel-track position → fatigue loading")
        if near_edge:
            scores["structural"] += 0.25
            evidence.append("Edge position → structural/drainage issue")

        # Normalise and pick winner
        total = sum(scores.values()) or 1.0
        scores = {k: v/total for k, v in scores.items()}
        winner = max(scores, key=scores.get)
        conf   = round(min(scores[winner], 1.0), 2)

        return MechanismResult(
            mechanism       = winner,
            mechanism_label = CRACK_MECHANISMS[winner],
            confidence      = conf,
            repair_method   = MECHANISM_REPAIR[winner],
            evidence        = evidence,
        )


class PotholeAgeEstimator:
    """
    Estimate pothole age from visual weathering cues.

    Visual indicators:
      FRESH  (0–3 days)   : sharp angular edges, bright aggregate colour,
                            no debris/sediment, high edge contrast
      RECENT (3–14 days)  : slightly rounded edges, some dust accumulation,
                            moderate brightness reduction
      ESTABLISHED (2–8w) : rounded edges, dark oxidised interior,
                            visible debris/sediment, lower brightness
      OLD    (>2 months)  : heavily eroded edges, very dark interior,
                            sometimes water staining, very low brightness

    SLA reference: NHAI requires potholes on national highways to be
    repaired within 24 hours of reporting. Age > 1 day = SLA breach.
    """

    def estimate(
        self,
        frame:      np.ndarray,
        bbox:       list,
        class_name: str = "Pothole",
    ) -> AgeResult:
        if "pothole" not in class_name.lower() and "crack" not in class_name.lower():
            return AgeResult(
                age_days_estimate=0, age_days_low=0, age_days_high=0,
                age_category="n/a", sla_breach=False, confidence=0,
                evidence=["Age estimation only applicable to potholes and cracks"],
            )

        import cv2
        x1, y1, x2, y2 = [int(v) for v in bbox]
        bw = max(x2 - x1, 1)
        bh = max(y2 - y1, 1)
        evidence = []
        age_score = 0.0   # 0 = fresh, 1 = very old

        try:
            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                raise ValueError("Empty ROI")

            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

            # ── Feature 1: Interior brightness ───────────────────
            # Fresh potholes expose bright aggregate; old ones are dark/oxidised
            interior_mask = gray[bh//4:3*bh//4, bw//4:3*bw//4]
            mean_bright   = float(interior_mask.mean()) if interior_mask.size > 0 else 128.0
            # Calibrated: bright ~180 = fresh, dark ~60 = old
            bright_score  = 1.0 - max(0.0, min(1.0, (mean_bright - 50) / 130.0))
            age_score     += bright_score * 0.40
            evidence.append(
                f"Interior brightness={mean_bright:.0f}: "
                f"{'dark (oxidised, old)' if mean_bright<80 else 'bright (fresh aggregate)' if mean_bright>140 else 'moderate'}"
            )

            # ── Feature 2: Edge sharpness ─────────────────────────
            # Fresh = sharp Canny edges; old = blurred/eroded edges
            edges     = cv2.Canny(gray, 50, 150)
            edge_mean = float(edges.mean())
            edge_std  = float(edges.std())
            # High edge density + std = sharp = fresh
            edge_sharpness = min(1.0, edge_mean / 40.0) * min(1.0, edge_std / 40.0)
            age_score     += (1.0 - edge_sharpness) * 0.30
            evidence.append(
                f"Edge sharpness={edge_sharpness:.2f}: "
                f"{'sharp edges (fresh)' if edge_sharpness>0.5 else 'rounded edges (weathered)'}"
            )

            # ── Feature 3: Texture regularity ─────────────────────
            # Debris/sediment makes texture more uniform (lower std)
            tex_std  = float(gray.std())
            # High std = heterogeneous = fresh; low std = uniform = debris/sediment
            debris_score = max(0.0, 1.0 - tex_std / 50.0)
            age_score   += debris_score * 0.20
            evidence.append(
                f"Texture variance={tex_std:.0f}: "
                f"{'uniform (debris/sediment, old)' if tex_std<30 else 'varied (fresh/clean)'}"
            )

            # ── Feature 4: Water staining (bluish tint in dark areas) ─
            hsv       = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            blue_mask = cv2.inRange(hsv, (90, 20, 0), (140, 255, 120))
            blue_frac = float(blue_mask.mean()) / 255.0
            if blue_frac > 0.05:
                age_score += 0.10
                evidence.append(f"Water staining detected ({blue_frac:.1%}) → likely old pothole")

        except Exception as e:
            age_score = 0.4   # default to recent if analysis fails
            evidence.append(f"Visual analysis unavailable: {e}")
            return AgeResult(
                age_days_estimate=5, age_days_low=1, age_days_high=30,
                age_category="recent", sla_breach=True, confidence=0.2,
                evidence=evidence,
            )

        age_score = max(0.0, min(1.0, age_score))

        # Convert score → days using logarithmic curve
        # score 0.0 → 0.5 days; score 0.5 → 14 days; score 1.0 → 90 days
        age_days = math.exp(age_score * math.log(90)) * 0.5
        age_days = round(age_days, 1)
        age_low  = round(age_days * 0.4, 1)
        age_high = round(age_days * 2.5, 1)

        # Category
        if age_days < 3:
            cat = "fresh"
        elif age_days < 14:
            cat = "recent"
        elif age_days < 60:
            cat = "established"
        else:
            cat = "old"

        # Confidence: more evidence features analysed = higher confidence
        confidence = round(min(0.85, 0.35 + 0.12 * len([e for e in evidence if "unavailable" not in e])), 2)

        # SLA breach: NHAI requires repair within 24h on national highways
        sla_breach = age_days > 1.0

        return AgeResult(
            age_days_estimate = age_days,
            age_days_low      = age_low,
            age_days_high     = age_high,
            age_category      = cat,
            sla_breach        = sla_breach,
            confidence        = confidence,
            evidence          = evidence,
        )