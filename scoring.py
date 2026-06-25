"""
scoring.py — Road Health Scoring Engine

Converts raw YOLO detections into a calibrated 0–100 road health score
using four weighted penalty factors:

  1. Area severity    — how large each damage region is relative to frame
  2. Confidence       — model certainty acts as a damage confidence weight
  3. Class severity   — different damage types carry different base penalties
  4. Density          — penalises frames crowded with many overlapping faults

Additionally tracks per-session trend data, emits structured ScoreResult
objects, and can aggregate scores across multiple frames into a segment-level
health report.
"""

from __future__ import annotations

import math
import logging
import numpy as np
from dataclasses import dataclass, field
from collections import deque
from typing import Optional

log = logging.getLogger(__name__)


# ── Class severity table ──────────────────────────────────────────────────────
# Maps detection class_id → base severity weight (0.0–1.0).
# Potholes are structurally dangerous; surface cracks less so.
# Override via RoadScorer(class_severity={...}).

DEFAULT_CLASS_SEVERITY: dict[int, float] = {
    0: 0.55,   # Longitudinal Crack  — structural risk, but gradual
    1: 0.60,   # Transverse Crack    — water ingress risk
    2: 0.80,   # Alligator Crack     — base failure indicator
    3: 0.95,   # Pothole             — immediate safety hazard
}

DEFAULT_CLASS_NAMES: dict[int, str] = {
    0: "Longitudinal Crack",
    1: "Transverse Crack",
    2: "Alligator Crack",
    3: "Pothole",
}

# Priority bands
PRIORITY_BANDS = [
    (80,  "Low",      "Routine monitoring — schedule next inspection in 6 months."),
    (60,  "Medium",   "Preventive repair — patch within 30 days."),
    (40,  "High",     "Urgent repair — address within 1 week."),
    (0,   "Critical", "Immediate closure risk — repair within 24 hours."),
]


# ── Result objects ────────────────────────────────────────────────────────────

@dataclass
class DetectionScore:
    """Per-detection breakdown for explainability."""
    class_id:        int
    class_name:      str
    confidence:      float
    area_px:         int
    area_norm:       float          # 0–1 relative to frame
    class_weight:    float
    combined_penalty: float         # final penalty contribution 0–1

    def to_dict(self) -> dict:
        return {
            "class":            self.class_name,
            "confidence":       round(self.confidence, 3),
            "area_norm":        round(self.area_norm, 4),
            "class_weight":     self.class_weight,
            "combined_penalty": round(self.combined_penalty, 4),
        }


@dataclass
class ScoreResult:
    """Full scoring output for one frame."""
    health_score:      float                    # 0–100
    priority:          str
    priority_message:  str
    n_detections:      int
    penalty_area:      float                    # 0–1 contribution
    penalty_conf:      float
    penalty_class:     float
    penalty_density:   float
    final_penalty:     float                    # weighted aggregate 0–1
    per_detection:     list[DetectionScore] = field(default_factory=list)
    frame_index:       Optional[int] = None

    @property
    def band(self) -> str:
        if self.health_score >= 80: return "good"
        if self.health_score >= 60: return "moderate"
        if self.health_score >= 40: return "poor"
        return "critical"

    @property
    def band_color_ansi(self) -> str:
        return {
            "good":     "\033[92m",
            "moderate": "\033[93m",
            "poor":     "\033[91m",
            "critical": "\033[95m",
        }[self.band]

    def to_dict(self) -> dict:
        return {
            "frame_index":    self.frame_index,
            "health_score":   self.health_score,
            "band":           self.band,
            "priority":       self.priority,
            "priority_message": self.priority_message,
            "n_detections":   self.n_detections,
            "penalties": {
                "area":    round(self.penalty_area,    4),
                "conf":    round(self.penalty_conf,    4),
                "class":   round(self.penalty_class,   4),
                "density": round(self.penalty_density, 4),
                "final":   round(self.final_penalty,   4),
            },
            "per_detection": [d.to_dict() for d in self.per_detection],
        }

    def __str__(self) -> str:
        rst = "\033[0m"
        c   = self.band_color_ansi
        return (
            f"{c}Score {self.health_score:>6.2f}/100  "
            f"[{self.band.upper():^8}]  "
            f"{self.priority} priority  "
            f"({self.n_detections} detections){rst}"
        )


# ── Segment aggregator ────────────────────────────────────────────────────────

@dataclass
class SegmentReport:
    """
    Aggregates ScoreResults across many frames into a road-segment summary.
    Use scorer.aggregate([result1, result2, ...]) to build one.
    """
    n_frames:           int
    avg_score:          float
    min_score:          float
    max_score:          float
    std_score:          float
    priority:           str
    priority_message:   str
    total_detections:   int
    class_breakdown:    dict[str, int]
    score_trend:        str           # "improving" | "stable" | "worsening"
    flagged_frames:     list[int]     # frame indices below 50

    def to_dict(self) -> dict:
        return {
            "n_frames":         self.n_frames,
            "avg_score":        round(self.avg_score, 2),
            "min_score":        round(self.min_score, 2),
            "max_score":        round(self.max_score, 2),
            "std_score":        round(self.std_score, 2),
            "priority":         self.priority,
            "priority_message": self.priority_message,
            "total_detections": self.total_detections,
            "class_breakdown":  self.class_breakdown,
            "score_trend":      self.score_trend,
            "flagged_frames":   self.flagged_frames,
        }

    def __str__(self) -> str:
        trend_sym = {"improving": "↑", "stable": "→", "worsening": "↓"}[self.score_trend]
        sep = "─" * 50
        lines = [
            f"\n{sep}",
            f"  SEGMENT HEALTH REPORT",
            f"{sep}",
            f"  Frames analysed  : {self.n_frames}",
            f"  Avg health score : {self.avg_score:.2f}/100  {trend_sym} {self.score_trend}",
            f"  Score range      : {self.min_score:.1f} – {self.max_score:.1f}  (σ={self.std_score:.1f})",
            f"  Priority         : {self.priority}",
            f"  Total detections : {self.total_detections}",
        ]
        if self.class_breakdown:
            lines.append(f"\n  Class breakdown:")
            for cls, cnt in sorted(self.class_breakdown.items(), key=lambda x: -x[1]):
                bar = "█" * min(cnt, 35)
                lines.append(f"    {cls:<22} {bar} {cnt}")
        if self.flagged_frames:
            lines.append(f"\n  ⚠  Flagged frames (score < 50): {len(self.flagged_frames)}")
            lines.append(f"     {self.flagged_frames[:10]}" +
                         (" …" if len(self.flagged_frames) > 10 else ""))
        lines.append(sep)
        return "\n".join(lines)


# ── Core scorer ───────────────────────────────────────────────────────────────

class RoadScorer:
    """
    Multi-factor road health scorer.

    Parameters
    ----------
    frame_w, frame_h : int
        Frame dimensions used to normalise bounding box area.
        If unknown, leave as None — area factor will use a fixed divisor.
    class_severity : dict
        Override DEFAULT_CLASS_SEVERITY with your own class → weight mapping.
    class_names : dict
        class_id → human label mapping.
    weights : dict
        Relative weights of the four penalty factors.
        Must contain keys: area, conf, class_, density.
    history_len : int
        Number of recent frame scores to keep for trend detection.
    """

    DEFAULT_WEIGHTS = {
        "area":    0.30,   # size of damage regions
        "conf":    0.25,   # model confidence (proxy for damage certainty)
        "class_":  0.30,   # damage type severity
        "density": 0.15,   # number of simultaneous detections
    }

    def __init__(
        self,
        frame_w: Optional[int] = None,
        frame_h: Optional[int] = None,
        class_severity: Optional[dict] = None,
        class_names:    Optional[dict] = None,
        weights:        Optional[dict] = None,
        history_len:    int = 120,
    ):
        self.frame_area    = (frame_w * frame_h) if (frame_w and frame_h) else None
        self.class_severity = {**DEFAULT_CLASS_SEVERITY, **(class_severity or {})}
        self.class_names    = {**DEFAULT_CLASS_NAMES,    **(class_names    or {})}
        self.weights        = {**self.DEFAULT_WEIGHTS,   **(weights        or {})}
        self._normalise_weights()

        self._history: deque[float] = deque(maxlen=history_len)
        self._frame_counter = 0

        log.debug(f"RoadScorer init — weights={self.weights}  "
                  f"frame_area={self.frame_area}")

    # ── Public API ────────────────────────────────────────────────────────────

    def score_frame(
        self,
        detections: list[dict],
        frame_index: Optional[int] = None,
    ) -> ScoreResult:
        """
        Score a single frame.

        Parameters
        ----------
        detections : list of dicts
            Each dict must have:
              bbox       : [x1, y1, x2, y2]
              confidence : float
              class      : int   (class_id)
        frame_index : int, optional
            Stored in the result for traceability.

        Returns
        -------
        ScoreResult
        """
        idx = frame_index if frame_index is not None else self._frame_counter
        self._frame_counter += 1

        if not detections:
            result = self._perfect_score(idx)
            self._history.append(result.health_score)
            return result

        per_det, p_area, p_conf, p_class = self._score_detections(detections)
        p_density = self._density_penalty(len(detections))

        # Weighted aggregate
        final_penalty = (
            self.weights["area"]    * p_area    +
            self.weights["conf"]    * p_conf    +
            self.weights["class_"]  * p_class   +
            self.weights["density"] * p_density
        )
        final_penalty = float(np.clip(final_penalty, 0.0, 1.0))

        health_score = round(max(0.0, 100.0 - final_penalty * 100.0), 2)
        priority, message = self._priority(health_score)

        result = ScoreResult(
            health_score=health_score,
            priority=priority,
            priority_message=message,
            n_detections=len(detections),
            penalty_area=p_area,
            penalty_conf=p_conf,
            penalty_class=p_class,
            penalty_density=p_density,
            final_penalty=final_penalty,
            per_detection=per_det,
            frame_index=idx,
        )

        self._history.append(health_score)
        return result

    # Convenience alias
    def compute_score(self, detections: list[dict]) -> float:
        """Drop-in replacement for the original API — returns score float."""
        return self.score_frame(detections).health_score

    def get_priority(self, score: float) -> str:
        """Drop-in replacement for the original API."""
        return self._priority(score)[0]

    def aggregate(self, results: list[ScoreResult]) -> SegmentReport:
        """
        Aggregate a list of ScoreResults into a SegmentReport.

        Typical use:
            results = [scorer.score_frame(dets, i) for i, dets in enumerate(frames)]
            report  = scorer.aggregate(results)
        """
        if not results:
            raise ValueError("Cannot aggregate empty result list.")

        scores  = [r.health_score for r in results]
        avg     = float(np.mean(scores))
        priority, message = self._priority(avg)

        class_breakdown: dict[str, int] = {}
        total_det = 0
        for r in results:
            total_det += r.n_detections
            for d in r.per_detection:
                class_breakdown[d.class_name] = class_breakdown.get(d.class_name, 0) + 1

        flagged = [r.frame_index for r in results
                   if r.frame_index is not None and r.health_score < 50]

        return SegmentReport(
            n_frames=len(results),
            avg_score=avg,
            min_score=float(np.min(scores)),
            max_score=float(np.max(scores)),
            std_score=float(np.std(scores)),
            priority=priority,
            priority_message=message,
            total_detections=total_det,
            class_breakdown=class_breakdown,
            score_trend=self._trend(scores),
            flagged_frames=flagged,
        )

    @property
    def rolling_average(self) -> Optional[float]:
        """Mean health score over the recent history window."""
        if not self._history:
            return None
        return round(float(np.mean(self._history)), 2)

    @property
    def trend(self) -> str:
        return self._trend(list(self._history))

    def reset_history(self) -> None:
        self._history.clear()
        self._frame_counter = 0

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _score_detections(
        self, detections: list[dict]
    ) -> tuple[list[DetectionScore], float, float, float]:
        """Returns (per_det list, area_penalty, conf_penalty, class_penalty)."""

        area_penalties  = []
        conf_penalties  = []
        class_penalties = []
        per_det         = []

        for d in detections:
            x1, y1, x2, y2 = d["bbox"]
            w    = max(x2 - x1, 0)
            h    = max(y2 - y1, 0)
            area = w * h
            conf = float(np.clip(d.get("confidence", 0.5), 0.0, 1.0))
            cls  = int(d.get("class", -1))

            # Area factor — normalised to frame if dimensions known
            if self.frame_area:
                area_norm = min(area / self.frame_area, 1.0)
                # Non-linear scaling: small damage should have less weight
                area_sev = 1.0 - math.exp(-5.0 * area_norm)
            else:
                area_norm = min(area / 80_000.0, 1.0)
                area_sev  = 1.0 - math.exp(-5.0 * area_norm)

            # Confidence factor — high confidence → reliable penalty
            conf_sev = conf  # directly use as weight

            # Class factor
            class_w = self.class_severity.get(cls, 0.60)

            # Combined per-detection penalty (geometric: all factors must be high)
            combined = area_sev * conf_sev * class_w

            area_penalties.append(area_sev)
            conf_penalties.append(conf_sev)
            class_penalties.append(class_w * conf_sev)  # weighted class penalty

            per_det.append(DetectionScore(
                class_id=cls,
                class_name=self.class_names.get(cls, f"class_{cls}"),
                confidence=conf,
                area_px=area,
                area_norm=area_norm,
                class_weight=class_w,
                combined_penalty=combined,
            ))

        # Aggregate — use 90th percentile to avoid single outlier dominating
        p_area  = float(np.percentile(area_penalties,  90))
        p_conf  = float(np.mean(conf_penalties))
        p_class = float(np.percentile(class_penalties, 90))

        return per_det, p_area, p_conf, p_class

    def _density_penalty(self, n: int, soft_cap: int = 8) -> float:
        """
        Sigmoid-shaped penalty that saturates around `soft_cap` detections.
        1 detection → ~0.12   4 → ~0.46   8 → ~0.73   15 → ~0.93
        """
        return float(1.0 / (1.0 + math.exp(-0.7 * (n - soft_cap / 2))))

    @staticmethod
    def _priority(score: float) -> tuple[str, str]:
        for threshold, label, message in PRIORITY_BANDS:
            if score >= threshold:
                return label, message
        return PRIORITY_BANDS[-1][1], PRIORITY_BANDS[-1][2]

    @staticmethod
    def _trend(scores: list[float], window: int = 10) -> str:
        if len(scores) < window * 2:
            return "stable"
        first_half = np.mean(scores[:window])
        last_half  = np.mean(scores[-window:])
        diff = last_half - first_half
        if diff > 3.0:  return "improving"
        if diff < -3.0: return "worsening"
        return "stable"

    def _perfect_score(self, idx: int) -> ScoreResult:
        return ScoreResult(
            health_score=100.0,
            priority="Low",
            priority_message=PRIORITY_BANDS[0][2],
            n_detections=0,
            penalty_area=0.0,
            penalty_conf=0.0,
            penalty_class=0.0,
            penalty_density=0.0,
            final_penalty=0.0,
            per_detection=[],
            frame_index=idx,
        )

    def _normalise_weights(self) -> None:
        total = sum(self.weights.values())
        if total <= 0:
            raise ValueError("Scoring weights must sum to a positive value.")
        self.weights = {k: v / total for k, v in self.weights.items()}


# ── Convenience factory ────────────────────────────────────────────────────────

def make_scorer(frame_w: int, frame_h: int, **kwargs) -> RoadScorer:
    """Shorthand: scorer = make_scorer(1920, 1080)"""
    return RoadScorer(frame_w=frame_w, frame_h=frame_h, **kwargs)


# ── Quick demo ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json

    scorer = make_scorer(frame_w=1280, frame_h=720)

    sample_frames = [
        [],  # clean frame
        [
            {"bbox": [100, 200, 250, 320], "confidence": 0.91, "class": 3},  # pothole
            {"bbox": [400, 100, 500, 160], "confidence": 0.65, "class": 1},  # transverse
        ],
        [
            {"bbox": [50,  50,  600, 400], "confidence": 0.88, "class": 2},  # alligator (large)
            {"bbox": [200, 300, 400, 450], "confidence": 0.77, "class": 3},  # pothole
            {"bbox": [700, 100, 900, 200], "confidence": 0.55, "class": 0},  # longitudinal
        ],
        [
            {"bbox": [10, 10, 80, 50], "confidence": 0.40, "class": 0},   # minor crack
        ],
    ]

    results = []
    print("\n── Per-frame results ──────────────────────────────")
    for i, dets in enumerate(sample_frames):
        res = scorer.score_frame(dets, frame_index=i)
        results.append(res)
        print(f"  Frame {i}: {res}")
        if res.per_detection:
            for d in res.per_detection:
                print(f"    └─ {d.class_name:<22} conf={d.confidence:.2f}  "
                      f"area_norm={d.area_norm:.4f}  penalty={d.combined_penalty:.4f}")

    print(f"\n  Rolling avg : {scorer.rolling_average}")
    print(f"  Trend       : {scorer.trend}")

    report = scorer.aggregate(results)
    print(report)
    print("\n── JSON export (first frame) ──────────────────────")
    print(json.dumps(results[1].to_dict(), indent=2))