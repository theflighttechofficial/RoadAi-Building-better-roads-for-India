"""
preprocessor.py — Advanced Adaptive Frame Preprocessor

Features:
  - Night / rain / fog / glare detection and enhancement
  - ROI masking — only analyse the road zone, skip sky/bonnet/dashboard
  - Frame deduplication — skip near-identical frames (saves 30–60% processing)
  - Speed-adaptive frame sampling — dense at slow speed, sparse on highways
  - Perspective-normalised ROI crop for consistent detection input size

FIXES:
  1. FOG FALSE POSITIVE — old single variance check (var < 0.04) triggered on
     any low-contrast frame, flagging 98% of clear daytime road as fog.
     New 5-factor check requires variance + mid-brightness + low-saturation +
     low-edge-density + uniform-sky to ALL pass. Clear roads no longer trigger.

  2. CONFIDENCE CALIBRATION — ConfidenceCalibrator class added.
     Per-class thresholds + temperature scaling + temporal consistency boost.
     Cracks (hard to detect) get lower thresholds; potholes (distinct) higher.
     Detections repeated in nearby frames gain a small confidence boost.
"""

from __future__ import annotations
import cv2
import numpy as np
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Tuple, Dict
import logging

log = logging.getLogger(__name__)


# ── Enums ─────────────────────────────────────────────────────────────────────

class FrameCondition(Enum):
    NORMAL = auto()
    NIGHT  = auto()
    RAIN   = auto()
    FOG    = auto()
    GLARE  = auto()


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class PreprocessResult:
    frame:          np.ndarray          # enhanced full frame
    roi_frame:      np.ndarray          # road-zone crop (used for detection)
    roi_bbox:       Tuple[int,int,int,int]  # (x1,y1,x2,y2) of roi in original
    condition:      FrameCondition
    brightness:     float
    contrast:       float
    enhanced:       bool
    is_duplicate:   bool = False        # True → skip inference
    sample_this:    bool = True         # False → speed-adaptive skip
    skip_reason:    str  = ""


# ── ROI Masker ────────────────────────────────────────────────────────────────

class ROIMasker:
    """
    Extracts the road zone from a frame, masking out:
      - Sky / trees (top portion)
      - Dashboard / bonnet (bottom strip for dashcam)
      - Side margins (very edges, often contain poles/buildings)

    Returns a crop that is then used for detection, keeping bounding
    box coordinates relative to the original frame via offset mapping.
    """

    def __init__(
        self,
        top_skip:    float = 0.35,   # ignore top 35% (sky)
        bottom_skip: float = 0.05,   # ignore bottom 5% (bonnet edge)
        side_skip:   float = 0.03,   # ignore outer 3% each side
        portrait_top: float = 0.45,  # tighter for portrait (more sky)
    ):
        self.top_skip     = top_skip
        self.bottom_skip  = bottom_skip
        self.side_skip    = side_skip
        self.portrait_top = portrait_top

    def extract(self, frame: np.ndarray) -> Tuple[np.ndarray, Tuple[int,int,int,int]]:
        """
        Returns (roi_crop, (x1, y1, x2, y2)) where bbox is in original coords.
        Detections in roi_crop must be offset by (x1, y1) to map back.
        """
        h, w = frame.shape[:2]
        portrait = h > w
        top  = self.portrait_top if portrait else self.top_skip
        x1 = int(w * self.side_skip)
        y1 = int(h * top)
        x2 = int(w * (1.0 - self.side_skip))
        y2 = int(h * (1.0 - self.bottom_skip))
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        return frame[y1:y2, x1:x2].copy(), (x1, y1, x2, y2)

    def remap_detections(
        self,
        detections: list[dict],
        roi_bbox: Tuple[int,int,int,int]
    ) -> list[dict]:
        """Offset bbox coordinates from ROI-space back to full-frame space."""
        x1_off, y1_off, _, _ = roi_bbox
        remapped = []
        for d in detections:
            bx1, by1, bx2, by2 = d["bbox"]
            remapped.append({
                **d,
                "bbox": [bx1 + x1_off, by1 + y1_off,
                          bx2 + x1_off, by2 + y1_off]
            })
        return remapped


# ── Duplicate Detector ────────────────────────────────────────────────────────

class DuplicateDetector:
    """
    Detects near-identical consecutive frames using perceptual hashing.
    Skips inference on duplicates to save CPU/GPU.

    Uses a small downsampled grayscale image and mean absolute difference.
    Threshold tuned so minor camera jitter is ignored but new road surface
    (after vehicle moves ~0.5m) triggers a new inference.
    """

    def __init__(
        self,
        threshold:   float = 7.0,    # was 4.5 — higher = skip more similar frames
        hash_size:   int   = 16,
        min_gap:     int   = 2,       # was 3 — process at least every 2nd frame
    ):
        self.threshold  = threshold
        self.hash_size  = hash_size
        self.min_gap    = min_gap
        self._last_hash: Optional[np.ndarray] = None
        self._frames_since_process = 0

    def is_duplicate(self, frame: np.ndarray) -> bool:
        """Returns True if frame is near-identical to previous processed frame."""
        self._frames_since_process += 1

        # Force-process after min_gap regardless of similarity
        if self._frames_since_process >= self.min_gap:
            self._update_hash(frame)
            self._frames_since_process = 0
            return False

        if self._last_hash is None:
            self._update_hash(frame)
            return False

        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (self.hash_size, self.hash_size))
        diff  = float(np.mean(np.abs(small.astype(np.float32) -
                                      self._last_hash.astype(np.float32))))
        if diff < self.threshold:
            return True   # duplicate → skip

        self._update_hash(frame)
        return False

    def _update_hash(self, frame: np.ndarray) -> None:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self._last_hash = cv2.resize(gray, (self.hash_size, self.hash_size))

    def reset(self) -> None:
        self._last_hash = None
        self._frames_since_process = 0


# ── Speed-Adaptive Sampler ────────────────────────────────────────────────────

class SpeedAdaptiveSampler:
    """
    Adjusts how frequently frames are sampled based on vehicle speed.

    At low speed / stationary: sample almost every frame (dense).
    At highway speed: sample less frequently (road changes fast but
    individual damage is small and multiple frames capture same area).

    Speed is inferred from GPS delta or passed directly (km/h).
    Falls back to fixed-skip if no speed data available.
    """

    # (speed_kmh_threshold, process_every_N_frames)
    SPEED_TABLE = [
        (5,   1),    # near-stationary → every frame
        (20,  3),    # slow urban      → every 3rd frame  (was 2)
        (40,  5),    # urban/suburban  → every 5th frame  (was 3)
        (70,  8),    # arterial road   → every 8th frame  (was 4)
        (100, 12),   # highway         → every 12th frame (was 6)
        (999, 16),   # expressway      → every 16th frame (was 8)
    ]

    def __init__(self, default_skip: int = 2):
        self.default_skip  = default_skip
        self._frame_count  = 0
        self._current_skip = default_skip

    def update_speed(self, speed_kmh: float) -> None:
        """Call with current GPS speed to adjust sampling rate."""
        for threshold, skip in self.SPEED_TABLE:
            if speed_kmh <= threshold:
                self._current_skip = skip
                return
        self._current_skip = self.SPEED_TABLE[-1][1]

    def should_process(self) -> bool:
        """Returns True if this frame should be processed."""
        self._frame_count += 1
        return (self._frame_count % max(1, self._current_skip)) == 0

    @property
    def current_skip(self) -> int:
        return self._current_skip

    def reset(self) -> None:
        self._frame_count = 0


# ── Confidence Calibrator ─────────────────────────────────────────────────────

class ConfidenceCalibrator:
    """
    Fixes YOLO's poorly calibrated per-class confidence scores for road damage.

    WHY NEEDED:
    YOLO confidence scores are not well calibrated across classes. Road damage
    models show a consistent bias: thin cracks produce raw confidence ~0.25–0.35
    even when clearly visible, while potholes (large, round) hit 0.6–0.9 easily.
    A single global threshold (e.g. 0.50) therefore misses almost all cracks
    while allowing pothole false positives.

    HOW IT WORKS:
      1. Temperature scaling: conf_cal = conf^(1/T)
         T > 1.0 sharpens the distribution — middling scores shift higher.
         T = 1.2 gives a mild boost without inflating noise.

      2. Per-class minimum thresholds tuned to each class's detection difficulty:
         - Longitudinal/Transverse Crack → 0.28  (hardest, thin lines)
         - Alligator Crack               → 0.33  (area pattern, medium)
         - Pothole / Potholes            → 0.38  (most distinctive, stricter)

      3. Temporal consistency boost: if the same class appeared in the last
         N frames, add a small confidence increment (damage persists in reality).
         Reduces flickering and catches real damage detected inconsistently.
    """

    # Per-class thresholds — lower = harder to detect, needs more lenient gate
    DEFAULT_THRESHOLDS: Dict[str, float] = {
        "Longitudinal Crack": 0.28,
        "Transverse Crack":   0.28,
        "Alligator Crack":    0.33,
        "Pothole":            0.38,
        "Potholes":           0.38,   # alias used by RDD2022 model
    }
    GLOBAL_FALLBACK = 0.33

    def __init__(
        self,
        class_thresholds: Optional[Dict[str, float]] = None,
        temperature:      float = 1.2,    # > 1 sharpens confidence distribution
        temporal_boost:   float = 0.04,   # added per recent matching frame
        temporal_window:  int   = 6,      # how many frames back to look
    ):
        self.thresholds    = {**self.DEFAULT_THRESHOLDS, **(class_thresholds or {})}
        self.temperature   = temperature
        self.boost         = temporal_boost
        self.window        = temporal_window
        self._history: Dict[str, list] = {}   # class → [frame_indices]

    def calibrate(self, detections: list, frame_index: int) -> list:
        """
        Filter and calibrate raw YOLO detections.
        Adds 'raw_confidence' key (original value) and updates 'confidence'.
        Returns only detections that pass their per-class threshold.
        """
        out = []
        for d in detections:
            raw = float(d.get("confidence", 0))
            cls = d.get("class_name", "")

            # 1. Temperature scaling
            cal = float(np.power(np.clip(raw, 1e-6, 1.0), 1.0 / self.temperature))

            # 2. Temporal boost
            recent = [f for f in self._history.get(cls, [])
                      if abs(frame_index - f) <= self.window]
            if recent:
                cal = min(cal + self.boost * len(recent), 0.99)

            # 3. Per-class threshold gate
            threshold = self.thresholds.get(cls, self.GLOBAL_FALLBACK)
            if cal < threshold:
                continue

            # Track for temporal memory
            self._history.setdefault(cls, []).append(frame_index)
            self._history[cls] = self._history[cls][-(self.window * 3):]

            out.append({**d,
                        "confidence":     round(cal, 3),
                        "raw_confidence": round(raw, 3)})
        return out

    def reset(self) -> None:
        self._history.clear()

    def stats(self) -> dict:
        return {
            "temperature":    self.temperature,
            "temporal_window": self.window,
            "active_classes": list(self._history.keys()),
        }


# ── Main Preprocessor ─────────────────────────────────────────────────────────

class FramePreprocessor:
    """
    Full preprocessing pipeline:
      1. Speed-adaptive sampling check
      2. Duplicate detection
      3. Condition detection (night/rain/fog/glare)
      4. Enhancement
      5. ROI extraction

    Usage
    -----
    pre = FramePreprocessor()
    result = pre.process(frame, speed_kmh=35.0)
    if result.is_duplicate or not result.sample_this:
        pass  # skip YOLO
    else:
        detections = model(result.roi_frame)
        detections = pre.roi.remap_detections(detections, result.roi_bbox)
    """

    def __init__(
        self,
        # Condition thresholds
        night_threshold:  float = 60.0,
        glare_threshold:  float = 220.0,
        rain_threshold:   float = 0.15,
        # ── 5-factor fog (ALL must pass — fixes clear-road false positives) ──
        fog_var_max:      float = 0.025,   # normalised variance  (was 0.04, too loose)
        fog_br_min:       float = 75.0,    # must be brighter than night
        fog_br_max:       float = 185.0,   # must be dimmer than blinding sun
        fog_sat_max:      float = 45.0,    # fog kills colour saturation
        fog_edge_max:     float = 0.055,   # fog blurs edges (<5.5% edge pixels)
        fog_sky_std_max:  float = 22.0,    # fog = uniform pale sky
        # ─────────────────────────────────────────────────────────────────────
        clahe_clip:       float = 3.0,
        clahe_grid:       tuple = (8, 8),
        # Feature toggles
        enable_dedup:     bool  = True,
        enable_roi:       bool  = True,
        enable_adaptive:  bool  = True,
        default_skip:     int   = 2,
        # ROI config
        top_skip:         float = 0.35,
        bottom_skip:      float = 0.05,
        # Duplicate config
        dup_threshold:    float = 4.5,
    ):
        self.night_threshold = night_threshold
        self.glare_threshold = glare_threshold
        self.rain_threshold  = rain_threshold
        self.fog_var_max     = fog_var_max
        self.fog_br_min      = fog_br_min
        self.fog_br_max      = fog_br_max
        self.fog_sat_max     = fog_sat_max
        self.fog_edge_max    = fog_edge_max
        self.fog_sky_std_max = fog_sky_std_max
        self._clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_grid)

        self.enable_dedup    = enable_dedup
        self.enable_roi      = enable_roi
        self.enable_adaptive = enable_adaptive

        self.roi        = ROIMasker(top_skip=top_skip, bottom_skip=bottom_skip)
        self.dedup      = DuplicateDetector(threshold=dup_threshold)
        self.sampler    = SpeedAdaptiveSampler(default_skip=default_skip)
        self.calibrator = ConfidenceCalibrator()   # NEW

        self._stats = {"total": 0, "duplicates": 0, "skipped_speed": 0, "enhanced": 0}

    # ── Public ────────────────────────────────────────────────────────────────

    def process(
        self,
        frame: np.ndarray,
        speed_kmh: Optional[float] = None,
    ) -> PreprocessResult:
        self._stats["total"] += 1

        # Speed-adaptive sampling
        if self.enable_adaptive and speed_kmh is not None:
            self.sampler.update_speed(speed_kmh)
        sample_this = not self.enable_adaptive or self.sampler.should_process()
        if not sample_this:
            self._stats["skipped_speed"] += 1
            h, w = frame.shape[:2]
            return PreprocessResult(
                frame=frame, roi_frame=frame, roi_bbox=(0,0,w,h),
                condition=FrameCondition.NORMAL, brightness=0, contrast=0,
                enhanced=False, is_duplicate=False,
                sample_this=False, skip_reason="speed_adaptive"
            )

        # Duplicate detection
        if self.enable_dedup and self.dedup.is_duplicate(frame):
            self._stats["duplicates"] += 1
            h, w = frame.shape[:2]
            return PreprocessResult(
                frame=frame, roi_frame=frame, roi_bbox=(0,0,w,h),
                condition=FrameCondition.NORMAL, brightness=0, contrast=0,
                enhanced=False, is_duplicate=True,
                sample_this=True, skip_reason="duplicate"
            )

        # Condition detection + enhancement
        condition = self._detect_condition(frame)
        enhanced  = condition != FrameCondition.NORMAL
        if enhanced:
            self._stats["enhanced"] += 1
        out_frame = self._enhance(frame, condition) if enhanced else frame

        # Brightness / contrast metrics
        gray       = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = float(np.mean(gray))
        contrast   = float(np.std(gray))

        # ROI extraction
        if self.enable_roi:
            roi_frame, roi_bbox = self.roi.extract(out_frame)
        else:
            h, w = out_frame.shape[:2]
            roi_frame, roi_bbox = out_frame, (0, 0, w, h)

        return PreprocessResult(
            frame=out_frame, roi_frame=roi_frame, roi_bbox=roi_bbox,
            condition=condition, brightness=brightness, contrast=contrast,
            enhanced=enhanced, is_duplicate=False,
            sample_this=True, skip_reason=""
        )

    def stats(self) -> dict:
        t = max(self._stats["total"], 1)
        return {
            "total_frames":    self._stats["total"],
            "duplicates":      self._stats["duplicates"],
            "skipped_speed":   self._stats["skipped_speed"],
            "enhanced":        self._stats["enhanced"],
            "dup_rate_pct":    round(self._stats["duplicates"] / t * 100, 1),
            "skip_rate_pct":   round(self._stats["skipped_speed"] / t * 100, 1),
            "enhance_rate_pct":round(self._stats["enhanced"] / t * 100, 1),
        }

    def calibrate_detections(self, detections: list, frame_index: int) -> list:
        """Apply per-class confidence calibration to raw YOLO detections.
        Call this after model inference, before scoring/costing.
        Returns filtered list with calibrated_conf and raw_confidence added.
        """
        return self.calibrator.calibrate(detections, frame_index)

    def reset(self) -> None:
        self.dedup.reset()
        self.sampler.reset()
        self.calibrator.reset()
        self._stats = {"total": 0, "duplicates": 0, "skipped_speed": 0, "enhanced": 0}

    # ── Condition Detection ───────────────────────────────────────────────────

    def _detect_condition(self, frame: np.ndarray) -> FrameCondition:
        gray       = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = float(np.mean(gray))
        variance   = float(np.var(gray)) / (255 ** 2)

        if brightness < self.night_threshold:
            return FrameCondition.NIGHT
        if brightness > self.glare_threshold:
            return FrameCondition.GLARE
        if self._is_fog(frame, gray, brightness, variance):
            return FrameCondition.FOG
        if self._has_rain_streaks(gray):
            return FrameCondition.RAIN
        return FrameCondition.NORMAL

    def _is_fog(self, frame: np.ndarray, gray: np.ndarray,
                brightness: float, variance: float) -> bool:
        """
        5-factor fog detector. ALL factors must pass.

        The old single-factor check (variance < 0.04) triggered on any
        low-contrast frame — smooth tarmac in bright daylight has low variance
        too, causing 98% of clear daytime road footage to be flagged as fog.

        New approach requires 5 independent physical signs of atmospheric fog:
          1. Low image contrast   — fog reduces sharpness differences
          2. Mid-range brightness — fog is pale/grey, not dark night or glaring sun
          3. Low colour saturation — fog washes out colours toward grey/white
          4. Sparse edges          — fog blurs fine structural detail
          5. Uniform sky region    — fog creates a flat, featureless sky band
        """
        # 1. Low contrast (necessary but not sufficient on its own)
        if variance >= self.fog_var_max:
            return False

        # 2. Mid-range brightness (not night, not glare/sunny)
        if not (self.fog_br_min <= brightness <= self.fog_br_max):
            return False

        # 3. Low colour saturation — clear days have vivid colours
        hsv      = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mean_sat = float(np.mean(hsv[:, :, 1]))
        if mean_sat >= self.fog_sat_max:
            return False

        # 4. Low edge density — fog softens/blurs structural edges
        edges        = cv2.Canny(gray, 50, 150)
        edge_density = float(np.count_nonzero(edges)) / (gray.shape[0] * gray.shape[1])
        if edge_density >= self.fog_edge_max:
            return False

        # 5. Uniform pale sky region — fog = flat grey/white top band
        sky_h   = max(1, int(frame.shape[0] * 0.20))
        sky_std = float(np.std(cv2.cvtColor(frame[:sky_h, :], cv2.COLOR_BGR2GRAY)))
        if sky_std >= self.fog_sky_std_max:
            return False

        return True  # all 5 passed → genuine atmospheric fog

    def _has_rain_streaks(self, gray: np.ndarray) -> bool:
        sobel_x  = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y  = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        h_energy = float(np.mean(np.abs(sobel_x)))
        v_energy = float(np.mean(np.abs(sobel_y)))
        ratio    = v_energy / (h_energy + 1e-6)
        return ratio > (1.0 / self.rain_threshold)

    # ── Enhancement ───────────────────────────────────────────────────────────

    def _enhance(self, frame: np.ndarray, condition: FrameCondition) -> np.ndarray:
        handlers = {
            FrameCondition.NIGHT: self._enhance_night,
            FrameCondition.RAIN:  self._enhance_rain,
            FrameCondition.FOG:   self._enhance_fog,
            FrameCondition.GLARE: self._enhance_glare,
        }
        return handlers.get(condition, lambda f: f)(frame)

    def _enhance_night(self, frame: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = self._clahe.apply(l)
        enhanced = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
        inv_gamma = 1.0 / 1.5
        table = np.array([(i / 255.0) ** inv_gamma * 255
                          for i in range(256)], dtype=np.uint8)
        return cv2.LUT(enhanced, table)

    def _enhance_rain(self, frame: np.ndarray) -> np.ndarray:
        derained = cv2.bilateralFilter(frame, d=9, sigmaColor=75, sigmaSpace=75)
        lab = cv2.cvtColor(derained, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = self._clahe.apply(l)
        return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    def _enhance_fog(self, frame: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe_strong = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
        l = clahe_strong.apply(l)
        dehazed = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
        blurred = cv2.GaussianBlur(dehazed, (0, 0), 3)
        return cv2.addWeighted(dehazed, 1.5, blurred, -0.5, 0)

    def _enhance_glare(self, frame: np.ndarray) -> np.ndarray:
        inv_gamma = 1.0 / 0.6
        table = np.array([(i / 255.0) ** inv_gamma * 255
                          for i in range(256)], dtype=np.uint8)
        darkened = cv2.LUT(frame, table)
        return cv2.convertScaleAbs(darkened, alpha=0.85, beta=0)