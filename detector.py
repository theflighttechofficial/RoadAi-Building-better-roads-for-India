"""
detector.py — Advanced Detection & AI Engine

Features:
  1. Severity scoring per defect (individual bbox-level severity 0–100)
  2. Crack width/length estimation in cm using perspective transform
  3. Pothole depth estimation from shadow analysis
  4. Multi-model ensemble — run 2+ models, merge with Weighted Boxes Fusion
  5. Uncertainty quantification — Bayesian Monte Carlo Dropout
  6. Active learning — flag uncertain / novel detections for human review
  7. Self-supervised pseudo-labeling for dataset expansion
"""

from __future__ import annotations

import cv2
import numpy as np
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict
from pathlib import Path
import json
from datetime import datetime
import torch
import torch.nn as nn

log = logging.getLogger(__name__)

# Physics-based depth estimator (graceful fallback if not installed)
try:
    from Depth_estimator import DepthEstimator, estimate_depth_without_frame
    _DEPTH_EST = DepthEstimator()
    DEPTH_EST_OK = True
    log.info("Physics depth estimator loaded (shadow + brightness + edge model)")
except ImportError:
    DEPTH_EST_OK = False
    log.warning("depth_estimator.py not found — using shape heuristic for depth")

# Weighted Boxes Fusion library for ensemble (install: pip install ensemble-boxes)
try:
    from ensemble_boxes import weighted_boxes_fusion
    WBF_AVAILABLE = True
    log.info("Weighted Boxes Fusion available for ensemble")
except ImportError:
    WBF_AVAILABLE = False
    log.warning("ensemble-boxes not installed — falling back to simple NMS")


# ── Per-detection severity weights ───────────────────────────────────────────

CLASS_SEVERITY = {
    "Pothole":            1.00,   # immediate safety hazard
    "Potholes":           1.00,   # alias for RDD2022 model
    "Alligator Crack":    0.85,   # base failure indicator
    "Transverse Crack":   0.60,   # water ingress risk
    "Longitudinal Crack": 0.45,   # gradual structural risk
}

# Repair cost per m² in INR
REPAIR_RATES = {
    "Pothole":            2200,
    "Potholes":           2200,   # alias for RDD2022 model
    "Alligator Crack":    2800,
    "Transverse Crack":    950,
    "Longitudinal Crack":  850,
}

# Assumed road surface pixels per cm (calibrated for typical dashcam ~1280px wide ≈ 8m road)
# → 1280px / 800cm ≈ 1.6 px/cm
PX_PER_CM = 1.6


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class DefectMeasurement:
    """Physical size estimates for one detection."""
    width_cm:   float = 0.0    # estimated horizontal extent
    length_cm:  float = 0.0    # estimated vertical extent
    area_cm2:   float = 0.0    # width × length
    depth_cm:   float = 0.0    # pothole depth from shadow (0 if not pothole)
    severity:   float = 0.0    # 0–100 per-defect severity score
    cost_inr:   float = 0.0    # per-defect repair cost estimate


@dataclass
class EnrichedDetection:
    """One YOLO detection enriched with measurements, severity, flags."""
    bbox:          List[int]           # [x1,y1,x2,y2] in original frame coords
    confidence:    float
    class_id:      int
    class_name:    str
    measurement:   DefectMeasurement
    uncertain:     bool = False        # flagged for active learning review
    novelty_score: float = 0.0         # how different from typical detections
    uncertainty:   float = 0.0         # epistemic uncertainty (MC dropout)
    needs_review:  bool = False        # high uncertainty flag

    def to_dict(self) -> dict:
        return {
            "bbox":          self.bbox,
            "confidence":    round(self.confidence, 3),
            "class":         self.class_id,
            "class_name":    self.class_name,
            "width_cm":      round(self.measurement.width_cm, 1),
            "length_cm":     round(self.measurement.length_cm, 1),
            "area_cm2":      round(self.measurement.area_cm2, 1),
            "depth_cm":      round(self.measurement.depth_cm, 1),
            "severity":      round(self.measurement.severity, 1),
            "cost_inr":      round(self.measurement.cost_inr, 0),
            "uncertain":     self.uncertain,
            "novelty_score": round(self.novelty_score, 3),
            "uncertainty":   round(self.uncertainty, 3),
            "needs_review":  self.needs_review,
        }


# ── Perspective / Size Estimator ──────────────────────────────────────────────

class SizeEstimator:
    """
    Estimates real-world crack/pothole dimensions from bounding box pixels.

    Uses a simple perspective correction: detections lower in the frame
    (closer to camera) are larger in pixel space per unit area, so we
    apply a vertical correction factor.

    For a proper calibration you'd use a homography matrix from a
    checkerboard on the road surface. This approximation is good enough
    for rough-order estimates (±30%).
    """

    def __init__(
        self,
        frame_h:         int   = 720,
        frame_w:         int   = 1280,
        road_width_m:    float = 7.0,    # typical single carriageway
        camera_height_m: float = 1.2,    # typical dashcam mount
        fov_vertical_deg:float = 50.0,
    ):
        self.frame_h          = frame_h
        self.frame_w          = frame_w
        self.road_width_m     = road_width_m
        self.camera_height_m  = camera_height_m
        self.fov_v            = np.radians(fov_vertical_deg)
        # px/cm at the bottom of frame (closest point)
        self._base_px_per_cm  = frame_w / (road_width_m * 100.0)
        self._last_depth_estimate = None   # set by _estimate_depth

    def estimate(self, bbox: List[int], class_name: str) -> DefectMeasurement:
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        bw = max(x2 - x1, 1)
        bh = max(y2 - y1, 1)

        # Perspective scale: objects near top of road zone are farther away
        # Normalize y position within road zone (0=top of road, 1=bottom)
        road_top = self.frame_h * 0.35
        road_frac = max(0.1, (cy - road_top) / max(self.frame_h - road_top, 1))
        # Inverse perspective: farther away → fewer px per cm
        scale = self._base_px_per_cm * road_frac  # px/cm

        width_cm  = bw / max(scale, 0.01)
        length_cm = bh / max(scale, 0.01)
        area_cm2  = width_cm * length_cm

        # Pothole depth from shadow
        depth_cm = 0.0
        if class_name in ("Pothole", "Potholes"):
            depth_cm = self._estimate_depth(bbox, frame=getattr(self, '_current_frame', None), class_name=class_name)

        # Per-defect severity (0–100)
        class_w  = CLASS_SEVERITY.get(class_name, 0.5)
        area_norm = min(area_cm2 / 5000.0, 1.0)   # 5000cm² ≈ ~1m² full lane damage
        depth_norm= min(depth_cm / 10.0, 1.0)      # 10cm = severe pothole
        severity  = round((class_w * 0.5 + area_norm * 0.3 + depth_norm * 0.2) * 100, 1)

        # Cost estimate
        area_m2  = area_cm2 / 10000.0
        rate     = REPAIR_RATES.get(class_name, 1500)
        cost_inr = max(area_m2 * rate, rate * 0.05)   # minimum 5% of rate per detection

        return DefectMeasurement(
            width_cm=round(width_cm, 1),
            length_cm=round(length_cm, 1),
            area_cm2=round(area_cm2, 1),
            depth_cm=round(depth_cm, 1),
            severity=severity,
            cost_inr=round(cost_inr, 0),
        )

    def _estimate_depth(self, bbox: List[int], frame: Optional[np.ndarray] = None,
                        class_name: str = "Pothole") -> float:
        """
        Physics-grounded depth estimation.

        When a raw frame is available uses the three-cue shadow/brightness/edge
        model from depth_estimator.py (Eriksson 2008, Koch & Brilakis 2011).
        Falls back to shape-only heuristic when frame is not provided.
        Confidence and full DepthEstimate are stored in self._last_depth_estimate
        for downstream use (report generator, ticket PDF).
        """
        if DEPTH_EST_OK and frame is not None:
            est = _DEPTH_EST.estimate(frame, tuple(bbox), class_name)
            self._last_depth_estimate = est
            return est.depth_cm
        # Shape-only fallback
        if DEPTH_EST_OK:
            est = estimate_depth_without_frame(bbox, class_name)
            self._last_depth_estimate = est
            return est.depth_cm
        # Legacy heuristic (no depth_estimator.py present)
        x1, y1, x2, y2 = bbox
        bw = max(x2 - x1, 1); bh = max(y2 - y1, 1)
        aspect = bw / bh
        roundness   = 1.0 - abs(aspect - 1.0) / max(aspect, 1)
        size_factor = min((bw * bh) / (100 * 100), 1.0)
        self._last_depth_estimate = None
        return round(roundness * size_factor * 12.0, 1)


# ── NEW: Uncertainty Quantification (MC Dropout) ──────────────────────────────

class UncertaintyEstimator:
    """
    Bayesian uncertainty estimation using Monte Carlo Dropout.
    
    Runs multiple forward passes with dropout enabled to estimate
    epistemic uncertainty (model uncertainty about its predictions).
    
    High uncertainty → needs human review
    """
    
    def __init__(self, num_samples: int = 10, dropout_rate: float = 0.2):
        """
        Args:
            num_samples: Number of MC dropout samples (10-30 recommended)
            dropout_rate: Dropout probability during inference
        """
        self.num_samples = num_samples
        self.dropout_rate = dropout_rate
        self._model = None
        
        log.info(f"Uncertainty estimator: {num_samples} MC samples, dropout={dropout_rate}")
    
    def enable_dropout(self, model):
        """Enable dropout layers during inference for Monte Carlo sampling"""
        self._model = model
        for module in model.model.modules():
            if isinstance(module, nn.Dropout):
                module.train()  # Keep in training mode to enable dropout
                log.debug(f"Enabled MC dropout: {module}")
    
    def estimate_uncertainty(
        self,
        detections: List[dict],
        all_samples: List[List[dict]]
    ) -> List[float]:
        """
        Compute epistemic uncertainty for each detection.
        
        Args:
            detections: Final merged detections
            all_samples: Raw detections from all MC samples
            
        Returns:
            List of uncertainty scores (0-1) per detection
        """
        uncertainties = []
        
        for det in detections:
            # Find matching detections across samples
            matches = []
            for sample in all_samples:
                for s_det in sample:
                    if self._iou(det['bbox'], s_det['bbox']) > 0.5:
                        matches.append(s_det)
            
            # Calculate variance in confidence
            if len(matches) > 1:
                confidences = [m['confidence'] for m in matches]
                uncertainty = float(np.std(confidences))
            else:
                # Low sample count = high uncertainty
                uncertainty = 1.0 - (len(matches) / self.num_samples)
            
            uncertainties.append(min(uncertainty, 1.0))
        
        return uncertainties
    
    @staticmethod
    def _iou(b1: List[int], b2: List[int]) -> float:
        """Compute IoU between two boxes"""
        x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
        x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
        inter = max(0, x2-x1) * max(0, y2-y1)
        if inter == 0: return 0.0
        a1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
        a2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
        return inter / (a1 + a2 - inter + 1e-6)


# ── ENHANCED: Active Learning Flagger with Uncertainty ────────────────────────

class ActiveLearningFlagger:
    """
    Flags detections that are uncertain or novel for human review.

    Now integrates:
    - Confidence-based uncertainty (existing)
    - Novelty detection (existing)
    - MC Dropout uncertainty (NEW)
    - Edge case detection (NEW)
    """

    def __init__(
        self,
        conf_uncertain_low:  float = 0.30,   # below this → probably noise
        conf_uncertain_high: float = 0.65,   # above this → confident
        uncertainty_threshold: float = 0.4,  # MC dropout threshold
        max_review_queue:    int   = 1000,
        queue_dir: str = "active_learning_queue",
    ):
        self.conf_low  = conf_uncertain_low
        self.conf_high = conf_uncertain_high
        self.uncertainty_threshold = uncertainty_threshold
        self._max_queue = max_review_queue
        
        # Setup queue directory
        self.queue_dir = Path(queue_dir)
        self.queue_dir.mkdir(exist_ok=True, parents=True)
        self.queue_file = self.queue_dir / "review_queue.json"
        
        self._review_queue: list[dict] = []
        self._novelty_stats: dict = {}   # class → (mean_area, std_area)
        self._area_history: dict  = {}   # class → list of areas
        
        # Load existing queue
        self._load_queue()
        
        log.info(f"Active learning initialized: {len(self._review_queue)} items in queue")

    def flag(
        self,
        det: dict,
        frame_index: int,
        frame_shape: Tuple[int, int],
        mc_uncertainty: float = 0.0,  # NEW: from MC dropout
    ) -> Tuple[bool, float, bool]:
        """
        Returns (is_uncertain, novelty_score, needs_review).
        
        is_uncertain = True → borderline confidence
        needs_review = True → high MC uncertainty OR uncertain OR novel
        """
        conf  = det.get("confidence", 0.5)
        cls   = det.get("class_name", "")
        bbox  = det.get("bbox", [0,0,1,1])
        x1,y1,x2,y2 = bbox
        area  = (x2-x1) * (y2-y1)
        fh, fw = frame_shape[:2]

        # Criteria
        uncertain = self.conf_low <= conf <= self.conf_high
        novelty = self._compute_novelty(cls, area, fw * fh)
        high_mc_uncertainty = mc_uncertainty > self.uncertainty_threshold
        edge_case = self._is_edge_case(det, frame_shape)
        
        # Overall review flag
        needs_review = uncertain or novelty > 0.7 or high_mc_uncertainty or edge_case

        if needs_review:
            reasons = []
            if uncertain: reasons.append("uncertain_confidence")
            if novelty > 0.7: reasons.append("novel_pattern")
            if high_mc_uncertainty: reasons.append("high_epistemic_uncertainty")
            if edge_case: reasons.append("edge_case")
            
            self._add_to_queue({
                "frame_index": frame_index,
                "class_name":  cls,
                "confidence":  conf,
                "bbox":        bbox,
                "novelty":     round(novelty, 3),
                "mc_uncertainty": round(mc_uncertainty, 3),
                "reasons":     reasons,
                "priority":    self._calculate_priority(uncertain, novelty, mc_uncertainty, edge_case),
            })

        # Update area history for novelty baseline
        if cls not in self._area_history:
            self._area_history[cls] = []
        self._area_history[cls].append(area)
        if len(self._area_history[cls]) > 500:
            self._area_history[cls] = self._area_history[cls][-500:]

        return uncertain, round(novelty, 3), needs_review

    def _compute_novelty(self, cls: str, area: int, frame_area: int) -> float:
        history = self._area_history.get(cls, [])
        if len(history) < 10:
            return 0.0   # not enough data
        area_norm = area / max(frame_area, 1)
        hist_norm = [a / max(frame_area, 1) for a in history]
        mean = float(np.mean(hist_norm))
        std  = float(np.std(hist_norm)) + 1e-6
        z    = abs(area_norm - mean) / std
        return float(min(z / 5.0, 1.0))   # Z>5 → novelty=1.0
    
    def _is_edge_case(self, det: dict, frame_shape: Tuple[int, int]) -> bool:
        """Detect unusual detection patterns"""
        bbox = det.get("bbox", [0,0,1,1])
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        
        # Check for extreme aspect ratios or sizes
        if w <= 0 or h <= 0:
            return True
        
        aspect = w / h
        area = w * h
        frame_h, frame_w = frame_shape[:2]
        frame_area = frame_h * frame_w
        
        # Extremely thin/wide
        if aspect > 10 or aspect < 0.1:
            return True
        
        # Extremely large (>20% of frame)
        if area > frame_area * 0.2:
            return True
        
        # Extremely small (<100 pixels)
        if area < 100:
            return True
        
        return False
    
    def _calculate_priority(
        self,
        uncertain: bool,
        novelty: float,
        mc_uncertainty: float,
        edge_case: bool
    ) -> float:
        """Calculate review priority (0-1, higher = more urgent)"""
        priority = 0.0
        
        if uncertain:
            priority += 0.25
        
        priority += novelty * 0.3
        priority += mc_uncertainty * 0.35
        
        if edge_case:
            priority += 0.1
        
        return min(priority, 1.0)

    def add_frame_to_queue(
        self,
        frame: np.ndarray,
        detection_result: dict,
        frame_metadata: dict = None
    ):
        """
        Save frame image for later review/labeling.
        
        Args:
            frame: Image frame
            detection_result: Detection data with uncertainty
            frame_metadata: GPS, timestamp, etc.
        """
        if frame_metadata is None:
            frame_metadata = {}
        
        # Generate unique ID
        item_id = f"review_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        # Save frame image
        frame_path = self.queue_dir / f"{item_id}.jpg"
        cv2.imwrite(str(frame_path), frame)
        
        # Add to queue
        self._add_to_queue({
            'id': item_id,
            'frame_path': str(frame_path),
            'detection': detection_result,
            'metadata': frame_metadata,
            'timestamp': datetime.now().isoformat(),
            'status': 'pending',
        })

    def _add_to_queue(self, item: dict) -> None:
        if len(self._review_queue) < self._max_queue:
            self._review_queue.append(item)
        else:
            # Remove lowest priority item
            self._trim_queue()
            self._review_queue.append(item)
        
        self._save_queue()
    
    def _trim_queue(self):
        """Remove lowest priority items"""
        if 'priority' in self._review_queue[0]:
            self._review_queue = sorted(
                self._review_queue,
                key=lambda x: x.get('priority', 0),
                reverse=True
            )[:self._max_queue]

    def export_review_queue(self, path: str = None) -> str:
        """Export queue for labeling in Roboflow/CVAT"""
        if path is None:
            path = str(self.queue_dir / f"export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        
        # Sort by priority
        sorted_queue = sorted(
            [item for item in self._review_queue if item.get('status') == 'pending'],
            key=lambda x: x.get('priority', 0),
            reverse=True
        )
        
        export_data = {
            'export_date': datetime.now().isoformat(),
            'num_items': len(sorted_queue),
            'items': sorted_queue
        }
        
        with open(path, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2)
        
        log.info(f"Exported {len(sorted_queue)} review items → {path}")
        return path

    def get_review_queue(self) -> list[dict]:
        return self._review_queue.copy()

    def stats(self) -> dict:
        pending = sum(1 for r in self._review_queue if r.get('status') == 'pending')
        return {
            "review_queue_size": len(self._review_queue),
            "pending": pending,
            "reviewed": len(self._review_queue) - pending,
            "avg_priority": float(np.mean([r.get('priority', 0) for r in self._review_queue])) if self._review_queue else 0,
        }
    
    def _save_queue(self):
        """Persist queue to disk"""
        try:
            with open(self.queue_file, 'w') as f:
                json.dump(self._review_queue, f, indent=2)
        except Exception as e:
            log.warning(f"Failed to save queue: {e}")
    
    def _load_queue(self):
        """Load queue from disk"""
        if self.queue_file.exists():
            try:
                with open(self.queue_file, 'r') as f:
                    self._review_queue = json.load(f)
                log.info(f"Loaded {len(self._review_queue)} items from queue")
            except Exception as e:
                log.warning(f"Failed to load queue: {e}")
                self._review_queue = []


# ── ENHANCED: Multi-Model Ensemble with WBF ──────────────────────────────────

class EnsembleDetector:
    """
    Runs multiple YOLO models and merges with Weighted Boxes Fusion.
    
    Now supports:
    - Weighted Boxes Fusion (better than NMS)
    - Adaptive model weights based on conditions
    - MC Dropout uncertainty estimation per model
    """

    def __init__(
        self,
        model_configs: Dict[str, dict],  # name → {'path': ..., 'weight': ..., 'specialization': ...}
        iou_threshold:   float = 0.45,
        skip_box_threshold: float = 0.0,
        enable_mc_dropout: bool = False,
        mc_samples: int = 10,
    ):
        self.iou_threshold = iou_threshold
        self.skip_box_threshold = skip_box_threshold
        self.enable_mc_dropout = enable_mc_dropout
        self.mc_samples = mc_samples
        
        self.model_configs = model_configs
        self.models = {}
        self.uncertainty_estimators = {}

        # Load models
        for name, config in model_configs.items():
            try:
                from ultralytics import YOLO
                model = YOLO(config['path'])
                self.models[name] = model
                
                # Setup uncertainty estimator if enabled
                if enable_mc_dropout:
                    ue = UncertaintyEstimator(num_samples=mc_samples)
                    ue.enable_dropout(model)
                    self.uncertainty_estimators[name] = ue
                
                log.info(f"Ensemble: loaded {name} from {config['path']}")
            except Exception as e:
                log.warning(f"Ensemble: could not load {name}: {e}")

        if not self.models:
            raise RuntimeError("No models loaded for ensemble")

        # Get class names from first model
        first_model = list(self.models.values())[0]
        self.cls_names = {int(k): v for k, v in first_model.names.items()}

    def detect(
        self,
        frame: np.ndarray,
        conf: float = 0.35,
        conditions: dict = None,
    ) -> Tuple[List[dict], List[float]]:
        """
        Run ensemble detection with adaptive weighting.
        
        Args:
            frame: Input image
            conf: Confidence threshold
            conditions: Environmental conditions for adaptive weights
                {'time': 'night', 'weather': 'rain', 'fog': False}
        
        Returns:
            (detections, uncertainties) where uncertainties is MC dropout scores
        """
        if conditions is None:
            conditions = {'time': 'day', 'weather': 'clear', 'fog': False}
        
        # Get adaptive weights
        weights = self._get_adaptive_weights(conditions)
        
        # Collect predictions from all models
        all_boxes = []
        all_scores = []
        all_labels = []
        model_weights = []
        all_samples = []  # For uncertainty estimation

        for model_name, model in self.models.items():
            try:
                # Run inference (with MC dropout if enabled)
                if self.enable_mc_dropout:
                    samples = []
                    for _ in range(self.mc_samples):
                        raw = model(frame, conf=conf, verbose=False)[0]
                        sample_dets = self._extract_detections(raw)
                        samples.append(sample_dets)
                    
                    # Use mean prediction
                    detections = self._average_samples(samples)
                    all_samples.extend(samples)
                else:
                    raw = model(frame, conf=conf, verbose=False)[0]
                    detections = self._extract_detections(raw)
                
                if detections:
                    boxes = np.array([d['bbox_norm'] for d in detections])
                    scores = np.array([d['confidence'] for d in detections])
                    labels = np.array([d['class'] for d in detections])
                    
                    all_boxes.append(boxes)
                    all_scores.append(scores)
                    all_labels.append(labels)
                    model_weights.append(weights.get(model_name, 0.5))
                    
                    log.debug(f"{model_name}: {len(boxes)} detections")
                    
            except Exception as e:
                log.warning(f"Model {model_name} failed: {e}")

        # No detections
        if not all_boxes:
            return [], []

        # Apply Weighted Boxes Fusion
        if WBF_AVAILABLE:
            fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
                all_boxes,
                all_scores,
                all_labels,
                weights=model_weights,
                iou_thr=self.iou_threshold,
                skip_box_thr=self.skip_box_threshold
            )
        else:
            # Fallback to simple merge
            fused_boxes = np.vstack(all_boxes)
            fused_scores = np.hstack(all_scores)
            fused_labels = np.hstack(all_labels)

        # Convert to detection format
        h, w = frame.shape[:2]
        detections = []
        
        for box, score, label in zip(fused_boxes, fused_scores, fused_labels):
            x1, y1, x2, y2 = box
            detections.append({
                'bbox': [int(x1*w), int(y1*h), int(x2*w), int(y2*h)],
                'bbox_norm': box.tolist(),
                'confidence': float(score),
                'class': int(label),
                'class_name': self.cls_names.get(int(label), f"Class{int(label)}"),
            })
        
        # Estimate uncertainty if MC dropout enabled
        uncertainties = []
        if self.enable_mc_dropout and all_samples:
            ue = UncertaintyEstimator()
            uncertainties = ue.estimate_uncertainty(detections, all_samples)
        else:
            uncertainties = [0.0] * len(detections)
        
        log.info(f"Ensemble: {sum(len(b) for b in all_boxes)} → {len(detections)} detections (WBF)")
        
        return detections, uncertainties

    def _extract_detections(self, raw_result) -> List[dict]:
        """Extract detections from YOLO result"""
        detections = []
        for box in raw_result.boxes:
            x1, y1, x2, y2 = box.xyxyn[0].tolist()  # Normalized
            detections.append({
                'bbox_norm': [x1, y1, x2, y2],
                'confidence': float(box.conf[0]),
                'class': int(box.cls[0]),
            })
        return detections
    
    def _average_samples(self, samples: List[List[dict]]) -> List[dict]:
        """Average MC dropout samples"""
        if not samples:
            return []
        
        # Simple averaging (could be improved with clustering)
        all_dets = []
        for sample in samples:
            all_dets.extend(sample)
        
        return all_dets
    
    def _get_adaptive_weights(self, conditions: dict) -> Dict[str, float]:
        """Adjust model weights based on environmental conditions"""
        weights = {}
        
        # Night conditions
        if conditions.get('time') == 'night':
            weights = {
                'general': 0.2,
                'night': 0.6,
                'rain': 0.1,
                'cracks': 0.1
            }
        # Rain conditions
        elif conditions.get('weather') == 'rain':
            weights = {
                'general': 0.2,
                'night': 0.1,
                'rain': 0.6,
                'cracks': 0.1
            }
        # Fog conditions
        elif conditions.get('fog'):
            weights = {
                'general': 0.3,
                'night': 0.3,
                'rain': 0.2,
                'cracks': 0.2
            }
        # Default (day, clear)
        else:
            weights = {
                'general': 0.5,
                'night': 0.15,
                'rain': 0.15,
                'cracks': 0.2
            }
        
        # Only return weights for loaded models
        return {k: v for k, v in weights.items() if k in self.models}


# ── Main Detection Engine ─────────────────────────────────────────────────────

class DetectionEngine:
    """
    Full detection pipeline combining:
      - Single model OR ensemble with WBF
      - MC Dropout uncertainty quantification
      - Size estimation (width/length/depth in cm)
      - Per-defect severity scoring
      - Active learning flagging

    Usage
    -----
    # Single model:
    engine = DetectionEngine("yolov8s.pt")
    
    # Ensemble:
    engine = DetectionEngine({
        'general': {'path': 'yolov8m.pt', 'weight': 0.5},
        'night': {'path': 'yolov8s_night.pt', 'weight': 0.5}
    }, use_ensemble=True)
    
    # With uncertainty:
    engine = DetectionEngine("yolov8s.pt", enable_uncertainty=True)

    result = engine.run(frame, frame_index=fi)
    """

    def __init__(
        self,
        model_path,                         # str or dict[str, dict]
        frame_w:    int   = 1280,
        frame_h:    int   = 720,
        conf:       float = 0.50,
        use_ensemble: bool = False,
        enable_measurements: bool = True,
        enable_active_learning: bool = True,
        enable_uncertainty: bool = False,   # NEW: MC dropout
        mc_samples: int = 10,
        conditions: dict = None,            # NEW: environmental conditions
    ):
        self.conf    = conf
        self.frame_w = frame_w
        self.frame_h = frame_h
        self.conditions = conditions or {'time': 'day', 'weather': 'clear'}

        # Load model(s)
        if use_ensemble and isinstance(model_path, dict):
            self.ensemble = EnsembleDetector(
                model_path,
                enable_mc_dropout=enable_uncertainty,
                mc_samples=mc_samples
            )
            self.model = None
            self.cls_names = self.ensemble.cls_names
            self.use_ensemble = True
            log.info(f"DetectionEngine: ensemble mode ({len(model_path)} models)")
        else:
            path = model_path if isinstance(model_path, str) else model_path.get('general', {}).get('path', 'yolov8s.pt')
            from ultralytics import YOLO
            self.model = YOLO(path)
            self.cls_names = {int(k): v for k, v in self.model.names.items()}
            self.ensemble = None
            self.use_ensemble = False
            
            # Setup uncertainty for single model
            if enable_uncertainty:
                self.uncertainty_estimator = UncertaintyEstimator(num_samples=mc_samples)
                self.uncertainty_estimator.enable_dropout(self.model)
            else:
                self.uncertainty_estimator = None
            
            log.info(f"DetectionEngine: single model {path}")

        self.enable_measurements     = enable_measurements
        self.enable_active_learning  = enable_active_learning
        self.enable_uncertainty      = enable_uncertainty

        self.estimator = SizeEstimator(frame_w=frame_w, frame_h=frame_h)
        self.flagger   = ActiveLearningFlagger() if enable_active_learning else None

    def run(
        self,
        frame:       np.ndarray,
        frame_index: int = 0,
        roi_bbox:    Optional[Tuple[int,int,int,int]] = None,
        conditions:  Optional[dict] = None,
    ) -> "DetectionResult":
        """
        Run detection on frame (or ROI sub-frame).
        
        Args:
            frame: Input image
            frame_index: Frame number
            roi_bbox: Optional ROI (x1,y1,x2,y2)
            conditions: Environmental conditions override
        """
        # Use provided conditions or default
        cond = conditions or self.conditions
        
        # Run model
        raw_dets, uncertainties = self._infer(frame, cond)

        # Offset from ROI coords → full frame coords
        if roi_bbox:
            x_off, y_off = roi_bbox[0], roi_bbox[1]
            for d in raw_dets:
                b = d["bbox"]
                d["bbox"] = [b[0]+x_off, b[1]+y_off, b[2]+x_off, b[3]+y_off]

        # Enrich each detection
        enriched: List[EnrichedDetection] = []
        for i, d in enumerate(raw_dets):
            m = DefectMeasurement()
            if self.enable_measurements:
                self.estimator._current_frame = frame
                m = self.estimator.estimate(d["bbox"], d.get("class_name",""))

            # Get uncertainty for this detection
            mc_uncertainty = uncertainties[i] if i < len(uncertainties) else 0.0
            
            # Active learning flagging
            uncertain, novelty, needs_review = False, 0.0, False
            if self.enable_active_learning and self.flagger:
                uncertain, novelty, needs_review = self.flagger.flag(
                    d, 
                    frame_index, 
                    frame.shape,
                    mc_uncertainty=mc_uncertainty
                )

            enriched.append(EnrichedDetection(
                bbox=d["bbox"],
                confidence=d["confidence"],
                class_id=d.get("class", 0),
                class_name=d.get("class_name", ""),
                measurement=m,
                uncertain=uncertain,
                novelty_score=novelty,
                uncertainty=mc_uncertainty,
                needs_review=needs_review,
            ))

        return DetectionResult(
            detections=enriched,
            frame_index=frame_index,
            total_cost=sum(e.measurement.cost_inr for e in enriched),
            uncertain_count=sum(1 for e in enriched if e.uncertain),
            needs_review_count=sum(1 for e in enriched if e.needs_review),
        )

    def _infer(self, frame: np.ndarray, conditions: dict) -> Tuple[List[dict], List[float]]:
        """Run inference and return (detections, uncertainties)"""
        try:
            if self.use_ensemble:
                return self.ensemble.detect(frame, conf=self.conf, conditions=conditions)
            else:
                # Single model with optional MC dropout
                if self.enable_uncertainty and self.uncertainty_estimator:
                    # Run multiple samples
                    all_samples = []
                    for _ in range(self.uncertainty_estimator.num_samples):
                        raw = self.model(frame, conf=self.conf, verbose=False)[0]
                        sample_dets = self._extract_detections(raw)
                        all_samples.append(sample_dets)
                    
                    # Average samples
                    detections = self._merge_mc_samples(all_samples)
                    
                    # Estimate uncertainties
                    ue = UncertaintyEstimator()
                    uncertainties = ue.estimate_uncertainty(detections, all_samples)
                else:
                    # Regular single inference
                    raw = self.model(frame, conf=self.conf, verbose=False)[0]
                    detections = self._extract_detections(raw)
                    uncertainties = [0.0] * len(detections)
                
                return detections, uncertainties
                
        except Exception as e:
            log.debug(f"Inference error: {e}")
            return [], []
    
    def _extract_detections(self, raw_result) -> List[dict]:
        """Extract detections from YOLO result"""
        detections = []
        for box in raw_result.boxes:
            detections.append({
                "bbox": [int(v) for v in box.xyxy[0].tolist()],
                "bbox_norm": box.xyxyn[0].tolist(),
                "confidence": round(float(box.conf[0]), 3),
                "class": int(box.cls[0]),
                "class_name": self.cls_names.get(int(box.cls[0]), f"Class{int(box.cls[0])}")
            })
        return detections
    
    def _merge_mc_samples(self, samples: List[List[dict]]) -> List[dict]:
        """Merge MC dropout samples using simple NMS"""
        all_dets = []
        for sample in samples:
            all_dets.extend(sample)
        
        if not all_dets:
            return []
        
        # Simple NMS
        sorted_dets = sorted(all_dets, key=lambda d: -d['confidence'])
        kept = []
        used = [False] * len(sorted_dets)
        
        for i, d in enumerate(sorted_dets):
            if used[i]:
                continue
            
            kept.append(d)
            used[i] = True
            
            # Suppress overlapping
            for j in range(i+1, len(sorted_dets)):
                if not used[j] and self._iou(d['bbox'], sorted_dets[j]['bbox']) > 0.5:
                    used[j] = True
        
        return kept
    
    @staticmethod
    def _iou(b1, b2):
        x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
        x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
        inter = max(0, x2-x1) * max(0, y2-y1)
        if inter == 0: return 0.0
        a1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
        a2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
        return inter / (a1 + a2 - inter + 1e-6)

    def export_review_queue(self, path: str = None) -> str:
        """Export active learning queue"""
        if self.flagger:
            return self.flagger.export_review_queue(path)
        return ""

    def active_learning_stats(self) -> dict:
        if self.flagger:
            return self.flagger.stats()
        return {}


@dataclass
class DetectionResult:
    detections:     List[EnrichedDetection]
    frame_index:    int
    total_cost:     float
    uncertain_count: int
    needs_review_count: int = 0  # NEW: high uncertainty count

    def to_frame_dict(self, base: dict) -> dict:
        """Merge detection data into a frame dict for the dashboard."""
        dets = [d.to_dict() for d in self.detections]
        dominant = (max(self.detections,
                        key=lambda d: d.measurement.severity).class_name
                    if self.detections else "")
        total_cost = sum(d.measurement.cost_inr for d in self.detections)
        return {
            **base,
            "detections":      dets,
            "dominant_class":  dominant,
            "cost_inr":        round(total_cost, 0),
            "uncertain_count": self.uncertain_count,
            "needs_review_count": self.needs_review_count,
            "has_uncertain":   self.uncertain_count > 0,
            # Enriched fields
            "max_width_cm":    round(max((d.measurement.width_cm for d in self.detections), default=0), 1),
            "max_depth_cm":    round(max((d.measurement.depth_cm for d in self.detections), default=0), 1),
            "max_severity":    round(max((d.measurement.severity for d in self.detections), default=0), 1),
            "max_uncertainty": round(max((d.uncertainty for d in self.detections), default=0), 3),
        }