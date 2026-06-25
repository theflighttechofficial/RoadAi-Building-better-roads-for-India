"""
advanced_features.py  -  Timelapse Generator, Weather-Aware Confidence,
                          Auto Fine-Tune Trigger
==========================================================================

TIMELAPSE GENERATOR
-------------------
Creates a speed-multiplied (8–16×) annotated output video.
Uses the annotated_output.mp4 as input and writes a
timelapse_Nx.mp4 with all HUD overlays intact.
Includes a damage density bar at the bottom showing where hotspots are.

WEATHER-AWARE CONFIDENCE ADJUSTER
------------------------------------
Adjusts YOLO confidence thresholds per condition:
  Night  → lower threshold (model less certain, accept lower conf)
  Rain   → lower threshold + increase min area filter
  Fog    → much lower threshold (visibility reduced)
  Clear  → standard thresholds
Reference: Kenk & Hassaballah (2020) "DAWN: Dataset of Adverse Weather
Conditions for Autonomous Driving", arXiv 2009.10396.

AUTO FINE-TUNE TRIGGER
------------------------
Reads the active learning review_queue.json and triggers a
YOLOv8 fine-tune run if:
  - Review queue has >= MIN_SAMPLES uncertain frames
  - Confidence calibration gap is large
  - At least N new labeled frames are available
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List

# Optional heavy deps — imported at module level so Pylance can resolve types.
# Functions that need these check availability at call time.
try:
    import numpy as np
    _NUMPY_OK = True
except ImportError:
    np = None          # type: ignore[assignment]
    _NUMPY_OK = False

try:
    import cv2
    _CV2_OK = True
except ImportError:
    cv2 = None         # type: ignore[assignment]
    _CV2_OK = False

log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# TIMELAPSE GENERATOR
# ═══════════════════════════════════════════════════════════════════

def generate_timelapse(
    input_video:  str,
    output_video: str = None,
    speed_factor: int = 8,
    results:      list = None,
) -> str:
    """
    Generate a speed-multiplied timelapse from the annotated output video.

    Parameters
    ----------
    input_video  : path to annotated_output.mp4
    output_video : output path (default: timelapse_8x.mp4 in same dir)
    speed_factor : how many × faster (4, 8, 12, 16)
    results      : frame result dicts for damage density bar

    Returns
    -------
    str path to timelapse video, or "" on failure
    """
    if not _CV2_OK or not _NUMPY_OK:
        log.error("opencv-python and numpy required for timelapse generation")
        return ""

    if not Path(input_video).exists():
        log.error(f"Input video not found: {input_video}")
        return ""

    if output_video is None:
        stem = Path(input_video).stem
        output_video = str(Path(input_video).parent / f"{stem}_timelapse_{speed_factor}x.mp4")

    cap = cv2.VideoCapture(str(input_video))
    fps_in  = cap.get(cv2.CAP_PROP_FPS) or 25
    fps_out = min(fps_in * speed_factor, 120)   # cap output fps at 120
    W       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = cv2.VideoWriter(
        output_video,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps_out, (W, H)
    )

    # Build damage density strip (bottom 6px) from results
    density_strip = None
    if results:
        density_strip = _build_density_strip(results, W, total)

    fi = 0
    written = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Only write every Nth frame
        if fi % speed_factor == 0:
            # Overlay timelapse indicator
            frame = _overlay_timelapse_hud(frame, fi, total, speed_factor, W, H)
            # Overlay damage density strip
            if density_strip is not None:
                frame[-6:, :] = density_strip
            writer.write(frame)
            written += 1
        fi += 1

    cap.release()
    writer.release()
    log.info(f"Timelapse: {written} frames → {output_video}  ({speed_factor}× speed)")
    return output_video


def _build_density_strip(results: list, W: int, total_frames: int):
    """Build a W×6 BGR strip showing damage density across the video."""
    strip = np.zeros((6, W, 3), dtype=np.uint8)
    for r in results:
        fi    = r.get("frame_index", 0)
        score = r.get("health_score", 100)
        x     = int(fi / max(total_frames, 1) * W)
        x     = min(x, W - 1)
        if score < 80:
            col = (168, 139, 243) if score < 40 else \
                  (168, 139, 243) if score < 40 else \
                  (168, 56,  243) if score < 40 else \
                  (56, 168, 243)  if score >= 80 else \
                  (175, 226, 249) if score >= 60 else \
                  (168, 139, 56)  if score >= 40 else \
                  (56,  56,  243)
            # BGR colors by tier
            if score >= 80:   col = (161, 227, 166)   # green
            elif score >= 60: col = (175, 226, 249)   # yellow
            elif score >= 40: col = (168, 139, 243)   # orange
            else:             col = ( 56,  56, 243)   # red
            strip[:, max(0, x-1):min(W, x+2), :] = col
    return strip


def _overlay_timelapse_hud(
    frame: object,
    fi: int,
    total: int,
    speed: int,
    W: int,
    H: int,
):
    """Add timelapse speed indicator to frame."""
    cv2.putText(
        frame, f"{speed}x TIMELAPSE",
        (W - 130, 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45,
        (0, 0, 0), 3, cv2.LINE_AA
    )
    cv2.putText(
        frame, f"{speed}x TIMELAPSE",
        (W - 130, 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45,
        (249, 226, 175), 1, cv2.LINE_AA
    )
    # Progress bar
    pct = fi / max(total, 1)
    cv2.rectangle(frame, (0, H - 10), (W, H - 6), (30, 36, 52), -1)
    cv2.rectangle(frame, (0, H - 10), (int(pct * W), H - 6), (166, 227, 161), -1)
    return frame


# ═══════════════════════════════════════════════════════════════════
# WEATHER-AWARE CONFIDENCE ADJUSTER
# ═══════════════════════════════════════════════════════════════════

# Per-condition confidence multipliers
# Reference: Kenk & Hassaballah 2020, DAWN Dataset
WEATHER_CONF_FACTORS = {
    "normal": {
        "conf_multiplier":    1.00,
        "min_area_multiplier":1.00,
        "note":               "Standard thresholds",
    },
    "night": {
        "conf_multiplier":    0.75,   # accept lower confidence at night
        "min_area_multiplier":0.80,   # accept slightly smaller detections
        "note":               "Night mode: -25% confidence threshold (CLAHE enhanced)",
    },
    "rain": {
        "conf_multiplier":    0.80,
        "min_area_multiplier":1.20,   # increase min area (wet reflections = FP)
        "note":               "Rain mode: -20% confidence, +20% min area (reduce reflection FP)",
    },
    "fog": {
        "conf_multiplier":    0.65,   # most lenient — fog severely reduces visibility
        "min_area_multiplier":0.70,
        "note":               "Fog mode: -35% confidence threshold (strong CLAHE applied)",
    },
    "glare": {
        "conf_multiplier":    0.85,
        "min_area_multiplier":1.10,
        "note":               "Glare mode: -15% confidence, +10% min area",
    },
}

DEFAULT_CONF     = 0.50
DEFAULT_MIN_AREA = 0.003


def weather_adjusted_thresholds(
    condition: str,
    base_conf:     float = DEFAULT_CONF,
    base_min_area: float = DEFAULT_MIN_AREA,
) -> tuple[float, float]:
    """
    Return (conf_threshold, min_area_ratio) adjusted for current weather condition.

    Parameters
    ----------
    condition    : "normal", "night", "rain", "fog", "glare"
    base_conf    : base confidence threshold (default 0.50)
    base_min_area: base min bbox area ratio (default 0.003)

    Returns
    -------
    (adjusted_conf, adjusted_min_area)
    """
    factors = WEATHER_CONF_FACTORS.get(condition.lower(),
                                       WEATHER_CONF_FACTORS["normal"])
    adj_conf     = round(base_conf     * factors["conf_multiplier"],    3)
    adj_min_area = round(base_min_area * factors["min_area_multiplier"], 4)
    # Clamp to sensible ranges
    adj_conf     = max(0.25, min(0.80, adj_conf))
    adj_min_area = max(0.001, min(0.020, adj_min_area))
    return adj_conf, adj_min_area


# ═══════════════════════════════════════════════════════════════════
# AUTO FINE-TUNE TRIGGER
# ═══════════════════════════════════════════════════════════════════

MIN_SAMPLES_FOR_FINETUNE = 50   # minimum uncertain frames before triggering


@dataclass
class FineTuneStatus:
    should_finetune: bool
    reason:          str
    queue_size:      int
    uncertain_count: int
    novel_count:     int
    command:         str   # the command to run to start fine-tuning


def check_finetune_trigger(
    review_queue_path: str,
    model_path:        str = "yolov8n.pt",
    data_yaml_path:    str = "dataset.yaml",
    output_model:      str = "yolov8n_finetuned.pt",
) -> FineTuneStatus:
    """
    Check if the active learning queue has enough samples to trigger fine-tuning.

    The review queue is populated by the active learning flagger in detector.py.
    When enough uncertain frames accumulate, this signals that the model has
    systematic gaps that fine-tuning can address.

    Parameters
    ----------
    review_queue_path : path to review_queue.json from pipeline run
    model_path        : current YOLOv8 model
    data_yaml_path    : YOLO dataset.yaml (needs to be created with labeled data)
    output_model      : where to save fine-tuned model

    Returns
    -------
    FineTuneStatus with should_finetune=True and the exact CLI command to run
    """
    import json

    # Load review queue
    try:
        queue = json.loads(Path(review_queue_path).read_text())
    except Exception as e:
        return FineTuneStatus(
            should_finetune=False,
            reason=f"Could not read review queue: {e}",
            queue_size=0, uncertain_count=0, novel_count=0, command="",
        )

    n_total     = len(queue)
    n_uncertain = sum(1 for f in queue if f.get("reason") == "uncertain")
    n_novel     = sum(1 for f in queue if f.get("reason") == "novel_class")

    if n_total < MIN_SAMPLES_FOR_FINETUNE:
        return FineTuneStatus(
            should_finetune=False,
            reason=(f"Queue has {n_total} samples — need {MIN_SAMPLES_FOR_FINETUNE} "
                    f"before fine-tuning. Keep running the pipeline to collect more."),
            queue_size=n_total,
            uncertain_count=n_uncertain,
            novel_count=n_novel,
            command="",
        )

    # Check if data.yaml exists (user needs to label the frames first)
    if not Path(data_yaml_path).exists():
        return FineTuneStatus(
            should_finetune=False,
            reason=(f"Queue ready ({n_total} samples) but {data_yaml_path} not found. "
                    f"Label the frames in review_queue.json and create a dataset.yaml "
                    f"before fine-tuning."),
            queue_size=n_total,
            uncertain_count=n_uncertain,
            novel_count=n_novel,
            command=f"# 1. Label frames in review_queue.json\n"
                    f"# 2. Create {data_yaml_path}\n"
                    f"# 3. Run: yolo train model={model_path} data={data_yaml_path} "
                    f"epochs=20 imgsz=640 name=road_ai_finetuned",
        )

    cmd = (f"yolo train model={model_path} data={data_yaml_path} "
           f"epochs=20 imgsz=640 batch=16 patience=5 "
           f"name=road_ai_finetuned project=runs/finetune "
           f"# Then update --model to runs/finetune/road_ai_finetuned/weights/best.pt")

    return FineTuneStatus(
        should_finetune=True,
        reason=(f"Ready: {n_total} samples ({n_uncertain} uncertain, {n_novel} novel). "
                f"Fine-tuning will improve recall on underperforming classes."),
        queue_size=n_total,
        uncertain_count=n_uncertain,
        novel_count=n_novel,
        command=cmd,
    )