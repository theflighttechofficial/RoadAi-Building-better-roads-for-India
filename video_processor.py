"""
video_processor.py — Road Damage Video Analyzer

Production-grade video analysis pipeline built on RoadDamageDetector.
Features:
  - Typed FrameResult and SessionSummary dataclasses
  - Frame skipping with configurable stride
  - ROI (region-of-interest) masking — analyse only a crop of each frame
  - Per-frame callback hook for real-time side effects (live display, streaming)
  - Generator-based streaming mode for memory-efficient processing
  - Rolling statistics updated every frame (avg detections, worst frame, etc.)
  - Graceful error isolation — one bad frame never kills the session
  - Full session summary with class breakdown, flagged frames, and trend
  - Drop-in compatible: process_video() still returns a list
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Generator, Iterator, Optional

import cv2
import numpy as np

from detector import RoadDamageDetector

log = logging.getLogger(__name__)


# ── Typed result objects ───────────────────────────────────────────────────────

@dataclass
class FrameResult:
    """Complete output for a single processed frame."""

    frame_index:      int
    timestamp_sec:    float                       # position in video
    detections:       list[dict]
    n_detections:     int       = field(init=False)
    detection_error:  Optional[str] = None        # set if inference threw

    # Derived on init
    dominant_class:   str  = field(init=False)
    max_confidence:   float = field(init=False)
    total_bbox_area:  int   = field(init=False)

    # Class name lookup injected by VideoAnalyzer
    _class_names: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        self.n_detections    = len(self.detections)
        self.max_confidence  = max((d.get("confidence", 0) for d in self.detections), default=0.0)
        self.total_bbox_area = sum(self._bbox_area(d["bbox"]) for d in self.detections)

        if self.detections:
            cls_counter: dict[str, int] = defaultdict(int)
            for d in self.detections:
                name = self._class_names.get(d.get("class", -1), f"class_{d.get('class', '?')}")
                cls_counter[name] += 1
                d["class_name"] = name          # enrich detection in-place
            self.dominant_class = max(cls_counter, key=cls_counter.get)
        else:
            self.dominant_class = ""

    @staticmethod
    def _bbox_area(bbox: list[int]) -> int:
        x1, y1, x2, y2 = bbox
        return max(0, x2 - x1) * max(0, y2 - y1)

    @property
    def has_detections(self) -> bool:
        return self.n_detections > 0

    @property
    def is_high_damage(self) -> bool:
        return self.n_detections >= 5 or self.max_confidence >= 0.85

    def to_dict(self) -> dict:
        return {
            "frame_index":     self.frame_index,
            "timestamp_sec":   round(self.timestamp_sec, 3),
            "n_detections":    self.n_detections,
            "dominant_class":  self.dominant_class,
            "max_confidence":  round(self.max_confidence, 4),
            "total_bbox_area": self.total_bbox_area,
            "detection_error": self.detection_error,
            "detections":      self.detections,
        }


@dataclass
class SessionSummary:
    """Aggregated analytics across the full video session."""

    video_path:         str
    model_path:         str
    total_frames:       int
    processed_frames:   int
    skipped_frames:     int
    error_frames:       int
    elapsed_sec:        float
    fps_throughput:     float

    total_detections:   int
    avg_det_per_frame:  float
    max_det_frame:      int             # frame index with most detections
    max_det_count:      int
    high_damage_frames: int             # frames flagged as high-damage

    class_breakdown:    dict[str, int]  # class_name → total count
    score_trend:        str             # "improving" | "stable" | "worsening"
    flagged_frames:     list[int]       # indices of high-damage frames

    def __str__(self) -> str:
        sep = "─" * 52
        trend_sym = {"improving": "↑", "stable": "→", "worsening": "↓"}.get(
            self.score_trend, "→"
        )
        lines = [
            f"\n{sep}",
            f"  VIDEO SESSION SUMMARY",
            sep,
            f"  Video           : {Path(self.video_path).name}",
            f"  Model           : {self.model_path}",
            f"  Frames total    : {self.total_frames}",
            f"  Frames processed: {self.processed_frames}  (skipped {self.skipped_frames})",
            f"  Error frames    : {self.error_frames}",
            f"  Elapsed         : {self.elapsed_sec:.2f}s  ({self.fps_throughput:.1f} fps throughput)",
            f"\n  Detections",
            f"    Total         : {self.total_detections}",
            f"    Avg / frame   : {self.avg_det_per_frame:.2f}",
            f"    Peak frame    : #{self.max_det_frame}  ({self.max_det_count} detections)",
            f"    High-damage   : {self.high_damage_frames} frame(s)",
            f"    Trend         : {trend_sym} {self.score_trend}",
        ]
        if self.class_breakdown:
            lines.append(f"\n  Class breakdown:")
            for cls, cnt in sorted(self.class_breakdown.items(), key=lambda x: -x[1]):
                bar = "█" * min(cnt, 36)
                lines.append(f"    {cls:<24} {bar} {cnt}")
        if self.flagged_frames:
            preview = self.flagged_frames[:8]
            lines.append(f"\n  ⚠  Flagged frames: {self.flagged_frames[:8]}"
                         + (" …" if len(self.flagged_frames) > 8 else ""))
        lines.append(sep)
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "video_path":         self.video_path,
            "model_path":         self.model_path,
            "total_frames":       self.total_frames,
            "processed_frames":   self.processed_frames,
            "skipped_frames":     self.skipped_frames,
            "error_frames":       self.error_frames,
            "elapsed_sec":        round(self.elapsed_sec, 3),
            "fps_throughput":     round(self.fps_throughput, 2),
            "total_detections":   self.total_detections,
            "avg_det_per_frame":  round(self.avg_det_per_frame, 3),
            "max_det_frame":      self.max_det_frame,
            "max_det_count":      self.max_det_count,
            "high_damage_frames": self.high_damage_frames,
            "class_breakdown":    self.class_breakdown,
            "score_trend":        self.score_trend,
            "flagged_frames":     self.flagged_frames,
        }


# ── Rolling stats tracker ──────────────────────────────────────────────────────

class _RollingStats:
    """Lightweight rolling accumulator — updated every frame."""

    def __init__(self, trend_window: int = 30):
        self.total_detections  = 0
        self.error_frames      = 0
        self.high_damage       = 0
        self.max_det_count     = 0
        self.max_det_frame     = 0
        self.class_counts: dict[str, int] = defaultdict(int)
        self.flagged: list[int] = []

        # For trend: rolling detection counts
        self._window           = trend_window
        self._recent: deque[int] = deque(maxlen=trend_window * 2)

    def update(self, result: FrameResult) -> None:
        n = result.n_detections
        self.total_detections += n
        self._recent.append(n)

        if result.detection_error:
            self.error_frames += 1

        if result.is_high_damage:
            self.high_damage  += 1
            self.flagged.append(result.frame_index)

        if n > self.max_det_count:
            self.max_det_count = n
            self.max_det_frame = result.frame_index

        for d in result.detections:
            self.class_counts[d.get("class_name", "Unknown")] += 1

    def trend(self) -> str:
        buf = list(self._recent)
        if len(buf) < self._window * 2:
            return "stable"
        first = sum(buf[: self._window]) / self._window
        last  = sum(buf[-self._window :]) / self._window
        diff  = last - first
        if diff > 1.5:  return "worsening"    # more damage lately
        if diff < -1.5: return "improving"
        return "stable"


# ── Progress bar ───────────────────────────────────────────────────────────────

class _ProgressBar:
    def __init__(self, total: int, width: int = 40):
        self.total = max(total, 1)
        self.width = width
        self.start = time.time()

    def update(self, n: int) -> None:
        pct     = n / self.total
        filled  = int(self.width * pct)
        bar     = "█" * filled + "░" * (self.width - filled)
        elapsed = time.time() - self.start
        eta     = (elapsed / max(n, 1)) * (self.total - n)
        print(f"\r  [{bar}] {pct:>6.1%}  frame {n}/{self.total}  ETA {eta:.0f}s",
              end="", flush=True)

    def done(self) -> None:
        print()


# ── Main class ────────────────────────────────────────────────────────────────

class VideoAnalyzer:
    """
    Road damage video analysis pipeline.

    Parameters
    ----------
    model_path : str
        Path to YOLO weights file (e.g. "best.pt").
    class_names : dict, optional
        Maps class_id (int) → human label (str).
        Defaults cover the four standard road-damage classes.
    conf_threshold : float
        Minimum confidence to keep a detection.
    skip_frames : int
        Process every Nth frame; pass-through the rest unmarked.
        1 = process all frames.
    roi : tuple (x1, y1, x2, y2), optional
        Pixel crop applied before inference — useful to ignore dashboards
        or sky regions that generate false positives.
    high_damage_threshold : int
        Detection count per frame above which a frame is flagged.
    show_progress : bool
        Print a live progress bar to stdout.
    on_frame : Callable[[FrameResult], None], optional
        Called after every processed frame — use for live display,
        websocket streaming, or per-frame scoring.
    """

    DEFAULT_CLASS_NAMES = {
        0: "Longitudinal Crack",
        1: "Transverse Crack",
        2: "Alligator Crack",
        3: "Pothole",
    }

    def __init__(
        self,
        model_path: str,
        class_names:            Optional[dict]                       = None,
        conf_threshold:         float                                = 0.35,
        skip_frames:            int                                  = 1,
        roi:                    Optional[tuple[int,int,int,int]]     = None,
        high_damage_threshold:  int                                  = 5,
        show_progress:          bool                                 = True,
        on_frame:               Optional[Callable[[FrameResult], None]] = None,
    ):
        if not Path(model_path).exists() and not model_path.startswith("yolov"):
            raise FileNotFoundError(f"Model weights not found: {model_path}")

        self.detector              = RoadDamageDetector(model_path)
        self.model_path            = model_path
        self.class_names           = {**self.DEFAULT_CLASS_NAMES, **(class_names or {})}
        self.conf_threshold        = conf_threshold
        self.skip_frames           = max(1, skip_frames)
        self.roi                   = roi
        self.high_damage_threshold = high_damage_threshold
        self.show_progress         = show_progress
        self.on_frame              = on_frame

        log.info(f"VideoAnalyzer ready — model={model_path}  "
                 f"conf≥{conf_threshold}  skip={skip_frames}")

    # ── Public API ─────────────────────────────────────────────────────────────

    def process_video(
        self,
        video_path: str,
        max_frames: Optional[int] = None,
    ) -> list[FrameResult]:
        """
        Process the full video and return all FrameResults as a list.
        Backward-compatible drop-in for the original API.

        Parameters
        ----------
        video_path  : path to the video file.
        max_frames  : stop after this many frames (None = process all).
        """
        return list(self.stream_video(video_path, max_frames=max_frames))

    def stream_video(
        self,
        video_path: str,
        max_frames: Optional[int] = None,
    ) -> Generator[FrameResult, None, None]:
        """
        Memory-efficient generator — yields one FrameResult per processed frame.
        Use when piping results directly into a scorer or writer without
        accumulating all frames in RAM.

        Example
        -------
        for result in analyzer.stream_video("road.mp4"):
            score = scorer.score_frame(result.detections)
            writer.write(annotate(frame, result))
        """
        cap, meta = self._open_video(video_path)
        stats     = _RollingStats()
        progress  = _ProgressBar(meta["total_frames"]) if self.show_progress else None

        log.info(f"Processing '{Path(video_path).name}'  "
                 f"{meta['width']}×{meta['height']} @ {meta['fps']:.1f}fps  "
                 f"({meta['total_frames']} frames)")

        t_start = time.time()
        frame_index = 0
        processed   = 0

        try:
            while True:
                ret, raw_frame = cap.read()
                if not ret or (max_frames and frame_index >= max_frames):
                    break

                timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

                # Skip
                if frame_index % self.skip_frames != 0:
                    frame_index += 1
                    if progress:
                        progress.update(frame_index)
                    continue

                # Crop to ROI if set
                frame = self._apply_roi(raw_frame)

                # Detect (isolated — one bad frame never kills the loop)
                detections, error = self._safe_detect(frame)

                result = FrameResult(
                    frame_index=frame_index,
                    timestamp_sec=timestamp,
                    detections=detections,
                    detection_error=error,
                    _class_names=self.class_names,
                )

                stats.update(result)
                processed += 1

                if self.on_frame:
                    try:
                        self.on_frame(result)
                    except Exception as e:
                        log.warning(f"on_frame callback raised at frame {frame_index}: {e}")

                if progress:
                    progress.update(frame_index + 1)

                yield result
                frame_index += 1

        except KeyboardInterrupt:
            log.warning("Processing interrupted — partial results returned.")
        finally:
            cap.release()
            if progress:
                progress.done()

        elapsed = time.time() - t_start
        summary = self._build_summary(
            video_path, meta, frame_index, processed, stats, elapsed
        )
        log.info(summary)

    def summarise(
        self,
        results: list[FrameResult],
        video_path: str = "",
        elapsed_sec: float = 0.0,
    ) -> SessionSummary:
        """
        Build a SessionSummary from a completed result list.
        Useful when you called process_video() and want analytics afterward.
        """
        stats = _RollingStats()
        for r in results:
            stats.update(r)

        total = len(results)
        avg   = stats.total_detections / max(total, 1)

        return SessionSummary(
            video_path=video_path,
            model_path=self.model_path,
            total_frames=total,
            processed_frames=total,
            skipped_frames=0,
            error_frames=stats.error_frames,
            elapsed_sec=elapsed_sec,
            fps_throughput=total / max(elapsed_sec, 1e-6),
            total_detections=stats.total_detections,
            avg_det_per_frame=round(avg, 3),
            max_det_frame=stats.max_det_frame,
            max_det_count=stats.max_det_count,
            high_damage_frames=stats.high_damage,
            class_breakdown=dict(stats.class_counts),
            score_trend=stats.trend(),
            flagged_frames=stats.flagged,
        )

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _open_video(self, video_path: str) -> tuple[cv2.VideoCapture, dict]:
        if not Path(video_path).exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        return cap, {
            "width":        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height":       int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps":          cap.get(cv2.CAP_PROP_FPS) or 30.0,
            "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        }

    def _apply_roi(self, frame: np.ndarray) -> np.ndarray:
        if self.roi is None:
            return frame
        x1, y1, x2, y2 = self.roi
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        return frame[y1:y2, x1:x2]

    def _safe_detect(self, frame: np.ndarray) -> tuple[list[dict], Optional[str]]:
        try:
            detections = self.detector.detect(frame)
            # Filter by confidence threshold
            detections = [
                d for d in detections
                if d.get("confidence", 1.0) >= self.conf_threshold
            ]
            return detections, None
        except Exception as e:
            log.warning(f"Detection error: {e}")
            return [], str(e)

    def _build_summary(
        self,
        video_path: str,
        meta: dict,
        total_frames: int,
        processed: int,
        stats: _RollingStats,
        elapsed: float,
    ) -> SessionSummary:
        avg = stats.total_detections / max(processed, 1)
        return SessionSummary(
            video_path=video_path,
            model_path=self.model_path,
            total_frames=meta["total_frames"],
            processed_frames=processed,
            skipped_frames=total_frames - processed,
            error_frames=stats.error_frames,
            elapsed_sec=round(elapsed, 2),
            fps_throughput=round(processed / max(elapsed, 1e-6), 2),
            total_detections=stats.total_detections,
            avg_det_per_frame=round(avg, 3),
            max_det_frame=stats.max_det_frame,
            max_det_count=stats.max_det_count,
            high_damage_frames=stats.high_damage,
            class_breakdown=dict(stats.class_counts),
            score_trend=stats.trend(),
            flagged_frames=stats.flagged,
        )


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    VIDEO = r"C:/Users/DELL/Downloads/road_damage_ai/3657637695-preview.mp4"
    MODEL = "best.pt"

    # Optional: live callback — e.g. print every high-damage frame
    def on_frame(result: FrameResult) -> None:
        if result.is_high_damage:
            print(f"  ⚠  Frame {result.frame_index:>5} @ {result.timestamp_sec:.2f}s — "
                  f"{result.n_detections} detections  [{result.dominant_class}]")

    analyzer = VideoAnalyzer(
        model_path=MODEL,
        conf_threshold=0.35,
        skip_frames=1,
        show_progress=True,
        on_frame=on_frame,
        # roi=(0, 200, 1280, 680),   # optional: ignore top sky & bottom dashboard
    )

    # ── Drop-in: returns list (original API) ──────────────────────────────────
    t0      = time.time()
    results = analyzer.process_video(VIDEO)
    elapsed = time.time() - t0

    summary = analyzer.summarise(results, VIDEO, elapsed)
    print(summary)

    # ── Export summary JSON ───────────────────────────────────────────────────
    out = Path("output/session_summary.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary.to_dict(), indent=2))
    log.info(f"Summary → {out.resolve()}")

    # ── Streaming example (memory-efficient) ─────────────────────────────────
    # for result in analyzer.stream_video(VIDEO):
    #     score = scorer.score_frame(result.detections, result.frame_index)
    #     writer.write(annotate(frame, result))