"""
stream_processor.py — Speed-Adaptive Processing + RTSP Live Stream

Handles:
  1. Speed-adaptive frame skipping — process densely in high-damage zones,
     skip aggressively in clean zones
  2. RTSP/webcam/file unified source abstraction
  3. Live stream reconnection with exponential backoff
"""

from __future__ import annotations

import logging
import time
import threading
import queue
from dataclasses import dataclass, field
from collections import deque
from typing import Optional, Callable, Generator
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)


# ── Video source abstraction ─────────────────────────────────────────────────

@dataclass
class StreamConfig:
    source:              str | int          # path, RTSP URL, or webcam index
    reconnect:           bool  = True       # auto-reconnect on RTSP drop
    reconnect_delay:     float = 2.0        # seconds between retries
    max_reconnects:      int   = 10
    buffer_size:         int   = 4          # frame buffer for async reads
    rtsp_transport:      str   = "tcp"      # "tcp" or "udp"

    @property
    def is_rtsp(self) -> bool:
        s = str(self.source)
        return s.startswith("rtsp://") or s.startswith("rtsps://")

    @property
    def is_file(self) -> bool:
        return Path(str(self.source)).exists()

    @property
    def is_webcam(self) -> bool:
        return isinstance(self.source, int)


class VideoSource:
    """
    Unified video source that handles files, webcams, and RTSP streams.
    RTSP streams use a background thread + queue to decouple decode from inference.
    """

    def __init__(self, cfg: StreamConfig):
        self.cfg     = cfg
        self._cap:   Optional[cv2.VideoCapture] = None
        self._q:     queue.Queue = queue.Queue(maxsize=cfg.buffer_size)
        self._thread: Optional[threading.Thread] = None
        self._stop    = threading.Event()
        self._reconnects = 0
        self._open()

    # ── Public ──────────────────────────────────────────────────────────────

    @property
    def fps(self) -> float:
        return self._cap.get(cv2.CAP_PROP_FPS) or 30.0 if self._cap else 30.0

    @property
    def width(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if self._cap else 0

    @property
    def height(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if self._cap else 0

    @property
    def total_frames(self) -> int:
        if self.cfg.is_rtsp or self.cfg.is_webcam:
            return -1   # unknown
        return int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT)) if self._cap else 0

    def read(self) -> tuple[bool, Optional[np.ndarray]]:
        if self.cfg.is_rtsp:
            try:
                frame = self._q.get(timeout=3.0)
                return True, frame
            except queue.Empty:
                log.warning("RTSP frame timeout.")
                return False, None
        if self._cap is None:
            return False, None
        return self._cap.read()

    def release(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)
        if self._cap:
            self._cap.release()

    # ── Internal ────────────────────────────────────────────────────────────

    def _open(self) -> bool:
        source = self.cfg.source
        if self.cfg.is_rtsp:
            source_str = str(source)
            cap = cv2.VideoCapture(source_str, cv2.CAP_FFMPEG)
            # Force TCP transport for reliability
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if self.cfg.rtsp_transport == "tcp":
                cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
        else:
            cap = cv2.VideoCapture(source)

        if not cap.isOpened():
            log.error(f"Cannot open source: {source}")
            return False

        self._cap = cap

        if self.cfg.is_rtsp:
            self._stop.clear()
            self._thread = threading.Thread(
                target=self._rtsp_reader, daemon=True
            )
            self._thread.start()

        log.info(f"Source opened: {source}  "
                 f"{self.width}×{self.height} @ {self.fps:.1f}fps")
        return True

    def _rtsp_reader(self) -> None:
        """Background thread — continuously reads RTSP frames into queue."""
        while not self._stop.is_set():
            ret, frame = self._cap.read()
            if not ret:
                if self.cfg.reconnect and self._reconnects < self.cfg.max_reconnects:
                    delay = self.cfg.reconnect_delay * (2 ** self._reconnects)
                    log.warning(f"RTSP stream lost. Reconnecting in {delay:.1f}s "
                                f"(attempt {self._reconnects + 1}/{self.cfg.max_reconnects})")
                    time.sleep(delay)
                    self._cap.release()
                    if self._open():
                        self._reconnects += 1
                        continue
                else:
                    log.error("RTSP reconnection limit reached — stopping.")
                    break
                continue
            self._reconnects = 0    # reset on successful read
            # Drop old frames if queue full (keep latest)
            if self._q.full():
                try:
                    self._q.get_nowait()
                except queue.Empty:
                    pass
            self._q.put(frame)


# ── Speed-adaptive skip controller ──────────────────────────────────────────

@dataclass
class AdaptiveSkipConfig:
    """Controls how aggressively frames are skipped based on damage activity."""
    base_skip:       int   = 1     # default: process every frame
    min_skip:        int   = 1     # always process at least every Nth
    max_skip:        int   = 8     # skip up to 8 frames when road is clean
    damage_window:   int   = 15    # look back N frames for activity
    damage_low:      float = 0.5   # avg det/frame below this → increase skip
    damage_high:     float = 3.0   # avg det/frame above this → decrease skip
    speed_factor:    float = 1.0   # multiply skip by GPS speed / 50 km/h


class AdaptiveSkipController:
    """
    Dynamically adjusts how many frames to skip between processed frames.

    - High damage activity   → skip fewer frames (don't miss anything)
    - Clean road / high speed → skip more frames (save GPU time)

    Usage
    -----
    ctrl = AdaptiveSkipController()
    for frame_index, frame in enumerate(source):
        if ctrl.should_process(frame_index):
            detections = detector.detect(frame)
            ctrl.update(len(detections))
    """

    def __init__(self, cfg: Optional[AdaptiveSkipConfig] = None):
        self.cfg     = cfg or AdaptiveSkipConfig()
        self._skip   = self.cfg.base_skip
        self._history: deque[int] = deque(maxlen=self.cfg.damage_window)
        self._next_process = 0

    @property
    def current_skip(self) -> int:
        return self._skip

    def should_process(self, frame_index: int) -> bool:
        return frame_index >= self._next_process

    def update(self, n_detections: int, speed_kmh: float = 50.0) -> int:
        """
        Call after processing a frame. Returns the new skip value.

        Parameters
        ----------
        n_detections : detections found in the last processed frame
        speed_kmh    : current GPS speed (higher speed → fewer frames/metre)
        """
        self._history.append(n_detections)
        avg = sum(self._history) / max(len(self._history), 1)

        # Speed adjustment: at 100 km/h we cover ground 2× faster
        speed_mult = max(0.5, min(2.0, speed_kmh / 50.0)) * self.cfg.speed_factor

        if avg >= self.cfg.damage_high:
            # Damage zone — process more frames
            self._skip = max(self.cfg.min_skip, self._skip - 1)
        elif avg <= self.cfg.damage_low:
            # Clean road — process fewer frames
            self._skip = min(self.cfg.max_skip,
                             int(self._skip * speed_mult) + 1)
        # else: hold current skip

        self._next_process += self._skip
        return self._skip

    def reset(self) -> None:
        self._skip = self.cfg.base_skip
        self._history.clear()
        self._next_process = 0

    def stats(self) -> dict:
        return {
            "current_skip":  self._skip,
            "avg_detections": round(
                sum(self._history) / max(len(self._history), 1), 2
            ),
            "frames_saved_pct": round((1 - 1/self._skip) * 100, 1),
        }


# ── Combined live pipeline ───────────────────────────────────────────────────

def stream_frames(
    source_cfg:  StreamConfig,
    skip_cfg:    Optional[AdaptiveSkipConfig] = None,
    max_frames:  Optional[int] = None,
    on_skip:     Optional[Callable[[int], None]] = None,
) -> Generator[tuple[int, float, np.ndarray], None, None]:
    """
    Generator yielding (frame_index, timestamp_sec, frame) for frames
    selected by the adaptive skip controller.

    Parameters
    ----------
    source_cfg  : StreamConfig defining the video/RTSP source.
    skip_cfg    : AdaptiveSkipConfig — pass None for fixed skip=1.
    max_frames  : stop after this many total frames read (None = unlimited).
    on_skip     : called with frame_index for each skipped frame.

    Yields
    ------
    (frame_index, timestamp_sec, frame_bgr)

    Usage
    -----
    for idx, ts, frame in stream_frames(StreamConfig("rtsp://192.168.1.10/stream")):
        detections = detector.detect(frame)
        skip_ctrl.update(len(detections), gps_speed)
    """
    src  = VideoSource(source_cfg)
    ctrl = AdaptiveSkipController(skip_cfg)

    frame_index = 0
    try:
        while True:
            ret, frame = src.read()
            if not ret or frame is None:
                break
            if max_frames and frame_index >= max_frames:
                break

            timestamp = frame_index / max(src.fps, 1e-6)

            if ctrl.should_process(frame_index):
                yield frame_index, timestamp, frame
            else:
                if on_skip:
                    on_skip(frame_index)

            frame_index += 1
    finally:
        src.release()
        log.info(f"Stream closed after {frame_index} frames.  "
                 f"Skip stats: {ctrl.stats()}")
