"""
video_writer.py — Road Damage Video Processor
Renders a professional HUD overlay on each frame with:
  - Severity-coded bounding boxes with corner brackets
  - Per-frame detection count + rolling damage density bar
  - Heads-up stats panel (frame #, FPS, total detections, worst class)
  - Semi-transparent alert banner on high-damage frames
  - End-of-run terminal summary
"""

import cv2
import sys
import time
import logging
import numpy as np
from pathlib import Path
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import Optional

from video_processor import VideoAnalyzer

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# ── Palette (BGR) ─────────────────────────────────────────────────────────────
class C:
    GOOD     = (0,   210, 100)   # green
    MODERATE = (0,   165, 255)   # orange
    POOR     = (45,   45, 220)   # red
    WHITE    = (240, 240, 240)
    BLACK    = (10,   10,  10)
    DIM      = (120, 120, 120)
    ACCENT   = (0,   210, 100)


# ── Config ────────────────────────────────────────────────────────────────────
@dataclass
class RenderConfig:
    video_path:    str   = r"C:/Users/DELL/Downloads/road_damage_ai/3657637695-preview.mp4"
    output_path:   str   = "output_video.mp4"
    model_path:    str   = "best.pt"
    conf_threshold: float = 0.35
    skip_frames:   int   = 1       # process every Nth frame (1 = all)
    alert_threshold: int = 5       # detections/frame to trigger alert banner
    history_len:   int   = 60      # frames to keep for rolling density bar

    # Class name map — adjust to match your model's labels
    class_names: dict = field(default_factory=lambda: {
        0: "Longitudinal Crack",
        1: "Transverse Crack",
        2: "Alligator Crack",
        3: "Pothole",
    })

    def validate(self) -> None:
        if not Path(self.video_path).exists():
            raise FileNotFoundError(f"Video not found: {self.video_path}")

    def severity(self, conf: float) -> str:
        if conf >= 0.75: return "poor"
        if conf >= 0.50: return "moderate"
        return "good"

    def sev_color(self, conf: float) -> tuple:
        return {
            "poor":     C.POOR,
            "moderate": C.MODERATE,
            "good":     C.GOOD,
        }[self.severity(conf)]


# ── Drawing helpers ───────────────────────────────────────────────────────────

def draw_corner_box(img, x1, y1, x2, y2, color, thickness=2, ratio=0.22):
    """Draws four corner brackets instead of a full rectangle."""
    w, h   = x2 - x1, y2 - y1
    lx, ly = max(8, int(w * ratio)), max(8, int(h * ratio))

    corners = [
        [(x1, y1 + ly), (x1, y1), (x1 + lx, y1)],
        [(x2 - lx, y1), (x2, y1), (x2, y1 + ly)],
        [(x1, y2 - ly), (x1, y2), (x1 + lx, y2)],
        [(x2 - lx, y2), (x2, y2), (x2, y2 - ly)],
    ]
    for pts in corners:
        for i in range(len(pts) - 1):
            cv2.line(img, pts[i], pts[i + 1], color, thickness, cv2.LINE_AA)


def draw_label_pill(img, text, x, y, color, font_scale=0.45, thickness=1):
    """Draws a filled pill label above a bounding box."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), bl = cv2.getTextSize(text, font, font_scale, thickness)
    pad = 4
    px1, py1 = x, y - th - pad * 2
    px2, py2 = x + tw + pad * 2, y

    overlay = img.copy()
    cv2.rectangle(overlay, (px1, py1), (px2, py2), color, -1)
    cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)
    cv2.putText(img, text, (px1 + pad, py2 - pad),
                font, font_scale, C.WHITE, thickness, cv2.LINE_AA)


def draw_semi_rect(img, x1, y1, x2, y2, color, alpha=0.35):
    """Transparent filled rectangle."""
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def put_text(img, text, x, y, color=C.WHITE, scale=0.5, thickness=1, mono=False):
    font = cv2.FONT_HERSHEY_DUPLEX if not mono else cv2.FONT_HERSHEY_PLAIN
    cv2.putText(img, text, (x, y), font, scale, C.BLACK, thickness + 1, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


# ── HUD Renderer ──────────────────────────────────────────────────────────────

class HUDRenderer:
    """Manages all on-frame overlay drawing."""

    PANEL_W  = 230
    PANEL_H  = 130
    MARGIN   = 12
    BAR_H    = 6

    def __init__(self, frame_w: int, frame_h: int, cfg: RenderConfig):
        self.fw, self.fh = frame_w, frame_h
        self.cfg         = cfg
        self.history: deque[int] = deque(maxlen=cfg.history_len)
        self.total_detections    = 0
        self.class_counts: defaultdict = defaultdict(int)

    def render(self, frame: np.ndarray, detections: list, frame_id: int,
               fps_live: float) -> np.ndarray:

        self.history.append(len(detections))
        self.total_detections += len(detections)
        for d in detections:
            self.class_counts[d.get("class_name", "Unknown")] += 1

        # 1. Bounding boxes
        self._draw_boxes(frame, detections)

        # 2. Stats panel (top-left)
        self._draw_stats_panel(frame, frame_id, fps_live, len(detections))

        # 3. Density bar (bottom strip)
        self._draw_density_bar(frame)

        # 4. Alert banner (conditional)
        if len(detections) >= self.cfg.alert_threshold:
            self._draw_alert(frame, len(detections))

        # 5. Watermark
        put_text(frame,
                 "ROAD-AI v1.0",
                 self.fw - 120, self.fh - 10,
                 C.DIM, scale=0.38)

        return frame

    # ── Boxes ─────────────────────────────────────────────────────────────────
    def _draw_boxes(self, frame, detections):
        for d in detections:
            x1, y1, x2, y2 = d["bbox"]
            conf  = d.get("confidence", 0.5)
            cls   = d.get("class_name", d.get("class", "Damage"))
            color = self.cfg.sev_color(conf)
            sev   = self.cfg.severity(conf)

            # Subtle fill
            draw_semi_rect(frame, x1, y1, x2, y2, color, alpha=0.08)

            # Corner bracket box
            draw_corner_box(frame, x1, y1, x2, y2, color, thickness=2)

            # Label
            label = f"{cls}  {conf:.0%}"
            draw_label_pill(frame, label, x1, y1, color)

            # Severity dot at bottom-right corner
            dot_x, dot_y = x2 - 6, y2 - 6
            cv2.circle(frame, (dot_x, dot_y), 4, color, -1, cv2.LINE_AA)

    # ── Stats panel ───────────────────────────────────────────────────────────
    def _draw_stats_panel(self, frame, frame_id, fps_live, n_det):
        m  = self.MARGIN
        pw = self.PANEL_W
        ph = self.PANEL_H

        # Panel background
        draw_semi_rect(frame, m, m, m + pw, m + ph, C.BLACK, alpha=0.65)
        cv2.rectangle(frame, (m, m), (m + pw, m + ph), C.DIM, 1)

        # Title bar
        cv2.rectangle(frame, (m, m), (m + pw, m + 22), C.DIM, -1)
        put_text(frame, "ROAD DAMAGE MONITOR",
                 m + 6, m + 15, C.WHITE, scale=0.42, thickness=1)

        # Live indicator dot
        dot_color = C.POOR if n_det >= self.cfg.alert_threshold else C.GOOD
        cv2.circle(frame, (m + pw - 10, m + 12), 4, dot_color, -1, cv2.LINE_AA)

        rows = [
            ("FRAME",      f"{frame_id:>6}"),
            ("FPS",        f"{fps_live:>5.1f}"),
            ("THIS FRAME", f"{n_det:>6}"),
            ("TOTAL DET.", f"{self.total_detections:>6}"),
        ]

        y = m + 36
        for key, val in rows:
            put_text(frame, key, m + 8,  y, C.DIM,   scale=0.38)
            put_text(frame, val, m + 130, y, C.WHITE, scale=0.42, thickness=1)
            y += 18

        # Top damage class
        if self.class_counts:
            top_cls = max(self.class_counts, key=self.class_counts.get)
            short   = top_cls[:18]
            put_text(frame, "TOP CLASS", m + 8,  y, C.DIM,       scale=0.38)
            put_text(frame, short,        m + 8,  y + 14, C.MODERATE, scale=0.36)

    # ── Density bar ───────────────────────────────────────────────────────────
    def _draw_density_bar(self, frame):
        if not self.history:
            return

        bw      = self.fw
        bh      = 28
        by      = self.fh - bh
        max_det = max(max(self.history), 1)
        n       = len(self.history)

        draw_semi_rect(frame, 0, by, bw, self.fh, C.BLACK, alpha=0.55)
        put_text(frame, "DAMAGE DENSITY", 6, by + 11, C.DIM, scale=0.35)

        seg_w = (bw - 120) / self.cfg.history_len
        ox    = 115

        for i, cnt in enumerate(self.history):
            ratio = cnt / max_det
            color = C.POOR if ratio > 0.7 else (C.MODERATE if ratio > 0.35 else C.GOOD)
            bar_h = int(ratio * (bh - 6))
            rx    = int(ox + i * seg_w)
            ry    = self.fh - 3 - bar_h
            rw    = max(int(seg_w) - 1, 1)
            cv2.rectangle(frame, (rx, ry), (rx + rw, self.fh - 3), color, -1)

        # Current count label
        put_text(frame,
                 f"{list(self.history)[-1]} det",
                 bw - 55, by + 17,
                 C.WHITE, scale=0.4)

    # ── Alert banner ──────────────────────────────────────────────────────────
    def _draw_alert(self, frame, n_det):
        bw = self.fw
        draw_semi_rect(frame, 0, 52, bw, 78, C.POOR, alpha=0.7)
        cv2.rectangle(frame, (0, 52), (bw, 78), C.POOR, 1)

        msg = f"  ⚠  HIGH DAMAGE DENSITY — {n_det} DETECTIONS THIS FRAME"
        put_text(frame, msg, 6, 70, C.WHITE, scale=0.52, thickness=1)


# ── Progress bar ──────────────────────────────────────────────────────────────

class ProgressBar:
    def __init__(self, total: int, width: int = 42):
        self.total = max(total, 1)
        self.width = width
        self.start = time.time()

    def update(self, n: int):
        pct     = n / self.total
        filled  = int(self.width * pct)
        bar     = "█" * filled + "░" * (self.width - filled)
        elapsed = time.time() - self.start
        eta     = (elapsed / max(n, 1)) * (self.total - n)
        print(f"\r  [{bar}] {pct:>6.1%}  frame {n}/{self.total}  ETA {eta:.1f}s",
              end="", flush=True)

    def done(self):
        print()


# ── Main pipeline ─────────────────────────────────────────────────────────────

def process_video(cfg: RenderConfig) -> dict:
    cfg.validate()

    log.info(f"Model  : {cfg.model_path}")
    log.info(f"Input  : {cfg.video_path}")

    analyzer = VideoAnalyzer(cfg.model_path)

    cap = cv2.VideoCapture(cfg.video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {cfg.video_path}")

    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps          = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    log.info(f"Video  : {width}×{height} @ {fps:.1f} fps  ({total_frames} frames)")

    Path(cfg.output_path).parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(cfg.output_path, fourcc, fps, (width, height))

    hud      = HUDRenderer(width, height, cfg)
    progress = ProgressBar(total_frames)

    frame_id     = 0
    written      = 0
    total_det    = 0
    tick         = time.time()
    fps_live     = 0.0
    fps_window   = deque(maxlen=30)

    log.info("Processing…")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # FPS measurement
            now = time.time()
            fps_window.append(now - tick)
            tick = now
            if len(fps_window) > 1:
                fps_live = len(fps_window) / sum(fps_window)

            # Skip frames
            if frame_id % cfg.skip_frames != 0:
                writer.write(frame)
                frame_id += 1
                written  += 1
                progress.update(frame_id)
                continue

            # Detect
            try:
                detections = analyzer.detector.detect(frame)
            except Exception as e:
                log.warning(f"Frame {frame_id} detection failed: {e}")
                detections = []

            # Attach class names
            for d in detections:
                d["class_name"] = cfg.class_names.get(d.get("class", -1), "Damage")

            total_det += len(detections)

            # Render HUD
            annotated = hud.render(frame.copy(), detections, frame_id, fps_live)
            writer.write(annotated)
            written  += 1
            frame_id += 1
            progress.update(frame_id)

    except KeyboardInterrupt:
        log.warning("Interrupted — partial video saved.")
    finally:
        cap.release()
        writer.release()
        progress.done()

    # ── Summary ───────────────────────────────────────────────────────────────
    stats = {
        "frames_total":     frame_id,
        "frames_written":   written,
        "total_detections": total_det,
        "avg_det_per_frame": round(total_det / max(frame_id, 1), 2),
        "class_breakdown":  dict(hud.class_counts),
        "output":           str(Path(cfg.output_path).resolve()),
    }

    _print_summary(stats)
    return stats


def _print_summary(s: dict):
    sep = "─" * 46
    print(f"\n{sep}")
    print(f"  ✅  Video processing complete")
    print(sep)
    print(f"  Frames total   : {s['frames_total']}")
    print(f"  Frames written : {s['frames_written']}")
    print(f"  Total detections     : {s['total_detections']}")
    print(f"  Avg detections/frame : {s['avg_det_per_frame']}")

    if s["class_breakdown"]:
        print(f"\n  Class breakdown:")
        for cls, cnt in sorted(s["class_breakdown"].items(), key=lambda x: -x[1]):
            bar = "█" * min(cnt, 30)
            print(f"    {cls:<22} {bar} {cnt}")

    print(f"\n  Output → {s['output']}")
    print(f"{sep}\n")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cfg = RenderConfig(
        video_path=r"C:/Users/DELL/Downloads/road_damage_ai/3657637695-preview.mp4",
        output_path="output/road_damage_output.mp4",
        model_path="best.pt",
        conf_threshold=0.35,
        skip_frames=1,
        alert_threshold=5,
        history_len=60,
    )

    process_video(cfg)