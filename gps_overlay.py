"""
gps_overlay.py — GPS Track Synchroniser

Loads a GPX file and maps each video frame's timestamp to a GPS coordinate.
Also supports live NMEA serial GPS for real-time dashcam use.

Dependencies: gpxpy, pyserial (optional for live GPS)
    pip install gpxpy
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)


@dataclass
class GPSPoint:
    lat:       float
    lon:       float
    elevation: float       = 0.0
    speed_kmh: float       = 0.0
    bearing:   float       = 0.0        # degrees 0–360
    timestamp: Optional[datetime] = None

    def to_dict(self) -> dict:
        return {
            "lat":       self.lat,
            "lon":       self.lon,
            "elevation": round(self.elevation, 1),
            "speed_kmh": round(self.speed_kmh, 1),
            "bearing":   round(self.bearing, 1),
        }


class GPXTrackLoader:
    """
    Loads a .gpx file and interpolates GPS coordinates for any video timestamp.

    Usage
    -----
    gps = GPXTrackLoader("track.gpx", video_start_utc="2024-02-26T10:30:00Z")
    point = gps.at_second(12.5)   # returns GPSPoint at 12.5s into video
    """

    def __init__(
        self,
        gpx_path:         str,
        video_start_utc:  Optional[str] = None,   # ISO-8601 string
        offset_seconds:   float         = 0.0,    # manual sync correction
    ):
        try:
            import gpxpy
        except ImportError:
            raise ImportError("Install gpxpy:  pip install gpxpy")

        self._points: list[tuple[float, GPSPoint]] = []   # (elapsed_sec, point)
        self.offset  = offset_seconds
        self._load(gpx_path, video_start_utc, gpxpy)

    # ── Public ──────────────────────────────────────────────────────────────

    def at_second(self, video_sec: float) -> Optional[GPSPoint]:
        """Return interpolated GPSPoint at `video_sec` seconds into the video."""
        t = video_sec + self.offset
        if not self._points:
            return None
        if t <= self._points[0][0]:
            return self._points[0][1]
        if t >= self._points[-1][0]:
            return self._points[-1][1]

        # Binary search for surrounding points
        lo, hi = 0, len(self._points) - 1
        while lo < hi - 1:
            mid = (lo + hi) // 2
            if self._points[mid][0] <= t:
                lo = mid
            else:
                hi = mid

        t0, p0 = self._points[lo]
        t1, p1 = self._points[hi]
        frac = (t - t0) / max(t1 - t0, 1e-9)

        return GPSPoint(
            lat=p0.lat + frac * (p1.lat - p0.lat),
            lon=p0.lon + frac * (p1.lon - p0.lon),
            elevation=p0.elevation + frac * (p1.elevation - p0.elevation),
            speed_kmh=p0.speed_kmh,
            bearing=self._interp_bearing(p0.bearing, p1.bearing, frac),
            timestamp=p0.timestamp,
        )

    def at_frame(self, frame_index: int, fps: float) -> Optional[GPSPoint]:
        return self.at_second(frame_index / max(fps, 1e-6))

    @property
    def total_distance_km(self) -> float:
        if len(self._points) < 2:
            return 0.0
        total = 0.0
        for i in range(1, len(self._points)):
            total += self._haversine(
                self._points[i-1][1], self._points[i][1]
            )
        return total

    @property
    def duration_sec(self) -> float:
        if not self._points:
            return 0.0
        return self._points[-1][0] - self._points[0][0]

    # ── Internal ────────────────────────────────────────────────────────────

    def _load(self, gpx_path: str, video_start_utc, gpxpy) -> None:
        with open(gpx_path, "r") as f:
            gpx = gpxpy.parse(f)

        all_pts = []
        for track in gpx.tracks:
            for seg in track.segments:
                all_pts.extend(seg.points)

        if not all_pts:
            log.warning("GPX file contains no track points.")
            return

        # Determine time reference
        if video_start_utc:
            try:
                ref = datetime.fromisoformat(
                    video_start_utc.replace("Z", "+00:00")
                )
            except ValueError:
                ref = all_pts[0].time
                log.warning("Could not parse video_start_utc — using first GPX point.")
        else:
            ref = all_pts[0].time

        prev = None
        for pt in all_pts:
            if pt.time is None:
                continue
            elapsed = (pt.time - ref).total_seconds()
            speed   = 0.0
            bearing = 0.0
            if prev is not None:
                dt = (pt.time - prev.time).total_seconds()
                if dt > 0:
                    dist_km = self._haversine_raw(
                        prev.latitude, prev.longitude,
                        pt.latitude,  pt.longitude,
                    )
                    speed   = (dist_km / dt) * 3600.0
                    bearing = self._calc_bearing(
                        prev.latitude, prev.longitude,
                        pt.latitude,  pt.longitude,
                    )
            self._points.append((elapsed, GPSPoint(
                lat=pt.latitude,
                lon=pt.longitude,
                elevation=pt.elevation or 0.0,
                speed_kmh=speed,
                bearing=bearing,
                timestamp=pt.time,
            )))
            prev = pt

        log.info(f"GPX loaded: {len(self._points)} points  "
                 f"{self.total_distance_km:.2f} km  {self.duration_sec:.0f}s")

    @staticmethod
    def _haversine(p1: GPSPoint, p2: GPSPoint) -> float:
        return GPXTrackLoader._haversine_raw(p1.lat, p1.lon, p2.lat, p2.lon)

    @staticmethod
    def _haversine_raw(lat1, lon1, lat2, lon2) -> float:
        import math
        R  = 6371.0
        φ1, φ2 = math.radians(lat1), math.radians(lat2)
        Δφ = math.radians(lat2 - lat1)
        Δλ = math.radians(lon2 - lon1)
        a  = math.sin(Δφ/2)**2 + math.cos(φ1)*math.cos(φ2)*math.sin(Δλ/2)**2
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

    @staticmethod
    def _calc_bearing(lat1, lon1, lat2, lon2) -> float:
        import math
        φ1, φ2 = math.radians(lat1), math.radians(lat2)
        Δλ = math.radians(lon2 - lon1)
        x  = math.sin(Δλ) * math.cos(φ2)
        y  = math.cos(φ1)*math.sin(φ2) - math.sin(φ1)*math.cos(φ2)*math.cos(Δλ)
        return (math.degrees(math.atan2(x, y)) + 360) % 360

    @staticmethod
    def _interp_bearing(b0: float, b1: float, frac: float) -> float:
        """Shortest-path bearing interpolation (handles 359→1 wrap)."""
        diff = ((b1 - b0 + 540) % 360) - 180
        return (b0 + frac * diff) % 360


# ── Stub for when no GPX is available ────────────────────────────────────────

class NullGPSLoader:
    """Drop-in replacement when no GPS data is available."""

    def at_second(self, _: float) -> None:
        return None

    def at_frame(self, _: int, __: float) -> None:
        return None

    @property
    def total_distance_km(self) -> float:
        return 0.0
