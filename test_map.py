"""
run_mapping.py — Road Health Map Runner

Production-grade entry point for the mapping pipeline. Connects scoring
outputs → location enrichment → map generation → export, with:

  - Typed LocationBuilder that enriches raw score dicts with metadata
  - Multi-source input: inline list, JSON file, CSV file, or scorer results
  - Validation with per-field error reporting
  - Batch mode: generate one map per district/zone grouping
  - GeoJSON + CSV export alongside the HTML map
  - Rich terminal summary with ASCII score distribution
"""

from __future__ import annotations

import csv
import json
import logging
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

from mapping import create_map, Location

log = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)


# ── Pipeline config ───────────────────────────────────────────────────────────

@dataclass
class MapPipelineConfig:
    """All knobs for one map-generation run."""

    # Required: list of raw location dicts (lat, lon, score, + optional fields)
    locations:      list[dict] = field(default_factory=list)

    # Alternatively, load from file (JSON array or CSV with headers)
    input_file:     Optional[str] = None

    # Output
    output_dir:     str  = "output/maps"
    map_filename:   str  = "road_health_map.html"
    title:          str  = "Road Health Monitor"

    # Map options (passed through to create_map)
    cluster:        bool = True
    heatmap:        bool = True
    tile_style:     str  = "CartoDB positron"
    zoom:           int  = 14

    # Export extras
    export_geojson: bool = True
    export_csv:     bool = True

    # Batch: if locations have a "zone" key, generate one map per zone
    batch_by_zone:  bool = False

    # Filtering
    min_score:      Optional[float] = None   # drop locations below this
    max_score:      Optional[float] = None
    only_priority:  Optional[str]   = None   # "Low" | "Medium" | "High" | "Critical"

    def validate(self) -> None:
        if not self.locations and not self.input_file:
            raise ValueError("Provide either `locations` list or `input_file` path.")
        if self.input_file and not Path(self.input_file).exists():
            raise FileNotFoundError(f"Input file not found: {self.input_file}")
        if self.min_score is not None and not (0 <= self.min_score <= 100):
            raise ValueError("min_score must be 0–100.")
        if self.max_score is not None and not (0 <= self.max_score <= 100):
            raise ValueError("max_score must be 0–100.")


# ── Location builder ──────────────────────────────────────────────────────────

PRIORITY_BANDS = [
    (80, "Low",      "#2ecc71"),
    (60, "Medium",   "#f1c40f"),
    (40, "High",     "#e67e22"),
    (0,  "Critical", "#e74c3c"),
]

REPAIR_MESSAGES = {
    "Low":      "Routine monitoring — re-inspect in 6 months.",
    "Medium":   "Preventive repair — patch within 30 days.",
    "High":     "Urgent repair — address within 1 week.",
    "Critical": "Immediate risk — repair within 24 hours.",
}


def get_priority(score: float) -> str:
    for threshold, label, _ in PRIORITY_BANDS:
        if score >= threshold:
            return label
    return "Critical"


@dataclass
class EnrichedLocation:
    """A raw dict enriched with derived fields before map rendering."""
    lat:             float
    lon:             float
    score:           float
    label:           str   = ""
    address:         str   = ""
    zone:            str   = "default"
    frame_index:     Optional[int]   = None
    n_detections:    int   = 0
    dominant_class:  str   = ""
    recorded_at:     str   = field(default_factory=lambda: datetime.now().isoformat())

    # Derived
    priority:        str   = field(init=False)
    repair_message:  str   = field(init=False)
    health_band:     str   = field(init=False)

    def __post_init__(self):
        self.score       = float(max(0, min(100, self.score)))
        self.priority    = get_priority(self.score)
        self.repair_message = REPAIR_MESSAGES[self.priority]
        if self.score >= 80:   self.health_band = "good"
        elif self.score >= 60: self.health_band = "moderate"
        elif self.score >= 40: self.health_band = "poor"
        else:                  self.health_band = "critical"

    @classmethod
    def from_dict(cls, d: dict) -> "EnrichedLocation":
        return cls(
            lat=float(d["lat"]),
            lon=float(d["lon"]),
            score=float(d["score"]),
            label=d.get("label", d.get("name", "")),
            address=d.get("address", ""),
            zone=d.get("zone", "default"),
            frame_index=d.get("frame_index"),
            n_detections=int(d.get("n_detections", 0)),
            dominant_class=d.get("dominant_class", ""),
            recorded_at=d.get("recorded_at", datetime.now().isoformat()),
        )

    def to_map_dict(self) -> dict:
        """Format expected by create_map() — includes all metadata fields."""
        return {
            "lat":            self.lat,
            "lon":            self.lon,
            "score":          self.score,
            "label":          self.label or f"Segment ({self.lat:.4f}, {self.lon:.4f})",
            "address":        self.address,
            "priority":       self.priority,
            "repair":         self.repair_message,
            "zone":           self.zone,
            "detections":     self.n_detections,
            "dominant_class": self.dominant_class,
            "recorded_at":    self.recorded_at,
        }

    def to_geojson_feature(self) -> dict:
        return {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [self.lon, self.lat],
            },
            "properties": asdict(self),
        }

    def to_csv_row(self) -> dict:
        return {
            "lat": self.lat, "lon": self.lon,
            "score": self.score, "band": self.health_band,
            "priority": self.priority, "label": self.label,
            "address": self.address, "zone": self.zone,
            "n_detections": self.n_detections,
            "dominant_class": self.dominant_class,
            "recorded_at": self.recorded_at,
        }


# ── Loaders ───────────────────────────────────────────────────────────────────

def _load_from_file(path: str) -> list[dict]:
    p = Path(path)
    if p.suffix.lower() == ".json":
        data = json.loads(p.read_text())
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and "locations" in data:
            return data["locations"]
        raise ValueError("JSON must be a list of location dicts or {locations: [...]}")
    if p.suffix.lower() == ".csv":
        with open(p, newline="") as f:
            return list(csv.DictReader(f))
    raise ValueError(f"Unsupported file type: {p.suffix}")


def _validate_raw(raw: list[dict]) -> list[EnrichedLocation]:
    """Parse, validate, and enrich all raw location dicts."""
    valid, errors = [], []
    required = {"lat", "lon", "score"}

    for i, d in enumerate(raw):
        missing = required - set(d.keys())
        if missing:
            errors.append(f"  Row {i}: missing fields {missing}")
            continue
        try:
            loc = EnrichedLocation.from_dict(d)
        except (ValueError, TypeError) as e:
            errors.append(f"  Row {i}: {e}")
            continue
        if not (-90 <= loc.lat <= 90):
            errors.append(f"  Row {i}: lat {loc.lat} out of range")
            continue
        if not (-180 <= loc.lon <= 180):
            errors.append(f"  Row {i}: lon {loc.lon} out of range")
            continue
        valid.append(loc)

    if errors:
        log.warning(f"{len(errors)} row(s) skipped due to validation errors:")
        for e in errors:
            log.warning(e)

    return valid


def _filter_locations(
    locs: list[EnrichedLocation],
    cfg: MapPipelineConfig,
) -> list[EnrichedLocation]:
    out = locs
    if cfg.min_score is not None:
        out = [l for l in out if l.score >= cfg.min_score]
    if cfg.max_score is not None:
        out = [l for l in out if l.score <= cfg.max_score]
    if cfg.only_priority:
        out = [l for l in out if l.priority == cfg.only_priority]
    removed = len(locs) - len(out)
    if removed:
        log.info(f"Filtered out {removed} location(s) by config rules.")
    return out


# ── Exports ───────────────────────────────────────────────────────────────────

def _export_geojson(locs: list[EnrichedLocation], out_dir: Path, stem: str) -> Path:
    path = out_dir / f"{stem}.geojson"
    fc = {
        "type": "FeatureCollection",
        "generated_at": datetime.now().isoformat(),
        "features": [l.to_geojson_feature() for l in locs],
    }
    path.write_text(json.dumps(fc, indent=2))
    log.info(f"GeoJSON → {path.resolve()}")
    return path


def _export_csv(locs: list[EnrichedLocation], out_dir: Path, stem: str) -> Path:
    path = out_dir / f"{stem}.csv"
    if not locs:
        return path
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=locs[0].to_csv_row().keys())
        writer.writeheader()
        writer.writerows(l.to_csv_row() for l in locs)
    log.info(f"CSV     → {path.resolve()}")
    return path


# ── Terminal summary ──────────────────────────────────────────────────────────

def _print_summary(locs: list[EnrichedLocation], outputs: list[str]) -> None:
    if not locs:
        return

    scores = [l.score for l in locs]
    avg    = sum(scores) / len(scores)
    bands  = {"good": 0, "moderate": 0, "poor": 0, "critical": 0}
    for l in locs:
        bands[l.health_band] += 1

    # ASCII score histogram (10 buckets)
    buckets = [0] * 10
    for s in scores:
        buckets[min(int(s // 10), 9)] += 1
    max_b = max(buckets) or 1

    sep = "─" * 52
    print(f"\n{sep}")
    print(f"  ROAD HEALTH MAP — PIPELINE SUMMARY")
    print(sep)
    print(f"  Locations   : {len(locs)}")
    print(f"  Avg score   : {avg:.1f} / 100")
    print(f"  Range       : {min(scores):.1f} – {max(scores):.1f}")
    print(f"\n  Condition breakdown:")
    icons = {"good": "✅", "moderate": "⚠️ ", "poor": "🔴", "critical": "🚨"}
    for band, count in bands.items():
        bar = "█" * min(count, 30)
        pct = count / len(locs) * 100
        print(f"    {icons[band]}  {band:<10} {bar:<30} {count}  ({pct:.0f}%)")

    print(f"\n  Score distribution (0–100):")
    for i, cnt in enumerate(buckets):
        label = f"  {i*10:>3}–{i*10+9:<3}"
        bar   = "▓" * int(cnt / max_b * 28)
        print(f"    {label} {bar} {cnt}")

    if outputs:
        print(f"\n  Output files:")
        for o in outputs:
            print(f"    → {o}")

    print(sep + "\n")


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_map_pipeline(cfg: MapPipelineConfig) -> list[str]:
    """
    Execute the full mapping pipeline.

    Returns
    -------
    list of output file paths generated.
    """
    cfg.validate()

    # 1. Load
    raw = cfg.locations or []
    if cfg.input_file:
        extra = _load_from_file(cfg.input_file)
        raw = raw + extra
        log.info(f"Loaded {len(extra)} location(s) from {cfg.input_file}")

    # 2. Validate & enrich
    locs = _validate_raw(raw)
    if not locs:
        log.error("No valid locations after validation — aborting.")
        return []

    # 3. Filter
    locs = _filter_locations(locs, cfg)
    if not locs:
        log.warning("All locations filtered out — nothing to map.")
        return []

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    outputs: list[str] = []

    # 4. Batch or single map
    groups = {}
    if cfg.batch_by_zone:
        for l in locs:
            groups.setdefault(l.zone, []).append(l)
        log.info(f"Batch mode: {len(groups)} zone(s) — {list(groups.keys())}")
    else:
        groups["default"] = locs

    for zone, zone_locs in groups.items():
        stem     = Path(cfg.map_filename).stem
        zone_tag = f"_{zone}" if cfg.batch_by_zone and zone != "default" else ""
        html_name = f"{stem}{zone_tag}.html"
        map_title = f"{cfg.title}" + (f" — {zone.title()}" if zone_tag else "")

        log.info(f"Generating map '{html_name}' ({len(zone_locs)} locations)…")

        map_dicts = [l.to_map_dict() for l in zone_locs]

        create_map(
            locations=map_dicts,
            output_file=str(out_dir / html_name),
            title=map_title,
            cluster=cfg.cluster,
            heatmap=cfg.heatmap,
            tile_style=cfg.tile_style,
            zoom=cfg.zoom,
            save_json=False,   # we handle exports ourselves below
        )

        outputs.append(str((out_dir / html_name).resolve()))

        if cfg.export_geojson:
            p = _export_geojson(zone_locs, out_dir, stem + zone_tag)
            outputs.append(str(p))

        if cfg.export_csv:
            p = _export_csv(zone_locs, out_dir, stem + zone_tag)
            outputs.append(str(p))

    _print_summary(locs, outputs)
    return outputs


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # ── Inline locations (can also point to a JSON / CSV file) ────────────────
    locations = [
        {
            "lat": 13.0827, "lon": 80.2707, "score": 90,
            "label": "Anna Salai",   "address": "Chennai, TN",
            "zone": "central",       "n_detections": 2,
            "dominant_class": "Longitudinal Crack",
        },
        {
            "lat": 13.0850, "lon": 80.2750, "score": 62,
            "label": "T. Nagar",    "address": "Chennai, TN",
            "zone": "central",      "n_detections": 8,
            "dominant_class": "Transverse Crack",
        },
        {
            "lat": 13.0800, "lon": 80.2600, "score": 30,
            "label": "Saidapet",    "address": "Chennai, TN",
            "zone": "south",        "n_detections": 21,
            "dominant_class": "Pothole",
        },
        {
            "lat": 13.0900, "lon": 80.2800, "score": 78,
            "label": "Nungambakkam","address": "Chennai, TN",
            "zone": "central",      "n_detections": 5,
            "dominant_class": "Alligator Crack",
        },
        {
            "lat": 13.0760, "lon": 80.2620, "score": 18,
            "label": "Vadapalani",  "address": "Chennai, TN",
            "zone": "south",        "n_detections": 34,
            "dominant_class": "Pothole",
        },
        {
            "lat": 13.0950, "lon": 80.2550, "score": 85,
            "label": "Koyambedu",   "address": "Chennai, TN",
            "zone": "west",         "n_detections": 3,
            "dominant_class": "Longitudinal Crack",
        },
    ]

    cfg = MapPipelineConfig(
        locations=locations,
        # input_file="locations.json",   # ← or load from file instead
        output_dir="output/maps",
        map_filename="road_health_map.html",
        title="Chennai Road Health Monitor",
        cluster=True,
        heatmap=True,
        zoom=14,
        export_geojson=True,
        export_csv=True,
        batch_by_zone=False,           # set True to get one map per zone
        # min_score=20,                # optional: filter low-quality entries
        # only_priority="High",        # optional: only show urgent segments
    )

    run_map_pipeline(cfg)