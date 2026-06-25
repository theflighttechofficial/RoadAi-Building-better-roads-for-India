# mapping.py

import json
import logging
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

import folium
from folium.plugins import MarkerCluster, HeatMap, MiniMap, Fullscreen

log = logging.getLogger(__name__)


# ─── Data Model ───────────────────────────────────────────────────────────────

@dataclass
class Location:
    lat:      float
    lon:      float
    score:    int                        # 0–100 road health score
    label:    str        = ""            # optional place name
    address:  str        = ""
    metadata: dict       = field(default_factory=dict)  # any extra payload

    # ── Derived ───────────────────────────────────────────────────────────────

    @property
    def health_band(self) -> str:
        if self.score >= 80:   return "good"
        elif self.score >= 50: return "moderate"
        else:                  return "poor"

    @property
    def color(self) -> str:
        return {"good": "#2ecc71", "moderate": "#f39c12", "poor": "#e74c3c"}[self.health_band]

    @property
    def icon(self) -> str:
        return {"good": "✅", "moderate": "⚠️", "poor": "🔴"}[self.health_band]

    @property
    def radius(self) -> int:
        """Larger marker for worse roads — draws attention to problem areas."""
        return {"good": 7, "moderate": 10, "poor": 14}[self.health_band]

    # ── Constructors ──────────────────────────────────────────────────────────

    @classmethod
    def from_dict(cls, d: dict) -> "Location":
        return cls(
            lat=float(d["lat"]),
            lon=float(d["lon"]),
            score=int(d["score"]),
            label=d.get("label", ""),
            address=d.get("address", ""),
            metadata={k: v for k, v in d.items()
                      if k not in {"lat", "lon", "score", "label", "address"}},
        )

    @classmethod
    def from_json_file(cls, path: str) -> list["Location"]:
        data = json.loads(Path(path).read_text())
        return [cls.from_dict(d) for d in data]

    def to_dict(self) -> dict:
        return {
            "lat": self.lat, "lon": self.lon, "score": self.score,
            "label": self.label, "address": self.address,
            "health_band": self.health_band, **self.metadata,
        }


# ─── Popup Builder ────────────────────────────────────────────────────────────

def _build_popup(loc: Location) -> folium.Popup:
    """Render a styled HTML card inside the marker popup."""
    band_colors = {"good": "#2ecc71", "moderate": "#f39c12", "poor": "#e74c3c"}
    band_color  = band_colors[loc.health_band]

    extra_rows = "".join(
        f"<tr><td style='color:#888'>{k}</td><td><b>{v}</b></td></tr>"
        for k, v in loc.metadata.items()
    )

    html = f"""
    <div style="font-family:Arial,sans-serif;min-width:200px;padding:4px">
      <div style="background:{band_color};color:#fff;padding:8px 12px;border-radius:6px 6px 0 0">
        <b style="font-size:15px">{loc.icon} {loc.label or 'Road Segment'}</b>
      </div>
      <div style="border:1px solid #ddd;border-top:none;padding:10px;border-radius:0 0 6px 6px">
        <table style="width:100%;border-collapse:collapse;font-size:13px">
          <tr>
            <td style="color:#888">Health Score</td>
            <td><b style="font-size:18px;color:{band_color}">{loc.score}/100</b></td>
          </tr>
          <tr>
            <td style="color:#888">Condition</td>
            <td><b>{loc.health_band.title()}</b></td>
          </tr>
          {'<tr><td style="color:#888">Address</td><td>' + loc.address + '</td></tr>' if loc.address else ''}
          <tr><td style="color:#888">Coords</td>
            <td style="font-size:11px">{loc.lat:.5f}, {loc.lon:.5f}</td>
          </tr>
          {extra_rows}
        </table>
      </div>
    </div>
    """
    return folium.Popup(folium.IFrame(html, width=240, height=200), max_width=260)


# ─── Legend ───────────────────────────────────────────────────────────────────

def _add_legend(m: folium.Map, summary: dict) -> None:
    total  = summary["total"]
    legend = f"""
    <div style="
        position:fixed;bottom:30px;right:10px;z-index:9999;
        background:#fff;padding:14px 18px;border-radius:10px;
        box-shadow:0 2px 12px rgba(0,0,0,0.2);font-family:Arial,sans-serif;
        min-width:190px">
      <b style="font-size:14px">🗺 Road Health Legend</b>
      <hr style="margin:8px 0">
      <div style="margin:4px 0">
        <span style="color:#2ecc71;font-size:18px">●</span>
        <b>Good</b> (≥80) &nbsp;&nbsp;
        <span style="float:right;color:#555">{summary['good']} ({summary['good']/max(total,1):.0%})</span>
      </div>
      <div style="margin:4px 0">
        <span style="color:#f39c12;font-size:18px">●</span>
        <b>Moderate</b> (50–79)
        <span style="float:right;color:#555">{summary['moderate']} ({summary['moderate']/max(total,1):.0%})</span>
      </div>
      <div style="margin:4px 0">
        <span style="color:#e74c3c;font-size:18px">●</span>
        <b>Poor</b> (&lt;50) &nbsp;&nbsp;
        <span style="float:right;color:#555">{summary['poor']} ({summary['poor']/max(total,1):.0%})</span>
      </div>
      <hr style="margin:8px 0">
      <div style="font-size:12px;color:#888">
        Total segments: <b>{total}</b><br>
        Avg score: <b>{summary['avg_score']:.1f}</b><br>
        Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
      </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend))


# ─── Core Builder ─────────────────────────────────────────────────────────────

def build_summary(locations: list[Location]) -> dict:
    scores = [loc.score for loc in locations]
    return {
        "total":    len(locations),
        "good":     sum(1 for loc in locations if loc.health_band == "good"),
        "moderate": sum(1 for loc in locations if loc.health_band == "moderate"),
        "poor":     sum(1 for loc in locations if loc.health_band == "poor"),
        "avg_score":   sum(scores) / max(len(scores), 1),
        "min_score":   min(scores),
        "max_score":   max(scores),
    }


def create_map(
    locations:    list[dict | Location],
    output_file:  str  = "road_health_map.html",
    title:        str  = "Road Health Map",
    cluster:      bool = True,
    heatmap:      bool = True,
    tile_style:   str  = "CartoDB positron",   # clean light basemap
    zoom:         int  = 14,
    save_json:    bool = False,
) -> Optional[folium.Map]:
    """
    Build and save an interactive road health map.

    Args:
        locations  : list of dicts with at least {lat, lon, score},
                     or pre-built Location objects.
        output_file: HTML output path.
        title      : Page title shown in browser tab.
        cluster    : Group nearby markers into clusters.
        heatmap    : Add a semi-transparent heatmap layer.
        tile_style : Folium tile provider name.
        zoom       : Initial zoom level.
        save_json  : Also dump a GeoJSON summary alongside the HTML.

    Returns:
        The folium.Map object (useful for further customization or testing).
    """

    # ── Normalise input ───────────────────────────────────────────────────────
    locs: list[Location] = [
        loc if isinstance(loc, Location) else Location.from_dict(loc)
        for loc in locations
    ]

    if not locs:
        log.warning("No locations provided — map not created.")
        return None

    # ── Validate ──────────────────────────────────────────────────────────────
    invalid = [i for i, l in enumerate(locs) if not (-90 <= l.lat <= 90 and -180 <= l.lon <= 180)]
    if invalid:
        raise ValueError(f"Invalid coordinates at indices: {invalid}")

    summary = build_summary(locs)
    log.info(f"Mapping {summary['total']} segments | "
             f"good={summary['good']} moderate={summary['moderate']} poor={summary['poor']} | "
             f"avg={summary['avg_score']:.1f}")

    # ── Center: mean of all points ────────────────────────────────────────────
    center_lat = sum(l.lat for l in locs) / len(locs)
    center_lon = sum(l.lon for l in locs) / len(locs)

    # ── Base map ──────────────────────────────────────────────────────────────
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=zoom,
        tiles=tile_style,
        prefer_canvas=True,
    )

    # ── Plugins ───────────────────────────────────────────────────────────────
    Fullscreen(position="topleft").add_to(m)
    MiniMap(toggle_display=True, position="bottomleft").add_to(m)

    # ── Heatmap layer ─────────────────────────────────────────────────────────
    if heatmap:
        heat_data = [
            [loc.lat, loc.lon, (100 - loc.score) / 100]   # invert: poor = hot
            for loc in locs
        ]
        HeatMap(
            heat_data,
            name="Damage Heatmap",
            min_opacity=0.3,
            radius=20,
            blur=15,
            gradient={0.4: "#2ecc71", 0.65: "#f39c12", 1.0: "#e74c3c"},
        ).add_to(m)

    # ── Marker layer ──────────────────────────────────────────────────────────
    marker_layer = MarkerCluster(name="Road Segments") if cluster else folium.FeatureGroup(name="Road Segments")

    for loc in locs:
        folium.CircleMarker(
            location=[loc.lat, loc.lon],
            radius=loc.radius,
            color=loc.color,
            weight=2,
            fill=True,
            fill_color=loc.color,
            fill_opacity=0.75,
            popup=_build_popup(loc),
            tooltip=folium.Tooltip(
                f"{'📍 ' + loc.label + ' — ' if loc.label else ''}Score: {loc.score}/100",
                sticky=False,
            ),
        ).add_to(marker_layer)

    marker_layer.add_to(m)

    # ── Layer control + Legend ────────────────────────────────────────────────
    folium.LayerControl(collapsed=False).add_to(m)
    _add_legend(m, summary)

    # ── Save HTML ─────────────────────────────────────────────────────────────
    out = Path(output_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    m.get_root().header.add_child(folium.Element(f"<title>{title}</title>"))
    m.save(str(out))
    log.info(f"Map saved → {out.resolve()}")

    # ── Optional GeoJSON export ───────────────────────────────────────────────
    if save_json:
        geojson_path = out.with_suffix(".geojson")
        geojson = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [loc.lon, loc.lat]},
                    "properties": loc.to_dict(),
                }
                for loc in locs
            ],
        }
        geojson_path.write_text(json.dumps(geojson, indent=2))
        log.info(f"GeoJSON saved → {geojson_path.resolve()}")

    print_summary(summary, output_file)
    return m


# ─── CLI Summary ──────────────────────────────────────────────────────────────

def print_summary(summary: dict, output_file: str) -> None:
    sep = "─" * 42
    print(f"\n{sep}")
    print(f"  Road Health Map — {Path(output_file).name}")
    print(sep)
    print(f"  Total segments : {summary['total']}")
    print(f"  Avg score      : {summary['avg_score']:.1f}/100")
    print(f"  Score range    : {summary['min_score']} – {summary['max_score']}")
    print(f"\n  ✅  Good     : {summary['good']}")
    print(f"  ⚠️   Moderate : {summary['moderate']}")
    print(f"  🔴  Poor     : {summary['poor']}")
    print(f"\n  Saved → {Path(output_file).resolve()}")
    print(f"{sep}\n")


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    sample_locations = [
        {"lat": 13.0827, "lon": 80.2707, "score": 90, "label": "Anna Salai",    "address": "Chennai, TN"},
        {"lat": 13.0850, "lon": 80.2750, "score": 62, "label": "T. Nagar",      "address": "Chennai, TN"},
        {"lat": 13.0780, "lon": 80.2680, "score": 35, "label": "Saidapet",      "address": "Chennai, TN"},
        {"lat": 13.0900, "lon": 80.2800, "score": 78, "label": "Nungambakkam",  "address": "Chennai, TN"},
        {"lat": 13.0760, "lon": 80.2600, "score": 20, "label": "Vadapalani",    "address": "Chennai, TN"},
        {"lat": 13.0950, "lon": 80.2550, "score": 85, "label": "Koyambedu",     "address": "Chennai, TN"},
    ]

    create_map(
        locations=sample_locations,
        output_file="output/road_health_map.html",
        title="Chennai Road Health Monitor",
        cluster=True,
        heatmap=True,
        save_json=True,
    )       