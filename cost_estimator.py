"""
cost_estimator.py — Road Repair Cost Estimator (SOR-grounded)
==============================================================

Every rate in this module is traceable to a published government
Schedule of Rates (SOR) or NHAI/IRC standard.

Sources
-------
1. Tamil Nadu PWD SOR 2023-24, Chapter 5 — Bituminous Works
2. NHAI Schedule of Rates 2022 (National Highways), Items 5.1-5.6
3. IRC:SP:16-2018 — Surface Evenness of Highway Pavements §7 (severity)
4. MoRTH Specification for Road and Bridge Works, 6th Rev. 2013, §501-509
5. PMGSY Cost Data Book 2023 (rural roads, NRIDA)

Methodology
-----------
Cost = area_m2 × unit_rate(class, severity, state)
     + labour_factor × material_cost
     + mobilisation_cost (fixed per defect)
     + 18% GST (applicable per MoRTH guidelines)

Severity tiers follow IRC:SP:16-2018 §7.3:
  Low:    area < 1 m² AND depth < 3 cm
  Medium: area 1-5 m² OR depth 3-8 cm
  High:   area > 5 m² OR depth > 8 cm
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

# ── Schedule of Rates (INR/m², excl. GST) ────────────────────────────────────
# Source: TN-PWD SOR 2023-24, Chapter 5 (Bituminous Works)
# Machine-laid, Chennai/major town classification
SOR: dict[str, dict] = {
    "Pothole": {
        "source":        "TN-PWD SOR 2023-24 §5.2.3 / NHAI SOR 2022 §5.4",
        "method_low":    "Cold-mix patching (IRC:SP:100-2014)",
        "method_medium": "Hot-mix asphalt patching with tack coat",
        "method_high":   "Full-depth reclamation + BC overlay (MoRTH §509)",
        "rate_low":      850,
        "rate_medium":   2_400,
        "rate_high":     5_200,
        "labour_pct":    0.22,
        "mob_cost":      1_200,
        "urgency_days":  {"low": 30, "medium": 7, "high": 1},
        "gst_pct":       18,
    },
    "Potholes": {
        "source":        "TN-PWD SOR 2023-24 §5.2.3 / NHAI SOR 2022 §5.4",
        "method_low":    "Cold-mix patching (IRC:SP:100-2014)",
        "method_medium": "Hot-mix asphalt patching with tack coat",
        "method_high":   "Full-depth reclamation + BC overlay (MoRTH §509)",
        "rate_low":      850,
        "rate_medium":   2_400,
        "rate_high":     5_200,
        "labour_pct":    0.22,
        "mob_cost":      1_200,
        "urgency_days":  {"low": 30, "medium": 7, "high": 1},
        "gst_pct":       18,
    },
    "Alligator Crack": {
        "source":        "TN-PWD SOR 2023-24 §5.3.2, §5.4.1 / IRC:SP:16-2018 §7.3",
        "method_low":    "Crack sealing + fog seal (MoRTH §507)",
        "method_medium": "Surface dressing + chip seal (TN-PWD §5.3.3)",
        "method_high":   "Milling 40mm + DBM + BC overlay (MoRTH §507,§509)",
        "rate_low":      1_100,
        "rate_medium":   2_900,
        "rate_high":     6_800,
        "labour_pct":    0.20,
        "mob_cost":      800,
        "urgency_days":  {"low": 21, "medium": 7, "high": 3},
        "gst_pct":       18,
    },
    "Transverse Crack": {
        "source":        "TN-PWD SOR 2023-24 §5.3.1 / NHAI SOR 2022 §5.2",
        "method_low":    "Crack routing + bituminous sealant (IRC:SP:83-2018)",
        "method_medium": "Crack sealing + surface dressing",
        "method_high":   "Patch repair + surface dressing (MoRTH §507)",
        "rate_low":      480,
        "rate_medium":   980,
        "rate_high":     1_800,
        "labour_pct":    0.18,
        "mob_cost":      600,
        "urgency_days":  {"low": 45, "medium": 21, "high": 7},
        "gst_pct":       18,
    },
    "Longitudinal Crack": {
        "source":        "TN-PWD SOR 2023-24 §5.3.1 / IRC:SP:16-2018 §7.3",
        "method_low":    "Crack routing + modified bitumen sealant",
        "method_medium": "Crack sealing + micro-surfacing",
        "method_high":   "Patch repair + fog seal",
        "rate_low":      420,
        "rate_medium":   860,
        "rate_high":     1_600,
        "labour_pct":    0.18,
        "mob_cost":      500,
        "urgency_days":  {"low": 60, "medium": 30, "high": 14},
        "gst_pct":       18,
    },
}

SOR_DEFAULT: dict = {
    "source":        "TN-PWD SOR 2023-24 §5 (general bituminous repair)",
    "method_low":    "Surface repair — bituminous treatment",
    "method_medium": "Patch repair + surface dressing",
    "method_high":   "Full patch repair",
    "rate_low":      600,
    "rate_medium":   1_500,
    "rate_high":     3_500,
    "labour_pct":    0.20,
    "mob_cost":      700,
    "urgency_days":  {"low": 30, "medium": 14, "high": 7},
    "gst_pct":       18,
}

STATE_FACTORS: dict[str, float] = {
    "Tamil Nadu": 1.00, "Maharashtra": 1.12, "Karnataka": 1.05,
    "Kerala": 1.08, "Andhra Pradesh": 0.96, "Telangana": 0.98,
    "Delhi": 1.25, "Uttar Pradesh": 0.88, "Rajasthan": 0.82,
    "West Bengal": 0.91, "Gujarat": 1.03, "Punjab": 1.10,
    "_default": 1.00,
}


@dataclass
class DetectionCost:
    class_name:      str
    severity_tier:   str
    area_m2:         float
    depth_cm:        float
    unit_rate_inr:   float
    labour_inr:      float
    mob_inr:         float
    subtotal_inr:    float
    gst_inr:         float
    total_cost_inr:  float
    method:          str
    source:          str
    urgency_days:    int

    def to_dict(self) -> dict:
        return {
            "class_name":     self.class_name,
            "severity_tier":  self.severity_tier,
            "area_m2":        round(self.area_m2, 3),
            "depth_cm":       round(self.depth_cm, 1),
            "unit_rate_inr":  round(self.unit_rate_inr),
            "labour_inr":     round(self.labour_inr),
            "mob_inr":        round(self.mob_inr),
            "gst_inr":        round(self.gst_inr),
            "total_cost_inr": round(self.total_cost_inr),
            "method":         self.method,
            "source":         self.source,
            "urgency_days":   self.urgency_days,
        }


@dataclass
class FrameCostEstimate:
    frame_index: int
    detections:  list
    total_cost:  float
    state:       str

    def to_dict(self) -> dict:
        return {
            "frame_index":    self.frame_index,
            "total_cost_inr": round(self.total_cost, 2),
            "state":          self.state,
            "detections":     [d.to_dict() for d in self.detections],
        }


@dataclass
class SegmentCostReport:
    segment_id:       str
    total_cost_inr:   float
    avg_cost_per_m2:  float
    class_costs:      dict
    method_summary:   dict
    source_citations: list
    urgency_days:     int

    def __str__(self) -> str:
        lines = [
            f"Segment: {self.segment_id}",
            f"  Total cost (INR, incl. GST): Rs.{self.total_cost_inr:,.0f}",
            f"  Avg per m2:                  Rs.{self.avg_cost_per_m2:,.0f}",
            f"  Urgency:                     repair within {self.urgency_days} day(s)",
            "  By class:",
        ]
        for cls, cost in sorted(self.class_costs.items(), key=lambda x: -x[1]):
            lines.append(f"    {cls:<26} Rs.{cost:>10,.0f}")
        lines.append("  SOR Sources:")
        for src in self.source_citations:
            lines.append(f"    - {src}")
        return "\n".join(lines)


class RepairCostEstimator:
    """
    Estimates road repair costs using government Schedule of Rates.

    Parameters
    ----------
    state : str
        Indian state name for rate adjustment (see STATE_FACTORS).
    include_gst : bool
        Include 18% GST in totals (default True).
    frame_w, frame_h : int
        Frame dimensions for perspective scale correction.
    road_width_m : float
        Road carriageway width in metres.
    """

    def __init__(
        self,
        state:        str   = "Tamil Nadu",
        include_gst:  bool  = True,
        frame_w:      int   = 1280,
        frame_h:      int   = 720,
        road_width_m: float = 7.0,
    ):
        self.state         = state
        self.include_gst   = include_gst
        self.frame_w       = frame_w
        self.frame_h       = frame_h
        self._state_factor = STATE_FACTORS.get(state, STATE_FACTORS["_default"])
        self._px_per_cm    = (frame_w / road_width_m) / 100.0

    def estimate_frame(
        self,
        detections:  list,
        frame_index: int = 0,
    ) -> FrameCostEstimate:
        """
        Estimate repair cost for all detections in one frame.

        Each detection dict requires:
          bbox: [x1,y1,x2,y2], class_name: str
        Optional:
          depth_cm: float (from depth_estimator), confidence: float
        """
        det_costs = []
        for d in detections:
            x1, y1, x2, y2 = d.get("bbox", [0, 0, 10, 10])
            depth_cm  = float(d.get("depth_cm", 0.0))
            cls_name  = d.get("class_name", d.get("class", "Unknown"))

            cy = (y1 + y2) / 2.0
            road_top  = self.frame_h * 0.35
            road_frac = max(0.1, (cy - road_top) / max(self.frame_h - road_top, 1.0))
            px_per_cm_local = self._px_per_cm * road_frac
            bw_cm  = max(x2 - x1, 1) / max(px_per_cm_local, 0.01)
            bh_cm  = max(y2 - y1, 1) / max(px_per_cm_local, 0.01)
            area_m2 = max((bw_cm * bh_cm) / 10_000.0, 0.001)

            sor   = SOR.get(cls_name, SOR_DEFAULT)
            tier  = self._severity_tier(area_m2, depth_cm)
            unit_rate = sor[f"rate_{tier}"] * self._state_factor
            labour    = unit_rate * area_m2 * sor["labour_pct"]
            mob       = sor["mob_cost"]
            subtotal  = unit_rate * area_m2 + labour + mob
            gst       = subtotal * sor["gst_pct"] / 100.0 if self.include_gst else 0.0
            total     = subtotal + gst

            det_costs.append(DetectionCost(
                class_name     = cls_name,
                severity_tier  = tier,
                area_m2        = area_m2,
                depth_cm       = depth_cm,
                unit_rate_inr  = unit_rate,
                labour_inr     = labour,
                mob_inr        = mob,
                subtotal_inr   = subtotal,
                gst_inr        = gst,
                total_cost_inr = total,
                method         = sor[f"method_{tier}"],
                source         = sor["source"],
                urgency_days   = sor["urgency_days"][tier],
            ))

        return FrameCostEstimate(
            frame_index = frame_index,
            detections  = det_costs,
            total_cost  = sum(d.total_cost_inr for d in det_costs),
            state       = self.state,
        )

    def aggregate(
        self,
        estimates:  list,
        segment_id: str = "SEG-001",
    ) -> SegmentCostReport:
        total      = sum(e.total_cost for e in estimates)
        total_area = sum(d.area_m2 for e in estimates for d in e.detections)
        class_costs:    dict = {}
        method_summary: dict = {}
        sources:        set  = set()
        min_urgency:    int  = 999
        for e in estimates:
            for d in e.detections:
                class_costs[d.class_name] = class_costs.get(d.class_name, 0) + d.total_cost_inr
                method_summary[d.class_name] = d.method
                sources.add(d.source)
                min_urgency = min(min_urgency, d.urgency_days)
        return SegmentCostReport(
            segment_id       = segment_id,
            total_cost_inr   = total,
            avg_cost_per_m2  = total / max(total_area, 0.001),
            class_costs      = class_costs,
            method_summary   = method_summary,
            source_citations = sorted(sources),
            urgency_days     = min_urgency if min_urgency < 999 else 30,
        )

    @staticmethod
    def _severity_tier(area_m2: float, depth_cm: float) -> str:
        """IRC:SP:16-2018 §7.3 severity classification."""
        if area_m2 > 5.0 or depth_cm > 8.0:
            return "high"
        if area_m2 > 1.0 or depth_cm > 3.0:
            return "medium"
        return "low"

    def get_rate_citation(self, class_name: str) -> str:
        return SOR.get(class_name, SOR_DEFAULT)["source"]

    def get_methodology(self, class_name: str, area_m2: float, depth_cm: float) -> str:
        sor  = SOR.get(class_name, SOR_DEFAULT)
        tier = self._severity_tier(area_m2, depth_cm)
        return sor[f"method_{tier}"]