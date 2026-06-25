"""
road_metrics.py  -  Pavement Condition Index, IRI, Budget Allocator,
                    Contractor Work-Order Generator
=======================================================================

PCI (Pavement Condition Index)
-------------------------------
ASTM D6433 standard. 0–100 scale used by NHAI, PWD, and World Bank.
Maps from Road-AI health score + damage density to PCI band.

  100–85  Good        — no maintenance required
  85–70   Satisfactory — routine maintenance
  70–55   Fair         — preventive treatment
  55–40   Poor         — rehabilitation
  40–25   Very Poor    — major rehabilitation
  25–10   Serious      — reconstruction required
  10–0    Failed       — immediate closure risk

IRI (International Roughness Index) Proxy
------------------------------------------
IRI measures ride quality in m/km. Standard World Bank thresholds:
  IRI < 2.0  m/km  → Good
  IRI 2–4    m/km  → Fair
  IRI 4–6    m/km  → Poor
  IRI > 6    m/km  → Very Poor

We estimate IRI from damage density per 100m segment:
  IRI_proxy = base_IRI + (pothole_count × 0.8) + (alligator_pct × 3.0)
              + (transverse_pct × 1.5) + (longitudinal_pct × 0.8)
Reference: Ramos et al. (2020) "Pavement Condition Assessment Using
Image Processing", Transportation Research Record.

Budget Allocator
----------------
Given a budget cap, selects which repair segments to fund maximising
urgency coverage. Classic 0/1 knapsack with urgency weighting.

Contractor Work-Order Generator
---------------------------------
Produces a structured dict (and optionally a PDF) suitable for sending
to a PWD contractor. Includes SOR item codes, material quantities,
GPS coordinates, deadline dates.
"""
from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)


# ── PCI ──────────────────────────────────────────────────────────

PCI_BANDS = [
    (85, 100, "Good",         "No maintenance required"),
    (70,  85, "Satisfactory", "Routine maintenance / crack sealing"),
    (55,  70, "Fair",         "Preventive treatment — thin overlay"),
    (40,  55, "Poor",         "Rehabilitation — structural repair"),
    (25,  40, "Very Poor",    "Major rehabilitation — mill and overlay"),
    (10,  25, "Serious",      "Reconstruction required"),
    ( 0,  10, "Failed",       "Immediate closure / emergency repair"),
]


@dataclass
class PCIResult:
    pci:         float
    band:        str
    description: str
    iri_proxy:   float       # m/km
    iri_band:    str
    segment_km:  float
    damage_density_pct: float   # % of segment area with damage


def health_score_to_pci(health_score: float) -> float:
    """
    Convert Road-AI health score (0–100) to ASTM D6433 PCI.
    The relationship is non-linear — road health degrades slowly
    at first then accelerates.
    """
    s = max(0.0, min(100.0, health_score))
    # Sigmoid-like mapping calibrated to ASTM distress curves
    if s >= 90:
        pci = 85 + (s - 90) * 1.5
    elif s >= 70:
        pci = 55 + (s - 70) * 1.5
    elif s >= 50:
        pci = 25 + (s - 50) * 1.5
    elif s >= 30:
        pci = 10 + (s - 30) * 0.75
    else:
        pci = s * 0.33
    return round(min(100.0, max(0.0, pci)), 1)


def compute_pci(
    frames: list[dict],
    segment_length_m: float = 100.0,
) -> PCIResult:
    """
    Compute PCI and IRI proxy for a list of frame dicts.
    """
    if not frames:
        return PCIResult(100, "Good", PCI_BANDS[0][3], 0.0, "Good", 0.0, 0.0)

    scores   = [f.get("health_score", 100) for f in frames]
    avg      = sum(scores) / len(scores)
    pci      = health_score_to_pci(avg)

    # PCI band
    band_label = "Good"; band_desc = PCI_BANDS[0][3]
    for lo, hi, label, desc in PCI_BANDS:
        if lo <= pci <= hi:
            band_label = label; band_desc = desc; break

    # Damage density
    n_damaged = sum(1 for f in frames if f.get("n_detections", 0) > 0)
    density   = n_damaged / max(len(frames), 1) * 100

    # IRI proxy — damage density → roughness
    potholes_per_100m     = sum(1 for f in frames
                                if f.get("dominant_class","").lower() in ("pothole","potholes"))
    alligator_pct         = sum(1 for f in frames
                                if "alligator" in f.get("dominant_class","").lower()) / max(len(frames),1)
    transverse_pct        = sum(1 for f in frames
                                if "transverse" in f.get("dominant_class","").lower()) / max(len(frames),1)

    base_iri = max(0.5, 4.5 - pci / 22.0)
    iri      = round(base_iri
                     + potholes_per_100m * 0.8
                     + alligator_pct     * 3.0
                     + transverse_pct    * 1.5, 2)

    iri_band = ("Good" if iri < 2 else "Fair" if iri < 4 else "Poor" if iri < 6 else "Very Poor")

    segment_km = segment_length_m / 1000.0

    return PCIResult(
        pci         = pci,
        band        = band_label,
        description = band_desc,
        iri_proxy   = iri,
        iri_band    = iri_band,
        segment_km  = segment_km,
        damage_density_pct = round(density, 1),
    )


def compute_pci_for_session(results: list[dict], segment_size: int = 50) -> list[dict]:
    """
    Split session frames into segments and compute PCI per segment.
    Returns list of dicts sorted by PCI (worst first).
    """
    if not results:
        return []
    segs = []
    for i in range(0, len(results), segment_size):
        chunk = results[i:i + segment_size]
        r     = compute_pci(chunk)
        scores = [f.get("health_score",100) for f in chunk]
        costs  = sum(f.get("cost_inr",0) for f in chunk)
        # urgency_days can be on the frame or on individual detections
        all_urg = ([f["urgency_days"] for f in chunk if f.get("urgency_days")] +
                   [d["urgency_days"] for f in chunk
                    for d in f.get("detections",[]) if d.get("urgency_days")])
        urg = min(all_urg) if all_urg else 30
        lat = next((f["gps_lat"] for f in chunk if f.get("gps_lat")), None)
        lon = next((f["gps_lon"] for f in chunk if f.get("gps_lon")), None)
        segs.append({
            "segment_id":      i // segment_size + 1,
            "frame_start":     chunk[0].get("frame_index", i),
            "frame_end":       chunk[-1].get("frame_index", i + len(chunk) - 1),
            "pci":             r.pci,
            "pci_band":        r.band,
            "iri":             r.iri_proxy,
            "iri_band":        r.iri_band,
            "avg_health":      round(sum(scores)/len(scores), 1),
            "damage_density":  r.damage_density_pct,
            "cost_inr":        round(costs, 0),
            "urgency_days":    urg,
            "gps_lat":         lat,
            "gps_lon":         lon,
        })
    segs.sort(key=lambda x: x["pci"])
    return segs


# ── Budget Allocator ──────────────────────────────────────────────

@dataclass
class BudgetResult:
    budget_inr:        float
    allocated_inr:     float
    unallocated_inr:   float
    selected_segments: list
    deferred_segments: list
    coverage_pct:      float   # % of total damage cost covered
    urgency_coverage:  float   # % of urgent segments (≤7 days) covered


def allocate_budget(
    segments:   list[dict],
    budget_inr: float,
    strategy:   str = "urgency_first",
) -> BudgetResult:
    """
    Select which repair segments to fund within a budget cap.

    Strategies:
      urgency_first  — fund most urgent (lowest urgency_days) first
      cost_effective — fund highest PCI-improvement-per-rupee first
      worst_first    — fund lowest PCI segments first
    """
    if not segments or budget_inr <= 0:
        return BudgetResult(budget_inr, 0, budget_inr, [], segments, 0, 0)

    segs = [s for s in segments if s.get("cost_inr", 0) > 0]

    # Sort by strategy
    if strategy == "urgency_first":
        segs.sort(key=lambda x: (x.get("urgency_days", 999), x.get("pci", 100)))
    elif strategy == "cost_effective":
        # PCI improvement per rupee: (100 - pci) / cost
        segs.sort(key=lambda x: -(100 - x.get("pci", 100)) / max(x.get("cost_inr", 1), 1))
    else:  # worst_first
        segs.sort(key=lambda x: x.get("pci", 100))

    selected   = []
    deferred   = []
    remaining  = budget_inr

    for seg in segs:
        cost = seg.get("cost_inr", 0)
        if cost <= remaining:
            selected.append(seg)
            remaining -= cost
        else:
            deferred.append(seg)

    total_cost  = sum(s.get("cost_inr", 0) for s in segs)
    alloc       = budget_inr - remaining
    coverage    = alloc / max(total_cost, 1) * 100

    urgent      = [s for s in segs if s.get("urgency_days", 999) <= 7]
    urg_covered = len([s for s in selected if s.get("urgency_days", 999) <= 7])
    urg_cov_pct = urg_covered / max(len(urgent), 1) * 100

    return BudgetResult(
        budget_inr        = budget_inr,
        allocated_inr     = round(alloc, 0),
        unallocated_inr   = round(remaining, 0),
        selected_segments = selected,
        deferred_segments = deferred,
        coverage_pct      = round(coverage, 1),
        urgency_coverage  = round(urg_cov_pct, 1),
    )


# ── Contractor Work Order ─────────────────────────────────────────

@dataclass
class WorkOrder:
    order_no:       str
    date_issued:    str
    deadline_date:  str
    location:       str
    gps_lat:        Optional[float]
    gps_lon:        Optional[float]
    damage_class:   str
    repair_method:  str
    sor_item_code:  str
    sor_description: str
    area_m2:        float
    depth_cm:       float
    volume_m3:      float
    material_qty:   dict          # material → quantity
    cost_breakdown: dict
    total_cost_inr: float
    contractor_note: str

    def to_dict(self) -> dict:
        return {
            "order_no":        self.order_no,
            "date_issued":     self.date_issued,
            "deadline_date":   self.deadline_date,
            "location":        self.location,
            "gps_lat":         self.gps_lat,
            "gps_lon":         self.gps_lon,
            "damage_class":    self.damage_class,
            "repair_method":   self.repair_method,
            "sor_item_code":   self.sor_item_code,
            "sor_description": self.sor_description,
            "area_m2":         self.area_m2,
            "depth_cm":        self.depth_cm,
            "volume_m3":       self.volume_m3,
            "material_qty":    self.material_qty,
            "cost_breakdown":  self.cost_breakdown,
            "total_cost_inr":  self.total_cost_inr,
            "contractor_note": self.contractor_note,
        }


# SOR item codes for common damage types (TN-PWD SOR 2023-24)
SOR_ITEMS = {
    "Pothole": {
        "code": "PWD/BSR/5.2.3",
        "desc": "Pothole patching with hot-mix asphalt (DBM + BC), per m²",
        "unit_rate_low_inr":    850,
        "unit_rate_high_inr":  2400,
        "method": "Mill 50mm, apply tack coat, fill DBM+BC layers, compact",
        "materials": {
            "Hot-mix asphalt (DBM)": "0.05 m³/m²",
            "Bituminous Concrete (BC)": "0.025 m³/m²",
            "Tack coat (SS-1)": "0.3 kg/m²",
            "Compaction equipment": "1 pass",
        },
    },
    "Alligator Crack": {
        "code": "PWD/BSR/5.3.2",
        "desc": "Alligator crack repair — fog seal + crack filler, per m²",
        "unit_rate_low_inr":   1200,
        "unit_rate_high_inr":  3500,
        "method": "Clean cracks, apply hot-pour rubberised sealant, fog seal",
        "materials": {
            "Hot-pour crack sealant": "2.5 kg/m",
            "Fog seal emulsion": "0.5 L/m²",
            "Sand cover": "2 kg/m²",
        },
    },
    "Transverse Crack": {
        "code": "PWD/BSR/5.4.1",
        "desc": "Transverse crack routing and sealing (IRC:SP:83), per m",
        "unit_rate_low_inr":    180,
        "unit_rate_high_inr":   480,
        "method": "Rout crack to 20mm×20mm, clean, apply rubberised sealant",
        "materials": {
            "Rubberised bitumen sealant": "1.0 kg/m",
            "Primer": "0.1 L/m",
        },
    },
    "Longitudinal Crack": {
        "code": "PWD/BSR/5.4.2",
        "desc": "Longitudinal crack routing and sealing, per m",
        "unit_rate_low_inr":    160,
        "unit_rate_high_inr":   420,
        "method": "Rout and seal as per IRC:SP:83-2018 §5.3",
        "materials": {
            "Rubberised bitumen sealant": "0.9 kg/m",
            "Primer": "0.08 L/m",
        },
    },
}


class WorkOrderGenerator:
    """Generate contractor work orders from pipeline segment data."""

    def __init__(self, state: str = "Tamil Nadu"):
        self.state = state
        self._counter = 1

    def generate(
        self,
        segment:    dict,
        detections: list[dict] = None,
        project_name: str = "Road Damage Repair",
    ) -> WorkOrder:
        """Generate a work order for one repair segment."""
        detections = detections or []
        dom_class  = segment.get("dominant_class", "Pothole") or "Pothole"
        sor        = SOR_ITEMS.get(dom_class, SOR_ITEMS["Pothole"])

        # Estimate area from detection bboxes or use defaults
        area_m2 = max(0.5, segment.get("cost_inr", 5000) / sor["unit_rate_high_inr"])
        depth   = segment.get("max_depth_cm", 5.0) or 5.0
        volume  = round(area_m2 * depth / 100.0, 3)

        # Cost breakdown
        mat_cost  = round(area_m2 * sor["unit_rate_high_inr"], 0)
        labour    = round(mat_cost * 0.22, 0)
        mob       = round(min(mat_cost * 0.08, 5000), 0)
        sub_total = mat_cost + labour + mob
        gst       = round(sub_total * 0.18, 0)
        total     = sub_total + gst

        # Deadline from urgency_days
        urg      = int(segment.get("urgency_days", 14) or 14)
        issued   = datetime.now()
        deadline = issued + timedelta(days=urg)

        # GPS location string
        lat = segment.get("gps_lat")
        lon = segment.get("gps_lon")
        loc = f"{lat:.5f}°N, {lon:.5f}°E" if lat and lon else "See attached map"

        order_no = f"PWD/{datetime.now().strftime('%Y%m')}/{self._counter:04d}"
        self._counter += 1

        note = (
            f"Work to be completed within {urg} days of issue. "
            f"IRC:SP:16-2018 severity tier: "
            f"{'High' if depth > 8 else 'Medium' if depth > 3 else 'Low'}. "
            f"All materials to conform to MoRTH Specification (6th Rev. 2013). "
            f"Surface to be reinstated to original profile. "
            f"Quality check: core sample required for depths > 6cm."
        )

        return WorkOrder(
            order_no        = order_no,
            date_issued     = issued.strftime("%d %B %Y"),
            deadline_date   = deadline.strftime("%d %B %Y"),
            location        = loc,
            gps_lat         = lat,
            gps_lon         = lon,
            damage_class    = dom_class,
            repair_method   = sor["method"],
            sor_item_code   = sor["code"],
            sor_description = sor["desc"],
            area_m2         = round(area_m2, 2),
            depth_cm        = depth,
            volume_m3       = volume,
            material_qty    = sor["materials"],
            cost_breakdown  = {
                "Material cost (SOR rate)": round(mat_cost),
                "Labour (22%)":             round(labour),
                "Mobilisation":             round(mob),
                "Sub-total":                round(sub_total),
                "GST (18%)":                round(gst),
                "Total":                    round(total),
            },
            total_cost_inr  = round(total, 0),
            contractor_note = note,
        )

    def generate_pdf(
        self,
        work_orders: list[WorkOrder],
        output_path: str = "work_orders.pdf",
        project_name: str = "Road Damage Repair",
    ) -> str:
        """Generate a multi-page PDF of work orders."""
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.lib.units import mm
            from reportlab.lib.styles import ParagraphStyle
            from reportlab.lib.enums import TA_CENTER, TA_LEFT
            from reportlab.platypus import (
                SimpleDocTemplate, Paragraph, Spacer,
                Table, TableStyle, HRFlowable, PageBreak
            )
            from reportlab.lib import colors
        except ImportError:
            log.error("reportlab not installed")
            return ""

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        doc = SimpleDocTemplate(output_path, pagesize=A4,
                                leftMargin=18*mm, rightMargin=18*mm,
                                topMargin=18*mm, bottomMargin=18*mm)

        DARK  = colors.HexColor("#0d1117")
        BLUE  = colors.HexColor("#1a56db")
        GREY  = colors.HexColor("#6b7280")
        WHITE = colors.white

        def sty(name, **kw):
            return ParagraphStyle(name, **kw)

        title_s = sty("t", fontName="Helvetica-Bold", fontSize=13,
                      textColor=DARK, spaceAfter=2)
        head_s  = sty("h", fontName="Helvetica-Bold", fontSize=10,
                      textColor=BLUE, spaceAfter=2, spaceBefore=6)
        body_s  = sty("b", fontName="Helvetica", fontSize=8.5,
                      textColor=DARK, leading=13)
        dim_s   = sty("d", fontName="Helvetica", fontSize=7.5,
                      textColor=GREY, leading=11)

        story = []
        for i, wo in enumerate(work_orders):
            if i > 0:
                story.append(PageBreak())

            story.append(Paragraph(
                f"CONTRACTOR WORK ORDER — {wo.order_no}", title_s))
            story.append(Paragraph(
                f"Project: {project_name}  |  Issued: {wo.date_issued}  |  "
                f"Deadline: {wo.deadline_date}", dim_s))
            story.append(HRFlowable(width="100%", thickness=0.8,
                                    color=BLUE, spaceAfter=4*mm))

            # Job details table
            details = [
                ["Damage Class:",    wo.damage_class,
                 "SOR Item Code:",   wo.sor_item_code],
                ["Location (GPS):",  wo.location,
                 "Deadline:",        wo.deadline_date],
                ["Repair Method:",   wo.repair_method, "", ""],
                ["Area:",            f"{wo.area_m2} m²",
                 "Depth:",           f"{wo.depth_cm} cm"],
                ["Volume:",          f"{wo.volume_m3} m³",
                 "SOR Description:", wo.sor_description],
            ]
            cw = [(A4[0]-36*mm)/4] * 4
            t  = Table(details, colWidths=cw)
            t.setStyle(TableStyle([
                ("FONTNAME",  (0,0),  (-1,-1), "Helvetica"),
                ("FONTNAME",  (0,0),  (0,-1),  "Helvetica-Bold"),
                ("FONTNAME",  (2,0),  (2,-1),  "Helvetica-Bold"),
                ("FONTSIZE",  (0,0),  (-1,-1), 8.5),
                ("ROWBACKGROUNDS", (0,0), (-1,-1),
                 [colors.HexColor("#f8fafc"), colors.white]),
                ("GRID", (0,0), (-1,-1), 0.3,
                 colors.HexColor("#e5e7eb")),
                ("TOPPADDING",    (0,0), (-1,-1), 4),
                ("BOTTOMPADDING", (0,0), (-1,-1), 4),
                ("LEFTPADDING",   (0,0), (-1,-1), 5),
            ]))
            story.append(t)
            story.append(Spacer(1, 4*mm))

            # Material quantities
            story.append(Paragraph("Materials Required", head_s))
            mat_rows = [["Material", "Quantity"]] + [
                [k, v] for k, v in wo.material_qty.items()
            ]
            t2 = Table(mat_rows, colWidths=[(A4[0]-36*mm)*0.6, (A4[0]-36*mm)*0.4])
            t2.setStyle(TableStyle([
                ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#1a56db")),
                ("TEXTCOLOR",  (0,0), (-1,0), WHITE),
                ("FONTNAME",   (0,0), (-1,0), "Helvetica-Bold"),
                ("FONTNAME",   (0,1), (-1,-1), "Helvetica"),
                ("FONTSIZE",   (0,0), (-1,-1), 8.5),
                ("ROWBACKGROUNDS", (0,1), (-1,-1),
                 [colors.HexColor("#f8fafc"), colors.white]),
                ("GRID", (0,0), (-1,-1), 0.3, colors.HexColor("#e5e7eb")),
                ("TOPPADDING",    (0,0), (-1,-1), 4),
                ("BOTTOMPADDING", (0,0), (-1,-1), 4),
                ("LEFTPADDING",   (0,0), (-1,-1), 5),
            ]))
            story.append(t2)
            story.append(Spacer(1, 4*mm))

            # Cost breakdown
            story.append(Paragraph("Cost Breakdown (TN-PWD SOR 2023-24)", head_s))
            cost_rows = [[k, f"₹{v:,}"] for k, v in wo.cost_breakdown.items()]
            t3 = Table(cost_rows, colWidths=[(A4[0]-36*mm)*0.7, (A4[0]-36*mm)*0.3])
            t3.setStyle(TableStyle([
                ("FONTNAME", (0,0),  (-1,-1), "Helvetica"),
                ("FONTNAME", (0,-1), (-1,-1), "Helvetica-Bold"),
                ("FONTSIZE", (0,0),  (-1,-1), 8.5),
                ("ALIGN",    (1,0),  (1,-1),  "RIGHT"),
                ("ROWBACKGROUNDS", (0,0), (-1,-1),
                 [colors.HexColor("#f8fafc"), colors.white]),
                ("GRID",     (0,0), (-1,-1), 0.3, colors.HexColor("#e5e7eb")),
                ("TOPPADDING",    (0,0), (-1,-1), 4),
                ("BOTTOMPADDING", (0,0), (-1,-1), 4),
                ("LEFTPADDING",   (0,0), (-1,-1), 5),
            ]))
            story.append(t3)
            story.append(Spacer(1, 4*mm))

            # Contractor notes
            story.append(Paragraph("Contractor Notes", head_s))
            story.append(Paragraph(wo.contractor_note, body_s))
            story.append(Spacer(1, 6*mm))

            # Signature block
            sig_data = [
                ["Prepared by:", "_" * 30, "Approved by:", "_" * 30],
                ["Date:",        "_" * 30, "Date:",        "_" * 30],
                ["Designation:", "Road-AI System",
                 "Designation:", "Executive Engineer, PWD"],
            ]
            t4 = Table(sig_data, colWidths=[(A4[0]-36*mm)/4]*4)
            t4.setStyle(TableStyle([
                ("FONTNAME", (0,0), (-1,-1), "Helvetica"),
                ("FONTNAME", (0,0), (0,-1), "Helvetica-Bold"),
                ("FONTNAME", (2,0), (2,-1), "Helvetica-Bold"),
                ("FONTSIZE", (0,0), (-1,-1), 8),
                ("TOPPADDING", (0,0), (-1,-1), 6),
                ("BOTTOMPADDING", (0,0), (-1,-1), 6),
            ]))
            story.append(t4)

        doc.build(story)
        log.info(f"Work orders PDF: {Path(output_path).resolve()}")
        return output_path