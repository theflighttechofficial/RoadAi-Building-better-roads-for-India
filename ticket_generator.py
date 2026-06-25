"""
ticket_generator.py — Road-AI Repair Ticket Generator

Generates official-style repair work order tickets for each damaged road segment.
Each ticket includes:
  - Unique ticket ID and QR-style reference
  - Location details (GPS + segment info)
  - Damage description and severity
  - Cost estimate with breakdown
  - Priority level and deadline
  - Contractor assignment section
  - Sign-off fields

Requires: reportlab (pip install reportlab)
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional
import json

log = logging.getLogger(__name__)

try:
    from qr_generator import qr_for_reportlab
    QR_OK = True
except ImportError:
    QR_OK = False

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import cm, mm
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        HRFlowable, PageBreak, KeepTogether
    )
    from reportlab.graphics.shapes import (
        Drawing, Rect, String, Line, Circle, Polygon
    )
    from reportlab.graphics import renderPDF
    REPORTLAB_OK = True
except ImportError:
    REPORTLAB_OK = False


# ── Palette ───────────────────────────────────────────────────────
class P:
    DARK    = colors.HexColor("#080b10")
    S1      = colors.HexColor("#0d1117")
    S2      = colors.HexColor("#13161e")
    BORDER  = colors.HexColor("#21262d")
    ACCENT  = colors.HexColor("#10d98a")
    GOOD    = colors.HexColor("#10d98a")
    MOD     = colors.HexColor("#f5a623")
    POOR    = colors.HexColor("#f04444")
    CRIT    = colors.HexColor("#a855f7")
    TEXT    = colors.HexColor("#e2e8f0")
    DIM     = colors.HexColor("#7c8299")
    WHITE   = colors.white
    ORANGE  = colors.HexColor("#f97316")


def priority_color(score):
    if score < 40:  return P.CRIT
    if score < 60:  return P.POOR
    if score < 80:  return P.MOD
    return P.GOOD

def priority_label(score):
    if score < 40:  return "P1 — CRITICAL"
    if score < 60:  return "P2 — HIGH"
    if score < 80:  return "P3 — MEDIUM"
    return "P4 — LOW"

def deadline_days(score):
    if score < 40:  return 3
    if score < 60:  return 7
    if score < 80:  return 30
    return 90


@dataclass
class DamageSegment:
    """A contiguous stretch of road with damage."""
    segment_id:    str
    ticket_no:     str
    start_frame:   int
    end_frame:     int
    start_time:    float
    end_time:      float
    avg_score:     float
    min_score:     float
    total_cost:    float
    total_dets:    int
    dominant_class: str
    class_counts:  Dict[str, int] = field(default_factory=dict)
    class_costs:   Dict[str, float] = field(default_factory=dict)
    gps_lat:       Optional[float] = None
    gps_lon:       Optional[float] = None
    length_m:      float = 0.0
    road_name:     str = "Unnamed Road"
    district:      str = "—"


def cluster_frames_into_segments(frames: List[Dict],
                                  min_score_threshold: float = 70,
                                  min_frames: int = 3,
                                  gap_frames: int = 10) -> List[DamageSegment]:
    """
    Group consecutive bad frames into damage segments.
    A segment starts when score drops below threshold and ends when
    it recovers above threshold for gap_frames consecutive frames.
    """
    segments = []
    bad_frames = []
    gap_count  = 0
    seg_counter = 1

    for f in frames:
        score = f.get("health_score", 100)
        if score < min_score_threshold:
            bad_frames.append(f)
            gap_count = 0
        else:
            if bad_frames:
                gap_count += 1
                if gap_count >= gap_frames:
                    if len(bad_frames) >= min_frames:
                        seg = _make_segment(bad_frames, seg_counter)
                        segments.append(seg)
                        seg_counter += 1
                    bad_frames = []
                    gap_count  = 0

    # Flush remaining
    if bad_frames and len(bad_frames) >= min_frames:
        segments.append(_make_segment(bad_frames, seg_counter))

    return segments


def _make_segment(frames: List[Dict], idx: int) -> DamageSegment:
    scores   = [f.get("health_score", 100) for f in frames]
    costs    = [f.get("cost_inr", 0) or 0 for f in frames]
    classes  = {}
    class_costs = {}

    for f in frames:
        cls  = f.get("dominant_class", "") or ""
        cost = f.get("cost_inr", 0) or 0
        ndet = f.get("n_detections", 0) or 0
        if cls and ndet > 0:
            classes[cls]     = classes.get(cls, 0) + ndet
            class_costs[cls] = class_costs.get(cls, 0.0) + cost

    dom = max(classes, key=classes.get) if classes else "Unknown"
    gps_frames = [f for f in frames if f.get("gps_lat")]
    mid = gps_frames[len(gps_frames)//2] if gps_frames else frames[len(frames)//2]

    # Estimate length: GPS distance or frame count proxy
    length_m = 0.0
    if len(gps_frames) > 1:
        pts = [(f["gps_lat"], f["gps_lon"]) for f in gps_frames]
        for i in range(len(pts)-1):
            length_m += _haversine(pts[i], pts[i+1])
    else:
        # ~5m per frame at 25 km/h, 30fps
        length_m = len(frames) * (25000/3600/30)

    ticket_no = f"RD-{datetime.now().strftime('%Y%m')}-{idx:04d}"

    return DamageSegment(
        segment_id    = f"SEG-{idx:03d}",
        ticket_no     = ticket_no,
        start_frame   = frames[0].get("frame_index", 0),
        end_frame     = frames[-1].get("frame_index", 0),
        start_time    = frames[0].get("timestamp_sec", 0),
        end_time      = frames[-1].get("timestamp_sec", 0),
        avg_score     = sum(scores)/len(scores),
        min_score     = min(scores),
        total_cost    = sum(costs),
        total_dets    = sum(f.get("n_detections",0) for f in frames),
        dominant_class= dom,
        class_counts  = classes,
        class_costs   = class_costs,
        gps_lat       = mid.get("gps_lat"),
        gps_lon       = mid.get("gps_lon"),
        length_m      = round(length_m, 1),
    )


def _haversine(p1, p2):
    R = 6371000
    lat1, lon1 = math.radians(p1[0]), math.radians(p1[1])
    lat2, lon2 = math.radians(p2[0]), math.radians(p2[1])
    dlat = lat2-lat1; dlon = lon2-lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))


# ── PDF ticket builder ────────────────────────────────────────────
class TicketGenerator:
    def __init__(self):
        if not REPORTLAB_OK:
            raise RuntimeError("reportlab not installed. pip install reportlab")

    def generate_all(self, segments: List[DamageSegment],
                     output_dir: str = "output/tickets") -> List[str]:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        paths = []
        for seg in segments:
            path = str(Path(output_dir) / f"ticket_{seg.ticket_no}.pdf")
            self.generate_ticket(seg, path)
            paths.append(path)
        log.info(f"Generated {len(paths)} repair tickets in {output_dir}")
        return paths

    def generate_summary(self, segments: List[DamageSegment],
                         output_path: str = "output/tickets_summary.pdf") -> str:
        """One PDF with all tickets back to back."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        def _template(canvas, doc):
            canvas.saveState()
            w, h = A4
            canvas.setFillColor(P.DARK)
            canvas.rect(0, h-20*mm, w, 20*mm, fill=1, stroke=0)
            canvas.setFont("Helvetica-Bold", 10)
            canvas.setFillColor(P.ACCENT)
            canvas.drawString(15*mm, h-12*mm, "ROAD-AI — REPAIR WORK ORDERS")
            canvas.setFont("Helvetica", 8)
            canvas.setFillColor(P.DIM)
            canvas.drawRightString(w-15*mm, h-12*mm,
                f"Page {doc.page}  |  {datetime.now().strftime('%d %b %Y')}")
            canvas.setStrokeColor(P.BORDER)
            canvas.line(15*mm, 12*mm, w-15*mm, 12*mm)
            canvas.setFont("Helvetica", 6)
            canvas.setFillColor(P.DIM)
            canvas.drawCentredString(w/2, 8*mm,
                "Road-AI Automated Work Order System — Official Use Only")
            canvas.restoreState()

        doc   = SimpleDocTemplate(output_path, pagesize=A4,
                    leftMargin=15*mm, rightMargin=15*mm,
                    topMargin=24*mm, bottomMargin=18*mm,
                    onFirstPage=_template, onLaterPages=_template)
        story = []
        styles = self._styles()

        # Cover
        story.append(Spacer(1, 8*mm))
        story.append(Paragraph("REPAIR WORK ORDER SUMMARY", styles["title"]))
        story.append(Paragraph(
            f"{len(segments)} segments identified  ·  "
            f"Total estimated cost: ₹{sum(s.total_cost for s in segments):,.0f}",
            styles["subtitle"]))
        story.append(HRFlowable(width="100%", thickness=1,
                                color=P.ACCENT, spaceAfter=6*mm))

        # Summary table
        rows = [["Ticket No", "Segment", "Priority", "Score", "Length", "Cost (₹)", "Deadline"]]
        for s in sorted(segments, key=lambda x: x.avg_score):
            deadline = (datetime.now() + timedelta(days=deadline_days(s.avg_score))).strftime("%d %b %Y")
            rows.append([
                s.ticket_no, s.segment_id,
                priority_label(s.avg_score),
                f"{s.avg_score:.1f}",
                f"{s.length_m:.0f}m",
                f"₹{s.total_cost:,.0f}",
                deadline,
            ])

        t = Table(rows, colWidths=[32*mm,22*mm,30*mm,16*mm,18*mm,24*mm,24*mm])
        t.setStyle(TableStyle([
            ("BACKGROUND",   (0,0),(-1,0),  P.DARK),
            ("TEXTCOLOR",    (0,0),(-1,0),  P.ACCENT),
            ("FONTNAME",     (0,0),(-1,0),  "Helvetica-Bold"),
            ("FONTNAME",     (0,1),(-1,-1), "Helvetica"),
            ("FONTSIZE",     (0,0),(-1,-1), 7.5),
            ("TEXTCOLOR",    (0,1),(-1,-1), P.TEXT),
            ("ROWBACKGROUNDS",(0,1),(-1,-1),
             [colors.HexColor("#0f1117"), colors.HexColor("#13161e")]),
            ("GRID",         (0,0),(-1,-1), 0.3, P.BORDER),
            ("ALIGN",        (0,0),(-1,-1), "CENTER"),
            ("VALIGN",       (0,0),(-1,-1), "MIDDLE"),
            ("TOPPADDING",   (0,0),(-1,-1), 4),
            ("BOTTOMPADDING",(0,0),(-1,-1), 4),
        ]))
        story.append(t)
        story.append(PageBreak())

        # Individual tickets
        for i, seg in enumerate(sorted(segments, key=lambda x: x.avg_score)):
            story += self._ticket_story(seg, styles)
            if i < len(segments)-1:
                story.append(PageBreak())

        doc.build(story)
        log.info(f"Ticket summary PDF → {output_path}")
        return output_path

    def generate_ticket(self, seg: DamageSegment, output_path: str) -> str:
        def _tpl(canvas, doc):
            canvas.saveState()
            w, h = A4
            canvas.setFillColor(P.DARK)
            canvas.rect(0, h-18*mm, w, 18*mm, fill=1, stroke=0)
            canvas.setFont("Helvetica-Bold", 9)
            canvas.setFillColor(P.ACCENT)
            canvas.drawString(12*mm, h-10*mm, "ROAD-AI REPAIR WORK ORDER")
            canvas.setFont("Helvetica", 8)
            canvas.setFillColor(P.DIM)
            canvas.drawRightString(w-12*mm, h-10*mm, seg.ticket_no)
            canvas.restoreState()

        doc = SimpleDocTemplate(output_path, pagesize=A4,
                  leftMargin=15*mm, rightMargin=15*mm,
                  topMargin=22*mm, bottomMargin=16*mm,
                  onFirstPage=_tpl, onLaterPages=_tpl)
        doc.build(self._ticket_story(seg, self._styles()))
        return output_path

    def _ticket_story(self, seg: DamageSegment, styles: dict) -> list:
        story = []
        col = priority_color(seg.avg_score)
        deadline = datetime.now() + timedelta(days=deadline_days(seg.avg_score))

        # Header band
        header_data = [[
            Paragraph(seg.ticket_no, styles["ticket_no"]),
            Paragraph(priority_label(seg.avg_score), styles["priority"]),
            Paragraph(f"Score: {seg.avg_score:.1f}/100", styles["score"]),
        ]]
        ht = Table(header_data, colWidths=[65*mm, 60*mm, 45*mm])
        ht.setStyle(TableStyle([
            ("BACKGROUND",  (0,0),(-1,-1), P.DARK),
            ("TOPPADDING",  (0,0),(-1,-1), 8),
            ("BOTTOMPADDING",(0,0),(-1,-1), 8),
            ("LEFTPADDING", (0,0),(-1,-1), 8),
            ("LINEBEFORE",  (1,0),(2,-1), 1, P.BORDER),
            ("VALIGN",      (0,0),(-1,-1), "MIDDLE"),
        ]))
        story.append(ht)
        story.append(HRFlowable(width="100%", thickness=2,
                                color=col, spaceAfter=4*mm))

        # Location & segment info
        loc_rows = [
            ["LOCATION DETAILS", ""],
            ["Segment ID",        seg.segment_id],
            ["Road Name",         seg.road_name],
            ["District",          seg.district],
            ["GPS Coordinates",   f"{seg.gps_lat:.6f}, {seg.gps_lon:.6f}"
                                  if seg.gps_lat else "Not available"],
            ["Maps Link",         f"https://maps.google.com/?q={seg.gps_lat},{seg.gps_lon}"
                                  if seg.gps_lat else "—"],
            ["Segment Length",    f"{seg.length_m:.0f} m"],
            ["Survey Timestamp",  f"{seg.start_time:.1f}s – {seg.end_time:.1f}s"],
            ["Frame Range",       f"#{seg.start_frame} – #{seg.end_frame}"],
        ]
        lt = Table(loc_rows, colWidths=[52*mm, 118*mm])
        lt.setStyle(self._info_style(header_col=P.ACCENT))
        story.append(lt)
        story.append(Spacer(1, 3*mm))

        # Damage details
        dam_rows = [
            ["DAMAGE DETAILS", ""],
            ["Dominant Type",      seg.dominant_class or "Unknown"],
            ["All Classes",        ", ".join(f"{k}×{v}" for k,v in seg.class_counts.items()) or "—"],
            ["Total Detections",   str(seg.total_dets)],
            ["Avg Health Score",   f"{seg.avg_score:.1f}/100"],
            ["Min Health Score",   f"{seg.min_score:.1f}/100"],
            ["Severity",           priority_label(seg.avg_score)],
        ]
        dt = Table(dam_rows, colWidths=[52*mm, 118*mm])
        dt.setStyle(self._info_style(header_col=P.POOR))
        story.append(dt)
        story.append(Spacer(1, 3*mm))

        # Cost breakdown
        cost_rows = [["COST ESTIMATE", "", ""]]
        cost_rows.append(["Damage Class", "Area Estimate", "Cost (INR)"])
        total_shown = 0
        for cls, cost in sorted(seg.class_costs.items(), key=lambda x:-x[1]):
            count = seg.class_counts.get(cls, 0)
            cost_rows.append([cls, f"~{count * 0.25:.1f} m²", f"₹{cost:,.0f}"])
            total_shown += cost
        cost_rows.append(["TOTAL ESTIMATED REPAIR COST", "",
                          f"₹{seg.total_cost:,.0f}"])

        ct = Table(cost_rows, colWidths=[80*mm, 50*mm, 40*mm])
        ct.setStyle(TableStyle([
            ("SPAN",         (0,0),(2,0)),
            ("BACKGROUND",   (0,0),(2,0), P.DARK),
            ("TEXTCOLOR",    (0,0),(2,0), P.MOD),
            ("FONTNAME",     (0,0),(2,0), "Helvetica-Bold"),
            ("BACKGROUND",   (0,1),(2,1), P.S2),
            ("TEXTCOLOR",    (0,1),(2,1), P.DIM),
            ("FONTNAME",     (0,1),(2,1), "Helvetica-Bold"),
            ("FONTNAME",     (0,2),(-1,-2),"Helvetica"),
            ("FONTNAME",     (0,-1),(-1,-1),"Helvetica-Bold"),
            ("BACKGROUND",   (0,-1),(2,-1), P.DARK),
            ("TEXTCOLOR",    (0,-1),(2,-1), P.MOD),
            ("FONTSIZE",     (0,0),(-1,-1), 8),
            ("TEXTCOLOR",    (0,2),(-1,-2), P.TEXT),
            ("ROWBACKGROUNDS",(0,2),(-1,-2),
             [colors.HexColor("#0f1117"), colors.HexColor("#13161e")]),
            ("GRID",         (0,0),(-1,-1), 0.3, P.BORDER),
            ("ALIGN",        (1,0),(-1,-1), "RIGHT"),
            ("ALIGN",        (0,0),(0,-1), "LEFT"),
            ("TOPPADDING",   (0,0),(-1,-1), 4),
            ("BOTTOMPADDING",(0,0),(-1,-1), 4),
            ("LEFTPADDING",  (0,0),(-1,-1), 6),
            ("RIGHTPADDING", (0,0),(-1,-1), 6),
        ]))
        story.append(ct)
        story.append(Spacer(1, 3*mm))

        # Action & sign-off
        action_rows = [
            ["ACTION REQUIRED", ""],
            ["Priority Level",    priority_label(seg.avg_score)],
            ["Repair Deadline",   deadline.strftime("%d %B %Y")],
            ["Days Remaining",    f"{deadline_days(seg.avg_score)} days"],
            ["Recommended Action", self._recommend(seg)],
            ["Assigned Contractor","_" * 35],
            ["Work Order Issued By","_" * 35],
            ["Issue Date",         datetime.now().strftime("%d %B %Y")],
            ["Completion Date",    "________________"],
            ["Verified By",        "________________"],
        ]
        at = Table(action_rows, colWidths=[52*mm, 118*mm])
        at.setStyle(self._info_style(header_col=P.ORANGE))
        story.append(at)
        story.append(Spacer(1, 4*mm))

        # ── QR Code block ─────────────────────────────────────────────────────
        # Two QR codes side by side:
        #   LEFT:  ticket number + segment ID (for field scanning / lookup)
        #   RIGHT: Google Maps deep link to GPS coordinates
        if QR_OK:
            ticket_qr_text = f"{seg.ticket_no} {seg.segment_id}"
            if seg.gps_lat:
                maps_url  = f"https://maps.google.com/?q={seg.gps_lat:.6f},{seg.gps_lon:.6f}"
                maps_label = f"{seg.gps_lat:.5f}, {seg.gps_lon:.5f}"
            else:
                maps_url   = f"ROAD-AI {seg.ticket_no} NO-GPS"
                maps_label = "GPS unavailable"

            qr_ticket = qr_for_reportlab(ticket_qr_text, width_pt=55, height_pt=55)
            qr_maps   = qr_for_reportlab(maps_url,       width_pt=55, height_pt=55)

            qr_table_data = [
                [qr_ticket,                                  qr_maps],
                [Paragraph("Scan: Ticket Lookup",  styles["qr_label"]),
                 Paragraph("Scan: Open in Maps",   styles["qr_label"])],
                [Paragraph(seg.ticket_no,           styles["qr_sub"]),
                 Paragraph(maps_label,              styles["qr_sub"])],
            ]
            qr_table = Table(qr_table_data, colWidths=[85*mm, 85*mm])
            qr_table.setStyle(TableStyle([
                ("ALIGN",        (0,0), (-1,-1), "CENTER"),
                ("VALIGN",       (0,0), (-1,-1), "MIDDLE"),
                ("BACKGROUND",   (0,0), (-1,-1), P.DARK),
                ("TOPPADDING",   (0,0), (-1,-1), 6),
                ("BOTTOMPADDING",(0,0), (-1,-1), 4),
                ("LINEBEFORE",   (1,0), (1,-1),  0.5, P.BORDER),
                ("BOX",          (0,0), (-1,-1),  0.5, P.BORDER),
            ]))
            story.append(qr_table)
            story.append(Spacer(1, 3*mm))

        # Footer note
        story.append(Paragraph(
            f"This work order was automatically generated by Road-AI on "
            f"{datetime.now().strftime('%d %B %Y %H:%M')}. "
            f"Field verification required before commencing repair. "
            f"Ref: {seg.ticket_no}",
            styles["footer"]
        ))
        return story

    def _info_style(self, header_col=None):
        hc = header_col or P.ACCENT
        return TableStyle([
            ("SPAN",          (0,0),(1,0)),
            ("BACKGROUND",    (0,0),(1,0),  P.DARK),
            ("TEXTCOLOR",     (0,0),(1,0),  hc),
            ("FONTNAME",      (0,0),(1,0),  "Helvetica-Bold"),
            ("FONTSIZE",      (0,0),(1,0),  8),
            ("FONTNAME",      (0,1),(0,-1), "Helvetica-Bold"),
            ("FONTNAME",      (1,1),(1,-1), "Helvetica"),
            ("FONTSIZE",      (0,1),(-1,-1),8),
            ("TEXTCOLOR",     (0,1),(0,-1), P.DIM),
            ("TEXTCOLOR",     (1,1),(1,-1), P.TEXT),
            ("ROWBACKGROUNDS",(0,1),(-1,-1),
             [colors.HexColor("#0f1117"), colors.HexColor("#13161e")]),
            ("GRID",          (0,0),(-1,-1), 0.3, P.BORDER),
            ("TOPPADDING",    (0,0),(-1,-1), 4),
            ("BOTTOMPADDING", (0,0),(-1,-1), 4),
            ("LEFTPADDING",   (0,0),(-1,-1), 6),
            ("RIGHTPADDING",  (0,0),(-1,-1), 6),
        ])

    def _recommend(self, seg: DamageSegment) -> str:
        cls = seg.dominant_class or ""
        if "Pothole" in cls:
            return "Hot-mix asphalt patching with full compaction. Temporary cold-mix if urgent."
        if "Alligator" in cls:
            return "Full-depth reclamation. Investigate sub-base failure before patching."
        if "Transverse" in cls:
            return "Crack sealing with rubberized asphalt. Monitor for widening."
        if "Longitudinal" in cls:
            return "Crack sealing. Check for edge subsidence and drainage issues."
        return "Field inspection required to determine appropriate repair method."

    def _styles(self) -> dict:
        base = getSampleStyleSheet()
        def s(name, **kw):
            return ParagraphStyle(name, parent=base["Normal"], **kw)
        return {
            "title":     s("t",  fontSize=16, fontName="Helvetica-Bold",
                           textColor=P.TEXT, alignment=TA_CENTER, spaceAfter=4),
            "subtitle":  s("st", fontSize=9,  fontName="Helvetica",
                           textColor=P.DIM,  alignment=TA_CENTER, spaceAfter=6),
            "ticket_no": s("tn", fontSize=13, fontName="Helvetica-Bold",
                           textColor=P.ACCENT),
            "priority":  s("pr", fontSize=11, fontName="Helvetica-Bold",
                           textColor=P.POOR, alignment=TA_CENTER),
            "score":     s("sc", fontSize=11, fontName="Helvetica-Bold",
                           textColor=P.TEXT, alignment=TA_RIGHT),
            "footer":    s("ft", fontSize=6.5, fontName="Helvetica",
                           textColor=P.DIM, alignment=TA_CENTER, leading=9),
            "qr_label":  s("ql", fontSize=7, fontName="Helvetica-Bold",
                           textColor=P.ACCENT, alignment=TA_CENTER),
            "qr_sub":    s("qs", fontSize=6.5, fontName="Helvetica",
                           textColor=P.DIM, alignment=TA_CENTER),
        }