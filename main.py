"""
main.py  -  Road-AI Master Pipeline
=====================================

HOW TO RUN
----------
  # Basic:
  python main.py --video road.mp4 --model yolov8n.pt

  # With ensemble (NEW):
  python main.py --video road.mp4 --model yolov8m.pt --model2 yolov8s_night.pt --ensemble

  # With uncertainty quantification (NEW):
  python main.py --video road.mp4 --model yolov8s.pt --uncertainty --mc-samples 10

  # Full stack (ensemble + uncertainty + active learning):
  python main.py --video road.mp4 --ensemble --uncertainty --active-learning-export

  # Self-supervised pseudo-labeling (NEW):
  python main.py --video unlabeled.mp4 --self-supervised --ssl-output pseudo_dataset

  # With gamification + blockchain + accessibility (NEW):
  python main.py --video road.mp4 --blockchain --accessibility --audio-alerts \
                 --citizen-mode --reporter-id user123 --ward "Anna Nagar"

  # With live dashboard:
  python dashboard_server.py &
  python main.py --video road.mp4 --model yolov8n.pt --dashboard

  # With GPS dead-reckoning:
  python main.py --video road.mp4 --gps 13.0827,80.2707 --gps-bearing 90

  # From GPX track file:
  python main.py --video road.mp4 --gps-file track.gpx

  # RTSP live stream:
  python main.py --video rtsp://192.168.1.1/stream --stream

  # Webcam:
  python main.py --video 0 --stream

  # Full output suite:
  python main.py --video road.mp4 --gps 13.08,80.27 --pdf --excel --tickets --map

  # Auto-email report after analysis:
  python main.py --video road.mp4 --email-to officer@pwd.gov.in
                 --smtp-from you@gmail.com --smtp-pass "xxxx xxxx xxxx xxxx"

  # With environmental conditions (for adaptive ensemble weights):
  python main.py --video night_road.mp4 --ensemble --condition-time night --condition-weather rain
"""
from __future__ import annotations

import argparse, json, logging, math, sys, time, threading, urllib.request, urllib.parse
from pathlib import Path
from datetime import datetime
import cv2

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("pipeline.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════
# NEW: Gamification, Blockchain, Accessibility
# ══════════════════════════════════════════════════════════════════════════

try:
    from gamification import GamificationEngine
    GAMIFICATION_AVAILABLE = True
except ImportError:
    GAMIFICATION_AVAILABLE = False
    log.warning("gamification.py not found - gamification features disabled")

try:
    from blockchain_tracker import SimpleBlockchain
    BLOCKCHAIN_AVAILABLE = True
except ImportError:
    BLOCKCHAIN_AVAILABLE = False
    log.warning("blockchain_tracker.py not found - blockchain features disabled")

try:
    from accessibility import AccessibilityMode
    ACCESSIBILITY_AVAILABLE = True
except ImportError:
    ACCESSIBILITY_AVAILABLE = False
    log.warning("accessibility.py not found - accessibility features disabled")

# ══════════════════════════════════════════════════════════════════════════


# =============================================================================
# GPS helpers
# =============================================================================

def parse_gps_arg(s: str):
    try:
        lat, lon = [float(x.strip()) for x in s.split(",")]
        return (lat, lon)
    except Exception:
        return None


def simulate_gps(start_lat, start_lon, frame_i, fps, speed_kmh=25.0, bearing=90.0):
    """Dead-reckoning GPS position from a known start point + bearing."""
    dist_km = (frame_i / max(fps, 1)) * (speed_kmh / 3600.0)
    R = 6371.0
    lat_r = math.radians(start_lat)
    brg_r = math.radians(bearing)
    d_r   = dist_km / R
    lat2  = math.asin(math.sin(lat_r)*math.cos(d_r) +
                      math.cos(lat_r)*math.sin(d_r)*math.cos(brg_r))
    lon2  = math.radians(start_lon) + math.atan2(
        math.sin(brg_r)*math.sin(d_r)*math.cos(lat_r),
        math.cos(d_r) - math.sin(lat_r)*math.sin(lat2))
    jlat = (hash(frame_i * 7919) % 1000 - 500) * 0.0000008
    jlon = (hash(frame_i * 6271) % 1000 - 500) * 0.0000008
    return round(math.degrees(lat2)+jlat, 6), round(math.degrees(lon2)+jlon, 6)


def load_gps_from_gpx(path: str):
    """Load GPS track from .gpx using gps_overlay, falling back to gpxpy."""
    try:
        from gps_overlay import GPXTrackLoader
        loader = GPXTrackLoader(path)
        duration = int(loader.duration_sec())
        pts = []
        for s in range(duration):
            try:
                p = loader.at_second(float(s))
                pts.append((p.lat, p.lon))
            except Exception:
                break
        log.info(f"GPX (gps_overlay): {len(pts)} points from {path}")
        return pts
    except Exception:
        try:
            import gpxpy
            with open(path) as f:
                gpx = gpxpy.parse(f)
            pts = [(pt.latitude, pt.longitude)
                   for track in gpx.tracks
                   for seg in track.segments
                   for pt in seg.points]
            log.info(f"GPX (gpxpy): {len(pts)} points from {path}")
            return pts
        except Exception as e:
            log.warning(f"GPX load failed: {e}")
            return []


# =============================================================================
# False-positive filter
# =============================================================================

def filter_detections(dets, frame_w, frame_h,
                      min_area_ratio=0.003, max_area_ratio=0.55):
    frame_area = max(frame_w * frame_h, 1)
    out = []
    for d in dets:
        x1, y1, x2, y2 = d["bbox"]
        bw = max(x2 - x1, 1)
        bh = max(y2 - y1, 1)
        ar = (bw * bh) / frame_area
        if ar < min_area_ratio:  continue
        if ar > max_area_ratio:  continue
        if bw / bh < 0.20:       continue
        out.append(d)
    return out


# =============================================================================
# Frame annotation (HUD)
# =============================================================================

def bgr(hx: str):
    hx = hx.lstrip("#")
    return int(hx[4:6], 16), int(hx[2:4], 16), int(hx[0:2], 16)


def annotate(frame, dets, score, cost, fi, gps=None,
             condition_label="", measurements=True, depth_info=None,
             uncertainty_info=None):  # NEW: uncertainty display
    h, w = frame.shape[:2]
    sc = (bgr("a6e3a1") if score >= 80 else bgr("f9e2af") if score >= 60
          else bgr("f38ba8") if score >= 40 else bgr("cba6f7"))

    for d in dets:
        x1, y1, x2, y2 = [int(v) for v in d["bbox"]]
        conf  = d.get("confidence", 0)
        
        # Color based on uncertainty (NEW)
        needs_review = d.get("needs_review", False)
        if needs_review:
            col = bgr("cba6f7")  # Purple for high uncertainty
        else:
            col = (bgr("a6e3a1") if conf >= 0.75 else
                   bgr("f9e2af") if conf >= 0.5 else bgr("f38ba8"))
        
        thick = 1 if d.get("uncertain") else 2
        
        # Dashed line for uncertain detections (NEW)
        if needs_review:
            # Draw dashed rectangle
            dash_length = 8
            for i in range(x1, x2, dash_length * 2):
                cv2.line(frame, (i, y1), (min(i + dash_length, x2), y1), col, thick)
                cv2.line(frame, (i, y2), (min(i + dash_length, x2), y2), col, thick)
            for i in range(y1, y2, dash_length * 2):
                cv2.line(frame, (x1, i), (x1, min(i + dash_length, y2)), col, thick)
                cv2.line(frame, (x2, i), (x2, min(i + dash_length, y2)), col, thick)
        else:
            cv2.rectangle(frame, (x1, y1), (x2, y2), col, thick)

        lbl = f"{d.get('class_name','?')} {conf:.0%}"
        if measurements:
            w_cm = d.get("width_cm", 0)
            d_cm = d.get("depth_cm", 0)
            if w_cm > 0:  lbl += f" {w_cm:.0f}cm"
            if d_cm > 0:  lbl += f" d:{d_cm:.1f}cm"
        
        # Add uncertainty indicator (NEW)
        if d.get("uncertain"):
            lbl += " ?"
        if needs_review:
            unc = d.get("uncertainty", 0)
            lbl += f" U:{unc:.2f}"

        (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
        cv2.rectangle(frame, (x1, y1-th-5), (x1+tw+4, y1), col, -1)
        cv2.putText(frame, lbl, (x1+2, y1-2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (16,16,24), 1, cv2.LINE_AA)

    # HUD panel - expanded for uncertainty info (NEW)
    hud_height = 134 if uncertainty_info else 118
    ov = frame.copy()
    cv2.rectangle(ov, (0,0), (270, hud_height), (6,8,16), -1)
    cv2.addWeighted(ov, 0.8, frame, 0.2, 0, frame)
    cv2.rectangle(frame, (0,0), (270, hud_height), (30,36,52), 1)

    def put(t, x, y, col=(205,214,244), s=0.4):
        cv2.putText(frame, t, (x,y), cv2.FONT_HERSHEY_SIMPLEX, s, (0,0,0), 2, cv2.LINE_AA)
        cv2.putText(frame, t, (x,y), cv2.FONT_HERSHEY_SIMPLEX, s, col,   1, cv2.LINE_AA)

    put("ROAD-AI",          6, 16, bgr("a6e3a1"), 0.48)
    put(f"Frame #{fi}",     6, 32, (98,114,164),  0.36)
    put(f"Score: {score:.1f}/100", 6, 50, sc, 0.43)
    put(f"Det:{len(dets)}  Rs{cost:.0f}", 6, 66, (166,173,200), 0.37)
    if condition_label and condition_label != "NORMAL":
        put(f"Cond: {condition_label}", 6, 82, (137,180,250), 0.33)
    if gps:
        put(f"GPS {gps[0]:.4f},{gps[1]:.4f}", 6, 98, (137,180,250), 0.28)
    if depth_info:
        put(f"Depth: {depth_info}", 6, 114, bgr("f9e2af"), 0.28)
    
    # NEW: Uncertainty info
    if uncertainty_info:
        put(f"Unc: {uncertainty_info}", 6, 130, bgr("cba6f7"), 0.28)

    cv2.rectangle(frame, (0, h-4), (int(score/100*w), h), sc, -1)
    return frame


# =============================================================================
# Dashboard client
# =============================================================================

class Dashboard:
    def __init__(self, port=8000):
        self.base = f"http://localhost:{port}/api"
        self.sid  = None

    def ping(self):
        try:
            urllib.request.urlopen(f"{self.base}/sessions", timeout=2)
            return True
        except Exception:
            return False

    def create_session(self, video):
        try:
            url = (f"{self.base}/sessions/create"
                   f"?video_path={urllib.parse.quote(str(video))}")
            res = urllib.request.urlopen(
                urllib.request.Request(url, method="POST"), timeout=5)
            self.sid = json.loads(res.read())["session_id"]
            log.info(f"Dashboard session: {self.sid}")
            return self.sid
        except Exception as e:
            log.warning(f"Dashboard session failed: {e}")
            return None

    def push(self, data, sync=False):
        if not self.sid:
            return
        def _do():
            try:
                body = json.dumps(data).encode()
                req  = urllib.request.Request(
                    f"{self.base}/sessions/{self.sid}/frame",
                    data=body, method="POST",
                    headers={"Content-Type": "application/json"})
                resp = urllib.request.urlopen(req, timeout=5)
            except urllib.error.HTTPError as e:
                if e.code == 429:
                    # Rate limited — back off and retry once after 100ms
                    import time as _t; _t.sleep(0.1)
                    try:
                        urllib.request.urlopen(
                            urllib.request.Request(
                                f"{self.base}/sessions/{self.sid}/frame",
                                data=body, method="POST",
                                headers={"Content-Type": "application/json"}),
                            timeout=5)
                    except Exception:
                        pass  # drop frame rather than block pipeline
            except Exception:
                pass
        if sync:
            _do()
        else:
            threading.Thread(target=_do, daemon=True).start()

    def done(self):
        if not self.sid:
            return
        try:
            urllib.request.urlopen(
                urllib.request.Request(
                    f"{self.base}/sessions/{self.sid}/done", method="POST"),
                timeout=3)
        except Exception:
            pass


# =============================================================================
# Output generators  (all optional)
# =============================================================================

def _class_breakdown(results: list) -> dict:
    counts: dict = {}
    for r in results:
        cls = r.get("dominant_class", "")
        if cls:
            counts[cls] = counts.get(cls, 0) + r.get("n_detections", 1)
    return counts


def generate_pdf(results, args, out_dir, dash_sid, worst_frame_imgs,
                 fusion_report=None, forecast_data=None,
                 depth_summary=None, sor_breakdown=None,
                 uncertainty_summary=None):  # NEW: uncertainty summary
    """PDF report with photo-evidence and research sections."""
    try:
        from report_generator import PDFReportGenerator, ReportData
        scores = [r["health_score"] for r in results]
        avg    = sum(scores)/len(scores) if scores else 100.0

        worst_sorted  = sorted(results, key=lambda x: x["health_score"])[:10]
        evidence_imgs = {
            f["frame_index"]: worst_frame_imgs[f["frame_index"]]
            for f in worst_sorted
            if f["frame_index"] in worst_frame_imgs
        }
        rd = ReportData(
            video_path        = str(args.video),
            processed_frames  = len(results),
            total_frames      = getattr(args, "_total_frames", len(results)),
            avg_health_score  = round(avg, 2),
            min_health_score  = round(min(scores, default=100), 2),
            max_health_score  = round(max(scores, default=100), 2),
            total_detections  = sum(r["n_detections"] for r in results),
            total_cost_inr    = round(sum(r["cost_inr"] for r in results), 2),
            flagged_frames    = [r["frame_index"] for r in results
                                 if r["health_score"] < 50],
            worst_frames      = worst_sorted,
            frame_images      = evidence_imgs,
            class_breakdown   = _class_breakdown(results),
            # Research sections
            fusion_report     = fusion_report or {},
            forecast          = forecast_data or {},
            depth_summary     = depth_summary or {},
            sor_breakdown     = sor_breakdown or {},
            uncertainty_summary = uncertainty_summary or {},  # NEW
        )
        sid_tag  = dash_sid or datetime.now().strftime("%Y%m%d_%H%M%S")
        pdf_path = str(out_dir / f"report_{sid_tag}.pdf")
        PDFReportGenerator().generate(rd, pdf_path)
        import shutil
        shutil.copy2(pdf_path, str(out_dir / "report.pdf"))
        log.info(f"PDF report: {pdf_path}")
        return pdf_path
    except Exception as e:
        log.warning(f"PDF generation failed: {e}")
        return None


def generate_excel(results, out_dir, dash_sid):
    """Excel report with per-frame data, cost breakdown, class analysis."""
    try:
        from excel_exporter import ExcelExporter
        try:
            from cost_estimator import SOR
        except ImportError:
            SOR = {}

        scores   = [r["health_score"] for r in results]
        avg      = sum(scores)/len(scores) if scores else 100.0
        tot_cost = sum(r["cost_inr"] for r in results)
        class_counts = _class_breakdown(results)

        min_urgency = 30
        for cls in class_counts:
            if cls in SOR:
                u = min(SOR[cls]["urgency_days"].values())
                min_urgency = min(min_urgency, u)

        summary = {
            "avg_health_score": round(avg, 2),
            "total_cost_inr":   round(tot_cost, 2),
            "total_detections": sum(r["n_detections"] for r in results),
            "condition":        ("critical" if avg < 40 else "poor" if avg < 60
                                 else "moderate" if avg < 80 else "good"),
            "urgency_days":     min_urgency,
            "budget_tier":      ("High" if tot_cost > 500_000
                                 else "Medium" if tot_cost > 100_000 else "Low"),
            "class_counts":     class_counts,
        }
        sid_tag = dash_sid or datetime.now().strftime("%Y%m%d_%H%M%S")
        xl_path = str(out_dir / f"report_{sid_tag}.xlsx")
        ExcelExporter().export(results, summary, xl_path)
        log.info(f"Excel report: {xl_path}")
        return xl_path
    except Exception as e:
        log.warning(f"Excel generation failed: {e}")
        return None


def generate_tickets(results, out_dir, dash_sid):
    """Repair ticket PDFs with QR codes, one per damage segment."""
    try:
        from ticket_generator import TicketGenerator, cluster_frames_into_segments
        segments = cluster_frames_into_segments(results)
        if not segments:
            log.info("No damage segments detected — no tickets generated")
            return []

        ticket_dir = out_dir / "tickets"
        ticket_dir.mkdir(exist_ok=True)
        paths = TicketGenerator().generate_all(segments, str(ticket_dir))

        # Combined summary PDF
        try:
            summary_path = str(out_dir / "tickets_summary.pdf")
            TicketGenerator().generate_summary(segments, summary_path)
            log.info(f"Ticket summary: {summary_path}")
        except Exception:
            pass

        log.info(f"Tickets: {len(paths)} PDFs → {ticket_dir}")
        return paths
    except Exception as e:
        log.warning(f"Ticket generation failed: {e}")
        return []


def generate_map(results, out_dir):
    """Interactive GPS heatmap HTML via run_mapping pipeline."""
    gps_results = [r for r in results if r.get("gps_lat") and r.get("gps_lon")]
    if not gps_results:
        log.info("No GPS data available — skipping map generation")
        return None
    try:
        from mapping import run_map_pipeline, MapPipelineConfig
        locations = [
            {
                "lat":            r["gps_lat"],
                "lon":            r["gps_lon"],
                "score":          r["health_score"],
                "label":          r.get("dominant_class", ""),
                "frame_index":    r["frame_index"],
                "n_detections":   r["n_detections"],
                "dominant_class": r.get("dominant_class", ""),
            }
            for r in gps_results
        ]
        cfg = MapPipelineConfig(
            locations      = locations,
            output_dir     = str(out_dir / "maps"),
            map_filename   = "road_health_map.html",
            title          = "Road-AI Health Map",
            cluster        = True,
            heatmap        = True,
            export_geojson = True,
            export_csv     = True,
        )
        outputs = run_map_pipeline(cfg)
        # Copy to output root for dashboard download endpoint
        map_src = out_dir / "maps" / "road_health_map.html"
        if map_src.exists():
            import shutil
            shutil.copy2(str(map_src), str(out_dir / "road_health_map.html"))
        log.info(f"Map outputs: {outputs}")
        return outputs
    except Exception as e:
        log.warning(f"Map generation failed: {e}")
        return None


def send_email_report(pdf_path, results, args):
    """Email the PDF report using EmailReporter with SOR-grounded cost summary."""
    if not (getattr(args, "email_to", None) and
            getattr(args, "smtp_from", None) and
            getattr(args, "smtp_pass", None)):
        return False
    try:
        from email_reporter import EmailReporter, EmailConfig
        from report_generator import ReportData
        scores   = [r["health_score"] for r in results]
        avg      = sum(scores)/len(scores) if scores else 100.0
        tot_cost = sum(r["cost_inr"] for r in results)
        rd = ReportData(
            avg_health_score = round(avg, 2),
            total_detections = sum(r["n_detections"] for r in results),
            total_cost_inr   = round(tot_cost, 2),
            class_breakdown  = _class_breakdown(results),
        )
        cfg = EmailConfig(
            smtp_host       = getattr(args, "smtp_host", "smtp.gmail.com"),
            smtp_port       = getattr(args, "smtp_port", 587),
            sender_email    = args.smtp_from,
            sender_password = args.smtp_pass,
        )
        ok = EmailReporter(cfg).send(
            to          = [args.email_to],
            report_data = rd,
            pdf_path    = pdf_path,
        )
        if ok:
            log.info(f"Email sent to {args.email_to}")
        else:
            log.warning("Email send failed — check credentials")
        return ok
    except Exception as e:
        log.warning(f"Email failed: {e}")
        return False


# =============================================================================
# NEW: Self-Supervised Learning Pipeline
# =============================================================================

def run_self_supervised_labeling(args):
    """
    Generate pseudo-labels from unlabeled video.
    Uses high-confidence detections from teacher model.
    """
    log.info("=" * 60)
    log.info("  SELF-SUPERVISED PSEUDO-LABELING MODE")
    log.info("=" * 60)
    
    try:
        # Import self-supervised trainer
        # (This would be in a new file: self_supervised.py based on Feature 4)
        import sys
        from pathlib import Path
        
        # Simple inline implementation for now
        class SimplePseudoLabeler:
            def __init__(self, teacher_model_path, confidence_threshold=0.85, output_dir="pseudo_labeled_data"):
                from ultralytics import YOLO
                self.teacher_model = YOLO(teacher_model_path)
                self.threshold = confidence_threshold
                self.output_dir = Path(output_dir)
                
                # Create dataset structure
                self.train_dir = self.output_dir / "train"
                self.train_images = self.train_dir / "images"
                self.train_labels = self.train_dir / "labels"
                
                for dir in [self.train_images, self.train_labels]:
                    dir.mkdir(exist_ok=True, parents=True)
                
                self.count = 0
            
            def process_video(self, video_path, frame_skip=30, max_frames=1000):
                cap = cv2.VideoCapture(str(video_path))
                frame_idx = 0
                
                while cap.isOpened() and self.count < max_frames:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Skip frames
                    if frame_idx % frame_skip != 0:
                        frame_idx += 1
                        continue
                    
                    # Run teacher model
                    results = self.teacher_model.predict(frame, conf=self.threshold, verbose=False)
                    
                    # Only keep high confidence
                    if len(results[0].boxes) > 0:
                        high_conf = [box for box in results[0].boxes if float(box.conf[0]) >= self.threshold]
                        
                        if high_conf:
                            # Save image and label
                            img_name = f"pseudo_{Path(video_path).stem}_frame_{frame_idx:06d}"
                            self._save_label(frame, high_conf, img_name)
                            self.count += 1
                            
                            if self.count % 100 == 0:
                                log.info(f"Generated {self.count} pseudo-labels...")
                    
                    frame_idx += 1
                
                cap.release()
                log.info(f"Pseudo-labeling complete: {self.count} frames")
                
                # Create data.yaml
                self._create_yaml()
                
                return self.count
            
            def _save_label(self, frame, boxes, img_name):
                import cv2
                
                # Save image
                img_path = self.train_images / f"{img_name}.jpg"
                cv2.imwrite(str(img_path), frame)
                
                # Save YOLO format label
                label_path = self.train_labels / f"{img_name}.txt"
                
                h, w = frame.shape[:2]
                
                with open(label_path, 'w') as f:
                    for box in boxes:
                        cls = int(box.cls.item())
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        
                        # YOLO format
                        x_center = ((x1 + x2) / 2) / w
                        y_center = ((y1 + y2) / 2) / h
                        box_width = (x2 - x1) / w
                        box_height = (y2 - y1) / h
                        
                        f.write(f"{cls} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}\n")
            
            def _create_yaml(self):
                import yaml
                
                yaml_content = {
                    'path': str(self.output_dir.absolute()),
                    'train': 'train/images',
                    'val': 'train/images',
                    'nc': 4,
                    'names': ['pothole', 'alligator_crack', 'transverse_crack', 'longitudinal_crack']
                }
                
                yaml_path = self.output_dir / 'data.yaml'
                
                with open(yaml_path, 'w') as f:
                    yaml.dump(yaml_content, f, default_flow_style=False)
                
                log.info(f"Created data.yaml: {yaml_path}")
        
        # Run pseudo-labeling
        output_dir = getattr(args, "ssl_output", "pseudo_labeled_data")
        labeler = SimplePseudoLabeler(
            teacher_model_path=args.model,
            confidence_threshold=getattr(args, "ssl_confidence", 0.85),
            output_dir=output_dir
        )
        
        count = labeler.process_video(
            args.video,
            frame_skip=getattr(args, "ssl_frame_skip", 30),
            max_frames=getattr(args, "ssl_max_frames", 1000)
        )
        
        print(f"\n{'='*60}")
        print(f"  PSEUDO-LABELING COMPLETE")
        print(f"  Generated: {count} labeled images")
        print(f"  Output: {output_dir}")
        print(f"  Ready to train with: {output_dir}/data.yaml")
        print(f"{'='*60}\n")
        
        return True
        
    except Exception as e:
        log.error(f"Self-supervised labeling failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# =============================================================================
# Main pipeline
# =============================================================================

def run(args):
    # NEW: Self-supervised mode (early exit)
    if getattr(args, "self_supervised", False):
        return run_self_supervised_labeling(args)
    
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Preprocessor ──────────────────────────────────────────────────────────
    from preprocessor import FramePreprocessor, ConfidenceCalibrator
    pre = FramePreprocessor(
        enable_dedup    = not getattr(args, "no_dedup",    False),
        enable_roi      = not getattr(args, "no_roi",      False),
        enable_adaptive = not getattr(args, "no_adaptive", False),
        default_skip    = args.skip,
    )
    calibrator = ConfidenceCalibrator()

    # ── Temporal fusion engine ─────────────────────────────────────────────────
    fusion_engine = None
    if not getattr(args, "no_fusion", False):
        try:
            from Temporal_fusion import TemporalFusionEngine
            fusion_engine = TemporalFusionEngine()
            log.info("Temporal fusion engine active (multi-frame severity consensus)")
        except ImportError:
            log.warning("temporal_fusion.py not found — single-frame scoring only")

    # ── Deterioration predictor ────────────────────────────────────────────────
    road_type = getattr(args, "road_type", "urban_mixed")
    try:
        from Deterioration_predictor import DeteriorationPredictor, forecast_session
        det_predictor = DeteriorationPredictor(road_type=road_type)
        log.info(f"Deterioration predictor active (road_type={road_type}, "
                 f"IRC:37-2018 / NHAI TC 2022)")
    except ImportError:
        det_predictor = None
        log.warning("deterioration_predictor.py not found — no forecast generated")
    log.info(
        f"Preprocessor  dedup={'ON' if pre.enable_dedup else 'OFF'} "
        f"roi={'ON' if pre.enable_roi else 'OFF'} "
        f"adaptive={'ON' if pre.enable_adaptive else 'OFF'}"
    )

    # ── Detection engine (NEW: with ensemble, uncertainty, conditions) ────────
    from detector import DetectionEngine
    
    # Build model configuration
    use_ensemble = getattr(args, "ensemble", False)
    enable_uncertainty = getattr(args, "uncertainty", False)
    mc_samples = getattr(args, "mc_samples", 10)
    
    # Environmental conditions (NEW)
    conditions = {
        'time': getattr(args, "condition_time", "day"),
        'weather': getattr(args, "condition_weather", "clear"),
        'fog': getattr(args, "condition_fog", False)
    }
    
    if use_ensemble:
        # Ensemble mode - build model config dict
        model_configs = {
            'general': {
                'path': args.model,
                'weight': 0.5,
                'specialization': 'general'
            }
        }
        
        if getattr(args, "model2", None):
            model_configs['model2'] = {
                'path': args.model2,
                'weight': 0.5,
                'specialization': 'general'
            }
        
        # Add more models if specified
        if getattr(args, "model_night", None):
            model_configs['night'] = {
                'path': args.model_night,
                'weight': 0.3,
                'specialization': 'night'
            }
        
        if getattr(args, "model_rain", None):
            model_configs['rain'] = {
                'path': args.model_rain,
                'weight': 0.3,
                'specialization': 'rain'
            }
        
        model_path_arg = model_configs
        log.info(f"Ensemble mode: {len(model_configs)} models")
        for name, cfg in model_configs.items():
            log.info(f"  - {name}: {cfg['path']} (weight: {cfg['weight']})")
    else:
        # Single model mode
        model_path_arg = args.model
        log.info(f"Single model mode: {args.model}")
    
    # Initialize detection engine with new features
    try:
        engine = DetectionEngine(
            model_path             = model_path_arg,
            conf                   = args.conf,
            use_ensemble           = use_ensemble,
            enable_measurements    = not getattr(args, "no_measurements",    False),
            enable_active_learning = not getattr(args, "no_active_learning", False),
            enable_uncertainty     = enable_uncertainty,  # NEW
            mc_samples             = mc_samples,           # NEW
            conditions             = conditions,           # NEW
        )
        log.info(f"Detection engine ready. Classes: {engine.cls_names}")
        
        # Log new features status
        if enable_uncertainty:
            log.info(f"  Uncertainty quantification: ON (MC samples: {mc_samples})")
        if use_ensemble:
            log.info(f"  Ensemble detection: ON (WBF fusion)")
        log.info(f"  Environmental conditions: {conditions}")
        
    except Exception as e:
        log.error(f"Model load failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # ── Physics depth estimator ────────────────────────────────────────────────
    depth_est = None
    if not getattr(args, "no_depth", False):
        try:
            from Depth_estimator import DepthEstimator
            depth_est = DepthEstimator(
                camera_height_m   = getattr(args, "camera_height", 1.2),
                sun_elevation_deg = getattr(args, "sun_elevation",  60.0),
            )
            log.info("Physics depth estimator active (shadow+brightness+edge, "
                     "Eriksson 2008 / Koch & Brilakis 2011)")
        except ImportError:
            log.warning("depth_estimator.py not found — shape heuristic used")

    # ── Crack mechanism classifier + pothole age estimator ────────────────────
    crack_classifier = None
    age_estimator    = None
    if not getattr(args, "no_damage_analysis", False):
        try:
            from Damage_analyzer import CrackMechanismClassifier, PotholeAgeEstimator
            crack_classifier = CrackMechanismClassifier()
            age_estimator    = PotholeAgeEstimator()
            log.info("Damage analyzer active (crack mechanism + pothole age)")
        except ImportError:
            log.warning("damage_analyzer.py not found")


    cost_est = None
    try:
        from cost_estimator import RepairCostEstimator
        cost_est = RepairCostEstimator(
            state       = getattr(args, "state", "Tamil Nadu"),
            include_gst = True,
        )
        state_name = getattr(args, "state", "Tamil Nadu")
        log.info(f"Cost estimator: {state_name} PWD SOR rates (incl. 18% GST)")
    except ImportError:
        log.warning("cost_estimator.py not found — simple cost estimate used")

    # ── RoadScorer ─────────────────────────────────────────────────────────────
    from scoring import RoadScorer

    # ── Road metrics (PCI, IRI, Budget, Work Orders) ───────────────────────────
    try:
        from Road_metrics import (compute_pci_for_session, allocate_budget,
                                   WorkOrderGenerator)
        wo_generator = WorkOrderGenerator(state=getattr(args,"state","Tamil Nadu"))
        log.info("Road metrics: PCI/IRI + budget allocator + work-order generator active")
    except ImportError:
        wo_generator = None
        log.warning("road_metrics.py not found")

    # ── Weather-aware confidence ───────────────────────────────────────────────
    try:
        from Advanced_features import weather_adjusted_thresholds
        _weather_conf_enabled = True
        log.info("Weather-aware confidence adjustment active")
    except ImportError:
        _weather_conf_enabled = False

    # ── GPS setup ──────────────────────────────────────────────────────────────
    gps_points  = []
    gps_start   = None
    gps_bearing = getattr(args, "gps_bearing", 90.0)

    if getattr(args, "gps_file", None):
        gps_points = load_gps_from_gpx(args.gps_file)
    elif getattr(args, "gps", None):
        gps_start = parse_gps_arg(args.gps)
        if gps_start:
            log.info(f"GPS dead-reckoning: start={gps_start}, bearing={gps_bearing}deg")
        else:
            log.warning(f"Bad --gps value: '{args.gps}'  Example: --gps 13.0827,80.2707")
    else:
        log.info("No GPS source. Use --gps lat,lon or --gps-file track.gpx")

    # ── Dashboard ──────────────────────────────────────────────────────────────
    dash     = Dashboard(args.port)
    use_dash = False
    if args.dashboard:
        if dash.ping():
            sid = dash.create_session(args.video)
            if sid:
                use_dash = True
                print(f"\n{'='*54}")
                print(f"  Dashboard connected!  Session: {sid}")
                print(f"  Open: http://localhost:{args.port}")
                print(f"{'='*54}\n")
                time.sleep(2.0)
        else:
            log.warning("Dashboard not reachable — run: python dashboard_server.py")

    # ══════════════════════════════════════════════════════════════════════════
    # NEW: Initialize Gamification, Blockchain, Accessibility
    # ══════════════════════════════════════════════════════════════════════════
    
    # ── Gamification Engine ───────────────────────────────────────────────────
    gamification = None
    if GAMIFICATION_AVAILABLE and not getattr(args, "no_gamification", False):
        try:
            gamification = GamificationEngine(
                db_path=str(out_dir / "gamification.db")
            )
            log.info("✅ Gamification engine active (points, badges, leaderboards)")
            
            # Create sample challenge if new database
            stats = gamification.get_stats()
            if stats['active_challenges'] == 0 and not is_stream:
                gamification.create_challenge(
                    title="Road Safety Week",
                    description="Report 100 potholes citywide this week",
                    goal_type="reports",
                    goal_value=100,
                    duration_days=7,
                    reward="₹10,000 donation to ward school"
                )
                log.info("Created sample challenge: Road Safety Week")
        except Exception as e:
            log.warning(f"Gamification initialization failed: {e}")
            gamification = None
    
    # ── Blockchain Tracker ────────────────────────────────────────────────────
    blockchain = None
    if BLOCKCHAIN_AVAILABLE and getattr(args, "blockchain", False):
        try:
            blockchain = SimpleBlockchain(
                db_path=str(out_dir / "blockchain.db")
            )
            log.info("✅ Blockchain tracker active (immutable audit trail)")
            
            # Verify chain integrity on startup
            integrity = blockchain.verify_chain_integrity()
            if integrity['valid']:
                log.info(f"Blockchain verified: {integrity['total_blocks']} blocks intact")
            else:
                log.warning(f"⚠️ Blockchain integrity issues: {len(integrity['issues'])} problems")
        except Exception as e:
            log.warning(f"Blockchain initialization failed: {e}")
            blockchain = None
    
    # ── Accessibility Mode ────────────────────────────────────────────────────
    accessibility = None
    if ACCESSIBILITY_AVAILABLE and getattr(args, "accessibility", False):
        try:
            accessibility = AccessibilityMode(
                default_language=getattr(args, "language", "en")
            )
            log.info(f"✅ Accessibility mode active (language: {getattr(args, 'language', 'en')})")
        except Exception as e:
            log.warning(f"Accessibility initialization failed: {e}")
            accessibility = None
    
    # ══════════════════════════════════════════════════════════════════════════

    # ── Open video / stream ────────────────────────────────────────────────────
    is_stream = getattr(args, "stream", False)
    video_src = args.video
    try:
        video_src = int(video_src)  # webcam index
        is_stream = True
    except (ValueError, TypeError):
        pass

    if is_stream:
        try:
            from stream_processor import VideoSource, StreamConfig
            vsrc  = VideoSource(StreamConfig(source=video_src))
            cap   = vsrc
            fps   = vsrc.fps()          or 30
            total = vsrc.total_frames() or 0
            W     = vsrc.width()        or 1280
            H     = vsrc.height()       or 720
            log.info(f"Stream source: {video_src}  {W}x{H} @ {fps:.1f}fps")
        except ImportError:
            cap   = cv2.VideoCapture(video_src)
            fps   = cap.get(cv2.CAP_PROP_FPS) or 30
            total = 0
            W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  or 1280
            H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
    else:
        cap = cv2.VideoCapture(str(video_src))
        if not cap.isOpened():
            log.error(f"Cannot open video: {video_src}")
            sys.exit(1)
        fps   = cap.get(cv2.CAP_PROP_FPS) or 30
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        log.info(f"Video: {W}x{H} @ {fps:.1f}fps  {total} frames")

    args._total_frames = total
    if H > W:
        log.warning(f"Portrait video ({W}x{H}) — tighter ROI applied automatically")

    # Sync frame dimensions to all estimators
    engine.estimator.frame_w = W
    engine.estimator.frame_h = H
    if cost_est:
        cost_est.frame_w = W
        cost_est.frame_h = H
    if depth_est:
        depth_est.frame_w = W
        depth_est.frame_h = H
    scorer = RoadScorer(frame_w=W, frame_h=H)

    # ── Video writer ───────────────────────────────────────────────────────────
    writer = cv2.VideoWriter(
        str(out_dir / "annotated_output.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps, (W, H),
    )

    # ── Main processing loop ───────────────────────────────────────────────────
    results          = []
    worst_frame_imgs = {}
    min_area         = getattr(args, "min_area", 0.003)
    speed_kmh        = 25.0
    t0 = time.time()
    fi = 0
    
    # NEW: Uncertainty tracking
    uncertainty_stats = {
        'total_detections': 0,
        'high_uncertainty': 0,
        'needs_review': 0,
        'avg_uncertainty': 0.0,
        'max_uncertainty': 0.0
    }

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            ts = fi / max(fps, 1)

            # GPS position for this frame
            gps = None
            if gps_points:
                idx = min(int(fi / max(total, 1) * len(gps_points)),
                          len(gps_points) - 1)
                gps = gps_points[idx]
            elif gps_start:
                gps = simulate_gps(gps_start[0], gps_start[1], fi, fps,
                                   speed_kmh=speed_kmh, bearing=gps_bearing)

            # ── Preprocessing ────────────────────────────────────────────────
            prep = pre.process(frame, speed_kmh=speed_kmh)

            if prep.is_duplicate:
                writer.write(frame); fi += 1; continue
            if not prep.sample_this:
                writer.write(frame); fi += 1; continue

            condition_label = prep.condition.name

            # ── Detection (with conditions for adaptive ensemble) ────────────
            det_result = engine.run(
                prep.roi_frame,
                frame_index = fi,
                roi_bbox    = prep.roi_bbox if pre.enable_roi else None,
                conditions  = conditions,  # NEW: pass conditions
            )
            dets = [d.to_dict() for d in det_result.detections]

            # Confidence calibration (per-class thresholds + temporal boost)
            dets = calibrator.calibrate(dets, frame_index=fi)

            # False-positive filter — weather-aware thresholds
            _w_conf, _w_area = (
                weather_adjusted_thresholds(condition_label.lower(),
                                            args.conf, min_area)
                if _weather_conf_enabled else (args.conf, min_area)
            )
            dets = filter_detections(dets, W, H,
                                     min_area_ratio=_w_area,
                                     max_area_ratio=0.55)

            # ── Temporal fusion (multi-frame consensus) ──────────────────────
            if fusion_engine is not None:
                dets = fusion_engine.update(fi, dets)
                # Suppress unconfirmed low-confidence detections
                if fi > 10:   # give tracker time to warm up
                    dets = [d for d in dets
                            if d.get("is_confirmed", True)
                            or d.get("confidence", 0) > 0.7]

            # ══════════════════════════════════════════════════════════════════
            # NEW: Blockchain Recording
            # ══════════════════════════════════════════════════════════════════
            if blockchain and dets and gps:
                try:
                    # Record most severe detection on blockchain
                    worst_det = max(dets, key=lambda d: d.get("severity", 0))
                    
                    # Calculate photo hash
                    import hashlib
                    photo_hash = hashlib.sha256(
                        f"frame_{fi}_{worst_det.get('class_name', '')}".encode()
                    ).hexdigest()
                    
                    blockchain_record = blockchain.record_detection(
                        gps=gps,
                        severity=worst_det.get("severity_class", "moderate"),
                        depth_cm=worst_det.get("depth_cm", 0),
                        photo_hash=photo_hash,
                        reporter_id=getattr(args, 'reporter_id', 'system_auto'),
                        metadata={
                            'frame_index': fi,
                            'confidence': worst_det.get("confidence", 0),
                            'class_name': worst_det.get("class_name", ""),
                            'cost_inr': worst_det.get("cost_inr", 0)
                        }
                    )
                    
                    # Add blockchain info to detection
                    worst_det['blockchain_id'] = blockchain_record['record_id']
                    worst_det['block_number'] = blockchain_record['block_number']
                    
                    if fi % 100 == 0:
                        log.debug(f"Blockchain: {blockchain_record['record_id']}")
                    
                except Exception as e:
                    if fi % 100 == 0:
                        log.debug(f"Blockchain recording failed: {e}")
            
            # ══════════════════════════════════════════════════════════════════
            # NEW: Gamification Points (for citizen reports)
            # ══════════════════════════════════════════════════════════════════
            if gamification and getattr(args, 'citizen_mode', False):
                try:
                    reporter_id = getattr(args, 'reporter_id', 'anonymous')
                    
                    # Award points for detection
                    if dets:
                        metadata = {
                            'severity': 'critical' if score < 50 else 'high' if score < 70 else 'moderate',
                            'time_of_day': 'night' if 'night' in condition_label.lower() else 'day',
                            'weather': 'rain' if 'rain' in condition_label.lower() else 'clear'
                        }
                        
                        # Register user if first detection
                        if fi == 1:
                            profile = gamification.get_user_profile(reporter_id)
                            if not profile:
                                gamification.register_user(
                                    user_id=reporter_id,
                                    name=reporter_id.replace('_', ' ').title(),
                                    ward=getattr(args, 'ward', None)
                                )
                        
                        # Award points
                        award = gamification.award_points(
                            user_id=reporter_id,
                            action='report_submitted',
                            metadata=metadata
                        )
                        
                        if fi % 100 == 0:
                            log.info(f"🎮 Awarded {award['points_earned']} points to {reporter_id}")
                        
                        # Check for new badges
                        if award['new_badges']:
                            for badge in award['new_badges']:
                                log.info(f"🏆 Badge unlocked: {badge['name']} (+{badge['points_reward']} pts)")
                    
                except Exception as e:
                    if fi % 100 == 0:
                        log.debug(f"Gamification award failed: {e}")
            
            # ══════════════════════════════════════════════════════════════════
            # NEW: Audio Accessibility (for real-time navigation)
            # ══════════════════════════════════════════════════════════════════
            if accessibility and getattr(args, 'audio_alerts', False):
                try:
                    if dets:
                        # Get most severe detection
                        worst_det = max(dets, key=lambda d: d.get("severity", 0))
                        
                        # Generate audio description for severe damage
                        if worst_det.get("severity", 0) > 70:
                            detection_dict = {
                                'class_name': worst_det.get("class_name", ""),
                                'severity': worst_det.get("severity", 0),
                                'severity_class': worst_det.get("severity_class", ""),
                                'depth_cm': worst_det.get("depth_cm", 0),
                                'width_cm': worst_det.get("width_cm", 0),
                                'distance_m': 20,  # Approximate
                                'lane_position': 'center'
                            }
                            
                            audio_desc = accessibility.audio_damage_description(
                                detection=detection_dict,
                                language=getattr(args, 'language', 'en'),
                                detailed=False  # Brief for real-time
                            )
                            
                            # Speak alert (non-blocking)
                            accessibility.speak_text(audio_desc.text)
                            
                            if fi % 100 == 0:
                                log.info(f"🔊 Audio: {audio_desc.text[:50]}...")
                
                except Exception as e:
                    if fi % 100 == 0:
                        log.debug(f"Audio alert failed: {e}")
            
            # ══════════════════════════════════════════════════════════════════

            # ── NEW: Collect uncertainty statistics ──────────────────────────
            if enable_uncertainty and dets:
                for d in dets:
                    unc = d.get("uncertainty", 0)
                    uncertainty_stats['total_detections'] += 1
                    uncertainty_stats['avg_uncertainty'] += unc
                    uncertainty_stats['max_uncertainty'] = max(uncertainty_stats['max_uncertainty'], unc)
                    
                    if unc > 0.4:
                        uncertainty_stats['high_uncertainty'] += 1
                    
                    if d.get("needs_review", False):
                        uncertainty_stats['needs_review'] += 1

            # ── Physics depth estimation ─────────────────────────────────────
            depth_info = None
            if depth_est and dets:
                for d in dets:
                    cls = d.get("class_name", "")
                    if cls in ("Pothole", "Potholes", "Alligator Crack"):
                        try:
                            de = depth_est.estimate(prep.frame, d["bbox"], cls)
                            d["depth_cm"]         = de.depth_cm
                            d["depth_low_cm"]     = de.depth_low_cm
                            d["depth_high_cm"]    = de.depth_high_cm
                            d["depth_confidence"] = de.confidence
                            d["severity_class"]   = de.severity_class
                            d["depth_urgency_days"] = de.urgency_days
                            if depth_info is None:
                                depth_info = (
                                    f"{de.depth_cm:.1f}cm "
                                    f"[{de.severity_class}] "
                                    f"CI:{de.depth_low_cm:.1f}-{de.depth_high_cm:.1f}"
                                )
                        except Exception:
                            pass

            # ── Crack mechanism + pothole age analysis ────────────────────────
            if dets and (crack_classifier or age_estimator):
                for d in dets:
                    cls = d.get("class_name", "")
                    try:
                        if crack_classifier and "crack" in cls.lower():
                            mech = crack_classifier.classify(
                                prep.frame, d["bbox"], cls, W, H)
                            d["crack_mechanism"]       = mech.mechanism
                            d["crack_mechanism_label"] = mech.mechanism_label
                            d["crack_repair_method"]   = mech.repair_method
                            d["mechanism_confidence"]  = mech.confidence
                    except Exception:
                        pass
                    try:
                        if age_estimator and "pothole" in cls.lower():
                            age = age_estimator.estimate(prep.frame, d["bbox"], cls)
                            d["age_days"]         = age.age_days_estimate
                            d["age_days_low"]     = age.age_days_low
                            d["age_days_high"]    = age.age_days_high
                            d["age_category"]     = age.age_category
                            d["sla_breach"]       = age.sla_breach
                            d["age_confidence"]   = age.confidence
                    except Exception:
                        pass


            if cost_est and dets:
                try:
                    fe = cost_est.estimate_frame(dets, frame_index=fi)
                    cost = fe.total_cost
                    for det_cost, d in zip(fe.detections, dets):
                        d["cost_inr"]      = det_cost.total_cost_inr
                        d["repair_method"] = det_cost.method
                        d["sor_source"]    = det_cost.source
                        d["severity_tier"] = det_cost.severity_tier
                        d["urgency_days"]  = det_cost.urgency_days
                except Exception:
                    cost = sum(d.get("cost_inr", 0) for d in dets)
            else:
                cost = sum(d.get("cost_inr", 0) for d in dets)

            # ── Scoring ──────────────────────────────────────────────────────
            score_result = scorer.score_frame(dets)
            score  = score_result.health_score
            dom    = (max(dets, key=lambda d: d.get("severity", 0))["class_name"]
                      if dets else "")

            # ── NEW: Prepare uncertainty info for HUD ─────────────────────────
            uncertainty_info = None
            if enable_uncertainty and dets:
                avg_unc = sum(d.get("uncertainty", 0) for d in dets) / len(dets)
                max_unc = max(d.get("uncertainty", 0) for d in dets)
                needs_rev = sum(1 for d in dets if d.get("needs_review", False))
                uncertainty_info = f"Avg:{avg_unc:.2f} Max:{max_unc:.2f} Rev:{needs_rev}"

            # ── Annotate & write frame ────────────────────────────────────────
            ann_frame = annotate(
                prep.frame.copy(), dets, score, cost, fi,
                gps             = gps,
                condition_label = condition_label,
                measurements    = not getattr(args, "no_measurements", False),
                depth_info      = depth_info,
                uncertainty_info = uncertainty_info,  # NEW
            )
            writer.write(ann_frame)

            # ── Capture worst frames for PDF photo evidence ───────────────────
            if dets and score < 85:
                _, jpg = cv2.imencode(".jpg", ann_frame,
                                      [cv2.IMWRITE_JPEG_QUALITY, 82])
                worst_frame_imgs[fi] = jpg.tobytes()
                if len(worst_frame_imgs) > 20:
                    worst_frame_imgs = dict(
                        sorted(worst_frame_imgs.items(),
                               key=lambda kv: next(
                                   (r["health_score"] for r in results
                                    if r["frame_index"] == kv[0]), 100))[:20]
                    )

            # ── Frame record ──────────────────────────────────────────────────
            fd = {
                "frame_index":     fi,
                "timestamp_sec":   round(ts, 2),
                "n_detections":    len(dets),
                "health_score":    score,
                "dominant_class":  dom,
                "cost_inr":        round(cost, 0),
                "condition":       condition_label.lower(),
                "gps_lat":         gps[0] if gps else None,
                "gps_lon":         gps[1] if gps else None,
                "speed_kmh":       speed_kmh,
                "max_width_cm":    max((d.get("width_cm",  0) for d in dets), default=0),
                "max_depth_cm":    max((d.get("depth_cm",  0) for d in dets), default=0),
                "max_severity":    max((d.get("severity",  0) for d in dets), default=0),
                "uncertain_count": det_result.uncertain_count,
                "needs_review_count": det_result.needs_review_count,  # NEW
                "max_uncertainty": max((d.get("uncertainty", 0) for d in dets), default=0),  # NEW
                "enhanced":        prep.enhanced,
                "detections":      dets,
            }
            results.append(fd)

            if use_dash:
                dash.push(fd, sync=(fi < 10))

            # ── Progress bar ──────────────────────────────────────────────────
            if total > 0:
                pct = fi / total
                bar = "█" * int(pct*28) + "░" * (28 - int(pct*28))
                eta = (time.time()-t0) / max(fi,1) * (total-fi)
                
                # NEW: Add uncertainty info to progress
                unc_str = ""
                if enable_uncertainty and uncertainty_stats['total_detections'] > 0:
                    unc_str = f"unc={uncertainty_stats['needs_review']}  "
                
                print(f"\r  [{bar}] {pct:5.1%}  det={len(dets)} "
                      f"score={score:.0f}  {unc_str}cond={condition_label[:3]}  "
                      f"ETA {eta:.0f}s  ",
                      end="", flush=True)
            fi += 1

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        if hasattr(cap, "release"):
            cap.release()
        writer.release()
        if use_dash:
            dash.done()

    # ==========================================================================
    # Post-processing summary & output generation
    # ==========================================================================

    elapsed  = time.time() - t0
    scores   = [r["health_score"] for r in results]
    avg      = sum(scores)/len(scores) if scores else 100.0
    tot_det  = sum(r["n_detections"] for r in results)
    tot_cost = sum(r["cost_inr"] for r in results)
    flagged  = sum(1 for s in scores if s < 50)
    pre_stats = pre.stats()

    print(f"\n\n{'='*60}")
    print(f"  ROAD-AI  COMPLETE")
    print(f"  Frames processed : {len(results)} / {fi} total")
    print(f"  Skipped (dedup)  : {pre_stats['duplicates']}  ({pre_stats['dup_rate_pct']}%)")
    print(f"  Skipped (speed)  : {pre_stats['skipped_speed']}  ({pre_stats['skip_rate_pct']}%)")
    print(f"  Enhanced frames  : {pre_stats['enhanced']}  ({pre_stats['enhance_rate_pct']}%)")
    print(f"  Avg Health Score : {avg:.1f}/100")
    print(f"  Total Detections : {tot_det}")
    print(f"  Flagged (<50)    : {flagged}")
    state_name = getattr(args, "state", "Tamil Nadu")
    print(f"  Est. Repair Cost : Rs.{tot_cost:,.0f}  (incl. 18% GST, {state_name} PWD SOR 2023-24)")
    
    # NEW: Uncertainty statistics
    if enable_uncertainty and uncertainty_stats['total_detections'] > 0:
        avg_unc = uncertainty_stats['avg_uncertainty'] / uncertainty_stats['total_detections']
        print(f"  Uncertainty stats:")
        print(f"    - Average      : {avg_unc:.3f}")
        print(f"    - Maximum      : {uncertainty_stats['max_uncertainty']:.3f}")
        print(f"    - High (>0.4)  : {uncertainty_stats['high_uncertainty']} ({uncertainty_stats['high_uncertainty']/uncertainty_stats['total_detections']*100:.1f}%)")
        print(f"    - Needs review : {uncertainty_stats['needs_review']} ({uncertainty_stats['needs_review']/uncertainty_stats['total_detections']*100:.1f}%)")
    
    al = engine.active_learning_stats()
    if al:
        print(f"  Review queue     : {al.get('review_queue_size', 0)} frames "
              f"({al.get('pending', 0)} pending)")
    print(f"  Time elapsed     : {elapsed:.1f}s  ({elapsed/max(fi,1)*1000:.1f}ms/frame)")
    print(f"{'='*60}")

    # Raw session JSON (always saved)
    session_json = out_dir / "session_data.json"
    session_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    log.info(f"Session data: {session_json}")

    # ══════════════════════════════════════════════════════════════════════════
    # NEW: Gamification, Blockchain, Accessibility Exports
    # ══════════════════════════════════════════════════════════════════════════
    
    # ── Export gamification data ──────────────────────────────────────────────
    if gamification:
        try:
            # Get leaderboard
            leaderboard = gamification.get_leaderboard(scope='global', limit=100)
            (out_dir / "leaderboard.json").write_text(
                json.dumps(leaderboard, indent=2), encoding="utf-8"
            )
            
            # Get active challenges
            challenges = gamification.get_active_challenges()
            (out_dir / "challenges.json").write_text(
                json.dumps(challenges, indent=2), encoding="utf-8"
            )
            
            # Get stats
            gam_stats = gamification.get_stats()
            
            print(f"  Gamification     : {gam_stats['total_users']} users, "
                  f"{gam_stats['total_points_awarded']:,} points awarded")
            
            log.info(f"Gamification data exported")
            
        except Exception as e:
            log.debug(f"Gamification export failed: {e}")
    
    # ── Export blockchain data ────────────────────────────────────────────────
    if blockchain:
        try:
            # Verify integrity
            integrity = blockchain.verify_chain_integrity()
            (out_dir / "blockchain_integrity.json").write_text(
                json.dumps(integrity, indent=2), encoding="utf-8"
            )
            
            # Export public explorer data
            explorer_data = blockchain.get_public_explorer_data()
            (out_dir / "blockchain_explorer.json").write_text(
                json.dumps(explorer_data, indent=2), encoding="utf-8"
            )
            
            print(f"  Blockchain       : {explorer_data['statistics']['total_detections']} records, "
                  f"₹{explorer_data['statistics']['total_allocated_inr']:,.0f} in contracts")
            
            if not integrity['valid']:
                print(f"  ⚠️  Blockchain integrity issues: {len(integrity['issues'])}")
            
            log.info(f"Blockchain data exported")
            
        except Exception as e:
            log.debug(f"Blockchain export failed: {e}")
    
    # ── Export accessibility resources ────────────────────────────────────────
    if accessibility:
        try:
            # Export keyboard shortcuts
            shortcuts = accessibility.get_keyboard_shortcuts()
            (out_dir / "keyboard_shortcuts.json").write_text(
                json.dumps(shortcuts, indent=2), encoding="utf-8"
            )
            
            # Export ARIA labels
            aria_labels = accessibility.get_aria_labels()
            (out_dir / "aria_labels.json").write_text(
                json.dumps(aria_labels, indent=2), encoding="utf-8"
            )
            
            # Generate high contrast CSS
            high_contrast = accessibility.get_high_contrast_theme()
            css_content = ":root.high-contrast {\n"
            for key, value in high_contrast.items():
                css_content += f"  {key}: {value};\n"
            css_content += "}\n\n"
            css_content += accessibility.get_large_text_css()
            
            (out_dir / "accessibility.css").write_text(css_content, encoding="utf-8")
            
            print(f"  Accessibility    : {len(shortcuts)} shortcuts, "
                  f"{len(aria_labels)} ARIA labels exported")
            
            log.info(f"Accessibility resources exported")
            
        except Exception as e:
            log.debug(f"Accessibility export failed: {e}")
    
    # ══════════════════════════════════════════════════════════════════════════

    # Active learning review queue (NEW: enhanced export)
    if getattr(args, "active_learning_export", False):
        queue_path = engine.export_review_queue(str(out_dir / "review_queue_export.json"))
        if queue_path:
            log.info(f"Active learning export: {queue_path}")
    else:
        engine.export_review_queue(str(out_dir / "review_queue.json"))

    fc_dict = {}  # deterioration forecast dict, set below if predictor ran

    # ── Temporal fusion report ─────────────────────────────────────────────────
    fusion_report = {}
    if fusion_engine is not None:
        fusion_report = fusion_engine.track_report()
        (out_dir / "temporal_fusion_report.json").write_text(
            json.dumps(fusion_report, indent=2), encoding="utf-8")
        stats = fusion_engine.stats()
        print(f"  Temporal fusion  : {stats['confirmed']} confirmed tracks, "
              f"{stats['suppressed_fp']} false positives suppressed, "
              f"avg depth {stats['avg_depth_cm']} cm")

    # ── Deterioration forecast ─────────────────────────────────────────────────
    if det_predictor is not None and results:
        fc = det_predictor.forecast(avg, tot_cost)
        fc_dict = fc.to_dict()
        (out_dir / "deterioration_forecast.json").write_text(
            json.dumps(fc_dict, indent=2), encoding="utf-8")
        print(f"  Deterioration    : {fc.urgency_label}")
        print(f"  Forecast 3m/6m   : {fc.score_3_months:.1f} / {fc.score_6_months:.1f}")
        print(f"  Cost if 6m defer : Rs.{fc.cost_if_deferred_6m:,.0f} "
              f"(+{(fc.cost_if_deferred_6m/max(tot_cost,1)-1)*100:.0f}%)")

    dash_sid = dash.sid if (use_dash and dash.sid) else None

    # ── PDF (default ON) ───────────────────────────────────────────────────────
    pdf_path = None
    if getattr(args, "pdf", True):
        # Gather research data for PDF sections
        _fc_dict = fc_dict if (det_predictor is not None and results) else {}
        # Build depth summary from results
        all_depths = [d for r in results for d in r.get('detections',[])
                      if d.get('depth_cm',0)>0]
        _depth_sum = {}
        if all_depths:
            depths_cm = [d['depth_cm'] for d in all_depths]
            _depth_sum = {
                'count':         len(all_depths),
                'avg_cm':        round(sum(depths_cm)/len(depths_cm),1),
                'max_cm':        round(max(depths_cm),1),
                'min_cm':        round(min(depths_cm),1),
                'severe_count':  sum(1 for d in all_depths if d.get('severity_class')=='severe'),
                'deep_count':    sum(1 for d in all_depths if d.get('severity_class') in ('deep','severe')),
                'moderate_count':sum(1 for d in all_depths if d.get('severity_class')=='moderate'),
            }
        # Build SOR breakdown from results
        _sor = {}
        by_class: dict = {}
        citations: dict = {}
        methods: dict = {}
        urgencies: dict = {}
        for r in results:
            for d in r.get('detections',[]):
                cls = d.get('class_name','')
                if cls:
                    by_class[cls] = by_class.get(cls,0) + d.get('cost_inr',0)
                    if d.get('sor_source'): citations[cls] = d['sor_source']
                    if d.get('repair_method'): methods[cls] = d['repair_method']
                    if d.get('urgency_days') is not None:
                        urgencies[cls] = min(urgencies.get(cls,999), d['urgency_days'])
        if by_class:
            _sor = {'by_class':by_class,'citations':citations,'methods':methods,
                    'urgencies':urgencies,'total_incl_gst':round(sum(by_class.values()),0),
                    'note':'TN-PWD SOR 2023-24 / NHAI SOR 2022. Incl. 18% GST.'}
        
        # NEW: Build uncertainty summary
        _unc_sum = {}
        if enable_uncertainty and uncertainty_stats['total_detections'] > 0:
            avg_unc = uncertainty_stats['avg_uncertainty'] / uncertainty_stats['total_detections']
            _unc_sum = {
                'total_detections': uncertainty_stats['total_detections'],
                'avg_uncertainty': round(avg_unc, 3),
                'max_uncertainty': round(uncertainty_stats['max_uncertainty'], 3),
                'high_uncertainty_count': uncertainty_stats['high_uncertainty'],
                'needs_review_count': uncertainty_stats['needs_review'],
                'high_uncertainty_pct': round(uncertainty_stats['high_uncertainty']/uncertainty_stats['total_detections']*100, 1),
                'needs_review_pct': round(uncertainty_stats['needs_review']/uncertainty_stats['total_detections']*100, 1),
                'note': f'Monte Carlo Dropout with {mc_samples} samples'
            }

        pdf_path = generate_pdf(results, args, out_dir, dash_sid, worst_frame_imgs,
                                fusion_report=fusion_report,
                                forecast_data=_fc_dict,
                                depth_summary=_depth_sum,
                                sor_breakdown=_sor,
                                uncertainty_summary=_unc_sum)  # NEW

    # ── Excel ─────────────────────────────────────────────────────────────────
    if getattr(args, "excel", False):
        generate_excel(results, out_dir, dash_sid)

    # ── Repair tickets with QR codes ──────────────────────────────────────────
    if getattr(args, "tickets", False):
        generate_tickets(results, out_dir, dash_sid)

    # ── Interactive GPS heatmap ───────────────────────────────────────────────
    if getattr(args, "map", False):
        generate_map(results, out_dir)

    # ── Email report ──────────────────────────────────────────────────────────
    if getattr(args, "email_to", None):
        send_email_report(pdf_path, results, args)

    # ── PCI / IRI metrics ─────────────────────────────────────────────────────
    if wo_generator is not None:
        try:
            from Road_metrics import compute_pci_for_session
            segments_pci = compute_pci_for_session(results, segment_size=50)
            (out_dir / "pci_segments.json").write_text(
                json.dumps(segments_pci, indent=2), encoding="utf-8")
            if segments_pci:
                worst_pci = segments_pci[0]
                print(f"  PCI (worst seg)  : {worst_pci['pci']:.1f} [{worst_pci['pci_band']}]"
                      f"  IRI: {worst_pci['iri']:.1f} m/km [{worst_pci['iri_band']}]")
            log.info(f"PCI segments: {len(segments_pci)} → pci_segments.json")
        except Exception as e:
            log.debug(f"PCI computation failed: {e}")

    # ── Budget allocation ─────────────────────────────────────────────────────
    budget_cap = getattr(args, "budget", None)
    if wo_generator is not None and budget_cap and 'segments_pci' in dir():
        try:
            from Road_metrics import allocate_budget
            budget_result = allocate_budget(
                segments_pci, budget_cap,
                strategy=getattr(args, "budget_strategy", "urgency_first"))
            alloc_path = out_dir / "budget_allocation.json"
            alloc_path.write_text(
                json.dumps({
                    "budget_inr":       budget_result.budget_inr,
                    "allocated_inr":    budget_result.allocated_inr,
                    "unallocated_inr":  budget_result.unallocated_inr,
                    "coverage_pct":     budget_result.coverage_pct,
                    "urgency_coverage": budget_result.urgency_coverage,
                    "selected":         budget_result.selected_segments,
                    "deferred":         budget_result.deferred_segments,
                }, indent=2), encoding="utf-8")
            print(f"  Budget Rs.{budget_cap:,.0f}: {len(budget_result.selected_segments)} segs funded "
                  f"({budget_result.coverage_pct:.0f}% cost coverage, "
                  f"{budget_result.urgency_coverage:.0f}% urgent segs covered)")
        except Exception as e:
            log.debug(f"Budget allocation failed: {e}")

    # ── Contractor work orders ─────────────────────────────────────────────────
    if wo_generator is not None and getattr(args, "work_orders", False):
        try:
            from Road_metrics import compute_pci_for_session
            if 'segments_pci' not in dir():
                segments_pci = compute_pci_for_session(results, segment_size=50)
            # Generate work order for each critical segment (PCI < 55)
            critical = [s for s in segments_pci if s.get("pci", 100) < 55][:10]
            if critical:
                # Build segment dicts with required fields
                wo_segs = []
                for seg in critical:
                    fi = seg.get("frame_start", 0)
                    frame_dets = next(
                        (r.get("detections",[]) for r in results
                         if r["frame_index"] == fi), [])
                    dom = (max(frame_dets, key=lambda d: d.get("severity",0))
                           .get("class_name","Pothole") if frame_dets else "Pothole")
                    wo_segs.append({**seg, "dominant_class": dom,
                                    "max_depth_cm": seg.get("max_depth_cm",5)})
                orders = [wo_generator.generate(s) for s in wo_segs]
                wo_pdf = str(out_dir / "work_orders.pdf")
                wo_generator.generate_pdf(orders, wo_pdf)
                log.info(f"Work orders: {len(orders)} PDFs → {wo_pdf}")
                print(f"  Work orders      : {len(orders)} critical segments → work_orders.pdf")
        except Exception as e:
            log.debug(f"Work order generation failed: {e}")

    # ── Timelapse video ────────────────────────────────────────────────────────
    if getattr(args, "timelapse", False):
        try:
            from Advanced_features import generate_timelapse
            input_vid = str(out_dir / "annotated_output.mp4")
            speed     = getattr(args, "timelapse_speed", 8)
            tl_path   = generate_timelapse(input_vid, speed_factor=speed,
                                           results=results)
            if tl_path:
                print(f"  Timelapse ({speed}×)   : {Path(tl_path).name}")
        except Exception as e:
            log.debug(f"Timelapse failed: {e}")

    # ── Auto fine-tune check ───────────────────────────────────────────────────
    try:
        from Advanced_features import check_finetune_trigger
        ft = check_finetune_trigger(
            str(out_dir / "review_queue.json"),
            model_path=args.model,
        )
        if ft.should_finetune:
            print(f"  Fine-tune ready  : {ft.reason}")
            print(f"  Command: {ft.command[:80]}...")
            (out_dir / "finetune_command.txt").write_text(
                ft.reason + "\n\n" + ft.command, encoding="utf-8")
        else:
            log.debug(f"Fine-tune not ready: {ft.reason}")
    except Exception as e:
        log.debug(f"Fine-tune check failed: {e}")


    # ── IEEE abstract ──────────────────────────────────────────────────────────
    try:
        from Ieee_abstract import AbstractData, generate_abstract, generate_abstract_pdf
        abs_data = AbstractData.from_session(
            results, fusion_report, fc_dict,
            location    = getattr(args, "location", "Chennai, Tamil Nadu"),
            authors     = getattr(args, "authors",  "Road-AI Research Team"),
            institution = getattr(args, "institution",
                                  "IIT Madras Road Health Showcase 2025"),
        )
        # Save plain text abstract
        abstract_txt = out_dir / "ieee_abstract.txt"
        abstract_txt.write_text(generate_abstract(abs_data), encoding="utf-8")
        # Save PDF abstract
        abstract_pdf = str(out_dir / "ieee_abstract.pdf")
        result = generate_abstract_pdf(abs_data, abstract_pdf)
        if result:
            log.info(f"IEEE abstract: {abstract_pdf}")
    except Exception as e:
        log.debug(f"IEEE abstract skipped: {e}")

    # ── Final output listing ──────────────────────────────────────────────────
    print(f"\n  Outputs in: {out_dir.resolve()}")
    for f in sorted(out_dir.rglob("*")):
        if f.is_file():
            rel = f.relative_to(out_dir)
            print(f"    {rel}  ({f.stat().st_size // 1024} KB)")
    print()


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Road-AI — Road damage detection and analysis pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Input
    p.add_argument("--video",   required=True,
                   help="Video file, RTSP URL, or webcam index")
    p.add_argument("--stream",  action="store_true",
                   help="Force stream mode (RTSP / webcam)")
    p.add_argument("--model",   default="best.pt")
    p.add_argument("--model2",  default=None,
                   help="Second YOLOv8 model for ensemble detection")
    
    # NEW: Additional ensemble models
    p.add_argument("--model-night", default=None, dest="model_night",
                   help="Specialized model for night conditions")
    p.add_argument("--model-rain", default=None, dest="model_rain",
                   help="Specialized model for rainy conditions")

    # GPS
    p.add_argument("--gps",         default=None,
                   help="Start GPS: 'lat,lon'  e.g. --gps 13.0827,80.2707")
    p.add_argument("--gps-file",    default=None, dest="gps_file",
                   help="GPX file path for GPS track")
    p.add_argument("--gps-bearing", default=90.0, type=float, dest="gps_bearing",
                   help="Travel bearing degrees (0=N 90=E 180=S 270=W). Default: 90")

    # Output
    p.add_argument("--output",  default="output")
    p.add_argument("--pdf",     action="store_true", default=True)
    p.add_argument("--no-pdf",  action="store_false", dest="pdf")
    p.add_argument("--excel",   action="store_true",
                   help="Generate Excel (.xlsx) report")
    p.add_argument("--tickets", action="store_true",
                   help="Generate repair ticket PDFs with QR codes")
    p.add_argument("--map",     action="store_true",
                   help="Generate interactive GPS heatmap HTML")

    # Dashboard
    p.add_argument("--dashboard", action="store_true")
    p.add_argument("--port",      default=8000, type=int)

    # Email
    p.add_argument("--email-to",  default=None, dest="email_to",
                   help="Recipient email — triggers auto-send after analysis")
    p.add_argument("--smtp-from", default=None, dest="smtp_from")
    p.add_argument("--smtp-pass", default=None, dest="smtp_pass",
                   help="SMTP App Password (Gmail 16-char app password)")
    p.add_argument("--smtp-host", default="smtp.gmail.com", dest="smtp_host")
    p.add_argument("--smtp-port", default=587, type=int,   dest="smtp_port")

    # Detection
    p.add_argument("--conf",      default=0.50, type=float)
    p.add_argument("--skip",      default=4,    type=int)
    p.add_argument("--min-area",  default=0.003, type=float, dest="min_area")

    # NEW: Ensemble & Uncertainty features
    p.add_argument("--ensemble", action="store_true",
                   help="Enable ensemble detection (uses --model + --model2)")
    p.add_argument("--uncertainty", action="store_true",
                   help="Enable uncertainty quantification (Monte Carlo Dropout)")
    p.add_argument("--mc-samples", default=10, type=int, dest="mc_samples",
                   help="Number of MC dropout samples for uncertainty (default: 10)")
    
    # NEW: Environmental conditions for adaptive ensemble
    p.add_argument("--condition-time", default="day", dest="condition_time",
                   choices=["day", "night", "dusk", "dawn"],
                   help="Time of day for adaptive ensemble weights (default: day)")
    p.add_argument("--condition-weather", default="clear", dest="condition_weather",
                   choices=["clear", "rain", "fog", "snow"],
                   help="Weather condition for adaptive ensemble (default: clear)")
    p.add_argument("--condition-fog", action="store_true", dest="condition_fog",
                   help="Flag foggy conditions for ensemble weight adjustment")
    
    # NEW: Active learning enhancements
    p.add_argument("--active-learning-export", action="store_true", dest="active_learning_export",
                   help="Export active learning queue with images for labeling")
    
    # NEW: Self-supervised learning
    p.add_argument("--self-supervised", action="store_true", dest="self_supervised",
                   help="Generate pseudo-labels from unlabeled video (teacher model mode)")
    p.add_argument("--ssl-output", default="pseudo_labeled_data", dest="ssl_output",
                   help="Output directory for pseudo-labeled dataset (default: pseudo_labeled_data)")
    p.add_argument("--ssl-confidence", default=0.85, type=float, dest="ssl_confidence",
                   help="Confidence threshold for pseudo-labels (default: 0.85)")
    p.add_argument("--ssl-frame-skip", default=30, type=int, dest="ssl_frame_skip",
                   help="Frame skip for pseudo-labeling (default: 30)")
    p.add_argument("--ssl-max-frames", default=1000, type=int, dest="ssl_max_frames",
                   help="Maximum frames to pseudo-label (default: 1000)")

    # Physics depth
    p.add_argument("--no-depth",       action="store_true", dest="no_depth",
                   help="Disable physics depth estimation")
    p.add_argument("--camera-height",  default=1.2,  type=float, dest="camera_height",
                   help="Camera mount height in metres (default: 1.2)")
    p.add_argument("--sun-elevation",  default=60.0, type=float, dest="sun_elevation",
                   help="Solar elevation angle degrees (default: 60)")

    # Cost
    p.add_argument("--state", default="Tamil Nadu",
                   help="State for PWD SOR cost rates (default: Tamil Nadu)")

    # Preprocessing toggles
    p.add_argument("--no-dedup",    action="store_true", dest="no_dedup")
    p.add_argument("--no-roi",      action="store_true", dest="no_roi")
    p.add_argument("--no-adaptive", action="store_true", dest="no_adaptive")

    # Detection feature toggles
    p.add_argument("--no-measurements",    action="store_true", dest="no_measurements")
    p.add_argument("--no-active-learning", action="store_true", dest="no_active_learning")
    p.add_argument("--no-damage-analysis", action="store_true", dest="no_damage_analysis",
                   help="Skip crack mechanism + pothole age analysis")

    # Temporal fusion + forecasting
    p.add_argument("--no-fusion",  action="store_true", dest="no_fusion",
                   help="Disable temporal fusion (single-frame scoring only)")
    p.add_argument("--road-type",  default="urban_mixed", dest="road_type",
                   choices=["national_heavy","national_medium","state_heavy",
                            "state_medium","urban_mixed","urban_light",
                            "rural_light","default"],
                   help="Road type for deterioration model (default: urban_mixed)")
    # PCI / budget / work orders
    p.add_argument("--budget",     default=None, type=float,
                   help="Budget cap in INR for budget allocation (e.g. 500000)")
    p.add_argument("--budget-strategy", default="urgency_first",
                   dest="budget_strategy",
                   choices=["urgency_first","cost_effective","worst_first"],
                   help="Budget allocation strategy (default: urgency_first)")
    p.add_argument("--work-orders", action="store_true", dest="work_orders",
                   help="Generate contractor work-order PDFs for critical segments")
    # Timelapse
    p.add_argument("--timelapse",       action="store_true",
                   help="Generate timelapse output video")
    p.add_argument("--timelapse-speed", default=8, type=int, dest="timelapse_speed",
                   help="Timelapse speed multiplier (default: 8)")

    # ══════════════════════════════════════════════════════════════════════════
    # NEW: Gamification, Blockchain, Accessibility Arguments
    # ══════════════════════════════════════════════════════════════════════════
    
    # Gamification
    p.add_argument("--no-gamification", action="store_true", dest="no_gamification",
                   help="Disable gamification features")
    p.add_argument("--citizen-mode", action="store_true", dest="citizen_mode",
                   help="Enable citizen reporting mode (awards points)")
    p.add_argument("--reporter-id", default=None, dest="reporter_id",
                   help="Reporter user ID for gamification points")
    p.add_argument("--ward", default=None,
                   help="Ward name for leaderboards")
    
    # Blockchain
    p.add_argument("--blockchain", action="store_true",
                   help="Enable blockchain tracking (immutable audit trail)")
    
    # Accessibility
    p.add_argument("--accessibility", action="store_true",
                   help="Enable accessibility features (audio, screen reader)")
    p.add_argument("--audio-alerts", action="store_true", dest="audio_alerts",
                   help="Enable real-time audio damage alerts")
    p.add_argument("--language", default="en", choices=["en", "hi", "ta"],
                   help="Language for audio descriptions (en/hi/ta, default: en)")
    
    # ══════════════════════════════════════════════════════════════════════════

    run(p.parse_args())