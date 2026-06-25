#!/usr/bin/env python3
"""
launch.py  -  Road-AI One-Command Launcher
==========================================

Starts the entire Road-AI system with a single command.

QUICK START
-----------
  python launch.py                       # opens dashboard, waits for upload
  python launch.py --demo                # runs with synthetic data (no model)
  python launch.py --video road.mp4      # analyse a video immediately
  python launch.py --video road.mp4 --gps 13.0827,80.2707 --excel --map

KEYBOARD SHORTCUTS (in browser after launch)
--------------------------------------------
  N   - New session        P - Download PDF
  G   - Download GeoJSON   D - Depth panel
  S   - SOR cost panel     B - Benchmark panel
  F   - Forecast panel     C - City heatmap
  X   - Compare frames     T - Temporal fusion
  ?   - Show help
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import signal
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path


# ── Terminal colour helpers ───────────────────────────────────────

def _c(code, text):
    return "\033[" + code + "m" + str(text) + "\033[0m" if sys.stdout.isatty() else str(text)

def grn(t): return _c("92", t)
def yel(t): return _c("93", t)
def cyn(t): return _c("96", t)
def red(t): return _c("91", t)
def bld(t): return _c("1",  t)
def dim(t): return _c("2",  t)


def banner():
    print()
    print(bld(cyn("  ██████╗  ██████╗  █████╗ ██████╗      █████╗ ██╗")))
    print(bld(cyn("  ██╔══██╗██╔═══██╗██╔══██╗██╔══██╗    ██╔══██╗██║")))
    print(bld(cyn("  ██████╔╝██║   ██║███████║██║  ██║    ███████║██║")))
    print(bld(cyn("  ██╔══██╗██║   ██║██╔══██║██║  ██║    ██╔══██║██║")))
    print(bld(cyn("  ██║  ██║╚██████╔╝██║  ██║██████╔╝    ██║  ██║██║")))
    print(bld(cyn("  ╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═╝╚═════╝     ╚═╝  ╚═╝╚═╝")))
    print()
    print(bld("  Road Damage Detection & Analysis System"))
    print(dim("  IIT Madras Road Health Showcase  -  v2.0"))
    print()


# ── Dependency checker ────────────────────────────────────────────

REQUIRED = {
    "cv2":       ("opencv-python", True),
    "numpy":     ("numpy",         True),
    "fastapi":   ("fastapi",       True),
    "uvicorn":   ("uvicorn",       True),
    "reportlab": ("reportlab",     True),
    "openpyxl":  ("openpyxl",      True),
}
OPTIONAL = {
    "ultralytics": "ultralytics",
    "gpxpy":       "gpxpy",
}


def check_deps():
    print(bld("  [1/4] Checking dependencies"))
    missing = []
    for module, (pkg, _) in REQUIRED.items():
        try:
            __import__(module)
            print("        " + grn("OK") + "  " + module)
        except ImportError:
            print("        " + red("MISSING") + "  " + module + "  ->  pip install " + pkg)
            missing.append(pkg)
    for module, pkg in OPTIONAL.items():
        try:
            __import__(module)
            print("        " + grn("OK") + "  " + module + "  (optional)")
        except ImportError:
            print("        " + yel("skip") + "  " + module + "  (optional - demo works without)")
    if missing:
        print()
        print(red("  Install missing packages:  pip install " + " ".join(missing)))
        sys.exit(1)
    print()


# ── Server launcher ───────────────────────────────────────────────

def start_server(port):
    print(bld("  [2/4] Starting dashboard server"))
    script = Path(__file__).parent / "dashboard_server.py"
    if not script.exists():
        print(red("  dashboard_server.py not found"))
        sys.exit(1)

    proc = subprocess.Popen(
        [sys.executable, str(script), "--port", str(port)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    # Drain server stdout silently
    threading.Thread(
        target=lambda: [line for line in proc.stdout],
        daemon=True
    ).start()

    # Poll until ready
    url = "http://localhost:" + str(port) + "/api/health"
    for attempt in range(40):
        time.sleep(0.5)
        try:
            urllib.request.urlopen(url, timeout=1)
            print("        " + grn("OK") + "  Server ready  ->  http://localhost:" + str(port))
            print()
            return proc
        except Exception:
            dots = "." * ((attempt % 3) + 1) + "   "
            print("\r        waiting" + dots, end="", flush=True)

    print(red("\n  Server failed to start"))
    proc.terminate()
    sys.exit(1)


# ── Browser opener ────────────────────────────────────────────────

def open_browser(port):
    print(bld("  [3/4] Opening dashboard"))
    url = "http://localhost:" + str(port)
    try:
        webbrowser.open(url)
        print("        " + grn("OK") + "  " + url)
    except Exception:
        print("        " + yel("info") + "  Open manually: " + cyn(url))
    print()


# ── Demo mode ─────────────────────────────────────────────────────

def run_demo(port):
    """Push 300 synthetic frames to the dashboard. No model needed."""
    print(bld("  [4/4] Demo mode  (synthetic data, no model required)"))
    print("        " + cyn("Simulating 300 frames of road survey..."))

    random.seed(42)
    base_url = "http://localhost:" + str(port) + "/api"

    # Create session
    try:
        res = urllib.request.urlopen(
            urllib.request.Request(
                base_url + "/sessions/create?video_path=DEMO_synthetic.mp4",
                method="POST"),
            timeout=5)
        sid = json.loads(res.read())["session_id"]
        print("        " + grn("OK") + "  Session: " + sid.upper())
    except Exception as e:
        print(red("  Could not create session: " + str(e)))
        return

    # Damage zones: (start_frame, end_frame, class, confidence, depth_cm)
    zones = [
        (40,  60,  "Pothole",            0.85, 7.2),
        (90,  110, "Alligator Crack",    0.78, 3.1),
        (150, 165, "Transverse Crack",   0.72, 0.0),
        (200, 230, "Pothole",            0.91, 9.4),
        (260, 280, "Longitudinal Crack", 0.68, 0.0),
    ]
    rates = {
        "Pothole": 2400, "Alligator Crack": 2900,
        "Transverse Crack": 980, "Longitudinal Crack": 860,
    }
    n = 300
    base_lat, base_lon = 13.0827, 80.2707

    for fi in range(n):
        # GPS dead-reckoning
        dist = fi / 25.0 * 0.04
        lat  = round(base_lat + dist * 0.009 + (random.random()-0.5)*0.0002, 6)
        lon  = round(base_lon + dist * 0.009 + (random.random()-0.5)*0.0002, 6)

        # Damage
        dets = 0; dom = ""; cost = 0.0; depth = 0.0
        score = round(95.0 + random.gauss(0, 2), 1)
        for zs, ze, cls, conf, dep in zones:
            if zs <= fi <= ze:
                dets  = random.randint(1, 3)
                dom   = cls
                depth = max(0.0, dep + random.gauss(0, 0.4))
                area  = random.uniform(0.3, 2.0)
                cost  = area * rates.get(cls, 1500) * 1.18
                score = max(20.0, 95.0 - dets*15 - depth*2 + random.gauss(0, 3))
                score = round(min(100.0, score), 1)
                break

        # Volume estimate (m3) for potholes
        vol = 0.0
        if dom in ("Pothole",) and depth > 0:
            area_m2 = random.uniform(0.3, 2.0)
            vol = round(area_m2 * depth / 100.0, 4)   # depth cm -> m

        cond = ("night" if fi < 20 else "fog" if 130 <= fi <= 145 else "normal")
        det_list = []
        if dets > 0:
            det_list = [{
                "class_name":      dom,
                "confidence":      round(random.uniform(0.65, 0.92), 2),
                "depth_cm":        round(depth, 1),
                "depth_low_cm":    round(max(0, depth - 1.5), 1),
                "depth_high_cm":   round(depth + 1.5, 1),
                "volume_m3":       vol,
                "severity_class":  ("deep" if depth > 6 else
                                    "moderate" if depth > 3 else "surface"),
                "cost_inr":        round(cost, 0),
                "repair_method":   "Hot-mix asphalt patching (TN-PWD SOR 2023-24 §5.2.3)",
                "sor_source":      "TN-PWD SOR 2023-24 §5.2.3 / NHAI SOR 2022 §5.4",
                "urgency_days":    3 if depth > 6 else 14,
                "bbox":            [300, 250, 600, 450],
                "n_observations":  random.randint(2, 8),
                "is_confirmed":    True,
                "temporal_confidence": round(random.uniform(0.65, 0.95), 2),
            }]

        fd = {
            "frame_index":     fi,
            "timestamp_sec":   round(fi / 25.0, 2),
            "n_detections":    dets,
            "health_score":    score,
            "dominant_class":  dom,
            "cost_inr":        round(cost, 0),
            "condition":       cond,
            "gps_lat":         lat,
            "gps_lon":         lon,
            "speed_kmh":       round(random.uniform(20, 45), 1),
            "max_depth_cm":    round(depth, 1),
            "max_volume_m3":   vol,
            "max_severity":    round(min(score * 0.8, 100), 1),
            "uncertain_count": 0,
            "enhanced":        False,
            "detections":      det_list,
        }

        try:
            body = json.dumps(fd).encode()
            req  = urllib.request.Request(
                base_url + "/sessions/" + sid + "/frame",
                data=body, method="POST",
                headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=3)
        except Exception:
            pass

        pct = (fi + 1) / n
        bar = "█" * int(pct * 28) + "░" * (28 - int(pct * 28))
        icon = "🌙" if cond == "night" else "🌫" if cond == "fog" else "  "
        print(
            "\r        [" + bar + "] " + str(round(pct*100)).rjust(3) + "%"
            "  score=" + str(int(score)) + "  " + icon + " ",
            end="", flush=True,
        )
        time.sleep(0.03)

    # Mark done
    try:
        urllib.request.urlopen(
            urllib.request.Request(
                base_url + "/sessions/" + sid + "/done", method="POST"),
            timeout=3)
    except Exception:
        pass

    print("\n        " + grn("OK") + "  Demo complete  -  " + str(n) + " frames pushed")
    print("        Session: " + cyn(sid.upper()) + "  (auto-selected in dashboard)")
    print()


# ── Pipeline runner ───────────────────────────────────────────────

def run_pipeline(args, port):
    print(bld("  [4/4] Starting analysis pipeline"))
    cmd = [
        sys.executable,
        str(Path(__file__).parent / "main.py"),
        "--video",     args.video,
        "--dashboard",
        "--port",      str(port),
        "--model",     args.model,
    ]
    if args.gps:     cmd += ["--gps",     args.gps]
    if args.output:  cmd += ["--output",  args.output]
    if args.excel:   cmd += ["--excel"]
    if args.tickets: cmd += ["--tickets"]
    if args.map:     cmd += ["--map"]
    print("        " + dim(" ".join(cmd)))
    print()
    subprocess.run(cmd)


# ── Status monitor ────────────────────────────────────────────────

def status_monitor(port, proc):
    while True:
        time.sleep(5)
        if proc.poll() is not None:
            break
        try:
            url  = "http://localhost:" + str(port) + "/api/sessions"
            res  = urllib.request.urlopen(url, timeout=2)
            sess = json.loads(res.read())
            n    = len(sess)
            if n:
                latest = sess[-1]
                avg    = round(latest.get("avg_health_score", 100), 1)
                det    = latest.get("total_detections", 0)
                ts     = time.strftime("%H:%M:%S")
                line   = ("  [" + ts + "]  Sessions: " + str(n) +
                          "   Score: " + str(avg) + "/100" +
                          "   Detections: " + str(det))
                print("\r" + dim(line) + "    ", end="", flush=True)
        except Exception:
            pass


# ── Main ──────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Road-AI - One-command launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--video",      default=None,
                   help="Video file to analyse immediately")
    p.add_argument("--demo",       action="store_true",
                   help="Run demo with synthetic data (no model needed)")
    p.add_argument("--model",      default="yolov8n.pt")
    p.add_argument("--gps",        default=None,
                   help="Start GPS: 'lat,lon'")
    p.add_argument("--output",     default="output")
    p.add_argument("--port",       default=8000, type=int)
    p.add_argument("--no-browser", action="store_true")
    p.add_argument("--excel",      action="store_true")
    p.add_argument("--tickets",    action="store_true")
    p.add_argument("--map",        action="store_true")
    args = p.parse_args()

    banner()
    check_deps()
    server = start_server(args.port)

    if not args.no_browser:
        open_browser(args.port)

    def _shutdown(sig, frame):
        print("\n\n  " + yel("Shutting down..."))
        server.terminate()
        try:
            server.wait(timeout=3)
        except Exception:
            server.kill()
        print("  " + grn("Done.") + "\n")
        sys.exit(0)

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    print(bld("  Road-AI is running"))
    print("  Dashboard  : " + cyn("http://localhost:" + str(args.port)))
    print("  API docs   : " + cyn("http://localhost:" + str(args.port) + "/docs"))
    print("  GeoJSON    : " + cyn("http://localhost:" + str(args.port) +
                                   "/api/sessions/{sid}/export/geojson"))
    print("  Press " + bld("Ctrl+C") + " to stop")
    print()

    if args.demo:
        run_demo(args.port)
        print("  " + grn("Demo loaded!") +
              "  Click D, S, B, F, C in the browser topbar to explore panels.")
        print("  Press ? in the browser for keyboard shortcut help.")
        print()
    elif args.video:
        run_pipeline(args, args.port)
    else:
        print("  " + yel("Tip:") + " Run  " + bld("python launch.py --demo") +
              "  for a full demo without a video file.")
        print("  " + yel("Tip:") + " Run  " + bld("python launch.py --video road.mp4") +
              "  to analyse a video.")
        print()
        threading.Thread(
            target=status_monitor, args=(args.port, server), daemon=True
        ).start()

    try:
        server.wait()
    except KeyboardInterrupt:
        _shutdown(None, None)


if __name__ == "__main__":
    main()