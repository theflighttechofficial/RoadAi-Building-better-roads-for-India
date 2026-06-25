# verify.py — End-to-end pipeline verification
import cv2
import time
from pathlib import Path

VIDEO = r"C:\Users\DELL\Downloads\road_damage_ai\3657637695-preview.mp4"
MODEL = "yolov8n.pt"   # or "yolov8n.pt" — whichever .pt file you have

print("\n" + "="*50)
print("  ROAD DAMAGE AI — PIPELINE VERIFICATION")
print("="*50)

# ── Step 1: Check files exist ──────────────────────
print("\n[1/5] Checking files...")
checks = {
    "Video"  : VIDEO,
    "Model"  : MODEL,
    "Detector": "detector.py",
    "Scorer" : "scoring.py",
    "Mapper" : "mapping.py",
}
all_ok = True
for name, path in checks.items():
    exists = Path(path).exists()
    status = "✅" if exists else "❌ MISSING"
    print(f"  {name:<12}: {status}  ({path})")
    if not exists:
        all_ok = False

if not all_ok:
    print("\n  ❌ Fix missing files before continuing.")
    exit(1)

# ── Step 2: Open video ─────────────────────────────
print("\n[2/5] Reading video...")
cap = cv2.VideoCapture(VIDEO)
if not cap.isOpened():
    print("  ❌ Cannot open video.")
    exit(1)

total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps    = cap.get(cv2.CAP_PROP_FPS)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
dur    = total / fps if fps else 0
cap.release()

print(f"  ✅ {width}×{height}  {fps:.1f} fps  "
      f"{total} frames  ({dur:.1f}s)")

# ── Step 3: Load model ─────────────────────────────
print("\n[3/5] Loading model...")
try:
    from detector import RoadDamageDetector
    detector = RoadDamageDetector(MODEL)
    print(f"  ✅ Model loaded: {MODEL}")
except Exception as e:
    print(f"  ❌ Model load failed: {e}")
    exit(1)

# ── Step 4: Run detection on first 5 frames ────────
print("\n[4/5] Running detection on first 5 frames...")
cap    = cv2.VideoCapture(VIDEO)
passed = 0
t0     = time.time()

for i in range(5):
    ret, frame = cap.read()
    if not ret:
        break
    try:
        detections = detector.detect(frame)
        status = f"{len(detections)} detection(s)"
        print(f"  Frame {i}: ✅  {status}")
        passed += 1
    except Exception as e:
        print(f"  Frame {i}: ❌  {e}")

cap.release()
elapsed = time.time() - t0
print(f"\n  {passed}/5 frames passed  ({elapsed:.2f}s)")

if passed == 0:
    print("  ❌ Detector not working. Check model + ultralytics install.")
    exit(1)

# ── Step 5: Test scorer + mapper ───────────────────
print("\n[5/5] Testing scorer and mapper...")

# Scorer
try:
    from scoring import RoadScorer
    scorer = RoadScorer()
    dummy  = [{"bbox": [100, 200, 300, 400], "confidence": 0.75, "class": 3}]
    result = scorer.score_frame(dummy)
    print(f"  ✅ Scorer  — score={result.health_score}  "
          f"priority={result.priority}")
except Exception as e:
    print(f"  ❌ Scorer failed: {e}")

# Mapper
try:
    from mapping import create_map
    test_locs = [
        {"lat": 13.0827, "lon": 80.2707, "score": 90, "label": "Test A"},
        {"lat": 13.0850, "lon": 80.2750, "score": 45, "label": "Test B"},
    ]
    create_map(test_locs, output_file="verify_map.html", title="Verify Test")
    print(f"  ✅ Mapper  — verify_map.html created")
except Exception as e:
    print(f"  ❌ Mapper failed: {e}")

# ── Final result ───────────────────────────────────
print("\n" + "="*50)
print("  ✅  ALL CHECKS PASSED — pipeline is working!")
print("  Next: run  python main.py  for full processing")
print("="*50 + "\n")