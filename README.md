# 🛣️ Road-AI — Intelligent Road Damage Detection & Smart Infrastructure Management

---

## 📌 Overview

Road-AI is a full-stack AI pipeline for automated road damage detection, severity classification, and smart repair management. It processes road imagery (photos, video frames, or drone footage) and produces actionable repair reports grounded in real government costing standards.

The system combines computer vision, physics-based depth estimation, and civic-tech tooling into a single end-to-end platform — from raw image input to PDF/Excel repair tickets with QR codes.

---

## 🚀 Key Features

- **Ensemble YOLOv8 Detection** — fine-tuned `best.pt` model for multi-class road damage (potholes, cracks, delamination, patches, etc.), outperforming the base `YOLOv8_Small_RDD.pt`
- **Temporal Fusion** — aggregates detections across video frames to reduce false positives
- **Monte Carlo Dropout** — uncertainty quantification on detections, flagging low-confidence predictions
- **Depth Estimation via Shadow-Gradient Physics** — estimates damage depth without LiDAR using shadow geometry and gradient analysis
- **Cost Estimation** — repair cost modelling grounded in **Tamil Nadu PWD** and **NHAI Schedule of Rates (SOR)**
- **Gamification Layer** — citizen reporting rewards and leaderboard system to crowdsource road condition data
- **Blockchain Repair Tracking** — immutable repair event log for accountability across agencies
- **Multilingual Accessibility** — UI and reports available in **English, Hindi, and Tamil**
- **Rich Output Formats** — PDF reports, Excel sheets, GeoJSON overlays, interactive HTML maps, QR-coded repair tickets

---

## 🧠 Tech Stack

| Layer | Tools |
|-------|-------|
| Detection | YOLOv8 (Ultralytics), OpenCV |
| Uncertainty | Monte Carlo Dropout |
| Depth Estimation | Shadow-gradient physics model |
| Cost Engine | Tamil Nadu PWD SOR, NHAI SOR |
| Mapping | GeoJSON, Folium (HTML maps) |
| Output | ReportLab (PDF), openpyxl (Excel), qrcode |
| Blockchain | [your chain/library here] |
| Language | Python 3.10+ |

---

## 📁 Project Structure

```
road-ai/
├── main.py                  # Main 1600+ line pipeline
├── best.pt                  # Fine-tuned YOLOv8 model (primary)
├── YOLOv8_Small_RDD.pt      # Base model (fallback)
├── detectors/
│   ├── ensemble.py          # Ensemble detection logic
│   ├── temporal_fusion.py   # Frame aggregation
│   └── uncertainty.py       # Monte Carlo Dropout
├── depth/
│   └── shadow_gradient.py   # Physics-based depth estimator
├── costing/
│   ├── tnpwd_sor.py         # Tamil Nadu PWD rates
│   └── nhai_sor.py          # NHAI rates
├── outputs/
│   ├── report_generator.py  # PDF + Excel reports
│   ├── geojson_export.py    # GeoJSON map output
│   └── qr_tickets.py        # QR-coded repair tickets
├── gamification/
│   └── leaderboard.py       # Citizen reporting rewards
├── blockchain/
│   └── tracker.py           # Repair event logging
├── ui/
│   └── multilingual.py      # EN / HI / TA support
└── requirements.txt
```

---

## ⚙️ Setup & Installation

```bash
# Clone the repository
git clone https://github.com/theflighttechofficial/road-ai.git
cd road-ai

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Requirements (key packages)
```
ultralytics
opencv-python
torch
torchvision
reportlab
openpyxl
folium
qrcode
Pillow
numpy
pandas
```

---

## 🖥️ Usage

```bash
# Run on an image
python main.py --input road_image.jpg --lang en

# Run on a video
python main.py --input road_video.mp4 --mode video --lang ta

# Output to specific directory
python main.py --input road_image.jpg --output ./reports/
```

### CLI Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--input` | Path to image or video file | required |
| `--mode` | `image` or `video` | `image` |
| `--lang` | Output language: `en`, `hi`, `ta` | `en` |
| `--output` | Output directory | `./outputs/` |
| `--model` | Model weights path | `best.pt` |
| `--uncertainty` | Enable MC Dropout uncertainty | `True` |

---

## 📊 Output Samples

The pipeline generates the following per run:

- `report_<timestamp>.pdf` — full damage report with severity breakdown and cost estimates
- `damage_log_<timestamp>.xlsx` — structured Excel sheet of all detections
- `map_<timestamp>.html` — interactive Folium map with damage markers
- `damages_<timestamp>.geojson` — GeoJSON for GIS integration
- `ticket_<id>.png` — QR-coded repair ticket per damage cluster

---

## 🎯 Model Performance

| Metric | Value |
|--------|-------|
| Model | YOLOv8 (fine-tuned) |
| Dataset | RDD2022 + custom collected frames |
| Classes | Pothole, Longitudinal Crack, Transverse Crack, Alligator Crack, Patch |
| mAP@0.5 | 
| Inference speed |

> Fine-tuned `best.pt` consistently outperforms base `YOLOv8_Small_RDD.pt` across all damage categories.

---

## 🏗️ Pipeline Architecture

```
Input Image/Video
       │
       ▼
Ensemble YOLOv8 Detection
       │
  ┌────┴─────┐
  │Temporal  │  (video mode)
  │Fusion    │
  └────┬─────┘
       │
MC Dropout Uncertainty Filtering
       │
Shadow-Gradient Depth Estimation
       │
Cost Estimation (PWD/NHAI SOR)
       │
  ┌────┴──────────────────────┐
  │  PDF │ Excel │ GeoJSON   │
  │  HTML Map │ QR Tickets   │
  └───────────────────────────┘
       │
Blockchain Repair Log
```

## 📸 Screenshots

### Dashboard
![Dashboard](images/Dashboard.png)

---

### Upload Road Image
![upload](images/upload.png)

---

### Damage Analysis
![analysis](images/analysis.png)

---

### Export Report
![export](images/export.png)

---

### Citizen Reporting Portal
![citizenreporting](images/citizenreporting.png)

---

### Project Code Structure
![vscode](images/vscode.png)

---

## 🌐 Multilingual Support

Reports and UI labels are available in:
- 🇬🇧 English (`--lang en`)
- 🇮🇳 Hindi (`--lang hi`)
- 🇮🇳 Tamil (`--lang ta`)

---

## 🙏 Acknowledgements

- **RDD2022 Dataset** — Road Damage Detection and Classification Challenge
- **Tamil Nadu PWD** and **NHAI** for public Schedule of Rates data
- IIT Madras Hackathon organizers

---

## 📜 License

This project is for academic and hackathon demonstration purposes.  
© 2025 S. Varun Vaibhav. All rights reserved.

---

## 👤 Author

**S. Varun Vaibhav**  
B.Tech CSE (AI & Data Analytics) — SRIHER Chennai  
🔗 [GitHub](https://github.com/theflighttechofficial) | [LinkedIn](https://linkedin.in/in/varun-vaibhav-s-11b69a2ba)
