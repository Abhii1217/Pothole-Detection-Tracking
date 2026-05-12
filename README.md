# 🚧 Pothole Detection & Tracking

A computer vision pipeline for automatic road defect detection, multi-object tracking, and GPS-based geolocation logging using **YOLOv8** and **ByteTrack**.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-brightgreen)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📌 Overview

This project processes dashcam footage alongside GPX GPS recordings to:
- Detect road defects (potholes, cracks) using a custom YOLOv8 model
- Track each defect across frames using ByteTrack
- Synchronize detections with real-world GPS coordinates
- Export results as CSV, annotated video, and GeoJSON for mapping

---

## 🎯 Features

- 🔍 **YOLOv8 Detection** — Detects 4 road defect classes:
  - Longitudinal Cracks
  - Transverse Cracks
  - Alligator Cracks
  - Potholes
- 🎯 **ByteTrack Multi-Object Tracking** — Assigns persistent IDs to each detected defect
- 📍 **GPS Synchronization** — Maps each detection to real-world coordinates via GPX file
- 🎬 **Annotated Video Output** — Saves a fully annotated video with bounding boxes and track IDs
- 📊 **CSV Logging** — Logs every pothole crossing with frame, timestamp, and GPS coordinates
- 🗺️ **GeoJSON Export** — Ready-to-use output for mapping tools like QGIS or Mapbox

---

## 📁 Project Structure

    pothole-detection-tracking/
    │
    ├── pothole_detection/        
    │   ├── __init__.py
    │   ├── config.py             
    │   ├── gps.py                
    │   └── tracker.py            
    │
    ├── config/
    │   └── default_config.yaml   
    │
    ├── data/                     
    ├── models/                   
    ├── outputs/                  
    │
    ├── run.py                    
    ├── requirements.txt
    └── README.md

## ⚙️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/Abhii1217/Pothole-Detection-Tracking.git
cd Pothole-Detection-Tracking
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Install ByteTrack
```bash
git clone https://github.com/ifzhang/ByteTrack.git
cd ByteTrack
pip install -r requirements.txt
python setup.py develop
cd ..
```

---

## 🗂️ Setup

### 1. Add your files
data/          ← place your input video (.mov/.mp4) and GPX file here
models/        ← place your YOLOv8 weights file (best.pt) here

### 2. Edit the config file
Open `config/default_config.yaml` and update the filenames:

```yaml
paths:
  model: "models/best.pt"
  video: "data/your_video.mov"
  gpx:   "data/your_recording.gpx"
```

---

## 🚀 Usage

```bash
python run.py
```

Or with a custom config:

```bash
python run.py --config config/default_config.yaml
```

---

## 📤 Outputs

| File | Description |
|------|-------------|
| `outputs/pothole_log.csv` | Frame, type, track ID, timestamp, GPS coordinates |
| `outputs/annotated_video.mp4` | Video with bounding boxes and track IDs overlaid |
| `outputs/potholes.geojson` | GeoJSON point features for each detected pothole |

---

## 🛠️ Configuration

All settings are in `config/default_config.yaml`:

```yaml
detection:
  confidence_threshold: 0.25   # Lower = more detections
  iou_threshold: 0.45
  image_size: 640

tracking:
  track_thresh: 0.4
  match_thresh: 0.8
  track_buffer: 30

classes:
  poi_class_ids: [3]            # 3 = Potholes only
```

---

## 🔮 Future Improvements

- [ ] Real-time webcam/live stream support
- [ ] Web dashboard for visualizing GeoJSON on an interactive map
- [ ] Severity scoring for each detected defect
- [ ] Mobile app integration for field data collection
- [ ] Support for multiple camera angles
- [ ] Automated road condition reports (PDF export)

---

## 🧰 Tech Stack

| Tool | Purpose |
|------|---------|
| [YOLOv8](https://github.com/ultralytics/ultralytics) | Object detection |
| [ByteTrack](https://github.com/ifzhang/ByteTrack) | Multi-object tracking |
| [OpenCV](https://opencv.org/) | Video processing |
| [gpxpy](https://github.com/tkrajina/gpxpy) | GPX file parsing |
| [PyYAML](https://pyyaml.org/) | Config management |

---

## 👤 Author

**Abhishek**
- GitHub: [@Abhii1217](https://github.com/Abhii1217)

---

## 📄 License

This project is licensed under the MIT License.
