<div align="center">

# 👁️ NAYAN
### Hazard-Prioritized Assistive Navigation

**Real-time navigation for the visually impaired — powered by YOLOv8n.**

Detects and tracks obstacles, ranks them by danger, and speaks only the most critical warning
through the smartphone — so users hear what matters first instead of everything at once.

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8n-Ultralytics-00FFFF?logo=yolo&logoColor=black)](https://docs.ultralytics.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](#-license)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen.svg)]()

*Project Exhibition (DSN2098) · Group 10 · VIT Bhopal University*

</div>

---

## 📖 Overview

Most assistive-vision tools answer *"What objects are present?"* A person walking down a street
needs a different answer: *"Which object is the most important threat right now?"*

**NAYAN** combines real-time object detection with a **hazard-prioritization engine**. Each detected
object is tracked across frames and scored on proximity, object risk, motion, path relevance and
Time-to-Collision (TTC). Only the highest-priority hazard is announced, which reduces cognitive load
and avoids burying the one warning that matters.

### ✨ Key Features

- 🎯 **YOLOv8n detection** — lightweight detector suited to smartphones
- 🔁 **ByteTrack tracking** — persistent IDs, so two people become *Person 1* and *Person 2*
- ⚠️ **Hazard prioritization** — proximity + object risk + motion + path relevance + TTC
- 🛣️ **Path-aware scoring** — objects in the central walking corridor matter more than objects at the sides
- ⏱️ **TTC urgency estimate** — derived from bounding-box growth across frames
- 🔊 **Audio alerts** — short spoken warnings via Text-to-Speech
- 🛟 **Failure handling** — says *"Camera view unclear"* instead of staying silent, so silence is never mistaken for safety
- ⚡ **Quantized for edge devices** — INT8 TensorFlow Lite model is 72.8% smaller than FP32
- 🧩 **Modular scripts** — each stage (convert → train → export → hazard logic → demo) runs standalone

---

## 🏗️ How It Works

```
Camera → YOLOv8n Detection → ByteTrack Tracking → Hazard Analysis → Path + TTC → Priority Selection → Voice Alert
```

For every frame, NAYAN scores each tracked object using five factors and announces the highest-scoring one:

| Factor | What it captures | Source |
|--------|------------------|--------|
| **Proximity** | Larger bounding box ≈ closer object | Bounding-box size |
| **Object risk** | Different classes carry different risk weights | Class label |
| **Motion** | Stationary, approaching, moving away, lateral movement | Tracked position across frames |
| **Path relevance** | Higher inside the central walking corridor | Bounding-box position |
| **TTC** | Rapid bounding-box growth = more urgent | Bounding-box size across frames |

> **Note:** TTC is an *estimated urgency indicator* from a single monocular camera, not a calibrated
> physical collision time. It is reported as `N/A` when there isn't enough temporal evidence.

**Example (live demo):** a person inside the walking corridor (path relevance 1.00) scored a hazard
of 0.84, while a fire hydrant off to the side (path relevance 0.20) scored 0.53.

---

## 🎬 Live Demo Screenshots

The demo draws the **walking corridor** (cyan), each detection with its class, confidence and tracking ID,
and a status bar with the **Hazard**, **Path** and **TTC** values. The console prints the spoken alert.

<table>
  <tr>
    <td align="center"><img src="screenshots/demo-car.png" alt="NAYAN live demo detecting two cars" width="480"></td>
    <td align="center"><img src="screenshots/demo-stop-sign.png" alt="NAYAN live demo detecting stop signs" width="300"></td>
  </tr>
  <tr>
    <td align="center"><sub><b>Car ahead</b> — two cars tracked (ID 38, ID 44); the one in the corridor scores Hazard 0.84, Path 0.99, TTC N/A → alert: <i>"car ahead"</i></sub></td>
    <td align="center"><sub><b>Stop sign</b> — two stop signs tracked (ID 26 and a second ID); Hazard 0.30, Path 0.88 → alert: <i>"stop sign ahead"</i></sub></td>
  </tr>
</table>

> The console also shows the fallback alert *"Camera view unclear, please check camera position"* when no confident detection is available.

---

## 📊 Results

Model: YOLOv8n, trained for 50 epochs on a Colab T4 GPU using **500 images, 11 classes**, with an
**80/20 held-out split**.

| Metric | FP32 (TFLite) | INT8 (TFLite) |
|--------|---------------|---------------|
| Model size | 11.70 MB | **3.18 MB** (≈ 72.8% smaller) |
| mAP@0.5 | 0.3743 | 0.3592 (≈ 96% retained) |
| mAP@0.5:0.95 | 0.2367 | 0.1982 (≈ 83.7% retained) |

<details>
<summary><b>Per-class detection performance</b></summary>

| Class | Score |
|-------|-------|
| Fire Hydrant | 0.828 |
| Stop Sign | 0.714 |
| Person | 0.673 |
| Bus | 0.535 |
| Bicycle | 0.425 |
| Car | 0.324 |
| Truck | 0.211 |
| Bench | 0.122 |
| Traffic Light | 0.118 |
| Chair | 0.076 |

Weak classes (Chair, Traffic Light, Bench, Truck) are the priority for additional data.

</details>

> These metrics come from the initial 500-image, 11-class model. The additional 600 labelled images
> (see below) are **not** included yet.

---

## 🗂️ Dataset

| Dataset | Images | Classes | Status |
|---------|--------|---------|--------|
| Current training set | 500 | 11 | Used for all reported metrics |
| Additional labelled set | 600 | 4 (2 new) | Labelled in YOLO format; integration and splitting pending |
| Expanded dataset | — | 13 | After integration and retraining |

The additional 600 images are 150 each of **Truck**, **Fire Hydrant**, **Stairs** (new) and **Pothole** (new).

---

## 🚧 Project Status

**Done**
- ✅ YOLOv8n trained and evaluated on held-out data
- ✅ FP32 and INT8 TensorFlow Lite export and evaluation
- ✅ ByteTrack multi-object tracking
- ✅ Path-aware hazard scoring with TTC-based urgency
- ✅ Live Python demo showing hazard score, path relevance and TTC
- ✅ Android project builds; debug APK generated and installed

**In progress / planned**
- ⬜ Integrate and split the 600 new images, then retrain on the 13-class dataset
- ⬜ Complete and validate the Android pipeline on-device (CameraX → TFLite → tracking → hazard → TTS), including latency and battery measurements
- ⬜ FP16 export to complete the FP32 / FP16 / INT8 comparison
- ⬜ Collect more data for weak classes (Bench, Chair, Traffic Light, Truck)
- ⬜ Test under varied lighting, occlusion and crowds, and with visually impaired users

**Out of scope for now:** GPS / map navigation, cloud inference, face or emotion recognition, wearable hardware.

---

## 🚀 Quick Start

### Prerequisites

| Tool | Version | Download |
|------|---------|----------|
| Python | 3.10 or higher | [python.org](https://www.python.org/downloads/) |
| Git | Latest | [git-scm.com](https://git-scm.com/downloads) |
| VS Code *(optional)* | Latest | [code.visualstudio.com](https://code.visualstudio.com/) |

### 1. Clone the repository

```bash
git clone https://github.com/anshikasingh2008/NAYAN--Hazard-Prioritized-Assistive-Navigation.git
cd NAYAN--Hazard-Prioritized-Assistive-Navigation
```

### 2. Create a virtual environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

> ⚠️ If PowerShell blocks the activation script, run this first:
> ```powershell
> Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
> ```

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Obtain the dataset

The `annotations/`, `images/`, and `labels/` folders (~500 MB) are **not** included in this
repository due to size limits. Request the dataset from the project owner, then place all
three folders in the **project root**:

```
NAYAN--Hazard-Prioritized-Assistive-Navigation/
├── annotations/     ← add this
├── images/          ← add this
├── labels/          ← add this
├── scripts/
├── data.yaml
└── requirements.txt
```

### 5. Run the live demo

```bash
python scripts/5_live_demo.py
```

The demo opens your webcam and overlays the walking corridor (cyan rectangle), each detection with
its class, confidence and tracking ID, and a bottom bar showing the **Hazard**, **Path** and **TTC** values.

---

## 📜 Scripts Reference

| # | Script | Description |
|---|--------|-------------|
| 1 | `scripts/1_convert_coco_to_yolo.py` | Convert COCO annotations → YOLO format |
| 2 | `scripts/2_train_yolov8.py` | Train the YOLOv8 model |
| 3 | `scripts/3_export_quantize.py` | Export & quantize (TFLite FP32 / INT8) |
| 4 | `scripts/4_hazard_priority.py` | Hazard prioritization logic |
| 5 | `scripts/5_live_demo.py` | Real-time live demo |
| – | `scripts/download_hazard_images.py` | Fetch sample hazard images |

---

## 🤝 Contributing (Team Workflow)

> **⚠️ Never push directly to `main`.** Always use feature branches and Pull Requests.

**1️⃣ Sync with the latest code**
```bash
git checkout main
git pull origin main
```

**2️⃣ Create a branch for your work**
```bash
git checkout -b feature/your-feature-name
```
*Examples:* `feature/audio-alerts`, `fix/training-bug`, `docs/readme-update`

**3️⃣ Make changes and commit**
```bash
git add .
git commit -m "feat: add proximity-based audio priority"
```
Follow [Conventional Commits](https://www.conventionalcommits.org/):
`feat:`, `fix:`, `docs:`, `refactor:`, `chore:`

**4️⃣ Push your branch**
```bash
git push origin feature/your-feature-name
```

**5️⃣ Open a Pull Request**
- Go to the repository on GitHub → click **"Compare & pull request"**
- Describe what you changed and why
- Request a review from a teammate
- Wait for approval → merge into `main`

**6️⃣ Clean up locally**
```bash
git checkout main
git pull origin main
git branch -d feature/your-feature-name
```

---

## 🚫 What NOT to Commit

The following are already listed in `.gitignore` — **do not force-add them**:

| Path | Reason |
|------|--------|
| `.venv/` | Platform-specific virtual environment |
| `annotations/`, `images/`, `labels/` | 500+ MB dataset |
| `runs/` | Training outputs (regenerable) |
| `*.pt`, `*.pth`, `*.onnx` | Large model weight files |
| `__pycache__/` | Python bytecode cache |
| `.env` | Secrets / API keys |

> ⚠️ If you accidentally commit a large file, **notify the team before force-pushing.**

---

## 🐛 Issues & Feature Requests

1. Go to the **Issues** tab on GitHub
2. Click **New issue**
3. Describe the problem with steps to reproduce
4. Assign it to yourself or a teammate

Use labels: `bug`, `enhancement`, `documentation`, `help wanted`

---

## 📁 Project Structure

```
NAYAN--Hazard-Prioritized-Assistive-Navigation/
│
├── 📂 scripts/                          # Python pipeline, one script per stage
│   ├── 1_convert_coco_to_yolo.py
│   ├── 2_train_yolov8.py
│   ├── 3_export_quantize.py
│   ├── 4_hazard_priority.py
│   ├── 5_live_demo.py
│   └── download_hazard_images.py
│
├── 📂 android/                          # Android app (CameraX → TFLite → hazard → TTS)
├── 📂 models/                           # Trained / exported models
├── 📂 results/                          # Evaluation outputs
├── 📂 additional_data/                  # Additional labelled images (not yet in training)
├── 📂 screenshots/                      # README screenshots
│
├── 📂 runs/detect/                      # Training runs (nayan_hazard_detector, -2, val) — regenerable
├── 📂 annotations/                      # (excluded — get from owner)
├── 📂 images/                           # (excluded — get from owner)
├── 📂 labels/                           # (excluded — get from owner)
│
├── 📄 data.yaml                         # YOLO dataset config
├── 📄 requirements.txt                  # Python dependencies
├── 📄 SRIJA_DATASET.md                  # Dataset notes
├── 📄 .gitignore
├── 📄 .gitattributes
└── 📄 README.md
```

> `.venv/` and `yolov8.pt` (base weights) also sit in the project root locally but are not committed.

---

## 🧠 Tech Stack

| Area | Tools |
|------|-------|
| Detection | YOLOv8n (Ultralytics) |
| Tracking | ByteTrack |
| Language / DL | Python 3.10+, PyTorch |
| Computer vision | OpenCV |
| Deployment | TensorFlow Lite (FP32 / INT8) |
| Mobile | Android Studio, CameraX, Android Text-to-Speech |
| Training | Google Colab (T4 GPU) |

---

## 👥 Team

**Group 10 — Project Exhibition (DSN2098), School of Computing Science and Engineering (AI-ML),
VIT Bhopal University**

| Name | Role |
|------|------|
| [Anshika Singh](https://github.com/anshikasingh2008) | Coordination & architecture |
| Srija Das | Dataset collection & labelling |
| Tushita Pathak | Model training & quantization |
| Anjali Kumari | Object tracking & Android integration |
| Shanya Kushwaha | Path-aware detection & TTC |
| Surbhi Kumari | Documentation, testing & presentation |



---

## 📄 License

This project is licensed under the **MIT License** — see the `LICENSE` file for details.

---

## 🙏 Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) for the detection framework
- [ByteTrack](https://github.com/ifzhang/ByteTrack) for multi-object tracking
- The open-source computer vision community
- Everyone contributing to assistive technology for the visually impaired

---

<div align="center">

**⭐ If you find this project useful, please give it a star! ⭐**

Made with ❤️ for accessibility

</div>
