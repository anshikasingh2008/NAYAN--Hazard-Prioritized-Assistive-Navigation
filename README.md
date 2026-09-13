<div align="center">

# 👁️ NAYAN
### Hazard-Prioritized Assistive Navigation

**Real-time navigation for the visually impaired — powered by YOLOv8.**

Identifies dangerous obstacles and delivers prioritized audio alerts via smartphone,
so users receive only the most critical warnings first.

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF?logo=yolo&logoColor=black)](https://docs.ultralytics.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](#-license)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen.svg)]()

</div>

---

## 📖 Overview

**NAYAN** is an assistive navigation system designed for visually impaired users. It combines
real-time object detection with a **hazard-prioritization engine** that ranks detected
obstacles by urgency, ensuring the user hears only what matters most at any given moment.

### ✨ Key Features

- 🎯 **YOLOv8-powered detection** — fast, accurate object recognition
- ⚠️ **Hazard prioritization** — critical obstacles announced first
- 🔊 **Audio feedback** — voice alerts through the smartphone speaker
- ⚡ **Optimized & quantized** — runs efficiently on edge devices
- 🧩 **Modular scripts** — each stage (convert → train → export → demo) is standalone

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

---

## 📜 Scripts Reference

| # | Script | Description |
|---|--------|-------------|
| 1 | `scripts/1_convert_coco_to_yolo.py` | Convert COCO annotations → YOLO format |
| 2 | `scripts/2_train_yolov8.py` | Train the YOLOv8 model |
| 3 | `scripts/3_export_quantize.py` | Export & quantize for deployment |
| 4 | `scripts/4_hazard_priority.py` | Hazard prioritization logic |
| 5 | `scripts/5_live_demo.py` | Real-time live demo |
| – | `scripts/download_hazard_images.py` | Fetch sample hazard images |

---

## 🤝 Contributing (Team Workflow)

> **⚠️ Never push directly to `main`.** Always use feature branches and Pull Requests.

### Step-by-step

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

Found a bug or have an idea? Open an issue:

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
├── 📂 scripts/                          # All Python scripts
│   ├── 1_convert_coco_to_yolo.py
│   ├── 2_train_yolov8.py
│   ├── 3_export_quantize.py
│   ├── 4_hazard_priority.py
│   ├── 5_live_demo.py
│   └── download_hazard_images.py
│
├── 📂 annotations/                      # (excluded — get from owner)
├── 📂 images/                           # (excluded — get from owner)
├── 📂 labels/                           # (excluded — get from owner)
│
├── 📄 data.yaml                         # YOLO dataset config
├── 📄 requirements.txt                  # Python dependencies
├── 📄 .gitignore
├── 📄 .gitattributes
└── 📄 README.md
```

---

## 🧠 Tech Stack

- **Model:** [YOLOv8](https://docs.ultralytics.com/) (Ultralytics)
- **Language:** Python 3.10+
- **Deep Learning:** PyTorch
- **Computer Vision:** OpenCV
- **Deployment Target:** Edge / mobile (quantized)

---

## 📄 License

This project is licensed under the **MIT License** — see the `LICENSE` file for details.

---

## 👤 Author

**Anshika Singh** — [@anshikasingh2008](https://github.com/anshikasingh2008)

---

## 🙏 Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) for the detection framework
- The open-source computer vision community
- Everyone contributing to assistive technology for the visually impaired

---

<div align="center">

**⭐ If you find this project useful, please give it a star! ⭐**

Made with ❤️ for accessibility

</div>
