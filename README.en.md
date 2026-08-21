# Detection-System

[中文说明](README.md)

Detection-System is a Windows desktop application for classifying detected grapes as `immature`, `semi-mature`, or `mature`. It combines a PyQt5 interface with an included YOLO checkpoint and supports images, folders, video files, and cameras.

## Highlights

- Background inference keeps the interface responsive.
- Each worker owns its YOLO instance; models are never shared across threads.
- The bundled model is verified with SHA-256 before loading.
- Corrupt or oversized media is rejected before inference.
- Image and batch results include CSV data with spreadsheet-formula neutralization.
- Video exports are written to a temporary file, verified, then atomically published.
- Canceling an export removes the partial file.
- The result table is capped at 500 rows to prevent unbounded memory growth.

## Quick start

Requirements: Windows 10/11 and CPython 3.10–3.12.

```powershell
git clone https://github.com/liuxiao20051106-prog/Detection-System.git
cd Detection-System
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python MainProgram.py
```

The default output directory is `%LOCALAPPDATA%\Detection-System\results`.

## Configuration

```powershell
$env:DETECTION_MODEL_PATH = "D:\models\best.pt"
$env:DETECTION_MODEL_SHA256 = "<64-character SHA-256>"
$env:DETECTION_SAVE_PATH = "D:\detection-results"
$env:DETECTION_CAMERA_ID = "0"
python MainProgram.py
```

Never use a checkpoint from an untrusted source. PyTorch checkpoints can contain executable serialized objects. This project verifies file identity, but that only proves a file matches the hash you supplied; it does not establish the source's trustworthiness.

## Known limitations

- Exported video does not retain the source audio.
- The included samples are demonstrations, not a labeled benchmark.
- The training dataset is not included, so the supplied metrics cannot be independently reproduced from this repository alone.
- Model results must not be used as a sole basis for food-safety, agricultural, financial, or other high-impact decisions.

See [MODEL_CARD.md](MODEL_CARD.md), [SECURITY.md](SECURITY.md), and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) before redistribution or commercial use. The repository currently has no declared project-level license; all rights remain reserved unless the owner adds one.

## Verification

```powershell
$env:QT_QPA_PLATFORM = "offscreen"
python -m unittest discover -s tests -v
python -m ruff check .
python -m pip_audit -r requirements.txt --strict
```

For contributions and release packaging, see [CONTRIBUTING.md](CONTRIBUTING.md).
