"""项目运行配置。"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "best.pt"
DEFAULT_MODEL_SHA256 = "ada66cb1215507aa4652d60b073ef64f9c6465daeac7e96475abe151b3d120c0"


def _default_save_path():
    if os.name == "nt":
        base = Path(os.getenv("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    else:
        base = Path(os.getenv("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return base / "Detection-System" / "results"


save_path = str(Path(os.getenv("DETECTION_SAVE_PATH", _default_save_path())).expanduser().resolve())
model_path = str(Path(os.getenv("DETECTION_MODEL_PATH", DEFAULT_MODEL_PATH)).expanduser().resolve())
model_sha256 = os.getenv("DETECTION_MODEL_SHA256", DEFAULT_MODEL_SHA256).strip().lower()
camera_id = int(os.getenv("DETECTION_CAMERA_ID", "0"))

names = {
    0: "immature",
    1: "semi-mature",
    2: "mature",
}
CH_names = ["未成熟", "半成熟", "成熟"]
