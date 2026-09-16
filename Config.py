"""项目运行配置。

所有路径都基于项目根目录解析，因此从任意工作目录启动程序都不会找不到资源。
路径类配置可用环境变量覆盖，便于部署到不同机器。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent


def _resolve(env_name: str, default: Path) -> Path:
    """读取环境变量指定的路径，缺省时使用项目内的默认位置。

    环境变量给出相对路径时按项目根目录解析，避免依赖启动目录。
    """
    raw = os.getenv(env_name)
    target = Path(raw).expanduser() if raw else default
    if not target.is_absolute():
        target = PROJECT_ROOT / target
    return target.resolve()


# 类别信息的唯一来源：(类别 id, 英文名称, 中文名称)。
# Config.names / Config.CH_names 由它派生，二者永远保持同序、等长。
CLASSES: Tuple[Tuple[int, str, str], ...] = (
    (0, 'immature', '未成熟'),
    (1, 'semi-mature', '半成熟'),
    (2, 'mature', '成熟'),
)

names: Dict[int, str] = {class_id: english for class_id, english, _chinese in CLASSES}
CH_names: List[str] = [chinese for _id, _english, chinese in CLASSES]

# 模型与结果输出位置，均可通过环境变量覆盖。
save_path = str(_resolve("DETECTION_SAVE_PATH", PROJECT_ROOT / "save_data"))
model_path = str(_resolve("DETECTION_MODEL_PATH", PROJECT_ROOT / "models" / "best.pt"))
font_path = str(PROJECT_ROOT / "Font" / "platech.ttf")

# 文件选择与参数设置相关的常量集中在此，界面逻辑不再散落魔法值。
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp')
VIDEO_EXTENSIONS = ('.avi', '.mp4', '.wmv', '.mkv')
IMAGE_FILTER = "图片文件 (*.jpg *.jpeg *.png *.bmp)"
VIDEO_FILTER = "视频文件 (*.avi *.mp4 *.wmv *.mkv)"

DEFAULT_CONF_THRES = 0.25
DEFAULT_IOU_THRES = 0.45
THRES_STEP = 0.05

# 界面刷新节流参数：视频/摄像头每帧都刷新列表会拖垮 UI，这里限制刷新频率。
FRAME_RENDER_INTERVAL_MS = 33          # 约 30 FPS
STREAM_TABLE_INTERVAL_MS = 400         # 流模式下结果表刷新间隔
REDETECT_DEBOUNCE_MS = 250             # 拖动阈值滑块时的去重抖延时
MAX_TABLE_ROWS = 5000                  # 表格行数上限，防止长时间录像检测时无限增长

CSV_HEADERS = ('序号', '文件路径', '类别', '置信度', '目标框坐标')
