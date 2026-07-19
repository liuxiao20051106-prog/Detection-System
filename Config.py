"""项目运行配置。"""

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent

# 可通过环境变量覆盖，便于部署到不同机器。
save_path = str(Path(os.getenv("DETECTION_SAVE_PATH", PROJECT_ROOT / "save_data")).expanduser().resolve())
model_path = str(Path(os.getenv("DETECTION_MODEL_PATH", PROJECT_ROOT / "models" / "best.pt")).expanduser().resolve())
font_path = str(PROJECT_ROOT / "Font" / "platech.ttf")

names = {
    0: 'immature',
    1: 'semi-mature',
    2: 'mature'
}
CH_names = [
    '未成熟',
    '半成熟',
    '成熟'
]
