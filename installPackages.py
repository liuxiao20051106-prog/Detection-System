"""使用当前 Python 环境安装项目依赖。"""

import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    requirements = Path(__file__).resolve().with_name("requirements.txt")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", str(requirements)],
        check=True,
    )
