# 葡萄成熟度检测系统

这是一个基于 YOLOv8 和 PyQt5 的桌面检测工具，可识别图片、视频和摄像头画面中的葡萄，并将目标分为“未成熟、半成熟、成熟”三类。

## 功能

- 单张图片检测与结果查看
- 文件夹批量检测
- 视频文件和摄像头实时检测
- 置信度、IoU 阈值及标签显示控制
- 检测结果图片、视频和 CSV 数据保存

## 环境要求

- Python 3.9 或 3.10（推荐 3.10）
- Windows 10/11（图形界面主要在 Windows 下验证）
- 可选：NVIDIA GPU 与匹配的 CUDA 环境；没有 GPU 时会自动使用 CPU

## 快速开始

```powershell
git clone https://github.com/liuxiao20051106-prog/Detection-System.git
cd Detection-System
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python MainProgram.py
```

也可以运行 `python installPackages.py` 安装依赖。该脚本始终使用当前 Python 解释器对应的 pip，不会调用错误的全局环境。

仓库默认从 `models/best.pt` 加载模型，并将结果保存到 `save_data/`。从其他目录启动程序也可以正常定位资源。

如需使用其他模型或保存目录，可设置环境变量：

```powershell
$env:DETECTION_MODEL_PATH = "D:\models\best.pt"
$env:DETECTION_SAVE_PATH = "D:\detection-results"
python MainProgram.py
```

## 使用说明

1. 启动程序后选择图片、视频、文件夹或摄像头。
2. 根据需要调整置信度和 IoU 阈值。
3. 在右侧查看类别、置信度和坐标信息。
4. 使用保存按钮导出检测结果。

首次推理通常会比后续推理慢。摄像头无法打开时，请先确认没有被其他软件占用，并尝试更换摄像头编号。

## 项目结构

```text
MainProgram.py       主程序和桌面界面逻辑
Config.py            模型、输出路径和类别配置
detect_tools.py      图像绘制、转换与 CSV 工具
UIProgram/           Qt 界面、样式和资源
models/best.pt       训练后的检测模型
TestFiles/           示例图片
train.py             模型训练入口
tests/               基础自动检查
```

## 自检

```powershell
python -m compileall -q .
python -m unittest discover -s tests -v
```

## 常见问题

- **提示找不到模型**：确认 `models/best.pt` 存在，或设置 `DETECTION_MODEL_PATH`。
- **安装 PyTorch 失败**：根据操作系统和 CUDA 版本，先从 PyTorch 官方安装向导安装匹配版本，再执行依赖安装。
- **界面文字或图标异常**：请保留 `Font/` 和 `UIProgram/` 目录，不要只复制 `MainProgram.py`。

## 模型说明

默认模型包含三个类别：`immature`、`semi-mature`、`mature`。替换模型时，请同步修改 `Config.py` 中的类别名称和顺序。

本项目使用 Ultralytics YOLO，请在分发或商业使用前确认其许可证以及模型、数据集的授权条件。
