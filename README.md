# 葡萄成熟度检测系统

[English](README.en.md)

这是一个基于 Ultralytics YOLO 和 PyQt5 的 Windows 桌面工具，用于在图片、文件夹、视频和摄像头画面中检测葡萄，并分为“未成熟、半成熟、成熟”三类。

## 主要功能

- 单图检测、文件夹批量检测、视频和摄像头实时预览。
- 调整置信度、IoU 阈值和标签显示；单图调整采用防抖重检测。
- 图片和批量结果导出标注图与 CSV，保存批量结果时不重复推理。
- 视频导出可取消；部分文件会自动删除，只在关闭编码器并重新验证后报告成功。
- 推理在后台线程运行，每个线程独占模型，界面事件循环不被阻塞。
- 模型 SHA-256、类别映射、媒体解码/尺寸/时长和 CSV 公式注入防护。
- 界面可缩放，小屏使用滚动区；表格最多保留 500 行，避免长视频无限占用内存。

## 快速开始

建议使用 Windows 10/11 和 CPython 3.10–3.12。CPU 可直接运行；NVIDIA GPU 用户应先按 PyTorch 官方安装向导选择匹配的 CUDA 组合。

```powershell
git clone https://github.com/liuxiao20051106-prog/Detection-System.git
cd Detection-System
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python MainProgram.py
```

也可运行 `python installPackages.py` 使用当前 Python 解释器安装依赖。结果默认保存到 `%LOCALAPPDATA%\Detection-System\results`。

## 使用方法

1. 启动后选择图片、文件夹、视频或摄像头。
2. 根据需要调整置信度、IoU 和标签显示。
3. 在画面和右侧表格查看类别、置信度与坐标。
4. 点击保存导出结果。同名文件会生成递增后缀，不覆盖原结果。

### 命令行工具

```powershell
python imgTest.py "D:\images\grape.jpg"
python VideoTest.py "D:\videos\grape.mp4"
python CameraTest.py --camera 0
python train.py --data "D:\dataset\data.yaml" --epochs 120 --batch 16
```

## 配置

```powershell
$env:DETECTION_MODEL_PATH = "D:\models\best.pt"
$env:DETECTION_MODEL_SHA256 = "<对应的64位 SHA-256>"
$env:DETECTION_SAVE_PATH = "D:\detection-results"
$env:DETECTION_CAMERA_ID = "0"
python MainProgram.py
```

默认模型为 `models/best.pt`，并已在 `Config.py` 锁定哈希。更换模型时必须提供新哈希，且类别顺序必须与 `Config.py` 一致。

> 安全提醒：PyTorch `.pt` 模型可包含可执行的序列化对象。哈希校验只能证明文件与给定哈希一致，不能证明来源可信。不要加载来源不明的权重。

## 项目结构

```text
MainProgram.py       桌面界面与任务调度
workers.py           独立模型的后台检测/导出线程
detection_core.py    输入校验、结果模型和安全导出
Config.py            模型、哈希、输出路径和类别配置
UIProgram/           Qt 界面、样式和内嵌资源
models/best.pt       默认推理权重
tests/               核心、线程、界面和导出回归测试
docs/decisions/      架构决策记录
```

## 验证

```powershell
$env:QT_QPA_PLATFORM = "offscreen"
python -m unittest discover -s tests -v
python -m ruff check .
python -m pip_audit -r requirements.txt --strict
```

CI 在 Windows 上对 Python 3.10 和 3.12 运行上述检查。打标签 `v*` 或手动启动发布工作流可生成 Windows 可执行产物。

## 已知限制

- 导出视频不保留原音频，界面会在导出前提示。
- 仓库中的示例图只用于演示，不是标注测试集。
- 训练数据未随仓库提供，无法仅依据本仓库独立复现模型指标。
- 检测结果不应成为食品安全、农业采收、质量认证或其他高影响决策的唯一依据。

详细阅读 [MODEL_CARD.md](MODEL_CARD.md)、[SECURITY.md](SECURITY.md) 和 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

## 授权状态

当前仓库没有项目所有者声明的 `LICENSE` 文件。在所有者明确选择许可证之前，请不要假定项目代码可自由再分发或商用。同时请单独审查 PyQt5、Ultralytics、视频编解码器、模型和训练数据的授权条件。
