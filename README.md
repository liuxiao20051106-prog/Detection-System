# 葡萄成熟度检测系统

基于 YOLOv8 与 PyQt5 的桌面检测工具，可识别图片、视频和摄像头画面中的葡萄，并将目标分为“未成熟、半成熟、成熟”三类。

当前代码经过一轮结构性重构：**推理、绘框和写盘全部搬到后台线程**，界面线程只做渲染与交互；同时补齐了参数集中管理、资源释放、日志与自动化测试。

## 功能

- 单张图片检测、文件夹批量检测
- 视频文件和摄像头实时检测
- 置信度、IoU 阈值与标签显示的实时开关
- 结果图片、结果视频与 CSV 数据的导出
- 检测过程全程可响应：批量 / 视频导出任务可随时取消

## 环境要求

- Python 3.9 或 3.10（推荐 3.10）
- Windows 10/11（图形界面主要在 Windows 下验证）
- 可选：NVIDIA GPU 与匹配的 CUDA 环境；没有 GPU 时会自动使用 CPU
- 界面进程不再直接依赖 torch，GPU 由 ultralytics 自动选择

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
2. 根据需要调整置信度和 IoU 阈值（松手约 0.25 秒后自动重算，不需要再点按钮）。
3. 在右侧查看类别、置信度和坐标信息，或用下拉框单独查看某个目标。
4. 使用保存按钮导出检测结果：图片会同时导出一份 CSV，文件夹会导出全部结果图与汇总 CSV，视频会另存为检测结果视频。

首次推理通常会比后续推理慢。摄像头无法打开时，请先确认没有被其他软件占用。

## 项目结构

```text
MainProgram.py       主窗口：只负责界面状态与信号编排
Config.py            模型、路径、类别与刷新参数的唯一来源
detect_tools.py      图像读写、检测框绘制、单帧结果数据结构
workers.py           后台线程：图片/批量检测、逐帧推理、共用的模型调用锁
UIProgram/           Qt 界面、样式和资源
scripts/demo_detect.py  命令行版检测（图片 / 视频 / 摄像头 / 文件夹）
models/best.pt       训练后的检测模型
TestFiles/           示例图片
train.py             训练入口，支持命令行参数
tests/               单元测试（不需要 torch / ultralytics）
```

## 架构与性能要点

**线程分工**（最关键的一条）：

- 界面线程：取视频帧、渲染画面、写结果文件；
- 后台线程：YOLO 推理与绘制检测框。

视频帧的采集刻意留在主线程——OpenCV 的视频后端（Windows 的 MSMF / DirectShow）对“在哪个线程里初始化”很敏感，
放进 QThread 容易出现卡死或设备打不开；而取帧本身开销很小，用 QTimer 驱动即可。
真正耗时的推理仍然全部在后台，所以窗口不会卡，导出任务也能边跑边看进度。

| 主题 | 旧实现 | 现在 |
| --- | --- | --- |
| 推理位置 | 界面线程直接调用 YOLO，大图/批量/视频时窗口卡死 | `workers.py` 中的 QThread，界面始终可响应 |
| 重复代码 | 同一套“检测 -> 解析 -> 刷新控件”逻辑复制了 4 份 | 收敛为一条流水线，保存时复用已检测结果 |
| 重复的推理 | 文件夹保存时会把每张图再推理一遍 | `persist` 模式一次完成检测与写盘，并顺带输出 CSV |
| 跨线程数据 | 界面长期持有 ultralytics `Results`（torch 张量） | 转换为纯 Python 的 `DetectionResult` 后立即释放张量 |
| 绘框 | 每个目标一次 PIL 转换，且依赖 `results.plot()` 的英文标签 | 一次 PIL 转换画完所有框，中文标签 + 置信度 |
| 视频播放节奏 | `QTimer(1ms)` 全速播放，CPU 拉满 | 按视频自身帧率取帧，摄像头按 30 FPS |
| 列表增长 | 每帧往表格追加行，长时间录像时内存只增不减 | 表格按 400ms 节流刷新并在 5000 行封顶 |
| 帧渲染 | 推理多快就渲染多快，消息队列被刷爆 | 33ms 定时器抽帧渲染（约 30 FPS），中间帧自动丢弃 |
| 资源释放 | 摄像头关闭不置空、异常时不释放 VideoWriter | 统一释放；关闭窗口时会等待后台线程退出 |
| 错误处理 | 启动时缺模型直接抛栈 | 弹窗提示 + logging 输出到控制台 |

## 自检

```powershell
# 开发依赖（不含 torch / ultralytics，秒级完成）
pip install -r requirements-dev.txt

python -m compileall -q .
python -m flake8 .
$env:QT_QPA_PLATFORM = "offscreen"
python -m unittest discover -s tests -v
```

`tests/` 全部使用桩对象替身 YOLO，因此跑测试不需要 GPU、模型推理依赖和显示器。
GitHub Actions（`.github/workflows/tests.yml`）在 Python 3.9 / 3.10 上执行同一套检查。

命令行快速验证：

```powershell
python scripts/demo_detect.py --source dir --input TestFiles --save
python scripts/demo_detect.py --source camera --input 0
```

## 常见问题

- **提示找不到模型**：确认 `models/best.pt` 存在，或设置 `DETECTION_MODEL_PATH`。
- **安装 PyTorch 失败**：根据操作系统和 CUDA 版本，先从 PyTorch 官方安装向导安装匹配版本，再执行依赖安装。
- **界面文字或图标异常**：请保留 `Font/` 和 `UIProgram/` 目录，不要只复制 `MainProgram.py`。
- **检测没有反应或很慢**：首次调用会初始化模型；可在 `save_data/` 之外查看日志输出，或直接跑 `scripts/demo_detect.py` 排查是模型还是界面的问题。

## 模型说明

默认模型包含三个类别：`immature`、`semi-mature`、`mature`。替换模型时，请同步修改 `Config.py` 中的 `CLASSES`，
`names` 与 `CH_names` 会由它自动派生，不会出现“英文表多了、中文表没改”的错位。

本项目使用 Ultralytics YOLO，请在分发或商业使用前确认其许可证以及模型、数据集的授权条件。
