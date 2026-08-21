# 贡献指南

## 开发环境

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

## 提交前检查

```powershell
$env:QT_QPA_PLATFORM = "offscreen"
python -m unittest discover -s tests -v
python -m ruff check .
python -m pip_audit -r requirements.txt --strict
```

修改界面资源后运行：

```powershell
pyrcc5 UIProgram\ui_sources.qrc -o UIProgram\ui_sources_rc.py
```

新功能需附带回归测试。不要提交输出文件、运行缓存、IDE 配置、未授权数据或来源不明的权重。更换默认模型时，必须更新 `Config.py` 中的 SHA-256、`MODEL_CARD.md` 和对应测试。

## 构建 Windows 可执行文件

```powershell
python -m pip install -r requirements-build.txt
pyinstaller --noconfirm --clean DetectionSystem.spec
dist\Detection-System.exe --self-test
```

产物位于 `dist/Detection-System.exe`。分发前必须完成 PyQt5、Ultralytics、模型和训练数据的授权审查。
