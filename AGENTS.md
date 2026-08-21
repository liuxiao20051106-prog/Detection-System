# Codex 项目指南

## 修改边界

- 保留现有 PyQt5 桌面应用形态和三类葡萄成熟度语义。
- 推理不得在 Qt 主线程执行，YOLO 实例不得跨线程共享。
- 不得绕过模型 SHA-256、类别映射或媒体输入校验。
- 不得覆盖已有输出；取消的导出不得留下部分文件。
- 不要替项目所有者猜测或添加项目许可证。

## 验收

所有代码变更至少运行：

```powershell
$env:QT_QPA_PLATFORM = "offscreen"
python -m unittest discover -s tests -v
python -m ruff check .
```

依赖变更额外运行 `python -m pip_audit -r requirements.txt --strict`。界面资源变更后重新生成 `UIProgram/ui_sources_rc.py` 并验证从仓库外的工作目录启动。
