# -*- coding: utf-8 -*-
"""任务进度对话框。"""

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QDialog, QHBoxLayout, QLabel, QProgressBar, QPushButton, QVBoxLayout


class ProgressBar(QDialog):
    """显示“当前项/总数 + 百分比”的进度条，支持点击取消回调终止后台任务。"""

    def __init__(self, parent=None, title: str = '视频保存进度信息', on_cancel=None) -> None:
        super().__init__(parent)
        self._on_cancel = on_cancel

        self.resize(350, 100)
        self.setWindowTitle(self.tr(title))
        if parent is not None:
            self.setWindowModality(Qt.WindowModal)

        self.TipLabel = QLabel(self.tr('当前帧/总帧数：0/0'))
        self.FeatLabel = QLabel(self.tr('保存进度：'))

        self.FeatProgressBar = QProgressBar(self)
        self.FeatProgressBar.setMinimum(0)
        self.FeatProgressBar.setMaximum(100)
        self.FeatProgressBar.setValue(0)

        tip_layout = QHBoxLayout()
        tip_layout.addWidget(self.TipLabel)

        feat_layout = QHBoxLayout()
        feat_layout.addWidget(self.FeatLabel)
        feat_layout.addWidget(self.FeatProgressBar)

        self.cancelButton = QPushButton(self.tr('取消保存'), self)
        self.cancelButton.clicked.connect(self.onCancel)

        button_layout = QHBoxLayout()
        button_layout.addStretch(1)
        button_layout.addWidget(self.cancelButton)

        layout = QVBoxLayout(self)
        layout.addLayout(feat_layout)
        layout.addLayout(tip_layout)
        layout.addLayout(button_layout)

    def setValue(self, start, end, progress: int) -> None:
        self.TipLabel.setText(self.tr('当前帧/总帧数：   %s/%s') % (start, end))
        self.FeatProgressBar.setValue(max(0, min(100, int(progress))))

    def reset(self) -> None:
        self.TipLabel.setText(self.tr('当前帧/总帧数：0/0'))
        self.FeatProgressBar.setValue(0)

    def onCancel(self) -> None:
        """点击取消时通知调用方停止线程。"""
        if self._on_cancel is not None:
            self._on_cancel()
        self.close()
