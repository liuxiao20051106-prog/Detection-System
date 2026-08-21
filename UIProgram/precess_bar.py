# -*- coding: utf-8 -*-
# 进度条
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QDialog, QHBoxLayout, QLabel, QProgressBar, QPushButton, QVBoxLayout


class ProgressBar(QDialog):
    cancel_requested = pyqtSignal()

    def __init__(self, parent=None):
        super(ProgressBar, self).__init__(parent)

        self._allow_close = False
        self._cancel_emitted = False

        self.resize(350, 100)
        self.setWindowTitle(self.tr("视频保存进度信息"))

        self.TipLabel = QLabel(self.tr("当前帧/总帧数:0/0"))
        self.FeatLabel = QLabel(self.tr("保存进度:"))

        self.FeatProgressBar = QProgressBar(self)
        self.FeatProgressBar.setMinimum(0)
        self.FeatProgressBar.setMaximum(100)  # 总进程换算为100
        self.FeatProgressBar.setValue(0)  # 进度条初始值为0

        TipLayout = QHBoxLayout()
        TipLayout.addWidget(self.TipLabel)

        FeatLayout = QHBoxLayout()
        FeatLayout.addWidget(self.FeatLabel)
        FeatLayout.addWidget(self.FeatProgressBar)

        self.cancelButton = QPushButton("取消保存", self)

        buttonlayout = QHBoxLayout()
        buttonlayout.addStretch(1)
        buttonlayout.addWidget(self.cancelButton)

        layout = QVBoxLayout()
        layout.addLayout(FeatLayout)
        layout.addLayout(TipLayout)
        layout.addLayout(buttonlayout)
        self.setLayout(layout)
        self.cancelButton.clicked.connect(self.onCancel)
        # self.show()

    def setValue(self, start, end):
        self.TipLabel.setText(self.tr("当前帧/总帧数:" + "   " + str(start) + "/" + str(end)))
        if end <= 0:
            self.FeatProgressBar.setRange(0, 0)
            return
        self.FeatProgressBar.setRange(0, 100)
        self.FeatProgressBar.setValue(min(100, int(start * 100 / end)))

    def onCancel(self):
        if self._cancel_emitted:
            return
        self._cancel_emitted = True
        self.cancelButton.setEnabled(False)
        self.cancelButton.setText(self.tr("正在取消…"))
        self.cancel_requested.emit()

    def finish(self):
        self._allow_close = True
        self.close()

    def closeEvent(self, event):
        if self._allow_close:
            event.accept()
            return
        self.onCancel()
        event.ignore()
