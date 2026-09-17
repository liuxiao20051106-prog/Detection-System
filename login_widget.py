"""可复用的登录组件（暂未接入主程序，供后续需要账号体系时直接使用）。"""

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (QApplication, QLabel, QLineEdit, QMessageBox,
                             QPushButton, QVBoxLayout, QWidget)


class LoginWidget(QWidget):
    """一个可重用的登录组件，可以集成到任何 PyQt5 应用程序中。"""

    login_successful = pyqtSignal(str, str)  # 用户名, 密码
    signup_requested = pyqtSignal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        app = QApplication.instance()
        if app is not None:
            self.setFont(app.font())
        self.init_ui()

    def init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(20, 20, 20, 20)

        self.title_label = QLabel(self.tr('用户登录'), self)
        self.title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.title_label)

        self.username_input = QLineEdit(self)
        self.username_input.setPlaceholderText(self.tr('请输入用户名'))
        self.username_input.setMinimumHeight(35)
        layout.addWidget(self.username_input)

        self.password_input = QLineEdit(self)
        self.password_input.setPlaceholderText(self.tr('请输入密码'))
        self.password_input.setEchoMode(QLineEdit.Password)
        self.password_input.setMinimumHeight(35)
        self.password_input.returnPressed.connect(self.handle_login)
        layout.addWidget(self.password_input)

        self.login_btn = QPushButton(self.tr('登录'), self)
        self.login_btn.setMinimumHeight(35)
        self.login_btn.clicked.connect(self.handle_login)
        layout.addWidget(self.login_btn)

        self.signup_btn = QPushButton(self.tr('还没有账号？点击注册'), self)
        self.signup_btn.setFlat(True)
        self.signup_btn.setCursor(Qt.PointingHandCursor)
        self.signup_btn.clicked.connect(self.handle_signup)
        layout.addWidget(self.signup_btn)

        self.setMinimumWidth(300)

    def handle_login(self) -> None:
        username = self.username_input.text().strip()
        password = self.password_input.text().strip()

        if not username or not password:
            QMessageBox.warning(self, self.tr('错误'), self.tr('请填写所有字段'))
            return

        self.login_successful.emit(username, password)

    def handle_signup(self) -> None:
        self.signup_requested.emit()

    def clear_fields(self) -> None:
        """清空输入字段。"""
        self.username_input.clear()
        self.password_input.clear()

    def set_login_button_text(self, text: str) -> None:
        self.login_btn.setText(text)

    def set_title_text(self, text: str) -> None:
        self.title_label.setText(text)
