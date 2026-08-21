from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class LoginWidget(QWidget):
    """
    一个可重用的登录组件，可以集成到任何PyQt5应用程序中。
    发出登录成功和注册请求的信号。
    """

    # 定义自定义信号
    login_successful = pyqtSignal(str, str)  # 用户名, 密码
    signup_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        # 使用应用程序的默认字体
        self.setFont(QApplication.instance().font())
        self.init_ui()

    def init_ui(self):
        # 创建主布局
        layout = QVBoxLayout()
        layout.setSpacing(10)
        layout.setContentsMargins(20, 20, 20, 20)

        # 标题
        title = QLabel("用户登录")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # 用户名输入
        self.username_input = QLineEdit()
        self.username_input.setPlaceholderText("请输入用户名")
        self.username_input.setMinimumHeight(35)
        layout.addWidget(self.username_input)

        # 密码输入
        self.password_input = QLineEdit()
        self.password_input.setPlaceholderText("请输入密码")
        self.password_input.setEchoMode(QLineEdit.Password)
        self.password_input.setMinimumHeight(35)
        layout.addWidget(self.password_input)

        # 登录按钮
        self.login_btn = QPushButton("登录")
        self.login_btn.setMinimumHeight(35)
        self.login_btn.clicked.connect(self.handle_login)
        layout.addWidget(self.login_btn)

        # 注册链接
        self.signup_btn = QPushButton("还没有账号？点击注册")
        self.signup_btn.setFlat(True)
        self.signup_btn.setCursor(Qt.PointingHandCursor)
        self.signup_btn.clicked.connect(self.handle_signup)
        layout.addWidget(self.signup_btn)

        # 设置布局
        self.setLayout(layout)

        # 设置最小尺寸
        self.setMinimumWidth(300)

    def handle_login(self):
        username = self.username_input.text().strip()
        password = self.password_input.text().strip()

        # 基本验证
        if not username or not password:
            QMessageBox.warning(self, "错误", "请填写所有字段")
            return

        # 发出登录信号
        self.login_successful.emit(username, password)

    def handle_signup(self):
        # 发出注册信号
        self.signup_requested.emit()

    def clear_fields(self):
        """清空输入字段"""
        self.username_input.clear()
        self.password_input.clear()

    def set_login_button_text(self, text):
        """自定义登录按钮文本"""
        self.login_btn.setText(text)

    def set_title_text(self, text):
        """自定义标题文本"""
        self.findChild(QLabel).setText(text)
