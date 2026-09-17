"""QSS 样式加载器。"""

from pathlib import Path


class QSSLoader:
    """读取 QSS 文件；文件缺失时返回空样式而不是让程序崩溃。"""

    @staticmethod
    def read_qss_file(qss_file_name) -> str:
        path = Path(qss_file_name)
        if not path.is_file():
            return ''
        return path.read_text(encoding='utf-8')
