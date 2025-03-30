import sys
from PyQt5.QtWidgets import QApplication
from ui.login_window import LoginWindow
from ui.main_window import MainWindow
import logging


def handle_exception(exc_type, exc_value, exc_traceback):
    logging.critical("未捕获异常", exc_info=(exc_type, exc_value, exc_traceback))
    sys.exit(1)


if __name__ == "__main__":
    # 配置全局异常处理
    sys.excepthook = handle_exception

    app = QApplication(sys.argv)
    try:
        # login = LoginWindow()
        # if login.exec_() == 1:  # QDialog.Accepted
            logging.info("登录成功，初始化主窗口...")
            main_win = MainWindow()
            main_win.show()
            sys.exit(app.exec_())
    except Exception as e:
        logging.critical(f"主程序崩溃: {str(e)}")
        sys.exit(1)