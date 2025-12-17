# %%
import sys
import math
import json
import requests
from http import HTTPStatus
from PyQt5.QtGui import QCursor, QIcon, QMovie
from PyQt5.QtWidgets import (
    QApplication, QWidget, QSystemTrayIcon, QMenu, 
    QActionGroup, QAction, QLabel, QDialog, QMessageBox,
    QVBoxLayout, QHBoxLayout, QTextEdit, QPushButton, QDialogButtonBox
)
from PyQt5.QtCore import (
    Qt, QTimer, QPointF, QThread, pyqtSignal, QSize, QMetaObject, Qt
)
def load_config():
    try:
        with open("config.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"配置加载失败：{e}")
        sys.exit(1)

CONFIG = load_config()

class RemoteAgentWorker(QThread):
    response_signal = pyqtSignal(str, list)
    error_signal = pyqtSignal(str)

    def __init__(self, prompt, history=[]):
        super().__init__()
        self.prompt = prompt
        self.history = history
        self.api_config = CONFIG["remote_agent"]

    def run(self):
        try:
            url = self.api_config["api_url"]
            headers = self.api_config["headers"]
            timeout = self.api_config["timeout"]
            
            request_data = {
                self.api_config["request_key"]: self.prompt,
                self.api_config["history_key"]: self.history
            }

            response = requests.post(
                url,
                headers=headers,
                json=request_data,
                timeout=timeout
            )

            if response.status_code == HTTPStatus.OK:
                result = response.json()
                response_keys = self.api_config["response_key"].split(".")
                content = result
                for k in response_keys:
                    content = content.get(k)
                    if content is None:
                        raise Exception(f"响应字段缺失：{k}")
                new_history = result.get("data", {}).get("history", [])
                self.response_signal.emit(str(content), new_history)
            else:
                raise Exception(f"接口错误 [{response.status_code}]：{response.text}")

        except Exception as e:
            self.error_signal.emit(f"请求失败：{str(e)}")

class LoadingDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("加载中")
        self.setFixedSize(200, 100)
        self.setModal(False)
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint)

        layout = QVBoxLayout()
        label = QLabel("正在请求金融顾问Agent，请稍候...")
        label.setAlignment(Qt.AlignCenter)
        layout.addWidget(label)
        self.setLayout(layout)
class ChatDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("和金融宠物聊天")
        self.setFixedSize(400, 300)

        layout = QVBoxLayout()
        self.input_box = QTextEdit()
        self.input_box.setPlaceholderText("输入你的问题，比如：什么是分散投资？")
        layout.addWidget(self.input_box)

        btn_layout = QHBoxLayout()
        self.send_btn = QPushButton("发送")
        self.cancel_btn = QPushButton("取消")
        btn_layout.addWidget(self.send_btn)
        btn_layout.addWidget(self.cancel_btn)
        layout.addLayout(btn_layout)
        self.setLayout(layout)

        self.is_busy = False
        self.loading_dialog = None

        self.send_btn.clicked.connect(self.on_send)
        self.cancel_btn.clicked.connect(self.reject)

    def on_send(self):
        """仅非忙碌状态下触发发送"""
        if not self.is_busy:
            self.accept()

    def get_input_text(self):
        """获取输入内容"""
        return self.input_box.toPlainText().strip()

    def clear_input(self):
        """清空输入并聚焦"""
        self.input_box.clear()
        self.input_box.setFocus()

    def set_busy(self, busy):
        """设置忙碌状态，控制按钮和加载框"""
        self.is_busy = busy
        self.send_btn.setEnabled(not busy)
        self.cancel_btn.setEnabled(not busy)

        if busy:
            self.loading_dialog = LoadingDialog(self)
            self.loading_dialog.move(self.x() + 100, self.y() + 50)
            self.loading_dialog.show()
        else:
            if self.loading_dialog and self.loading_dialog.isVisible():
                QMetaObject.invokeMethod(self.loading_dialog, "close", Qt.QueuedConnection)
                self.loading_dialog = None

class DesktopPet(QWidget):
    def __init__(self):
        super().__init__()
        self.pet_cfg = CONFIG["pet"]
        self.agent_cfg = CONFIG["remote_agent"]
        self.chat_history = []

        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setFixedSize(*self.pet_cfg["size"])
        self.move(*self.pet_cfg["initial_pos"])
        self._init_animation()
        self._init_movement()
        self._init_tray()
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.show_right_menu)

    def _init_animation(self):
        self.pet_label = QLabel(self)
        self.pet_label.setGeometry(0, 0, *self.pet_cfg["size"])
        self.normal_movie = QMovie(self.pet_cfg["original_gif"])
        self.close_movie = QMovie(self.pet_cfg["close_gif"])
        
        for movie in [self.normal_movie, self.close_movie]:
            movie.setScaledSize(QSize(*self.pet_cfg["size"]))
            movie.setCacheMode(QMovie.CacheAll)
            movie.start()
        
        self.close_movie.setPaused(True)
        self.pet_label.setMovie(self.normal_movie)

    def _init_movement(self):
        self.chase_mode = True
        self.dragging = False
        self.last_pos = QPointF(*self.pet_cfg["initial_pos"])
        self.target_pos = QPointF(*self.pet_cfg["initial_pos"])

        self.timer = QTimer(self)
        self.timer.setInterval(self.pet_cfg["timer_interval"])
        self.timer.timeout.connect(self.update_pos)
        self.timer.start()

    def _init_tray(self):
        self.tray_icon = QSystemTrayIcon(QIcon(self.pet_cfg["tray_icon"]), self)
        self.tray_icon.setToolTip("金融投资顾问宠物")

        tray_menu = QMenu()
        self.toggle_act = QAction("隐藏宠物", self)
        self.toggle_act.triggered.connect(self.toggle_visible)
        tray_menu.addAction(self.toggle_act)
        mode_menu = QMenu("运行模式", self)
        self.chase_act = QAction("追逐模式", self, checkable=True, checked=True)
        self.fixed_act = QAction("固定模式", self, checkable=True)
        mode_group = QActionGroup(self)
        mode_group.addAction(self.chase_act)
        mode_group.addAction(self.fixed_act)
        self.chase_act.triggered.connect(lambda: self.switch_mode(True))
        self.fixed_act.triggered.connect(lambda: self.switch_mode(False))
        mode_menu.addActions([self.chase_act, self.fixed_act])
        tray_menu.addMenu(mode_menu)
        chat_act = QAction("一起来聊天", self)
        chat_act.triggered.connect(self.open_chat)
        tray_menu.addAction(chat_act)
        exit_act = QAction("退出", self)
        exit_act.triggered.connect(self.quit_app)
        tray_menu.addAction(exit_act)

        self.tray_icon.setContextMenu(tray_menu)
        self.tray_icon.show()
    def toggle_visible(self):
        if self.isVisible():
            self.hide()
            self.toggle_act.setText("显示宠物")
        else:
            self.show()
            self.toggle_act.setText("隐藏宠物")

    def switch_mode(self, is_chase):
        self.chase_mode = is_chase
        if is_chase:
            self.timer.start()
            self.normal_movie.setPaused(False)
            self.close_movie.setPaused(True)
            self.pet_label.setMovie(self.normal_movie)
        else:
            self.timer.stop()
            self.normal_movie.setPaused(True)
            self.close_movie.setPaused(False)
            self.pet_label.setMovie(self.close_movie)
            self.last_pos = QPointF(self.pos().x(), self.pos().y())

    def update_pos(self):
        """追逐模式更新位置"""
        if not self.chase_mode:
            return
        mouse_pos = QCursor.pos()
        target_x = mouse_pos.x() - self.pet_cfg["size"][0]/2
        target_y = mouse_pos.y() - self.pet_cfg["size"][1]/2
        self.target_pos = QPointF(target_x, target_y)

        dx = self.target_pos.x() - self.last_pos.x()
        dy = self.target_pos.y() - self.last_pos.y()
        distance = math.hypot(dx, dy)

        if distance < self.pet_cfg["distance_threshold"]:
            if self.pet_label.movie() != self.close_movie:
                self.normal_movie.setPaused(True)
                self.close_movie.setPaused(False)
                self.pet_label.setMovie(self.close_movie)
        else:
            if self.pet_label.movie() != self.normal_movie:
                self.close_movie.setPaused(True)
                self.normal_movie.setPaused(False)
                self.pet_label.setMovie(self.normal_movie)
        if distance > 5:
            step = self.pet_cfg["move_speed"] * 0.02
            self.last_pos += QPointF(dx*step/distance, dy*step/distance)
            self.move(int(self.last_pos.x()), int(self.last_pos.y()))

    def show_right_menu(self, pos):
        self.tray_icon.contextMenu().exec_(self.mapToGlobal(pos))

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton and not self.chase_mode:
            self.dragging = True
            self.drag_pos = e.globalPos() - self.pos()
            e.accept()

    def mouseMoveEvent(self, e):
        if self.dragging and not self.chase_mode:
            self.move(e.globalPos() - self.drag_pos)
            e.accept()

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.LeftButton:
            self.dragging = False
            self.last_pos = QPointF(self.pos().x(), self.pos().y())
            e.accept()
    def open_chat(self):
        chat_dialog = ChatDialog(self)
        
        while True:
            result = chat_dialog.exec_()
            if result != QDialog.Accepted:
                break
            input_text = chat_dialog.get_input_text()
            if not input_text:
                QMessageBox.warning(self, "提示", "请输入你的问题！")
                continue
            chat_dialog.set_busy(True)
            worker = RemoteAgentWorker(input_text, self.chat_history)
            worker.response_signal.connect(lambda c, h: self.on_agent_response(c, h, chat_dialog))
            worker.error_signal.connect(lambda e: self.on_agent_error(e, chat_dialog))
            worker.start()

    def on_agent_response(self, content, new_history, chat_dialog):
        chat_dialog.set_busy(False)
        self.chat_history = new_history
        QMessageBox.information(self, "金融顾问回复", content)
        chat_dialog.clear_input()

    def on_agent_error(self, error_msg, chat_dialog):
        chat_dialog.set_busy(False)
        QMessageBox.critical(self, "请求失败", error_msg)

    def quit_app(self):
        self.timer.stop()
        self.tray_icon.hide()
        QApplication.quit()
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)
    pet = DesktopPet()
    pet.show()
    sys.exit(app.exec_())


