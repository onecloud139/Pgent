import sys
import math
from PyQt5.QtCore import Qt, QTimer, QPointF, QSize
from PyQt5.QtGui import QCursor, QIcon, QMovie  # 移除 QGraphicsDropShadowEffect
from PyQt5.QtWidgets import (
    QWidget, QLabel, QSystemTrayIcon, QMenu, QAction, 
    QActionGroup, QMessageBox, QGraphicsDropShadowEffect  # 修正：从这里导入
)
from config import CONFIG
from chat_dialog import ChatDialog
from agent_worker import RemoteAgentWorker

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
        """初始化动画"""
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
        """初始化移动逻辑"""
        self.chase_mode = True
        self.dragging = False
        self.last_pos = QPointF(*self.pet_cfg["initial_pos"])
        self.target_pos = QPointF(*self.pet_cfg["initial_pos"])

        self.timer = QTimer(self)
        self.timer.setInterval(self.pet_cfg["timer_interval"])
        self.timer.timeout.connect(self.update_pos)
        self.timer.start()

    def _init_tray(self):
        """初始化系统托盘"""
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
        """切换可见性"""
        if self.isVisible():
            self.hide()
            self.toggle_act.setText("显示宠物")
        else:
            self.show()
            self.toggle_act.setText("隐藏宠物")

    def switch_mode(self, is_chase):
        """切换运行模式"""
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
        """显示右键菜单"""
        self.tray_icon.contextMenu().exec_(self.mapToGlobal(pos))

    def mousePressEvent(self, e):
        """鼠标按下事件"""
        if e.button() == Qt.LeftButton and not self.chase_mode:
            self.dragging = True
            self.drag_pos = e.globalPos() - self.pos()
            e.accept()

    def mouseMoveEvent(self, e):
        """鼠标移动事件"""
        if self.dragging and not self.chase_mode:
            self.move(e.globalPos() - self.drag_pos)
            e.accept()

    def mouseReleaseEvent(self, e):
        """鼠标释放事件"""
        if e.button() == Qt.LeftButton:
            self.dragging = False
            self.last_pos = QPointF(self.pos().x(), self.pos().y())
            e.accept()

    def open_chat(self):
        """打开聊天对话框"""
        chat_dialog = ChatDialog(self)
        chat_dialog.exec_()

    def on_agent_response(self, content, new_history, chat_dialog):
        """处理Agent响应"""
        chat_dialog.set_busy(False)
        self.chat_history = new_history
        QMessageBox.information(self, "金融顾问回复", content)
        chat_dialog.clear_input()

    def on_agent_error(self, error_msg, chat_dialog):
        """处理错误"""
        chat_dialog.set_busy(False)
        QMessageBox.critical(self, "请求失败", error_msg)

    def quit_app(self):
        """退出应用"""
        self.timer.stop()
        self.tray_icon.hide()
        from PyQt5.QtWidgets import QApplication
        QApplication.quit()