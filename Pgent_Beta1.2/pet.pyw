# %%
import sys
import math
from PyQt5.QtCore import Qt, QTimer, QPointF, QPoint,QThread,QObject, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QCursor, QIcon
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QSystemTrayIcon, QMenu, QActionGroup,QAction
from PyQt5.QtCore import QPropertyAnimation
import requests
from http import HTTPStatus
import json
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QTextEdit, QPushButton, QMessageBox,QHBoxLayout

class DesktopPet(QWidget):
    def __init__(self):
        super().__init__()
        self.last_distance_state = None
        self.chat_worker = None  # 新增成员变量
        self.setMouseTracking(True)
        self.setWindowFlags(
            Qt.FramelessWindowHint | 
            Qt.WindowStaysOnTopHint | 
            Qt.Tool |
            Qt.SubWindow
        )
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_ShowWithoutActivating, True)

        self.setAutoFillBackground(False)

        self.pet_image = QLabel(self)
        pixmap = QPixmap("./pet.png")
        self.pet_image.setPixmap(pixmap)
        self.resize(pixmap.width(), pixmap.height())
        self.original_pixmap = QPixmap("./pet.png")
        self.close_pixmap = QPixmap("./close_pet.png") 
        self.is_close_image = False
        
        # 鼠标初始化
        self.current_pos = QPointF(400.0, 300.0)
        self.target_pos = QPointF(400.0, 300.0)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_position)
        self.timer.start(20)  # 每20ms更新一次位置
        
        self.tray_icon = QSystemTrayIcon(self)
        self.tray_icon.setIcon(QIcon("./icon.png"))
        self.tray_icon.setToolTip("电子宠物")
        self.tray_icon.show()
        
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True) 
        self.move_speed = 300 

        # 菜单系统
        self.create_tray_menu()  # 系统托盘菜单
        self.create_context_menu()  # 宠物右键菜单
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.show_context_menu)

        self.chase_mode = True  # 默认追逐模式
        self.fixed_position = QPointF(400, 300)  # 固定模式默认位置

    def create_tray_menu(self):
        self.tray_menu = QMenu()
        quit_action = self.tray_menu.addAction("退出")
        quit_action.triggered.connect(self.cleanup)

        self.toggle_visibility_action = self.tray_menu.addAction("隐藏宠物")
        self.toggle_visibility_action.triggered.connect(self.toggle_visibility)

        mode_menu = self.tray_menu.addMenu("运行模式")
        self.chase_action = QAction('追逐模式', checkable=True)
        self.fixed_action = QAction('固定模式', checkable=True)
        
        mode_group = QActionGroup(self)
        mode_group.addAction(self.chase_action)
        mode_group.addAction(self.fixed_action)
        self.chase_action.setChecked(True)
        
        self.chase_action.triggered.connect(self.enable_chase_mode)
        self.fixed_action.triggered.connect(self.enable_fixed_mode)
        mode_menu.addAction(self.chase_action)
        mode_menu.addAction(self.fixed_action)

        self.tray_icon.setContextMenu(self.tray_menu)
        
        chat_action = self.tray_menu.addAction("一起来聊天")
        chat_action.triggered.connect(self.start_chat_session)

    def show_context_menu(self, pos):
        self.context_menu.exec_(self.mapToGlobal(pos))
    def create_context_menu(self):
        """宠物右键菜单"""
        self.context_menu = QMenu(self)
        self.feed_action = self.context_menu.addAction("喂食")
        self.bathe_action = self.context_menu.addAction("洗澡")
        self.context_menu.addSeparator()
        
        mode_menu = self.context_menu.addMenu("运行模式")
        self.context_chase_action = QAction('追逐模式', checkable=True)
        self.context_fixed_action = QAction('固定模式', checkable=True)

        context_mode_group = QActionGroup(self)
        context_mode_group.addAction(self.context_chase_action)
        context_mode_group.addAction(self.context_fixed_action)
        
        self.context_chase_action.setChecked(self.chase_action.isChecked())
        self.context_fixed_action.setChecked(self.fixed_action.isChecked())
        
        self.chase_action.triggered.connect(lambda: self.context_chase_action.setChecked(True))
        self.fixed_action.triggered.connect(lambda: self.context_fixed_action.setChecked(True))
        self.context_chase_action.triggered.connect(self.enable_chase_mode)
        self.context_fixed_action.triggered.connect(self.enable_fixed_mode)
        
        mode_menu.addAction(self.context_chase_action)
        mode_menu.addAction(self.context_fixed_action)
        self.context_menu.addSeparator()
        self.skin_action = self.context_menu.addAction("切换皮肤")
        
        chat_action = self.context_menu.addAction("一起来聊天")
        chat_action.triggered.connect(self.start_chat_session)

    def toggle_visibility(self):
        """切换可见性"""
        if self.isVisible():
            self.hide()
            self.toggle_visibility_action.setText("显示宠物")
        else:
            self.show()
            self.toggle_visibility_action.setText("隐藏宠物")
    def enable_chase_mode(self):
        """启用追逐模式"""
        self.chase_mode = True
        self.timer.start(20)
        self.pet_image.setPixmap(self.original_pixmap)
        
    def enable_fixed_mode(self):
        """启用固定模式"""
        self.chase_mode = False
        self.timer.stop()
        self.setCursor(Qt.ArrowCursor)
        self.fixed_position = self.pos()
        self.pet_image.setPixmap(self.close_pixmap)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, False)
    def mouseMoveEvent(self, event):
        if not self.chase_mode and event.buttons() & Qt.LeftButton:
            anim = QPropertyAnimation(self, b"pos")
            anim.setDuration(100)
            anim.setStartValue(self.pos())
            anim.setEndValue(event.globalPos() - self.drag_start_pos)
            anim.start()
    def mousePressEvent(self, event):
        """拖动起始位置"""
        if not self.chase_mode and event.button() == Qt.LeftButton:
            self.drag_start_pos = event.globalPos() - self.pos()
        super().mousePressEvent(event)
    def mouseMoveEvent(self, event):
        """固定模式拖动"""
        if not self.chase_mode and event.buttons() & Qt.LeftButton:
            delta = event.globalPos() - self.drag_start_pos
            self.move(delta.x(), delta.y())
            self.fixed_position = self.pos()
        else:
            super().mouseMoveEvent(event)
            
	
    def start_chat_session(self):
        dialog = ChatDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            prompt = dialog.get_input()
            if prompt:
                self.chat_worker = AIWorker(prompt)
                self.chat_worker.response_signal.connect(self.show_response)
                self.chat_worker.error_signal.connect(self.show_error)
                self.chat_worker.start()
    
    def show_response(self, text):
        QMessageBox.information(self, "AI回复", text)
    
    def show_error(self, error_msg):
        QMessageBox.warning(self, "连接错误", error_msg)
        self.chat_worker = None

    def cleanup(self):
        self.timer.stop() 
        self.tray_icon.hide() 
        
        self.setParent(None)
        self.deleteLater()
        
        QApplication.processEvents()
        QApplication.quit()

    def update_position(self):
        global_pos = QCursor.pos()
        self.target_pos = QPointF(
            global_pos.x() - self.width() / 2,
            global_pos.y() - self.height() / 2
        )
        
        dx = self.target_pos.x() - self.current_pos.x()
        dy = self.target_pos.y() - self.current_pos.y()
        distance = math.sqrt(dx**2 + dy**2)
        
        threshold = 30  # 触发距离p
        current_state = "close" if distance < threshold else "normal"
    
        if current_state != self.last_distance_state:
            if current_state == "close":
                self.pet_image.setPixmap(self.close_pixmap)
            else:
                self.pet_image.setPixmap(self.original_pixmap)
            self.last_distance_state = current_state

        if distance > 5:
            step_x = dx * (self.move_speed * 0.02) / distance
            step_y = dy * (self.move_speed * 0.02) / distance
            self.current_pos += QPointF(step_x, step_y)
            self.move(self.current_pos.toPoint())

# %%
from http import HTTPStatus
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QTextEdit, QPushButton, QMessageBox

class ChatDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("聊天（测试版）")
        
        layout = QVBoxLayout()
        self.input_box = QTextEdit()
        self.input_box.setPlaceholderText("输入你想说的话...")
        self.ok_btn = QPushButton("发送")
        self.cancel_btn = QPushButton("取消")
        
        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.ok_btn)
        btn_layout.addWidget(self.cancel_btn)
        
        layout.addWidget(self.input_box)
        layout.addLayout(btn_layout)
        self.setLayout(layout)
        
        self.ok_btn.clicked.connect(self.accept)
        self.cancel_btn.clicked.connect(self.reject)
    
    def get_input(self):
        return self.input_box.toPlainText().strip()

class AIWorker(QThread):
    response_signal = pyqtSignal(str)
    error_signal = pyqtSignal(str)
    
    def __init__(self, prompt):
        super().__init__()
        self.prompt = prompt
        self.api_key = r"sk-"  
        self.base_url = r"https://api.moonshot.cn/v1/chat/completions"

    def run(self):
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}" 
            }
            payload = {
                "model": "moonshot-v1-8k", 
                "messages": [
                    {"role": "system", "content": "你是 Kimi 智能助手"},
                    {"role": "user", "content": self.prompt}
                ],
                "temperature": 0.3, 
                "max_tokens": 2048
            }
            
            response = requests.post(
                self.base_url,
                headers=headers,
                data=json.dumps(payload),
                timeout=30 
            )
            
            if response.status_code == HTTPStatus.OK:
                result = response.json()
                content = result['choices'][0]['message']['content']
                self.response_signal.emit(content)
            else:
                error_msg = f"API错误[{response.status_code}]: {response.text}"
                raise Exception(error_msg)
                
        except Exception as e:
            self.error_signal.emit(f"连接失败：{str(e)}")

# %%
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)
    pet = DesktopPet()
    pet.show()
    app.exec_() 



