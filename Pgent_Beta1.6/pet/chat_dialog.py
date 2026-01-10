import json
from PyQt5.QtCore import Qt, QDateTime
from PyQt5.QtWidgets import (
    QDialog, QTextEdit, QLabel, QPushButton, QVBoxLayout, 
    QHBoxLayout, QMessageBox, QDialogButtonBox, QWidget, 
    QGraphicsDropShadowEffect
)

class ChatDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("💬 金融投资顾问")
        self.setMinimumSize(600, 450)
        self.setMaximumSize(1800, 1600)
        
        # 配置模式：True=使用模拟数据，False=调用真实服务
        self.MOCK_MODE = True  # 测试时改为True，部署时改为False
        
        self._setup_ui()
        self._connect_signals()
        self.is_busy = False
        self.loading_dialog = None
        self.load_chat_history()

    def _setup_ui(self):
        """设置UI界面"""
        main_layout = QVBoxLayout()
        
        # 历史对话显示区域 - 修正后的样式
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setMinimumHeight(200)
        self.chat_display.setAcceptRichText(True)
        self.chat_display.setHtml("")
        
        # 关键修正：简化样式表，让HTML完全控制显示
        self.chat_display.setStyleSheet("""
            QTextEdit {
                background-color: #f7f7f7;  /* 更接近微信的背景色 */
                border: 1px solid #e1e1e1;
                border-radius: 8px;
                padding: 5px;  /* 改为较小的内边距 */
                font-family: "Segoe UI", "Microsoft YaHei", "PingFang SC", "Helvetica Neue", sans-serif;
                font-size: 14px;  /* 统一字体大小 */
                line-height: 1.6;
            }
            QScrollBar:vertical {
                background-color: #f0f0f0;
                width: 12px;
                margin: 0px;
            }
            QScrollBar::handle:vertical {
                background-color: #c8c8c8;
                border-radius: 6px;
                min-height: 30px;
            }
        """)
        
        chat_scroll = QVBoxLayout()
        chat_scroll.addWidget(QLabel("💭 对话历史"))
        chat_scroll.addWidget(self.chat_display)
        
        # 输入区域
        input_label = QLabel("📝 输入问题：")
        input_label.setStyleSheet("font-weight: bold; margin-top: 5px;")
        
        self.input_box = QTextEdit()
        self.input_box.setPlaceholderText("请输入您的问题，例如：什么是分散投资？如何开始基金投资？")
        self.input_box.setMinimumHeight(80)
        self.input_box.setMaximumHeight(120)
        self.input_box.setStyleSheet("""
            QTextEdit {
                border: 1px solid #4CAF50;
                border-radius: 6px;
                padding: 10px;
                font-size: 14px;
                font-family: "Segoe UI", "Microsoft YaHei", "PingFang SC", "Helvetica Neue", sans-serif;
                background-color: white;
                selection-background-color: #c8e6c9;
            }
            QTextEdit:focus {
                border: 2px solid #45a049;
            }
        """)
        
        # 按钮区域
        button_layout = QHBoxLayout()
        
        self.clear_btn = QPushButton("🗑️ 清除历史")
        self.clear_btn.setStyleSheet("""
            QPushButton {
                background-color: #607D8B;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 8px 15px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #455A64;
            }
        """)
        
        self.cancel_btn = QPushButton("❌ 关闭")
        self.cancel_btn.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 8px 20px;
                font-size: 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #d32f2f;
            }
        """)
        
        self.send_btn = QPushButton("🚀 发送")
        self.send_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 8px 20px;
                font-size: 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        
        button_layout.addWidget(self.clear_btn)
        button_layout.addStretch()
        button_layout.addWidget(self.cancel_btn)
        button_layout.addWidget(self.send_btn)
        
        # 组装所有部件
        main_layout.addLayout(chat_scroll)
        main_layout.addWidget(input_label)
        main_layout.addWidget(self.input_box)
        main_layout.addLayout(button_layout)
        self.setLayout(main_layout)
        
        # 美化对话框
        self.setStyleSheet("""
            QDialog {
                background-color: #f9f9f9;
            }
            QLabel {
                color: #333333;
                font-family: "Segoe UI", "Microsoft YaHei", sans-serif;
            }
        """)
        
        self.setWindowFlags(self.windowFlags() | 
                          Qt.WindowMinimizeButtonHint | 
                          Qt.WindowMaximizeButtonHint |
                          Qt.WindowCloseButtonHint)
        
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(10)
        shadow.setOffset(0, 0)
        self.setGraphicsEffect(shadow)

    def _connect_signals(self):
        self.send_btn.clicked.connect(self.on_send)
        self.cancel_btn.clicked.connect(self.reject)
        self.clear_btn.clicked.connect(self.clear_history)
        self.input_box.installEventFilter(self)

    def eventFilter(self, obj, event):
        """捕获Enter键发送消息（Ctrl+Enter换行）"""
        if obj == self.input_box and event.type() == event.KeyPress:
            if event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter:
                if event.modifiers() & Qt.ControlModifier:
                    return False
                else:
                    if not self.is_busy and self.send_btn.isEnabled():
                        self.on_send()
                    return True
        return super().eventFilter(obj, event)
    
    def load_chat_history(self):
        """加载聊天历史"""
        if not self.parent():
            return
            
        for message in self.parent().chat_history:
            if isinstance(message, dict) and 'content' in message:
                # 修复：直接传递内容，不添加任何前缀
                is_user = (message.get('role') == 'user')
                self.append_message(message['content'], is_user=is_user)
    
    def append_message(self, message, is_user=True):
        """使用QTextDocument API确保对齐正确"""
        timestamp = QDateTime.currentDateTime().toString("hh:mm")
        
        # 处理特殊字符
        message = message.replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")
        
        # 获取文本光标
        cursor = self.chat_display.textCursor()
        cursor.movePosition(cursor.End)
        
        # 创建块格式并设置对齐方式
        from PyQt5.QtGui import QTextBlockFormat, QTextCharFormat, QTextCursor
        block_format = QTextBlockFormat()
        
        if is_user:
            # 用户消息右对齐
            block_format.setAlignment(Qt.AlignRight)
        else:
            # AI消息左对齐
            block_format.setAlignment(Qt.AlignLeft)
        
        # 设置块格式
        cursor.insertBlock(block_format)
        
        # 插入时间戳
        timestamp_format = QTextCharFormat()
        timestamp_format.setFontPointSize(10)
        timestamp_format.setForeground(Qt.gray)
        cursor.insertText(f"{timestamp}\n", timestamp_format)
        
        # 插入消息内容
        if is_user:
            # 用户消息使用特殊格式
            cursor.insertHtml(f"""
                <span style="
                    background: #95ec69;
                    border-radius: 25px;                /* 完全圆润的气泡 */
                    padding: 16px 22px;                 /* 更大的内边距 */
                    display: inline-block;
                    text-align: left;
                    font-size: 16px;
                    margin-right: 10px;
                    max-width: 80%;
                    word-wrap: break-word;
                    line-height: 1.5;                   /* 更舒适的行高 */
                    border: 1.5px solid #7bc957;        /* 稍微粗一点的边框 */
                    box-shadow: 0 3px 8px rgba(0,0,0,0.15); /* 更明显的阴影 */
                ">{message}</span>
            """)
        else:
            # AI消息直接插入
            cursor.insertHtml(f"""
                <div style="
                    font-size: 16px;
                    line-height: 1.6;
                    text-align: left;
                    margin-left: 10px;
                    margin-right: 20px;
                    word-wrap: break-word;
                    white-space: normal;
                ">{message}</div>
            """)
        
        # 添加一些间距
        cursor.insertBlock()
        
        # 确保滚动到底部
        self.chat_display.setTextCursor(cursor)
        self.chat_display.ensureCursorVisible()

    def on_send(self):
        """发送消息"""
        if self.is_busy:
            return
        
        text = self.get_input_text()
        if not text:
            QMessageBox.warning(self, "提示", "请输入您的问题！")
            return
        
        # 显示用户消息
        self.append_message(text, is_user=True)
        self.set_busy(True)
        
        # 保存到历史记录
        if self.parent():
            self.parent().chat_history.append({"role": "user", "content": text})
        
        if self.MOCK_MODE:
            # 模拟回复模式
            from PyQt5.QtCore import QTimer
            
            mock_responses = {
                "默认": "您好！我是您的金融投资顾问。当前处于测试模式，后端服务尚未连接。\n\n当服务启动后，我将为您提供专业的投资建议，包括：\n• 投资组合分析\n• 市场趋势解读\n• 风险管理建议\n• 资产配置策略"
            }
            
            response = mock_responses.get(text, mock_responses["默认"])
            
            # 模拟网络延迟
            QTimer.singleShot(1000, lambda: self._show_mock_response(response))
        else:
            # 真实请求模式
            try:
                from agent_worker import RemoteAgentWorker
                worker = RemoteAgentWorker(text, self.parent().chat_history if self.parent() else [])
                worker.response_signal.connect(self.on_agent_response)
                worker.error_signal.connect(self.on_agent_error)
                worker.start()
            except Exception as e:
                self.on_agent_error(f"启动失败：{str(e)}")
    
    def _show_mock_response(self, response):
        """显示模拟回复 - 确保整条消息在一个气泡中"""
        self.set_busy(False)
        if isinstance(response, list):
            response = "\n".join(response)      
        if self.parent():
            self.parent().chat_history.append({"role": "assistant", "content": response})
        self.append_message(response, is_user=False)
        self.clear_input()
    
    def clear_history(self):
        """清除对话历史"""
        reply = QMessageBox.question(
            self, "确认", 
            "确定要清除所有对话历史吗？",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.chat_display.clear()
            if self.parent():
                self.parent().chat_history = []
            # 修复：不再显示"已清空"提示消息
    
    def get_input_text(self):
        """获取输入文本"""
        return self.input_box.toPlainText().strip()
    
    def clear_input(self):
        """清空输入框"""
        self.input_box.clear()
        self.input_box.setFocus()
    
    def set_busy(self, busy):
        """设置忙碌状态"""
        self.is_busy = busy
        self.send_btn.setEnabled(not busy)
        self.clear_btn.setEnabled(not busy)
        self.input_box.setReadOnly(busy)
        
        if busy:
            self.send_btn.setText("⏳ 处理中...")
            try:
                from loading_dialog import LoadingDialog
                self.loading_dialog = LoadingDialog(self)
                self.loading_dialog.move(
                    self.x() + (self.width() - 200) // 2,
                    self.y() + (self.height() - 100) // 2
                )
                self.loading_dialog.show()
            except ImportError:
                pass
        else:
            self.send_btn.setText("🚀 发送")
            if self.loading_dialog and self.loading_dialog.isVisible():
                self.loading_dialog.close()
                self.loading_dialog = None
    
    def on_agent_response(self, content, new_history):
        """处理Agent响应"""
        self.set_busy(False)
        if self.parent():
            self.parent().chat_history = new_history
        self.append_message(content, is_user=False)
        self.clear_input()
    
    def on_agent_error(self, error_msg):
        """处理错误"""
        self.set_busy(False)
        self.append_message(f"❌ {error_msg}", is_user=False)