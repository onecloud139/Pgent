# worker.py
import json
import requests
import socket
from http import HTTPStatus
from PyQt5.QtCore import QThread, pyqtSignal
from config import CONFIG

class RemoteAgentWorker(QThread):
    response_signal = pyqtSignal(str, list)
    error_signal = pyqtSignal(str)

    def __init__(self, prompt, history=None):
        super().__init__()
        self.prompt = prompt
        self.history = history if history is not None else []
        self.api_config = CONFIG["remote_agent"]

    def run(self):
        try:
            # 首先检查服务是否可用（快速失败）
            self._check_service_available()
            
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

        except requests.exceptions.ConnectionError:
            self.error_signal.emit("无法连接到金融顾问服务\n请确认服务是否启动：\n1. 运行后端服务程序\n2. 检查端口 8005 是否开放")
        except requests.exceptions.Timeout:
            self.error_signal.emit("请求超时，服务响应过慢")
        except socket.error as e:
            self.error_signal.emit(f"网络错误：{str(e)}")
        except Exception as e:
            self.error_signal.emit(f"请求失败：{str(e)}")

    def _check_service_available(self):
        """快速检查服务是否可连接"""
        url = self.api_config["api_url"]
        parsed = requests.compat.urlparse(url)
        host = parsed.hostname
        port = parsed.port or 80
        
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)  # 2秒快速检测
        try:
            sock.connect((host, port))
            sock.close()
        except:
            sock.close()
            raise requests.exceptions.ConnectionError("服务未启动")