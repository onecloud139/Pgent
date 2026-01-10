# config.py
import json
import sys

def load_config():
    try:
        with open("config.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"配置加载失败：{e}")
        sys.exit(1)

CONFIG = load_config()