"""
记忆系统配置文件
定义常量、默认值和路径配置。
"""

import os
from datetime import datetime
from pathlib import Path

# 项目根目录（memory_system 所在的目录）
PROJECT_ROOT = Path(__file__).parent.parent

# 数据存储目录
DATA_DIR = PROJECT_ROOT / "data"
# 日志存储目录
LOG_DIR = DATA_DIR / "logs"
# 系统日志存储目录（与data同级）
SYSTEM_LOG_DIR = PROJECT_ROOT / "log"
# 用户信息文件
USER_INFO_FILE = DATA_DIR / "user.md"
# 压缩记忆存储文件
COMPRESSED_MEMORY_FILE = DATA_DIR / "compressed_memory.md"

# 对话窗口最大容量（最近对话轮次）
MAX_WINDOWS = 10

# 默认 LLM 配置（用户可覆盖）
DEFAULT_LLM_CONFIG = {
    "model_name": "Qwen3-32B-FP8",
    "api_key": None,
    "base_url": "https://api.chat.csu.edu.cn/v1",
    "temperature": 0.0,
    "stream": False
}

# 确保必要的目录存在
def init_directories():
    """初始化所需的目录结构"""
    directories = [DATA_DIR, LOG_DIR, SYSTEM_LOG_DIR]
    for dir_path in directories:
        dir_path.mkdir(parents=True, exist_ok=True)

# 获取当天日志文件路径
def get_daily_log_path() -> Path:
    """返回当天日志文件的路径，格式为 YYYY-MM-DD.md"""
    today = datetime.now().strftime("%Y-%m-%d")
    return LOG_DIR / f"{today}.md"