"""
记忆系统工具函数
提供日志配置、文件操作等辅助功能。
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from .config import SYSTEM_LOG_DIR


def setup_logging(
    log_file: Optional[Path] = None,
    level: int = logging.INFO,
    console: bool = True
) -> logging.Logger:
    """
    配置日志系统。

    Args:
        log_file: 日志文件路径，如果为 None 则使用默认系统日志文件
        level: 日志级别，默认为 INFO
        console: 是否在控制台输出日志

    Returns:
        配置好的 logger 实例
    """
    # 如果未指定日志文件，使用默认系统日志文件
    if log_file is None:
        SYSTEM_LOG_DIR.mkdir(parents=True, exist_ok=True)
        log_file = SYSTEM_LOG_DIR / "system.log"

    logger = logging.getLogger('memory_system')
    logger.setLevel(level)

    # 清除现有处理器
    logger.handlers.clear()

    # 创建格式化器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 控制台处理器
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # 文件处理器
    if log_file:
        # 确保日志目录存在
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def ensure_directory(dir_path: Path) -> Path:
    """
    确保目录存在，如果不存在则创建。

    Args:
        dir_path: 目录路径

    Returns:
        存在的目录路径
    """
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def read_file_safe(file_path: Path, default: str = "") -> str:
    """
    安全读取文件内容，如果文件不存在或读取失败返回默认值。

    Args:
        file_path: 文件路径
        default: 默认返回值

    Returns:
        文件内容或默认值
    """
    try:
        if file_path.exists():
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
    except Exception as e:
        logging.getLogger(__name__).warning(f"读取文件失败 {file_path}: {e}")
    return default


def write_file_safe(file_path: Path, content: str) -> bool:
    """
    安全写入文件内容。

    Args:
        file_path: 文件路径
        content: 要写入的内容

    Returns:
        是否成功
    """
    try:
        # 确保目录存在
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    except Exception as e:
        logging.getLogger(__name__).error(f"写入文件失败 {file_path}: {e}")
        return False


def format_timestamp(timestamp: Optional[datetime] = None) -> str:
    """
    格式化时间戳为字符串。

    Args:
        timestamp: 时间戳，如果为 None 使用当前时间

    Returns:
        格式化的时间字符串 (YYYY-MM-DD HH:MM:SS)
    """
    if timestamp is None:
        timestamp = datetime.now()
    return timestamp.strftime("%Y-%m-%d %H:%M:%S")


def parse_timestamp(timestamp_str: str) -> Optional[datetime]:
    """
    解析时间字符串为 datetime 对象。

    Args:
        timestamp_str: 时间字符串，格式应为 YYYY-MM-DD HH:MM:SS

    Returns:
        datetime 对象或 None（解析失败时）
    """
    try:
        return datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        try:
            # 尝试其他常见格式
            return datetime.strptime(timestamp_str, "%Y-%m-%d")
        except ValueError:
            return None