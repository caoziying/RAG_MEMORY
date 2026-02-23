"""
记忆系统核心模块。
包含配置、工具函数和基础记忆系统。
"""

from .config import (
    MAX_WINDOWS,
    DATA_DIR,
    LOG_DIR,
    USER_INFO_FILE,
    COMPRESSED_MEMORY_FILE,
    get_daily_log_path,
    init_directories,
    DEFAULT_LLM_CONFIG
)

from .utils import (
    setup_logging,
    ensure_directory,
    read_file_safe,
    write_file_safe,
    format_timestamp,
    parse_timestamp
)

from .memory import (
    MemorySystem,
    create_default_memory_system
)

__all__ = [
    # 配置
    "MAX_WINDOWS",
    "DATA_DIR",
    "LOG_DIR",
    "USER_INFO_FILE",
    "COMPRESSED_MEMORY_FILE",
    "get_daily_log_path",
    "init_directories",
    "DEFAULT_LLM_CONFIG",

    # 工具函数
    "setup_logging",
    "ensure_directory",
    "read_file_safe",
    "write_file_safe",
    "format_timestamp",
    "parse_timestamp",

    # 记忆系统
    "MemorySystem",
    "create_default_memory_system",
]