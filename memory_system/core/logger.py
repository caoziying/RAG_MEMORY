"""
统一的日志管理模块
提供日志配置、模型信息记录和统一的日志接口。
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

from .config import SYSTEM_LOG_DIR


class SystemLogger:
    """
    系统日志管理器，提供统一的日志接口和模型信息记录。

    功能：
    1. 统一的日志配置管理
    2. 模型使用信息记录
    3. 性能统计
    4. 错误追踪
    """

    # 单例实例
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SystemLogger, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        """初始化日志管理器"""
        if not self._initialized:
            self._initialized = True
            self.loggers: Dict[str, logging.Logger] = {}
            self.model_info: Dict[str, Any] = {}
            self.performance_stats: Dict[str, Any] = {}
            self._setup_default_logger()

    def _setup_default_logger(self):
        """设置默认的logger"""
        # 使用memory_system作为默认logger名称
        self.default_logger = logging.getLogger('memory_system')
        if not self.default_logger.handlers:
            # 确保系统日志目录存在
            SYSTEM_LOG_DIR.mkdir(parents=True, exist_ok=True)

            # 创建格式化器
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )

            # 添加控制台处理器
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            self.default_logger.addHandler(console_handler)

            # 添加文件处理器
            log_file = SYSTEM_LOG_DIR / "system.log"
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setFormatter(formatter)
            self.default_logger.addHandler(file_handler)

            self.default_logger.setLevel(logging.INFO)

    def setup_logging(
        self,
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

        self.default_logger = logger
        return logger

    def get_logger(self, name: str = 'memory_system') -> logging.Logger:
        """
        获取指定名称的logger。

        Args:
            name: logger名称

        Returns:
            logging.Logger实例
        """
        if name not in self.loggers:
            logger = logging.getLogger(name)
            # 如果没有处理器，添加默认处理器
            if not logger.handlers:
                logger.setLevel(self.default_logger.level)
                for handler in self.default_logger.handlers:
                    logger.addHandler(handler)
            self.loggers[name] = logger
        return self.loggers[name]

    def log_model_info(self, model_type: str, model_name: str, status: str = "active", **kwargs):
        """
        记录模型使用信息。

        Args:
            model_type: 模型类型，如 'chat', 'embedding', 'rerank'
            model_name: 模型名称
            status: 模型状态，如 'active', 'fallback', 'disabled'
            **kwargs: 额外信息，如降级原因、版本等
        """
        timestamp = datetime.now().isoformat()
        model_key = f"{model_type}_{model_name}"

        self.model_info[model_key] = {
            "type": model_type,
            "name": model_name,
            "status": status,
            "timestamp": timestamp,
            **kwargs
        }

        # 记录到日志
        logger = self.get_logger('memory_system.models')
        extra_info = ", ".join([f"{k}={v}" for k, v in kwargs.items()])
        log_message = f"模型信息 - 类型: {model_type}, 名称: {model_name}, 状态: {status}"
        if extra_info:
            log_message += f", {extra_info}"

        logger.info(log_message)

    def log_performance(self, operation: str, duration_ms: float, details: Optional[Dict] = None):
        """
        记录性能统计信息。

        Args:
            operation: 操作名称，如 'rerank', 'embedding', 'retrieval'
            duration_ms: 耗时（毫秒）
            details: 额外详情
        """
        if operation not in self.performance_stats:
            self.performance_stats[operation] = {
                "count": 0,
                "total_duration_ms": 0,
                "avg_duration_ms": 0,
                "min_duration_ms": float('inf'),
                "max_duration_ms": 0,
                "last_execution": None
            }

        stats = self.performance_stats[operation]
        stats["count"] += 1
        stats["total_duration_ms"] += duration_ms
        stats["avg_duration_ms"] = stats["total_duration_ms"] / stats["count"]
        stats["min_duration_ms"] = min(stats["min_duration_ms"], duration_ms)
        stats["max_duration_ms"] = max(stats["max_duration_ms"], duration_ms)
        stats["last_execution"] = datetime.now().isoformat()

        if details:
            stats.setdefault("details", []).append(details)

        # 记录到日志（可选，避免日志过多）
        if duration_ms > 1000:  # 如果操作耗时超过1秒，记录警告
            logger = self.get_logger('memory_system.performance')
            logger.warning(f"性能警告 - 操作: {operation}, 耗时: {duration_ms:.2f}ms")

    def log_error(self, error_type: str, error_message: str, context: Optional[Dict] = None):
        """
        记录错误信息。

        Args:
            error_type: 错误类型，如 'api_error', 'connection_error', 'validation_error'
            error_message: 错误消息
            context: 错误上下文信息
        """
        logger = self.get_logger('memory_system.errors')

        log_message = f"错误 - 类型: {error_type}, 消息: {error_message}"
        if context:
            context_str = ", ".join([f"{k}={v}" for k, v in context.items()])
            log_message += f", 上下文: {context_str}"

        logger.error(log_message)

    def get_model_summary(self) -> Dict[str, Any]:
        """
        获取模型使用情况摘要。

        Returns:
            模型信息摘要
        """
        summary = {
            "total_models": len(self.model_info),
            "models_by_type": {},
            "models_by_status": {}
        }

        for model_key, info in self.model_info.items():
            model_type = info["type"]
            status = info["status"]

            # 按类型统计
            if model_type not in summary["models_by_type"]:
                summary["models_by_type"][model_type] = 0
            summary["models_by_type"][model_type] += 1

            # 按状态统计
            if status not in summary["models_by_status"]:
                summary["models_by_status"][status] = 0
            summary["models_by_status"][status] += 1

        return summary

    def get_performance_summary(self) -> Dict[str, Any]:
        """
        获取性能统计摘要。

        Returns:
            性能统计摘要
        """
        return self.performance_stats

    def reset_stats(self):
        """重置统计信息"""
        self.performance_stats.clear()


# 全局单例实例
logger_manager = SystemLogger()

# 便捷函数
def setup_logging(log_file: Optional[Path] = None, level: int = logging.INFO, console: bool = True) -> logging.Logger:
    """配置日志系统（兼容现有代码）"""
    return logger_manager.setup_logging(log_file, level, console)

def get_logger(name: str = 'memory_system') -> logging.Logger:
    """获取logger实例（兼容现有代码）"""
    return logger_manager.get_logger(name)

def log_model_info(model_type: str, model_name: str, status: str = "active", **kwargs):
    """记录模型信息"""
    logger_manager.log_model_info(model_type, model_name, status, **kwargs)