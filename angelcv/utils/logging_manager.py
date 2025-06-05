"""
Logging manager for AngelCV.

This module provides a comprehensive logging system that follows Python logging best practices
while integrating seamlessly with PyTorch Lightning's experiment tracking capabilities.

Features:
- Hierarchical logger management
- Experiment directory integration
- File and console logging with rotation
- Distributed training support (rank-zero logging)
- Structured logging with context
- Configurable log levels and formats
"""

from datetime import datetime
import logging
import logging.config
import os
from pathlib import Path
from typing import Any, Self

from lightning.pytorch.utilities.rank_zero import rank_zero_only
from rich.console import Console
from rich.logging import RichHandler


class LoggingManager:
    """
    Centralized logging manager for AngelCV.

    Provides consistent logging configuration across the application with support for:
    - Console logging with Rich formatting
    - File logging with rotation
    - Experiment-specific log directories
    - Distributed training (rank-zero only file writing)
    """

    _instance: Self | None = None
    _initialized: bool = False
    _experiment_dir: Path | None = None
    _loggers: dict[str, logging.Logger] = {}

    def __new__(cls) -> "LoggingManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        # Prevent re-initialization
        if self._initialized:
            return
        self._initialized = True

        # Default configuration
        self._base_level = logging.INFO
        self._console_level = logging.INFO
        self._file_level = logging.DEBUG
        self._max_bytes = 10 * 1024 * 1024  # 10MB
        self._backup_count = 5

        # Setup default console logging
        self._setup_default_console_logging()

    def configure(
        self,
        experiment_dir: str | Path | None = None,
        base_level: int = logging.INFO,
        console_level: int = logging.INFO,
        file_level: int = logging.DEBUG,
        max_bytes: int = 10 * 1024 * 1024,
        backup_count: int = 5,
        enable_file_logging: bool = True,
    ) -> None:
        """
        Configure the logging system.

        Args:
            experiment_dir: Directory for storing log files (if None, uses current working directory)
            base_level: Base logging level for all loggers
            console_level: Logging level for console output
            file_level: Logging level for file output
            max_bytes: Maximum size of log files before rotation
            backup_count: Number of backup files to keep
            enable_file_logging: Whether to enable file logging
        """
        self._base_level = base_level
        self._console_level = console_level
        self._file_level = file_level
        self._max_bytes = max_bytes
        self._backup_count = backup_count

        if experiment_dir:
            self._experiment_dir = Path(experiment_dir)
            self._experiment_dir.mkdir(parents=True, exist_ok=True)

        if enable_file_logging and self._experiment_dir:
            self._setup_file_logging()

    def _setup_default_console_logging(self) -> None:
        """Setup default console logging with Rich formatting."""
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(self._base_level)

        # Remove existing handlers to avoid duplicates
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        # Add Rich console handler (rank zero only for distributed training)
        if rank_zero_only.rank == 0:
            console_handler = RichHandler(
                console=Console(),
                show_time=True,
                show_level=True,
                show_path=True,
                markup=True,
                rich_tracebacks=True,
            )
            console_handler.setLevel(self._console_level)

            # Custom formatter for better structure
            formatter = logging.Formatter(
                fmt="%(message)s",
                datefmt="[%X]",
            )
            console_handler.setFormatter(formatter)
            root_logger.addHandler(console_handler)

    @rank_zero_only
    def _setup_file_logging(self) -> None:
        """Setup file logging with rotation (rank zero only)."""
        if not self._experiment_dir:
            return

        # Create logs subdirectory
        log_dir = self._experiment_dir / "logs"
        log_dir.mkdir(exist_ok=True)

        # Setup rotating file handler
        from logging.handlers import RotatingFileHandler

        log_file = log_dir / "angelcv.log"
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=self._max_bytes,
            backupCount=self._backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(self._file_level)

        # Detailed formatter for file logs
        file_formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(funcName)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_formatter)

        # Add to root logger
        root_logger = logging.getLogger()
        root_logger.addHandler(file_handler)

        # Log system information
        self._log_system_info()

    @rank_zero_only
    def _log_system_info(self) -> None:
        """Log system and environment information."""
        import platform
        import sys

        import torch

        logger = self.get_logger("system")
        logger.info("=" * 80)
        logger.info("ANGELCV LOGGING SESSION STARTED")
        logger.info("=" * 80)
        logger.info(f"Timestamp: {datetime.now().isoformat()}")
        logger.info(f"Platform: {platform.platform()}")
        logger.info(f"Python: {sys.version}")
        logger.info(f"PyTorch: {torch.__version__}")
        logger.info(f"CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"CUDA Version: {torch.version.cuda}")
            logger.info(f"GPU Count: {torch.cuda.device_count()}")
        logger.info(f"Working Directory: {os.getcwd()}")
        if self._experiment_dir:
            logger.info(f"Experiment Directory: {self._experiment_dir}")
        logger.info("=" * 80)

    def get_logger(self, name: str) -> logging.Logger:
        """
        Get or create a logger with the specified name.

        Args:
            name: Logger name (typically __name__ of the module)

        Returns:
            Configured logger instance
        """
        if name not in self._loggers:
            logger = logging.getLogger(name)
            logger.setLevel(self._base_level)
            self._loggers[name] = logger

        return self._loggers[name]

    def set_experiment_dir(self, experiment_dir: str | Path) -> None:
        """
        Set the experiment directory and reconfigure file logging.

        Args:
            experiment_dir: Path to experiment directory
        """
        self.configure(experiment_dir=experiment_dir, enable_file_logging=True)

    def get_experiment_dir(self) -> Path | None:
        """Get the current experiment directory."""
        return self._experiment_dir

    def set_level(self, level: int, logger_name: str | None = None) -> None:
        """
        Set logging level for a specific logger or all loggers.

        Args:
            level: Logging level (e.g., logging.DEBUG, logging.INFO)
            logger_name: Specific logger name, or None for root logger
        """
        if logger_name:
            logger = self.get_logger(logger_name)
            logger.setLevel(level)
        else:
            logging.getLogger().setLevel(level)
            self._base_level = level

    @rank_zero_only
    def log_config(self, config: dict[str, Any], logger_name: str = "config") -> None:
        """
        Log configuration information to a dedicated config logger.

        Args:
            config: Configuration dictionary to log
            logger_name: Name of the logger to use
        """
        logger = self.get_logger(logger_name)
        logger.info("Configuration:")
        logger.info("-" * 40)

        def log_dict_recursive(d: dict[str, Any], indent: str = "") -> None:
            for key, value in d.items():
                if isinstance(value, dict):
                    logger.info(f"{indent}{key}:")
                    log_dict_recursive(value, indent + "  ")
                else:
                    logger.info(f"{indent}{key}: {value}")

        log_dict_recursive(config)
        logger.info("-" * 40)

    @rank_zero_only
    def close(self) -> None:
        """Close all logging handlers and cleanup."""
        logger = self.get_logger("system")
        logger.info("=" * 80)
        logger.info("ANGELCV LOGGING SESSION ENDED")
        logger.info(f"Timestamp: {datetime.now().isoformat()}")
        logger.info("=" * 80)

        # Close all handlers
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            handler.close()
            root_logger.removeHandler(handler)

        # Reset state
        self._loggers.clear()
        self._experiment_dir = None


# Global instance
_logging_manager = LoggingManager()


def setup_logging(
    experiment_dir: str | Path | None = None,
    level: int = logging.INFO,
    enable_file_logging: bool = True,
) -> LoggingManager:
    """
    Convenient function to setup logging with default configuration.

    Args:
        experiment_dir: Directory for storing log files
        level: Base logging level
        enable_file_logging: Whether to enable file logging

    Returns:
        Configured LoggingManager instance
    """
    _logging_manager.configure(
        experiment_dir=experiment_dir,
        base_level=level,
        enable_file_logging=enable_file_logging,
    )
    return _logging_manager


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the specified name.

    This is the recommended way to get loggers throughout the application.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Configured logger instance

    Example:
        ```python
        from angelcv.utils.logging_manager import get_logger

        logger = get_logger(__name__)
        logger.info("This is an info message")
        ```
    """
    return _logging_manager.get_logger(name)


def set_experiment_dir(experiment_dir: str | Path) -> None:
    """
    Set the experiment directory for file logging.

    Args:
        experiment_dir: Path to experiment directory
    """
    _logging_manager.set_experiment_dir(experiment_dir)


def log_config(config: dict[str, Any]) -> None:
    """
    Log configuration information.

    Args:
        config: Configuration dictionary to log
    """
    _logging_manager.log_config(config)


def close_logging() -> None:
    """Close all logging handlers and cleanup."""
    _logging_manager.close()
