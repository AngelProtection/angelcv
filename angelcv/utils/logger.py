"""
Logging configuration module for AngelCV.

This module sets up a rich console logger with rank-zero filtering for distributed training.
The logger provides formatted output with timestamps, log levels, and source information.
"""

import logging

from lightning.pytorch.utilities.rank_zero import rank_zero_only
from rich.console import Console
from rich.logging import RichHandler


def setup_logger(
    name: str = "AngelCV",
    level: int = logging.DEBUG,
    *,  # Forces following arguments to be keyword-only
    show_time: bool = True,
    show_level: bool = True,
    show_path: bool = True,
    markup: bool = True,
) -> logging.Logger:
    """
    Configure and return a logger with rich console formatting.

    Args:
        name (str): Name of the logger. Defaults to "AngelCV".
        level (int): Logging level. Defaults to DEBUG.
        show_time (bool): Show timestamp in logs. Defaults to True.
        show_level (bool): Show log level in logs. Defaults to True.
        show_path (bool): Show source path in logs. Defaults to True.
        markup (bool): Enable rich text markup. Defaults to True.

    Returns:
        logging.Logger: Configured logger instance
    """
    # Create logger instance
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    # Add rich handler only for rank 0 process and if no handlers exist
    if rank_zero_only.rank == 0 and not logger.hasHandlers():
        logger.addHandler(
            RichHandler(
                console=Console(),
                show_time=show_time,
                show_level=show_level,
                show_path=show_path,
                markup=markup,
            )
        )

    return logger


# TODO [MID]: streamline how logger is imported, some places imported from here others from logger directly

# Create default logger instance
logger = setup_logger()
