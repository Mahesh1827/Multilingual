"""
Centralized logging configuration using loguru.

Usage in any module:
    from query.logging_config import logger
    logger.info("message")

Or continue using standard logging — loguru intercepts it automatically.
"""

import sys
import logging
from loguru import logger

# ──────────────────────────────────────────────
# Remove default loguru handler
# ──────────────────────────────────────────────
logger.remove()

# ──────────────────────────────────────────────
# Console output: human-readable, colored
# ──────────────────────────────────────────────
logger.add(
    sys.stderr,
    level="INFO",
    format=(
        "<green>{time:HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    ),
    colorize=True,
)

# ──────────────────────────────────────────────
# File output: structured JSON for analysis
# ──────────────────────────────────────────────
logger.add(
    ".cache/logs/query_{time:YYYY-MM-DD}.log",
    level="DEBUG",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
    rotation="10 MB",
    retention="30 days",
    compression="zip",
)

# ──────────────────────────────────────────────
# Intercept standard library logging → loguru
# ──────────────────────────────────────────────

class _InterceptHandler(logging.Handler):
    """Forwards standard library log records to loguru."""
    def emit(self, record):
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        
        frame, depth = logging.currentframe(), 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1
        
        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


def setup_logging():
    """Call once at application startup to intercept all standard logging."""
    logging.basicConfig(handlers=[_InterceptHandler()], level=0, force=True)
    
    # Silence noisy libraries
    for name in ["httpx", "httpcore", "urllib3", "sentence_transformers", "asyncio"]:
        logging.getLogger(name).setLevel(logging.WARNING)
