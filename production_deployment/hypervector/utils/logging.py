"""Logging utilities for monitoring and debugging."""

import logging
import sys
import os
from typing import Optional
from datetime import datetime


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    format_string: Optional[str] = None
) -> None:
    """Setup logging configuration.
    
    Args:
        level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
        log_file: Optional log file path
        format_string: Custom format string
    """
    if format_string is None:
        format_string = (
            '%(asctime)s - %(name)s - %(levelname)s - '
            '%(filename)s:%(lineno)d - %(message)s'
        )
    
    # Configure logging
    logging_config = {
        'level': getattr(logging, level.upper()),
        'format': format_string,
        'datefmt': '%Y-%m-%d %H:%M:%S'
    }
    
    # Setup handlers
    handlers = [logging.StreamHandler(sys.stdout)]
    
    if log_file:
        # Create log directory if needed
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
            
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(handlers=handlers, **logging_config)


def get_logger(name: str) -> logging.Logger:
    """Get logger instance for a module.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


class PerformanceTimer:
    """Context manager for timing operations."""
    
    def __init__(self, operation_name: str, logger: Optional[logging.Logger] = None):
        self.operation_name = operation_name
        self.logger = logger or get_logger(__name__)
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        self.logger.debug(f"Starting {self.operation_name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = datetime.now()
        duration = (self.end_time - self.start_time).total_seconds()
        
        if exc_type is None:
            self.logger.info(f"Completed {self.operation_name} in {duration:.3f}s")
        else:
            self.logger.error(
                f"Failed {self.operation_name} after {duration:.3f}s: {exc_val}"
            )
    
    @property
    def duration(self) -> Optional[float]:
        """Get operation duration in seconds."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None


class MetricsLogger:
    """Logger for performance and usage metrics."""
    
    def __init__(self, name: str):
        self.logger = get_logger(f"{name}.metrics")
        self.counters = {}
        self.timers = {}
    
    def increment_counter(self, name: str, value: int = 1) -> None:
        """Increment a counter metric."""
        self.counters[name] = self.counters.get(name, 0) + value
        self.logger.debug(f"Counter {name}: {self.counters[name]}")
    
    def record_time(self, name: str, duration: float) -> None:
        """Record a timing metric."""
        if name not in self.timers:
            self.timers[name] = []
        self.timers[name].append(duration)
        self.logger.debug(f"Timer {name}: {duration:.3f}s")
    
    def log_summary(self) -> None:
        """Log summary of all metrics."""
        self.logger.info("=== Metrics Summary ===")
        
        for name, count in self.counters.items():
            self.logger.info(f"Counter {name}: {count}")
        
        for name, times in self.timers.items():
            avg_time = sum(times) / len(times)
            total_time = sum(times)
            self.logger.info(
                f"Timer {name}: avg={avg_time:.3f}s, total={total_time:.3f}s, count={len(times)}"
            )
    
    def reset(self) -> None:
        """Reset all metrics."""
        self.counters.clear()
        self.timers.clear()
        self.logger.debug("Metrics reset")


def log_function_call(func):
    """Decorator to log function calls."""
    logger = get_logger(func.__module__)
    
    def wrapper(*args, **kwargs):
        func_name = f"{func.__module__}.{func.__name__}"
        logger.debug(f"Calling {func_name} with args={len(args)}, kwargs={len(kwargs)}")
        
        try:
            result = func(*args, **kwargs)
            logger.debug(f"Successfully completed {func_name}")
            return result
        except Exception as e:
            logger.error(f"Error in {func_name}: {e}")
            raise
    
    return wrapper


def log_errors(logger: Optional[logging.Logger] = None):
    """Decorator to log errors from functions."""
    def decorator(func):
        nonlocal logger
        if logger is None:
            logger = get_logger(func.__module__)
        
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(
                    f"Error in {func.__name__}: {type(e).__name__}: {e}",
                    exc_info=True
                )
                raise
        
        return wrapper
    return decorator