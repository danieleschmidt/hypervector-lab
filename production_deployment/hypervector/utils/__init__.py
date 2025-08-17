"""Utility functions for robust operation."""

from .validation import validate_input, validate_dimensions
from .logging import get_logger, setup_logging
from .config import Config, load_config
from .security import sanitize_input, validate_file_path

__all__ = [
    "validate_input",
    "validate_dimensions", 
    "get_logger",
    "setup_logging",
    "Config",
    "load_config",
    "sanitize_input",
    "validate_file_path"
]

def safe_import(module_name, fallback=None):
    """Safely import a module with fallback"""
    try:
        return __import__(module_name)
    except ImportError as e:
        logging.warning(f"Failed to import {module_name}: {e}")
        if fallback:
            logging.info(f"Using fallback for {module_name}")
            return fallback
        raise

def get_available_device():
    """Get the best available computing device"""
    try:
        import torch
        if torch.cuda.is_available():
            return 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'
        else:
            return 'cpu'
    except ImportError:
        logging.warning("PyTorch not available, using CPU fallback")
        return 'cpu'
