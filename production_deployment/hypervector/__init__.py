"""
HyperVector-Lab: Production-ready tooling for Hyperdimensional Computing (HDC) in PyTorch 2.5

A comprehensive library for hyperdimensional computing with:
- Multi-modal encoders (text, vision, EEG)
- Advanced applications (BCI, cross-modal retrieval)
- Production-ready features (security, monitoring, acceleration)
- Comprehensive benchmarking and profiling
- Deployment utilities for scalable systems
"""

# Core components
from .core.system import HDCSystem
from .core.hypervector import HyperVector
from .core.operations import bind, bundle, permute, cosine_similarity

# Encoders
from .encoders import TextEncoder, VisionEncoder, EEGEncoder

# Applications
from .applications import BCIClassifier, CrossModalRetrieval

# Version info
__version__ = "1.0.0"
__author__ = "HyperVector Lab Team"
__email__ = "contact@hyperdimensional.co"
__description__ = "Production-ready Hyperdimensional Computing in PyTorch"

# Main exports
__all__ = [
    # Core HDC
    "HDCSystem",
    "HyperVector", 
    "bind",
    "bundle",
    "permute",
    "cosine_similarity",
    
    # Encoders
    "TextEncoder",
    "VisionEncoder", 
    "EEGEncoder",
    
    # Applications
    "BCIClassifier",
    "CrossModalRetrieval",
]

# Submodule imports for advanced users
try:
    from . import utils
    from . import accelerators
    from . import benchmark
    from . import deployment
    
    __all__.extend([
        "utils",
        "accelerators", 
        "benchmark",
        "deployment"
    ])
except ImportError:
    # Optional dependencies not available
    pass

# Configuration
def get_version():
    """Get version string."""
    return __version__

def get_device_info():
    """Get device information."""
    import torch
    device_info = {
        'cpu_available': True,
        'cuda_available': torch.cuda.is_available(),
        'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
    }
    
    if torch.cuda.is_available():
        device_info['cuda_version'] = torch.version.cuda
        device_info['current_device'] = torch.cuda.current_device()
        device_info['device_name'] = torch.cuda.get_device_name()
    
    return device_info

def print_system_info():
    """Print system information."""
    import torch
    
    print(f"HyperVector-Lab {__version__}")
    print(f"PyTorch version: {torch.__version__}")
    
    device_info = get_device_info()
    print(f"CUDA available: {device_info['cuda_available']}")
    
    if device_info['cuda_available']:
        print(f"CUDA devices: {device_info['cuda_device_count']}")
        print(f"Current device: {device_info['device_name']}")
    
    print(f"Recommended device: {'cuda' if device_info['cuda_available'] else 'cpu'}")

# Initialize logging if imported directly
def _setup_default_logging():
    """Setup default logging configuration."""
    try:
        from .utils.logging import setup_logging
        setup_logging(level="WARNING")  # Quiet by default
    except ImportError:
        pass  # utils not available

_setup_default_logging()