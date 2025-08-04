"""Core HyperVector class implementation."""

import torch
from typing import Union, List, Optional
import numpy as np
from .exceptions import DimensionMismatchError, InvalidModeError, DeviceError
from ..utils.validation import validate_input, validate_dimensions, validate_positive
from ..utils.logging import get_logger, log_errors

logger = get_logger(__name__)


class HyperVector:
    """
    A hyperdimensional vector with efficient operations for HDC.
    
    Supports both binary {-1, 1} and ternary {-1, 0, 1} representations.
    """
    
    @log_errors(logger)
    def __init__(
        self, 
        data: Union[torch.Tensor, np.ndarray, List[float]], 
        device: Optional[str] = None,
        dtype: torch.dtype = torch.float32,
        mode: str = "dense"
    ):
        """Initialize HyperVector.
        
        Args:
            data: Vector data
            device: Target device ('cpu', 'cuda', etc.)
            dtype: Data type
            mode: Storage mode ('dense', 'binary', 'ternary')
            
        Raises:
            InvalidModeError: If mode is not supported
            DeviceError: If device is invalid
            ValueError: If data is invalid
        """
        # Validate mode
        valid_modes = ['dense', 'binary', 'ternary']
        if mode not in valid_modes:
            raise InvalidModeError(f"Mode must be one of {valid_modes}, got '{mode}'")
        
        # Validate and convert data
        try:
            if isinstance(data, (list, np.ndarray)):
                data = torch.tensor(data, dtype=dtype)
            elif isinstance(data, torch.Tensor):
                data = data.to(dtype=dtype)
            else:
                raise ValueError(f"Unsupported data type: {type(data)}")
        except Exception as e:
            logger.error(f"Failed to convert data to tensor: {e}")
            raise ValueError(f"Invalid data: {e}")
            
        # Validate device and move data
        if device is not None:
            try:
                data = data.to(device)
            except Exception as e:
                raise DeviceError(f"Failed to move data to device '{device}': {e}")
            
        # Validate data shape
        if data.ndim != 1:
            raise ValueError(f"Data must be 1-dimensional, got {data.ndim} dimensions")
        
        if data.shape[0] == 0:
            raise ValueError("Data cannot be empty")
            
        self.data = data
        self.mode = mode
        self.dim = data.shape[-1]
        
        logger.debug(f"Created HyperVector: dim={self.dim}, mode={self.mode}, device={self.data.device}")
        
    @classmethod
    def random(
        cls, 
        dim: int, 
        device: Optional[str] = None, 
        mode: str = "dense",
        seed: Optional[int] = None
    ) -> "HyperVector":
        """Generate random hypervector."""
        if seed is not None:
            torch.manual_seed(seed)
            
        if mode == "binary":
            data = torch.randint(0, 2, (dim,), device=device, dtype=torch.float32) * 2 - 1
        elif mode == "ternary":
            data = torch.randint(-1, 2, (dim,), device=device, dtype=torch.float32)
        else:  # dense
            data = torch.randn(dim, device=device, dtype=torch.float32)
            data = torch.nn.functional.normalize(data, dim=-1)
            
        return cls(data, device=device, mode=mode)
    
    @classmethod
    def zeros(cls, dim: int, device: Optional[str] = None) -> "HyperVector":
        """Create zero hypervector."""
        data = torch.zeros(dim, device=device, dtype=torch.float32)
        return cls(data, device=device)
    
    def to(self, device: str) -> "HyperVector":
        """Move to device."""
        return HyperVector(self.data.to(device), mode=self.mode)
    
    def normalize(self) -> "HyperVector":
        """Normalize vector."""
        if self.mode in ["binary", "ternary"]:
            return self  # Already normalized
        normalized_data = torch.nn.functional.normalize(self.data, dim=-1)
        return HyperVector(normalized_data, mode=self.mode)
    
    def binarize(self, threshold: float = 0.0) -> "HyperVector":
        """Convert to binary representation."""
        binary_data = torch.where(self.data > threshold, 1.0, -1.0)
        return HyperVector(binary_data, mode="binary")
    
    def ternarize(self, low_threshold: float = -0.5, high_threshold: float = 0.5) -> "HyperVector":
        """Convert to ternary representation."""
        ternary_data = torch.where(
            self.data > high_threshold, 1.0,
            torch.where(self.data < low_threshold, -1.0, 0.0)
        )
        return HyperVector(ternary_data, mode="ternary")
    
    def __add__(self, other: "HyperVector") -> "HyperVector":
        """Element-wise addition (bundling)."""
        result_data = self.data + other.data
        return HyperVector(result_data, mode="dense")
    
    def __mul__(self, other: Union["HyperVector", float]) -> "HyperVector":
        """Element-wise multiplication (binding) or scalar multiplication."""
        if isinstance(other, HyperVector):
            result_data = self.data * other.data
            return HyperVector(result_data, mode=self.mode)
        else:
            result_data = self.data * other
            return HyperVector(result_data, mode=self.mode)
    
    def __rmul__(self, other: float) -> "HyperVector":
        """Right scalar multiplication."""
        return self.__mul__(other)
    
    def __repr__(self) -> str:
        return f"HyperVector(dim={self.dim}, mode={self.mode}, device={self.data.device})"
    
    def cosine_similarity(self, other: "HyperVector") -> torch.Tensor:
        """Compute cosine similarity with another hypervector."""
        dot_product = torch.dot(self.data, other.data)
        norm_product = torch.norm(self.data) * torch.norm(other.data)
        return dot_product / (norm_product + 1e-8)
    
    def hamming_distance(self, other: "HyperVector") -> torch.Tensor:
        """Compute Hamming distance (for binary/ternary vectors)."""
        if self.mode not in ["binary", "ternary"] or other.mode not in ["binary", "ternary"]:
            raise ValueError("Hamming distance only defined for binary/ternary vectors")
        different = torch.sum(self.data != other.data)
        return different / self.dim