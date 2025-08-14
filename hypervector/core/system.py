"""Main HDC System class that orchestrates all components."""

import torch
from typing import Optional, Union, List, Dict, Any
try:
    import numpy as np
except ImportError:
    # Fallback for environments with fake numpy
    class FakeNumpy:
        def __getattr__(self, name):
            if name == 'ndarray':
                return torch.Tensor
            raise AttributeError(f"module 'numpy' has no attribute '{name}'")
    np = FakeNumpy()

try:
    from PIL import Image
except ImportError:
    # Fallback for environments without PIL
    class FakeImage:
        def __getattr__(self, name):
            return object()
    Image = FakeImage()

from .hypervector import HyperVector
from .operations import bind, bundle, permute, cosine_similarity
from ..encoders.text import TextEncoder
from ..encoders.vision import VisionEncoder
from ..encoders.eeg import EEGEncoder


class HDCSystem:
    """
    Main HDC system that provides high-level interface for hyperdimensional computing.
    """
    
    def __init__(
        self,
        dim: int = 10000,
        device: Optional[str] = None,
        dtype: torch.dtype = torch.float32,
        mode: str = "dense"
    ):
        """Initialize HDC system.
        
        Args:
            dim: Hypervector dimensionality
            device: Compute device ('cpu', 'cuda', etc.)
            dtype: Data type for computations
            mode: Default hypervector mode ('dense', 'binary', 'ternary')
        """
        self.dim = dim
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = dtype
        self.mode = mode
        
        # Initialize encoders
        self.text_encoder = TextEncoder(dim=dim, device=self.device)
        self.vision_encoder = VisionEncoder(dim=dim, device=self.device)
        self.eeg_encoder = EEGEncoder(dim=dim, device=self.device)
        
        # Memory for storing hypervectors
        self.memory: Dict[str, HyperVector] = {}
        
    def encode_text(self, text: str, method: str = "token") -> HyperVector:
        """Encode text into hypervector."""
        return self.text_encoder.encode(text, method=method)
    
    def encode_image(self, image: Union[torch.Tensor, "np.ndarray", Image.Image]) -> HyperVector:
        """Encode image into hypervector."""
        return self.vision_encoder.encode(image)
    
    def encode_eeg(
        self, 
        signal: Union[torch.Tensor, "np.ndarray"], 
        sampling_rate: float = 250.0
    ) -> HyperVector:
        """Encode EEG signal into hypervector."""
        return self.eeg_encoder.encode(signal, sampling_rate=sampling_rate)
    
    def bind(self, hvs: List[HyperVector]) -> HyperVector:
        """Bind multiple hypervectors."""
        if len(hvs) < 2:
            raise ValueError("Need at least 2 hypervectors to bind")
        
        result = hvs[0]
        for hv in hvs[1:]:
            result = bind(result, hv)
        return result
    
    def bundle(self, hvs: List[HyperVector], normalize: bool = True) -> HyperVector:
        """Bundle multiple hypervectors."""
        return bundle(hvs, normalize=normalize)
    
    def permute(self, hv: HyperVector, shift: int = 1) -> HyperVector:
        """Permute hypervector."""
        return permute(hv, shift=shift)
    
    def cosine_similarity(self, hv1: HyperVector, hv2: HyperVector) -> torch.Tensor:
        """Compute cosine similarity."""
        return cosine_similarity(hv1, hv2)
    
    def store(self, key: str, hv: HyperVector) -> None:
        """Store hypervector in memory."""
        self.memory[key] = hv
    
    def retrieve(self, key: str) -> Optional[HyperVector]:
        """Retrieve hypervector from memory."""
        return self.memory.get(key)
    
    def query_memory(self, query: HyperVector, top_k: int = 1) -> List[tuple[str, float]]:
        """Query memory for most similar stored hypervectors."""
        if not self.memory:
            return []
        
        similarities = []
        for key, stored_hv in self.memory.items():
            sim = self.cosine_similarity(query, stored_hv).item()
            similarities.append((key, sim))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def clear_memory(self) -> None:
        """Clear all stored hypervectors."""
        self.memory.clear()
    
    def random_hypervector(self, seed: Optional[int] = None) -> HyperVector:
        """Generate random hypervector."""
        return HyperVector.random(
            dim=self.dim, 
            device=self.device, 
            mode=self.mode, 
            seed=seed
        )
    
    def to(self, device: str) -> "HDCSystem":
        """Move system to different device."""
        new_system = HDCSystem(
            dim=self.dim,
            device=device,
            dtype=self.dtype,
            mode=self.mode
        )
        
        # Move stored memories
        for key, hv in self.memory.items():
            new_system.memory[key] = hv.to(device)
            
        return new_system
    
    def __repr__(self) -> str:
        return (f"HDCSystem(dim={self.dim}, device={self.device}, "
                f"mode={self.mode}, memory_size={len(self.memory)})")