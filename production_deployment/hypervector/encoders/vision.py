"""Vision encoding for hyperdimensional computing."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
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
from typing import Union, Optional, Tuple

from ..core.hypervector import HyperVector
from ..core.operations import bind, bundle


class VisionEncoder:
    """
    Encode images into hypervectors using patch-based or holistic approaches.
    """
    
    def __init__(self, dim: int = 10000, device: Optional[str] = None):
        """Initialize vision encoder.
        
        Args:
            dim: Hypervector dimensionality
            device: Compute device
        """
        self.dim = dim
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Patch encoding parameters
        self.patch_size = 16
        self.num_patches = (224 // self.patch_size) ** 2
        
        # Position encoding for patches
        self.position_vectors = {}
        self._initialize_position_vectors()
        
        # Feature extraction network (simple CNN)
        self.feature_extractor = self._create_feature_extractor()
        
    def _create_feature_extractor(self) -> nn.Module:
        """Create simple CNN for feature extraction."""
        return nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(256 * 7 * 7, 1024),
            nn.ReLU(),
            nn.Linear(1024, self.dim)
        ).to(self.device)
    
    def _initialize_position_vectors(self) -> None:
        """Initialize position vectors for patches."""
        for i in range(self.num_patches):
            self.position_vectors[i] = HyperVector.random(
                dim=self.dim,
                device=self.device,
                seed=(hash("patch_pos") + i) % 2**31
            )
    
    def _preprocess_image(self, image: Union[torch.Tensor, "np.ndarray", Image.Image]) -> torch.Tensor:
        """Preprocess image for encoding."""
        if isinstance(image, np.ndarray):
            if image.ndim == 3 and image.shape[2] == 3:
                # Convert HWC to CHW
                image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            else:
                image = torch.from_numpy(image).float()
        elif isinstance(image, Image.Image):
            image = self.transform(image)
        elif isinstance(image, torch.Tensor):
            if image.ndim == 3 and image.shape[0] == 3:
                # Already in CHW format
                if image.max() > 1.0:
                    image = image.float() / 255.0
            else:
                raise ValueError(f"Unexpected tensor shape: {image.shape}")
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
        
        # Ensure we have a batch dimension
        if image.ndim == 3:
            image = image.unsqueeze(0)
            
        return image.to(self.device)
    
    def encode_holistic(self, image: Union[torch.Tensor, "np.ndarray", Image.Image]) -> HyperVector:
        """Encode entire image as single hypervector."""
        processed_image = self._preprocess_image(image)
        
        with torch.no_grad():
            features = self.feature_extractor(processed_image)
            
        # Normalize features
        features = F.normalize(features, dim=-1)
        
        return HyperVector(features.squeeze(0), device=self.device)
    
    def encode_patches(self, image: Union[torch.Tensor, "np.ndarray", Image.Image]) -> HyperVector:
        """Encode image using patch-based approach."""
        processed_image = self._preprocess_image(image)
        
        # Extract patches
        patches = F.unfold(
            processed_image, 
            kernel_size=self.patch_size, 
            stride=self.patch_size
        )
        patches = patches.transpose(1, 2).reshape(-1, 3, self.patch_size, self.patch_size)
        
        # Encode each patch
        patch_hvs = []
        for i, patch in enumerate(patches):
            if i >= len(self.position_vectors):
                break
                
            # Simple patch encoding: flatten and project
            patch_flat = patch.flatten()
            
            # Project to hypervector dimension
            if not hasattr(self, 'patch_projector'):
                self.patch_projector = nn.Linear(
                    3 * self.patch_size * self.patch_size, 
                    self.dim
                ).to(self.device)
            
            with torch.no_grad():
                patch_features = self.patch_projector(patch_flat.unsqueeze(0))
                patch_features = F.normalize(patch_features, dim=-1)
            
            patch_hv = HyperVector(patch_features.squeeze(0), device=self.device)
            
            # Bind with position
            pos_hv = self.position_vectors[i]
            positioned_patch = bind(patch_hv, pos_hv)
            patch_hvs.append(positioned_patch)
        
        if not patch_hvs:
            return HyperVector.zeros(self.dim, device=self.device)
        
        # Bundle all positioned patches
        return bundle(patch_hvs, normalize=True)
    
    def encode(
        self, 
        image: Union[torch.Tensor, "np.ndarray", Image.Image],
        method: str = "holistic"
    ) -> HyperVector:
        """Encode image using specified method.
        
        Args:
            image: Input image
            method: Encoding method ('holistic', 'patches')
        """
        if method == "holistic":
            return self.encode_holistic(image)
        elif method == "patches":
            return self.encode_patches(image)
        else:
            raise ValueError(f"Unknown encoding method: {method}")
    
    def similarity(
        self, 
        image1: Union[torch.Tensor, "np.ndarray", Image.Image],
        image2: Union[torch.Tensor, "np.ndarray", Image.Image],
        method: str = "holistic"
    ) -> float:
        """Compute similarity between two images."""
        hv1 = self.encode(image1, method=method)
        hv2 = self.encode(image2, method=method)
        return hv1.cosine_similarity(hv2).item()