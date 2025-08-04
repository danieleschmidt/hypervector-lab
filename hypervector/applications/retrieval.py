"""Cross-modal retrieval system using hyperdimensional computing."""

import torch
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any
from PIL import Image
import pickle

from ..core.hypervector import HyperVector
from ..core.operations import cosine_similarity, bind
from ..encoders.text import TextEncoder
from ..encoders.vision import VisionEncoder
from ..encoders.eeg import EEGEncoder


class CrossModalRetrieval:
    """
    Cross-modal retrieval system supporting text, images, and EEG signals.
    
    Enables querying with one modality and retrieving relevant items from other modalities.
    """
    
    def __init__(self, dim: int = 10000, device: Optional[str] = None):
        """Initialize cross-modal retrieval system.
        
        Args:
            dim: Hypervector dimensionality
            device: Compute device
        """
        self.dim = dim
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize encoders
        self.text_encoder = TextEncoder(dim=dim, device=self.device)
        self.vision_encoder = VisionEncoder(dim=dim, device=self.device)
        self.eeg_encoder = EEGEncoder(dim=dim, device=self.device)
        
        # Storage for indexed items
        self.indexed_items: Dict[str, Dict[str, Any]] = {}  # item_id -> item_data
        self.text_index: Dict[str, HyperVector] = {}        # item_id -> text_hv
        self.image_index: Dict[str, HyperVector] = {}       # item_id -> image_hv
        self.eeg_index: Dict[str, HyperVector] = {}         # item_id -> eeg_hv
        self.multimodal_index: Dict[str, HyperVector] = {}  # item_id -> combined_hv
        
        # Cross-modal alignment matrices (learned from paired data)
        self.text_to_image_alignment = None
        self.text_to_eeg_alignment = None
        self.image_to_eeg_alignment = None
    
    def add_item(
        self,
        item_id: str,
        text: Optional[str] = None,
        image: Optional[Union[torch.Tensor, np.ndarray, Image.Image]] = None,
        eeg: Optional[Union[torch.Tensor, np.ndarray]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add multimodal item to the index.
        
        Args:
            item_id: Unique identifier for the item
            text: Text description
            image: Image data
            eeg: EEG signal data
            metadata: Additional metadata
        """
        if not any([text, image is not None, eeg is not None]):
            raise ValueError("At least one modality must be provided")
        
        # Store item data
        self.indexed_items[item_id] = {
            'text': text,
            'image': image,
            'eeg': eeg,
            'metadata': metadata or {}
        }
        
        # Encode each modality
        modality_hvs = []
        
        if text:
            text_hv = self.text_encoder.encode(text)
            self.text_index[item_id] = text_hv
            modality_hvs.append(text_hv)
        
        if image is not None:
            image_hv = self.vision_encoder.encode(image)
            self.image_index[item_id] = image_hv
            modality_hvs.append(image_hv)
        
        if eeg is not None:
            eeg_hv = self.eeg_encoder.encode(eeg)
            self.eeg_index[item_id] = eeg_hv
            modality_hvs.append(eeg_hv)
        
        # Create multimodal representation by binding all modalities
        if len(modality_hvs) > 1:
            multimodal_hv = modality_hvs[0]
            for hv in modality_hvs[1:]:
                multimodal_hv = bind(multimodal_hv, hv)
            self.multimodal_index[item_id] = multimodal_hv
        elif len(modality_hvs) == 1:
            self.multimodal_index[item_id] = modality_hvs[0]
    
    def index_dataset(
        self,
        images: Optional[List[Union[torch.Tensor, np.ndarray, Image.Image]]] = None,
        texts: Optional[List[str]] = None,
        eeg_samples: Optional[List[Union[torch.Tensor, np.ndarray]]] = None,
        item_ids: Optional[List[str]] = None
    ) -> None:
        """Index a dataset with multiple items.
        
        Args:
            images: List of images
            texts: List of texts  
            eeg_samples: List of EEG signals
            item_ids: List of item IDs (generated if not provided)
        """
        # Determine dataset size
        sizes = []
        if images: sizes.append(len(images))
        if texts: sizes.append(len(texts))
        if eeg_samples: sizes.append(len(eeg_samples))
        
        if not sizes:
            raise ValueError("At least one modality list must be provided")
        
        dataset_size = max(sizes)
        
        # Generate item IDs if not provided
        if item_ids is None:
            item_ids = [f"item_{i:06d}" for i in range(dataset_size)]
        elif len(item_ids) != dataset_size:
            raise ValueError("Number of item_ids must match dataset size")
        
        # Index each item
        for i in range(dataset_size):
            text = texts[i] if texts and i < len(texts) else None
            image = images[i] if images and i < len(images) else None
            eeg = eeg_samples[i] if eeg_samples and i < len(eeg_samples) else None
            
            self.add_item(item_ids[i], text=text, image=image, eeg=eeg)
    
    def query_by_text(
        self, 
        query_text: str, 
        modality: str = "all",
        top_k: int = 10
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Query using text and retrieve items.
        
        Args:
            query_text: Query text
            modality: Target modality ('text', 'image', 'eeg', 'multimodal', 'all')
            top_k: Number of results to return
            
        Returns:
            List of (item_id, similarity_score, item_data) tuples
        """
        query_hv = self.text_encoder.encode(query_text)
        return self._search(query_hv, modality, top_k)
    
    def query_by_image(
        self,
        query_image: Union[torch.Tensor, np.ndarray, Image.Image],
        modality: str = "all",
        top_k: int = 10
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Query using image and retrieve items."""
        query_hv = self.vision_encoder.encode(query_image)
        return self._search(query_hv, modality, top_k)
    
    def query_by_eeg(
        self,
        query_eeg: Union[torch.Tensor, np.ndarray],
        sampling_rate: float = 250.0,
        modality: str = "all",
        top_k: int = 10
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Query using EEG and retrieve items."""
        query_hv = self.eeg_encoder.encode(query_eeg, sampling_rate)
        return self._search(query_hv, modality, top_k)
    
    def _search(
        self,
        query_hv: HyperVector,
        modality: str,
        top_k: int
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Internal search function."""
        results = []
        
        # Determine which indices to search
        if modality == "text":
            indices = [self.text_index]
        elif modality == "image":
            indices = [self.image_index]
        elif modality == "eeg":
            indices = [self.eeg_index]
        elif modality == "multimodal":
            indices = [self.multimodal_index]
        elif modality == "all":
            indices = [self.text_index, self.image_index, self.eeg_index, self.multimodal_index]
        else:
            raise ValueError(f"Unknown modality: {modality}")
        
        # Search all specified indices
        for index in indices:
            for item_id, item_hv in index.items():
                similarity = cosine_similarity(query_hv, item_hv).item()
                item_data = self.indexed_items[item_id]
                results.append((item_id, similarity, item_data))
        
        # Remove duplicates (keep highest similarity)
        unique_results = {}
        for item_id, similarity, item_data in results:
            if item_id not in unique_results or similarity > unique_results[item_id][1]:
                unique_results[item_id] = (item_id, similarity, item_data)
        
        # Sort by similarity and return top_k
        sorted_results = sorted(unique_results.values(), key=lambda x: x[1], reverse=True)
        return sorted_results[:top_k]
    
    def compute_cross_modal_similarity(
        self,
        text: str,
        image: Union[torch.Tensor, np.ndarray, Image.Image]
    ) -> float:
        """Compute similarity between text and image."""
        text_hv = self.text_encoder.encode(text)
        image_hv = self.vision_encoder.encode(image)
        return cosine_similarity(text_hv, image_hv).item()
    
    def get_statistics(self) -> Dict[str, int]:
        """Get indexing statistics."""
        return {
            'total_items': len(self.indexed_items),
            'text_items': len(self.text_index),
            'image_items': len(self.image_index),
            'eeg_items': len(self.eeg_index),
            'multimodal_items': len(self.multimodal_index)
        }
    
    def clear_index(self) -> None:
        """Clear all indexed items."""
        self.indexed_items.clear()
        self.text_index.clear()
        self.image_index.clear()
        self.eeg_index.clear()
        self.multimodal_index.clear()
    
    def save_index(self, filepath: str) -> None:
        """Save the index to file."""
        index_data = {
            'indexed_items': self.indexed_items,
            'text_index': {k: v.data.cpu() for k, v in self.text_index.items()},
            'image_index': {k: v.data.cpu() for k, v in self.image_index.items()},
            'eeg_index': {k: v.data.cpu() for k, v in self.eeg_index.items()},
            'multimodal_index': {k: v.data.cpu() for k, v in self.multimodal_index.items()},
            'config': {
                'dim': self.dim,
                'device': self.device
            }
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(index_data, f)
    
    def load_index(self, filepath: str) -> None:
        """Load index from file."""
        with open(filepath, 'rb') as f:
            index_data = pickle.load(f)
        
        self.indexed_items = index_data['indexed_items']
        
        # Restore hypervector indices
        for k, v in index_data['text_index'].items():
            self.text_index[k] = HyperVector(v.to(self.device), device=self.device)
        
        for k, v in index_data['image_index'].items():
            self.image_index[k] = HyperVector(v.to(self.device), device=self.device)
        
        for k, v in index_data['eeg_index'].items():
            self.eeg_index[k] = HyperVector(v.to(self.device), device=self.device)
        
        for k, v in index_data['multimodal_index'].items():
            self.multimodal_index[k] = HyperVector(v.to(self.device), device=self.device)
    
    def __repr__(self) -> str:
        stats = self.get_statistics()
        return (f"CrossModalRetrieval(dim={self.dim}, "
                f"total_items={stats['total_items']}, "
                f"text={stats['text_items']}, "
                f"image={stats['image_items']}, "
                f"eeg={stats['eeg_items']})")