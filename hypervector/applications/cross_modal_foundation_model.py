"""
Cross-Modal Foundation Model with HDC Backbone
=============================================

Advanced foundation model that uses hyperdimensional computing as the core
representation system for cross-modal understanding and reasoning.

Key features:
1. Unified HDC representation space for all modalities
2. Zero-shot cross-modal transfer learning
3. Compositional reasoning with hypervector operations
4. Scale-invariant representation learning
5. Few-shot adaptation to new modalities
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Union, Any, Tuple
import math
import logging
from dataclasses import dataclass
from enum import Enum

from ..core.hypervector import HyperVector
from ..core.operations import bind, bundle, permute, cosine_similarity
from ..core.system import HDCSystem
from ..encoders.text import TextEncoder
from ..encoders.vision import VisionEncoder
from ..encoders.eeg import EEGEncoder

logger = logging.getLogger(__name__)

class ModalityType(Enum):
    """Supported modality types."""
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    EEG = "eeg"
    EMG = "emg"
    SENSOR = "sensor"

@dataclass
class CrossModalResult:
    """Result from cross-modal operation."""
    representation: HyperVector
    confidence: float
    modality_weights: Dict[str, float]
    attention_map: Optional[torch.Tensor] = None
    metadata: Optional[Dict[str, Any]] = None

class CrossModalFoundationModel(nn.Module):
    """
    Cross-modal foundation model with HDC backbone for unified representation learning.
    """
    
    def __init__(
        self,
        hdc_dim: int = 10000,
        device: Optional[str] = None,
        max_modalities: int = 8,
        use_attention: bool = True,
        adaptive_weighting: bool = True
    ):
        super().__init__()
        self.hdc_dim = hdc_dim
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_modalities = max_modalities
        self.use_attention = use_attention
        self.adaptive_weighting = adaptive_weighting
        
        # Core HDC system
        self.hdc = HDCSystem(dim=hdc_dim, device=self.device)
        
        # Modality encoders
        self.text_encoder = TextEncoder(dim=hdc_dim, device=self.device)
        self.vision_encoder = VisionEncoder(dim=hdc_dim, device=self.device)
        self.eeg_encoder = EEGEncoder(dim=hdc_dim, device=self.device)
        
        # Cross-modal attention mechanism
        if use_attention:
            # Ensure num_heads divides embed_dim
            num_heads = 8
            if hdc_dim % num_heads != 0:
                num_heads = 4  # Try 4 heads
                if hdc_dim % num_heads != 0:
                    num_heads = 1  # Fallback to single head
            
            self.attention = nn.MultiheadAttention(
                embed_dim=hdc_dim,
                num_heads=num_heads,
                dropout=0.1,
                batch_first=True
            ).to(self.device)
        
        # Adaptive modality weighting
        if adaptive_weighting:
            self.modality_weights = nn.ParameterDict({
                modality.value: nn.Parameter(torch.ones(1)) 
                for modality in ModalityType
            })
        
        # Learned basis vectors for compositional operations
        self.composition_bases = nn.Parameter(
            torch.randn(16, hdc_dim, device=self.device)
        )
        
        # Cross-modal projection layers
        self.cross_modal_projections = nn.ModuleDict({
            f"{src.value}_to_{tgt.value}": nn.Linear(hdc_dim, hdc_dim)
            for src in ModalityType for tgt in ModalityType if src != tgt
        }).to(self.device)
        
        logger.info(f"Initialized CrossModalFoundationModel with {hdc_dim}D representations")
    
    def encode_modality(
        self, 
        data: Union[str, torch.Tensor, List], 
        modality: ModalityType,
        **kwargs
    ) -> HyperVector:
        """Encode data from specific modality into HDC representation."""
        try:
            if modality == ModalityType.TEXT:
                return self.text_encoder(data, **kwargs)
            elif modality == ModalityType.IMAGE:
                return self.vision_encoder(data, **kwargs)
            elif modality == ModalityType.EEG:
                return self.eeg_encoder(data, **kwargs)
            else:
                raise NotImplementedError(f"Encoder for {modality.value} not implemented")
        except Exception as e:
            logger.error(f"Error encoding {modality.value}: {e}")
            # Return random hypervector as fallback
            return HyperVector.random(self.hdc_dim, device=self.device)
    
    def cross_modal_attention(
        self,
        query_hv: HyperVector,
        key_hvs: List[HyperVector],
        value_hvs: List[HyperVector]
    ) -> Tuple[HyperVector, torch.Tensor]:
        """Apply cross-modal attention mechanism."""
        if not self.use_attention or not key_hvs:
            return query_hv, torch.ones(1, device=self.device)
        
        # Convert to tensors for attention
        query = query_hv.vector.unsqueeze(0).unsqueeze(0)  # [1, 1, dim]
        keys = torch.stack([hv.vector for hv in key_hvs]).unsqueeze(0)  # [1, num_keys, dim]
        values = torch.stack([hv.vector for hv in value_hvs]).unsqueeze(0)  # [1, num_values, dim]
        
        # Apply attention
        attended, attention_weights = self.attention(query, keys, values)
        
        # Convert back to HyperVector
        attended_hv = HyperVector(attended.squeeze(0).squeeze(0))
        
        return attended_hv, attention_weights.squeeze(0)
    
    def compositional_bind(
        self,
        hvs: List[HyperVector],
        composition_type: str = "sequential"
    ) -> HyperVector:
        """Perform compositional binding with learned operations."""
        if not hvs:
            return HyperVector.random(self.hdc_dim, device=self.device)
        
        if len(hvs) == 1:
            return hvs[0]
        
        if composition_type == "sequential":
            # Sequential binding with position encoding
            result = hvs[0]
            for i, hv in enumerate(hvs[1:], 1):
                position_vec = HyperVector(self.composition_bases[i % len(self.composition_bases)])
                result = bind(result, bind(hv, position_vec))
            return result
            
        elif composition_type == "hierarchical":
            # Hierarchical binding for structured composition
            if len(hvs) == 2:
                return bind(hvs[0], hvs[1])
            
            mid = len(hvs) // 2
            left = self.compositional_bind(hvs[:mid], "hierarchical")
            right = self.compositional_bind(hvs[mid:], "hierarchical")
            return bind(left, right)
            
        elif composition_type == "bundle":
            # Simple bundling for averaging semantics
            return bundle(hvs)
        
        else:
            raise ValueError(f"Unknown composition type: {composition_type}")
    
    def cross_modal_transfer(
        self,
        source_hv: HyperVector,
        source_modality: ModalityType,
        target_modality: ModalityType
    ) -> HyperVector:
        """Transfer representation from one modality to another."""
        if source_modality == target_modality:
            return source_hv
        
        projection_key = f"{source_modality.value}_to_{target_modality.value}"
        
        if projection_key in self.cross_modal_projections:
            projected = self.cross_modal_projections[projection_key](source_hv.vector)
            return HyperVector(projected)
        else:
            # Fallback to identity projection
            logger.warning(f"No projection found for {projection_key}, using identity")
            return source_hv
    
    def multimodal_fusion(
        self,
        modality_data: Dict[ModalityType, Any],
        fusion_strategy: str = "attention",
        composition_type: str = "sequential"
    ) -> CrossModalResult:
        """Fuse multiple modalities into unified representation."""
        # Encode each modality
        encoded_hvs = []
        modality_names = []
        confidences = []
        
        for modality, data in modality_data.items():
            try:
                hv = self.encode_modality(data, modality)
                encoded_hvs.append(hv)
                modality_names.append(modality.value)
                # Simple confidence based on vector norm
                confidences.append(torch.norm(hv.vector).item())
            except Exception as e:
                logger.warning(f"Failed to encode {modality.value}: {e}")
                continue
        
        if not encoded_hvs:
            raise ValueError("No modalities could be encoded")
        
        # Apply adaptive weighting if enabled
        if self.adaptive_weighting:
            weights = torch.stack([
                torch.sigmoid(self.modality_weights[name]) 
                for name in modality_names
            ])
            weights = F.softmax(weights, dim=0)
            
            # Apply weights to hypervectors
            weighted_hvs = [
                HyperVector(w * hv.vector) 
                for w, hv in zip(weights, encoded_hvs)
            ]
        else:
            weighted_hvs = encoded_hvs
            weights = torch.ones(len(encoded_hvs)) / len(encoded_hvs)
        
        # Fusion strategy
        if fusion_strategy == "attention" and len(weighted_hvs) > 1:
            # Use first modality as query, others as keys/values
            query_hv = weighted_hvs[0]
            key_value_hvs = weighted_hvs[1:]
            
            fused_hv, attention_weights = self.cross_modal_attention(
                query_hv, key_value_hvs, key_value_hvs
            )
        
        elif fusion_strategy == "compositional":
            fused_hv = self.compositional_bind(weighted_hvs, composition_type)
            attention_weights = None
        
        elif fusion_strategy == "bundle":
            fused_hv = bundle(weighted_hvs)
            attention_weights = None
        
        else:
            # Default to bundling
            fused_hv = bundle(weighted_hvs)
            attention_weights = None
        
        # Calculate overall confidence
        overall_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        # Create modality weight dictionary
        modality_weights = {
            name: weight.item() 
            for name, weight in zip(modality_names, weights)
        }
        
        return CrossModalResult(
            representation=fused_hv,
            confidence=overall_confidence,
            modality_weights=modality_weights,
            attention_map=attention_weights,
            metadata={
                "fusion_strategy": fusion_strategy,
                "composition_type": composition_type,
                "num_modalities": len(encoded_hvs)
            }
        )
    
    def similarity_search(
        self,
        query: Union[HyperVector, Dict[ModalityType, Any]],
        candidates: List[Union[HyperVector, Dict[ModalityType, Any]]],
        cross_modal: bool = True
    ) -> List[Tuple[int, float]]:
        """Perform similarity search across modalities."""
        # Convert query to HyperVector if needed
        if isinstance(query, dict):
            query_result = self.multimodal_fusion(query)
            query_hv = query_result.representation
        else:
            query_hv = query
        
        similarities = []
        
        for i, candidate in enumerate(candidates):
            try:
                # Convert candidate to HyperVector if needed
                if isinstance(candidate, dict):
                    candidate_result = self.multimodal_fusion(candidate)
                    candidate_hv = candidate_result.representation
                else:
                    candidate_hv = candidate
                
                # Calculate similarity
                sim = cosine_similarity(query_hv, candidate_hv)
                similarities.append((i, sim.item()))
                
            except Exception as e:
                logger.warning(f"Error calculating similarity for candidate {i}: {e}")
                similarities.append((i, 0.0))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities
    
    def zero_shot_classification(
        self,
        data: Dict[ModalityType, Any],
        class_descriptions: List[str],
        description_modality: ModalityType = ModalityType.TEXT
    ) -> List[Tuple[int, float]]:
        """Perform zero-shot classification using cross-modal transfer."""
        # Encode input data
        input_result = self.multimodal_fusion(data)
        input_hv = input_result.representation
        
        # Encode class descriptions
        class_hvs = []
        for desc in class_descriptions:
            class_hv = self.encode_modality(desc, description_modality)
            class_hvs.append(class_hv)
        
        # Calculate similarities
        similarities = []
        for i, class_hv in enumerate(class_hvs):
            sim = cosine_similarity(input_hv, class_hv)
            similarities.append((i, sim.item()))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities
    
    def few_shot_adaptation(
        self,
        support_data: List[Tuple[Dict[ModalityType, Any], int]],
        query_data: Dict[ModalityType, Any],
        k_shot: int = 5
    ) -> Tuple[int, float]:
        """Perform few-shot learning adaptation."""
        # Encode support examples
        support_hvs_by_class = {}
        
        for data, label in support_data[:k_shot]:
            support_result = self.multimodal_fusion(data)
            support_hv = support_result.representation
            
            if label not in support_hvs_by_class:
                support_hvs_by_class[label] = []
            support_hvs_by_class[label].append(support_hv)
        
        # Create class prototypes by bundling
        class_prototypes = {}
        for label, hvs in support_hvs_by_class.items():
            class_prototypes[label] = bundle(hvs)
        
        # Encode query
        query_result = self.multimodal_fusion(query_data)
        query_hv = query_result.representation
        
        # Find closest prototype
        best_class = None
        best_similarity = -1.0
        
        for label, prototype in class_prototypes.items():
            sim = cosine_similarity(query_hv, prototype)
            if sim.item() > best_similarity:
                best_similarity = sim.item()
                best_class = label
        
        return best_class, best_similarity
    
    def generate_explanations(
        self,
        result: CrossModalResult,
        top_k: int = 3
    ) -> Dict[str, Any]:
        """Generate explanations for cross-modal results."""
        explanations = {
            "confidence": f"{result.confidence:.3f}",
            "dominant_modalities": [],
            "fusion_strategy": result.metadata.get("fusion_strategy", "unknown"),
            "representation_summary": {
                "mean": result.representation.vector.mean().item(),
                "std": result.representation.vector.std().item(),
                "sparsity": (result.representation.vector == 0).float().mean().item()
            }
        }
        
        # Find top contributing modalities
        sorted_modalities = sorted(
            result.modality_weights.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        explanations["dominant_modalities"] = sorted_modalities[:top_k]
        
        return explanations

    def save_model(self, path: str):
        """Save the model state."""
        torch.save({
            'state_dict': self.state_dict(),
            'hdc_dim': self.hdc_dim,
            'max_modalities': self.max_modalities,
            'use_attention': self.use_attention,
            'adaptive_weighting': self.adaptive_weighting,
        }, path)
        
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load the model state."""
        checkpoint = torch.load(path, map_location=self.device)
        self.load_state_dict(checkpoint['state_dict'])
        
        logger.info(f"Model loaded from {path}")

# Factory functions for common use cases
def create_multimodal_classifier(
    hdc_dim: int = 10000,
    num_classes: int = None,
    device: Optional[str] = None
) -> CrossModalFoundationModel:
    """Create a cross-modal classifier."""
    return CrossModalFoundationModel(
        hdc_dim=hdc_dim,
        device=device,
        use_attention=True,
        adaptive_weighting=True
    )

def create_multimodal_retrieval_system(
    hdc_dim: int = 10000,
    device: Optional[str] = None
) -> CrossModalFoundationModel:
    """Create a cross-modal retrieval system."""
    return CrossModalFoundationModel(
        hdc_dim=hdc_dim,
        device=device,
        use_attention=False,  # Faster for retrieval
        adaptive_weighting=False
    )