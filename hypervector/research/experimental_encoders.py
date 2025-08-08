"""Experimental encoders for advanced data modalities."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Optional, Tuple, Any
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

from ..core.hypervector import HyperVector
from ..core.operations import bind, bundle, permute, cosine_similarity
from ..utils.logging import get_logger

logger = get_logger(__name__)


class GraphEncoder:
    """
    Encode graph structures into hyperdimensional representations.
    
    Novel research contribution: Graph-aware encoding that preserves
    structural properties and enables graph similarity comparisons.
    """
    
    def __init__(self, dim: int = 10000, max_nodes: int = 1000, device: str = "cpu"):
        """Initialize graph encoder.
        
        Args:
            dim: Hypervector dimensionality
            max_nodes: Maximum number of nodes to support
            device: Compute device
        """
        self.dim = dim
        self.max_nodes = max_nodes
        self.device = device
        
        # Node and edge type vocabularies
        self.node_vectors = {}
        self.edge_vectors = {}
        
        # Structural encoding vectors
        self.degree_vectors = self._create_degree_vectors()
        self.position_vectors = self._create_position_vectors()
        
        logger.info(f"Initialized GraphEncoder with dim={dim}, max_nodes={max_nodes}")
    
    def _create_degree_vectors(self) -> Dict[int, HyperVector]:
        """Create hypervectors for different node degrees."""
        degree_vectors = {}
        max_degree = 50  # Support up to degree 50
        
        for degree in range(max_degree + 1):
            seed = hash(f"degree_{degree}") % (2**31)
            degree_vectors[degree] = HyperVector.random(
                dim=self.dim, device=self.device, seed=seed
            )
        
        return degree_vectors
    
    def _create_position_vectors(self) -> Dict[int, HyperVector]:
        """Create hypervectors for node positions in graph."""
        position_vectors = {}
        
        for pos in range(self.max_nodes):
            seed = hash(f"position_{pos}") % (2**31)
            position_vectors[pos] = HyperVector.random(
                dim=self.dim, device=self.device, seed=seed
            )
        
        return position_vectors
    
    def encode_graph(self, 
                    adjacency_matrix: torch.Tensor,
                    node_features: Optional[torch.Tensor] = None,
                    edge_features: Optional[torch.Tensor] = None) -> HyperVector:
        """Encode a graph into a hypervector.
        
        Args:
            adjacency_matrix: Graph adjacency matrix (n_nodes x n_nodes)
            node_features: Optional node feature matrix (n_nodes x n_features)
            edge_features: Optional edge feature matrix (n_edges x n_features)
            
        Returns:
            Graph hypervector representation
        """
        n_nodes = adjacency_matrix.shape[0]
        if n_nodes > self.max_nodes:
            logger.warning(f"Graph has {n_nodes} nodes, truncating to {self.max_nodes}")
            n_nodes = self.max_nodes
            adjacency_matrix = adjacency_matrix[:n_nodes, :n_nodes]
        
        graph_encoding = torch.zeros(self.dim, device=self.device)
        
        # Encode nodes with structural information
        for i in range(n_nodes):
            node_hv = self._encode_node(i, adjacency_matrix, node_features)
            graph_encoding += node_hv.data
        
        # Encode edges
        edges = torch.nonzero(adjacency_matrix, as_tuple=False)
        for edge_idx, (i, j) in enumerate(edges):
            if edge_idx >= len(edges) // 2:  # Avoid double counting for undirected graphs
                break
            edge_hv = self._encode_edge(i.item(), j.item(), edge_features, edge_idx)
            graph_encoding += edge_hv.data
        
        # Normalize
        graph_encoding = graph_encoding / (n_nodes + len(edges))
        
        return HyperVector(graph_encoding, device=self.device)
    
    def _encode_node(self, node_id: int, adjacency_matrix: torch.Tensor, 
                    node_features: Optional[torch.Tensor]) -> HyperVector:
        """Encode individual node with structural context."""
        # Get node degree
        degree = adjacency_matrix[node_id].sum().int().item()
        degree = min(degree, max(self.degree_vectors.keys()))
        
        # Start with degree vector
        node_hv = self.degree_vectors[degree]
        
        # Bind with position
        if node_id < len(self.position_vectors):
            node_hv = bind(node_hv, self.position_vectors[node_id])
        
        # Add node features if available
        if node_features is not None and node_id < node_features.shape[0]:
            feature_hv = self._encode_features(node_features[node_id])
            node_hv = bind(node_hv, feature_hv)
        
        # Encode neighborhood structure
        neighbors = torch.nonzero(adjacency_matrix[node_id], as_tuple=False).flatten()
        if len(neighbors) > 0:
            neighbor_hvs = []
            for neighbor in neighbors[:10]:  # Limit to 10 neighbors for efficiency
                neighbor_pos_hv = self.position_vectors.get(neighbor.item())
                if neighbor_pos_hv:
                    neighbor_hvs.append(neighbor_pos_hv)
            
            if neighbor_hvs:
                neighborhood_hv = bundle(neighbor_hvs)
                node_hv = bind(node_hv, neighborhood_hv)
        
        return node_hv
    
    def _encode_edge(self, src: int, dst: int, edge_features: Optional[torch.Tensor], 
                    edge_idx: int) -> HyperVector:
        """Encode individual edge."""
        # Bind source and destination positions
        src_hv = self.position_vectors.get(src, self.position_vectors[0])
        dst_hv = self.position_vectors.get(dst, self.position_vectors[0])
        
        edge_hv = bind(src_hv, dst_hv)
        
        # Add edge features if available
        if edge_features is not None and edge_idx < edge_features.shape[0]:
            feature_hv = self._encode_features(edge_features[edge_idx])
            edge_hv = bind(edge_hv, feature_hv)
        
        return edge_hv
    
    def _encode_features(self, features: torch.Tensor) -> HyperVector:
        """Encode feature vector into hypervector."""
        # Simple feature encoding: quantize and bind
        quantized = torch.round(features * 10) / 10
        
        feature_hvs = []
        for i, value in enumerate(quantized):
            seed = hash(f"feature_{i}_{value.item()}") % (2**31)
            feature_hv = HyperVector.random(dim=self.dim, device=self.device, seed=seed)
            feature_hvs.append(feature_hv)
        
        return bundle(feature_hvs) if feature_hvs else HyperVector.zeros(self.dim, device=self.device)
    
    def graph_similarity(self, graph_hv1: HyperVector, graph_hv2: HyperVector) -> float:
        """Compute structural similarity between graph hypervectors."""
        return cosine_similarity(graph_hv1, graph_hv2).item()


class SequenceEncoder:
    """
    Advanced sequence encoding with attention and memory mechanisms.
    
    Novel research contribution: Position-aware encoding with learned
    attention patterns for long sequences.
    """
    
    def __init__(self, dim: int = 10000, max_length: int = 1000, 
                 attention_heads: int = 8, device: str = "cpu"):
        """Initialize sequence encoder.
        
        Args:
            dim: Hypervector dimensionality
            max_length: Maximum sequence length
            attention_heads: Number of attention heads
            device: Compute device
        """
        self.dim = dim
        self.max_length = max_length
        self.attention_heads = attention_heads
        self.device = device
        
        # Position encoding
        self.position_encodings = self._create_position_encodings()
        
        # Attention mechanism
        self.attention_dim = dim // attention_heads
        self.attention_matrices = self._create_attention_matrices()
        
        # Memory bank for sequence patterns
        self.sequence_memory = {}
        
        logger.info(f"Initialized SequenceEncoder with dim={dim}, max_length={max_length}")
    
    def _create_position_encodings(self) -> torch.Tensor:
        """Create learnable position encodings."""
        # Sinusoidal position encodings like in Transformer
        pos_encodings = torch.zeros(self.max_length, self.dim, device=self.device)
        
        position = torch.arange(0, self.max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.dim, 2).float() * 
                           (-math.log(10000.0) / self.dim))
        
        pos_encodings[:, 0::2] = torch.sin(position * div_term)
        pos_encodings[:, 1::2] = torch.cos(position * div_term)
        
        return pos_encodings
    
    def _create_attention_matrices(self) -> Dict[str, torch.Tensor]:
        """Create attention projection matrices."""
        return {
            'query': torch.randn(self.attention_heads, self.dim, self.attention_dim, device=self.device),
            'key': torch.randn(self.attention_heads, self.dim, self.attention_dim, device=self.device),
            'value': torch.randn(self.attention_heads, self.dim, self.attention_dim, device=self.device)
        }
    
    def encode_sequence(self, sequence: List[HyperVector], 
                       use_attention: bool = True) -> HyperVector:
        """Encode sequence with position and attention.
        
        Args:
            sequence: List of hypervectors in sequence
            use_attention: Whether to use attention mechanism
            
        Returns:
            Sequence hypervector representation
        """
        seq_length = min(len(sequence), self.max_length)
        
        if seq_length == 0:
            return HyperVector.zeros(self.dim, device=self.device)
        
        # Add position encodings
        positioned_sequence = []
        for i, hv in enumerate(sequence[:seq_length]):
            pos_encoding = HyperVector(self.position_encodings[i], device=self.device)
            positioned_hv = bind(hv, pos_encoding)
            positioned_sequence.append(positioned_hv)
        
        if use_attention:
            return self._apply_attention(positioned_sequence)
        else:
            # Simple bundling with position decay
            weighted_hvs = []
            for i, hv in enumerate(positioned_sequence):
                weight = 1.0 / (1.0 + i * 0.1)  # Decay with position
                weighted_hv = HyperVector(hv.data * weight, device=self.device)
                weighted_hvs.append(weighted_hv)
            
            return bundle(weighted_hvs)
    
    def _apply_attention(self, sequence: List[HyperVector]) -> HyperVector:
        """Apply multi-head attention to sequence."""
        seq_matrix = torch.stack([hv.data for hv in sequence])  # [seq_len, dim]
        
        attended_results = []
        
        for head in range(self.attention_heads):
            # Project to query, key, value
            Q = torch.matmul(seq_matrix, self.attention_matrices['query'][head])
            K = torch.matmul(seq_matrix, self.attention_matrices['key'][head])
            V = torch.matmul(seq_matrix, self.attention_matrices['value'][head])
            
            # Compute attention scores
            attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.attention_dim)
            attention_weights = F.softmax(attention_scores, dim=-1)
            
            # Apply attention
            attended = torch.matmul(attention_weights, V)
            attended_results.append(attended)
        
        # Combine attention heads
        combined_attention = torch.cat(attended_results, dim=-1)
        
        # Global pooling
        pooled = combined_attention.mean(dim=0)
        
        return HyperVector(pooled, device=self.device)
    
    def sequence_similarity(self, seq_hv1: HyperVector, seq_hv2: HyperVector,
                          temporal_weight: float = 0.3) -> float:
        """Compute sequence similarity with temporal awareness."""
        # Spatial similarity
        spatial_sim = cosine_similarity(seq_hv1, seq_hv2).item()
        
        # Temporal similarity based on position encoding alignment
        temporal_sim = 0.0
        for pos in range(min(10, self.max_length)):  # Check first 10 positions
            pos_hv = HyperVector(self.position_encodings[pos], device=self.device)
            
            proj1 = cosine_similarity(seq_hv1, pos_hv).item()
            proj2 = cosine_similarity(seq_hv2, pos_hv).item()
            
            temporal_sim += proj1 * proj2
        
        temporal_sim /= 10
        
        return (1 - temporal_weight) * spatial_sim + temporal_weight * temporal_sim


class AttentionBasedEncoder:
    """
    Encoder with learned attention mechanisms for multi-modal fusion.
    
    Novel research contribution: Cross-modal attention for binding
    information from different modalities with learned importance.
    """
    
    def __init__(self, dim: int = 10000, modalities: List[str] = None, device: str = "cpu"):
        """Initialize attention-based encoder.
        
        Args:
            dim: Hypervector dimensionality
            modalities: List of modality names
            device: Compute device
        """
        self.dim = dim
        self.modalities = modalities or ['text', 'vision', 'audio']
        self.device = device
        
        # Modality-specific encoders
        self.modality_encoders = self._create_modality_encoders()
        
        # Cross-modal attention matrices
        self.attention_matrices = self._create_cross_modal_attention()
        
        # Learned fusion weights
        self.fusion_weights = torch.ones(len(self.modalities), device=device)
        
        logger.info(f"Initialized AttentionBasedEncoder for modalities: {self.modalities}")
    
    def _create_modality_encoders(self) -> Dict[str, torch.Tensor]:
        """Create modality-specific encoding matrices."""
        encoders = {}
        
        for modality in self.modalities:
            # Random projection matrix for each modality
            seed = hash(f"modality_{modality}") % (2**31)
            torch.manual_seed(seed)
            encoder_matrix = torch.randn(self.dim, self.dim, device=self.device)
            
            # Orthogonalize for stability
            encoder_matrix = torch.qr(encoder_matrix)[0]
            encoders[modality] = encoder_matrix
        
        return encoders
    
    def _create_cross_modal_attention(self) -> Dict[str, torch.Tensor]:
        """Create cross-modal attention matrices."""
        attention_matrices = {}
        
        for mod1 in self.modalities:
            for mod2 in self.modalities:
                if mod1 != mod2:
                    key = f"{mod1}_to_{mod2}"
                    attention_matrices[key] = torch.randn(self.dim, self.dim, device=self.device) * 0.1
        
        return attention_matrices
    
    def encode_multimodal(self, modality_data: Dict[str, HyperVector],
                         attention_temperature: float = 1.0) -> HyperVector:
        """Encode multi-modal data with cross-modal attention.
        
        Args:
            modality_data: Dict mapping modality names to hypervectors
            attention_temperature: Temperature for attention softmax
            
        Returns:
            Fused multi-modal hypervector
        """
        # Encode each modality
        encoded_modalities = {}
        for modality, hv in modality_data.items():
            if modality in self.modality_encoders:
                encoder = self.modality_encoders[modality]
                encoded_hv = torch.matmul(encoder, hv.data)
                encoded_modalities[modality] = HyperVector(encoded_hv, device=self.device)
        
        # Apply cross-modal attention
        attended_modalities = {}
        for target_mod in encoded_modalities:
            attention_sum = torch.zeros(self.dim, device=self.device)
            attention_weights_sum = 0.0
            
            for source_mod in encoded_modalities:
                if source_mod != target_mod:
                    attention_key = f"{source_mod}_to_{target_mod}"
                    if attention_key in self.attention_matrices:
                        # Compute attention weight
                        attention_matrix = self.attention_matrices[attention_key]
                        attention_score = torch.matmul(
                            encoded_modalities[source_mod].data,
                            torch.matmul(attention_matrix, encoded_modalities[target_mod].data)
                        ).sum()
                        
                        attention_weight = torch.exp(attention_score / attention_temperature)
                        
                        # Apply attention
                        attended_vector = torch.matmul(attention_matrix, encoded_modalities[source_mod].data)
                        attention_sum += attention_weight * attended_vector
                        attention_weights_sum += attention_weight
            
            # Normalize and combine with self-attention
            if attention_weights_sum > 0:
                attention_sum /= attention_weights_sum
                self_weight = 0.7
                attended_modalities[target_mod] = HyperVector(
                    self_weight * encoded_modalities[target_mod].data + 
                    (1 - self_weight) * attention_sum,
                    device=self.device
                )
            else:
                attended_modalities[target_mod] = encoded_modalities[target_mod]
        
        # Weighted fusion
        fusion_result = torch.zeros(self.dim, device=self.device)
        total_weight = 0.0
        
        for i, modality in enumerate(self.modalities):
            if modality in attended_modalities:
                weight = self.fusion_weights[i]
                fusion_result += weight * attended_modalities[modality].data
                total_weight += weight
        
        if total_weight > 0:
            fusion_result /= total_weight
        
        return HyperVector(fusion_result, device=self.device)
    
    def update_attention_weights(self, modality_data: Dict[str, HyperVector],
                               target_hv: HyperVector, learning_rate: float = 0.01):
        """Update attention weights based on target similarity.
        
        Args:
            modality_data: Input modality data
            target_hv: Target hypervector for supervision
            learning_rate: Learning rate for weight updates
        """
        # Compute current fusion
        current_fusion = self.encode_multimodal(modality_data)
        
        # Compute similarity to target
        current_similarity = cosine_similarity(current_fusion, target_hv).item()
        
        # Update fusion weights based on individual modality similarities
        for i, modality in enumerate(self.modalities):
            if modality in modality_data:
                modality_similarity = cosine_similarity(modality_data[modality], target_hv).item()
                
                # Gradient-like update
                weight_gradient = modality_similarity - current_similarity
                self.fusion_weights[i] += learning_rate * weight_gradient
        
        # Normalize weights
        self.fusion_weights = F.softmax(self.fusion_weights, dim=0)
    
    def modality_importance(self) -> Dict[str, float]:
        """Get current modality importance weights."""
        return {modality: self.fusion_weights[i].item() 
                for i, modality in enumerate(self.modalities)}