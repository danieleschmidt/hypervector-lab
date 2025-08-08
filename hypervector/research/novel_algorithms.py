"""Novel HDC algorithms for research and experimentation."""

import torch
import math
from typing import List, Optional, Dict, Any, Tuple
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


class HierarchicalHDC:
    """
    Hierarchical hyperdimensional computing with multi-level representations.
    
    Novel research contribution: Enables composition of concepts at multiple 
    abstraction levels with adaptive resolution.
    """
    
    def __init__(self, base_dim: int = 10000, levels: int = 3, device: str = "cpu"):
        """Initialize hierarchical HDC system.
        
        Args:
            base_dim: Base dimensionality for lowest level
            levels: Number of hierarchical levels
            device: Compute device
        """
        self.base_dim = base_dim
        self.levels = levels
        self.device = device
        
        # Create hierarchical dimensions (powers of 2 progression)
        self.level_dims = [base_dim * (2 ** i) for i in range(levels)]
        
        # Initialize level-specific operators
        self.level_operators = {}
        for level in range(levels):
            self.level_operators[level] = {
                'projection': self._create_projection_matrix(level),
                'abstraction': self._create_abstraction_operator(level)
            }
        
        logger.info(f"Initialized HierarchicalHDC with {levels} levels: {self.level_dims}")
    
    def _create_projection_matrix(self, level: int) -> torch.Tensor:
        """Create projection matrix for level transitions."""
        if level == 0:
            return torch.eye(self.level_dims[0], device=self.device)
        
        # Random projection from lower to higher level
        lower_dim = self.level_dims[level - 1]
        higher_dim = self.level_dims[level]
        
        projection = torch.randn(higher_dim, lower_dim, device=self.device)
        # Normalize to preserve magnitude
        projection = projection / torch.norm(projection, dim=0, keepdim=True)
        
        return projection
    
    def _create_abstraction_operator(self, level: int) -> Dict[str, Any]:
        """Create abstraction operator for concept formation."""
        return {
            'grouping_threshold': 0.7 - (level * 0.1),  # More abstract levels are more permissive
            'compression_factor': 1.5 + (level * 0.5),   # Higher levels compress more
            'temporal_span': 2 ** level                   # Higher levels see longer patterns
        }
    
    def encode_hierarchical(self, data: torch.Tensor, start_level: int = 0) -> List[HyperVector]:
        """Encode data at multiple hierarchical levels.
        
        Args:
            data: Input data tensor
            start_level: Starting level for encoding
            
        Returns:
            List of hypervectors at each level
        """
        encodings = []
        current_data = data.to(self.device)
        
        for level in range(start_level, self.levels):
            # Project to current level dimensions
            projection = self.level_operators[level]['projection']
            if level > 0:
                # Use previous level as input
                current_data = torch.matmul(projection, current_data.flatten()).unsqueeze(0)
            
            # Apply abstraction operator
            abstraction_ops = self.level_operators[level]['abstraction']
            compression = abstraction_ops['compression_factor']
            
            # Create hypervector with abstraction-aware encoding
            if current_data.numel() < self.level_dims[level]:
                # Pad if necessary
                padded = torch.zeros(self.level_dims[level], device=self.device)
                padded[:current_data.numel()] = current_data.flatten()
                hv_data = padded
            else:
                # Compress if necessary
                compressed_size = int(self.level_dims[level] / compression)
                hv_data = current_data.flatten()[:compressed_size]
                # Pad to exact dimension
                padded = torch.zeros(self.level_dims[level], device=self.device)
                padded[:len(hv_data)] = hv_data
                hv_data = padded
            
            hv = HyperVector(hv_data, device=self.device)
            encodings.append(hv)
            
            # Prepare for next level
            current_data = hv.data
        
        return encodings
    
    def hierarchical_similarity(self, hvs1: List[HyperVector], hvs2: List[HyperVector]) -> List[float]:
        """Compute hierarchical similarity across all levels."""
        similarities = []
        
        for level in range(min(len(hvs1), len(hvs2))):
            sim = cosine_similarity(hvs1[level], hvs2[level]).item()
            # Weight by level importance (higher levels get more weight)
            weighted_sim = sim * (1.0 + level * 0.3)
            similarities.append(weighted_sim)
        
        return similarities
    
    def aggregate_similarity(self, hierarchical_sims: List[float]) -> float:
        """Aggregate hierarchical similarities into single score."""
        if not hierarchical_sims:
            return 0.0
        
        # Weighted average favoring higher levels
        weights = [1.0 + i * 0.5 for i in range(len(hierarchical_sims))]
        weighted_sum = sum(sim * weight for sim, weight in zip(hierarchical_sims, weights))
        weight_sum = sum(weights)
        
        return weighted_sum / weight_sum


class AdaptiveBindingOperator:
    """
    Adaptive binding operator that adjusts binding strength based on context.
    
    Novel research contribution: Context-sensitive binding that preserves 
    important information while reducing interference.
    """
    
    def __init__(self, dim: int = 10000, adaptation_rate: float = 0.1, device: str = "cpu"):
        """Initialize adaptive binding operator.
        
        Args:
            dim: Hypervector dimensionality
            adaptation_rate: Rate of adaptation to context
            device: Compute device
        """
        self.dim = dim
        self.adaptation_rate = adaptation_rate
        self.device = device
        
        # Learnable binding matrix
        self.binding_matrix = torch.eye(dim, device=device) + \
                             torch.randn(dim, dim, device=device) * 0.1
        
        # Context memory for adaptation
        self.context_memory = []
        self.max_context_size = 100
        
        # Importance weights for different binding patterns
        self.importance_weights = torch.ones(dim, device=device)
        
        logger.info(f"Initialized AdaptiveBindingOperator with dim={dim}")
    
    def adaptive_bind(self, hv1: HyperVector, hv2: HyperVector, context: Optional[HyperVector] = None) -> HyperVector:
        """Perform context-adaptive binding.
        
        Args:
            hv1, hv2: Hypervectors to bind
            context: Optional context hypervector
            
        Returns:
            Adaptively bound hypervector
        """
        # Standard element-wise binding as baseline
        standard_binding = bind(hv1, hv2)
        
        # Compute context-sensitive binding
        if context is not None:
            self._update_context(context)
            adaptive_strength = self._compute_adaptive_strength(hv1, hv2, context)
        else:
            adaptive_strength = torch.ones_like(hv1.data)
        
        # Apply adaptive binding matrix
        combined = torch.cat([hv1.data.unsqueeze(1), hv2.data.unsqueeze(1)], dim=1)
        
        # Context-weighted matrix multiplication
        adapted_matrix = self.binding_matrix * adaptive_strength.unsqueeze(1)
        adaptive_result = torch.matmul(adapted_matrix, combined.mean(dim=1))
        
        # Blend standard and adaptive binding
        blend_factor = 0.7  # Favor adaptive binding
        final_result = (blend_factor * adaptive_result + 
                       (1 - blend_factor) * standard_binding.data)
        
        return HyperVector(final_result, device=self.device)
    
    def _update_context(self, context: HyperVector):
        """Update context memory for future adaptations."""
        self.context_memory.append(context.data.clone())
        
        # Maintain memory size
        if len(self.context_memory) > self.max_context_size:
            self.context_memory.pop(0)
        
        # Update importance weights based on context frequency
        if len(self.context_memory) > 10:
            context_stack = torch.stack(self.context_memory[-10:])
            importance_update = torch.abs(context_stack).mean(dim=0)
            
            self.importance_weights = (
                (1 - self.adaptation_rate) * self.importance_weights +
                self.adaptation_rate * importance_update
            )
    
    def _compute_adaptive_strength(self, hv1: HyperVector, hv2: HyperVector, context: HyperVector) -> torch.Tensor:
        """Compute context-sensitive binding strength."""
        # Measure alignment with context
        context_alignment1 = torch.abs(cosine_similarity(hv1, context))
        context_alignment2 = torch.abs(cosine_similarity(hv2, context))
        
        # Measure importance based on magnitude and memory
        magnitude_importance = torch.abs(hv1.data) * torch.abs(hv2.data)
        memory_importance = self.importance_weights
        
        # Combine factors
        adaptive_strength = (
            context_alignment1 * 0.3 +
            context_alignment2 * 0.3 +
            magnitude_importance * 0.2 +
            memory_importance * 0.2
        )
        
        # Normalize to [0.5, 1.5] range
        adaptive_strength = 0.5 + adaptive_strength / adaptive_strength.max()
        
        return adaptive_strength


class QuantumInspiredHDC:
    """
    Quantum-inspired hyperdimensional computing with superposition and entanglement.
    
    Novel research contribution: Quantum-like superposition states for 
    representing uncertainty and quantum entanglement for non-local correlations.
    """
    
    def __init__(self, dim: int = 10000, coherence_time: float = 100.0, device: str = "cpu"):
        """Initialize quantum-inspired HDC system.
        
        Args:
            dim: Hypervector dimensionality
            coherence_time: Time scale for quantum coherence decay
            device: Compute device
        """
        self.dim = dim
        self.coherence_time = coherence_time
        self.device = device
        
        # Quantum state representation: complex amplitudes
        self.quantum_dim = dim // 2  # Complex numbers use 2 reals
        
        # Entanglement registry
        self.entangled_pairs = []
        
        logger.info(f"Initialized QuantumInspiredHDC with quantum_dim={self.quantum_dim}")
    
    def create_superposition(self, hvs: List[HyperVector], amplitudes: Optional[List[complex]] = None) -> 'QuantumHyperVector':
        """Create quantum superposition of hypervectors.
        
        Args:
            hvs: List of hypervectors to superpose
            amplitudes: Complex amplitudes (normalized automatically)
            
        Returns:
            QuantumHyperVector in superposition state
        """
        if amplitudes is None:
            # Equal superposition
            amplitudes = [complex(1.0, 0.0) / math.sqrt(len(hvs))] * len(hvs)
        else:
            # Normalize amplitudes
            norm = math.sqrt(sum(abs(amp)**2 for amp in amplitudes))
            amplitudes = [amp / norm for amp in amplitudes]
        
        # Create quantum state
        quantum_state = torch.zeros(self.quantum_dim, dtype=torch.complex64, device=self.device)
        
        for hv, amplitude in zip(hvs, amplitudes):
            # Convert real hypervector to complex representation
            real_part = hv.data[:self.quantum_dim]
            imag_part = torch.zeros_like(real_part)
            
            if len(hv.data) > self.quantum_dim:
                imag_part = hv.data[self.quantum_dim:2*self.quantum_dim]
            
            complex_hv = torch.complex(real_part, imag_part)
            quantum_state += amplitude * complex_hv
        
        return QuantumHyperVector(quantum_state, self.coherence_time, self.device)
    
    def quantum_bind(self, qhv1: 'QuantumHyperVector', qhv2: 'QuantumHyperVector') -> 'QuantumHyperVector':
        """Quantum binding operation preserving superposition."""
        # Element-wise multiplication in quantum space
        bound_state = qhv1.quantum_state * qhv2.quantum_state
        
        # Apply quantum phase rotation
        phase_rotation = torch.exp(1j * torch.randn(self.quantum_dim, device=self.device) * 0.1)
        bound_state *= phase_rotation
        
        # Combine coherence times
        combined_coherence = min(qhv1.coherence_time, qhv2.coherence_time)
        
        return QuantumHyperVector(bound_state, combined_coherence, self.device)
    
    def entangle(self, qhv1: 'QuantumHyperVector', qhv2: 'QuantumHyperVector') -> Tuple['QuantumHyperVector', 'QuantumHyperVector']:
        """Create quantum entanglement between two hypervectors."""
        # Create Bell-like entangled state
        entangled_state1 = (qhv1.quantum_state + qhv2.quantum_state) / math.sqrt(2)
        entangled_state2 = (qhv1.quantum_state - qhv2.quantum_state) / math.sqrt(2)
        
        # Apply entanglement phase
        entanglement_phase = torch.exp(1j * torch.pi / 4)
        entangled_state1 *= entanglement_phase
        entangled_state2 *= entanglement_phase.conj()
        
        qhv1_entangled = QuantumHyperVector(entangled_state1, qhv1.coherence_time, self.device)
        qhv2_entangled = QuantumHyperVector(entangled_state2, qhv2.coherence_time, self.device)
        
        # Register entanglement
        self.entangled_pairs.append((qhv1_entangled, qhv2_entangled))
        
        return qhv1_entangled, qhv2_entangled
    
    def measure(self, qhv: 'QuantumHyperVector') -> HyperVector:
        """Quantum measurement collapse to classical hypervector."""
        # Apply decoherence
        decoherence_factor = torch.exp(-torch.tensor(1.0) / qhv.coherence_time)
        
        # Measurement collapse with Born rule
        probabilities = torch.abs(qhv.quantum_state)**2
        
        # Stochastic collapse
        measurement_noise = torch.randn_like(probabilities) * 0.1
        collapsed_amplitudes = torch.real(qhv.quantum_state) + measurement_noise
        
        # Convert back to real hypervector
        real_hv_data = torch.zeros(self.dim, device=self.device)
        real_hv_data[:self.quantum_dim] = collapsed_amplitudes
        real_hv_data[self.quantum_dim:2*self.quantum_dim] = torch.imag(qhv.quantum_state)
        
        return HyperVector(real_hv_data, device=self.device)


class QuantumHyperVector:
    """Quantum hypervector with complex amplitudes and coherence."""
    
    def __init__(self, quantum_state: torch.Tensor, coherence_time: float, device: str):
        self.quantum_state = quantum_state.to(device)
        self.coherence_time = coherence_time
        self.device = device
        self.creation_time = 0  # Could be set to actual time for decay simulation
    
    def probability_amplitude(self, index: int) -> complex:
        """Get probability amplitude at specific index."""
        return self.quantum_state[index].item()
    
    def measurement_probability(self, index: int) -> float:
        """Get measurement probability at specific index."""
        return (torch.abs(self.quantum_state[index])**2).item()
    
    def quantum_similarity(self, other: 'QuantumHyperVector') -> complex:
        """Compute quantum similarity (inner product)."""
        return torch.vdot(self.quantum_state, other.quantum_state).item()


class TemporalHDC:
    """
    Temporal hyperdimensional computing with time-aware operations.
    
    Novel research contribution: Explicit temporal dynamics and 
    time-sensitive similarity measures.
    """
    
    def __init__(self, dim: int = 10000, temporal_resolution: int = 100, device: str = "cpu"):
        """Initialize temporal HDC system.
        
        Args:
            dim: Hypervector dimensionality
            temporal_resolution: Number of temporal bins
            device: Compute device
        """
        self.dim = dim
        self.temporal_resolution = temporal_resolution
        self.device = device
        
        # Create temporal basis vectors
        self.temporal_basis = self._create_temporal_basis()
        
        # Temporal memory buffer
        self.temporal_memory = {}
        
        logger.info(f"Initialized TemporalHDC with resolution={temporal_resolution}")
    
    def _create_temporal_basis(self) -> torch.Tensor:
        """Create orthogonal temporal basis vectors."""
        # Use Fourier-like basis for temporal representation
        basis = torch.zeros(self.temporal_resolution, self.dim, device=self.device)
        
        for t in range(self.temporal_resolution):
            # Create sinusoidal patterns with different frequencies
            for freq in range(1, self.dim // self.temporal_resolution + 1):
                start_idx = (freq - 1) * self.temporal_resolution
                end_idx = min(start_idx + self.temporal_resolution, self.dim)
                
                if start_idx < self.dim:
                    phase = 2 * math.pi * freq * t / self.temporal_resolution
                    amplitude = 1.0 / math.sqrt(self.temporal_resolution)
                    
                    basis[t, start_idx:end_idx] = amplitude * torch.sin(
                        torch.tensor(phase + torch.arange(end_idx - start_idx) * 0.1)
                    )
        
        return basis
    
    def encode_temporal_sequence(self, sequence: List[HyperVector], timestamps: Optional[List[float]] = None) -> HyperVector:
        """Encode temporal sequence of hypervectors.
        
        Args:
            sequence: List of hypervectors in temporal order
            timestamps: Optional timestamps (uniform spacing if None)
            
        Returns:
            Temporal hypervector encoding
        """
        if timestamps is None:
            timestamps = list(range(len(sequence)))
        
        # Normalize timestamps to temporal resolution
        min_time, max_time = min(timestamps), max(timestamps)
        time_range = max_time - min_time
        
        if time_range == 0:
            normalized_times = [0] * len(timestamps)
        else:
            normalized_times = [
                int((t - min_time) / time_range * (self.temporal_resolution - 1))
                for t in timestamps
            ]
        
        # Encode sequence with temporal binding
        temporal_encoding = torch.zeros(self.dim, device=self.device)
        
        for i, (hv, time_idx) in enumerate(zip(sequence, normalized_times)):
            # Bind with temporal basis vector
            temporal_basis_hv = HyperVector(self.temporal_basis[time_idx], device=self.device)
            temporal_bound = bind(hv, temporal_basis_hv)
            
            # Add to sequence encoding
            temporal_encoding += temporal_bound.data
        
        # Normalize
        temporal_encoding = temporal_encoding / len(sequence)
        
        return HyperVector(temporal_encoding, device=self.device)
    
    def temporal_similarity(self, thv1: HyperVector, thv2: HyperVector, time_weight: float = 0.5) -> float:
        """Compute time-aware similarity between temporal hypervectors.
        
        Args:
            thv1, thv2: Temporal hypervectors
            time_weight: Weight for temporal vs spatial similarity
            
        Returns:
            Time-weighted similarity score
        """
        # Standard spatial similarity
        spatial_sim = cosine_similarity(thv1, thv2).item()
        
        # Temporal similarity based on temporal basis alignment
        temporal_alignment = 0.0
        
        for t in range(self.temporal_resolution):
            basis_hv = HyperVector(self.temporal_basis[t], device=self.device)
            
            # Project onto temporal basis
            proj1 = cosine_similarity(thv1, basis_hv).item()
            proj2 = cosine_similarity(thv2, basis_hv).item()
            
            # Accumulate temporal alignment
            temporal_alignment += proj1 * proj2
        
        temporal_alignment /= self.temporal_resolution
        
        # Combine spatial and temporal similarities
        combined_similarity = (
            (1 - time_weight) * spatial_sim +
            time_weight * temporal_alignment
        )
        
        return combined_similarity
    
    def predict_temporal_evolution(self, current_hv: HyperVector, time_steps: int) -> List[HyperVector]:
        """Predict temporal evolution of hypervector.
        
        Args:
            current_hv: Current hypervector state
            time_steps: Number of future time steps to predict
            
        Returns:
            List of predicted future states
        """
        predictions = []
        current_state = current_hv.data.clone()
        
        # Simple temporal evolution model (could be learned)
        evolution_matrix = torch.eye(self.dim, device=self.device) + \
                          torch.randn(self.dim, self.dim, device=self.device) * 0.01
        
        for step in range(time_steps):
            # Apply evolution
            current_state = torch.matmul(evolution_matrix, current_state)
            
            # Add temporal basis contribution
            time_idx = step % self.temporal_resolution
            temporal_contribution = self.temporal_basis[time_idx] * 0.1
            current_state += temporal_contribution
            
            # Normalize to maintain hypervector properties
            current_state = current_state / torch.norm(current_state)
            
            predictions.append(HyperVector(current_state.clone(), device=self.device))
        
        return predictions