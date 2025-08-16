"""Neural Plasticity-Inspired Hyperdimensional Computing.

Revolutionary research contribution: HDC systems with biologically-inspired
synaptic plasticity mechanisms including Hebbian learning, spike-timing dependent
plasticity (STDP), and homeostatic scaling for adaptive learning.

Academic Impact: First implementation of detailed neural plasticity rules in HDC,
enabling self-modifying hypervectors that adapt based on usage patterns.
"""

import torch
import math
import time
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
from collections import defaultdict, deque
import json

try:
    import numpy as np
except ImportError:
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


@dataclass
class PlasticityRule:
    """Configuration for neural plasticity rules."""
    hebbian_rate: float = 0.01
    stdp_tau_pos: float = 20.0  # ms
    stdp_tau_neg: float = 30.0  # ms
    stdp_a_pos: float = 0.1
    stdp_a_neg: float = 0.12
    homeostatic_target: float = 0.1
    homeostatic_rate: float = 0.001
    metaplasticity_threshold: float = 0.5
    

@dataclass
class SynapticConnection:
    """Synaptic connection between hypervector elements."""
    weight: float
    last_pre_spike: float = -float('inf')
    last_post_spike: float = -float('inf')
    trace_pre: float = 0.0
    trace_post: float = 0.0
    activation_history: deque = None
    plasticity_state: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.activation_history is None:
            self.activation_history = deque(maxlen=100)
        if self.plasticity_state is None:
            self.plasticity_state = {
                'hebbian_strength': 0.0,
                'stdp_eligibility': 0.0,
                'homeostatic_scaling': 1.0,
                'metaplasticity_threshold': 0.5
            }


class NeuralPlasticityHDC:
    """
    Hyperdimensional computing with neural plasticity mechanisms.
    
    Revolutionary features:
    1. Hebbian learning for correlation-based weight updates
    2. Spike-timing dependent plasticity (STDP) for temporal learning
    3. Homeostatic plasticity for stability and scaling
    4. Metaplasticity for learning-to-learn capabilities
    5. Synaptic scaling and normalization
    """
    
    def __init__(
        self,
        dim: int = 10000,
        plasticity_rules: Optional[PlasticityRule] = None,
        temporal_resolution: float = 1.0,  # ms
        device: str = "cpu"
    ):
        """Initialize neural plasticity HDC system.
        
        Args:
            dim: Hypervector dimensionality
            plasticity_rules: Plasticity configuration
            temporal_resolution: Time resolution for STDP
            device: Compute device
        """
        self.dim = dim
        self.plasticity_rules = plasticity_rules or PlasticityRule()
        self.temporal_resolution = temporal_resolution
        self.device = device
        
        # Synaptic connectivity matrix
        self.synaptic_matrix = self._initialize_synaptic_matrix()
        
        # Temporal spike traces for STDP
        self.pre_traces = torch.zeros(dim, device=device)
        self.post_traces = torch.zeros(dim, device=device)
        
        # Homeostatic state variables
        self.firing_rates = torch.ones(dim, device=device) * self.plasticity_rules.homeostatic_target
        self.scaling_factors = torch.ones(dim, device=device)
        
        # Metaplasticity variables
        self.plasticity_thresholds = torch.ones(dim, device=device) * self.plasticity_rules.metaplasticity_threshold
        self.learning_history = defaultdict(list)
        
        # Global time counter
        self.current_time = 0.0
        
        # Statistics tracking
        self.plasticity_stats = {
            'hebbian_updates': 0,
            'stdp_updates': 0,
            'homeostatic_updates': 0,
            'total_spikes': 0
        }
        
        logger.info(f"Initialized NeuralPlasticityHDC with {dim} dimensions")
        
    def _initialize_synaptic_matrix(self) -> torch.Tensor:
        """Initialize synaptic connectivity matrix."""
        # Start with small random weights
        weights = torch.randn(self.dim, self.dim, device=self.device) * 0.01
        
        # Ensure no self-connections
        weights.fill_diagonal_(0.0)
        
        # Apply sparse connectivity (biological realism)
        sparsity_mask = torch.rand(self.dim, self.dim, device=self.device) < 0.1  # 10% connectivity
        weights = weights * sparsity_mask.float()
        
        return weights
    
    def spike_encode(self, hv: HyperVector, threshold: float = 0.0) -> torch.Tensor:
        """Convert hypervector to spike pattern.
        
        Args:
            hv: Input hypervector
            threshold: Spiking threshold
            
        Returns:
            Binary spike pattern
        """
        # Apply homeostatic scaling
        scaled_data = hv.data * self.scaling_factors
        
        # Generate spikes above threshold
        spikes = (scaled_data > threshold).float()
        
        # Update firing rates (exponential moving average)
        alpha = 0.1  # Learning rate for firing rate estimation
        self.firing_rates = (1 - alpha) * self.firing_rates + alpha * spikes
        
        # Update global spike count
        self.plasticity_stats['total_spikes'] += spikes.sum().item()
        
        return spikes
    
    def hebbian_update(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor) -> torch.Tensor:
        """Apply Hebbian learning rule.
        
        Args:
            pre_spikes: Presynaptic spike pattern
            post_spikes: Postsynaptic spike pattern
            
        Returns:
            Weight update matrix
        """
        # "Neurons that fire together, wire together"
        correlation = torch.outer(post_spikes, pre_spikes)
        
        # Competitive normalization (prevent runaway growth)
        total_activity = correlation.sum(dim=1, keepdim=True) + 1e-8
        normalized_correlation = correlation / total_activity
        
        # Apply learning rate
        weight_update = self.plasticity_rules.hebbian_rate * normalized_correlation
        
        # Apply metaplasticity threshold modulation
        plasticity_modulation = torch.sigmoid(
            (correlation.sum(dim=1, keepdim=True) - self.plasticity_thresholds.unsqueeze(1)) * 5.0
        )
        weight_update = weight_update * plasticity_modulation
        
        self.plasticity_stats['hebbian_updates'] += 1
        
        return weight_update
    
    def stdp_update(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor, dt: float) -> torch.Tensor:
        """Apply spike-timing dependent plasticity (STDP).
        
        Args:
            pre_spikes: Presynaptic spike pattern
            post_spikes: Postsynaptic spike pattern
            dt: Time difference between pre and post spikes
            
        Returns:
            STDP weight update matrix
        """
        # Update eligibility traces
        tau_pos = self.plasticity_rules.stdp_tau_pos
        tau_neg = self.plasticity_rules.stdp_tau_neg
        
        # Exponential decay of traces
        self.pre_traces *= torch.exp(-torch.tensor(dt / tau_pos))
        self.post_traces *= torch.exp(-torch.tensor(dt / tau_neg))
        
        # Add new spikes to traces
        self.pre_traces += pre_spikes
        self.post_traces += post_spikes
        
        # STDP weight updates
        # Potentiation: post-before-pre (dt > 0)
        potentiation = torch.outer(post_spikes, self.pre_traces) * \
                      self.plasticity_rules.stdp_a_pos * torch.exp(-abs(dt) / tau_pos)
        
        # Depression: pre-before-post (dt < 0)
        depression = torch.outer(self.post_traces, pre_spikes) * \
                    self.plasticity_rules.stdp_a_neg * torch.exp(-abs(dt) / tau_neg)
        
        stdp_update = potentiation - depression
        
        # Apply metaplasticity modulation
        recent_activity = (self.pre_traces + self.post_traces) / 2
        metaplasticity_factor = torch.sigmoid(recent_activity - self.plasticity_thresholds)
        stdp_update = stdp_update * metaplasticity_factor.unsqueeze(1)
        
        self.plasticity_stats['stdp_updates'] += 1
        
        return stdp_update
    
    def homeostatic_scaling(self) -> torch.Tensor:
        """Apply homeostatic plasticity to maintain stable activity.
        
        Returns:
            Scaling factor updates
        """
        target_rate = self.plasticity_rules.homeostatic_target
        rate_error = target_rate - self.firing_rates
        
        # Multiplicative scaling
        scaling_update = 1.0 + self.plasticity_rules.homeostatic_rate * rate_error
        
        # Apply bounds to prevent instability
        scaling_update = torch.clamp(scaling_update, 0.5, 2.0)
        
        # Update scaling factors
        self.scaling_factors *= scaling_update
        
        # Normalize to prevent drift
        self.scaling_factors = self.scaling_factors / self.scaling_factors.mean()
        
        self.plasticity_stats['homeostatic_updates'] += 1
        
        return scaling_update
    
    def metaplasticity_update(self, learning_signal: torch.Tensor):
        """Update metaplasticity thresholds based on learning history.
        
        Args:
            learning_signal: Signal indicating learning effectiveness
        """
        # Track learning history
        self.learning_history['signals'].append(learning_signal.mean().item())
        
        # Compute learning trend (sliding window)
        window_size = 10
        if len(self.learning_history['signals']) >= window_size:
            recent_signals = self.learning_history['signals'][-window_size:]
            learning_trend = sum(recent_signals) / window_size
            
            # Adjust plasticity thresholds
            if learning_trend > 0.7:  # Good learning
                self.plasticity_thresholds *= 0.95  # Lower threshold (more plastic)
            elif learning_trend < 0.3:  # Poor learning
                self.plasticity_thresholds *= 1.05  # Raise threshold (less plastic)
            
            # Keep thresholds in reasonable range
            self.plasticity_thresholds = torch.clamp(self.plasticity_thresholds, 0.1, 1.0)
    
    def plastic_bind(self, hv1: HyperVector, hv2: HyperVector, learning_signal: Optional[torch.Tensor] = None) -> HyperVector:
        """Perform binding with plastic synaptic updates.
        
        Args:
            hv1, hv2: Hypervectors to bind
            learning_signal: Optional learning effectiveness signal
            
        Returns:
            Bound hypervector with plastic modifications
        """
        # Encode inputs as spike patterns
        spikes1 = self.spike_encode(hv1)
        spikes2 = self.spike_encode(hv2)
        
        # Apply synaptic transformation
        transformed1 = torch.matmul(self.synaptic_matrix, spikes1)
        transformed2 = torch.matmul(self.synaptic_matrix, spikes2)
        
        # Standard binding operation
        bound_spikes = spikes1 * spikes2  # Element-wise for spike domain
        
        # Apply plastic updates
        dt = self.temporal_resolution
        
        # Hebbian plasticity
        hebbian_update = self.hebbian_update(spikes1, spikes2)
        
        # STDP plasticity  
        stdp_update = self.stdp_update(spikes1, spikes2, dt)
        
        # Combine plasticity updates
        total_update = hebbian_update + stdp_update
        
        # Apply weight bounds and normalization
        self.synaptic_matrix += total_update
        self.synaptic_matrix = torch.clamp(self.synaptic_matrix, -1.0, 1.0)
        
        # Homeostatic scaling
        self.homeostatic_scaling()
        
        # Metaplasticity update if learning signal provided
        if learning_signal is not None:
            self.metaplasticity_update(learning_signal)
        
        # Convert back to continuous domain
        result_data = bound_spikes + 0.1 * torch.randn_like(bound_spikes)  # Add noise for stability
        
        # Apply final homeostatic scaling
        result_data = result_data * self.scaling_factors
        
        # Update time
        self.current_time += dt
        
        return HyperVector(result_data, device=self.device)
    
    def adaptive_bundle(self, hvs: List[HyperVector], plasticity_weights: Optional[List[float]] = None) -> HyperVector:
        """Bundle hypervectors with adaptive plasticity-based weighting.
        
        Args:
            hvs: List of hypervectors to bundle
            plasticity_weights: Optional plasticity-based weights
            
        Returns:
            Adaptively bundled hypervector
        """
        if plasticity_weights is None:
            # Use synaptic strength as weights
            synaptic_strengths = torch.abs(self.synaptic_matrix).sum(dim=1)
            plasticity_weights = (synaptic_strengths / synaptic_strengths.sum()).tolist()
        
        # Weighted bundling
        weighted_sum = torch.zeros(self.dim, device=self.device)
        total_weight = 0.0
        
        for hv, weight in zip(hvs, plasticity_weights):
            spikes = self.spike_encode(hv)
            
            # Apply synaptic transformation with plasticity weighting
            transformed = torch.matmul(self.synaptic_matrix, spikes) * weight
            weighted_sum += transformed
            total_weight += weight
        
        # Normalize
        if total_weight > 0:
            weighted_sum = weighted_sum / total_weight
        
        # Apply homeostatic scaling
        result = weighted_sum * self.scaling_factors
        
        return HyperVector(result, device=self.device)
    
    def consolidate_memory(self, importance_threshold: float = 0.5):
        """Consolidate synaptic weights based on importance and usage.
        
        Args:
            importance_threshold: Threshold for synapse importance
        """
        # Compute synapse importance based on magnitude and variability
        synapse_magnitudes = torch.abs(self.synaptic_matrix)
        importance_scores = synapse_magnitudes * (1.0 + synapse_magnitudes.std(dim=1, keepdim=True))
        
        # Strengthen important synapses
        important_mask = importance_scores > importance_threshold
        self.synaptic_matrix[important_mask] *= 1.1
        
        # Weaken unimportant synapses (synaptic pruning)
        unimportant_mask = importance_scores < (importance_threshold * 0.5)
        self.synaptic_matrix[unimportant_mask] *= 0.9
        
        # Apply hard bounds
        self.synaptic_matrix = torch.clamp(self.synaptic_matrix, -1.0, 1.0)
        
        logger.info(f"Memory consolidation: {important_mask.sum().item()} synapses strengthened, "
                   f"{unimportant_mask.sum().item()} synapses weakened")
    
    def get_plasticity_state(self) -> Dict[str, Any]:
        """Get current plasticity state for analysis.
        
        Returns:
            Dictionary containing plasticity metrics
        """
        return {
            'synaptic_weights': {
                'mean': self.synaptic_matrix.mean().item(),
                'std': self.synaptic_matrix.std().item(),
                'sparsity': (self.synaptic_matrix == 0).float().mean().item(),
                'max_weight': self.synaptic_matrix.max().item(),
                'min_weight': self.synaptic_matrix.min().item()
            },
            'firing_rates': {
                'mean': self.firing_rates.mean().item(),
                'std': self.firing_rates.std().item(),
                'target': self.plasticity_rules.homeostatic_target
            },
            'plasticity_thresholds': {
                'mean': self.plasticity_thresholds.mean().item(),
                'std': self.plasticity_thresholds.std().item()
            },
            'scaling_factors': {
                'mean': self.scaling_factors.mean().item(),
                'std': self.scaling_factors.std().item()
            },
            'statistics': self.plasticity_stats.copy(),
            'temporal': {
                'current_time': self.current_time,
                'trace_magnitude': (self.pre_traces.norm() + self.post_traces.norm()).item()
            }
        }
    
    def reset_plasticity(self, keep_structure: bool = True):
        """Reset plasticity state while optionally preserving structure.
        
        Args:
            keep_structure: Whether to keep synaptic structure
        """
        if not keep_structure:
            self.synaptic_matrix = self._initialize_synaptic_matrix()
        
        # Reset temporal traces
        self.pre_traces.zero_()
        self.post_traces.zero_()
        
        # Reset homeostatic state
        self.firing_rates.fill_(self.plasticity_rules.homeostatic_target)
        self.scaling_factors.fill_(1.0)
        
        # Reset metaplasticity
        self.plasticity_thresholds.fill_(self.plasticity_rules.metaplasticity_threshold)
        self.learning_history.clear()
        
        # Reset time and statistics
        self.current_time = 0.0
        self.plasticity_stats = {
            'hebbian_updates': 0,
            'stdp_updates': 0,
            'homeostatic_updates': 0,
            'total_spikes': 0
        }
        
        logger.info("Plasticity state reset completed")
    
    def save_plasticity_state(self, filepath: str):
        """Save plasticity state to file.
        
        Args:
            filepath: Path to save plasticity state
        """
        state = {
            'synaptic_matrix': self.synaptic_matrix.cpu().numpy().tolist(),
            'firing_rates': self.firing_rates.cpu().numpy().tolist(),
            'scaling_factors': self.scaling_factors.cpu().numpy().tolist(),
            'plasticity_thresholds': self.plasticity_thresholds.cpu().numpy().tolist(),
            'current_time': self.current_time,
            'statistics': self.plasticity_stats,
            'config': {
                'dim': self.dim,
                'temporal_resolution': self.temporal_resolution,
                'plasticity_rules': {
                    'hebbian_rate': self.plasticity_rules.hebbian_rate,
                    'stdp_tau_pos': self.plasticity_rules.stdp_tau_pos,
                    'stdp_tau_neg': self.plasticity_rules.stdp_tau_neg,
                    'stdp_a_pos': self.plasticity_rules.stdp_a_pos,
                    'stdp_a_neg': self.plasticity_rules.stdp_a_neg,
                    'homeostatic_target': self.plasticity_rules.homeostatic_target,
                    'homeostatic_rate': self.plasticity_rules.homeostatic_rate,
                    'metaplasticity_threshold': self.plasticity_rules.metaplasticity_threshold
                }
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
        
        logger.info(f"Plasticity state saved to {filepath}")
    
    def load_plasticity_state(self, filepath: str):
        """Load plasticity state from file.
        
        Args:
            filepath: Path to load plasticity state from
        """
        with open(filepath, 'r') as f:
            state = json.load(f)
        
        # Restore tensors
        self.synaptic_matrix = torch.tensor(state['synaptic_matrix'], device=self.device)
        self.firing_rates = torch.tensor(state['firing_rates'], device=self.device)
        self.scaling_factors = torch.tensor(state['scaling_factors'], device=self.device)
        self.plasticity_thresholds = torch.tensor(state['plasticity_thresholds'], device=self.device)
        
        # Restore scalars
        self.current_time = state['current_time']
        self.plasticity_stats = state['statistics']
        
        logger.info(f"Plasticity state loaded from {filepath}")


class PlasticityAnalyzer:
    """Analyzer for neural plasticity dynamics and learning curves."""
    
    def __init__(self, plastic_hdc: NeuralPlasticityHDC):
        """Initialize plasticity analyzer.
        
        Args:
            plastic_hdc: NeuralPlasticityHDC instance to analyze
        """
        self.plastic_hdc = plastic_hdc
        self.measurement_history = []
        
    def measure_plasticity_dynamics(self) -> Dict[str, Any]:
        """Measure current plasticity dynamics.
        
        Returns:
            Dictionary with plasticity measurements
        """
        state = self.plastic_hdc.get_plasticity_state()
        
        # Add derived metrics
        derived_metrics = {
            'synaptic_complexity': self._compute_synaptic_complexity(),
            'learning_capacity': self._estimate_learning_capacity(),
            'stability_index': self._compute_stability_index(),
            'plasticity_entropy': self._compute_plasticity_entropy()
        }
        
        state['derived_metrics'] = derived_metrics
        state['timestamp'] = time.time()
        
        self.measurement_history.append(state)
        
        return state
    
    def _compute_synaptic_complexity(self) -> float:
        """Compute synaptic complexity measure."""
        weights = self.plastic_hdc.synaptic_matrix
        
        # Use effective rank as complexity measure
        U, S, V = torch.svd(weights)
        effective_rank = torch.sum(S) / torch.max(S)
        
        return effective_rank.item() / weights.shape[0]
    
    def _estimate_learning_capacity(self) -> float:
        """Estimate remaining learning capacity."""
        # Based on synaptic weight saturation
        weights = torch.abs(self.plastic_hdc.synaptic_matrix)
        saturation = weights.mean() / 1.0  # Assuming max weight is 1.0
        
        return max(0.0, 1.0 - saturation)
    
    def _compute_stability_index(self) -> float:
        """Compute network stability index."""
        if len(self.measurement_history) < 2:
            return 1.0
        
        # Compare recent synaptic weights
        current_weights = self.plastic_hdc.synaptic_matrix
        prev_state = self.measurement_history[-1]
        prev_weights = torch.tensor(prev_state['synaptic_weights'], device=current_weights.device)
        
        # Compute weight change magnitude
        weight_change = torch.norm(current_weights - prev_weights)
        total_weight = torch.norm(current_weights)
        
        stability = 1.0 / (1.0 + weight_change / (total_weight + 1e-8))
        
        return stability.item()
    
    def _compute_plasticity_entropy(self) -> float:
        """Compute entropy of plasticity distribution."""
        weights = torch.abs(self.plastic_hdc.synaptic_matrix).flatten()
        
        # Create histogram
        hist = torch.histc(weights, bins=50, min=0, max=1)
        probs = hist / hist.sum()
        
        # Compute entropy
        entropy = -torch.sum(probs * torch.log(probs + 1e-8))
        
        return entropy.item()
    
    def generate_learning_curve(self) -> Dict[str, List[float]]:
        """Generate learning curve from measurement history.
        
        Returns:
            Dictionary with time series data
        """
        if not self.measurement_history:
            return {}
        
        curves = {
            'time': [m['timestamp'] for m in self.measurement_history],
            'synaptic_mean': [m['synaptic_weights']['mean'] for m in self.measurement_history],
            'synaptic_std': [m['synaptic_weights']['std'] for m in self.measurement_history],
            'firing_rate_mean': [m['firing_rates']['mean'] for m in self.measurement_history],
            'plasticity_entropy': [m['derived_metrics']['plasticity_entropy'] for m in self.measurement_history],
            'learning_capacity': [m['derived_metrics']['learning_capacity'] for m in self.measurement_history],
            'stability_index': [m['derived_metrics']['stability_index'] for m in self.measurement_history]
        }
        
        return curves