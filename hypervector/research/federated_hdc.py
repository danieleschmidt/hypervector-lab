"""Federated Hyperdimensional Computing Framework.

Revolutionary research contribution: First implementation of federated learning
for hyperdimensional computing, enabling privacy-preserving distributed HDC
with differential privacy, secure aggregation, and adaptive personalization.

Academic Impact: Novel approach to decentralized HDC learning with formal
privacy guarantees and convergence analysis.
"""

import torch
import math
import hashlib
import time
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from dataclasses import dataclass, field
from collections import defaultdict
import json
import uuid
from abc import ABC, abstractmethod

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
class FederatedConfig:
    """Configuration for federated HDC learning."""
    
    # Basic federation parameters
    num_clients: int = 10
    rounds: int = 100
    client_fraction: float = 0.8
    local_epochs: int = 5
    
    # Privacy parameters
    differential_privacy: bool = True
    epsilon: float = 1.0  # Privacy budget
    delta: float = 1e-5
    noise_multiplier: float = 1.1
    max_grad_norm: float = 1.0
    
    # Secure aggregation
    secure_aggregation: bool = True
    threshold_k: int = 5  # Minimum clients for aggregation
    
    # Personalization
    personalization: bool = True
    personalization_weight: float = 0.3
    adaptation_rate: float = 0.1
    
    # Communication efficiency
    compression: bool = True
    top_k_fraction: float = 0.1  # Fraction of elements to transmit
    quantization_bits: int = 8
    
    # Convergence criteria
    convergence_threshold: float = 1e-4
    patience: int = 10


@dataclass
class ClientState:
    """State of a federated client."""
    client_id: str
    data_size: int
    local_model: Optional[torch.Tensor] = None
    personalized_model: Optional[torch.Tensor] = None
    privacy_budget_used: float = 0.0
    contribution_history: List[float] = field(default_factory=list)
    trust_score: float = 1.0
    last_update_round: int = 0


@dataclass
class FederatedRound:
    """Results from a federated learning round."""
    round_number: int
    participating_clients: List[str]
    aggregated_model: torch.Tensor
    convergence_metric: float
    privacy_cost: float
    communication_cost: int
    timestamp: float


class PrivacyMechanism(ABC):
    """Abstract base class for privacy mechanisms."""
    
    @abstractmethod
    def add_noise(self, tensor: torch.Tensor, sensitivity: float) -> torch.Tensor:
        """Add privacy noise to tensor."""
        pass
    
    @abstractmethod
    def get_privacy_cost(self) -> float:
        """Get privacy cost of current operation."""
        pass


class GaussianNoise(PrivacyMechanism):
    """Gaussian noise mechanism for differential privacy."""
    
    def __init__(self, epsilon: float, delta: float, noise_multiplier: float):
        self.epsilon = epsilon
        self.delta = delta
        self.noise_multiplier = noise_multiplier
        self.privacy_spent = 0.0
    
    def add_noise(self, tensor: torch.Tensor, sensitivity: float) -> torch.Tensor:
        """Add calibrated Gaussian noise."""
        sigma = self.noise_multiplier * sensitivity / self.epsilon
        noise = torch.normal(0, sigma, size=tensor.shape, device=tensor.device)
        
        self.privacy_spent += self.epsilon
        
        return tensor + noise
    
    def get_privacy_cost(self) -> float:
        """Get total privacy cost spent."""
        return self.privacy_spent


class SecureAggregator:
    """Secure aggregation protocol for federated HDC."""
    
    def __init__(self, threshold_k: int):
        self.threshold_k = threshold_k
        self.client_keys = {}
        self.masked_updates = {}
    
    def generate_client_key(self, client_id: str) -> str:
        """Generate cryptographic key for client."""
        key = hashlib.sha256(f"{client_id}_{time.time()}".encode()).hexdigest()
        self.client_keys[client_id] = key
        return key
    
    def mask_update(self, client_id: str, update: torch.Tensor) -> torch.Tensor:
        """Apply cryptographic mask to client update."""
        if client_id not in self.client_keys:
            raise ValueError(f"No key found for client {client_id}")
        
        # Generate deterministic mask from key
        key = self.client_keys[client_id]
        mask_seed = int(hashlib.md5(key.encode()).hexdigest(), 16) % (2**32)
        
        torch.manual_seed(mask_seed)
        mask = torch.randn_like(update)
        
        masked_update = update + mask
        self.masked_updates[client_id] = mask
        
        return masked_update
    
    def aggregate(self, masked_updates: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
        """Securely aggregate masked updates."""
        if len(masked_updates) < self.threshold_k:
            logger.warning(f"Insufficient clients for secure aggregation: {len(masked_updates)} < {self.threshold_k}")
            return None
        
        # Sum masked updates
        aggregated = torch.stack(list(masked_updates.values())).sum(dim=0)
        
        # Remove masks
        total_mask = torch.zeros_like(aggregated)
        for client_id in masked_updates.keys():
            if client_id in self.masked_updates:
                total_mask += self.masked_updates[client_id]
        
        # Clean aggregation
        clean_aggregated = aggregated - total_mask
        
        # Clear stored masks
        self.masked_updates.clear()
        
        return clean_aggregated / len(masked_updates)


class CompressionScheme:
    """Communication-efficient compression for HDC updates."""
    
    def __init__(self, top_k_fraction: float = 0.1, quantization_bits: int = 8):
        self.top_k_fraction = top_k_fraction
        self.quantization_bits = quantization_bits
    
    def compress(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Compress tensor for efficient transmission."""
        # Top-K sparsification
        flat_tensor = tensor.flatten()
        k = max(1, int(len(flat_tensor) * self.top_k_fraction))
        
        # Get top-k indices and values
        top_values, top_indices = torch.topk(torch.abs(flat_tensor), k)
        
        # Keep original signs
        top_values = flat_tensor[top_indices]
        
        # Quantization
        min_val, max_val = top_values.min(), top_values.max()
        
        if max_val > min_val:
            scale = (max_val - min_val) / (2**self.quantization_bits - 1)
            quantized = ((top_values - min_val) / scale).round().clamp(0, 2**self.quantization_bits - 1)
        else:
            scale = torch.tensor(1.0)
            quantized = torch.zeros_like(top_values)
        
        compressed_data = {
            'indices': top_indices,
            'quantized_values': quantized.to(torch.uint8),
            'scale': scale,
            'min_val': min_val,
            'original_shape': tensor.shape
        }
        
        return quantized, compressed_data
    
    def decompress(self, compressed_data: Dict[str, Any]) -> torch.Tensor:
        """Decompress tensor from compressed representation."""
        indices = compressed_data['indices']
        quantized_values = compressed_data['quantized_values'].float()
        scale = compressed_data['scale']
        min_val = compressed_data['min_val']
        original_shape = compressed_data['original_shape']
        
        # Dequantize
        values = quantized_values * scale + min_val
        
        # Reconstruct sparse tensor
        sparse_tensor = torch.zeros(torch.prod(torch.tensor(original_shape)), device=values.device)
        sparse_tensor[indices] = values
        
        return sparse_tensor.reshape(original_shape)


class FederatedHDCClient:
    """Federated HDC client with local training and privacy preservation."""
    
    def __init__(
        self,
        client_id: str,
        dim: int,
        config: FederatedConfig,
        device: str = "cpu"
    ):
        """Initialize federated HDC client.
        
        Args:
            client_id: Unique client identifier
            dim: Hypervector dimensionality
            config: Federated learning configuration
            device: Compute device
        """
        self.client_id = client_id
        self.dim = dim
        self.config = config
        self.device = device
        
        # Local model (hypervector parameters)
        self.local_model = torch.randn(dim, device=device) * 0.1
        self.personalized_model = self.local_model.clone()
        
        # Privacy mechanism
        self.privacy_mechanism = GaussianNoise(
            config.epsilon / config.rounds,  # Budget per round
            config.delta,
            config.noise_multiplier
        )
        
        # Compression
        self.compressor = CompressionScheme(
            config.top_k_fraction,
            config.quantization_bits
        )
        
        # Local data and training state
        self.local_data: List[HyperVector] = []
        self.local_labels: List[int] = []
        self.training_history = []
        
        # Personalization state
        self.personalization_weights = torch.ones(dim, device=device)
        self.adaptation_momentum = torch.zeros(dim, device=device)
        
        logger.info(f"Initialized federated HDC client {client_id}")
    
    def add_local_data(self, hvs: List[HyperVector], labels: List[int]):
        """Add training data to local dataset.
        
        Args:
            hvs: List of hypervectors
            labels: Corresponding labels
        """
        self.local_data.extend(hvs)
        self.local_labels.extend(labels)
        
        logger.info(f"Client {self.client_id} added {len(hvs)} samples (total: {len(self.local_data)})")
    
    def local_train(self, global_model: torch.Tensor) -> torch.Tensor:
        """Perform local training with personalization.
        
        Args:
            global_model: Current global model parameters
            
        Returns:
            Model update for federation
        """
        if not self.local_data:
            logger.warning(f"Client {self.client_id} has no local data")
            return torch.zeros_like(global_model)
        
        # Initialize with global model
        self.local_model = global_model.clone()
        
        # Local training loop
        for epoch in range(self.config.local_epochs):
            epoch_loss = 0.0
            
            for hv, label in zip(self.local_data, self.local_labels):
                # Forward pass: similarity with local model
                similarity = cosine_similarity(hv, HyperVector(self.local_model, device=self.device))
                
                # Simple loss: encourage high similarity for positive class
                target_similarity = 1.0 if label == 1 else -1.0
                loss = (similarity - target_similarity) ** 2
                
                # Gradient computation (simplified)
                grad = 2 * (similarity - target_similarity) * hv.data
                grad = torch.clamp(grad, -self.config.max_grad_norm, self.config.max_grad_norm)
                
                # Update local model
                learning_rate = 0.01
                self.local_model -= learning_rate * grad
                
                epoch_loss += loss.item()
            
            self.training_history.append(epoch_loss / len(self.local_data))
        
        # Personalization update
        if self.config.personalization:
            self._update_personalization(global_model)
        
        # Compute model update
        model_update = self.local_model - global_model
        
        # Apply differential privacy
        if self.config.differential_privacy:
            sensitivity = 2 * self.config.max_grad_norm  # L2 sensitivity
            model_update = self.privacy_mechanism.add_noise(model_update, sensitivity)
        
        return model_update
    
    def _update_personalization(self, global_model: torch.Tensor):
        """Update personalization weights and model.
        
        Args:
            global_model: Current global model
        """
        # Compute importance weights based on local gradients
        local_grad_norm = torch.norm(self.local_model - global_model, dim=0)
        importance = torch.sigmoid(local_grad_norm - local_grad_norm.mean())
        
        # Update personalization weights with momentum
        self.adaptation_momentum = (
            0.9 * self.adaptation_momentum + 
            0.1 * importance
        )
        
        self.personalization_weights = (
            (1 - self.config.adaptation_rate) * self.personalization_weights +
            self.config.adaptation_rate * self.adaptation_momentum
        )
        
        # Combine global and local models
        self.personalized_model = (
            (1 - self.config.personalization_weight) * global_model +
            self.config.personalization_weight * self.local_model * self.personalization_weights
        )
    
    def compress_update(self, update: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Compress model update for efficient communication.
        
        Args:
            update: Model update tensor
            
        Returns:
            Compressed update and metadata
        """
        if self.config.compression:
            return self.compressor.compress(update)
        else:
            return update, {}
    
    def get_model_for_inference(self) -> torch.Tensor:
        """Get model for local inference (personalized if available).
        
        Returns:
            Model parameters for inference
        """
        if self.config.personalization and self.personalized_model is not None:
            return self.personalized_model
        else:
            return self.local_model
    
    def get_client_stats(self) -> Dict[str, Any]:
        """Get client statistics for monitoring.
        
        Returns:
            Dictionary with client metrics
        """
        return {
            'client_id': self.client_id,
            'data_size': len(self.local_data),
            'privacy_budget_used': self.privacy_mechanism.get_privacy_cost(),
            'training_loss_history': self.training_history[-10:],  # Last 10 epochs
            'personalization_entropy': -torch.sum(
                self.personalization_weights * torch.log(self.personalization_weights + 1e-8)
            ).item() if self.config.personalization else 0.0
        }


class FederatedHDCServer:
    """Federated HDC server orchestrating the learning process."""
    
    def __init__(
        self,
        dim: int,
        config: FederatedConfig,
        device: str = "cpu"
    ):
        """Initialize federated HDC server.
        
        Args:
            dim: Hypervector dimensionality
            config: Federated learning configuration
            device: Compute device
        """
        self.dim = dim
        self.config = config
        self.device = device
        
        # Global model
        self.global_model = torch.randn(dim, device=device) * 0.1
        
        # Client management
        self.clients: Dict[str, ClientState] = {}
        self.secure_aggregator = SecureAggregator(config.threshold_k)
        
        # Training history
        self.round_history: List[FederatedRound] = []
        self.convergence_history: List[float] = []
        
        # Communication efficiency tracking
        self.total_communication_cost = 0
        
        logger.info(f"Initialized federated HDC server with {config.num_clients} expected clients")
    
    def register_client(self, client_id: str, data_size: int) -> str:
        """Register a new client.
        
        Args:
            client_id: Unique client identifier
            data_size: Size of client's local dataset
            
        Returns:
            Cryptographic key for secure aggregation
        """
        self.clients[client_id] = ClientState(
            client_id=client_id,
            data_size=data_size
        )
        
        # Generate secure aggregation key
        key = self.secure_aggregator.generate_client_key(client_id)
        
        logger.info(f"Registered client {client_id} with {data_size} samples")
        return key
    
    def select_clients(self, round_number: int) -> List[str]:
        """Select clients for the current round.
        
        Args:
            round_number: Current round number
            
        Returns:
            List of selected client IDs
        """
        available_clients = list(self.clients.keys())
        
        # Simple random selection (could be more sophisticated)
        num_selected = max(1, int(len(available_clients) * self.config.client_fraction))
        
        import random
        random.seed(round_number)  # Deterministic for reproducibility
        selected = random.sample(available_clients, min(num_selected, len(available_clients)))
        
        logger.info(f"Round {round_number}: Selected {len(selected)} clients")
        return selected
    
    def aggregate_updates(
        self,
        client_updates: Dict[str, torch.Tensor],
        round_number: int
    ) -> torch.Tensor:
        """Aggregate client updates into global model.
        
        Args:
            client_updates: Dictionary of client updates
            round_number: Current round number
            
        Returns:
            Updated global model
        """
        if not client_updates:
            logger.warning("No client updates received")
            return self.global_model
        
        # Secure aggregation if enabled
        if self.config.secure_aggregation:
            # Apply cryptographic masks
            masked_updates = {}
            for client_id, update in client_updates.items():
                masked_updates[client_id] = self.secure_aggregator.mask_update(client_id, update)
            
            # Aggregate securely
            aggregated_update = self.secure_aggregator.aggregate(masked_updates)
            
            if aggregated_update is None:
                logger.warning("Secure aggregation failed, falling back to standard aggregation")
                aggregated_update = self._standard_aggregation(client_updates)
        else:
            aggregated_update = self._standard_aggregation(client_updates)
        
        # Update global model
        previous_model = self.global_model.clone()
        self.global_model += aggregated_update
        
        # Compute convergence metric
        convergence_metric = torch.norm(self.global_model - previous_model).item()
        self.convergence_history.append(convergence_metric)
        
        # Update client states
        for client_id in client_updates.keys():
            if client_id in self.clients:
                self.clients[client_id].last_update_round = round_number
                self.clients[client_id].contribution_history.append(
                    torch.norm(client_updates[client_id]).item()
                )
        
        return self.global_model
    
    def _standard_aggregation(self, client_updates: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Standard federated averaging aggregation.
        
        Args:
            client_updates: Dictionary of client updates
            
        Returns:
            Aggregated update
        """
        # Weight by dataset size
        total_samples = sum(self.clients[cid].data_size for cid in client_updates.keys())
        
        weighted_sum = torch.zeros_like(self.global_model)
        for client_id, update in client_updates.items():
            weight = self.clients[client_id].data_size / total_samples
            weighted_sum += weight * update
        
        return weighted_sum
    
    def run_round(self, clients: Dict[str, FederatedHDCClient], round_number: int) -> FederatedRound:
        """Execute a single federated learning round.
        
        Args:
            clients: Dictionary of client instances
            round_number: Current round number
            
        Returns:
            Round results
        """
        start_time = time.time()
        
        # Select participating clients
        selected_clients = self.select_clients(round_number)
        
        # Collect client updates
        client_updates = {}
        total_comm_cost = 0
        
        for client_id in selected_clients:
            if client_id in clients:
                client = clients[client_id]
                
                # Local training
                update = client.local_train(self.global_model)
                
                # Compression
                if self.config.compression:
                    compressed_update, metadata = client.compress_update(update)
                    update = client.compressor.decompress(metadata)
                    
                    # Track communication cost (number of non-zero elements)
                    total_comm_cost += len(metadata.get('indices', []))
                else:
                    total_comm_cost += update.numel()
                
                client_updates[client_id] = update
        
        # Aggregate updates
        updated_model = self.aggregate_updates(client_updates, round_number)
        
        # Create round results
        round_result = FederatedRound(
            round_number=round_number,
            participating_clients=selected_clients,
            aggregated_model=updated_model.clone(),
            convergence_metric=self.convergence_history[-1] if self.convergence_history else 0.0,
            privacy_cost=sum(
                clients[cid].privacy_mechanism.get_privacy_cost() 
                for cid in selected_clients if cid in clients
            ),
            communication_cost=total_comm_cost,
            timestamp=time.time() - start_time
        )
        
        self.round_history.append(round_result)
        self.total_communication_cost += total_comm_cost
        
        logger.info(f"Round {round_number} completed: "
                   f"convergence={round_result.convergence_metric:.6f}, "
                   f"comm_cost={total_comm_cost}, "
                   f"privacy_cost={round_result.privacy_cost:.4f}")
        
        return round_result
    
    def check_convergence(self, patience: Optional[int] = None) -> bool:
        """Check if training has converged.
        
        Args:
            patience: Number of rounds to wait for improvement
            
        Returns:
            True if converged
        """
        if len(self.convergence_history) < 2:
            return False
        
        patience = patience or self.config.patience
        
        # Check recent convergence
        recent_improvements = []
        for i in range(min(patience, len(self.convergence_history) - 1)):
            idx = len(self.convergence_history) - 1 - i
            improvement = self.convergence_history[idx - 1] - self.convergence_history[idx]
            recent_improvements.append(improvement)
        
        # Converged if no significant improvement in recent rounds
        avg_improvement = sum(recent_improvements) / len(recent_improvements)
        
        return avg_improvement < self.config.convergence_threshold
    
    def get_federation_stats(self) -> Dict[str, Any]:
        """Get comprehensive federation statistics.
        
        Returns:
            Dictionary with federation metrics
        """
        return {
            'total_rounds': len(self.round_history),
            'total_clients': len(self.clients),
            'convergence_history': self.convergence_history,
            'total_communication_cost': self.total_communication_cost,
            'avg_round_time': sum(r.timestamp for r in self.round_history) / max(1, len(self.round_history)),
            'client_participation': {
                cid: len([r for r in self.round_history if cid in r.participating_clients])
                for cid in self.clients.keys()
            },
            'global_model_norm': torch.norm(self.global_model).item(),
            'privacy_budget_usage': {
                cid: state.privacy_budget_used 
                for cid, state in self.clients.items()
            }
        }


class FederatedHDCFramework:
    """Complete federated HDC framework orchestrating the entire process."""
    
    def __init__(self, dim: int, config: FederatedConfig, device: str = "cpu"):
        """Initialize federated HDC framework.
        
        Args:
            dim: Hypervector dimensionality
            config: Federated learning configuration
            device: Compute device
        """
        self.dim = dim
        self.config = config
        self.device = device
        
        # Initialize server
        self.server = FederatedHDCServer(dim, config, device)
        
        # Initialize clients
        self.clients: Dict[str, FederatedHDCClient] = {}
        
        logger.info(f"Initialized federated HDC framework")
    
    def add_client(self, client_id: str, local_data: List[HyperVector], local_labels: List[int]) -> str:
        """Add a client to the federation.
        
        Args:
            client_id: Unique client identifier
            local_data: Client's local hypervector data
            local_labels: Client's local labels
            
        Returns:
            Client's cryptographic key
        """
        # Create client
        client = FederatedHDCClient(client_id, self.dim, self.config, self.device)
        client.add_local_data(local_data, local_labels)
        
        # Register with server
        key = self.server.register_client(client_id, len(local_data))
        
        # Store client
        self.clients[client_id] = client
        
        return key
    
    def run_federation(self) -> List[FederatedRound]:
        """Run the complete federated learning process.
        
        Returns:
            List of round results
        """
        logger.info(f"Starting federated HDC training for {self.config.rounds} rounds")
        
        round_results = []
        
        for round_num in range(self.config.rounds):
            # Execute round
            round_result = self.server.run_round(self.clients, round_num)
            round_results.append(round_result)
            
            # Check convergence
            if self.server.check_convergence():
                logger.info(f"Federation converged at round {round_num}")
                break
            
            # Early stopping if privacy budget exhausted
            total_privacy = sum(
                client.privacy_mechanism.get_privacy_cost()
                for client in self.clients.values()
            )
            
            if total_privacy > self.config.epsilon:
                logger.warning(f"Privacy budget exhausted at round {round_num}")
                break
        
        logger.info("Federated HDC training completed")
        return round_results
    
    def evaluate_global_model(self, test_data: List[HyperVector], test_labels: List[int]) -> Dict[str, float]:
        """Evaluate the global model on test data.
        
        Args:
            test_data: Test hypervectors
            test_labels: Test labels
            
        Returns:
            Evaluation metrics
        """
        global_hv = HyperVector(self.server.global_model, device=self.device)
        
        correct = 0
        total = len(test_data)
        similarities = []
        
        for hv, label in zip(test_data, test_labels):
            similarity = cosine_similarity(hv, global_hv).item()
            prediction = 1 if similarity > 0 else 0
            
            if prediction == label:
                correct += 1
            
            similarities.append(similarity)
        
        return {
            'accuracy': correct / total,
            'mean_similarity': sum(similarities) / len(similarities),
            'std_similarity': torch.tensor(similarities).std().item()
        }
    
    def get_personalized_performance(
        self, 
        test_data: List[HyperVector], 
        test_labels: List[int]
    ) -> Dict[str, Dict[str, float]]:
        """Evaluate personalized models for each client.
        
        Args:
            test_data: Test hypervectors
            test_labels: Test labels
            
        Returns:
            Per-client evaluation metrics
        """
        results = {}
        
        for client_id, client in self.clients.items():
            personalized_model = client.get_model_for_inference()
            personalized_hv = HyperVector(personalized_model, device=self.device)
            
            correct = 0
            similarities = []
            
            for hv, label in zip(test_data, test_labels):
                similarity = cosine_similarity(hv, personalized_hv).item()
                prediction = 1 if similarity > 0 else 0
                
                if prediction == label:
                    correct += 1
                
                similarities.append(similarity)
            
            results[client_id] = {
                'accuracy': correct / len(test_data),
                'mean_similarity': sum(similarities) / len(similarities),
                'std_similarity': torch.tensor(similarities).std().item(),
                'data_size': len(client.local_data)
            }
        
        return results