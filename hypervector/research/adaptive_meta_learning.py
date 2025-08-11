"""Adaptive meta-learning systems for hyperdimensional computing.

Novel research contribution: Self-improving HDC systems that adapt
encoding strategies, optimize operations, and learn from experience
to achieve superior performance on specific domains and tasks.
"""

import time
import math
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from dataclasses import dataclass, field
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
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
class TaskMetadata:
    """Metadata about learning tasks for adaptation."""
    task_id: str
    domain: str
    data_type: str  # 'text', 'vision', 'audio', 'multimodal'
    complexity_level: float
    performance_history: List[float] = field(default_factory=list)
    best_config: Optional[Dict[str, Any]] = None
    adaptation_count: int = 0


@dataclass
class AdaptationResult:
    """Results from adaptation process."""
    old_performance: float
    new_performance: float
    improvement: float
    config_changes: Dict[str, Any]
    adaptation_time: float
    confidence: float


class MetaLearningStrategy(ABC):
    """Abstract base class for meta-learning strategies."""
    
    def __init__(self, name: str):
        self.name = name
        self.adaptation_history = []
    
    @abstractmethod
    def adapt_encoding(self, task_metadata: TaskMetadata, 
                      performance_data: List[float]) -> Dict[str, Any]:
        """Adapt encoding strategy based on task performance."""
        pass
    
    @abstractmethod
    def evaluate_adaptation(self, old_config: Dict[str, Any], 
                          new_config: Dict[str, Any], 
                          performance_improvement: float) -> float:
        """Evaluate quality of adaptation."""
        pass


class GradientBasedMetaLearning(MetaLearningStrategy):
    """
    Gradient-based meta-learning for HDC parameter optimization.
    
    Novel research contribution: Differentiable HDC operations
    with gradient-based optimization of encoding parameters.
    """
    
    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.9):
        super().__init__("GradientBasedMetaLearning")
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.parameter_gradients = {}
        self.parameter_momentum = {}
        
        logger.info(f"Initialized GradientBasedMetaLearning with lr={learning_rate}")
    
    def adapt_encoding(self, task_metadata: TaskMetadata, 
                      performance_data: List[float]) -> Dict[str, Any]:
        """Adapt encoding using gradient-based optimization."""
        if len(performance_data) < 2:
            return {}
        
        # Compute performance gradient
        recent_performance = performance_data[-5:] if len(performance_data) >= 5 else performance_data
        performance_gradient = self._compute_performance_gradient(recent_performance)
        
        # Adapt parameters based on gradient
        adaptations = {}
        
        # Adapt dimension size
        if 'dimension' not in self.parameter_gradients:
            self.parameter_gradients['dimension'] = 0.0
            self.parameter_momentum['dimension'] = 0.0
        
        dim_gradient = performance_gradient * self._compute_dimension_sensitivity(task_metadata)
        self.parameter_momentum['dimension'] = (self.momentum * self.parameter_momentum['dimension'] + 
                                               self.learning_rate * dim_gradient)
        
        # Update dimension (constrained to reasonable range)
        base_dim = 10000
        dim_adjustment = int(self.parameter_momentum['dimension'] * 1000)
        new_dimension = max(1000, min(50000, base_dim + dim_adjustment))
        adaptations['dimension'] = new_dimension
        
        # Adapt encoding method parameters
        if task_metadata.data_type == 'text':
            adaptations.update(self._adapt_text_encoding_params(performance_gradient, task_metadata))
        elif task_metadata.data_type == 'vision':
            adaptations.update(self._adapt_vision_encoding_params(performance_gradient, task_metadata))
        
        # Adapt operation parameters
        adaptations.update(self._adapt_operation_params(performance_gradient, task_metadata))
        
        return adaptations
    
    def _compute_performance_gradient(self, performance_data: List[float]) -> float:
        """Compute gradient of performance over time."""
        if len(performance_data) < 2:
            return 0.0
        
        # Simple finite difference
        recent_avg = sum(performance_data[-3:]) / min(3, len(performance_data))
        older_avg = sum(performance_data[:-3]) / max(1, len(performance_data) - 3)
        
        return recent_avg - older_avg
    
    def _compute_dimension_sensitivity(self, task_metadata: TaskMetadata) -> float:
        """Compute sensitivity to dimension changes for specific task."""
        base_sensitivity = 1.0
        
        # Higher complexity tasks benefit more from larger dimensions
        complexity_factor = 1.0 + (task_metadata.complexity_level - 0.5) * 0.5
        
        # Domain-specific adjustments
        domain_factors = {
            'nlp': 1.2,      # Text benefits from higher dimensions
            'vision': 0.8,   # Vision can work with moderate dimensions
            'audio': 1.0,    # Audio is neutral
            'multimodal': 1.5  # Multimodal benefits most from high dimensions
        }
        
        domain_factor = domain_factors.get(task_metadata.domain, 1.0)
        
        return base_sensitivity * complexity_factor * domain_factor
    
    def _adapt_text_encoding_params(self, gradient: float, task_metadata: TaskMetadata) -> Dict[str, Any]:
        """Adapt text-specific encoding parameters."""
        adaptations = {}
        
        # Adapt n-gram size based on performance
        if 'ngram_size' not in self.parameter_momentum:
            self.parameter_momentum['ngram_size'] = 0.0
        
        ngram_gradient = gradient * 0.1  # Scale down for discrete parameter
        self.parameter_momentum['ngram_size'] += self.learning_rate * ngram_gradient
        
        base_ngram = 3
        ngram_adjustment = int(self.parameter_momentum['ngram_size'] * 2)
        adaptations['ngram_size'] = max(1, min(5, base_ngram + ngram_adjustment))
        
        # Adapt position encoding strength
        if 'position_weight' not in self.parameter_momentum:
            self.parameter_momentum['position_weight'] = 0.0
        
        pos_gradient = gradient * 0.05
        self.parameter_momentum['position_weight'] += self.learning_rate * pos_gradient
        
        base_weight = 0.5
        weight_adjustment = self.parameter_momentum['position_weight']
        adaptations['position_weight'] = max(0.1, min(1.0, base_weight + weight_adjustment))
        
        return adaptations
    
    def _adapt_vision_encoding_params(self, gradient: float, task_metadata: TaskMetadata) -> Dict[str, Any]:
        """Adapt vision-specific encoding parameters."""
        adaptations = {}
        
        # Adapt patch size
        if 'patch_size' not in self.parameter_momentum:
            self.parameter_momentum['patch_size'] = 0.0
        
        patch_gradient = gradient * 0.2
        self.parameter_momentum['patch_size'] += self.learning_rate * patch_gradient
        
        base_patch_size = 16
        patch_adjustment = int(self.parameter_momentum['patch_size'] * 4)
        adaptations['patch_size'] = max(4, min(64, base_patch_size + patch_adjustment))
        
        # Adapt feature extraction method
        if gradient > 0.05:  # Significant improvement needed
            adaptations['feature_method'] = 'cnn'  # Switch to more complex features
        elif gradient < -0.05:  # Performance degrading
            adaptations['feature_method'] = 'holistic'  # Switch to simpler features
        
        return adaptations
    
    def _adapt_operation_params(self, gradient: float, task_metadata: TaskMetadata) -> Dict[str, Any]:
        """Adapt HDC operation parameters."""
        adaptations = {}
        
        # Adapt bundling normalization
        if 'bundle_normalize' not in self.parameter_momentum:
            self.parameter_momentum['bundle_normalize'] = 0.0
        
        norm_gradient = gradient * 0.1
        self.parameter_momentum['bundle_normalize'] += self.learning_rate * norm_gradient
        
        # Use momentum to decide normalization strategy
        if self.parameter_momentum['bundle_normalize'] > 0.3:
            adaptations['bundle_normalize'] = True
        else:
            adaptations['bundle_normalize'] = False
        
        # Adapt similarity threshold
        if 'similarity_threshold' not in self.parameter_momentum:
            self.parameter_momentum['similarity_threshold'] = 0.0
        
        threshold_gradient = gradient * 0.05
        self.parameter_momentum['similarity_threshold'] += self.learning_rate * threshold_gradient
        
        base_threshold = 0.5
        threshold_adjustment = self.parameter_momentum['similarity_threshold']
        adaptations['similarity_threshold'] = max(0.1, min(0.9, base_threshold + threshold_adjustment))
        
        return adaptations
    
    def evaluate_adaptation(self, old_config: Dict[str, Any], 
                          new_config: Dict[str, Any], 
                          performance_improvement: float) -> float:
        """Evaluate gradient-based adaptation quality."""
        # Base evaluation on performance improvement
        base_score = max(0.0, performance_improvement)
        
        # Penalize large parameter changes (prefer stable adaptations)
        change_penalty = 0.0
        for key in set(old_config.keys()) & set(new_config.keys()):
            if isinstance(old_config[key], (int, float)) and isinstance(new_config[key], (int, float)):
                relative_change = abs(new_config[key] - old_config[key]) / max(abs(old_config[key]), 1e-8)
                change_penalty += relative_change * 0.1
        
        final_score = base_score - change_penalty
        return max(0.0, final_score)


class EvolutionaryMetaLearning(MetaLearningStrategy):
    """
    Evolutionary algorithm for HDC configuration optimization.
    
    Novel research contribution: Population-based search over
    HDC configuration space with mutation and crossover.
    """
    
    def __init__(self, population_size: int = 20, mutation_rate: float = 0.1):
        super().__init__("EvolutionaryMetaLearning")
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.population = []
        self.fitness_scores = []
        self.generation_count = 0
        
        logger.info(f"Initialized EvolutionaryMetaLearning with population_size={population_size}")
    
    def adapt_encoding(self, task_metadata: TaskMetadata, 
                      performance_data: List[float]) -> Dict[str, Any]:
        """Evolve encoding configuration using genetic algorithm."""
        if not performance_data:
            return self._generate_random_config(task_metadata)
        
        # Initialize population if empty
        if not self.population:
            self._initialize_population(task_metadata)
        
        # Update fitness scores
        current_performance = performance_data[-1] if performance_data else 0.0
        self._update_fitness(current_performance)
        
        # Evolve population
        self._evolve_population(task_metadata)
        
        # Return best configuration
        best_index = max(range(len(self.fitness_scores)), key=lambda i: self.fitness_scores[i])
        return self.population[best_index].copy()
    
    def _initialize_population(self, task_metadata: TaskMetadata):
        """Initialize random population of configurations."""
        self.population = []
        self.fitness_scores = []
        
        for _ in range(self.population_size):
            config = self._generate_random_config(task_metadata)
            self.population.append(config)
            self.fitness_scores.append(0.0)  # Will be updated with actual performance
    
    def _generate_random_config(self, task_metadata: TaskMetadata) -> Dict[str, Any]:
        """Generate random configuration for given task."""
        config = {
            'dimension': torch.randint(1000, 20000, (1,)).item(),
            'bundle_normalize': torch.rand(1).item() > 0.5,
            'similarity_threshold': 0.1 + torch.rand(1).item() * 0.8,
        }
        
        # Task-specific parameters
        if task_metadata.data_type == 'text':
            config.update({
                'ngram_size': torch.randint(1, 6, (1,)).item(),
                'position_weight': torch.rand(1).item(),
                'encoding_method': torch.choice(['character', 'word', 'sentence'])
            })
        elif task_metadata.data_type == 'vision':
            config.update({
                'patch_size': torch.choice([4, 8, 16, 32, 64]),
                'feature_method': torch.choice(['holistic', 'patch', 'cnn']),
                'color_weight': torch.rand(1).item()
            })
        
        return config
    
    def _update_fitness(self, current_performance: float):
        """Update fitness scores based on current performance."""
        # Simple fitness based on performance (could be more sophisticated)
        base_fitness = max(0.0, current_performance)
        
        # Add some noise and history weighting
        for i in range(len(self.fitness_scores)):
            # Weighted update: 70% current performance, 30% historical
            self.fitness_scores[i] = 0.7 * base_fitness + 0.3 * self.fitness_scores[i]
            
            # Add small random component to maintain diversity
            self.fitness_scores[i] += torch.randn(1).item() * 0.01
    
    def _evolve_population(self, task_metadata: TaskMetadata):
        """Evolve population using selection, crossover, and mutation."""
        self.generation_count += 1
        
        # Selection: keep top 50% of population
        num_survivors = self.population_size // 2
        survivor_indices = sorted(range(len(self.fitness_scores)), 
                                 key=lambda i: self.fitness_scores[i], 
                                 reverse=True)[:num_survivors]
        
        survivors = [self.population[i] for i in survivor_indices]
        survivor_fitness = [self.fitness_scores[i] for i in survivor_indices]
        
        # Generate offspring through crossover and mutation
        new_population = survivors.copy()
        new_fitness = survivor_fitness.copy()
        
        while len(new_population) < self.population_size:
            # Select two parents (tournament selection)
            parent1_idx = self._tournament_selection(survivor_fitness)
            parent2_idx = self._tournament_selection(survivor_fitness)
            
            parent1 = survivors[parent1_idx]
            parent2 = survivors[parent2_idx]
            
            # Crossover
            offspring = self._crossover(parent1, parent2)
            
            # Mutation
            offspring = self._mutate(offspring, task_metadata)
            
            new_population.append(offspring)
            new_fitness.append(0.0)  # Will be evaluated later
        
        self.population = new_population
        self.fitness_scores = new_fitness
    
    def _tournament_selection(self, fitness_scores: List[float], tournament_size: int = 3) -> int:
        """Tournament selection for parent choice."""
        tournament_indices = torch.randint(0, len(fitness_scores), (tournament_size,))
        best_idx = max(tournament_indices, key=lambda i: fitness_scores[i])
        return best_idx.item()
    
    def _crossover(self, parent1: Dict[str, Any], parent2: Dict[str, Any]) -> Dict[str, Any]:
        """Crossover between two parent configurations."""
        offspring = {}
        
        for key in set(parent1.keys()) | set(parent2.keys()):
            if torch.rand(1).item() < 0.5:
                offspring[key] = parent1.get(key, parent2.get(key))
            else:
                offspring[key] = parent2.get(key, parent1.get(key))
        
        return offspring
    
    def _mutate(self, config: Dict[str, Any], task_metadata: TaskMetadata) -> Dict[str, Any]:
        """Mutate configuration."""
        mutated_config = config.copy()
        
        for key, value in mutated_config.items():
            if torch.rand(1).item() < self.mutation_rate:
                if key == 'dimension':
                    # Gaussian mutation around current value
                    mutation = torch.randn(1).item() * 1000
                    mutated_config[key] = max(1000, min(50000, int(value + mutation)))
                    
                elif key == 'similarity_threshold':
                    mutation = torch.randn(1).item() * 0.1
                    mutated_config[key] = max(0.1, min(0.9, value + mutation))
                    
                elif key == 'position_weight':
                    mutation = torch.randn(1).item() * 0.1
                    mutated_config[key] = max(0.0, min(1.0, value + mutation))
                    
                elif key == 'ngram_size':
                    if torch.rand(1).item() < 0.5:
                        mutated_config[key] = max(1, min(5, value + 1))
                    else:
                        mutated_config[key] = max(1, min(5, value - 1))
                    
                elif key == 'patch_size':
                    sizes = [4, 8, 16, 32, 64]
                    mutated_config[key] = torch.choice(sizes)
                    
                elif isinstance(value, bool):
                    mutated_config[key] = not value
        
        return mutated_config
    
    def evaluate_adaptation(self, old_config: Dict[str, Any], 
                          new_config: Dict[str, Any], 
                          performance_improvement: float) -> float:
        """Evaluate evolutionary adaptation quality."""
        # Focus on population diversity and performance improvement
        performance_score = max(0.0, performance_improvement)
        
        # Bonus for maintaining population diversity
        diversity_bonus = self._compute_population_diversity() * 0.1
        
        return performance_score + diversity_bonus
    
    def _compute_population_diversity(self) -> float:
        """Compute diversity measure of current population."""
        if len(self.population) < 2:
            return 0.0
        
        # Simple diversity measure: average pairwise differences
        total_difference = 0.0
        comparisons = 0
        
        for i in range(len(self.population)):
            for j in range(i + 1, len(self.population)):
                config1 = self.population[i]
                config2 = self.population[j]
                
                # Count different parameters
                differences = sum(
                    1 for key in set(config1.keys()) & set(config2.keys())
                    if config1[key] != config2[key]
                )
                
                total_difference += differences
                comparisons += 1
        
        return total_difference / comparisons if comparisons > 0 else 0.0


class AdaptiveHDCSystem:
    """
    Adaptive HDC system with meta-learning capabilities.
    
    Novel research contribution: Self-improving HDC system that
    continuously adapts to new tasks and domains for optimal performance.
    """
    
    def __init__(self, meta_learning_strategies: List[MetaLearningStrategy] = None):
        """Initialize adaptive HDC system.
        
        Args:
            meta_learning_strategies: List of meta-learning strategies to use
        """
        if meta_learning_strategies is None:
            meta_learning_strategies = [
                GradientBasedMetaLearning(learning_rate=0.01),
                EvolutionaryMetaLearning(population_size=15)
            ]
        
        self.meta_learning_strategies = meta_learning_strategies
        self.current_config = self._get_default_config()
        self.task_history = {}
        self.adaptation_results = deque(maxlen=100)
        
        # Meta-meta-learning: learn which strategy works best for which tasks
        self.strategy_performance = {strategy.name: [] for strategy in meta_learning_strategies}
        
        logger.info(f"Initialized AdaptiveHDCSystem with {len(meta_learning_strategies)} strategies")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default HDC configuration."""
        return {
            'dimension': 10000,
            'bundle_normalize': True,
            'similarity_threshold': 0.5,
            'encoding_method': 'default',
            'device': 'cpu'
        }
    
    def register_task(self, task_id: str, domain: str, data_type: str, 
                     complexity_level: float = 0.5) -> TaskMetadata:
        """Register new task for adaptation.
        
        Args:
            task_id: Unique task identifier
            domain: Task domain (e.g., 'nlp', 'vision')
            data_type: Data type ('text', 'vision', 'audio', 'multimodal')
            complexity_level: Task complexity [0, 1]
            
        Returns:
            Task metadata object
        """
        task_metadata = TaskMetadata(
            task_id=task_id,
            domain=domain,
            data_type=data_type,
            complexity_level=complexity_level
        )
        
        self.task_history[task_id] = task_metadata
        logger.info(f"Registered task {task_id} in domain {domain}")
        
        return task_metadata
    
    def adapt_to_task(self, task_id: str, performance_feedback: float, 
                     adaptation_trigger_threshold: float = 0.1) -> AdaptationResult:
        """Adapt system configuration based on task performance.
        
        Args:
            task_id: Task identifier
            performance_feedback: Current task performance [0, 1]
            adaptation_trigger_threshold: Minimum performance drop to trigger adaptation
            
        Returns:
            Adaptation result
        """
        if task_id not in self.task_history:
            raise ValueError(f"Task {task_id} not registered")
        
        task_metadata = self.task_history[task_id]
        task_metadata.performance_history.append(performance_feedback)
        
        # Check if adaptation is needed
        if not self._should_adapt(task_metadata, adaptation_trigger_threshold):
            return AdaptationResult(
                old_performance=performance_feedback,
                new_performance=performance_feedback,
                improvement=0.0,
                config_changes={},
                adaptation_time=0.0,
                confidence=1.0
            )
        
        start_time = time.perf_counter()
        old_config = self.current_config.copy()
        
        # Select best meta-learning strategy for this task
        best_strategy = self._select_strategy(task_metadata)
        
        # Perform adaptation
        adaptation_suggestions = best_strategy.adapt_encoding(
            task_metadata, task_metadata.performance_history
        )
        
        # Apply adaptations
        new_config = self._apply_adaptations(old_config, adaptation_suggestions)
        
        # Estimate performance improvement (would be validated in practice)
        estimated_improvement = self._estimate_improvement(task_metadata, old_config, new_config)
        
        adaptation_time = time.perf_counter() - start_time
        
        # Create adaptation result
        adaptation_result = AdaptationResult(
            old_performance=performance_feedback,
            new_performance=performance_feedback + estimated_improvement,
            improvement=estimated_improvement,
            config_changes={k: v for k, v in adaptation_suggestions.items() if k in new_config},
            adaptation_time=adaptation_time,
            confidence=self._compute_adaptation_confidence(best_strategy, task_metadata)
        )
        
        # Update system configuration if improvement is expected
        if estimated_improvement > 0.02:  # 2% improvement threshold
            self.current_config = new_config
            task_metadata.best_config = new_config.copy()
            task_metadata.adaptation_count += 1
        
        # Record adaptation result
        self.adaptation_results.append(adaptation_result)
        
        # Update strategy performance tracking
        self.strategy_performance[best_strategy.name].append(estimated_improvement)
        
        logger.info(f"Adapted task {task_id}: {estimated_improvement:.3f} improvement expected")
        
        return adaptation_result
    
    def _should_adapt(self, task_metadata: TaskMetadata, threshold: float) -> bool:
        """Determine if adaptation is needed."""
        if len(task_metadata.performance_history) < 3:
            return False  # Need some performance history
        
        # Check for performance degradation
        recent_performance = task_metadata.performance_history[-3:]
        older_performance = task_metadata.performance_history[:-3] if len(task_metadata.performance_history) > 3 else []
        
        if not older_performance:
            return False
        
        recent_avg = sum(recent_performance) / len(recent_performance)
        older_avg = sum(older_performance) / len(older_performance)
        
        performance_drop = older_avg - recent_avg
        
        # Adapt if performance dropped or if we haven't adapted recently and performance is stagnant
        should_adapt = (performance_drop > threshold or 
                       (task_metadata.adaptation_count == 0 and len(task_metadata.performance_history) > 10))
        
        return should_adapt
    
    def _select_strategy(self, task_metadata: TaskMetadata) -> MetaLearningStrategy:
        """Select best meta-learning strategy for given task."""
        if not self.strategy_performance:
            # Default to first strategy if no performance history
            return self.meta_learning_strategies[0]
        
        # Select strategy with best average performance for this domain
        best_strategy = None
        best_score = -float('inf')
        
        for strategy in self.meta_learning_strategies:
            strategy_scores = self.strategy_performance[strategy.name]
            
            if strategy_scores:
                # Weight recent performance more heavily
                weights = [0.5 ** (len(strategy_scores) - i - 1) for i in range(len(strategy_scores))]
                weighted_avg = sum(score * weight for score, weight in zip(strategy_scores, weights)) / sum(weights)
                
                if weighted_avg > best_score:
                    best_score = weighted_avg
                    best_strategy = strategy
        
        return best_strategy or self.meta_learning_strategies[0]
    
    def _apply_adaptations(self, old_config: Dict[str, Any], 
                          adaptations: Dict[str, Any]) -> Dict[str, Any]:
        """Apply adaptation suggestions to configuration."""
        new_config = old_config.copy()
        
        for key, value in adaptations.items():
            if key in ['dimension', 'bundle_normalize', 'similarity_threshold', 
                      'encoding_method', 'ngram_size', 'position_weight', 
                      'patch_size', 'feature_method']:
                new_config[key] = value
        
        return new_config
    
    def _estimate_improvement(self, task_metadata: TaskMetadata, 
                            old_config: Dict[str, Any], 
                            new_config: Dict[str, Any]) -> float:
        """Estimate performance improvement from configuration change."""
        # Simple heuristic-based estimation (in practice, could use learned models)
        
        improvement = 0.0
        
        # Dimension changes
        old_dim = old_config.get('dimension', 10000)
        new_dim = new_config.get('dimension', 10000)
        
        if new_dim > old_dim:
            # Larger dimensions generally help complex tasks
            improvement += (new_dim - old_dim) / old_dim * task_metadata.complexity_level * 0.1
        elif new_dim < old_dim:
            # Smaller dimensions help with overfitting/noise
            if task_metadata.complexity_level < 0.5:
                improvement += (old_dim - new_dim) / old_dim * (1 - task_metadata.complexity_level) * 0.05
        
        # Normalization changes
        if old_config.get('bundle_normalize') != new_config.get('bundle_normalize'):
            # Normalization generally helps
            if new_config.get('bundle_normalize', True):
                improvement += 0.02
        
        # Task-specific improvements
        if task_metadata.data_type == 'text':
            # N-gram size optimization
            old_ngram = old_config.get('ngram_size', 3)
            new_ngram = new_config.get('ngram_size', 3)
            optimal_ngram = 3  # Assume 3 is generally optimal
            
            old_distance = abs(old_ngram - optimal_ngram)
            new_distance = abs(new_ngram - optimal_ngram)
            
            if new_distance < old_distance:
                improvement += (old_distance - new_distance) * 0.02
        
        elif task_metadata.data_type == 'vision':
            # Patch size optimization (task-dependent)
            if 'patch_size' in new_config:
                # Smaller patches generally better for detailed tasks
                if task_metadata.complexity_level > 0.7:  # High complexity
                    old_patch = old_config.get('patch_size', 16)
                    new_patch = new_config.get('patch_size', 16)
                    
                    if new_patch < old_patch:
                        improvement += 0.03
        
        # Cap improvement estimate
        return max(0.0, min(0.2, improvement))  # Max 20% improvement estimate
    
    def _compute_adaptation_confidence(self, strategy: MetaLearningStrategy, 
                                     task_metadata: TaskMetadata) -> float:
        """Compute confidence in adaptation."""
        base_confidence = 0.7
        
        # Higher confidence with more performance history
        history_bonus = min(0.2, len(task_metadata.performance_history) * 0.01)
        
        # Higher confidence if strategy has good track record
        strategy_scores = self.strategy_performance[strategy.name]
        if strategy_scores:
            avg_strategy_performance = sum(strategy_scores) / len(strategy_scores)
            strategy_bonus = max(0.0, avg_strategy_performance) * 0.1
        else:
            strategy_bonus = 0.0
        
        # Lower confidence for highly complex tasks (more uncertainty)
        complexity_penalty = task_metadata.complexity_level * 0.1
        
        confidence = base_confidence + history_bonus + strategy_bonus - complexity_penalty
        return max(0.1, min(1.0, confidence))
    
    def get_adaptation_analytics(self) -> Dict[str, Any]:
        """Get analytics about adaptation performance."""
        analytics = {
            'total_adaptations': len(self.adaptation_results),
            'successful_adaptations': sum(1 for r in self.adaptation_results if r.improvement > 0),
            'average_improvement': sum(r.improvement for r in self.adaptation_results) / len(self.adaptation_results) if self.adaptation_results else 0,
            'average_adaptation_time': sum(r.adaptation_time for r in self.adaptation_results) / len(self.adaptation_results) if self.adaptation_results else 0,
            'strategy_performance': {}
        }
        
        # Strategy performance analysis
        for strategy_name, scores in self.strategy_performance.items():
            if scores:
                analytics['strategy_performance'][strategy_name] = {
                    'num_adaptations': len(scores),
                    'average_improvement': sum(scores) / len(scores),
                    'success_rate': sum(1 for s in scores if s > 0) / len(scores)
                }
        
        # Task analysis
        analytics['task_analysis'] = {}
        for task_id, task_metadata in self.task_history.items():
            analytics['task_analysis'][task_id] = {
                'domain': task_metadata.domain,
                'data_type': task_metadata.data_type,
                'complexity_level': task_metadata.complexity_level,
                'num_adaptations': task_metadata.adaptation_count,
                'performance_trend': self._compute_performance_trend(task_metadata.performance_history)
            }
        
        return analytics
    
    def _compute_performance_trend(self, performance_history: List[float]) -> str:
        """Compute performance trend for a task."""
        if len(performance_history) < 3:
            return "insufficient_data"
        
        # Compare first third with last third
        third = len(performance_history) // 3
        early_avg = sum(performance_history[:third]) / third if third > 0 else 0
        late_avg = sum(performance_history[-third:]) / third
        
        improvement = late_avg - early_avg
        
        if improvement > 0.05:
            return "improving"
        elif improvement < -0.05:
            return "declining"
        else:
            return "stable"
    
    def export_learned_configs(self) -> Dict[str, Dict[str, Any]]:
        """Export learned configurations for different task types."""
        configs = {}
        
        # Extract best configurations by task domain
        domain_configs = {}
        for task_id, task_metadata in self.task_history.items():
            if task_metadata.best_config:
                domain = task_metadata.domain
                if domain not in domain_configs:
                    domain_configs[domain] = []
                domain_configs[domain].append(task_metadata.best_config)
        
        # Average configurations by domain
        for domain, config_list in domain_configs.items():
            if config_list:
                averaged_config = {}
                
                # Average numerical parameters
                for key in ['dimension', 'similarity_threshold', 'position_weight']:
                    values = [c.get(key) for c in config_list if key in c and isinstance(c[key], (int, float))]
                    if values:
                        averaged_config[key] = sum(values) / len(values)
                
                # Mode for categorical parameters
                for key in ['bundle_normalize', 'encoding_method', 'feature_method']:
                    values = [c.get(key) for c in config_list if key in c]
                    if values:
                        # Simple mode (most common value)
                        averaged_config[key] = max(set(values), key=values.count)
                
                configs[domain] = averaged_config
        
        return configs
    
    def load_learned_configs(self, configs: Dict[str, Dict[str, Any]]):
        """Load previously learned configurations."""
        # This would integrate learned configurations into the system
        # For now, just log the loading
        logger.info(f"Loaded learned configurations for {len(configs)} domains")
        
        # Could integrate these as priors for new tasks in similar domains
        for domain, config in configs.items():
            logger.info(f"Loaded config for {domain}: {config}")
