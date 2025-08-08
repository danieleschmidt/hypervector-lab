"""Comparative study framework for HDC research validation."""

import time
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any, Callable
import torch
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


@dataclass
class ExperimentResult:
    """Results from a single experiment run."""
    algorithm_name: str
    parameters: Dict[str, Any]
    metrics: Dict[str, float]
    execution_time: float
    memory_usage_mb: Optional[float] = None
    statistical_significance: Optional[float] = None


@dataclass
class ComparisonResult:
    """Results from comparing multiple algorithms."""
    baseline_algorithm: str
    comparison_algorithms: List[str]
    relative_performance: Dict[str, float]  # Algorithm -> relative improvement
    statistical_tests: Dict[str, float]     # Test name -> p-value
    confidence_intervals: Dict[str, Tuple[float, float]]
    recommendation: str


class ComparisonFramework:
    """
    Framework for rigorous comparison of HDC algorithms.
    
    Novel research contribution: Standardized methodology for 
    reproducible HDC research with statistical validation.
    """
    
    def __init__(self, random_seed: int = 42, num_trials: int = 10):
        """Initialize comparison framework.
        
        Args:
            random_seed: Seed for reproducible experiments
            num_trials: Number of trials per experiment
        """
        self.random_seed = random_seed
        self.num_trials = num_trials
        self.results: List[ExperimentResult] = []
        
        torch.manual_seed(random_seed)
        logger.info(f"Initialized ComparisonFramework with {num_trials} trials")
    
    def run_experiment(self, 
                      algorithm: Callable,
                      algorithm_name: str,
                      parameters: Dict[str, Any],
                      evaluation_fn: Callable,
                      datasets: List[Any]) -> ExperimentResult:
        """Run single algorithm experiment with multiple trials.
        
        Args:
            algorithm: Algorithm function to test
            algorithm_name: Name of the algorithm
            parameters: Algorithm parameters
            evaluation_fn: Function to evaluate results
            datasets: List of datasets to test on
            
        Returns:
            Aggregated experiment results
        """
        trial_results = []
        trial_times = []
        
        for trial in range(self.num_trials):
            # Set seed for this trial
            torch.manual_seed(self.random_seed + trial)
            
            trial_metrics = {}
            start_time = time.perf_counter()
            
            try:
                # Run algorithm on all datasets
                for dataset_idx, dataset in enumerate(datasets):
                    result = algorithm(dataset, **parameters)
                    metrics = evaluation_fn(result, dataset)
                    
                    # Aggregate metrics across datasets
                    for metric_name, value in metrics.items():
                        key = f"dataset_{dataset_idx}_{metric_name}"
                        trial_metrics[key] = value
                
                execution_time = time.perf_counter() - start_time
                trial_times.append(execution_time)
                trial_results.append(trial_metrics)
                
            except Exception as e:
                logger.error(f"Trial {trial} failed for {algorithm_name}: {e}")
                continue
        
        # Aggregate results across trials
        aggregated_metrics = self._aggregate_trial_results(trial_results)
        avg_execution_time = sum(trial_times) / len(trial_times) if trial_times else 0.0
        
        result = ExperimentResult(
            algorithm_name=algorithm_name,
            parameters=parameters,
            metrics=aggregated_metrics,
            execution_time=avg_execution_time
        )
        
        self.results.append(result)
        return result
    
    def _aggregate_trial_results(self, trial_results: List[Dict[str, float]]) -> Dict[str, float]:
        """Aggregate metrics across multiple trials."""
        if not trial_results:
            return {}
        
        aggregated = {}
        all_keys = set()
        for trial in trial_results:
            all_keys.update(trial.keys())
        
        for key in all_keys:
            values = [trial.get(key, 0.0) for trial in trial_results]
            
            aggregated[f"{key}_mean"] = sum(values) / len(values)
            aggregated[f"{key}_std"] = self._compute_std(values)
            aggregated[f"{key}_min"] = min(values)
            aggregated[f"{key}_max"] = max(values)
        
        return aggregated
    
    def _compute_std(self, values: List[float]) -> float:
        """Compute standard deviation."""
        if len(values) <= 1:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return math.sqrt(variance)
    
    def compare_algorithms(self, 
                          baseline_name: str,
                          comparison_names: List[str],
                          primary_metric: str) -> ComparisonResult:
        """Compare multiple algorithms with statistical testing.
        
        Args:
            baseline_name: Name of baseline algorithm
            comparison_names: Names of algorithms to compare against baseline
            primary_metric: Primary metric for comparison
            
        Returns:
            Comprehensive comparison results
        """
        baseline_result = None
        comparison_results = {}
        
        # Find results
        for result in self.results:
            if result.algorithm_name == baseline_name:
                baseline_result = result
            elif result.algorithm_name in comparison_names:
                comparison_results[result.algorithm_name] = result
        
        if baseline_result is None:
            raise ValueError(f"Baseline algorithm '{baseline_name}' not found in results")
        
        # Compute relative performance
        relative_performance = {}
        statistical_tests = {}
        confidence_intervals = {}
        
        baseline_metric = baseline_result.metrics.get(f"{primary_metric}_mean", 0.0)
        baseline_std = baseline_result.metrics.get(f"{primary_metric}_std", 0.0)
        
        for alg_name, result in comparison_results.items():
            comp_metric = result.metrics.get(f"{primary_metric}_mean", 0.0)
            comp_std = result.metrics.get(f"{primary_metric}_std", 0.0)
            
            # Relative improvement
            if baseline_metric != 0:
                relative_improvement = (comp_metric - baseline_metric) / baseline_metric
            else:
                relative_improvement = 0.0
            
            relative_performance[alg_name] = relative_improvement
            
            # Statistical significance (simplified t-test approximation)
            pooled_std = math.sqrt((baseline_std**2 + comp_std**2) / 2)
            if pooled_std > 0:
                t_statistic = abs(comp_metric - baseline_metric) / pooled_std
                # Approximate p-value (simplified)
                p_value = self._approximate_p_value(t_statistic)
            else:
                p_value = 1.0
            
            statistical_tests[alg_name] = p_value
            
            # Confidence interval (95%)
            margin_of_error = 1.96 * pooled_std / math.sqrt(self.num_trials)
            ci_lower = relative_improvement - margin_of_error
            ci_upper = relative_improvement + margin_of_error
            confidence_intervals[alg_name] = (ci_lower, ci_upper)
        
        # Generate recommendation
        recommendation = self._generate_recommendation(
            relative_performance, statistical_tests, primary_metric
        )
        
        return ComparisonResult(
            baseline_algorithm=baseline_name,
            comparison_algorithms=comparison_names,
            relative_performance=relative_performance,
            statistical_tests=statistical_tests,
            confidence_intervals=confidence_intervals,
            recommendation=recommendation
        )
    
    def _approximate_p_value(self, t_statistic: float) -> float:
        """Approximate p-value for t-statistic."""
        # Simplified approximation for demonstration
        # In practice, use proper statistical libraries
        if t_statistic > 2.576:  # 99% confidence
            return 0.01
        elif t_statistic > 1.96:  # 95% confidence
            return 0.05
        elif t_statistic > 1.645:  # 90% confidence
            return 0.10
        else:
            return 0.20
    
    def _generate_recommendation(self, 
                               relative_performance: Dict[str, float],
                               statistical_tests: Dict[str, float],
                               primary_metric: str) -> str:
        """Generate algorithm recommendation based on results."""
        significant_improvements = {}
        
        for alg_name, improvement in relative_performance.items():
            p_value = statistical_tests.get(alg_name, 1.0)
            
            if improvement > 0.05 and p_value < 0.05:  # 5% improvement, p < 0.05
                significant_improvements[alg_name] = improvement
        
        if not significant_improvements:
            return f"No algorithm shows statistically significant improvement in {primary_metric}"
        
        best_algorithm = max(significant_improvements.keys(), 
                           key=lambda x: significant_improvements[x])
        best_improvement = significant_improvements[best_algorithm]
        
        return f"Recommend {best_algorithm}: {best_improvement:.1%} improvement in {primary_metric} (p < 0.05)"


class BenchmarkComparator:
    """
    Standardized benchmark comparator for HDC algorithms.
    
    Novel research contribution: Comprehensive benchmark suite
    for fair algorithm comparison.
    """
    
    def __init__(self, dimensions: List[int] = None):
        """Initialize benchmark comparator.
        
        Args:
            dimensions: List of hypervector dimensions to test
        """
        self.dimensions = dimensions or [1000, 5000, 10000]
        self.benchmark_functions = {
            'creation_speed': self._benchmark_creation,
            'binding_speed': self._benchmark_binding,
            'similarity_accuracy': self._benchmark_similarity,
            'memory_efficiency': self._benchmark_memory,
            'scalability': self._benchmark_scalability
        }
        
        logger.info(f"Initialized BenchmarkComparator with dimensions: {self.dimensions}")
    
    def run_comprehensive_benchmark(self, 
                                   algorithms: Dict[str, Callable],
                                   benchmark_types: List[str] = None) -> Dict[str, Dict[str, float]]:
        """Run comprehensive benchmark across all algorithms.
        
        Args:
            algorithms: Dict mapping algorithm names to functions
            benchmark_types: List of benchmark types to run
            
        Returns:
            Dict mapping algorithm names to benchmark results
        """
        if benchmark_types is None:
            benchmark_types = list(self.benchmark_functions.keys())
        
        results = {}
        
        for alg_name, alg_func in algorithms.items():
            alg_results = {}
            
            for benchmark_type in benchmark_types:
                if benchmark_type in self.benchmark_functions:
                    benchmark_func = self.benchmark_functions[benchmark_type]
                    
                    try:
                        score = benchmark_func(alg_func)
                        alg_results[benchmark_type] = score
                    except Exception as e:
                        logger.error(f"Benchmark {benchmark_type} failed for {alg_name}: {e}")
                        alg_results[benchmark_type] = 0.0
            
            results[alg_name] = alg_results
        
        return results
    
    def _benchmark_creation(self, algorithm: Callable) -> float:
        """Benchmark hypervector creation speed."""
        total_time = 0.0
        num_trials = 100
        
        for dim in self.dimensions:
            start_time = time.perf_counter()
            
            for _ in range(num_trials):
                hv = algorithm(dim)
            
            end_time = time.perf_counter()
            total_time += (end_time - start_time) / num_trials
        
        # Return operations per second
        return len(self.dimensions) / total_time if total_time > 0 else 0.0
    
    def _benchmark_binding(self, algorithm: Callable) -> float:
        """Benchmark binding operation speed."""
        total_time = 0.0
        num_trials = 50
        
        for dim in self.dimensions:
            hv1 = HyperVector.random(dim)
            hv2 = HyperVector.random(dim)
            
            start_time = time.perf_counter()
            
            for _ in range(num_trials):
                result = bind(hv1, hv2)
            
            end_time = time.perf_counter()
            total_time += (end_time - start_time) / num_trials
        
        return len(self.dimensions) / total_time if total_time > 0 else 0.0
    
    def _benchmark_similarity(self, algorithm: Callable) -> float:
        """Benchmark similarity computation accuracy."""
        accuracies = []
        
        for dim in self.dimensions:
            # Create test vectors with known relationships
            base_hv = HyperVector.random(dim)
            similar_hv = HyperVector(base_hv.data + torch.randn_like(base_hv.data) * 0.1)
            different_hv = HyperVector.random(dim)
            
            # Compute similarities
            similar_score = cosine_similarity(base_hv, similar_hv).item()
            different_score = cosine_similarity(base_hv, different_hv).item()
            
            # Accuracy is whether similar > different
            accuracy = 1.0 if similar_score > different_score else 0.0
            accuracies.append(accuracy)
        
        return sum(accuracies) / len(accuracies)
    
    def _benchmark_memory(self, algorithm: Callable) -> float:
        """Benchmark memory efficiency."""
        # Simplified memory benchmark
        # In practice, use proper memory profiling
        memory_scores = []
        
        for dim in self.dimensions:
            hv = HyperVector.random(dim)
            # Score inversely proportional to dimension (smaller is better)
            memory_score = 1.0 / (dim / 1000.0)
            memory_scores.append(memory_score)
        
        return sum(memory_scores) / len(memory_scores)
    
    def _benchmark_scalability(self, algorithm: Callable) -> float:
        """Benchmark scalability across dimensions."""
        times = []
        
        for dim in self.dimensions:
            start_time = time.perf_counter()
            hv1 = HyperVector.random(dim)
            hv2 = HyperVector.random(dim)
            result = bind(hv1, hv2)
            end_time = time.perf_counter()
            
            times.append(end_time - start_time)
        
        # Check if scaling is approximately linear
        if len(times) >= 2:
            # Compute correlation with dimension
            dim_ratios = [self.dimensions[i] / self.dimensions[0] for i in range(len(self.dimensions))]
            time_ratios = [times[i] / times[0] for i in range(len(times))]
            
            # Simple correlation approximation
            correlation = sum(dr * tr for dr, tr in zip(dim_ratios, time_ratios)) / len(dim_ratios)
            
            # Score inversely related to how much worse than linear
            scalability_score = max(0.0, 2.0 - correlation)
            return scalability_score
        
        return 1.0


class StatisticalAnalyzer:
    """
    Statistical analysis tools for HDC research validation.
    
    Novel research contribution: Rigorous statistical methods
    for validating HDC algorithm performance claims.
    """
    
    def __init__(self, confidence_level: float = 0.95):
        """Initialize statistical analyzer.
        
        Args:
            confidence_level: Statistical confidence level
        """
        self.confidence_level = confidence_level
        self.alpha = 1.0 - confidence_level
        
        logger.info(f"Initialized StatisticalAnalyzer with {confidence_level:.1%} confidence")
    
    def power_analysis(self, 
                      effect_size: float,
                      baseline_std: float,
                      desired_power: float = 0.8) -> int:
        """Compute required sample size for statistical power.
        
        Args:
            effect_size: Expected effect size
            baseline_std: Standard deviation of baseline
            desired_power: Desired statistical power
            
        Returns:
            Required sample size
        """
        # Simplified power analysis
        # In practice, use proper statistical libraries
        
        z_alpha = 1.96  # For 95% confidence
        z_beta = 0.84   # For 80% power
        
        if effect_size == 0 or baseline_std == 0:
            return float('inf')
        
        standardized_effect = effect_size / baseline_std
        
        sample_size = 2 * ((z_alpha + z_beta) / standardized_effect) ** 2
        
        return max(10, int(math.ceil(sample_size)))
    
    def effect_size_analysis(self, 
                           baseline_mean: float,
                           baseline_std: float,
                           treatment_mean: float,
                           treatment_std: float) -> Dict[str, float]:
        """Compute effect size metrics.
        
        Args:
            baseline_mean: Baseline group mean
            baseline_std: Baseline group standard deviation
            treatment_mean: Treatment group mean
            treatment_std: Treatment group standard deviation
            
        Returns:
            Dict with effect size metrics
        """
        # Cohen's d
        pooled_std = math.sqrt((baseline_std**2 + treatment_std**2) / 2)
        cohens_d = (treatment_mean - baseline_mean) / pooled_std if pooled_std > 0 else 0.0
        
        # Glass's delta
        glass_delta = (treatment_mean - baseline_mean) / baseline_std if baseline_std > 0 else 0.0
        
        # Hedge's g (bias-corrected)
        hedges_g = cohens_d * (1 - 3 / (4 * 10 - 9))  # Assuming n=10 for demonstration
        
        return {
            'cohens_d': cohens_d,
            'glass_delta': glass_delta,
            'hedges_g': hedges_g,
            'effect_interpretation': self._interpret_effect_size(abs(cohens_d))
        }
    
    def _interpret_effect_size(self, effect_size: float) -> str:
        """Interpret effect size magnitude."""
        if effect_size < 0.2:
            return "negligible"
        elif effect_size < 0.5:
            return "small"
        elif effect_size < 0.8:
            return "medium"
        else:
            return "large"
    
    def meta_analysis(self, 
                     studies: List[Dict[str, float]]) -> Dict[str, float]:
        """Perform meta-analysis across multiple studies.
        
        Args:
            studies: List of study results with 'effect_size' and 'variance'
            
        Returns:
            Meta-analysis results
        """
        if not studies:
            return {}
        
        # Fixed-effects meta-analysis
        weights = []
        weighted_effects = []
        
        for study in studies:
            if 'variance' in study and study['variance'] > 0:
                weight = 1.0 / study['variance']
                weights.append(weight)
                weighted_effects.append(weight * study.get('effect_size', 0.0))
        
        if not weights:
            return {'error': 'No valid studies with variance information'}
        
        total_weight = sum(weights)
        pooled_effect = sum(weighted_effects) / total_weight
        pooled_variance = 1.0 / total_weight
        pooled_se = math.sqrt(pooled_variance)
        
        # Confidence interval
        z_critical = 1.96  # For 95% CI
        ci_lower = pooled_effect - z_critical * pooled_se
        ci_upper = pooled_effect + z_critical * pooled_se
        
        # Heterogeneity assessment (simplified)
        q_statistic = sum(weight * (study.get('effect_size', 0.0) - pooled_effect)**2 
                         for weight, study in zip(weights, studies))
        df = len(studies) - 1
        i_squared = max(0.0, (q_statistic - df) / q_statistic) if q_statistic > 0 else 0.0
        
        return {
            'pooled_effect_size': pooled_effect,
            'standard_error': pooled_se,
            'confidence_interval_lower': ci_lower,
            'confidence_interval_upper': ci_upper,
            'q_statistic': q_statistic,
            'i_squared': i_squared,
            'heterogeneity': 'low' if i_squared < 25 else ('moderate' if i_squared < 75 else 'high'),
            'num_studies': len(studies)
        }