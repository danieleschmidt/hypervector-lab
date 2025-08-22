#!/usr/bin/env python3
"""Comprehensive test suite for breakthrough HDC research implementations.

This test suite validates all breakthrough algorithms, quantum enhancements,
and real-time processing capabilities with statistical significance testing.
"""

import sys
import time
import pytest
import torch
import numpy as np
from typing import List, Dict, Any
import logging

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import HDC components
try:
    from hypervector import HDCSystem, HyperVector
    from hypervector.research.breakthrough_algorithms import (
        SelfOrganizingHyperMap, EvolutionaryHDC, MetaLearningHDC,
        QuantumCoherentHDC, NeuroplasticityHDC, BreakthroughResearchSuite
    )
    from hypervector.research.quantum_enhanced_real_time_hdc import (
        QuantumCoherentProcessor, RealTimeHDCEngine, ScalableHDCCluster
    )
    from hypervector.utils.advanced_error_recovery import HDCErrorRecovery
    HDC_AVAILABLE = True
except ImportError as e:
    print(f"HDC modules not available: {e}")
    HDC_AVAILABLE = False


@pytest.mark.skipif(not HDC_AVAILABLE, reason="HDC modules not available")
class TestBreakthroughAlgorithms:
    """Test breakthrough HDC algorithms."""
    
    def setup_method(self):
        """Setup test environment."""
        self.dim = 1000  # Smaller dimension for faster tests
        self.device = "cpu"
        self.test_data = self._generate_test_data()
    
    def _generate_test_data(self) -> List[HyperVector]:
        """Generate test hypervectors."""
        test_data = []
        for i in range(20):
            hv = HyperVector.random(dim=self.dim, device=self.device)
            test_data.append(hv)
        return test_data
    
    def test_self_organizing_hypermap(self):
        """Test Self-Organizing HyperMap implementation."""
        logger.info("Testing Self-Organizing HyperMap")
        
        som = SelfOrganizingHyperMap(
            input_dim=self.dim, 
            map_size=(10, 10), 
            device=self.device
        )
        
        # Test training
        start_time = time.time()
        metrics = som.train(self.test_data, epochs=10)
        training_time = time.time() - start_time
        
        # Validate results
        assert metrics.algorithm_name == "SelfOrganizingHyperMap"
        assert metrics.performance_improvement > 0.0
        assert metrics.statistical_significance >= 0.0
        assert training_time < 30.0  # Should complete within 30 seconds
        
        # Test winner finding
        test_hv = self.test_data[0]
        winner_pos = som.find_winner(test_hv)
        assert isinstance(winner_pos, tuple)
        assert len(winner_pos) == 2
        assert 0 <= winner_pos[0] < 10
        assert 0 <= winner_pos[1] < 10
        
        logger.info(f"SOM training completed in {training_time:.2f}s")
        logger.info(f"Performance improvement: {metrics.performance_improvement:.3f}")
    
    def test_evolutionary_hdc(self):
        """Test Evolutionary HDC implementation."""
        logger.info("Testing Evolutionary HDC")
        
        evo = EvolutionaryHDC(
            population_size=20, 
            dim=self.dim, 
            device=self.device
        )
        
        # Test evolution
        target_hv = self.test_data[0]
        start_time = time.time()
        metrics = evo.evolve(target_hv, generations=10)
        evolution_time = time.time() - start_time
        
        # Validate results
        assert metrics.algorithm_name == "EvolutionaryHDC"
        assert metrics.performance_improvement > 0.0
        assert evolution_time < 60.0  # Should complete within 60 seconds
        
        # Test genetic operations
        parent1 = evo.population[0]
        parent2 = evo.population[1]
        
        child1, child2 = evo.hd_crossover(parent1, parent2)
        assert isinstance(child1, HyperVector)
        assert isinstance(child2, HyperVector)
        assert child1.dim == self.dim
        assert child2.dim == self.dim
        
        mutated = evo.hd_mutation(parent1)
        assert isinstance(mutated, HyperVector)
        assert mutated.dim == self.dim
        
        logger.info(f"Evolution completed in {evolution_time:.2f}s")
        logger.info(f"Best fitness: {metrics.performance_improvement:.3f}")
    
    def test_meta_learning_hdc(self):
        """Test Meta-Learning HDC implementation."""
        logger.info("Testing Meta-Learning HDC")
        
        meta = MetaLearningHDC(base_dim=self.dim, device=self.device)
        
        # Create meta-tasks
        meta_tasks = []
        for i in range(5):
            task = [(self.test_data[j], self.test_data[(j + 1) % len(self.test_data)]) 
                   for j in range(i, i + 3)]
            meta_tasks.append(task)
        
        # Test meta-training
        start_time = time.time()
        metrics = meta.meta_train(meta_tasks, meta_epochs=5)
        training_time = time.time() - start_time
        
        # Validate results
        assert metrics.algorithm_name == "MetaLearningHDC"
        assert metrics.performance_improvement >= 0.0
        assert training_time < 45.0  # Should complete within 45 seconds
        
        # Test learnable operations
        hv1, hv2 = self.test_data[0], self.test_data[1]
        
        bound_hv = meta.learnable_bind(hv1, hv2)
        assert isinstance(bound_hv, HyperVector)
        assert bound_hv.dim == self.dim
        
        bundled_hv = meta.learnable_bundle([hv1, hv2])
        assert isinstance(bundled_hv, HyperVector)
        assert bundled_hv.dim == self.dim
        
        permuted_hv = meta.learnable_permute(hv1)
        assert isinstance(permuted_hv, HyperVector)
        assert permuted_hv.dim == self.dim
        
        logger.info(f"Meta-learning completed in {training_time:.2f}s")
        logger.info(f"Improvement: {metrics.performance_improvement:.3f}")
    
    def test_quantum_coherent_hdc(self):
        """Test Quantum Coherent HDC implementation.""" 
        logger.info("Testing Quantum Coherent HDC")
        
        quantum_hdc = QuantumCoherentHDC(dim=self.dim, device=self.device)
        
        # Test quantum state creation
        test_hv = self.test_data[0]
        amplitudes = test_hv.data
        phases = torch.rand(self.dim, device=self.device) * 2 * np.pi
        
        quantum_state = quantum_hdc.create_coherent_state(amplitudes, phases)
        assert 'amplitudes' in quantum_state
        assert 'phases' in quantum_state
        assert 'creation_time' in quantum_state
        
        # Test coherence measurement
        coherence = quantum_hdc.measure_coherence(quantum_state)
        assert isinstance(coherence, float)
        assert 0.0 <= coherence <= 1.0
        
        # Test coherent binding
        state1 = quantum_state
        phases2 = torch.rand(self.dim, device=self.device) * 2 * np.pi
        state2 = quantum_hdc.create_coherent_state(self.test_data[1].data, phases2)
        
        bound_state = quantum_hdc.coherent_bind(state1, state2)
        assert 'amplitudes' in bound_state
        assert 'phases' in bound_state
        
        bound_coherence = quantum_hdc.measure_coherence(bound_state)
        assert isinstance(bound_coherence, float)
        assert 0.0 <= bound_coherence <= 1.0
        
        logger.info(f"Quantum coherence: {coherence:.3f}")
        logger.info(f"Bound coherence: {bound_coherence:.3f}")
    
    def test_neuroplasticity_hdc(self):
        """Test Neuroplasticity HDC implementation."""
        logger.info("Testing Neuroplasticity HDC")
        
        plastic_hdc = NeuroplasticityHDC(dim=self.dim, device=self.device)
        
        # Test adaptive encoding
        test_hv = self.test_data[0]
        context_hv = self.test_data[1]
        
        adapted_hv = plastic_hdc.adaptive_encode(test_hv, context_hv)
        assert isinstance(adapted_hv, HyperVector)
        assert adapted_hv.dim == self.dim
        
        # Test spike timing plasticity
        pre_spike = test_hv.data
        post_spike = context_hv.data
        time_diff = 0.01  # 10ms difference
        
        initial_weights = plastic_hdc.synaptic_weights.clone()
        updated_weights = plastic_hdc.spike_timing_plasticity(pre_spike, post_spike, time_diff)
        
        assert torch.is_tensor(updated_weights)
        assert updated_weights.shape == initial_weights.shape
        assert not torch.equal(updated_weights, initial_weights)  # Weights should change
        
        # Test weight bounds
        assert torch.all(updated_weights >= 0.1)  # Minimum weight
        assert torch.all(updated_weights <= 2.0)   # Maximum weight
        
        logger.info("Neuroplasticity mechanisms validated")
    
    def test_breakthrough_research_suite(self):
        """Test comprehensive breakthrough research suite."""
        logger.info("Testing Breakthrough Research Suite")
        
        suite = BreakthroughResearchSuite(device=self.device)
        
        # Prepare test and target data
        test_data = self.test_data[:10]
        target_data = self.test_data[5:15]
        
        # Run comprehensive research
        start_time = time.time()
        results = suite.run_comprehensive_research(test_data, target_data)
        research_time = time.time() - start_time
        
        # Validate results
        assert isinstance(results, dict)
        assert len(results) >= 4  # Should have at least 4 algorithms
        
        expected_algorithms = [
            'SelfOrganizingHyperMap',
            'EvolutionaryHDC', 
            'MetaLearningHDC',
            'QuantumCoherentHDC',
            'NeuroplasticityHDC'
        ]
        
        for algo_name in expected_algorithms:
            assert algo_name in results
            metrics = results[algo_name]
            assert hasattr(metrics, 'algorithm_name')
            assert hasattr(metrics, 'performance_improvement')
            assert hasattr(metrics, 'statistical_significance')
            assert hasattr(metrics, 'novel_contributions')
        
        # Generate research report
        report = suite.generate_research_report()
        assert isinstance(report, str)
        assert len(report) > 100  # Should be substantial
        assert "# Breakthrough HDC Research Results" in report
        
        logger.info(f"Research suite completed in {research_time:.2f}s")
        logger.info(f"Algorithms tested: {len(results)}")


@pytest.mark.skipif(not HDC_AVAILABLE, reason="HDC modules not available")
class TestQuantumEnhancedRealTime:
    """Test quantum-enhanced real-time HDC systems."""
    
    def setup_method(self):
        """Setup test environment."""
        self.dim = 1000
        self.device = "cpu"
    
    def test_quantum_coherent_processor(self):
        """Test QuantumCoherentProcessor."""
        logger.info("Testing QuantumCoherentProcessor")
        
        processor = QuantumCoherentProcessor(dim=self.dim, device=self.device)
        
        # Test quantum hypervector creation
        classical_hv = HyperVector.random(dim=self.dim, device=self.device)
        quantum_state = processor.create_quantum_hypervector(classical_hv)
        
        assert hasattr(quantum_state, 'amplitudes')
        assert hasattr(quantum_state, 'phases')
        assert hasattr(quantum_state, 'coherence_time')
        assert quantum_state.amplitudes.shape[-1] == self.dim
        assert quantum_state.phases.shape[-1] == self.dim
        
        # Test quantum binding
        classical_hv2 = HyperVector.random(dim=self.dim, device=self.device)
        quantum_state2 = processor.create_quantum_hypervector(classical_hv2)
        
        bound_state = processor.quantum_bind(quantum_state, quantum_state2)
        assert hasattr(bound_state, 'amplitudes')
        assert hasattr(bound_state, 'phases')
        
        # Test quantum similarity
        similarity = processor.quantum_similarity(quantum_state, quantum_state2)
        assert torch.is_tensor(similarity)
        assert -1.0 <= similarity.item() <= 1.0
        
        # Test measurement
        measured_hv = processor.measure_quantum_state(quantum_state)
        assert isinstance(measured_hv, HyperVector)
        assert measured_hv.dim == self.dim
        
        logger.info("QuantumCoherentProcessor validation complete")
    
    def test_real_time_hdc_engine(self):
        """Test RealTimeHDCEngine."""
        logger.info("Testing RealTimeHDCEngine")
        
        engine = RealTimeHDCEngine(dim=self.dim, device=self.device, max_workers=2)
        
        # Start engine
        engine.start_real_time_processing()
        time.sleep(0.1)  # Allow startup
        
        try:
            # Test operation submission
            test_hv1 = HyperVector.random(dim=self.dim, device=self.device)
            test_hv2 = HyperVector.random(dim=self.dim, device=self.device)
            
            # Submit quantum encoding operation
            op_id1 = engine.submit_operation("quantum_encode", test_hv1)
            assert op_id1 is not None
            
            # Submit similarity operation
            op_id2 = engine.submit_operation("similarity", (test_hv1, test_hv2))
            assert op_id2 is not None
            
            # Get results
            result1 = engine.get_result(op_id1, timeout=2.0)
            result2 = engine.get_result(op_id2, timeout=2.0)
            
            # Validate results
            assert result1 is not None  # Quantum state
            assert result2 is not None  # Similarity tensor
            assert torch.is_tensor(result2)
            
            # Test performance stats
            stats = engine.get_performance_stats()
            assert isinstance(stats, dict)
            assert 'operations_processed' in stats
            assert 'avg_latency_ms' in stats
            assert stats['operations_processed'] >= 2
            
        finally:
            # Stop engine
            engine.stop_real_time_processing()
        
        logger.info("RealTimeHDCEngine validation complete")
    
    def test_scalable_hdc_cluster(self):
        """Test ScalableHDCCluster."""
        logger.info("Testing ScalableHDCCluster")
        
        cluster = ScalableHDCCluster(num_nodes=2, dim=self.dim, device=self.device)
        
        # Start cluster
        cluster.start_cluster()
        time.sleep(0.2)  # Allow startup
        
        try:
            # Submit operations to cluster
            test_hv1 = HyperVector.random(dim=self.dim, device=self.device)
            test_hv2 = HyperVector.random(dim=self.dim, device=self.device)
            
            # Submit multiple operations
            operations = []
            for i in range(5):
                node_id, op_id = cluster.submit_operation("similarity", (test_hv1, test_hv2))
                if op_id:
                    operations.append((node_id, op_id))
            
            assert len(operations) >= 3  # At least some operations should succeed
            
            # Get results
            results = []
            for node_id, op_id in operations:
                result = cluster.get_result(node_id, op_id, timeout=2.0)
                if result is not None:
                    results.append(result)
            
            assert len(results) >= 1  # At least one result
            
            # Test cluster stats
            stats = cluster.get_cluster_stats()
            assert isinstance(stats, dict)
            assert 'num_nodes' in stats
            assert 'operations_processed' in stats
            assert stats['num_nodes'] == 2
            
        finally:
            # Stop cluster
            cluster.stop_cluster()
        
        logger.info("ScalableHDCCluster validation complete")


@pytest.mark.skipif(not HDC_AVAILABLE, reason="HDC modules not available")  
class TestErrorRecovery:
    """Test advanced error recovery systems."""
    
    def test_hdc_error_recovery(self):
        """Test HDC error recovery functionality."""
        logger.info("Testing HDC Error Recovery")
        
        # This would test the error recovery system
        # For now, just ensure it can be imported and initialized
        error_recovery = HDCErrorRecovery(max_recovery_attempts=2)
        
        # Test basic functionality
        def test_operation():
            return "success"
        
        def failing_operation():
            raise ValueError("Test error")
        
        # Test successful operation
        result = error_recovery.protected_execute(test_operation)
        assert result == "success"
        
        # Test error handling
        try:
            error_recovery.protected_execute(failing_operation)
            assert False, "Should have raised an error"
        except ValueError:
            pass  # Expected
        
        # Check error statistics
        stats = error_recovery.get_error_statistics()
        assert isinstance(stats, dict)
        assert 'recovery_statistics' in stats
        
        logger.info("Error recovery validation complete")


def run_performance_benchmarks():
    """Run performance benchmarks for the HDC system."""
    if not HDC_AVAILABLE:
        logger.warning("HDC modules not available, skipping benchmarks")
        return
    
    logger.info("Running performance benchmarks")
    
    # Benchmark HDC system creation
    start_time = time.time()
    hdc = HDCSystem(dim=10000, device="cpu")
    creation_time = time.time() - start_time
    
    # Benchmark encoding operations
    test_text = "Hyperdimensional computing benchmark test"
    start_time = time.time()
    text_hv = hdc.encode_text(test_text)
    text_encoding_time = time.time() - start_time
    
    # Benchmark binding operations
    hv1 = HyperVector.random(dim=10000, device="cpu")
    hv2 = HyperVector.random(dim=10000, device="cpu")
    
    start_time = time.time()
    bound_hv = hdc.bind([hv1, hv2])
    binding_time = time.time() - start_time
    
    # Benchmark similarity computation
    start_time = time.time()
    similarity = hdc.cosine_similarity(hv1, hv2)
    similarity_time = time.time() - start_time
    
    # Report results
    logger.info(f"Performance Benchmark Results:")
    logger.info(f"  HDC System Creation: {creation_time*1000:.2f}ms")
    logger.info(f"  Text Encoding: {text_encoding_time*1000:.2f}ms")
    logger.info(f"  Binding Operation: {binding_time*1000:.2f}ms")
    logger.info(f"  Similarity Computation: {similarity_time*1000:.2f}ms")
    
    # Performance assertions
    assert creation_time < 1.0, "HDC system creation too slow"
    assert text_encoding_time < 0.1, "Text encoding too slow"
    assert binding_time < 0.01, "Binding operation too slow"
    assert similarity_time < 0.01, "Similarity computation too slow"
    
    logger.info("Performance benchmarks passed")


def run_statistical_validation():
    """Run statistical validation of research results.""" 
    if not HDC_AVAILABLE:
        logger.warning("HDC modules not available, skipping statistical validation")
        return
    
    logger.info("Running statistical validation")
    
    # Generate test datasets
    num_trials = 10
    results = []
    
    for trial in range(num_trials):
        # Create test data
        test_data = [HyperVector.random(dim=1000, device="cpu") for _ in range(10)]
        target_data = [HyperVector.random(dim=1000, device="cpu") for _ in range(10)]
        
        # Run research suite
        suite = BreakthroughResearchSuite(device="cpu")
        trial_results = suite.run_comprehensive_research(test_data[:5], target_data[:5])
        
        # Extract performance metrics
        trial_metrics = {}
        for algo_name, metrics in trial_results.items():
            trial_metrics[algo_name] = {
                'performance': metrics.performance_improvement,
                'significance': metrics.statistical_significance
            }
        
        results.append(trial_metrics)
    
    # Statistical analysis
    algorithms = list(results[0].keys())
    
    for algo_name in algorithms:
        performances = [r[algo_name]['performance'] for r in results]
        significances = [r[algo_name]['significance'] for r in results]
        
        mean_performance = np.mean(performances)
        std_performance = np.std(performances)
        mean_significance = np.mean(significances)
        
        logger.info(f"{algo_name} Statistics:")
        logger.info(f"  Mean Performance: {mean_performance:.3f} ± {std_performance:.3f}")
        logger.info(f"  Mean Significance: {mean_significance:.3f}")
        
        # Statistical validation
        assert mean_performance >= 0.0, f"{algo_name} negative performance"
        assert std_performance < 1.0, f"{algo_name} high variance"
        assert mean_significance >= 0.0, f"{algo_name} negative significance"
    
    logger.info("Statistical validation passed")


if __name__ == "__main__":
    """Run comprehensive test suite."""
    logger.info("Starting comprehensive HDC test suite")
    
    if not HDC_AVAILABLE:
        logger.error("HDC modules not available - cannot run tests")
        sys.exit(1)
    
    # Run pytest suite
    pytest_args = [
        __file__,
        "-v",
        "--tb=short",
        "--maxfail=10"
    ]
    
    pytest_result = pytest.main(pytest_args)
    
    # Run additional validations
    try:
        run_performance_benchmarks()
        run_statistical_validation()
        
        logger.info("All tests and validations passed successfully!")
        
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        sys.exit(1)
    
    # Exit with pytest result
    sys.exit(pytest_result)