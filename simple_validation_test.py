#!/usr/bin/env python3
"""Simple validation test for HDC system without external dependencies."""

import sys
import time
import traceback
import torch
import numpy as np

def test_basic_hdc_functionality():
    """Test basic HDC functionality."""
    print("🧪 Testing basic HDC functionality...")
    
    try:
        # Test imports
        from hypervector import HDCSystem, HyperVector
        from hypervector.core.operations import bind, bundle, cosine_similarity
        print("✅ HDC imports successful")
        
        # Test HDC system creation
        hdc = HDCSystem(dim=1000, device="cpu")
        print("✅ HDC system created")
        
        # Test hypervector creation
        hv1 = HyperVector.random(dim=1000, device="cpu")
        hv2 = HyperVector.random(dim=1000, device="cpu")
        print("✅ Random hypervectors created")
        
        # Test binding
        bound_hv = bind(hv1, hv2)
        assert isinstance(bound_hv, HyperVector)
        assert bound_hv.dim == 1000
        print("✅ Binding operation successful")
        
        # Test bundling
        bundled_hv = bundle([hv1, hv2])
        assert isinstance(bundled_hv, HyperVector)
        assert bundled_hv.dim == 1000
        print("✅ Bundling operation successful")
        
        # Test similarity
        similarity = cosine_similarity(hv1, hv2)
        assert torch.is_tensor(similarity)
        assert -1.0 <= similarity.item() <= 1.0
        print("✅ Similarity computation successful")
        
        # Test encoding
        text_hv = hdc.encode_text("test text")
        assert isinstance(text_hv, HyperVector)
        print("✅ Text encoding successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic HDC test failed: {e}")
        traceback.print_exc()
        return False


def test_breakthrough_algorithms():
    """Test breakthrough algorithms."""
    print("\n🔬 Testing breakthrough algorithms...")
    
    try:
        from hypervector.research.breakthrough_algorithms import (
            SelfOrganizingHyperMap, EvolutionaryHDC, MetaLearningHDC,
            QuantumCoherentHDC, NeuroplasticityHDC
        )
        print("✅ Breakthrough algorithm imports successful")
        
        # Generate test data
        test_data = [HyperVector.random(dim=500, device="cpu") for _ in range(5)]
        
        # Test Self-Organizing HyperMap
        som = SelfOrganizingHyperMap(input_dim=500, map_size=(5, 5), device="cpu")
        som_metrics = som.train(test_data, epochs=3)
        assert hasattr(som_metrics, 'algorithm_name')
        print("✅ Self-Organizing HyperMap test passed")
        
        # Test Evolutionary HDC
        evo = EvolutionaryHDC(population_size=10, dim=500, device="cpu")
        evo_metrics = evo.evolve(test_data[0], generations=3)
        assert hasattr(evo_metrics, 'algorithm_name')
        print("✅ Evolutionary HDC test passed")
        
        # Test Meta-Learning HDC
        meta = MetaLearningHDC(base_dim=500, device="cpu")
        meta_tasks = [[(test_data[i], test_data[(i + 1) % len(test_data)])] for i in range(2)]
        meta_metrics = meta.meta_train(meta_tasks, meta_epochs=2)
        assert hasattr(meta_metrics, 'algorithm_name')
        print("✅ Meta-Learning HDC test passed")
        
        # Test Quantum Coherent HDC
        quantum_hdc = QuantumCoherentHDC(dim=500, device="cpu")
        amplitudes = test_data[0].data
        phases = torch.rand(500) * 2 * np.pi
        quantum_state = quantum_hdc.create_coherent_state(amplitudes, phases)
        assert 'amplitudes' in quantum_state
        print("✅ Quantum Coherent HDC test passed")
        
        # Test Neuroplasticity HDC
        plastic_hdc = NeuroplasticityHDC(dim=500, device="cpu")
        adapted_hv = plastic_hdc.adaptive_encode(test_data[0], test_data[1])
        assert isinstance(adapted_hv, HyperVector)
        print("✅ Neuroplasticity HDC test passed")
        
        return True
        
    except Exception as e:
        print(f"❌ Breakthrough algorithms test failed: {e}")
        traceback.print_exc()
        return False


def test_quantum_real_time():
    """Test quantum-enhanced real-time systems."""
    print("\n⚡ Testing quantum-enhanced real-time systems...")
    
    try:
        from hypervector.research.quantum_enhanced_real_time_hdc import (
            QuantumCoherentProcessor, RealTimeHDCEngine
        )
        print("✅ Quantum real-time imports successful")
        
        # Test Quantum Coherent Processor
        processor = QuantumCoherentProcessor(dim=500, device="cpu")
        classical_hv = HyperVector.random(dim=500, device="cpu")
        quantum_state = processor.create_quantum_hypervector(classical_hv)
        assert hasattr(quantum_state, 'amplitudes')
        print("✅ Quantum Coherent Processor test passed")
        
        # Test Real-Time HDC Engine (basic functionality)
        engine = RealTimeHDCEngine(dim=500, device="cpu", max_workers=2)
        stats = engine.get_performance_stats()
        assert isinstance(stats, dict)
        print("✅ Real-Time HDC Engine test passed")
        
        return True
        
    except Exception as e:
        print(f"❌ Quantum real-time test failed: {e}")
        traceback.print_exc()
        return False


def test_error_recovery():
    """Test error recovery systems."""
    print("\n🛡️ Testing error recovery systems...")
    
    try:
        from hypervector.utils.advanced_error_recovery import HDCErrorRecovery
        print("✅ Error recovery imports successful")
        
        # Test error recovery initialization
        error_recovery = HDCErrorRecovery(max_recovery_attempts=2)
        
        # Test successful operation
        def test_operation():
            return "success"
        
        result = error_recovery.protected_execute(test_operation)
        assert result == "success"
        print("✅ Error recovery test passed")
        
        return True
        
    except Exception as e:
        print(f"❌ Error recovery test failed: {e}")
        traceback.print_exc()
        return False


def test_performance_optimization():
    """Test performance optimization systems."""
    print("\n🚀 Testing performance optimization...")
    
    try:
        from hypervector.accelerators.performance_optimizer import (
            PerformanceProfiler, BatchProcessor, ParallelProcessor
        )
        print("✅ Performance optimization imports successful")
        
        # Test Performance Profiler
        profiler = PerformanceProfiler()
        
        @profiler.profile_operation("test_op")
        def test_operation(x):
            return x * 2
        
        result = test_operation(5)
        assert result == 10
        print("✅ Performance profiler test passed")
        
        # Test Batch Processor
        batch_processor = BatchProcessor(max_batch_size=10)
        items = list(range(15))
        results = batch_processor.process_batch(items, lambda batch: [x * 2 for x in batch])
        assert len(results) == 15
        print("✅ Batch processor test passed")
        
        # Test Parallel Processor
        parallel_processor = ParallelProcessor(max_workers=2)
        items = list(range(10))
        results = parallel_processor.parallel_map(lambda x: x * 2, items)
        assert len(results) == 10
        parallel_processor.shutdown()
        print("✅ Parallel processor test passed")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance optimization test failed: {e}")
        traceback.print_exc()
        return False


def run_performance_benchmark():
    """Run simple performance benchmark."""
    print("\n📊 Running performance benchmark...")
    
    try:
        from hypervector import HDCSystem, HyperVector
        
        # Benchmark system creation
        start_time = time.time()
        hdc = HDCSystem(dim=10000, device="cpu")
        creation_time = (time.time() - start_time) * 1000
        
        # Benchmark operations
        hv1 = HyperVector.random(dim=10000, device="cpu")
        hv2 = HyperVector.random(dim=10000, device="cpu")
        
        # Binding benchmark
        start_time = time.time()
        bound_hv = hdc.bind([hv1, hv2])
        binding_time = (time.time() - start_time) * 1000
        
        # Similarity benchmark
        start_time = time.time()
        similarity = hdc.cosine_similarity(hv1, hv2)
        similarity_time = (time.time() - start_time) * 1000
        
        print(f"📈 Performance Results:")
        print(f"   System Creation: {creation_time:.2f}ms")
        print(f"   Binding: {binding_time:.2f}ms")
        print(f"   Similarity: {similarity_time:.2f}ms")
        
        # Performance assertions
        assert creation_time < 5000, f"System creation too slow: {creation_time:.2f}ms"
        assert binding_time < 100, f"Binding too slow: {binding_time:.2f}ms"
        assert similarity_time < 50, f"Similarity too slow: {similarity_time:.2f}ms"
        
        print("✅ Performance benchmark passed")
        return True
        
    except Exception as e:
        print(f"❌ Performance benchmark failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all validation tests."""
    print("🧬 HDC System Comprehensive Validation Suite")
    print("=" * 50)
    
    tests = [
        ("Basic HDC Functionality", test_basic_hdc_functionality),
        ("Breakthrough Algorithms", test_breakthrough_algorithms),
        ("Quantum Real-Time Systems", test_quantum_real_time),
        ("Error Recovery", test_error_recovery),
        ("Performance Optimization", test_performance_optimization),
        ("Performance Benchmark", run_performance_benchmark)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'=' * 50}")
        print(f"Running: {test_name}")
        print('=' * 50)
        
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} FAILED with exception: {e}")
            traceback.print_exc()
    
    print(f"\n{'=' * 50}")
    print(f"VALIDATION SUMMARY")
    print('=' * 50)
    print(f"Tests Passed: {passed}/{total}")
    print(f"Success Rate: {(passed/total)*100:.1f}%")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! HDC system is ready for production.")
        return 0
    else:
        print("⚠️  Some tests failed. Review implementation before deployment.")
        return 1


if __name__ == "__main__":
    sys.exit(main())