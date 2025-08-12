#!/usr/bin/env python3
"""Test enhanced validation and error handling."""

import sys
sys.path.insert(0, '.')

import torch
import numpy as np
import hypervector as hv
from hypervector.core.exceptions import DimensionMismatchError, InvalidModeError, DeviceError


def test_enhanced_validation():
    """Test enhanced validation features."""
    print("Testing enhanced validation and error handling...")
    
    # Test invalid data validation
    try:
        hv.HyperVector([])  # Empty list
        assert False, "Should have raised ValueError for empty data"
    except ValueError as e:
        print(f"✓ Caught empty data error: {e}")
    
    # Test NaN detection
    try:
        data_with_nan = torch.tensor([1.0, float('nan'), 3.0])
        hv.HyperVector(data_with_nan)
        assert False, "Should have raised ValueError for NaN data"
    except ValueError as e:
        print(f"✓ Caught NaN data error: {e}")
    
    # Test infinite values detection
    try:
        data_with_inf = torch.tensor([1.0, float('inf'), 3.0])
        hv.HyperVector(data_with_inf)
        assert False, "Should have raised ValueError for infinite data"
    except ValueError as e:
        print(f"✓ Caught infinite data error: {e}")
    
    # Test device validation with invalid CUDA device
    try:
        if not torch.cuda.is_available():
            hv.HyperVector([1.0, 2.0, 3.0], device='cuda:0')
            assert False, "Should have raised DeviceError for unavailable CUDA"
    except DeviceError as e:
        print(f"✓ Caught device error: {e}")
    
    # Test dimension mismatch in operations
    try:
        hv1 = hv.HyperVector(torch.randn(100))
        hv2 = hv.HyperVector(torch.randn(200))
        result = hv1 + hv2
        assert False, "Should have raised DimensionMismatchError"
    except DimensionMismatchError as e:
        print(f"✓ Caught dimension mismatch error: {e}")
    
    # Test type validation in operations
    try:
        hv1 = hv.HyperVector(torch.randn(100))
        result = hv1 + "invalid"
        assert False, "Should have raised TypeError"
    except TypeError as e:
        print(f"✓ Caught type error: {e}")
    
    # Test zero vector similarity warning
    print("\nTesting zero vector handling...")
    hv_zero = hv.HyperVector(torch.zeros(100))
    hv_normal = hv.HyperVector(torch.randn(100))
    
    # This should log a warning and return 0
    similarity = hv_zero.cosine_similarity(hv_normal)
    print(f"Zero vector similarity: {similarity}")
    
    # Test enhanced memory validation
    print("\nTesting memory management...")
    memory_manager = hv.accelerators.MemoryManager(max_memory_gb=0.001)  # Very small limit
    
    # Test memory usage tracking
    current_memory = memory_manager.get_current_memory_usage()
    print(f"Current memory usage: {current_memory:.3f}GB")
    
    # Test memory cleanup
    freed_memory = memory_manager.cleanup_memory(force=True)
    print(f"Freed memory: {freed_memory:.3f}GB")
    
    print("\n✅ All enhanced validation tests passed!")


def test_security_features():
    """Test security features."""
    print("\nTesting security features...")
    
    from hypervector.utils.security import sanitize_input, validate_file_path, SecurityContext
    
    # Test input sanitization
    unsafe_input = "Hello\x00World\x1F"
    safe_input = sanitize_input(unsafe_input)
    print(f"Sanitized input: '{safe_input}'")
    
    # Test security context
    with SecurityContext(validate_inputs=True, memory_limit_gb=1.0) as ctx:
        ctx.validate_input_data("test string")
        print("✓ Security context validation passed")
    
    print("✅ Security tests passed!")


def test_monitoring_features():
    """Test monitoring and profiling features."""
    print("\nTesting monitoring features...")
    
    from hypervector.production.monitoring import PerformanceMonitor, MetricsCollector
    
    # Test performance monitoring
    monitor = PerformanceMonitor(retention_hours=1)
    
    # Record some metrics
    monitor.record_metric("test_metric", 123.45, {"test": "tag"})
    
    # Get system health
    health = monitor.get_system_health()
    print(f"System health: {health.overall_status}")
    print(f"Memory usage: {health.memory_usage_mb:.1f}MB")
    
    # Test metrics collector
    collector = MetricsCollector()
    collector.define_metric("custom_metric", "Test metric", "count")
    collector.record_custom_metric("custom_metric", 42.0, {"env": "test"})
    
    summary = collector.get_metric_summary()
    print(f"Metrics summary: {summary}")
    
    print("✅ Monitoring tests passed!")


def test_performance_optimization():
    """Test performance optimization features."""
    print("\nTesting performance optimization...")
    
    # Create optimized HDC system
    hdc_system, components = hv.accelerators.create_optimized_hdc_system(
        dim=1000,
        device='cpu',
        enable_profiling=True,
        enable_batching=True,
        enable_parallel=True
    )
    
    # Test batch processing
    batch_processor = components['batch_processor']
    items = list(range(100))
    
    def process_item(item):
        return item * 2
    
    results = batch_processor.process_batch(items, lambda batch: [process_item(x) for x in batch])
    print(f"Batch processing result length: {len(results)}")
    
    # Test parallel processing
    parallel_processor = components['parallel_processor']
    parallel_results = parallel_processor.parallel_map(lambda x: x ** 2, items[:10])
    print(f"Parallel processing result: {parallel_results[:5]}...")
    
    # Test adaptive optimizer
    optimizer = components['adaptive_optimizer']
    optimal_params = optimizer.get_optimal_parameters("test_op", 1000)
    print(f"Optimal parameters: {optimal_params}")
    
    print("✅ Performance optimization tests passed!")


if __name__ == "__main__":
    try:
        test_enhanced_validation()
        test_security_features()
        test_monitoring_features()
        test_performance_optimization()
        
        print("\n🎉 All enhanced feature tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)