#!/usr/bin/env python3
"""Comprehensive quality gates and validation testing."""

import sys
sys.path.insert(0, '.')

import time
import torch
import numpy as np
import hypervector as hv
from hypervector.utils.security import SecurityContext


def run_performance_benchmarks():
    """Run performance benchmarks."""
    print("🚀 Running performance benchmarks...")
    
    # Create optimized HDC system
    hdc_system, components = hv.accelerators.create_optimized_hdc_system(
        dim=10000,
        device='cpu',
        enable_profiling=True
    )
    
    # Benchmark text encoding
    start_time = time.perf_counter()
    for i in range(100):
        text_hv = hdc_system.encode_text(f"Benchmark text {i}")
    text_encoding_time = time.perf_counter() - start_time
    
    # Benchmark operations
    hv1 = hdc_system.random_hypervector()
    hv2 = hdc_system.random_hypervector()
    
    start_time = time.perf_counter()
    for i in range(1000):
        sim = hdc_system.cosine_similarity(hv1, hv2)
    similarity_time = time.perf_counter() - start_time
    
    print(f"✓ Text encoding (100 ops): {text_encoding_time:.3f}s ({text_encoding_time*10:.1f}ms/op)")
    print(f"✓ Cosine similarity (1000 ops): {similarity_time:.3f}s ({similarity_time:.1f}ms/op)")
    
    # Verify performance thresholds
    assert text_encoding_time < 10.0, f"Text encoding too slow: {text_encoding_time:.3f}s"
    assert similarity_time < 2.0, f"Similarity computation too slow: {similarity_time:.3f}s"
    
    return True


def run_memory_stress_test():
    """Run memory stress test."""
    print("🧠 Running memory stress test...")
    
    with SecurityContext(memory_limit_gb=1.0) as ctx:
        # Create multiple large hypervectors
        large_hvs = []
        
        for i in range(10):
            hv_large = hv.HyperVector(torch.randn(10000))
            large_hvs.append(hv_large)
            ctx.validate_input_data(f"test_{i}")
        
        # Test bundling operation with many vectors
        bundled = hv.bundle(large_hvs[:5])
        
        print(f"✓ Created {len(large_hvs)} large hypervectors")
        print(f"✓ Bundled result dimension: {bundled.dim}")
        
        # Cleanup
        del large_hvs
        del bundled
    
    return True


def run_accuracy_tests():
    """Run accuracy and correctness tests."""
    print("🎯 Running accuracy tests...")
    
    # Test mathematical properties
    hdc = hv.HDCSystem(dim=1000)
    
    # Test binding properties
    a = hdc.random_hypervector()
    b = hdc.random_hypervector()
    
    # Binding should be commutative
    ab = hdc.bind([a, b])
    ba = hdc.bind([b, a])
    binding_similarity = hdc.cosine_similarity(ab, ba)
    
    print(f"✓ Binding commutativity similarity: {binding_similarity:.3f}")
    assert binding_similarity > 0.95, f"Binding not commutative enough: {binding_similarity:.3f}"
    
    # Test bundling properties
    hvs = [hdc.random_hypervector() for _ in range(10)]
    bundled = hdc.bundle(hvs)
    
    # Each component should have some similarity to bundle
    similarities = [hdc.cosine_similarity(bundled, hv).item() for hv in hvs]
    avg_similarity = sum(similarities) / len(similarities)
    
    print(f"✓ Average component similarity to bundle: {avg_similarity:.3f}")
    assert avg_similarity > 0.1, f"Bundle similarity too low: {avg_similarity:.3f}"
    
    # Test permutation properties - permutation should create dissimilar vectors
    original = hdc.random_hypervector()
    permuted = hdc.permute(original, shift=1)
    perm_similarity = hdc.cosine_similarity(original, permuted)
    
    print(f"✓ Permutation similarity: {perm_similarity:.3f}")
    # Permutation should create dissimilar vectors (low similarity)
    assert abs(perm_similarity) < 0.3, f"Permutation similarity too high: {perm_similarity:.3f}"
    
    return True


def run_robustness_tests():
    """Run robustness and edge case tests."""
    print("🛡️ Running robustness tests...")
    
    hdc = hv.HDCSystem(dim=1000)
    
    # Test with very small numbers
    small_data = torch.randn(1000) * 1e-10
    small_hv = hv.HyperVector(small_data)
    
    # Test with very large numbers
    large_data = torch.randn(1000) * 1e10
    large_hv = hv.HyperVector(large_data)
    
    # Test operations with extreme values
    try:
        result = hdc.cosine_similarity(small_hv, large_hv)
        print(f"✓ Extreme value similarity: {result:.6f}")
    except Exception as e:
        print(f"✗ Extreme value test failed: {e}")
        return False
    
    # Test with identical vectors
    identical = hdc.random_hypervector()
    identical_sim = hdc.cosine_similarity(identical, identical)
    print(f"✓ Identical vector similarity: {identical_sim:.6f}")
    assert abs(identical_sim - 1.0) < 1e-6, f"Identical similarity not 1.0: {identical_sim:.6f}"
    
    # Test with orthogonal vectors (binary case)
    binary1 = hv.HyperVector(torch.ones(1000), mode="binary")
    binary2 = hv.HyperVector(-torch.ones(1000), mode="binary")
    orthogonal_sim = binary1.cosine_similarity(binary2)
    print(f"✓ Orthogonal binary similarity: {orthogonal_sim:.6f}")
    assert abs(orthogonal_sim + 1.0) < 1e-6, f"Orthogonal similarity not -1.0: {orthogonal_sim:.6f}"
    
    return True


def run_multimodal_integration_test():
    """Run multimodal integration test."""
    print("🎭 Running multimodal integration test...")
    
    hdc = hv.HDCSystem(dim=1000)
    
    # Test text encoding
    text_hv = hdc.encode_text("The quick brown fox")
    
    # Test image encoding (random image)
    fake_image = torch.randn(3, 224, 224)
    image_hv = hdc.encode_image(fake_image)
    
    # Test EEG encoding (synthetic EEG)
    fake_eeg = torch.randn(64, 1000) * 100  # 64 channels, 1000 samples
    eeg_hv = hdc.encode_eeg(fake_eeg, sampling_rate=250.0)
    
    # Test multimodal binding
    multimodal_hv = hdc.bind([text_hv, image_hv, eeg_hv])
    
    print(f"✓ Text HV dimension: {text_hv.dim}")
    print(f"✓ Image HV dimension: {image_hv.dim}")
    print(f"✓ EEG HV dimension: {eeg_hv.dim}")
    print(f"✓ Multimodal HV dimension: {multimodal_hv.dim}")
    
    # Test cross-modal similarities
    text_image_sim = hdc.cosine_similarity(text_hv, image_hv)
    text_eeg_sim = hdc.cosine_similarity(text_hv, eeg_hv)
    image_eeg_sim = hdc.cosine_similarity(image_hv, eeg_hv)
    
    print(f"✓ Text-Image similarity: {text_image_sim:.3f}")
    print(f"✓ Text-EEG similarity: {text_eeg_sim:.3f}")
    print(f"✓ Image-EEG similarity: {image_eeg_sim:.3f}")
    
    # All should be relatively uncorrelated
    assert abs(text_image_sim) < 0.3, f"Text-Image too similar: {text_image_sim:.3f}"
    assert abs(text_eeg_sim) < 0.3, f"Text-EEG too similar: {text_eeg_sim:.3f}"
    assert abs(image_eeg_sim) < 0.3, f"Image-EEG too similar: {image_eeg_sim:.3f}"
    
    return True


def run_production_readiness_test():
    """Run production readiness test."""
    print("🏭 Running production readiness test...")
    
    # Test monitoring and alerting
    from hypervector.production.monitoring import PerformanceMonitor, AlertManager
    
    monitor = PerformanceMonitor()
    monitor.start_monitoring()
    
    # Simulate some workload
    hdc = hv.HDCSystem(dim=1000)
    for i in range(50):
        text_hv = hdc.encode_text(f"Production test {i}")
        if i % 10 == 0:
            monitor.record_metric(f"test_metric_{i}", float(i), {"test": "production"})
    
    # Get health status
    health = monitor.get_system_health()
    print(f"✓ System health: {health.overall_status}")
    print(f"✓ CPU usage: {health.cpu_usage:.1f}%")
    print(f"✓ Memory usage: {health.memory_usage_mb:.1f}MB")
    
    monitor.stop_monitoring()
    
    # Test alert manager
    alert_manager = AlertManager()
    
    # Define a test alert
    def high_memory_condition(context):
        return context.get('memory_usage_mb', 0) > 5000
    
    alert_manager.define_alert_rule(
        "high_memory",
        high_memory_condition,
        severity="warning",
        description="Memory usage is high"
    )
    
    # Test alert triggering
    alert_manager.check_alerts({'memory_usage_mb': 6000})
    alert_manager.check_alerts({'memory_usage_mb': 1000})  # Should resolve
    
    summary = alert_manager.get_alert_summary()
    print(f"✓ Alert summary: {summary}")
    
    return True


def main():
    """Run all quality gates."""
    print("🔥 TERRAGON SDLC QUALITY GATES 🔥")
    print("=" * 50)
    
    quality_gates = [
        ("Performance Benchmarks", run_performance_benchmarks),
        ("Memory Stress Test", run_memory_stress_test),
        ("Accuracy Tests", run_accuracy_tests),
        ("Robustness Tests", run_robustness_tests),
        ("Multimodal Integration", run_multimodal_integration_test),
        ("Production Readiness", run_production_readiness_test),
    ]
    
    passed_gates = 0
    total_gates = len(quality_gates)
    
    for gate_name, gate_function in quality_gates:
        print(f"\n📋 {gate_name}")
        print("-" * 30)
        
        try:
            start_time = time.perf_counter()
            success = gate_function()
            end_time = time.perf_counter()
            
            if success:
                print(f"✅ PASSED in {end_time - start_time:.2f}s")
                passed_gates += 1
            else:
                print(f"❌ FAILED in {end_time - start_time:.2f}s")
                
        except Exception as e:
            print(f"❌ FAILED with error: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 50)
    print(f"🎯 QUALITY GATE SUMMARY: {passed_gates}/{total_gates} PASSED")
    
    if passed_gates == total_gates:
        print("🎉 ALL QUALITY GATES PASSED! READY FOR PRODUCTION! 🚀")
        return True
    else:
        print("⚠️  SOME QUALITY GATES FAILED. REVIEW REQUIRED.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)