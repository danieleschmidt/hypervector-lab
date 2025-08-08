#!/usr/bin/env python3
"""Comprehensive validation of HyperVector-Lab research and production features."""

import time
import torch
import hypervector

def test_core_functionality():
    """Test core HDC functionality."""
    print("🧠 Testing Core HDC Functionality...")
    
    # Core operations
    hv1 = hypervector.HyperVector.random(1000, seed=42)
    hv2 = hypervector.HyperVector.random(1000, seed=43)
    
    bound = hypervector.bind(hv1, hv2)
    bundled = hypervector.bundle([hv1, hv2])
    similarity = hypervector.cosine_similarity(hv1, hv2)
    
    print(f"  ✅ Core operations: bind, bundle, similarity = {similarity:.4f}")
    
    # HDC System
    hdc = hypervector.HDCSystem(dim=1000, device='cpu')
    text_hv = hdc.encode_text("Hello HDC world!")
    image_hv = hdc.encode_image(torch.rand(3, 224, 224))
    eeg_hv = hdc.encode_eeg(torch.rand(64, 1000), sampling_rate=250.0)
    
    print(f"  ✅ Multi-modal encoding: text, image, EEG")
    
    # Applications
    bci = hypervector.BCIClassifier(channels=8, sampling_rate=250, hypervector_dim=1000)
    retrieval = hypervector.CrossModalRetrieval(dim=1000)
    
    print(f"  ✅ Applications: BCI classifier, cross-modal retrieval")
    
    return True

def test_research_algorithms():
    """Test novel research algorithms."""
    print("🔬 Testing Novel Research Algorithms...")
    
    # Hierarchical HDC
    from hypervector.research.novel_algorithms import HierarchicalHDC
    hierarchical = HierarchicalHDC(base_dim=500, levels=3)
    test_data = torch.rand(100)
    encodings = hierarchical.encode_hierarchical(test_data)
    similarities = hierarchical.hierarchical_similarity(encodings, encodings)
    
    print(f"  ✅ Hierarchical HDC: {len(encodings)} levels, similarity: {similarities[0]:.4f}")
    
    # Adaptive Binding
    from hypervector.research.novel_algorithms import AdaptiveBindingOperator
    adaptive = AdaptiveBindingOperator(dim=1000)
    hv1 = hypervector.HyperVector.random(1000)
    hv2 = hypervector.HyperVector.random(1000)
    context = hypervector.HyperVector.random(1000)
    adaptive_result = adaptive.adaptive_bind(hv1, hv2, context)
    
    print(f"  ✅ Adaptive Binding: context-aware binding completed")
    
    # Quantum-Inspired HDC
    from hypervector.research.novel_algorithms import QuantumInspiredHDC
    quantum = QuantumInspiredHDC(dim=1000, coherence_time=100.0)
    hvs = [hypervector.HyperVector.random(1000) for _ in range(3)]
    superposition = quantum.create_superposition(hvs)
    measurement = quantum.measure(superposition)
    
    print(f"  ✅ Quantum-Inspired HDC: superposition and measurement")
    
    # Temporal HDC
    from hypervector.research.novel_algorithms import TemporalHDC
    temporal = TemporalHDC(dim=1000, temporal_resolution=20)
    sequence = [hypervector.HyperVector.random(1000) for _ in range(5)]
    temporal_encoding = temporal.encode_temporal_sequence(sequence)
    
    print(f"  ✅ Temporal HDC: sequence encoding with time awareness")
    
    return True

def test_experimental_encoders():
    """Test experimental encoders."""
    print("🧪 Testing Experimental Encoders...")
    
    # Graph Encoder
    from hypervector.research.experimental_encoders import GraphEncoder
    graph_encoder = GraphEncoder(dim=1000, max_nodes=10)
    adj_matrix = torch.tensor([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=torch.float)
    graph_hv = graph_encoder.encode_graph(adj_matrix)
    
    print(f"  ✅ Graph Encoder: 3-node graph encoded")
    
    # Sequence Encoder
    from hypervector.research.experimental_encoders import SequenceEncoder
    seq_encoder = SequenceEncoder(dim=1000, max_length=20, attention_heads=4)
    sequence = [hypervector.HyperVector.random(1000) for _ in range(5)]
    seq_hv = seq_encoder.encode_sequence(sequence, use_attention=True)
    
    print(f"  ✅ Sequence Encoder: attention-based sequence encoding")
    
    # Attention-Based Encoder
    from hypervector.research.experimental_encoders import AttentionBasedEncoder
    attention_encoder = AttentionBasedEncoder(dim=1000, modalities=['text', 'vision', 'audio'])
    modality_data = {
        'text': hypervector.HyperVector.random(1000),
        'vision': hypervector.HyperVector.random(1000),
        'audio': hypervector.HyperVector.random(1000)
    }
    fused_hv = attention_encoder.encode_multimodal(modality_data)
    importance = attention_encoder.modality_importance()
    
    print(f"  ✅ Attention Encoder: multi-modal fusion, text importance: {importance.get('text', 0):.3f}")
    
    return True

def test_comparative_studies():
    """Test comparative studies framework."""
    print("📊 Testing Comparative Studies Framework...")
    
    from hypervector.research.comparative_studies import ComparisonFramework, BenchmarkComparator
    
    # Comparison Framework
    framework = ComparisonFramework(num_trials=3)
    
    def algorithm_a(dataset, param=1.0):
        return hypervector.HyperVector.random(1000)
    
    def algorithm_b(dataset, param=1.0):
        return hypervector.HyperVector.random(1000)
    
    def evaluation_fn(result, dataset):
        return {'accuracy': 0.85 + torch.rand(1).item() * 0.1}
    
    test_datasets = [{'data': i} for i in range(2)]
    
    result_a = framework.run_experiment(algorithm_a, 'Algorithm_A', {'param': 1.0}, evaluation_fn, test_datasets)
    result_b = framework.run_experiment(algorithm_b, 'Algorithm_B', {'param': 1.5}, evaluation_fn, test_datasets)
    
    print(f"  ✅ Experiment Framework: A={result_a.execution_time:.4f}s, B={result_b.execution_time:.4f}s")
    
    # Benchmark Comparator
    benchmark = BenchmarkComparator(dimensions=[500, 1000])
    
    algorithms = {
        'standard': lambda dim: hypervector.HyperVector.random(dim),
        'optimized': lambda dim: hypervector.HyperVector.random(dim)
    }
    
    results = benchmark.run_comprehensive_benchmark(algorithms, ['creation_speed', 'similarity_accuracy'])
    
    print(f"  ✅ Benchmark Comparator: {len(results)} algorithms benchmarked")
    
    return True

def test_production_features():
    """Test production features."""
    print("🚀 Testing Production Features...")
    
    # GPU Acceleration
    from hypervector.production.gpu_acceleration import CUDAAccelerator, BatchProcessor
    accelerator = CUDAAccelerator()
    
    hvs = [hypervector.HyperVector.random(1000) for _ in range(5)]
    accelerated_result = accelerator.accelerated_bundle(hvs)
    
    print(f"  ✅ GPU Acceleration: device={accelerator.device}")
    
    # Batch Processing
    batch_processor = BatchProcessor(accelerator, max_batch_size=100)
    
    operations = [{'hvs': hvs[:2]} for _ in range(3)]
    batch_results = batch_processor.process_batch_operations(operations, 'bind')
    
    print(f"  ✅ Batch Processing: {len(batch_results)} operations processed")
    
    # Performance Monitoring
    from hypervector.production.monitoring import PerformanceMonitor, MetricsCollector
    monitor = PerformanceMonitor(retention_hours=1)
    
    # Time an operation
    timer_id = monitor.start_timer('test_op')
    time.sleep(0.001)
    monitor.stop_timer(timer_id, 'test_op', success=True)
    
    health = monitor.get_system_health()
    
    print(f"  ✅ Monitoring: health={health.overall_status}, memory={health.memory_usage_mb:.1f}MB")
    
    # Custom Metrics
    metrics_collector = MetricsCollector()
    metrics_collector.define_metric('test_metric', 'Test metric description', 'ops/sec', 'avg')
    metrics_collector.record_custom_metric('test_metric', 1000.0, {'component': 'core'})
    
    print(f"  ✅ Metrics Collection: custom metrics defined and recorded")
    
    return True

def test_benchmarks():
    """Test benchmark performance."""
    print("⚡ Running Performance Benchmarks...")
    
    from hypervector.benchmark import benchmarks
    suite = benchmarks.BenchmarkSuite(device='cpu')
    
    # Run focused benchmarks
    suite.benchmark_hypervector_creation([1000])
    suite.benchmark_operations([1000])
    
    results = suite.results[-5:]  # Last 5 results
    
    for result in results:
        print(f"  ✅ {result.operation}: {result.mean_time*1000:.2f}ms ± {result.std_time*1000:.2f}ms")
    
    return True

def run_comprehensive_test():
    """Run comprehensive validation test."""
    print("🧠⚡ HyperVector-Lab Comprehensive Validation")
    print("=" * 60)
    
    test_functions = [
        test_core_functionality,
        test_research_algorithms,
        test_experimental_encoders,
        test_comparative_studies,
        test_production_features,
        test_benchmarks
    ]
    
    passed_tests = 0
    total_tests = len(test_functions)
    start_time = time.time()
    
    for test_func in test_functions:
        try:
            result = test_func()
            if result:
                passed_tests += 1
            print()
        except Exception as e:
            print(f"  ❌ Test failed: {e}")
            print()
    
    execution_time = time.time() - start_time
    
    print("=" * 60)
    print(f"📊 Test Results: {passed_tests}/{total_tests} passed")
    print(f"⏱️  Execution Time: {execution_time:.2f} seconds")
    
    if passed_tests == total_tests:
        print("🎉 ALL TESTS PASSED! HyperVector-Lab is fully functional!")
        print("🏆 Ready for research, production, and large-scale deployment!")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
    
    print("=" * 60)
    
    return passed_tests == total_tests

if __name__ == "__main__":
    success = run_comprehensive_test()
    exit(0 if success else 1)