#!/usr/bin/env python3
"""
Advanced HyperVector-Lab Examples
Demonstrates production-ready features and optimizations.
"""

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
import torch
from pathlib import Path

# Import HyperVector-Lab components
import hypervector as hv
from hypervector.utils.config import Config, get_config, set_config
from hypervector.utils.logging import setup_logging, get_logger, PerformanceTimer
from hypervector.utils.security import SecurityContext
from hypervector.accelerators import CPUAccelerator, BatchProcessor, MemoryManager
from hypervector.benchmark import run_benchmarks, HDCProfiler
from hypervector.deployment import DeploymentConfig

# Setup logging
setup_logging(level="INFO")
logger = get_logger(__name__)


def example_1_production_configuration():
    """Example 1: Production-ready configuration management."""
    print("\n" + "="*60)
    print("EXAMPLE 1: Production Configuration")
    print("="*60)
    
    # Create custom configuration
    config = Config({
        'hypervector': {
            'default_dim': 8000,
            'default_mode': 'dense',
            'default_device': 'cpu'
        },
        'performance': {
            'batch_size': 64,
            'num_workers': 8,
            'memory_limit_gb': 6
        },
        'logging': {
            'level': 'INFO',
            'metrics_enabled': True
        }
    })
    
    # Set as global configuration
    set_config(config)
    
    # Initialize HDC system with config
    hdc = hv.HDCSystem(
        dim=config.get('hypervector.default_dim'),
        device=config.get('hypervector.default_device'),
        mode=config.get('hypervector.default_mode')
    )
    
    print(f"✓ HDC System initialized with {hdc.dim}D vectors on {hdc.device}")
    print(f"✓ Configuration loaded: batch_size={config.get('performance.batch_size')}")
    
    # Save configuration for deployment
    config.save('config/production.json')
    print("✓ Configuration saved to config/production.json")


def example_2_security_and_validation():
    """Example 2: Security features and input validation."""
    print("\n" + "="*60)
    print("EXAMPLE 2: Security and Validation")
    print("="*60)
    
    # Use security context for safe operations
    with SecurityContext(validate_inputs=True, memory_limit_gb=4.0) as security:
        
        # Initialize components with validation
        try:
            hdc = hv.HDCSystem(dim=5000, device='cpu')
            
            # Test text encoding with validation
            texts = [
                "Valid input text",
                "Another valid example",
                "Safe hyperdimensional computing"
            ]
            
            for text in texts:
                security.validate_input_data(text, "text_input")
                text_hv = hdc.encode_text(text)
                print(f"✓ Safely encoded text: '{text[:30]}...' -> {text_hv.dim}D vector")
            
            # Test EEG encoding with validation
            eeg_signal = np.random.randn(16, 250) * 50  # Realistic EEG amplitudes
            eeg_hv = hdc.encode_eeg(eeg_signal, sampling_rate=250.0)
            print(f"✓ Safely encoded EEG: {eeg_signal.shape} -> {eeg_hv.dim}D vector")
            
        except Exception as e:
            logger.error(f"Security validation failed: {e}")
    
    print("✓ All operations completed within security context")


def example_3_performance_optimization():
    """Example 3: Performance optimization and acceleration."""
    print("\n" + "="*60)
    print("EXAMPLE 3: Performance Optimization")
    print("="*60)
    
    # Initialize accelerators
    cpu_accelerator = CPUAccelerator(num_threads=4)
    batch_processor = BatchProcessor()
    memory_manager = MemoryManager(max_memory_gb=4.0, cache_size=500)
    
    # Create test data
    hvs = [hv.HyperVector.random(5000, device='cpu') for _ in range(100)]
    
    with PerformanceTimer("Optimized Operations") as timer:
        
        # 1. Optimized bundling
        with memory_manager:
            bundled_hv = cpu_accelerator.vectorized_bundle(hvs[:10])
            print(f"✓ Vectorized bundling: 10 vectors -> {bundled_hv.dim}D")
            
            # 2. Batch similarity computation
            query_hv = hv.HyperVector.random(5000, device='cpu')
            similarities = cpu_accelerator.batch_cosine_similarity(query_hv, hvs[:20])
            print(f"✓ Batch similarity: 1 query vs 20 candidates -> {len(similarities)} scores")
            
            # 3. Memory-efficient processing
            def process_chunk(chunk_hvs):
                return [cpu_accelerator.vectorized_bind(chunk_hvs[0], hv) for hv in chunk_hvs[1:]]
            
            results = cpu_accelerator.chunked_operation(
                process_chunk, 
                hvs[:50], 
                chunk_size=10
            )
            print(f"✓ Chunked processing: 50 vectors in chunks of 10")
    
    # Get memory statistics
    memory_stats = memory_manager.get_memory_stats()
    print(f"✓ Memory usage: {memory_stats['current_memory_gb']:.2f}GB")
    print(f"✓ Cache utilization: {len(memory_manager.cache)}/{memory_manager.cache_size}")


def example_4_advanced_applications():
    """Example 4: Advanced application scenarios."""
    print("\n" + "="*60)
    print("EXAMPLE 4: Advanced Applications")
    print("="*60)
    
    # 1. Real-time BCI with online learning
    bci_system = hv.BCIClassifier(
        channels=32,
        sampling_rate=250,
        window_size=125,
        hypervector_dim=8000,
        adaptation_rate=0.15
    )
    
    # Simulate training data
    print("Training BCI classifier...")
    for i in range(50):
        # Generate realistic EEG patterns
        if i % 2 == 0:
            # Motor imagery left (higher activity in C3)
            eeg_data = np.random.randn(32, 125) * 20
            eeg_data[10, :] += 30  # C3 activation
            label = "motor_left"
        else:
            # Motor imagery right (higher activity in C4)
            eeg_data = np.random.randn(32, 125) * 20
            eeg_data[22, :] += 30  # C4 activation
            label = "motor_right"
        
        bci_system.add_training_sample(eeg_data, label)
    
    # Test classification
    test_eeg = np.random.randn(32, 125) * 20
    test_eeg[10, :] += 25  # Simulate left motor imagery
    
    prediction, confidence = bci_system.classify(test_eeg)
    print(f"✓ BCI Classification: {prediction} (confidence: {confidence:.3f})")
    print(f"✓ Model accuracy: {bci_system.get_accuracy():.3f}")
    
    # 2. Large-scale cross-modal retrieval
    retrieval_system = hv.CrossModalRetrieval(dim=6000)
    
    # Index diverse content
    print("Indexing multimodal content...")
    contents = [
        ("Neural interfaces for brain-computer interaction", "bci_research"),
        ("Machine learning algorithms for pattern recognition", "ml_algorithms"),
        ("Hyperdimensional computing in neuromorphic systems", "hdc_neuro"),
        ("Signal processing techniques for EEG analysis", "eeg_processing"),
        ("Artificial intelligence and cognitive computing", "ai_cognitive")
    ]
    
    for text, item_id in contents:
        # Add text with synthetic image and EEG
        image = torch.rand(3, 224, 224)
        eeg = np.random.randn(8, 250)
        
        retrieval_system.add_item(
            item_id,
            text=text,
            image=image,
            eeg=eeg
        )
    
    # Test cross-modal queries
    query_results = retrieval_system.query_by_text(
        "brain computer interface EEG",
        modality="all",
        top_k=3
    )
    
    print(f"✓ Cross-modal retrieval found {len(query_results)} relevant items:")
    for item_id, similarity, _ in query_results:
        print(f"  - {item_id}: similarity={similarity:.3f}")


def example_5_benchmarking_and_profiling():
    """Example 5: Comprehensive benchmarking and profiling."""
    print("\n" + "="*60)
    print("EXAMPLE 5: Benchmarking and Profiling")
    print("="*60)
    
    # Initialize profiler
    profiler = HDCProfiler("AdvancedExample")
    
    # Create test system
    hdc = hv.HDCSystem(dim=8000, device='cpu')
    
    # Profile different operations
    with profiler.profile("text_encoding_batch"):
        texts = [f"Sample text {i} for encoding benchmark" for i in range(20)]
        text_hvs = [hdc.encode_text(text) for text in texts]
    
    with profiler.profile("image_encoding"):
        image = torch.rand(3, 224, 224)
        image_hv = hdc.encode_image(image)
    
    with profiler.profile("complex_operations"):
        # Simulate complex HDC workflow
        hvs = [hv.HyperVector.random(8000) for _ in range(50)]
        
        # Bundle in groups
        groups = [hvs[i:i+10] for i in range(0, len(hvs), 10)]
        group_hvs = [hv.bundle(group) for group in groups]
        
        # Bind groups
        final_hv = group_hvs[0]
        for group_hv in group_hvs[1:]:
            final_hv = hv.bind(final_hv, group_hv)
    
    # Print profiling summary
    profiler.print_summary()
    
    # Run comprehensive benchmarks
    print("\nRunning comprehensive benchmarks...")
    benchmark_results = run_benchmarks(device='cpu', verbose=False)
    
    # Analyze results
    fast_ops = [r for r in benchmark_results if r.mean_time < 0.001]  # < 1ms
    slow_ops = [r for r in benchmark_results if r.mean_time > 0.1]    # > 100ms
    
    print(f"✓ Fast operations (< 1ms): {len(fast_ops)}")
    print(f"✓ Slow operations (> 100ms): {len(slow_ops)}")
    
    if fast_ops:
        fastest = min(fast_ops, key=lambda x: x.mean_time)
        print(f"✓ Fastest operation: {fastest.operation} ({fastest.mean_time*1e6:.1f}μs)")


def example_6_deployment_ready():
    """Example 6: Deployment-ready configuration."""
    print("\n" + "="*60)
    print("EXAMPLE 6: Deployment Configuration")
    print("="*60)
    
    # Create deployment configuration
    deployment_config = DeploymentConfig()
    
    # Add model configurations
    from hypervector.deployment.deployment_config import ModelConfig
    
    bci_model = ModelConfig(
        model_path="models/bci_classifier.pth",
        model_type="bci",
        dimension=10000,
        device="cpu",
        batch_size=32,
        max_memory_gb=2.0
    )
    
    retrieval_model = ModelConfig(
        model_path="models/retrieval_system.pkl",
        model_type="retrieval", 
        dimension=8000,
        device="cpu",
        batch_size=16,
        max_memory_gb=4.0
    )
    
    deployment_config.models = [bci_model, retrieval_model]
    
    # Configure for production
    deployment_config.environment = "production"
    deployment_config.server.host = "0.0.0.0"
    deployment_config.server.port = 8080
    deployment_config.server.workers = 4
    
    deployment_config.monitoring.enable_metrics = True
    deployment_config.monitoring.metrics_port = 9090
    
    deployment_config.security.enable_authentication = True
    deployment_config.security.rate_limit_requests_per_minute = 1000
    
    deployment_config.cache.enable_caching = True
    deployment_config.cache.cache_size = 2000
    
    # Save deployment configuration
    Path("config").mkdir(exist_ok=True)
    deployment_config.save("config/deployment.yaml")
    
    print("✓ Deployment configuration created:")
    print(f"  - Server: {deployment_config.server.host}:{deployment_config.server.port}")
    print(f"  - Workers: {deployment_config.server.workers}")
    print(f"  - Models: {len(deployment_config.models)}")
    print(f"  - Authentication: {deployment_config.security.enable_authentication}")
    print(f"  - Caching: {deployment_config.cache.enable_caching}")
    print("✓ Configuration saved to config/deployment.yaml")


def main():
    """Run all advanced examples."""
    print("🧠 HyperVector-Lab Advanced Examples")
    print("Production-Ready Hyperdimensional Computing")
    
    try:
        example_1_production_configuration()
        example_2_security_and_validation()
        example_3_performance_optimization()
        example_4_advanced_applications()
        example_5_benchmarking_and_profiling()
        example_6_deployment_ready()
        
        print("\n" + "="*60)
        print("🎉 ALL ADVANCED EXAMPLES COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("Your HyperVector-Lab system is ready for production deployment.")
        
    except Exception as e:
        logger.error(f"Example failed: {e}")
        raise


if __name__ == "__main__":
    main()