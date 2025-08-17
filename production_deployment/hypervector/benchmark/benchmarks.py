"""Comprehensive benchmarking suite for HDC operations."""

import time
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
from typing import Dict, List, Tuple, Optional, Any
import statistics
from dataclasses import dataclass

from ..core import HyperVector, bind, bundle, permute, cosine_similarity, HDCSystem
from ..encoders import TextEncoder, VisionEncoder, EEGEncoder
from ..applications import BCIClassifier, CrossModalRetrieval
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""
    operation: str
    mean_time: float
    std_time: float
    min_time: float
    max_time: float
    throughput: Optional[float] = None
    memory_usage_mb: Optional[float] = None
    device: str = "cpu"
    parameters: Dict[str, Any] = None


class BenchmarkSuite:
    """Comprehensive benchmark suite for HDC operations."""
    
    def __init__(self, device: str = "cpu", warmup_runs: int = 3):
        """Initialize benchmark suite.
        
        Args:
            device: Device to run benchmarks on
            warmup_runs: Number of warmup runs before timing
        """
        self.device = device
        self.warmup_runs = warmup_runs
        self.results: List[BenchmarkResult] = []
    
    def _time_operation(
        self, 
        operation_fn, 
        num_runs: int = 10,
        measure_memory: bool = True
    ) -> Tuple[List[float], Optional[float]]:
        """Time an operation multiple times.
        
        Args:
            operation_fn: Function to time
            num_runs: Number of timing runs
            measure_memory: Whether to measure memory usage
            
        Returns:
            Tuple of (execution_times, memory_usage_mb)
        """
        # Warmup runs
        for _ in range(self.warmup_runs):
            try:
                operation_fn()
            except Exception as e:
                logger.warning(f"Warmup run failed: {e}")
                return [], None
        
        # Clear cache if CUDA
        if self.device.startswith('cuda') and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Measure memory before
        memory_before = self._get_memory_usage()
        
        # Timing runs
        times = []
        for _ in range(num_runs):
            start_time = time.perf_counter()
            
            try:
                operation_fn()
            except Exception as e:
                logger.error(f"Benchmark run failed: {e}")
                continue
            
            # Synchronize for CUDA
            if self.device.startswith('cuda') and torch.cuda.is_available():
                torch.cuda.synchronize()
            
            end_time = time.perf_counter()
            times.append(end_time - start_time)
        
        # Measure memory after
        memory_after = self._get_memory_usage()
        memory_usage = memory_after - memory_before if memory_before and memory_after else None
        
        return times, memory_usage
    
    def _get_memory_usage(self) -> Optional[float]:
        """Get current memory usage in MB."""
        try:
            if self.device.startswith('cuda') and torch.cuda.is_available():
                return torch.cuda.memory_allocated() / (1024**2)
            else:
                import psutil
                return psutil.Process().memory_info().rss / (1024**2)
        except ImportError:
            return None
    
    def benchmark_hypervector_creation(self, dimensions: List[int]) -> None:
        """Benchmark HyperVector creation."""
        for dim in dimensions:
            # Random generation
            def create_random():
                return HyperVector.random(dim, device=self.device)
            
            times, memory = self._time_operation(create_random)
            if times:
                result = BenchmarkResult(
                    operation=f"hypervector_random_{dim}",
                    mean_time=statistics.mean(times),
                    std_time=statistics.stdev(times) if len(times) > 1 else 0,
                    min_time=min(times),
                    max_time=max(times),
                    memory_usage_mb=memory,
                    device=self.device,
                    parameters={"dimension": dim}
                )
                self.results.append(result)
                logger.info(f"Random HyperVector ({dim}D): {result.mean_time*1000:.2f}ms")
    
    def benchmark_operations(self, dimensions: List[int]) -> None:
        """Benchmark core HDC operations."""
        for dim in dimensions:
            hv1 = HyperVector.random(dim, device=self.device)
            hv2 = HyperVector.random(dim, device=self.device)
            hvs = [HyperVector.random(dim, device=self.device) for _ in range(10)]
            
            # Bind operation
            def bind_op():
                return bind(hv1, hv2)
            
            times, memory = self._time_operation(bind_op)
            if times:
                throughput = dim / (statistics.mean(times) * 1e6)  # Operations per microsecond
                result = BenchmarkResult(
                    operation=f"bind_{dim}",
                    mean_time=statistics.mean(times),
                    std_time=statistics.stdev(times) if len(times) > 1 else 0,
                    min_time=min(times),
                    max_time=max(times),
                    throughput=throughput,
                    memory_usage_mb=memory,
                    device=self.device,
                    parameters={"dimension": dim}
                )
                self.results.append(result)
                logger.info(f"Bind ({dim}D): {result.mean_time*1e6:.2f}μs")
            
            # Bundle operation
            def bundle_op():
                return bundle(hvs)
            
            times, memory = self._time_operation(bundle_op)
            if times:
                throughput = len(hvs) * dim / (statistics.mean(times) * 1e6)
                result = BenchmarkResult(
                    operation=f"bundle_{dim}_{len(hvs)}",
                    mean_time=statistics.mean(times),
                    std_time=statistics.stdev(times) if len(times) > 1 else 0,
                    min_time=min(times),
                    max_time=max(times),
                    throughput=throughput,
                    memory_usage_mb=memory,
                    device=self.device,
                    parameters={"dimension": dim, "num_vectors": len(hvs)}
                )
                self.results.append(result)
                logger.info(f"Bundle ({dim}D, {len(hvs)} vectors): {result.mean_time*1000:.2f}ms")
            
            # Similarity operation
            def similarity_op():
                return cosine_similarity(hv1, hv2)
            
            times, memory = self._time_operation(similarity_op)
            if times:
                result = BenchmarkResult(
                    operation=f"similarity_{dim}",
                    mean_time=statistics.mean(times),
                    std_time=statistics.stdev(times) if len(times) > 1 else 0,
                    min_time=min(times),
                    max_time=max(times),
                    memory_usage_mb=memory,
                    device=self.device,
                    parameters={"dimension": dim}
                )
                self.results.append(result)
                logger.info(f"Similarity ({dim}D): {result.mean_time*1e6:.2f}μs")
    
    def benchmark_encoders(self, dimension: int = 10000) -> None:
        """Benchmark encoder performance."""
        # Text encoder
        text_encoder = TextEncoder(dim=dimension, device=self.device)
        test_texts = [
            "Hello world",
            "The quick brown fox jumps over the lazy dog",
            "Hyperdimensional computing enables efficient similarity-preserving mappings",
            "A" * 100,  # Long text
            "Machine learning and artificial intelligence"
        ]
        
        for i, text in enumerate(test_texts):
            def encode_text():
                return text_encoder.encode(text)
            
            times, memory = self._time_operation(encode_text, num_runs=5)
            if times:
                result = BenchmarkResult(
                    operation=f"text_encode_len_{len(text)}",
                    mean_time=statistics.mean(times),
                    std_time=statistics.stdev(times) if len(times) > 1 else 0,
                    min_time=min(times),
                    max_time=max(times),
                    memory_usage_mb=memory,
                    device=self.device,
                    parameters={"text_length": len(text), "dimension": dimension}
                )
                self.results.append(result)
                logger.info(f"Text encoding ({len(text)} chars): {result.mean_time*1000:.2f}ms")
        
        # Vision encoder
        vision_encoder = VisionEncoder(dim=dimension, device=self.device)
        test_images = [
            torch.rand(3, 224, 224, device=self.device),
            torch.rand(3, 512, 512, device=self.device),
        ]
        
        for i, image in enumerate(test_images):
            def encode_image():
                return vision_encoder.encode(image)
            
            times, memory = self._time_operation(encode_image, num_runs=3)
            if times:
                h, w = image.shape[-2:]
                result = BenchmarkResult(
                    operation=f"vision_encode_{h}x{w}",
                    mean_time=statistics.mean(times),
                    std_time=statistics.stdev(times) if len(times) > 1 else 0,
                    min_time=min(times),
                    max_time=max(times),
                    memory_usage_mb=memory,
                    device=self.device,
                    parameters={"image_size": f"{h}x{w}", "dimension": dimension}
                )
                self.results.append(result)
                logger.info(f"Vision encoding ({h}x{w}): {result.mean_time*1000:.2f}ms")
        
        # EEG encoder
        eeg_encoder = EEGEncoder(dim=dimension, device=self.device)
        test_signals = [
            np.random.randn(8, 250),   # 8 channels, 1 second at 250Hz
            np.random.randn(64, 1000), # 64 channels, 4 seconds at 250Hz
        ]
        
        for signal in test_signals:
            def encode_eeg():
                return eeg_encoder.encode(signal, sampling_rate=250.0)
            
            times, memory = self._time_operation(encode_eeg, num_runs=3)
            if times:
                channels, samples = signal.shape
                result = BenchmarkResult(
                    operation=f"eeg_encode_{channels}ch_{samples}samples",
                    mean_time=statistics.mean(times),
                    std_time=statistics.stdev(times) if len(times) > 1 else 0,
                    min_time=min(times),
                    max_time=max(times),
                    memory_usage_mb=memory,
                    device=self.device,
                    parameters={"channels": channels, "samples": samples, "dimension": dimension}
                )
                self.results.append(result)
                logger.info(f"EEG encoding ({channels}ch, {samples} samples): {result.mean_time*1000:.2f}ms")
    
    def benchmark_applications(self, dimension: int = 5000) -> None:
        """Benchmark application performance."""
        # BCI Classifier
        bci = BCIClassifier(
            channels=16,
            hypervector_dim=dimension,
            window_size=125
        )
        
        # Add some training data
        for i in range(10):
            eeg_data = np.random.randn(16, 125)
            label = "class_A" if i < 5 else "class_B"
            bci.add_training_sample(eeg_data, label)
        
        # Benchmark classification
        test_eeg = np.random.randn(16, 125)
        
        def classify_eeg():
            return bci.classify(test_eeg)
        
        times, memory = self._time_operation(classify_eeg)
        if times:
            result = BenchmarkResult(
                operation="bci_classify",
                mean_time=statistics.mean(times),
                std_time=statistics.stdev(times) if len(times) > 1 else 0,
                min_time=min(times),
                max_time=max(times),
                memory_usage_mb=memory,
                device=self.device,
                parameters={"channels": 16, "samples": 125, "dimension": dimension}
            )
            self.results.append(result)
            logger.info(f"BCI classification: {result.mean_time*1000:.2f}ms")
        
        # Cross-modal retrieval
        retrieval = CrossModalRetrieval(dim=dimension)
        
        # Index some items
        for i in range(100):
            text = f"Sample text {i}"
            retrieval.add_item(f"item_{i}", text=text)
        
        # Benchmark query
        def query_retrieval():
            return retrieval.query_by_text("sample query", top_k=10)
        
        times, memory = self._time_operation(query_retrieval)
        if times:
            result = BenchmarkResult(
                operation="retrieval_query",
                mean_time=statistics.mean(times),
                std_time=statistics.stdev(times) if len(times) > 1 else 0,
                min_time=min(times),
                max_time=max(times),
                memory_usage_mb=memory,
                device=self.device,
                parameters={"index_size": 100, "top_k": 10, "dimension": dimension}
            )
            self.results.append(result)
            logger.info(f"Retrieval query (100 items): {result.mean_time*1000:.2f}ms")
    
    def run_all_benchmarks(self) -> List[BenchmarkResult]:
        """Run all benchmarks and return results."""
        logger.info(f"Starting comprehensive benchmarks on {self.device}")
        
        dimensions = [1000, 5000, 10000]
        
        self.benchmark_hypervector_creation(dimensions)
        self.benchmark_operations(dimensions)
        self.benchmark_encoders(dimension=10000)
        self.benchmark_applications(dimension=5000)
        
        logger.info(f"Completed {len(self.results)} benchmarks")
        return self.results
    
    def print_summary(self) -> None:
        """Print benchmark summary."""
        if not self.results:
            print("No benchmark results available.")
            return
        
        print(f"\n{'='*60}")
        print(f"BENCHMARK SUMMARY ({self.device.upper()})")
        print(f"{'='*60}")
        
        # Group by operation type
        groups = {}
        for result in self.results:
            op_type = result.operation.split('_')[0]
            if op_type not in groups:
                groups[op_type] = []
            groups[op_type].append(result)
        
        for op_type, results in groups.items():
            print(f"\n{op_type.upper()} Operations:")
            print("-" * 40)
            
            for result in sorted(results, key=lambda x: x.mean_time):
                time_str = f"{result.mean_time*1000:.2f}ms"
                if result.throughput:
                    time_str += f" ({result.throughput:.1f} ops/μs)"
                
                print(f"  {result.operation:<25} {time_str}")
        
        # Overall statistics
        all_times = [r.mean_time for r in self.results]
        total_memory = sum(r.memory_usage_mb for r in self.results if r.memory_usage_mb)
        
        print(f"\n{'='*60}")
        print(f"OVERALL STATISTICS")
        print(f"{'='*60}")
        print(f"Total benchmarks: {len(self.results)}")
        print(f"Fastest operation: {min(all_times)*1000:.2f}ms")
        print(f"Slowest operation: {max(all_times)*1000:.2f}ms")
        print(f"Average time: {statistics.mean(all_times)*1000:.2f}ms")
        if total_memory:
            print(f"Total memory usage: {total_memory:.1f}MB")


def run_benchmarks(
    device: str = "cpu",
    dimensions: Optional[List[int]] = None,
    verbose: bool = True
) -> List[BenchmarkResult]:
    """Run HDC benchmarks.
    
    Args:
        device: Device to run on ('cpu', 'cuda')
        dimensions: List of dimensions to test
        verbose: Whether to print results
        
    Returns:
        List of benchmark results
    """
    suite = BenchmarkSuite(device=device)
    results = suite.run_all_benchmarks()
    
    if verbose:
        suite.print_summary()
    
    return results