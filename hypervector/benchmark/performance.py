from ..core.hypervector import HyperVector

import time
import torch
import statistics
from typing import Dict, List, Callable, Any
import json
from datetime import datetime
import platform

class PerformanceBenchmark:
    """Comprehensive performance benchmarking for HDC operations"""
    
    def __init__(self, warmup_runs: int = 3, benchmark_runs: int = 10):
        self.warmup_runs = warmup_runs
        self.benchmark_runs = benchmark_runs
        self.results = {}
        
    def benchmark_operation(self, name: str, operation: Callable, *args, **kwargs) -> Dict[str, float]:
        """Benchmark a single operation"""
        print(f"Benchmarking {name}...")
        
        # Warmup runs
        for _ in range(self.warmup_runs):
            try:
                operation(*args, **kwargs)
            except Exception as e:
                print(f"Warmup failed for {name}: {e}")
                return {"error": str(e)}
        
        # Benchmark runs
        times = []
        memory_usage = []
        
        for run in range(self.benchmark_runs):
            # Clear cache before each run
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # Memory before
            mem_before = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            
            # Time the operation
            start_time = time.perf_counter()
            
            try:
                result = operation(*args, **kwargs)
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                end_time = time.perf_counter()
                
                # Memory after
                mem_after = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
                
                times.append(end_time - start_time)
                memory_usage.append(mem_after - mem_before)
                
            except Exception as e:
                print(f"Benchmark run {run} failed for {name}: {e}")
                continue
        
        if not times:
            return {"error": "All benchmark runs failed"}
        
        # Calculate statistics
        stats = {
            "mean_time": statistics.mean(times),
            "median_time": statistics.median(times),
            "min_time": min(times),
            "max_time": max(times),
            "std_time": statistics.stdev(times) if len(times) > 1 else 0,
            "mean_memory_mb": statistics.mean(memory_usage) / (1024 * 1024),
            "runs": len(times)
        }
        
        self.results[name] = stats
        return stats
    
    def benchmark_hdc_operations(self, dim: int = 10000, device: str = 'cpu'):
        """Benchmark core HDC operations"""
        print(f"Benchmarking HDC operations (dim={dim}, device={device})")
        
        # Create test data
        hv1 = HyperVector.random(dim, device=device)
        hv2 = HyperVector.random(dim, device=device)
        hvs = [HyperVector.random(dim, device=device) for _ in range(100)]
        
        # Benchmark individual operations
        benchmarks = {
            "random_generation": lambda: HyperVector.random(dim, device=device),
            "binding": lambda: hv1 * hv2,
            "bundling_small": lambda: sum(hvs[:10]) / len(hvs[:10]),
            "bundling_large": lambda: sum(hvs) / len(hvs),
            "cosine_similarity": lambda: hv1.cosine_similarity(hv2),
            "normalization": lambda: hv1.normalize(),
            "binarization": lambda: hv1.binarize(),
            "ternarization": lambda: hv1.ternarize(),
        }
        
        results = {}
        for name, operation in benchmarks.items():
            results[name] = self.benchmark_operation(name, operation)
        
        return results
    
    def benchmark_scaling(self, dimensions: List[int] = None, devices: List[str] = None):
        """Benchmark scaling across dimensions and devices"""
        if dimensions is None:
            dimensions = [1000, 5000, 10000, 20000, 50000]
        
        if devices is None:
            devices = ['cpu']
            if torch.cuda.is_available():
                devices.append('cuda')
        
        scaling_results = {}
        
        for dim in dimensions:
            for device in devices:
                key = f"dim_{dim}_{device}"
                print(f"Scaling benchmark: {key}")
                
                try:
                    scaling_results[key] = self.benchmark_hdc_operations(dim, device)
                except Exception as e:
                    print(f"Scaling benchmark failed for {key}: {e}")
                    scaling_results[key] = {"error": str(e)}
        
        return scaling_results
    
    def generate_report(self, save_path: str = None) -> Dict[str, Any]:
        """Generate comprehensive benchmark report"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "system_info": {
                "platform": platform.platform(),
                "python_version": platform.python_version(),
                "torch_version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
            },
            "benchmark_config": {
                "warmup_runs": self.warmup_runs,
                "benchmark_runs": self.benchmark_runs
            },
            "results": self.results
        }
        
        if torch.cuda.is_available():
            report["system_info"]["cuda_device_count"] = torch.cuda.device_count()
            report["system_info"]["cuda_device_name"] = torch.cuda.get_device_name()
        
        if save_path:
            with open(save_path, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"Benchmark report saved to {save_path}")
        
        return report
    
    def compare_implementations(self, implementations: Dict[str, Callable], *args, **kwargs):
        """Compare different implementations of the same operation"""
        comparison = {}
        
        for name, impl in implementations.items():
            comparison[name] = self.benchmark_operation(f"compare_{name}", impl, *args, **kwargs)
        
        # Calculate relative performance
        if len(comparison) > 1:
            baseline = min(comparison.values(), key=lambda x: x.get("mean_time", float('inf')))
            baseline_time = baseline.get("mean_time", 1.0)
            
            for name, result in comparison.items():
                if "mean_time" in result:
                    result["speedup"] = baseline_time / result["mean_time"]
        
        return comparison

def run_comprehensive_benchmark():
    """Run comprehensive performance benchmark"""
    benchmark = PerformanceBenchmark()
    
    print("Starting comprehensive HDC performance benchmark...")
    
    # Basic operations benchmark
    basic_results = benchmark.benchmark_hdc_operations()
    
    # Scaling benchmark
    scaling_results = benchmark.benchmark_scaling()
    
    # Generate and save report
    report = benchmark.generate_report("hdc_benchmark_report.json")
    
    print("\nBenchmark Summary:")
    print("=" * 50)
    for name, result in basic_results.items():
        if "error" not in result:
            print(f"{name:20s}: {result['mean_time']*1000:.2f}ms ± {result['std_time']*1000:.2f}ms")
        else:
            print(f"{name:20s}: ERROR - {result['error']}")
    
    return report

if __name__ == "__main__":
    run_comprehensive_benchmark()
