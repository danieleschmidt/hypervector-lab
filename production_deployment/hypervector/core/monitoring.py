"""Production monitoring and health checks for HDC systems."""

import time
import threading
import logging
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass
from collections import defaultdict, deque
import torch


@dataclass
class HealthCheck:
    """Health check result."""
    name: str
    status: str  # 'healthy', 'warning', 'critical'
    message: str
    timestamp: float
    details: Optional[Dict[str, Any]] = None


class MetricsCollector:
    """Collect and track system metrics."""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.metrics = defaultdict(lambda: deque(maxlen=max_history))
        self.counters = defaultdict(int)
        self._lock = threading.Lock()
        
    def record_metric(self, name: str, value: float, timestamp: Optional[float] = None):
        """Record a metric value."""
        if timestamp is None:
            timestamp = time.time()
            
        with self._lock:
            self.metrics[name].append((timestamp, value))
    
    def increment_counter(self, name: str, amount: int = 1):
        """Increment a counter."""
        with self._lock:
            self.counters[name] += amount
    
    def get_metric_stats(self, name: str, window_seconds: Optional[float] = None) -> Dict[str, float]:
        """Get statistics for a metric."""
        with self._lock:
            if name not in self.metrics:
                return {}
                
            values = list(self.metrics[name])
            
            # Filter by time window if specified
            if window_seconds:
                cutoff = time.time() - window_seconds
                values = [(t, v) for t, v in values if t >= cutoff]
            
            if not values:
                return {}
            
            numbers = [v for _, v in values]
            return {
                'count': len(numbers),
                'min': min(numbers),
                'max': max(numbers),
                'avg': sum(numbers) / len(numbers),
                'latest': numbers[-1]
            }
    
    def get_all_stats(self) -> Dict[str, Any]:
        """Get all metrics and counters."""
        with self._lock:
            return {
                'metrics': {name: self.get_metric_stats(name) for name in self.metrics},
                'counters': dict(self.counters)
            }


class HealthMonitor:
    """Monitor system health and performance."""
    
    def __init__(self):
        self.metrics = MetricsCollector()
        self.health_checks: Dict[str, Callable[[], HealthCheck]] = {}
        self.last_health_status = {}
        self._logger = logging.getLogger(__name__)
        
    def register_health_check(self, name: str, check_func: Callable[[], HealthCheck]):
        """Register a health check function."""
        self.health_checks[name] = check_func
        
    def check_memory_usage(self) -> HealthCheck:
        """Check system memory usage."""
        try:
            if torch.cuda.is_available():
                # GPU memory
                allocated = torch.cuda.memory_allocated()
                reserved = torch.cuda.memory_reserved()
                total = torch.cuda.get_device_properties(0).total_memory
                
                usage_percent = (allocated / total) * 100
                
                if usage_percent > 90:
                    status = 'critical'
                    message = f"GPU memory usage critical: {usage_percent:.1f}%"
                elif usage_percent > 75:
                    status = 'warning'
                    message = f"GPU memory usage high: {usage_percent:.1f}%"
                else:
                    status = 'healthy'
                    message = f"GPU memory usage normal: {usage_percent:.1f}%"
                
                details = {
                    'allocated_gb': allocated / (1024**3),
                    'reserved_gb': reserved / (1024**3),
                    'total_gb': total / (1024**3),
                    'usage_percent': usage_percent
                }
            else:
                # CPU memory (simplified)
                status = 'healthy'
                message = "CPU memory check passed"
                details = {'gpu_available': False}
                
            return HealthCheck(
                name='memory_usage',
                status=status,
                message=message,
                timestamp=time.time(),
                details=details
            )
            
        except Exception as e:
            return HealthCheck(
                name='memory_usage',
                status='critical',
                message=f"Memory check failed: {str(e)}",
                timestamp=time.time()
            )
    
    def check_compute_performance(self) -> HealthCheck:
        """Check compute performance."""
        try:
            # Simple compute test
            start_time = time.time()
            
            # Create a test tensor operation
            if torch.cuda.is_available():
                device = 'cuda'
                test_tensor = torch.randn(1000, 1000, device=device)
                result = torch.matmul(test_tensor, test_tensor.T)
                torch.cuda.synchronize()
            else:
                device = 'cpu'
                test_tensor = torch.randn(500, 500)  # Smaller for CPU
                result = torch.matmul(test_tensor, test_tensor.T)
            
            compute_time = time.time() - start_time
            
            # Record metric
            self.metrics.record_metric('compute_latency_ms', compute_time * 1000)
            
            # Determine status
            threshold = 0.1 if device == 'cuda' else 0.5  # Different thresholds
            
            if compute_time > threshold * 3:
                status = 'critical'
                message = f"Compute performance critical: {compute_time:.3f}s"
            elif compute_time > threshold:
                status = 'warning'
                message = f"Compute performance slow: {compute_time:.3f}s"
            else:
                status = 'healthy'
                message = f"Compute performance good: {compute_time:.3f}s"
            
            return HealthCheck(
                name='compute_performance',
                status=status,
                message=message,
                timestamp=time.time(),
                details={
                    'device': device,
                    'compute_time_ms': compute_time * 1000,
                    'threshold_ms': threshold * 1000
                }
            )
            
        except Exception as e:
            return HealthCheck(
                name='compute_performance',
                status='critical',
                message=f"Compute test failed: {str(e)}",
                timestamp=time.time()
            )
    
    def run_all_health_checks(self) -> Dict[str, HealthCheck]:
        """Run all registered health checks."""
        results = {}
        
        # Built-in checks
        results['memory_usage'] = self.check_memory_usage()
        results['compute_performance'] = self.check_compute_performance()
        
        # Custom registered checks
        for name, check_func in self.health_checks.items():
            try:
                results[name] = check_func()
            except Exception as e:
                results[name] = HealthCheck(
                    name=name,
                    status='critical',
                    message=f"Health check failed: {str(e)}",
                    timestamp=time.time()
                )
        
        # Update status tracking
        self.last_health_status = results
        
        return results
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status."""
        health_results = self.run_all_health_checks()
        
        # Count status types
        status_counts = defaultdict(int)
        for check in health_results.values():
            status_counts[check.status] += 1
        
        # Determine overall status
        if status_counts['critical'] > 0:
            overall_status = 'critical'
        elif status_counts['warning'] > 0:
            overall_status = 'warning'
        else:
            overall_status = 'healthy'
        
        return {
            'overall_status': overall_status,
            'timestamp': time.time(),
            'health_checks': {name: {
                'status': check.status,
                'message': check.message,
                'timestamp': check.timestamp,
                'details': check.details
            } for name, check in health_results.items()},
            'status_summary': dict(status_counts),
            'metrics_summary': self.metrics.get_all_stats()
        }


class PerformanceProfiler:
    """Profile performance of HDC operations."""
    
    def __init__(self, monitor: HealthMonitor):
        self.monitor = monitor
        self._active_profiles = {}
        
    def start_profile(self, operation_name: str) -> str:
        """Start profiling an operation."""
        profile_id = f"{operation_name}_{int(time.time() * 1000000)}"
        self._active_profiles[profile_id] = {
            'operation': operation_name,
            'start_time': time.time(),
            'start_memory': self._get_memory_usage()
        }
        return profile_id
    
    def end_profile(self, profile_id: str) -> Dict[str, Any]:
        """End profiling and record metrics."""
        if profile_id not in self._active_profiles:
            raise ValueError(f"Profile {profile_id} not found")
        
        profile_data = self._active_profiles.pop(profile_id)
        end_time = time.time()
        end_memory = self._get_memory_usage()
        
        duration = end_time - profile_data['start_time']
        memory_delta = end_memory - profile_data['start_memory']
        
        # Record metrics
        operation = profile_data['operation']
        self.monitor.metrics.record_metric(f'{operation}_duration_ms', duration * 1000)
        self.monitor.metrics.record_metric(f'{operation}_memory_mb', memory_delta / (1024**2))
        self.monitor.metrics.increment_counter(f'{operation}_count')
        
        result = {
            'operation': operation,
            'duration_ms': duration * 1000,
            'memory_delta_mb': memory_delta / (1024**2),
            'start_time': profile_data['start_time'],
            'end_time': end_time
        }
        
        return result
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in bytes."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated()
        else:
            # For CPU, we'd use psutil if available, but return 0 for now
            return 0.0
    
    def profile_function(self, func_name: str):
        """Decorator to profile a function."""
        def decorator(func):
            def wrapper(*args, **kwargs):
                profile_id = self.start_profile(func_name)
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    self.end_profile(profile_id)
            return wrapper
        return decorator


# Global instances
_global_monitor = None
_global_profiler = None

def get_monitor() -> HealthMonitor:
    """Get global health monitor instance."""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = HealthMonitor()
    return _global_monitor

def get_profiler() -> PerformanceProfiler:
    """Get global performance profiler instance."""
    global _global_profiler
    if _global_profiler is None:
        _global_profiler = PerformanceProfiler(get_monitor())
    return _global_profiler

def profile_operation(operation_name: str):
    """Decorator to profile an operation."""
    return get_profiler().profile_function(operation_name)

import time
from functools import wraps

def monitor_performance(func):
    """Decorator to monitor function performance"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            logging.info(f"{func.__name__} completed in {duration:.4f}s")
            return result
        except Exception as e:
            duration = time.time() - start_time
            logging.error(f"{func.__name__} failed after {duration:.4f}s: {e}")
            raise
    return wrapper

def log_memory_usage():
    """Log current memory usage"""
    try:
        import psutil
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        logging.info(f"Memory usage: {memory_mb:.2f} MB")
    except ImportError:
        logging.warning("psutil not available for memory monitoring")
