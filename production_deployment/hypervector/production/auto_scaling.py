"""Auto-scaling and load balancing for HDC systems."""

import time
import threading
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
from queue import Queue, Empty
import logging


@dataclass
class ScalingMetrics:
    """Metrics for auto-scaling decisions."""
    cpu_utilization: float
    memory_utilization: float
    gpu_utilization: float
    queue_depth: int
    response_time_p95: float
    throughput: float
    error_rate: float
    timestamp: float


@dataclass
class ScalingPolicy:
    """Auto-scaling policy configuration."""
    min_instances: int = 1
    max_instances: int = 10
    target_cpu_utilization: float = 70.0
    target_memory_utilization: float = 80.0
    target_response_time_ms: float = 1000.0
    scale_up_threshold: float = 0.8
    scale_down_threshold: float = 0.5
    cooldown_period: float = 300.0  # 5 minutes
    metrics_window: float = 300.0   # 5 minutes


class WorkerInstance:
    """Represents a worker instance for processing."""
    
    def __init__(self, instance_id: str, worker_func: Callable):
        self.instance_id = instance_id
        self.worker_func = worker_func
        self.is_active = False
        self.last_used = time.time()
        self.request_count = 0
        self.error_count = 0
        self.total_processing_time = 0.0
        self._thread = None
        self._stop_event = threading.Event()
        
    def start(self, task_queue: Queue, result_queue: Queue):
        """Start the worker instance."""
        self.is_active = True
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._worker_loop,
            args=(task_queue, result_queue),
            daemon=True
        )
        self._thread.start()
        
    def stop(self):
        """Stop the worker instance."""
        self.is_active = False
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=30)
    
    def _worker_loop(self, task_queue: Queue, result_queue: Queue):
        """Main worker loop."""
        while not self._stop_event.is_set():
            try:
                # Get task with timeout
                task = task_queue.get(timeout=1.0)
                if task is None:  # Poison pill
                    break
                
                # Process task
                start_time = time.time()
                try:
                    result = self.worker_func(task)
                    processing_time = time.time() - start_time
                    
                    # Update metrics
                    self.request_count += 1
                    self.total_processing_time += processing_time
                    self.last_used = time.time()
                    
                    # Send result
                    result_queue.put({
                        'status': 'success',
                        'result': result,
                        'processing_time': processing_time,
                        'worker_id': self.instance_id
                    })
                    
                except Exception as e:
                    processing_time = time.time() - start_time
                    self.error_count += 1
                    self.last_used = time.time()
                    
                    result_queue.put({
                        'status': 'error',
                        'error': str(e),
                        'processing_time': processing_time,
                        'worker_id': self.instance_id
                    })
                
                finally:
                    task_queue.task_done()
                    
            except Empty:
                continue  # Timeout, check stop event
            except Exception as e:
                logging.error(f"Worker {self.instance_id} error: {e}")
                break
    
    def get_stats(self) -> Dict[str, float]:
        """Get worker statistics."""
        avg_processing_time = (
            self.total_processing_time / max(self.request_count, 1)
        )
        error_rate = self.error_count / max(self.request_count, 1)
        
        return {
            'request_count': self.request_count,
            'error_count': self.error_count,
            'error_rate': error_rate,
            'avg_processing_time': avg_processing_time,
            'last_used': self.last_used,
            'is_active': self.is_active
        }


class AutoScaler:
    """Auto-scaling manager for HDC workloads."""
    
    def __init__(self, worker_func: Callable, policy: Optional[ScalingPolicy] = None):
        self.worker_func = worker_func
        self.policy = policy or ScalingPolicy()
        
        self.workers: Dict[str, WorkerInstance] = {}
        self.task_queue = Queue()
        self.result_queue = Queue()
        
        self.metrics_history: List[ScalingMetrics] = []
        self.last_scale_action = 0.0
        self.next_worker_id = 1
        
        self._monitoring_thread = None
        self._stop_monitoring = threading.Event()
        self._logger = logging.getLogger(__name__)
        
        # Start with minimum instances
        for _ in range(self.policy.min_instances):
            self._add_worker()
    
    def start_monitoring(self):
        """Start auto-scaling monitoring."""
        self._stop_monitoring.clear()
        self._monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self._monitoring_thread.start()
        self._logger.info("Auto-scaling monitoring started")
    
    def stop_monitoring(self):
        """Stop auto-scaling monitoring."""
        self._stop_monitoring.set()
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=30)
        self._logger.info("Auto-scaling monitoring stopped")
    
    def submit_task(self, task: Any) -> str:
        """Submit a task for processing."""
        task_id = f"task_{int(time.time() * 1000000)}"
        self.task_queue.put((task_id, task))
        return task_id
    
    def get_result(self, timeout: Optional[float] = None) -> Dict[str, Any]:
        """Get a result from processing."""
        try:
            return self.result_queue.get(timeout=timeout)
        except Empty:
            raise TimeoutError("No result available within timeout")
    
    def _add_worker(self) -> str:
        """Add a new worker instance."""
        worker_id = f"worker_{self.next_worker_id}"
        self.next_worker_id += 1
        
        worker = WorkerInstance(worker_id, self.worker_func)
        worker.start(self.task_queue, self.result_queue)
        
        self.workers[worker_id] = worker
        self._logger.info(f"Added worker {worker_id}")
        
        return worker_id
    
    def _remove_worker(self, worker_id: str):
        """Remove a worker instance."""
        if worker_id in self.workers:
            worker = self.workers[worker_id]
            worker.stop()
            del self.workers[worker_id]
            self._logger.info(f"Removed worker {worker_id}")
    
    def _collect_metrics(self) -> ScalingMetrics:
        """Collect current system metrics."""
        current_time = time.time()
        
        # Calculate queue depth
        queue_depth = self.task_queue.qsize()
        
        # Collect worker stats
        worker_stats = [w.get_stats() for w in self.workers.values()]
        
        # Calculate aggregated metrics
        total_requests = sum(s['request_count'] for s in worker_stats)
        total_errors = sum(s['error_count'] for s in worker_stats)
        
        error_rate = total_errors / max(total_requests, 1)
        
        # Average processing time (simplified)
        avg_response_time = sum(
            s['avg_processing_time'] for s in worker_stats
        ) / max(len(worker_stats), 1)
        
        # Simplified utilization metrics (would be more complex in practice)
        cpu_utilization = min(80.0, queue_depth * 10.0)  # Simplified
        memory_utilization = min(85.0, len(self.workers) * 8.0)  # Simplified
        gpu_utilization = min(90.0, total_requests * 0.1)  # Simplified
        
        # Throughput (requests per second in last window)
        window_start = current_time - 60.0  # Last minute
        recent_requests = sum(
            s['request_count'] for s in worker_stats
            if s['last_used'] >= window_start
        )
        throughput = recent_requests / 60.0
        
        return ScalingMetrics(
            cpu_utilization=cpu_utilization,
            memory_utilization=memory_utilization,
            gpu_utilization=gpu_utilization,
            queue_depth=queue_depth,
            response_time_p95=avg_response_time * 1000,  # Convert to ms
            throughput=throughput,
            error_rate=error_rate,
            timestamp=current_time
        )
    
    def _should_scale_up(self, metrics: ScalingMetrics) -> bool:
        """Determine if we should scale up."""
        if len(self.workers) >= self.policy.max_instances:
            return False
        
        # Check cooldown period
        if time.time() - self.last_scale_action < self.policy.cooldown_period:
            return False
        
        # Scale up conditions
        conditions = [
            metrics.cpu_utilization > self.policy.target_cpu_utilization * self.policy.scale_up_threshold,
            metrics.memory_utilization > self.policy.target_memory_utilization * self.policy.scale_up_threshold,
            metrics.response_time_p95 > self.policy.target_response_time_ms * self.policy.scale_up_threshold,
            metrics.queue_depth > len(self.workers) * 2,  # Queue building up
            metrics.error_rate > 0.05  # High error rate
        ]
        
        # Scale up if any critical condition is met
        return any(conditions)
    
    def _should_scale_down(self, metrics: ScalingMetrics) -> bool:
        """Determine if we should scale down."""
        if len(self.workers) <= self.policy.min_instances:
            return False
        
        # Check cooldown period
        if time.time() - self.last_scale_action < self.policy.cooldown_period:
            return False
        
        # Scale down conditions (all must be true)
        conditions = [
            metrics.cpu_utilization < self.policy.target_cpu_utilization * self.policy.scale_down_threshold,
            metrics.memory_utilization < self.policy.target_memory_utilization * self.policy.scale_down_threshold,
            metrics.response_time_p95 < self.policy.target_response_time_ms * self.policy.scale_down_threshold,
            metrics.queue_depth < len(self.workers) * 0.5,  # Low queue
            metrics.error_rate < 0.01  # Low error rate
        ]
        
        # Scale down only if all conditions are met
        return all(conditions)
    
    def _monitoring_loop(self):
        """Main monitoring and scaling loop."""
        while not self._stop_monitoring.is_set():
            try:
                # Collect metrics
                metrics = self._collect_metrics()
                self.metrics_history.append(metrics)
                
                # Trim metrics history
                cutoff_time = time.time() - self.policy.metrics_window
                self.metrics_history = [
                    m for m in self.metrics_history 
                    if m.timestamp >= cutoff_time
                ]
                
                # Make scaling decisions
                if self._should_scale_up(metrics):
                    self._add_worker()
                    self.last_scale_action = time.time()
                    self._logger.info(f"Scaled up to {len(self.workers)} workers")
                    
                elif self._should_scale_down(metrics):
                    # Remove least recently used worker
                    worker_to_remove = min(
                        self.workers.values(),
                        key=lambda w: w.last_used
                    )
                    self._remove_worker(worker_to_remove.instance_id)
                    self.last_scale_action = time.time()
                    self._logger.info(f"Scaled down to {len(self.workers)} workers")
                
                # Log current status
                if len(self.metrics_history) % 12 == 0:  # Every 12 cycles (1 minute)
                    self._logger.info(
                        f"AutoScaler status: {len(self.workers)} workers, "
                        f"queue: {metrics.queue_depth}, "
                        f"CPU: {metrics.cpu_utilization:.1f}%, "
                        f"response: {metrics.response_time_p95:.1f}ms"
                    )
                
            except Exception as e:
                self._logger.error(f"Monitoring loop error: {e}")
            
            # Wait before next check
            self._stop_monitoring.wait(5.0)  # Check every 5 seconds
    
    def get_status(self) -> Dict[str, Any]:
        """Get current auto-scaler status."""
        if self.metrics_history:
            latest_metrics = self.metrics_history[-1]
        else:
            latest_metrics = self._collect_metrics()
        
        worker_stats = {
            worker_id: worker.get_stats()
            for worker_id, worker in self.workers.items()
        }
        
        return {
            'worker_count': len(self.workers),
            'policy': {
                'min_instances': self.policy.min_instances,
                'max_instances': self.policy.max_instances,
                'target_cpu': self.policy.target_cpu_utilization,
                'target_memory': self.policy.target_memory_utilization
            },
            'current_metrics': {
                'cpu_utilization': latest_metrics.cpu_utilization,
                'memory_utilization': latest_metrics.memory_utilization,
                'queue_depth': latest_metrics.queue_depth,
                'response_time_p95': latest_metrics.response_time_p95,
                'throughput': latest_metrics.throughput,
                'error_rate': latest_metrics.error_rate
            },
            'workers': worker_stats,
            'last_scale_action': self.last_scale_action
        }
    
    def shutdown(self):
        """Shutdown the auto-scaler."""
        self.stop_monitoring()
        
        # Stop all workers
        for worker in self.workers.values():
            worker.stop()
        
        self.workers.clear()
        self._logger.info("Auto-scaler shutdown complete")

import time
import threading
import queue
from typing import List, Callable, Any, Optional
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
import psutil

class AutoScaler:
    """Auto-scaling system for HDC operations"""
    
    def __init__(self, min_workers: int = 2, max_workers: int = None, target_cpu_percent: float = 75.0):
        self.min_workers = min_workers
        self.max_workers = max_workers or mp.cpu_count()
        self.target_cpu_percent = target_cpu_percent
        self.current_workers = min_workers
        
        self.thread_pool = ThreadPoolExecutor(max_workers=self.current_workers)
        self.process_pool = ProcessPoolExecutor(max_workers=self.current_workers)
        
        self.task_queue = queue.Queue()
        self.metrics = {
            'completed_tasks': 0,
            'failed_tasks': 0,
            'avg_task_time': 0.0,
            'queue_size': 0
        }
        
        self.monitor_thread = threading.Thread(target=self._monitor_performance, daemon=True)
        self.monitor_thread.start()
    
    def _monitor_performance(self):
        """Monitor system performance and adjust scaling"""
        while True:
            try:
                cpu_percent = psutil.cpu_percent(interval=1)
                memory_percent = psutil.virtual_memory().percent
                queue_size = self.task_queue.qsize()
                
                # Scale up conditions
                if (cpu_percent > self.target_cpu_percent and 
                    queue_size > self.current_workers and 
                    self.current_workers < self.max_workers):
                    self._scale_up()
                
                # Scale down conditions
                elif (cpu_percent < self.target_cpu_percent * 0.5 and 
                      queue_size < self.current_workers // 2 and 
                      self.current_workers > self.min_workers):
                    self._scale_down()
                
                self.metrics['queue_size'] = queue_size
                
                time.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                print(f"Monitoring error: {e}")
                time.sleep(10)
    
    def _scale_up(self):
        """Increase worker count"""
        new_workers = min(self.current_workers + 2, self.max_workers)
        if new_workers > self.current_workers:
            self.current_workers = new_workers
            self._restart_pools()
            print(f"Scaled up to {self.current_workers} workers")
    
    def _scale_down(self):
        """Decrease worker count"""
        new_workers = max(self.current_workers - 1, self.min_workers)
        if new_workers < self.current_workers:
            self.current_workers = new_workers
            self._restart_pools()
            print(f"Scaled down to {self.current_workers} workers")
    
    def _restart_pools(self):
        """Restart thread and process pools with new worker count"""
        # Shutdown old pools
        self.thread_pool.shutdown(wait=False)
        self.process_pool.shutdown(wait=False)
        
        # Create new pools
        self.thread_pool = ThreadPoolExecutor(max_workers=self.current_workers)
        self.process_pool = ProcessPoolExecutor(max_workers=self.current_workers)
    
    def submit_io_task(self, func: Callable, *args, **kwargs):
        """Submit I/O bound task to thread pool"""
        start_time = time.time()
        future = self.thread_pool.submit(func, *args, **kwargs)
        
        def on_complete(fut):
            duration = time.time() - start_time
            try:
                result = fut.result()
                self.metrics['completed_tasks'] += 1
                self._update_avg_time(duration)
            except Exception as e:
                self.metrics['failed_tasks'] += 1
                print(f"Task failed: {e}")
        
        future.add_done_callback(on_complete)
        return future
    
    def submit_cpu_task(self, func: Callable, *args, **kwargs):
        """Submit CPU bound task to process pool"""
        start_time = time.time()
        future = self.process_pool.submit(func, *args, **kwargs)
        
        def on_complete(fut):
            duration = time.time() - start_time
            try:
                result = fut.result()
                self.metrics['completed_tasks'] += 1
                self._update_avg_time(duration)
            except Exception as e:
                self.metrics['failed_tasks'] += 1
                print(f"Task failed: {e}")
        
        future.add_done_callback(on_complete)
        return future
    
    def _update_avg_time(self, new_time: float):
        """Update average task time"""
        total_tasks = self.metrics['completed_tasks']
        if total_tasks == 1:
            self.metrics['avg_task_time'] = new_time
        else:
            # Moving average
            alpha = 0.1  # Smoothing factor
            self.metrics['avg_task_time'] = (
                alpha * new_time + 
                (1 - alpha) * self.metrics['avg_task_time']
            )
    
    def get_metrics(self) -> dict:
        """Get current performance metrics"""
        return self.metrics.copy()
    
    def shutdown(self):
        """Shutdown all pools"""
        self.thread_pool.shutdown(wait=True)
        self.process_pool.shutdown(wait=True)

class LoadBalancer:
    """Load balancer for distributing HDC operations"""
    
    def __init__(self, workers: List[Any]):
        self.workers = workers
        self.current_index = 0
        self.worker_loads = [0] * len(workers)
        self.lock = threading.Lock()
    
    def get_least_loaded_worker(self):
        """Get worker with least current load"""
        with self.lock:
            min_load_index = min(range(len(self.worker_loads)), key=lambda i: self.worker_loads[i])
            self.worker_loads[min_load_index] += 1
            return self.workers[min_load_index], min_load_index
    
    def release_worker(self, worker_index: int):
        """Release worker after task completion"""
        with self.lock:
            if 0 <= worker_index < len(self.worker_loads):
                self.worker_loads[worker_index] = max(0, self.worker_loads[worker_index] - 1)
    
    def get_round_robin_worker(self):
        """Get next worker in round-robin fashion"""
        with self.lock:
            worker = self.workers[self.current_index]
            worker_index = self.current_index
            self.current_index = (self.current_index + 1) % len(self.workers)
            return worker, worker_index

# Global auto-scaler
_auto_scaler = None

def get_auto_scaler(**kwargs) -> AutoScaler:
    """Get global auto-scaler instance"""
    global _auto_scaler
    if _auto_scaler is None:
        _auto_scaler = AutoScaler(**kwargs)
    return _auto_scaler
