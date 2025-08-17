"""Distributed processing for large-scale HDC operations."""

import os
import time
import threading
from typing import Dict, List, Optional, Callable, Any, Tuple, Union
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import queue
import multiprocessing as mp
import torch
import torch.distributed as dist
import torch.multiprocessing as torch_mp


@dataclass
class DistributedConfig:
    """Configuration for distributed processing."""
    backend: str = 'nccl'  # 'nccl', 'gloo', 'mpi'
    world_size: int = 1
    rank: int = 0
    master_addr: str = 'localhost'
    master_port: str = '12355'
    use_cuda: bool = True
    chunk_size: int = 1000
    overlap_communication: bool = True


class WorkerPool:
    """Manage pool of worker processes/threads."""
    
    def __init__(
        self,
        num_workers: Optional[int] = None,
        worker_type: str = 'process',  # 'process', 'thread'
        use_cuda: bool = False
    ):
        self.num_workers = num_workers or mp.cpu_count()
        self.worker_type = worker_type
        self.use_cuda = use_cuda
        
        self.executor = None
        self.is_running = False
        
    def start(self):
        """Start the worker pool."""
        if self.is_running:
            return
        
        if self.worker_type == 'process':
            # Use spawn method for CUDA compatibility
            mp_context = mp.get_context('spawn') if self.use_cuda else None
            self.executor = ProcessPoolExecutor(
                max_workers=self.num_workers,
                mp_context=mp_context
            )
        else:
            self.executor = ThreadPoolExecutor(max_workers=self.num_workers)
        
        self.is_running = True
    
    def stop(self):
        """Stop the worker pool."""
        if not self.is_running:
            return
        
        if self.executor:
            self.executor.shutdown(wait=True)
            self.executor = None
        
        self.is_running = False
    
    def submit(self, func: Callable, *args, **kwargs):
        """Submit a task to the worker pool."""
        if not self.is_running:
            self.start()
        
        return self.executor.submit(func, *args, **kwargs)
    
    def map(self, func: Callable, iterable, chunksize: int = 1):
        """Map function over iterable using worker pool."""
        if not self.is_running:
            self.start()
        
        if self.worker_type == 'process':
            # ProcessPoolExecutor doesn't have map method in all Python versions
            futures = [self.submit(func, item) for item in iterable]
            return [future.result() for future in futures]
        else:
            return list(self.executor.map(func, iterable, chunksize=chunksize))


def init_distributed_process(rank: int, world_size: int, config: DistributedConfig):
    """Initialize distributed process."""
    os.environ['MASTER_ADDR'] = config.master_addr
    os.environ['MASTER_PORT'] = config.master_port
    
    # Initialize process group
    dist.init_process_group(
        backend=config.backend,
        rank=rank,
        world_size=world_size
    )
    
    # Set device for CUDA
    if config.use_cuda and torch.cuda.is_available():
        device_id = rank % torch.cuda.device_count()
        torch.cuda.set_device(device_id)


class DistributedHDCProcessor:
    """Distributed processor for HDC operations."""
    
    def __init__(self, config: Optional[DistributedConfig] = None):
        self.config = config or DistributedConfig()
        self.is_initialized = False
        self.worker_pool = None
        
    def initialize(self):
        """Initialize distributed processing."""
        if self.is_initialized:
            return
        
        # Check if we're in a distributed environment
        if 'WORLD_SIZE' in os.environ:
            self.config.world_size = int(os.environ['WORLD_SIZE'])
            self.config.rank = int(os.environ['RANK'])
            
            # Initialize distributed processing
            init_distributed_process(
                self.config.rank, 
                self.config.world_size, 
                self.config
            )
        
        # Initialize worker pool for local parallelism
        self.worker_pool = WorkerPool(
            worker_type='thread' if self.config.use_cuda else 'process',
            use_cuda=self.config.use_cuda
        )
        
        self.is_initialized = True
    
    def distributed_bind(
        self, 
        x_list: List[torch.Tensor], 
        y_list: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """Perform distributed binding operation."""
        if not self.is_initialized:
            self.initialize()
        
        if len(x_list) != len(y_list):
            raise ValueError("Input lists must have same length")
        
        # Split work across processes
        chunks = self._split_work(list(zip(x_list, y_list)))
        
        if dist.is_initialized():
            # Distributed execution
            local_chunk = chunks[self.config.rank]
            local_results = self._process_bind_chunk(local_chunk)
            
            # Gather results from all processes
            all_results = [None] * self.config.world_size
            dist.all_gather_object(all_results, local_results)
            
            # Flatten results
            results = []
            for chunk_results in all_results:
                if chunk_results:
                    results.extend(chunk_results)
            
            return results
        else:
            # Local parallel execution
            futures = []
            for chunk in chunks:
                future = self.worker_pool.submit(self._process_bind_chunk, chunk)
                futures.append(future)
            
            results = []
            for future in as_completed(futures):
                chunk_results = future.result()
                results.extend(chunk_results)
            
            return results
    
    def distributed_bundle(
        self, 
        tensor_groups: List[List[torch.Tensor]],
        weights: Optional[List[torch.Tensor]] = None
    ) -> List[torch.Tensor]:
        """Perform distributed bundling operation."""
        if not self.is_initialized:
            self.initialize()
        
        # Prepare work items
        work_items = []
        for i, tensors in enumerate(tensor_groups):
            group_weights = weights[i] if weights else None
            work_items.append((tensors, group_weights))
        
        # Split work across processes
        chunks = self._split_work(work_items)
        
        if dist.is_initialized():
            # Distributed execution
            local_chunk = chunks[self.config.rank]
            local_results = self._process_bundle_chunk(local_chunk)
            
            # Gather results
            all_results = [None] * self.config.world_size
            dist.all_gather_object(all_results, local_results)
            
            # Flatten results
            results = []
            for chunk_results in all_results:
                if chunk_results:
                    results.extend(chunk_results)
            
            return results
        else:
            # Local parallel execution
            futures = []
            for chunk in chunks:
                future = self.worker_pool.submit(self._process_bundle_chunk, chunk)
                futures.append(future)
            
            results = []
            for future in as_completed(futures):
                chunk_results = future.result()
                results.extend(chunk_results)
            
            return results
    
    def distributed_similarity_matrix(
        self, 
        x_tensors: List[torch.Tensor], 
        y_tensors: List[torch.Tensor]
    ) -> torch.Tensor:
        """Compute similarity matrix in distributed fashion."""
        if not self.is_initialized:
            self.initialize()
        
        n_x, n_y = len(x_tensors), len(y_tensors)
        
        # Create work items (pairs of tensor indices)
        work_items = []
        for i in range(n_x):
            for j in range(n_y):
                work_items.append((i, j))
        
        # Split work across processes
        chunks = self._split_work(work_items)
        
        if dist.is_initialized():
            # Distributed execution
            local_chunk = chunks[self.config.rank]
            local_results = self._process_similarity_chunk(
                local_chunk, x_tensors, y_tensors
            )
            
            # Gather results
            all_results = [None] * self.config.world_size
            dist.all_gather_object(all_results, local_results)
            
            # Combine results into matrix
            similarity_matrix = torch.zeros(n_x, n_y)
            for chunk_results in all_results:
                if chunk_results:
                    for (i, j), sim_value in chunk_results:
                        similarity_matrix[i, j] = sim_value
            
            return similarity_matrix
        else:
            # Local parallel execution
            futures = []
            for chunk in chunks:
                future = self.worker_pool.submit(
                    self._process_similarity_chunk, 
                    chunk, x_tensors, y_tensors
                )
                futures.append(future)
            
            # Combine results
            similarity_matrix = torch.zeros(n_x, n_y)
            for future in as_completed(futures):
                chunk_results = future.result()
                for (i, j), sim_value in chunk_results:
                    similarity_matrix[i, j] = sim_value
            
            return similarity_matrix
    
    def _split_work(self, work_items: List[Any]) -> List[List[Any]]:
        """Split work items across available workers."""
        if dist.is_initialized():
            num_workers = self.config.world_size
        else:
            num_workers = self.worker_pool.num_workers if self.worker_pool else 1
        
        chunk_size = max(1, len(work_items) // num_workers)
        chunks = []
        
        for i in range(0, len(work_items), chunk_size):
            chunk = work_items[i:i + chunk_size]
            chunks.append(chunk)
        
        # Ensure we have exactly num_workers chunks
        while len(chunks) < num_workers:
            chunks.append([])
        
        return chunks[:num_workers]
    
    def _process_bind_chunk(self, chunk: List[Tuple[torch.Tensor, torch.Tensor]]) -> List[torch.Tensor]:
        """Process a chunk of binding operations."""
        results = []
        for x, y in chunk:
            # Simple element-wise multiplication for binding
            result = x * y
            results.append(result)
        return results
    
    def _process_bundle_chunk(self, chunk: List[Tuple[List[torch.Tensor], Optional[torch.Tensor]]]) -> List[torch.Tensor]:
        """Process a chunk of bundling operations."""
        results = []
        for tensors, weights in chunk:
            if not tensors:
                continue
            
            if weights is not None:
                # Weighted average
                result = torch.zeros_like(tensors[0])
                for tensor, weight in zip(tensors, weights):
                    result += weight * tensor
            else:
                # Simple average
                result = torch.stack(tensors).mean(dim=0)
            
            results.append(result)
        return results
    
    def _process_similarity_chunk(
        self, 
        chunk: List[Tuple[int, int]], 
        x_tensors: List[torch.Tensor], 
        y_tensors: List[torch.Tensor]
    ) -> List[Tuple[Tuple[int, int], float]]:
        """Process a chunk of similarity computations."""
        results = []
        for i, j in chunk:
            x = x_tensors[i]
            y = y_tensors[j]
            
            # Compute cosine similarity
            similarity = torch.cosine_similarity(x, y, dim=0)
            results.append(((i, j), similarity.item()))
        
        return results
    
    def cleanup(self):
        """Cleanup distributed resources."""
        if self.worker_pool:
            self.worker_pool.stop()
        
        if dist.is_initialized():
            dist.destroy_process_group()
        
        self.is_initialized = False


class AsyncProcessor:
    """Asynchronous processor for non-blocking HDC operations."""
    
    def __init__(self, max_concurrent: int = 10):
        self.max_concurrent = max_concurrent
        self.task_queue = queue.Queue()
        self.result_futures = {}
        self.worker_threads = []
        self.is_running = False
        
    def start(self):
        """Start async processing."""
        if self.is_running:
            return
        
        self.is_running = True
        
        # Start worker threads
        for i in range(self.max_concurrent):
            thread = threading.Thread(
                target=self._worker_loop,
                name=f"AsyncWorker-{i}",
                daemon=True
            )
            thread.start()
            self.worker_threads.append(thread)
    
    def stop(self):
        """Stop async processing."""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Add poison pills to stop workers
        for _ in self.worker_threads:
            self.task_queue.put(None)
        
        # Wait for workers to finish
        for thread in self.worker_threads:
            thread.join(timeout=5.0)
        
        self.worker_threads.clear()
    
    def submit_async(
        self, 
        func: Callable, 
        *args, 
        task_id: Optional[str] = None,
        **kwargs
    ) -> str:
        """Submit task for async processing."""
        if not self.is_running:
            self.start()
        
        if task_id is None:
            task_id = f"task_{int(time.time() * 1000000)}"
        
        # Create future for result
        future = threading.Event()
        self.result_futures[task_id] = {
            'event': future,
            'result': None,
            'error': None,
            'start_time': time.time()
        }
        
        # Add task to queue
        task = {
            'id': task_id,
            'func': func,
            'args': args,
            'kwargs': kwargs
        }
        self.task_queue.put(task)
        
        return task_id
    
    def get_result(self, task_id: str, timeout: Optional[float] = None) -> Any:
        """Get result of async task."""
        if task_id not in self.result_futures:
            raise ValueError(f"Unknown task ID: {task_id}")
        
        future_info = self.result_futures[task_id]
        
        # Wait for completion
        if not future_info['event'].wait(timeout):
            raise TimeoutError(f"Task {task_id} timed out")
        
        # Check for errors
        if future_info['error']:
            raise future_info['error']
        
        return future_info['result']
    
    def is_ready(self, task_id: str) -> bool:
        """Check if task is ready."""
        if task_id not in self.result_futures:
            return False
        
        return self.result_futures[task_id]['event'].is_set()
    
    def _worker_loop(self):
        """Main worker loop."""
        while self.is_running:
            try:
                task = self.task_queue.get(timeout=1.0)
                if task is None:  # Poison pill
                    break
                
                task_id = task['id']
                func = task['func']
                args = task['args']
                kwargs = task['kwargs']
                
                future_info = self.result_futures[task_id]
                
                try:
                    # Execute task
                    result = func(*args, **kwargs)
                    future_info['result'] = result
                    
                except Exception as e:
                    future_info['error'] = e
                
                finally:
                    # Signal completion
                    future_info['event'].set()
                    
            except queue.Empty:
                continue
            except Exception as e:
                # Log unexpected errors
                import logging
                logging.error(f"Async worker error: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get async processor statistics."""
        pending_tasks = sum(
            1 for info in self.result_futures.values()
            if not info['event'].is_set()
        )
        
        completed_tasks = len(self.result_futures) - pending_tasks
        
        return {
            'is_running': self.is_running,
            'worker_count': len(self.worker_threads),
            'queue_size': self.task_queue.qsize(),
            'pending_tasks': pending_tasks,
            'completed_tasks': completed_tasks,
            'total_tasks': len(self.result_futures)
        }


# Global instances
_global_distributed_processor = None
_global_async_processor = None

def get_distributed_processor() -> DistributedHDCProcessor:
    """Get global distributed processor."""
    global _global_distributed_processor
    if _global_distributed_processor is None:
        _global_distributed_processor = DistributedHDCProcessor()
    return _global_distributed_processor

def get_async_processor() -> AsyncProcessor:
    """Get global async processor."""
    global _global_async_processor
    if _global_async_processor is None:
        _global_async_processor = AsyncProcessor()
    return _global_async_processor