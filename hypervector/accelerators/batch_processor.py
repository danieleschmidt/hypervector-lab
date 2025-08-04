"""Batch processing utilities for efficient HDC operations."""

import torch
import numpy as np
from typing import List, Iterator, Callable, Optional, Dict, Any, Union
from dataclasses import dataclass
import queue
import threading
from concurrent.futures import Future, ThreadPoolExecutor

from ..core.hypervector import HyperVector
from ..utils.logging import get_logger
from ..benchmark.profiler import profile_operation

logger = get_logger(__name__)


@dataclass
class BatchConfig:
    """Configuration for batch processing."""
    batch_size: int = 32
    max_queue_size: int = 100
    num_workers: int = 4
    prefetch_batches: int = 2
    drop_last: bool = False
    shuffle: bool = False


class BatchProcessor:
    """Efficient batch processing for HDC operations."""
    
    def __init__(self, config: Optional[BatchConfig] = None):
        """Initialize batch processor.
        
        Args:
            config: Batch processing configuration
        """
        self.config = config or BatchConfig()
        self.logger = get_logger(f"{__name__}.BatchProcessor")
    
    def create_batches(
        self, 
        data: List[Any], 
        batch_size: Optional[int] = None,
        drop_last: Optional[bool] = None
    ) -> Iterator[List[Any]]:
        """Create batches from data.
        
        Args:
            data: Input data list
            batch_size: Batch size (uses config default if None)
            drop_last: Whether to drop last incomplete batch
            
        Yields:
            Batches of data
        """
        batch_size = batch_size or self.config.batch_size
        drop_last = drop_last if drop_last is not None else self.config.drop_last
        
        if self.config.shuffle:
            indices = torch.randperm(len(data)).tolist()
            data = [data[i] for i in indices]
        
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            
            if drop_last and len(batch) < batch_size:
                break
            
            yield batch
    
    @profile_operation("batch_encode")
    def batch_encode(
        self, 
        encoder,
        inputs: List[Union[str, torch.Tensor, np.ndarray]],
        batch_size: Optional[int] = None
    ) -> List[HyperVector]:
        """Encode inputs in batches.
        
        Args:
            encoder: Encoder with encode method
            inputs: List of inputs to encode
            batch_size: Batch size for processing
            
        Returns:
            List of encoded hypervectors
        """
        batch_size = batch_size or self.config.batch_size
        results = []
        
        for batch in self.create_batches(inputs, batch_size):
            self.logger.debug(f"Encoding batch of size {len(batch)}")
            
            # Process batch
            if hasattr(encoder, 'batch_encode'):
                # Use encoder's batch method if available
                batch_results = encoder.batch_encode(batch)
            else:
                # Fallback to individual encoding
                batch_results = [encoder.encode(item) for item in batch]
            
            results.extend(batch_results)
        
        return results
    
    @profile_operation("batch_similarity")
    def batch_similarity(
        self,
        query_hvs: List[HyperVector],
        candidate_hvs: List[HyperVector],
        batch_size: Optional[int] = None
    ) -> torch.Tensor:
        """Compute similarities between queries and candidates in batches.
        
        Args:
            query_hvs: List of query hypervectors
            candidate_hvs: List of candidate hypervectors
            batch_size: Batch size for processing
            
        Returns:
            Similarity matrix of shape (len(query_hvs), len(candidate_hvs))
        """
        batch_size = batch_size or self.config.batch_size
        num_queries = len(query_hvs)
        num_candidates = len(candidate_hvs)
        
        # Initialize result matrix
        similarity_matrix = torch.zeros(num_queries, num_candidates)
        
        # Process query batches
        for q_start in range(0, num_queries, batch_size):
            q_end = min(q_start + batch_size, num_queries)
            query_batch = query_hvs[q_start:q_end]
            
            # Stack query batch
            query_data = torch.stack([hv.data for hv in query_batch])
            query_norm = torch.nn.functional.normalize(query_data, dim=-1)
            
            # Process candidate batches
            for c_start in range(0, num_candidates, batch_size):
                c_end = min(c_start + batch_size, num_candidates)
                candidate_batch = candidate_hvs[c_start:c_end]
                
                # Stack candidate batch
                candidate_data = torch.stack([hv.data for hv in candidate_batch])
                candidate_norm = torch.nn.functional.normalize(candidate_data, dim=-1)
                
                # Compute batch similarities
                batch_similarities = torch.mm(query_norm, candidate_norm.t())
                
                # Store in result matrix
                similarity_matrix[q_start:q_end, c_start:c_end] = batch_similarities
        
        return similarity_matrix
    
    @profile_operation("batch_operations")
    def batch_operation(
        self,
        operation_fn: Callable,
        data: List[Any],
        batch_size: Optional[int] = None,
        **operation_kwargs
    ) -> List[Any]:
        """Apply operation to data in batches.
        
        Args:
            operation_fn: Function to apply to each batch
            data: Input data
            batch_size: Batch size
            **operation_kwargs: Additional arguments for operation_fn
            
        Returns:
            Results from all batches
        """
        batch_size = batch_size or self.config.batch_size
        results = []
        
        for batch in self.create_batches(data, batch_size):
            batch_result = operation_fn(batch, **operation_kwargs)
            
            if isinstance(batch_result, list):
                results.extend(batch_result)
            else:
                results.append(batch_result)
        
        return results


class AsyncBatchProcessor:
    """Asynchronous batch processor with prefetching."""
    
    def __init__(self, config: Optional[BatchConfig] = None):
        """Initialize async batch processor.
        
        Args:
            config: Batch processing configuration
        """
        self.config = config or BatchConfig()
        self.logger = get_logger(f"{__name__}.AsyncBatchProcessor")
        
        # Threading components
        self.batch_queue = queue.Queue(maxsize=self.config.max_queue_size)
        self.result_queue = queue.Queue()
        self.executor = ThreadPoolExecutor(max_workers=self.config.num_workers)
        self.shutdown_event = threading.Event()
    
    def start_processing(
        self, 
        operation_fn: Callable,
        data: List[Any],
        **operation_kwargs
    ) -> Iterator[Any]:
        """Start asynchronous batch processing.
        
        Args:
            operation_fn: Function to apply to each batch
            data: Input data
            **operation_kwargs: Additional arguments for operation_fn
            
        Yields:
            Results as they become available
        """
        # Start producer thread
        producer_thread = threading.Thread(
            target=self._produce_batches,
            args=(data,)
        )
        producer_thread.start()
        
        # Submit processing tasks
        futures = []
        
        try:
            while True:
                try:
                    batch = self.batch_queue.get(timeout=1.0)
                    if batch is None:  # Sentinel value for end
                        break
                    
                    future = self.executor.submit(operation_fn, batch, **operation_kwargs)
                    futures.append(future)
                    
                except queue.Empty:
                    if not producer_thread.is_alive():
                        break
                    continue
            
            # Yield results as they complete
            for future in futures:
                result = future.result()
                if isinstance(result, list):
                    for item in result:
                        yield item
                else:
                    yield result
        
        finally:
            producer_thread.join()
            self.shutdown()
    
    def _produce_batches(self, data: List[Any]) -> None:
        """Produce batches in background thread."""
        batch_processor = BatchProcessor(self.config)
        
        try:
            for batch in batch_processor.create_batches(data):
                if self.shutdown_event.is_set():
                    break
                
                self.batch_queue.put(batch)
            
            # Send sentinel to indicate end
            self.batch_queue.put(None)
            
        except Exception as e:
            self.logger.error(f"Error in batch producer: {e}")
            self.batch_queue.put(None)
    
    def shutdown(self) -> None:
        """Shutdown async processor."""
        self.shutdown_event.set()
        self.executor.shutdown(wait=True)


class StreamingProcessor:
    """Streaming processor for real-time HDC operations."""
    
    def __init__(self, buffer_size: int = 1000):
        """Initialize streaming processor.
        
        Args:
            buffer_size: Size of internal buffer
        """
        self.buffer_size = buffer_size
        self.buffer = []
        self.logger = get_logger(f"{__name__}.StreamingProcessor")
    
    def add_item(self, item: Any) -> Optional[List[Any]]:
        """Add item to stream and return batch if ready.
        
        Args:
            item: Item to add to stream
            
        Returns:
            Batch if buffer is full, None otherwise
        """
        self.buffer.append(item)
        
        if len(self.buffer) >= self.buffer_size:
            batch = self.buffer.copy()
            self.buffer.clear()
            return batch
        
        return None
    
    def flush(self) -> List[Any]:
        """Flush remaining items in buffer.
        
        Returns:
            Remaining items in buffer
        """
        remaining = self.buffer.copy()
        self.buffer.clear()
        return remaining
    
    def process_stream(
        self,
        data_stream: Iterator[Any],
        operation_fn: Callable,
        **operation_kwargs
    ) -> Iterator[Any]:
        """Process streaming data.
        
        Args:
            data_stream: Iterator of input data
            operation_fn: Function to apply to batches
            **operation_kwargs: Additional arguments for operation_fn
            
        Yields:
            Processed results
        """
        for item in data_stream:
            batch = self.add_item(item)
            
            if batch is not None:
                results = operation_fn(batch, **operation_kwargs)
                
                if isinstance(results, list):
                    for result in results:
                        yield result
                else:
                    yield results
        
        # Process remaining items
        remaining = self.flush()
        if remaining:
            results = operation_fn(remaining, **operation_kwargs)
            
            if isinstance(results, list):
                for result in results:
                    yield result
            else:
                yield results


class MemoryEfficientBatcher:
    """Memory-efficient batcher that minimizes memory usage."""
    
    def __init__(self, max_memory_mb: float = 1000.0):
        """Initialize memory-efficient batcher.
        
        Args:
            max_memory_mb: Maximum memory usage in MB
        """
        self.max_memory_mb = max_memory_mb
        self.logger = get_logger(f"{__name__}.MemoryEfficientBatcher")
    
    def estimate_memory_usage(self, item: Any) -> float:
        """Estimate memory usage of an item in MB.
        
        Args:
            item: Item to estimate
            
        Returns:
            Estimated memory usage in MB
        """
        if isinstance(item, torch.Tensor):
            return item.numel() * item.element_size() / (1024**2)
        elif isinstance(item, np.ndarray):
            return item.nbytes / (1024**2)
        elif isinstance(item, HyperVector):
            return self.estimate_memory_usage(item.data)
        elif isinstance(item, str):
            return len(item.encode('utf-8')) / (1024**2)
        else:
            # Rough estimate for other types
            return 0.001  # 1KB
    
    def create_memory_aware_batches(
        self, 
        data: List[Any]
    ) -> Iterator[List[Any]]:
        """Create batches based on memory constraints.
        
        Args:
            data: Input data
            
        Yields:
            Memory-constrained batches
        """
        current_batch = []
        current_memory = 0.0
        
        for item in data:
            item_memory = self.estimate_memory_usage(item)
            
            if current_memory + item_memory > self.max_memory_mb and current_batch:
                # Yield current batch if adding item would exceed memory limit
                self.logger.debug(
                    f"Yielding batch of {len(current_batch)} items "
                    f"({current_memory:.1f}MB)"
                )
                yield current_batch
                current_batch = [item]
                current_memory = item_memory
            else:
                current_batch.append(item)
                current_memory += item_memory
        
        # Yield remaining batch
        if current_batch:
            self.logger.debug(
                f"Yielding final batch of {len(current_batch)} items "
                f"({current_memory:.1f}MB)"
            )
            yield current_batch