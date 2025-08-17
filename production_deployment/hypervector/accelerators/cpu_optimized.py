"""CPU-optimized implementations for HDC operations."""

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
from typing import List, Union, Optional
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

from ..core.hypervector import HyperVector
from ..utils.logging import get_logger
from ..benchmark.profiler import profile_operation

logger = get_logger(__name__)


class CPUAccelerator:
    """CPU-optimized implementations for HDC operations."""
    
    def __init__(self, num_threads: Optional[int] = None):
        """Initialize CPU accelerator.
        
        Args:
            num_threads: Number of threads to use (defaults to CPU count)
        """
        self.num_threads = num_threads or mp.cpu_count()
        logger.info(f"Initialized CPU accelerator with {self.num_threads} threads")
    
    @profile_operation("cpu_vectorized_bind")
    def vectorized_bind(self, hv1: HyperVector, hv2: HyperVector) -> HyperVector:
        """Optimized element-wise multiplication (binding).
        
        Args:
            hv1: First hypervector
            hv2: Second hypervector
            
        Returns:
            Bound hypervector
        """
        # Use torch's optimized operations
        result_data = torch.mul(hv1.data, hv2.data)
        return HyperVector(result_data, device=hv1.data.device.type, mode=hv1.mode)
    
    @profile_operation("cpu_vectorized_bundle")
    def vectorized_bundle(self, hvs: List[HyperVector], normalize: bool = True) -> HyperVector:
        """Optimized bundling using vectorized operations.
        
        Args:
            hvs: List of hypervectors to bundle
            normalize: Whether to normalize the result
            
        Returns:
            Bundled hypervector
        """
        if not hvs:
            raise ValueError("Cannot bundle empty list")
        
        # Stack all vectors for efficient computation
        stacked_data = torch.stack([hv.data for hv in hvs], dim=0)
        
        # Sum along the first dimension
        result_data = torch.sum(stacked_data, dim=0)
        
        # Normalize if requested
        if normalize:
            result_data = torch.nn.functional.normalize(result_data, dim=-1)
        
        return HyperVector(result_data, device=hvs[0].data.device.type, mode="dense")
    
    @profile_operation("cpu_batch_similarity")
    def batch_cosine_similarity(
        self, 
        query_hv: HyperVector, 
        hvs: List[HyperVector]
    ) -> torch.Tensor:
        """Compute cosine similarity between query and multiple vectors efficiently.
        
        Args:
            query_hv: Query hypervector
            hvs: List of hypervectors to compare against
            
        Returns:
            Tensor of similarity scores
        """
        if not hvs:
            return torch.tensor([])
        
        # Stack all vectors
        stacked_data = torch.stack([hv.data for hv in hvs], dim=0)
        
        # Normalize query and target vectors
        query_norm = torch.nn.functional.normalize(query_hv.data, dim=-1)
        targets_norm = torch.nn.functional.normalize(stacked_data, dim=-1)
        
        # Compute batch dot product
        similarities = torch.mv(targets_norm, query_norm)
        
        return similarities
    
    @profile_operation("cpu_parallel_encode")
    def parallel_encode(
        self, 
        encoder,
        inputs: List[Union[str, torch.Tensor, "np.ndarray"]],
        max_workers: Optional[int] = None
    ) -> List[HyperVector]:
        """Encode multiple inputs in parallel.
        
        Args:
            encoder: Encoder instance with encode method
            inputs: List of inputs to encode
            max_workers: Maximum number of worker threads
            
        Returns:
            List of encoded hypervectors
        """
        max_workers = max_workers or min(self.num_threads, len(inputs))
        
        if max_workers == 1 or len(inputs) == 1:
            # Single-threaded execution
            return [encoder.encode(inp) for inp in inputs]
        
        # Parallel execution
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(encoder.encode, inp) for inp in inputs]
            results = [future.result() for future in futures]
        
        return results
    
    @profile_operation("cpu_chunked_operation")
    def chunked_operation(
        self,
        operation_fn,
        data: List,
        chunk_size: int = 1000,
        combine_fn = None
    ):
        """Process large datasets in chunks to manage memory.
        
        Args:
            operation_fn: Function to apply to each chunk
            data: List of data to process
            chunk_size: Size of each chunk
            combine_fn: Function to combine chunk results (defaults to list concatenation)
            
        Returns:
            Combined results
        """
        if combine_fn is None:
            combine_fn = lambda results: [item for chunk in results for item in chunk]
        
        chunk_results = []
        
        for i in range(0, len(data), chunk_size):
            chunk = data[i:i + chunk_size]
            logger.debug(f"Processing chunk {i//chunk_size + 1}/{(len(data)-1)//chunk_size + 1}")
            
            chunk_result = operation_fn(chunk)
            chunk_results.append(chunk_result)
        
        return combine_fn(chunk_results)
    
    @profile_operation("cpu_memory_efficient_bind")
    def memory_efficient_bind(
        self, 
        hv1: HyperVector, 
        hv2: HyperVector,
        inplace: bool = False
    ) -> HyperVector:
        """Memory-efficient binding operation.
        
        Args:
            hv1: First hypervector
            hv2: Second hypervector
            inplace: Whether to modify hv1 in-place
            
        Returns:
            Bound hypervector
        """
        if inplace:
            hv1.data.mul_(hv2.data)
            return hv1
        else:
            result_data = torch.mul(hv1.data, hv2.data)
            return HyperVector(result_data, device=hv1.data.device.type, mode=hv1.mode)
    
    def optimize_for_inference(self, model) -> None:
        """Optimize model for inference on CPU.
        
        Args:
            model: Model to optimize
        """
        # Set to eval mode
        if hasattr(model, 'eval'):
            model.eval()
        
        # Disable gradients for inference
        if hasattr(model, 'requires_grad_'):
            for param in model.parameters():
                param.requires_grad_(False)
        
        logger.info("Optimized model for CPU inference")
    
    @profile_operation("cpu_quantized_operations")
    def quantized_operations(self, hv: HyperVector, bits: int = 8) -> HyperVector:
        """Quantize hypervector for reduced memory usage.
        
        Args:
            hv: Input hypervector
            bits: Number of bits for quantization
            
        Returns:
            Quantized hypervector
        """
        if bits == 8:
            # Quantize to int8 range
            min_val = torch.min(hv.data)
            max_val = torch.max(hv.data)
            
            # Scale to [0, 255] range
            scaled = (hv.data - min_val) / (max_val - min_val) * 255
            quantized = scaled.round().to(torch.uint8)
            
            # Dequantize back to float
            dequantized = quantized.to(torch.float32) / 255 * (max_val - min_val) + min_val
            
            return HyperVector(dequantized, device=hv.data.device.type, mode=hv.mode)
        
        else:
            logger.warning(f"Quantization to {bits} bits not implemented, returning original")
            return hv


class AVXOptimizedOperations:
    """AVX-optimized operations for x86 CPUs."""
    
    @staticmethod
    def is_avx_available() -> bool:
        """Check if AVX instructions are available."""
        try:
            import cpuinfo
            info = cpuinfo.get_cpu_info()
            return 'avx' in info.get('flags', [])
        except ImportError:
            logger.warning("cpuinfo package not available, cannot detect AVX support")
            return False
    
    @staticmethod
    @profile_operation("avx_dot_product")
    def avx_dot_product(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """AVX-optimized dot product (uses PyTorch's optimized BLAS)."""
        return torch.dot(a, b)
    
    @staticmethod
    @profile_operation("avx_elementwise_mul")
    def avx_elementwise_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """AVX-optimized element-wise multiplication."""
        return torch.mul(a, b)


class AdaptiveProcessor:
    """Adaptive processor that chooses optimal implementation based on data size."""
    
    def __init__(self, cpu_accelerator: CPUAccelerator):
        self.cpu_accelerator = cpu_accelerator
        self.thresholds = {
            'parallel_threshold': 100,    # Use parallel processing above this size
            'chunking_threshold': 10000,  # Use chunking above this size
            'memory_threshold': 1000000   # Use memory-efficient ops above this size
        }
    
    def adaptive_bundle(self, hvs: List[HyperVector]) -> HyperVector:
        """Adaptively choose bundling strategy based on input size."""
        num_vectors = len(hvs)
        vector_dim = hvs[0].dim if hvs else 0
        
        if num_vectors * vector_dim > self.thresholds['memory_threshold']:
            # Use chunked processing for very large datasets
            logger.debug(f"Using chunked bundling for {num_vectors} vectors")
            
            def chunk_bundle(chunk_hvs):
                return self.cpu_accelerator.vectorized_bundle(chunk_hvs, normalize=False)
            
            def combine_chunks(chunk_results):
                return self.cpu_accelerator.vectorized_bundle(chunk_results, normalize=True)
            
            return self.cpu_accelerator.chunked_operation(
                chunk_bundle, 
                hvs, 
                chunk_size=1000,
                combine_fn=combine_chunks
            )
        
        elif num_vectors > self.thresholds['parallel_threshold']:
            # Use vectorized bundling for medium datasets
            logger.debug(f"Using vectorized bundling for {num_vectors} vectors")
            return self.cpu_accelerator.vectorized_bundle(hvs)
        
        else:
            # Use simple bundling for small datasets
            logger.debug(f"Using simple bundling for {num_vectors} vectors")
            result = hvs[0]
            for hv in hvs[1:]:
                result = result + hv
            return result.normalize()
    
    def adaptive_similarity_search(
        self, 
        query: HyperVector, 
        candidates: List[HyperVector],
        top_k: int = 10
    ) -> List[tuple]:
        """Adaptively search for most similar vectors."""
        num_candidates = len(candidates)
        
        if num_candidates > self.thresholds['chunking_threshold']:
            # Use chunked processing for very large candidate sets
            logger.debug(f"Using chunked similarity search for {num_candidates} candidates")
            
            def chunk_search(chunk_candidates):
                similarities = self.cpu_accelerator.batch_cosine_similarity(query, chunk_candidates)
                return list(zip(chunk_candidates, similarities))
            
            def combine_chunks(chunk_results):
                all_results = [item for chunk in chunk_results for item in chunk]
                all_results.sort(key=lambda x: x[1], reverse=True)
                return all_results[:top_k]
            
            return self.cpu_accelerator.chunked_operation(
                chunk_search,
                candidates,
                chunk_size=5000,
                combine_fn=combine_chunks
            )
        
        else:
            # Use batch similarity for manageable datasets
            logger.debug(f"Using batch similarity search for {num_candidates} candidates")
            similarities = self.cpu_accelerator.batch_cosine_similarity(query, candidates)
            
            # Get top-k results
            top_indices = torch.topk(similarities, min(top_k, len(candidates))).indices
            results = [(candidates[i], similarities[i].item()) for i in top_indices]
            
            return results