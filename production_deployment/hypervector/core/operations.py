"""Core HDC operations: bind, bundle, permute, and similarity metrics."""

import torch
from typing import List, Union
from .hypervector import HyperVector


def bind(hv1: HyperVector, hv2: HyperVector) -> HyperVector:
    """
    Bind two hypervectors using element-wise multiplication.
    
    Binding creates a dissimilar vector from two similar inputs,
    and is approximately its own inverse.
    """
    return hv1 * hv2


def bundle(hvs: List[HyperVector], normalize: bool = True) -> HyperVector:
    """
    Bundle multiple hypervectors using element-wise addition.
    
    Bundling creates a vector similar to all inputs.
    """
    if not hvs:
        raise ValueError("Cannot bundle empty list of hypervectors")
    
    result = hvs[0]
    for hv in hvs[1:]:
        result = result + hv
    
    if normalize:
        result = result.normalize()
        
    return result


def permute(hv: HyperVector, shift: int = 1) -> HyperVector:
    """
    Permute hypervector by circular shift.
    
    Permutation creates a dissimilar vector that preserves
    distances between vectors.
    """
    if shift == 0:
        return hv
        
    shifted_data = torch.roll(hv.data, shifts=shift, dims=-1)
    return HyperVector(shifted_data, mode=hv.mode)


def cosine_similarity(hv1: HyperVector, hv2: HyperVector) -> torch.Tensor:
    """Compute cosine similarity between two hypervectors."""
    return hv1.cosine_similarity(hv2)


def hamming_distance(hv1: HyperVector, hv2: HyperVector) -> torch.Tensor:
    """Compute Hamming distance between two hypervectors."""
    return hv1.hamming_distance(hv2)


def cleanup_memory(
    query: HyperVector, 
    memory: List[HyperVector], 
    similarity_fn: str = "cosine"
) -> tuple[HyperVector, float]:
    """
    Find the most similar vector in memory to the query.
    
    Args:
        query: Query hypervector
        memory: List of stored hypervectors
        similarity_fn: Similarity function ('cosine' or 'hamming')
        
    Returns:
        Tuple of (most_similar_vector, similarity_score)
    """
    if not memory:
        raise ValueError("Memory cannot be empty")
    
    best_similarity = float('-inf') if similarity_fn == "cosine" else float('inf')
    best_match = memory[0]
    
    for hv in memory:
        if similarity_fn == "cosine":
            sim = cosine_similarity(query, hv)
            if sim > best_similarity:
                best_similarity = sim
                best_match = hv
        elif similarity_fn == "hamming":
            dist = hamming_distance(query, hv)
            if dist < best_similarity:
                best_similarity = dist
                best_match = hv
        else:
            raise ValueError(f"Unknown similarity function: {similarity_fn}")
    
    return best_match, best_similarity.item()


def majority_vote(hvs: List[HyperVector], threshold: float = 0.5) -> HyperVector:
    """
    Perform majority voting on binary/ternary hypervectors.
    
    Each dimension is set to the value that appears most frequently
    across all input vectors.
    """
    if not hvs:
        raise ValueError("Cannot vote on empty list")
    
    # Stack all vectors
    stacked = torch.stack([hv.data for hv in hvs], dim=0)
    
    # For binary vectors, use sign of sum
    if all(hv.mode == "binary" for hv in hvs):
        result_data = torch.sign(torch.sum(stacked, dim=0))
        return HyperVector(result_data, mode="binary")
    
    # For ternary vectors, more complex voting
    elif all(hv.mode == "ternary" for hv in hvs):
        # Count votes for each value (-1, 0, 1)
        neg_ones = torch.sum(stacked == -1, dim=0)
        zeros = torch.sum(stacked == 0, dim=0)
        pos_ones = torch.sum(stacked == 1, dim=0)
        
        # Find maximum vote count
        vote_counts = torch.stack([neg_ones, zeros, pos_ones], dim=0)
        max_votes = torch.argmax(vote_counts, dim=0)
        
        # Map back to values
        result_data = torch.where(max_votes == 0, -1.0,
                                torch.where(max_votes == 1, 0.0, 1.0))
        return HyperVector(result_data, mode="ternary")
    
    else:
        # For dense vectors, use thresholded average
        avg = torch.mean(stacked, dim=0)
        binary_data = torch.where(avg > threshold, 1.0, -1.0)
        return HyperVector(binary_data, mode="binary")

# High-performance operation implementations
import torch
import torch.nn.functional as F
from typing import List, Optional
from functools import lru_cache
import threading
from concurrent.futures import ThreadPoolExecutor

class OptimizedOperations:
    """High-performance implementations of HDC operations"""
    
    _cache = {}
    _cache_lock = threading.Lock()
    _thread_pool = ThreadPoolExecutor(max_workers=4)
    
    @staticmethod
    @lru_cache(maxsize=1024)
    def cached_bind(hv1_hash: int, hv2_hash: int, dim: int, device: str):
        """Cached binding operation for frequently used vectors"""
        return torch.randn(dim, device=device)  # Placeholder for actual cached result
    
    @staticmethod 
    def optimized_bind(hv1, hv2):
        """Optimized binding using element-wise operations"""
        if hv1.data.device != hv2.data.device:
            hv2 = hv2.to(str(hv1.data.device))
        
        # Use optimized kernels for different vector types
        if hv1.mode == "binary" and hv2.mode == "binary":
            # XOR for binary vectors (fastest)
            result_data = torch.where(hv1.data == hv2.data, 1.0, -1.0)
        else:
            # Element-wise multiplication (general case)
            result_data = hv1.data * hv2.data
            
        return HyperVector(result_data, mode=hv1.mode)
    
    @staticmethod
    def optimized_bundle(hvs: List, normalize: bool = True, batch_size: int = 32):
        """Optimized bundling with batching for large collections"""
        if not hvs:
            raise ValueError("Cannot bundle empty list")
        
        device = hvs[0].data.device
        dim = hvs[0].dim
        
        # Process in batches to manage memory
        result = torch.zeros(dim, device=device, dtype=torch.float32)
        
        for i in range(0, len(hvs), batch_size):
            batch = hvs[i:i + batch_size]
            batch_data = torch.stack([hv.data for hv in batch])
            batch_sum = torch.sum(batch_data, dim=0)
            result += batch_sum
        
        if normalize and len(hvs) > 1:
            result = F.normalize(result, dim=-1)
            
        return HyperVector(result)
    
    @staticmethod
    def parallel_similarity_search(query_hv, memory_hvs: List, top_k: int = 5):
        """Parallel similarity computation for large memory searches"""
        if not memory_hvs:
            return []
        
        # Batch similarity computation
        memory_data = torch.stack([hv.data for hv in memory_hvs])
        query_data = query_hv.data.unsqueeze(0)
        
        # Efficient batched cosine similarity
        similarities = F.cosine_similarity(query_data, memory_data, dim=1)
        
        # Get top-k results
        top_k = min(top_k, len(memory_hvs))
        top_scores, top_indices = torch.topk(similarities, top_k)
        
        return [(i.item(), score.item()) for i, score in zip(top_indices, top_scores)]
    
    @staticmethod
    def memory_efficient_permutation(hv, shift: int):
        """Memory-efficient permutation using torch.roll"""
        # Use built-in roll for optimal performance
        permuted_data = torch.roll(hv.data, shifts=shift, dims=0)
        return HyperVector(permuted_data, mode=hv.mode)
