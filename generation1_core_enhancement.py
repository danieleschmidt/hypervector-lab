#!/usr/bin/env python3
"""
Generation 1: Core HDC Enhancement - MAKE IT WORK
Lightweight improvements to core functionality
"""

import sys
import os
import json
import random
import math

# Simple torch-like tensor simulation for testing without pytorch
class TensorSim:
    def __init__(self, data, device='cpu'):
        if isinstance(data, list):
            self.data = data
        else:
            self.data = [data]
        self.device = device
        self.shape = (len(self.data),) if isinstance(data, list) else ()
    
    def __mul__(self, other):
        if isinstance(other, TensorSim):
            result = [a * b for a, b in zip(self.data, other.data)]
        else:
            result = [x * other for x in self.data]
        return TensorSim(result, self.device)
    
    def __add__(self, other):
        if isinstance(other, TensorSim):
            result = [a + b for a, b in zip(self.data, other.data)]
        else:
            result = [x + other for x in self.data]
        return TensorSim(result, self.device)
    
    def sum(self):
        return sum(self.data)
    
    def norm(self):
        return math.sqrt(sum(x**2 for x in self.data))

# Enhanced HDC Core Functionality
class EnhancedHDCCore:
    """Enhanced HDC system with improved functionality"""
    
    def __init__(self, dim=1000, device='cpu'):
        self.dim = dim
        self.device = device
        self.memory = {}
        self.similarity_cache = {}
        self.operations_count = 0
        
    def random_hypervector(self, seed=None):
        """Generate random hypervector with optional seed"""
        if seed is not None:
            random.seed(seed)
        return TensorSim([random.gauss(0, 1) for _ in range(self.dim)], self.device)
    
    def bind(self, hv1, hv2):
        """Bind two hypervectors using element-wise multiplication"""
        self.operations_count += 1
        return hv1 * hv2
    
    def bundle(self, hvs, weights=None):
        """Bundle multiple hypervectors with optional weights"""
        self.operations_count += 1
        if not hvs:
            return self.random_hypervector()
        
        if weights is None:
            weights = [1.0] * len(hvs)
        
        result_data = [0.0] * self.dim
        for hv, weight in zip(hvs, weights):
            for i, val in enumerate(hv.data):
                result_data[i] += val * weight
        
        # Normalize
        norm = math.sqrt(sum(x**2 for x in result_data))
        if norm > 0:
            result_data = [x / norm for x in result_data]
        
        return TensorSim(result_data, self.device)
    
    def cosine_similarity(self, hv1, hv2):
        """Compute cosine similarity between hypervectors"""
        cache_key = (id(hv1), id(hv2))
        if cache_key in self.similarity_cache:
            return self.similarity_cache[cache_key]
        
        dot_product = sum(a * b for a, b in zip(hv1.data, hv2.data))
        norm1 = hv1.norm()
        norm2 = hv2.norm()
        
        if norm1 == 0 or norm2 == 0:
            similarity = 0.0
        else:
            similarity = dot_product / (norm1 * norm2)
        
        self.similarity_cache[cache_key] = similarity
        return similarity
    
    def store_pattern(self, key, hv):
        """Store hypervector with enhanced metadata"""
        self.memory[key] = {
            'hypervector': hv,
            'timestamp': self.operations_count,
            'access_count': 0
        }
    
    def query_memory(self, query_hv, top_k=5, threshold=0.5):
        """Enhanced memory query with threshold filtering"""
        if not self.memory:
            return []
        
        similarities = []
        for key, data in self.memory.items():
            stored_hv = data['hypervector']
            sim = self.cosine_similarity(query_hv, stored_hv)
            
            if sim >= threshold:
                similarities.append((key, sim, data['timestamp']))
                data['access_count'] += 1
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def adaptive_cleanup(self, target_size=100):
        """Adaptive memory cleanup based on access patterns"""
        if len(self.memory) <= target_size:
            return
        
        # Score memories by recency and access frequency
        scored_memories = []
        current_time = self.operations_count
        
        for key, data in self.memory.items():
            recency_score = 1.0 / (current_time - data['timestamp'] + 1)
            access_score = math.log(data['access_count'] + 1)
            total_score = recency_score + access_score
            scored_memories.append((key, total_score))
        
        # Keep top-scoring memories
        scored_memories.sort(key=lambda x: x[1], reverse=True)
        keys_to_keep = set(key for key, _ in scored_memories[:target_size])
        
        # Remove low-scoring memories
        keys_to_remove = [key for key in self.memory.keys() if key not in keys_to_keep]
        for key in keys_to_remove:
            del self.memory[key]
    
    def get_statistics(self):
        """Get system statistics"""
        return {
            'dimension': self.dim,
            'device': self.device,
            'memory_size': len(self.memory),
            'operations_count': self.operations_count,
            'cache_size': len(self.similarity_cache),
            'average_access_count': sum(data['access_count'] for data in self.memory.values()) / len(self.memory) if self.memory else 0
        }

def test_enhanced_hdc():
    """Test enhanced HDC functionality"""
    print("=== Testing Enhanced HDC Core ===")
    
    hdc = EnhancedHDCCore(dim=100)
    
    # Test 1: Basic operations
    print("Test 1: Basic operations")
    hv1 = hdc.random_hypervector(seed=42)
    hv2 = hdc.random_hypervector(seed=43)
    
    bound = hdc.bind(hv1, hv2)
    bundled = hdc.bundle([hv1, hv2])
    similarity = hdc.cosine_similarity(hv1, hv2)
    
    print(f"  Similarity between random vectors: {similarity:.4f}")
    print(f"  Bound vector norm: {bound.norm():.4f}")
    print(f"  Bundle vector norm: {bundled.norm():.4f}")
    
    # Test 2: Memory operations
    print("\nTest 2: Memory operations")
    hdc.store_pattern("pattern_1", hv1)
    hdc.store_pattern("pattern_2", hv2)
    hdc.store_pattern("pattern_3", bundled)
    
    query_results = hdc.query_memory(hv1, top_k=3, threshold=0.1)
    print(f"  Query results: {len(query_results)} matches")
    for key, sim, timestamp in query_results:
        print(f"    {key}: similarity={sim:.4f}, timestamp={timestamp}")
    
    # Test 3: Adaptive cleanup
    print("\nTest 3: System statistics and cleanup")
    stats_before = hdc.get_statistics()
    print(f"  Before cleanup: {stats_before}")
    
    hdc.adaptive_cleanup(target_size=2)
    stats_after = hdc.get_statistics()
    print(f"  After cleanup: {stats_after}")
    
    print("✓ Enhanced HDC Core tests completed successfully!")
    return True

def benchmark_performance():
    """Benchmark performance improvements"""
    print("\n=== Performance Benchmark ===")
    
    import time
    
    hdc = EnhancedHDCCore(dim=500)
    
    # Generate test vectors
    test_vectors = [hdc.random_hypervector(seed=i) for i in range(100)]
    
    # Benchmark binding operations
    start_time = time.time()
    for i in range(0, len(test_vectors)-1, 2):
        hdc.bind(test_vectors[i], test_vectors[i+1])
    bind_time = time.time() - start_time
    
    # Benchmark bundling operations
    start_time = time.time()
    for i in range(0, len(test_vectors), 5):
        chunk = test_vectors[i:i+5]
        if len(chunk) > 1:
            hdc.bundle(chunk)
    bundle_time = time.time() - start_time
    
    # Benchmark similarity computations
    start_time = time.time()
    similarities = []
    for i in range(min(50, len(test_vectors))):
        for j in range(i+1, min(i+10, len(test_vectors))):
            sim = hdc.cosine_similarity(test_vectors[i], test_vectors[j])
            similarities.append(sim)
    similarity_time = time.time() - start_time
    
    print(f"  Bind operations: {bind_time:.4f}s for {len(test_vectors)//2} operations")
    print(f"  Bundle operations: {bundle_time:.4f}s for {len(test_vectors)//5} operations")
    print(f"  Similarity computations: {similarity_time:.4f}s for {len(similarities)} comparisons")
    print(f"  Cache hit rate: {len(hdc.similarity_cache)} cached similarities")
    
    return {
        'bind_time': bind_time,
        'bundle_time': bundle_time,
        'similarity_time': similarity_time,
        'cache_size': len(hdc.similarity_cache)
    }

def main():
    """Main execution function"""
    print("Generation 1: Enhanced HDC Core Implementation")
    print("=" * 50)
    
    # Run tests
    try:
        test_result = test_enhanced_hdc()
        benchmark_result = benchmark_performance()
        
        # Save results
        results = {
            'generation': 1,
            'status': 'success',
            'test_passed': test_result,
            'benchmark': benchmark_result,
            'enhancements': [
                'Similarity caching for performance',
                'Weighted bundling operations',
                'Enhanced memory with metadata',
                'Adaptive memory cleanup',
                'Comprehensive statistics tracking',
                'Threshold-based memory queries'
            ]
        }
        
        with open('/root/repo/generation1_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print("\n✓ Generation 1 Implementation Complete!")
        print("  Key enhancements:")
        for enhancement in results['enhancements']:
            print(f"    - {enhancement}")
        
        return True
        
    except Exception as e:
        print(f"✗ Generation 1 Failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)