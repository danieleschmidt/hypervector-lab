
import pickle
import hashlib
import os
import threading
from typing import Dict, Any, Optional
from functools import wraps
import time

class HyperVectorCache:
    """Intelligent caching system for hypervectors and computations"""
    
    def __init__(self, max_memory_mb: int = 512, disk_cache_dir: str = ".hvc_cache"):
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.memory_cache: Dict[str, Any] = {}
        self.cache_stats = {"hits": 0, "misses": 0, "evictions": 0}
        self.access_times: Dict[str, float] = {}
        self.cache_lock = threading.RLock()
        self.disk_cache_dir = disk_cache_dir
        
        if not os.path.exists(disk_cache_dir):
            os.makedirs(disk_cache_dir)
    
    def _get_cache_key(self, *args, **kwargs) -> str:
        """Generate cache key from arguments"""
        key_data = str(args) + str(sorted(kwargs.items()))
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _estimate_size(self, obj) -> int:
        """Estimate object size in bytes"""
        try:
            return len(pickle.dumps(obj))
        except:
            return 1024  # Default estimate
    
    def _evict_lru(self, required_space: int):
        """Evict least recently used items"""
        current_size = sum(self._estimate_size(v) for v in self.memory_cache.values())
        
        if current_size + required_space <= self.max_memory_bytes:
            return
        
        # Sort by access time (LRU first)
        sorted_keys = sorted(self.access_times.items(), key=lambda x: x[1])
        
        for key, _ in sorted_keys:
            if key in self.memory_cache:
                # Move to disk cache before evicting
                self._save_to_disk(key, self.memory_cache[key])
                del self.memory_cache[key]
                del self.access_times[key]
                self.cache_stats["evictions"] += 1
                
                current_size = sum(self._estimate_size(v) for v in self.memory_cache.values())
                if current_size + required_space <= self.max_memory_bytes:
                    break
    
    def _save_to_disk(self, key: str, value: Any):
        """Save value to disk cache"""
        try:
            cache_file = os.path.join(self.disk_cache_dir, f"{key}.cache")
            with open(cache_file, 'wb') as f:
                pickle.dump(value, f)
        except Exception as e:
            print(f"Failed to save to disk cache: {e}")
    
    def _load_from_disk(self, key: str) -> Optional[Any]:
        """Load value from disk cache"""
        try:
            cache_file = os.path.join(self.disk_cache_dir, f"{key}.cache")
            if os.path.exists(cache_file):
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            print(f"Failed to load from disk cache: {e}")
        return None
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        with self.cache_lock:
            # Check memory cache first
            if key in self.memory_cache:
                self.access_times[key] = time.time()
                self.cache_stats["hits"] += 1
                return self.memory_cache[key]
            
            # Check disk cache
            value = self._load_from_disk(key)
            if value is not None:
                # Move back to memory cache
                self.put(key, value)
                self.cache_stats["hits"] += 1
                return value
            
            self.cache_stats["misses"] += 1
            return None
    
    def put(self, key: str, value: Any):
        """Put value in cache"""
        with self.cache_lock:
            value_size = self._estimate_size(value)
            self._evict_lru(value_size)
            
            self.memory_cache[key] = value
            self.access_times[key] = time.time()
    
    def clear(self):
        """Clear all caches"""
        with self.cache_lock:
            self.memory_cache.clear()
            self.access_times.clear()
            # Clear disk cache
            for filename in os.listdir(self.disk_cache_dir):
                if filename.endswith('.cache'):
                    os.remove(os.path.join(self.disk_cache_dir, filename))
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.cache_stats["hits"] + self.cache_stats["misses"]
        hit_rate = self.cache_stats["hits"] / total_requests if total_requests > 0 else 0
        
        return {
            "hit_rate": hit_rate,
            "memory_items": len(self.memory_cache),
            "stats": self.cache_stats.copy()
        }

# Global cache instance
_global_cache = HyperVectorCache()

def cached_operation(cache_key_func=None):
    """Decorator for caching expensive operations"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            if cache_key_func:
                cache_key = cache_key_func(*args, **kwargs)
            else:
                cache_key = _global_cache._get_cache_key(func.__name__, *args, **kwargs)
            
            # Try to get from cache
            result = _global_cache.get(cache_key)
            if result is not None:
                return result
            
            # Compute and cache result
            result = func(*args, **kwargs)
            _global_cache.put(cache_key, result)
            return result
        
        return wrapper
    return decorator
