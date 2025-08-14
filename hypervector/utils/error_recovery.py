"""Error recovery and resilience utilities."""

import time
import random
import logging
from typing import Any, Callable, Optional, Type, Union, List
from functools import wraps
from dataclasses import dataclass


@dataclass
class RetryConfig:
    """Configuration for retry logic."""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    backoff_strategy: str = 'exponential'  # 'linear', 'exponential', 'constant'


class CircuitBreaker:
    """Circuit breaker pattern for fault tolerance."""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exceptions: tuple = (Exception,)
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exceptions = expected_exceptions
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'CLOSED'  # CLOSED, OPEN, HALF_OPEN
        
        self._logger = logging.getLogger(__name__)
    
    def __call__(self, func):
        """Decorator to apply circuit breaker."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            return self._call_with_circuit_breaker(func, *args, **kwargs)
        return wrapper
    
    def _call_with_circuit_breaker(self, func, *args, **kwargs):
        """Execute function with circuit breaker logic."""
        if self.state == 'OPEN':
            if self._should_attempt_reset():
                self.state = 'HALF_OPEN'
                self._logger.info("Circuit breaker entering HALF_OPEN state")
            else:
                raise CircuitBreakerOpenError("Circuit breaker is OPEN")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
            
        except self.expected_exceptions as e:
            self._on_failure()
            raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if we should attempt to reset the circuit breaker."""
        return (
            self.last_failure_time is not None and
            time.time() - self.last_failure_time >= self.recovery_timeout
        )
    
    def _on_success(self):
        """Handle successful execution."""
        if self.state == 'HALF_OPEN':
            self.state = 'CLOSED'
            self._logger.info("Circuit breaker reset to CLOSED state")
        
        self.failure_count = 0
    
    def _on_failure(self):
        """Handle failed execution."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = 'OPEN'
            self._logger.warning(
                f"Circuit breaker opened after {self.failure_count} failures"
            )


class RetryableError(Exception):
    """Exception that indicates an operation should be retried."""
    pass


class CircuitBreakerOpenError(Exception):
    """Exception raised when circuit breaker is open."""
    pass


def retry_with_backoff(
    config: Optional[RetryConfig] = None,
    exceptions: tuple = (Exception,),
    on_retry: Optional[Callable[[int, Exception], None]] = None
):
    """Decorator to retry function calls with configurable backoff."""
    if config is None:
        config = RetryConfig()
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(config.max_attempts):
                try:
                    return func(*args, **kwargs)
                    
                except exceptions as e:
                    last_exception = e
                    
                    if attempt == config.max_attempts - 1:
                        # Last attempt failed
                        break
                    
                    # Calculate delay
                    delay = _calculate_delay(config, attempt)
                    
                    # Call retry callback if provided
                    if on_retry:
                        on_retry(attempt + 1, e)
                    
                    # Log retry
                    logger = logging.getLogger(__name__)
                    logger.warning(
                        f"Attempt {attempt + 1} failed for {func.__name__}: {e}. "
                        f"Retrying in {delay:.2f}s..."
                    )
                    
                    time.sleep(delay)
            
            # All attempts failed
            raise last_exception
            
        return wrapper
    return decorator


def _calculate_delay(config: RetryConfig, attempt: int) -> float:
    """Calculate delay for retry attempt."""
    if config.backoff_strategy == 'constant':
        delay = config.base_delay
    elif config.backoff_strategy == 'linear':
        delay = config.base_delay * (attempt + 1)
    elif config.backoff_strategy == 'exponential':
        delay = config.base_delay * (config.exponential_base ** attempt)
    else:
        raise ValueError(f"Unknown backoff strategy: {config.backoff_strategy}")
    
    # Apply maximum delay
    delay = min(delay, config.max_delay)
    
    # Add jitter if enabled
    if config.jitter:
        jitter_range = delay * 0.1  # 10% jitter
        delay += random.uniform(-jitter_range, jitter_range)
    
    return max(0, delay)  # Ensure non-negative


class GracefulDegradation:
    """Handle graceful degradation when services are unavailable."""
    
    def __init__(self, fallback_func: Optional[Callable] = None):
        self.fallback_func = fallback_func
        self._logger = logging.getLogger(__name__)
    
    def __call__(self, func):
        """Decorator to apply graceful degradation."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                self._logger.warning(
                    f"Primary function {func.__name__} failed: {e}. "
                    "Attempting graceful degradation..."
                )
                
                if self.fallback_func:
                    try:
                        result = self.fallback_func(*args, **kwargs)
                        self._logger.info("Fallback succeeded")
                        return result
                    except Exception as fallback_error:
                        self._logger.error(f"Fallback also failed: {fallback_error}")
                        raise e  # Raise original exception
                else:
                    # No fallback available
                    raise e
        
        return wrapper


class ErrorRecoveryManager:
    """Centralized error recovery management."""
    
    def __init__(self):
        self.recovery_strategies = {}
        self._logger = logging.getLogger(__name__)
    
    def register_recovery_strategy(
        self,
        exception_type: Type[Exception],
        recovery_func: Callable[[Exception], Any]
    ):
        """Register a recovery strategy for an exception type."""
        self.recovery_strategies[exception_type] = recovery_func
    
    def attempt_recovery(self, exception: Exception) -> Any:
        """Attempt to recover from an exception."""
        exception_type = type(exception)
        
        # Look for exact match first
        if exception_type in self.recovery_strategies:
            recovery_func = self.recovery_strategies[exception_type]
            self._logger.info(f"Attempting recovery for {exception_type.__name__}")
            return recovery_func(exception)
        
        # Look for parent class matches
        for registered_type, recovery_func in self.recovery_strategies.items():
            if isinstance(exception, registered_type):
                self._logger.info(
                    f"Attempting recovery for {exception_type.__name__} "
                    f"using {registered_type.__name__} strategy"
                )
                return recovery_func(exception)
        
        # No recovery strategy found
        self._logger.warning(f"No recovery strategy for {exception_type.__name__}")
        raise exception
    
    def with_recovery(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with automatic error recovery."""
        try:
            return func(*args, **kwargs)
        except Exception as e:
            return self.attempt_recovery(e)


# Example recovery strategies
def memory_cleanup_recovery(exception: Exception):
    """Recovery strategy for memory errors."""
    import gc
    import torch
    
    # Force garbage collection
    gc.collect()
    
    # Clear GPU cache if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Log recovery action
    logger = logging.getLogger(__name__)
    logger.info("Memory cleanup recovery executed")
    
    # Return a default value or re-raise with context
    raise RetryableError(f"Memory cleaned up, retry recommended: {exception}")


def device_fallback_recovery(exception: Exception):
    """Recovery strategy for device errors."""
    # This could switch from GPU to CPU processing
    logger = logging.getLogger(__name__)
    logger.warning("Device error detected, consider fallback to CPU")
    
    # Return information about the fallback
    return {
        'status': 'fallback_recommended',
        'device': 'cpu',
        'original_error': str(exception)
    }


# Global error recovery manager
_global_recovery_manager = None

def get_recovery_manager() -> ErrorRecoveryManager:
    """Get global error recovery manager."""
    global _global_recovery_manager
    if _global_recovery_manager is None:
        _global_recovery_manager = ErrorRecoveryManager()
        
        # Register default recovery strategies
        _global_recovery_manager.register_recovery_strategy(
            MemoryError, memory_cleanup_recovery
        )
        _global_recovery_manager.register_recovery_strategy(
            RuntimeError, device_fallback_recovery
        )
    
    return _global_recovery_manager