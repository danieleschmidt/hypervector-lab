"""Advanced error recovery and resilience systems for HDC.

Provides fault tolerance, automatic recovery, and graceful degradation
for production hyperdimensional computing systems.
"""

import torch
import traceback
import time
import threading
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass
from enum import Enum
from contextlib import contextmanager
from functools import wraps
import pickle
import json
from pathlib import Path

from ..core.hypervector import HyperVector
from ..core.operations import cosine_similarity
from .logging import get_logger

logger = get_logger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels for recovery decisions."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RecoveryStrategy(Enum):
    """Available recovery strategies."""
    RETRY = "retry"
    FALLBACK = "fallback"
    RESTART = "restart"
    GRACEFUL_DEGRADE = "graceful_degrade"
    CHECKPOINT_RESTORE = "checkpoint_restore"


@dataclass
class ErrorContext:
    """Context information for error analysis and recovery."""
    error_type: str
    error_message: str
    stack_trace: str
    timestamp: float
    severity: ErrorSeverity
    recovery_strategy: RecoveryStrategy
    metadata: Dict[str, Any]


@dataclass
class SystemCheckpoint:
    """System state checkpoint for recovery."""
    timestamp: float
    state_data: Dict[str, Any]
    checksum: str
    metadata: Dict[str, Any]


class CircuitBreaker:
    """Circuit breaker pattern for fault tolerance."""
    
    def __init__(self, failure_threshold: int = 5, timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = "closed"  # closed, open, half-open
        self._lock = threading.Lock()
    
    def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        with self._lock:
            if self.state == "open":
                if time.time() - self.last_failure_time >= self.timeout:
                    self.state = "half-open"
                    logger.info("Circuit breaker moving to half-open state")
                else:
                    raise RuntimeError("Circuit breaker is open")
            
            try:
                result = func(*args, **kwargs)
                
                if self.state == "half-open":
                    self.state = "closed"
                    self.failure_count = 0
                    logger.info("Circuit breaker closed - service recovered")
                
                return result
                
            except Exception as e:
                self.failure_count += 1
                self.last_failure_time = time.time()
                
                if self.failure_count >= self.failure_threshold:
                    self.state = "open"
                    logger.warning(f"Circuit breaker opened after {self.failure_count} failures")
                
                raise e


class CheckpointManager:
    """Manages system checkpoints for recovery."""
    
    def __init__(self, checkpoint_dir: str = "checkpoints", max_checkpoints: int = 10):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self.checkpoints: List[SystemCheckpoint] = []
        self._load_existing_checkpoints()
    
    def create_checkpoint(self, system_state: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> str:
        """Create a new system checkpoint."""
        timestamp = time.time()
        
        # Serialize state
        try:
            state_data = self._serialize_state(system_state)
            checksum = self._compute_checksum(state_data)
            
            checkpoint = SystemCheckpoint(
                timestamp=timestamp,
                state_data=state_data,
                checksum=checksum,
                metadata=metadata or {}
            )
            
            # Save to disk
            checkpoint_id = f"checkpoint_{int(timestamp)}"
            checkpoint_path = self.checkpoint_dir / f"{checkpoint_id}.pkl"
            
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(checkpoint, f)
            
            self.checkpoints.append(checkpoint)
            self._cleanup_old_checkpoints()
            
            logger.info(f"Created checkpoint {checkpoint_id} with checksum {checksum[:8]}")
            return checkpoint_id
            
        except Exception as e:
            logger.error(f"Failed to create checkpoint: {e}")
            raise
    
    def restore_checkpoint(self, checkpoint_id: Optional[str] = None) -> Dict[str, Any]:
        """Restore system state from checkpoint."""
        try:
            if checkpoint_id:
                checkpoint_path = self.checkpoint_dir / f"{checkpoint_id}.pkl"
                if not checkpoint_path.exists():
                    raise FileNotFoundError(f"Checkpoint {checkpoint_id} not found")
            else:
                # Use latest checkpoint
                if not self.checkpoints:
                    raise RuntimeError("No checkpoints available")
                checkpoint = self.checkpoints[-1]
                checkpoint_path = self.checkpoint_dir / f"checkpoint_{int(checkpoint.timestamp)}.pkl"
            
            with open(checkpoint_path, 'rb') as f:
                checkpoint = pickle.load(f)
            
            # Verify checksum
            computed_checksum = self._compute_checksum(checkpoint.state_data)
            if computed_checksum != checkpoint.checksum:
                raise RuntimeError(f"Checkpoint corruption detected: {computed_checksum} != {checkpoint.checksum}")
            
            restored_state = self._deserialize_state(checkpoint.state_data)
            
            logger.info(f"Restored checkpoint from {time.ctime(checkpoint.timestamp)}")
            return restored_state
            
        except Exception as e:
            logger.error(f"Failed to restore checkpoint: {e}")
            raise
    
    def _serialize_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize system state for checkpointing."""
        serialized = {}
        
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                serialized[key] = {
                    'type': 'tensor',
                    'data': value.cpu().numpy().tolist(),
                    'shape': list(value.shape),
                    'dtype': str(value.dtype)
                }
            elif isinstance(value, HyperVector):
                serialized[key] = {
                    'type': 'hypervector',
                    'data': value.data.cpu().numpy().tolist(),
                    'mode': value.mode
                }
            else:
                serialized[key] = {'type': 'other', 'data': value}
        
        return serialized
    
    def _deserialize_state(self, serialized: Dict[str, Any]) -> Dict[str, Any]:
        """Deserialize system state from checkpoint."""
        state = {}
        
        for key, value in serialized.items():
            if value['type'] == 'tensor':
                data = torch.tensor(value['data'], dtype=getattr(torch, value['dtype'].split('.')[-1]))
                state[key] = data.reshape(value['shape'])
            elif value['type'] == 'hypervector':
                data = torch.tensor(value['data'])
                state[key] = HyperVector(data, mode=value['mode'])
            else:
                state[key] = value['data']
        
        return state
    
    def _compute_checksum(self, data: Dict[str, Any]) -> str:
        """Compute checksum for data integrity."""
        import hashlib
        serialized = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode()).hexdigest()
    
    def _load_existing_checkpoints(self):
        """Load existing checkpoints from disk."""
        for checkpoint_file in self.checkpoint_dir.glob("checkpoint_*.pkl"):
            try:
                with open(checkpoint_file, 'rb') as f:
                    checkpoint = pickle.load(f)
                self.checkpoints.append(checkpoint)
            except Exception as e:
                logger.warning(f"Failed to load checkpoint {checkpoint_file}: {e}")
        
        # Sort by timestamp
        self.checkpoints.sort(key=lambda c: c.timestamp)
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints to maintain limit."""
        while len(self.checkpoints) > self.max_checkpoints:
            old_checkpoint = self.checkpoints.pop(0)
            checkpoint_file = self.checkpoint_dir / f"checkpoint_{int(old_checkpoint.timestamp)}.pkl"
            if checkpoint_file.exists():
                checkpoint_file.unlink()


class ErrorAnalyzer:
    """Analyzes errors and determines recovery strategies."""
    
    def __init__(self):
        self.error_patterns = {
            'memory_error': {
                'patterns': ['out of memory', 'memory allocation', 'cuda out of memory'],
                'severity': ErrorSeverity.HIGH,
                'strategy': RecoveryStrategy.GRACEFUL_DEGRADE
            },
            'cuda_error': {
                'patterns': ['cuda', 'gpu', 'device-side assert'],
                'severity': ErrorSeverity.MEDIUM,
                'strategy': RecoveryStrategy.FALLBACK
            },
            'numerical_error': {
                'patterns': ['nan', 'inf', 'overflow', 'underflow'],
                'severity': ErrorSeverity.MEDIUM,
                'strategy': RecoveryStrategy.CHECKPOINT_RESTORE
            },
            'io_error': {
                'patterns': ['file not found', 'permission denied', 'disk full'],
                'severity': ErrorSeverity.LOW,
                'strategy': RecoveryStrategy.RETRY
            }
        }
        
        self.error_history: List[ErrorContext] = []
    
    def analyze_error(self, error: Exception, context: Dict[str, Any]) -> ErrorContext:
        """Analyze error and determine recovery strategy."""
        error_message = str(error).lower()
        error_type = type(error).__name__
        stack_trace = traceback.format_exc()
        
        # Pattern matching
        severity = ErrorSeverity.LOW
        strategy = RecoveryStrategy.RETRY
        
        for pattern_name, pattern_info in self.error_patterns.items():
            if any(pattern in error_message for pattern in pattern_info['patterns']):
                severity = pattern_info['severity']
                strategy = pattern_info['strategy']
                break
        
        # Check error frequency
        recent_errors = [e for e in self.error_history 
                        if time.time() - e.timestamp < 300]  # Last 5 minutes
        
        if len(recent_errors) > 10:
            severity = ErrorSeverity.CRITICAL
            strategy = RecoveryStrategy.RESTART
        
        error_context = ErrorContext(
            error_type=error_type,
            error_message=str(error),
            stack_trace=stack_trace,
            timestamp=time.time(),
            severity=severity,
            recovery_strategy=strategy,
            metadata=context
        )
        
        self.error_history.append(error_context)
        return error_context


class FallbackProvider:
    """Provides fallback implementations for failed operations."""
    
    def __init__(self):
        self.fallback_implementations = {
            'bind': self._fallback_bind,
            'bundle': self._fallback_bundle,
            'similarity': self._fallback_similarity,
            'encode': self._fallback_encode
        }
    
    def get_fallback(self, operation: str, *args, **kwargs):
        """Get fallback implementation for operation."""
        if operation in self.fallback_implementations:
            logger.warning(f"Using fallback implementation for {operation}")
            return self.fallback_implementations[operation](*args, **kwargs)
        else:
            raise NotImplementedError(f"No fallback available for {operation}")
    
    def _fallback_bind(self, hv1: HyperVector, hv2: HyperVector) -> HyperVector:
        """Simple fallback binding using element-wise multiplication."""
        try:
            result_data = hv1.data * hv2.data
            return HyperVector(result_data)
        except Exception:
            # Ultra-simple fallback
            result_data = torch.sign(hv1.data) * torch.sign(hv2.data)
            return HyperVector(result_data, mode="binary")
    
    def _fallback_bundle(self, hvs: List[HyperVector]) -> HyperVector:
        """Simple fallback bundling using averaging."""
        if not hvs:
            return HyperVector(torch.zeros(1000))
        
        try:
            # Stack and average
            stacked = torch.stack([hv.data for hv in hvs])
            result_data = torch.mean(stacked, dim=0)
            return HyperVector(result_data)
        except Exception:
            # Return first vector as last resort
            return hvs[0]
    
    def _fallback_similarity(self, hv1: HyperVector, hv2: HyperVector) -> torch.Tensor:
        """Simple fallback similarity using dot product."""
        try:
            return torch.dot(hv1.data.flatten(), hv2.data.flatten())
        except Exception:
            return torch.tensor(0.0)
    
    def _fallback_encode(self, data: Any, dim: int = 1000) -> HyperVector:
        """Fallback encoder for unknown data types."""
        try:
            if isinstance(data, (int, float)):
                # Encode scalar as position in hypervector
                result = torch.zeros(dim)
                pos = int(abs(data) * 100) % dim
                result[pos] = 1.0 if data >= 0 else -1.0
                return HyperVector(result, mode="binary")
            else:
                # Random hypervector as ultimate fallback
                result = torch.randn(dim)
                return HyperVector(torch.sign(result), mode="binary")
        except Exception:
            return HyperVector(torch.zeros(dim))


class ResilientSystem:
    """Main resilient system with integrated error recovery."""
    
    def __init__(self, enable_checkpoints: bool = True, checkpoint_interval: float = 300.0):
        self.circuit_breaker = CircuitBreaker()
        self.checkpoint_manager = CheckpointManager() if enable_checkpoints else None
        self.error_analyzer = ErrorAnalyzer()
        self.fallback_provider = FallbackProvider()
        
        self.checkpoint_interval = checkpoint_interval
        self.last_checkpoint = time.time()
        self.system_state = {}
        
        self.recovery_stats = {
            'total_errors': 0,
            'successful_recoveries': 0,
            'failed_recoveries': 0,
            'fallback_uses': 0,
            'checkpoint_restores': 0
        }
    
    @contextmanager
    def resilient_execution(self, operation_name: str, context: Optional[Dict[str, Any]] = None):
        """Context manager for resilient operation execution."""
        context = context or {}
        context['operation'] = operation_name
        
        try:
            # Auto-checkpoint if interval passed
            if (self.checkpoint_manager and 
                time.time() - self.last_checkpoint > self.checkpoint_interval):
                self._auto_checkpoint()
            
            yield self
            
        except Exception as e:
            self.recovery_stats['total_errors'] += 1
            
            # Analyze error
            error_context = self.error_analyzer.analyze_error(e, context)
            logger.error(f"Error in {operation_name}: {error_context.error_message}")
            
            # Attempt recovery
            try:
                self._attempt_recovery(error_context, context)
                self.recovery_stats['successful_recoveries'] += 1
                logger.info(f"Successfully recovered from {operation_name} error")
            except Exception as recovery_error:
                self.recovery_stats['failed_recoveries'] += 1
                logger.error(f"Recovery failed: {recovery_error}")
                raise recovery_error
    
    def _attempt_recovery(self, error_context: ErrorContext, context: Dict[str, Any]):
        """Attempt recovery based on error analysis."""
        strategy = error_context.recovery_strategy
        
        if strategy == RecoveryStrategy.RETRY:
            self._retry_operation(context)
        
        elif strategy == RecoveryStrategy.FALLBACK:
            self._use_fallback(context)
        
        elif strategy == RecoveryStrategy.CHECKPOINT_RESTORE:
            self._restore_from_checkpoint()
        
        elif strategy == RecoveryStrategy.GRACEFUL_DEGRADE:
            self._graceful_degrade(context)
        
        elif strategy == RecoveryStrategy.RESTART:
            self._restart_system()
        
        else:
            raise RuntimeError(f"Unknown recovery strategy: {strategy}")
    
    def _retry_operation(self, context: Dict[str, Any]):
        """Retry the failed operation."""
        operation = context.get('operation', 'unknown')
        logger.info(f"Retrying operation: {operation}")
        # Implementation depends on context
        time.sleep(1)  # Brief delay before retry
    
    def _use_fallback(self, context: Dict[str, Any]):
        """Use fallback implementation."""
        self.recovery_stats['fallback_uses'] += 1
        operation = context.get('operation', 'unknown')
        logger.info(f"Using fallback for operation: {operation}")
        
        # Return fallback result through context
        if 'args' in context and 'operation_type' in context:
            result = self.fallback_provider.get_fallback(
                context['operation_type'], 
                *context['args']
            )
            context['fallback_result'] = result
    
    def _restore_from_checkpoint(self):
        """Restore system state from latest checkpoint."""
        if not self.checkpoint_manager:
            raise RuntimeError("Checkpoint manager not enabled")
        
        self.recovery_stats['checkpoint_restores'] += 1
        logger.info("Restoring from checkpoint")
        
        restored_state = self.checkpoint_manager.restore_checkpoint()
        self.system_state.update(restored_state)
    
    def _graceful_degrade(self, context: Dict[str, Any]):
        """Gracefully degrade system performance."""
        logger.info("Initiating graceful degradation")
        
        # Reduce system complexity
        if 'batch_size' in context:
            context['batch_size'] = max(1, context['batch_size'] // 2)
        
        if 'precision' in context:
            context['precision'] = 'low'
        
        # Switch to CPU if GPU fails
        if 'device' in context and 'cuda' in str(context['device']):
            context['device'] = 'cpu'
            logger.info("Switched to CPU due to GPU error")
    
    def _restart_system(self):
        """Restart critical system components."""
        logger.warning("Initiating system restart")
        
        # Clear caches
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Reset system state
        self.system_state.clear()
        
        # Reset circuit breaker
        self.circuit_breaker.failure_count = 0
        self.circuit_breaker.state = "closed"
    
    def _auto_checkpoint(self):
        """Create automatic checkpoint."""
        if self.checkpoint_manager and self.system_state:
            try:
                checkpoint_id = self.checkpoint_manager.create_checkpoint(
                    self.system_state,
                    {'type': 'auto', 'timestamp': time.time()}
                )
                self.last_checkpoint = time.time()
                logger.debug(f"Auto-checkpoint created: {checkpoint_id}")
            except Exception as e:
                logger.warning(f"Auto-checkpoint failed: {e}")
    
    def update_system_state(self, key: str, value: Any):
        """Update system state for checkpointing."""
        self.system_state[key] = value
    
    def get_recovery_stats(self) -> Dict[str, Any]:
        """Get recovery statistics."""
        stats = self.recovery_stats.copy()
        if stats['total_errors'] > 0:
            stats['recovery_rate'] = stats['successful_recoveries'] / stats['total_errors']
        else:
            stats['recovery_rate'] = 1.0
        
        return stats


def resilient_operation(operation_name: str):
    """Decorator for making operations resilient."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            resilient_system = getattr(wrapper, '_resilient_system', None)
            if not resilient_system:
                resilient_system = ResilientSystem()
                wrapper._resilient_system = resilient_system
            
            context = {
                'function': func.__name__,
                'args': args,
                'kwargs': kwargs,
                'operation_type': operation_name
            }
            
            with resilient_system.resilient_execution(operation_name, context):
                try:
                    result = func(*args, **kwargs)
                    resilient_system.update_system_state(f"{operation_name}_result", result)
                    return result
                except Exception as e:
                    # Check if fallback result available
                    if 'fallback_result' in context:
                        return context['fallback_result']
                    raise
        
        return wrapper
    return decorator


# Example usage decorators
@resilient_operation('bind')
def resilient_bind(hv1: HyperVector, hv2: HyperVector) -> HyperVector:
    """Resilient binding operation."""
    from ..core.operations import bind
    return bind(hv1, hv2)


@resilient_operation('similarity')
def resilient_similarity(hv1: HyperVector, hv2: HyperVector) -> torch.Tensor:
    """Resilient similarity computation."""
    return cosine_similarity(hv1, hv2)
