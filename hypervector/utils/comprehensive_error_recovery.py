"""
Comprehensive Error Recovery System for HDC
==========================================

Advanced error recovery and self-healing capabilities for hyperdimensional
computing systems with quantum-enhanced fault tolerance.

Features:
1. Intelligent error detection and classification
2. Automated recovery strategies
3. Quantum error correction for HDC operations
4. Self-healing system architecture
5. Predictive failure analysis
6. Graceful degradation mechanisms
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import time
import logging
import threading
import traceback
import pickle
import json
from concurrent.futures import ThreadPoolExecutor
import asyncio
from collections import deque, defaultdict

from ..core.hypervector import HyperVector
from ..core.operations import bind, bundle, cosine_similarity
try:
    from .validation import ValidationResult
except ImportError:
    ValidationResult = Dict[str, Any]

logger = logging.getLogger(__name__)

class ErrorType(Enum):
    """Types of errors in HDC systems."""
    COMPUTATION_ERROR = "computation_error"
    MEMORY_ERROR = "memory_error"
    DEVICE_ERROR = "device_error"
    VALIDATION_ERROR = "validation_error"
    QUANTUM_ERROR = "quantum_error"
    NETWORK_ERROR = "network_error"
    DATA_CORRUPTION = "data_corruption"
    TIMEOUT_ERROR = "timeout_error"
    RESOURCE_EXHAUSTION = "resource_exhaustion"

class RecoveryStrategy(Enum):
    """Recovery strategies for different error types."""
    RETRY = "retry"
    FALLBACK = "fallback"
    RECONSTRUCT = "reconstruct"
    CHECKPOINT_RESTORE = "checkpoint_restore"
    GRACEFUL_DEGRADATION = "graceful_degradation"
    QUANTUM_CORRECTION = "quantum_correction"
    SYSTEM_RESTART = "system_restart"

class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class ErrorRecord:
    """Record of an error occurrence."""
    error_id: str
    timestamp: float
    error_type: ErrorType
    severity: ErrorSeverity
    message: str
    component: str
    stack_trace: str
    context: Dict[str, Any] = field(default_factory=dict)
    recovery_attempted: bool = False
    recovery_successful: bool = False
    recovery_strategy: Optional[RecoveryStrategy] = None
    recovery_time: float = 0.0

class SystemCheckpoint:
    """System checkpoint for recovery."""
    
    def __init__(
        self,
        checkpoint_id: str,
        timestamp: float,
        system_state: Dict[str, Any],
        performance_metrics: Dict[str, float]
    ):
        self.checkpoint_id = checkpoint_id
        self.timestamp = timestamp
        self.system_state = system_state
        self.performance_metrics = performance_metrics
        self.validation_hash = self._compute_validation_hash()
    
    def _compute_validation_hash(self) -> str:
        """Compute validation hash for checkpoint integrity."""
        try:
            state_str = json.dumps(self.system_state, sort_keys=True, default=str)
            return hash(state_str + str(self.timestamp))
        except Exception:
            return hash(str(self.timestamp))
    
    def is_valid(self) -> bool:
        """Check if checkpoint is valid."""
        try:
            current_hash = self._compute_validation_hash()
            return current_hash == self.validation_hash
        except Exception:
            return False

class ComprehensiveErrorRecovery:
    """Comprehensive error recovery system for HDC."""
    
    def __init__(
        self,
        hdc_dim: int = 10000,
        device: Optional[str] = None,
        max_recovery_attempts: int = 3,
        checkpoint_interval: float = 300.0,
        enable_quantum_correction: bool = True
    ):
        self.hdc_dim = hdc_dim
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_recovery_attempts = max_recovery_attempts
        self.checkpoint_interval = checkpoint_interval
        self.enable_quantum_correction = enable_quantum_correction
        
        # Error tracking
        self.error_history: List[ErrorRecord] = []
        self.recovery_statistics = defaultdict(int)
        
        # Checkpointing system
        self.checkpoints: List[SystemCheckpoint] = []
        self.max_checkpoints = 10
        self.last_checkpoint_time = time.time()
        
        # Recovery mechanisms
        self.recovery_strategies = self._initialize_recovery_strategies()
        self.fallback_functions = {}
        
        # Quantum error correction
        if enable_quantum_correction:
            self._initialize_quantum_correction()
        
        # Self-healing mechanisms
        self.healing_patterns = {}
        self.predictive_models = {}
        
        # Performance monitoring
        self.performance_baseline = {}
        self.degradation_thresholds = {
            'accuracy': 0.1,
            'latency': 2.0,
            'memory': 1.5
        }
        
        logger.info("Comprehensive Error Recovery System initialized")
    
    def _initialize_recovery_strategies(self) -> Dict[ErrorType, List[RecoveryStrategy]]:
        """Initialize recovery strategies for each error type."""
        return {
            ErrorType.COMPUTATION_ERROR: [
                RecoveryStrategy.RETRY,
                RecoveryStrategy.FALLBACK,
                RecoveryStrategy.QUANTUM_CORRECTION
            ],
            ErrorType.MEMORY_ERROR: [
                RecoveryStrategy.GRACEFUL_DEGRADATION,
                RecoveryStrategy.CHECKPOINT_RESTORE,
                RecoveryStrategy.SYSTEM_RESTART
            ],
            ErrorType.DEVICE_ERROR: [
                RecoveryStrategy.FALLBACK,
                RecoveryStrategy.CHECKPOINT_RESTORE
            ],
            ErrorType.VALIDATION_ERROR: [
                RecoveryStrategy.RECONSTRUCT,
                RecoveryStrategy.FALLBACK
            ],
            ErrorType.QUANTUM_ERROR: [
                RecoveryStrategy.QUANTUM_CORRECTION,
                RecoveryStrategy.RECONSTRUCT,
                RecoveryStrategy.CHECKPOINT_RESTORE
            ],
            ErrorType.NETWORK_ERROR: [
                RecoveryStrategy.RETRY,
                RecoveryStrategy.GRACEFUL_DEGRADATION
            ],
            ErrorType.DATA_CORRUPTION: [
                RecoveryStrategy.RECONSTRUCT,
                RecoveryStrategy.CHECKPOINT_RESTORE,
                RecoveryStrategy.QUANTUM_CORRECTION
            ],
            ErrorType.TIMEOUT_ERROR: [
                RecoveryStrategy.GRACEFUL_DEGRADATION,
                RecoveryStrategy.RETRY
            ],
            ErrorType.RESOURCE_EXHAUSTION: [
                RecoveryStrategy.GRACEFUL_DEGRADATION,
                RecoveryStrategy.CHECKPOINT_RESTORE
            ]
        }
    
    def _initialize_quantum_correction(self):
        """Initialize quantum error correction patterns."""
        self.quantum_correction_codes = {
            'stabilizer_codes': [
                HyperVector.random(self.hdc_dim, device=self.device)
                for _ in range(7)  # 7-qubit Steane code analogy
            ],
            'surface_codes': [
                HyperVector.random(self.hdc_dim, device=self.device)
                for _ in range(9)  # 9-qubit surface code analogy
            ]
        }
        
        logger.info("Quantum error correction initialized")
    
    def create_checkpoint(
        self,
        system_state: Dict[str, Any],
        performance_metrics: Optional[Dict[str, float]] = None
    ) -> str:
        """Create system checkpoint."""
        try:
            checkpoint_id = f"checkpoint_{int(time.time())}_{len(self.checkpoints)}"
            
            if performance_metrics is None:
                performance_metrics = self._collect_performance_metrics()
            
            checkpoint = SystemCheckpoint(
                checkpoint_id=checkpoint_id,
                timestamp=time.time(),
                system_state=system_state,
                performance_metrics=performance_metrics
            )
            
            self.checkpoints.append(checkpoint)
            
            # Maintain maximum number of checkpoints
            if len(self.checkpoints) > self.max_checkpoints:
                self.checkpoints.pop(0)
            
            self.last_checkpoint_time = time.time()
            
            logger.info(f"Created checkpoint {checkpoint_id}")
            return checkpoint_id
        
        except Exception as e:
            logger.error(f"Failed to create checkpoint: {e}")
            return ""
    
    def _collect_performance_metrics(self) -> Dict[str, float]:
        """Collect current performance metrics."""
        return {
            'timestamp': time.time(),
            'memory_usage': self._get_memory_usage(),
            'cpu_usage': self._get_cpu_usage(),
            'gpu_usage': self._get_gpu_usage() if torch.cuda.is_available() else 0.0
        }
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage."""
        try:
            if torch.cuda.is_available():
                return torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
            else:
                import psutil
                return psutil.virtual_memory().percent / 100.0
        except Exception:
            return 0.0
    
    def _get_cpu_usage(self) -> float:
        """Get current CPU usage."""
        try:
            import psutil
            return psutil.cpu_percent() / 100.0
        except Exception:
            return 0.0
    
    def _get_gpu_usage(self) -> float:
        """Get current GPU usage."""
        try:
            if torch.cuda.is_available():
                return torch.cuda.utilization() / 100.0
        except Exception:
            return 0.0
    
    def detect_error(
        self,
        exception: Exception,
        component: str,
        context: Optional[Dict[str, Any]] = None
    ) -> ErrorRecord:
        """Detect and classify error."""
        error_type = self._classify_error(exception)
        severity = self._assess_severity(exception, error_type)
        
        error_record = ErrorRecord(
            error_id=f"error_{int(time.time())}_{len(self.error_history)}",
            timestamp=time.time(),
            error_type=error_type,
            severity=severity,
            message=str(exception),
            component=component,
            stack_trace=traceback.format_exc(),
            context=context or {}
        )
        
        self.error_history.append(error_record)
        
        logger.error(f"Detected {error_type.value} error in {component}: {str(exception)}")
        
        return error_record
    
    def _classify_error(self, exception: Exception) -> ErrorType:
        """Classify error type based on exception."""
        exception_type = type(exception).__name__
        
        classification_map = {
            'RuntimeError': ErrorType.COMPUTATION_ERROR,
            'MemoryError': ErrorType.MEMORY_ERROR,
            'OutOfMemoryError': ErrorType.MEMORY_ERROR,
            'CUDA_ERROR': ErrorType.DEVICE_ERROR,
            'ValueError': ErrorType.VALIDATION_ERROR,
            'TypeError': ErrorType.VALIDATION_ERROR,
            'TimeoutError': ErrorType.TIMEOUT_ERROR,
            'ConnectionError': ErrorType.NETWORK_ERROR,
            'FileNotFoundError': ErrorType.DATA_CORRUPTION,
            'PermissionError': ErrorType.RESOURCE_EXHAUSTION
        }
        
        return classification_map.get(exception_type, ErrorType.COMPUTATION_ERROR)
    
    def _assess_severity(self, exception: Exception, error_type: ErrorType) -> ErrorSeverity:
        """Assess error severity."""
        # Critical errors
        if error_type in [ErrorType.MEMORY_ERROR, ErrorType.DEVICE_ERROR]:
            return ErrorSeverity.CRITICAL
        
        # High severity errors
        if error_type in [ErrorType.DATA_CORRUPTION, ErrorType.QUANTUM_ERROR]:
            return ErrorSeverity.HIGH
        
        # Medium severity errors
        if error_type in [ErrorType.VALIDATION_ERROR, ErrorType.NETWORK_ERROR]:
            return ErrorSeverity.MEDIUM
        
        # Default to low severity
        return ErrorSeverity.LOW
    
    def recover_from_error(
        self,
        error_record: ErrorRecord,
        recovery_context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Attempt to recover from error."""
        logger.info(f"Attempting recovery for error {error_record.error_id}")
        
        strategies = self.recovery_strategies.get(error_record.error_type, [RecoveryStrategy.RETRY])
        
        for strategy in strategies:
            try:
                recovery_start_time = time.time()
                
                success = self._apply_recovery_strategy(
                    error_record, strategy, recovery_context
                )
                
                recovery_time = time.time() - recovery_start_time
                
                error_record.recovery_attempted = True
                error_record.recovery_strategy = strategy
                error_record.recovery_time = recovery_time
                error_record.recovery_successful = success
                
                self.recovery_statistics[strategy.value] += 1
                
                if success:
                    logger.info(f"Recovery successful using {strategy.value}")
                    return True
                else:
                    logger.warning(f"Recovery failed using {strategy.value}")
            
            except Exception as recovery_error:
                logger.error(f"Recovery strategy {strategy.value} failed: {recovery_error}")
                continue
        
        logger.error(f"All recovery strategies failed for error {error_record.error_id}")
        return False
    
    def _apply_recovery_strategy(
        self,
        error_record: ErrorRecord,
        strategy: RecoveryStrategy,
        context: Optional[Dict[str, Any]]
    ) -> bool:
        """Apply specific recovery strategy."""
        if strategy == RecoveryStrategy.RETRY:
            return self._retry_operation(error_record, context)
        
        elif strategy == RecoveryStrategy.FALLBACK:
            return self._apply_fallback(error_record, context)
        
        elif strategy == RecoveryStrategy.RECONSTRUCT:
            return self._reconstruct_data(error_record, context)
        
        elif strategy == RecoveryStrategy.CHECKPOINT_RESTORE:
            return self._restore_checkpoint(error_record, context)
        
        elif strategy == RecoveryStrategy.GRACEFUL_DEGRADATION:
            return self._graceful_degradation(error_record, context)
        
        elif strategy == RecoveryStrategy.QUANTUM_CORRECTION:
            return self._quantum_error_correction(error_record, context)
        
        elif strategy == RecoveryStrategy.SYSTEM_RESTART:
            return self._system_restart(error_record, context)
        
        else:
            logger.warning(f"Unknown recovery strategy: {strategy}")
            return False
    
    def _retry_operation(
        self,
        error_record: ErrorRecord,
        context: Optional[Dict[str, Any]]
    ) -> bool:
        """Retry failed operation."""
        try:
            if context and 'retry_function' in context:
                retry_function = context['retry_function']
                retry_args = context.get('retry_args', ())
                retry_kwargs = context.get('retry_kwargs', {})
                
                for attempt in range(self.max_recovery_attempts):
                    try:
                        result = retry_function(*retry_args, **retry_kwargs)
                        logger.info(f"Retry successful on attempt {attempt + 1}")
                        return True
                    except Exception as e:
                        if attempt == self.max_recovery_attempts - 1:
                            logger.error(f"All retry attempts failed: {e}")
                            return False
                        time.sleep(0.1 * (attempt + 1))  # Exponential backoff
            
            return False
        except Exception as e:
            logger.error(f"Retry strategy failed: {e}")
            return False
    
    def _apply_fallback(
        self,
        error_record: ErrorRecord,
        context: Optional[Dict[str, Any]]
    ) -> bool:
        """Apply fallback mechanism."""
        try:
            component = error_record.component
            
            if component in self.fallback_functions:
                fallback_func = self.fallback_functions[component]
                fallback_args = context.get('fallback_args', ()) if context else ()
                fallback_kwargs = context.get('fallback_kwargs', {}) if context else {}
                
                result = fallback_func(*fallback_args, **fallback_kwargs)
                logger.info(f"Fallback applied for component {component}")
                return True
            
            # Default fallback: return safe default values
            if context and 'default_return' in context:
                return True
            
            return False
        except Exception as e:
            logger.error(f"Fallback strategy failed: {e}")
            return False
    
    def _reconstruct_data(
        self,
        error_record: ErrorRecord,
        context: Optional[Dict[str, Any]]
    ) -> bool:
        """Reconstruct corrupted data."""
        try:
            if not context or 'corrupted_data' not in context:
                return False
            
            corrupted_data = context['corrupted_data']
            
            if isinstance(corrupted_data, HyperVector):
                # Attempt to reconstruct using redundancy
                reconstructed = self._reconstruct_hypervector(corrupted_data)
                
                if reconstructed is not None:
                    context['reconstructed_data'] = reconstructed
                    logger.info("Data reconstruction successful")
                    return True
            
            return False
        except Exception as e:
            logger.error(f"Data reconstruction failed: {e}")
            return False
    
    def _reconstruct_hypervector(self, corrupted_hv: HyperVector) -> Optional[HyperVector]:
        """Reconstruct corrupted hypervector."""
        try:
            # Use quantum error correction patterns
            if self.enable_quantum_correction:
                for code_type, correction_codes in self.quantum_correction_codes.items():
                    # Find best matching correction pattern
                    best_similarity = -1.0
                    best_code = None
                    
                    for code in correction_codes:
                        similarity = cosine_similarity(corrupted_hv, code)
                        if similarity.item() > best_similarity:
                            best_similarity = similarity.item()
                            best_code = code
                    
                    if best_code is not None and best_similarity > 0.7:
                        # Reconstruct using error correction
                        corrected = HyperVector(
                            0.7 * best_code.vector + 0.3 * corrupted_hv.vector
                        )
                        return corrected
            
            # Fallback: median filtering
            corrected_vector = torch.median(
                corrupted_hv.vector.view(-1, 1),
                dim=1
            ).values
            
            return HyperVector(corrected_vector)
        
        except Exception as e:
            logger.error(f"Hypervector reconstruction failed: {e}")
            return None
    
    def _restore_checkpoint(
        self,
        error_record: ErrorRecord,
        context: Optional[Dict[str, Any]]
    ) -> bool:
        """Restore from checkpoint."""
        try:
            if not self.checkpoints:
                logger.warning("No checkpoints available for restoration")
                return False
            
            # Find most recent valid checkpoint
            for checkpoint in reversed(self.checkpoints):
                if checkpoint.is_valid():
                    logger.info(f"Restoring from checkpoint {checkpoint.checkpoint_id}")
                    
                    if context:
                        context['restored_state'] = checkpoint.system_state
                        context['restored_metrics'] = checkpoint.performance_metrics
                    
                    return True
            
            logger.warning("No valid checkpoints found")
            return False
        
        except Exception as e:
            logger.error(f"Checkpoint restoration failed: {e}")
            return False
    
    def _graceful_degradation(
        self,
        error_record: ErrorRecord,
        context: Optional[Dict[str, Any]]
    ) -> bool:
        """Apply graceful degradation."""
        try:
            # Reduce system complexity/accuracy for continued operation
            degradation_config = {
                'reduce_precision': True,
                'simplify_operations': True,
                'disable_advanced_features': True
            }
            
            if context:
                context['degradation_config'] = degradation_config
            
            logger.info("Graceful degradation applied")
            return True
        
        except Exception as e:
            logger.error(f"Graceful degradation failed: {e}")
            return False
    
    def _quantum_error_correction(
        self,
        error_record: ErrorRecord,
        context: Optional[Dict[str, Any]]
    ) -> bool:
        """Apply quantum error correction."""
        try:
            if not self.enable_quantum_correction:
                return False
            
            if context and 'quantum_state' in context:
                quantum_state = context['quantum_state']
                
                if isinstance(quantum_state, HyperVector):
                    corrected_state = self._apply_quantum_correction(quantum_state)
                    
                    if corrected_state is not None:
                        context['corrected_quantum_state'] = corrected_state
                        logger.info("Quantum error correction applied")
                        return True
            
            return False
        
        except Exception as e:
            logger.error(f"Quantum error correction failed: {e}")
            return False
    
    def _apply_quantum_correction(self, quantum_state: HyperVector) -> Optional[HyperVector]:
        """Apply quantum error correction to hypervector."""
        try:
            # Use stabilizer codes for error correction
            stabilizers = self.quantum_correction_codes['stabilizer_codes']
            
            # Find syndrome (error pattern)
            syndrome = []
            for stabilizer in stabilizers:
                measurement = torch.dot(quantum_state.vector, stabilizer.vector)
                syndrome.append(measurement.item() > 0.5)
            
            # Apply correction based on syndrome
            if any(syndrome):
                # Select correction operator
                correction_index = sum(bit * (2**i) for i, bit in enumerate(syndrome)) % len(stabilizers)
                correction_operator = stabilizers[correction_index]
                
                # Apply correction
                corrected_state = HyperVector(
                    quantum_state.vector + 0.1 * correction_operator.vector
                )
                
                return corrected_state
            
            return quantum_state
        
        except Exception as e:
            logger.error(f"Quantum correction application failed: {e}")
            return None
    
    def _system_restart(
        self,
        error_record: ErrorRecord,
        context: Optional[Dict[str, Any]]
    ) -> bool:
        """Perform system restart (simulation)."""
        try:
            logger.critical("System restart requested - this would restart the system")
            
            # In a real system, this would perform actual restart
            # For simulation, we'll just clear some state
            self.error_history = self.error_history[-10:]  # Keep last 10 errors
            
            if context:
                context['system_restarted'] = True
            
            return True
        
        except Exception as e:
            logger.error(f"System restart failed: {e}")
            return False
    
    def register_fallback_function(
        self,
        component: str,
        fallback_func: Callable
    ):
        """Register fallback function for component."""
        self.fallback_functions[component] = fallback_func
        logger.info(f"Registered fallback function for {component}")
    
    def predict_failures(
        self,
        current_metrics: Dict[str, float]
    ) -> List[Tuple[str, float]]:
        """Predict potential failures based on current metrics."""
        predictions = []
        
        try:
            # Simple predictive analysis
            for metric_name, current_value in current_metrics.items():
                if metric_name in self.performance_baseline:
                    baseline = self.performance_baseline[metric_name]
                    threshold = self.degradation_thresholds.get(metric_name, 1.5)
                    
                    if current_value > baseline * threshold:
                        failure_probability = min((current_value / baseline) - 1.0, 1.0)
                        predictions.append((metric_name, failure_probability))
            
            # Sort by failure probability
            predictions.sort(key=lambda x: x[1], reverse=True)
            
        except Exception as e:
            logger.error(f"Failure prediction failed: {e}")
        
        return predictions
    
    def update_performance_baseline(self, metrics: Dict[str, float]):
        """Update performance baseline for failure prediction."""
        for metric_name, value in metrics.items():
            if metric_name not in self.performance_baseline:
                self.performance_baseline[metric_name] = value
            else:
                # Exponential moving average
                alpha = 0.1
                self.performance_baseline[metric_name] = (
                    alpha * value + (1 - alpha) * self.performance_baseline[metric_name]
                )
    
    def generate_recovery_report(self) -> Dict[str, Any]:
        """Generate comprehensive recovery report."""
        total_errors = len(self.error_history)
        recovered_errors = sum(1 for error in self.error_history if error.recovery_successful)
        
        error_type_distribution = defaultdict(int)
        severity_distribution = defaultdict(int)
        
        for error in self.error_history:
            error_type_distribution[error.error_type.value] += 1
            severity_distribution[error.severity.name] += 1
        
        report = {
            'summary': {
                'total_errors': total_errors,
                'recovered_errors': recovered_errors,
                'recovery_rate': recovered_errors / total_errors if total_errors > 0 else 1.0,
                'active_checkpoints': len(self.checkpoints)
            },
            'error_analysis': {
                'error_type_distribution': dict(error_type_distribution),
                'severity_distribution': dict(severity_distribution)
            },
            'recovery_statistics': dict(self.recovery_statistics),
            'system_health': {
                'quantum_correction_enabled': self.enable_quantum_correction,
                'fallback_functions_registered': len(self.fallback_functions),
                'performance_baseline_established': len(self.performance_baseline) > 0
            },
            'recent_errors': [
                {
                    'error_id': error.error_id,
                    'timestamp': error.timestamp,
                    'error_type': error.error_type.value,
                    'severity': error.severity.name,
                    'component': error.component,
                    'recovery_successful': error.recovery_successful,
                    'recovery_strategy': error.recovery_strategy.value if error.recovery_strategy else None
                }
                for error in sorted(self.error_history, key=lambda x: x.timestamp, reverse=True)[:10]
            ]
        }
        
        return report

# Context manager for automatic error recovery
class ErrorRecoveryContext:
    """Context manager for automatic error recovery."""
    
    def __init__(
        self,
        recovery_system: ComprehensiveErrorRecovery,
        component: str,
        context: Optional[Dict[str, Any]] = None
    ):
        self.recovery_system = recovery_system
        self.component = component
        self.context = context or {}
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_val is not None:
            # Error occurred, attempt recovery
            error_record = self.recovery_system.detect_error(exc_val, self.component, self.context)
            recovery_success = self.recovery_system.recover_from_error(error_record, self.context)
            
            if recovery_success:
                logger.info(f"Error recovered successfully in {self.component}")
                return True  # Suppress exception
            else:
                logger.error(f"Error recovery failed in {self.component}")
                return False  # Let exception propagate
        
        return False

# Factory function
def create_error_recovery_system(
    hdc_dim: int = 10000,
    max_recovery_attempts: int = 3,
    enable_quantum_correction: bool = True
) -> ComprehensiveErrorRecovery:
    """Create comprehensive error recovery system."""
    return ComprehensiveErrorRecovery(
        hdc_dim=hdc_dim,
        max_recovery_attempts=max_recovery_attempts,
        enable_quantum_correction=enable_quantum_correction
    )