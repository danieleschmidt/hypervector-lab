"""Advanced monitoring and metrics collection for HDC systems."""

import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
import json
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

from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PerformanceMetric:
    """Performance metric data structure."""
    name: str
    value: float
    timestamp: datetime
    tags: Dict[str, str]
    unit: str = "ms"


@dataclass
class SystemHealth:
    """System health status."""
    overall_status: str  # "healthy", "warning", "critical"
    cpu_usage: float
    memory_usage_mb: float
    gpu_usage: Optional[float] = None
    gpu_memory_mb: Optional[float] = None
    active_operations: int = 0
    error_rate: float = 0.0
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class PerformanceMonitor:
    """
    Real-time performance monitoring for HDC operations.
    
    Production enhancement: Comprehensive metrics collection with
    real-time alerting and automatic performance optimization.
    """
    
    def __init__(self, retention_hours: int = 24, sampling_interval: float = 1.0):
        """Initialize performance monitor.
        
        Args:
            retention_hours: How long to retain metrics
            sampling_interval: Sampling interval in seconds
        """
        self.retention_hours = retention_hours
        self.sampling_interval = sampling_interval
        
        # Metrics storage
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=int(retention_hours * 3600 / sampling_interval)))
        self.aggregated_metrics: Dict[str, Dict[str, float]] = defaultdict(dict)
        
        # Performance tracking
        self.operation_timers: Dict[str, float] = {}
        self.operation_counts: Dict[str, int] = defaultdict(int)
        self.error_counts: Dict[str, int] = defaultdict(int)
        
        # Health monitoring
        self.health_history: deque = deque(maxlen=1000)
        self.alert_thresholds = {
            'memory_usage_mb': 8000,  # 8GB
            'error_rate': 0.05,       # 5%
            'response_time_ms': 1000   # 1 second
        }
        
        # Monitoring thread
        self.monitoring_active = False
        self.monitoring_thread = None
        
        logger.info(f"Initialized PerformanceMonitor with {retention_hours}h retention")
    
    def start_monitoring(self):
        """Start background monitoring thread."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        logger.info("Started performance monitoring")
    
    def stop_monitoring(self):
        """Stop background monitoring."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        logger.info("Stopped performance monitoring")
    
    def record_metric(self, name: str, value: float, tags: Optional[Dict[str, str]] = None, unit: str = "ms"):
        """Record a performance metric.
        
        Args:
            name: Metric name
            value: Metric value
            tags: Optional tags for categorization
            unit: Metric unit
        """
        metric = PerformanceMetric(
            name=name,
            value=value,
            timestamp=datetime.now(),
            tags=tags or {},
            unit=unit
        )
        
        self.metrics[name].append(metric)
        self._update_aggregated_metrics(name)
    
    def start_timer(self, operation: str) -> str:
        """Start timing an operation.
        
        Args:
            operation: Operation name
            
        Returns:
            Timer ID for stopping
        """
        timer_id = f"{operation}_{time.time()}"
        self.operation_timers[timer_id] = time.perf_counter()
        return timer_id
    
    def stop_timer(self, timer_id: str, operation: str, success: bool = True):
        """Stop timing an operation and record metrics.
        
        Args:
            timer_id: Timer ID from start_timer
            operation: Operation name
            success: Whether operation succeeded
        """
        if timer_id not in self.operation_timers:
            logger.warning(f"Timer {timer_id} not found")
            return
        
        # Calculate duration
        duration_ms = (time.perf_counter() - self.operation_timers[timer_id]) * 1000
        del self.operation_timers[timer_id]
        
        # Record metrics
        self.record_metric(f"{operation}_duration", duration_ms, {"operation": operation})
        self.operation_counts[operation] += 1
        
        if not success:
            self.error_counts[operation] += 1
            self.record_metric(f"{operation}_error", 1.0, {"operation": operation}, "count")
    
    def time_operation(self, operation: str):
        """Decorator for timing operations.
        
        Args:
            operation: Operation name
        """
        def decorator(func):
            def wrapper(*args, **kwargs):
                timer_id = self.start_timer(operation)
                try:
                    result = func(*args, **kwargs)
                    self.stop_timer(timer_id, operation, success=True)
                    return result
                except Exception as e:
                    self.stop_timer(timer_id, operation, success=False)
                    raise
            return wrapper
        return decorator
    
    def get_system_health(self) -> SystemHealth:
        """Get current system health status."""
        # CPU and memory usage
        cpu_usage = self._get_cpu_usage()
        memory_usage = self._get_memory_usage()
        
        # GPU usage if available
        gpu_usage, gpu_memory = self._get_gpu_usage()
        
        # Calculate error rate
        total_ops = sum(self.operation_counts.values())
        total_errors = sum(self.error_counts.values())
        error_rate = total_errors / total_ops if total_ops > 0 else 0.0
        
        # Determine overall status
        status = "healthy"
        if (memory_usage > self.alert_thresholds['memory_usage_mb'] or 
            error_rate > self.alert_thresholds['error_rate']):
            status = "warning"
        
        if memory_usage > self.alert_thresholds['memory_usage_mb'] * 1.5:
            status = "critical"
        
        health = SystemHealth(
            overall_status=status,
            cpu_usage=cpu_usage,
            memory_usage_mb=memory_usage,
            gpu_usage=gpu_usage,
            gpu_memory_mb=gpu_memory,
            active_operations=len(self.operation_timers),
            error_rate=error_rate
        )
        
        self.health_history.append(health)
        return health
    
    def get_metrics_summary(self, time_window_minutes: int = 60) -> Dict[str, Dict[str, float]]:
        """Get metrics summary for specified time window.
        
        Args:
            time_window_minutes: Time window in minutes
            
        Returns:
            Dict with metric summaries
        """
        cutoff_time = datetime.now() - timedelta(minutes=time_window_minutes)
        summary = {}
        
        for metric_name, metric_deque in self.metrics.items():
            # Filter metrics within time window
            recent_metrics = [m for m in metric_deque if m.timestamp >= cutoff_time]
            
            if not recent_metrics:
                continue
            
            values = [m.value for m in recent_metrics]
            
            summary[metric_name] = {
                'count': len(values),
                'mean': sum(values) / len(values),
                'min': min(values),
                'max': max(values),
                'std': self._calculate_std(values),
                'p50': self._calculate_percentile(values, 0.5),
                'p95': self._calculate_percentile(values, 0.95),
                'p99': self._calculate_percentile(values, 0.99)
            }
        
        return summary
    
    def _monitoring_loop(self):
        """Background monitoring loop."""
        while self.monitoring_active:
            try:
                # Collect system health
                health = self.get_system_health()
                
                # Record system metrics
                self.record_metric("system_cpu_usage", health.cpu_usage, unit="%")
                self.record_metric("system_memory_usage", health.memory_usage_mb, unit="MB")
                
                if health.gpu_usage is not None:
                    self.record_metric("gpu_usage", health.gpu_usage, unit="%")
                    self.record_metric("gpu_memory_usage", health.gpu_memory_mb, unit="MB")
                
                self.record_metric("active_operations", health.active_operations, unit="count")
                self.record_metric("error_rate", health.error_rate, unit="%")
                
                # Check for alerts
                self._check_alerts(health)
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
            
            time.sleep(self.sampling_interval)
    
    def _update_aggregated_metrics(self, metric_name: str):
        """Update aggregated metrics for a given metric."""
        metrics = list(self.metrics[metric_name])
        if not metrics:
            return
        
        values = [m.value for m in metrics]
        
        self.aggregated_metrics[metric_name] = {
            'count': len(values),
            'mean': sum(values) / len(values),
            'min': min(values),
            'max': max(values),
            'latest': values[-1] if values else 0.0
        }
    
    def _get_cpu_usage(self) -> float:
        """Get CPU usage percentage."""
        try:
            import psutil
            return psutil.cpu_percent(interval=0.1)
        except ImportError:
            # Fallback estimation
            return 50.0  # Placeholder
    
    def _get_memory_usage(self) -> float:
        """Get memory usage in MB."""
        try:
            import psutil
            memory = psutil.virtual_memory()
            return memory.used / (1024 ** 2)
        except ImportError:
            # Fallback using torch
            if torch.cuda.is_available():
                return torch.cuda.memory_allocated() / (1024 ** 2)
            return 1000.0  # Placeholder
    
    def _get_gpu_usage(self) -> tuple[Optional[float], Optional[float]]:
        """Get GPU usage and memory."""
        if not torch.cuda.is_available():
            return None, None
        
        try:
            # Get GPU utilization
            gpu_usage = 0.0  # Would need nvidia-ml-py for real GPU utilization
            
            # Get GPU memory
            gpu_memory = torch.cuda.memory_allocated() / (1024 ** 2)
            
            return gpu_usage, gpu_memory
        except Exception:
            return None, None
    
    def _calculate_std(self, values: List[float]) -> float:
        """Calculate standard deviation."""
        if len(values) <= 1:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return variance ** 0.5
    
    def _calculate_percentile(self, values: List[float], percentile: float) -> float:
        """Calculate percentile value."""
        if not values:
            return 0.0
        
        sorted_values = sorted(values)
        index = int(percentile * (len(sorted_values) - 1))
        return sorted_values[index]
    
    def _check_alerts(self, health: SystemHealth):
        """Check for alert conditions."""
        alerts = []
        
        if health.memory_usage_mb > self.alert_thresholds['memory_usage_mb']:
            alerts.append(f"High memory usage: {health.memory_usage_mb:.1f}MB")
        
        if health.error_rate > self.alert_thresholds['error_rate']:
            alerts.append(f"High error rate: {health.error_rate:.1%}")
        
        # Check recent response times
        recent_durations = []
        for metric_name in self.metrics:
            if "duration" in metric_name:
                recent_metrics = list(self.metrics[metric_name])[-10:]  # Last 10
                recent_durations.extend([m.value for m in recent_metrics])
        
        if recent_durations:
            avg_duration = sum(recent_durations) / len(recent_durations)
            if avg_duration > self.alert_thresholds['response_time_ms']:
                alerts.append(f"High response time: {avg_duration:.1f}ms")
        
        # Log alerts
        for alert in alerts:
            logger.warning(f"ALERT: {alert}")


class MetricsCollector:
    """
    Advanced metrics collection and aggregation system.
    
    Production enhancement: Custom metrics with flexible aggregation
    and export capabilities.
    """
    
    def __init__(self, export_interval: int = 300):
        """Initialize metrics collector.
        
        Args:
            export_interval: Export interval in seconds
        """
        self.export_interval = export_interval
        
        # Custom metrics
        self.custom_metrics: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.metric_definitions: Dict[str, Dict[str, Any]] = {}
        
        # Aggregation functions
        self.aggregation_functions = {
            'sum': sum,
            'avg': lambda x: sum(x) / len(x) if x else 0,
            'min': min,
            'max': max,
            'count': len,
            'rate': lambda x: len(x) / (self.export_interval / 60)  # per minute
        }
        
        # Export handlers
        self.export_handlers: List[Callable] = []
        
        logger.info(f"Initialized MetricsCollector with {export_interval}s export interval")
    
    def define_metric(self, name: str, description: str, unit: str = "count", 
                     aggregation: str = "sum", tags: Optional[List[str]] = None):
        """Define a custom metric.
        
        Args:
            name: Metric name
            description: Metric description
            unit: Metric unit
            aggregation: Aggregation method
            tags: Allowed tag keys
        """
        self.metric_definitions[name] = {
            'description': description,
            'unit': unit,
            'aggregation': aggregation,
            'tags': tags or [],
            'created_at': datetime.now()
        }
        
        logger.info(f"Defined metric: {name} ({unit}, {aggregation})")
    
    def record_custom_metric(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Record a custom metric value.
        
        Args:
            name: Metric name
            value: Metric value
            tags: Optional tags
        """
        if name not in self.metric_definitions:
            logger.warning(f"Undefined metric: {name}")
            return
        
        metric_data = {
            'value': value,
            'timestamp': datetime.now(),
            'tags': tags or {}
        }
        
        self.custom_metrics[name].append(metric_data)
    
    def aggregate_metrics(self, time_window_minutes: int = 5) -> Dict[str, Any]:
        """Aggregate metrics over time window.
        
        Args:
            time_window_minutes: Time window for aggregation
            
        Returns:
            Aggregated metrics
        """
        cutoff_time = datetime.now() - timedelta(minutes=time_window_minutes)
        aggregated = {}
        
        for metric_name, metric_data in self.custom_metrics.items():
            if metric_name not in self.metric_definitions:
                continue
            
            # Filter recent data
            recent_data = [
                d for d in metric_data 
                if d['timestamp'] >= cutoff_time
            ]
            
            if not recent_data:
                continue
            
            # Get aggregation function
            aggregation_method = self.metric_definitions[metric_name]['aggregation']
            agg_func = self.aggregation_functions.get(aggregation_method, sum)
            
            # Aggregate by tags
            tag_groups = defaultdict(list)
            for data in recent_data:
                tag_key = json.dumps(data['tags'], sort_keys=True)
                tag_groups[tag_key].append(data['value'])
            
            # Apply aggregation
            aggregated_values = {}
            for tag_key, values in tag_groups.items():
                try:
                    aggregated_values[tag_key] = agg_func(values)
                except Exception as e:
                    logger.error(f"Aggregation failed for {metric_name}: {e}")
                    aggregated_values[tag_key] = 0
            
            aggregated[metric_name] = {
                'values': aggregated_values,
                'definition': self.metric_definitions[metric_name],
                'window_minutes': time_window_minutes,
                'timestamp': datetime.now()
            }
        
        return aggregated
    
    def add_export_handler(self, handler: Callable[[Dict[str, Any]], None]):
        """Add metrics export handler.
        
        Args:
            handler: Function to handle metric export
        """
        self.export_handlers.append(handler)
        logger.info(f"Added export handler: {handler.__name__}")
    
    def export_metrics(self):
        """Export metrics using all registered handlers."""
        aggregated = self.aggregate_metrics()
        
        for handler in self.export_handlers:
            try:
                handler(aggregated)
            except Exception as e:
                logger.error(f"Export handler {handler.__name__} failed: {e}")
    
    def get_metric_summary(self) -> Dict[str, Any]:
        """Get summary of all defined metrics."""
        summary = {
            'definitions': self.metric_definitions,
            'data_points': {name: len(data) for name, data in self.custom_metrics.items()},
            'last_export': datetime.now(),
            'export_handlers': len(self.export_handlers)
        }
        
        return summary


class AlertManager:
    """
    Intelligent alerting system for HDC operations.
    
    Production enhancement: Smart alerting with escalation,
    deduplication, and automatic resolution.
    """
    
    def __init__(self, cooldown_minutes: int = 5):
        """Initialize alert manager.
        
        Args:
            cooldown_minutes: Cooldown period between similar alerts
        """
        self.cooldown_minutes = cooldown_minutes
        
        # Alert rules
        self.alert_rules: Dict[str, Dict[str, Any]] = {}
        
        # Alert history
        self.active_alerts: Dict[str, Dict[str, Any]] = {}
        self.alert_history: List[Dict[str, Any]] = []
        
        # Alert handlers
        self.alert_handlers: List[Callable] = []
        
        logger.info(f"Initialized AlertManager with {cooldown_minutes}min cooldown")
    
    def define_alert_rule(self, name: str, condition: Callable[[Any], bool], 
                         severity: str = "warning", description: str = ""):
        """Define an alert rule.
        
        Args:
            name: Alert rule name
            condition: Function that returns True if alert should fire
            severity: Alert severity (info, warning, critical)
            description: Alert description
        """
        self.alert_rules[name] = {
            'condition': condition,
            'severity': severity,
            'description': description,
            'created_at': datetime.now()
        }
        
        logger.info(f"Defined alert rule: {name} ({severity})")
    
    def check_alerts(self, context: Dict[str, Any]):
        """Check all alert rules against current context.
        
        Args:
            context: Current system context for evaluation
        """
        for rule_name, rule in self.alert_rules.items():
            try:
                should_alert = rule['condition'](context)
                
                if should_alert:
                    self._trigger_alert(rule_name, rule, context)
                else:
                    self._resolve_alert(rule_name)
                    
            except Exception as e:
                logger.error(f"Alert rule {rule_name} evaluation failed: {e}")
    
    def _trigger_alert(self, rule_name: str, rule: Dict[str, Any], context: Dict[str, Any]):
        """Trigger an alert."""
        now = datetime.now()
        
        # Check cooldown
        if rule_name in self.active_alerts:
            last_alert = self.active_alerts[rule_name]['timestamp']
            if (now - last_alert).total_seconds() < self.cooldown_minutes * 60:
                return  # Still in cooldown
        
        # Create alert
        alert = {
            'rule_name': rule_name,
            'severity': rule['severity'],
            'description': rule['description'],
            'context': context.copy(),
            'timestamp': now,
            'resolved': False
        }
        
        # Update active alerts
        self.active_alerts[rule_name] = alert
        
        # Add to history
        self.alert_history.append(alert.copy())
        
        # Notify handlers
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Alert handler failed: {e}")
        
        logger.warning(f"ALERT TRIGGERED: {rule_name} ({rule['severity']})")
    
    def _resolve_alert(self, rule_name: str):
        """Resolve an active alert."""
        if rule_name in self.active_alerts:
            alert = self.active_alerts[rule_name]
            alert['resolved'] = True
            alert['resolved_at'] = datetime.now()
            
            # Remove from active alerts
            del self.active_alerts[rule_name]
            
            logger.info(f"ALERT RESOLVED: {rule_name}")
    
    def add_alert_handler(self, handler: Callable[[Dict[str, Any]], None]):
        """Add alert handler.
        
        Args:
            handler: Function to handle alerts
        """
        self.alert_handlers.append(handler)
        logger.info(f"Added alert handler: {handler.__name__}")
    
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get list of active alerts."""
        return list(self.active_alerts.values())
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get alert system summary."""
        now = datetime.now()
        recent_cutoff = now - timedelta(hours=24)
        
        recent_alerts = [
            alert for alert in self.alert_history 
            if alert['timestamp'] >= recent_cutoff
        ]
        
        summary = {
            'active_alerts': len(self.active_alerts),
            'total_rules': len(self.alert_rules),
            'recent_alerts_24h': len(recent_alerts),
            'alert_handlers': len(self.alert_handlers),
            'severities': {
                'critical': len([a for a in recent_alerts if a['severity'] == 'critical']),
                'warning': len([a for a in recent_alerts if a['severity'] == 'warning']),
                'info': len([a for a in recent_alerts if a['severity'] == 'info'])
            }
        }
        
        return summary


def create_console_alert_handler() -> Callable:
    """Create a console alert handler."""
    def console_handler(alert: Dict[str, Any]):
        severity = alert['severity'].upper()
        timestamp = alert['timestamp'].strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {severity}: {alert['rule_name']} - {alert['description']}")
    
    return console_handler


def create_log_export_handler() -> Callable:
    """Create a log-based metrics export handler."""
    def log_handler(metrics: Dict[str, Any]):
        for metric_name, metric_data in metrics.items():
            values = metric_data['values']
            unit = metric_data['definition']['unit']
            
            for tag_key, value in values.items():
                logger.info(f"METRIC: {metric_name}={value}{unit} tags={tag_key}")
    
    return log_handler