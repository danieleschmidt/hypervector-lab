"""Load balancing and traffic management for HDC services."""

import time
import random
import threading
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import logging


class BalancingStrategy(Enum):
    """Load balancing strategies."""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_RESPONSE_TIME = "least_response_time"
    RANDOM = "random"
    HASH = "hash"


@dataclass
class ServerNode:
    """Represents a server node in the load balancer."""
    node_id: str
    host: str
    port: int
    weight: float = 1.0
    max_connections: int = 100
    is_healthy: bool = True
    last_health_check: float = field(default_factory=time.time)
    
    # Statistics
    active_connections: int = 0
    total_requests: int = 0
    total_errors: int = 0
    total_response_time: float = 0.0
    response_times: deque = field(default_factory=lambda: deque(maxlen=100))
    
    def __post_init__(self):
        if isinstance(self.response_times, deque) and len(self.response_times) == 0:
            self.response_times = deque(maxlen=100)
    
    @property
    def average_response_time(self) -> float:
        """Get average response time."""
        if not self.response_times:
            return 0.0
        return sum(self.response_times) / len(self.response_times)
    
    @property
    def error_rate(self) -> float:
        """Get error rate."""
        if self.total_requests == 0:
            return 0.0
        return self.total_errors / self.total_requests
    
    @property
    def load_factor(self) -> float:
        """Get current load factor (0.0 to 1.0)."""
        if self.max_connections == 0:
            return 0.0
        return self.active_connections / self.max_connections
    
    def record_request(self, response_time: float, success: bool = True):
        """Record a request result."""
        self.total_requests += 1
        self.total_response_time += response_time
        self.response_times.append(response_time)
        
        if not success:
            self.total_errors += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get server statistics."""
        return {
            'node_id': self.node_id,
            'host': self.host,
            'port': self.port,
            'is_healthy': self.is_healthy,
            'active_connections': self.active_connections,
            'total_requests': self.total_requests,
            'total_errors': self.total_errors,
            'error_rate': self.error_rate,
            'average_response_time': self.average_response_time,
            'load_factor': self.load_factor,
            'weight': self.weight
        }


class HealthChecker:
    """Health checker for server nodes."""
    
    def __init__(self, check_interval: float = 30.0):
        self.check_interval = check_interval
        self.health_checks: Dict[str, Callable[[ServerNode], bool]] = {}
        self._running = False
        self._thread = None
        self._logger = logging.getLogger(__name__)
    
    def register_health_check(self, name: str, check_func: Callable[[ServerNode], bool]):
        """Register a health check function."""
        self.health_checks[name] = check_func
    
    def start(self, nodes: Dict[str, ServerNode]):
        """Start health checking."""
        if self._running:
            return
        
        self._running = True
        self.nodes = nodes
        self._thread = threading.Thread(target=self._health_check_loop, daemon=True)
        self._thread.start()
        self._logger.info("Health checker started")
    
    def stop(self):
        """Stop health checking."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=10)
        self._logger.info("Health checker stopped")
    
    def _health_check_loop(self):
        """Main health check loop."""
        while self._running:
            try:
                for node in self.nodes.values():
                    self._check_node_health(node)
                
                time.sleep(self.check_interval)
                
            except Exception as e:
                self._logger.error(f"Health check error: {e}")
                time.sleep(min(self.check_interval, 10.0))
    
    def _check_node_health(self, node: ServerNode):
        """Check health of a single node."""
        try:
            # Run all registered health checks
            is_healthy = True
            
            for check_name, check_func in self.health_checks.items():
                try:
                    if not check_func(node):
                        is_healthy = False
                        self._logger.warning(
                            f"Health check '{check_name}' failed for node {node.node_id}"
                        )
                        break
                except Exception as e:
                    is_healthy = False
                    self._logger.error(
                        f"Health check '{check_name}' error for node {node.node_id}: {e}"
                    )
                    break
            
            # Update node health status
            if node.is_healthy != is_healthy:
                node.is_healthy = is_healthy
                status = "healthy" if is_healthy else "unhealthy"
                self._logger.info(f"Node {node.node_id} is now {status}")
            
            node.last_health_check = time.time()
            
        except Exception as e:
            self._logger.error(f"Error checking node {node.node_id}: {e}")
            node.is_healthy = False


class LoadBalancer:
    """Advanced load balancer for HDC services."""
    
    def __init__(self, strategy: BalancingStrategy = BalancingStrategy.ROUND_ROBIN):
        self.strategy = strategy
        self.nodes: Dict[str, ServerNode] = {}
        self.current_index = 0
        self.health_checker = HealthChecker()
        
        # Circuit breaker state per node
        self.circuit_breakers: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {
                'state': 'CLOSED',  # CLOSED, OPEN, HALF_OPEN
                'failure_count': 0,
                'last_failure_time': 0.0,
                'failure_threshold': 5,
                'recovery_timeout': 60.0
            }
        )
        
        self._lock = threading.Lock()
        self._logger = logging.getLogger(__name__)
    
    def add_node(
        self, 
        node_id: str, 
        host: str, 
        port: int, 
        weight: float = 1.0,
        max_connections: int = 100
    ):
        """Add a server node."""
        with self._lock:
            node = ServerNode(
                node_id=node_id,
                host=host,
                port=port,
                weight=weight,
                max_connections=max_connections
            )
            self.nodes[node_id] = node
            self._logger.info(f"Added node {node_id} ({host}:{port})")
    
    def remove_node(self, node_id: str):
        """Remove a server node."""
        with self._lock:
            if node_id in self.nodes:
                del self.nodes[node_id]
                if node_id in self.circuit_breakers:
                    del self.circuit_breakers[node_id]
                self._logger.info(f"Removed node {node_id}")
    
    def get_node(self, request_key: Optional[str] = None) -> Optional[ServerNode]:
        """Get a server node based on the balancing strategy."""
        with self._lock:
            # Get healthy nodes
            healthy_nodes = [
                node for node in self.nodes.values()
                if node.is_healthy and self._is_circuit_closed(node.node_id)
            ]
            
            if not healthy_nodes:
                self._logger.warning("No healthy nodes available")
                return None
            
            # Apply balancing strategy
            if self.strategy == BalancingStrategy.ROUND_ROBIN:
                return self._round_robin_select(healthy_nodes)
            elif self.strategy == BalancingStrategy.LEAST_CONNECTIONS:
                return self._least_connections_select(healthy_nodes)
            elif self.strategy == BalancingStrategy.WEIGHTED_ROUND_ROBIN:
                return self._weighted_round_robin_select(healthy_nodes)
            elif self.strategy == BalancingStrategy.LEAST_RESPONSE_TIME:
                return self._least_response_time_select(healthy_nodes)
            elif self.strategy == BalancingStrategy.RANDOM:
                return random.choice(healthy_nodes)
            elif self.strategy == BalancingStrategy.HASH:
                return self._hash_select(healthy_nodes, request_key or "")
            else:
                return healthy_nodes[0]  # Fallback
    
    def record_request_start(self, node: ServerNode):
        """Record start of a request."""
        with self._lock:
            node.active_connections += 1
    
    def record_request_end(
        self, 
        node: ServerNode, 
        response_time: float, 
        success: bool = True
    ):
        """Record end of a request."""
        with self._lock:
            node.active_connections = max(0, node.active_connections - 1)
            node.record_request(response_time, success)
            
            # Update circuit breaker
            if success:
                self._record_success(node.node_id)
            else:
                self._record_failure(node.node_id)
    
    def _round_robin_select(self, nodes: List[ServerNode]) -> ServerNode:
        """Round-robin selection."""
        node = nodes[self.current_index % len(nodes)]
        self.current_index += 1
        return node
    
    def _least_connections_select(self, nodes: List[ServerNode]) -> ServerNode:
        """Select node with least active connections."""
        return min(nodes, key=lambda n: n.active_connections)
    
    def _weighted_round_robin_select(self, nodes: List[ServerNode]) -> ServerNode:
        """Weighted round-robin selection."""
        # Build weighted list
        weighted_nodes = []
        for node in nodes:
            weight_count = max(1, int(node.weight * 10))  # Scale weights
            weighted_nodes.extend([node] * weight_count)
        
        if not weighted_nodes:
            return nodes[0]
        
        node = weighted_nodes[self.current_index % len(weighted_nodes)]
        self.current_index += 1
        return node
    
    def _least_response_time_select(self, nodes: List[ServerNode]) -> ServerNode:
        """Select node with lowest average response time."""
        return min(nodes, key=lambda n: n.average_response_time or float('inf'))
    
    def _hash_select(self, nodes: List[ServerNode], key: str) -> ServerNode:
        """Hash-based selection for sticky sessions."""
        hash_value = hash(key)
        index = hash_value % len(nodes)
        return nodes[index]
    
    def _is_circuit_closed(self, node_id: str) -> bool:
        """Check if circuit breaker is closed for node."""
        cb = self.circuit_breakers[node_id]
        
        if cb['state'] == 'CLOSED':
            return True
        elif cb['state'] == 'OPEN':
            # Check if we should try half-open
            if time.time() - cb['last_failure_time'] > cb['recovery_timeout']:
                cb['state'] = 'HALF_OPEN'
                self._logger.info(f"Circuit breaker for {node_id} entering HALF_OPEN")
                return True
            return False
        elif cb['state'] == 'HALF_OPEN':
            return True
        
        return False
    
    def _record_success(self, node_id: str):
        """Record successful request for circuit breaker."""
        cb = self.circuit_breakers[node_id]
        
        if cb['state'] == 'HALF_OPEN':
            cb['state'] = 'CLOSED'
            cb['failure_count'] = 0
            self._logger.info(f"Circuit breaker for {node_id} reset to CLOSED")
        elif cb['state'] == 'CLOSED':
            cb['failure_count'] = max(0, cb['failure_count'] - 1)
    
    def _record_failure(self, node_id: str):
        """Record failed request for circuit breaker."""
        cb = self.circuit_breakers[node_id]
        cb['failure_count'] += 1
        cb['last_failure_time'] = time.time()
        
        if cb['failure_count'] >= cb['failure_threshold']:
            cb['state'] = 'OPEN'
            self._logger.warning(
                f"Circuit breaker for {node_id} opened after {cb['failure_count']} failures"
            )
    
    def start_health_checking(self):
        """Start health checking for all nodes."""
        # Register default health checks
        self.health_checker.register_health_check(
            'basic_connectivity', 
            self._basic_health_check
        )
        self.health_checker.register_health_check(
            'load_check', 
            self._load_health_check
        )
        
        self.health_checker.start(self.nodes)
    
    def stop_health_checking(self):
        """Stop health checking."""
        self.health_checker.stop()
    
    def _basic_health_check(self, node: ServerNode) -> bool:
        """Basic health check - connection and response time."""
        # In a real implementation, this would make an actual HTTP/TCP check
        # For now, we'll use simple heuristics
        
        # Check if error rate is too high
        if node.error_rate > 0.1:  # 10% error rate threshold
            return False
        
        # Check if response time is too high
        if node.average_response_time > 5000:  # 5 second threshold
            return False
        
        return True
    
    def _load_health_check(self, node: ServerNode) -> bool:
        """Load-based health check."""
        # Check if node is overloaded
        if node.load_factor > 0.95:  # 95% capacity threshold
            return False
        
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """Get load balancer statistics."""
        with self._lock:
            node_stats = [node.get_stats() for node in self.nodes.values()]
            
            # Circuit breaker stats
            cb_stats = {
                node_id: {
                    'state': cb['state'],
                    'failure_count': cb['failure_count'],
                    'last_failure_time': cb['last_failure_time']
                }
                for node_id, cb in self.circuit_breakers.items()
            }
            
            # Aggregate stats
            total_requests = sum(node.total_requests for node in self.nodes.values())
            total_errors = sum(node.total_errors for node in self.nodes.values())
            active_connections = sum(node.active_connections for node in self.nodes.values())
            
            return {
                'strategy': self.strategy.value,
                'node_count': len(self.nodes),
                'healthy_nodes': sum(1 for node in self.nodes.values() if node.is_healthy),
                'total_requests': total_requests,
                'total_errors': total_errors,
                'overall_error_rate': total_errors / max(total_requests, 1),
                'active_connections': active_connections,
                'nodes': node_stats,
                'circuit_breakers': cb_stats
            }


class RequestRouter:
    """Route requests to appropriate HDC services."""
    
    def __init__(self):
        self.load_balancers: Dict[str, LoadBalancer] = {}
        self.routing_rules: Dict[str, str] = {}  # pattern -> service_name
        self._logger = logging.getLogger(__name__)
    
    def add_service(
        self, 
        service_name: str, 
        strategy: BalancingStrategy = BalancingStrategy.ROUND_ROBIN
    ):
        """Add a service with its load balancer."""
        self.load_balancers[service_name] = LoadBalancer(strategy)
        self._logger.info(f"Added service {service_name} with strategy {strategy.value}")
    
    def add_routing_rule(self, pattern: str, service_name: str):
        """Add a routing rule."""
        self.routing_rules[pattern] = service_name
        self._logger.info(f"Added routing rule: {pattern} -> {service_name}")
    
    def route_request(self, request_path: str, request_key: Optional[str] = None) -> Optional[ServerNode]:
        """Route a request to appropriate server."""
        # Find matching service
        service_name = None
        for pattern, svc in self.routing_rules.items():
            if pattern in request_path:
                service_name = svc
                break
        
        if not service_name or service_name not in self.load_balancers:
            self._logger.warning(f"No service found for path: {request_path}")
            return None
        
        # Get node from load balancer
        lb = self.load_balancers[service_name]
        return lb.get_node(request_key)
    
    def record_request(
        self, 
        service_name: str, 
        node: ServerNode, 
        response_time: float, 
        success: bool = True
    ):
        """Record request completion."""
        if service_name in self.load_balancers:
            self.load_balancers[service_name].record_request_end(
                node, response_time, success
            )
    
    def get_service_stats(self) -> Dict[str, Any]:
        """Get statistics for all services."""
        return {
            service_name: lb.get_stats()
            for service_name, lb in self.load_balancers.items()
        }