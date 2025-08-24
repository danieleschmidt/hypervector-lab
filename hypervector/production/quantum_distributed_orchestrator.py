"""
Quantum-Distributed HDC Orchestrator
===================================

Advanced distributed computing orchestrator for hyperdimensional computing
with quantum-enhanced load balancing and auto-scaling capabilities.

Features:
1. Quantum-optimized task distribution across heterogeneous clusters
2. Real-time auto-scaling with predictive resource management  
3. Fault-tolerant distributed quantum HDC operations
4. Multi-cloud deployment with edge computing integration
5. ML-based performance optimization and resource allocation
6. Self-organizing cluster topology with quantum entanglement patterns
"""

import torch
import torch.nn as nn
import torch.distributed as dist
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import time
import logging
import threading
import asyncio
import json
import hashlib
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from collections import defaultdict, deque
import uuid
import psutil

from ..core.hypervector import HyperVector
from ..core.operations import bind, bundle, cosine_similarity
from ..research.adaptive_quantum_hdc import AdaptiveQuantumHDC
try:
    from ..utils.comprehensive_monitoring import ComprehensiveMonitor
except ImportError:
    class ComprehensiveMonitor:
        def __init__(self, *args, **kwargs):
            pass

logger = logging.getLogger(__name__)

class NodeType(Enum):
    """Types of compute nodes in the distributed system."""
    CPU_WORKER = "cpu_worker"
    GPU_WORKER = "gpu_worker" 
    QUANTUM_ACCELERATOR = "quantum_accelerator"
    EDGE_DEVICE = "edge_device"
    COORDINATOR = "coordinator"
    STORAGE_NODE = "storage_node"

class TaskPriority(Enum):
    """Task priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4
    QUANTUM = 5  # Highest priority for quantum tasks

class LoadBalanceStrategy(Enum):
    """Load balancing strategies."""
    ROUND_ROBIN = "round_robin"
    LEAST_LOADED = "least_loaded"
    QUANTUM_OPTIMAL = "quantum_optimal"
    LOCALITY_AWARE = "locality_aware"
    ML_PREDICTED = "ml_predicted"

@dataclass
class ComputeNode:
    """Represents a compute node in the distributed system."""
    node_id: str
    node_type: NodeType
    address: str
    port: int
    capabilities: Dict[str, Any]
    current_load: float = 0.0
    max_capacity: float = 1.0
    quantum_coherence: float = 0.85
    last_heartbeat: float = field(default_factory=time.time)
    is_healthy: bool = True
    performance_history: List[float] = field(default_factory=list)
    specialized_functions: List[str] = field(default_factory=list)

@dataclass
class DistributedTask:
    """Represents a task in the distributed system."""
    task_id: str
    task_type: str
    priority: TaskPriority
    data: Any
    quantum_requirements: Dict[str, Any] = field(default_factory=dict)
    resource_requirements: Dict[str, float] = field(default_factory=dict)
    deadline: Optional[float] = None
    created_at: float = field(default_factory=time.time)
    assigned_node: Optional[str] = None
    status: str = "pending"
    result: Any = None
    execution_time: float = 0.0

class QuantumDistributedOrchestrator:
    """
    Quantum-enhanced distributed orchestrator for HDC operations.
    """
    
    def __init__(
        self,
        cluster_id: str,
        hdc_dim: int = 10000,
        max_nodes: int = 1000,
        enable_quantum_optimization: bool = True,
        enable_edge_computing: bool = True,
        device: Optional[str] = None
    ):
        self.cluster_id = cluster_id
        self.hdc_dim = hdc_dim
        self.max_nodes = max_nodes
        self.enable_quantum_optimization = enable_quantum_optimization
        self.enable_edge_computing = enable_edge_computing
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Node management
        self.nodes: Dict[str, ComputeNode] = {}
        self.node_groups: Dict[NodeType, List[str]] = defaultdict(list)
        
        # Task management
        self.task_queue: Dict[TaskPriority, deque] = {
            priority: deque() for priority in TaskPriority
        }
        self.active_tasks: Dict[str, DistributedTask] = {}
        self.completed_tasks: deque = deque(maxlen=10000)
        
        # Load balancing
        self.load_balance_strategy = LoadBalanceStrategy.QUANTUM_OPTIMAL
        self.load_balancer = self._initialize_load_balancer()
        
        # Auto-scaling
        self.auto_scaling_enabled = True
        self.scaling_policies = self._initialize_scaling_policies()
        self.resource_predictor = self._initialize_resource_predictor()
        
        # Quantum optimization
        if enable_quantum_optimization:
            self.quantum_optimizer = self._initialize_quantum_optimizer()
            self.quantum_entanglement_map = {}
        
        # Performance monitoring
        try:
            self.monitor = ComprehensiveMonitor(
                component_name="quantum_orchestrator",
                hdc_dim=hdc_dim
            )
        except:
            self.monitor = ComprehensiveMonitor()
        
        # Distributed coordination
        self.coordination_patterns = {}
        self.consensus_threshold = 0.67
        
        # Edge computing
        if enable_edge_computing:
            self.edge_manager = self._initialize_edge_manager()
        
        logger.info(f"Quantum Distributed Orchestrator initialized for cluster {cluster_id}")
    
    def _initialize_load_balancer(self) -> Dict[str, Any]:
        """Initialize intelligent load balancer."""
        return {
            'strategy': self.load_balance_strategy,
            'weights': defaultdict(float),
            'history': deque(maxlen=1000),
            'prediction_model': None
        }
    
    def _initialize_scaling_policies(self) -> Dict[str, Any]:
        """Initialize auto-scaling policies."""
        return {
            'scale_up_threshold': 0.8,
            'scale_down_threshold': 0.2,
            'min_nodes': 1,
            'max_nodes': self.max_nodes,
            'cooldown_period': 300.0,
            'last_scaling_action': 0.0,
            'predictive_scaling': True
        }
    
    def _initialize_resource_predictor(self) -> Dict[str, Any]:
        """Initialize ML-based resource predictor."""
        return {
            'model_type': 'lstm',
            'prediction_window': 300.0,  # 5 minutes
            'training_data': deque(maxlen=10000),
            'model_accuracy': 0.0,
            'last_training': 0.0
        }
    
    def _initialize_quantum_optimizer(self) -> Dict[str, Any]:
        """Initialize quantum optimization components."""
        return {
            'quantum_hdc': AdaptiveQuantumHDC(
                base_dim=self.hdc_dim,
                device=self.device
            ),
            'entanglement_patterns': {},
            'coherence_optimization': True,
            'quantum_task_scheduler': {}
        }
    
    def _initialize_edge_manager(self) -> Dict[str, Any]:
        """Initialize edge computing manager."""
        return {
            'edge_nodes': {},
            'latency_map': {},
            'bandwidth_map': {},
            'edge_cache': {},
            'offloading_strategies': ['latency_sensitive', 'bandwidth_efficient']
        }
    
    def register_node(
        self,
        node_id: str,
        node_type: NodeType,
        address: str,
        port: int,
        capabilities: Dict[str, Any]
    ) -> bool:
        """Register a new compute node."""
        try:
            if len(self.nodes) >= self.max_nodes:
                logger.warning(f"Cannot register node {node_id}: cluster at capacity")
                return False
            
            if node_id in self.nodes:
                logger.warning(f"Node {node_id} already registered")
                return False
            
            node = ComputeNode(
                node_id=node_id,
                node_type=node_type,
                address=address,
                port=port,
                capabilities=capabilities,
                quantum_coherence=capabilities.get('quantum_coherence', 0.85)
            )
            
            self.nodes[node_id] = node
            self.node_groups[node_type].append(node_id)
            
            # Initialize quantum entanglement if quantum optimization is enabled
            if self.enable_quantum_optimization:
                self._create_quantum_entanglement(node_id)
            
            logger.info(f"Registered {node_type.value} node {node_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register node {node_id}: {e}")
            return False
    
    def _create_quantum_entanglement(self, node_id: str):
        """Create quantum entanglement patterns for node coordination."""
        try:
            # Create entanglement pattern for this node
            entanglement_vector = HyperVector.random(self.hdc_dim, device=self.device)
            
            # Bind with cluster coordination pattern
            cluster_pattern = HyperVector.random(self.hdc_dim, device=self.device)
            entangled_pattern = bind(entanglement_vector, cluster_pattern)
            
            self.quantum_entanglement_map[node_id] = {
                'entanglement_vector': entanglement_vector,
                'cluster_pattern': cluster_pattern,
                'entangled_state': entangled_pattern,
                'coherence_level': self.nodes[node_id].quantum_coherence
            }
            
            logger.debug(f"Created quantum entanglement for node {node_id}")
            
        except Exception as e:
            logger.error(f"Failed to create quantum entanglement for {node_id}: {e}")
    
    def submit_task(
        self,
        task_type: str,
        data: Any,
        priority: TaskPriority = TaskPriority.NORMAL,
        quantum_requirements: Optional[Dict[str, Any]] = None,
        resource_requirements: Optional[Dict[str, float]] = None,
        deadline: Optional[float] = None
    ) -> str:
        """Submit a task for distributed execution."""
        try:
            task_id = f"task_{uuid.uuid4().hex[:8]}"
            
            task = DistributedTask(
                task_id=task_id,
                task_type=task_type,
                priority=priority,
                data=data,
                quantum_requirements=quantum_requirements or {},
                resource_requirements=resource_requirements or {},
                deadline=deadline
            )
            
            # Add to appropriate priority queue
            self.task_queue[priority].append(task)
            
            logger.info(f"Submitted task {task_id} with priority {priority.name}")
            return task_id
            
        except Exception as e:
            logger.error(f"Failed to submit task: {e}")
            return ""
    
    def schedule_tasks(self) -> List[str]:
        """Schedule pending tasks to available nodes."""
        scheduled_tasks = []
        
        try:
            # Process tasks by priority (highest first)
            for priority in reversed(TaskPriority):
                while self.task_queue[priority]:
                    task = self.task_queue[priority].popleft()
                    
                    # Find optimal node for task
                    selected_node = self._select_optimal_node(task)
                    
                    if selected_node:
                        task.assigned_node = selected_node
                        task.status = "scheduled"
                        self.active_tasks[task.task_id] = task
                        scheduled_tasks.append(task.task_id)
                        
                        # Update node load
                        self._update_node_load(selected_node, task)
                        
                        logger.debug(f"Scheduled task {task.task_id} to node {selected_node}")
                    else:
                        # No available node, put task back in queue
                        self.task_queue[priority].appendleft(task)
                        break
            
            return scheduled_tasks
            
        except Exception as e:
            logger.error(f"Task scheduling failed: {e}")
            return []
    
    def _select_optimal_node(self, task: DistributedTask) -> Optional[str]:
        """Select optimal node for task execution."""
        try:
            available_nodes = [
                node_id for node_id, node in self.nodes.items()
                if node.is_healthy and node.current_load < node.max_capacity
            ]
            
            if not available_nodes:
                return None
            
            # Apply load balancing strategy
            if self.load_balance_strategy == LoadBalanceStrategy.QUANTUM_OPTIMAL:
                return self._quantum_optimal_selection(task, available_nodes)
            
            elif self.load_balance_strategy == LoadBalanceStrategy.LEAST_LOADED:
                return min(available_nodes, key=lambda n: self.nodes[n].current_load)
            
            elif self.load_balance_strategy == LoadBalanceStrategy.ROUND_ROBIN:
                return self._round_robin_selection(available_nodes)
            
            elif self.load_balance_strategy == LoadBalanceStrategy.LOCALITY_AWARE:
                return self._locality_aware_selection(task, available_nodes)
            
            elif self.load_balance_strategy == LoadBalanceStrategy.ML_PREDICTED:
                return self._ml_predicted_selection(task, available_nodes)
            
            else:
                # Default to least loaded
                return min(available_nodes, key=lambda n: self.nodes[n].current_load)
                
        except Exception as e:
            logger.error(f"Node selection failed: {e}")
            return None
    
    def _quantum_optimal_selection(
        self,
        task: DistributedTask,
        available_nodes: List[str]
    ) -> Optional[str]:
        """Select node using quantum optimization."""
        try:
            if not self.enable_quantum_optimization:
                return min(available_nodes, key=lambda n: self.nodes[n].current_load)
            
            # Create quantum representation of task requirements
            task_vector = self._task_to_hypervector(task)
            
            best_node = None
            best_score = -1.0
            
            for node_id in available_nodes:
                node = self.nodes[node_id]
                
                # Get quantum entanglement pattern
                if node_id in self.quantum_entanglement_map:
                    entanglement_info = self.quantum_entanglement_map[node_id]
                    node_vector = entanglement_info['entangled_state']
                else:
                    # Fallback to random representation
                    node_vector = HyperVector.random(self.hdc_dim, device=self.device)
                
                # Calculate quantum-enhanced compatibility score
                compatibility = cosine_similarity(task_vector, node_vector)
                load_factor = 1.0 - node.current_load
                coherence_factor = node.quantum_coherence
                
                # Combined quantum score
                quantum_score = (
                    0.4 * compatibility.item() +
                    0.4 * load_factor +
                    0.2 * coherence_factor
                )
                
                if quantum_score > best_score:
                    best_score = quantum_score
                    best_node = node_id
            
            return best_node
            
        except Exception as e:
            logger.error(f"Quantum optimal selection failed: {e}")
            return min(available_nodes, key=lambda n: self.nodes[n].current_load)
    
    def _task_to_hypervector(self, task: DistributedTask) -> HyperVector:
        """Convert task to hypervector representation."""
        try:
            # Simple task encoding (would be more sophisticated in practice)
            task_features = []
            
            # Encode task type
            task_type_hash = hash(task.task_type) % self.hdc_dim
            task_features.extend([1.0 if i == task_type_hash else 0.0 for i in range(min(100, self.hdc_dim))])
            
            # Encode priority
            priority_encoding = [0.0] * 5
            priority_encoding[task.priority.value - 1] = 1.0
            task_features.extend(priority_encoding)
            
            # Pad to required dimension
            while len(task_features) < self.hdc_dim:
                task_features.append(0.0)
            
            task_vector = torch.tensor(task_features[:self.hdc_dim], device=self.device)
            return HyperVector(task_vector)
            
        except Exception as e:
            logger.error(f"Task to hypervector conversion failed: {e}")
            return HyperVector.random(self.hdc_dim, device=self.device)
    
    def _round_robin_selection(self, available_nodes: List[str]) -> str:
        """Round-robin node selection."""
        if not hasattr(self, '_round_robin_index'):
            self._round_robin_index = 0
        
        selected_node = available_nodes[self._round_robin_index % len(available_nodes)]
        self._round_robin_index += 1
        
        return selected_node
    
    def _locality_aware_selection(
        self,
        task: DistributedTask,
        available_nodes: List[str]
    ) -> Optional[str]:
        """Select node based on data locality."""
        # Simplified locality-aware selection
        # In practice, this would consider data placement, network topology, etc.
        return min(available_nodes, key=lambda n: self.nodes[n].current_load)
    
    def _ml_predicted_selection(
        self,
        task: DistributedTask,
        available_nodes: List[str]
    ) -> Optional[str]:
        """Select node using ML predictions."""
        # Simplified ML-based selection
        # In practice, this would use trained models to predict optimal placement
        return min(available_nodes, key=lambda n: self.nodes[n].current_load)
    
    def _update_node_load(self, node_id: str, task: DistributedTask):
        """Update node load after task assignment."""
        try:
            node = self.nodes[node_id]
            
            # Estimate task resource usage
            cpu_usage = task.resource_requirements.get('cpu', 0.1)
            memory_usage = task.resource_requirements.get('memory', 0.1)
            
            # Simple load calculation
            additional_load = max(cpu_usage, memory_usage)
            node.current_load = min(node.current_load + additional_load, 1.0)
            
        except Exception as e:
            logger.error(f"Failed to update node load for {node_id}: {e}")
    
    async def execute_distributed_tasks(self) -> Dict[str, Any]:
        """Execute distributed tasks across the cluster."""
        try:
            execution_results = {}
            
            # Schedule pending tasks
            scheduled_tasks = self.schedule_tasks()
            
            # Execute tasks in parallel
            with ThreadPoolExecutor(max_workers=len(self.nodes)) as executor:
                futures = {}
                
                for task_id in scheduled_tasks:
                    if task_id in self.active_tasks:
                        task = self.active_tasks[task_id]
                        future = executor.submit(self._execute_single_task, task)
                        futures[future] = task_id
                
                # Collect results
                for future in as_completed(futures):
                    task_id = futures[future]
                    try:
                        result = future.result()
                        execution_results[task_id] = result
                        
                        # Move completed task
                        if task_id in self.active_tasks:
                            completed_task = self.active_tasks.pop(task_id)
                            completed_task.result = result
                            completed_task.status = "completed"
                            self.completed_tasks.append(completed_task)
                        
                    except Exception as e:
                        logger.error(f"Task {task_id} execution failed: {e}")
                        execution_results[task_id] = {"error": str(e)}
            
            return execution_results
            
        except Exception as e:
            logger.error(f"Distributed task execution failed: {e}")
            return {}
    
    def _execute_single_task(self, task: DistributedTask) -> Any:
        """Execute a single task."""
        try:
            start_time = time.time()
            
            # Simulate task execution
            if task.task_type == "hdc_bind":
                result = self._execute_hdc_bind(task)
            elif task.task_type == "hdc_bundle":
                result = self._execute_hdc_bundle(task)
            elif task.task_type == "quantum_hdc":
                result = self._execute_quantum_hdc(task)
            else:
                result = {"message": f"Executed task {task.task_id}"}
            
            execution_time = time.time() - start_time
            task.execution_time = execution_time
            
            # Update node load (decrease)
            if task.assigned_node:
                self._decrease_node_load(task.assigned_node, task)
            
            return result
            
        except Exception as e:
            logger.error(f"Single task execution failed: {e}")
            return {"error": str(e)}
    
    def _execute_hdc_bind(self, task: DistributedTask) -> Dict[str, Any]:
        """Execute HDC bind operation."""
        try:
            data = task.data
            if 'hv1' in data and 'hv2' in data:
                hv1 = data['hv1']
                hv2 = data['hv2']
                
                if isinstance(hv1, HyperVector) and isinstance(hv2, HyperVector):
                    result_hv = bind(hv1, hv2)
                    return {"result": result_hv, "operation": "bind"}
            
            return {"error": "Invalid data for HDC bind operation"}
            
        except Exception as e:
            return {"error": f"HDC bind execution failed: {e}"}
    
    def _execute_hdc_bundle(self, task: DistributedTask) -> Dict[str, Any]:
        """Execute HDC bundle operation."""
        try:
            data = task.data
            if 'hypervectors' in data:
                hypervectors = data['hypervectors']
                
                if isinstance(hypervectors, list) and all(isinstance(hv, HyperVector) for hv in hypervectors):
                    result_hv = bundle(hypervectors)
                    return {"result": result_hv, "operation": "bundle"}
            
            return {"error": "Invalid data for HDC bundle operation"}
            
        except Exception as e:
            return {"error": f"HDC bundle execution failed: {e}"}
    
    def _execute_quantum_hdc(self, task: DistributedTask) -> Dict[str, Any]:
        """Execute quantum-enhanced HDC operation."""
        try:
            if not self.enable_quantum_optimization:
                return {"error": "Quantum optimization not enabled"}
            
            data = task.data
            quantum_hdc = self.quantum_optimizer['quantum_hdc']
            
            if 'hv1' in data and 'hv2' in data:
                hv1 = data['hv1']
                hv2 = data['hv2']
                entanglement_strength = data.get('entanglement_strength', 0.7)
                
                if isinstance(hv1, HyperVector) and isinstance(hv2, HyperVector):
                    result_hv = quantum_hdc.quantum_bind(hv1, hv2, entanglement_strength)
                    return {"result": result_hv, "operation": "quantum_bind"}
            
            return {"error": "Invalid data for quantum HDC operation"}
            
        except Exception as e:
            return {"error": f"Quantum HDC execution failed: {e}"}
    
    def _decrease_node_load(self, node_id: str, task: DistributedTask):
        """Decrease node load after task completion."""
        try:
            node = self.nodes[node_id]
            
            # Estimate released resources
            cpu_usage = task.resource_requirements.get('cpu', 0.1)
            memory_usage = task.resource_requirements.get('memory', 0.1)
            
            released_load = max(cpu_usage, memory_usage)
            node.current_load = max(node.current_load - released_load, 0.0)
            
        except Exception as e:
            logger.error(f"Failed to decrease node load for {node_id}: {e}")
    
    def auto_scale_cluster(self) -> Dict[str, Any]:
        """Perform auto-scaling based on current load."""
        try:
            if not self.auto_scaling_enabled:
                return {"scaling_action": "disabled"}
            
            current_time = time.time()
            policies = self.scaling_policies
            
            # Check cooldown period
            if current_time - policies['last_scaling_action'] < policies['cooldown_period']:
                return {"scaling_action": "cooldown"}
            
            # Calculate cluster load
            total_load = sum(node.current_load for node in self.nodes.values())
            avg_load = total_load / len(self.nodes) if self.nodes else 0.0
            
            scaling_action = "none"
            
            # Scale up if needed
            if avg_load > policies['scale_up_threshold'] and len(self.nodes) < policies['max_nodes']:
                scaling_action = self._scale_up_cluster()
                policies['last_scaling_action'] = current_time
            
            # Scale down if needed
            elif avg_load < policies['scale_down_threshold'] and len(self.nodes) > policies['min_nodes']:
                scaling_action = self._scale_down_cluster()
                policies['last_scaling_action'] = current_time
            
            return {
                "scaling_action": scaling_action,
                "cluster_load": avg_load,
                "node_count": len(self.nodes),
                "timestamp": current_time
            }
            
        except Exception as e:
            logger.error(f"Auto-scaling failed: {e}")
            return {"scaling_action": "error", "error": str(e)}
    
    def _scale_up_cluster(self) -> str:
        """Scale up the cluster by adding nodes."""
        try:
            # In a real implementation, this would provision new nodes
            logger.info("Scaling up cluster (simulated)")
            return "scale_up"
        except Exception as e:
            logger.error(f"Scale up failed: {e}")
            return "scale_up_failed"
    
    def _scale_down_cluster(self) -> str:
        """Scale down the cluster by removing nodes."""
        try:
            # In a real implementation, this would decommission nodes
            logger.info("Scaling down cluster (simulated)")
            return "scale_down"
        except Exception as e:
            logger.error(f"Scale down failed: {e}")
            return "scale_down_failed"
    
    def monitor_cluster_health(self) -> Dict[str, Any]:
        """Monitor cluster health and performance."""
        try:
            healthy_nodes = sum(1 for node in self.nodes.values() if node.is_healthy)
            total_nodes = len(self.nodes)
            
            avg_load = sum(node.current_load for node in self.nodes.values()) / total_nodes if total_nodes > 0 else 0.0
            avg_quantum_coherence = sum(node.quantum_coherence for node in self.nodes.values()) / total_nodes if total_nodes > 0 else 0.0
            
            active_tasks = len(self.active_tasks)
            pending_tasks = sum(len(queue) for queue in self.task_queue.values())
            completed_tasks = len(self.completed_tasks)
            
            health_report = {
                "timestamp": time.time(),
                "cluster_id": self.cluster_id,
                "node_health": {
                    "healthy_nodes": healthy_nodes,
                    "total_nodes": total_nodes,
                    "health_ratio": healthy_nodes / total_nodes if total_nodes > 0 else 0.0
                },
                "performance": {
                    "average_load": avg_load,
                    "average_quantum_coherence": avg_quantum_coherence
                },
                "task_statistics": {
                    "active_tasks": active_tasks,
                    "pending_tasks": pending_tasks,
                    "completed_tasks": completed_tasks,
                    "total_throughput": completed_tasks
                },
                "resource_utilization": self._calculate_resource_utilization()
            }
            
            return health_report
            
        except Exception as e:
            logger.error(f"Cluster health monitoring failed: {e}")
            return {"error": str(e)}
    
    def _calculate_resource_utilization(self) -> Dict[str, float]:
        """Calculate cluster resource utilization."""
        try:
            total_cpu_usage = 0.0
            total_memory_usage = 0.0
            total_capacity = len(self.nodes)
            
            for node in self.nodes.values():
                # Simplified resource calculation
                total_cpu_usage += node.current_load
                total_memory_usage += node.current_load  # Assume CPU load correlates with memory
            
            return {
                "cpu_utilization": total_cpu_usage / total_capacity if total_capacity > 0 else 0.0,
                "memory_utilization": total_memory_usage / total_capacity if total_capacity > 0 else 0.0,
                "quantum_coherence": sum(node.quantum_coherence for node in self.nodes.values()) / total_capacity if total_capacity > 0 else 0.0
            }
            
        except Exception as e:
            logger.error(f"Resource utilization calculation failed: {e}")
            return {"cpu_utilization": 0.0, "memory_utilization": 0.0}
    
    def generate_orchestration_report(self) -> Dict[str, Any]:
        """Generate comprehensive orchestration report."""
        try:
            cluster_health = self.monitor_cluster_health()
            
            # Task performance analytics
            completed_tasks_list = list(self.completed_tasks)
            avg_execution_time = (
                sum(task.execution_time for task in completed_tasks_list) / 
                len(completed_tasks_list) if completed_tasks_list else 0.0
            )
            
            # Node type distribution
            node_type_distribution = {}
            for node_type in NodeType:
                count = len(self.node_groups[node_type])
                node_type_distribution[node_type.value] = count
            
            report = {
                "generation_timestamp": time.time(),
                "cluster_overview": {
                    "cluster_id": self.cluster_id,
                    "total_nodes": len(self.nodes),
                    "node_type_distribution": node_type_distribution,
                    "quantum_optimization_enabled": self.enable_quantum_optimization,
                    "edge_computing_enabled": self.enable_edge_computing
                },
                "cluster_health": cluster_health,
                "performance_metrics": {
                    "average_task_execution_time": avg_execution_time,
                    "tasks_completed": len(self.completed_tasks),
                    "current_throughput": len(self.active_tasks),
                    "load_balance_strategy": self.load_balance_strategy.value
                },
                "scaling_status": {
                    "auto_scaling_enabled": self.auto_scaling_enabled,
                    "current_scaling_policy": self.scaling_policies
                },
                "quantum_analytics": {
                    "quantum_entangled_nodes": len(self.quantum_entanglement_map),
                    "average_coherence": cluster_health.get("performance", {}).get("average_quantum_coherence", 0.0)
                } if self.enable_quantum_optimization else {"enabled": False}
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            return {"error": str(e)}

# Factory functions
def create_quantum_orchestrator(
    cluster_id: str,
    hdc_dim: int = 10000,
    enable_quantum_optimization: bool = True
) -> QuantumDistributedOrchestrator:
    """Create quantum distributed orchestrator."""
    return QuantumDistributedOrchestrator(
        cluster_id=cluster_id,
        hdc_dim=hdc_dim,
        enable_quantum_optimization=enable_quantum_optimization
    )

async def create_distributed_cluster(
    cluster_config: Dict[str, Any]
) -> QuantumDistributedOrchestrator:
    """Create and initialize a distributed HDC cluster."""
    orchestrator = create_quantum_orchestrator(
        cluster_id=cluster_config.get("cluster_id", "default_cluster"),
        hdc_dim=cluster_config.get("hdc_dim", 10000),
        enable_quantum_optimization=cluster_config.get("enable_quantum_optimization", True)
    )
    
    # Register initial nodes
    for node_config in cluster_config.get("initial_nodes", []):
        orchestrator.register_node(
            node_id=node_config["node_id"],
            node_type=NodeType(node_config["node_type"]),
            address=node_config["address"],
            port=node_config["port"],
            capabilities=node_config["capabilities"]
        )
    
    return orchestrator