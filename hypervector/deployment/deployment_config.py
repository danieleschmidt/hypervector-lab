"""Production deployment configuration."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import os
from pathlib import Path


@dataclass
class ModelConfig:
    """Configuration for HDC models."""
    model_path: str
    model_type: str  # 'bci', 'retrieval', 'encoder'
    dimension: int = 10000
    device: str = 'cpu'
    batch_size: int = 32
    max_memory_gb: float = 4.0


@dataclass
class ServerConfig:
    """Configuration for model server."""
    host: str = '0.0.0.0'
    port: int = 8080
    workers: int = 4
    max_requests_per_worker: int = 100
    request_timeout: float = 30.0
    keepalive_timeout: float = 2.0
    max_request_size: int = 16 * 1024 * 1024  # 16MB


@dataclass
class MonitoringConfig:
    """Configuration for monitoring and health checks."""
    enable_metrics: bool = True
    metrics_port: int = 9090
    health_check_interval: float = 30.0
    performance_log_interval: float = 60.0
    alert_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'memory_usage_percent': 80.0,
        'cpu_usage_percent': 85.0,
        'response_time_ms': 1000.0,
        'error_rate_percent': 5.0
    })


@dataclass
class SecurityConfig:
    """Configuration for security settings."""
    enable_authentication: bool = False
    api_key_header: str = 'X-API-Key'
    allowed_origins: List[str] = field(default_factory=list)
    rate_limit_requests_per_minute: int = 1000
    enable_request_logging: bool = True
    log_request_bodies: bool = False


@dataclass
class CacheConfig:
    """Configuration for caching."""
    enable_caching: bool = True
    cache_size: int = 1000
    cache_ttl_seconds: float = 3600.0
    memory_cache_size_gb: float = 2.0


@dataclass
class DeploymentConfig:
    """Complete deployment configuration."""
    models: List[ModelConfig] = field(default_factory=list)
    server: ServerConfig = field(default_factory=ServerConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    
    # Environment settings
    environment: str = 'production'  # 'development', 'staging', 'production'
    log_level: str = 'INFO'
    log_file: Optional[str] = None
    
    @classmethod
    def from_env(cls) -> 'DeploymentConfig':
        """Create configuration from environment variables."""
        config = cls()
        
        # Server config from env
        config.server.host = os.getenv('HDC_HOST', config.server.host)
        config.server.port = int(os.getenv('HDC_PORT', config.server.port))
        config.server.workers = int(os.getenv('HDC_WORKERS', config.server.workers))
        
        # Monitoring config from env
        config.monitoring.enable_metrics = os.getenv('HDC_ENABLE_METRICS', 'true').lower() == 'true'
        config.monitoring.metrics_port = int(os.getenv('HDC_METRICS_PORT', config.monitoring.metrics_port))
        
        # Security config from env
        config.security.enable_authentication = os.getenv('HDC_ENABLE_AUTH', 'false').lower() == 'true'
        config.security.rate_limit_requests_per_minute = int(
            os.getenv('HDC_RATE_LIMIT', config.security.rate_limit_requests_per_minute)
        )
        
        # Cache config from env
        config.cache.enable_caching = os.getenv('HDC_ENABLE_CACHE', 'true').lower() == 'true'
        config.cache.cache_size = int(os.getenv('HDC_CACHE_SIZE', config.cache.cache_size))
        
        # Environment settings
        config.environment = os.getenv('HDC_ENVIRONMENT', config.environment)
        config.log_level = os.getenv('HDC_LOG_LEVEL', config.log_level)
        config.log_file = os.getenv('HDC_LOG_FILE', config.log_file)
        
        return config
    
    @classmethod
    def from_file(cls, config_path: str) -> 'DeploymentConfig':
        """Load configuration from file."""
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        if config_path.suffix == '.json':
            import json
            with open(config_path) as f:
                data = json.load(f)
        elif config_path.suffix in ['.yaml', '.yml']:
            import yaml
            with open(config_path) as f:
                data = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")
        
        return cls.from_dict(data)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DeploymentConfig':
        """Create configuration from dictionary."""
        config = cls()
        
        # Update models
        if 'models' in data:
            config.models = [ModelConfig(**model_data) for model_data in data['models']]
        
        # Update server config
        if 'server' in data:
            for key, value in data['server'].items():
                if hasattr(config.server, key):
                    setattr(config.server, key, value)
        
        # Update monitoring config
        if 'monitoring' in data:
            for key, value in data['monitoring'].items():
                if hasattr(config.monitoring, key):
                    setattr(config.monitoring, key, value)
        
        # Update security config
        if 'security' in data:
            for key, value in data['security'].items():
                if hasattr(config.security, key):
                    setattr(config.security, key, value)
        
        # Update cache config
        if 'cache' in data:
            for key, value in data['cache'].items():
                if hasattr(config.cache, key):
                    setattr(config.cache, key, value)
        
        # Update environment settings
        config.environment = data.get('environment', config.environment)
        config.log_level = data.get('log_level', config.log_level)
        config.log_file = data.get('log_file', config.log_file)
        
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'models': [
                {
                    'model_path': model.model_path,
                    'model_type': model.model_type,
                    'dimension': model.dimension,
                    'device': model.device,
                    'batch_size': model.batch_size,
                    'max_memory_gb': model.max_memory_gb
                }
                for model in self.models
            ],
            'server': {
                'host': self.server.host,
                'port': self.server.port,
                'workers': self.server.workers,
                'max_requests_per_worker': self.server.max_requests_per_worker,
                'request_timeout': self.server.request_timeout,
                'keepalive_timeout': self.server.keepalive_timeout,
                'max_request_size': self.server.max_request_size
            },
            'monitoring': {
                'enable_metrics': self.monitoring.enable_metrics,
                'metrics_port': self.monitoring.metrics_port,
                'health_check_interval': self.monitoring.health_check_interval,
                'performance_log_interval': self.monitoring.performance_log_interval,
                'alert_thresholds': self.monitoring.alert_thresholds
            },
            'security': {
                'enable_authentication': self.security.enable_authentication,
                'api_key_header': self.security.api_key_header,
                'allowed_origins': self.security.allowed_origins,
                'rate_limit_requests_per_minute': self.security.rate_limit_requests_per_minute,
                'enable_request_logging': self.security.enable_request_logging,
                'log_request_bodies': self.security.log_request_bodies
            },
            'cache': {
                'enable_caching': self.cache.enable_caching,
                'cache_size': self.cache.cache_size,
                'cache_ttl_seconds': self.cache.cache_ttl_seconds,
                'memory_cache_size_gb': self.cache.memory_cache_size_gb
            },
            'environment': self.environment,
            'log_level': self.log_level,
            'log_file': self.log_file
        }
    
    def save(self, config_path: str) -> None:
        """Save configuration to file."""
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = self.to_dict()
        
        if config_path.suffix == '.json':
            import json
            with open(config_path, 'w') as f:
                json.dump(data, f, indent=2)
        elif config_path.suffix in ['.yaml', '.yml']:
            import yaml
            with open(config_path, 'w') as f:
                yaml.dump(data, f, default_flow_style=False, indent=2)
        else:
            raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")
    
    def validate(self) -> None:
        """Validate configuration."""
        # Validate models
        if not self.models:
            raise ValueError("At least one model must be configured")
        
        for model in self.models:
            if not Path(model.model_path).exists():
                raise FileNotFoundError(f"Model file not found: {model.model_path}")
            
            if model.model_type not in ['bci', 'retrieval', 'encoder']:
                raise ValueError(f"Invalid model type: {model.model_type}")
            
            if model.dimension <= 0:
                raise ValueError(f"Invalid dimension: {model.dimension}")
        
        # Validate server config
        if self.server.port <= 0 or self.server.port > 65535:
            raise ValueError(f"Invalid port: {self.server.port}")
        
        if self.server.workers <= 0:
            raise ValueError(f"Invalid number of workers: {self.server.workers}")
        
        # Validate monitoring config
        if self.monitoring.metrics_port <= 0 or self.monitoring.metrics_port > 65535:
            raise ValueError(f"Invalid metrics port: {self.monitoring.metrics_port}")
        
        # Validate cache config
        if self.cache.cache_size <= 0:
            raise ValueError(f"Invalid cache size: {self.cache.cache_size}")
        
        if self.cache.cache_ttl_seconds <= 0:
            raise ValueError(f"Invalid cache TTL: {self.cache.cache_ttl_seconds}")
    
    def get_model_config(self, model_type: str) -> Optional[ModelConfig]:
        """Get configuration for specific model type."""
        for model in self.models:
            if model.model_type == model_type:
                return model
        return None
    
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment == 'production'
    
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment == 'development'