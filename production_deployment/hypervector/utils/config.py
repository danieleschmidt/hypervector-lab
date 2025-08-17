"""Configuration management for hyperdimensional computing."""

import os
import json
import yaml
from typing import Dict, Any, Optional, Union
from pathlib import Path


class ConfigError(Exception):
    """Configuration-related errors."""
    pass


class Config:
    """Configuration manager for HDC system."""
    
    DEFAULT_CONFIG = {
        'hypervector': {
            'default_dim': 10000,
            'default_mode': 'dense',
            'default_device': 'auto'  # 'auto', 'cpu', 'cuda'
        },
        'encoders': {
            'text': {
                'max_positions': 512,
                'cache_size': 1000
            },
            'vision': {
                'patch_size': 16,
                'image_size': 224
            },
            'eeg': {
                'max_channels': 128,
                'max_time_steps': 1024,
                'freq_bands': {
                    'delta': [0.5, 4],
                    'theta': [4, 8], 
                    'alpha': [8, 13],
                    'beta': [13, 30],
                    'gamma': [30, 100]
                }
            }
        },
        'applications': {
            'bci': {
                'default_channels': 64,
                'default_sampling_rate': 250.0,
                'default_window_size': 250,
                'adaptation_rate': 0.1
            },
            'retrieval': {
                'default_top_k': 10,
                'cache_size': 10000
            }
        },
        'performance': {
            'batch_size': 32,
            'num_workers': 4,
            'memory_limit_gb': 8
        },
        'logging': {
            'level': 'INFO',
            'log_file': None,
            'metrics_enabled': True
        },
        'security': {
            'validate_inputs': True,
            'sanitize_file_paths': True,
            'max_file_size_mb': 100
        }
    }
    
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """Initialize configuration.
        
        Args:
            config_dict: Configuration dictionary (uses defaults if None)
        """
        self._config = self.DEFAULT_CONFIG.copy()
        if config_dict:
            self._deep_update(self._config, config_dict)
    
    def _deep_update(self, base_dict: Dict, update_dict: Dict) -> None:
        """Deep update nested dictionary."""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """Get configuration value using dot notation.
        
        Args:
            key_path: Dot-separated key path (e.g., 'hypervector.default_dim')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key_path.split('.')
        current = self._config
        
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default
        
        return current
    
    def set(self, key_path: str, value: Any) -> None:
        """Set configuration value using dot notation.
        
        Args:
            key_path: Dot-separated key path
            value: Value to set
        """
        keys = key_path.split('.')
        current = self._config
        
        # Navigate to parent of target key
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        # Set the value
        current[keys[-1]] = value
    
    def update(self, config_dict: Dict[str, Any]) -> None:
        """Update configuration with new values.
        
        Args:
            config_dict: Dictionary of configuration updates
        """
        self._deep_update(self._config, config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Get configuration as dictionary."""
        return self._config.copy()
    
    def save(self, file_path: Union[str, Path], format: str = 'auto') -> None:
        """Save configuration to file.
        
        Args:
            file_path: Output file path
            format: File format ('json', 'yaml', 'auto')
        """
        file_path = Path(file_path)
        
        if format == 'auto':
            format = file_path.suffix.lower()
            if format == '.yml':
                format = '.yaml'
        
        # Create directory if needed
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format in ['.json']:
            with open(file_path, 'w') as f:
                json.dump(self._config, f, indent=2)
        elif format in ['.yaml', '.yml']:
            with open(file_path, 'w') as f:
                yaml.dump(self._config, f, default_flow_style=False, indent=2)
        else:
            raise ConfigError(f"Unsupported format: {format}")
    
    @classmethod
    def load(cls, file_path: Union[str, Path], format: str = 'auto') -> 'Config':
        """Load configuration from file.
        
        Args:
            file_path: Input file path
            format: File format ('json', 'yaml', 'auto')
            
        Returns:
            Config instance
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise ConfigError(f"Configuration file not found: {file_path}")
        
        if format == 'auto':
            format = file_path.suffix.lower()
            if format == '.yml':
                format = '.yaml'
        
        try:
            if format in ['.json']:
                with open(file_path, 'r') as f:
                    config_dict = json.load(f)
            elif format in ['.yaml', '.yml']:
                with open(file_path, 'r') as f:
                    config_dict = yaml.safe_load(f)
            else:
                raise ConfigError(f"Unsupported format: {format}")
        except Exception as e:
            raise ConfigError(f"Failed to load config from {file_path}: {e}")
        
        return cls(config_dict)
    
    @classmethod
    def from_env(cls, prefix: str = 'HYPERVECTOR_') -> 'Config':
        """Create configuration from environment variables.
        
        Args:
            prefix: Environment variable prefix
            
        Returns:
            Config instance
        """
        config_dict = {}
        
        for key, value in os.environ.items():
            if key.startswith(prefix):
                # Convert HYPERVECTOR_ENCODERS_TEXT_CACHE_SIZE to encoders.text.cache_size
                config_key = key[len(prefix):].lower().replace('_', '.')
                
                # Try to parse as JSON, fallback to string
                try:
                    parsed_value = json.loads(value)
                except (json.JSONDecodeError, ValueError):
                    parsed_value = value
                
                # Set nested value
                keys = config_key.split('.')
                current = config_dict
                for k in keys[:-1]:
                    if k not in current:
                        current[k] = {}
                    current = current[k]
                current[keys[-1]] = parsed_value
        
        return cls(config_dict)
    
    def validate(self) -> None:
        """Validate configuration values."""
        # Validate hypervector dimension
        dim = self.get('hypervector.default_dim')
        if not isinstance(dim, int) or dim <= 0:
            raise ConfigError(f"Invalid hypervector dimension: {dim}")
        
        # Validate device
        device = self.get('hypervector.default_device')
        if device not in ['auto', 'cpu', 'cuda']:
            raise ConfigError(f"Invalid device: {device}")
        
        # Validate logging level
        level = self.get('logging.level')
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if level not in valid_levels:
            raise ConfigError(f"Invalid logging level: {level}")
        
        # Validate memory limit
        memory_limit = self.get('performance.memory_limit_gb')
        if not isinstance(memory_limit, (int, float)) or memory_limit <= 0:
            raise ConfigError(f"Invalid memory limit: {memory_limit}")


def load_config(
    config_path: Optional[Union[str, Path]] = None,
    env_prefix: str = 'HYPERVECTOR_'
) -> Config:
    """Load configuration from file and environment.
    
    Args:
        config_path: Optional configuration file path
        env_prefix: Environment variable prefix
        
    Returns:
        Config instance
    """
    # Start with defaults
    config = Config()
    
    # Load from file if specified
    if config_path:
        file_config = Config.load(config_path)
        config.update(file_config.to_dict())
    
    # Override with environment variables
    env_config = Config.from_env(env_prefix)
    config.update(env_config.to_dict())
    
    # Validate final configuration
    config.validate()
    
    return config


# Global configuration instance
_global_config = None


def get_config() -> Config:
    """Get global configuration instance."""
    global _global_config
    if _global_config is None:
        _global_config = load_config()
    return _global_config


def set_config(config: Config) -> None:
    """Set global configuration instance."""
    global _global_config
    _global_config = config

import json
import os
from typing import Dict, Any

class ConfigManager:
    """Robust configuration management"""
    
    def __init__(self, config_path: str = None):
        self.config_path = config_path or os.getenv('HDC_CONFIG_PATH', 'config.json')
        self.config = self.load_config()
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration with error handling"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    return json.load(f)
            else:
                logging.info(f"Config file {self.config_path} not found, using defaults")
                return self.get_default_config()
        except Exception as e:
            logging.error(f"Failed to load config: {e}")
            return self.get_default_config()
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'default_dim': 10000,
            'default_device': 'auto',
            'logging_level': 'INFO',
            'performance_monitoring': True,
            'memory_limit_mb': 1024
        }
    
    def get(self, key: str, default=None):
        """Get configuration value with fallback"""
        return self.config.get(key, default)
    
    def validate_config(self):
        """Validate configuration parameters"""
        required_keys = ['default_dim', 'default_device']
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required config key: {key}")
