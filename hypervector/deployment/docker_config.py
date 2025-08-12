"""Docker deployment configuration and utilities."""

import os
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict

from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class DockerConfig:
    """Docker deployment configuration."""
    base_image: str = "python:3.12-slim"
    python_version: str = "3.12"
    working_directory: str = "/app"
    exposed_ports: List[int] = None
    environment_variables: Dict[str, str] = None
    install_packages: List[str] = None
    cuda_support: bool = False
    memory_limit: str = "2g"
    cpu_limit: str = "1.0"
    
    def __post_init__(self):
        if self.exposed_ports is None:
            self.exposed_ports = [8000]
        if self.environment_variables is None:
            self.environment_variables = {}
        if self.install_packages is None:
            self.install_packages = []


class DockerfileGenerator:
    """Generate optimized Dockerfiles for HDC applications."""
    
    def __init__(self, config: DockerConfig):
        """Initialize Dockerfile generator.
        
        Args:
            config: Docker configuration
        """
        self.config = config
        logger.info(f"Initialized DockerfileGenerator with base image: {config.base_image}")
    
    def generate_dockerfile(self, output_path: Path) -> Path:
        """Generate Dockerfile.
        
        Args:
            output_path: Path to write Dockerfile
            
        Returns:
            Path to generated Dockerfile
        """
        dockerfile_content = self._build_dockerfile_content()
        
        dockerfile_path = output_path / "Dockerfile"
        dockerfile_path.write_text(dockerfile_content)
        
        logger.info(f"Generated Dockerfile at: {dockerfile_path}")
        return dockerfile_path
    
    def _build_dockerfile_content(self) -> str:
        """Build Dockerfile content."""
        lines = []
        
        # Base image
        lines.append(f"FROM {self.config.base_image}")
        lines.append("")
        
        # Metadata
        lines.append("LABEL maintainer=\"HyperVector Lab Team\"")
        lines.append("LABEL description=\"HyperVector-Lab Production Deployment\"")
        lines.append("LABEL version=\"1.0.0\"")
        lines.append("")
        
        # System packages
        if self.config.install_packages:
            lines.append("# Install system packages")
            lines.append("RUN apt-get update && apt-get install -y \\")
            for i, package in enumerate(self.config.install_packages):
                suffix = " \\" if i < len(self.config.install_packages) - 1 else ""
                lines.append(f"    {package}{suffix}")
            lines.append("    && rm -rf /var/lib/apt/lists/*")
            lines.append("")
        
        # CUDA support
        if self.config.cuda_support:
            lines.append("# CUDA support")
            lines.append("RUN apt-get update && apt-get install -y \\")
            lines.append("    cuda-toolkit-12-0 \\")
            lines.append("    && rm -rf /var/lib/apt/lists/*")
            lines.append("")
        
        # Python dependencies
        lines.append("# Install Python dependencies")
        lines.append("COPY requirements.txt .")
        lines.append("RUN pip install --no-cache-dir -r requirements.txt")
        
        if self.config.cuda_support:
            lines.append("RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu118")
        else:
            lines.append("RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu")
        
        lines.append("")
        
        # Working directory
        lines.append(f"WORKDIR {self.config.working_directory}")
        lines.append("")
        
        # Copy application code
        lines.append("# Copy application code")
        lines.append("COPY . .")
        lines.append("")
        
        # Install hypervector package
        lines.append("# Install HyperVector package")
        lines.append("RUN pip install -e .")
        lines.append("")
        
        # Environment variables
        if self.config.environment_variables:
            lines.append("# Environment variables")
            for key, value in self.config.environment_variables.items():
                lines.append(f"ENV {key}={value}")
            lines.append("")
        
        # Expose ports
        if self.config.exposed_ports:
            lines.append("# Expose ports")
            for port in self.config.exposed_ports:
                lines.append(f"EXPOSE {port}")
            lines.append("")
        
        # Health check
        lines.append("# Health check")
        lines.append("HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \\")
        lines.append("    CMD python -c \"import hypervector; print('Health check passed')\" || exit 1")
        lines.append("")
        
        # User setup for security
        lines.append("# Create non-root user")
        lines.append("RUN groupadd -g 1000 appuser && useradd -r -u 1000 -g appuser appuser")
        lines.append("RUN chown -R appuser:appuser /app")
        lines.append("USER appuser")
        lines.append("")
        
        # Default command
        lines.append("# Default command")
        lines.append("CMD [\"python\", \"-c\", \"import hypervector; print('HyperVector container started successfully')\"]")
        
        return "\n".join(lines)
    
    def generate_docker_compose(self, output_path: Path) -> Path:
        """Generate docker-compose.yml file.
        
        Args:
            output_path: Path to write docker-compose.yml
            
        Returns:
            Path to generated file
        """
        compose_config = {
            'version': '3.8',
            'services': {
                'hypervector-app': {
                    'build': '.',
                    'ports': [f"{port}:{port}" for port in self.config.exposed_ports],
                    'environment': self.config.environment_variables,
                    'volumes': [
                        './data:/app/data',
                        './logs:/app/logs'
                    ],
                    'restart': 'unless-stopped',
                    'mem_limit': self.config.memory_limit,
                    'cpus': self.config.cpu_limit
                }
            },
            'volumes': {
                'hypervector-data': {},
                'hypervector-logs': {}
            }
        }
        
        if self.config.cuda_support:
            compose_config['services']['hypervector-app']['runtime'] = 'nvidia'
            compose_config['services']['hypervector-app']['environment']['NVIDIA_VISIBLE_DEVICES'] = 'all'
        
        compose_path = output_path / "docker-compose.yml"
        
        import yaml
        with open(compose_path, 'w') as f:
            yaml.dump(compose_config, f, default_flow_style=False)
        
        logger.info(f"Generated docker-compose.yml at: {compose_path}")
        return compose_path
    
    def generate_requirements_txt(self, output_path: Path) -> Path:
        """Generate requirements.txt file.
        
        Args:
            output_path: Path to write requirements.txt
            
        Returns:
            Path to generated file
        """
        requirements = [
            "torch>=2.0.0",
            "numpy>=1.21.0",
            "einops>=0.6.0",
            "transformers>=4.20.0",
            "Pillow>=9.0.0",
            "scipy>=1.8.0",
            "scikit-learn>=1.1.0",
            "matplotlib>=3.5.0",
            "tqdm>=4.64.0",
            "fastapi>=0.68.0",
            "uvicorn>=0.15.0",
            "pydantic>=1.8.0",
            "psutil>=5.8.0",
            "pyyaml>=6.0",
        ]
        
        if self.config.cuda_support:
            requirements.extend([
                "cupy-cuda12x>=12.0.0",
                "nvidia-ml-py>=12.0.0",
            ])
        
        requirements_path = output_path / "requirements.txt"
        requirements_path.write_text("\n".join(requirements))
        
        logger.info(f"Generated requirements.txt at: {requirements_path}")
        return requirements_path


class KubernetesGenerator:
    """Generate Kubernetes deployment manifests."""
    
    def __init__(self, config: DockerConfig, app_name: str = "hypervector-app"):
        """Initialize Kubernetes generator.
        
        Args:
            config: Docker configuration
            app_name: Application name
        """
        self.config = config
        self.app_name = app_name
        logger.info(f"Initialized KubernetesGenerator for app: {app_name}")
    
    def generate_deployment(self, output_path: Path) -> Path:
        """Generate Kubernetes deployment manifest.
        
        Args:
            output_path: Path to write deployment.yaml
            
        Returns:
            Path to generated file
        """
        deployment = {
            'apiVersion': 'apps/v1',
            'kind': 'Deployment',
            'metadata': {
                'name': self.app_name,
                'labels': {
                    'app': self.app_name,
                    'version': '1.0.0'
                }
            },
            'spec': {
                'replicas': 3,
                'selector': {
                    'matchLabels': {
                        'app': self.app_name
                    }
                },
                'template': {
                    'metadata': {
                        'labels': {
                            'app': self.app_name
                        }
                    },
                    'spec': {
                        'containers': [{
                            'name': self.app_name,
                            'image': f"{self.app_name}:latest",
                            'ports': [{'containerPort': port} for port in self.config.exposed_ports],
                            'env': [
                                {'name': key, 'value': value}
                                for key, value in self.config.environment_variables.items()
                            ],
                            'resources': {
                                'requests': {
                                    'memory': '1Gi',
                                    'cpu': '0.5'
                                },
                                'limits': {
                                    'memory': self.config.memory_limit,
                                    'cpu': self.config.cpu_limit
                                }
                            },
                            'livenessProbe': {
                                'exec': {
                                    'command': [
                                        'python', '-c',
                                        'import hypervector; print("Health check passed")'
                                    ]
                                },
                                'initialDelaySeconds': 60,
                                'periodSeconds': 30
                            },
                            'readinessProbe': {
                                'exec': {
                                    'command': [
                                        'python', '-c',
                                        'import hypervector; print("Ready")'
                                    ]
                                },
                                'initialDelaySeconds': 30,
                                'periodSeconds': 10
                            }
                        }],
                        'securityContext': {
                            'runAsNonRoot': True,
                            'runAsUser': 1000,
                            'fsGroup': 1000
                        }
                    }
                }
            }
        }
        
        if self.config.cuda_support:
            deployment['spec']['template']['spec']['containers'][0]['resources']['limits']['nvidia.com/gpu'] = 1
        
        deployment_path = output_path / "deployment.yaml"
        
        import yaml
        with open(deployment_path, 'w') as f:
            yaml.dump(deployment, f, default_flow_style=False)
        
        logger.info(f"Generated deployment.yaml at: {deployment_path}")
        return deployment_path
    
    def generate_service(self, output_path: Path) -> Path:
        """Generate Kubernetes service manifest.
        
        Args:
            output_path: Path to write service.yaml
            
        Returns:
            Path to generated file
        """
        service = {
            'apiVersion': 'v1',
            'kind': 'Service',
            'metadata': {
                'name': f"{self.app_name}-service",
                'labels': {
                    'app': self.app_name
                }
            },
            'spec': {
                'selector': {
                    'app': self.app_name
                },
                'ports': [
                    {
                        'port': port,
                        'targetPort': port,
                        'protocol': 'TCP'
                    }
                    for port in self.config.exposed_ports
                ],
                'type': 'LoadBalancer'
            }
        }
        
        service_path = output_path / "service.yaml"
        
        import yaml
        with open(service_path, 'w') as f:
            yaml.dump(service, f, default_flow_style=False)
        
        logger.info(f"Generated service.yaml at: {service_path}")
        return service_path
    
    def generate_configmap(self, output_path: Path, config_data: Dict[str, Any]) -> Path:
        """Generate Kubernetes ConfigMap manifest.
        
        Args:
            output_path: Path to write configmap.yaml
            config_data: Configuration data
            
        Returns:
            Path to generated file
        """
        configmap = {
            'apiVersion': 'v1',
            'kind': 'ConfigMap',
            'metadata': {
                'name': f"{self.app_name}-config",
                'labels': {
                    'app': self.app_name
                }
            },
            'data': {
                key: str(value) for key, value in config_data.items()
            }
        }
        
        configmap_path = output_path / "configmap.yaml"
        
        import yaml
        with open(configmap_path, 'w') as f:
            yaml.dump(configmap, f, default_flow_style=False)
        
        logger.info(f"Generated configmap.yaml at: {configmap_path}")
        return configmap_path


def create_production_deployment(
    output_dir: Path,
    cuda_support: bool = False,
    memory_limit: str = "4g",
    cpu_limit: str = "2.0",
    expose_ports: List[int] = None
) -> Dict[str, Path]:
    """Create complete production deployment configuration.
    
    Args:
        output_dir: Output directory for deployment files
        cuda_support: Enable CUDA support
        memory_limit: Memory limit
        cpu_limit: CPU limit
        expose_ports: Ports to expose
        
    Returns:
        Dictionary mapping file types to generated file paths
    """
    if expose_ports is None:
        expose_ports = [8000]
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create configuration
    config = DockerConfig(
        base_image="python:3.12-slim",
        exposed_ports=expose_ports,
        environment_variables={
            "HYPERVECTOR_LOG_LEVEL": "INFO",
            "HYPERVECTOR_CACHE_SIZE": "1000",
            "HYPERVECTOR_MAX_MEMORY_GB": memory_limit.rstrip('g'),
            "PYTHONUNBUFFERED": "1"
        },
        install_packages=[
            "build-essential",
            "curl",
            "git",
            "libopenblas-dev",
            "libgomp1"
        ] + (["nvidia-cuda-toolkit"] if cuda_support else []),
        cuda_support=cuda_support,
        memory_limit=memory_limit,
        cpu_limit=cpu_limit
    )
    
    # Generate files
    dockerfile_gen = DockerfileGenerator(config)
    k8s_gen = KubernetesGenerator(config)
    
    generated_files = {}
    
    # Docker files
    generated_files['dockerfile'] = dockerfile_gen.generate_dockerfile(output_dir)
    generated_files['docker_compose'] = dockerfile_gen.generate_docker_compose(output_dir)
    generated_files['requirements'] = dockerfile_gen.generate_requirements_txt(output_dir)
    
    # Kubernetes files
    generated_files['k8s_deployment'] = k8s_gen.generate_deployment(output_dir)
    generated_files['k8s_service'] = k8s_gen.generate_service(output_dir)
    generated_files['k8s_configmap'] = k8s_gen.generate_configmap(output_dir, {
        'log_level': 'INFO',
        'cache_size': '1000',
        'max_memory_gb': memory_limit.rstrip('g')
    })
    
    # Create deployment scripts
    deployment_script = output_dir / "deploy.sh"
    deployment_script.write_text("""#!/bin/bash
set -e

echo "🚀 Starting HyperVector deployment..."

# Build Docker image
echo "📦 Building Docker image..."
docker build -t hypervector-app:latest .

# Deploy with docker-compose
echo "🐳 Deploying with Docker Compose..."
docker-compose up -d

echo "✅ Deployment completed!"
echo "🌐 Application should be available on the configured ports"
""")
    deployment_script.chmod(0o755)
    generated_files['deploy_script'] = deployment_script
    
    # Create Kubernetes deployment script
    k8s_script = output_dir / "deploy-k8s.sh"
    k8s_script.write_text("""#!/bin/bash
set -e

echo "🚀 Starting Kubernetes deployment..."

# Apply ConfigMap
echo "⚙️  Applying ConfigMap..."
kubectl apply -f configmap.yaml

# Apply Deployment
echo "🚀 Applying Deployment..."
kubectl apply -f deployment.yaml

# Apply Service
echo "🌐 Applying Service..."
kubectl apply -f service.yaml

echo "✅ Kubernetes deployment completed!"
echo "📊 Check status with: kubectl get pods,services"
""")
    k8s_script.chmod(0o755)
    generated_files['k8s_deploy_script'] = k8s_script
    
    # Create monitoring configuration
    monitoring_config = {
        'prometheus': {
            'enabled': True,
            'port': 9090,
            'metrics_path': '/metrics'
        },
        'grafana': {
            'enabled': True,
            'port': 3000
        },
        'alerts': {
            'memory_threshold': '80%',
            'cpu_threshold': '80%',
            'error_rate_threshold': '5%'
        }
    }
    
    monitoring_path = output_dir / "monitoring.yaml"
    import yaml
    with open(monitoring_path, 'w') as f:
        yaml.dump(monitoring_config, f, default_flow_style=False)
    generated_files['monitoring_config'] = monitoring_path
    
    logger.info(f"✅ Created complete production deployment in: {output_dir}")
    logger.info(f"📁 Generated {len(generated_files)} files:")
    for file_type, file_path in generated_files.items():
        logger.info(f"   - {file_type}: {file_path.name}")
    
    return generated_files