import platform

import time
import logging
from typing import Dict, List, Any

class HealthChecker:
    """System health checker and diagnostics"""
    
    def __init__(self, hdc_system=None):
        self.hdc_system = hdc_system
        self.checks = []
        self.last_check_time = None
        self.health_status = "unknown"
    
    def add_check(self, name: str, check_func, critical: bool = False):
        """Add a health check"""
        self.checks.append({
            'name': name,
            'function': check_func,
            'critical': critical,
            'last_result': None,
            'last_run': None
        })
    
    def run_health_checks(self) -> Dict[str, Any]:
        """Run all health checks"""
        results = {
            'timestamp': time.time(),
            'overall_status': 'healthy',
            'checks': [],
            'errors': [],
            'warnings': []
        }
        
        for check in self.checks:
            try:
                start_time = time.time()
                result = check['function']()
                duration = time.time() - start_time
                
                check_result = {
                    'name': check['name'],
                    'status': 'pass' if result else 'fail',
                    'duration': duration,
                    'critical': check['critical']
                }
                
                if not result:
                    if check['critical']:
                        results['overall_status'] = 'critical'
                        results['errors'].append(f"Critical check failed: {check['name']}")
                    else:
                        if results['overall_status'] == 'healthy':
                            results['overall_status'] = 'warning'
                        results['warnings'].append(f"Check failed: {check['name']}")
                
                check['last_result'] = result
                check['last_run'] = time.time()
                results['checks'].append(check_result)
                
            except Exception as e:
                logging.error(f"Health check {check['name']} failed with exception: {e}")
                results['errors'].append(f"Check exception: {check['name']} - {e}")
                results['overall_status'] = 'critical'
        
        self.health_status = results['overall_status']
        self.last_check_time = time.time()
        return results
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information"""
        info = {
            'health_status': self.health_status,
            'last_check': self.last_check_time,
            'python_version': platform.python_version(),
            'platform': platform.platform(),
        }
        
        try:
            import torch
            info['torch_version'] = torch.__version__
            info['cuda_available'] = torch.cuda.is_available()
            if torch.cuda.is_available():
                info['cuda_device_count'] = torch.cuda.device_count()
        except ImportError:
            info['torch_available'] = False
        
        try:
            import psutil
            info['memory_usage'] = psutil.virtual_memory()._asdict()
            info['cpu_usage'] = psutil.cpu_percent()
        except ImportError:
            info['system_monitoring'] = 'unavailable'
        
        return info

# Default health checks
def check_memory_usage():
    """Check if memory usage is within acceptable limits"""
    try:
        import psutil
        memory = psutil.virtual_memory()
        return memory.percent < 90  # Less than 90% memory usage
    except ImportError:
        return True  # Can't check, assume OK

def check_disk_space():
    """Check if disk space is adequate"""
    try:
        import shutil
        _, _, free = shutil.disk_usage('/')
        free_gb = free / (1024**3)
        return free_gb > 1.0  # At least 1GB free
    except:
        return True

def check_torch_availability():
    """Check if PyTorch is available and working"""
    try:
        import torch
        # Try a simple tensor operation
        x = torch.randn(10)
        y = x + 1
        return True
    except:
        return False
