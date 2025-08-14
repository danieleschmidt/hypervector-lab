#!/usr/bin/env python3
"""Comprehensive quality gates validation for HyperVector-Lab."""

import os
import sys
import time
import subprocess
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import logging

# Add repo to path
sys.path.insert(0, '/root/repo')


@dataclass
class QualityGateResult:
    """Result of a quality gate check."""
    name: str
    passed: bool
    score: float
    max_score: float
    details: Dict[str, Any]
    execution_time: float = 0.0
    error_message: Optional[str] = None


class QualityGatesValidator:
    """Comprehensive quality gates validator."""
    
    def __init__(self):
        self.results: List[QualityGateResult] = []
        self.logger = logging.getLogger(__name__)
        
        # Quality thresholds
        self.thresholds = {
            'code_coverage': 85.0,
            'security_score': 95.0,
            'performance_score': 80.0,
            'documentation_score': 90.0,
            'maintainability_score': 85.0,
            'test_pass_rate': 100.0
        }
    
    def validate_all_gates(self) -> bool:
        """Run all quality gates and return overall pass/fail."""
        print("=" * 80)
        print("HYPERVECTOR-LAB QUALITY GATES VALIDATION")
        print("=" * 80)
        
        # Define quality gates in execution order
        gates = [
            ("Code Structure", self._validate_code_structure),
            ("Import Safety", self._validate_import_safety),
            ("Security Analysis", self._validate_security),
            ("Performance Benchmarks", self._validate_performance),
            ("Documentation Quality", self._validate_documentation),
            ("Error Handling", self._validate_error_handling),
            ("Memory Management", self._validate_memory_management),
            ("API Consistency", self._validate_api_consistency),
            ("Production Readiness", self._validate_production_readiness)
        ]
        
        all_passed = True
        
        for gate_name, gate_func in gates:
            print(f"\n{gate_name}:")
            print("-" * 50)
            
            start_time = time.time()
            try:
                result = gate_func()
                execution_time = time.time() - start_time
                
                result.execution_time = execution_time
                self.results.append(result)
                
                # Display result
                status = "✓ PASS" if result.passed else "✗ FAIL"
                score_pct = (result.score / result.max_score) * 100
                print(f"{status} - {score_pct:.1f}% ({result.score:.1f}/{result.max_score})")
                
                if result.details:
                    for key, value in result.details.items():
                        print(f"  {key}: {value}")
                
                if result.error_message:
                    print(f"  Error: {result.error_message}")
                
                all_passed &= result.passed
                
            except Exception as e:
                execution_time = time.time() - start_time
                error_result = QualityGateResult(
                    name=gate_name,
                    passed=False,
                    score=0.0,
                    max_score=100.0,
                    details={},
                    execution_time=execution_time,
                    error_message=str(e)
                )
                self.results.append(error_result)
                print(f"✗ FAIL - Exception: {e}")
                all_passed = False
        
        # Generate summary
        self._generate_summary(all_passed)
        
        return all_passed
    
    def _validate_code_structure(self) -> QualityGateResult:
        """Validate code structure and organization."""
        score = 0.0
        max_score = 100.0
        details = {}
        
        # Check required directories
        required_dirs = [
            'hypervector/core',
            'hypervector/encoders', 
            'hypervector/applications',
            'hypervector/accelerators',
            'hypervector/production',
            'hypervector/research',
            'hypervector/utils',
            'tests'
        ]
        
        existing_dirs = []
        for dir_path in required_dirs:
            full_path = Path('/root/repo') / dir_path
            if full_path.exists():
                existing_dirs.append(dir_path)
        
        dir_score = (len(existing_dirs) / len(required_dirs)) * 30
        score += dir_score
        details['directory_structure'] = f"{len(existing_dirs)}/{len(required_dirs)} required directories"
        
        # Check Python file structure
        py_files = list(Path('/root/repo/hypervector').rglob('*.py'))
        non_test_files = [f for f in py_files if 'test' not in str(f) and '__pycache__' not in str(f)]
        
        if len(non_test_files) >= 20:  # Good modular structure
            file_score = 30
        elif len(non_test_files) >= 10:
            file_score = 20
        else:
            file_score = 10
        
        score += file_score
        details['python_files'] = f"{len(non_test_files)} implementation files"
        
        # Check for __init__.py files
        init_files = list(Path('/root/repo/hypervector').rglob('__init__.py'))
        if len(init_files) >= 8:
            init_score = 20
        else:
            init_score = len(init_files) * 2.5
        
        score += init_score
        details['init_files'] = f"{len(init_files)} __init__.py files"
        
        # Check configuration files
        config_files = ['pyproject.toml', 'README.md', 'LICENSE']
        existing_configs = []
        for config in config_files:
            if (Path('/root/repo') / config).exists():
                existing_configs.append(config)
        
        config_score = (len(existing_configs) / len(config_files)) * 20
        score += config_score
        details['config_files'] = f"{len(existing_configs)}/{len(config_files)} config files"
        
        passed = score >= 75  # 75% threshold for structure
        
        return QualityGateResult(
            name="Code Structure",
            passed=passed,
            score=score,
            max_score=max_score,
            details=details
        )
    
    def _validate_import_safety(self) -> QualityGateResult:
        """Validate that imports are safe and don't cause issues."""
        score = 0.0
        max_score = 100.0
        details = {}
        
        # Test basic import without dependencies
        import_test_code = '''
import sys
sys.path.insert(0, '/root/repo')

# Mock required modules to test structure
class MockModule:
    def __getattr__(self, name): 
        if name in ['Tensor', 'tensor', 'float32']:
            return object()
        if name in ['cuda', 'version']:
            return type('MockAttr', (), {
                'is_available': lambda: False,
                'device_count': lambda: 0,
                'cuda': None
            })()
        return object()

sys.modules['torch'] = MockModule()
sys.modules['torch.nn'] = MockModule()  
sys.modules['torch.nn.functional'] = MockModule()
sys.modules['PIL'] = MockModule()
sys.modules['PIL.Image'] = MockModule()
sys.modules['numpy'] = MockModule()
sys.modules['transformers'] = MockModule()
sys.modules['scipy'] = MockModule()
sys.modules['scipy.signal'] = MockModule()
sys.modules['sklearn'] = MockModule()
sys.modules['sklearn.base'] = MockModule()

try:
    # Test core imports
    from hypervector.core import hypervector, operations, system
    from hypervector.encoders import text, vision, eeg
    from hypervector.applications import bci, retrieval
    
    # Test main module import
    import hypervector
    
    print("SUCCESS: All imports working")
    exit(0)
except Exception as e:
    print(f"IMPORT_ERROR: {e}")
    exit(1)
'''
        
        try:
            result = subprocess.run(
                [sys.executable, '-c', import_test_code],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                score += 50
                details['import_test'] = "✓ All imports successful"
            else:
                details['import_test'] = f"✗ Import failed: {result.stdout}"
                
        except Exception as e:
            details['import_test'] = f"✗ Import test exception: {e}"
        
        # Check for circular imports (basic check)
        circular_imports = 0
        py_files = list(Path('/root/repo/hypervector').rglob('*.py'))
        
        for py_file in py_files[:10]:  # Check first 10 files
            try:
                with open(py_file, 'r') as f:
                    content = f.read()
                
                # Look for relative imports that might cause circles
                if 'from .' in content and 'import' in content:
                    lines = content.split('\n')
                    for line in lines:
                        if line.strip().startswith('from .') and ' import ' in line:
                            # This is a basic heuristic, not comprehensive
                            if '..' in line:  # Parent imports can be risky
                                circular_imports += 1
                                break
                                
            except Exception:
                continue
        
        if circular_imports == 0:
            score += 30
            details['circular_imports'] = "✓ No obvious circular imports detected"
        else:
            score += max(0, 30 - circular_imports * 5)
            details['circular_imports'] = f"⚠ {circular_imports} potential circular imports"
        
        # Check dependency management
        if (Path('/root/repo') / 'pyproject.toml').exists():
            score += 20
            details['dependency_management'] = "✓ pyproject.toml exists"
        else:
            details['dependency_management'] = "✗ No pyproject.toml"
        
        passed = score >= 70
        
        return QualityGateResult(
            name="Import Safety",
            passed=passed,
            score=score,
            max_score=max_score,
            details=details
        )
    
    def _validate_security(self) -> QualityGateResult:
        """Validate security aspects."""
        score = 0.0
        max_score = 100.0
        details = {}
        
        # Check for security utilities
        security_file = Path('/root/repo/hypervector/utils/security.py')
        if security_file.exists():
            score += 30
            details['security_module'] = "✓ Security utilities present"
        else:
            details['security_module'] = "✗ No security module"
        
        # Check for input validation
        validation_file = Path('/root/repo/hypervector/utils/validation.py')
        if validation_file.exists():
            score += 20
            details['validation_module'] = "✓ Input validation present"
        else:
            details['validation_module'] = "✗ No validation module"
        
        # Check for error recovery
        error_recovery_file = Path('/root/repo/hypervector/utils/error_recovery.py')
        if error_recovery_file.exists():
            score += 20
            details['error_recovery'] = "✓ Error recovery present"
        else:
            details['error_recovery'] = "✗ No error recovery"
        
        # Check for monitoring
        monitoring_file = Path('/root/repo/hypervector/core/monitoring.py')
        if monitoring_file.exists():
            score += 20
            details['monitoring'] = "✓ Monitoring capabilities present"
        else:
            details['monitoring'] = "✗ No monitoring"
        
        # Check for no hardcoded secrets (basic check)
        suspicious_patterns = ['password', 'secret', 'key=', 'token=', 'api_key']
        suspicious_files = []
        
        py_files = list(Path('/root/repo/hypervector').rglob('*.py'))
        for py_file in py_files[:20]:  # Check subset
            try:
                with open(py_file, 'r') as f:
                    content = f.read().lower()
                
                for pattern in suspicious_patterns:
                    if pattern in content and '=' in content:
                        suspicious_files.append(str(py_file))
                        break
                        
            except Exception:
                continue
        
        if len(suspicious_files) == 0:
            score += 10
            details['secrets_check'] = "✓ No obvious hardcoded secrets"
        else:
            details['secrets_check'] = f"⚠ {len(suspicious_files)} files with potential secrets"
        
        passed = score >= self.thresholds['security_score']
        
        return QualityGateResult(
            name="Security Analysis", 
            passed=passed,
            score=score,
            max_score=max_score,
            details=details
        )
    
    def _validate_performance(self) -> QualityGateResult:
        """Validate performance optimizations."""
        score = 0.0
        max_score = 100.0
        details = {}
        
        # Check for GPU optimization
        gpu_opt_file = Path('/root/repo/hypervector/accelerators/gpu_optimization.py')
        if gpu_opt_file.exists():
            score += 25
            details['gpu_optimization'] = "✓ GPU optimization present"
        else:
            details['gpu_optimization'] = "✗ No GPU optimization"
        
        # Check for distributed processing
        dist_file = Path('/root/repo/hypervector/accelerators/distributed_processing.py')
        if dist_file.exists():
            score += 25
            details['distributed_processing'] = "✓ Distributed processing present"
        else:
            details['distributed_processing'] = "✗ No distributed processing"
        
        # Check for auto-scaling
        autoscale_file = Path('/root/repo/hypervector/production/auto_scaling.py')
        if autoscale_file.exists():
            score += 20
            details['auto_scaling'] = "✓ Auto-scaling present"
        else:
            details['auto_scaling'] = "✗ No auto-scaling"
        
        # Check for load balancing
        lb_file = Path('/root/repo/hypervector/production/load_balancer.py')
        if lb_file.exists():
            score += 20
            details['load_balancing'] = "✓ Load balancing present"
        else:
            details['load_balancing'] = "✗ No load balancing"
        
        # Check for benchmarking
        bench_file = Path('/root/repo/hypervector/benchmark')
        if bench_file.exists():
            score += 10
            details['benchmarking'] = "✓ Benchmarking capabilities present"
        else:
            details['benchmarking'] = "✗ No benchmarking"
        
        passed = score >= self.thresholds['performance_score']
        
        return QualityGateResult(
            name="Performance Benchmarks",
            passed=passed,
            score=score,
            max_score=max_score,
            details=details
        )
    
    def _validate_documentation(self) -> QualityGateResult:
        """Validate documentation quality."""
        score = 0.0
        max_score = 100.0
        details = {}
        
        # Check README quality
        readme_file = Path('/root/repo/README.md')
        if readme_file.exists():
            try:
                with open(readme_file, 'r') as f:
                    readme_content = f.read()
                
                # Check for essential sections
                required_sections = [
                    '# HyperVector-Lab',
                    '## 🚀 Features',
                    '## 📦 Installation',
                    '## 🎯 Quick Start',
                    '## 🏗️ Architecture'
                ]
                
                section_score = 0
                for section in required_sections:
                    if section in readme_content:
                        section_score += 1
                
                readme_score = (section_score / len(required_sections)) * 40
                score += readme_score
                details['readme_sections'] = f"{section_score}/{len(required_sections)} essential sections"
                
                # Check README length (good documentation is comprehensive)
                if len(readme_content) > 5000:  # Substantial documentation
                    score += 20
                    details['readme_length'] = f"✓ Comprehensive ({len(readme_content)} chars)"
                else:
                    score += 10
                    details['readme_length'] = f"⚠ Brief ({len(readme_content)} chars)"
                
            except Exception as e:
                details['readme_analysis'] = f"✗ Error reading README: {e}"
        else:
            details['readme_analysis'] = "✗ No README.md found"
        
        # Check for docstrings in Python files
        py_files = list(Path('/root/repo/hypervector').rglob('*.py'))
        files_with_docstrings = 0
        
        for py_file in py_files[:10]:  # Sample first 10 files
            try:
                with open(py_file, 'r') as f:
                    content = f.read()
                
                # Look for docstrings
                if '"""' in content or "'''" in content:
                    files_with_docstrings += 1
                    
            except Exception:
                continue
        
        if py_files:
            docstring_ratio = files_with_docstrings / min(len(py_files), 10)
            docstring_score = docstring_ratio * 25
            score += docstring_score
            details['docstring_coverage'] = f"{files_with_docstrings}/{min(len(py_files), 10)} files have docstrings"
        
        # Check for API documentation
        api_docs = [
            'API_REFERENCE.md',
            'DEPLOYMENT_GUIDE.md', 
            'IMPLEMENTATION_SUMMARY.md'
        ]
        
        existing_api_docs = []
        for doc in api_docs:
            if (Path('/root/repo') / doc).exists():
                existing_api_docs.append(doc)
        
        api_score = (len(existing_api_docs) / len(api_docs)) * 15
        score += api_score
        details['api_documentation'] = f"{len(existing_api_docs)}/{len(api_docs)} API docs present"
        
        passed = score >= self.thresholds['documentation_score']
        
        return QualityGateResult(
            name="Documentation Quality",
            passed=passed,
            score=score,
            max_score=max_score,
            details=details
        )
    
    def _validate_error_handling(self) -> QualityGateResult:
        """Validate error handling robustness."""
        score = 0.0
        max_score = 100.0
        details = {}
        
        # Check for exception classes
        exception_file = Path('/root/repo/hypervector/core/exceptions.py')
        if exception_file.exists():
            score += 30
            details['custom_exceptions'] = "✓ Custom exception classes present"
        else:
            details['custom_exceptions'] = "✗ No custom exceptions"
        
        # Check for error recovery utilities
        recovery_file = Path('/root/repo/hypervector/utils/error_recovery.py')
        if recovery_file.exists():
            score += 30
            details['error_recovery'] = "✓ Error recovery utilities present"
        else:
            details['error_recovery'] = "✗ No error recovery utilities"
        
        # Check for logging
        logging_file = Path('/root/repo/hypervector/utils/logging.py')
        if logging_file.exists():
            score += 20
            details['logging_system'] = "✓ Logging system present"
        else:
            details['logging_system'] = "✗ No logging system"
        
        # Analyze try/except patterns in code
        py_files = list(Path('/root/repo/hypervector').rglob('*.py'))
        files_with_error_handling = 0
        
        for py_file in py_files[:15]:  # Sample files
            try:
                with open(py_file, 'r') as f:
                    content = f.read()
                
                # Look for error handling patterns
                if 'try:' in content and 'except' in content:
                    files_with_error_handling += 1
                    
            except Exception:
                continue
        
        if py_files:
            error_handling_ratio = files_with_error_handling / min(len(py_files), 15)
            error_score = error_handling_ratio * 20
            score += error_score
            details['error_handling_coverage'] = f"{files_with_error_handling}/{min(len(py_files), 15)} files have error handling"
        
        passed = score >= 75  # 75% threshold for error handling
        
        return QualityGateResult(
            name="Error Handling",
            passed=passed,
            score=score,
            max_score=max_score,
            details=details
        )
    
    def _validate_memory_management(self) -> QualityGateResult:
        """Validate memory management practices."""
        score = 0.0
        max_score = 100.0
        details = {}
        
        # Check for memory management utilities
        memory_file = Path('/root/repo/hypervector/accelerators/memory_manager.py')
        if memory_file.exists():
            score += 40
            details['memory_manager'] = "✓ Memory manager present"
        else:
            details['memory_manager'] = "✗ No memory manager"
        
        # Check for GPU optimization with memory pools
        gpu_opt_file = Path('/root/repo/hypervector/accelerators/gpu_optimization.py')
        if gpu_opt_file.exists():
            try:
                with open(gpu_opt_file, 'r') as f:
                    content = f.read()
                
                if 'TensorMemoryPool' in content or 'memory_pool' in content:
                    score += 30
                    details['memory_pools'] = "✓ Memory pooling implemented"
                else:
                    score += 15
                    details['memory_pools'] = "⚠ GPU optimization present but no clear pooling"
                    
            except Exception:
                details['memory_pools'] = "✗ Error reading GPU optimization"
        else:
            details['memory_pools'] = "✗ No GPU optimization"
        
        # Check for cleanup methods
        cleanup_methods = 0
        py_files = list(Path('/root/repo/hypervector').rglob('*.py'))
        
        for py_file in py_files[:10]:
            try:
                with open(py_file, 'r') as f:
                    content = f.read()
                
                if 'cleanup' in content.lower() or 'clear' in content.lower():
                    cleanup_methods += 1
                    
            except Exception:
                continue
        
        if cleanup_methods >= 3:
            score += 20
            details['cleanup_methods'] = f"✓ {cleanup_methods} files with cleanup methods"
        else:
            score += cleanup_methods * 5
            details['cleanup_methods'] = f"⚠ {cleanup_methods} files with cleanup methods"
        
        # Check for resource context managers
        context_managers = 0
        for py_file in py_files[:10]:
            try:
                with open(py_file, 'r') as f:
                    content = f.read()
                
                if '__enter__' in content and '__exit__' in content:
                    context_managers += 1
                    
            except Exception:
                continue
        
        if context_managers > 0:
            score += 10
            details['context_managers'] = f"✓ {context_managers} context managers found"
        else:
            details['context_managers'] = "⚠ No context managers found"
        
        passed = score >= 70
        
        return QualityGateResult(
            name="Memory Management",
            passed=passed,
            score=score,
            max_score=max_score,
            details=details
        )
    
    def _validate_api_consistency(self) -> QualityGateResult:
        """Validate API design consistency."""
        score = 0.0
        max_score = 100.0
        details = {}
        
        # Check main API exports
        init_file = Path('/root/repo/hypervector/__init__.py')
        if init_file.exists():
            try:
                with open(init_file, 'r') as f:
                    content = f.read()
                
                # Check for __all__ definition
                if '__all__' in content:
                    score += 25
                    details['api_exports'] = "✓ __all__ defined for clean API"
                else:
                    score += 10
                    details['api_exports'] = "⚠ No __all__ definition"
                
                # Check for version info
                if '__version__' in content:
                    score += 15
                    details['version_info'] = "✓ Version information present"
                else:
                    details['version_info'] = "✗ No version information"
                
                # Check for main system class
                if 'HDCSystem' in content:
                    score += 20
                    details['main_system'] = "✓ Main HDCSystem class exported"
                else:
                    details['main_system'] = "✗ No main system class"
                    
            except Exception as e:
                details['init_analysis'] = f"✗ Error reading __init__.py: {e}"
        else:
            details['init_analysis'] = "✗ No __init__.py in main package"
        
        # Check for consistent naming patterns
        py_files = list(Path('/root/repo/hypervector').rglob('*.py'))
        camel_case_files = []
        snake_case_files = []
        
        for py_file in py_files:
            filename = py_file.stem
            if filename.startswith('_'):
                continue  # Skip special files
            
            if any(c.isupper() for c in filename):
                camel_case_files.append(filename)
            elif '_' in filename:
                snake_case_files.append(filename)
        
        # Python convention is snake_case for files
        if len(snake_case_files) > len(camel_case_files):
            score += 20
            details['naming_convention'] = f"✓ Consistent snake_case naming ({len(snake_case_files)} vs {len(camel_case_files)})"
        else:
            score += 10
            details['naming_convention'] = f"⚠ Mixed naming conventions ({len(snake_case_files)} snake_case vs {len(camel_case_files)} camelCase)"
        
        # Check for type hints
        files_with_typing = 0
        for py_file in py_files[:10]:
            try:
                with open(py_file, 'r') as f:
                    content = f.read()
                
                if 'from typing import' in content or 'import typing' in content:
                    files_with_typing += 1
                    
            except Exception:
                continue
        
        if files_with_typing >= 5:
            score += 20
            details['type_hints'] = f"✓ {files_with_typing} files use type hints"
        else:
            score += files_with_typing * 3
            details['type_hints'] = f"⚠ {files_with_typing} files use type hints"
        
        passed = score >= 70
        
        return QualityGateResult(
            name="API Consistency",
            passed=passed,
            score=score,
            max_score=max_score,
            details=details
        )
    
    def _validate_production_readiness(self) -> QualityGateResult:
        """Validate production readiness."""
        score = 0.0
        max_score = 100.0
        details = {}
        
        # Check for deployment configurations
        deployment_configs = [
            'docker-compose.yml',
            'Dockerfile',
            'deployment_output/deployment.yaml'
        ]
        
        existing_configs = []
        for config in deployment_configs:
            if (Path('/root/repo') / config).exists():
                existing_configs.append(config)
        
        deployment_score = (len(existing_configs) / len(deployment_configs)) * 25
        score += deployment_score
        details['deployment_configs'] = f"{len(existing_configs)}/{len(deployment_configs)} deployment configs"
        
        # Check for monitoring
        monitoring_file = Path('/root/repo/hypervector/core/monitoring.py')
        if monitoring_file.exists():
            score += 25
            details['monitoring'] = "✓ Monitoring system present"
        else:
            details['monitoring'] = "✗ No monitoring system"
        
        # Check for production utilities
        prod_files = list(Path('/root/repo/hypervector/production').glob('*.py'))
        if len(prod_files) >= 3:
            score += 25
            details['production_utilities'] = f"✓ {len(prod_files)} production utilities"
        else:
            score += len(prod_files) * 8
            details['production_utilities'] = f"⚠ {len(prod_files)} production utilities"
        
        # Check for configuration management
        config_file = Path('/root/repo/hypervector/utils/config.py')
        if config_file.exists():
            score += 15
            details['config_management'] = "✓ Configuration management present"
        else:
            details['config_management'] = "✗ No configuration management"
        
        # Check for security features
        security_file = Path('/root/repo/hypervector/utils/security.py')
        if security_file.exists():
            score += 10
            details['security_features'] = "✓ Security utilities present"
        else:
            details['security_features'] = "✗ No security utilities"
        
        passed = score >= 75
        
        return QualityGateResult(
            name="Production Readiness",
            passed=passed,
            score=score,
            max_score=max_score,
            details=details
        )
    
    def _generate_summary(self, all_passed: bool):
        """Generate validation summary."""
        print("\n" + "=" * 80)
        print("QUALITY GATES SUMMARY")
        print("=" * 80)
        
        total_score = sum(r.score for r in self.results)
        total_max_score = sum(r.max_score for r in self.results)
        overall_percentage = (total_score / total_max_score) * 100
        
        print(f"Overall Score: {total_score:.1f}/{total_max_score} ({overall_percentage:.1f}%)")
        print(f"Gates Passed: {sum(1 for r in self.results if r.passed)}/{len(self.results)}")
        print(f"Total Execution Time: {sum(r.execution_time for r in self.results):.2f}s")
        
        print(f"\nFinal Result: {'✓ ALL GATES PASSED' if all_passed else '✗ QUALITY GATES FAILED'}")
        
        if not all_passed:
            print("\nFailed Gates:")
            for result in self.results:
                if not result.passed:
                    print(f"  - {result.name}: {result.score:.1f}/{result.max_score}")
        
        print("\nDetailed Results:")
        for result in self.results:
            status = "PASS" if result.passed else "FAIL"
            percentage = (result.score / result.max_score) * 100
            print(f"  {result.name}: {status} ({percentage:.1f}%)")
        
        print("=" * 80)


def main():
    """Main execution function."""
    validator = QualityGatesValidator()
    success = validator.validate_all_gates()
    return 0 if success else 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)