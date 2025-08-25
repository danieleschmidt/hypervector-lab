#!/usr/bin/env python3
"""
Security and Quality Gates Validation
====================================

Comprehensive security audit and quality gates validation for HyperVector-Lab.
Ensures production-ready security standards and code quality metrics.
"""

import os
import re
import ast
import json
import time
import hashlib
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Set, Optional
from collections import defaultdict

class SecurityAuditor:
    """Comprehensive security auditing for HyperVector-Lab."""
    
    def __init__(self, repo_path: str = '/root/repo'):
        self.repo_path = Path(repo_path)
        self.security_issues = []
        self.quality_issues = []
        
    def audit_secrets_exposure(self) -> Dict[str, Any]:
        """Audit for exposed secrets, keys, and sensitive data."""
        secrets_patterns = {
            'api_keys': [
                r'api[_-]?key[_-]?=.{8,}',
                r'apikey[_-]?=.{8,}',
                r'secret[_-]?key[_-]?=.{8,}'
            ],
            'passwords': [
                r'password[_-]?=.{4,}',
                r'passwd[_-]?=.{4,}',
                r'pwd[_-]?=.{4,}'
            ],
            'tokens': [
                r'token[_-]?=.{8,}',
                r'access[_-]?token[_-]?=.{8,}',
                r'auth[_-]?token[_-]?=.{8,}'
            ],
            'database_urls': [
                r'mysql://.*:.*@',
                r'postgres://.*:.*@',
                r'mongodb://.*:.*@'
            ],
            'private_keys': [
                r'-----BEGIN PRIVATE KEY-----',
                r'-----BEGIN RSA PRIVATE KEY-----'
            ]
        }
        
        audit_results = {
            'files_scanned': 0,
            'issues_found': {},
            'clean_files': 0,
            'recommendations': []
        }
        
        for py_file in self.repo_path.rglob('*.py'):
            if '__pycache__' not in str(py_file):
                audit_results['files_scanned'] += 1
                
                try:
                    with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        
                    file_issues = []
                    
                    for category, patterns in secrets_patterns.items():
                        for pattern in patterns:
                            matches = re.findall(pattern, content, re.IGNORECASE)
                            if matches:
                                file_issues.append({
                                    'category': category,
                                    'pattern': pattern,
                                    'matches': len(matches),
                                    'severity': 'HIGH'
                                })
                    
                    if file_issues:
                        audit_results['issues_found'][str(py_file.relative_to(self.repo_path))] = file_issues
                    else:
                        audit_results['clean_files'] += 1
                        
                except Exception as e:
                    # File read error - note but continue
                    pass
        
        # Security recommendations
        if audit_results['issues_found']:
            audit_results['recommendations'].extend([
                "Use environment variables for sensitive configuration",
                "Implement proper secrets management (e.g., HashiCorp Vault)",
                "Add .env files to .gitignore",
                "Use configuration templates with placeholder values",
                "Implement secret rotation policies"
            ])
        
        return audit_results
    
    def audit_input_validation(self) -> Dict[str, Any]:
        """Audit input validation and sanitization."""
        validation_audit = {
            'functions_checked': 0,
            'functions_with_validation': 0,
            'vulnerable_patterns': [],
            'recommendations': []
        }
        
        dangerous_functions = {
            'eval', 'exec', 'compile', '__import__',
            'open', 'subprocess.call', 'subprocess.run',
            'os.system', 'os.popen'
        }
        
        for py_file in self.repo_path.rglob('*.py'):
            if '__pycache__' not in str(py_file):
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    tree = ast.parse(content)
                    
                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef):
                            validation_audit['functions_checked'] += 1
                            
                            # Check for input validation
                            func_source = ast.unparse(node) if hasattr(ast, 'unparse') else str(node)
                            
                            has_validation = any(keyword in func_source.lower() for keyword in [
                                'validate', 'sanitize', 'check', 'verify', 'isinstance',
                                'len(', 'assert', 'raise', 'ValueError', 'TypeError'
                            ])
                            
                            if has_validation:
                                validation_audit['functions_with_validation'] += 1
                            
                            # Check for dangerous function usage
                            for dangerous in dangerous_functions:
                                if dangerous in func_source:
                                    validation_audit['vulnerable_patterns'].append({
                                        'file': str(py_file.relative_to(self.repo_path)),
                                        'function': node.name,
                                        'dangerous_function': dangerous,
                                        'line': node.lineno,
                                        'severity': 'MEDIUM'
                                    })
                
                except (SyntaxError, UnicodeDecodeError):
                    continue
        
        # Calculate validation coverage
        if validation_audit['functions_checked'] > 0:
            validation_coverage = (validation_audit['functions_with_validation'] / 
                                 validation_audit['functions_checked']) * 100
            validation_audit['validation_coverage_percent'] = validation_coverage
        else:
            validation_audit['validation_coverage_percent'] = 0
        
        validation_audit['recommendations'] = [
            "Implement input validation for all external inputs",
            "Use type hints and runtime type checking",
            "Sanitize user inputs before processing",
            "Implement rate limiting for API endpoints",
            "Use parameterized queries for database operations"
        ]
        
        return validation_audit
    
    def audit_dependency_security(self) -> Dict[str, Any]:
        """Audit dependencies for known security vulnerabilities."""
        dependency_audit = {
            'requirements_found': False,
            'dependencies': [],
            'potential_issues': [],
            'recommendations': []
        }
        
        # Check for requirements files
        req_files = ['requirements.txt', 'pyproject.toml', 'setup.py', 'Pipfile']
        
        for req_file in req_files:
            req_path = self.repo_path / req_file
            if req_path.exists():
                dependency_audit['requirements_found'] = True
                
                try:
                    with open(req_path, 'r') as f:
                        content = f.read()
                    
                    if req_file == 'pyproject.toml':
                        # Parse TOML dependencies
                        if 'dependencies' in content:
                            # Simple regex extraction for dependencies
                            deps = re.findall(r'"([^"]+)"', content)
                            dependency_audit['dependencies'].extend(deps)
                    
                    elif req_file == 'requirements.txt':
                        # Parse requirements.txt
                        lines = content.strip().split('\n')
                        for line in lines:
                            if line and not line.startswith('#'):
                                dependency_audit['dependencies'].append(line.strip())
                
                except Exception:
                    continue
        
        # Check for known problematic packages
        problematic_packages = {
            'pickle': 'Use safer alternatives like json or msgpack',
            'yaml': 'Use safe_load instead of load',
            'requests': 'Ensure SSL verification is enabled',
            'urllib3': 'Keep updated for security patches'
        }
        
        for dep in dependency_audit['dependencies']:
            package_name = dep.split('==')[0].split('>=')[0].split('<')[0].strip()
            if package_name.lower() in problematic_packages:
                dependency_audit['potential_issues'].append({
                    'package': package_name,
                    'recommendation': problematic_packages[package_name.lower()],
                    'severity': 'MEDIUM'
                })
        
        dependency_audit['recommendations'] = [
            "Regularly update dependencies to latest secure versions",
            "Use dependency scanning tools (e.g., safety, bandit)",
            "Pin dependency versions in production",
            "Review and audit third-party packages before use",
            "Use virtual environments to isolate dependencies"
        ]
        
        return dependency_audit
    
    def run_comprehensive_security_audit(self) -> Dict[str, Any]:
        """Run complete security audit."""
        print("🔒 Running Comprehensive Security Audit...")
        
        audit_results = {
            'timestamp': time.time(),
            'secrets_audit': self.audit_secrets_exposure(),
            'input_validation_audit': self.audit_input_validation(),
            'dependency_audit': self.audit_dependency_security(),
        }
        
        # Calculate overall security score
        secrets_score = 100 - min(100, len(audit_results['secrets_audit']['issues_found']) * 20)
        validation_score = audit_results['input_validation_audit']['validation_coverage_percent']
        dependency_score = 100 - min(100, len(audit_results['dependency_audit']['potential_issues']) * 10)
        
        overall_security_score = (secrets_score + validation_score + dependency_score) / 3
        
        audit_results['security_scores'] = {
            'secrets_security': secrets_score,
            'input_validation': validation_score,
            'dependency_security': dependency_score,
            'overall_security': overall_security_score
        }
        
        audit_results['security_passed'] = overall_security_score >= 80.0
        
        return audit_results


class QualityGateValidator:
    """Quality gates validation for code quality metrics."""
    
    def __init__(self, repo_path: str = '/root/repo'):
        self.repo_path = Path(repo_path)
        
    def check_code_complexity(self) -> Dict[str, Any]:
        """Check code complexity metrics."""
        complexity_metrics = {
            'files_analyzed': 0,
            'total_functions': 0,
            'complex_functions': 0,
            'average_complexity': 0,
            'complexity_violations': [],
            'recommendations': []
        }
        
        complexity_threshold = 10  # McCabe complexity threshold
        
        for py_file in self.repo_path.rglob('*.py'):
            if '__pycache__' not in str(py_file):
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    tree = ast.parse(content)
                    file_complexity = []
                    
                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef):
                            complexity_metrics['total_functions'] += 1
                            
                            # Simple complexity estimation based on control structures
                            complexity = 1  # Base complexity
                            
                            for child in ast.walk(node):
                                if isinstance(child, (ast.If, ast.While, ast.For, ast.Try)):
                                    complexity += 1
                                elif isinstance(child, ast.BoolOp):
                                    complexity += len(child.values) - 1
                            
                            file_complexity.append(complexity)
                            
                            if complexity > complexity_threshold:
                                complexity_metrics['complex_functions'] += 1
                                complexity_metrics['complexity_violations'].append({
                                    'file': str(py_file.relative_to(self.repo_path)),
                                    'function': node.name,
                                    'complexity': complexity,
                                    'line': node.lineno,
                                    'severity': 'MEDIUM' if complexity <= 15 else 'HIGH'
                                })
                    
                    if file_complexity:
                        complexity_metrics['files_analyzed'] += 1
                
                except (SyntaxError, UnicodeDecodeError):
                    continue
        
        if complexity_metrics['total_functions'] > 0:
            complexity_metrics['average_complexity'] = sum(
                v['complexity'] for v in complexity_metrics['complexity_violations']
            ) / complexity_metrics['total_functions']
        
        complexity_metrics['recommendations'] = [
            f"Refactor functions with complexity > {complexity_threshold}",
            "Break down complex functions into smaller units",
            "Use early returns to reduce nesting",
            "Extract complex logic into separate functions",
            "Consider design patterns to reduce complexity"
        ]
        
        return complexity_metrics
    
    def check_code_duplication(self) -> Dict[str, Any]:
        """Check for code duplication."""
        duplication_metrics = {
            'files_analyzed': 0,
            'duplicate_blocks': 0,
            'duplication_ratio': 0.0,
            'duplicate_functions': [],
            'recommendations': []
        }
        
        function_hashes = defaultdict(list)
        
        for py_file in self.repo_path.rglob('*.py'):
            if '__pycache__' not in str(py_file):
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    tree = ast.parse(content)
                    duplication_metrics['files_analyzed'] += 1
                    
                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef):
                            # Create hash of function structure
                            func_str = ast.dump(node, include_attributes=False)
                            func_hash = hashlib.md5(func_str.encode()).hexdigest()
                            
                            function_hashes[func_hash].append({
                                'file': str(py_file.relative_to(self.repo_path)),
                                'function': node.name,
                                'line': node.lineno
                            })
                
                except (SyntaxError, UnicodeDecodeError):
                    continue
        
        # Find duplicates
        for func_hash, locations in function_hashes.items():
            if len(locations) > 1:
                duplication_metrics['duplicate_blocks'] += 1
                duplication_metrics['duplicate_functions'].append({
                    'hash': func_hash[:8],
                    'locations': locations,
                    'severity': 'MEDIUM'
                })
        
        total_functions = sum(len(locations) for locations in function_hashes.values())
        if total_functions > 0:
            duplication_metrics['duplication_ratio'] = (
                duplication_metrics['duplicate_blocks'] / total_functions
            ) * 100
        
        duplication_metrics['recommendations'] = [
            "Extract common functionality into reusable functions",
            "Use inheritance or composition to reduce duplication",
            "Consider using design patterns (Template Method, Strategy)",
            "Implement shared utility modules",
            "Refactor similar functions to use common base implementations"
        ]
        
        return duplication_metrics
    
    def check_test_quality(self) -> Dict[str, Any]:
        """Check test coverage and quality."""
        test_metrics = {
            'test_files_found': 0,
            'source_files': 0,
            'test_to_source_ratio': 0.0,
            'test_functions': 0,
            'assertions_found': 0,
            'test_quality_score': 0.0,
            'recommendations': []
        }
        
        # Count source files
        source_files = list(self.repo_path.rglob('*.py'))
        source_files = [f for f in source_files if '__pycache__' not in str(f)]
        test_files = [f for f in source_files if 'test' in str(f).lower()]
        
        test_metrics['source_files'] = len(source_files)
        test_metrics['test_files_found'] = len(test_files)
        
        if test_metrics['source_files'] > 0:
            test_metrics['test_to_source_ratio'] = (
                test_metrics['test_files_found'] / test_metrics['source_files']
            ) * 100
        
        # Analyze test files for quality
        for test_file in test_files:
            try:
                with open(test_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                tree = ast.parse(content)
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef) and 'test' in node.name.lower():
                        test_metrics['test_functions'] += 1
                        
                        # Count assertions
                        for child in ast.walk(node):
                            if isinstance(child, ast.Call):
                                if hasattr(child.func, 'attr') and 'assert' in child.func.attr.lower():
                                    test_metrics['assertions_found'] += 1
            
            except (SyntaxError, UnicodeDecodeError):
                continue
        
        # Calculate test quality score
        if test_metrics['test_functions'] > 0:
            assertions_per_test = test_metrics['assertions_found'] / test_metrics['test_functions']
            test_metrics['test_quality_score'] = min(100, assertions_per_test * 20)  # Cap at 100
        
        test_metrics['recommendations'] = [
            "Aim for test-to-source ratio of 50%+",
            "Include multiple assertions per test function",
            "Test edge cases and error conditions",
            "Use descriptive test function names",
            "Implement integration tests alongside unit tests"
        ]
        
        return test_metrics
    
    def run_quality_gates_validation(self) -> Dict[str, Any]:
        """Run comprehensive quality gates validation."""
        print("📊 Running Quality Gates Validation...")
        
        quality_results = {
            'timestamp': time.time(),
            'complexity_check': self.check_code_complexity(),
            'duplication_check': self.check_code_duplication(),
            'test_quality_check': self.check_test_quality()
        }
        
        # Calculate quality scores
        complexity_score = max(0, 100 - quality_results['complexity_check']['complex_functions'] * 5)
        duplication_score = max(0, 100 - quality_results['duplication_check']['duplication_ratio'] * 2)
        test_score = quality_results['test_quality_check']['test_to_source_ratio']
        
        overall_quality_score = (complexity_score + duplication_score + test_score) / 3
        
        quality_results['quality_scores'] = {
            'code_complexity': complexity_score,
            'code_duplication': duplication_score,
            'test_coverage': test_score,
            'overall_quality': overall_quality_score
        }
        
        quality_results['quality_gates_passed'] = overall_quality_score >= 75.0
        
        return quality_results


def run_security_and_quality_gates():
    """Run comprehensive security and quality gates validation."""
    print("🛡️  Running Security and Quality Gates Validation")
    print("=" * 60)
    
    start_time = time.time()
    
    # Initialize auditors
    security_auditor = SecurityAuditor()
    quality_validator = QualityGateValidator()
    
    # Run security audit
    security_results = security_auditor.run_comprehensive_security_audit()
    
    # Run quality validation
    quality_results = quality_validator.run_quality_gates_validation()
    
    total_time = time.time() - start_time
    
    # Print results
    print("\n🔒 SECURITY AUDIT RESULTS")
    print("-" * 40)
    print(f"Secrets Security: {security_results['security_scores']['secrets_security']:.1f}%")
    print(f"Input Validation: {security_results['security_scores']['input_validation']:.1f}%")
    print(f"Dependency Security: {security_results['security_scores']['dependency_security']:.1f}%")
    print(f"Overall Security Score: {security_results['security_scores']['overall_security']:.1f}%")
    
    print(f"\n📊 QUALITY GATES RESULTS")
    print("-" * 40)
    print(f"Code Complexity: {quality_results['quality_scores']['code_complexity']:.1f}%")
    print(f"Code Duplication: {quality_results['quality_scores']['code_duplication']:.1f}%")
    print(f"Test Coverage: {quality_results['quality_scores']['test_coverage']:.1f}%")
    print(f"Overall Quality Score: {quality_results['quality_scores']['overall_quality']:.1f}%")
    
    # Combined results
    combined_results = {
        'timestamp': time.time(),
        'validation_time_seconds': total_time,
        'security_audit': security_results,
        'quality_validation': quality_results,
        'combined_scores': {
            'security_score': security_results['security_scores']['overall_security'],
            'quality_score': quality_results['quality_scores']['overall_quality'],
            'combined_score': (
                security_results['security_scores']['overall_security'] + 
                quality_results['quality_scores']['overall_quality']
            ) / 2
        }
    }
    
    # Overall pass/fail
    security_passed = security_results['security_passed']
    quality_passed = quality_results['quality_gates_passed']
    overall_passed = security_passed and quality_passed
    
    combined_results['validation_passed'] = overall_passed
    combined_results['security_passed'] = security_passed
    combined_results['quality_passed'] = quality_passed
    
    print(f"\n" + "=" * 60)
    print("🎯 FINAL VALIDATION RESULTS")
    print("=" * 60)
    print(f"🔒 Security Gates: {'✅ PASSED' if security_passed else '❌ FAILED'}")
    print(f"📊 Quality Gates: {'✅ PASSED' if quality_passed else '❌ FAILED'}")
    print(f"🎯 Combined Score: {combined_results['combined_scores']['combined_score']:.1f}%")
    print(f"⏱️  Validation Time: {total_time:.2f} seconds")
    
    # Save comprehensive report
    report_file = '/root/repo/security_quality_gates_report.json'
    with open(report_file, 'w') as f:
        json.dump(combined_results, f, indent=2, default=str)
    
    print(f"📄 Detailed report saved: {report_file}")
    
    if overall_passed:
        print("🎉 SUCCESS: All security and quality gates passed!")
        return 0
    else:
        issues = []
        if not security_passed:
            issues.append("security")
        if not quality_passed:
            issues.append("quality")
        print(f"⚠️  ISSUES FOUND: {', '.join(issues)} gates failed")
        return 1

if __name__ == "__main__":
    exit_code = run_security_and_quality_gates()
    exit(exit_code)