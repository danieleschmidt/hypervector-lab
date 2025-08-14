#!/usr/bin/env python3
"""Enhanced validation suite with error handling and security checks."""

import os
import sys
import hashlib
import re
from pathlib import Path

sys.path.insert(0, '/root/repo')

class SecurityValidator:
    """Security validation for code safety."""
    
    DANGEROUS_PATTERNS = [
        r'exec\s*\(',
        r'eval\s*\(',
        r'__import__\s*\(',
        r'subprocess\.call',
        r'os\.system',
        r'shell=True',
        r'pickle\.loads',
        r'yaml\.load\(',
        r'input\s*\(',
        r'raw_input\s*\('
    ]
    
    def __init__(self):
        self.issues = []
    
    def scan_file(self, filepath):
        """Scan a single file for security issues."""
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
            for pattern in self.DANGEROUS_PATTERNS:
                matches = re.finditer(pattern, content, re.IGNORECASE)
                for match in matches:
                    line_num = content[:match.start()].count('\n') + 1
                    self.issues.append({
                        'file': filepath,
                        'line': line_num,
                        'pattern': pattern,
                        'severity': 'HIGH'
                    })
        except Exception as e:
            self.issues.append({
                'file': filepath,
                'error': str(e),
                'severity': 'ERROR'
            })
    
    def scan_directory(self, directory):
        """Scan directory for Python files."""
        for py_file in Path(directory).rglob('*.py'):
            if 'hdc_env' not in str(py_file):  # Skip virtual env
                self.scan_file(str(py_file))
    
    def report(self):
        """Generate security report."""
        if not self.issues:
            return True, "✓ No security issues found"
        
        report = "✗ Security issues detected:\n"
        for issue in self.issues:
            if 'error' in issue:
                report += f"  ERROR in {issue['file']}: {issue['error']}\n"
            else:
                report += f"  {issue['severity']} in {issue['file']}:{issue['line']} - {issue['pattern']}\n"
        
        return False, report

class PerformanceValidator:
    """Performance and efficiency validation."""
    
    def __init__(self):
        self.metrics = {}
    
    def check_import_efficiency(self):
        """Check for efficient import patterns."""
        issues = []
        
        # Check for imports in loops (basic detection)
        for py_file in Path('/root/repo/hypervector').rglob('*.py'):
            try:
                with open(py_file, 'r') as f:
                    lines = f.readlines()
                
                in_loop = False
                for i, line in enumerate(lines):
                    stripped = line.strip()
                    
                    # Detect loop start
                    if any(keyword in stripped for keyword in ['for ', 'while ']):
                        in_loop = True
                    
                    # Detect function/class definition (reset loop state)
                    if stripped.startswith(('def ', 'class ')):
                        in_loop = False
                    
                    # Check for imports in loops
                    if in_loop and stripped.startswith(('import ', 'from ')):
                        issues.append(f"{py_file}:{i+1} - Import inside loop")
                        
            except Exception:
                continue
        
        self.metrics['import_efficiency'] = len(issues) == 0
        return len(issues) == 0, issues
    
    def check_memory_patterns(self):
        """Check for memory-efficient patterns."""
        issues = []
        
        # Look for potential memory issues
        for py_file in Path('/root/repo/hypervector').rglob('*.py'):
            try:
                with open(py_file, 'r') as f:
                    content = f.read()
                
                # Check for list comprehensions vs generators where appropriate
                if re.search(r'\[\s*.*\s+for\s+.*\s+in\s+.*\]', content):
                    # Large list comprehensions might benefit from generators
                    pass  # This is a complex heuristic, skipping for now
                
                # Check for string concatenation in loops
                if '+=' in content and 'str' in content:
                    issues.append(f"{py_file} - Potential string concatenation in loop")
                        
            except Exception:
                continue
        
        self.metrics['memory_patterns'] = len(issues) == 0
        return len(issues) == 0, issues

class RobustnessValidator:
    """Robustness and error handling validation."""
    
    def __init__(self):
        self.results = {}
    
    def check_error_handling(self):
        """Check for proper error handling."""
        files_with_errors = 0
        files_with_proper_handling = 0
        
        for py_file in Path('/root/repo/hypervector').rglob('*.py'):
            try:
                with open(py_file, 'r') as f:
                    content = f.read()
                
                has_exceptions = 'raise ' in content or 'except ' in content
                has_try_except = 'try:' in content and 'except' in content
                
                if has_exceptions:
                    files_with_errors += 1
                    if has_try_except:
                        files_with_proper_handling += 1
                        
            except Exception:
                continue
        
        coverage = files_with_proper_handling / max(files_with_errors, 1)
        self.results['error_handling_coverage'] = coverage
        
        return coverage > 0.7, f"Error handling coverage: {coverage:.2%}"
    
    def check_input_validation(self):
        """Check for input validation patterns."""
        validation_patterns = [
            r'validate_',
            r'assert\s+',
            r'isinstance\s*\(',
            r'if.*is None',
            r'if.*not.*:',
            r'raise.*Error'
        ]
        
        files_with_validation = 0
        total_files = 0
        
        for py_file in Path('/root/repo/hypervector').rglob('*.py'):
            try:
                with open(py_file, 'r') as f:
                    content = f.read()
                
                total_files += 1
                
                for pattern in validation_patterns:
                    if re.search(pattern, content):
                        files_with_validation += 1
                        break
                        
            except Exception:
                continue
        
        coverage = files_with_validation / max(total_files, 1)
        self.results['input_validation_coverage'] = coverage
        
        return coverage > 0.5, f"Input validation coverage: {coverage:.2%}"

def main():
    """Run enhanced validation suite."""
    print("=" * 60)
    print("ENHANCED ROBUSTNESS VALIDATION SUITE")
    print("=" * 60)
    
    all_passed = True
    
    # Security validation
    print("\n1. SECURITY SCAN")
    print("-" * 30)
    security = SecurityValidator()
    security.scan_directory('/root/repo/hypervector')
    security_passed, security_report = security.report()
    print(security_report)
    all_passed &= security_passed
    
    # Performance validation
    print("\n2. PERFORMANCE VALIDATION")
    print("-" * 30)
    perf = PerformanceValidator()
    
    import_ok, import_issues = perf.check_import_efficiency()
    print(f"Import efficiency: {'✓' if import_ok else '✗'}")
    if not import_ok:
        for issue in import_issues[:3]:  # Show first 3
            print(f"  {issue}")
    
    memory_ok, memory_issues = perf.check_memory_patterns()
    print(f"Memory patterns: {'✓' if memory_ok else '✗'}")
    
    all_passed &= import_ok and memory_ok
    
    # Robustness validation
    print("\n3. ROBUSTNESS VALIDATION")
    print("-" * 30)
    robust = RobustnessValidator()
    
    error_ok, error_msg = robust.check_error_handling()
    print(f"Error handling: {'✓' if error_ok else '✗'} - {error_msg}")
    
    validation_ok, validation_msg = robust.check_input_validation()
    print(f"Input validation: {'✓' if validation_ok else '✗'} - {validation_msg}")
    
    all_passed &= error_ok and validation_ok
    
    # Summary
    print("\n" + "=" * 60)
    print(f"ENHANCED VALIDATION RESULT: {'✓ PASSED' if all_passed else '✗ ISSUES DETECTED'}")
    print("=" * 60)
    
    return all_passed

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)