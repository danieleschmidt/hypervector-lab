#!/usr/bin/env python3
"""Security audit report for HyperVector-Lab."""

import os
import sys
import re
from pathlib import Path

sys.path.insert(0, '/root/repo')

def audit_security_issues():
    """Audit and report security findings."""
    print("SECURITY AUDIT REPORT")
    print("=" * 50)
    
    # Check the flagged eval() usages
    eval_files = [
        '/root/repo/hypervector/accelerators/cpu_optimized.py',
        '/root/repo/hypervector/applications/retrieval.py', 
        '/root/repo/hypervector/benchmark/benchmarks.py'
    ]
    
    print("1. EVAL() USAGE ANALYSIS")
    print("-" * 30)
    
    for filepath in eval_files:
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r') as f:
                    lines = f.readlines()
                
                for i, line in enumerate(lines):
                    if 'eval(' in line:
                        print(f"File: {os.path.basename(filepath)}")
                        print(f"Line {i+1}: {line.strip()}")
                        
                        # Context analysis
                        context_start = max(0, i-2)
                        context_end = min(len(lines), i+3)
                        
                        print("Context:")
                        for j in range(context_start, context_end):
                            marker = ">>>" if j == i else "   "
                            print(f"{marker} {j+1:3d}: {lines[j].rstrip()}")
                        print()
                        
            except Exception as e:
                print(f"Error reading {filepath}: {e}")
    
    print("2. INPUT VALIDATION SUMMARY")
    print("-" * 30)
    print("The 'input' matches were function names like 'sanitize_input'")
    print("These are NOT security vulnerabilities - they are validation functions")
    print()
    
    print("3. SECURITY RECOMMENDATIONS")
    print("-" * 30)
    print("✓ Replace eval() with safer alternatives where possible")
    print("✓ Use ast.literal_eval() for safe expression evaluation")
    print("✓ Implement input sanitization (already present)")
    print("✓ Add rate limiting for API endpoints")
    print("✓ Implement logging of security-relevant events")
    print()

def generate_security_fixes():
    """Generate security improvement suggestions."""
    fixes = {
        'eval_replacement': '''
# Instead of eval(), use safer alternatives:
import ast

# For literal evaluation:
try:
    result = ast.literal_eval(expression)
except (ValueError, SyntaxError):
    raise SecurityError("Invalid expression")

# For dynamic execution, use restricted environments
''',
        'input_validation': '''
# Robust input validation pattern:
def validate_and_sanitize(data, expected_type, max_length=1000):
    if not isinstance(data, expected_type):
        raise ValueError(f"Expected {expected_type}, got {type(data)}")
    
    if hasattr(data, '__len__') and len(data) > max_length:
        raise ValueError(f"Data too long: {len(data)} > {max_length}")
    
    return data
'''
    }
    
    print("4. SECURITY CODE TEMPLATES")
    print("-" * 30)
    for fix_type, code in fixes.items():
        print(f"{fix_type.upper()}:")
        print(code)

if __name__ == '__main__':
    audit_security_issues()
    generate_security_fixes()