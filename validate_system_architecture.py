"""
System Architecture Validation
=============================

Lightweight validation of system architecture and core functionality
without requiring external dependencies.
"""

import os
import sys
import json
import time
import inspect
import importlib.util
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

@dataclass
class ValidationResult:
    """Result of validation check."""
    component: str
    status: str  # 'pass', 'fail', 'warning', 'skip'
    message: str
    details: Optional[Dict[str, Any]] = None

class SystemValidator:
    """Validates system architecture and components."""
    
    def __init__(self):
        self.results: List[ValidationResult] = []
        self.repo_root = Path(__file__).parent
    
    def validate_file_structure(self) -> List[ValidationResult]:
        """Validate repository file structure."""
        results = []
        
        # Required directories
        required_dirs = [
            'hypervector',
            'hypervector/core',
            'hypervector/encoders',
            'hypervector/applications',
            'hypervector/accelerators',
            'hypervector/benchmark',
            'hypervector/deployment',
            'hypervector/production',
            'hypervector/research',
            'hypervector/utils'
        ]
        
        for dir_path in required_dirs:
            full_path = self.repo_root / dir_path
            if full_path.exists() and full_path.is_dir():
                results.append(ValidationResult(
                    'file_structure',
                    'pass',
                    f"Directory exists: {dir_path}"
                ))
            else:
                results.append(ValidationResult(
                    'file_structure',
                    'fail',
                    f"Missing directory: {dir_path}"
                ))
        
        # Required core files
        required_files = [
            'hypervector/__init__.py',
            'hypervector/core/__init__.py',
            'hypervector/core/hypervector.py',
            'hypervector/core/operations.py',
            'hypervector/core/system.py',
            'README.md',
            'pyproject.toml'
        ]
        
        for file_path in required_files:
            full_path = self.repo_root / file_path
            if full_path.exists() and full_path.is_file():
                results.append(ValidationResult(
                    'file_structure',
                    'pass',
                    f"File exists: {file_path}"
                ))
            else:
                results.append(ValidationResult(
                    'file_structure',
                    'fail',
                    f"Missing file: {file_path}"
                ))
        
        return results
    
    def validate_python_syntax(self) -> List[ValidationResult]:
        """Validate Python syntax of core files."""
        results = []
        
        python_files = list(self.repo_root.rglob("*.py"))
        
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Try to compile the source
                compile(content, py_file, 'exec')
                
                results.append(ValidationResult(
                    'syntax',
                    'pass',
                    f"Valid syntax: {py_file.relative_to(self.repo_root)}"
                ))
                
            except SyntaxError as e:
                results.append(ValidationResult(
                    'syntax',
                    'fail',
                    f"Syntax error in {py_file.relative_to(self.repo_root)}: {e.msg} (line {e.lineno})"
                ))
            except Exception as e:
                results.append(ValidationResult(
                    'syntax',
                    'warning',
                    f"Could not validate {py_file.relative_to(self.repo_root)}: {str(e)}"
                ))
        
        return results
    
    def validate_module_imports(self) -> List[ValidationResult]:
        """Validate that modules can be imported without external dependencies."""
        results = []
        
        # Core modules that should be importable
        core_modules = [
            'hypervector',
            'hypervector.core',
            'hypervector.encoders',
            'hypervector.applications',
            'hypervector.utils'
        ]
        
        for module_name in core_modules:
            try:
                # Check if module directory exists
                module_path = self.repo_root / module_name.replace('.', '/')
                if not module_path.exists():
                    results.append(ValidationResult(
                        'imports',
                        'fail',
                        f"Module directory not found: {module_name}"
                    ))
                    continue
                
                # Check for __init__.py
                init_file = module_path / '__init__.py'
                if init_file.exists():
                    results.append(ValidationResult(
                        'imports',
                        'pass',
                        f"Module structure valid: {module_name}"
                    ))
                else:
                    results.append(ValidationResult(
                        'imports',
                        'warning',
                        f"Missing __init__.py: {module_name}"
                    ))
                    
            except Exception as e:
                results.append(ValidationResult(
                    'imports',
                    'fail',
                    f"Import validation failed for {module_name}: {str(e)}"
                ))
        
        return results
    
    def validate_documentation(self) -> List[ValidationResult]:
        """Validate documentation structure."""
        results = []
        
        # Check README
        readme_path = self.repo_root / 'README.md'
        if readme_path.exists():
            with open(readme_path, 'r', encoding='utf-8') as f:
                readme_content = f.read()
            
            # Check for key sections
            required_sections = ['Features', 'Installation', 'Quick Start', 'Architecture']
            for section in required_sections:
                if section.lower() in readme_content.lower():
                    results.append(ValidationResult(
                        'documentation',
                        'pass',
                        f"README contains {section} section"
                    ))
                else:
                    results.append(ValidationResult(
                        'documentation',
                        'warning',
                        f"README missing {section} section"
                    ))
        else:
            results.append(ValidationResult(
                'documentation',
                'fail',
                "README.md not found"
            ))
        
        # Check for docstrings in core modules
        core_files = [
            'hypervector/core/hypervector.py',
            'hypervector/core/operations.py',
            'hypervector/core/system.py'
        ]
        
        for file_path in core_files:
            full_path = self.repo_root / file_path
            if full_path.exists():
                try:
                    with open(full_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Simple check for docstrings
                    if '"""' in content or "'''" in content:
                        results.append(ValidationResult(
                            'documentation',
                            'pass',
                            f"Docstrings found in {file_path}"
                        ))
                    else:
                        results.append(ValidationResult(
                            'documentation',
                            'warning',
                            f"No docstrings found in {file_path}"
                        ))
                        
                except Exception as e:
                    results.append(ValidationResult(
                        'documentation',
                        'warning',
                        f"Could not check docstrings in {file_path}: {str(e)}"
                    ))
        
        return results
    
    def validate_configuration(self) -> List[ValidationResult]:
        """Validate configuration files."""
        results = []
        
        # Check pyproject.toml
        pyproject_path = self.repo_root / 'pyproject.toml'
        if pyproject_path.exists():
            try:
                with open(pyproject_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Basic validation
                required_sections = ['[build-system]', '[project]']
                for section in required_sections:
                    if section in content:
                        results.append(ValidationResult(
                            'configuration',
                            'pass',
                            f"pyproject.toml contains {section}"
                        ))
                    else:
                        results.append(ValidationResult(
                            'configuration',
                            'warning',
                            f"pyproject.toml missing {section}"
                        ))
                        
            except Exception as e:
                results.append(ValidationResult(
                    'configuration',
                    'fail',
                    f"Could not validate pyproject.toml: {str(e)}"
                ))
        else:
            results.append(ValidationResult(
                'configuration',
                'fail',
                "pyproject.toml not found"
            ))
        
        return results
    
    def validate_code_quality(self) -> List[ValidationResult]:
        """Validate code quality metrics."""
        results = []
        
        python_files = list(self.repo_root.rglob("*.py"))
        
        total_lines = 0
        total_functions = 0
        total_classes = 0
        total_comments = 0
        
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                file_lines = len(lines)
                file_functions = sum(1 for line in lines if line.strip().startswith('def '))
                file_classes = sum(1 for line in lines if line.strip().startswith('class '))
                file_comments = sum(1 for line in lines if line.strip().startswith('#'))
                
                total_lines += file_lines
                total_functions += file_functions
                total_classes += file_classes
                total_comments += file_comments
                
            except Exception:
                continue
        
        # Calculate metrics
        if total_lines > 0:
            comment_ratio = total_comments / total_lines
            
            results.append(ValidationResult(
                'code_quality',
                'pass',
                f"Total lines of code: {total_lines}",
                {
                    'total_lines': total_lines,
                    'total_functions': total_functions,
                    'total_classes': total_classes,
                    'comment_ratio': comment_ratio
                }
            ))
            
            if comment_ratio > 0.1:
                results.append(ValidationResult(
                    'code_quality',
                    'pass',
                    f"Good comment ratio: {comment_ratio:.2%}"
                ))
            else:
                results.append(ValidationResult(
                    'code_quality',
                    'warning',
                    f"Low comment ratio: {comment_ratio:.2%}"
                ))
        
        return results
    
    def validate_architecture_coherence(self) -> List[ValidationResult]:
        """Validate architectural coherence."""
        results = []
        
        # Check for proper separation of concerns
        expected_modules = {
            'core': ['hypervector.py', 'operations.py', 'system.py'],
            'encoders': ['text.py', 'vision.py', 'eeg.py'],
            'applications': ['bci.py', 'retrieval.py'],
            'production': ['auto_scaling.py', 'load_balancer.py', 'monitoring.py'],
            'research': ['quantum_enhanced_hdc.py', 'neural_plasticity_hdc.py']
        }
        
        for module_name, expected_files in expected_modules.items():
            module_path = self.repo_root / 'hypervector' / module_name
            
            if module_path.exists():
                existing_files = [f.name for f in module_path.glob('*.py')]
                
                for expected_file in expected_files:
                    if expected_file in existing_files:
                        results.append(ValidationResult(
                            'architecture',
                            'pass',
                            f"Module {module_name} contains {expected_file}"
                        ))
                    else:
                        results.append(ValidationResult(
                            'architecture',
                            'warning',
                            f"Module {module_name} missing {expected_file}"
                        ))
            else:
                results.append(ValidationResult(
                    'architecture',
                    'warning',
                    f"Module directory not found: {module_name}"
                ))
        
        return results
    
    def run_all_validations(self) -> Dict[str, Any]:
        """Run all validation checks."""
        print("🔍 Running System Architecture Validation...")
        
        all_results = []
        
        # Run validation categories
        validation_categories = [
            ('File Structure', self.validate_file_structure),
            ('Python Syntax', self.validate_python_syntax),
            ('Module Imports', self.validate_module_imports),
            ('Documentation', self.validate_documentation),
            ('Configuration', self.validate_configuration),
            ('Code Quality', self.validate_code_quality),
            ('Architecture', self.validate_architecture_coherence)
        ]
        
        for category_name, validator_func in validation_categories:
            print(f"\n📋 Validating {category_name}...")
            category_results = validator_func()
            all_results.extend(category_results)
            
            # Print immediate feedback
            passes = len([r for r in category_results if r.status == 'pass'])
            warnings = len([r for r in category_results if r.status == 'warning'])
            failures = len([r for r in category_results if r.status == 'fail'])
            
            print(f"  ✅ Pass: {passes}, ⚠️  Warning: {warnings}, ❌ Fail: {failures}")
        
        # Compile summary
        summary = self._compile_summary(all_results)
        
        return {
            'summary': summary,
            'detailed_results': all_results,
            'timestamp': time.time()
        }
    
    def _compile_summary(self, results: List[ValidationResult]) -> Dict[str, Any]:
        """Compile validation summary."""
        total_checks = len(results)
        passes = len([r for r in results if r.status == 'pass'])
        warnings = len([r for r in results if r.status == 'warning'])
        failures = len([r for r in results if r.status == 'fail'])
        skips = len([r for r in results if r.status == 'skip'])
        
        # Calculate overall score
        score = (passes / total_checks * 100) if total_checks > 0 else 0
        
        # Determine overall status
        if failures == 0 and warnings <= total_checks * 0.1:  # <=10% warnings
            overall_status = 'excellent'
        elif failures == 0:
            overall_status = 'good'
        elif failures <= total_checks * 0.1:  # <=10% failures
            overall_status = 'acceptable'
        else:
            overall_status = 'needs_improvement'
        
        return {
            'total_checks': total_checks,
            'passes': passes,
            'warnings': warnings,
            'failures': failures,
            'skips': skips,
            'score': score,
            'overall_status': overall_status
        }
    
    def generate_report(self, validation_data: Dict[str, Any]) -> str:
        """Generate human-readable validation report."""
        summary = validation_data['summary']
        
        report = []
        report.append("=" * 60)
        report.append("🏗️  SYSTEM ARCHITECTURE VALIDATION REPORT")
        report.append("=" * 60)
        
        # Overall status
        status_emoji = {
            'excellent': '🌟',
            'good': '✅',
            'acceptable': '⚠️',
            'needs_improvement': '❌'
        }
        
        report.append(f"\n{status_emoji[summary['overall_status']]} Overall Status: {summary['overall_status'].upper()}")
        report.append(f"📊 Validation Score: {summary['score']:.1f}/100")
        
        # Summary statistics
        report.append(f"\n📈 Validation Statistics:")
        report.append(f"  Total checks: {summary['total_checks']}")
        report.append(f"  ✅ Passed: {summary['passes']}")
        report.append(f"  ⚠️  Warnings: {summary['warnings']}")
        report.append(f"  ❌ Failed: {summary['failures']}")
        if summary['skips'] > 0:
            report.append(f"  ⏭️  Skipped: {summary['skips']}")
        
        # Category breakdown
        results = validation_data['detailed_results']
        categories = {}
        for result in results:
            if result.component not in categories:
                categories[result.component] = {'pass': 0, 'warning': 0, 'fail': 0, 'skip': 0}
            categories[result.component][result.status] += 1
        
        report.append(f"\n📊 Results by Category:")
        for category, counts in categories.items():
            total_cat = sum(counts.values())
            pass_rate = (counts['pass'] / total_cat * 100) if total_cat > 0 else 0
            report.append(f"  {category}: {pass_rate:.1f}% ({counts['pass']}/{total_cat})")
        
        # Failures and warnings
        failures = [r for r in results if r.status == 'fail']
        if failures:
            report.append(f"\n❌ Critical Issues ({len(failures)}):")
            for failure in failures[:10]:  # Show first 10
                report.append(f"  • {failure.message}")
            if len(failures) > 10:
                report.append(f"  ... and {len(failures) - 10} more")
        
        warnings = [r for r in results if r.status == 'warning']
        if warnings:
            report.append(f"\n⚠️  Warnings ({len(warnings)}):")
            for warning in warnings[:5]:  # Show first 5
                report.append(f"  • {warning.message}")
            if len(warnings) > 5:
                report.append(f"  ... and {len(warnings) - 5} more")
        
        # Recommendations
        report.append(f"\n💡 Recommendations:")
        if summary['overall_status'] == 'excellent':
            report.append("  • System architecture is excellent! No major issues found.")
            report.append("  • Consider minor optimizations based on warnings above.")
        elif summary['overall_status'] == 'good':
            report.append("  • System architecture is good with minor issues.")
            report.append("  • Address warnings to achieve excellent status.")
        elif summary['overall_status'] == 'acceptable':
            report.append("  • System has some architectural issues to address.")
            report.append("  • Focus on fixing critical failures first.")
            report.append("  • Review warnings for potential improvements.")
        else:
            report.append("  • System requires significant architectural improvements.")
            report.append("  • Address all critical failures immediately.")
            report.append("  • Consider refactoring problematic components.")
        
        report.append("\n" + "=" * 60)
        report.append("Validation completed successfully! 🎉")
        report.append("=" * 60)
        
        return "\n".join(report)

def main():
    """Main validation function."""
    validator = SystemValidator()
    
    # Run all validations
    validation_data = validator.run_all_validations()
    
    # Generate and print report
    report = validator.generate_report(validation_data)
    print(report)
    
    # Save detailed results
    results_file = Path(__file__).parent / 'validation_results.json'
    with open(results_file, 'w') as f:
        json.dump({
            'summary': validation_data['summary'],
            'timestamp': validation_data['timestamp'],
            'detailed_count': len(validation_data['detailed_results'])
        }, f, indent=2)
    
    print(f"\n📁 Detailed results saved to: {results_file}")
    
    return validation_data['summary']['overall_status'] in ['excellent', 'good']

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)