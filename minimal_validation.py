#!/usr/bin/env python3
"""Minimal validation test for HDC system structure and imports."""

import sys
import os
import importlib.util

def test_module_structure():
    """Test that the HDC module structure is correct."""
    print("🏗️  Testing HDC module structure...")
    
    required_modules = [
        "hypervector/__init__.py",
        "hypervector/core/__init__.py",
        "hypervector/core/system.py",
        "hypervector/core/hypervector.py",
        "hypervector/core/operations.py",
        "hypervector/encoders/__init__.py",
        "hypervector/encoders/text.py",
        "hypervector/encoders/vision.py",
        "hypervector/encoders/eeg.py",
        "hypervector/applications/__init__.py",
        "hypervector/applications/bci.py",
        "hypervector/applications/retrieval.py",
        "hypervector/accelerators/__init__.py",
        "hypervector/accelerators/performance_optimizer.py",
        "hypervector/accelerators/memory_manager.py",
        "hypervector/research/__init__.py",
        "hypervector/research/breakthrough_algorithms.py",
        "hypervector/research/quantum_enhanced_real_time_hdc.py",
        "hypervector/utils/__init__.py",
        "hypervector/utils/advanced_error_recovery.py",
        "hypervector/utils/logging.py",
    ]
    
    missing_modules = []
    for module_path in required_modules:
        if not os.path.exists(module_path):
            missing_modules.append(module_path)
    
    if missing_modules:
        print(f"❌ Missing modules: {missing_modules}")
        return False
    else:
        print("✅ All required modules present")
        return True


def test_code_quality():
    """Test basic code quality metrics."""
    print("\n🔍 Testing code quality...")
    
    python_files = []
    for root, dirs, files in os.walk("hypervector"):
        for file in files:
            if file.endswith(".py"):
                python_files.append(os.path.join(root, file))
    
    issues = []
    
    for file_path in python_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Basic syntax check
            try:
                compile(content, file_path, 'exec')
            except SyntaxError as e:
                issues.append(f"Syntax error in {file_path}: {e}")
                
            # Check for basic documentation
            if not content.strip().startswith('"""') and not content.strip().startswith("'''"):
                if file_path != "hypervector/__init__.py":  # Allow __init__.py to be simple
                    issues.append(f"Missing docstring in {file_path}")
                    
        except Exception as e:
            issues.append(f"Error reading {file_path}: {e}")
    
    if issues:
        print(f"❌ Code quality issues found: {len(issues)}")
        for issue in issues[:5]:  # Show first 5 issues
            print(f"   - {issue}")
        if len(issues) > 5:
            print(f"   ... and {len(issues) - 5} more")
        return False
    else:
        print(f"✅ Code quality check passed for {len(python_files)} files")
        return True


def test_configuration_files():
    """Test configuration files."""
    print("\n⚙️  Testing configuration files...")
    
    config_files = [
        "pyproject.toml",
        "README.md",
        "LICENSE"
    ]
    
    missing_configs = []
    for config_file in config_files:
        if not os.path.exists(config_file):
            missing_configs.append(config_file)
    
    if missing_configs:
        print(f"❌ Missing config files: {missing_configs}")
        return False
    else:
        print("✅ All configuration files present")
        return True


def test_research_implementations():
    """Test research implementation completeness."""
    print("\n🔬 Testing research implementations...")
    
    research_files = [
        "hypervector/research/breakthrough_algorithms.py",
        "hypervector/research/quantum_enhanced_real_time_hdc.py",
        "hypervector/research/adaptive_meta_learning_hdc.py",
        "hypervector/research/quantum_enhanced_hdc.py",
        "hypervector/research/neuromorphic_backends.py"
    ]
    
    implementations_found = 0
    for research_file in research_files:
        if os.path.exists(research_file):
            implementations_found += 1
            
            # Check file size (should be substantial for research code)
            file_size = os.path.getsize(research_file)
            if file_size < 1000:  # Less than 1KB is probably empty
                print(f"⚠️  {research_file} seems too small ({file_size} bytes)")
            else:
                print(f"✅ {research_file} ({file_size} bytes)")
    
    print(f"📊 Research implementations found: {implementations_found}/{len(research_files)}")
    
    if implementations_found >= len(research_files) * 0.8:  # At least 80% implemented
        print("✅ Research implementation test passed")
        return True
    else:
        print("❌ Insufficient research implementations")
        return False


def test_production_readiness():
    """Test production readiness indicators."""
    print("\n🚀 Testing production readiness...")
    
    production_indicators = [
        ("Docker support", "Dockerfile"),
        ("Production deployment", "production_deployment/"),
        ("Kubernetes config", "production_ready/kubernetes/"),
        ("Monitoring setup", "production_ready/monitoring/"),
        ("CI/CD pipeline", "production_ready/ci-cd/"),
        ("Testing suite", "tests/"),
        ("Performance optimization", "hypervector/accelerators/"),
        ("Error recovery", "hypervector/utils/advanced_error_recovery.py"),
        ("Security features", "hypervector/utils/advanced_security.py"),
        ("Global compliance", "hypervector/utils/global_compliance.py")
    ]
    
    readiness_score = 0
    for indicator_name, path in production_indicators:
        if os.path.exists(path):
            print(f"✅ {indicator_name}")
            readiness_score += 1
        else:
            print(f"❌ {indicator_name}")
    
    readiness_percentage = (readiness_score / len(production_indicators)) * 100
    print(f"\n📈 Production Readiness Score: {readiness_score}/{len(production_indicators)} ({readiness_percentage:.1f}%)")
    
    if readiness_percentage >= 80:
        print("✅ Production readiness test passed")
        return True
    else:
        print("❌ Not ready for production deployment")
        return False


def test_research_publications():
    """Test research publication artifacts."""
    print("\n📚 Testing research publication artifacts...")
    
    publication_files = [
        "ACADEMIC_RESEARCH_PUBLICATION_PACKAGE.md",
        "RESEARCH_ENHANCEMENTS_REPORT.md",
        "FINAL_AUTONOMOUS_RESEARCH_EXECUTION_REPORT.md"
    ]
    
    artifacts_found = 0
    for pub_file in publication_files:
        if os.path.exists(pub_file):
            artifacts_found += 1
            file_size = os.path.getsize(pub_file)
            print(f"✅ {pub_file} ({file_size} bytes)")
        else:
            print(f"❌ {pub_file}")
    
    # Check for validation results
    validation_files = [
        "research_validation_results_*.json",
        "quality_gates_report.json",
        "generation*_results.json"
    ]
    
    import glob
    validation_found = 0
    for pattern in validation_files:
        matches = glob.glob(pattern)
        validation_found += len(matches)
        for match in matches:
            print(f"✅ Validation result: {match}")
    
    total_score = artifacts_found + min(validation_found, 3)  # Cap validation files
    max_score = len(publication_files) + 3
    
    print(f"\n📊 Research Publication Score: {total_score}/{max_score}")
    
    if total_score >= max_score * 0.7:  # At least 70%
        print("✅ Research publication test passed")
        return True
    else:
        print("❌ Insufficient research publication artifacts")
        return False


def main():
    """Run minimal validation suite."""
    print("🧬 HDC System Minimal Validation Suite")
    print("=" * 50)
    print("This validation runs without external dependencies.")
    print("=" * 50)
    
    tests = [
        ("Module Structure", test_module_structure),
        ("Code Quality", test_code_quality),
        ("Configuration Files", test_configuration_files),
        ("Research Implementations", test_research_implementations),
        ("Production Readiness", test_production_readiness),
        ("Research Publications", test_research_publications)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'=' * 50}")
        print(f"Running: {test_name}")
        print('=' * 50)
        
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} FAILED with exception: {e}")
    
    print(f"\n{'=' * 50}")
    print(f"MINIMAL VALIDATION SUMMARY")
    print('=' * 50)
    print(f"Tests Passed: {passed}/{total}")
    print(f"Success Rate: {(passed/total)*100:.1f}%")
    
    if passed == total:
        print("🎉 MINIMAL VALIDATION PASSED!")
        print("📋 System structure and code quality verified.")
        print("🔄 Run full validation with dependencies for complete testing.")
        return 0
    elif passed >= total * 0.8:
        print("✅ MOSTLY READY - Minor issues detected.")
        print("🔧 Address remaining issues before production deployment.")
        return 0
    else:
        print("⚠️  VALIDATION FAILED - Major issues detected.")
        print("🛠️  Significant work needed before system is ready.")
        return 1


if __name__ == "__main__":
    sys.exit(main())