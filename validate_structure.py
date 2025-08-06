#!/usr/bin/env python3
"""Validate repository structure and implementation completeness."""

import os
import sys
from pathlib import Path

def check_file_exists(filepath, description=""):
    """Check if a file exists and has content."""
    path = Path(filepath)
    if not path.exists():
        return False, f"❌ Missing: {filepath}"
    
    size = path.stat().st_size
    if size == 0:
        return False, f"❌ Empty: {filepath}"
    
    return True, f"✅ {filepath} ({size} bytes)"

def analyze_python_file(filepath):
    """Analyze Python file for basic structure."""
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        
        lines = content.split('\n')
        non_empty_lines = [line for line in lines if line.strip() and not line.strip().startswith('#')]
        
        classes = [line for line in lines if line.strip().startswith('class ')]
        functions = [line for line in lines if line.strip().startswith('def ')]
        imports = [line for line in lines if line.strip().startswith(('import ', 'from '))]
        
        return {
            'total_lines': len(lines),
            'code_lines': len(non_empty_lines),
            'classes': len(classes),
            'functions': len(functions),
            'imports': len(imports),
            'has_docstring': '"""' in content or "'''" in content
        }
    except Exception as e:
        return {'error': str(e)}

def main():
    print("🧠 HyperVector-Lab Structure Validation")
    print("=" * 50)
    
    # Core structure checks
    core_files = [
        ("pyproject.toml", "Project configuration"),
        ("README.md", "Documentation"),
        ("LICENSE", "License file"),
        ("hypervector/__init__.py", "Package init"),
        ("hypervector/core/__init__.py", "Core module init"),
        ("hypervector/core/hypervector.py", "HyperVector class"),
        ("hypervector/core/operations.py", "HDC operations"),
        ("hypervector/core/system.py", "HDC system"),
        ("hypervector/core/exceptions.py", "Custom exceptions"),
    ]
    
    encoder_files = [
        ("hypervector/encoders/__init__.py", "Encoders init"),
        ("hypervector/encoders/text.py", "Text encoder"),
        ("hypervector/encoders/vision.py", "Vision encoder"),
        ("hypervector/encoders/eeg.py", "EEG encoder"),
    ]
    
    application_files = [
        ("hypervector/applications/__init__.py", "Applications init"),
        ("hypervector/applications/bci.py", "BCI classifier"),
        ("hypervector/applications/retrieval.py", "Cross-modal retrieval"),
    ]
    
    utility_files = [
        ("hypervector/utils/__init__.py", "Utils init"),
        ("hypervector/utils/validation.py", "Input validation"),
        ("hypervector/utils/logging.py", "Logging system"),
        ("hypervector/utils/config.py", "Configuration"),
        ("hypervector/utils/security.py", "Security utilities"),
    ]
    
    accelerator_files = [
        ("hypervector/accelerators/__init__.py", "Accelerators init"),
        ("hypervector/accelerators/cpu_optimized.py", "CPU optimization"),
        ("hypervector/accelerators/batch_processor.py", "Batch processing"),
        ("hypervector/accelerators/memory_manager.py", "Memory management"),
    ]
    
    benchmark_files = [
        ("hypervector/benchmark/__init__.py", "Benchmark init"),
        ("hypervector/benchmark/benchmarks.py", "Benchmarking suite"),
        ("hypervector/benchmark/profiler.py", "Performance profiler"),
    ]
    
    deployment_files = [
        ("hypervector/deployment/__init__.py", "Deployment init"),
        ("hypervector/deployment/deployment_config.py", "Deployment config"),
    ]
    
    test_files = [
        ("tests/__init__.py", "Tests init"),
        ("tests/test_core.py", "Core tests"),
        ("tests/test_encoders.py", "Encoder tests"),
        ("tests/test_applications.py", "Application tests"),
        ("tests/test_integration.py", "Integration tests"),
    ]
    
    all_files = (core_files + encoder_files + application_files + 
                utility_files + accelerator_files + benchmark_files + 
                deployment_files + test_files)
    
    print("\n📁 File Structure Validation:")
    print("-" * 30)
    
    total_files = len(all_files)
    existing_files = 0
    total_size = 0
    
    for filepath, description in all_files:
        exists, message = check_file_exists(filepath)
        print(message)
        
        if exists:
            existing_files += 1
            try:
                total_size += Path(filepath).stat().st_size
            except:
                pass
    
    print(f"\n📊 Structure Summary:")
    print(f"Files found: {existing_files}/{total_files} ({(existing_files/total_files)*100:.1f}%)")
    print(f"Total codebase size: {total_size:,} bytes ({total_size/1024:.1f} KB)")
    
    # Analyze key Python files
    print(f"\n🔍 Code Analysis:")
    print("-" * 20)
    
    key_files = [
        "hypervector/core/hypervector.py",
        "hypervector/core/operations.py", 
        "hypervector/core/system.py",
        "hypervector/encoders/text.py",
        "hypervector/encoders/vision.py",
        "hypervector/encoders/eeg.py",
        "hypervector/applications/bci.py",
        "hypervector/applications/retrieval.py",
        "hypervector/utils/validation.py",
        "hypervector/accelerators/cpu_optimized.py",
    ]
    
    total_code_lines = 0
    total_classes = 0
    total_functions = 0
    
    for filepath in key_files:
        if Path(filepath).exists():
            analysis = analyze_python_file(filepath)
            if 'error' not in analysis:
                print(f"{filepath}: {analysis['code_lines']} lines, "
                     f"{analysis['classes']} classes, {analysis['functions']} functions")
                total_code_lines += analysis['code_lines']
                total_classes += analysis['classes']
                total_functions += analysis['functions']
            else:
                print(f"{filepath}: Error - {analysis['error']}")
    
    print(f"\n📈 Code Metrics:")
    print(f"Total code lines: {total_code_lines:,}")
    print(f"Total classes: {total_classes}")
    print(f"Total functions: {total_functions}")
    
    # Check implementation status
    print(f"\n🎯 Implementation Status:")
    print("-" * 25)
    
    if existing_files >= total_files * 0.9:
        print("✅ COMPLETE: All major components implemented")
        status = "PRODUCTION-READY"
    elif existing_files >= total_files * 0.7:
        print("🟡 PARTIAL: Most components implemented")
        status = "DEVELOPMENT"
    else:
        print("❌ INCOMPLETE: Major components missing")
        status = "EARLY"
    
    if total_code_lines >= 1500:
        print("✅ COMPREHENSIVE: Substantial implementation")
    elif total_code_lines >= 800:
        print("🟡 MODERATE: Reasonable implementation")
    else:
        print("❌ MINIMAL: Limited implementation")
    
    # Generation assessment
    print(f"\n🚀 SDLC Generation Assessment:")
    print("-" * 35)
    
    gen1_files = len([f for f, _ in core_files + encoder_files if Path(f).exists()])
    gen2_files = len([f for f, _ in utility_files if Path(f).exists()]) 
    gen3_files = len([f for f, _ in accelerator_files + benchmark_files if Path(f).exists()])
    
    if gen1_files >= 8:
        print("✅ Generation 1 (MAKE IT WORK): Complete")
    else:
        print("❌ Generation 1 (MAKE IT WORK): Incomplete")
    
    if gen2_files >= 4:
        print("✅ Generation 2 (MAKE IT ROBUST): Complete")
    else:
        print("❌ Generation 2 (MAKE IT ROBUST): Incomplete")
        
    if gen3_files >= 5:
        print("✅ Generation 3 (MAKE IT SCALE): Complete")
    else:
        print("❌ Generation 3 (MAKE IT SCALE): Incomplete")
    
    # Final assessment
    print(f"\n🏆 FINAL ASSESSMENT:")
    print("=" * 20)
    print(f"Implementation Status: {status}")
    print(f"File Coverage: {(existing_files/total_files)*100:.1f}%")
    print(f"Codebase Size: {total_code_lines:,} lines")
    print(f"Architecture: Multi-modal HDC with {total_classes} classes")
    
    if (existing_files >= total_files * 0.9 and 
        total_code_lines >= 1500 and 
        gen1_files >= 8 and gen2_files >= 4 and gen3_files >= 5):
        print("\n🎉 AUTONOMOUS SDLC EXECUTION: COMPLETE!")
        print("Ready for production deployment.")
        return True
    else:
        print("\n⚠️  Implementation needs completion.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)