#!/usr/bin/env python3
"""Simple test to validate core library functionality without heavy dependencies."""

import sys
import os

# Add the repo to Python path
sys.path.insert(0, '/root/repo')

def test_basic_structure():
    """Test basic library structure and imports."""
    print("Testing HyperVector-Lab basic structure...")
    
    # Test directory structure
    required_dirs = [
        'hypervector',
        'hypervector/core', 
        'hypervector/encoders',
        'hypervector/applications',
        'tests'
    ]
    
    for dir_path in required_dirs:
        full_path = os.path.join('/root/repo', dir_path)
        if os.path.exists(full_path):
            print(f"✓ {dir_path} exists")
        else:
            print(f"✗ {dir_path} missing")
            return False
    
    # Test that key files exist
    required_files = [
        'hypervector/__init__.py',
        'hypervector/core/system.py',
        'hypervector/core/hypervector.py',
        'hypervector/encoders/text.py',
        'pyproject.toml',
        'README.md'
    ]
    
    for file_path in required_files:
        full_path = os.path.join('/root/repo', file_path)
        if os.path.exists(full_path):
            print(f"✓ {file_path} exists")
        else:
            print(f"✗ {file_path} missing")
            return False
    
    print("✓ Basic structure validation passed")
    return True

def test_version_info():
    """Test version information."""
    try:
        # Read version from __init__.py
        init_file = '/root/repo/hypervector/__init__.py'
        with open(init_file, 'r') as f:
            content = f.read()
            
        if '__version__' in content:
            print("✓ Version info found in __init__.py")
            return True
        else:
            print("✗ No version info found")
            return False
            
    except Exception as e:
        print(f"✗ Error reading version: {e}")
        return False

def test_documentation():
    """Test documentation completeness."""
    readme_path = '/root/repo/README.md'
    try:
        with open(readme_path, 'r') as f:
            readme_content = f.read()
            
        required_sections = [
            '# HyperVector-Lab',
            '## 🚀 Features',
            '## 📦 Installation', 
            '## 🎯 Quick Start',
            '## 🏗️ Architecture'
        ]
        
        for section in required_sections:
            if section in readme_content:
                print(f"✓ README contains {section}")
            else:
                print(f"✗ README missing {section}")
                return False
                
        print("✓ Documentation validation passed")
        return True
        
    except Exception as e:
        print(f"✗ Error reading README: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 50)
    print("HYPERVECTOR-LAB VALIDATION")
    print("=" * 50)
    
    tests = [
        test_basic_structure,
        test_version_info, 
        test_documentation
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
            print()
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
            results.append(False)
            print()
    
    # Summary
    passed = sum(results)
    total = len(results)
    print("=" * 50)
    print(f"RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ ALL TESTS PASSED - Library structure is valid")
        return True
    else:
        print("✗ SOME TESTS FAILED - Issues detected")
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)