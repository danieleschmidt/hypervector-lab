#!/usr/bin/env python3
"""Simple integration test for HyperVector-Lab functionality."""

import sys
import traceback

def test_basic_functionality():
    """Test basic hyperdimensional computing functionality."""
    
    results = []
    
    # Test 1: Basic imports
    try:
        import hypervector
        import torch
        print("✅ Test 1: Basic imports successful")
        results.append(True)
    except Exception as e:
        print(f"❌ Test 1: Import failed: {e}")
        results.append(False)
        return results
    
    # Test 2: HDC System creation
    try:
        hdc = hypervector.HDCSystem(dim=1000, device='cpu')
        print(f"✅ Test 2: HDC system created: {hdc}")
        results.append(True)
    except Exception as e:
        print(f"❌ Test 2: HDC system creation failed: {e}")
        traceback.print_exc()
        results.append(False)
        return results
    
    # Test 3: HyperVector creation
    try:
        # Create random hypervector
        hv1 = hypervector.HyperVector.random(1000, device='cpu', seed=42)
        hv2 = hypervector.HyperVector.random(1000, device='cpu', seed=43)
        print(f"✅ Test 3: Random hypervectors created")
        results.append(True)
    except Exception as e:
        print(f"❌ Test 3: HyperVector creation failed: {e}")
        traceback.print_exc()
        results.append(False)
        return results
    
    # Test 4: Basic HDC operations
    try:
        # Test binding
        bound = hypervector.bind(hv1, hv2)
        
        # Test bundling
        bundled = hypervector.bundle([hv1, hv2])
        
        # Test similarity
        similarity = hypervector.cosine_similarity(hv1, hv2)
        
        print(f"✅ Test 4: HDC operations successful - similarity: {similarity:.4f}")
        results.append(True)
    except Exception as e:
        print(f"❌ Test 4: HDC operations failed: {e}")
        traceback.print_exc()
        results.append(False)
        return results
    
    # Test 5: Text encoding
    try:
        text_hv = hdc.encode_text("Hello HDC world!")
        print(f"✅ Test 5: Text encoding successful - shape: {text_hv.data.shape}")
        results.append(True)
    except Exception as e:
        print(f"❌ Test 5: Text encoding failed: {e}")
        traceback.print_exc()
        results.append(False)
    
    # Test 6: Image encoding
    try:
        import torch
        # Create dummy image tensor (3, 224, 224)
        dummy_image = torch.rand(3, 224, 224)
        image_hv = hdc.encode_image(dummy_image)
        print(f"✅ Test 6: Image encoding successful - shape: {image_hv.data.shape}")
        results.append(True)
    except Exception as e:
        print(f"❌ Test 6: Image encoding failed: {e}")
        traceback.print_exc()
        results.append(False)
    
    # Test 7: EEG encoding
    try:
        import torch
        # Create dummy EEG signal (64 channels, 1000 samples)
        dummy_eeg = torch.rand(64, 1000)
        eeg_hv = hdc.encode_eeg(dummy_eeg, sampling_rate=250.0)
        print(f"✅ Test 7: EEG encoding successful - shape: {eeg_hv.data.shape}")
        results.append(True)
    except Exception as e:
        print(f"❌ Test 7: EEG encoding failed: {e}")
        traceback.print_exc()
        results.append(False)
    
    # Test 8: BCI Application
    try:
        bci = hypervector.BCIClassifier(
            channels=8,
            sampling_rate=250,
            window_size=250,
            hypervector_dim=1000
        )
        print(f"✅ Test 8: BCI classifier created successfully")
        results.append(True)
    except Exception as e:
        print(f"❌ Test 8: BCI classifier creation failed: {e}")
        traceback.print_exc()
        results.append(False)
    
    # Test 9: Cross-modal retrieval
    try:
        retrieval = hypervector.CrossModalRetrieval(dim=1000)
        print(f"✅ Test 9: Cross-modal retrieval created successfully")
        results.append(True)
    except Exception as e:
        print(f"❌ Test 9: Cross-modal retrieval creation failed: {e}")
        traceback.print_exc()
        results.append(False)
    
    return results

if __name__ == "__main__":
    print("🧠 HyperVector-Lab Integration Test")
    print("=" * 50)
    
    results = test_basic_functionality()
    
    print("\n" + "=" * 50)
    print(f"Test Results: {sum(results)}/{len(results)} passed")
    
    if sum(results) == len(results):
        print("🎉 All tests PASSED! HyperVector-Lab is working correctly.")
        sys.exit(0)
    else:
        print("❌ Some tests failed. Check the error messages above.")
        sys.exit(1)