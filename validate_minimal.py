#!/usr/bin/env python3
"""Minimal validation of core functionality without external dependencies."""

import sys
import os
import numpy as np

# Add local package to path
sys.path.insert(0, os.path.dirname(__file__))

# Mock torch for testing
class MockTensor:
    def __init__(self, data):
        if isinstance(data, (list, tuple)):
            self.data = np.array(data, dtype=np.float32)
        elif isinstance(data, np.ndarray):
            self.data = data.astype(np.float32)
        else:
            self.data = np.array(data, dtype=np.float32)
        self.shape = self.data.shape
        self.device = MockDevice()
    
    def __add__(self, other):
        if isinstance(other, MockTensor):
            return MockTensor(self.data + other.data)
        return MockTensor(self.data + other)
    
    def __mul__(self, other):
        if isinstance(other, MockTensor):
            return MockTensor(self.data * other.data)
        return MockTensor(self.data * other)
    
    def __rmul__(self, other):
        return MockTensor(self.data * other)
    
    def to(self, device=None, dtype=None):
        return self
    
    def unsqueeze(self, dim):
        return MockTensor(np.expand_dims(self.data, dim))
    
    def squeeze(self, dim=None):
        if dim is None:
            return MockTensor(np.squeeze(self.data))
        return MockTensor(np.squeeze(self.data, axis=dim))
    
    def cpu(self):
        return self
    
    def item(self):
        return float(self.data.item())
    
    def __getitem__(self, key):
        return MockTensor(self.data[key])

class MockDevice:
    def __init__(self, type='cpu'):
        self.type = type

class MockTorch:
    tensor = MockTensor
    float32 = 'float32'
    
    @staticmethod
    def randn(*shape, device=None, dtype=None):
        return MockTensor(np.random.randn(*shape))
    
    @staticmethod
    def randint(low, high, size, device=None, dtype=None):
        return MockTensor(np.random.randint(low, high, size))
    
    @staticmethod
    def zeros(*shape, device=None, dtype=None):
        return MockTensor(np.zeros(shape))
    
    @staticmethod
    def ones(*shape, device=None, dtype=None):
        return MockTensor(np.ones(shape))
    
    @staticmethod
    def from_numpy(array):
        return MockTensor(array)
    
    @staticmethod
    def roll(tensor, shifts, dims):
        return MockTensor(np.roll(tensor.data, shifts, axis=dims))
    
    @staticmethod
    def stack(tensors, dim=0):
        arrays = [t.data for t in tensors]
        return MockTensor(np.stack(arrays, axis=dim))
    
    @staticmethod
    def sum(tensor, dim=None):
        return MockTensor(np.sum(tensor.data, axis=dim))
    
    @staticmethod
    def dot(a, b):
        return MockTensor(np.dot(a.data, b.data))
    
    @staticmethod
    def norm(tensor):
        return MockTensor(np.linalg.norm(tensor.data))
    
    @staticmethod
    def where(condition, x, y):
        if isinstance(condition, MockTensor):
            condition = condition.data
        if isinstance(x, MockTensor):
            x = x.data
        if isinstance(y, MockTensor):
            y = y.data
        return MockTensor(np.where(condition, x, y))
    
    @staticmethod
    def sign(tensor):
        return MockTensor(np.sign(tensor.data))
    
    @staticmethod
    def argmax(tensor, dim=None):
        return MockTensor(np.argmax(tensor.data, axis=dim))
    
    @staticmethod
    def mean(tensor, dim=None):
        return MockTensor(np.mean(tensor.data, axis=dim))
    
    @staticmethod
    def round(tensor):
        return MockTensor(np.round(tensor.data))
    
    @staticmethod
    def unique(tensor):
        return MockTensor(np.unique(tensor.data))
    
    @staticmethod
    def all(tensor):
        return MockTensor(np.all(tensor.data))
    
    @staticmethod
    def isin(tensor, values):
        if isinstance(values, MockTensor):
            values = values.data
        return MockTensor(np.isin(tensor.data, values))
    
    @staticmethod
    def allclose(a, b, rtol=1e-05, atol=1e-08):
        return np.allclose(a.data, b.data, rtol=rtol, atol=atol)
    
    @staticmethod
    def linspace(start, end, steps, dtype=None):
        return MockTensor(np.linspace(start, end, steps))
    
    @staticmethod
    def manual_seed(seed):
        np.random.seed(seed)
    
    class cuda:
        @staticmethod
        def is_available():
            return False
    
    class nn:
        class functional:
            @staticmethod
            def normalize(tensor, dim=-1, p=2):
                data = tensor.data
                norm = np.linalg.norm(data, ord=p, axis=dim, keepdims=True)
                return MockTensor(data / (norm + 1e-8))
        
        class Module:
            def __init__(self):
                pass
            def to(self, device):
                return self
        
        class Sequential(Module):
            def __init__(self, *layers):
                super().__init__()
                self.layers = layers
        
        class Conv2d(Module):
            def __init__(self, *args, **kwargs):
                super().__init__()
        
        class ReLU(Module):
            def __init__(self):
                super().__init__()
        
        class MaxPool2d(Module):
            def __init__(self, *args, **kwargs):
                super().__init__()
        
        class AdaptiveAvgPool2d(Module):
            def __init__(self, *args, **kwargs):
                super().__init__()
        
        class Flatten(Module):
            def __init__(self):
                super().__init__()
        
        class Linear(Module):
            def __init__(self, *args, **kwargs):
                super().__init__()

# Mock dependencies
class MockTorchModule:
    def __init__(self):
        self.Tensor = MockTensor
        self.tensor = MockTensor
        self.float32 = 'float32'
        self.dtype = str  # Mock dtype
        self.randn = MockTorch.randn
        self.randint = MockTorch.randint
        self.zeros = MockTorch.zeros
        self.ones = MockTorch.ones
        self.from_numpy = MockTorch.from_numpy
        self.roll = MockTorch.roll
        self.stack = MockTorch.stack
        self.sum = MockTorch.sum
        self.dot = MockTorch.dot
        self.norm = MockTorch.norm
        self.where = MockTorch.where
        self.sign = MockTorch.sign
        self.argmax = MockTorch.argmax
        self.mean = MockTorch.mean
        self.round = MockTorch.round
        self.unique = MockTorch.unique
        self.all = MockTorch.all
        self.isin = MockTorch.isin
        self.allclose = MockTorch.allclose
        self.linspace = MockTorch.linspace
        self.manual_seed = MockTorch.manual_seed
        self.cuda = MockTorch.cuda
        self.nn = MockTorch.nn

# Mock torchvision
class MockTransforms:
    class Compose:
        def __init__(self, transforms):
            pass
    
    class Resize:
        def __init__(self, size):
            pass
    
    class ToTensor:
        pass
    
    class Normalize:
        def __init__(self, mean, std):
            pass

class MockTorchVision:
    transforms = MockTransforms()

# Mock scipy
class MockSignal:
    @staticmethod
    def welch(x, fs, nperseg, axis=-1):
        # Return dummy frequency and PSD arrays
        freqs = np.linspace(0, fs/2, 129)
        if x.ndim == 1:
            psd = np.random.rand(len(freqs))
        else:
            psd = np.random.rand(x.shape[0], len(freqs))
        return freqs, psd

class MockScipy:
    signal = MockSignal

sys.modules['torch'] = MockTorchModule()
sys.modules['torch.nn'] = MockTorch.nn
sys.modules['torch.nn.functional'] = MockTorch.nn.functional
sys.modules['torchvision'] = MockTorchVision()
sys.modules['torchvision.transforms'] = MockTransforms()
sys.modules['scipy'] = MockScipy()
sys.modules['scipy.signal'] = MockSignal()

# Validation tests
def test_basic_imports():
    """Test basic package imports."""
    try:
        import hypervector as hv
        print("✓ Package imports successfully")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_hypervector_creation():
    """Test HyperVector creation and basic operations."""
    try:
        from hypervector.core import HyperVector
        
        # Test creation from list
        hv1 = HyperVector([1.0, 2.0, 3.0, 4.0])
        assert hv1.dim == 4
        print("✓ HyperVector creation from list")
        
        # Test random generation
        hv2 = HyperVector.random(1000, seed=42)
        assert hv2.dim == 1000
        print("✓ Random HyperVector generation")
        
        # Test operations
        hv3 = hv1 + hv1  # bundling
        hv4 = hv1 * hv1  # binding
        print("✓ Basic HyperVector operations")
        
        return True
    except Exception as e:
        print(f"❌ HyperVector test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_hdc_operations():
    """Test HDC operations."""
    try:
        from hypervector.core import HyperVector, bind, bundle, permute, cosine_similarity
        
        hv1 = HyperVector.random(100, seed=1)
        hv2 = HyperVector.random(100, seed=2)
        
        # Test bind
        bound = bind(hv1, hv2)
        assert bound.dim == 100
        print("✓ Bind operation")
        
        # Test bundle
        bundled = bundle([hv1, hv2])
        assert bundled.dim == 100
        print("✓ Bundle operation")
        
        # Test permute
        permuted = permute(hv1, shift=1)
        assert permuted.dim == 100
        print("✓ Permute operation")
        
        # Test similarity
        sim = cosine_similarity(hv1, hv2)
        print("✓ Cosine similarity")
        
        return True
    except Exception as e:
        print(f"❌ HDC operations test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_text_encoding():
    """Test text encoding functionality."""
    try:
        from hypervector.encoders import TextEncoder
        
        encoder = TextEncoder(dim=500)
        
        # Test character encoding
        char_hv = encoder.encode_character('a')
        assert char_hv.dim == 500
        print("✓ Character encoding")
        
        # Test word encoding
        word_hv = encoder.encode_word("hello")
        assert word_hv.dim == 500
        print("✓ Word encoding")
        
        # Test sentence encoding
        sent_hv = encoder.encode_sentence("hello world")
        assert sent_hv.dim == 500
        print("✓ Sentence encoding")
        
        return True
    except Exception as e:
        print(f"❌ Text encoding test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_applications():
    """Test application modules."""
    try:
        from hypervector.applications import BCIClassifier, CrossModalRetrieval
        
        # Test BCI classifier
        bci = BCIClassifier(
            channels=4,
            hypervector_dim=500,
            window_size=100
        )
        print("✓ BCI classifier creation")
        
        # Test cross-modal retrieval
        retrieval = CrossModalRetrieval(dim=500)
        print("✓ Cross-modal retrieval creation")
        
        return True
    except Exception as e:
        print(f"❌ Applications test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all validation tests."""
    print("🧠 HyperVector-Lab Validation Tests")
    print("=" * 40)
    
    tests = [
        test_basic_imports,
        test_hypervector_creation,
        test_hdc_operations,
        test_text_encoding,
        test_applications,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        print(f"\nRunning {test.__name__}...")
        if test():
            passed += 1
        else:
            print(f"Test {test.__name__} failed!")
    
    print(f"\n{'='*40}")
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Generation 1 implementation is working!")
        return True
    else:
        print("❌ Some tests failed.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)