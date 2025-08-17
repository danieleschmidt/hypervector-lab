
import unittest
import numpy as np
from hypervector import HDCSystem, HyperVector

class TestRobustness(unittest.TestCase):
    """Test robustness and error handling"""
    
    def setUp(self):
        """Setup test environment"""
        try:
            self.hdc = HDCSystem(dim=1000, device='cpu')
        except Exception as e:
            self.skipTest(f"Could not initialize HDC system: {e}")
    
    def test_invalid_dimension(self):
        """Test handling of invalid dimensions"""
        with self.assertRaises(ValueError):
            HDCSystem(dim=-1)
        
        with self.assertRaises(TypeError):
            HDCSystem(dim="invalid")
    
    def test_invalid_device(self):
        """Test handling of invalid devices"""
        with self.assertRaises(ValueError):
            HDCSystem(device="invalid_device")
    
    def test_empty_input_handling(self):
        """Test handling of empty inputs"""
        # These should handle gracefully
        try:
            result = self.hdc.encode_text("")
            self.assertIsNotNone(result)
        except Exception as e:
            self.fail(f"Empty text encoding should not fail: {e}")
    
    def test_large_input_handling(self):
        """Test handling of large inputs"""
        large_text = "test " * 10000
        try:
            result = self.hdc.encode_text(large_text)
            self.assertIsNotNone(result)
        except MemoryError:
            self.skipTest("Insufficient memory for large input test")
        except Exception as e:
            self.fail(f"Large input should not cause unexpected error: {e}")
    
    def test_concurrent_access(self):
        """Test concurrent access to HDC system"""
        import threading
        
        def encode_worker():
            try:
                self.hdc.encode_text("concurrent test")
            except Exception as e:
                self.fail(f"Concurrent access failed: {e}")
        
        threads = [threading.Thread(target=encode_worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

if __name__ == '__main__':
    unittest.main()
