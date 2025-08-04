"""Tests for core HDC functionality."""

import pytest
import torch
import numpy as np
from hypervector.core import HyperVector, bind, bundle, permute, cosine_similarity, HDCSystem


class TestHyperVector:
    """Tests for HyperVector class."""
    
    def test_init_from_tensor(self):
        """Test initialization from torch tensor."""
        data = torch.randn(1000)
        hv = HyperVector(data)
        assert hv.dim == 1000
        assert torch.allclose(hv.data, data)
    
    def test_init_from_numpy(self):
        """Test initialization from numpy array."""
        data = np.random.randn(1000)
        hv = HyperVector(data)
        assert hv.dim == 1000
        assert torch.allclose(hv.data, torch.from_numpy(data).float())
    
    def test_init_from_list(self):
        """Test initialization from list."""
        data = [1.0, -1.0, 0.5, -0.5]
        hv = HyperVector(data)
        assert hv.dim == 4
        assert torch.allclose(hv.data, torch.tensor(data))
    
    def test_random_generation(self):
        """Test random hypervector generation."""
        dim = 10000
        hv1 = HyperVector.random(dim)
        hv2 = HyperVector.random(dim)
        
        assert hv1.dim == dim
        assert hv2.dim == dim
        
        # Should be different
        similarity = cosine_similarity(hv1, hv2)
        assert abs(similarity.item()) < 0.1  # Should be nearly orthogonal
    
    def test_random_reproducibility(self):
        """Test random generation with seed."""
        dim = 1000
        seed = 42
        
        hv1 = HyperVector.random(dim, seed=seed)
        hv2 = HyperVector.random(dim, seed=seed)
        
        assert torch.allclose(hv1.data, hv2.data)
    
    def test_binary_mode(self):
        """Test binary hypervector generation."""
        hv = HyperVector.random(1000, mode="binary")
        unique_values = torch.unique(hv.data)
        
        # Should only contain -1 and 1
        assert len(unique_values) <= 2
        assert torch.all(torch.isin(unique_values, torch.tensor([-1.0, 1.0])))
    
    def test_ternary_mode(self):
        """Test ternary hypervector generation."""
        hv = HyperVector.random(1000, mode="ternary")
        unique_values = torch.unique(hv.data)
        
        # Should only contain -1, 0, and 1
        assert len(unique_values) <= 3
        assert torch.all(torch.isin(unique_values, torch.tensor([-1.0, 0.0, 1.0])))
    
    def test_binarization(self):
        """Test vector binarization."""
        data = torch.tensor([0.8, -0.3, 0.1, -0.9])
        hv = HyperVector(data)
        binary_hv = hv.binarize(threshold=0.0)
        
        expected = torch.tensor([1.0, -1.0, 1.0, -1.0])
        assert torch.allclose(binary_hv.data, expected)
    
    def test_ternarization(self):
        """Test vector ternarization."""
        data = torch.tensor([0.8, -0.3, 0.1, -0.9])
        hv = HyperVector(data)
        ternary_hv = hv.ternarize(low_threshold=-0.5, high_threshold=0.5)
        
        expected = torch.tensor([1.0, 0.0, 0.0, -1.0])
        assert torch.allclose(ternary_hv.data, expected)
    
    def test_addition(self):
        """Test hypervector addition (bundling)."""
        hv1 = HyperVector([1.0, 2.0, 3.0])
        hv2 = HyperVector([4.0, 5.0, 6.0])
        result = hv1 + hv2
        
        expected = torch.tensor([5.0, 7.0, 9.0])
        assert torch.allclose(result.data, expected)
    
    def test_multiplication(self):
        """Test hypervector multiplication (binding)."""
        hv1 = HyperVector([1.0, 2.0, 3.0])
        hv2 = HyperVector([4.0, 5.0, 6.0])
        result = hv1 * hv2
        
        expected = torch.tensor([4.0, 10.0, 18.0])
        assert torch.allclose(result.data, expected)
    
    def test_cosine_similarity(self):
        """Test cosine similarity computation."""
        hv1 = HyperVector([1.0, 0.0, 0.0])
        hv2 = HyperVector([0.0, 1.0, 0.0])
        hv3 = HyperVector([1.0, 0.0, 0.0])
        
        # Orthogonal vectors
        sim12 = hv1.cosine_similarity(hv2)
        assert abs(sim12.item()) < 1e-6
        
        # Identical vectors
        sim13 = hv1.cosine_similarity(hv3)
        assert abs(sim13.item() - 1.0) < 1e-6


class TestOperations:
    """Tests for HDC operations."""
    
    def test_bind_operation(self):
        """Test binding operation."""
        hv1 = HyperVector.random(1000, seed=1)
        hv2 = HyperVector.random(1000, seed=2)
        
        bound = bind(hv1, hv2)
        
        # Binding should produce dissimilar result
        sim1 = cosine_similarity(bound, hv1).item()
        sim2 = cosine_similarity(bound, hv2).item() 
        
        assert abs(sim1) < 0.2
        assert abs(sim2) < 0.2
    
    def test_bind_inverse_property(self):
        """Test that binding is approximately its own inverse."""
        hv1 = HyperVector.random(10000, seed=1, mode="binary")
        hv2 = HyperVector.random(10000, seed=2, mode="binary")
        
        # A * B * B should be approximately A
        bound = bind(hv1, hv2)
        unbound = bind(bound, hv2)
        
        similarity = cosine_similarity(hv1, unbound).item()
        assert similarity > 0.8  # Should be quite similar
    
    def test_bundle_operation(self):
        """Test bundling operation."""
        hv1 = HyperVector.random(1000, seed=1)
        hv2 = HyperVector.random(1000, seed=2)
        hv3 = HyperVector.random(1000, seed=3)
        
        bundled = bundle([hv1, hv2, hv3])
        
        # Bundling should produce similar result to inputs
        sim1 = cosine_similarity(bundled, hv1).item()
        sim2 = cosine_similarity(bundled, hv2).item()
        sim3 = cosine_similarity(bundled, hv3).item()
        
        assert sim1 > 0.3
        assert sim2 > 0.3  
        assert sim3 > 0.3
    
    def test_permute_operation(self):
        """Test permutation operation."""
        data = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        hv = HyperVector(data)
        
        # Shift by 1
        permuted = permute(hv, shift=1)
        expected = torch.tensor([5.0, 1.0, 2.0, 3.0, 4.0])
        assert torch.allclose(permuted.data, expected)
        
        # Shift by -2
        permuted = permute(hv, shift=-2)
        expected = torch.tensor([3.0, 4.0, 5.0, 1.0, 2.0])
        assert torch.allclose(permuted.data, expected)
    
    def test_permute_preserves_similarity(self):
        """Test that permutation preserves pairwise similarities."""
        hv1 = HyperVector.random(1000, seed=1)
        hv2 = HyperVector.random(1000, seed=2)
        
        original_sim = cosine_similarity(hv1, hv2).item()
        
        # Permute both vectors by same amount
        perm_hv1 = permute(hv1, shift=5)
        perm_hv2 = permute(hv2, shift=5)
        
        permuted_sim = cosine_similarity(perm_hv1, perm_hv2).item()
        
        # Similarity should be preserved
        assert abs(original_sim - permuted_sim) < 0.01


class TestHDCSystem:
    """Tests for HDCSystem class."""
    
    def test_initialization(self):
        """Test HDC system initialization."""
        hdc = HDCSystem(dim=5000)
        assert hdc.dim == 5000
        assert len(hdc.memory) == 0
    
    def test_memory_operations(self):
        """Test memory store and retrieve."""
        hdc = HDCSystem(dim=1000)
        hv = HyperVector.random(1000)
        
        # Store and retrieve
        hdc.store("test_key", hv)
        retrieved = hdc.retrieve("test_key")
        
        assert retrieved is not None
        assert torch.allclose(retrieved.data, hv.data)
    
    def test_memory_query(self):
        """Test memory querying."""
        hdc = HDCSystem(dim=1000)
        
        # Store some vectors
        hv1 = HyperVector.random(1000, seed=1)
        hv2 = HyperVector.random(1000, seed=2) 
        hv3 = HyperVector.random(1000, seed=3)
        
        hdc.store("item1", hv1)
        hdc.store("item2", hv2)
        hdc.store("item3", hv3)
        
        # Query with similar vector to hv1
        query = hv1 * 0.9 + HyperVector.random(1000, seed=99) * 0.1
        results = hdc.query_memory(query, top_k=2)
        
        assert len(results) == 2
        assert results[0][0] == "item1"  # Should find item1 as most similar
    
    def test_clear_memory(self):
        """Test memory clearing."""
        hdc = HDCSystem(dim=1000)
        hv = HyperVector.random(1000)
        
        hdc.store("test", hv)
        assert len(hdc.memory) == 1
        
        hdc.clear_memory()
        assert len(hdc.memory) == 0
    
    def test_device_transfer(self):
        """Test moving system to different device."""
        hdc = HDCSystem(dim=1000, device='cpu')
        hv = HyperVector.random(1000, device='cpu')
        hdc.store("test", hv)
        
        # Move to CPU (same device, but tests the functionality)
        new_hdc = hdc.to('cpu')
        
        assert new_hdc.device == 'cpu'
        assert len(new_hdc.memory) == 1
        retrieved = new_hdc.retrieve("test")
        assert retrieved.data.device.type == 'cpu'


if __name__ == "__main__":
    pytest.main([__file__])