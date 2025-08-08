"""Tests for multi-modal encoders."""

import pytest
import torch
try:
    import numpy as np
except ImportError:
    # Fallback for environments with fake numpy
    class FakeNumpy:
        def __getattr__(self, name):
            if name == 'ndarray':
                return torch.Tensor
            raise AttributeError(f"module 'numpy' has no attribute '{name}'")
    np = FakeNumpy()
from PIL import Image
from hypervector.encoders import TextEncoder, VisionEncoder, EEGEncoder


class TestTextEncoder:
    """Tests for text encoding."""
    
    def test_initialization(self):
        """Test text encoder initialization."""
        encoder = TextEncoder(dim=5000)
        assert encoder.dim == 5000
        assert len(encoder.char_vectors) > 0
    
    def test_character_encoding(self):
        """Test single character encoding."""
        encoder = TextEncoder(dim=1000)
        
        # Test basic characters
        hv_a = encoder.encode_character('a')
        hv_b = encoder.encode_character('b')
        
        assert hv_a.dim == 1000
        assert hv_b.dim == 1000
        
        # Different characters should have different encodings
        similarity = hv_a.cosine_similarity(hv_b).item()
        assert abs(similarity) < 0.2  # Should be dissimilar
    
    def test_word_encoding(self):
        """Test word encoding."""
        encoder = TextEncoder(dim=1000)
        
        hv_hello = encoder.encode_word("hello")
        hv_world = encoder.encode_word("world")
        
        assert hv_hello.dim == 1000
        assert hv_world.dim == 1000
        
        # Different words should be dissimilar
        similarity = hv_hello.cosine_similarity(hv_world).item()
        assert abs(similarity) < 0.3
    
    def test_sentence_encoding(self):
        """Test sentence encoding."""
        encoder = TextEncoder(dim=1000)
        
        hv1 = encoder.encode_sentence("Hello world")
        hv2 = encoder.encode_sentence("Goodbye world")
        
        assert hv1.dim == 1000
        assert hv2.dim == 1000
        
        # Similar sentences should have some similarity
        similarity = hv1.cosine_similarity(hv2).item()
        assert 0.1 < similarity < 0.8  # Some similarity due to "world"
    
    def test_encoding_methods(self):
        """Test different encoding methods."""
        encoder = TextEncoder(dim=1000)
        text = "hello"
        
        hv_char = encoder.encode(text, method="character")
        hv_token = encoder.encode(text, method="token")
        
        assert hv_char.dim == 1000
        assert hv_token.dim == 1000
        
        # Different methods should produce different results
        similarity = hv_char.cosine_similarity(hv_token).item()
        assert abs(similarity) < 0.8
    
    def test_empty_text(self):
        """Test encoding of empty text."""
        encoder = TextEncoder(dim=1000)
        
        hv = encoder.encode("", method="token")
        assert hv.dim == 1000
        
        # Should be zero vector
        assert torch.allclose(hv.data, torch.zeros(1000))
    
    def test_similarity_function(self):
        """Test text similarity function."""
        encoder = TextEncoder(dim=1000)
        
        sim1 = encoder.similarity("hello world", "hello world")
        sim2 = encoder.similarity("hello world", "goodbye world")
        sim3 = encoder.similarity("hello world", "random text")
        
        # Identical texts should have high similarity
        assert sim1 > 0.9
        
        # Related texts should have medium similarity
        assert 0.1 < sim2 < 0.8
        
        # Unrelated texts should have low similarity
        assert abs(sim3) < 0.3


class TestVisionEncoder:
    """Tests for vision encoding."""
    
    def test_initialization(self):
        """Test vision encoder initialization."""
        encoder = VisionEncoder(dim=5000)
        assert encoder.dim == 5000
        assert encoder.patch_size == 16
    
    def test_image_preprocessing(self):
        """Test image preprocessing."""
        encoder = VisionEncoder(dim=1000)
        
        # Test with random numpy array
        img_np = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        processed = encoder._preprocess_image(img_np)
        
        assert processed.shape == (1, 3, 224, 224)
        assert processed.device.type == encoder.device
    
    def test_holistic_encoding(self):
        """Test holistic image encoding."""
        encoder = VisionEncoder(dim=1000)
        
        # Create random image
        img = torch.rand(3, 224, 224)
        hv = encoder.encode_holistic(img)
        
        assert hv.dim == 1000
    
    def test_patch_encoding(self):
        """Test patch-based image encoding."""
        encoder = VisionEncoder(dim=1000)
        
        # Create random image
        img = torch.rand(3, 224, 224)
        hv = encoder.encode_patches(img)
        
        assert hv.dim == 1000
    
    def test_encoding_methods(self):
        """Test different encoding methods."""
        encoder = VisionEncoder(dim=1000)
        img = torch.rand(3, 224, 224)
        
        hv_holistic = encoder.encode(img, method="holistic")
        hv_patches = encoder.encode(img, method="patches")
        
        assert hv_holistic.dim == 1000
        assert hv_patches.dim == 1000
        
        # Different methods should produce different results
        similarity = hv_holistic.cosine_similarity(hv_patches).item()
        assert abs(similarity) < 0.9
    
    def test_pil_image_support(self):
        """Test PIL Image support."""
        encoder = VisionEncoder(dim=1000)
        
        # Create PIL image
        img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        pil_img = Image.fromarray(img_array)
        
        hv = encoder.encode(pil_img)
        assert hv.dim == 1000
    
    def test_similarity_function(self):
        """Test image similarity function."""
        encoder = VisionEncoder(dim=1000)
        
        img1 = torch.rand(3, 224, 224)
        img2 = torch.rand(3, 224, 224)
        
        # Same image should have high similarity
        sim1 = encoder.similarity(img1, img1)
        assert sim1 > 0.99
        
        # Different images should have lower similarity
        sim2 = encoder.similarity(img1, img2)
        assert sim2 < 0.9


class TestEEGEncoder:
    """Tests for EEG signal encoding."""
    
    def test_initialization(self):
        """Test EEG encoder initialization."""
        encoder = EEGEncoder(dim=5000)
        assert encoder.dim == 5000
        assert len(encoder.freq_bands) == 5  # delta, theta, alpha, beta, gamma
    
    def test_signal_preprocessing(self):
        """Test EEG signal preprocessing."""
        encoder = EEGEncoder(dim=1000)
        
        # Single channel signal
        signal_1d = np.random.randn(1000)
        processed = encoder._preprocess_signal(signal_1d, 250.0)
        
        assert processed.shape == (1, 1000)
        assert processed.device.type == encoder.device
        
        # Multi-channel signal
        signal_2d = np.random.randn(64, 1000)
        processed = encoder._preprocess_signal(signal_2d, 250.0)
        
        assert processed.shape == (64, 1000)
    
    def test_spectral_features(self):
        """Test spectral feature extraction."""
        encoder = EEGEncoder(dim=1000)
        
        # Create synthetic EEG with known frequency content
        fs = 250.0
        t = np.arange(0, 4, 1/fs)  # 4 seconds
        
        # 10 Hz alpha wave
        signal = np.sin(2 * np.pi * 10 * t).reshape(1, -1)
        signal_tensor = torch.from_numpy(signal).float()
        
        band_powers = encoder.extract_spectral_features(signal_tensor, fs)
        
        # Alpha band should have highest power
        alpha_power = band_powers['alpha'].item()
        assert alpha_power > band_powers['delta'].item()
        assert alpha_power > band_powers['theta'].item()
    
    def test_temporal_encoding(self):
        """Test temporal encoding."""
        encoder = EEGEncoder(dim=1000)
        
        # Random EEG signal
        signal = np.random.randn(4, 500)  # 4 channels, 500 samples
        hv = encoder.encode_temporal(signal, sampling_rate=250.0)
        
        assert hv.dim == 1000
    
    def test_spectral_encoding(self):
        """Test spectral encoding."""
        encoder = EEGEncoder(dim=1000)
        
        # Random EEG signal
        signal = np.random.randn(4, 500)
        hv = encoder.encode_spectral(signal, sampling_rate=250.0)
        
        assert hv.dim == 1000
    
    def test_combined_encoding(self):
        """Test combined temporal and spectral encoding."""
        encoder = EEGEncoder(dim=1000)
        
        signal = np.random.randn(4, 500)
        hv = encoder.encode(signal, sampling_rate=250.0, method="combined")
        
        assert hv.dim == 1000
    
    def test_encoding_methods(self):
        """Test different encoding methods."""
        encoder = EEGEncoder(dim=1000)
        signal = np.random.randn(2, 250)
        
        hv_temporal = encoder.encode(signal, method="temporal")
        hv_spectral = encoder.encode(signal, method="spectral")
        hv_combined = encoder.encode(signal, method="combined")
        
        assert hv_temporal.dim == 1000
        assert hv_spectral.dim == 1000
        assert hv_combined.dim == 1000
        
        # Different methods should produce different results
        sim1 = hv_temporal.cosine_similarity(hv_spectral).item()
        sim2 = hv_temporal.cosine_similarity(hv_combined).item()
        
        assert abs(sim1) < 0.8
        assert abs(sim2) < 0.8
    
    def test_similarity_function(self):
        """Test EEG similarity function."""
        encoder = EEGEncoder(dim=1000)
        
        signal1 = np.random.randn(4, 250)
        signal2 = np.random.randn(4, 250)
        
        # Same signal should have high similarity
        sim1 = encoder.similarity(signal1, signal1)
        assert sim1 > 0.99
        
        # Different signals should have lower similarity
        sim2 = encoder.similarity(signal1, signal2)
        assert sim2 < 0.9
    
    def test_single_channel_signal(self):
        """Test encoding of single channel EEG."""
        encoder = EEGEncoder(dim=1000)
        
        # Single channel signal (1D array)
        signal = np.random.randn(500)
        hv = encoder.encode(signal, sampling_rate=250.0)
        
        assert hv.dim == 1000


if __name__ == "__main__":
    pytest.main([__file__])