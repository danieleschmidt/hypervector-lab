"""Tests for application modules."""

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
import tempfile
import os

from hypervector.applications import BCIClassifier, CrossModalRetrieval


class TestBCIClassifier:
    """Tests for BCI classifier."""
    
    def test_initialization(self):
        """Test BCI classifier initialization."""
        bci = BCIClassifier(
            channels=32,
            sampling_rate=250.0,
            window_size=125,
            hypervector_dim=5000
        )
        
        assert bci.channels == 32
        assert bci.sampling_rate == 250.0
        assert bci.window_size == 125
        assert len(bci.class_prototypes) == 0
    
    def test_training_sample_addition(self):
        """Test adding training samples."""
        bci = BCIClassifier(hypervector_dim=1000)
        
        # Add training samples
        eeg_data1 = np.random.randn(8, 250)  # 8 channels, 250 samples
        eeg_data2 = np.random.randn(8, 250)
        
        bci.add_training_sample(eeg_data1, "class_A")
        bci.add_training_sample(eeg_data2, "class_B")
        
        assert "class_A" in bci.class_prototypes
        assert "class_B" in bci.class_prototypes
        assert bci.class_counts["class_A"] == 1
        assert bci.class_counts["class_B"] == 1
    
    def test_batch_training(self):
        """Test batch training."""
        bci = BCIClassifier(hypervector_dim=1000)
        
        # Create batch of training data
        eeg_samples = [np.random.randn(4, 125) for _ in range(10)]
        labels = ["class_A"] * 5 + ["class_B"] * 5
        
        bci.train_batch(eeg_samples, labels)
        
        assert len(bci.class_prototypes) == 2
        assert bci.class_counts["class_A"] == 5
        assert bci.class_counts["class_B"] == 5
    
    def test_classification(self):
        """Test EEG classification."""
        bci = BCIClassifier(hypervector_dim=1000)
        
        # Train with some data
        for i in range(5):
            eeg_a = np.random.randn(4, 125) + np.array([[1], [0], [0], [0]])  # Bias for class A
            eeg_b = np.random.randn(4, 125) + np.array([[0], [1], [0], [0]])  # Bias for class B
            bci.add_training_sample(eeg_a, "class_A")
            bci.add_training_sample(eeg_b, "class_B")
        
        # Test classification
        test_eeg = np.random.randn(4, 125)
        prediction, confidence = bci.classify(test_eeg, return_confidence=True)
        
        assert prediction in ["class_A", "class_B"]
        assert 0.0 <= confidence <= 1.0
    
    def test_streaming_classification(self):
        """Test streaming EEG classification."""
        bci = BCIClassifier(
            channels=1,
            window_size=10,
            hypervector_dim=1000
        )
        
        # Add some training data
        train_data = np.random.randn(1, 10)
        bci.add_training_sample(train_data, "test_class")
        
        # Feed streaming data
        results = []
        for i in range(15):
            result = bci.classify_streaming(np.random.randn())
            if result is not None:
                results.append(result)
        
        # Should get at least one classification when buffer fills
        assert len(results) >= 1
    
    def test_online_learning(self):
        """Test online learning and adaptation."""
        bci = BCIClassifier(hypervector_dim=1000, adaptation_rate=0.2)
        
        # Initial training
        eeg_data = np.random.randn(4, 125)
        bci.add_training_sample(eeg_data, "class_A")
        
        # Make prediction
        test_eeg = np.random.randn(4, 125)
        prediction, confidence = bci.classify(test_eeg)
        
        # Update with feedback (incorrect prediction)
        bci.update_online(test_eeg, "class_B", prediction)
        
        # Should have added new class
        assert "class_B" in bci.class_prototypes
        assert bci.get_accuracy() <= 1.0
    
    def test_confidence_tracking(self):
        """Test confidence tracking."""
        bci = BCIClassifier(hypervector_dim=1000)
        
        # Add training data
        eeg_data = np.random.randn(4, 125)
        bci.add_training_sample(eeg_data, "test_class")
        
        # Make several predictions
        for _ in range(5):
            test_eeg = np.random.randn(4, 125)
            bci.classify(test_eeg)
        
        # Should have confidence history
        confidence = bci.get_confidence()
        assert 0.0 <= confidence <= 1.0
    
    def test_model_saving_loading(self):
        """Test model save/load functionality."""
        bci = BCIClassifier(hypervector_dim=1000)
        
        # Train model
        eeg_data = np.random.randn(4, 125)
        bci.add_training_sample(eeg_data, "test_class")
        
        # Save model
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            temp_path = f.name
        
        try:
            bci.save_model(temp_path)
            
            # Create new classifier and load
            bci2 = BCIClassifier(hypervector_dim=1000)
            bci2.load_model(temp_path)
            
            # Should have same prototypes
            assert len(bci2.class_prototypes) == len(bci.class_prototypes)
            assert "test_class" in bci2.class_prototypes
        finally:
            os.unlink(temp_path)
    
    def test_metrics_tracking(self):
        """Test performance metrics tracking."""
        bci = BCIClassifier(hypervector_dim=1000)
        
        # Add training data
        eeg_data = np.random.randn(4, 125)
        bci.add_training_sample(eeg_data, "class_A")
        
        # Make predictions and update
        for i in range(10):
            test_eeg = np.random.randn(4, 125)
            prediction, _ = bci.classify(test_eeg)
            
            # Simulate correct/incorrect predictions
            true_label = "class_A" if i < 7 else "class_B"
            bci.update_online(test_eeg, true_label, prediction)
        
        accuracy = bci.get_accuracy()
        assert 0.0 <= accuracy <= 1.0
        
        class_accuracy = bci.get_class_accuracy("class_A")
        assert 0.0 <= class_accuracy <= 1.0


class TestCrossModalRetrieval:
    """Tests for cross-modal retrieval system."""
    
    def test_initialization(self):
        """Test retrieval system initialization."""
        retrieval = CrossModalRetrieval(dim=5000)
        assert retrieval.dim == 5000
        assert len(retrieval.indexed_items) == 0
    
    def test_item_addition(self):
        """Test adding multimodal items."""
        retrieval = CrossModalRetrieval(dim=1000)
        
        # Add text-only item
        retrieval.add_item("item1", text="Hello world")
        assert "item1" in retrieval.indexed_items
        assert "item1" in retrieval.text_index
        
        # Add image-only item
        image = torch.rand(3, 224, 224)
        retrieval.add_item("item2", image=image)
        assert "item2" in retrieval.image_index
        
        # Add multimodal item
        eeg_signal = np.random.randn(4, 250)
        retrieval.add_item("item3", text="EEG data", eeg=eeg_signal)
        assert "item3" in retrieval.text_index
        assert "item3" in retrieval.eeg_index
        assert "item3" in retrieval.multimodal_index
    
    def test_dataset_indexing(self):
        """Test batch dataset indexing."""
        retrieval = CrossModalRetrieval(dim=1000)
        
        # Create dataset
        texts = ["text one", "text two", "text three"]
        images = [torch.rand(3, 224, 224) for _ in range(3)]
        
        retrieval.index_dataset(images=images, texts=texts)
        
        assert len(retrieval.indexed_items) == 3
        assert len(retrieval.text_index) == 3
        assert len(retrieval.image_index) == 3
    
    def test_text_query(self):
        """Test querying by text."""
        retrieval = CrossModalRetrieval(dim=1000)
        
        # Index some items
        retrieval.add_item("doc1", text="machine learning algorithms")
        retrieval.add_item("doc2", text="deep neural networks")
        retrieval.add_item("doc3", text="cooking recipes")
        
        # Query with related text
        results = retrieval.query_by_text("artificial intelligence", top_k=2)
        
        assert len(results) <= 2
        assert all(len(result) == 3 for result in results)  # (id, score, data)
        
        # Results should be sorted by similarity
        if len(results) > 1:
            assert results[0][1] >= results[1][1]
    
    def test_image_query(self):
        """Test querying by image."""
        retrieval = CrossModalRetrieval(dim=1000)
        
        # Index some images
        for i in range(3):
            image = torch.rand(3, 224, 224)
            retrieval.add_item(f"img{i}", image=image)
        
        # Query with new image
        query_image = torch.rand(3, 224, 224)
        results = retrieval.query_by_image(query_image, top_k=2)
        
        assert len(results) <= 2
    
    def test_eeg_query(self):
        """Test querying by EEG."""
        retrieval = CrossModalRetrieval(dim=1000)
        
        # Index some EEG signals
        for i in range(3):
            eeg = np.random.randn(4, 250)
            retrieval.add_item(f"eeg{i}", eeg=eeg)
        
        # Query with new EEG
        query_eeg = np.random.randn(4, 250)
        results = retrieval.query_by_eeg(query_eeg, top_k=2)
        
        assert len(results) <= 2
    
    def test_cross_modal_similarity(self):
        """Test cross-modal similarity computation."""
        retrieval = CrossModalRetrieval(dim=1000)
        
        text = "test text"
        image = torch.rand(3, 224, 224)
        
        similarity = retrieval.compute_cross_modal_similarity(text, image)
        assert -1.0 <= similarity <= 1.0
    
    def test_modality_specific_search(self):
        """Test searching within specific modalities."""
        retrieval = CrossModalRetrieval(dim=1000)
        
        # Add multimodal items
        for i in range(3):
            retrieval.add_item(
                f"item{i}",
                text=f"text {i}",
                image=torch.rand(3, 224, 224)
            )
        
        # Search only in text modality
        text_results = retrieval.query_by_text("text", modality="text", top_k=3)
        assert len(text_results) == 3
        
        # Search only in image modality  
        image_results = retrieval.query_by_text("text", modality="image", top_k=3)
        assert len(image_results) == 3
        
        # Results might be different due to different modalities
        # (though with random data, hard to guarantee)
    
    def test_statistics(self):
        """Test retrieval statistics."""
        retrieval = CrossModalRetrieval(dim=1000)
        
        # Add various items
        retrieval.add_item("text_only", text="hello")
        retrieval.add_item("image_only", image=torch.rand(3, 224, 224))
        retrieval.add_item("multimodal", text="world", image=torch.rand(3, 224, 224))
        
        stats = retrieval.get_statistics()
        
        assert stats['total_items'] == 3
        assert stats['text_items'] == 2
        assert stats['image_items'] == 2
        assert stats['multimodal_items'] == 2  # text_+image and multimodal items
    
    def test_index_clearing(self):
        """Test index clearing."""
        retrieval = CrossModalRetrieval(dim=1000)
        
        # Add some items
        retrieval.add_item("test", text="hello")
        assert len(retrieval.indexed_items) == 1
        
        # Clear index
        retrieval.clear_index()
        assert len(retrieval.indexed_items) == 0
        assert len(retrieval.text_index) == 0
    
    def test_save_load_index(self):
        """Test index save/load functionality."""
        retrieval = CrossModalRetrieval(dim=1000)
        
        # Add some items
        retrieval.add_item("test1", text="hello world")
        retrieval.add_item("test2", image=torch.rand(3, 224, 224))
        
        # Save index
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            temp_path = f.name
        
        try:
            retrieval.save_index(temp_path)
            
            # Create new system and load
            retrieval2 = CrossModalRetrieval(dim=1000)
            retrieval2.load_index(temp_path)
            
            # Should have same items
            assert len(retrieval2.indexed_items) == 2
            assert "test1" in retrieval2.text_index
            assert "test2" in retrieval2.image_index
        finally:
            os.unlink(temp_path)
    
    def test_empty_query_handling(self):
        """Test handling of queries on empty index."""
        retrieval = CrossModalRetrieval(dim=1000)
        
        # Query empty index
        results = retrieval.query_by_text("test query")
        assert len(results) == 0
    
    def test_pil_image_support(self):
        """Test PIL Image support in retrieval."""
        retrieval = CrossModalRetrieval(dim=1000)
        
        # Create PIL image
        img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        pil_img = Image.fromarray(img_array)
        
        # Add item with PIL image
        retrieval.add_item("pil_test", image=pil_img)
        
        assert "pil_test" in retrieval.image_index
        
        # Query with PIL image
        results = retrieval.query_by_image(pil_img)
        assert len(results) == 1


if __name__ == "__main__":
    pytest.main([__file__])