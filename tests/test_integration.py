"""Integration tests for complete workflows."""

import pytest
import torch
import numpy as np
from PIL import Image
import tempfile
import os

import hypervector as hv
from hypervector.core import HDCSystem
from hypervector.applications import BCIClassifier, CrossModalRetrieval


class TestIntegration:
    """Integration tests for complete HDC workflows."""
    
    def test_readme_quickstart_example(self):
        """Test the example from README.md."""
        # Initialize HDC system with 10,000 dimensions
        hdc = HDCSystem(dim=10000, device='cpu')  # Force CPU for testing
        
        # Multi-modal encoding
        text_hv = hdc.encode_text("Hyperdimensional computing is fascinating")
        image_hv = hdc.encode_image(torch.rand(3, 224, 224))
        eeg_hv = hdc.encode_eeg(np.random.randn(4, 250), sampling_rate=250)
        
        # Verify encodings
        assert text_hv.dim == 10000
        assert image_hv.dim == 10000
        assert eeg_hv.dim == 10000
        
        # Bind modalities
        multimodal_hv = hdc.bind([text_hv, image_hv, eeg_hv])
        assert multimodal_hv.dim == 10000
        
        # Create query vector for similarity
        query_hv = hdc.random_hypervector()
        
        # Similarity search
        similarity = hdc.cosine_similarity(multimodal_hv, query_hv)
        assert -1.0 <= similarity.item() <= 1.0
    
    def test_bci_workflow(self):
        """Test complete BCI classification workflow."""
        # Real-time EEG classification example from README
        bci_system = BCIClassifier(
            channels=4,  # Reduced for testing
            sampling_rate=250,
            window_size=125,  # Reduced for testing
            hypervector_dim=5000  # Reduced for testing
        )
        
        # Generate synthetic training data
        n_samples = 20
        eeg_class_a = []
        eeg_class_b = []
        
        for i in range(n_samples):
            # Class A: higher activity in first channel
            eeg_a = np.random.randn(4, 125)
            eeg_a[0, :] += 2.0  # Add bias to first channel
            eeg_class_a.append(eeg_a)
            
            # Class B: higher activity in second channel
            eeg_b = np.random.randn(4, 125)
            eeg_b[1, :] += 2.0  # Add bias to second channel
            eeg_class_b.append(eeg_b)
        
        # Train the system
        all_samples = eeg_class_a + eeg_class_b
        all_labels = ["motor_imagery_left"] * n_samples + ["motor_imagery_right"] * n_samples
        
        bci_system.train_batch(all_samples, all_labels)
        
        # Test classification
        test_eeg = np.random.randn(4, 125)
        test_eeg[0, :] += 1.5  # Should be closer to class A
        
        prediction, confidence = bci_system.classify(test_eeg)
        assert prediction in ["motor_imagery_left", "motor_imagery_right"]
        assert 0.0 <= confidence <= 1.0
        
        # Test streaming classification
        for sample in test_eeg.flatten()[:15]:  # Feed first 15 samples
            result = bci_system.classify_streaming(sample)
            if result is not None:
                pred, conf = result
                assert pred in ["motor_imagery_left", "motor_imagery_right"]
                assert 0.0 <= conf <= 1.0
    
    def test_multimodal_retrieval_workflow(self):
        """Test complete multimodal retrieval workflow."""
        # Cross-modal retrieval example from README
        retrieval_system = CrossModalRetrieval(dim=5000)  # Reduced for testing
        
        # Create synthetic multimodal dataset
        images = [torch.rand(3, 224, 224) for _ in range(5)]
        texts = [
            "mountain landscape with snow",
            "city skyline at night", 
            "ocean waves crashing",
            "forest with tall trees",
            "desert sand dunes"
        ]
        eeg_samples = [np.random.randn(4, 250) for _ in range(5)]
        
        # Index the dataset
        retrieval_system.index_dataset(images, texts, eeg_samples)
        
        # Verify indexing
        stats = retrieval_system.get_statistics()
        assert stats['total_items'] == 5
        assert stats['text_items'] == 5
        assert stats['image_items'] == 5
        assert stats['eeg_items'] == 5
        
        # Query with text
        results = retrieval_system.query_by_text("mountain landscape")
        assert len(results) > 0
        
        # Should find the mountain landscape item with highest similarity
        best_match = results[0]
        item_id, similarity, item_data = best_match
        assert 'text' in item_data
        assert similarity >= -1.0 and similarity <= 1.0
        
        # Query with image
        query_image = torch.rand(3, 224, 224)
        image_results = retrieval_system.query_by_image(query_image, top_k=3)
        assert len(image_results) <= 3
        
        # Query with EEG
        query_eeg = np.random.randn(4, 250)
        eeg_results = retrieval_system.query_by_eeg(query_eeg, top_k=2)
        assert len(eeg_results) <= 2
    
    def test_cross_modal_binding_workflow(self):
        """Test binding across different modalities."""
        hdc = HDCSystem(dim=5000, device='cpu')
        
        # Encode different aspects of the same concept
        visual_hv = hdc.encode_image(torch.rand(3, 224, 224))
        semantic_hv = hdc.encode_text("beautiful sunset")
        neural_hv = hdc.encode_eeg(np.random.randn(8, 500), sampling_rate=250)
        
        # Create concept representation by binding all modalities
        concept_hv = hdc.bind([visual_hv, semantic_hv, neural_hv])
        
        # Store in memory
        hdc.store("sunset_concept", concept_hv)
        
        # Test retrieval
        retrieved = hdc.retrieve("sunset_concept")
        assert retrieved is not None
        assert torch.allclose(retrieved.data, concept_hv.data)
        
        # Test similarity-based querying
        query_hv = hdc.encode_text("sunset")  # Related query
        results = hdc.query_memory(query_hv, top_k=1)
        
        assert len(results) == 1
        assert results[0][0] == "sunset_concept"
    
    def test_online_learning_adaptation(self):
        """Test online learning and adaptation workflow."""
        bci = BCIClassifier(
            channels=2,
            hypervector_dim=2000,
            adaptation_rate=0.3
        )
        
        # Initial training with limited data
        initial_eeg = np.random.randn(2, 125)
        bci.add_training_sample(initial_eeg, "rest")
        
        # Simulate online usage with feedback
        accuracy_history = []
        
        for trial in range(20):
            # Generate test signal
            test_eeg = np.random.randn(2, 125)
            
            # Add some bias based on "true" class
            true_class = "rest" if trial % 3 == 0 else "active"
            if true_class == "active":
                test_eeg[0, :] += 1.0  # Add activation pattern
            
            # Make prediction
            prediction, confidence = bci.classify(test_eeg)
            
            # Provide feedback (online learning)
            bci.update_online(test_eeg, true_class, prediction)
            
            # Track accuracy
            current_accuracy = bci.get_accuracy()
            accuracy_history.append(current_accuracy)
        
        # System should have learned both classes
        assert len(bci.class_prototypes) == 2
        assert "rest" in bci.class_prototypes
        assert "active" in bci.class_prototypes
        
        # Accuracy should be reasonable (not necessarily perfect due to random data)
        final_accuracy = bci.get_accuracy()
        assert 0.0 <= final_accuracy <= 1.0
    
    def test_memory_and_similarity_workflow(self):
        """Test memory storage and similarity-based retrieval."""
        hdc = HDCSystem(dim=3000, device='cpu')
        
        # Create and store multiple concept vectors
        concepts = {
            "dog": hdc.encode_text("loyal pet animal"),
            "cat": hdc.encode_text("independent feline pet"),
            "bird": hdc.encode_text("flying animal with wings"),
            "fish": hdc.encode_text("swimming aquatic animal")
        }
        
        # Store all concepts
        for name, concept_hv in concepts.items():
            hdc.store(name, concept_hv)
        
        # Test similarity-based queries
        pet_query = hdc.encode_text("domestic pet")
        pet_results = hdc.query_memory(pet_query, top_k=2)
        
        # Should find pet-related concepts (dog, cat) with higher similarity
        assert len(pet_results) == 2
        top_matches = [result[0] for result in pet_results]
        
        # Test animal query
        animal_query = hdc.encode_text("animal")
        animal_results = hdc.query_memory(animal_query, top_k=4)
        
        assert len(animal_results) == 4  # Should find all animals
        
        # Test direct retrieval
        dog_hv = hdc.retrieve("dog")
        assert dog_hv is not None
        
        # Verify similarity
        dog_similarity = hdc.cosine_similarity(concepts["dog"], dog_hv)
        assert dog_similarity.item() > 0.99  # Should be nearly identical
    
    def test_error_handling_and_edge_cases(self):
        """Test error handling and edge cases."""
        hdc = HDCSystem(dim=1000, device='cpu')
        
        # Test empty text encoding
        empty_hv = hdc.encode_text("")
        assert empty_hv.dim == 1000
        assert torch.allclose(empty_hv.data, torch.zeros(1000))
        
        # Test single-sample EEG
        single_sample = np.array([[1.0]])  # 1 channel, 1 sample
        single_hv = hdc.encode_eeg(single_sample)
        assert single_hv.dim == 1000
        
        # Test BCI with no training data
        bci = BCIClassifier(hypervector_dim=1000)
        test_eeg = np.random.randn(4, 125)
        
        with pytest.raises(RuntimeError):
            bci.classify(test_eeg)
        
        # Test empty retrieval system
        retrieval = CrossModalRetrieval(dim=1000)
        empty_results = retrieval.query_by_text("test")
        assert len(empty_results) == 0
        
        # Test memory queries on empty system
        empty_memory_results = hdc.query_memory(hdc.random_hypervector())
        assert len(empty_memory_results) == 0
    
    def test_device_consistency(self):
        """Test device consistency across operations."""
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
            
        hdc = HDCSystem(dim=1000, device=device)
        
        # All encodings should be on correct device
        text_hv = hdc.encode_text("test")
        assert text_hv.data.device.type == device
        
        image_hv = hdc.encode_image(torch.rand(3, 224, 224))
        assert image_hv.data.device.type == device
        
        eeg_hv = hdc.encode_eeg(np.random.randn(2, 100))
        assert eeg_hv.data.device.type == device
        
        # Operations should preserve device
        bound_hv = hdc.bind([text_hv, image_hv])
        assert bound_hv.data.device.type == device
        
        # Memory operations should preserve device
        hdc.store("test", bound_hv)
        retrieved = hdc.retrieve("test")
        assert retrieved.data.device.type == device


if __name__ == "__main__":
    pytest.main([__file__])