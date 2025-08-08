"""Brain-Computer Interface classifier using hyperdimensional computing."""

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
from typing import Dict, List, Optional, Union, Tuple
from collections import defaultdict, deque

from ..core.hypervector import HyperVector
from ..core.operations import bundle, cosine_similarity
from ..encoders.eeg import EEGEncoder


class BCIClassifier:
    """
    Real-time EEG classification system using hyperdimensional computing.
    
    Supports online learning and adaptation for brain-computer interfaces.
    """
    
    def __init__(
        self,
        channels: int = 64,
        sampling_rate: float = 1000.0,
        window_size: int = 250,
        hypervector_dim: int = 10000,
        device: Optional[str] = None,
        adaptation_rate: float = 0.1
    ):
        """Initialize BCI classifier.
        
        Args:
            channels: Number of EEG channels
            sampling_rate: Sampling rate in Hz
            window_size: Window size in samples
            hypervector_dim: Hypervector dimensionality
            device: Compute device
            adaptation_rate: Learning rate for online adaptation
        """
        self.channels = channels
        self.sampling_rate = sampling_rate
        self.window_size = window_size
        self.adaptation_rate = adaptation_rate
        
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize EEG encoder
        self.encoder = EEGEncoder(dim=hypervector_dim, device=self.device)
        
        # Class prototypes (learned hypervectors for each class)
        self.class_prototypes: Dict[str, HyperVector] = {}
        self.class_counts: Dict[str, int] = defaultdict(int)
        
        # Buffer for streaming data
        self.buffer = deque(maxlen=window_size)
        
        # Confidence tracking
        self.confidence_history = deque(maxlen=100)
        
        # Performance metrics
        self.metrics = {
            'total_predictions': 0,
            'correct_predictions': 0,
            'class_accuracies': defaultdict(list)
        }
    
    def add_training_sample(
        self, 
        eeg_data: Union[torch.Tensor, "np.ndarray"], 
        label: str
    ) -> None:
        """Add labeled training sample to build class prototypes.
        
        Args:
            eeg_data: EEG data (channels x samples)
            label: Class label
        """
        # Encode EEG data
        sample_hv = self.encoder.encode(eeg_data, self.sampling_rate)
        
        if label not in self.class_prototypes:
            # First sample for this class
            self.class_prototypes[label] = sample_hv
            self.class_counts[label] = 1
        else:
            # Update prototype using incremental averaging
            old_prototype = self.class_prototypes[label]
            count = self.class_counts[label]
            
            # Weighted average: new_proto = (old_proto * count + new_sample) / (count + 1)
            updated_proto = (old_prototype * count + sample_hv) * (1.0 / (count + 1))
            self.class_prototypes[label] = updated_proto.normalize()
            self.class_counts[label] += 1
    
    def train_batch(
        self, 
        eeg_samples: List[Union[torch.Tensor, "np.ndarray"]], 
        labels: List[str]
    ) -> None:
        """Train on batch of labeled samples."""
        if len(eeg_samples) != len(labels):
            raise ValueError("Number of samples must match number of labels")
        
        for eeg_data, label in zip(eeg_samples, labels):
            self.add_training_sample(eeg_data, label)
    
    def classify(
        self, 
        eeg_data: Union[torch.Tensor, "np.ndarray"],
        return_confidence: bool = True
    ) -> Union[str, Tuple[str, float]]:
        """Classify single EEG sample.
        
        Args:
            eeg_data: EEG data (channels x samples)
            return_confidence: Whether to return confidence score
            
        Returns:
            Predicted class label or (label, confidence) tuple
        """
        if not self.class_prototypes:
            raise RuntimeError("No training data available. Call add_training_sample first.")
        
        # Encode sample
        sample_hv = self.encoder.encode(eeg_data, self.sampling_rate)
        
        # Compute similarities to all class prototypes
        similarities = {}
        for label, prototype in self.class_prototypes.items():
            sim = cosine_similarity(sample_hv, prototype).item()
            similarities[label] = sim
        
        # Find best match
        predicted_label = max(similarities, key=similarities.get)
        confidence = similarities[predicted_label]
        
        # Update metrics
        self.metrics['total_predictions'] += 1
        self.confidence_history.append(confidence)
        
        if return_confidence:
            return predicted_label, confidence
        else:
            return predicted_label
    
    def classify_streaming(self, new_sample: float) -> Optional[Tuple[str, float]]:
        """Classify from streaming single-channel data.
        
        Args:
            new_sample: New EEG sample value
            
        Returns:
            (predicted_label, confidence) if buffer is full, None otherwise
        """
        self.buffer.append(new_sample)
        
        if len(self.buffer) == self.window_size:
            # Convert buffer to tensor and add channel dimension
            eeg_window = torch.tensor(list(self.buffer)).unsqueeze(0)
            return self.classify(eeg_window, return_confidence=True)
        
        return None
    
    def update_online(
        self, 
        eeg_data: Union[torch.Tensor, "np.ndarray"], 
        true_label: str,
        predicted_label: str
    ) -> None:
        """Update model online based on feedback.
        
        Args:
            eeg_data: EEG data that was classified
            true_label: Ground truth label
            predicted_label: Model's prediction
        """
        # Update accuracy metrics
        is_correct = (true_label == predicted_label)
        if is_correct:
            self.metrics['correct_predictions'] += 1
        
        self.metrics['class_accuracies'][true_label].append(is_correct)
        
        # Adaptive learning: update prototype based on error
        if not is_correct:
            # Encode the sample
            sample_hv = self.encoder.encode(eeg_data, self.sampling_rate)
            
            # Move true class prototype toward sample
            if true_label in self.class_prototypes:
                old_prototype = self.class_prototypes[true_label]
                updated_prototype = (
                    old_prototype * (1 - self.adaptation_rate) + 
                    sample_hv * self.adaptation_rate
                )
                self.class_prototypes[true_label] = updated_prototype.normalize()
            else:
                # New class discovered online
                self.class_prototypes[true_label] = sample_hv
                self.class_counts[true_label] = 1
    
    def get_confidence(self) -> float:
        """Get recent average confidence."""
        if not self.confidence_history:
            return 0.0
        return float(np.mean(self.confidence_history))
    
    def get_accuracy(self) -> float:
        """Get overall classification accuracy."""
        if self.metrics['total_predictions'] == 0:
            return 0.0
        return self.metrics['correct_predictions'] / self.metrics['total_predictions']
    
    def get_class_accuracy(self, class_label: str) -> float:
        """Get accuracy for specific class."""
        if class_label not in self.metrics['class_accuracies']:
            return 0.0
        accuracies = self.metrics['class_accuracies'][class_label]
        if not accuracies:
            return 0.0
        return sum(accuracies) / len(accuracies)
    
    def reset_metrics(self) -> None:
        """Reset performance metrics."""
        self.metrics = {
            'total_predictions': 0,
            'correct_predictions': 0,
            'class_accuracies': defaultdict(list)
        }
        self.confidence_history.clear()
    
    def save_model(self, filepath: str) -> None:
        """Save trained model."""
        model_data = {
            'class_prototypes': {k: v.data.cpu() for k, v in self.class_prototypes.items()},
            'class_counts': dict(self.class_counts),
            'config': {
                'channels': self.channels,
                'sampling_rate': self.sampling_rate,
                'window_size': self.window_size,
                'hypervector_dim': self.encoder.dim,
                'adaptation_rate': self.adaptation_rate
            }
        }
        torch.save(model_data, filepath)
    
    def load_model(self, filepath: str) -> None:
        """Load trained model."""
        model_data = torch.load(filepath, map_location=self.device)
        
        # Restore prototypes
        self.class_prototypes = {}
        for label, data in model_data['class_prototypes'].items():
            self.class_prototypes[label] = HyperVector(
                data.to(self.device), 
                device=self.device
            )
        
        self.class_counts = defaultdict(int, model_data['class_counts'])
        
        # Update config if provided
        if 'config' in model_data:
            config = model_data['config']
            self.channels = config.get('channels', self.channels)
            self.sampling_rate = config.get('sampling_rate', self.sampling_rate)
            self.window_size = config.get('window_size', self.window_size)
            self.adaptation_rate = config.get('adaptation_rate', self.adaptation_rate)
    
    def __repr__(self) -> str:
        return (f"BCIClassifier(channels={self.channels}, "
                f"sampling_rate={self.sampling_rate}, "
                f"classes={len(self.class_prototypes)}, "
                f"accuracy={self.get_accuracy():.3f})")