# HyperVector-Lab API Reference 📚

## Core Classes

### HyperVector

The fundamental hyperdimensional vector class.

```python
from hypervector import HyperVector

# Create from data
hv = HyperVector([1.0, 2.0, 3.0, 4.0])

# Create random vector
hv = HyperVector.random(dim=10000, seed=42)

# Properties
print(hv.dim)        # Dimensionality
print(hv.mode)       # Storage mode ('dense', 'binary', 'ternary')
print(hv.device)     # Compute device
```

**Methods:**
- `to(device)` - Move to device
- `normalize()` - L2 normalize
- `to_binary()` - Convert to binary {-1, 1}
- `to_ternary()` - Convert to ternary {-1, 0, 1}
- `similarity(other)` - Cosine similarity

### HDCSystem

High-level orchestration for HDC operations.

```python
from hypervector import HDCSystem

hdc = HDCSystem(dim=10000, device='cuda')

# Multi-modal encoding
text_hv = hdc.encode_text("Hello world")
image_hv = hdc.encode_image(image_tensor)
eeg_hv = hdc.encode_eeg(eeg_data, sampling_rate=250)
```

**Methods:**
- `encode_text(text)` - Encode text to hypervector
- `encode_image(image)` - Encode image to hypervector  
- `encode_eeg(data, fs)` - Encode EEG to hypervector
- `bind(hv1, hv2)` - Bind two hypervectors
- `bundle(hvs)` - Bundle multiple hypervectors

## Core Operations

### Binding
```python
from hypervector import bind

# Element-wise multiplication
result = bind(hv1, hv2)
```

### Bundling
```python
from hypervector import bundle

# Superposition of vectors
result = bundle([hv1, hv2, hv3])
```

### Permutation
```python
from hypervector import permute

# Circular shift
shifted = permute(hv, shift=1)
```

### Similarity
```python
from hypervector import cosine_similarity

# Compute similarity
sim = cosine_similarity(hv1, hv2)
```

## Encoders

### TextEncoder

```python
from hypervector.encoders import TextEncoder

encoder = TextEncoder(dim=10000, device='cuda')

# Character-level encoding
char_hv = encoder.encode_character('a')

# Word-level encoding
word_hv = encoder.encode_word("hello")

# Sentence-level encoding
sent_hv = encoder.encode_sentence("Hello world")
```

**Parameters:**
- `dim`: Hypervector dimensionality
- `device`: Compute device
- `position_encoding`: Include position information
- `cache_size`: Number of vectors to cache

### VisionEncoder

```python
from hypervector.encoders import VisionEncoder

encoder = VisionEncoder(dim=10000, device='cuda')

# Holistic image encoding
image_hv = encoder.encode_image(image_tensor)

# Patch-based encoding
patches_hv = encoder.encode_patches(image_tensor, patch_size=16)
```

**Parameters:**
- `dim`: Hypervector dimensionality
- `device`: Compute device
- `patch_size`: Size of image patches
- `feature_dim`: CNN feature dimension

### EEGEncoder

```python
from hypervector.encoders import EEGEncoder

encoder = EEGEncoder(dim=10000, device='cuda')

# Temporal encoding
temporal_hv = encoder.encode_temporal(eeg_data)

# Spectral encoding
spectral_hv = encoder.encode_spectral(eeg_data, fs=250)

# Combined encoding
combined_hv = encoder.encode_combined(eeg_data, fs=250)
```

**Parameters:**
- `dim`: Hypervector dimensionality
- `device`: Compute device
- `freq_bands`: EEG frequency bands
- `window_size`: Temporal window size

## Applications

### BCIClassifier

Real-time brain-computer interface classification.

```python
from hypervector.applications import BCIClassifier

bci = BCIClassifier(
    channels=64,
    sampling_rate=1000,
    window_size=250,
    hypervector_dim=10000
)

# Train classifier
bci.fit(eeg_windows, labels)

# Real-time classification
prediction = bci.classify(eeg_window)
confidence = bci.get_confidence()
```

**Methods:**
- `fit(X, y)` - Train classifier
- `classify(eeg_window)` - Classify EEG window
- `update(eeg_window, label)` - Online learning
- `get_confidence()` - Get prediction confidence

### CrossModalRetrieval

Multi-modal similarity search and retrieval.

```python
from hypervector.applications import CrossModalRetrieval

retrieval = CrossModalRetrieval(dim=10000)

# Index multi-modal data
retrieval.index_texts(texts)
retrieval.index_images(images) 
retrieval.index_eeg(eeg_samples)

# Cross-modal queries
results = retrieval.query_by_text("mountain landscape")
results = retrieval.query_by_image(query_image)
```

**Methods:**
- `index_texts(texts)` - Index text documents
- `index_images(images)` - Index images
- `index_eeg(samples)` - Index EEG samples
- `query_by_text(text)` - Text-based retrieval
- `query_by_image(image)` - Image-based retrieval

## Performance Optimization

### CPUAccelerator

```python
from hypervector.accelerators import CPUAccelerator

accelerator = CPUAccelerator(num_threads=8)

# Vectorized operations
result = accelerator.vectorized_bind(hv1, hv2)
bundled = accelerator.vectorized_bundle(hvs)
```

### BatchProcessor

```python
from hypervector.accelerators import BatchProcessor

processor = BatchProcessor(
    batch_size=1000,
    num_workers=4,
    device='cuda'
)

# Batch processing
results = processor.process_texts(large_text_list)
```

### MemoryManager

```python
from hypervector.accelerators import MemoryManager

manager = MemoryManager(
    pool_size=10000,
    cleanup_threshold=0.8
)

# Get managed vectors
hv = manager.get_vector(dim=10000)
manager.return_vector(hv)
```

## Benchmarking

### Running Benchmarks

```python
from hypervector.benchmark import run_benchmarks

# Run all benchmarks
results = run_benchmarks(device='cuda')

# Specific benchmarks
results = run_benchmarks(
    operations=['bind', 'bundle'],
    dims=[1000, 10000],
    device='cuda'
)
```

### Profiling

```python
from hypervector.benchmark import start_profiling, get_profile_report

# Enable profiling
start_profiling()

# Your code here
result = hdc.encode_text("example")

# Get report
report = get_profile_report()
```

## Utilities

### Configuration

```python
from hypervector.utils.config import get_config, set_config

# Get current config
config = get_config()

# Set configuration
set_config('device', 'cuda')
set_config('log_level', 'DEBUG')
```

### Logging

```python
from hypervector.utils.logging import setup_logging, get_logger

# Setup logging
setup_logging(level='INFO', file='hypervector.log')

# Get logger
logger = get_logger(__name__)
```

### Validation

```python
from hypervector.utils.validation import validate_input, validate_dimensions

# Validate inputs
validate_input(data, expected_type=torch.Tensor)
validate_dimensions(hv1, hv2)  # Check compatibility
```

## Error Handling

### Custom Exceptions

```python
from hypervector.core.exceptions import (
    HDCError,
    DimensionMismatchError,
    InvalidModeError,
    DeviceError,
    EncodingError
)

try:
    result = bind(hv1, hv2)
except DimensionMismatchError as e:
    print(f"Dimension mismatch: {e}")
```

## Examples

### Basic Usage
```python
import hypervector as hv

# Create HDC system
hdc = hv.HDCSystem(dim=10000, device='cuda')

# Multi-modal encoding
text_hv = hdc.encode_text("artificial intelligence")
image_hv = hdc.encode_image(ai_image)

# Bind modalities
multimodal = hdc.bind(text_hv, image_hv)

# Similarity search
similarity = hdc.cosine_similarity(multimodal, query_hv)
```

### BCI Application
```python
from hypervector.applications import BCIClassifier

# Setup BCI system
bci = BCIClassifier(
    channels=32,
    sampling_rate=500,
    hypervector_dim=10000
)

# Train on labeled EEG data
bci.fit(training_eeg, training_labels)

# Real-time classification
for eeg_window in eeg_stream:
    prediction = bci.classify(eeg_window)
    print(f"Predicted class: {prediction}")
```

### Cross-Modal Search
```python
from hypervector.applications import CrossModalRetrieval

# Setup retrieval system
retrieval = CrossModalRetrieval(dim=10000)

# Index multimedia content
retrieval.index_texts(documents)
retrieval.index_images(image_collection)

# Search with text query
results = retrieval.query_by_text("sunset over mountains")
print(f"Found {len(results)} similar images")
```