"""Input validation utilities for robust operation."""

import torch
import numpy as np
from typing import Union, List, Any, Optional
from PIL import Image


class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass


def validate_input(
    data: Any,
    expected_types: Union[type, tuple],
    name: str = "input"
) -> None:
    """Validate input data type.
    
    Args:
        data: Input data to validate
        expected_types: Expected type(s)
        name: Name of the input for error messages
        
    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(data, expected_types):
        if isinstance(expected_types, tuple):
            type_names = [t.__name__ for t in expected_types]
            expected_str = " or ".join(type_names)
        else:
            expected_str = expected_types.__name__
            
        raise ValidationError(
            f"{name} must be {expected_str}, got {type(data).__name__}"
        )


def validate_dimensions(
    data: Union[torch.Tensor, np.ndarray],
    expected_dims: Union[int, List[int]],
    name: str = "data"
) -> None:
    """Validate tensor/array dimensions.
    
    Args:
        data: Input tensor or array
        expected_dims: Expected number of dimensions (or list of valid dimensions)
        name: Name of the input for error messages
        
    Raises:
        ValidationError: If validation fails
    """
    actual_dims = data.ndim if hasattr(data, 'ndim') else len(data.shape)
    
    if isinstance(expected_dims, int):
        expected_dims = [expected_dims]
    
    if actual_dims not in expected_dims:
        raise ValidationError(
            f"{name} must have {expected_dims} dimensions, got {actual_dims}"
        )


def validate_range(
    value: Union[int, float],
    min_val: Optional[Union[int, float]] = None,
    max_val: Optional[Union[int, float]] = None,
    name: str = "value"
) -> None:
    """Validate value is within specified range.
    
    Args:
        value: Value to validate
        min_val: Minimum allowed value (inclusive)
        max_val: Maximum allowed value (inclusive)
        name: Name of the value for error messages
        
    Raises:
        ValidationError: If validation fails
    """
    if min_val is not None and value < min_val:
        raise ValidationError(f"{name} must be >= {min_val}, got {value}")
    
    if max_val is not None and value > max_val:
        raise ValidationError(f"{name} must be <= {max_val}, got {value}")


def validate_positive(value: Union[int, float], name: str = "value") -> None:
    """Validate value is positive.
    
    Args:
        value: Value to validate
        name: Name of the value for error messages
        
    Raises:
        ValidationError: If validation fails
    """
    if value <= 0:
        raise ValidationError(f"{name} must be positive, got {value}")


def validate_non_empty(
    data: Union[List, str, torch.Tensor, np.ndarray],
    name: str = "data"
) -> None:
    """Validate data is not empty.
    
    Args:
        data: Data to validate
        name: Name of the data for error messages
        
    Raises:
        ValidationError: If validation fails
    """
    if len(data) == 0:
        raise ValidationError(f"{name} cannot be empty")


def validate_eeg_signal(
    signal: Union[torch.Tensor, np.ndarray],
    sampling_rate: float,
    name: str = "EEG signal"
) -> None:
    """Validate EEG signal data.
    
    Args:
        signal: EEG signal data
        sampling_rate: Sampling rate in Hz
        name: Name for error messages
        
    Raises:
        ValidationError: If validation fails
    """
    validate_input(signal, (torch.Tensor, np.ndarray), name)
    validate_dimensions(signal, [1, 2], name)
    validate_positive(sampling_rate, "sampling_rate")
    
    # Check for reasonable sampling rate
    validate_range(sampling_rate, 1.0, 10000.0, "sampling_rate")
    
    # Check for reasonable signal values (assuming microvolts)
    if hasattr(signal, 'data'):
        signal_data = signal.data
    else:
        signal_data = signal
        
    max_val = np.max(np.abs(signal_data))
    if max_val > 1000:  # Likely in wrong units
        raise ValidationError(
            f"{name} values seem too large (max={max_val}). "
            "Expected microvolts range."
        )


def validate_image(
    image: Union[torch.Tensor, np.ndarray, Image.Image],
    name: str = "image"
) -> None:
    """Validate image data.
    
    Args:
        image: Image data
        name: Name for error messages
        
    Raises:
        ValidationError: If validation fails
    """
    validate_input(image, (torch.Tensor, np.ndarray, Image.Image), name)
    
    if isinstance(image, Image.Image):
        # PIL image is valid
        return
    
    # Validate tensor/array dimensions
    if hasattr(image, 'ndim'):
        ndim = image.ndim
        shape = image.shape
    else:
        ndim = len(image.shape)
        shape = image.shape
    
    if ndim not in [2, 3, 4]:
        raise ValidationError(
            f"{name} must have 2, 3, or 4 dimensions, got {ndim}"
        )
    
    # Check reasonable image dimensions
    if ndim >= 2:
        h, w = shape[-2:]
        if h < 1 or w < 1 or h > 10000 or w > 10000:
            raise ValidationError(
                f"{name} has unreasonable dimensions: {h}x{w}"
            )
    
    # Check color channels if present
    if ndim == 3 and shape[0] not in [1, 3, 4]:
        raise ValidationError(
            f"{name} must have 1, 3, or 4 color channels, got {shape[0]}"
        )


def validate_text(text: str, name: str = "text") -> None:
    """Validate text input.
    
    Args:
        text: Text to validate
        name: Name for error messages
        
    Raises:
        ValidationError: If validation fails
    """
    validate_input(text, str, name)
    
    # Check for reasonable length
    if len(text) > 1000000:  # 1MB of text
        raise ValidationError(f"{name} is too long ({len(text)} characters)")
    
    # Check for valid encoding
    try:
        text.encode('utf-8')
    except UnicodeEncodeError as e:
        raise ValidationError(f"{name} contains invalid characters: {e}")


def validate_hypervector_dim(dim: int) -> None:
    """Validate hypervector dimensionality.
    
    Args:
        dim: Hypervector dimensionality
        
    Raises:
        ValidationError: If validation fails
    """
    validate_input(dim, int, "hypervector dimension")
    validate_positive(dim, "hypervector dimension")
    
    # Check reasonable range
    if dim < 100:
        raise ValidationError(
            f"Hypervector dimension too small ({dim}). "
            "Recommended minimum: 100"
        )
    
    if dim > 100000:
        raise ValidationError(
            f"Hypervector dimension too large ({dim}). "
            "Recommended maximum: 100000"
        )


def validate_device(device: str) -> None:
    """Validate device specification.
    
    Args:
        device: Device string (e.g., 'cpu', 'cuda', 'cuda:0')
        
    Raises:
        ValidationError: If validation fails
    """
    validate_input(device, str, "device")
    
    valid_devices = ['cpu']
    if torch.cuda.is_available():
        valid_devices.extend([f'cuda:{i}' for i in range(torch.cuda.device_count())])
        valid_devices.append('cuda')
    
    if device not in valid_devices:
        raise ValidationError(
            f"Invalid device '{device}'. Valid options: {valid_devices}"
        )