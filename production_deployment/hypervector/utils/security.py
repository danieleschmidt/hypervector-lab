"""Security utilities for safe operation."""

import os
import re
from pathlib import Path
from typing import Union, Any
from .validation import ValidationError


def sanitize_input(text: str, max_length: int = 10000) -> str:
    """Sanitize text input for safety.
    
    Args:
        text: Input text to sanitize
        max_length: Maximum allowed length
        
    Returns:
        Sanitized text
        
    Raises:
        ValidationError: If input is invalid
    """
    if not isinstance(text, str):
        raise ValidationError("Input must be string")
    
    if len(text) > max_length:
        raise ValidationError(f"Input too long: {len(text)} > {max_length}")
    
    # Remove control characters except whitespace
    sanitized = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
    
    # Limit consecutive whitespace
    sanitized = re.sub(r'\s+', ' ', sanitized)
    
    return sanitized.strip()


def validate_file_path(
    file_path: Union[str, Path],
    allowed_extensions: list = None,
    max_size_mb: float = 100.0
) -> Path:
    """Validate and sanitize file path.
    
    Args:
        file_path: File path to validate
        allowed_extensions: List of allowed file extensions
        max_size_mb: Maximum file size in MB
        
    Returns:
        Validated Path object
        
    Raises:
        ValidationError: If path is invalid or unsafe
    """
    if not isinstance(file_path, (str, Path)):
        raise ValidationError("File path must be string or Path")
    
    path = Path(file_path).resolve()
    
    # Check for path traversal attempts
    try:
        path.relative_to(Path.cwd())
    except ValueError:
        # Allow absolute paths, but check they're not trying to escape
        if '..' in str(path):
            raise ValidationError("Path traversal not allowed")
    
    # Check file extension if specified
    if allowed_extensions:
        if path.suffix.lower() not in [ext.lower() for ext in allowed_extensions]:
            raise ValidationError(
                f"File extension {path.suffix} not allowed. "
                f"Allowed: {allowed_extensions}"
            )
    
    # Check file size if file exists
    if path.exists() and path.is_file():
        size_mb = path.stat().st_size / (1024 * 1024)
        if size_mb > max_size_mb:
            raise ValidationError(
                f"File too large: {size_mb:.1f}MB > {max_size_mb}MB"
            )
    
    return path


def sanitize_filename(filename: str, max_length: int = 255) -> str:
    """Sanitize filename for safety.
    
    Args:
        filename: Original filename
        max_length: Maximum filename length
        
    Returns:
        Sanitized filename
    """
    if not isinstance(filename, str):
        raise ValidationError("Filename must be string")
    
    # Remove/replace dangerous characters
    sanitized = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', filename)
    
    # Remove leading/trailing dots and spaces
    sanitized = sanitized.strip('. ')
    
    # Truncate if too long
    if len(sanitized) > max_length:
        name, ext = os.path.splitext(sanitized)
        max_name_length = max_length - len(ext)
        sanitized = name[:max_name_length] + ext
    
    # Ensure not empty
    if not sanitized:
        sanitized = "file"
    
    return sanitized


def check_memory_usage(limit_gb: float = 8.0) -> None:
    """Check current memory usage against limit.
    
    Args:
        limit_gb: Memory limit in GB
        
    Raises:
        ValidationError: If memory usage exceeds limit
    """
    try:
        import psutil
        process = psutil.Process()
        memory_gb = process.memory_info().rss / (1024**3)
        
        if memory_gb > limit_gb:
            raise ValidationError(
                f"Memory usage {memory_gb:.1f}GB exceeds limit {limit_gb}GB"
            )
    except ImportError:
        # psutil not available, skip check
        pass


def validate_model_file(file_path: Union[str, Path]) -> Path:
    """Validate model file for loading.
    
    Args:
        file_path: Path to model file
        
    Returns:
        Validated Path object
        
    Raises:
        ValidationError: If file is invalid
    """
    allowed_extensions = ['.pth', '.pt', '.pkl', '.pickle', '.h5', '.hdf5']
    path = validate_file_path(file_path, allowed_extensions, max_size_mb=500.0)
    
    if not path.exists():
        raise ValidationError(f"Model file not found: {path}")
    
    return path


def sanitize_config_value(value: Any) -> Any:
    """Sanitize configuration value.
    
    Args:
        value: Configuration value to sanitize
        
    Returns:
        Sanitized value
    """
    if isinstance(value, str):
        return sanitize_input(value, max_length=1000)
    elif isinstance(value, dict):
        return {k: sanitize_config_value(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [sanitize_config_value(item) for item in value]
    else:
        return value


class SecurityContext:
    """Security context manager for safe operations."""
    
    def __init__(
        self,
        validate_inputs: bool = True,
        memory_limit_gb: float = 8.0,
        file_size_limit_mb: float = 100.0
    ):
        self.validate_inputs = validate_inputs
        self.memory_limit_gb = memory_limit_gb
        self.file_size_limit_mb = file_size_limit_mb
    
    def __enter__(self):
        if self.validate_inputs:
            check_memory_usage(self.memory_limit_gb)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Clean up if needed
        pass
    
    def validate_input_data(self, data: Any, name: str = "data") -> None:
        """Validate input data within security context."""
        if not self.validate_inputs:
            return
        
        if isinstance(data, str):
            sanitize_input(data)
        
        # Check memory usage periodically
        check_memory_usage(self.memory_limit_gb)


def create_secure_temp_dir() -> Path:
    """Create secure temporary directory.
    
    Returns:
        Path to temporary directory
    """
    import tempfile
    
    temp_dir = Path(tempfile.mkdtemp(prefix='hypervector_'))
    
    # Set restrictive permissions (owner only)
    temp_dir.chmod(0o700)
    
    return temp_dir