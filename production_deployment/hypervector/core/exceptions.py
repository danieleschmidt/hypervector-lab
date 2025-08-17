"""Custom exceptions for hyperdimensional computing."""


class HDCError(Exception):
    """Base exception for HDC-related errors."""
    pass


class DimensionMismatchError(HDCError):
    """Raised when hypervector dimensions don't match."""
    pass


class InvalidModeError(HDCError):
    """Raised when an invalid hypervector mode is specified."""
    pass


class EncodingError(HDCError):
    """Raised when encoding fails."""
    pass


class DeviceError(HDCError):
    """Raised when device operations fail."""
    pass


class ModelError(HDCError):
    """Raised when model operations fail."""
    pass


class ClassificationError(HDCError):
    """Raised when classification fails."""
    pass


class RetrievalError(HDCError):
    """Raised when retrieval operations fail."""
    pass