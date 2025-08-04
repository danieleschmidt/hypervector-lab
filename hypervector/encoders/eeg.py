"""EEG signal encoding for hyperdimensional computing."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Union, Optional, List, Tuple
from scipy import signal

from ..core.hypervector import HyperVector
from ..core.operations import bind, bundle


class EEGEncoder:
    """
    Encode EEG signals into hypervectors using temporal and spectral features.
    """
    
    def __init__(self, dim: int = 10000, device: Optional[str] = None):
        """Initialize EEG encoder.
        
        Args:
            dim: Hypervector dimensionality
            device: Compute device
        """
        self.dim = dim
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Standard EEG frequency bands
        self.freq_bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 100)
        }
        
        # Channel position vectors (for multi-channel EEG)
        self.channel_vectors = {}
        self.max_channels = 128
        self._initialize_channel_vectors()
        
        # Time position vectors
        self.time_vectors = {}
        self.max_time_steps = 1024
        self._initialize_time_vectors()
        
        # Frequency band vectors
        self.band_vectors = {}
        self._initialize_band_vectors()
    
    def _initialize_channel_vectors(self) -> None:
        """Initialize position vectors for EEG channels."""
        for ch in range(self.max_channels):
            self.channel_vectors[ch] = HyperVector.random(
                dim=self.dim,
                device=self.device,
                seed=(hash("channel") + ch) % 2**31
            )
    
    def _initialize_time_vectors(self) -> None:
        """Initialize time position vectors."""
        for t in range(self.max_time_steps):
            self.time_vectors[t] = HyperVector.random(
                dim=self.dim,
                device=self.device,
                seed=(hash("time") + t) % 2**31
            )
    
    def _initialize_band_vectors(self) -> None:
        """Initialize frequency band vectors."""
        for i, band in enumerate(self.freq_bands.keys()):
            self.band_vectors[band] = HyperVector.random(
                dim=self.dim,
                device=self.device,
                seed=(hash(band)) % 2**31
            )
    
    def _preprocess_signal(
        self, 
        signal_data: Union[torch.Tensor, np.ndarray],
        sampling_rate: float
    ) -> torch.Tensor:
        """Preprocess EEG signal."""
        if isinstance(signal_data, np.ndarray):
            signal_data = torch.from_numpy(signal_data).float()
        
        if signal_data.ndim == 1:
            # Single channel, add channel dimension
            signal_data = signal_data.unsqueeze(0)
        
        signal_data = signal_data.to(self.device)
        
        # Basic preprocessing: normalization
        signal_data = F.normalize(signal_data, dim=-1, p=2)
        
        return signal_data
    
    def extract_spectral_features(
        self, 
        signal_data: torch.Tensor, 
        sampling_rate: float
    ) -> Dict[str, torch.Tensor]:
        """Extract power spectral density features for each frequency band."""
        signal_np = signal_data.cpu().numpy()
        n_channels, n_samples = signal_np.shape
        
        # Compute power spectral density
        freqs, psd = signal.welch(
            signal_np, 
            fs=sampling_rate, 
            nperseg=min(256, n_samples//4),
            axis=-1
        )
        
        band_powers = {}
        for band_name, (low_freq, high_freq) in self.freq_bands.items():
            # Find frequency indices
            freq_mask = (freqs >= low_freq) & (freqs <= high_freq)
            if freq_mask.sum() > 0:
                band_power = np.mean(psd[:, freq_mask], axis=-1)
                band_powers[band_name] = torch.from_numpy(band_power).to(self.device)
            else:
                band_powers[band_name] = torch.zeros(n_channels, device=self.device)
        
        return band_powers
    
    def encode_temporal(
        self, 
        signal_data: Union[torch.Tensor, np.ndarray],
        sampling_rate: float = 250.0
    ) -> HyperVector:
        """Encode EEG signal using temporal features."""
        processed_signal = self._preprocess_signal(signal_data, sampling_rate)
        n_channels, n_samples = processed_signal.shape
        
        # Downsample if signal is too long
        max_samples = min(n_samples, self.max_time_steps)
        if n_samples > max_samples:
            # Simple downsampling
            indices = torch.linspace(0, n_samples-1, max_samples, dtype=torch.long)
            processed_signal = processed_signal[:, indices]
        
        # Encode each time step
        channel_hvs = []
        for ch in range(min(n_channels, self.max_channels)):
            time_hvs = []
            for t in range(processed_signal.shape[1]):
                # Encode amplitude value
                amplitude = processed_signal[ch, t].item()
                
                # Simple amplitude encoding: quantize to discrete levels
                quantized_amp = torch.round(amplitude * 10) / 10
                amp_hv = HyperVector.random(
                    dim=self.dim,
                    device=self.device,
                    seed=hash(f"amp_{quantized_amp}") % 2**31
                )
                
                # Bind with time position
                time_hv = self.time_vectors.get(t, self.time_vectors[0])
                temporal_hv = bind(amp_hv, time_hv)
                time_hvs.append(temporal_hv)
            
            if time_hvs:
                # Bundle temporal features for this channel
                channel_temporal = bundle(time_hvs, normalize=True)
                
                # Bind with channel position
                channel_hv = self.channel_vectors.get(ch, self.channel_vectors[0])
                positioned_channel = bind(channel_temporal, channel_hv)
                channel_hvs.append(positioned_channel)
        
        if not channel_hvs:
            return HyperVector.zeros(self.dim, device=self.device)
        
        # Bundle all channels
        return bundle(channel_hvs, normalize=True)
    
    def encode_spectral(
        self, 
        signal_data: Union[torch.Tensor, np.ndarray],
        sampling_rate: float = 250.0
    ) -> HyperVector:
        """Encode EEG signal using spectral features."""
        processed_signal = self._preprocess_signal(signal_data, sampling_rate)
        
        # Extract band powers
        band_powers = self.extract_spectral_features(processed_signal, sampling_rate)
        
        # Encode each channel's spectral features
        channel_hvs = []
        n_channels = processed_signal.shape[0]
        
        for ch in range(min(n_channels, self.max_channels)):
            band_hvs = []
            
            for band_name, power in band_powers.items():
                if ch < len(power):
                    # Quantize power value
                    quantized_power = torch.round(power[ch] * 100) / 100
                    
                    # Create power encoding
                    power_hv = HyperVector.random(
                        dim=self.dim,
                        device=self.device,
                        seed=hash(f"power_{quantized_power}") % 2**31
                    )
                    
                    # Bind with frequency band
                    band_hv = self.band_vectors[band_name]
                    spectral_hv = bind(power_hv, band_hv)
                    band_hvs.append(spectral_hv)
            
            if band_hvs:
                # Bundle spectral features for this channel
                channel_spectral = bundle(band_hvs, normalize=True)
                
                # Bind with channel position
                channel_hv = self.channel_vectors.get(ch, self.channel_vectors[0])
                positioned_channel = bind(channel_spectral, channel_hv)
                channel_hvs.append(positioned_channel)
        
        if not channel_hvs:
            return HyperVector.zeros(self.dim, device=self.device)
        
        # Bundle all channels
        return bundle(channel_hvs, normalize=True)
    
    def encode(
        self, 
        signal_data: Union[torch.Tensor, np.ndarray],
        sampling_rate: float = 250.0,
        method: str = "combined"
    ) -> HyperVector:
        """Encode EEG signal using specified method.
        
        Args:
            signal_data: EEG signal data (channels x samples)
            sampling_rate: Sampling rate in Hz
            method: Encoding method ('temporal', 'spectral', 'combined')
        """
        if method == "temporal":
            return self.encode_temporal(signal_data, sampling_rate)
        elif method == "spectral":
            return self.encode_spectral(signal_data, sampling_rate)
        elif method == "combined":
            temporal_hv = self.encode_temporal(signal_data, sampling_rate)
            spectral_hv = self.encode_spectral(signal_data, sampling_rate)
            return bind(temporal_hv, spectral_hv)
        else:
            raise ValueError(f"Unknown encoding method: {method}")
    
    def similarity(
        self,
        signal1: Union[torch.Tensor, np.ndarray],
        signal2: Union[torch.Tensor, np.ndarray],
        sampling_rate: float = 250.0,
        method: str = "combined"
    ) -> float:
        """Compute similarity between two EEG signals."""
        hv1 = self.encode(signal1, sampling_rate, method)
        hv2 = self.encode(signal2, sampling_rate, method)
        return hv1.cosine_similarity(hv2).item()