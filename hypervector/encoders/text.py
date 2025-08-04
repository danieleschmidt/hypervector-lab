"""Text encoding for hyperdimensional computing."""

import torch
import torch.nn as nn
import string
from typing import Dict, List, Optional
from ..core.hypervector import HyperVector
from ..core.operations import bind, bundle


class TextEncoder:
    """
    Encode text into hypervectors using token-level or character-level encoding.
    """
    
    def __init__(self, dim: int = 10000, device: Optional[str] = None):
        """Initialize text encoder.
        
        Args:
            dim: Hypervector dimensionality
            device: Compute device
        """
        self.dim = dim
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize character-level encoding table
        self.char_vectors: Dict[str, HyperVector] = {}
        self._initialize_char_vectors()
        
        # Initialize position vectors for sequence encoding
        self.position_vectors: Dict[int, HyperVector] = {}
        self.max_positions = 512
        self._initialize_position_vectors()
        
        # Simple word tokenization (can be replaced with more sophisticated tokenizers)
        self.word_vectors: Dict[str, HyperVector] = {}
        
    def _initialize_char_vectors(self) -> None:
        """Initialize random hypervectors for each character."""
        # ASCII printable characters
        chars = string.printable
        for i, char in enumerate(chars):
            self.char_vectors[char] = HyperVector.random(
                dim=self.dim, 
                device=self.device, 
                seed=hash(char) % 2**31
            )
    
    def _initialize_position_vectors(self) -> None:
        """Initialize position encoding vectors."""
        for pos in range(self.max_positions):
            self.position_vectors[pos] = HyperVector.random(
                dim=self.dim,
                device=self.device,
                seed=(hash("position") + pos) % 2**31
            )
    
    def encode_character(self, char: str) -> HyperVector:
        """Encode single character."""
        if char not in self.char_vectors:
            # Create vector for unknown character
            self.char_vectors[char] = HyperVector.random(
                dim=self.dim,
                device=self.device,
                seed=hash(char) % 2**31
            )
        return self.char_vectors[char]
    
    def encode_word(self, word: str) -> HyperVector:
        """Encode word using character composition."""
        if not word:
            return HyperVector.zeros(self.dim, device=self.device)
        
        # Check cache first
        if word in self.word_vectors:
            return self.word_vectors[word]
        
        # Encode each character and bind with position
        char_hvs = []
        for i, char in enumerate(word):
            char_hv = self.encode_character(char)
            pos_hv = self.position_vectors.get(i % self.max_positions, 
                                              self.position_vectors[0])
            positioned_char = bind(char_hv, pos_hv)
            char_hvs.append(positioned_char)
        
        # Bundle all positioned characters
        word_hv = bundle(char_hvs, normalize=True)
        
        # Cache for future use
        self.word_vectors[word] = word_hv
        return word_hv
    
    def encode_sentence(self, sentence: str) -> HyperVector:
        """Encode sentence using word composition."""
        if not sentence.strip():
            return HyperVector.zeros(self.dim, device=self.device)
        
        # Simple tokenization (split by whitespace and punctuation)
        words = sentence.lower().split()
        
        if not words:
            return HyperVector.zeros(self.dim, device=self.device)
        
        # Encode each word and bind with position
        word_hvs = []
        for i, word in enumerate(words):
            # Clean word (remove punctuation)
            clean_word = ''.join(c for c in word if c.isalnum())
            if clean_word:
                word_hv = self.encode_word(clean_word)
                pos_hv = self.position_vectors.get(i % self.max_positions,
                                                  self.position_vectors[0])
                positioned_word = bind(word_hv, pos_hv)
                word_hvs.append(positioned_word)
        
        if not word_hvs:
            return HyperVector.zeros(self.dim, device=self.device)
        
        # Bundle all positioned words
        return bundle(word_hvs, normalize=True)
    
    def encode(self, text: str, method: str = "token") -> HyperVector:
        """Encode text using specified method.
        
        Args:
            text: Input text
            method: Encoding method ('token', 'character', 'sentence')
        """
        if method == "character":
            # Character-level encoding
            char_hvs = [self.encode_character(char) for char in text]
            if not char_hvs:
                return HyperVector.zeros(self.dim, device=self.device)
            return bundle(char_hvs, normalize=True)
            
        elif method == "token" or method == "word":
            # Word-level encoding
            return self.encode_sentence(text)
            
        elif method == "sentence":
            # Sentence-level encoding (same as token for now)
            return self.encode_sentence(text)
            
        else:
            raise ValueError(f"Unknown encoding method: {method}")
    
    def similarity(self, text1: str, text2: str, method: str = "token") -> float:
        """Compute similarity between two texts."""
        hv1 = self.encode(text1, method=method)
        hv2 = self.encode(text2, method=method)
        return hv1.cosine_similarity(hv2).item()
    
    def clear_cache(self) -> None:
        """Clear cached word vectors."""
        self.word_vectors.clear()