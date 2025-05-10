import json
import numpy as np
import ctypes
import logging
from typing import List, Dict, Optional, Union

logger = logging.getLogger("OPENtransformer.core.asm.tokenizer")

class BasicTokenizer:
    """Basic tokenizer implementation that wraps HuggingFace's tokenizer."""
    
    def __init__(self, vocab_size: int = 32000, max_length: int = 2048, hf_tokenizer=None):
        """
        Initialize the tokenizer.
        
        Args:
            vocab_size: Size of the vocabulary
            max_length: Maximum sequence length
            hf_tokenizer: HuggingFace tokenizer to use
        """
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.hf_tokenizer = hf_tokenizer
        
        # Initialize special tokens based on HuggingFace tokenizer if available
        if hf_tokenizer is not None:
            # Use HuggingFace's special tokens
            self.eos_token = hf_tokenizer.eos_token or "<|endoftext|>"
            self.pad_token = hf_tokenizer.pad_token or "<|pad|>"
            self.unk_token = hf_tokenizer.unk_token or "<|unk|>"
            self.bos_token = hf_tokenizer.bos_token or "<s>"
            
            # Get special token IDs from HuggingFace tokenizer
            self.eos_token_id = hf_tokenizer.eos_token_id if hf_tokenizer.eos_token_id is not None else 0
            self.pad_token_id = hf_tokenizer.pad_token_id if hf_tokenizer.pad_token_id is not None else 1
            self.unk_token_id = hf_tokenizer.unk_token_id if hf_tokenizer.unk_token_id is not None else 2
            self.bos_token_id = hf_tokenizer.bos_token_id if hf_tokenizer.bos_token_id is not None else 1
            
            logger.info(f"Initialized tokenizer wrapper with HuggingFace tokenizer")
        else:
            # Default special tokens
            self.eos_token = "<|endoftext|>"
            self.pad_token = "<|pad|>"
            self.unk_token = "<|unk|>"
            self.bos_token = "<s>"
            
            # Set special token IDs
            self.eos_token_id = 0
            self.pad_token_id = 1
            self.unk_token_id = 2
            self.bos_token_id = 1
            
            logger.warning("No HuggingFace tokenizer provided, using default implementation")
            
        # Initialize empty maps for fallback tokenizer
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        
        # Initialize special tokens
        self.token_to_id[self.eos_token] = self.eos_token_id
        self.token_to_id[self.pad_token] = self.pad_token_id
        self.token_to_id[self.unk_token] = self.unk_token_id
        self.token_to_id[self.bos_token] = self.bos_token_id
        
        self.id_to_token[self.eos_token_id] = self.eos_token
        self.id_to_token[self.pad_token_id] = self.pad_token
        self.id_to_token[self.unk_token_id] = self.unk_token
        self.id_to_token[self.bos_token_id] = self.bos_token
        
        # Add common whitespace tokens
        self.token_to_id["\n"] = 10
        self.token_to_id["\t"] = 9
        self.token_to_id["\r"] = 13
        self.id_to_token[10] = "\n"
        self.id_to_token[9] = "\t"
        self.id_to_token[13] = "\r"
        
        # If HuggingFace tokenizer is provided, load its vocabulary for fallback
        if hf_tokenizer is not None:
            self._load_hf_vocabulary(hf_tokenizer)
        else:
            # Initialize basic vocabulary with ASCII characters
            for i in range(32, 127):  # Printable ASCII characters
                char = chr(i)
                token_id = i
                self.token_to_id[char] = token_id
                self.id_to_token[token_id] = char
    
    def _load_hf_vocabulary(self, hf_tokenizer):
        """Load vocabulary from a HuggingFace tokenizer for fallback."""
        try:
            # Get the vocabulary from the HuggingFace tokenizer
            vocab = hf_tokenizer.get_vocab()
            
            # Update our fallback vocabulary with the HuggingFace tokens
            for token, hf_id in vocab.items():
                if hf_id < self.vocab_size:
                    self.token_to_id[token] = hf_id
                    self.id_to_token[hf_id] = token
            
            logger.info(f"Successfully loaded fallback vocabulary from HuggingFace tokenizer with {len(vocab)} tokens")
        except Exception as e:
            logger.error(f"Error loading HuggingFace vocabulary: {e}")
    
    def tokenize(self, text: str) -> List[int]:
        """
        Tokenize the input text.
        
        Args:
            text: Input text to tokenize
            
        Returns:
            List of token IDs
        """
        # If HuggingFace tokenizer is available, use it
        if self.hf_tokenizer is not None:
            try:
                return self.hf_tokenizer.encode(text)
            except Exception as e:
                logger.warning(f"Error using HuggingFace tokenizer: {e}. Falling back to basic implementation.")
        
        # Fallback to basic implementation
        return self._fallback_tokenize(text)
    
    def _fallback_tokenize(self, text: str) -> List[int]:
        """Fallback tokenization implementation in Python."""
        tokens = []
        current_pos = 0
        text_len = len(text)
        
        while current_pos < text_len:
            # Try to find the longest matching token
            max_token_len = 0
            max_token_id = self.unk_token_id
            
            # Get the next character and its byte length
            char = text[current_pos]
            char_len = len(char.encode('utf-8'))
            
            # First try exact character match
            if char in self.token_to_id:
                max_token_len = char_len
                max_token_id = self.token_to_id[char]
            
            # Then try longer token matches
            for token, token_id in self.token_to_id.items():
                token_len = len(token)
                if current_pos + token_len <= text_len:
                    if text[current_pos:current_pos + token_len] == token:
                        if token_len > max_token_len:
                            max_token_len = token_len
                            max_token_id = token_id
            
            tokens.append(max_token_id)
            # Advance by at least one character's worth of bytes
            current_pos += max(max_token_len, char_len)
        
        return tokens
    
    def decode(self, token_ids: List[int]) -> str:
        """
        Decode token IDs back to text.
        
        Args:
            token_ids: List of token IDs
            
        Returns:
            Decoded text
        """
        # Use HuggingFace tokenizer for decoding if available
        if self.hf_tokenizer is not None:
            try:
                return self.hf_tokenizer.decode(token_ids)
            except Exception as e:
                logger.warning(f"Error using HuggingFace decoder: {e}. Falling back to basic implementation.")
        
        # Fallback to basic implementation
        result = []
        for token_id in token_ids:
            if token_id in self.id_to_token:
                token = self.id_to_token[token_id]
                # Skip special tokens in output
                if token not in [self.eos_token, self.pad_token, self.unk_token, self.bos_token]:
                    result.append(token)
        return ''.join(result)
    
    def save(self, path: str):
        """Save tokenizer vocabulary to a JSON file."""
        with open(path, 'w') as f:
            json.dump({
                'vocab_size': self.vocab_size,
                'token_to_id': self.token_to_id,
                'id_to_token': {str(k): v for k, v in self.id_to_token.items()}
            }, f)
    
    def load(self, path: str):
        """Load tokenizer vocabulary from a JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
            self.vocab_size = data['vocab_size']
            self.token_to_id = data['token_to_id']
            self.id_to_token = {int(k): v for k, v in data['id_to_token'].items()}

def test_tokenizer():
    """Test the tokenizer implementation."""
    logging.basicConfig(level=logging.INFO)
    logger.info("Testing tokenizer...")
    
    # Create tokenizer
    tokenizer = BasicTokenizer()
    
    # Test texts
    test_cases = [
        "Hello, world!",
        "Testing UTF-8: 你好，世界！",
        "Mixed text with numbers 123 and symbols @#$",
        "Multiple\nlines\nof\ntext",
        "Emojis: 👋 🌍 🎉"
    ]
    
    for text in test_cases:
        logger.info(f"\nTesting text: {text}")
        try:
            # Tokenize
            tokens = tokenizer.tokenize(text)
            logger.info(f"Number of tokens: {len(tokens)}")
            logger.info(f"Token IDs: {tokens}")
            
            # Detokenize
            reconstructed = tokenizer.decode(tokens)
            logger.info(f"Reconstructed text: {reconstructed}")
            
            # Check if the text was preserved
            if reconstructed == text:
                logger.info("✓ Text perfectly preserved")
            else:
                logger.info("⚠ Text changed during tokenization/detokenization")
                
        except Exception as e:
            logger.error(f"Error processing text: {str(e)}")

if __name__ == "__main__":
    test_tokenizer() 