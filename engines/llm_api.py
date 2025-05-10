#!/usr/bin/env python3
import os
import sys
import torch
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Union, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
import time

# Add the project root to Python path
project_root = str(Path(__file__).parent.parent.absolute())
sys.path.insert(0, project_root)

# Import after path setup
from OPENtransformer.core.asm.kernels.transformer import Transformer as TransformerLayer

# API Key Management
class APIKeyManager:
    """Centralized API key management for all models"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.keys = {}
        return cls._instance
    
    @classmethod
    def set_key(cls, key_name: str, key_value: str) -> None:
        """Set an API key"""
        instance = cls()
        instance.keys[key_name] = key_value
    
    @classmethod
    def get_key(cls, key_name: str) -> Optional[str]:
        """Get an API key"""
        instance = cls()
        return instance.keys.get(key_name)
    
    @classmethod
    def load_keys_from_env(cls) -> None:
        """Load API keys from environment variables"""
        instance = cls()
        for key in os.environ:
            if key.startswith("API_KEY_"):
                instance.keys[key[8:]] = os.environ[key]
    
    @classmethod
    def validate_key(cls, key_name: str) -> bool:
        """Validate if an API key exists"""
        instance = cls()
        return key_name in instance.keys

# Initialize API key manager
api_key_manager = APIKeyManager()
api_key_manager.load_keys_from_env()

def load_model(
    model_name: str,
    hf_token: str,
    device_map: str = "auto",
    torch_dtype: torch.dtype = torch.float32,
    **kwargs
) -> Tuple[AutoModelForCausalLM, AutoTokenizer, int, int, int]:
    """
    Load a model and tokenizer from HuggingFace
    
    Args:
        model_name: Name of the model to load
        hf_token: HuggingFace token for authentication
        device_map: Device mapping strategy
        torch_dtype: Torch data type
        **kwargs: Additional arguments to pass to from_pretrained
        
    Returns:
        Tuple of (model, tokenizer, d_model, n_heads, n_layers)
    """
    print(f"Loading {model_name}...")
    
    # Load tokenizer with English language settings
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=hf_token,
        trust_remote_code=True,
        padding_side="left",
        truncation_side="left",
        model_max_length=2048,
        use_fast=True
    )
    
    # Set special tokens for English chat
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    # Load model with memory optimization
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=hf_token,
        device_map=device_map,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        **kwargs
    )
    
    # Get model parameters
    config = model.config
    d_model = config.hidden_size
    n_heads = config.num_attention_heads
    n_layers = config.num_hidden_layers
    
    return model, tokenizer, d_model, n_heads, n_layers

def convert_weights(
    hf_model: AutoModelForCausalLM,
    d_model: int,
    n_heads: int,
    n_layers: int,
    tokenizer: AutoTokenizer,
    max_context_length: int = 2048
) -> List[TransformerLayer]:
    """Convert HuggingFace model weights to our format with 8-bit quantization."""
    
    # Initialize transformer layers
    layers = []
    for _ in range(n_layers):
        layer = TransformerLayer(
            embed_dim=d_model,
            num_heads=n_heads
        )
        layers.append(layer)
    
    # Get model state dict
    state_dict = hf_model.state_dict()
    
    # Convert embeddings with quantization
    token_embeddings = state_dict['transformer.wte.weight'].numpy()
    position_embeddings = state_dict['transformer.wpe.weight'].numpy()
    
    # Quantize embeddings
    token_embeddings = np.clip(
        np.round(token_embeddings * 127.0 / np.max(np.abs(token_embeddings))),
        -127, 127
    ).astype(np.int8)
    
    position_embeddings = np.clip(
        np.round(position_embeddings * 127.0 / np.max(np.abs(position_embeddings))),
        -127, 127
    ).astype(np.int8)
    
    # Convert layer weights with quantization
    for i, layer in enumerate(layers):
        layer_key = f'layer_{i}'
        hf_layer = hf_model.transformer.h[i]
        
        # Get attention weights
        q_proj = hf_layer.attn.q_proj.weight.detach().cpu().numpy()
        k_proj = hf_layer.attn.k_proj.weight.detach().cpu().numpy()
        v_proj = hf_layer.attn.v_proj.weight.detach().cpu().numpy()
        out_proj = hf_layer.attn.out_proj.weight.detach().cpu().numpy()
        
        # Quantize attention weights
        weights = {
            'q_proj': np.clip(
                np.round(q_proj * 127.0 / np.max(np.abs(q_proj))),
                -127, 127
            ).astype(np.int8),
            'k_proj': np.clip(
                np.round(k_proj * 127.0 / np.max(np.abs(k_proj))),
                -127, 127
            ).astype(np.int8),
            'v_proj': np.clip(
                np.round(v_proj * 127.0 / np.max(np.abs(v_proj))),
                -127, 127
            ).astype(np.int8),
            'out_proj': np.clip(
                np.round(out_proj * 127.0 / np.max(np.abs(out_proj))),
                -127, 127
            ).astype(np.int8)
        }
        
        # Get feed-forward weights
        gate_proj = hf_layer.mlp.gate_proj.weight.detach().cpu().numpy()
        up_proj = hf_layer.mlp.up_proj.weight.detach().cpu().numpy()
        down_proj = hf_layer.mlp.down_proj.weight.detach().cpu().numpy()
        
        # Quantize feed-forward weights
        weights['ff1'] = np.clip(
            np.round(np.concatenate([gate_proj, up_proj], axis=0) * 127.0 / np.max(np.abs(np.concatenate([gate_proj, up_proj], axis=0)))),
            -127, 127
        ).astype(np.int8)
        
        weights['ff2'] = np.clip(
            np.round(down_proj * 127.0 / np.max(np.abs(down_proj))),
            -127, 127
        ).astype(np.int8)
        
        # Get layer norm weights
        weights['norm1'] = np.clip(
            np.round(hf_layer.input_layernorm.weight.detach().cpu().numpy() * 127.0 / np.max(np.abs(hf_layer.input_layernorm.weight.detach().cpu().numpy()))),
            -127, 127
        ).astype(np.int8)
        
        weights['norm2'] = np.clip(
            np.round(hf_layer.post_attention_layernorm.weight.detach().cpu().numpy() * 127.0 / np.max(np.abs(hf_layer.post_attention_layernorm.weight.detach().cpu().numpy()))),
            -127, 127
        ).astype(np.int8)
        
        # Set biases to zero (not used in RMSNorm)
        weights['norm1_bias'] = np.zeros(d_model, dtype=np.int8)
        weights['norm2_bias'] = np.zeros(d_model, dtype=np.int8)
        
        # Set weights for this layer
        layer.set_weights(weights)
    
    return layers

class LLMAPI:
    """A unified API for loading, converting and using transformer models"""
    
    def __init__(self, model_name: str, hf_token: Optional[str] = None):
        """
        Initialize the LLM API with a model name and optional HuggingFace token
        
        Args:
            model_name: Name of the model to load (e.g. "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
            hf_token: Optional HuggingFace token for authentication
        """
        self.model_name = model_name
        self.start_time = None
        self.token_count = 0
        
        # Use provided token or get from API key manager
        if hf_token:
            self.hf_token = hf_token
            api_key_manager.set_key("HUGGINGFACE_TOKEN", hf_token)
        else:
            self.hf_token = api_key_manager.get_key("HUGGINGFACE_TOKEN")
            if not self.hf_token:
                raise ValueError("Please set the HUGGINGFACE_TOKEN using APIKeyManager.set_key() or provide it during initialization")
        
        self.model = None
        self.tokenizer = None
        self.conversation_history = []
        self.device = None
    
    def _update_tps_display(self, new_tokens: int = 1):
        """Update and display the current TPS rate"""
        if self.start_time is None:
            self.start_time = time.time()
            self.token_count = 0
        
        self.token_count += new_tokens
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        
        if elapsed_time > 0:
            tps = self.token_count / elapsed_time
            # Save cursor position
            print("\033[s", end="", flush=True)
            # Move cursor to fixed position (3 lines down from top)
            print("\033[3;1H", end="", flush=True)
            # Clear the line and print TPS info
            print("\033[K", end="", flush=True)  # Clear the line
            print(f"[TPS: {tps:.1f} | Tokens: {self.token_count} | Time: {elapsed_time:.1f}s]", end="", flush=True)
            # Restore cursor position
            print("\033[u", end="", flush=True)

    def load_model(self, device_map: str = "auto", torch_dtype: torch.dtype = torch.float16) -> None:
        """Load the model and tokenizer"""
        print(f"Loading {self.model_name}...")
        
        # Load tokenizer with English language settings
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            token=self.hf_token,
            trust_remote_code=True,
            padding_side="left",
            truncation_side="left",
            model_max_length=2048,
            use_fast=True
        )
        
        # Set special tokens for English chat
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        
        # Load model with memory optimization
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            token=self.hf_token,
            device_map=device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        # Get model parameters
        config = self.model.config
        self.d_model = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.n_layers = config.num_hidden_layers
        
        print(f"Model loaded successfully! (d_model={self.d_model}, n_heads={self.n_heads}, n_layers={self.n_layers})")
        
        # Get device
        self.device = next(self.model.parameters()).device
        print(f"Model loaded on {self.device}")
        
        # Initialize conversation history
        self.conversation_history = []
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Enable model for inference
        with torch.no_grad():
            # Test inference with a simple prompt
            test_input = self.tokenizer("Hello", return_tensors="pt").to(self.device)
            _ = self.model(**test_input)
        
        print("Model initialized for inference!")
    
    def chat(self, message: str, max_new_tokens: int = 256, temperature: float = 0.7, top_p: float = 0.9) -> str:
        """
        Chat with the model
        
        Args:
            message: User message
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            
        Returns:
            Model response
        """
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Reset TPS timer
        self.start_time = None
        self.token_count = 0
        
        # Prepare input with English chat format
        prompt = f"User: {message}\nAssistant:"
        
        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
            add_special_tokens=True
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Create custom streamer class that updates TPS
        class TPSStreamer(TextStreamer):
            def __init__(self, tokenizer, parent):
                super().__init__(tokenizer, skip_prompt=True)
                self.parent = parent
            
            def on_finalized_text(self, text: str, stream_end: bool = False):
                super().on_finalized_text(text, stream_end)
                # Count tokens in the new text
                new_tokens = len(self.tokenizer(text, add_special_tokens=False)['input_ids'])
                self.parent._update_tps_display(new_tokens)
        
        # Create streamer for token-by-token output
        streamer = TPSStreamer(self.tokenizer, self)
        
        # Generate response with improved parameters
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
            length_penalty=1.0,
            num_beams=1,
            streamer=streamer
        )
        
        # Decode response for return value
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract assistant's response
        response = response.split("Assistant:")[-1].strip()
        
        return response
    
    def clear_history(self) -> None:
        """Clear conversation history"""
        self.conversation_history = []

# Example usage:
if __name__ == "__main__":
    # Ensure the Hugging Face token is set as an environment variable HUGGINGFACE_TOKEN
    # or manage it using APIKeyManager.set_key("HUGGINGFACE_TOKEN", "your_token")
    # before running this example.

    # Initialize API with TinyLlama
    # The API will automatically load the token from the environment via APIKeyManager
    api = LLMAPI(
        model_name="meta-llama/Llama-2-7b-chat-hf",
        hf_token=api_key_manager.get_key("HUGGINGFACE_TOKEN")
    )

    # Load model
    api.load_model()
    
    # Example chat
    print("\nWelcome to TinyLlama chat! Type 'quit' to exit")
    print("="*50)
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == 'quit':
            break
        
        response = api.chat(user_input) 