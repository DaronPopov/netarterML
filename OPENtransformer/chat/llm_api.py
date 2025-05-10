"""
LLM API engine for the AI Studio application.
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class LLMAPI:
    def __init__(self, model_name, hf_token=None):
        """Initialize the LLM API engine.
        
        Args:
            model_name (str): The model name to use
            hf_token (str, optional): HuggingFace token for authentication
        """
        self.model_name = model_name
        self.hf_token = hf_token
        self.tokenizer = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def load_model(self):
        """Load the model and tokenizer."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                token=self.hf_token,
                trust_remote_code=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                token=self.hf_token,
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            ).to(self.device)
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def chat(self, message):
        """Generate a response for the given message.
        
        Args:
            message (str): The input message
            
        Returns:
            str: The generated response
        """
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model not loaded")
        
        inputs = self.tokenizer(message, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_length=2048,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True
        )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response 