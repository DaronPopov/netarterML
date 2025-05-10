import logging
import json
from pathlib import Path
from finlib.core.model_converter import ConvertedModelLoader
from finlib.core.asm.kernels.fused_transformer_op import create_fully_fused_transformer_op
from finlib.core.asm.assembler.builder import build_and_jit
import numpy as np
import os
import traceback
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Optional, Tuple

# Configure logging with clear formatting
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('chat_inference.log')
    ]
)
logger = logging.getLogger(__name__)

# Build and compile the fused transformer operation once
logger.info("Initializing fused transformer operation...")
try:
    FUSED_OP = create_fully_fused_transformer_op(build_and_jit)
    logger.info("Successfully initialized fused transformer operation")
except Exception as e:
    logger.error(f"Failed to initialize fused transformer operation: {str(e)}")
    raise

def create_builder_func():
    """Create a builder function that returns the pre-compiled fused transformer op."""
    def builder(code, name):
        return FUSED_OP
    return builder

def calculate_gflops(
    sequence_length: int,
    d_model: int,
    n_heads: int,
    n_layers: int,
    batch_size: int = 1
) -> float:
    """
    Calculate theoretical GFLOPS for transformer forward pass.
    
    Args:
        sequence_length: Length of input sequence
        d_model: Model dimension
        n_heads: Number of attention heads
        n_layers: Number of transformer layers
        batch_size: Batch size
        
    Returns:
        Theoretical GFLOPS
    """
    # Attention computation
    attention_flops = 2 * sequence_length * sequence_length * d_model * batch_size
    
    # FFN computation
    ffn_flops = 8 * sequence_length * d_model * d_model * batch_size
    
    # Total flops per layer
    layer_flops = attention_flops + ffn_flops
    
    # Total flops for all layers
    total_flops = layer_flops * n_layers
    
    # Convert to GFLOPS
    return total_flops / 1e9

class ChatInference:
    def __init__(
        self,
        model_dir: str,
        max_seq_length: int = 2048,
        temperature: float = 0.7,  # Lower temperature for more focused outputs
        top_p: float = 0.9,  # Nucleus sampling
        top_k: int = 50,  # Top-k sampling
        repetition_penalty: float = 1.2,  # Moderate repetition penalty
        presence_penalty: float = 0.0,  # No presence penalty
        frequency_penalty: float = 0.0,  # No frequency penalty
        length_penalty: float = 1.0  # No length penalty
    ):
        """
        Initialize chat inference with a converted model.
        
        Args:
            model_dir: Directory containing the converted model
            max_seq_length: Maximum sequence length for inference
            temperature: Sampling temperature (lower = more focused)
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            repetition_penalty: Penalty for repeated tokens
            presence_penalty: Penalty for presence of tokens
            frequency_penalty: Penalty for frequency of tokens
            length_penalty: Penalty for sequence length
        """
        try:
            # Load inference setup
            setup_path = os.path.join(model_dir, "inference_setup.json")
            if not os.path.exists(setup_path):
                raise FileNotFoundError(f"Model setup file not found at {setup_path}")
            
            with open(setup_path, "r") as f:
                self.setup = json.load(f)
            
            # Initialize tokenizer with proper BLOOM settings
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.setup["model_name"],
                use_fast=True,
                model_max_length=max_seq_length,
                padding_side="left",
                truncation_side="left"
            )
            
            # Set special tokens for BLOOM
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "left"
            
            # Load model
            builder_func = create_builder_func()
            self.loader = ConvertedModelLoader(model_dir)
            self.loader.load_model(builder_func)
            
            # Get model dimensions
            dimensions = self.loader.get_model_dimensions()
            self.d_model = dimensions["d_model"]
            self.n_heads = dimensions["n_heads"]
            self.n_layers = dimensions["n_layers"]
            
            # Set generation parameters
            self.max_seq_length = max_seq_length
            self.temperature = temperature
            self.top_p = top_p
            self.top_k = top_k
            self.repetition_penalty = repetition_penalty
            self.presence_penalty = presence_penalty
            self.frequency_penalty = frequency_penalty
            self.length_penalty = length_penalty
            
            # Get special tokens
            self.eos_token_id = self.tokenizer.eos_token_id
            self.bos_token_id = self.tokenizer.bos_token_id if self.tokenizer.bos_token_id is not None else self.eos_token_id
            
            logger.info(f"Successfully initialized chat inference with model dimensions:")
            logger.info(f"- d_model: {self.d_model}")
            logger.info(f"- n_heads: {self.n_heads}")
            logger.info(f"- n_layers: {self.n_layers}")
            logger.info(f"Sampling parameters:")
            logger.info(f"- temperature: {temperature}")
            logger.info(f"- top_p: {top_p}")
            logger.info(f"- top_k: {top_k}")
            logger.info(f"- repetition_penalty: {repetition_penalty}")
            
        except Exception as e:
            logger.error(f"Failed to initialize chat inference: {str(e)}")
            raise

    def apply_sampling_penalties(
        self,
        logits: np.ndarray,
        generated_ids: List[int],
        current_length: int
    ) -> np.ndarray:
        """
        Apply various sampling penalties to the logits.
        
        Args:
            logits: Raw logits from the model
            generated_ids: List of generated token IDs
            current_length: Current sequence length
            
        Returns:
            Modified logits with penalties applied
        """
        # Apply temperature
        logits = logits / self.temperature
        
        # Apply repetition penalty
        if self.repetition_penalty != 1.0:
            for token_id in set(generated_ids):
                logits[token_id] /= self.repetition_penalty
        
        # Apply presence penalty
        if self.presence_penalty != 0.0:
            for token_id in set(generated_ids):
                logits[token_id] -= self.presence_penalty
        
        # Apply frequency penalty
        if self.frequency_penalty != 0.0:
            token_counts = {}
            for token_id in generated_ids:
                token_counts[token_id] = token_counts.get(token_id, 0) + 1
            for token_id, count in token_counts.items():
                logits[token_id] -= self.frequency_penalty * count
        
        # Apply length penalty
        if self.length_penalty != 1.0:
            logits = logits * (1.0 / (current_length ** self.length_penalty))
        
        return logits

    def sample_next_token(
        self,
        logits: np.ndarray,
        generated_ids: List[int],
        current_length: int
    ) -> int:
        """
        Sample the next token using configured sampling strategies.
        
        Args:
            logits: Raw logits from the model
            generated_ids: List of generated token IDs
            current_length: Current sequence length
            
        Returns:
            Sampled token ID
        """
        # Apply all penalties
        logits = self.apply_sampling_penalties(logits, generated_ids, current_length)
        
        # Apply top-k sampling
        if self.top_k > 0:
            top_k_indices = np.argsort(logits)[-self.top_k:]
            logits = np.where(np.isin(np.arange(len(logits)), top_k_indices), logits, float('-inf'))
        
        # Apply top-p (nucleus) sampling
        probs = np.exp(logits)
        probs = probs / np.sum(probs)
        sorted_probs = np.sort(probs)[::-1]
        cumulative_probs = np.cumsum(sorted_probs)
        sorted_indices = np.argsort(probs)[::-1]
        
        # Find the cutoff point
        cutoff_idx = np.where(cumulative_probs <= self.top_p)[0]
        if len(cutoff_idx) > 0:
            cutoff_idx = cutoff_idx[-1] + 1
        else:
            cutoff_idx = 1
            
        # Get the filtered probabilities and indices
        filtered_probs = sorted_probs[:cutoff_idx]
        filtered_indices = sorted_indices[:cutoff_idx]
        
        # Normalize filtered probabilities
        filtered_probs = filtered_probs / np.sum(filtered_probs)
        
        # Sample from filtered distribution
        return int(np.random.choice(filtered_indices, p=filtered_probs))

    def generate_response(
        self,
        prompt: str,
        max_length: int = 100,
        stop_sequences: Optional[List[str]] = None
    ) -> Tuple[str, Dict[str, float]]:
        """
        Generate a response to the given prompt.
        
        Args:
            prompt: Input prompt text
            max_length: Maximum length of generated response
            stop_sequences: Optional list of sequences to stop generation at
            
        Returns:
            Tuple of (generated response text, performance metrics)
        """
        try:
            # Format prompt with BLOOM chat format
            formatted_prompt = f"Human: {prompt}\nAssistant:"
            
            # Encode input prompt with proper padding
            inputs = self.tokenizer(
                formatted_prompt,
                return_tensors="np",
                padding=True,
                truncation=True,
                max_length=self.max_seq_length - max_length
            )
            
            # Start with BOS token if available
            generated_ids = [self.bos_token_id] if self.bos_token_id != self.eos_token_id else []
            generated_ids.extend(inputs["input_ids"][0].tolist())
            
            # Initialize timing variables
            start_time = time.time()
            total_tokens = 0
            total_flops = 0
            
            # Generate tokens
            for _ in range(max_length):
                # Convert generated_ids to numpy array with batch dimension
                input_array = np.array([generated_ids], dtype=np.int64)
                
                # Get next token logits
                logits = self.loader.get_next_token_logits(input_array)
                logits = logits[0]  # Get logits for the current batch
                
                # Sample next token
                next_token = self.sample_next_token(logits, generated_ids, len(generated_ids))
                
                # Add token to sequence
                generated_ids.append(next_token)
                
                # Update metrics
                total_tokens += 1
                total_flops += calculate_gflops(
                    len(generated_ids),
                    self.d_model,
                    self.n_heads,
                    self.n_layers
                )
                
                # Check for end of sequence
                if next_token == self.eos_token_id:
                    break
                
                # Check for stop sequences
                if stop_sequences:
                    current_text = self.tokenizer.decode(generated_ids)
                    if any(seq in current_text for seq in stop_sequences):
                        break
            
            # Calculate timing metrics
            end_time = time.time()
            generation_time = end_time - start_time
            tokens_per_second = total_tokens / generation_time if generation_time > 0 else 0
            gflops_per_second = total_flops / generation_time if generation_time > 0 else 0
            
            # Decode the generated sequence
            generated_text = self.tokenizer.decode(generated_ids)
            
            # Extract only the Assistant's response
            try:
                response_parts = generated_text.split("Assistant:")
                if len(response_parts) > 1:
                    response = response_parts[-1].strip()
                else:
                    response = generated_text.strip()
                
                # Clean up the response
                response = response.replace("\n", " ").strip()
                
            except Exception as e:
                logger.warning(f"Could not extract clean response: {str(e)}")
                response = generated_text.strip()
            
            # Prepare performance metrics
            metrics = {
                "generation_time": generation_time,
                "tokens_per_second": tokens_per_second,
                "gflops_per_second": gflops_per_second,
                "total_tokens": total_tokens,
                "total_gflops": total_flops
            }
            
            return response, metrics
            
        except Exception as e:
            logger.error(f"Error during text generation: {str(e)}")
            raise

def main():
    """Main function to run chat inference."""
    try:
        # Initialize chat inference with optimized sampling parameters
        model_dir = "converted_models/bloom_1b1"  # Update this path as needed
        chat = ChatInference(
            model_dir,
            temperature=0.7,  # Lower temperature for more focused outputs
            top_p=0.9,  # Nucleus sampling
            top_k=50,  # Top-k sampling
            repetition_penalty=1.2,  # Moderate repetition penalty
            presence_penalty=0.0,  # No presence penalty
            frequency_penalty=0.0,  # No frequency penalty
            length_penalty=1.0,  # No length penalty
            batch_size=4
        )
        
        logger.info("Chat inference initialized. Type 'quit' to exit.")
        logger.info("Enter your message:")
        
        while True:
            try:
                user_input = input("> ").strip()
                
                if user_input.lower() == 'quit':
                    break
                
                if not user_input:
                    continue
                
                # Generate response
                response, metrics = chat.generate_response(user_input)
                print(f"\nAssistant: {response}\n")
                
                # Print performance metrics
                print(f"Performance metrics:")
                print(f"- Generation time: {metrics['generation_time']:.2f} seconds")
                print(f"- Tokens per second: {metrics['tokens_per_second']:.2f}")
                print(f"- GFLOPS: {metrics['gflops_per_second']:.2f}")
                print(f"- Total tokens: {metrics['total_tokens']}")
                print(f"- Total GFLOPS: {metrics['total_gflops']:.2f}\n")
                
            except KeyboardInterrupt:
                logger.info("\nChat session terminated by user.")
                break
            except Exception as e:
                logger.error(f"Error during chat: {str(e)}")
                logger.error("Please try again or type 'quit' to exit.")
                
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        raise

if __name__ == "__main__":
    main() 