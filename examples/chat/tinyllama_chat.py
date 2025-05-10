import os
import sys
import torch
from pathlib import Path

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import the custom inference engine from the engines package
try:
    from engines.llm_api import LLMAPI
    print("Successfully imported custom LLM inference engine")
except ImportError as e:
    print(f"Error importing custom inference engine: {e}")
    sys.exit(1)

def main():
    # Initialize the model
    print("Loading TinyLlama model...")
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    
    # Get Hugging Face token from environment
    hf_token = os.getenv('HUGGINGFACE_TOKEN')
    if not hf_token:
        print("Error: HUGGINGFACE_TOKEN environment variable not set")
        sys.exit(1)
    
    # Initialize the API with SIMD optimizations
    chat = LLMAPI(model_name=model_name, hf_token=hf_token)
    
    # Load the model with FP16 precision
    chat.load_model(device_map="cpu", torch_dtype=torch.float16)  # Use FP16 for better performance
    
    print("\nTinyLlama Chat initialized with FP16 precision!")
    print("----------------------------------------")
    
    # Chat loop
    while True:
        # Get user input
        user_input = input("\nYou: ").strip()
        
        # Check if user wants to quit
        if user_input.lower() == 'quit':
            print("\nGoodbye!")
            break
        
        try:
            # Get response from the model using SIMD optimizations
            response = chat.chat(
                message=user_input,
                max_new_tokens=256,
                temperature=0.7,
                top_p=0.9
            )
            
            print("\nTinyLlama:", response)
            
        except Exception as e:
            print(f"\nError: {str(e)}")
            print("Please try again.")

if __name__ == "__main__":
    main() 